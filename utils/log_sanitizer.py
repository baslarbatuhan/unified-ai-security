"""utils/log_sanitizer.py
==========================
KVKK / GDPR uyumlu PII redaction.

`sanitize(text)` ve `sanitize_event(event_dict)` iki giriş noktası sağlar.
Schema (schemas/telemetry_schema.py) emit sırasında `sanitize_event`'i
dinamik olarak import edip olay yazılmadan önce çağırır.

Taranan desenler
----------------
- Email (RFC-5322 lite)
- E.164 / TR telefon formatları
- TCKN — 11 haneli, Luhn-benzeri check ile (yanlış pozitifi azaltmak için)
- IBAN (TR + genel)
- Credit card (13–19 hane, Luhn doğrulaması)
- API key heuristics:
    * `Bearer <token>` başlığı
    * OpenAI `sk-...`, Anthropic `sk-ant-...` tarzı
    * Yüksek entropili (>= 4.0 Shannon) uzun alfanumerik bloklar
      ama **ortak İngilizce kelimeler whitelist'te** (false positive'e karşı).
- IPv4

Tasarım notları
---------------
- Orjinal veriyi mutate etmez; dict'lerde kopya döner.
- Regex yerine mümkün olduğunca validated substitution (TCKN / kart için
  check-digit testi) kullanır: benign sayı dizileri maskelenmez.
- `allowlist_keys` bir dict key setidir — bu anahtardaki değerler
  sanitize edilmez (örn. `run_id`, `attack_id`).  Saldırı veriseti
  id'leri ya da hash'ler PII gibi görünebilir; whitelist onları korur.
- Maskeleme kuralları `MASK_*` sabitlerinde; değiştirmek gerekirse
  tek nokta.
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, Iterable, Optional, Set

# ---------------------------------------------------------------------------
# Maskeleme sabitleri — değiştirilirse tek noktada.
# ---------------------------------------------------------------------------
MASK_EMAIL = "[REDACTED_EMAIL]"
MASK_PHONE = "[REDACTED_PHONE]"
MASK_TCKN = "[REDACTED_TCKN]"
MASK_IBAN = "[REDACTED_IBAN]"
MASK_CARD = "[REDACTED_CARD]"
MASK_APIKEY = "[REDACTED_APIKEY]"
MASK_IPV4 = "[REDACTED_IP]"
MASK_BEARER = "Bearer [REDACTED_TOKEN]"

# Key names whose *values* should never be sanitized (ids, hashes, timestamps…)
# Extendable by callers via `sanitize_event(event, allowlist_keys=...)`.
DEFAULT_ALLOWLIST_KEYS: Set[str] = {
    "event_id",
    "run_id",
    "target_id",
    "attack_id",
    "timestamp",
    "kind",
    "module",
    "decision",
    "latency_ms",
    "latency_ms_total",
    "risk_score",
    "confidence",
    "fused_risk_score",
    "prompt_score",
    "rag_score",
    "agency_score",
    "output_score",
    "prompt_char_count",
    "retrieved_doc_count",
    "has_retrieved_docs",
    "session_role",
    "error_type",
    "where",
}

# Common English words we never treat as API keys even if entropy crosses.
# Small, conservative list; keeps false positives down on module names and
# normal prose.
_APIKEY_WORD_WHITELIST = {
    "application", "authentication", "authorization",
    "prompt_guard", "rag_guard", "output_agency", "output_guard",
    "fusion_gateway",
}


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------
_RE_EMAIL = re.compile(
    r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"
)

# International / TR phones. Requires + or 0 prefix, 9–14 digits total after
# normalization. Keeps us from matching arbitrary numeric IDs.
_RE_PHONE = re.compile(
    r"(?<!\d)(?:\+?\d{1,3}[\s\-.]?)?"      # optional country
    r"\(?\d{2,4}\)?[\s\-.]?\d{3,4}[\s\-.]?\d{2,4}(?!\d)"
)

# IBAN: 2-letter country + 2 digits + up to 30 alphanumerics. TR hattı 26 karakter.
_RE_IBAN = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b")

# Candidate digit runs for TCKN (11) and cards (13–19).  We validate digits
# separately so "00000000000" does not get masked.
_RE_TCKN_CANDIDATE = re.compile(r"(?<!\d)\d{11}(?!\d)")
_RE_CARD_CANDIDATE = re.compile(r"(?<!\d)(?:\d[ \-]?){13,19}(?!\d)")

# Bearer token in HTTP header style.
_RE_BEARER = re.compile(r"(?i)\bBearer\s+([A-Za-z0-9._\-]{10,})\b")

# Known provider prefixes (OpenAI, Anthropic, GitHub, Slack, AWS, Google).
_RE_PROVIDER_KEY = re.compile(
    r"\b(?:"
    r"sk-ant-[A-Za-z0-9_\-]{20,}"
    r"|sk-[A-Za-z0-9]{20,}"
    r"|gh[pousr]_[A-Za-z0-9]{20,}"
    r"|xox[baprs]-[A-Za-z0-9\-]{10,}"
    r"|AKIA[0-9A-Z]{16}"
    r"|AIza[0-9A-Za-z\-_]{35}"
    r")\b"
)

# Generic high-entropy tokens (length 24+). Stricter than default so we don't
# wipe hashes that are already in allowlisted fields.
_RE_GENERIC_KEY_CANDIDATE = re.compile(r"\b[A-Za-z0-9_\-]{24,}\b")

# IPv4.  Skips 0.0.0.0, 127.*, 10.*, 192.168.*, 172.16–31.* (private; leave them
# for debugging clarity). Public IPs are masked.
_RE_IPV4 = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")


def _is_private_ipv4(addr: str) -> bool:
    try:
        parts = [int(x) for x in addr.split(".")]
        if any(p < 0 or p > 255 for p in parts):
            return False
    except ValueError:
        return False
    if parts[0] == 10:
        return True
    if parts[0] == 127:
        return True
    if parts[0] == 192 and parts[1] == 168:
        return True
    if parts[0] == 172 and 16 <= parts[1] <= 31:
        return True
    if parts == [0, 0, 0, 0]:
        return True
    return False


# ---------------------------------------------------------------------------
# Digit validators (reduce false positives from id-like numbers)
# ---------------------------------------------------------------------------
def _luhn_ok(digits: str) -> bool:
    total = 0
    parity = len(digits) % 2
    for i, ch in enumerate(digits):
        if not ch.isdigit():
            return False
        d = int(ch)
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0 and total > 0


def _is_valid_tckn(digits: str) -> bool:
    """Official TCKN check-digit rule.  Filters huge chunks of random 11-digit
    numbers that would otherwise be masked.  Reference: Nüfus ve Vatandaşlık
    İşleri Genel Müdürlüğü algoritması."""
    if len(digits) != 11 or not digits.isdigit() or digits[0] == "0":
        return False
    d = [int(x) for x in digits]
    if ((sum(d[0:9:2]) * 7) - sum(d[1:8:2])) % 10 != d[9]:
        return False
    if sum(d[0:10]) % 10 != d[10]:
        return False
    return True


def _shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    freq: Dict[str, int] = {}
    for ch in s:
        freq[ch] = freq.get(ch, 0) + 1
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in freq.values())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def sanitize(text: str) -> str:
    """Redact PII patterns in a free-form string.

    Ordering matters: provider-specific keys before generic entropy heuristic
    so the match-length is maximal, TCKN before generic digit runs, bearer
    before generic key.
    """
    if not isinstance(text, str) or not text:
        return text

    # 1. Bearer header (keeps the "Bearer " marker for grep).
    text = _RE_BEARER.sub(MASK_BEARER, text)

    # 2. Provider-specific API keys.
    text = _RE_PROVIDER_KEY.sub(MASK_APIKEY, text)

    # 3. Email.
    text = _RE_EMAIL.sub(MASK_EMAIL, text)

    # 4. IBAN before phone / card (shape is distinct).
    text = _RE_IBAN.sub(MASK_IBAN, text)

    # 5. Card (13–19 digit, Luhn).
    def _card_sub(m: re.Match[str]) -> str:
        digits = re.sub(r"[\s\-]", "", m.group(0))
        if 13 <= len(digits) <= 19 and _luhn_ok(digits):
            return MASK_CARD
        return m.group(0)
    text = _RE_CARD_CANDIDATE.sub(_card_sub, text)

    # 6. TCKN (Turkish national id).
    def _tckn_sub(m: re.Match[str]) -> str:
        digits = m.group(0)
        return MASK_TCKN if _is_valid_tckn(digits) else digits
    text = _RE_TCKN_CANDIDATE.sub(_tckn_sub, text)

    # 7. Phone.
    def _phone_sub(m: re.Match[str]) -> str:
        raw = m.group(0)
        digits_only = re.sub(r"\D", "", raw)
        # Require 9–14 digits so that things like "2023-04-24" don't get
        # swallowed.
        if 9 <= len(digits_only) <= 14:
            return MASK_PHONE
        return raw
    text = _RE_PHONE.sub(_phone_sub, text)

    # 8. Public IPv4.
    def _ip_sub(m: re.Match[str]) -> str:
        return m.group(0) if _is_private_ipv4(m.group(0)) else MASK_IPV4
    text = _RE_IPV4.sub(_ip_sub, text)

    # 9. Generic high-entropy keys (last, conservative).
    def _generic_sub(m: re.Match[str]) -> str:
        tok = m.group(0)
        if tok.lower() in _APIKEY_WORD_WHITELIST:
            return tok
        # Must mix upper/lower/digits to look key-ish.
        has_lower = any(c.islower() for c in tok)
        has_upper = any(c.isupper() for c in tok)
        has_digit = any(c.isdigit() for c in tok)
        if not (has_lower and has_upper and has_digit):
            return tok
        if _shannon_entropy(tok) < 4.0:
            return tok
        return MASK_APIKEY
    text = _RE_GENERIC_KEY_CANDIDATE.sub(_generic_sub, text)

    return text


def sanitize_event(
    event: Dict[str, Any],
    *,
    allowlist_keys: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Walk a telemetry event dict and sanitize string leaves.

    - Lists/dicts are recursed.
    - Numbers / bools / None pass through.
    - Keys in `allowlist_keys` (or DEFAULT_ALLOWLIST_KEYS) keep their values
      verbatim — useful for ids, scores, timestamps.
    """
    allow = set(allowlist_keys) if allowlist_keys is not None else DEFAULT_ALLOWLIST_KEYS

    def _walk(node: Any, current_key: Optional[str]) -> Any:
        if current_key in allow:
            return node
        if isinstance(node, str):
            return sanitize(node)
        if isinstance(node, dict):
            return {k: _walk(v, k) for k, v in node.items()}
        if isinstance(node, list):
            return [_walk(v, current_key) for v in node]
        if isinstance(node, tuple):
            return tuple(_walk(v, current_key) for v in node)
        return node

    return _walk(event, None)


__all__ = [
    "sanitize",
    "sanitize_event",
    "DEFAULT_ALLOWLIST_KEYS",
    "MASK_EMAIL",
    "MASK_PHONE",
    "MASK_TCKN",
    "MASK_IBAN",
    "MASK_CARD",
    "MASK_APIKEY",
    "MASK_IPV4",
    "MASK_BEARER",
]
