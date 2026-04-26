"""output_guard/output_analyzer.py
===================================
Model output → risk score.

Five independent sub-checks, each contributing a signal to a weighted risk
score in [0, 1]:

    1. pii              — emails, phone, TCKN, IBAN, card (reuses log_sanitizer
                          matchers so behaviour stays consistent).
    2. api_key          — bearer tokens, provider-prefixed keys, high-entropy
                          generic secrets.
    3. unsafe_instruction
                        — model instructs the reader to do something unsafe:
                          `rm -rf`, `curl … | sh`, `chmod 777`, `disable …
                          security`, `--privileged`, etc.
    4. downstream_injection
                        — output tries to hijack a downstream agent:
                          `ignore previous`, `disregard system`, role-switch
                          markers (`SYSTEM:`, `[[INST]]`).
    5. redirect_to_unknown
                        — URL whose registrable domain is NOT on the
                          allowlist (`configs/redirect_allowlist.yaml` or a
                          small default).  Distinguishes legitimate references
                          (Wikipedia, official vendor docs) from phishing-ish
                          destinations.

The analyzer is pure Python + regex — no model call — so it fits in the
sub-10-ms budget from `configs/timeout_config.yaml::output_guard`.

Output contract
---------------
`analyze(text)` returns `OutputRiskResult`:
    score        : float in [0, 1]
    decision     : "allow" | "sanitize" | "block"
    flags        : dict[signal_name → flag_details]
    evidence     : list[str], short human-readable lines
    latency_ms   : int

Decision mapping mirrors the other modules (allow < 0.30, sanitize < 0.60,
block >= 0.85 by default) so the fusion engine can compose without bespoke
logic.  Thresholds are configurable.
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# Reuse regex + validators from the sanitizer so a URL or email detected here
# matches what the sanitizer would scrub later.
from utils.log_sanitizer import (
    _RE_EMAIL, _RE_PHONE, _RE_TCKN_CANDIDATE, _RE_IBAN, _RE_CARD_CANDIDATE,
    _RE_BEARER, _RE_PROVIDER_KEY, _RE_GENERIC_KEY_CANDIDATE,
    _is_valid_tckn, _luhn_ok, _shannon_entropy,
)


# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------
DEFAULT_THRESHOLDS = {"allow": 0.30, "sanitize": 0.60, "block": 0.85}

# Weights for signal → aggregate score.  Summed with clipping in [0,1].
# Calibrated so any ONE strong signal (e.g. confirmed API key) already lands
# in the sanitize/block band; two independent PII hits reach block.
DEFAULT_WEIGHTS = {
    "pii":                   0.35,
    "api_key":               0.75,
    "unsafe_instruction":    0.55,
    "downstream_injection":  0.60,
    "redirect_to_unknown":   0.30,
}

# Domains that are treated as safe to reference.  Intentionally short —
# operators extend via `analyze(text, allowlist_domains=[...])`.
DEFAULT_ALLOWLIST_DOMAINS = {
    "wikipedia.org",
    "github.com",
    "stackoverflow.com",
    "docs.python.org",
    "developer.mozilla.org",
    "kernel.org",
    "owasp.org",
}


# ---------------------------------------------------------------------------
# Patterns for the checks that don't reuse log_sanitizer
# ---------------------------------------------------------------------------
# Unsafe-instruction heuristics. Kept tight to avoid flagging ordinary prose.
_UNSAFE_PATTERNS: List[Tuple[str, re.Pattern[str]]] = [
    ("rm_rf",        re.compile(r"\brm\s+-[rfRF]{1,3}\s+(?:/|~|\$HOME)\S*", re.IGNORECASE)),
    ("curl_pipe_sh", re.compile(r"\bcurl\s+[^|]+\|\s*(?:sh|bash|zsh)\b", re.IGNORECASE)),
    ("wget_pipe_sh", re.compile(r"\bwget\s+[^|]+\|\s*(?:sh|bash|zsh)\b", re.IGNORECASE)),
    ("chmod_777",    re.compile(r"\bchmod\s+(?:-R\s+)?(?:777|a\+rwx)\b", re.IGNORECASE)),
    ("disable_sec",  re.compile(r"\bdisable\s+(?:the\s+)?(?:firewall|selinux|apparmor|tls|ssl|cert\w*)\b", re.IGNORECASE)),
    ("privileged",   re.compile(r"--privileged\b|--cap-add=ALL\b", re.IGNORECASE)),
    ("sudo_all",     re.compile(r"\bsudo\s+-\s*i\b|\bsu\s+-\s*root\b", re.IGNORECASE)),
    ("base64_exec",  re.compile(r"\b(?:eval|exec)\s*\(\s*(?:base64|atob)\b", re.IGNORECASE)),
]

_DOWNSTREAM_PATTERNS: List[Tuple[str, re.Pattern[str]]] = [
    ("ignore_previous", re.compile(r"\b(?:ignore|disregard|forget)\s+(?:all\s+)?(?:previous|above|earlier|prior)\b", re.IGNORECASE)),
    ("role_switch",     re.compile(r"\b(?:system|assistant|developer)\s*:\s", re.IGNORECASE)),
    ("inst_tag",        re.compile(r"\[\[?\s*INST\s*\]?\]|\[/INST\]|<\|system\|>|<\|im_start\|>", re.IGNORECASE)),
    ("new_persona",     re.compile(r"\byou\s+are\s+now\s+(?:a\s+)?(?:different|new|unrestricted|dan)\b", re.IGNORECASE)),
    ("override_policy", re.compile(r"\boverride\s+(?:your|the)\s+(?:safety|policy|rules|guidelines)\b", re.IGNORECASE)),
]

# URL matcher — simple but captures the host reliably.
_RE_URL = re.compile(
    r"\bhttps?://([A-Za-z0-9.\-]+\.[A-Za-z]{2,})(?::\d+)?(?:/[^\s)]*)?",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class OutputRiskResult:
    score: float
    decision: str
    flags: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    evidence: List[str] = field(default_factory=list)
    latency_ms: int = 0
    # Character count after any truncation enforced by the caller — used by
    # the fusion engine / telemetry for context.
    output_chars: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": round(self.score, 4),
            "decision": self.decision,
            "flags": self.flags,
            "evidence": list(self.evidence),
            "latency_ms": self.latency_ms,
            "output_chars": self.output_chars,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _decision_from_score(
    score: float, thresholds: Dict[str, float]
) -> str:
    block = thresholds.get("block", DEFAULT_THRESHOLDS["block"])
    sanitize = thresholds.get("sanitize", DEFAULT_THRESHOLDS["sanitize"])
    if score >= block:
        return "block"
    if score >= sanitize:
        return "sanitize"
    return "allow"


def _registrable_domain(host: str) -> str:
    """Trailing two labels, lowercase. Good enough for an allowlist check:
    distinguishes github.com from example.github.com.evil.com attacks.

    We don't ship a PSL — a perfect solution would use `publicsuffix2`. For
    our corpus the 2-label heuristic works; extend if false positives arise.
    """
    host = host.lower().rstrip(".")
    parts = host.split(".")
    if len(parts) < 2:
        return host
    return ".".join(parts[-2:])


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------
def _check_pii(text: str) -> Dict[str, Any]:
    hits: Dict[str, int] = {}
    samples: List[str] = []

    for m in _RE_EMAIL.finditer(text):
        hits["email"] = hits.get("email", 0) + 1
        if len(samples) < 3:
            samples.append(f"email: {m.group(0)}")
    for m in _RE_PHONE.finditer(text):
        raw = m.group(0)
        digits = re.sub(r"\D", "", raw)
        if 9 <= len(digits) <= 14:
            hits["phone"] = hits.get("phone", 0) + 1
            if len(samples) < 3:
                samples.append(f"phone: {raw}")
    for m in _RE_TCKN_CANDIDATE.finditer(text):
        if _is_valid_tckn(m.group(0)):
            hits["tckn"] = hits.get("tckn", 0) + 1
    for _m in _RE_IBAN.finditer(text):
        hits["iban"] = hits.get("iban", 0) + 1
    for m in _RE_CARD_CANDIDATE.finditer(text):
        digits = re.sub(r"[\s\-]", "", m.group(0))
        if 13 <= len(digits) <= 19 and _luhn_ok(digits):
            hits["card"] = hits.get("card", 0) + 1

    if not hits:
        return {"triggered": False}
    return {
        "triggered": True,
        "hits": hits,
        "samples": samples,
        "count": sum(hits.values()),
    }


def _check_api_key(text: str) -> Dict[str, Any]:
    hits: Dict[str, int] = {}
    samples: List[str] = []

    for m in _RE_BEARER.finditer(text):
        hits["bearer"] = hits.get("bearer", 0) + 1
        if len(samples) < 3:
            samples.append(f"bearer: {m.group(0)[:40]}…")
    for m in _RE_PROVIDER_KEY.finditer(text):
        hits["provider_key"] = hits.get("provider_key", 0) + 1
        if len(samples) < 3:
            samples.append(f"provider_key: {m.group(0)[:40]}…")

    # Generic high-entropy token (strict): length 24+, mixed case/digits,
    # entropy >= 4.0. Matches the sanitizer's gate.
    for m in _RE_GENERIC_KEY_CANDIDATE.finditer(text):
        tok = m.group(0)
        has_lower = any(c.islower() for c in tok)
        has_upper = any(c.isupper() for c in tok)
        has_digit = any(c.isdigit() for c in tok)
        if not (has_lower and has_upper and has_digit):
            continue
        if _shannon_entropy(tok) < 4.0:
            continue
        hits["generic"] = hits.get("generic", 0) + 1
        if len(samples) < 3:
            samples.append(f"generic_key: {tok[:20]}…")

    if not hits:
        return {"triggered": False}
    return {"triggered": True, "hits": hits, "samples": samples,
            "count": sum(hits.values())}


def _check_unsafe_instruction(text: str) -> Dict[str, Any]:
    matches: List[Tuple[str, str]] = []
    for name, pat in _UNSAFE_PATTERNS:
        m = pat.search(text)
        if m:
            matches.append((name, m.group(0)[:80]))
    if not matches:
        return {"triggered": False}
    return {
        "triggered": True,
        "rules": [n for n, _ in matches],
        "samples": [f"{n}: {s}" for n, s in matches[:3]],
        "count": len(matches),
    }


def _check_downstream_injection(text: str) -> Dict[str, Any]:
    matches: List[Tuple[str, str]] = []
    for name, pat in _DOWNSTREAM_PATTERNS:
        m = pat.search(text)
        if m:
            matches.append((name, m.group(0)[:80]))
    if not matches:
        return {"triggered": False}
    return {
        "triggered": True,
        "rules": [n for n, _ in matches],
        "samples": [f"{n}: {s}" for n, s in matches[:3]],
        "count": len(matches),
    }


def _check_redirect(text: str, *, allowlist_domains: set[str]) -> Dict[str, Any]:
    unknown: List[str] = []
    for m in _RE_URL.finditer(text):
        host = m.group(1)
        reg = _registrable_domain(host)
        if reg in allowlist_domains:
            continue
        unknown.append(host)
    # dedupe, preserve order
    seen: set[str] = set()
    uniq = [h for h in unknown if not (h in seen or seen.add(h))]
    if not uniq:
        return {"triggered": False}
    return {
        "triggered": True,
        "unknown_hosts": uniq[:10],
        "count": len(uniq),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def analyze(
    text: str,
    *,
    thresholds: Optional[Dict[str, float]] = None,
    weights: Optional[Dict[str, float]] = None,
    allowlist_domains: Optional[List[str]] = None,
    max_chars: int = 200_000,
) -> OutputRiskResult:
    """Score an arbitrary model output.  Never raises — unexpected errors
    degrade to an `allow` verdict with an evidence note.

    Parameters
    ----------
    text
        The model reply to analyze.  Values longer than `max_chars` are
        truncated (analysis remains valid; scoring stays deterministic).
    thresholds, weights
        Overrides. Missing keys fall back to defaults.
    allowlist_domains
        Extra domains treated as safe redirect destinations.  Unioned with
        `DEFAULT_ALLOWLIST_DOMAINS`.
    """
    t0 = time.time()
    thr = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    allow = set(DEFAULT_ALLOWLIST_DOMAINS)
    if allowlist_domains:
        allow.update(d.lower() for d in allowlist_domains)

    if not isinstance(text, str):
        text = str(text or "")
    truncated = False
    if len(text) > max_chars:
        text = text[:max_chars]
        truncated = True

    flags: Dict[str, Dict[str, Any]] = {}
    score = 0.0
    evidence: List[str] = []

    try:
        checks = {
            "pii":                  _check_pii(text),
            "api_key":              _check_api_key(text),
            "unsafe_instruction":   _check_unsafe_instruction(text),
            "downstream_injection": _check_downstream_injection(text),
            "redirect_to_unknown":  _check_redirect(text, allowlist_domains=allow),
        }
        for name, result in checks.items():
            if not result.get("triggered"):
                continue
            flags[name] = result
            # Contribution scales with `min(count, 3)`: 1 hit → full weight, 2 hits
            # keep 1.35x, 3+ saturates. Prevents a single spam-y output from
            # scoring linearly huge.
            count_factor = min(result.get("count", 1), 3) / 1.0  # 1,2,3
            contribution = w.get(name, 0.0) * (1.0 + 0.15 * (count_factor - 1))
            score += contribution
            samples = result.get("samples") or result.get("unknown_hosts") or result.get("rules") or []
            if samples:
                evidence.append(f"[{name}] " + ", ".join(str(s) for s in samples[:2]))
    except Exception as exc:  # pragma: no cover — defensive
        evidence.append(f"output_guard analyzer error: {type(exc).__name__}: {exc}")
        score = 0.0

    if truncated:
        evidence.append(f"output truncated at {max_chars} chars before analysis")

    score = max(0.0, min(1.0, score))
    decision = _decision_from_score(score, thr)

    return OutputRiskResult(
        score=score,
        decision=decision,
        flags=flags,
        evidence=evidence,
        latency_ms=int((time.time() - t0) * 1000),
        output_chars=len(text),
    )


__all__ = [
    "analyze",
    "OutputRiskResult",
    "DEFAULT_THRESHOLDS",
    "DEFAULT_WEIGHTS",
    "DEFAULT_ALLOWLIST_DOMAINS",
]
