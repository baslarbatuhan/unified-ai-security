"""utils/config_builder.py
===========================
UI state → reproducible gateway config snapshot.

Purpose
-------
Her test çalıştırmasında hangi ayarların kullanıldığını kaydetmek.
Dashboard kullanıcı arayüzünden gelen seçimler bir dict olarak buraya
verilir; çıktı `runs/<run_id>/config_used.yaml` dosyasıdır.  Aynı UI
state'i iki kez göndermek aynı `run_id`'yi üretir (hash kararlı).

Şema
----
UI state dict alanları (hepsi opsiyonel, default'lar altta):

    {
        "target_id": "internal_chatbot_api",
        # one of: prompt_injection | rag_poisoning | agency_social | all | single
        "attack_suite": "prompt_injection",
        "modules": {
            "prompt_guard":  True,
            "rag_guard":     True,
            "output_agency": True,
            "output_guard":  True,
        },
        "model": "qwen2.5:7b",
        "fusion": {
            "weights": {"prompt_guard": 0.25, "rag_guard": 0.25, "output_agency": 0.25, "output_guard": 0.25},
            "thresholds": {"allow": 0.30, "sanitize": 0.60, "block": 0.85},
        },
        "timeout_profile": "standard",  # lookup key into configs/timeout_config.yaml
        "notes": "optional free text",
    }

Çıktı şeması `configs/secure_balanced.yaml` ile uyumlu: engine YAML
loader'ı (fusion_gateway/engine.py) bu dosyayı da yükleyebilir.
Farklılıkları minimumda tutuyoruz.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASE_CONFIG = _PROJECT_ROOT / "configs" / "secure_balanced.yaml"
DEFAULT_RUNS_DIR = _PROJECT_ROOT / "runs"


# ---------------------------------------------------------------------------
# Defaults — mirror configs/secure_balanced.yaml so a blank UI state still
# produces a working config.
# ---------------------------------------------------------------------------
_DEFAULT_UI_STATE: Dict[str, Any] = {
    "target_id": "mock_echo",
    # Must be a valid suite name from external_eval/attack_suites.SUITE_LOADERS
    # (or "all" / "single") — `"default"` is not a real loader and would fail
    # at run time if it ever reached `load_suite()`.
    "attack_suite": "prompt_injection",
    "modules": {
        "prompt_guard": True,
        "rag_guard": True,
        "output_agency": True,
        "output_guard": True,
    },
    "model": "qwen2.5:7b",
    "fusion": {
        "weights": {
            "prompt_guard": 0.30,
            "rag_guard": 0.30,
            "output_agency": 0.40,
            # output_guard is additive on top; keep 0.0 until Phase 1B-β lands.
            "output_guard": 0.0,
        },
        "thresholds": {"allow": 0.30, "sanitize": 0.60, "block": 0.85},
    },
    "timeout_profile": "standard",
    "notes": "",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def normalize_ui_state(ui_state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge user-provided UI state over the defaults, with shallow+nested
    dict merging for `modules` and `fusion`.  The result is *always* valid
    for `build_config`."""
    out: Dict[str, Any] = deepcopy(_DEFAULT_UI_STATE)
    if not ui_state:
        return out

    for key, val in ui_state.items():
        if key in {"modules", "fusion"} and isinstance(val, dict):
            existing = out.get(key, {})
            if not isinstance(existing, dict):
                existing = {}
            merged = deepcopy(existing)
            for k2, v2 in val.items():
                if isinstance(v2, dict) and isinstance(merged.get(k2), dict):
                    sub = deepcopy(merged[k2])
                    sub.update(v2)
                    merged[k2] = sub
                else:
                    merged[k2] = v2
            out[key] = merged
        else:
            out[key] = val
    return out


def _load_base_config(path: Optional[Path] = None) -> Dict[str, Any]:
    path = path or DEFAULT_BASE_CONFIG
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _stable_json(obj: Any) -> str:
    """Canonical JSON string: sorted keys, no trailing whitespace — for hashing."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def config_hash(config: Dict[str, Any]) -> str:
    """SHA-256 of the canonicalized config. First 12 hex chars used for run_id."""
    return hashlib.sha256(_stable_json(config).encode("utf-8")).hexdigest()


def make_run_id(config: Dict[str, Any], *, prefix: str = "run") -> str:
    """`<prefix>_YYYYmmddHHMMSS_<hash12>` — sortable + reproducibility tied to config."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    return f"{prefix}_{ts}_{config_hash(config)[:12]}"


def build_config(
    ui_state: Optional[Dict[str, Any]] = None,
    *,
    base_config_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Produce the gateway config dict that mirrors the UI selections.

    Starts from `configs/secure_balanced.yaml` and overlays UI-driven changes.
    Disabled modules get `enabled: false`; fusion weights / thresholds are
    replaced wholesale.
    """
    ui = normalize_ui_state(ui_state)
    base = _load_base_config(base_config_path)

    cfg: Dict[str, Any] = deepcopy(base) if base else {}
    cfg.setdefault("llm", {})
    cfg["llm"]["model"] = ui["model"]

    cfg.setdefault("policy", {}).setdefault("fusion", {})
    fusion = cfg["policy"]["fusion"]
    fusion["type"] = fusion.get("type", "weighted_sum")
    fusion["weights"] = deepcopy(ui["fusion"]["weights"])
    fusion["thresholds"] = deepcopy(ui["fusion"]["thresholds"])

    modules = cfg.setdefault("modules", {})
    for mod_name, enabled in ui["modules"].items():
        mod_cfg = modules.setdefault(mod_name, {})
        if isinstance(mod_cfg, dict):
            mod_cfg["enabled"] = bool(enabled)

    # UI-supplied metadata lives under `run_metadata` so it's preserved in the
    # snapshot without contaminating the gateway's live config consumers.
    cfg["run_metadata"] = {
        "target_id": ui["target_id"],
        "attack_suite": ui["attack_suite"],
        "timeout_profile": ui["timeout_profile"],
        "notes": ui.get("notes", ""),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    return cfg


def write_snapshot(
    config: Dict[str, Any],
    *,
    run_id: Optional[str] = None,
    runs_dir: Optional[Path] = None,
) -> Tuple[str, Path]:
    """Write `runs/<run_id>/config_used.yaml`. Returns `(run_id, path)`.

    If `run_id` is None it is derived from the config hash; repeated calls
    with an identical config write to the same file (idempotent).
    """
    runs_dir = runs_dir or DEFAULT_RUNS_DIR
    rid = run_id or make_run_id(config)
    out_dir = runs_dir / rid
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "config_used.yaml"
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)
    return rid, out_path


def snapshot_from_ui(
    ui_state: Optional[Dict[str, Any]] = None,
    *,
    base_config_path: Optional[Path] = None,
    runs_dir: Optional[Path] = None,
) -> Tuple[str, Path, Dict[str, Any]]:
    """End-to-end helper: UI state → snapshot on disk.

    Returns `(run_id, snapshot_path, config_dict)`.  The dashboard's "Start"
    button should call this exactly once per run and hand `run_id` off to the
    gateway so every telemetry event carries it.
    """
    cfg = build_config(ui_state, base_config_path=base_config_path)
    rid, path = write_snapshot(cfg, runs_dir=runs_dir)
    return rid, path, cfg


# ---------------------------------------------------------------------------
# Inverse: YAML → UI state. Lets the dashboard re-hydrate a prior run.
# ---------------------------------------------------------------------------
def ui_state_from_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    modules_cfg = cfg.get("modules", {}) or {}
    module_flags = {
        name: bool(mod.get("enabled", True)) if isinstance(mod, dict) else bool(mod)
        for name, mod in modules_cfg.items()
    }
    fusion_cfg = (cfg.get("policy") or {}).get("fusion") or {}
    meta = cfg.get("run_metadata") or {}
    return {
        "target_id": meta.get("target_id", _DEFAULT_UI_STATE["target_id"]),
        "attack_suite": meta.get("attack_suite", _DEFAULT_UI_STATE["attack_suite"]),
        "modules": module_flags or deepcopy(_DEFAULT_UI_STATE["modules"]),
        "model": (cfg.get("llm") or {}).get("model", _DEFAULT_UI_STATE["model"]),
        "fusion": {
            "weights": fusion_cfg.get("weights", _DEFAULT_UI_STATE["fusion"]["weights"]),
            "thresholds": fusion_cfg.get("thresholds", _DEFAULT_UI_STATE["fusion"]["thresholds"]),
        },
        "timeout_profile": meta.get("timeout_profile", _DEFAULT_UI_STATE["timeout_profile"]),
        "notes": meta.get("notes", ""),
    }


__all__ = [
    "normalize_ui_state",
    "build_config",
    "config_hash",
    "make_run_id",
    "write_snapshot",
    "snapshot_from_ui",
    "ui_state_from_config",
    "DEFAULT_BASE_CONFIG",
    "DEFAULT_RUNS_DIR",
]
