"""Shared fusion policy thresholds (single source for fusion + RAG module decisions)."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

_DEFAULT = {"allow": 0.30, "sanitize": 0.60, "block": 0.85}


def load_fusion_thresholds() -> Dict[str, float]:
    """Load policy.fusion.thresholds from secure_balanced.yaml with defaults."""
    config_path = Path(__file__).resolve().parent / "secure_balanced.yaml"
    out = dict(_DEFAULT)
    if not config_path.exists():
        return out
    try:
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        ft = (cfg.get("policy") or {}).get("fusion", {}).get("thresholds") or {}
        for k in _DEFAULT:
            if k in ft:
                out[k] = float(ft[k])
    except Exception:
        pass
    return out
