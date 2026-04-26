"""datasets/dataset_loaders.py
================================
Loader + metric helpers for curated regression datasets.

`prompt_regression_set.json` is the current target. Each case has an
explicit `label` (benign/attack) and `expected_decision` so we can compute
precision/recall/F1 without joining external ground truth.

This module is intentionally IO-only: no pipeline imports, no heavy deps,
so unit tests can exercise the schema validation in ~0ms.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROMPT_REGRESSION_SET = _PROJECT_ROOT / "datasets" / "prompt_regression_set.json"

_VALID_LABELS = {"benign", "attack"}
_VALID_DECISIONS = {"allow", "sanitize", "flag", "block"}


@dataclass
class RegressionCase:
    id: str
    prompt: str
    label: str                    # "benign" | "attack"
    expected_decision: str        # one of _VALID_DECISIONS
    category: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_attack(self) -> bool:
        return self.label == "attack"

    @property
    def is_benign(self) -> bool:
        return self.label == "benign"


def load_prompt_regression_set(path: Optional[Path] = None) -> List[RegressionCase]:
    """Load + validate the curated regression set.

    Raises ValueError on any schema violation so pytest/CI fails loudly when
    someone edits the JSON by hand and gets it wrong.
    """
    p = Path(path) if path else DEFAULT_PROMPT_REGRESSION_SET
    data = json.loads(p.read_text(encoding="utf-8"))
    cases_raw = data.get("cases")
    if not isinstance(cases_raw, list) or not cases_raw:
        raise ValueError(f"{p}: 'cases' must be a non-empty list")

    seen_ids: set = set()
    cases: List[RegressionCase] = []
    for i, c in enumerate(cases_raw):
        for key in ("id", "prompt", "label", "expected_decision"):
            if key not in c:
                raise ValueError(f"{p}: case #{i} missing required field '{key}'")
        cid = str(c["id"])
        if cid in seen_ids:
            raise ValueError(f"{p}: duplicate case id {cid!r}")
        seen_ids.add(cid)

        label = str(c["label"])
        if label not in _VALID_LABELS:
            raise ValueError(
                f"{p}: case {cid!r} has invalid label {label!r} (want one of {sorted(_VALID_LABELS)})"
            )
        decision = str(c["expected_decision"])
        if decision not in _VALID_DECISIONS:
            raise ValueError(
                f"{p}: case {cid!r} expected_decision={decision!r} not in {sorted(_VALID_DECISIONS)}"
            )
        cases.append(RegressionCase(
            id=cid,
            prompt=str(c["prompt"]),
            label=label,
            expected_decision=decision,
            category=str(c.get("category", "")),
            metadata={k: v for k, v in c.items()
                      if k not in ("id", "prompt", "label", "expected_decision", "category")},
        ))
    return cases


# ---------------------------------------------------------------------------
# Metric helpers — "observed" is the module/gateway decision for each case.
# ---------------------------------------------------------------------------
def _is_positive(decision: str, *, strict: bool) -> bool:
    """Treat a decision as 'flagged' (positive class).

    strict=True  → only 'block' counts (user-visible denial)
    strict=False → any of sanitize/flag/block counts (module-level action)
    """
    if strict:
        return decision == "block"
    return decision != "allow"


def compute_confusion(
    cases: Iterable[RegressionCase],
    observed: Dict[str, str],
    *,
    strict: bool = True,
) -> Dict[str, int]:
    """Build the binary confusion matrix (attack = positive class)."""
    tp = fp = tn = fn = 0
    for c in cases:
        if c.id not in observed:
            raise KeyError(f"missing observed decision for case {c.id!r}")
        predicted_positive = _is_positive(observed[c.id], strict=strict)
        actual_positive = c.is_attack
        if actual_positive and predicted_positive:
            tp += 1
        elif actual_positive and not predicted_positive:
            fn += 1
        elif not actual_positive and predicted_positive:
            fp += 1
        else:
            tn += 1
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def compute_metrics(confusion: Dict[str, int]) -> Dict[str, float]:
    tp, fp, tn, fn = (confusion[k] for k in ("tp", "fp", "tn", "fn"))
    total = tp + fp + tn + fn
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / total if total else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(accuracy, 4),
        "fpr": round(fpr, 4),
        "total": float(total),
    }


def per_category_breakdown(
    cases: Iterable[RegressionCase],
    observed: Dict[str, str],
    *,
    strict: bool = True,
) -> Dict[str, Dict[str, float]]:
    """Group cases by category and compute metrics for each group."""
    buckets: Dict[str, List[RegressionCase]] = {}
    for c in cases:
        buckets.setdefault(c.category or "uncategorized", []).append(c)
    out: Dict[str, Dict[str, float]] = {}
    for cat, group in buckets.items():
        cm = compute_confusion(group, observed, strict=strict)
        metrics = compute_metrics(cm)
        metrics.update({f"n_{k}": float(v) for k, v in cm.items()})
        out[cat] = metrics
    return out


__all__ = [
    "RegressionCase",
    "DEFAULT_PROMPT_REGRESSION_SET",
    "load_prompt_regression_set",
    "compute_confusion",
    "compute_metrics",
    "per_category_breakdown",
]
