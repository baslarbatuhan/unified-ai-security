"""rag_guard/chunk_router.py
==============================
Decide what to do with each chunk *before* paying for an LLM judge call.

The RAG pipeline already chunks long docs (`LLMJudge.analyze_chunked`) and
supports a simple `embedding_gate_threshold`. This router generalises that
gate into an explicit three-way routing decision so:

  * cheap triage happens in one place,
  * the policy is auditable (every chunk produces a `RouteDecision` with
    the reason it was routed where it was),
  * metrics_writer can log the route for every chunk without leaking
    embedding-internal details into the call site.

Routes
------
SKIP        — embedding_score below the lower gate; chunk contributes 0 to
              judge_score and no LLM call is made.
FAST_JUDGE  — ambiguous band; one judge call, result used as-is.
DEEP_JUDGE  — embedding score already looks bad; one judge call plus an
              override-multiplier so a judge "abstain" (near 0.5) can't
              wash out a strong embedding signal.

Thresholds come from the pipeline config; callers that don't want routing
can keep the existing behaviour by passing `lower=0.0, upper=1.0` (every
chunk is then FAST_JUDGE).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class Route(str, Enum):
    SKIP = "skip"
    FAST_JUDGE = "fast_judge"
    DEEP_JUDGE = "deep_judge"


@dataclass
class RouteDecision:
    idx: int
    route: Route
    embedding_score: float
    reason: str
    # Populated by the caller after running the judge (or skipping) so the
    # same struct flows into metrics_writer without a separate join.
    judge_score: Optional[float] = None
    used_override: bool = False
    text_preview: str = ""


@dataclass
class RouterConfig:
    """Thresholds bounding the three routes on the embedding axis."""
    skip_below: float = 0.15       # below → Route.SKIP
    deep_above: float = 0.55       # above → Route.DEEP_JUDGE
    override_multiplier: float = 0.85

    def __post_init__(self) -> None:
        if not (0.0 <= self.skip_below <= self.deep_above <= 1.0):
            raise ValueError(
                f"RouterConfig invariants violated: "
                f"0 ≤ skip_below({self.skip_below}) ≤ deep_above({self.deep_above}) ≤ 1"
            )


class ChunkRouter:
    """Stateless policy object — one instance per pipeline is fine."""

    def __init__(self, config: Optional[RouterConfig] = None):
        self.config = config or RouterConfig()

    def decide(self, idx: int, embedding_score: float, text_preview: str = "") -> RouteDecision:
        """Classify a single chunk by its embedding-only poison score."""
        s = float(max(0.0, min(1.0, embedding_score)))
        cfg = self.config
        if s < cfg.skip_below:
            return RouteDecision(
                idx=idx, route=Route.SKIP, embedding_score=s,
                reason=f"embedding {s:.3f} < skip_below {cfg.skip_below:.3f}",
                text_preview=text_preview,
            )
        if s > cfg.deep_above:
            return RouteDecision(
                idx=idx, route=Route.DEEP_JUDGE, embedding_score=s,
                reason=f"embedding {s:.3f} > deep_above {cfg.deep_above:.3f}",
                text_preview=text_preview,
            )
        return RouteDecision(
            idx=idx, route=Route.FAST_JUDGE, embedding_score=s,
            reason=f"embedding {s:.3f} in ambiguous band",
            text_preview=text_preview,
        )

    def decide_batch(
        self,
        embedding_scores: List[float],
        chunks: Optional[List[str]] = None,
    ) -> List[RouteDecision]:
        chunks = chunks or []
        return [
            self.decide(i, s, text_preview=(chunks[i][:120] if i < len(chunks) else ""))
            for i, s in enumerate(embedding_scores)
        ]

    # ------------------------------------------------------------------
    # Post-judge finalisation
    # ------------------------------------------------------------------
    def finalize(self, decision: RouteDecision, judge_score: Optional[float]) -> float:
        """Apply the override rule on DEEP_JUDGE.

        Returns the *effective* judge score the caller should use when
        fusing embedding + judge. Mutates `decision` to record the outcome
        so downstream loggers see a consistent view.
        """
        if decision.route == Route.SKIP:
            decision.judge_score = 0.0
            return 0.0

        js = float(judge_score if judge_score is not None else 0.0)
        decision.judge_score = js
        if decision.route == Route.DEEP_JUDGE:
            # If the judge abstains (returns something near its mid-band),
            # the embedding evidence should still dominate — use the max
            # of `js` and `embedding * override_multiplier`.
            overridden = max(js, decision.embedding_score * self.config.override_multiplier)
            if overridden > js:
                decision.used_override = True
            return overridden
        return js

    # ------------------------------------------------------------------
    # Dashboard-friendly summary
    # ------------------------------------------------------------------
    @staticmethod
    def summarize(decisions: List[RouteDecision]) -> Dict[str, int]:
        counts = {"skip": 0, "fast_judge": 0, "deep_judge": 0, "overrides": 0}
        for d in decisions:
            counts[d.route.value] += 1
            if d.used_override:
                counts["overrides"] += 1
        return counts


__all__ = ["ChunkRouter", "RouterConfig", "Route", "RouteDecision"]
