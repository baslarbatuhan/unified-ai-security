"""tests/test_dashboard_components.py
=======================================
Hafta 13 — pure-fn tests for the dashboard helpers. We don't try to
render Streamlit widgets here; Streamlit lives behind a session context
that's awkward to stand up in pytest. Instead we test:

  * `components` pure helpers: decision_color / decision_icon /
    format_registry_label / risk_level_from_score
  * `recommendations` policy: composite_security_score formula edges
    + each rule's trigger conditions
"""
from __future__ import annotations

import pytest

from dashboard.lib.components import (
    DECISION_COLORS,
    DECISION_ICONS,
    SEVERITY_ICONS,
    decision_color,
    decision_icon,
    format_registry_label,
    risk_level_from_score,
)
from dashboard.lib.recommendations import (
    HIGH_FALSE_POSITIVE_RATE,
    HIGH_FLAG_BAND_RATIO,
    HIGH_MISS_RATE,
    LATENCY_BUDGET_HOT_PCT,
    LOW_AGENCY_ENGAGEMENT,
    LOW_ROUTING_SAVINGS_PCT,
    MODERATE_MISS_RATE,
    _load_score_config,
    composite_security_score,
    generate_recommendations,
)


# ---------------------------------------------------------------------------
# components.decision_color / decision_icon
# ---------------------------------------------------------------------------
class TestDecisionAccessors:
    def test_known_decisions_map_to_distinct_colors(self) -> None:
        seen = {decision_color(d) for d in ("allow", "sanitize", "flag", "block")}
        assert len(seen) == 4, "every decision must have a distinct colour"

    def test_unknown_decision_returns_fallback_color(self) -> None:
        c = decision_color("???")
        assert c.startswith("#"), "fallback should still be a hex colour"
        assert c not in DECISION_COLORS.values(), "fallback != any known colour"

    def test_case_insensitive(self) -> None:
        assert decision_color("BLOCK") == decision_color("block")
        assert decision_icon("Allow") == decision_icon("allow")

    def test_severity_icons_present_for_all_levels(self) -> None:
        for level in ("info", "warn", "critical"):
            assert level in SEVERITY_ICONS


# ---------------------------------------------------------------------------
# components.format_registry_label
# ---------------------------------------------------------------------------
class TestFormatRegistryLabel:
    def test_includes_target_suite_n_runid(self) -> None:
        entry = {
            "run_id": "ext_mock_echo_20260506T010203Z",
            "target_id": "mock_echo",
            "suite": "rag_poisoning",
            "n_rows": 5,
            "ended_at": "2026-05-06T01:02:03+00:00",
        }
        lbl = format_registry_label(entry)
        assert "mock_echo" in lbl
        assert "rag_poisoning" in lbl
        assert "n=5" in lbl
        assert "ext_mock_echo_20260506T010203Z" in lbl
        assert "2026-05-06 01:02:03" in lbl  # T → space

    def test_handles_missing_fields_gracefully(self) -> None:
        lbl = format_registry_label({})
        assert "?" in lbl
        assert "n=0" in lbl

    def test_falls_back_to_started_at_when_no_ended_at(self) -> None:
        entry = {
            "run_id": "x", "target_id": "t", "suite": "s",
            "n_rows": 1, "started_at": "2026-05-06T05:00:00+00:00",
        }
        lbl = format_registry_label(entry)
        assert "2026-05-06 05:00:00" in lbl


# ---------------------------------------------------------------------------
# components.risk_level_from_score
# ---------------------------------------------------------------------------
class TestRiskLevel:
    def test_band_boundaries(self) -> None:
        assert risk_level_from_score(100) == "LOW"
        assert risk_level_from_score(85) == "LOW"
        assert risk_level_from_score(84.9) == "MEDIUM"
        assert risk_level_from_score(70) == "MEDIUM"
        assert risk_level_from_score(69.9) == "HIGH"
        assert risk_level_from_score(50) == "HIGH"
        assert risk_level_from_score(49.9) == "CRITICAL"
        assert risk_level_from_score(0) == "CRITICAL"

    def test_garbage_input_lands_in_critical(self) -> None:
        # `None` → 0.0 → CRITICAL (worst-case is the safe default).
        assert risk_level_from_score(None) == "CRITICAL"


# ---------------------------------------------------------------------------
# recommendations.composite_security_score formula
# ---------------------------------------------------------------------------
class TestCompositeScore:
    def test_perfect_inputs_gives_100(self) -> None:
        s = composite_security_score({
            "miss_rate": 0.0, "latency_breach_rate": 0.0,
            "precision": 1.0, "recall": 1.0,
        })
        assert s == 100.0

    def test_miss_50pct_halves_score(self) -> None:
        s = composite_security_score({
            "miss_rate": 0.5, "latency_breach_rate": 0.0,
            "precision": 1.0, "recall": 1.0,
        })
        assert s == 50.0

    def test_latency_breach_pulls_down(self) -> None:
        clean = composite_security_score({
            "miss_rate": 0.0, "latency_breach_rate": 0.0,
            "precision": 1.0, "recall": 1.0,
        })
        breach = composite_security_score({
            "miss_rate": 0.0, "latency_breach_rate": 0.25,
            "precision": 1.0, "recall": 1.0,
        })
        assert breach < clean
        assert breach == 75.0  # 100 * 1 * 0.75 * 1

    def test_missing_precision_recall_treated_as_unknown_not_punished(self) -> None:
        """When no eval has run, we shouldn't punish a fresh install."""
        s = composite_security_score({
            "miss_rate": 0.0, "latency_breach_rate": 0.0,
            # both omitted → 0/0 → treated as 1.0 (neutral).
        })
        assert s == 100.0

    def test_zero_precision_recall_still_neutral_when_both_zero(self) -> None:
        """Distinguish 'no data' from 'data exists but is bad'."""
        s = composite_security_score({"precision": 0.0, "recall": 0.0})
        assert s == 100.0  # both 0 → "unknown" → no penalty

    def test_partial_quality_pulls_score_down(self) -> None:
        s = composite_security_score({
            "miss_rate": 0.0, "latency_breach_rate": 0.0,
            "precision": 0.8, "recall": 0.5,
        })
        # 100 * 1 * 1 * (0.8 * 0.5) = 40
        assert s == 40.0

    def test_garbage_inputs_default_safely(self) -> None:
        s = composite_security_score({"miss_rate": "not-a-number"})
        assert 0.0 <= s <= 100.0

    def test_clipped_to_unit_range(self) -> None:
        """Anything > 1 must be clamped down so a bogus metric can't
        push the score above 100 (or below 0)."""
        s = composite_security_score({
            "miss_rate": -5.0, "latency_breach_rate": -5.0,
            "precision": 5.0, "recall": 5.0,
        })
        assert s == 100.0


# ---------------------------------------------------------------------------
# Hafta 15: yaml-driven score config — weights, neutrality flag, max_score
# ---------------------------------------------------------------------------
class TestScoreConfigYaml:
    def _cfg(self, **overrides):
        base = {
            "weights": {"miss": 1.0, "latency": 1.0, "quality": 1.0},
            "treat_zero_precision_recall_as_neutral": True,
            "max_score": 100.0,
        }
        base.update(overrides)
        return base

    def test_weight_zero_disables_factor(self) -> None:
        """weight=0 → factor**0 = 1.0 → that axis is neutral. Setting
        latency weight to 0 makes a 50% breach rate not affect the score."""
        cfg = self._cfg(weights={"miss": 1.0, "latency": 0.0, "quality": 1.0})
        s = composite_security_score({
            "miss_rate": 0.0, "latency_breach_rate": 0.50,
            "precision": 1.0, "recall": 1.0,
        }, config=cfg)
        assert s == 100.0

    def test_weight_amplifies_penalty(self) -> None:
        """Doubling the miss exponent should make a 10% miss hurt more."""
        cfg_neutral = self._cfg()
        cfg_strict = self._cfg(weights={"miss": 2.0, "latency": 1.0, "quality": 1.0})
        m = {"miss_rate": 0.10, "latency_breach_rate": 0.0,
             "precision": 1.0, "recall": 1.0}
        s_normal = composite_security_score(m, config=cfg_neutral)
        s_strict = composite_security_score(m, config=cfg_strict)
        assert s_strict < s_normal

    def test_max_score_override(self) -> None:
        """max_score=10 should clamp output accordingly."""
        cfg = self._cfg(max_score=10.0)
        s = composite_security_score({
            "miss_rate": 0.0, "latency_breach_rate": 0.0,
            "precision": 1.0, "recall": 1.0,
        }, config=cfg)
        assert s == 10.0

    def test_disable_zero_neutral_treats_unknown_as_zero(self) -> None:
        """When the flag is off, precision=recall=0 → score=0 (no
        'unknown ≠ bad' grace)."""
        cfg = self._cfg(treat_zero_precision_recall_as_neutral=False)
        s = composite_security_score({
            "miss_rate": 0.0, "latency_breach_rate": 0.0,
            "precision": 0.0, "recall": 0.0,
        }, config=cfg)
        assert s == 0.0


# ---------------------------------------------------------------------------
# Hafta 15: config loader robustness
# ---------------------------------------------------------------------------
class TestScoreConfigLoader:
    def test_missing_yaml_returns_defaults(self, tmp_path) -> None:
        bogus = tmp_path / "nope.yaml"
        cfg = _load_score_config(path=bogus)
        assert cfg["weights"] == {"miss": 1.0, "latency": 1.0, "quality": 1.0}
        assert cfg["max_score"] == 100.0
        assert cfg["treat_zero_precision_recall_as_neutral"] is True

    def test_partial_yaml_per_key_fallback(self, tmp_path) -> None:
        """yaml with only `weights.miss` overrides that one; other
        weights stay at default."""
        p = tmp_path / "partial.yaml"
        p.write_text("weights:\n  miss: 2.5\n", encoding="utf-8")
        cfg = _load_score_config(path=p)
        assert cfg["weights"]["miss"] == 2.5
        assert cfg["weights"]["latency"] == 1.0   # default
        assert cfg["weights"]["quality"] == 1.0   # default

    def test_garbage_weight_ignored(self, tmp_path) -> None:
        p = tmp_path / "bad.yaml"
        p.write_text("weights:\n  miss: not-a-number\n  latency: -1.0\n",
                     encoding="utf-8")
        cfg = _load_score_config(path=p)
        # Non-numeric → default; negative → default (must be >=0).
        assert cfg["weights"]["miss"] == 1.0
        assert cfg["weights"]["latency"] == 1.0

    def test_shipped_yaml_loads_to_defaults(self) -> None:
        """The committed yaml at `configs/security_score_weights.yaml`
        should preserve the Hafta 13 hardcoded formula behaviour
        (defaults across the board)."""
        cfg = _load_score_config()  # uses default path
        assert cfg["weights"]["miss"] == 1.0
        assert cfg["weights"]["latency"] == 1.0
        assert cfg["weights"]["quality"] == 1.0
        assert cfg["max_score"] == 100.0


# ---------------------------------------------------------------------------
# recommendations.generate_recommendations — each rule's trigger
# ---------------------------------------------------------------------------
class TestRecommendationRules:
    def test_high_miss_rate_produces_critical_reco(self) -> None:
        recos = generate_recommendations({
            "miss_rate": HIGH_MISS_RATE + 0.05,  # well above 10%
        })
        assert any(r["severity"] == "critical" for r in recos)
        assert any("recall" in r["text"].lower() for r in recos)

    def test_moderate_miss_rate_produces_warn(self) -> None:
        recos = generate_recommendations({
            "miss_rate": MODERATE_MISS_RATE + 0.01,
        })
        assert any(r["severity"] == "warn" for r in recos)
        assert not any(r["severity"] == "critical" for r in recos)

    def test_clean_miss_rate_no_miss_reco(self) -> None:
        recos = generate_recommendations({"miss_rate": 0.0})
        assert not any("recall" in r["text"].lower() for r in recos)

    def test_latency_above_budget_warns(self) -> None:
        recos = generate_recommendations({
            "module_latency_ms": {"rag_guard": 18_000},
            "module_budget_ms": {"rag_guard": 20_000},  # 90% of budget
        })
        assert any("rag_guard" in r["text"] and "budget" in r["text"] for r in recos)

    def test_latency_well_under_budget_no_reco(self) -> None:
        recos = generate_recommendations({
            "module_latency_ms": {"rag_guard": 5_000},
            "module_budget_ms": {"rag_guard": 20_000},  # 25% of budget
        })
        assert not any("budget" in r["text"] for r in recos)

    def test_low_routing_savings_emits_info(self) -> None:
        recos = generate_recommendations({
            "routing_savings_pct": LOW_ROUTING_SAVINGS_PCT - 5,
        })
        assert any("routing" in r["text"].lower() for r in recos)

    def test_agency_engagement_zero_warns(self) -> None:
        recos = generate_recommendations({
            "avg_agency_score": 0.0,
            "n_agency_cases": 20,
        })
        assert any("agency" in r["text"].lower() for r in recos)

    def test_agency_warn_skipped_when_no_cases(self) -> None:
        """Don't nag about agency when no agency cases ran."""
        recos = generate_recommendations({
            "avg_agency_score": 0.0,
            "n_agency_cases": 0,
        })
        assert not any("agency" in r["text"].lower() for r in recos)

    def test_flag_band_dominates_emits_info(self) -> None:
        recos = generate_recommendations({
            "flag_band_count": 18,
            "block_band_count": 2,  # 90% flag-tier
        })
        assert any("flag" in r["text"].lower() for r in recos)

    def test_flag_band_below_threshold_no_reco(self) -> None:
        recos = generate_recommendations({
            "flag_band_count": 4,
            "block_band_count": 16,
        })
        assert not any("flag" in r["text"].lower() for r in recos)

    def test_high_false_positive_warns(self) -> None:
        recos = generate_recommendations({
            "fp_rate": HIGH_FALSE_POSITIVE_RATE + 0.02,
        })
        assert any("false-positive" in r["text"].lower() for r in recos)

    def test_severity_ordering(self) -> None:
        recos = generate_recommendations({
            "miss_rate": HIGH_MISS_RATE + 0.05,      # critical
            "fp_rate": HIGH_FALSE_POSITIVE_RATE + 0.02,  # warn
            "routing_savings_pct": 10.0,             # info
        })
        # critical → warn → info ordering enforced.
        severities = [r["severity"] for r in recos]
        # No info before warn, no warn before critical.
        for i, s in enumerate(severities):
            for ahead in severities[:i]:
                rank = {"critical": 0, "warn": 1, "info": 2}
                assert rank[ahead] <= rank[s]

    def test_no_recos_for_healthy_system(self) -> None:
        recos = generate_recommendations({
            "miss_rate": 0.01,
            "fp_rate": 0.01,
            "module_latency_ms": {"rag_guard": 5_000},
            "module_budget_ms": {"rag_guard": 20_000},
            "routing_savings_pct": 60.0,
            "avg_agency_score": 0.5,
            "n_agency_cases": 10,
            "flag_band_count": 3,
            "block_band_count": 10,
        })
        assert recos == []

    def test_limit_caps_output(self) -> None:
        recos = generate_recommendations({
            "miss_rate": HIGH_MISS_RATE + 0.05,
            "fp_rate": HIGH_FALSE_POSITIVE_RATE + 0.02,
            "routing_savings_pct": 10.0,
        }, limit=1)
        assert len(recos) == 1


# ---------------------------------------------------------------------------
# Robustness — recommendations must not crash on garbage
# ---------------------------------------------------------------------------
class TestRobustness:
    def test_empty_metrics_returns_empty(self) -> None:
        assert generate_recommendations({}) == []

    def test_none_inputs_dont_crash(self) -> None:
        recos = generate_recommendations({
            "miss_rate": None,
            "fp_rate": None,
            "module_latency_ms": None,
            "module_budget_ms": None,
        })
        # Should not raise; may or may not emit recos.
        assert isinstance(recos, list)
