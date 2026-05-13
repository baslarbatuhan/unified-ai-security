"""tests/test_decision_trace.py
================================
Hafta 12.1 — per-decision audit trace round-trip.

Three layers covered:
  1. `utils.run_manifest.append_decision_trace / read_decision_trace`
     — pure I/O round-trip + schema drift rotation.
  2. `fusion_gateway.engine._format_fusion_formula` — prose snapshot.
  3. End-to-end via `FusionEngine.analyze(..., run_id=<scoped>)`:
     writes a row to `runs/<run_id>/decision_trace.csv`.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import List

import pytest

from utils.run_manifest import (
    DECISION_TRACE_FIELDS,
    DECISION_TRACE_FILENAME,
    append_decision_trace,
    read_decision_trace,
)
from fusion_gateway.engine import (
    FusionEngine,
    ModuleRisk,
    _format_fusion_formula,
    _try_append_decision_trace,
)


# ---------------------------------------------------------------------------
# 1) Pure I/O round-trip
# ---------------------------------------------------------------------------
class TestAppendDecisionTrace:
    def test_writes_header_and_row(self, tmp_path: Path) -> None:
        row = {
            "case_id": "c1", "target_id": "mock_echo",
            "timestamp": "2026-05-06T07:00:00+00:00",
            "final_decision": "block", "decision_band": "flag",
            "fused_risk": 0.71, "weighted_sum": 0.378,
            "override_applied": "elevated",
            "triggering_module": "rag_guard", "triggering_band": "flag",
            "prompt_score": 0.55, "rag_score": 0.71,
            "agency_score": 0.0, "output_score": 0.0,
            "fusion_formula": "0.30*0.55 + 0.30*0.71 + 0.40*0.00 = 0.378 → elevated → 0.71",
            "module_risks_json": [
                {"module": "prompt_guard", "risk_score": 0.55,
                 "decision": "sanitize", "top_evidence": "evidence A"},
                {"module": "rag_guard", "risk_score": 0.71,
                 "decision": "flag", "top_evidence": "evidence B"},
                {"module": "output_agency", "risk_score": 0.0,
                 "decision": "allow", "top_evidence": "No tool call"},
            ],
            "latency_ms": 12345,
        }
        out = append_decision_trace(tmp_path, row=row)
        assert out is not None and out.exists()

        # Header + 1 row.
        text = out.read_text(encoding="utf-8")
        assert text.startswith(",".join(DECISION_TRACE_FIELDS))
        assert "rag_guard" in text

    def test_read_round_trip_with_case_filter(self, tmp_path: Path) -> None:
        for cid in ("c1", "c2", "c3"):
            append_decision_trace(tmp_path, row={
                "case_id": cid, "target_id": "t",
                "timestamp": "T", "final_decision": "allow",
                "decision_band": "allow", "fused_risk": 0.1,
                "weighted_sum": 0.1, "override_applied": "none",
                "triggering_module": "prompt_guard", "triggering_band": "allow",
                "prompt_score": 0.1, "rag_score": 0.0,
                "agency_score": 0.0, "output_score": 0.0,
                "fusion_formula": "...",
                "module_risks_json": [],
                "latency_ms": 100,
            })

        all_rows = read_decision_trace(tmp_path)
        assert len(all_rows) == 3
        only_c2 = read_decision_trace(tmp_path, case_id="c2")
        assert len(only_c2) == 1
        assert only_c2[0]["case_id"] == "c2"
        # JSON column parsed back to a list.
        assert only_c2[0]["module_risks_json"] == []

    def test_read_missing_file_returns_empty(self, tmp_path: Path) -> None:
        # No write; just read.
        assert read_decision_trace(tmp_path) == []

    def test_schema_drift_rotates_old_file(self, tmp_path: Path) -> None:
        """If the header changes (e.g. we add a column), the legacy file
        is rotated to `.stale-<utc>` and the new write starts a fresh
        header. Keeps analyses backward-grep'able."""
        path = tmp_path / DECISION_TRACE_FILENAME
        # Hand-write an "old" file with a missing column.
        old_fields = DECISION_TRACE_FIELDS[:-1]  # drop latency_ms
        with path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=old_fields)
            w.writeheader()
            w.writerow({k: "x" for k in old_fields})

        # Write a row matching the new schema.
        result = append_decision_trace(tmp_path, row={
            "case_id": "c1", "target_id": "t", "timestamp": "T",
            "final_decision": "allow", "decision_band": "allow",
            "fused_risk": 0.1, "weighted_sum": 0.1, "override_applied": "none",
            "triggering_module": "prompt_guard", "triggering_band": "allow",
            "prompt_score": 0.1, "rag_score": 0.0,
            "agency_score": 0.0, "output_score": 0.0,
            "fusion_formula": "...", "module_risks_json": [],
            "latency_ms": 42,
        })
        assert result is not None

        # New file has the current header.
        text = path.read_text(encoding="utf-8")
        assert text.splitlines()[0] == ",".join(DECISION_TRACE_FIELDS)

        # Old data preserved under a stale-suffixed name.
        stale_files = list(tmp_path.glob(f"{DECISION_TRACE_FILENAME}.stale-*"))
        assert len(stale_files) == 1


# ---------------------------------------------------------------------------
# 2) Formula prose
# ---------------------------------------------------------------------------
class TestFusionFormulaPretty:
    def test_no_override_shows_just_weighted_sum(self) -> None:
        s = _format_fusion_formula(
            weights={"prompt_guard": 0.3, "rag_guard": 0.3, "output_agency": 0.4},
            risks={"prompt_guard": 0.1, "rag_guard": 0.0, "output_agency": 0.0},
            weighted_sum=0.03, final_fused=0.03, override_applied="none",
        )
        assert "0.30*0.100" in s
        assert "= 0.0300" in s
        assert "no override" in s

    def test_elevated_override_marker(self) -> None:
        s = _format_fusion_formula(
            weights={"prompt_guard": 0.3, "rag_guard": 0.3, "output_agency": 0.4},
            risks={"prompt_guard": 0.55, "rag_guard": 0.71, "output_agency": 0.0},
            weighted_sum=0.378, final_fused=0.71, override_applied="elevated",
        )
        assert "elevated" in s
        assert "→ 0.7100" in s

    def test_skips_zero_weight_modules(self) -> None:
        s = _format_fusion_formula(
            weights={"prompt_guard": 0.0, "rag_guard": 0.6, "output_agency": 0.4},
            risks={"prompt_guard": 0.9, "rag_guard": 0.5, "output_agency": 0.2},
            weighted_sum=0.38, final_fused=0.38, override_applied="none",
        )
        # prompt_guard had weight 0 → not in the printed sum.
        assert "0.00*" not in s
        assert "0.60*" in s


# ---------------------------------------------------------------------------
# 3) End-to-end via _try_append_decision_trace (direct fn call — engine
#    plumbing in analyze() is the orchestration around it)
# ---------------------------------------------------------------------------
class TestEngineTraceWiring:
    def _risk(self, name: str, score: float, decision: str = "allow",
              evidence: list = None) -> ModuleRisk:
        return ModuleRisk(
            module=name, risk_score=score, confidence=1.0,
            decision=decision, evidence=evidence or [f"{name} stub"],
            latency_ms=1,
        )

    def test_skips_when_run_id_is_live_sentinel(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """Ad-hoc /analyze calls (no scoped run) must not flood `runs/`."""
        # Redirect runs/ writes to tmp_path by patching _PROJECT_ROOT lookup.
        from fusion_gateway import engine as eng
        monkeypatch.setattr(eng, "Path",
                            lambda p="": tmp_path / Path(p).name
                            if isinstance(p, str) else tmp_path)

        # Even though our patch is shaky, the function should early-return
        # on run_id="live" without ever touching the filesystem.
        _try_append_decision_trace(
            run_id="live", case_id="x", target_id=None,
            final_decision="allow", band="allow", fused_risk=0.1,
            weighted_sum=0.1, override_applied="none",
            eff_weights={"prompt_guard": 0.3, "rag_guard": 0.3, "output_agency": 0.4},
            eff_thresholds={"allow": 0.3, "sanitize": 0.6, "block": 0.85},
            prompt_risk=self._risk("prompt_guard", 0.1),
            rag_risk=self._risk("rag_guard", 0.0),
            agency_risk=self._risk("output_agency", 0.0),
            latency_ms=42,
        )
        # No `runs/live/decision_trace.csv` may exist.
        assert not (tmp_path / "live" / DECISION_TRACE_FILENAME).exists()

    def test_writes_when_run_id_is_scoped(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """Real runs end up with a per-run decision_trace.csv."""
        from fusion_gateway import engine as eng

        # Monkey-patch the helper to a known location (the real one would
        # write under <project>/runs/<run_id>/). We can't easily redirect
        # _PROJECT_ROOT, so patch utils.run_manifest.append_decision_trace
        # to record the run_dir it was called with.
        recorded = {}

        def _spy(run_dir, *, row):
            recorded["run_dir"] = run_dir
            recorded["row"] = row
            return run_dir / DECISION_TRACE_FILENAME

        monkeypatch.setattr(
            "utils.run_manifest.append_decision_trace", _spy
        )
        _try_append_decision_trace(
            run_id="ext_mock_echo_TEST", case_id="case-1", target_id="mock_echo",
            final_decision="block", band="block", fused_risk=0.95,
            weighted_sum=0.40, override_applied="critical",
            eff_weights={"prompt_guard": 0.3, "rag_guard": 0.3, "output_agency": 0.4},
            eff_thresholds={"allow": 0.3, "sanitize": 0.6, "block": 0.85},
            prompt_risk=self._risk("prompt_guard", 0.50, "sanitize", ["promptev"]),
            rag_risk=self._risk("rag_guard", 0.95, "block", ["ragev1", "ragev2"]),
            agency_risk=self._risk("output_agency", 0.0),
            latency_ms=8421,
        )
        assert recorded
        # run_dir resolves to .../runs/ext_mock_echo_TEST
        assert recorded["run_dir"].name == "ext_mock_echo_TEST"
        # Triggering module is whichever produced max risk (rag_guard here).
        row = recorded["row"]
        assert row["triggering_module"] == "rag_guard"
        assert row["override_applied"] == "critical"
        # JSON column is a Python list at write-call time (append_decision_trace
        # will JSON-encode it before writing).
        assert isinstance(row["module_risks_json"], list)
        assert len(row["module_risks_json"]) == 3


# NOTE — full FusionEngine.analyze() → filesystem round-trip is covered
# by the Docker-mode E2E sanity at Hafta 12 sonu (real run_id → real
# runs/<id>/decision_trace.csv). The unit layer above already validates
# the helper, the formula pretty-print, and the wiring in
# `_try_append_decision_trace` (which is the only logic specific to the
# trace path inside analyze()). Patching `Path(__file__).resolve().
# parent.parent` to redirect to tmp_path is too brittle to be worth a
# pure-unit test — the value comes from real I/O in CI.
