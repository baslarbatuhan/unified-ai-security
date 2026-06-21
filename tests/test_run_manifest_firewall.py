"""tests/test_run_manifest_firewall.py
=======================================
Sprint 11 — the run registry/manifest must persist the firewall mode so the
dashboard run pickers can tag firewall runs without opening each manifest.
"""

from __future__ import annotations

from utils.run_manifest import (
    append_registry_entry,
    read_registry,
    write_manifest,
    read_manifest,
)


def test_registry_entry_persists_mode(tmp_path):
    append_registry_entry(
        tmp_path,
        run_id="r1", target_id="mock_echo", suite="prompt_injection",
        started_at="2026-06-09T00:00:00+00:00", ended_at="2026-06-09T00:01:00+00:00",
        exit_code=0, n_cases=3, n_rows=3, relative_path="runs/r1",
        mode="firewall",
    )
    entries = read_registry(tmp_path)
    assert entries[0]["mode"] == "firewall"


def test_registry_entry_mode_optional(tmp_path):
    # Legacy callers that omit mode must still produce a valid entry.
    append_registry_entry(
        tmp_path,
        run_id="r2", target_id="mock_echo", suite="prompt_injection",
        started_at="2026-06-09T00:00:00+00:00", ended_at="2026-06-09T00:01:00+00:00",
        exit_code=0, n_cases=1, n_rows=1, relative_path="runs/r2",
    )
    assert read_registry(tmp_path)[0]["mode"] is None


def test_manifest_extra_carries_firewall_fields(tmp_path):
    run_dir = tmp_path / "r3"
    write_manifest(
        run_dir,
        run_id="r3", target_id="mock_echo", suite="prompt_injection",
        started_at=None, ended_at=None, exit_code=0, n_cases=2, n_rows=2,
        extra={"mode": "firewall", "firewall_blocked": 2},
    )
    m = read_manifest(run_dir)
    assert m["extra"]["mode"] == "firewall"
    assert m["extra"]["firewall_blocked"] == 2
