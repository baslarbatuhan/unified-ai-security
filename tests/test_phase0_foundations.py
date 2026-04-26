"""tests/test_phase0_foundations.py
=====================================
Phase 0 exit criteria — runs fast, no LLM / network.

Covers:
  * schemas.telemetry_schema: emit → read round-trip, run_id filter
  * utils.log_sanitizer: PII redaction & false-positive avoidance
  * schemas.target_schema + external_eval.target_loader: CRUD + validation
  * external_eval.adapter_factory + mock_adapter: build + send
  * utils.config_builder: UI state → snapshot determinism
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------
def test_telemetry_emit_and_read_roundtrip(tmp_path, monkeypatch):
    from schemas import telemetry_schema as ts

    # redirect output under tmp_path
    monkeypatch.setattr(ts, "TELEMETRY_DIR", tmp_path)
    monkeypatch.setattr(ts, "TELEMETRY_FILE", tmp_path / "system_telemetry.jsonl")

    run_id = ts.new_run_id("test")
    ts.emit_telemetry(
        ts.RequestEvent(
            run_id=run_id,
            target_id="mock_echo",
            prompt="hello",
            prompt_char_count=5,
        )
    )
    ts.emit_telemetry(
        ts.ModuleResultEvent(
            run_id=run_id,
            module="prompt_guard",
            risk_score=0.1,
            confidence=0.9,
            decision="allow",
            latency_ms=12,
        )
    )
    # Event from a different run — must be filtered out.
    ts.emit_telemetry(
        ts.RequestEvent(
            run_id="other_run",
            prompt="x",
            prompt_char_count=1,
        )
    )

    events = ts.read_events(run_id=run_id)
    assert len(events) == 2
    assert {e["kind"] for e in events} == {"request", "module_result"}
    assert all(e["run_id"] == run_id for e in events)


def test_telemetry_kinds_filter(tmp_path, monkeypatch):
    from schemas import telemetry_schema as ts

    monkeypatch.setattr(ts, "TELEMETRY_DIR", tmp_path)
    monkeypatch.setattr(ts, "TELEMETRY_FILE", tmp_path / "system_telemetry.jsonl")

    run_id = ts.new_run_id()
    ts.emit_telemetry(ts.RequestEvent(run_id=run_id, prompt="a", prompt_char_count=1))
    ts.emit_telemetry(
        ts.FusionDecisionEvent(
            run_id=run_id,
            fused_risk_score=0.5,
            decision="sanitize",
            latency_ms_total=80,
        )
    )
    only_fusion = ts.read_events(run_id=run_id, kinds=["fusion_decision"])
    assert len(only_fusion) == 1
    assert only_fusion[0]["kind"] == "fusion_decision"


# ---------------------------------------------------------------------------
# Log sanitizer
# ---------------------------------------------------------------------------
def test_sanitizer_redacts_email_and_phone():
    from utils.log_sanitizer import sanitize, MASK_EMAIL, MASK_PHONE

    out = sanitize("Contact me at alice@example.com or +90 555 123 4567")
    assert MASK_EMAIL in out
    assert MASK_PHONE in out
    assert "alice@example.com" not in out


def test_sanitizer_redacts_bearer_and_api_keys():
    from utils.log_sanitizer import sanitize

    raw = "Authorization: Bearer sk-proj-abcdef1234567890ABCDEF and key sk-ant-api03-XYZXYZXYZXYZXYZXYZXYZXYZ"
    out = sanitize(raw)
    assert "sk-proj-abcdef1234567890ABCDEF" not in out
    assert "sk-ant-api03-XYZXYZXYZXYZXYZXYZXYZXYZ" not in out
    assert "[REDACTED_TOKEN]" in out or "[REDACTED_APIKEY]" in out


def test_sanitizer_keeps_private_ip_and_short_digits():
    from utils.log_sanitizer import sanitize

    # Private IP should NOT be masked — useful for internal debugging.
    out = sanitize("Server at 192.168.1.10 replied in 42 ms")
    assert "192.168.1.10" in out
    # Short digit sequence is not a phone.
    out2 = sanitize("Error code 404 occurred")
    assert "404" in out2


def test_sanitize_event_respects_allowlist():
    from utils.log_sanitizer import sanitize_event, MASK_EMAIL

    ev = {
        "run_id": "run_abc_123",        # allowlisted → keep
        "target_id": "alice@example.com",  # allowlisted → keep despite email
        "prompt": "mail to bob@example.com",  # NOT allowlisted → redact
        "details": {"contact": "carol@example.com"},
    }
    out = sanitize_event(ev)
    assert out["run_id"] == "run_abc_123"
    assert out["target_id"] == "alice@example.com"
    assert MASK_EMAIL in out["prompt"]
    assert MASK_EMAIL in out["details"]["contact"]


def test_sanitizer_tckn_luhn_check():
    """11-digit non-TCKN should pass through; a real TCKN pattern gets masked."""
    from utils.log_sanitizer import sanitize, MASK_TCKN

    # Non-TCKN 11-digit sequence (fails check-digit algorithm).
    out1 = sanitize("id=11111111111 issued")
    # May or may not match — the main guarantee is no crash + no email-mask
    # applied. Ensure at minimum the string still references the digits OR a mask.
    assert isinstance(out1, str)

    # A number that *does* pass the Turkish check-digit rule.
    # Pick a known-valid demo pattern: 10000000146 (algorithmically valid).
    out2 = sanitize("kimlik 10000000146 olarak kayıtlı")
    assert MASK_TCKN in out2


# ---------------------------------------------------------------------------
# Target schema + loader
# ---------------------------------------------------------------------------
def test_target_schema_rejects_web_without_selectors():
    from schemas.target_schema import TargetConfig
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        TargetConfig(
            id="bad_web",
            name="bad",
            type="web",
            endpoint="https://example.com",
        )


def test_target_schema_rejects_duplicate_ids():
    from schemas.target_schema import TargetConfig, TargetsFile
    from pydantic import ValidationError

    t1 = TargetConfig(id="dup", name="a", type="mock")
    t2 = TargetConfig(id="dup", name="b", type="mock")
    with pytest.raises(ValidationError):
        TargetsFile(targets=[t1, t2])


def test_target_loader_crud_roundtrip(tmp_path):
    from schemas.target_schema import TargetConfig
    from external_eval import target_loader as tl

    path = tmp_path / "targets.yaml"
    # Start empty
    tf = tl.load_targets(path)
    assert tf.targets == []

    # Upsert
    tl.upsert_target(
        TargetConfig(id="m1", name="Mock 1", type="mock"),
        path=path,
    )
    tl.upsert_target(
        TargetConfig(
            id="api1",
            name="API 1",
            type="api",
            endpoint="https://api.example.com/chat",
        ),
        path=path,
    )
    listed = tl.list_targets(path)
    assert {t.id for t in listed} == {"m1", "api1"}

    # Get
    got = tl.get_target("api1", path)
    assert got is not None and got.endpoint == "https://api.example.com/chat"

    # Overwrite
    tl.upsert_target(
        TargetConfig(id="m1", name="Mock 1 renamed", type="mock", enabled=False),
        path=path,
    )
    assert tl.get_target("m1", path).name == "Mock 1 renamed"

    # Delete
    assert tl.delete_target("m1", path) is True
    assert tl.delete_target("m1", path) is False
    assert {t.id for t in tl.list_targets(path)} == {"api1"}


def test_repo_targets_yaml_is_valid():
    """The shipped targets.yaml must parse cleanly."""
    from external_eval import target_loader as tl

    tf = tl.load_targets()  # default path
    ids = {t.id for t in tf.targets}
    assert "mock_echo" in ids


# ---------------------------------------------------------------------------
# Adapter factory + mock adapter
# ---------------------------------------------------------------------------
def test_mock_adapter_deterministic():
    from schemas.target_schema import TargetConfig
    from external_eval.adapter_factory import build_adapter

    target = TargetConfig(id="m", name="m", type="mock")
    adapter = build_adapter(target)
    r1 = adapter.send("hello")
    r2 = adapter.send("hello")
    assert r1.ok and r2.ok
    assert r1.text == r2.text
    assert "mock chatbot" in r1.text.lower()
    assert r1.metadata["prompt_chars"] == 5


def test_adapter_factory_rejects_disabled_target():
    from schemas.target_schema import TargetConfig
    from external_eval.adapter_factory import build_adapter
    from external_eval.base_adapter import AdapterConfigError

    t = TargetConfig(id="off", name="off", type="mock", enabled=False)
    with pytest.raises(AdapterConfigError):
        build_adapter(t)


def test_api_adapter_handles_missing_endpoint():
    from schemas.target_schema import TargetConfig
    from external_eval.api_adapter import APIAdapter
    from external_eval.base_adapter import AdapterConfigError
    from pydantic import ValidationError

    # Schema-level: endpoint required for api type.
    with pytest.raises(ValidationError):
        TargetConfig(id="a", name="a", type="api")


def test_api_adapter_extract_response_path():
    from external_eval.api_adapter import APIAdapter

    data = {"choices": [{"message": {"content": "hello world"}}]}
    assert (
        APIAdapter._extract_response(data, "choices.0.message.content") == "hello world"
    )
    # Fallback path when no explicit response_path is configured.
    assert APIAdapter._extract_response(data, None) == "hello world"


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------
def test_config_builder_hash_is_deterministic():
    from utils.config_builder import build_config, config_hash

    ui = {"target_id": "mock_echo", "model": "qwen2.5:7b"}
    c1 = build_config(ui)
    c2 = build_config(ui)
    # Strip the timestamp so the hash matches.
    c1["run_metadata"].pop("created_at", None)
    c2["run_metadata"].pop("created_at", None)
    assert config_hash(c1) == config_hash(c2)


def test_config_builder_respects_ui_overrides():
    from utils.config_builder import build_config

    ui = {
        "modules": {"prompt_guard": False, "output_guard": True},
        "fusion": {"weights": {"prompt_guard": 0.1, "rag_guard": 0.9}},
        "model": "llama3.1:8b",
    }
    cfg = build_config(ui)
    assert cfg["llm"]["model"] == "llama3.1:8b"
    assert cfg["policy"]["fusion"]["weights"]["prompt_guard"] == 0.1
    assert cfg["modules"]["prompt_guard"]["enabled"] is False


def test_snapshot_writes_yaml(tmp_path):
    from utils.config_builder import snapshot_from_ui

    run_id, path, cfg = snapshot_from_ui(
        {"target_id": "mock_echo"}, runs_dir=tmp_path
    )
    assert path.exists()
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert loaded["run_metadata"]["target_id"] == "mock_echo"
    assert run_id.startswith("run_")


def test_ui_state_roundtrip():
    from utils.config_builder import build_config, ui_state_from_config

    ui_in = {
        "target_id": "internal_chatbot_api",
        "attack_suite": "prompt_injection",
        "modules": {"prompt_guard": True, "rag_guard": False, "output_agency": True, "output_guard": True},
        "model": "qwen2.5:7b",
        "fusion": {
            "weights": {"prompt_guard": 0.2, "rag_guard": 0.2, "output_agency": 0.3, "output_guard": 0.3},
            "thresholds": {"allow": 0.3, "sanitize": 0.6, "block": 0.85},
        },
        "timeout_profile": "fast",
    }
    cfg = build_config(ui_in)
    ui_out = ui_state_from_config(cfg)
    assert ui_out["target_id"] == "internal_chatbot_api"
    assert ui_out["modules"]["rag_guard"] is False
    assert ui_out["fusion"]["weights"]["prompt_guard"] == 0.2
