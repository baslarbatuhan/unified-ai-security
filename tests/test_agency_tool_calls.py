"""tests/test_agency_tool_calls.py
====================================
Hafta 12.3 — three new agency tool families exercised by the gateway's
ParameterValidator. Each tool maps to a distinct attack class:

  weather_forecast  → numeric-bound IDOR / param manipulation
  stock_quote       → regex shape + max_length + enumeration setup
  calc_evaluate     → code injection caught by allow-list regex

Tests bypass the LLM judge entirely — they live at the ParameterValidator
layer, which the gateway hits BEFORE invoking the actual tool. End-to-end
fusion behaviour is covered by the existing agency suite tests.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from output_agency_defense.parameter_validation import ParameterValidator
from fusion_gateway.engine import _register_gateway_demo_schemas


@pytest.fixture
def validator() -> ParameterValidator:
    v = ParameterValidator()
    _register_gateway_demo_schemas(v)
    return v


# ---------------------------------------------------------------------------
# 1) weather_forecast — float bounds + type coercion attacks
# ---------------------------------------------------------------------------
class TestWeatherForecast:
    def test_valid_coords_pass(self, validator) -> None:
        r = validator.validate("weather_forecast", {
            "latitude": 41.01, "longitude": 28.95, "current_weather": True,
        })
        assert r.is_valid, f"unexpected violations: {r.violations}"

    def test_latitude_above_max_blocks(self, validator) -> None:
        r = validator.validate("weather_forecast", {
            "latitude": 999.0, "longitude": 28.95,
        })
        assert not r.is_valid
        assert any("above maximum" in v for v in r.violations)

    def test_latitude_below_min_blocks(self, validator) -> None:
        r = validator.validate("weather_forecast", {
            "latitude": -1000.0, "longitude": 0.0,
        })
        assert not r.is_valid
        assert any("below minimum" in v for v in r.violations)

    def test_longitude_out_of_range_blocks(self, validator) -> None:
        r = validator.validate("weather_forecast", {
            "latitude": 0.0, "longitude": 999.0,
        })
        assert not r.is_valid

    def test_path_traversal_string_in_float_field_blocks(self, validator) -> None:
        """String in a float-typed slot is a type-coercion attempt."""
        r = validator.validate("weather_forecast", {
            "latitude": 41.01, "longitude": "../../etc/passwd",
        })
        assert not r.is_valid
        assert any("Wrong type" in v for v in r.violations)

    def test_missing_longitude_blocks(self, validator) -> None:
        r = validator.validate("weather_forecast", {"latitude": 41.01})
        assert not r.is_valid
        assert any("Missing required parameter" in v for v in r.violations)


# ---------------------------------------------------------------------------
# 2) stock_quote — regex + max_length + payload patterns
# ---------------------------------------------------------------------------
class TestStockQuote:
    def test_valid_aapl_passes(self, validator) -> None:
        r = validator.validate("stock_quote", {"symbol": "AAPL"})
        assert r.is_valid, r.violations

    def test_valid_dotted_symbol_passes(self, validator) -> None:
        r = validator.validate("stock_quote", {"symbol": "BRK.A"})
        assert r.is_valid, r.violations

    def test_sql_injection_pattern_blocked(self, validator) -> None:
        r = validator.validate("stock_quote", {
            "symbol": "AAPL'; DROP TABLE quotes;--",
        })
        assert not r.is_valid

    def test_path_traversal_blocked(self, validator) -> None:
        r = validator.validate("stock_quote", {"symbol": "../../etc/passwd"})
        assert not r.is_valid

    def test_oversized_symbol_blocked(self, validator) -> None:
        # max_length=10
        r = validator.validate("stock_quote", {"symbol": "A" * 18})
        assert not r.is_valid

    def test_lowercase_symbol_blocked_by_regex(self, validator) -> None:
        """Regex requires uppercase letters — lowercase fails the shape check."""
        r = validator.validate("stock_quote", {"symbol": "aapl"})
        assert not r.is_valid


# ---------------------------------------------------------------------------
# 3) calc_evaluate — allow-list regex catches code injection
# ---------------------------------------------------------------------------
class TestCalcEvaluate:
    def test_legit_arithmetic_passes(self, validator) -> None:
        r = validator.validate("calc_evaluate", {"expression": "1 + 1"})
        assert r.is_valid, r.violations

    def test_complex_arithmetic_passes(self, validator) -> None:
        r = validator.validate("calc_evaluate", {
            "expression": "(2 + 3) * 4 / (5 - 1)",
        })
        assert r.is_valid

    def test_dunder_import_blocked(self, validator) -> None:
        r = validator.validate("calc_evaluate", {
            "expression": "__import__('os').system('rm -rf /')",
        })
        assert not r.is_valid

    def test_open_blocked(self, validator) -> None:
        r = validator.validate("calc_evaluate", {
            "expression": "open('/etc/shadow').read()",
        })
        assert not r.is_valid

    def test_eval_blocked(self, validator) -> None:
        r = validator.validate("calc_evaluate", {"expression": "eval('1+1')"})
        assert not r.is_valid

    def test_subprocess_blocked(self, validator) -> None:
        r = validator.validate("calc_evaluate", {
            "expression": "subprocess.run(['sh','-c','id'])",
        })
        assert not r.is_valid

    def test_lambda_blocked(self, validator) -> None:
        r = validator.validate("calc_evaluate", {
            "expression": "(lambda x: x)(1)",
        })
        assert not r.is_valid

    def test_backticks_blocked(self, validator) -> None:
        r = validator.validate("calc_evaluate", {"expression": "`id`"})
        assert not r.is_valid

    def test_oversized_expression_blocked(self, validator) -> None:
        # max_length=200; build something longer.
        r = validator.validate("calc_evaluate", {
            "expression": "1+1" * 200,
        })
        assert not r.is_valid


# ---------------------------------------------------------------------------
# 4) Scenario file integrity — the new 20 entries load + reference real tools
# ---------------------------------------------------------------------------
class TestScenarioFile:
    """Defence against bit-rot: scenarios.json must stay loadable and
    every `tool` referenced must have a registered schema (otherwise the
    runner silently skips the test with `No schema registered`)."""

    _SCENARIOS_PATH = (
        Path(__file__).resolve().parents[1]
        / "datasets" / "output_agency_attacks" / "agency_attack_scenarios.json"
    )

    def test_file_parses(self) -> None:
        data = json.loads(self._SCENARIOS_PATH.read_text(encoding="utf-8"))
        assert "scenarios" in data
        # Hafta 12.3 expanded the file from 30 → 50.
        assert len(data["scenarios"]) >= 50

    def test_atk_031_to_050_present(self) -> None:
        data = json.loads(self._SCENARIOS_PATH.read_text(encoding="utf-8"))
        ids = {s["id"] for s in data["scenarios"]}
        for i in range(31, 51):
            atk = f"ATK-{i:03d}"
            assert atk in ids, f"missing scenario {atk}"

    def test_new_tools_have_schemas(self, validator) -> None:
        # Each of the three new tools must be reachable through the
        # validator — otherwise the runner can't enforce param checks.
        registered = set(validator._schemas.keys())
        for tool in ("weather_forecast", "stock_quote", "calc_evaluate"):
            assert tool in registered, f"{tool} schema not registered"

    def test_scenarios_reference_real_tools(self, validator) -> None:
        """Every non-empty tool name in scenarios.json should be
        registered, OR be deliberately invalid (e.g. ATK-019 uses
        `execute_shell` to test the unknown-tool path)."""
        data = json.loads(self._SCENARIOS_PATH.read_text(encoding="utf-8"))
        registered = set(validator._schemas.keys())
        deliberately_unknown = {
            # ATK-019/020/021 deliberately probe unknown tools.
            "execute_shell", "read_file", "reset_database", "",
            # ATK-011 (role_misuse) targets an admin-only tool — schema
            # absence is the point: gateway rejects on "no schema".
            "delete_user",
        }
        seen_tools = {s["tool"] for s in data["scenarios"]}
        unknown_but_not_deliberate = (
            seen_tools - registered - deliberately_unknown
        )
        assert not unknown_but_not_deliberate, (
            f"tools referenced but not registered: {unknown_but_not_deliberate}"
        )
