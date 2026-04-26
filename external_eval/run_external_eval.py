"""external_eval/run_external_eval.py
======================================
Send an attack suite through a chatbot target and record the target's
responses + gateway verdicts into telemetry + CSV.

Flow per case:
    1. build adapter for the target
    2. adapter.send(prompt)            ← external chatbot reply
    3. FusionEngine.analyze(prompt)    ← our gateway's own verdict (optional)
    4. emit RequestEvent + FusionDecisionEvent (Phase 0.1 telemetry)
    5. append a row to runs/external_eval_results.csv

Usage
-----
    # Quickest smoke test — uses the in-process mock target.
    python external_eval/run_external_eval.py --target mock_echo --suite prompt_injection --max-attacks 3

    # Full external run once the internal chatbot is live:
    python external_eval/run_external_eval.py --target internal_chatbot_api --suite all

Flags that matter
-----------------
    --gateway-analyze/--no-gateway-analyze  Run our own FusionEngine on each
                                            prompt.  Default ON for richer
                                            reports; turn off to benchmark
                                            raw target behaviour only.
    --target-has-tools                      Override detection — by default
                                            targets are assumed to NOT have
                                            tool calling.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_RUNS_DIR = _PROJECT_ROOT / "runs"

sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(_PROJECT_ROOT / ".env")
except Exception:
    pass

from schemas.target_schema import TargetConfig
from schemas.telemetry_schema import (
    RequestEvent,
    FusionDecisionEvent,
    ErrorEvent,
    emit_telemetry,
    new_run_id,
)
from external_eval.target_loader import get_target
from external_eval.adapter_factory import build_adapter
from external_eval.attack_suites import (
    AttackCase,
    load_suite,
    filter_for_target,
)


# ---------------------------------------------------------------------------
# Gateway wrapper — lazy so the runner still works on minimal installs
# (e.g. CI with no torch) by falling back to `--no-gateway-analyze`.
# ---------------------------------------------------------------------------
def _build_gateway():  # -> FusionEngine | None
    try:
        from fusion_gateway.engine import FusionEngine  # noqa: WPS433
        return FusionEngine()
    except Exception as exc:
        print(f"[warn] FusionEngine unavailable: {exc}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Row emitter
# ---------------------------------------------------------------------------
_CSV_FIELDS = [
    "run_id",
    "case_id",
    "suite",
    "category",
    "target_id",
    "expected_decision",
    "adapter_ok",
    "adapter_latency_ms",
    "adapter_error",
    "response_chars",
    "gateway_decision",
    "gateway_fused_score",
    "gateway_prompt_score",
    "gateway_rag_score",
    "gateway_agency_score",
    "gateway_latency_ms",
    "gateway_miss",              # 1 if the gateway let a block/sanitize case through.
    "evidence_top",              # first 120 chars of gateway evidence joined
]


def _classify_gateway_miss(expected: str, gateway_decision: Optional[str]) -> int:
    """Gateway-miss proxy: 1 when the expected decision is block/sanitize
    but the gateway returned `allow`. Gateway not run → abstain → 0.

    NB: This is a *protector* metric — whether our gateway caught the attempt.
    It does NOT measure whether the external chatbot actually complied with
    the attack. True end-to-end "attack success" requires content analysis
    on the chatbot's response (Phase 4 scope)."""
    if gateway_decision is None:
        return 0
    if expected in {"block", "sanitize"} and gateway_decision == "allow":
        return 1
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="External chatbot evaluation runner.")
    ap.add_argument("--target", required=True, help="Target id from external_eval/targets.yaml")
    ap.add_argument(
        "--suite",
        default="prompt_injection",
        help="Attack suite: prompt_injection | rag_poisoning | agency_social | all",
    )
    ap.add_argument("--max-attacks", type=int, default=0, help="0 = all")
    ap.add_argument(
        "--target-has-tools",
        action="store_true",
        help="Treat the target as tool-calling-capable (keeps agency_social cases).",
    )
    ap.add_argument(
        "--no-gateway-analyze",
        action="store_true",
        help="Skip FusionEngine analysis; only collect chatbot replies.",
    )
    ap.add_argument(
        "--output-csv",
        default=str(_RUNS_DIR / "external_eval_results.csv"),
        help="CSV output path. Appended to if it exists.",
    )
    ap.add_argument(
        "--run-id",
        default=None,
        help="Override run_id. Otherwise a fresh id is generated (Phase 0 format).",
    )
    ap.add_argument(
        "--config-yaml",
        default=None,
        help=(
            "Path to a config snapshot (e.g. runs/<run_id>/config_used.yaml) "
            "whose policy.fusion + modules sections become per-request overrides. "
            "Lets the dashboard's Run Test page reach the gateway without mutating it."
        ),
    )
    ap.add_argument("--verbose", action="store_true")
    return ap.parse_args(argv)


def _load_overrides_from_yaml(path: Optional[str]) -> Optional[Dict[str, Any]]:
    """Convert a snapshot YAML into the override dict the engine consumes.

    Returns None when path is missing/empty so callers can skip the override
    keyword without branching at the call site.
    """
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        print(f"[warn] --config-yaml {path} not found; running with defaults", file=sys.stderr)
        return None
    import yaml as _yaml  # local import — runner already imports yaml elsewhere
    cfg = _yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    fusion = ((cfg.get("policy") or {}).get("fusion") or {})
    modules = (cfg.get("modules") or {})
    out: Dict[str, Any] = {}
    if fusion.get("weights"):
        out["weights"] = dict(fusion["weights"])
    if fusion.get("thresholds"):
        out["thresholds"] = dict(fusion["thresholds"])
    if fusion.get("override"):
        out["override"] = dict(fusion["override"])
    if modules:
        flags: Dict[str, bool] = {}
        for name, val in modules.items():
            if isinstance(val, dict) and "enabled" in val:
                flags[name] = bool(val["enabled"])
        if flags:
            out["modules_enabled"] = flags
    return out or None


def run(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    target = get_target(args.target)
    if target is None:
        print(f"[error] unknown target {args.target!r}", file=sys.stderr)
        return 2
    if not target.enabled:
        print(f"[error] target {args.target!r} is disabled", file=sys.stderr)
        return 2

    cases_raw = load_suite(args.suite)
    cases = filter_for_target(cases_raw, target_has_tools=args.target_has_tools)
    if args.max_attacks and args.max_attacks > 0:
        cases = cases[: args.max_attacks]
    if not cases:
        print(
            f"[error] no attack cases after filtering (suite={args.suite!r}, "
            f"target_has_tools={args.target_has_tools})",
            file=sys.stderr,
        )
        return 2

    skipped = len(cases_raw) - len(filter_for_target(cases_raw, target_has_tools=args.target_has_tools))
    if skipped:
        print(
            f"[info] skipped {skipped} tool-requiring cases "
            f"(target_has_tools=False). Pass --target-has-tools to include."
        )

    run_id = args.run_id or new_run_id(prefix=f"ext_{args.target}")
    print(f"[run] id={run_id} target={args.target} suite={args.suite} n_cases={len(cases)}")

    gateway = None if args.no_gateway_analyze else _build_gateway()
    overrides = _load_overrides_from_yaml(args.config_yaml)
    if overrides:
        print(f"[run] applying overrides from {args.config_yaml}: {sorted(overrides.keys())}")

    adapter = build_adapter(target)
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_path.exists()

    summary = {
        "total": 0,
        "adapter_ok": 0,
        "gateway_block": 0,
        "gateway_sanitize": 0,
        "gateway_allow": 0,
        "gateway_miss": 0,
        "adapter_latency_sum_ms": 0,
        "gateway_latency_sum_ms": 0,
    }

    try:
        with output_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
            if write_header:
                writer.writeheader()

            for i, case in enumerate(cases, start=1):
                t_case = time.time()
                summary["total"] += 1

                # 1. Telemetry: request event.
                emit_telemetry(
                    RequestEvent(
                        run_id=run_id,
                        target_id=target.id,
                        attack_id=case.id,
                        prompt=case.prompt,
                        prompt_char_count=len(case.prompt),
                    )
                )

                # 2. Send to target.
                adapter_resp = adapter.send(case.prompt)
                if adapter_resp.ok:
                    summary["adapter_ok"] += 1
                summary["adapter_latency_sum_ms"] += adapter_resp.latency_ms

                # 3. Gateway analysis (optional).
                gw_decision: Optional[str] = None
                gw_fused = gw_prompt = gw_rag = gw_agency = 0.0
                gw_latency_ms = 0
                gw_evidence: List[str] = []
                if gateway is not None:
                    t_gw = time.time()
                    try:
                        gw_result = gateway.analyze(user_input=case.prompt, overrides=overrides)
                        gw_latency_ms = int((time.time() - t_gw) * 1000)
                        gw_decision = _attr(gw_result, "final_decision") or _attr(gw_result, "decision")
                        gw_fused = float(
                            _attr(gw_result, "fused_risk", None)
                            or _attr(gw_result, "fused_risk_score", 0.0)
                            or 0.0
                        )
                        # FusionEngineResponse.module_risks is a list of dicts,
                        # each carrying {module, risk_score, evidence, ...}.
                        module_risks = _attr(gw_result, "module_risks", []) or []
                        for mr in module_risks:
                            name = (mr or {}).get("module", "")
                            score = float((mr or {}).get("risk_score", 0.0) or 0.0)
                            if name == "prompt_guard":
                                gw_prompt = score
                            elif name == "rag_guard":
                                gw_rag = score
                            elif name == "output_agency":
                                gw_agency = score
                            ev = (mr or {}).get("evidence") or []
                            if isinstance(ev, list):
                                gw_evidence.extend(str(x) for x in ev)
                    except Exception as exc:
                        emit_telemetry(
                            ErrorEvent(
                                run_id=run_id,
                                target_id=target.id,
                                attack_id=case.id,
                                where="fusion_engine.analyze_full",
                                error_type=type(exc).__name__,
                                message=str(exc)[:300],
                            )
                        )

                    if gw_decision:
                        emit_telemetry(
                            FusionDecisionEvent(
                                run_id=run_id,
                                target_id=target.id,
                                attack_id=case.id,
                                fused_risk_score=gw_fused,
                                decision=gw_decision,  # type: ignore[arg-type]
                                prompt_score=gw_prompt,
                                rag_score=gw_rag,
                                agency_score=gw_agency,
                                evidence=gw_evidence[:6],
                                latency_ms_total=gw_latency_ms,
                            )
                        )
                        key = f"gateway_{gw_decision}"
                        if key in summary:
                            summary[key] += 1
                        summary["gateway_latency_sum_ms"] += gw_latency_ms

                gateway_miss = _classify_gateway_miss(case.expected, gw_decision)
                summary["gateway_miss"] += gateway_miss

                writer.writerow(
                    {
                        "run_id": run_id,
                        "case_id": case.id,
                        "suite": case.suite,
                        "category": case.category,
                        "target_id": target.id,
                        "expected_decision": case.expected,
                        "adapter_ok": int(adapter_resp.ok),
                        "adapter_latency_ms": adapter_resp.latency_ms,
                        "adapter_error": (adapter_resp.error_message or "")[:200],
                        "response_chars": len(adapter_resp.text or ""),
                        "gateway_decision": gw_decision or "",
                        "gateway_fused_score": round(gw_fused, 4),
                        "gateway_prompt_score": round(gw_prompt, 4),
                        "gateway_rag_score": round(gw_rag, 4),
                        "gateway_agency_score": round(gw_agency, 4),
                        "gateway_latency_ms": gw_latency_ms,
                        "gateway_miss": gateway_miss,
                        "evidence_top": " | ".join(gw_evidence)[:120],
                    }
                )

                if args.verbose:
                    print(
                        f"  [{i}/{len(cases)}] {case.id} "
                        f"adapter_ok={adapter_resp.ok} "
                        f"gw={gw_decision or '-'} "
                        f"fused={gw_fused:.2f} "
                        f"dt={int((time.time()-t_case)*1000)}ms"
                    )
                else:
                    # One-line progress every 10 cases keeps CI logs readable.
                    if i % 10 == 0 or i == len(cases):
                        print(
                            f"  progress {i}/{len(cases)} "
                            f"adapter_ok={summary['adapter_ok']} "
                            f"gateway_miss={summary['gateway_miss']}"
                        )
    finally:
        adapter.close()

    # Summary line.
    n = max(summary["total"], 1)
    print(
        "\n[done] "
        f"total={summary['total']} "
        f"adapter_ok={summary['adapter_ok']} "
        f"gw_block={summary['gateway_block']} "
        f"gw_sanitize={summary['gateway_sanitize']} "
        f"gw_allow={summary['gateway_allow']} "
        f"gateway_miss={summary['gateway_miss']} "
        f"avg_adapter_ms={summary['adapter_latency_sum_ms']//n} "
        f"avg_gateway_ms={summary['gateway_latency_sum_ms']//n}"
    )
    print(f"[saved] {output_path}")
    return 0


def _attr(obj: Any, name: str, default: Any = None) -> Any:
    """Read attribute from a pydantic model or a dict-like return value."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


if __name__ == "__main__":
    raise SystemExit(run())
