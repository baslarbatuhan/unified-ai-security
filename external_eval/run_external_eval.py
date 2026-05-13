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
from datetime import datetime, timezone
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
#
# We wire SecurityGateway (not FusionEngine directly) so every call goes
# through the same telemetry path the HTTP /analyze endpoint uses:
#   * ModuleResultEvent × N  → Live monitor "Per-module latency" graph
#   * FusionDecisionEvent     → recent-runs feed (with target_id + attack_id)
# Without this, suite runs only emitted FusionDecisionEvent (manually below)
# and the per-module latency chart was always empty during a suite.
# ---------------------------------------------------------------------------
def _build_gateway():  # -> SecurityGateway | None
    try:
        from api.security_gateway import SecurityGateway  # noqa: WPS433
        return SecurityGateway()
    except Exception as exc:
        print(f"[warn] SecurityGateway unavailable: {exc}", file=sys.stderr)
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
    "gateway_decision_band",      # flag/block split for audit trail; None on legacy rows
    "gateway_fused_score",
    "gateway_prompt_score",
    "gateway_rag_score",
    "gateway_agency_score",
    "gateway_latency_ms",
    "gateway_miss",              # 1 if the gateway let a block/sanitize case through.
    "evidence_top",              # first 120 chars of gateway evidence joined
    # Hafta 14: tools_local target metadata. Empty on chatbot runs.
    "tool_executed",             # 1 if gateway allowed and the tool actually ran
    "tool_latency_ms",
    "tool_response_chars",
    "tool_response_preview",     # first 200 chars of JSON response
    "tool_error",                # any tool-level error message
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
    started_at_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"[run] id={run_id} target={args.target} suite={args.suite} n_cases={len(cases)}")

    gateway = None if args.no_gateway_analyze else _build_gateway()
    overrides = _load_overrides_from_yaml(args.config_yaml)
    if overrides:
        print(f"[run] applying overrides from {args.config_yaml}: {sorted(overrides.keys())}")

    adapter = build_adapter(target)
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Rotate on schema drift: if the existing header doesn't match the
    # current `_CSV_FIELDS` (e.g. we just added gateway_decision_band), move
    # the legacy file aside and start fresh. Mirrors the live writers'
    # `.stale-<utc>` rotation pattern so analysts can still grep history.
    write_header = not output_path.exists()
    if output_path.exists():
        try:
            with output_path.open("r", encoding="utf-8") as _f:
                first = _f.readline().strip()
            existing_cols = first.split(",") if first else []
            if existing_cols != _CSV_FIELDS:
                stale = output_path.with_suffix(
                    output_path.suffix + f".stale-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
                )
                output_path.rename(stale)
                print(f"[run] schema drift on {output_path.name}; rotated to {stale.name}")
                write_header = True
        except OSError:
            # Best-effort: fall through to append; DictWriter will still work,
            # readers may need a manual fix-up. Rare.
            pass

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
    # Mirror every row into a per-run buffer so we can write
    # `runs/<run_id>/results.csv` + manifest at end-of-run without
    # re-reading the aggregate file. The legacy aggregate keeps flowing
    # in parallel — nothing here changes that contract.
    run_rows: List[Dict[str, Any]] = []

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
                gw_decision_band: Optional[str] = None
                gw_fused = gw_prompt = gw_rag = gw_agency = 0.0
                gw_latency_ms = 0
                gw_evidence: List[str] = []
                if gateway is not None:
                    t_gw = time.time()
                    try:
                        # rag_poisoning cases pack the doc into `case.prompt`
                        # for the adapter, but the gateway scores rag_guard
                        # only when retrieved_context is passed separately.
                        # AttackCase.gateway_* fields carry that split (None
                        # for prompt_injection / agency_social → defaults).
                        # Build AnalyzeRequest for SecurityGateway. The
                        # gateway's `analyze()` emits ModuleResultEvent × N
                        # plus a FusionDecisionEvent, all tagged with
                        # run_id/target_id/case_id we set below — so the
                        # manual FusionDecisionEvent emission further down
                        # is dropped (would have duplicated).
                        from schemas.risk_schema import (
                            AnalyzeRequest,
                            ConfigOverrides,
                            SessionContext,
                            ToolRequest,
                        )
                        gw_user_input = case.gateway_user_input or case.prompt
                        # Session context: agency cases carry user_id/role
                        # in metadata; default to anonymous/basic otherwise.
                        session_kwargs: Dict[str, Any] = {}
                        if case.gateway_user_id:
                            session_kwargs["user_id"] = case.gateway_user_id
                        if case.gateway_role:
                            session_kwargs["role"] = case.gateway_role
                        req_kwargs: Dict[str, Any] = {
                            "prompt": gw_user_input,
                            "session_context": SessionContext(**session_kwargs),
                            "run_id": run_id,
                            "case_id": case.id,
                            "target_id": target.id,
                        }
                        if case.gateway_retrieved_context:
                            req_kwargs["context"] = case.gateway_retrieved_context
                        # Structured tool call for agency_social cases.
                        # Without this the gateway's agency module
                        # short-circuits with "No tool call" → 0.0 score.
                        if case.gateway_tool_call:
                            req_kwargs["tool_request"] = ToolRequest(
                                tool=case.gateway_tool_call.get("tool", ""),
                                params=case.gateway_tool_call.get("args") or {},
                            )
                        # Hafta 14: for tools_local targets, disable
                        # prompt_guard. The agency_social template
                        # ("You are assisting user X. Invoke the Y
                        # tool...") semantically resembles jailbreak
                        # signatures and triggers a 0.65 FP score on
                        # every case. tools_local's threat model is
                        # tool_call manipulation — prompt content is
                        # not the attack surface, so muting prompt_guard
                        # produces honest results.
                        merged_overrides = dict(overrides or {})
                        if target.type == "tools_local":
                            modules_en = dict(merged_overrides.get("modules_enabled") or {})
                            modules_en.setdefault("prompt_guard", False)
                            merged_overrides["modules_enabled"] = modules_en
                        if merged_overrides:
                            try:
                                req_kwargs["config_overrides"] = (
                                    ConfigOverrides.model_validate(merged_overrides)
                                )
                            except Exception:
                                # Validation should not break the run; just
                                # skip override propagation if shape is off.
                                pass
                        gw_result = gateway.analyze(AnalyzeRequest(**req_kwargs))
                        gw_latency_ms = int((time.time() - t_gw) * 1000)
                        gw_decision = _attr(gw_result, "final_decision") or _attr(gw_result, "decision")
                        gw_decision_band = _attr(gw_result, "decision_band", None)
                        gw_fused = float(
                            _attr(gw_result, "fused_risk", None)
                            or _attr(gw_result, "fused_risk_score", 0.0)
                            or 0.0
                        )
                        # SecurityGateway returns Pydantic ModuleRisk objects;
                        # the legacy FusionEngineResponse used plain dicts.
                        # `_attr` accesses both shapes uniformly.
                        module_risks = _attr(gw_result, "module_risks", []) or []
                        for mr in module_risks:
                            name = _attr(mr, "module", "") or ""
                            score = float(_attr(mr, "risk_score", 0.0) or 0.0)
                            if name == "prompt_guard":
                                gw_prompt = score
                            elif name == "rag_guard":
                                gw_rag = score
                            elif name == "output_agency":
                                gw_agency = score
                            ev = _attr(mr, "evidence", []) or []
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
                        # SecurityGateway already emits ModuleResultEvent × N
                        # and FusionDecisionEvent (with run_id/target_id/
                        # attack_id) inside .analyze(), so we don't re-emit
                        # FusionDecisionEvent here — would have duplicated.
                        # Counter bookkeeping for the [done] summary line still
                        # belongs to the runner.
                        key = f"gateway_{gw_decision}"
                        if key in summary:
                            summary[key] += 1
                        summary["gateway_latency_sum_ms"] += gw_latency_ms

                gateway_miss = _classify_gateway_miss(case.expected, gw_decision)
                summary["gateway_miss"] += gateway_miss

                # ------------------------------------------------------------------
                # Hafta 14: tools_local target → if the gateway allowed (or
                # sanitised) the call, actually execute the tool via the local
                # registry. BLOCK / no-tool-call cases skip execution.
                # ------------------------------------------------------------------
                tool_executed = 0
                tool_latency_ms = 0
                tool_response_chars = 0
                tool_response_preview = ""
                tool_error = ""
                if (
                    target.type == "tools_local"
                    and case.gateway_tool_call
                    and gw_decision in ("allow", "sanitize")
                ):
                    try:
                        from tools import invoke as _tool_invoke
                        import json as _json
                        t_tool = time.time()
                        tool_name = case.gateway_tool_call.get("tool", "")
                        tool_args = case.gateway_tool_call.get("args") or {}
                        tool_result = _tool_invoke(tool_name, tool_args)
                        tool_latency_ms = int((time.time() - t_tool) * 1000)
                        preview = _json.dumps(tool_result, ensure_ascii=False)
                        tool_response_chars = len(preview)
                        tool_response_preview = preview[:200]
                        if isinstance(tool_result, dict) and "error" in tool_result:
                            tool_error = str(tool_result["error"])[:200]
                            tool_executed = 1  # we did invoke; the tool returned an error
                        else:
                            tool_executed = 1
                    except Exception as exc:  # noqa: BLE001
                        tool_error = f"{type(exc).__name__}: {exc}"[:200]
                        tool_executed = 0
                        emit_telemetry(ErrorEvent(
                            run_id=run_id, target_id=target.id, attack_id=case.id,
                            where="external_eval.tools_local.invoke",
                            error_type=type(exc).__name__,
                            message=str(exc)[:300],
                        ))

                row = {
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
                    "gateway_decision_band": gw_decision_band or "",
                    "gateway_fused_score": round(gw_fused, 4),
                    "gateway_prompt_score": round(gw_prompt, 4),
                    "gateway_rag_score": round(gw_rag, 4),
                    "gateway_agency_score": round(gw_agency, 4),
                    "gateway_latency_ms": gw_latency_ms,
                    "gateway_miss": gateway_miss,
                    "evidence_top": " | ".join(gw_evidence)[:120],
                    "tool_executed": tool_executed,
                    "tool_latency_ms": tool_latency_ms,
                    "tool_response_chars": tool_response_chars,
                    "tool_response_preview": tool_response_preview,
                    "tool_error": tool_error,
                }
                writer.writerow(row)
                run_rows.append(row)

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

    # ------------------------------------------------------------------
    # Per-run artefacts: results.csv + manifest.json + registry entry.
    # Written after the aggregate save so a crash here can't lose data.
    # All errors swallowed — manifest layer is best-effort, the run
    # itself succeeded by this point.
    # ------------------------------------------------------------------
    try:
        from utils.run_manifest import (
            write_results_csv as _wm_write_results,
            write_manifest as _wm_write_manifest,
            append_registry_entry as _wm_append_registry,
        )
        runs_dir = output_path.parent
        run_dir = runs_dir / run_id
        ended_at_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
        _wm_write_results(run_dir, run_rows, _CSV_FIELDS)
        _wm_write_manifest(
            run_dir,
            run_id=run_id,
            target_id=target.id,
            suite=args.suite,
            started_at=started_at_iso,
            ended_at=ended_at_iso,
            exit_code=0,
            n_cases=len(cases),
            n_rows=len(run_rows),
            sources={
                "external_eval_results": str(output_path.name),
            },
        )
        _wm_append_registry(
            runs_dir,
            run_id=run_id,
            target_id=target.id,
            suite=args.suite,
            started_at=started_at_iso,
            ended_at=ended_at_iso,
            exit_code=0,
            n_cases=len(cases),
            n_rows=len(run_rows),
            relative_path=str(run_dir.relative_to(runs_dir.parent)),
        )
        print(f"[manifest] wrote {run_dir}/results.csv + manifest.json + registry entry")
    except Exception as exc:  # noqa: BLE001 — manifest is best-effort
        print(f"[warn] manifest layer failed: {exc}", file=sys.stderr)

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
