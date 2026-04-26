# Glossary

## Gateway / evaluation

- **Pre-LLM screening** — Analysis before the **application** calls its own target LLM: `POST /analyze`. No `model_output` is required.
- **Post-LLM screening** — Analysis of the model’s **completion** after the client has called the target LLM: `POST /analyze-output` with the same request fields as `/analyze` **plus** `model_output`.
- **Output agency (`output_agency` module)** — Tool-call–centric checks: which tool, parameters, role, object authorization, anti-enumeration, and related prompt hints when a tool is in play. Fusion module name `output_agency`; *not* the same as free-text “output safety” in output guard.
- **Output guard (`output_guard` module)** — Heuristic analysis of the **string** the model returned: PII-like content, key-like content, unsafe instructions, downstream-injection phrasing, suspicious redirects. Part of the four-module fusion on `analyze_with_output` only.
- **Gateway miss (`gateway_miss`)** — In `runs/external_eval_results.csv`, a row-level flag (0/1) for cases where the **expected** policy would block or sanitize but the **gateway** returned `allow`. Measures protector coverage, not “did the external chatbot comply with the attack.”
- **ASR (attack success rate) — RAG** — In `rag_guard/rag_baseline` and related tests, the fraction of poisoned documents that appeared in top-*k* retrieval. **Do not** confuse with `gateway_miss` in external eval; different system boundary.

## Observability

- **Fusion decision event** — Telemetry record of fused decision, per-module scores, and (on the post-LLM path) `output_score` for the output-guard contribution.
