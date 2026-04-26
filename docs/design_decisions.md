# Design Decisions

Architectural choices that shape the current codebase. For day-to-day operational trade-offs (log size, dashboard polling), see `reports/ops_notes.md`.

## Separate `POST /analyze-output` instead of optional `model_output` on `POST /analyze`

- **Context:** The fusion engine already exposes `analyze()` and `analyze_with_output()` as distinct methods; the HTTP layer mirrors that split.
- **Decision:** Add a **second route** for post-LLM screening instead of overloading a single endpoint with an optional field.
- **Rationale:** One URL keeps one semantic — pre-LLM input screening vs post-LLM output screening. Clients, OpenAPI, and telemetry stay clear (`output_score` is meaningful only on `/analyze-output`). Avoids “same endpoint, different behavior” depending on which JSON fields are present.
- **Alternative rejected:** Optional `model_output` on `POST /analyze` would multiplex two contracts on one path and complicate dashboard queries (e.g. `output_score` sometimes unset vs always present as `0.0` on input-only calls).

## Additive response schema

- `AnalyzeResponse` / `AnalyzeResponseModel` include `output_score` for all responses; **plain `/analyze` returns `0.0`**, so existing clients stay compatible.

## `gateway_miss` (external eval) vs RAG ASR

- **Decision:** `external_eval` CSV column and summaries use `gateway_miss`, not `attack_success`, for the “gateway let a block/sanitize case through” indicator.
- **Rationale:** “Attack success” in an end-to-end sense implies analyzing the **external** chatbot’s reply; the harness primarily compares expected vs **gateway** decision. RAG pipeline metrics that truly measure retrieval compromise keep the name `attack_success_rate` (ASR) where appropriate.

## Dashboard: full-file read for v1

- **Decision:** Implement dashboard helpers as read-the-file-and-tail in Python for thesis-scale data sizes.
- **Rationale:** Simplest correct behavior; `reports/ops_notes.md` records file-size and latency triggers for a future streamed tail implementation.
