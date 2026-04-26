# Security Objectives

This document is a placeholder; concrete metrics and thresholds live in
`configs/secure_balanced.yaml` and in evaluation scripts. For the **current**
gateway behavior (pre-LLM `POST /analyze` vs post-LLM `POST /analyze-output`,
`gateway_miss` in external eval, RAG ASR in baseline scripts), see
`README.md` and `docs/glossary.md`.

**Suggested** thesis-level goals (tune to your data):

- Constrain false positives on `/analyze` for production prompts (role-dependent).
- Lower **gateway misses** (expected `block`/`sanitize`, observed `allow`) in `external_eval_results.csv`.
- Keep batch **output guard** false positives within acceptable range when screening `model_output` offline or via `/analyze-output`.
- Meet latency SLOs in `configs/timeout_config.yaml` for each module, including `output_guard`.
