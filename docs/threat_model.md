# Threat Model

High-level module-specific notes live in the other `docs/*_threat_model.md`
files (prompt, RAG, agency, etc.).

**Gateway API surface (current build):** attackers can send arbitrary prompts
to `POST /analyze` and, if a victim application forwards LLM output, arbitrary
`model_output` to `POST /analyze-output`. Defenses: input-side **prompt, RAG,
and agency** modules; output-side **output guard** (plus re-run of the three
input modules) on the post-LLM path. See `docs/architecture_v1.md` and
`docs/glossary.md` for boundaries between **output agency** (tool misuse) and
**output guard** (toxic/secret exfil text).
