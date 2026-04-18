#!/usr/bin/env bash
# Wraps `docker compose config` and masks secret env values before printing,
# so the rendered output is safe to paste into reports / chats / PRs.
#
# Usage:
#   ./compose-config-safe.sh                         # dev (base + override)
#   ./compose-config-safe.sh -f docker-compose.yml   # reproducibility (base only)
set -euo pipefail

cd "$(dirname "$0")"

# Names of env vars whose values must be redacted in the rendered output.
SECRETS=(HF_TOKEN HUGGING_FACE_HUB_TOKEN OPENAI_API_KEY ANTHROPIC_API_KEY)

SED_ARGS=()
for name in "${SECRETS[@]}"; do
    # Matches both YAML map form (`HF_TOKEN: value`) and list form (`- HF_TOKEN=value`).
    SED_ARGS+=(-e "s|\(${name}[:=] *\).*|\1***REDACTED***|")
done

docker compose "$@" config | sed "${SED_ARGS[@]}"
