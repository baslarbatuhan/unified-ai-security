#!/usr/bin/env bash
# Wraps `docker compose config` and masks secret env values before printing,
# so the rendered output is safe to paste into reports / chats / PRs.
#
# Usage:
#   ./compose-config-safe.sh                         # dev (base + override)
#   ./compose-config-safe.sh -f docker-compose.yml   # reproducibility (base only)
set -euo pipefail

cd "$(dirname "$0")"

# Redact by NAME PATTERN rather than a hard-coded provider list. Any
# env-var whose name ends in (or contains) KEY / TOKEN / SECRET /
# PASSWORD / PASSWD gets its value replaced — so user-named secrets
# like `MY_GEMINI_KEY` or `INTERNAL_BEARER_TOKEN` are covered without
# editing this script.
#
# The regex matches both YAML map form (`MY_KEY: value`) and list form
# (`- MY_KEY=value`). Case-insensitive (sed -E + ignore-case via [Kk]
# etc. is awkward; we rely on the fact that env-var names are
# conventionally uppercase — anyone using lowercase secret names is
# already off the well-trodden path).
PATTERN='([A-Z][A-Z0-9_]*(KEY|TOKEN|SECRET|PASSWORD|PASSWD))'

docker compose "$@" config \
  | sed -E "s|(- ?${PATTERN}=).*|\1***REDACTED***|; s|(${PATTERN}: *)[^[:space:]].*|\1***REDACTED***|"
