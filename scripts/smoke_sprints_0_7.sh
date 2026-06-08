#!/usr/bin/env bash
# smoke_sprints_0_7.sh — post-recreate API smoke for Targets/Vault sprints 0–7.
#
# Usage:
#   cd infra && docker compose up -d --force-recreate gateway dashboard
#   # wait ~90s for gateway warm-up, then:
#   ../scripts/smoke_sprints_0_7.sh
#
# Options:
#   GATEWAY_URL=http://localhost:8000  — gateway base URL
#   DASHBOARD_URL=http://localhost:8501 — Streamlit (optional check)
#   RUN_GEMINI=1                        — also probe gemini_flash_test (needs vault key)
#   RUN_PYTEST=1                        — run sprint unit tests inside container
#
# Exit code: 0 if all required checks pass, 1 otherwise.

set -euo pipefail

GATEWAY_URL="${GATEWAY_URL:-http://localhost:8000}"
DASHBOARD_URL="${DASHBOARD_URL:-http://localhost:8501}"
RUN_GEMINI="${RUN_GEMINI:-0}"
RUN_PYTEST="${RUN_PYTEST:-0}"
SMOKE_SECRET="UAIS_SMOKE_VAULT_KEY_$$"
SMOKE_TARGET="smoke_sprint_tmp_$$"

PASS=0
FAIL=0
SKIP=0

_red() { printf '\033[31m'; }
_grn() { printf '\033[32m'; }
_ylw() { printf '\033[33m'; }
_rst() { printf '\033[0m'; }

pass() { _grn; echo "  PASS  $*"; _rst; PASS=$((PASS + 1)); }
fail() { _red; echo "  FAIL  $*"; _rst; FAIL=$((FAIL + 1)); }
skip() { _ylw; echo "  SKIP  $*"; _rst; SKIP=$((SKIP + 1)); }
info() { echo "  ....  $*"; }

json_get() {
  local expr="$1"
  python3 -c "
import json, sys
d = json.load(sys.stdin)
def walk(obj, path):
    cur = obj
    for p in path.split('.'):
        if p == '':
            continue
        if p.endswith(']'):
            key, idx = p[:-1].split('[')
            cur = cur[key][int(idx)]
        else:
            cur = cur[p]
    return cur
try:
    print(walk(d, '$expr'))
except Exception:
    sys.exit(1)
" 2>/dev/null
}

curl_json() {
  local method="$1" url="$2"
  shift 2
  curl -sS -m 30 -X "$method" "$url" \
    -H 'Content-Type: application/json' \
    "$@"
}

wait_healthy() {
  info "Waiting for gateway /health (up to 120s)…"
  local i
  for i in $(seq 1 24); do
    if curl -sf -m 5 "${GATEWAY_URL}/health" >/dev/null 2>&1; then
      return 0
    fi
    sleep 5
  done
  return 1
}

section() {
  echo ""
  echo "=== $* ==="
}

# ---------------------------------------------------------------------------
section "0 — Preflight"
if ! command -v python3 >/dev/null; then
  echo "python3 required for JSON parsing"; exit 1
fi
if ! wait_healthy; then
  fail "Gateway not healthy at ${GATEWAY_URL}/health"
  echo "Start stack: cd infra && docker compose up -d gateway dashboard"
  exit 1
fi
pass "Gateway /health reachable"

status=$(curl_json GET "${GATEWAY_URL}/health" | json_get status || true)
if [[ "$status" == "HEALTHY" ]]; then
  pass "Gateway status=HEALTHY"
else
  fail "Expected HEALTHY, got: ${status:-<none>}"
fi

# ---------------------------------------------------------------------------
section "Sprint 0 — Vault API"
curl_json PUT "${GATEWAY_URL}/secrets/${SMOKE_SECRET}" \
  -d '{"value":"smoke-not-a-real-secret"}' >/dev/null
pass "PUT /secrets/{name}"

names=$(curl_json GET "${GATEWAY_URL}/secrets" | json_get names || true)
if echo "$names" | grep -q "$SMOKE_SECRET"; then
  pass "GET /secrets lists new name (no value in response)"
else
  fail "SMOKE secret name not in /secrets names"
fi

src=$(curl_json GET "${GATEWAY_URL}/secrets/${SMOKE_SECRET}" | json_get source || true)
if [[ "$src" == "vault" ]]; then
  pass "GET /secrets/{name} source=vault"
else
  fail "Expected source=vault for ${SMOKE_SECRET}, got: ${src:-<none>}"
fi

# Values must never appear in list response body
body=$(curl_json GET "${GATEWAY_URL}/secrets")
if echo "$body" | grep -q 'smoke-not-a-real-secret'; then
  fail "Secret VALUE leaked in GET /secrets body"
else
  pass "No secret value leak in GET /secrets"
fi

code=$(curl -sS -o /dev/null -w '%{http_code}' -X DELETE "${GATEWAY_URL}/secrets/${SMOKE_SECRET}")
if [[ "$code" == "200" || "$code" == "204" ]]; then
  pass "DELETE /secrets/{name}"
else
  fail "DELETE /secrets returned HTTP ${code}"
fi

# ---------------------------------------------------------------------------
section "Sprint 3 — /targets/test + auth_sources"
mock_body=$(curl_json POST "${GATEWAY_URL}/targets/test" -d '{
  "target": {"id":"smoke_mock","name":"Smoke Mock","type":"mock","timeout_seconds":5},
  "probe_prompt": "ping"
}')
mock_ok=$(echo "$mock_body" | json_get ok || true)
if [[ "$mock_ok" == "True" || "$mock_ok" == "true" ]]; then
  pass "POST /targets/test mock ok=true"
else
  fail "Mock probe failed: $(echo "$mock_body" | head -c 200)"
fi
if echo "$mock_body" | grep -q 'auth_sources'; then
  pass "Mock probe includes auth_sources field"
else
  fail "auth_sources missing from /targets/test response"
fi

curl_json PUT "${GATEWAY_URL}/secrets/${SMOKE_SECRET}" \
  -d '{"value":"bearer-smoke-token"}' >/dev/null
bearer_body=$(curl_json POST "${GATEWAY_URL}/targets/test" -d "{
  \"target\": {
    \"id\": \"smoke_bearer\",
    \"name\": \"Smoke Bearer\",
    \"type\": \"api\",
    \"endpoint\": \"https://httpbin.org/anything\",
    \"http_method\": \"POST\",
    \"request_template\": {\"msg\": \"{prompt}\"},
    \"auth\": {\"type\": \"bearer\", \"token_env\": \"${SMOKE_SECRET}\"}
  },
  \"probe_prompt\": \"ping\"
}")
bearer_sources=$(echo "$bearer_body" | python3 -c "
import json,sys
d=json.load(sys.stdin)
s=d.get('auth_sources') or {}
for k,v in s.items():
    if 'token_env' in k and v=='vault':
        sys.exit(0)
sys.exit(1)
" 2>/dev/null && echo yes || echo no)
if [[ "$bearer_sources" == "yes" ]]; then
  pass "auth_sources reports vault for bearer token_env"
else
  fail "auth_sources did not show vault for bearer (check secret_resolver mount)"
fi
curl -sS -X DELETE "${GATEWAY_URL}/secrets/${SMOKE_SECRET}" >/dev/null

schema_body=$(curl_json POST "${GATEWAY_URL}/targets/test" -d '{
  "target": {"id":"bad","type":"api","endpoint":"http://x.test"}
}')
schema_ok=$(echo "$schema_body" | json_get ok || true)
schema_cat=$(echo "$schema_body" | json_get category || true)
if [[ ("$schema_ok" == "False" || "$schema_ok" == "false") && "$schema_cat" == "schema" ]]; then
  pass "Invalid target → ok=false category=schema (not HTTP 422)"
else
  fail "Schema probe expected ok=false/category=schema"
fi

# ---------------------------------------------------------------------------
section "Sprint 0/1 — Targets CRUD (YAML round-trip)"
# Gateway middleware may return 429 if smoke is run back-to-back; pause briefly.
sleep 2
create_payload=$(cat <<EOF
{
  "id": "${SMOKE_TARGET}",
  "name": "Smoke Temp",
  "type": "mock",
  "enabled": false,
  "timeout_seconds": 5.0,
  "auth": {"type": "none", "extra_headers": {}}
}
EOF
)
code=$(curl -sS -o /dev/null -w '%{http_code}' -X POST "${GATEWAY_URL}/targets" \
  -H 'Content-Type: application/json' -d "$create_payload")
if [[ "$code" == "200" || "$code" == "201" ]]; then
  pass "POST /targets (create mock)"
elif [[ "$code" == "429" ]]; then
  skip "POST /targets rate-limited (429) — wait 60s and re-run CRUD section"
else
  fail "POST /targets returned HTTP ${code}"
fi

listed=$(curl_json GET "${GATEWAY_URL}/targets" | python3 -c "
import json,sys
ids=[t['id'] for t in json.load(sys.stdin).get('targets',[])]
print('yes' if '${SMOKE_TARGET}' in ids else 'no')
")
if [[ "$listed" == "yes" ]]; then
  pass "GET /targets includes new id"
else
  fail "Created target not listed"
fi

code=$(curl -sS -o /dev/null -w '%{http_code}' -X DELETE "${GATEWAY_URL}/targets/${SMOKE_TARGET}")
if [[ "$code" == "200" ]]; then
  pass "DELETE /targets/{id} (EBUSY/bind-mount fix)"
else
  fail "DELETE /targets returned HTTP ${code} (read-only or EBUSY mount?)"
fi

# ---------------------------------------------------------------------------
section "Sprint 1 — Header \${VAR} (unit tests in container)"
if [[ "$RUN_PYTEST" == "1" ]]; then
  if docker exec uais-gateway python -m pytest \
      tests/test_api_adapter_auth.py::TestHeaderInterpolation \
      tests/test_secret_resolver.py -q --tb=no 2>/dev/null; then
    pass "Container pytest: header interpolation + resolver"
  else
    fail "Container pytest failed (image may lack new tests — rebuild gateway)"
  fi
else
  skip "RUN_PYTEST=1 to run header interpolation tests in container"
fi

# ---------------------------------------------------------------------------
section "Optional — Live Gemini (your vault E2E target)"
if [[ "$RUN_GEMINI" == "1" ]]; then
  gemini_src=$(curl_json GET "${GATEWAY_URL}/secrets/GEMINI_API_KEY" 2>/dev/null | json_get source || echo missing)
  if [[ "$gemini_src" != "vault" && "$gemini_src" != "env" ]]; then
    skip "GEMINI_API_KEY not in vault/env — dashboard paste first"
  else
    gemini_body=$(curl_json POST "${GATEWAY_URL}/targets/test" -d '{
      "target": {
        "id": "gemini_flash_test",
        "name": "Gemini Flash",
        "type": "api",
        "enabled": true,
        "endpoint": "https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent",
        "http_method": "POST",
        "request_template": {"contents": [{"parts": [{"text": "{prompt}"}]}]},
        "response_path": "candidates.0.content.parts.0.text",
        "auth": {"type": "query", "query_key": "key", "query_value_env": "GEMINI_API_KEY"}
      },
      "probe_prompt": "ping"
    }' 2>/dev/null || echo '{}')
    g_ok=$(echo "$gemini_body" | json_get ok || true)
    if [[ "$g_ok" == "True" || "$g_ok" == "true" ]]; then
      pass "Gemini probe ok=true (vault/query auth)"
    else
      fail "Gemini probe failed — check key/quota: $(echo "$gemini_body" | head -c 180)"
    fi
  fi
else
  skip "RUN_GEMINI=1 to probe gemini_flash_test (needs GEMINI_API_KEY in vault)"
fi

# ---------------------------------------------------------------------------
section "Dashboard (optional)"
if curl -sf -m 5 "${DASHBOARD_URL}/_stcore/health" >/dev/null 2>&1; then
  pass "Dashboard Streamlit health OK"
else
  skip "Dashboard not reachable at ${DASHBOARD_URL}"
fi

# ---------------------------------------------------------------------------
section "Summary"
echo ""
echo "Passed: ${PASS}  Failed: ${FAIL}  Skipped: ${SKIP}"
if [[ "$FAIL" -eq 0 && "$SKIP" -gt 0 ]]; then
  _ylw
  echo "Some optional checks were skipped."
  _rst
fi
if [[ "$FAIL" -gt 0 ]]; then
  _red
  echo "SMOKE FAILED"
  _rst
  exit 1
fi
_grn
echo "SMOKE OK"
_rst
exit 0
