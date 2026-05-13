"""tools/weather.py — open-meteo public weather API.

Free, no API key required. Schema enforced by the gateway's
ParameterValidator (lat ∈ [-90,90], lon ∈ [-180,180], float types).
By the time this handler runs, the call has already cleared the
parameter validator — out-of-range coords and type-coercion attacks
are blocked upstream. This module trusts its inputs.

Example call site:
    >>> from tools import invoke
    >>> r = invoke("weather_forecast", {"latitude": 41.01, "longitude": 28.95, "current_weather": True})
    >>> r["current_weather"]["temperature"]
    18.3

Failure modes (handled — return `{"error": "..."}`):
  * Network timeout (10s default)
  * HTTP non-2xx
  * Malformed JSON
"""
from __future__ import annotations

import os
from typing import Any, Dict

import httpx

from tools import _register


# Configurable via env so a CI run can point the test at a mock server
# without monkey-patching httpx.
_ENDPOINT = os.environ.get(
    "WEATHER_API_URL", "https://api.open-meteo.com/v1/forecast"
)
_TIMEOUT_S = float(os.environ.get("WEATHER_API_TIMEOUT_S", "10"))


def call(
    *,
    latitude: float,
    longitude: float,
    current_weather: bool = False,
    **_: Any,
) -> Dict[str, Any]:
    """Fetch a forecast for the given lat/long.

    Extra kwargs are tolerated but ignored — the gateway schema rejects
    unexpected parameters before we get here. The `**_` catch-all
    protects against schema additions that ship before this handler is
    updated.
    """
    params: Dict[str, Any] = {
        "latitude": float(latitude),
        "longitude": float(longitude),
    }
    if current_weather:
        # open-meteo wants string "true"/"false" for booleans on the wire.
        params["current_weather"] = "true"

    try:
        with httpx.Client(timeout=_TIMEOUT_S) as client:
            resp = client.get(_ENDPOINT, params=params)
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPStatusError as exc:
        # API responded but said no (e.g. 400 with detail string).
        body = ""
        try:
            body = exc.response.text[:200]
        except Exception:  # noqa: BLE001
            pass
        return {"error": f"open-meteo {exc.response.status_code}: {body}"}
    except httpx.HTTPError as exc:
        return {"error": f"network: {type(exc).__name__}: {exc}"}
    except ValueError as exc:
        return {"error": f"bad-json: {exc}"}

    # Surface only the fields callers typically need so the CSV preview
    # stays readable. The full body is available via the raw response
    # below in case a downstream consumer wants it.
    out: Dict[str, Any] = {
        "latitude": data.get("latitude"),
        "longitude": data.get("longitude"),
        "timezone": data.get("timezone"),
        "elevation": data.get("elevation"),
    }
    if "current_weather" in data:
        out["current_weather"] = data["current_weather"]
    return out


_register("weather_forecast", call)


__all__ = ["call"]
