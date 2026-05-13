"""tools/stock.py — Yahoo Finance public chart endpoint.

Stayed off `yfinance` on purpose: it's a heavy dep, pins a specific
pandas range, and the test fixtures it ships fight with our existing
pandas version. The unofficial chart endpoint returns the same data
in a smaller JSON payload that's easy to forward to the dashboard.

Schema enforced by the gateway's ParameterValidator:
    `^[A-Z]{1,5}(\\.[A-Z]{1,3})?$`  e.g. AAPL, MSFT, BRK.A
By the time this runs, malformed symbols (path traversal, SQL
injection, lowercase, oversized) have already been blocked. The
endpoint itself is rate-limited per-IP (~2k requests/hour, plenty for
a suite run).

Failure modes (handled): network timeout, non-2xx, unparsable JSON.
"""
from __future__ import annotations

import os
from typing import Any, Dict

import httpx

from tools import _register


# Yahoo's public chart endpoint. Returns OHLC + meta. Configurable so
# tests can point at a local mock without monkey-patching httpx.
_ENDPOINT_TEMPLATE = os.environ.get(
    "STOCK_API_URL_TEMPLATE",
    "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
)
_TIMEOUT_S = float(os.environ.get("STOCK_API_TIMEOUT_S", "10"))


def call(*, symbol: str, **_: Any) -> Dict[str, Any]:
    """Fetch the latest quote summary for `symbol`.

    Returns a slimmed-down dict — last price + currency + exchange —
    so the dashboard preview stays under the CSV cell limit. Callers
    that need the full chart series can hit the endpoint directly.
    """
    sym = str(symbol).strip().upper()
    if not sym:
        return {"error": "empty symbol"}

    url = _ENDPOINT_TEMPLATE.format(symbol=sym)
    params = {"interval": "1d", "range": "1d"}

    try:
        # Yahoo blocks requests without a UA on some PoPs — set one.
        headers = {"User-Agent": "Mozilla/5.0 (compatible; uais-tools/1.0)"}
        with httpx.Client(timeout=_TIMEOUT_S, headers=headers) as client:
            resp = client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPStatusError as exc:
        body = ""
        try:
            body = exc.response.text[:200]
        except Exception:  # noqa: BLE001
            pass
        return {"error": f"yahoo {exc.response.status_code}: {body}"}
    except httpx.HTTPError as exc:
        return {"error": f"network: {type(exc).__name__}: {exc}"}
    except ValueError as exc:
        return {"error": f"bad-json: {exc}"}

    # The chart JSON nests result under .chart.result[0]. Yahoo wraps
    # errors under .chart.error so check that first.
    chart = (data.get("chart") or {})
    err = chart.get("error")
    if err:
        return {
            "error": f"yahoo: {err.get('code', '?')}: {err.get('description', '')}",
            "symbol": sym,
        }
    results = chart.get("result") or []
    if not results:
        return {"error": "yahoo returned no results", "symbol": sym}
    meta = (results[0].get("meta") or {})
    return {
        "symbol": sym,
        "currency": meta.get("currency"),
        "exchange_name": meta.get("exchangeName") or meta.get("fullExchangeName"),
        "instrument_type": meta.get("instrumentType"),
        "regular_market_price": meta.get("regularMarketPrice"),
        "previous_close": meta.get("chartPreviousClose"),
    }


_register("stock_quote", call)


__all__ = ["call"]
