"""external_eval/web_adapter.py
=================================
Playwright-backed adapter for web chatbots.

Designed to tolerate UI churn:
  * Primary + fallback selectors (try one, then the next).
  * Response wait policy: fixed delay after submit (selectors.response_wait_ms)
    plus a polling read that captures the "latest message" when it stops
    growing for 500ms.
  * One persistent browser context per adapter instance — `close()` releases
    it. The runner reuses an adapter across an entire suite to amortize
    browser startup (~1–2s).

Playwright is an optional dependency. The import is lazy so the package
loads on systems where Playwright is not installed; `WebAdapter.__init__`
raises `AdapterConfigError` then.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

from schemas.target_schema import TargetConfig
from external_eval.base_adapter import (
    AdapterConfigError,
    AdapterError,
    AdapterTimeout,
    AdapterTransportError,
    ChatbotAdapter,
)


class WebAdapter(ChatbotAdapter):
    """Playwright adapter.  One browser context, many sends.

    Thread-safety: Playwright's sync API binds to the thread that created
    the instance; do not share `WebAdapter` across threads.
    """

    def __init__(self, target: TargetConfig):
        super().__init__(target)
        if target.type != "web":
            raise AdapterConfigError(
                f"WebAdapter requires type='web', got {target.type!r}"
            )
        if target.selectors is None:
            raise AdapterConfigError(
                f"WebAdapter target {target.id!r} missing selectors"
            )
        if not target.endpoint:
            raise AdapterConfigError(
                f"WebAdapter target {target.id!r} missing endpoint (page URL)"
            )

        try:
            from playwright.sync_api import sync_playwright  # noqa: F401
        except ImportError as exc:
            raise AdapterConfigError(
                "Playwright is required for WebAdapter. Install with "
                "`pip install playwright && playwright install chromium`."
            ) from exc

        self._pw_ctx = None
        self._browser = None
        self._page = None
        self._started = False

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def _ensure_started(self) -> None:
        if self._started:
            return
        from playwright.sync_api import sync_playwright

        self._pw_ctx = sync_playwright().start()
        self._browser = self._pw_ctx.chromium.launch(headless=True)
        context = self._browser.new_context()
        self._page = context.new_page()
        self._page.set_default_timeout(int(self.target.timeout_seconds * 1000))
        try:
            self._page.goto(self.target.endpoint, wait_until="domcontentloaded")
        except Exception as exc:
            self._teardown()
            raise AdapterTransportError(f"failed to open {self.target.endpoint}: {exc}") from exc
        self._started = True

    def _teardown(self) -> None:
        try:
            if self._browser is not None:
                self._browser.close()
        except Exception:
            pass
        try:
            if self._pw_ctx is not None:
                self._pw_ctx.stop()
        except Exception:
            pass
        self._browser = None
        self._pw_ctx = None
        self._page = None
        self._started = False

    def close(self) -> None:
        self._teardown()

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _selector_chain(self, primary: str, fallbacks: List[str]) -> List[str]:
        return [primary, *fallbacks]

    def _find_first(self, selectors: List[str]):
        """Try selectors in order; return the first Locator that resolves."""
        assert self._page is not None
        last_exc: Optional[Exception] = None
        for sel in selectors:
            try:
                loc = self._page.locator(sel)
                if loc.count() > 0:
                    return loc.first, sel
            except Exception as exc:
                last_exc = exc
                continue
        raise AdapterError(
            f"no selector from {selectors!r} matched"
            + (f" (last error: {last_exc})" if last_exc else "")
        )

    # ------------------------------------------------------------------
    # ChatbotAdapter API
    # ------------------------------------------------------------------
    def _send_impl(
        self, prompt: str, session_context: Dict[str, Any]
    ) -> Tuple[str, Dict[str, Any]]:
        self._ensure_started()
        assert self._page is not None
        selectors = self.target.selectors
        assert selectors is not None

        # Locate input
        in_chain = self._selector_chain(selectors.input, selectors.fallback_input)
        input_el, input_sel = self._find_first(in_chain)
        try:
            input_el.fill(prompt)
        except Exception as exc:
            raise AdapterError(f"input fill failed on {input_sel!r}: {exc}") from exc

        # Submit
        if selectors.submit:
            try:
                self._page.locator(selectors.submit).first.click()
            except Exception as exc:
                raise AdapterError(f"submit click failed on {selectors.submit!r}: {exc}") from exc
        else:
            try:
                input_el.press("Enter")
            except Exception as exc:
                raise AdapterError(f"Enter key failed: {exc}") from exc

        # Wait for response. Two stages:
        #   1. Fixed delay for the UI to start streaming.
        #   2. Poll until the response text stops growing for >= 500ms or the
        #      overall target timeout elapses.
        delay_s = selectors.response_wait_ms / 1000.0
        deadline = time.time() + self.target.timeout_seconds
        if delay_s > 0:
            time.sleep(min(delay_s, max(0.0, deadline - time.time())))

        out_chain = self._selector_chain(selectors.response, selectors.fallback_response)

        last_text = ""
        stable_since: Optional[float] = None
        response_sel_used: Optional[str] = None

        while time.time() < deadline:
            try:
                resp_el, response_sel_used = self._find_first(out_chain)
                text = (resp_el.inner_text() or "").strip()
            except AdapterError:
                text = ""
            if text and text == last_text:
                stable_since = stable_since or time.time()
                if time.time() - stable_since >= 0.5:
                    break
            else:
                last_text = text
                stable_since = None
            time.sleep(0.2)

        if not last_text:
            raise AdapterTimeout(
                f"no response text within {self.target.timeout_seconds}s"
            )

        metadata = {
            "input_selector": input_sel,
            "response_selector": response_sel_used,
            "endpoint": self.target.endpoint,
        }
        return last_text, metadata


__all__ = ["WebAdapter"]
