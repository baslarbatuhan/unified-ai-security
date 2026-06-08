"""scripts/export_webui_state.py
==================================
Capture a Playwright ``storage_state`` JSON from a logged-in browser
session — the input for Sprint 8 W4 (cookie / storage_state web auth).

Why a script instead of hand-editing DevTools output: Playwright's
``storage_state`` has a specific shape (``{"cookies": [...],
"origins": [...]}``) covering cookies *and* localStorage/sessionStorage.
Open WebUI (and most SPAs) keep the session token in localStorage, so a
cookies-only hand export silently misses it. ``context.storage_state()``
gets all of it in one call.

Usage
-----
Run on the HOST (needs a display for the headed browser):

    pip install playwright && playwright install chromium
    python scripts/export_webui_state.py \\
        --url http://localhost:3000 \\
        --out runs/open_webui_state.json

The script opens a real browser window, you log in by hand, then press
Enter in the terminal — the authenticated state is dumped to ``--out``.

Then register / edit the web target with::

    auth:
      type: cookie
      storage_state_path: /app/runs/open_webui_state.json

(The container path — ``runs/`` is bind-mounted to ``/app/runs``.)

Security
--------
``runs/`` is git-ignored, so the exported state never reaches the repo.
The file *does* contain a live session token — treat it like a password
and delete it when the session expires.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from urllib.parse import urlparse

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_OUT = _PROJECT_ROOT / "runs" / "open_webui_state.json"


def _rewrite_state(path: Path, login_url: str, target_url: str) -> None:
    """Rewrite cookie ``domain`` and localStorage ``origin`` from the
    host you logged in on to the host the WebAdapter will visit from
    inside Docker.

    Why this is mandatory: a browser scopes cookies by hostname and
    localStorage by origin (scheme + host + port). When you log in at
    ``http://localhost:3000`` (Open WebUI on the host) and the gateway
    container later visits ``http://open-webui:8080`` (in-network),
    the session token sits on the wrong scope and the chatbot redirects
    you back to /login. Cookies don't carry a port, so the domain swap
    is enough for them; origins carry scheme+host+port and need the
    full URL substitution.

    Idempotent — re-running on an already-rewritten file is a no-op.
    """
    login_host = urlparse(login_url).hostname or "localhost"
    target_host = urlparse(target_url).hostname or "open-webui"
    target_scheme_is_https = urlparse(target_url).scheme == "https"

    with path.open() as f:
        state = json.load(f)

    for cookie in state.get("cookies", []) or []:
        dom = (cookie.get("domain") or "").lstrip(".")
        if dom == login_host:
            cookie["domain"] = target_host
            # In-network traffic is HTTP unless the user wired TLS;
            # an originally-secure cookie cannot be sent over HTTP, so
            # downgrade unless the target itself uses HTTPS.
            if not target_scheme_is_https:
                cookie["secure"] = False

    for origin_entry in state.get("origins", []) or []:
        if origin_entry.get("origin") == login_url.rstrip("/"):
            origin_entry["origin"] = target_url.rstrip("/")

    with path.open("w") as f:
        json.dump(state, f, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export a Playwright storage_state JSON from a "
                    "manually logged-in browser session.",
    )
    parser.add_argument(
        "--url",
        default="http://localhost:3000",
        help="Page to open for login (default: Open WebUI on host :3000).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=_DEFAULT_OUT,
        help=f"Output path for the storage_state JSON (default: {_DEFAULT_OUT}).",
    )
    parser.add_argument(
        "--target-url",
        default="http://open-webui:8080",
        help=(
            "URL the WebAdapter will visit from inside the gateway "
            "container.  After the manual login, cookies and "
            "localStorage origins captured under --url are rewritten to "
            "this host so the in-network probe sees the session. Set to "
            "the same value as --url to skip the rewrite."
        ),
    )
    parser.add_argument(
        "--rewrite-only",
        type=Path,
        default=None,
        help=(
            "Skip the browser; just rewrite an existing storage_state "
            "JSON in place from --url to --target-url scopes. Use when "
            "you already exported but forgot to set --target-url."
        ),
    )
    args = parser.parse_args()

    # ----- rewrite-only mode (no browser, no login) -----
    if args.rewrite_only:
        if not args.rewrite_only.exists():
            print(f"ERROR: {args.rewrite_only} does not exist", file=sys.stderr)
            return 1
        _rewrite_state(args.rewrite_only, args.url, args.target_url)
        print(f"✅ rewrote scopes in: {args.rewrite_only}")
        print(f"   {args.url}  →  {args.target_url}")
        return 0

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print(
            "ERROR: Playwright is not installed on this host.\n"
            "  pip install playwright && playwright install chromium",
            file=sys.stderr,
        )
        return 2

    out_path: Path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Opening {args.url} in a browser window…")
    print("→ Log in by hand, get to the chat screen, THEN return here.")

    with sync_playwright() as pw:
        # Headed: the whole point is an interactive manual login.
        browser = pw.chromium.launch(headless=False)
        context = browser.new_context()
        page = context.new_page()
        try:
            page.goto(args.url, wait_until="domcontentloaded")
        except Exception as exc:  # noqa: BLE001 — surface any nav error plainly
            print(f"ERROR: could not open {args.url}: {exc}", file=sys.stderr)
            browser.close()
            return 1

        # Block until the operator confirms they have logged in. input()
        # on a headed run is the simplest reliable "are you done?" gate —
        # auto-detecting "logged in" would need an app-specific selector.
        try:
            input("\nPress Enter once you are logged in and on the chat page… ")
        except (EOFError, KeyboardInterrupt):
            print("\nAborted — no state written.", file=sys.stderr)
            browser.close()
            return 1

        context.storage_state(path=str(out_path))
        browser.close()

    # Scope rewrite for in-network access. The user logs in on the
    # host (e.g. http://localhost:3000) but the WebAdapter visits the
    # in-network URL (http://open-webui:8080), so cookies and
    # localStorage need their scope adjusted — without it, the
    # adapter lands on /login.
    if args.url.rstrip("/") != args.target_url.rstrip("/"):
        _rewrite_state(out_path, args.url, args.target_url)
        print(f"   scopes rewritten: {args.url}  →  {args.target_url}")

    print(f"\n✅ storage_state written to: {out_path}")
    print("Register the web target with:")
    print("  auth:")
    print("    type: cookie")
    print(f"    storage_state_path: /app/runs/{out_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
