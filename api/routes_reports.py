"""api/routes_reports.py
Reports listing + download.

Exposes the contents of the `reports/` directory (markdown files generated
by `reporting/report_generator.py` and the thesis writeups). Read-only.

Routes
------
GET /reports                       — list available reports with metadata
GET /reports/{name}                — return raw markdown body
GET /reports/{name}/download       — same body but with attachment headers
POST /reports/regenerate           — re-run reporting.report_generator on
                                     the live telemetry log
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse, Response


router = APIRouter(prefix="/reports", tags=["reports"])

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_REPORTS_DIR = _PROJECT_ROOT / "reports"

# Path-traversal guard: reject any name containing separators or relative
# components. Reports are bare *.md filenames.
_SAFE_NAME_CHARS = set(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-."
)


def _validate_name(name: str) -> None:
    if not name or any(c not in _SAFE_NAME_CHARS for c in name) or name.startswith("."):
        raise HTTPException(status_code=400, detail="invalid report name")
    if not name.endswith(".md"):
        raise HTTPException(status_code=400, detail="only .md reports are exposed")


def _resolve(name: str) -> Path:
    _validate_name(name)
    p = _REPORTS_DIR / name
    # Confirm the resolved path is still inside _REPORTS_DIR (defence in depth).
    try:
        p.resolve().relative_to(_REPORTS_DIR.resolve())
    except (ValueError, OSError):
        raise HTTPException(status_code=400, detail="invalid report path")
    if not p.exists() or not p.is_file():
        raise HTTPException(status_code=404, detail=f"report not found: {name}")
    return p


def _summarise(p: Path) -> Dict[str, Any]:
    """List entry: name, size, mtime, first heading (best-effort)."""
    title = p.stem.replace("_", " ").title()
    try:
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("# "):
                    title = line[2:].strip()
                    break
    except Exception:
        pass
    stat = p.stat()
    return {
        "name": p.name,
        "title": title,
        "size_bytes": stat.st_size,
        "modified_at": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(timespec="seconds"),
    }


@router.get("")
def list_reports() -> Dict[str, Any]:
    if not _REPORTS_DIR.exists():
        return {"reports": [], "total": 0}
    items = [_summarise(p) for p in sorted(_REPORTS_DIR.glob("*.md"))]
    items.sort(key=lambda r: r["modified_at"], reverse=True)
    return {"reports": items, "total": len(items)}


@router.get("/{name}", response_class=PlainTextResponse)
def get_report(name: str) -> PlainTextResponse:
    p = _resolve(name)
    return PlainTextResponse(p.read_text(encoding="utf-8"), media_type="text/markdown; charset=utf-8")


@router.get("/{name}/download")
def download_report(name: str) -> Response:
    p = _resolve(name)
    body = p.read_text(encoding="utf-8")
    return Response(
        content=body,
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{p.name}"'},
    )


@router.post("/regenerate")
def regenerate_report() -> Dict[str, Any]:
    """Re-run the auto-generated chatbot security report from live telemetry."""
    from reporting.report_generator import generate_report
    out = generate_report()
    stat = out.stat()
    return {
        "wrote": out.name,
        "size_bytes": stat.st_size,
        "modified_at": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(timespec="seconds"),
    }
