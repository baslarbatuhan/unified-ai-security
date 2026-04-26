"""Reports — list, view, download (markdown / PDF), regenerate, executive summary."""
from __future__ import annotations

import io
import sys
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st

from dashboard.lib.gateway_client import GatewayError, get_default_client


st.set_page_config(page_title="Reports", page_icon=":memo:", layout="wide")
st.title("Reports")
client = get_default_client()


# ---------------------------------------------------------------------------
# Listing + regenerate
# ---------------------------------------------------------------------------
try:
    raw = client.get_json("/reports") or []
except GatewayError as exc:
    st.error(f"Gateway unreachable: {exc}")
    st.stop()

# /reports may return a list or a {"reports": [...]} envelope — handle both.
listing = raw.get("reports") if isinstance(raw, dict) else raw
listing = listing or []

if listing:
    st.dataframe(listing, width="stretch", hide_index=True)
else:
    st.info("No reports yet.")

col_a, col_b = st.columns([3, 1])
names = [item.get("name") for item in listing if isinstance(item, dict) and item.get("name")]
chosen = col_a.selectbox("Open report", names) if names else None

if col_b.button("Regenerate", width="stretch"):
    try:
        result = client.post_json("/reports/regenerate", {})
    except GatewayError as exc:
        st.error(f"Regeneration failed: {exc}")
    else:
        st.success(f"Regenerated: {result}")
        st.rerun()


# ---------------------------------------------------------------------------
# Executive summary card — uses the in-process generator so it always
# matches the latest telemetry, even if the markdown report is stale.
# ---------------------------------------------------------------------------
with st.expander("Executive summary (live)", expanded=True):
    try:
        from reporting.summary_generator import build_summary, render_summary
        from schemas import telemetry_schema as ts

        events = ts.read_events(limit=5000)
        summary = build_summary(events)
        st.markdown(render_summary(summary))
    except Exception as exc:  # noqa: BLE001 — never break the page on a missing log
        st.info(f"Executive summary unavailable: {exc}")


# ---------------------------------------------------------------------------
# Open / download (markdown + PDF)
# ---------------------------------------------------------------------------
if not chosen:
    st.stop()

try:
    body = client.get_text(f"/reports/{chosen}")
except GatewayError as exc:
    st.error(f"Could not load: {exc}")
    st.stop()

c_md, c_pdf = st.columns(2)
c_md.download_button(
    "Download .md",
    data=body.encode("utf-8"),
    file_name=chosen,
    mime="text/markdown",
    width="stretch",
)

# PDF export: try optional dep `markdown` + `weasyprint` — fall back to a
# plain-text PDF via reportlab if neither is installed. The dashboard
# explicitly states which path was taken so the user knows what they got.
def _pdf_bytes(md_text: str, *, title: str) -> tuple[bytes | None, str]:
    try:
        import markdown  # type: ignore
        from weasyprint import HTML  # type: ignore
        html = "<style>body{font-family:sans-serif;}</style>" + markdown.markdown(md_text)
        return HTML(string=html).write_pdf(), "weasyprint+markdown"
    except Exception:
        pass
    try:
        from reportlab.lib.pagesizes import A4  # type: ignore
        from reportlab.pdfgen import canvas  # type: ignore
        buf = io.BytesIO()
        c = canvas.Canvas(buf, pagesize=A4)
        width, height = A4
        y = height - 50
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, y, title); y -= 22
        c.setFont("Helvetica", 9)
        for line in md_text.splitlines():
            for wrap in [line[i:i + 110] for i in range(0, max(1, len(line)), 110)]:
                if y < 50:
                    c.showPage(); y = height - 50; c.setFont("Helvetica", 9)
                c.drawString(50, y, wrap); y -= 12
        c.save()
        return buf.getvalue(), "reportlab (plain-text)"
    except Exception as exc:  # noqa: BLE001
        return None, f"unavailable: {exc}"


pdf_bytes, pdf_engine = _pdf_bytes(body, title=chosen)
if pdf_bytes:
    pdf_name = chosen.rsplit(".", 1)[0] + ".pdf"
    c_pdf.download_button(
        f"Download .pdf ({pdf_engine})",
        data=pdf_bytes,
        file_name=pdf_name,
        mime="application/pdf",
        width="stretch",
    )
else:
    c_pdf.info(
        "PDF export needs `weasyprint`+`markdown` or `reportlab`. "
        f"Status: {pdf_engine}"
    )

st.markdown(body)
