"""reporting/
Automated thesis-grade report generation from on-disk artefacts.

Three small modules:
    summary_generator      — snapshot → human-readable markdown summary
    recommendation_engine  — snapshot → ranked actionable recommendations
    report_generator       — composes the two + appendices into one report

Every module is a pure function (apart from the top-level orchestrator that
reads telemetry/CSV files), so unit tests stub inputs without I/O.
"""
from __future__ import annotations

from reporting.summary_generator import (  # noqa: F401
    Summary,
    build_summary,
    render_summary,
)
from reporting.recommendation_engine import (  # noqa: F401
    Recommendation,
    derive_recommendations,
    render_recommendations,
)
from reporting.report_generator import (  # noqa: F401
    generate_report,
    render_report,
)

__all__ = [
    "Summary",
    "build_summary",
    "render_summary",
    "Recommendation",
    "derive_recommendations",
    "render_recommendations",
    "generate_report",
    "render_report",
]
