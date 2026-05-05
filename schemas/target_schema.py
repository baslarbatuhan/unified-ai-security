"""schemas/target_schema.py
============================
Pydantic schema for `external_eval/targets.yaml`.

A "target" is any chatbot we can probe: REST API, web UI, or a local mock
used for offline development. The schema is intentionally loose about
authentication (any dict goes) because vendors vary; it is strict about
what the evaluation pipeline needs (id, type, endpoint, timeout).

Validation rules (enforced):
    * `id` is unique within a `TargetsFile`.
    * `type` ∈ {api, web, mock}.
    * `endpoint` is required for api/web; ignored for mock.
    * `timeout_seconds` > 0 and <= 600.
    * `web` targets must carry selectors.input and selectors.response.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator


TargetType = Literal["api", "web", "mock"]


class WebSelectors(BaseModel):
    """CSS / text selectors used by the Playwright adapter."""

    input: str = Field(..., description="Selector for the prompt input element.")
    submit: Optional[str] = Field(
        default=None,
        description="Selector for the send/submit button. If omitted, the adapter presses Enter.",
    )
    response: str = Field(
        ...,
        description="Selector that isolates the assistant's latest reply.",
    )
    # Fallback selectors tried in order if the primary ones fail. Survives UI
    # tweaks without requiring a code release.
    fallback_input: List[str] = Field(default_factory=list)
    fallback_response: List[str] = Field(default_factory=list)
    # Delay between submit and response read (ms). Some UIs stream tokens.
    response_wait_ms: int = Field(default=3000, ge=0, le=120_000)


class RateLimit(BaseModel):
    """Optional per-target throttle honored by the runner."""

    requests_per_minute: int = Field(default=60, ge=1)
    burst: int = Field(default=10, ge=1)


class TargetConfig(BaseModel):
    id: str = Field(
        ...,
        description="Stable identifier used in telemetry + results. Slug-like.",
    )
    name: str = Field(..., description="Human-readable label for the UI.")
    type: TargetType
    enabled: bool = True

    endpoint: Optional[str] = Field(
        default=None,
        description="Base URL for api targets, page URL for web targets. Ignored for mock.",
    )
    timeout_seconds: float = Field(default=30.0, gt=0.0, le=600.0)
    has_tools: bool = Field(
        default=False,
        description=(
            "True if the target exposes tool-calling capabilities. "
            "Controls whether agency_social cases (requires_tools=True) "
            "are included in the evaluation run."
        ),
    )

    # Authentication is vendor-shaped; we accept any dict and let the adapter
    # interpret it (`{"type": "bearer", "token": "..."}`,
    # `{"type": "header", "headers": {...}}`, etc.)
    auth: Dict[str, Any] = Field(default_factory=dict)

    # API-only
    http_method: Literal["POST", "GET"] = Field(
        default="POST",
        description=(
            "HTTP method for `api` targets. POST sends `request_template` "
            "as a JSON body; GET sends `query_template` as a URL-encoded "
            "query string. (Web/mock targets ignore this.)"
        ),
    )
    request_template: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Optional request body template for POST-style api targets. "
            "Use {prompt} placeholder. Example: "
            "{'messages':[{'role':'user','content':'{prompt}'}]}"
        ),
    )
    query_template: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Optional query-string template for GET-style api targets. "
            "Values may contain {prompt} placeholder. Example: "
            "{'message': '{prompt}'} → ?message=<encoded-prompt>"
        ),
    )
    response_path: Optional[str] = Field(
        default=None,
        description=(
            "Dot-path into the JSON response that holds the assistant text. "
            "Example: 'choices.0.message.content'. Leave blank for plain-text "
            "responses or to let the adapter try common shapes."
        ),
    )

    # Web-only
    selectors: Optional[WebSelectors] = None

    rate_limit: Optional[RateLimit] = None
    # Free-form metadata for dashboards (tags, notes).
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_type_specific(self) -> "TargetConfig":
        if self.type == "api":
            if not self.endpoint:
                raise ValueError(f"target {self.id!r}: api targets require `endpoint`.")
            # GET-style requires query_template — otherwise the adapter
            # has nothing to render and would always send an empty query.
            if self.http_method == "GET" and not self.query_template:
                raise ValueError(
                    f"target {self.id!r}: api+GET requires `query_template` "
                    "(e.g. {'message': '{prompt}'}). Use POST if you want "
                    "to send a JSON body via `request_template` instead."
                )
        elif self.type == "web":
            if not self.endpoint:
                raise ValueError(f"target {self.id!r}: web targets require `endpoint` (page URL).")
            if self.selectors is None:
                raise ValueError(
                    f"target {self.id!r}: web targets require `selectors` with input + response."
                )
        # mock: nothing required
        return self


class TargetsFile(BaseModel):
    """Top-level YAML shape."""

    version: int = Field(default=1, ge=1)
    targets: List[TargetConfig] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unique_ids(self) -> "TargetsFile":
        seen: Dict[str, int] = {}
        for t in self.targets:
            seen[t.id] = seen.get(t.id, 0) + 1
        dupes = [k for k, n in seen.items() if n > 1]
        if dupes:
            raise ValueError(f"duplicate target ids: {dupes}")
        return self


__all__ = [
    "TargetType",
    "WebSelectors",
    "RateLimit",
    "TargetConfig",
    "TargetsFile",
]
