"""external_eval/openwebui_chat.py
=====================================
Make eval runs show up as a real, readable conversation in the Open WebUI
sidebar.

Open WebUI's OpenAI-compatible ``POST /api/chat/completions`` endpoint runs
the model and returns a reply, but it does NOT create a sidebar-visible chat.
The sidebar is driven by the ``/api/v1/chats`` API. To make our forwarded
attack prompts appear (with the model's answer next to them), we:

  1. Create a chat once per run via ``POST /api/v1/chats/new`` → real UUID.
  2. Run each prompt through ``/api/chat/completions`` as usual (single-turn,
     so eval semantics are unchanged — no context bleed between attacks).
  3. After the reply lands, append the user+assistant pair to an in-memory
     transcript and ``POST /api/v1/chats/{id}`` the FULL document with the
     assistant content filled in (``done=True``).

This is the piece a previous attempt missed: the assistant reply must be
written back AFTER the completion, and the full accumulated history must be
re-sent each turn (a partial post overwrites the chat). Verified against a
live Open WebUI instance: the readback returns both messages with content.

Everything here is best-effort from the adapter's perspective — if the chat
API is unreachable the completion (and therefore the gateway verdict /
A-B evidence) still succeeds; only the sidebar nicety is lost.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import httpx

from schemas.target_schema import TargetConfig
from external_eval.base_adapter import AdapterError


def is_openwebui_target(target: TargetConfig) -> bool:
    """True when the target speaks to Open WebUI's chat completions API."""
    if target.metadata.get("openwebui"):
        return True
    ep = (target.endpoint or "").lower()
    return "open-webui" in ep and "chat/completions" in ep


def base_url(endpoint: str) -> str:
    """``http://host:8080/api/chat/completions`` → ``http://host:8080``."""
    parsed = urlparse(endpoint)
    if not parsed.scheme or not parsed.netloc:
        raise AdapterError(f"cannot derive Open WebUI base URL from {endpoint!r}")
    return f"{parsed.scheme}://{parsed.netloc}"


@dataclass
class OpenWebUISession:
    """Per-adapter Open WebUI chat state for a single eval run."""

    model: str = "qwen2.5:3b"
    title: str = "UAIS Eval"
    chat_id: Optional[str] = None
    # node_id → message node (both user and assistant)
    nodes: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # display order of node ids: [user1, asst1, user2, asst2, ...]
    order: List[str] = field(default_factory=list)
    last_id: Optional[str] = None  # currentId = newest assistant node


def create_chat(
    client: httpx.Client, *, base: str, headers: Dict[str, str], title: str, model: str,
) -> str:
    """Create an empty but UI-loadable chat. Returns the server-assigned UUID."""
    skeleton = {
        "chat": {
            "title": title,
            "models": [model],
            "messages": [],
            "history": {"messages": {}, "currentId": None},
        }
    }
    url = f"{base.rstrip('/')}/api/v1/chats/new"
    try:
        resp = client.post(url, headers=headers, json=skeleton)
    except httpx.HTTPError as exc:
        raise AdapterError(f"Open WebUI chat create failed: {exc}") from exc
    if resp.status_code >= 400:
        raise AdapterError(
            f"Open WebUI POST /api/v1/chats/new HTTP {resp.status_code}: {resp.text[:200]}"
        )
    data = resp.json()
    chat_id = data.get("id") or (data.get("chat") or {}).get("id")
    if not chat_id:
        raise AdapterError(f"Open WebUI /chats/new response missing id: {data!r}")
    return str(chat_id)


def append_turn(session: OpenWebUISession, prompt: str, reply: str) -> None:
    """Append a completed user→assistant exchange to the in-memory transcript."""
    ts = int(time.time())
    user_id = str(uuid.uuid4())
    asst_id = str(uuid.uuid4())

    user_node = {
        "id": user_id,
        "parentId": session.last_id,
        "childrenIds": [asst_id],
        "role": "user",
        "content": prompt,
        "timestamp": ts,
        "models": [session.model],
    }
    asst_node = {
        "id": asst_id,
        "parentId": user_id,
        "childrenIds": [],
        "role": "assistant",
        "content": reply,
        "model": session.model,
        "modelName": session.model,
        "modelIdx": 0,
        "timestamp": ts,
        "done": True,
    }
    # Link the previous assistant node to this new user turn.
    if session.last_id and session.last_id in session.nodes:
        session.nodes[session.last_id].setdefault("childrenIds", []).append(user_id)

    session.nodes[user_id] = user_node
    session.nodes[asst_id] = asst_node
    session.order.extend([user_id, asst_id])
    session.last_id = asst_id


def chat_document(session: OpenWebUISession) -> Dict[str, Any]:
    """Full ``{chat: {...}}`` document the Open WebUI frontend renders."""
    return {
        "chat": {
            "id": session.chat_id,
            "title": session.title,
            "models": [session.model],
            "messages": [session.nodes[i] for i in session.order],
            "history": {
                "messages": session.nodes,
                "currentId": session.last_id,
            },
        }
    }


def persist(
    client: httpx.Client, *, base: str, headers: Dict[str, str], session: OpenWebUISession,
) -> None:
    """POST the full accumulated chat document so the sidebar stays in sync."""
    if not session.chat_id:
        raise AdapterError("Open WebUI session missing chat_id")
    url = f"{base.rstrip('/')}/api/v1/chats/{session.chat_id}"
    try:
        resp = client.post(url, headers=headers, json=chat_document(session))
    except httpx.HTTPError as exc:
        raise AdapterError(f"Open WebUI chat persist failed: {exc}") from exc
    if resp.status_code >= 400:
        raise AdapterError(
            f"Open WebUI POST /api/v1/chats/{session.chat_id} "
            f"HTTP {resp.status_code}: {resp.text[:200]}"
        )


__all__ = [
    "OpenWebUISession",
    "is_openwebui_target",
    "base_url",
    "create_chat",
    "append_turn",
    "chat_document",
    "persist",
]
