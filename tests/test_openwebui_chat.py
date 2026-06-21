"""tests/test_openwebui_chat.py
================================
Unit tests for the Open WebUI sidebar-chat helpers (pure functions — no
network). The live HTTP round-trip was validated separately against a
running Open WebUI instance; here we lock the transcript-building logic that
makes the chat render correctly: parent/child links, ordering, and the full
document shape with assistant content filled in.
"""

from __future__ import annotations

from schemas.target_schema import TargetConfig
from external_eval import openwebui_chat as owui


def _target(endpoint="http://open-webui:8080/api/chat/completions", meta=None):
    return TargetConfig(id="t", name="t", type="api", endpoint=endpoint,
                        metadata=meta or {})


# ---------------------------------------------------------------------------
# Detection + URL derivation
# ---------------------------------------------------------------------------
def test_is_openwebui_by_metadata_flag():
    assert owui.is_openwebui_target(_target(endpoint="http://x/api/chat/completions",
                                            meta={"openwebui": True}))


def test_is_openwebui_by_endpoint_heuristic():
    assert owui.is_openwebui_target(_target())


def test_not_openwebui_for_plain_api():
    assert not owui.is_openwebui_target(_target(endpoint="https://api.openai.com/v1/chat"))


def test_base_url_strips_path():
    assert owui.base_url("http://open-webui:8080/api/chat/completions") == "http://open-webui:8080"


# ---------------------------------------------------------------------------
# Transcript building — the part the previous attempt got wrong
# ---------------------------------------------------------------------------
def test_append_turn_fills_assistant_content():
    s = owui.OpenWebUISession(model="qwen2.5:3b", title="UAIS Eval r1")
    owui.append_turn(s, "attack prompt", "model reply")

    assert len(s.order) == 2
    user_id, asst_id = s.order
    assert s.nodes[user_id]["role"] == "user"
    assert s.nodes[user_id]["content"] == "attack prompt"
    assert s.nodes[asst_id]["role"] == "assistant"
    # The reply MUST be persisted (Cursor's version left this empty).
    assert s.nodes[asst_id]["content"] == "model reply"
    assert s.nodes[asst_id]["done"] is True
    assert s.last_id == asst_id


def test_append_turn_links_parent_child_across_turns():
    s = owui.OpenWebUISession()
    owui.append_turn(s, "p1", "r1")
    u1, a1 = s.order
    owui.append_turn(s, "p2", "r2")
    u2, a2 = s.order[2], s.order[3]

    # user1 → asst1 → user2 → asst2 chain.
    assert s.nodes[u1]["childrenIds"] == [a1]
    assert a1 in s.nodes and u2 in s.nodes[a1]["childrenIds"]
    assert s.nodes[u2]["parentId"] == a1
    assert s.nodes[a2]["parentId"] == u2
    assert s.last_id == a2


def test_chat_document_shape():
    s = owui.OpenWebUISession(model="m", title="T")
    s.chat_id = "cid-1"
    owui.append_turn(s, "p", "r")
    doc = owui.chat_document(s)["chat"]

    assert doc["id"] == "cid-1"
    assert doc["title"] == "T"
    assert doc["models"] == ["m"]
    # Linear messages list mirrors history, in display order.
    assert [m["content"] for m in doc["messages"]] == ["p", "r"]
    assert doc["history"]["currentId"] == s.last_id
    assert set(doc["history"]["messages"]) == set(s.order)
