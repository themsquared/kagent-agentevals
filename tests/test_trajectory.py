"""Unit tests for the ADK-event -> agentevals-trajectory converter.

The fixture in tests/fixtures/ is a real 14-event session captured from a
live kagent cluster, so these tests pin the behaviour against
data the runtime actually produces rather than against a guess at the schema.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kagent_evals.trajectory import (
    event_stream_to_trajectory,
    split_by_invocation,
    tool_call_names,
)

FIXTURE = Path(__file__).parent / "fixtures" / "github-assistant-session.jsonl"


@pytest.fixture(scope="module")
def real_events() -> list[dict]:
    return [json.loads(line) for line in FIXTURE.read_text().splitlines() if line.strip()]


def _event(author: str, parts: list[dict], **extra) -> dict:
    role = "user" if author == "user" else "model"
    return {"author": author, "content": {"role": role, "parts": parts}, **extra}


def _call(name: str, args: dict, call_id: str = "c1") -> dict:
    return {"function_call": {"id": call_id, "name": name, "args": args}}


def _response(name: str, response, call_id: str = "c1") -> dict:
    return {"function_response": {"id": call_id, "name": name, "response": response}}


# --- real-session behaviour ------------------------------------------------


def test_real_session_shape(real_events):
    trajectory = event_stream_to_trajectory(real_events)
    assert [m["role"] for m in trajectory] == [
        "user", "assistant", "tool", "assistant", "user", "assistant", "tool", "assistant"
    ]
    assert tool_call_names(trajectory) == ["ask_user", "search_users"]


def test_adk_internal_tools_are_dropped_by_default(real_events):
    """adk_request_confirmation is HITL plumbing, not an agent decision."""
    names = tool_call_names(event_stream_to_trajectory(real_events))
    assert "adk_request_confirmation" not in names

    kept = tool_call_names(
        event_stream_to_trajectory(real_events, drop_internal_tools=False)
    )
    assert "adk_request_confirmation" in kept


def test_extra_denylist_drops_named_tool(real_events):
    names = tool_call_names(
        event_stream_to_trajectory(real_events, extra_tool_denylist=("ask_user",))
    )
    assert names == ["search_users"]


def test_mcp_envelope_is_unwrapped(real_events):
    trajectory = event_stream_to_trajectory(real_events)
    tool_message = next(
        m for m in trajectory if m.get("name") == "search_users"
    )
    # Raw form is {"content": [{"type": "text", "text": "..."}], "isError": false}
    assert "isError" not in tool_message["content"]
    assert json.loads(tool_message["content"])["total_count"] == 1


def test_invocation_split_matches_session(real_events):
    grouped = split_by_invocation(real_events)
    per_invocation = {
        inv: tool_call_names(event_stream_to_trajectory(evs))
        for inv, evs in grouped.items()
    }
    assert sorted(v for v in per_invocation.values() if v) == [["ask_user"], ["search_users"]]


# --- synthetic edge cases -------------------------------------------------


def test_role_comes_from_part_type_not_author():
    """kagent emits agent function calls under role/author "user" during HITL."""
    events = [
        _event("user", [_call("do_thing", {"a": 1})]),
        _event("user", [_response("do_thing", {"result": "ok"})]),
    ]
    trajectory = event_stream_to_trajectory(events)
    assert [m["role"] for m in trajectory] == ["assistant", "tool"]
    assert trajectory[0]["tool_calls"][0]["function"]["name"] == "do_thing"


def test_partial_events_are_skipped_by_default():
    events = [
        _event("agent", [{"text": "par"}], partial=True),
        _event("agent", [{"text": "partial then final"}]),
    ]
    assert [m["content"] for m in event_stream_to_trajectory(events)] == [
        "partial then final"
    ]
    assert len(event_stream_to_trajectory(events, include_partial=True)) == 2


def test_contentless_and_system_events_are_skipped():
    events = [
        {"author": "system", "invocation_id": "header_update"},
        {"author": "agent", "content": {"role": "model", "parts": []}},
        {"author": "agent", "content": None},
        _event("user", [{"text": "hello"}]),
    ]
    assert event_stream_to_trajectory(events) == [{"role": "user", "content": "hello"}]


def test_text_and_calls_in_one_event_merge_into_one_message():
    events = [_event("agent", [{"text": "let me check"}, _call("lookup", {})])]
    trajectory = event_stream_to_trajectory(events)
    assert len(trajectory) == 1
    assert trajectory[0]["content"] == "let me check"
    assert tool_call_names(trajectory) == ["lookup"]


def test_tool_args_are_json_encoded_deterministically():
    events = [_event("agent", [_call("t", {"b": 2, "a": 1})])]
    arguments = event_stream_to_trajectory(events)[0]["tool_calls"][0]["function"]["arguments"]
    assert arguments == '{"a": 1, "b": 2}'


def test_mcp_error_is_marked():
    events = [
        _event(
            "agent",
            [_response("t", {"content": [{"type": "text", "text": "boom"}], "isError": True})],
        )
    ]
    assert event_stream_to_trajectory(events)[0]["content"] == "[tool error] boom"


def test_empty_input_produces_empty_trajectory():
    assert event_stream_to_trajectory([]) == []


def test_state_delta_never_reaches_the_trajectory():
    """kagent stores request headers (including the caller's bearer token) in
    header_update events' actions.state_delta. Those events carry no content, so
    they must never contribute to a trajectory that gets shipped to an LLM judge
    or written to a CI artifact."""
    events = [
        {
            "author": "system",
            "invocation_id": "header_update",
            "actions": {"state_delta": {"headers": {"authorization": "Bearer secret-token"}}},
        },
        _event("user", [{"text": "hello"}]),
    ]
    trajectory = event_stream_to_trajectory(events)
    assert trajectory == [{"role": "user", "content": "hello"}]
    assert "secret-token" not in json.dumps(trajectory)


def test_fixture_carries_no_bearer_tokens():
    """Guards against re-capturing a fixture without scrubbing it."""
    raw = FIXTURE.read_text()
    assert "eyJhbGciOi" not in raw, "fixture contains a JWT; scrub before committing"
