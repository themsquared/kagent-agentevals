"""Convert kagent (Google ADK) session events into agentevals trajectories.

kagent stores one row per ADK event in the ``event`` table of its database,
with the raw ADK event JSON in ``event.data``. An ADK event looks like::

    {"author": "github_assistant",
     "invocation_id": "e-525e129f-...",
     "partial": null,
     "content": {"role": "model",
                 "parts": [{"function_call": {"id": "toolu_01...",
                                              "name": "search_users",
                                              "args": {"query": "themsquared"}}}]}}

agentevals wants a flat list of OpenAI-format chat messages. The mapping is
driven by *part type*, not by ``author`` or ``content.role``: kagent emits
function-call and function-response parts under ``role: "user"`` in several
cases (HITL confirmation round-trips in particular), so trusting the role
produces a trajectory with the tool calls attributed to the wrong speaker.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

__all__ = [
    "ADK_INTERNAL_PREFIX",
    "EventSummary",
    "event_stream_to_trajectory",
    "split_by_invocation",
    "summarize_events",
    "tool_call_names",
]

# ADK's own control-flow tools are injected by the runtime rather than chosen by
# the agent (adk_request_confirmation is the HITL approval round-trip). They are
# noise in a trajectory comparison, so they are dropped by default.
ADK_INTERNAL_PREFIX = "adk_"

# Envelope keys that wrap a tool's real payload. kagent's builtin tools return
# {"result": ...}; MCP tools (via kmcp) return the richer
# {"content": [{"type": "text", "text": "..."}], "isError": false}.
# Unwrapping these makes LLM-judge prompts far more legible.
_ENVELOPE_KEYS = ("result", "response", "output", "content")

# Prefix kept on unwrapped MCP payloads when isError is set, so a failed tool
# call still reads as a failure to an LLM judge instead of looking like data.
_ERROR_PREFIX = "[tool error] "


def _join_mcp_text(blocks: list[Any]) -> str | None:
    """Join an MCP ``content`` list of text blocks, or None if not that shape."""
    if not blocks or not all(
        isinstance(block, dict) and block.get("type") == "text" for block in blocks
    ):
        return None
    return "\n".join(str(block.get("text", "")) for block in blocks)


def _stringify_tool_output(response: Any, unwrap: bool) -> str:
    """Render a function_response payload as a string for a ``tool`` message."""
    value = response
    prefix = ""

    if unwrap and isinstance(value, dict):
        # MCP shape: {"content": [text blocks], "isError": bool}. Handle it before
        # the generic single-key unwrap, since isError makes it a two-key dict.
        if isinstance(value.get("content"), list):
            joined = _join_mcp_text(value["content"])
            if joined is not None and set(value) <= {"content", "isError"}:
                if value.get("isError"):
                    prefix = _ERROR_PREFIX
                value = joined
        # Generic single-key envelope from kagent's builtin tools.
        if isinstance(value, dict) and len(value) == 1:
            key = next(iter(value))
            if key in _ENVELOPE_KEYS:
                value = value[key]

    if unwrap and isinstance(value, list):
        joined = _join_mcp_text(value)
        if joined is not None:
            value = joined

    if isinstance(value, str):
        return prefix + value
    return prefix + json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )


def _is_internal(name: str, drop_internal: bool, extra_denylist: Sequence[str]) -> bool:
    if name in extra_denylist:
        return True
    return drop_internal and name.startswith(ADK_INTERNAL_PREFIX)


def _skip_event(event: dict[str, Any], include_partial: bool) -> bool:
    if not include_partial and event.get("partial"):
        # Streaming delta; the final aggregated event carries the same content.
        return True
    content = event.get("content")
    if not isinstance(content, dict):
        return True
    parts = content.get("parts")
    return not isinstance(parts, list) or not parts


class _AssistantBuffer:
    """Accumulates text and tool_calls from one event into one assistant message."""

    def __init__(self) -> None:
        self.text: list[str] = []
        self.tool_calls: list[dict[str, Any]] = []

    def empty(self) -> bool:
        return not self.text and not self.tool_calls

    def drain(self) -> dict[str, Any] | None:
        if self.empty():
            return None
        message: dict[str, Any] = {"role": "assistant", "content": "\n".join(self.text)}
        if self.tool_calls:
            message["tool_calls"] = self.tool_calls
        self.text = []
        self.tool_calls = []
        return message


def event_stream_to_trajectory(
    events: Iterable[dict[str, Any]],
    *,
    drop_internal_tools: bool = True,
    extra_tool_denylist: Sequence[str] = (),
    include_partial: bool = False,
    unwrap_tool_output: bool = True,
) -> list[dict[str, Any]]:
    """Convert chronologically ordered ADK events into an agentevals trajectory.

    Args:
        events: ADK event dicts, oldest first (``ORDER BY created_at ASC``).
        drop_internal_tools: Drop calls/responses for ``adk_*`` runtime tools.
        extra_tool_denylist: Additional tool names to drop, e.g. ``("ask_user",)``.
        include_partial: Keep streaming partial events. Off by default; leaving
            them in duplicates every assistant message.
        unwrap_tool_output: Unwrap MCP/kagent result envelopes in tool messages.

    Returns:
        A list of OpenAI-format messages suitable for any agentevals evaluator.
    """
    trajectory: list[dict[str, Any]] = []
    buffer = _AssistantBuffer()

    def flush() -> None:
        message = buffer.drain()
        if message is not None:
            trajectory.append(message)

    for event in events:
        if _skip_event(event, include_partial):
            continue

        author = event.get("author") or ""
        for part in event["content"]["parts"]:
            if not isinstance(part, dict):
                continue

            call = part.get("function_call")
            response = part.get("function_response")
            text = part.get("text")

            if isinstance(response, dict):
                name = response.get("name") or ""
                if _is_internal(name, drop_internal_tools, extra_tool_denylist):
                    continue
                flush()
                message: dict[str, Any] = {
                    "role": "tool",
                    "content": _stringify_tool_output(
                        response.get("response"), unwrap_tool_output
                    ),
                }
                if response.get("id"):
                    message["tool_call_id"] = response["id"]
                if name:
                    message["name"] = name
                trajectory.append(message)

            elif isinstance(call, dict):
                name = call.get("name") or ""
                if _is_internal(name, drop_internal_tools, extra_tool_denylist):
                    continue
                entry: dict[str, Any] = {
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(
                            call.get("args") or {}, ensure_ascii=False, sort_keys=True
                        ),
                    },
                }
                if call.get("id"):
                    entry["id"] = call["id"]
                buffer.tool_calls.append(entry)

            elif isinstance(text, str) and text.strip():
                if author == "user":
                    flush()
                    trajectory.append({"role": "user", "content": text})
                else:
                    buffer.text.append(text)

        flush()

    return trajectory


def split_by_invocation(
    events: Iterable[dict[str, Any]],
) -> "OrderedDict[str, list[dict[str, Any]]]":
    """Group events by ``invocation_id``, preserving first-seen order.

    One invocation is one user turn plus everything the agent did in response,
    which is usually the right unit to score.
    """
    grouped: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    for event in events:
        grouped.setdefault(event.get("invocation_id") or "", []).append(event)
    return grouped


def tool_call_names(trajectory: Sequence[dict[str, Any]]) -> list[str]:
    """The ordered tool-call names in a trajectory. Handy for assertions."""
    names: list[str] = []
    for message in trajectory:
        for call in message.get("tool_calls") or []:
            names.append(call.get("function", {}).get("name", ""))
    return names


@dataclass
class EventSummary:
    """One row of a human-readable account of what an event contributed."""

    index: int
    author: str
    role: str
    invocation_id: str
    parts: list[str]
    reason: str  # empty when the event contributed at least one message

    @property
    def kept(self) -> bool:
        return not self.reason


def summarize_events(
    events: Iterable[dict[str, Any]],
    *,
    drop_internal_tools: bool = True,
    extra_tool_denylist: Sequence[str] = (),
    include_partial: bool = False,
) -> list[EventSummary]:
    """Explain, event by event, what the converter does with a session.

    Raw ADK events serialise every optional field, so dumping them is unreadable.
    This is the view you actually want when working out why a trajectory looks
    the way it does. It reuses the converter's own skip predicates, so the two
    cannot drift apart.
    """
    summaries: list[EventSummary] = []

    for index, event in enumerate(events):
        author = str(event.get("author") or "")
        content = event.get("content")
        role = str(content.get("role") or "") if isinstance(content, dict) else ""
        invocation = str(event.get("invocation_id") or "")

        if _skip_event(event, include_partial):
            if not include_partial and event.get("partial"):
                reason = "streaming partial"
            elif not isinstance(content, dict):
                reason = "no content"
            else:
                reason = "no parts"
            summaries.append(
                EventSummary(index, author, role, invocation, [], reason)
            )
            continue

        parts: list[str] = []
        dropped_names: list[str] = []
        for part in content["parts"]:
            if not isinstance(part, dict):
                continue
            call = part.get("function_call")
            response = part.get("function_response")
            text = part.get("text")

            if isinstance(response, dict):
                name = response.get("name") or "?"
                if _is_internal(name, drop_internal_tools, extra_tool_denylist):
                    dropped_names.append(name)
                else:
                    parts.append(f"result {name}")
            elif isinstance(call, dict):
                name = call.get("name") or "?"
                if _is_internal(name, drop_internal_tools, extra_tool_denylist):
                    dropped_names.append(name)
                else:
                    parts.append(f"call {name}")
            elif isinstance(text, str) and text.strip():
                snippet = " ".join(text.split())
                if len(snippet) > 44:
                    snippet = snippet[:43] + "…"
                parts.append(f'text "{snippet}"')

        reason = ""
        if not parts:
            reason = (
                f"filtered: {', '.join(sorted(set(dropped_names)))}"
                if dropped_names
                else "nothing scorable"
            )
        summaries.append(EventSummary(index, author, role, invocation, parts, reason))

    return summaries
