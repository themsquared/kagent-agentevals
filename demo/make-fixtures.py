#!/usr/bin/env python3
"""Derive the demo fixtures from one real captured kagent session.

The grounded fixture is a real session, scrubbed. The two regressions are
mechanical edits of it, so it is obvious what changed and why the suite catches
it. Re-run against your own capture:

    kagent-evals extract <session-id> --raw -o /tmp/session.jsonl
    python demo/make-fixtures.py /tmp/session.jsonl
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE / "fixtures"

# Request headers kagent records on header_update events. Never commit the real
# values: authorization carries the caller's bearer token.
SCRUB_HEADERS = {
    "authorization": "Bearer REDACTED",
    "x-user-id": "00000000-0000-0000-0000-000000000000",
    "x-kagent-parent-context-id": "00000000-0000-0000-0000-000000000000",
    "x-kagent-root-context-id": "00000000-0000-0000-0000-000000000000",
    "traceparent": "00-00000000000000000000000000000000-0000000000000000-01",
}


def scrub(event: dict) -> dict:
    headers = ((event.get("actions") or {}).get("state_delta") or {}).get("headers")
    if isinstance(headers, dict):
        for key, placeholder in SCRUB_HEADERS.items():
            if key in headers:
                headers[key] = placeholder
    return event


def parts_of(event: dict) -> list:
    content = event.get("content")
    if not isinstance(content, dict):
        return []
    return content.get("parts") or []


def has_kind(event: dict, kind: str) -> bool:
    return any(isinstance(p, dict) and p.get(kind) is not None for p in parts_of(event))


def write(name: str, events: list[dict], note: str) -> None:
    path = OUT / name
    path.write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in events) + "\n")
    print(f"  {name:<32} {len(events)} events  {note}")


def main(source: Path) -> int:
    events = [json.loads(l) for l in source.read_text().splitlines() if l.strip()]
    events = [scrub(e) for e in events]

    OUT.mkdir(exist_ok=True)
    print(f"from {source} ({len(events)} events)")

    # 1. Grounded: the real session, untouched apart from header scrubbing.
    write("weather-grounded.jsonl", events, "real capture, headers scrubbed")

    # 2. Ungrounded: strip the tool call and its response, and swap the final
    #    answer for one the agent could not have known. This is the failure mode
    #    that matters -- the agent stopped checking and started asserting.
    ungrounded = []
    for event in events:
        if has_kind(event, "function_call") or has_kind(event, "function_response"):
            continue
        event = json.loads(json.dumps(event))
        for part in parts_of(event):
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                if "London" in part["text"] and "°F" in part["text"]:
                    part["text"] = (
                        "Currently in **London**, it's around **68°F** with light cloud "
                        "and a gentle breeze. Typical for this time of year!"
                    )
        ungrounded.append(event)
    write("weather-ungrounded.jsonl", ungrounded, "tool call removed, answer invented")

    # 3. Wrong city: the tool was called, but for somewhere the user never asked
    #    about. Catches a class of bug that a tools_used check sails straight past.
    wrong = json.loads(json.dumps(events))
    for event in wrong:
        for part in parts_of(event):
            if not isinstance(part, dict):
                continue
            call = part.get("function_call")
            if isinstance(call, dict) and isinstance(call.get("args"), dict):
                if call["args"].get("city") == "London":
                    call["args"]["city"] = "Paris"
    write("weather-wrong-city.jsonl", wrong, "same tool, wrong argument")

    print("\nreminder: read any new fixture before committing it")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(main(Path(sys.argv[1])))
