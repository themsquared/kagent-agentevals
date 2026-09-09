"""Guards for the customer demo.

The demo's entire point is that one suite goes green on a good session and red
on each regression. If a fixture is edited and the demo quietly stops
demonstrating anything, these fail.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kagent_evals.report import exit_code
from kagent_evals.runner import run_suite
from kagent_evals.sources import build_source
from kagent_evals.suite import load_suite
from kagent_evals.trajectory import summarize_events, tool_call_names, event_stream_to_trajectory

ROOT = Path(__file__).parents[1]
DEMO = ROOT / "demo"
SUITE = DEMO / "suite.yaml"
GROUNDED = DEMO / "fixtures" / "weather-grounded.jsonl"
UNGROUNDED = DEMO / "fixtures" / "weather-ungrounded.jsonl"
WRONG_CITY = DEMO / "fixtures" / "weather-wrong-city.jsonl"


def _run(fixture: Path):
    return run_suite(load_suite(SUITE), build_source({"kind": "fixture", "path": str(fixture)}))


def _events(fixture: Path) -> list[dict]:
    return [json.loads(l) for l in fixture.read_text().splitlines() if l.strip()]


def test_grounded_session_passes_every_case():
    results = _run(GROUNDED)
    assert exit_code(results) == 0, [(r.case, r.error) for r in results]
    assert len(results) == 3


def test_ungrounded_session_is_caught():
    """The headline regression: the agent answered without calling the tool."""
    results = _run(UNGROUNDED)
    assert exit_code(results) == 1
    failed = {r.case for r in results if not r.passed}
    assert "answer-is-grounded-in-a-tool-call" in failed
    assert "looked-up-the-city-the-user-asked-about" in failed


def test_ungrounded_session_slips_past_the_tool_budget_check():
    """Demo talking point, pinned: a subset check cannot catch an agent that did
    nothing, because nothing is a subset of anything. If this ever starts
    failing, act 5's narration is wrong."""
    results = _run(UNGROUNDED)
    budget = next(r for r in results if r.case == "stayed-within-its-tool-budget")
    assert budget.passed
    assert budget.tools == []


def test_wrong_city_is_caught_only_by_the_strict_case():
    """Demo talking point, pinned: tool-name checks ignore arguments."""
    results = _run(WRONG_CITY)
    assert exit_code(results) == 1
    failed = {r.case for r in results if not r.passed}
    assert failed == {"looked-up-the-city-the-user-asked-about"}


def test_regressions_differ_from_the_good_session_in_the_advertised_way():
    grounded = event_stream_to_trajectory(_events(GROUNDED))
    ungrounded = event_stream_to_trajectory(_events(UNGROUNDED))
    wrong = event_stream_to_trajectory(_events(WRONG_CITY))

    assert tool_call_names(grounded) == ["get-weather-by-city_get-weather-by-city"]
    assert tool_call_names(ungrounded) == []
    assert tool_call_names(wrong) == tool_call_names(grounded)

    def city(trajectory):
        args = trajectory[1]["tool_calls"][0]["function"]["arguments"]
        return json.loads(args)["city"]

    assert city(grounded) == "London"
    assert city(wrong) == "Paris"


@pytest.mark.parametrize("fixture", [GROUNDED, UNGROUNDED, WRONG_CITY])
def test_demo_fixtures_carry_no_credentials(fixture: Path):
    raw = fixture.read_text()
    assert "eyJhbGci" not in raw, f"{fixture.name} contains a JWT"
    for placeholder in ('"authorization"', '"x-user-id"'):
        if placeholder in raw:
            assert "REDACTED" in raw or "00000000-0000-0000-0000-000000000000" in raw


def test_summary_explains_every_event():
    """--summary must account for all 14 events of the messy session, and name
    the adk_ tool it filtered."""
    events = _events(ROOT / "tests" / "fixtures" / "github-assistant-session.jsonl")
    summaries = summarize_events(events)
    assert len(summaries) == len(events) == 14
    assert sum(1 for s in summaries if s.kept) == 8

    filtered = [s for s in summaries if "filtered" in s.reason]
    assert len(filtered) == 2
    assert all("adk_request_confirmation" in s.reason for s in filtered)

    reasons = {s.reason for s in summaries if not s.kept}
    assert "no content" in reasons


def test_summary_reason_is_empty_exactly_when_the_event_contributed():
    events = _events(GROUNDED)
    for item in summarize_events(events):
        assert bool(item.parts) == item.kept
