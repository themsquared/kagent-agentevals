"""Tests for suite parsing and evaluator wiring (no cluster, no LLM)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kagent_evals.runner import CaseResult, run_case, _reference_from_tool_names
from kagent_evals.suite import Case, EvaluatorSpec, load_suite

FIXTURE = Path(__file__).parent / "fixtures" / "github-assistant-session.jsonl"


class StubSource:
    """A SessionSource backed by the captured fixture."""

    def __init__(self, events: list[dict] | None = None) -> None:
        self._events = events if events is not None else [
            json.loads(l) for l in FIXTURE.read_text().splitlines() if l.strip()
        ]

    def list_sessions(self):
        return []

    def events(self, session_id: str):
        if session_id == "missing":
            return []
        if session_id == "boom":
            raise RuntimeError("connection refused")
        return self._events


def _case(**kwargs) -> Case:
    return Case(name="t", session_id="s", **kwargs)


def test_tools_used_superset_passes_when_tool_was_called():
    case = _case(
        evaluators=[EvaluatorSpec(type="tools_used", mode="superset", expected=["search_users"])]
    )
    result = run_case(case, StubSource())
    assert result.passed
    assert result.outcomes[0].key == "trajectory_superset_match"


def test_tools_used_superset_fails_for_uncalled_tool():
    case = _case(
        evaluators=[EvaluatorSpec(type="tools_used", mode="superset", expected=["delete_repo"])]
    )
    result = run_case(case, StubSource())
    assert not result.passed


def test_tools_used_subset_catches_an_unexpected_tool():
    """subset means: the agent called nothing outside the expected set."""
    case = _case(
        evaluators=[EvaluatorSpec(type="tools_used", mode="subset", expected=["search_users"])]
    )
    # ask_user was also called, so the real trajectory is not a subset.
    assert not run_case(case, StubSource()).passed


def test_invocation_scoping_narrows_the_trajectory():
    case = _case(
        invocation_id="e-c51ee408-e4c2-4bee-91aa-feb52996931c",
        evaluators=[EvaluatorSpec(type="tools_used", mode="subset", expected=["search_users"])],
    )
    result = run_case(case, StubSource())
    assert result.passed
    assert result.tools == ["search_users"]


def test_unknown_invocation_reports_available_ids():
    case = _case(
        invocation_id="nope",
        evaluators=[EvaluatorSpec(type="tools_used", mode="superset", expected=["x"])],
    )
    result = run_case(case, StubSource())
    assert not result.passed
    assert "not in session" in result.error
    assert "e-c51ee408" in result.error


def test_load_failure_is_reported_not_raised():
    case = Case(name="t", session_id="boom", evaluators=[
        EvaluatorSpec(type="tools_used", mode="superset", expected=["x"])
    ])
    result = run_case(case, StubSource())
    assert not result.passed
    assert "connection refused" in result.error


def test_empty_session_is_reported():
    case = Case(name="t", session_id="missing", evaluators=[
        EvaluatorSpec(type="tools_used", mode="superset", expected=["x"])
    ])
    assert "no events" in run_case(case, StubSource()).error


def test_reference_synthesis_produces_one_call_per_name():
    reference = _reference_from_tool_names(["a", "b"])
    assert [m["tool_calls"][0]["function"]["name"] for m in reference] == ["a", "b"]


# --- suite validation ----------------------------------------------------


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "s.yaml"
    path.write_text(body)
    return path


def test_suite_rejects_trajectory_match_without_reference(tmp_path):
    path = _write(tmp_path, """
name: s
cases:
  - name: c
    session_id: abc
    evaluators:
      - type: trajectory_match
        mode: strict
""")
    with pytest.raises(ValueError, match="needs a `reference`"):
        load_suite(path)


def test_suite_rejects_tools_used_with_strict_mode(tmp_path):
    path = _write(tmp_path, """
name: s
cases:
  - name: c
    session_id: abc
    evaluators:
      - type: tools_used
        mode: strict
        expected: [a]
""")
    with pytest.raises(ValueError, match="tools_used mode must be"):
        load_suite(path)


def test_suite_rejects_unknown_evaluator_type(tmp_path):
    path = _write(tmp_path, """
name: s
cases:
  - name: c
    session_id: abc
    evaluators:
      - type: vibes
""")
    with pytest.raises(ValueError, match="evaluator type must be one of"):
        load_suite(path)


def test_suite_requires_session_id(tmp_path):
    path = _write(tmp_path, """
name: s
cases:
  - name: c
    evaluators:
      - type: tools_used
        expected: [a]
""")
    with pytest.raises(ValueError, match="session_id is required"):
        load_suite(path)


def test_judge_model_falls_back_to_defaults(tmp_path):
    path = _write(tmp_path, """
name: s
defaults:
  judge_model: anthropic:claude-sonnet-5
cases:
  - name: c
    session_id: abc
    evaluators:
      - type: llm_judge
""")
    suite = load_suite(path)
    assert suite.cases[0].evaluators[0].model == "anthropic:claude-sonnet-5"


# --- llm_judge wiring (no network) ---------------------------------------


def test_judge_picks_reference_prompt_only_when_a_reference_is_given(monkeypatch):
    import agentevals.trajectory.llm as llm

    from kagent_evals.runner import _build_evaluator

    seen: dict = {}
    monkeypatch.setattr(
        llm,
        "create_trajectory_llm_as_judge",
        lambda **kw: seen.update(kw) or (lambda **_: {"key": "k", "score": 1.0}),
    )

    _build_evaluator(EvaluatorSpec(type="llm_judge", model="m"))
    assert seen["prompt"] is llm.TRAJECTORY_ACCURACY_PROMPT

    _build_evaluator(
        EvaluatorSpec(type="llm_judge", model="m", reference=[{"role": "user", "content": "x"}])
    )
    assert seen["prompt"] is llm.TRAJECTORY_ACCURACY_PROMPT_WITH_REFERENCE

    _build_evaluator(EvaluatorSpec(type="llm_judge", model="m", prompt="custom {outputs}"))
    assert seen["prompt"] == "custom {outputs}"


def test_evaluator_construction_failure_is_captured_per_evaluator():
    """A missing provider package must fail one evaluator, not the whole run."""
    case = Case(
        name="t",
        session_id="s",
        evaluators=[
            EvaluatorSpec(type="tools_used", mode="superset", expected=["search_users"]),
            EvaluatorSpec(type="llm_judge", model="not-a-real-provider:nope"),
        ],
    )
    result = run_case(case, StubSource())
    assert not result.passed
    assert result.outcomes[0].passed is True   # the offline evaluator still ran
    assert result.outcomes[1].error            # the judge reported why it could not


def test_continuous_score_uses_threshold():
    from kagent_evals.runner import _score_passed

    assert _score_passed(0.8, None) is True
    assert _score_passed(0.4, None) is False
    assert _score_passed(0.8, 0.9) is False
    assert _score_passed(True, None) is True
    assert _score_passed(False, 0.1) is False


# --- fixture source -------------------------------------------------------


def test_fixture_source_round_trips_the_quickstart_suite(tmp_path):
    """The quickstart suite must stay green with no cluster and no API key."""
    from kagent_evals.report import exit_code
    from kagent_evals.runner import run_suite
    from kagent_evals.sources import build_source

    suite = load_suite(Path(__file__).parents[1] / "suites" / "quickstart.yaml")
    source = build_source(
        {"kind": "fixture", "path": str(FIXTURE)}
    )
    results = run_suite(suite, source)
    assert exit_code(results) == 0, [
        (r.case, r.error, [(o.evaluator, o.score) for o in r.outcomes]) for r in results
    ]


def test_fixture_source_reports_a_missing_file(tmp_path):
    from kagent_evals.sources import build_source

    with pytest.raises(RuntimeError, match="fixture not found"):
        build_source({"kind": "fixture", "path": str(tmp_path / "nope.jsonl")})
