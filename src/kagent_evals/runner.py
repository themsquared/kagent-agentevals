"""Build agentevals evaluators from a suite and run them over kagent sessions."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from .sources import SessionSource
from .suite import Case, EvaluatorSpec, Suite
from .trajectory import event_stream_to_trajectory, split_by_invocation, tool_call_names

__all__ = ["CaseResult", "EvaluationOutcome", "run_case", "run_suite"]


@dataclass
class EvaluationOutcome:
    evaluator: str
    key: str
    score: Any
    passed: bool
    comment: str = ""
    error: str = ""


@dataclass
class CaseResult:
    case: str
    session_id: str
    invocation_id: str | None
    trajectory: list[dict[str, Any]] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)
    outcomes: list[EvaluationOutcome] = field(default_factory=list)
    error: str = ""

    @property
    def passed(self) -> bool:
        return not self.error and all(o.passed for o in self.outcomes)


def _reference_from_tool_names(names: list[str]) -> list[dict[str, Any]]:
    """Synthesise a reference trajectory that contains exactly these tool calls.

    subset/superset/unordered scoring only inspects tool calls, so a reference
    built from bare names is sufficient and lets a suite assert tool usage
    without transcribing a whole conversation.
    """
    return [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": f"ref-{index}",
                    "type": "function",
                    "function": {"name": name, "arguments": "{}"},
                }
            ],
        }
        for index, name in enumerate(names)
    ]


def _score_passed(score: Any, threshold: float | None) -> bool:
    if isinstance(score, bool):
        return score
    if isinstance(score, (int, float)):
        return score >= (threshold if threshold is not None else 0.5)
    return bool(score)


def _build_evaluator(spec: EvaluatorSpec) -> Callable[..., dict[str, Any]]:
    if spec.type in ("trajectory_match", "tools_used"):
        from agentevals.trajectory.match import create_trajectory_match_evaluator

        # tools_used only ever supplies tool *names*, so its synthesised
        # reference has empty args. Comparing args would fail every time, so the
        # mode is forced here rather than left to the caller or the YAML parser.
        args_mode = "ignore" if spec.type == "tools_used" else spec.tool_args_match_mode
        return create_trajectory_match_evaluator(
            trajectory_match_mode=spec.mode,
            tool_args_match_mode=args_mode,
        )

    if spec.type == "llm_judge":
        from agentevals.trajectory.llm import (
            TRAJECTORY_ACCURACY_PROMPT,
            TRAJECTORY_ACCURACY_PROMPT_WITH_REFERENCE,
            create_trajectory_llm_as_judge,
        )

        if spec.prompt:
            prompt = spec.prompt
        elif spec.reference:
            prompt = TRAJECTORY_ACCURACY_PROMPT_WITH_REFERENCE
        else:
            prompt = TRAJECTORY_ACCURACY_PROMPT
        return create_trajectory_llm_as_judge(
            prompt=prompt,
            model=spec.model,
            continuous=spec.continuous,
        )

    raise ValueError(f"unsupported evaluator type: {spec.type!r}")


def _evaluator_kwargs(spec: EvaluatorSpec, trajectory: list[dict[str, Any]]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"outputs": trajectory}
    if spec.type == "tools_used":
        kwargs["reference_outputs"] = _reference_from_tool_names(spec.expected)
    elif spec.reference:
        kwargs["reference_outputs"] = spec.reference
    return kwargs


def run_case(case: Case, source: SessionSource) -> CaseResult:
    """Load one session, build its trajectory, and run every evaluator on it."""
    result = CaseResult(
        case=case.name, session_id=case.session_id, invocation_id=case.invocation_id
    )

    try:
        events = source.events(case.session_id)
    except Exception as exc:
        result.error = f"failed to load session: {exc}"
        return result

    if not events:
        result.error = "session has no events"
        return result

    if case.invocation_id:
        grouped = split_by_invocation(events)
        if case.invocation_id not in grouped:
            result.error = (
                f"invocation {case.invocation_id!r} not in session "
                f"(available: {', '.join(k for k in grouped if k) or 'none'})"
            )
            return result
        events = grouped[case.invocation_id]

    result.trajectory = event_stream_to_trajectory(
        events,
        drop_internal_tools=case.drop_internal_tools,
        extra_tool_denylist=tuple(case.extra_tool_denylist),
    )
    result.tools = tool_call_names(result.trajectory)

    if not result.trajectory:
        result.error = "no scorable messages after conversion (all events were empty or filtered)"
        return result

    for spec in case.evaluators:
        try:
            evaluator = _build_evaluator(spec)
            raw = evaluator(**_evaluator_kwargs(spec, result.trajectory))
        except Exception as exc:
            result.outcomes.append(
                EvaluationOutcome(
                    evaluator=spec.label(), key="", score=None, passed=False, error=str(exc)
                )
            )
            continue

        score = raw.get("score") if isinstance(raw, dict) else raw
        result.outcomes.append(
            EvaluationOutcome(
                evaluator=spec.label(),
                key=(raw.get("key") if isinstance(raw, dict) else "") or spec.type,
                score=score,
                passed=_score_passed(score, spec.threshold),
                comment=(raw.get("comment") if isinstance(raw, dict) else "") or "",
            )
        )

    return result


def run_suite(suite: Suite, source: SessionSource) -> list[CaseResult]:
    return [run_case(case, source) for case in suite.cases]
