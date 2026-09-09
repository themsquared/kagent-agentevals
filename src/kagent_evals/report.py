"""Render evaluation results as a terminal table, JSON, or JUnit XML."""

from __future__ import annotations

import json
from typing import Sequence
from xml.etree import ElementTree as ET

from .runner import CaseResult

__all__ = ["render_table", "render_json", "render_junit", "exit_code"]

_PASS = "PASS"
_FAIL = "FAIL"


def _score_text(score: object) -> str:
    if isinstance(score, bool):
        return "true" if score else "false"
    if isinstance(score, float):
        return f"{score:.2f}"
    return "-" if score is None else str(score)


def render_table(results: Sequence[CaseResult], *, show_trajectory: bool = False) -> str:
    lines: list[str] = []
    for result in results:
        header = f"{result.case}  [session {result.session_id[:8]}"
        if result.invocation_id:
            header += f" / inv {result.invocation_id[:12]}"
        header += "]"
        lines.append(header)

        if result.error:
            lines.append(f"  {_FAIL}  {result.error}")
            lines.append("")
            continue

        lines.append(
            f"  {len(result.trajectory)} messages, "
            f"tools: {', '.join(result.tools) if result.tools else '(none)'}"
        )
        if show_trajectory:
            for message in result.trajectory:
                calls = ",".join(
                    call["function"]["name"] for call in message.get("tool_calls") or []
                )
                body = (message.get("content") or "").replace("\n", " ")[:60]
                marker = f"[{calls}] " if calls else ""
                lines.append(f"    {message['role']:<9} {marker}{body}")

        for outcome in result.outcomes:
            status = _PASS if outcome.passed else _FAIL
            line = f"  {status}  {outcome.evaluator:<24} {outcome.key:<28} {_score_text(outcome.score)}"
            lines.append(line)
            if outcome.error:
                lines.append(f"          error: {outcome.error}")
            elif outcome.comment:
                comment = outcome.comment.strip().replace("\n", " ")
                lines.append(f"          {comment[:160]}")
        lines.append("")

    passed = sum(1 for r in results if r.passed)
    lines.append(f"{passed}/{len(results)} cases passed")
    return "\n".join(lines)


def render_json(results: Sequence[CaseResult], *, show_trajectory: bool = False) -> str:
    payload = []
    for result in results:
        entry = {
            "case": result.case,
            "session_id": result.session_id,
            "invocation_id": result.invocation_id,
            "passed": result.passed,
            "error": result.error,
            "message_count": len(result.trajectory),
            "tools": result.tools,
            "evaluations": [
                {
                    "evaluator": o.evaluator,
                    "key": o.key,
                    "score": o.score,
                    "passed": o.passed,
                    "comment": o.comment,
                    "error": o.error,
                }
                for o in result.outcomes
            ],
        }
        if show_trajectory:
            entry["trajectory"] = result.trajectory
        payload.append(entry)
    return json.dumps(payload, indent=2, ensure_ascii=False)


def render_junit(results: Sequence[CaseResult], *, suite_name: str = "kagent-evals") -> str:
    total_failures = sum(
        1 for r in results for o in r.outcomes if not o.passed
    ) + sum(1 for r in results if r.error)
    suite = ET.Element(
        "testsuite",
        name=suite_name,
        tests=str(sum(max(len(r.outcomes), 1) for r in results)),
        failures=str(total_failures),
    )
    for result in results:
        if result.error:
            case = ET.SubElement(suite, "testcase", classname=result.case, name="load")
            ET.SubElement(case, "failure", message=result.error).text = result.error
            continue
        for outcome in result.outcomes:
            case = ET.SubElement(suite, "testcase", classname=result.case, name=outcome.evaluator)
            if not outcome.passed:
                detail = outcome.error or outcome.comment or f"score={outcome.score}"
                ET.SubElement(case, "failure", message=f"score={outcome.score}").text = detail
    return ET.tostring(suite, encoding="unicode")


def exit_code(results: Sequence[CaseResult]) -> int:
    return 0 if all(r.passed for r in results) else 1
