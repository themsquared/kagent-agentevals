"""Golden-suite definition: YAML in, validated dataclasses out."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

__all__ = ["Suite", "Case", "EvaluatorSpec", "load_suite"]

MATCH_MODES = ("strict", "unordered", "subset", "superset")
ARGS_MODES = ("exact", "ignore", "subset", "superset")
EVALUATOR_TYPES = ("trajectory_match", "tools_used", "llm_judge")


@dataclass
class EvaluatorSpec:
    type: str
    name: str = ""
    # trajectory_match / tools_used
    mode: str = "superset"
    tool_args_match_mode: str = "exact"
    reference: list[dict[str, Any]] = field(default_factory=list)
    expected: list[str] = field(default_factory=list)
    # llm_judge
    model: str = ""
    continuous: bool = False
    prompt: str = ""
    threshold: float | None = None

    def label(self) -> str:
        return self.name or self.type


@dataclass
class Case:
    name: str
    session_id: str
    invocation_id: str | None = None
    evaluators: list[EvaluatorSpec] = field(default_factory=list)
    drop_internal_tools: bool = True
    extra_tool_denylist: list[str] = field(default_factory=list)


@dataclass
class Suite:
    name: str
    cases: list[Case]
    source: dict[str, Any] = field(default_factory=dict)
    path: Path | None = None


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _parse_evaluator(raw: dict[str, Any], where: str, defaults: dict[str, Any]) -> EvaluatorSpec:
    _require(isinstance(raw, dict), f"{where}: each evaluator must be a mapping")
    kind = raw.get("type")
    _require(
        kind in EVALUATOR_TYPES,
        f"{where}: evaluator type must be one of {EVALUATOR_TYPES}, got {kind!r}",
    )

    spec = EvaluatorSpec(
        type=kind,
        name=raw.get("name", ""),
        mode=raw.get("mode", "superset" if kind == "tools_used" else "strict"),
        tool_args_match_mode=raw.get(
            "tool_args_match_mode",
            "ignore" if kind == "tools_used" else defaults.get("tool_args_match_mode", "exact"),
        ),
        reference=list(raw.get("reference") or []),
        expected=list(raw.get("expected") or []),
        model=raw.get("model") or defaults.get("judge_model", ""),
        continuous=bool(raw.get("continuous", False)),
        prompt=raw.get("prompt", ""),
        threshold=raw.get("threshold"),
    )

    if kind in ("trajectory_match", "tools_used"):
        _require(
            spec.mode in MATCH_MODES,
            f"{where}: mode must be one of {MATCH_MODES}, got {spec.mode!r}",
        )
        _require(
            spec.tool_args_match_mode in ARGS_MODES,
            f"{where}: tool_args_match_mode must be one of {ARGS_MODES}",
        )
    if kind == "trajectory_match":
        _require(bool(spec.reference), f"{where}: trajectory_match needs a `reference` trajectory")
    if kind == "tools_used":
        _require(bool(spec.expected), f"{where}: tools_used needs a non-empty `expected` list")
        _require(
            spec.mode in ("subset", "superset", "unordered"),
            f"{where}: tools_used mode must be subset, superset or unordered "
            f"(strict compares whole messages, which a tool-name list cannot express)",
        )
    if kind == "llm_judge":
        _require(bool(spec.model), f"{where}: llm_judge needs `model` or defaults.judge_model")
    return spec


def load_suite(path: str | Path) -> Suite:
    """Load and validate a suite YAML file."""
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("load_suite needs PyYAML: pip install pyyaml") from exc

    path = Path(path)
    raw = yaml.safe_load(path.read_text()) or {}
    _require(isinstance(raw, dict), f"{path}: top level must be a mapping")

    defaults = raw.get("defaults") or {}
    _require(isinstance(defaults, dict), f"{path}: defaults must be a mapping")

    raw_cases = raw.get("cases") or []
    _require(isinstance(raw_cases, list) and raw_cases, f"{path}: needs a non-empty `cases` list")

    cases: list[Case] = []
    for index, raw_case in enumerate(raw_cases):
        where = f"{path}: cases[{index}]"
        _require(isinstance(raw_case, dict), f"{where} must be a mapping")
        name = raw_case.get("name") or f"case-{index}"
        session_id = raw_case.get("session_id")
        _require(bool(session_id), f"{where} ({name}): session_id is required")

        raw_evaluators = raw_case.get("evaluators") or []
        _require(
            isinstance(raw_evaluators, list) and raw_evaluators,
            f"{where} ({name}): needs a non-empty `evaluators` list",
        )
        cases.append(
            Case(
                name=name,
                session_id=str(session_id),
                invocation_id=raw_case.get("invocation_id"),
                evaluators=[
                    _parse_evaluator(e, f"{where} ({name})", defaults) for e in raw_evaluators
                ],
                drop_internal_tools=bool(
                    raw_case.get(
                        "drop_internal_tools", defaults.get("drop_internal_tools", True)
                    )
                ),
                extra_tool_denylist=list(
                    raw_case.get(
                        "extra_tool_denylist", defaults.get("extra_tool_denylist", [])
                    )
                ),
            )
        )

    return Suite(
        name=raw.get("name") or path.stem,
        cases=cases,
        source=raw.get("source") or {},
        path=path,
    )
