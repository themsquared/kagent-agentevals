"""kagent-evals: score kagent agent trajectories with agentevals."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .report import exit_code, render_json, render_junit, render_table
from .runner import run_suite
from .sources import build_source
from .suite import load_suite
from .trajectory import event_stream_to_trajectory, split_by_invocation, tool_call_names


def _source_overrides(args: argparse.Namespace) -> dict[str, object]:
    spec: dict[str, object] = {}
    if getattr(args, "fixture", None):
        spec["kind"] = "fixture"
        spec["path"] = args.fixture
    if getattr(args, "source", None):
        spec["kind"] = args.source
    if getattr(args, "namespace", None):
        spec["namespace"] = args.namespace
    if getattr(args, "context", None):
        spec["context"] = args.context
    return spec


def _add_source_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--source",
        choices=("fixture", "kubectl", "postgres", "api"),
        help="where to read sessions from (default: kubectl, or KAGENT_EVALS_SOURCE)",
    )
    parser.add_argument(
        "--fixture",
        help="path to a JSONL file of ADK events; implies --source fixture",
    )
    parser.add_argument("--namespace", help="kagent namespace (default: kagent)")
    parser.add_argument("--context", help="kubeconfig context")


def cmd_sessions(args: argparse.Namespace) -> int:
    source = build_source(_source_overrides(args))
    sessions = source.list_sessions()
    if args.agent:
        sessions = [s for s in sessions if args.agent in s.agent_id]
    if not sessions:
        print("no sessions found", file=sys.stderr)
        return 1
    print(f"{'SESSION ID':<38} {'EVENTS':>6}  AGENT")
    for info in sessions[: args.limit]:
        count = "?" if info.event_count < 0 else str(info.event_count)
        print(f"{info.id:<38} {count:>6}  {info.agent_id}")
    return 0


def cmd_extract(args: argparse.Namespace) -> int:
    source = build_source(_source_overrides(args))
    events = source.events(args.session_id)
    if not events:
        print(f"session {args.session_id} has no events", file=sys.stderr)
        return 1

    if args.list_invocations:
        for invocation, group in split_by_invocation(events).items():
            trajectory = event_stream_to_trajectory(group)
            print(
                f"{invocation or '(none)':<40} events={len(group):<4} "
                f"messages={len(trajectory):<4} tools={','.join(tool_call_names(trajectory)) or '-'}"
            )
        return 0

    if args.invocation_id:
        grouped = split_by_invocation(events)
        if args.invocation_id not in grouped:
            print(f"invocation {args.invocation_id} not found", file=sys.stderr)
            return 1
        events = grouped[args.invocation_id]

    if args.raw:
        lines = "\n".join(json.dumps(e, ensure_ascii=False) for e in events)
        if args.output:
            Path(args.output).write_text(lines + "\n")
            print(f"wrote {len(events)} raw events to {args.output}", file=sys.stderr)
            print(
                "review it for credentials before committing: kagent stores request "
                "headers in actions.state_delta",
                file=sys.stderr,
            )
        else:
            print(lines)
        return 0

    trajectory = event_stream_to_trajectory(
        events,
        drop_internal_tools=not args.include_internal,
        extra_tool_denylist=tuple(args.drop_tool or ()),
    )
    payload = json.dumps(trajectory, indent=2, ensure_ascii=False)
    if args.output:
        Path(args.output).write_text(payload + "\n")
        print(f"wrote {len(trajectory)} messages to {args.output}", file=sys.stderr)
    else:
        print(payload)
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    suite = load_suite(args.suite)
    overrides = _source_overrides(args)
    source = build_source({**suite.source, **overrides})
    results = run_suite(suite, source)

    if args.format == "json":
        print(render_json(results, show_trajectory=args.show_trajectory))
    elif args.format == "junit":
        print(render_junit(results, suite_name=suite.name))
    else:
        print(render_table(results, show_trajectory=args.show_trajectory))
    return exit_code(results)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kagent-evals",
        description="Score kagent agent trajectories with agentevals.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_sessions = sub.add_parser("sessions", help="list sessions available to score")
    _add_source_args(p_sessions)
    p_sessions.add_argument("--agent", help="substring filter on agent id")
    p_sessions.add_argument("--limit", type=int, default=50)
    p_sessions.set_defaults(func=cmd_sessions)

    p_extract = sub.add_parser(
        "extract", help="convert a session to an agentevals trajectory and print it"
    )
    _add_source_args(p_extract)
    p_extract.add_argument("session_id")
    p_extract.add_argument("--invocation-id", help="score just this invocation (one user turn)")
    p_extract.add_argument(
        "--list-invocations", action="store_true", help="list invocations instead of converting"
    )
    p_extract.add_argument(
        "--include-internal", action="store_true", help="keep adk_* runtime tool calls"
    )
    p_extract.add_argument(
        "--drop-tool", action="append", help="also drop this tool name (repeatable)"
    )
    p_extract.add_argument(
        "--raw",
        action="store_true",
        help="emit the raw ADK events as JSONL instead of a converted trajectory "
        "(use this to capture a fixture)",
    )
    p_extract.add_argument("-o", "--output", help="write output here instead of stdout")
    p_extract.set_defaults(func=cmd_extract)

    p_run = sub.add_parser("run", help="run a golden suite")
    _add_source_args(p_run)
    p_run.add_argument("suite", help="path to a suite YAML file")
    p_run.add_argument("--format", choices=("table", "json", "junit"), default="table")
    p_run.add_argument("--show-trajectory", action="store_true")
    p_run.set_defaults(func=cmd_run)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
