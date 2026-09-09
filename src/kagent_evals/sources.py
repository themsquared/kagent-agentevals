"""Load kagent session events from a cluster.

Four loaders, same interface:

* :class:`FixtureSource`  -- a local JSONL file of ADK events. No cluster at
  all, which is what the quickstart suite uses.
* :class:`KubectlSource`  -- ``kubectl exec`` into the kagent Postgres pod. No
  Python DB driver, no port-forward, no OIDC token. This is the one that works
  on a laptop against a k3d/kind demo cluster with nothing else set up.
* :class:`PostgresSource` -- a real DSN via ``psycopg``. Use this in-cluster.
* :class:`ApiSource`      -- the kagent controller REST API. Needs a bearer
  token when the enterprise OIDC middleware is enabled.

All three return raw ADK event dicts in chronological order, ready for
:func:`kagent_evals.trajectory.event_stream_to_trajectory`.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Protocol

__all__ = [
    "SessionSource",
    "FixtureSource",
    "KubectlSource",
    "PostgresSource",
    "ApiSource",
    "build_source",
]

_EVENTS_SQL = (
    "select data from event where session_id = %(sid)s "
    "and deleted_at is null order by created_at asc"
)
_SESSIONS_SQL = (
    "select s.id, coalesce(s.name, ''), coalesce(s.agent_id, ''), count(e.id) "
    "from session s left join event e "
    "on e.session_id = s.id and e.deleted_at is null "
    "where s.deleted_at is null "
    "group by s.id, s.name, s.agent_id order by count(e.id) desc"
)


@dataclass(frozen=True)
class SessionInfo:
    id: str
    name: str
    agent_id: str
    event_count: int


class SessionSource(Protocol):
    def list_sessions(self) -> list[SessionInfo]: ...
    def events(self, session_id: str) -> list[dict[str, Any]]: ...


def _parse_events(rows: list[str]) -> list[dict[str, Any]]:
    """Parse each row of event.data JSON, skipping blanks."""
    events: list[dict[str, Any]] = []
    for row in rows:
        row = row.strip()
        if not row:
            continue
        events.append(json.loads(row))
    return events


class KubectlSource:
    """Reads events by running psql inside the kagent Postgres pod."""

    def __init__(
        self,
        *,
        namespace: str = "kagent",
        workload: str = "deploy/kagent-postgresql",
        user: str = "kagent",
        database: str = "kagent",
        password: str | None = None,
        context: str | None = None,
    ) -> None:
        if shutil.which("kubectl") is None:
            raise RuntimeError("kubectl not found on PATH")
        self.namespace = namespace
        self.workload = workload
        self.user = user
        self.database = database
        # Defaults to the chart's bundled-Postgres convention (user == password).
        self.password = password if password is not None else user
        self.context = context

    def _psql(self, sql: str) -> list[str]:
        cmd = ["kubectl"]
        if self.context:
            cmd += ["--context", self.context]
        cmd += [
            "exec", "-n", self.namespace, self.workload, "--",
            "env", f"PGPASSWORD={self.password}",
            "psql", "-U", self.user, "-d", self.database,
            # -t no header, -A unaligned, -F record separator
            "-t", "-A", "-F", "\x1f", "-c", sql,
        ]
        done = subprocess.run(cmd, capture_output=True, text=True)
        if done.returncode != 0:
            raise RuntimeError(f"kubectl exec psql failed: {done.stderr.strip()}")
        return done.stdout.splitlines()

    def list_sessions(self) -> list[SessionInfo]:
        # psycopg-style placeholders are not available here; inline the static query.
        rows = self._psql(_SESSIONS_SQL.replace("%(sid)s", "''"))
        out: list[SessionInfo] = []
        for row in rows:
            if not row.strip():
                continue
            parts = row.split("\x1f")
            if len(parts) != 4:
                continue
            out.append(SessionInfo(parts[0], parts[1], parts[2], int(parts[3] or 0)))
        return out

    def events(self, session_id: str) -> list[dict[str, Any]]:
        if "'" in session_id:
            raise ValueError(f"refusing suspicious session id: {session_id!r}")
        sql = _EVENTS_SQL.replace("%(sid)s", f"'{session_id}'")
        return _parse_events(self._psql(sql))


class PostgresSource:
    """Reads events over a real Postgres connection. Requires ``psycopg``."""

    def __init__(self, dsn: str) -> None:
        try:
            import psycopg  # noqa: F401
        except ImportError as exc:  # pragma: no cover - depends on extras
            raise RuntimeError(
                "PostgresSource needs psycopg: pip install 'kagent-agentevals[postgres]'"
            ) from exc
        self.dsn = dsn

    def _query(self, sql: str, params: dict[str, Any] | None = None) -> list[tuple]:
        import psycopg

        with psycopg.connect(self.dsn) as conn, conn.cursor() as cur:
            cur.execute(sql, params or {})
            return cur.fetchall()

    def list_sessions(self) -> list[SessionInfo]:
        return [
            SessionInfo(str(r[0]), str(r[1]), str(r[2]), int(r[3]))
            for r in self._query(_SESSIONS_SQL)
        ]

    def events(self, session_id: str) -> list[dict[str, Any]]:
        rows = self._query(_EVENTS_SQL, {"sid": session_id})
        return _parse_events([r[0] if isinstance(r[0], str) else json.dumps(r[0]) for r in rows])


class ApiSource:
    """Reads events from the kagent controller REST API.

    Endpoints (from kagent's httpserver route table):
      ``GET /api/sessions?user_id=...``            -> sessions
      ``GET /api/sessions/{id}?order=asc&user_id=...`` -> {session, events}

    Responses are wrapped as ``{"error": bool, "data": ..., "message": str}``.
    Each event carries its ADK payload in ``data``, which older builds serialise
    as a JSON *string* and newer ones inline as an object; both are handled.
    """

    def __init__(
        self,
        base_url: str,
        *,
        user_id: str = "admin@kagent.dev",
        token: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.user_id = user_id
        self.token = token
        self.timeout = timeout

    def _get(self, path: str, params: dict[str, str]) -> Any:
        import urllib.parse
        import urllib.request

        query = urllib.parse.urlencode({**params, "user_id": self.user_id})
        request = urllib.request.Request(f"{self.base_url}{path}?{query}")
        if self.token:
            request.add_header("Authorization", f"Bearer {self.token}")
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            body = json.loads(response.read().decode())
        if isinstance(body, dict) and body.get("error"):
            raise RuntimeError(f"kagent API error: {body.get('message')}")
        # Unwrap StandardResponse; tolerate an already-unwrapped payload.
        return body.get("data", body) if isinstance(body, dict) else body

    def list_sessions(self) -> list[SessionInfo]:
        payload = self._get("/api/sessions", {})
        sessions = payload if isinstance(payload, list) else payload.get("sessions", [])
        return [
            SessionInfo(
                str(s.get("id", "")),
                str(s.get("name") or ""),
                str(s.get("agent_id") or ""),
                -1,  # not reported by the list endpoint
            )
            for s in sessions
        ]

    def events(self, session_id: str) -> list[dict[str, Any]]:
        payload = self._get(f"/api/sessions/{session_id}", {"order": "asc"})
        raw = payload.get("events", []) if isinstance(payload, dict) else []
        events: list[dict[str, Any]] = []
        for item in raw:
            data = item.get("data") if isinstance(item, dict) else item
            if isinstance(data, str):
                data = json.loads(data)
            if isinstance(data, dict):
                events.append(data)
        return events



class FixtureSource:
    """Reads events from a local JSONL file: one ADK event per line.

    This is what makes the tool runnable with no cluster — handy for CI, for
    demoing on a plane, and for regression-testing the converter against a
    session you captured once. Produce a file with::

        kagent-evals extract <session-id> --raw -o session.jsonl
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise RuntimeError(f"fixture not found: {self.path}")

    def _load(self) -> list[dict[str, Any]]:
        return _parse_events(self.path.read_text().splitlines())

    def list_sessions(self) -> list[SessionInfo]:
        events = self._load()
        return [SessionInfo(self.path.stem, self.path.name, "(fixture)", len(events))]

    def events(self, session_id: str) -> list[dict[str, Any]]:
        # A fixture holds one session; the id is accepted for interface parity.
        return self._load()


def build_source(spec: dict[str, Any] | None = None) -> SessionSource:
    """Build a source from a suite's ``source:`` block, with env fallbacks.

    ``kind`` is one of ``kubectl`` (default), ``postgres``, ``api``.
    """
    spec = dict(spec or {})
    kind = spec.pop("kind", None) or os.environ.get("KAGENT_EVALS_SOURCE", "kubectl")

    if kind == "fixture":
        path = spec.get("path") or os.environ.get("KAGENT_EVALS_FIXTURE")
        if not path:
            raise RuntimeError("fixture source needs source.path or KAGENT_EVALS_FIXTURE")
        return FixtureSource(path)
    if kind == "kubectl":
        return KubectlSource(
            namespace=spec.get("namespace", os.environ.get("KAGENT_NAMESPACE", "kagent")),
            workload=spec.get("workload", "deploy/kagent-postgresql"),
            user=spec.get("user", "kagent"),
            database=spec.get("database", "kagent"),
            password=spec.get("password") or os.environ.get("KAGENT_DB_PASSWORD"),
            context=spec.get("context") or os.environ.get("KAGENT_CONTEXT"),
        )
    if kind == "postgres":
        dsn = spec.get("dsn") or os.environ.get("KAGENT_DB_DSN")
        if not dsn:
            raise RuntimeError("postgres source needs source.dsn or KAGENT_DB_DSN")
        return PostgresSource(dsn)
    if kind == "api":
        base = spec.get("base_url") or os.environ.get("KAGENT_URL")
        if not base:
            raise RuntimeError("api source needs source.base_url or KAGENT_URL")
        return ApiSource(
            base,
            user_id=spec.get("user_id", os.environ.get("KAGENT_USER_ID", "admin@kagent.dev")),
            token=spec.get("token") or os.environ.get("KAGENT_TOKEN"),
        )
    raise ValueError(f"unknown source kind: {kind!r}")
