# kagent-agentevals

Score [kagent](https://github.com/kagent-dev/kagent) agent trajectories with
LangChain's [agentevals](https://github.com/langchain-ai/agentevals).

[![ci](https://github.com/themsquared/kagent-agentevals/actions/workflows/ci.yaml/badge.svg)](https://github.com/themsquared/kagent-agentevals/actions/workflows/ci.yaml)

> 📖 **Read the write-up:** [Regression Tests for kagent Agents with agentevals](https://webofmike.com/kagent-trajectory-evals/)

agentevals is an offline scoring library: hand it a trajectory as a list of
OpenAI-format chat messages and it tells you whether the agent did the right
thing. kagent records its sessions as Google ADK events. This is the bridge —
plus a golden-suite runner so the scoring is repeatable and drops into CI.

It reads sessions kagent has already recorded. It does not modify your cluster,
your agents, or kagent itself.

## Try it in 30 seconds, no cluster needed

```bash
git clone https://github.com/themsquared/kagent-agentevals
cd kagent-agentevals
pip install -e .
kagent-evals run suites/quickstart.yaml
```

```
first-attempt-asks-the-user  [session fixture / inv e-525e129f-0]
  4 messages, tools: ask_user
  PASS  asked-instead-of-searching trajectory_subset_match      true

retry-searches-github  [session fixture / inv e-c51ee408-e]
  4 messages, tools: search_users
  PASS  calls-search-users       trajectory_superset_match    true

whole-session-tool-budget  [session fixture]
  8 messages, tools: ask_user, search_users
  PASS  no-unexpected-tools      trajectory_subset_match      true

retry-shape-and-args  [session fixture / inv e-c51ee408-e]
  4 messages, tools: search_users
  PASS  strict-with-args         trajectory_strict_match      true

4/4 cases passed
```

That suite scores a real captured session out of `tests/fixtures/`, so there is
nothing to install and nothing to stand up. Exit code is non-zero on failure.

## The demo

The quickstart above only ever goes green, which proves the plumbing works but
not that the evals are worth anything. `make demo` is the version that shows the
point: **one suite, three sessions, two caught regressions.**

```bash
make demo          # paused between acts, for talking over
make demo-fast     # straight through, to check it still works
```

Same three assertions throughout. Only the session changes:

| Session | Result |
| --- | --- |
| the real captured session | 3/3 pass |
| the agent answered without calling the tool | **caught** — exit 1 |
| the agent called the tool for the wrong city | **caught** — exit 1 |

The middle one is the failure mode that matters: a confident, plausible,
completely invented answer that still looks fine in the UI. See
[demo/README.md](demo/README.md) for the talk track and how to point it at your
own agent.

## What the conversion actually involves

kagent writes one row per ADK event, with the event JSON in `event.data`:

```json
{"author": "github_assistant",
 "invocation_id": "e-c51ee408-...",
 "content": {"role": "model",
             "parts": [{"function_call": {"id": "toolu_01...",
                                          "name": "search_users",
                                          "args": {"query": "themsquared"}}}]}}
```

`event_stream_to_trajectory()` turns a session's events into what agentevals
wants. Three things make this more than a field rename, all of them found by
running against real sessions rather than by reading the schema:

1. **Role comes from the part type, not from `author` or `content.role`.**
   kagent emits agent function calls and responses under `role: "user"` during
   human-in-the-loop confirmation round-trips. Trusting the role attributes the
   agent's own tool calls to the user.
2. **`adk_*` tool calls are runtime plumbing.** `adk_request_confirmation` is
   injected by ADK for HITL approval — it is not a decision the agent made, and
   left in it pollutes every trajectory comparison. Dropped by default via
   `drop_internal_tools`. `ask_user` *is* a real kagent tool, so it stays.
3. **Tool output arrives double-wrapped.** MCP tools return
   `{"content": [{"type": "text", "text": "..."}], "isError": false}`; builtin
   tools return `{"result": ...}`. Both are unwrapped so an LLM judge sees the
   payload instead of the envelope, and an `isError: true` result keeps a
   `[tool error]` prefix so a failed call still reads as a failure.

Contentless `system`/`header_update` events and streaming `partial` events are
skipped.

## Reading sessions from a cluster

Four interchangeable sources:

| kind | how | when |
| --- | --- | --- |
| `fixture` | a local JSONL file of ADK events | quickstart, CI, demos with no cluster |
| `kubectl` | `kubectl exec` into the kagent Postgres pod | laptop against a kind/k3d cluster; no port-forward, no token |
| `postgres` | `psycopg` + a DSN | in-cluster CronJob |
| `api` | the kagent controller REST API + bearer token | when you have a token |

Pick one with `--source` (or `--fixture <path>`, or `KAGENT_EVALS_SOURCE`).

```bash
# what is there to score?
kagent-evals sessions --agent weather

# what does the converter do with this session, and why is anything dropped?
kagent-evals extract <session-id> --summary

# split a session into invocations (one invocation = one user turn plus
# everything the agent did in response — usually the right unit to score)
kagent-evals extract <session-id> --list-invocations

# look at the converted trajectory
kagent-evals extract <session-id>

# capture a session as a fixture so it can be scored forever after
kagent-evals extract <session-id> --raw -o tests/fixtures/my-session.jsonl
```

## Writing a suite

```yaml
name: my-suite
source:
  kind: fixture
  path: tests/fixtures/my-session.jsonl
defaults:
  judge_model: anthropic:claude-haiku-4-5-20251001
cases:
  - name: uses-the-weather-tool
    session_id: fixture
    invocation_id: null          # omit to score the whole session
    evaluators:
      - type: tools_used
        mode: superset
        expected: [get_weather]
```

Three evaluator types:

- **`tools_used`** — assert on tool *names*. `superset` = "called at least
  these"; `subset` = "called nothing outside these". Arguments are always
  ignored, since the suite never supplies them.
- **`trajectory_match`** — compare against a full `reference` trajectory.
  `mode: strict` is the strongest form.
- **`llm_judge`** — grade with a model. Reference-free by default; supply a
  `reference` and it switches to the comparison prompt automatically. Set
  `continuous: true` for a 0–1 score plus a `threshold`.

**What the match modes actually compare.** Worth knowing before writing a suite,
because it is not what the names suggest:

- `subset` / `superset` / `unordered` inspect **tool calls only**. Roles, message
  order and text are ignored entirely.
- `strict` compares message count, roles and tool calls (names, and arguments
  per `tool_args_match_mode`). It does **not** compare assistant prose — so the
  `content` fields in a `strict` reference are documentation for the next person
  reading the suite, not assertions.

Nothing here asserts on the agent's wording. If you need that, use `llm_judge`.

Output: `--format table` (default), `--format junit` for CI, `--format json`,
`--show-trajectory` to print the messages that were scored.

## In-cluster

`deploy/cronjob.yaml` runs a suite on a schedule using the `postgres` source, so
the Job needs no RBAC on `pods/exec` — only network access to kagent's Postgres.
Build the image with the included `Dockerfile`. Replace the image reference, the
DSN secret and the placeholder session id before applying.

## Capturing your own fixtures

`kagent-evals extract --raw` writes the raw events, which include the request
headers kagent records on `header_update` events. **Read the file before you
commit it** — the CLI reminds you on every capture, and there is a test that
fails if a fixture in this repo contains a JWT. The checked-in fixture has its
`authorization` header and user id replaced with placeholders.

The converter never lifts anything out of `actions.state_delta` into a
trajectory, and a test pins that, so captured headers cannot leak into an LLM
judge prompt or a CI artifact. Scrub them anyway.

## Tests

```bash
pip install -e '.[dev]'
python -m pytest
```

42 tests, including guards that fail if the demo stops demonstrating anything.
The fixture is a real 14-event session captured from a live kagent
cluster, so the converter is pinned against data the runtime actually produces
rather than against a guess at the schema.

## Status

Validated against a live kagent cluster (kind/k3d, bundled Postgres):

- the `fixture` and `kubectl` sources; `sessions`, `extract` and `run`
- `tools_used` and `trajectory_match` in `strict`, `subset` and `superset` modes
- a negative-control suite, confirming failures fail and the exit code is
  non-zero
- `deploy/cronjob.yaml` via `kubectl apply --dry-run=client`

Not yet validated live:

- **`llm_judge` against a real model.** Prompt selection, scoring and thresholds
  are unit-tested, and a missing provider package surfaces as a per-evaluator
  error rather than a crash — but run `suites/demo-llm-judge.yaml` yourself
  before trusting it in CI.
- **the `api` source**, built from kagent's route table and response types
  rather than from a live call. It tolerates `event.data` arriving as either a
  JSON string or an inlined object, since that differs across builds.
- **the `postgres` source and the Docker image**, used by the CronJob.

## Licence

Apache-2.0
