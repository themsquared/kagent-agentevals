# The demo

One eval suite, three sessions, two caught regressions. No cluster, no API key,
no network. About a minute with pauses.

```bash
git clone https://github.com/themsquared/kagent-agentevals
cd kagent-agentevals
pip install -e .
make demo
```

`make demo` pauses between acts so you can talk over it. `make demo-fast` runs
straight through — use that to check it still works before you walk into a room.

## What it shows

| Act | Shows |
| --- | --- |
| 1 | What kagent stored: 6 ADK events, 4 of which carry anything |
| 2 | The same session as 4 OpenAI-format messages |
| 3 | A messier session: 14 events to 8 messages, HITL plumbing filtered |
| 4 | Three assertions against the good session — green |
| 5 | The agent answers without calling the tool — **caught**, exit 1 |
| 6 | The agent calls the tool for the wrong city — **caught**, exit 1 |
| 7 | The same suite against a live cluster |

The assertions never change between acts 4, 5 and 6. Only the session changes.
That is the whole idea: an eval suite is a contract you point at a session.

## The talk track

Lines worth actually saying, in order:

**Act 1** — "This is what the agent runtime wrote down. Six events, and two of
them are empty. The tool call is buried three levels deep in `content.parts`. If
you dump the raw JSON it's a wall of nulls, because ADK serialises every
optional field. So there's nothing here you can assert on yet."

**Act 2** — "Same session, converted. Four messages: the question, the tool call
with its arguments, the tool result, the answer. Now it's something you can
write a test against."

**Act 3** — "Real agents aren't that tidy. This one asked a clarifying question
first, which pulls ADK's human-in-the-loop machinery into the transcript.
Fourteen events, eight messages. `adk_request_confirmation` is plumbing the
runtime injected — the agent never chose it — so it gets filtered. Leave it in
and it breaks every reference trajectory you write. And look at events 3, 5 and
11: role says `user`, but they're carrying the agent's own tool calls. Trust the
role field and you'll score the wrong speaker."

**Act 4** — "Three assertions. A weather answer has to come from the weather
tool. The agent stays inside its tool budget. And it looks up the city the user
actually asked about. Green."

**Act 5** — *this is the one that lands* — "Now the same suite, same question,
but the agent skipped the tool and answered from memory. It's a confident,
plausible, completely made-up answer. That's the failure mode that ships to
production quietly, because the output still looks fine. Caught, and it exits
non-zero, so CI stops the build."

Then the honest bit: "Notice the middle check still passes. The agent called
nothing, and nothing is a subset of anything — a tool-budget check can't catch
an agent that did no work. That's why you assert in both directions."

**Act 6** — "Now it does call the tool, but for Paris, when the user asked about
London. Only the strict check catches it. Both tool-name checks sail straight
past, because they compare names and ignore arguments. If the arguments matter,
you have to say so."

**Act 7** — "All of that read a JSONL file, so it's what CI runs on every push:
no cluster, no model spend. Against a real cluster it's the same suite with one
flag changed — capture a session once and it becomes a permanent regression test
for that agent's behaviour."

## Where the fixtures came from

`weather-grounded.jsonl` is a real captured session from a live kagent cluster,
with the request headers scrubbed. The two regressions are mechanical edits of
it, produced by `make-fixtures.py` so it is obvious what changed:

- `weather-ungrounded.jsonl` — tool call and result removed, final answer
  replaced with one the agent could not have known.
- `weather-wrong-city.jsonl` — identical, except the tool argument says Paris.

To rebuild them from your own capture:

```bash
kagent-evals extract <session-id> --raw -o /tmp/session.jsonl
CAPTURE=/tmp/session.jsonl make fixtures
```

**Read any new capture before committing it.** kagent records inbound request
headers on `header_update` events, so a raw capture can contain the bearer token
of whoever was chatting. `make-fixtures.py` scrubs the ones it knows about, the
CLI warns you on every `--raw`, and `tests/test_demo.py` fails if a demo fixture
contains a JWT.

## Running it against your own agent

```bash
kagent-evals sessions --agent my-agent          # what is there to score
kagent-evals extract <session-id> --summary      # what the converter does with it
kagent-evals extract <session-id> --raw -o demo/fixtures/mine.jsonl
kagent-evals run demo/suite.yaml --fixture demo/fixtures/mine.jsonl
```

Then edit `demo/suite.yaml` so the expected tool names and arguments match your
agent, and you have a demo in your customer's own domain rather than the
weather.
