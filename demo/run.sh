#!/usr/bin/env bash
# Narrated demo: one eval suite, three sessions, two caught regressions.
# No cluster, no API key, no network. Runs in about a minute with pauses.
#
#   ./demo/run.sh            # paused, for talking over
#   ./demo/run.sh --fast     # no pauses, for CI or a quick self-check
set -uo pipefail

cd "$(dirname "$0")/.."

PAUSE=1
[ "${1:-}" = "--fast" ] && PAUSE=0
[ "${DEMO_PAUSE:-1}" = "0" ] && PAUSE=0

if [ -t 1 ] && [ "${NO_COLOR:-}" = "" ]; then
  B=$'\033[1m'; DIM=$'\033[2m'; CYAN=$'\033[36m'; AMBER=$'\033[33m'; R=$'\033[0m'
else
  B=''; DIM=''; CYAN=''; AMBER=''; R=''
fi

EVALS="kagent-evals"
if ! command -v "$EVALS" >/dev/null 2>&1; then
  if [ -x .venv/bin/kagent-evals ]; then
    EVALS=".venv/bin/kagent-evals"
  else
    echo "kagent-evals is not installed. Run:  pip install -e ." >&2
    exit 1
  fi
fi

act () {
  echo
  echo "${CYAN}${B}── $1 ${R}"
  [ -n "${2:-}" ] && echo "${DIM}$2${R}"
  echo
}

say () { echo "${DIM}$1${R}"; }

run () {
  echo "${AMBER}\$ $*${R}"
  echo
  "$@"
  echo
}

beat () {
  [ "$PAUSE" = "1" ] || return 0
  printf '%s' "${DIM}   ↵ ${R}"
  read -r _ </dev/tty 2>/dev/null || true
}

# ---------------------------------------------------------------------------

act "1. What kagent actually stored" \
    "A real session: someone asked a weather agent about London."

run $EVALS extract fixture --fixture demo/fixtures/weather-grounded.jsonl --summary
say "Six ADK events, four of which carry anything. The tool call and its result are"
say "buried in content.parts as function_call and function_response, and the raw JSON"
say "serialises every optional field, so it is unreadable at full size. Nothing here"
say "can be scored directly."
beat

act "2. The same session, converted" \
    "agentevals wants OpenAI-format chat messages."

run $EVALS extract fixture --fixture demo/fixtures/weather-grounded.jsonl
say "Four messages: the question, the tool call with its arguments, the tool result,"
say "the grounded answer. That is a trajectory you can make assertions about."
beat

act "3. A messier session, handled the same way" \
    "Real agents are not this tidy. This one asked a clarifying question first,"$'\n'"which drags ADK's human-in-the-loop machinery into the transcript."

run $EVALS extract fixture --fixture tests/fixtures/github-assistant-session.jsonl --summary
say "Fourteen events down to eight messages. adk_request_confirmation is plumbing the"
say "runtime injected, not a decision the agent made, so it is filtered — left in, it"
say "would break every reference trajectory you ever write. Note events 3, 5 and 11:"
say "role says user, but they carry the agent's own tool calls. Trust the role and you"
say "score the wrong speaker."
beat

act "4. The contract" \
    "Back to the weather agent. Three assertions, in demo/suite.yaml. A weather answer must come from the"$'\n'"weather tool, the agent must stay inside its tool budget, and it must look up"$'\n'"the city the user actually asked about."

run $EVALS run demo/suite.yaml --fixture demo/fixtures/weather-grounded.jsonl
say "Green. Now the interesting part."
beat

act "5. The agent stops checking and starts asserting" \
    "Same question. Same suite. But this time the agent skipped the tool and"$'\n'"answered from memory — the failure mode that quietly ships to production."

run $EVALS run demo/suite.yaml --fixture demo/fixtures/weather-ungrounded.jsonl
say "Caught, and it exits non-zero, so CI stops the build."
say ""
say "Note the middle case still passes: the agent called nothing, and nothing is a"
say "subset of anything. A tool-budget check cannot catch an agent that did no work."
say "That is why the suite asserts in both directions."
beat

act "6. The tool ran, but for the wrong place" \
    "Now it calls the weather tool — for Paris, when the user asked about London."

run $EVALS run demo/suite.yaml --fixture demo/fixtures/weather-wrong-city.jsonl
say "Only the strict case fails. Both tool-name checks sail straight past it, because"
say "they compare names and ignore arguments. If arguments matter, assert on them."
beat

act "7. Where this lives" \
    "Every one of those runs read a JSONL fixture, so this is what CI runs on"$'\n'"every push — no cluster and no model spend."

say "Against a real cluster it is the same suite with one flag changed:"
echo
echo "${AMBER}\$ kagent-evals sessions${R}"
echo "${AMBER}\$ kagent-evals extract <session-id> --raw -o demo/fixtures/mine.jsonl${R}"
echo "${AMBER}\$ kagent-evals run demo/suite.yaml --fixture demo/fixtures/mine.jsonl${R}"
echo
say "Capture a session once and it becomes a permanent regression test for that"
say "agent's behaviour."
echo
