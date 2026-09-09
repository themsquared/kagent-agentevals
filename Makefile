.PHONY: help install test demo demo-fast quickstart fixtures

help:
	@echo "install     install the package in editable mode"
	@echo "test        run the unit tests"
	@echo "quickstart  score the checked-in session (no cluster needed)"
	@echo "demo        narrated customer demo: one suite, three sessions"
	@echo "demo-fast   the same demo with no pauses"
	@echo "fixtures    regenerate the demo fixtures from a raw capture"

install:
	pip install -e '.[dev]'

test:
	python -m pytest -q

quickstart:
	kagent-evals run suites/quickstart.yaml

demo:
	./demo/run.sh

demo-fast:
	./demo/run.sh --fast

# CAPTURE=/tmp/session.jsonl make fixtures
fixtures:
	python demo/make-fixtures.py $(CAPTURE)
