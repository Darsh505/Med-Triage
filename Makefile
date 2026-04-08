.PHONY: install test lint format play

install:
	pip install -e ".[dev]"
	pip install rich

test:
	pytest tests/ -v

lint:
	ruff check triage_env tests

format:
	usort format triage_env tests
	ruff format triage_env tests

play:
	python examples/play.py
