UV := uv

.PHONY: help sync upgrade test lint format format-check build clean distclean check

help:
	@printf '%s\n' \
		'make sync         Install project and test dependencies' \
		'make upgrade      Upgrade locked dependencies and resync' \
		'make test         Run the test suite' \
		'make lint         Run Ruff lint checks' \
		'make format       Apply Ruff fixes and formatting' \
		'make format-check Check formatting without changing files' \
		'make build        Build source and wheel distributions' \
		'make clean        Remove generated files and caches' \
		'make distclean    Remove generated files and the virtual environment' \
		'make check        Run formatting, linting, tests, and build'

sync:
	$(UV) sync --group test

upgrade:
	$(UV) lock --upgrade
	$(UV) sync --group test

test:
	$(UV) run --group test pytest

lint:
	$(UV) run --group test ruff check .

format:
	$(UV) run --group test ruff check --fix .
	$(UV) run --group test ruff format .

format-check:
	$(UV) run --group test ruff format --check .

build: clean
	$(UV) build --no-sources

clean:
	rm -rf build dist .pytest_cache .ruff_cache .coverage htmlcov
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type f -name '*.py[co]' -delete

distclean: clean
	rm -rf .venv

check: format-check lint test build
