UV := uv
RUN := $(UV) run --locked --extra examples

.PHONY: help sync upgrade test test-cov lint typecheck format format-check build clean distclean check

help:
	@printf '%s\n' \
		'make sync         Install project, example, and development dependencies' \
		'make upgrade      Upgrade locked dependencies and resync' \
		'make test         Run the test suite' \
		'make test-cov     Run the test suite with coverage' \
		'make lint         Run Ruff lint checks' \
		'make typecheck    Run the advisory basedpyright check' \
		'make format       Apply Ruff fixes and formatting' \
		'make format-check Check formatting without changing files' \
		'make build        Build source and wheel distributions' \
		'make clean        Remove generated files and caches' \
		'make distclean    Remove generated files and the virtual environment' \
		'make check        Run all repository checks and build'

sync:
	$(UV) sync --locked --extra examples

upgrade:
	$(UV) lock --upgrade
	$(UV) sync --extra examples

test:
	$(RUN) pytest

test-cov:
	$(RUN) pytest --cov=jraph --cov-report=term-missing

lint:
	$(RUN) ruff check .

typecheck:
	$(RUN) basedpyright

format:
	$(RUN) ruff check --fix .
	$(RUN) ruff format .

format-check:
	$(RUN) ruff format --check .

build: clean
	$(UV) build --no-sources

clean:
	rm -rf build dist .pytest_cache .ruff_cache .coverage htmlcov
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type f -name '*.py[co]' -delete

distclean: clean
	rm -rf .venv

check:
	$(MAKE) format-check
	$(MAKE) lint
	$(MAKE) test
	$(MAKE) build
