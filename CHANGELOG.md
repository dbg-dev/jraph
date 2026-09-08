# Changelog

All notable changes to this independent continuation of Jraph are documented
in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- Flax NNX and Optax implementations of the Zachary's karate club, e-voting,
  SAT and Higgs-detection examples.
- A two-layer NNX graph convolutional network and training workflow for the Cora
  citation dataset.
- Shared training, evaluation, graph-preparation and padding utilities for the
  Open Graph Benchmark examples.
- Pytest coverage for the core library, experimental sharding and retained
  examples.
- Ruff formatting and linting, basedpyright as an advisory check, Make targets
  for common development tasks, and a unified CI check.

### Changed

- Migrated packaging and dependency management from `setup.py` to
  `pyproject.toml`, uv and the uv build backend.
- Updated the project to Python 3.14 and current JAX APIs.
- Reorganised the package around a private `jraph._src` implementation while
  preserving the public `jraph` API.
- Migrated the test suite from Abseil to pytest.
- Modernised the OGB single-device, Flax and `pmap` training paths to use
  current JAX and Optax APIs.
- Converted retained examples into importable, executable modules with
  deterministic test configurations.
- Updated the experimental sharded graph network for current JAX.
- Reworked the README, contribution guide, project metadata and CI around the
  maintained fork.

### Fixed

- `pad_with_graphs()` now preserves NumPy and JAX array backends and dtypes
  instead of silently converting JAX graphs to NumPy.
- Restored explicit and complete public API exports through `jraph.__all__`.
- Updated removed or deprecated JAX operations throughout the library,
  examples and tests.
- Made the multi-device test environment explicit and reproducible.

### Removed

- Abseil test and command-line dependencies.
- The obsolete upstream PyPI publishing workflow.
- The CIGRE power-network example, which now lives in the standalone
  [CIGRE GNN repository](https://github.com/dbg-dev/cigre-gnn).
