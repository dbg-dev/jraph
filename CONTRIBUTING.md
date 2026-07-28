# Contributing to Jraph

Thank you for considering a contribution.

This repository is an independent continuation of the original
[DeepMind Jraph project](https://github.com/google-deepmind/jraph). The aim is
to preserve Jraph's small, explicit graph-processing model while keeping it
usable with current JAX and adding a clear upgrade path to Flax NNX.

## Project principles

Contributions should generally support one or more of these goals:

- preserve the existing `GraphsTuple` data model and functional graph utilities;
- maintain behavioural compatibility with the original Jraph API where practical;
- support current versions of JAX and Python;
- add Flax NNX integration without turning Jraph into a larger graph framework;
- keep the implementation small, readable, and close to the underlying mathematics.

Large abstractions or broad API redesigns should be discussed in an issue before
implementation.

## Development setup

The project uses [uv](https://docs.astral.sh/uv/) for dependency management.

Clone the repository and create the development environment:

```bash
git clone <repository-url>
cd jraph
uv sync
```

Run the core test suite:

```bash
uv run pytest jraph/_src
```

Build the package:

```bash
uv build
```

Before submitting a change, also check for whitespace and merge-marker problems:

```bash
git diff --check
```

## Making changes

Keep changes focused and separate unrelated work into different commits.

When modifying existing behaviour:

- add or update tests that demonstrate the intended behaviour;
- preserve public API compatibility unless there is a clear reason not to;
- document any intentional compatibility break;
- avoid relying on private JAX or Flax APIs;
- prefer ordinary JAX transformations and explicit graph operations over hidden
  framework machinery.

When adding NNX support:

- keep graph data and aggregation utilities functional;
- use NNX for parameterised modules and model state;
- avoid duplicating existing Jraph functionality unless the NNX interface
  requires it;
- include equivalence tests against the existing functional implementation when
  possible.

## Tests

Tests for core library behaviour belong under:

```text
jraph/_src/
```

A contribution should normally include tests for:

- new public behaviour;
- bug fixes;
- compatibility changes;
- edge cases introduced by the change.

The core test suite must pass:

```bash
uv run pytest jraph/_src
```

Legacy examples may be modernised separately and are not necessarily part of the
core compatibility baseline.

## Documentation

Update docstrings and user-facing documentation when behaviour or public APIs
change.

Keep examples small and executable. Prefer examples that expose the relationship
between the code and the corresponding graph-network operation.

## Pull requests

A pull request should:

- explain the problem being solved;
- describe the chosen approach;
- identify any compatibility implications;
- include relevant tests;
- avoid unrelated formatting or refactoring changes.

Small pull requests are easier to review and less likely to introduce accidental
behaviour changes.

## Commit messages

Use clear, imperative commit messages. Conventional prefixes are welcome, for
example:

```text
fix: update aggregation for current JAX
feat: add NNX graph network module
test: add compatibility coverage for batching
docs: clarify migration from original Jraph
chore: update development tooling
```

## Licence and attribution

The project is licensed under the Apache License 2.0.

By contributing, you agree that your contribution may be distributed under the
same licence.

The original Jraph authorship, licence, and Git history are preserved. New
contributors should not remove or obscure existing attribution.
