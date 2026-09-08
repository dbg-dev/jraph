# Contributing to Jraph

Thank you for considering a contribution.

This repository is an independent continuation of the original
[DeepMind Jraph project](https://github.com/google-deepmind/jraph). The aim is
to preserve Jraph's small, explicit graph-processing model while keeping it
usable with current JAX and providing examples built with current Flax NNX APIs.

## Project principles

Contributions should generally support one or more of these goals:

- preserve the existing `GraphsTuple` data model and functional graph utilities;
- maintain behavioural compatibility with the original Jraph API where practical;
- support current versions of JAX and Python;
- maintain the Flax NNX examples without turning Jraph into a larger graph
  framework;
- keep the implementation small, readable, and close to the underlying mathematics.

Large abstractions or broad API redesigns should be discussed in an issue before
implementation.

## Development setup

The project requires Python 3.14 and uses
[uv](https://docs.astral.sh/uv/) for dependency management.

Clone the repository and create the locked development environment:

```bash
git clone https://github.com/dbg-dev/jraph.git
cd jraph
make sync
```

Run the complete repository check before submitting a change:

```bash
make check
```

This checks formatting, runs Ruff, executes the full test suite, and builds the
source and wheel distributions. Individual checks are also available:

```bash
make format-check
make lint
make test
make build
```

Static type checking is currently advisory while the project establishes a
useful boundary for JAX and PyTree types:

```bash
make typecheck
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

When modifying NNX examples:

- keep graph data and aggregation utilities functional;
- use NNX for parameterised modules and model state;
- avoid duplicating existing Jraph functionality unless the NNX interface
  requires it;
- include equivalence tests against the existing functional implementation when
  possible.

## Tests

Tests are organised by responsibility:

```text
tests/jraph/               Core library behaviour
tests/experimental/        Experimental sharding
tests/examples/            Executable examples
```

A contribution should normally include tests for:

- new public behaviour;
- bug fixes;
- compatibility changes;
- edge cases introduced by the change.

The retained examples form part of the test suite. Changes to shared utilities
or public APIs should preserve both core-library behaviour and the tested
example workflows.

Run all required checks with:

```bash
make check
```

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
