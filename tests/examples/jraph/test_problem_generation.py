"""Tests for voting and SAT problem generation."""

import jax
import numpy as np
import pytest

from examples.jraph import e_voting
from examples.jraph import sat


def _assert_trees_equal(actual: object, expected: object) -> None:
    jax.tree.map(np.testing.assert_array_equal, actual, expected)


def test_voting_problem_is_reproducible() -> None:
    first = e_voting.get_voting_problem(
        2,
        15,
        rng=np.random.default_rng(42),
    )
    second = e_voting.get_voting_problem(
        2,
        15,
        rng=np.random.default_rng(42),
    )

    _assert_trees_equal(first, second)


def test_voting_problem_labels_real_graph_winner() -> None:
    problem = e_voting.get_voting_problem(
        15,
        15,
        rng=np.random.default_rng(42),
    )

    real_nodes = int(np.asarray(problem.graph.n_node)[0])
    votes = np.asarray(problem.graph.nodes)[:real_nodes]
    expected_winner = int(np.argmax(votes.sum(axis=0)))

    assert problem.labels.shape == problem.mask.shape
    assert problem.labels.dtype == jax.numpy.int32
    assert problem.mask.dtype == jax.numpy.bool_
    assert int(problem.mask.sum()) == 1
    assert int(problem.labels[0]) == expected_winner


def test_sat_problem_is_reproducible() -> None:
    first = sat.get_2sat_problem(
        2,
        15,
        rng=np.random.default_rng(42),
    )
    second = sat.get_2sat_problem(
        2,
        15,
        rng=np.random.default_rng(42),
    )

    _assert_trees_equal(first, second)


def test_sat_problem_masks_only_literal_nodes() -> None:
    problem = sat.get_2sat_problem(
        8,
        8,
        rng=np.random.default_rng(42),
    )

    n_literals = int(problem.mask.sum())
    n_constraints = n_literals * (n_literals - 1) // 2

    assert n_literals == 8
    assert problem.labels.shape == problem.mask.shape
    assert problem.labels.dtype == jax.numpy.int32
    assert problem.mask.dtype == jax.numpy.bool_
    np.testing.assert_array_equal(
        problem.graph.n_node,
        np.asarray([n_literals + n_constraints, 1]),
    )
    np.testing.assert_array_equal(
        problem.graph.n_edge,
        np.asarray([2 * n_constraints, 0]),
    )


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (
            lambda: e_voting.get_voting_problem(
                0,
                5,
                rng=np.random.default_rng(42),
            ),
            "min_n_voters must be positive",
        ),
        (
            lambda: sat.get_2sat_problem(
                1,
                5,
                rng=np.random.default_rng(42),
            ),
            "min_n_literals must be at least 2",
        ),
    ],
)
def test_invalid_problem_ranges(call, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        call()
