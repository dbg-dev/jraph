"""Tests for examples.basic."""

import jax
import numpy as np
import pytest

from examples.jraph import basic


@pytest.fixture(scope="module")
def result() -> basic.BasicExampleResult:
    return basic.run(log_output=False)


def _assert_trees_allclose(actual: object, expected: object) -> None:
    jax.tree.map(np.testing.assert_allclose, actual, expected)


def test_batching_and_padding_round_trips(
    result: basic.BasicExampleResult,
) -> None:
    assert len(result.unbatched_graphs) == 3

    np.testing.assert_array_equal(
        result.implicitly_batched_graph.n_node,
        np.asarray([3, 3, 1]),
    )
    np.testing.assert_array_equal(
        result.implicitly_batched_graph.n_edge,
        np.asarray([2, 2, 1]),
    )

    assert int(np.asarray(result.padded_graph.n_node).sum()) == 10
    assert int(np.asarray(result.padded_graph.n_edge).sum()) == 5
    assert len(result.padded_graph.n_node) == 4

    _assert_trees_allclose(result.unpadded_graph, result.single_graph)


def test_identity_graph_network_preserves_features(
    result: basic.BasicExampleResult,
) -> None:
    _assert_trees_allclose(result.updated_single_graph, result.single_graph)
    _assert_trees_allclose(result.updated_nested_graph, result.nested_graph)
    _assert_trees_allclose(
        result.updated_batched_graph,
        result.implicitly_batched_graph,
    )
    _assert_trees_allclose(result.updated_padded_graph, result.padded_graph)


def test_jitted_graph_network_matches_eager_execution(
    result: basic.BasicExampleResult,
) -> None:
    _assert_trees_allclose(
        result.jitted_updated_padded_graph,
        result.updated_padded_graph,
    )


def test_explicit_batch_shapes(
    result: basic.BasicExampleResult,
) -> None:
    graph = result.explicitly_batched_graph

    assert graph.nodes.shape == (2, 3, 4)
    assert graph.edges.shape == (2, 2, 5)
    assert graph.globals.shape == (2, 1, 6)
    assert graph.senders.shape == (2, 2)
    assert graph.receivers.shape == (2, 2)
