"""Tests for examples.lstm."""

import jax
import numpy as np
import pytest

from examples.jraph import lstm


@pytest.fixture(scope="module")
def result() -> lstm.LSTMExampleResult:
    return lstm.run(seed=42, num_message_passing_steps=2)

def _assert_trees_allclose(
    actual: object,
    expected: object,
    *,
    rtol: float = 2e-5, # note: numpy's default is 1e-7
    atol: float = 1e-7,
) -> None:
    jax.tree.map(
        lambda actual_leaf, expected_leaf: np.testing.assert_allclose(
            actual_leaf,
            expected_leaf,
            rtol=rtol,
            atol=atol,
        ),
        actual,
        expected,
    )


def test_random_graph_is_deterministic_for_seed() -> None:
    graph_a = lstm.get_random_graph(np.random.default_rng(7))
    graph_b = lstm.get_random_graph(np.random.default_rng(7))

    _assert_trees_allclose(graph_a, graph_b)


def test_random_graph_has_valid_structure(
    result: lstm.LSTMExampleResult,
) -> None:
    graph = result.input_graph

    assert graph.nodes.shape == (lstm.NUM_NODES, lstm.EMBEDDING_SIZE)
    assert graph.edges.shape == (lstm.NUM_EDGES, lstm.EMBEDDING_SIZE)
    assert graph.senders.shape == (lstm.NUM_EDGES,)
    assert graph.receivers.shape == (lstm.NUM_EDGES,)
    assert np.all(np.asarray(graph.senders) < lstm.NUM_NODES)
    assert np.all(np.asarray(graph.receivers) < lstm.NUM_NODES)


def test_recurrent_output_shapes(
    result: lstm.LSTMExampleResult,
) -> None:
    graph = result.eager_output_graph

    assert isinstance(graph.nodes, lstm.StatefulField)
    assert graph.nodes.embedding.shape == (
        lstm.NUM_NODES,
        lstm.EMBEDDING_SIZE,
    )
    assert graph.nodes.state is None

    assert isinstance(graph.edges, lstm.StatefulField)
    assert graph.edges.embedding.shape == (
        lstm.NUM_EDGES,
        lstm.EMBEDDING_SIZE,
    )
    assert graph.edges.state is not None
    assert graph.edges.state.hidden.shape == (
        lstm.NUM_EDGES,
        lstm.HIDDEN_SIZE,
    )
    assert graph.edges.state.cell.shape == (
        lstm.NUM_EDGES,
        lstm.HIDDEN_SIZE,
    )

    np.testing.assert_array_equal(graph.n_node, result.input_graph.n_node)
    np.testing.assert_array_equal(graph.n_edge, result.input_graph.n_edge)
    np.testing.assert_array_equal(graph.senders, result.input_graph.senders)
    np.testing.assert_array_equal(graph.receivers, result.input_graph.receivers)


def test_jitted_execution_matches_eager(
    result: lstm.LSTMExampleResult,
) -> None:
    _assert_trees_allclose(
        result.jitted_output_graph,
        result.eager_output_graph,
    )


def test_message_passing_steps_must_be_positive() -> None:
    with pytest.raises(
        ValueError,
        match="num_message_passing_steps must be at least 1",
    ):
        lstm.make_network_definition(num_message_passing_steps=0)
