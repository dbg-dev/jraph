"""Tests for examples.game_of_life."""

import jax
import numpy as np
import pytest

from examples.jraph import game_of_life


def _live_cells(graph) -> set[tuple[int, int]]:
    num_nodes = int(np.asarray(graph.n_node).sum())
    size = int(np.sqrt(num_nodes))
    nodes = np.asarray(graph.nodes).reshape(size, size)
    rows, columns = np.nonzero(nodes == 1.0)
    return set(zip(rows.tolist(), columns.tolist(), strict=True))


@pytest.mark.parametrize("current_state", [0.0, 1.0])
@pytest.mark.parametrize("live_neighbours", range(9))
def test_conway_mlp_matches_rule(
    current_state: float,
    live_neighbours: int,
) -> None:
    result = game_of_life.conway_mlp(
        jax.numpy.asarray([current_state, live_neighbours])
    )

    expected = float(
        live_neighbours == 3
        or (current_state == 1.0 and live_neighbours == 2)
    )
    np.testing.assert_array_equal(result, np.asarray([expected]))


def test_graph_has_eight_incoming_edges_per_cell() -> None:
    size = 5
    graph = game_of_life.conway_graph(size, live_cells=())

    assert graph.nodes.shape == (size**2, 1)
    assert graph.edges.shape == (8 * size**2, 1)
    assert graph.senders.shape == (8 * size**2,)
    assert graph.receivers.shape == (8 * size**2,)

    receiver_counts = np.bincount(
        np.asarray(graph.receivers),
        minlength=size**2,
    )
    np.testing.assert_array_equal(receiver_counts, np.full(size**2, 8))


def test_block_is_a_still_life() -> None:
    block = {(1, 1), (1, 2), (2, 1), (2, 2)}
    graph = game_of_life.conway_graph(5, live_cells=block)

    next_graph = game_of_life.step(graph)

    assert _live_cells(next_graph) == block


def test_blinker_oscillates() -> None:
    horizontal = {(2, 1), (2, 2), (2, 3)}
    vertical = {(1, 2), (2, 2), (3, 2)}
    graph = game_of_life.conway_graph(5, live_cells=horizontal)

    history = game_of_life.simulate(graph, num_steps=2, use_jit=False)

    assert _live_cells(history[1]) == vertical
    assert _live_cells(history[2]) == horizontal


def test_jitted_step_matches_eager_step() -> None:
    graph = game_of_life.conway_graph(6)

    eager = game_of_life.step(graph)
    jitted = jax.jit(game_of_life.step)(graph)

    jax.tree.map(np.testing.assert_allclose, jitted, eager)


def test_simulation_includes_initial_graph() -> None:
    graph = game_of_life.conway_graph(5)

    history = game_of_life.simulate(graph, num_steps=3)

    assert len(history) == 4
    jax.tree.map(np.testing.assert_array_equal, history[0], graph)


def test_render_graph() -> None:
    graph = game_of_life.conway_graph(
        3,
        live_cells={(0, 0), (1, 1), (2, 2)},
    )

    assert game_of_life.render_graph(graph) == "---\nx  \n x \n  x"


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (
            lambda: game_of_life.conway_graph(0),
            "size must be at least 1",
        ),
        (
            lambda: game_of_life.conway_graph(3, live_cells={(3, 0)}),
            "outside a 3x3 board",
        ),
        (
            lambda: game_of_life.simulate(
                game_of_life.conway_graph(3),
                num_steps=-1,
            ),
            "num_steps must be non-negative",
        ),
    ],
)
def test_invalid_arguments(call, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        call()
