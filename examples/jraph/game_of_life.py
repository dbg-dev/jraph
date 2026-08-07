# Copyright 2020 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Conway's Game of Life implemented as a Jraph ``InteractionNetwork``.

Each cell is represented by a node. Eight directed edges carry the state of
the neighbouring cells to each receiver node, and an exactly specified MLP
implements Conway's update rule.
"""

import argparse
from collections.abc import Iterable, Sequence
import math
import time

import jax
import jax.numpy as jnp
from jraph import GraphsTuple, concatenated_args, InteractionNetwork
import numpy as np


DEFAULT_GLIDER = (
    (0, 0),
    (0, 1),
    (0, 2),
    (1, 2),
    (2, 1),
)

_NEIGHBOUR_OFFSETS = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


def conway_mlp(features: jax.Array) -> jax.Array:
    """Applies Conway's update rule using a fixed two-layer ReLU MLP.

    ``features`` contains the current cell state followed by the number of
    live neighbours. The returned scalar is either zero or one.
    """

    first_layer_weights = jnp.asarray(
        [
            [0.0, -1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, -1.0],
            [1.0, 1.0],
            [1.0, 1.0],
        ]
    )
    first_layer_bias = jnp.asarray([3.5, -3.5, -1.5, 1.5, -2.5, -3.5])

    hidden = jax.nn.relu(first_layer_weights @ features + first_layer_bias)

    output_weights = jnp.asarray([[2.0, -4.0, 2.0, -4.0, 2.0, -4.0]])
    output_bias = jnp.asarray([-4.0])

    return jax.nn.relu(output_weights @ hidden + output_bias)


def conway_graph(
    size: int,
    *,
    live_cells: Iterable[tuple[int, int]] = DEFAULT_GLIDER,
) -> GraphsTuple:
    """Builds a square, toroidal Game of Life graph.

    Args:
        size: Width and height of the square board.
        live_cells: ``(row, column)`` coordinates of initially live cells.

    Returns:
        A single-graph ``GraphsTuple`` with one scalar feature per node.
    """

    if size < 1:
        raise ValueError("size must be at least 1")

    live_cells = tuple(live_cells)
    for row, column in live_cells:
        if not 0 <= row < size or not 0 <= column < size:
            raise ValueError(
                f"live cell {(row, column)} is outside a {size}x{size} board"
            )

    num_nodes = size**2
    node_indices = np.arange(num_nodes, dtype=np.int32)
    rows, columns = np.divmod(node_indices, size)

    sender_groups = [
        ((rows + row_offset) % size) * size
        + ((columns + column_offset) % size)
        for row_offset, column_offset in _NEIGHBOUR_OFFSETS
    ]
    senders = np.stack(sender_groups, axis=1).reshape(-1)
    receivers = np.repeat(node_indices, len(_NEIGHBOUR_OFFSETS))

    nodes = np.zeros((num_nodes, 1), dtype=np.float32)
    for row, column in live_cells:
        nodes[row * size + column, 0] = 1.0

    num_edges = len(senders)
    return GraphsTuple(
        n_node=jnp.asarray([num_nodes], dtype=jnp.int32),
        n_edge=jnp.asarray([num_edges], dtype=jnp.int32),
        nodes=jnp.asarray(nodes),
        edges=jnp.zeros((num_edges, 1), dtype=jnp.float32),
        globals=None,
        senders=jnp.asarray(senders),
        receivers=jnp.asarray(receivers),
    )


def _update_edge(
    edge: jax.Array,
    sender_node: jax.Array,
    receiver_node: jax.Array,
) -> jax.Array:
    del edge, receiver_node
    return sender_node


_UPDATE_NODE = jax.vmap(concatenated_args(conway_mlp))
_CONWAY_NETWORK = InteractionNetwork(
    update_edge_fn=_update_edge,
    update_node_fn=_UPDATE_NODE,
)


def step(graph: GraphsTuple) -> GraphsTuple:
    """Advances a Game of Life graph by one generation."""

    return _CONWAY_NETWORK(graph)


def simulate(
    graph: GraphsTuple,
    num_steps: int,
    *,
    use_jit: bool = True,
) -> tuple[GraphsTuple, ...]:
    """Returns the initial graph followed by ``num_steps`` generations."""

    if num_steps < 0:
        raise ValueError("num_steps must be non-negative")

    step_fn = jax.jit(step) if use_jit else step
    history = [graph]

    for _ in range(num_steps):
        graph = step_fn(graph)
        history.append(graph)

    return tuple(history)


def render_graph(graph: GraphsTuple) -> str:
    """Returns an ASCII representation of a square Game of Life graph."""

    num_nodes = int(np.asarray(graph.n_node).sum())
    size = math.isqrt(num_nodes)
    if size * size != num_nodes:
        raise ValueError("graph does not contain a square number of nodes")

    nodes = np.asarray(graph.nodes).reshape(size, size)
    rows = [
        "".join("x" if value == 1.0 else " " for value in row)
        for row in nodes
    ]
    return "-" * size + "\n" + "\n".join(rows)


def display_graph(graph: GraphsTuple) -> None:
    """Prints an ASCII representation of the graph."""

    print(render_graph(graph))


def animate(
    *,
    size: int = 20,
    num_steps: int = 100,
    delay: float = 0.05,
) -> None:
    """Runs and displays the default glider simulation."""

    if delay < 0:
        raise ValueError("delay must be non-negative")

    graph = conway_graph(size)
    step_fn = jax.jit(step)

    display_graph(graph)
    for _ in range(num_steps):
        if delay:
            time.sleep(delay)
        graph = step_fn(graph)
        display_graph(graph)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=20)
    parser.add_argument("--num-steps", type=int, default=100)
    parser.add_argument("--delay", type=float, default=0.05)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Runs the animated example from the command line."""

    args = _parse_args(argv)
    animate(
        size=args.size,
        num_steps=args.num_steps,
        delay=args.delay,
    )


if __name__ == "__main__":
    main()
