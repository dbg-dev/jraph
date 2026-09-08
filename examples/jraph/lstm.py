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
"""Use an LSTM as recurrent edge state in an ``InteractionNetwork``.

Jraph features may be arbitrary pytrees. This example stores both an embedding
and recurrent state in each edge feature, allowing an LSTM to retain information
across message-passing steps.

The same pattern can be used for node or global recurrent state.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import NamedTuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from jraph import GraphsTuple, InteractionNetwork

NUM_NODES = 5
NUM_EDGES = 7
NUM_MESSAGE_PASSING_STEPS = 10
EMBEDDING_SIZE = 32
HIDDEN_SIZE = 128


class StatefulField(NamedTuple):
    """A feature embedding together with optional recurrent state."""

    embedding: jax.Array
    state: hk.LSTMState | None


@dataclass(frozen=True)
class LSTMExampleResult:
    """Values produced by :func:`run` for smoke testing and inspection."""

    input_graph: GraphsTuple
    eager_output_graph: GraphsTuple
    jitted_output_graph: GraphsTuple


def get_random_graph(
    rng: np.random.Generator,
    *,
    num_nodes: int = NUM_NODES,
    num_edges: int = NUM_EDGES,
    embedding_size: int = EMBEDDING_SIZE,
) -> GraphsTuple:
    """Returns a deterministic random graph for a supplied NumPy generator."""

    return GraphsTuple(
        n_node=jnp.asarray([num_nodes], dtype=jnp.int32),
        n_edge=jnp.asarray([num_edges], dtype=jnp.int32),
        nodes=jnp.asarray(
            rng.normal(size=(num_nodes, embedding_size)),
            dtype=jnp.float32,
        ),
        edges=jnp.asarray(
            rng.normal(size=(num_edges, embedding_size)),
            dtype=jnp.float32,
        ),
        globals=None,
        senders=jnp.asarray(
            rng.integers(0, num_nodes, size=num_edges, dtype=np.int32)
        ),
        receivers=jnp.asarray(
            rng.integers(0, num_nodes, size=num_edges, dtype=np.int32)
        ),
    )


def make_network_definition(
    *,
    num_message_passing_steps: int = NUM_MESSAGE_PASSING_STEPS,
) -> Callable[[GraphsTuple], GraphsTuple]:
    """Builds an ``InteractionNetwork`` with recurrent edge state."""

    if num_message_passing_steps < 1:
        raise ValueError("num_message_passing_steps must be at least 1")

    def network_definition(graph: GraphsTuple) -> GraphsTuple:
        edge_lstm = hk.LSTM(hidden_size=HIDDEN_SIZE)
        edge_mlp = hk.nets.MLP([HIDDEN_SIZE, EMBEDDING_SIZE])
        node_mlp = hk.nets.MLP([HIDDEN_SIZE, EMBEDDING_SIZE])

        graph = graph._replace(
            edges=StatefulField(
                embedding=graph.edges,
                state=edge_lstm.initial_state(graph.edges.shape[0]),
            ),
            nodes=StatefulField(
                embedding=graph.nodes,
                state=None,
            ),
        )

        def update_edge_fn(
            edges: StatefulField,
            sender_nodes: StatefulField,
            receiver_nodes: StatefulField,
        ) -> StatefulField:
            edge_inputs = jnp.concatenate(
                [
                    edges.embedding,
                    sender_nodes.embedding,
                    receiver_nodes.embedding,
                ],
                axis=-1,
            )
            lstm_output, updated_state = edge_lstm(edge_inputs, edges.state)
            return StatefulField(
                embedding=edge_mlp(lstm_output),
                state=updated_state,
            )

        def update_node_fn(
            nodes: StatefulField,
            received_edges: StatefulField,
        ) -> StatefulField:
            node_inputs = jnp.concatenate(
                [nodes.embedding, received_edges.embedding],
                axis=-1,
            )
            return StatefulField(
                embedding=node_mlp(node_inputs),
                state=None,
            )

        recurrent_network = InteractionNetwork(
            update_edge_fn=update_edge_fn,
            update_node_fn=update_node_fn,
        )

        for _ in range(num_message_passing_steps):
            graph = recurrent_network(graph)

        return graph

    return network_definition


def run(
    *,
    seed: int = 42,
    num_message_passing_steps: int = NUM_MESSAGE_PASSING_STEPS,
) -> LSTMExampleResult:
    """Runs the recurrent graph network in eager and JIT-compiled modes."""

    input_graph = get_random_graph(np.random.default_rng(seed))
    network_definition = make_network_definition(
        num_message_passing_steps=num_message_passing_steps
    )
    network = hk.without_apply_rng(hk.transform(network_definition))

    params = network.init(jax.random.PRNGKey(seed), input_graph)
    eager_output_graph = network.apply(params, input_graph)
    jitted_output_graph = jax.jit(network.apply)(params, input_graph)

    return LSTMExampleResult(
        input_graph=input_graph,
        eager_output_graph=eager_output_graph,
        jitted_output_graph=jitted_output_graph,
    )


def main() -> None:
    """Runs the example and prints the output pytree shapes."""

    result = run()
    shapes = jax.tree.map(lambda value: value.shape, result.eager_output_graph)
    print(shapes)


if __name__ == "__main__":
    main()
