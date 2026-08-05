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
"""Train a GCN on Zachary's karate club graph.

The model is supervised only by the known assignments of Mr. Hi (node 0) and
John A. (node 33), then evaluated against the full club partition.
"""

from dataclasses import dataclass
import logging
from typing import Any

import haiku as hk
import jax
import jax.numpy as jnp
import jraph
import optax


NUM_CLUB_MEMBERS = 34
NUM_CLASSES = 2
SUPERVISED_NODES = (0, 33)


@dataclass(frozen=True)
class TrainResult:
    """Summary metrics from a Karate Club training run."""

    initial_loss: float
    final_loss: float
    initial_accuracy: float
    final_accuracy: float


def get_zacharys_karate_club() -> jraph.GraphsTuple:
    """Returns a ``GraphsTuple`` representing Zachary's karate club."""

    social_graph = [
        (1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2),
        (4, 0), (5, 0), (6, 0), (6, 4), (6, 5), (7, 0), (7, 1),
        (7, 2), (7, 3), (8, 0), (8, 2), (9, 2), (10, 0), (10, 4),
        (10, 5), (11, 0), (12, 0), (12, 3), (13, 0), (13, 1),
        (13, 2), (13, 3), (16, 5), (16, 6), (17, 0), (17, 1),
        (19, 0), (19, 1), (21, 0), (21, 1), (25, 23), (25, 24),
        (27, 2), (27, 23), (27, 24), (28, 2), (29, 23), (29, 26),
        (30, 1), (30, 8), (31, 0), (31, 24), (31, 25), (31, 28),
        (32, 2), (32, 8), (32, 14), (32, 15), (32, 18), (32, 20),
        (32, 22), (32, 23), (32, 29), (32, 30), (32, 31), (33, 8),
        (33, 9), (33, 13), (33, 14), (33, 15), (33, 18), (33, 19),
        (33, 20), (33, 22), (33, 23), (33, 26), (33, 27), (33, 28),
        (33, 29), (33, 30), (33, 31), (33, 32),
    ]
    social_graph += [(receiver, sender) for sender, receiver in social_graph]

    return jraph.GraphsTuple(
        n_node=jnp.asarray([NUM_CLUB_MEMBERS], dtype=jnp.int32),
        n_edge=jnp.asarray([len(social_graph)], dtype=jnp.int32),
        nodes=jnp.eye(NUM_CLUB_MEMBERS, dtype=jnp.float32),
        edges=None,
        globals=None,
        senders=jnp.asarray(
            [sender for sender, _ in social_graph],
            dtype=jnp.int32,
        ),
        receivers=jnp.asarray(
            [receiver for _, receiver in social_graph],
            dtype=jnp.int32,
        ),
    )


def get_ground_truth_assignments_for_zacharys_karate_club() -> jax.Array:
    """Returns the known two-way partition of the club members."""

    return jnp.asarray(
        [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0,
            0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        ],
        dtype=jnp.int32,
    )


def network_definition(graph: jraph.GraphsTuple) -> jax.Array:
    """Applies the two-layer GCN used in the original example."""

    graph = jraph.GraphConvolution(
        update_node_fn=hk.Linear(5, with_bias=False),
        add_self_edges=True,
    )(graph)
    graph = graph._replace(nodes=jax.nn.relu(graph.nodes))
    graph = jraph.GraphConvolution(
        update_node_fn=hk.Linear(NUM_CLASSES, with_bias=False),
    )(graph)
    return graph.nodes


def build_network() -> Any:
    """Returns the transformed Haiku network."""

    return hk.without_apply_rng(hk.transform(network_definition))


def prediction_loss(
    params: hk.Params,
    network: Any,
    graph: jraph.GraphsTuple,
) -> jax.Array:
    """Returns the loss on the two supervised club members."""

    logits = network.apply(params, graph)
    log_probabilities = jax.nn.log_softmax(logits)
    return -(
        log_probabilities[SUPERVISED_NODES[0], 0]
        + log_probabilities[SUPERVISED_NODES[1], 1]
    )


def prediction_accuracy(
    params: hk.Params,
    network: Any,
    graph: jraph.GraphsTuple,
    labels: jax.Array,
) -> jax.Array:
    """Returns accuracy against the full known partition."""

    logits = network.apply(params, graph)
    return jnp.mean(jnp.argmax(logits, axis=-1) == labels)


def train(
    *,
    num_steps: int = 30,
    seed: int = 42,
    learning_rate: float = 1e-2,
    log_every: int | None = 1,
) -> TrainResult:
    """Trains the original Haiku GCN and returns summary metrics."""

    if num_steps < 0:
        raise ValueError("num_steps must be non-negative")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if log_every is not None and log_every < 1:
        raise ValueError("log_every must be positive or None")

    graph = get_zacharys_karate_club()
    labels = get_ground_truth_assignments_for_zacharys_karate_club()
    network = build_network()
    params = network.init(jax.random.PRNGKey(seed), graph)

    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)

    def loss_fn(current_params: hk.Params) -> jax.Array:
        return prediction_loss(current_params, network, graph)

    @jax.jit
    def update(
        current_params: hk.Params,
        current_opt_state: optax.OptState,
    ) -> tuple[hk.Params, optax.OptState, jax.Array]:
        loss, gradients = jax.value_and_grad(loss_fn)(current_params)
        updates, updated_opt_state = optimizer.update(
            gradients,
            current_opt_state,
            current_params,
        )
        updated_params = optax.apply_updates(current_params, updates)
        return updated_params, updated_opt_state, loss

    accuracy_fn = jax.jit(
        lambda current_params: prediction_accuracy(
            current_params,
            network,
            graph,
            labels,
        )
    )

    initial_loss = float(loss_fn(params))
    initial_accuracy = float(accuracy_fn(params))

    for step in range(num_steps):
        params, opt_state, loss = update(params, opt_state)

        if log_every is not None and step % log_every == 0:
            logging.info(
                "step %d loss %.6f accuracy %.4f",
                step,
                float(loss),
                float(accuracy_fn(params)),
            )

    return TrainResult(
        initial_loss=initial_loss,
        final_loss=float(loss_fn(params)),
        initial_accuracy=initial_accuracy,
        final_accuracy=float(accuracy_fn(params)),
    )


def main() -> None:
    """Runs the example with its original training length."""

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    result = train()
    logging.info(
        "final loss %.6f accuracy %.4f",
        result.final_loss,
        result.final_accuracy,
    )


if __name__ == "__main__":
    main()
