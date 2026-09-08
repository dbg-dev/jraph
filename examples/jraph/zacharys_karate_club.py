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
"""Train the reusable PyGCN model on Zachary's karate club graph.

The model is supervised only by the known assignments of Mr. Hi (node 0) and
John A. (node 33), then evaluated against the full club partition.
"""

import logging
from dataclasses import dataclass
from typing import cast

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from examples.pygcn.model import TwoLayerGCN
from examples.pygcn.training import eval_step, train_step
from jraph import GraphsTuple

NUM_CLUB_MEMBERS = 34
NUM_CLASSES = 2
HIDDEN_FEATURES = 5
SUPERVISED_NODES = (0, 33)


@dataclass(frozen=True, slots=True)
class TrainResult:
    """Summary metrics from a Karate Club training run."""

    initial_loss: float
    final_loss: float
    initial_accuracy: float
    final_accuracy: float


def get_zacharys_karate_club() -> GraphsTuple:
    """Return a ``GraphsTuple`` representing Zachary's karate club."""

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

    return GraphsTuple(
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
    """Return the known two-way partition of the club members."""

    return jnp.asarray(
        [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0,
            0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        ],
        dtype=jnp.int32,
    )


def get_supervision_mask() -> jax.Array:
    """Return a mask selecting the two nodes used for training."""

    return (
        jnp.zeros(NUM_CLUB_MEMBERS, dtype=jnp.bool_)
        .at[jnp.asarray(SUPERVISED_NODES)]
        .set(True)
    )


def build_model(
    *,
    seed: int = 42,
    dropout_rate: float = 0.0,
) -> TwoLayerGCN:
    """Build the shared two-layer NNX GCN for Karate Club."""

    return TwoLayerGCN(
        in_features=NUM_CLUB_MEMBERS,
        hidden_features=HIDDEN_FEATURES,
        out_features=NUM_CLASSES,
        dropout_rate=dropout_rate,
        rngs=nnx.Rngs(seed),
    )


def node_logits(
    model: TwoLayerGCN,
    graph: GraphsTuple,
) -> jax.Array:
    """Return per-node class logits."""

    return cast(jax.Array, model(graph).nodes)


def train(
    *,
    num_steps: int = 30,
    seed: int = 42,
    learning_rate: float = 1e-2,
    dropout_rate: float = 0.0,
    log_every: int | None = 1,
) -> TrainResult:
    """Train the shared PyGCN model and return summary metrics."""

    if num_steps < 0:
        raise ValueError("num_steps must be non-negative")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if not 0.0 <= dropout_rate < 1.0:
        raise ValueError("dropout_rate must be in [0, 1)")
    if log_every is not None and log_every < 1:
        raise ValueError("log_every must be positive or None")

    graph = get_zacharys_karate_club()
    labels = get_ground_truth_assignments_for_zacharys_karate_club()
    supervision_mask = get_supervision_mask()
    full_mask = jnp.ones(NUM_CLUB_MEMBERS, dtype=jnp.bool_)

    model = build_model(seed=seed, dropout_rate=dropout_rate)

    # Both views share the same parameters. Training enables dropout while
    # evaluation disables it.
    train_model = nnx.view(model, deterministic=False)
    eval_model = nnx.view(model, deterministic=True)

    optimizer = nnx.Optimizer(
        model,
        optax.adam(learning_rate),
        wrt=nnx.Param,
    )

    initial_supervised = eval_step(
        eval_model,
        graph,
        labels,
        supervision_mask,
    )
    initial_full = eval_step(
        eval_model,
        graph,
        labels,
        full_mask,
    )

    for step in range(num_steps):
        metrics = train_step(
            train_model,
            optimizer,
            graph,
            labels,
            supervision_mask,
        )

        if log_every is not None and step % log_every == 0:
            full_metrics = eval_step(
                eval_model,
                graph,
                labels,
                full_mask,
            )
            logging.info(
                "step %d loss %.6f accuracy %.4f",
                step,
                float(metrics.loss),
                float(full_metrics.accuracy),
            )

    final_supervised = eval_step(
        eval_model,
        graph,
        labels,
        supervision_mask,
    )
    final_full = eval_step(
        eval_model,
        graph,
        labels,
        full_mask,
    )

    return TrainResult(
        initial_loss=float(initial_supervised.loss),
        final_loss=float(final_supervised.loss),
        initial_accuracy=float(initial_full.accuracy),
        final_accuracy=float(final_full.accuracy),
    )


def main() -> None:
    """Run the example with the original training length."""

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    result = train()
    logging.info(
        "final loss %.6f accuracy %.4f",
        result.final_loss,
        result.final_accuracy,
    )


if __name__ == "__main__":
    main()
