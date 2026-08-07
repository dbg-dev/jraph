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
r"""Recover planted assignments from randomly generated satisfiable 2-SAT graphs.

The graph is bipartite: literal nodes connect to constraint nodes. Edge
features indicate whether a literal is positive or negated in its constraint.
"""

from dataclasses import dataclass
import logging
from typing import NamedTuple, cast

from flax import nnx
import jax
import jax.numpy as jnp
import jraph
import numpy as np
import optax

from examples.jraph._graphs import pad_with_graphs_as_jax
from examples.jraph._random import make_random_streams
from examples.jraph._train import (
    ClassificationMetrics,
    eval_step,
    train_step,
)


TRAIN_DATASET = (2, 15)
TEST_DATASET = (16, 20)

INPUT_FEATURES = 2
EMBEDDING_FEATURES = 16
MESSAGE_FEATURES = 10
NUM_CLASSES = 2
NUM_MESSAGE_PASSING_STEPS = 5


class Problem(NamedTuple):
    """A padded SAT graph with literal labels and a supervision mask."""

    graph: jraph.GraphsTuple
    labels: jax.Array
    mask: jax.Array


@dataclass(frozen=True, slots=True)
class TrainResult:
    """Metrics for fixed in-distribution and extrapolation problems."""

    in_distribution: ClassificationMetrics
    extrapolation: ClassificationMetrics


class UpdateMLP(nnx.Module):
    """Three-layer MLP matching the original Haiku update function."""

    def __init__(
        self,
        in_features: int,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.layers = nnx.List(
            [
                nnx.Linear(
                    in_features,
                    MESSAGE_FEATURES,
                    rngs=rngs,
                ),
                nnx.Linear(
                    MESSAGE_FEATURES,
                    MESSAGE_FEATURES,
                    rngs=rngs,
                ),
                nnx.Linear(
                    MESSAGE_FEATURES,
                    MESSAGE_FEATURES,
                    rngs=rngs,
                ),
            ]
        )

    def __call__(self, features: jax.Array) -> jax.Array:
        for layer in self.layers:
            features = jax.nn.relu(layer(features))
        return features


class InteractionBlock(nnx.Module):
    """One edge-then-node interaction-network update."""

    def __init__(
        self,
        *,
        node_features: int,
        edge_features: int,
        rngs: nnx.Rngs,
    ) -> None:
        self.edge_mlp = UpdateMLP(
            edge_features + 2 * node_features,
            rngs=rngs,
        )
        self.node_mlp = UpdateMLP(
            node_features + 2 * MESSAGE_FEATURES,
            rngs=rngs,
        )

    def __call__(
        self,
        graph: jraph.GraphsTuple,
    ) -> jraph.GraphsTuple:
        def update_edges(
            edges: jax.Array,
            sender_nodes: jax.Array,
            receiver_nodes: jax.Array,
        ) -> jax.Array:
            return self.edge_mlp(
                jnp.concatenate(
                    (edges, sender_nodes, receiver_nodes),
                    axis=-1,
                )
            )

        def update_nodes(
            nodes: jax.Array,
            sent_edges: jax.Array,
            received_edges: jax.Array,
        ) -> jax.Array:
            return self.node_mlp(
                jnp.concatenate(
                    (nodes, sent_edges, received_edges),
                    axis=-1,
                )
            )

        return jraph.InteractionNetwork(
            update_edge_fn=update_edges,
            update_node_fn=update_nodes,
            include_sent_messages_in_node_update=True,
        )(graph)


class SATModel(nnx.Module):
    """NNX interaction network for planted 2-SAT assignments."""

    def __init__(
        self,
        *,
        num_message_passing_steps: int = NUM_MESSAGE_PASSING_STEPS,
        rngs: nnx.Rngs,
    ) -> None:
        if num_message_passing_steps < 1:
            raise ValueError(
                "num_message_passing_steps must be positive"
            )

        self.node_embedder = nnx.Linear(
            INPUT_FEATURES,
            EMBEDDING_FEATURES,
            rngs=rngs,
        )
        self.edge_embedder = nnx.Linear(
            INPUT_FEATURES,
            EMBEDDING_FEATURES,
            rngs=rngs,
        )

        blocks: list[InteractionBlock] = [
            InteractionBlock(
                node_features=EMBEDDING_FEATURES,
                edge_features=EMBEDDING_FEATURES,
                rngs=rngs,
            )
        ]
        blocks.extend(
            InteractionBlock(
                node_features=MESSAGE_FEATURES,
                edge_features=MESSAGE_FEATURES,
                rngs=rngs,
            )
            for _ in range(num_message_passing_steps - 1)
        )
        self.blocks = nnx.List(blocks)

        self.decoder = nnx.Linear(
            MESSAGE_FEATURES,
            NUM_CLASSES,
            rngs=rngs,
        )

    def __call__(self, graph: jraph.GraphsTuple) -> jax.Array:
        nodes = cast(jax.Array, graph.nodes)
        edges = cast(jax.Array, graph.edges)

        graph = graph._replace(
            nodes=self.node_embedder(nodes),
            edges=self.edge_embedder(edges),
        )

        for block in self.blocks:
            graph = block(graph)

        return self.decoder(cast(jax.Array, graph.nodes))


def get_2sat_problem(
    min_n_literals: int,
    max_n_literals: int,
    *,
    rng: np.random.Generator,
) -> Problem:
    """Create a planted satisfiable 2-SAT problem using an explicit RNG."""

    if min_n_literals < 2:
        raise ValueError("min_n_literals must be at least 2")
    if max_n_literals < min_n_literals:
        raise ValueError(
            "max_n_literals must be at least min_n_literals"
        )

    n_literals = int(
        rng.integers(min_n_literals, max_n_literals + 1)
    )
    n_literals_true = int(rng.integers(1, n_literals))
    n_constraints = n_literals * (n_literals - 1) // 2
    n_node = n_literals + n_constraints

    node_types = np.concatenate(
        (
            np.zeros(n_literals, dtype=np.int32),
            np.ones(n_constraints, dtype=np.int32),
        )
    )

    edge_types: list[int] = []
    senders: list[int] = []
    for literal_node1 in range(n_literals):
        for literal_node2 in range(literal_node1 + 1, n_literals):
            senders.extend((literal_node1, literal_node2))
            edge_types.extend(
                (
                    int(literal_node1 < n_literals_true),
                    int(literal_node2 < n_literals_true),
                )
            )

    graph = jraph.GraphsTuple(
        n_node=jnp.asarray([n_node], dtype=jnp.int32),
        n_edge=jnp.asarray(
            [2 * n_constraints],
            dtype=jnp.int32,
        ),
        nodes=jnp.eye(INPUT_FEATURES, dtype=jnp.float32)[
            jnp.asarray(node_types)
        ],
        edges=jnp.eye(INPUT_FEATURES, dtype=jnp.float32)[
            jnp.asarray(edge_types)
        ],
        globals=None,
        senders=jnp.asarray(senders, dtype=jnp.int32),
        receivers=jnp.repeat(
            jnp.arange(n_constraints, dtype=jnp.int32) + n_literals,
            2,
        ),
    )

    max_n_constraints = (
        max_n_literals * (max_n_literals - 1) // 2
    )
    max_nodes = max_n_literals + max_n_constraints + 1
    max_edges = 2 * max_n_constraints
    graph = pad_with_graphs_as_jax(
        graph,
        n_node=max_nodes,
        n_edge=max_edges,
    )

    labels = (
        jnp.arange(max_nodes, dtype=jnp.int32) < n_literals_true
    ).astype(jnp.int32)
    mask = jnp.arange(max_nodes) < n_literals
    return Problem(graph=graph, labels=labels, mask=mask)


def build_model(
    *,
    seed: int = 42,
    num_message_passing_steps: int = NUM_MESSAGE_PASSING_STEPS,
) -> SATModel:
    """Build the NNX interaction-network SAT model."""

    return SATModel(
        num_message_passing_steps=num_message_passing_steps,
        rngs=nnx.Rngs(seed),
    )


def evaluate(
    model: SATModel,
    problems: tuple[Problem, ...],
) -> ClassificationMetrics:
    """Evaluate against a fixed collection of generated problems."""

    metrics = [
        eval_step(
            model,
            problem.graph,
            problem.labels,
            problem.mask,
        )
        for problem in problems
    ]
    return ClassificationMetrics(
        loss=jnp.mean(
            jnp.asarray([metric.loss for metric in metrics])
        ),
        accuracy=jnp.mean(
            jnp.asarray([metric.accuracy for metric in metrics])
        ),
    )


def train(
    num_steps: int,
    *,
    seed: int = 42,
    learning_rate: float = 2e-4,
    log_every: int | None = 1000,
    num_eval_problems: int = 100,
    num_message_passing_steps: int = NUM_MESSAGE_PASSING_STEPS,
) -> TrainResult:
    """Train on fresh SAT problems and evaluate on fixed problem sets."""

    if num_steps < 0:
        raise ValueError("num_steps must be non-negative")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if log_every is not None and log_every < 1:
        raise ValueError("log_every must be positive or None")
    if num_eval_problems < 1:
        raise ValueError("num_eval_problems must be positive")

    streams = make_random_streams(seed)

    in_distribution_problems = tuple(
        get_2sat_problem(
            *TRAIN_DATASET,
            rng=streams.in_distribution_evaluation,
        )
        for _ in range(num_eval_problems)
    )
    extrapolation_problems = tuple(
        get_2sat_problem(
            *TEST_DATASET,
            rng=streams.extrapolation_evaluation,
        )
        for _ in range(num_eval_problems)
    )

    model = build_model(
        seed=seed,
        num_message_passing_steps=num_message_passing_steps,
    )
    optimizer = nnx.Optimizer(
        model,
        optax.adam(learning_rate),
        wrt=nnx.Param,
    )

    for step in range(num_steps):
        problem = get_2sat_problem(
            *TRAIN_DATASET,
            rng=streams.train,
        )
        train_step(
            model,
            optimizer,
            problem.graph,
            problem.labels,
            problem.mask,
        )

        if log_every is not None and step % log_every == 0:
            in_distribution_metrics = evaluate(
                model,
                in_distribution_problems,
            )
            extrapolation_metrics = evaluate(
                model,
                extrapolation_problems,
            )
            logging.info(
                (
                    "step %d in-distribution loss %.4f accuracy %.4f "
                    "extrapolation loss %.4f accuracy %.4f"
                ),
                step,
                float(in_distribution_metrics.loss),
                float(in_distribution_metrics.accuracy),
                float(extrapolation_metrics.loss),
                float(extrapolation_metrics.accuracy),
            )

    return TrainResult(
        in_distribution=evaluate(
            model,
            in_distribution_problems,
        ),
        extrapolation=evaluate(
            model,
            extrapolation_problems,
        ),
    )


def main() -> None:
    """Run the original long training configuration."""

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    result = train(num_steps=10_000)
    logging.info(
        (
            "final in-distribution loss %.4f accuracy %.4f "
            "extrapolation loss %.4f accuracy %.4f"
        ),
        float(result.in_distribution.loss),
        float(result.in_distribution.accuracy),
        float(result.extrapolation.loss),
        float(result.extrapolation.accuracy),
    )


if __name__ == "__main__":
    main()
