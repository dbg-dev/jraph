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
r"""Electronic Voting Example.

In this example we use DeepSets to estimate the winner of an election.
Each vote is represented by a one-hot encoded vector.

It goes without saying, but don't use this in a real election!
Seriously, don't!
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import NamedTuple, cast

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from examples.jraph._graphs import pad_with_graphs_as_jax
from examples.jraph._random import make_random_streams
from examples.jraph._train import (
    ClassificationMetrics,
    eval_step,
    train_step,
)
from jraph import DeepSets, GraphsTuple, get_graph_padding_mask, segment_mean

NUM_CANDIDATES = 20
TRAIN_DATASET = (2, 15)
TEST_DATASET = (16, 20)


class Problem(NamedTuple):
    """A padded election graph and graph-classification targets."""

    graph: GraphsTuple
    labels: jax.Array
    mask: jax.Array


@dataclass(frozen=True, slots=True)
class TrainResult:
    """Metrics for fixed in-distribution and extrapolation problems."""

    in_distribution: ClassificationMetrics
    extrapolation: ClassificationMetrics


class MLP(nnx.Module):
    """Two-layer MLP matching the original Haiku update network."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.hidden = nnx.Linear(
            in_features,
            out_features,
            rngs=rngs,
        )
        self.output = nnx.Linear(
            out_features,
            out_features,
            rngs=rngs,
        )

    def __call__(self, features: jax.Array) -> jax.Array:
        return self.output(jax.nn.relu(self.hidden(features)))


class DeepSetsBlock(nnx.Module):
    """One learned node-to-global DeepSets update."""

    def __init__(
        self,
        feature_size: int,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.node_mlp = MLP(
            2 * feature_size,
            feature_size,
            rngs=rngs,
        )
        self.global_mlp = MLP(
            feature_size,
            feature_size,
            rngs=rngs,
        )

    def __call__(
        self,
        graph: GraphsTuple,
    ) -> GraphsTuple:
        def update_nodes(
            nodes: jax.Array,
            globals_: jax.Array,
        ) -> jax.Array:
            return self.node_mlp(
                jnp.concatenate((nodes, globals_), axis=-1)
            )

        def update_globals(
            aggregated_nodes: jax.Array,
        ) -> jax.Array:
            return self.global_mlp(aggregated_nodes)

        return DeepSets(
            update_node_fn=update_nodes,
            update_global_fn=update_globals,
            aggregate_nodes_for_globals_fn=segment_mean,
        )(graph)


class VotingModel(nnx.Module):
    """NNX DeepSets classifier for graph-level election winners."""

    def __init__(
        self,
        *,
        num_message_passing_steps: int = 1,
        rngs: nnx.Rngs,
    ) -> None:
        if num_message_passing_steps < 1:
            raise ValueError(
                "num_message_passing_steps must be positive"
            )

        self.blocks = nnx.List(
            [
                DeepSetsBlock(NUM_CANDIDATES, rngs=rngs)
                for _ in range(num_message_passing_steps)
            ]
        )
        self.decoder = nnx.Linear(
            NUM_CANDIDATES,
            NUM_CANDIDATES,
            rngs=rngs,
        )

    def __call__(self, graph: GraphsTuple) -> jax.Array:
        for block in self.blocks:
            graph = block(graph)

        globals_ = cast(jax.Array, graph.globals)
        return self.decoder(globals_)


def build_voting_problem(
    votes: Sequence[int] | np.ndarray,
    *,
    max_n_voters: int,
) -> Problem:
    """Build a padded voting problem from an explicit sequence of votes."""

    votes_array = np.asarray(votes, dtype=np.int32)
    if votes_array.ndim != 1:
        raise ValueError("votes must be a one-dimensional sequence")
    if votes_array.size < 1:
        raise ValueError("votes must not be empty")
    if max_n_voters < votes_array.size:
        raise ValueError(
            "max_n_voters must be at least the number of votes"
        )
    if np.any(votes_array < 0) or np.any(
        votes_array >= NUM_CANDIDATES
    ):
        raise ValueError(
            f"votes must be in [0, {NUM_CANDIDATES})"
        )

    n_voters = int(votes_array.size)
    one_hot_votes = np.eye(
        NUM_CANDIDATES,
        dtype=np.float32,
    )[votes_array]
    winner = int(np.argmax(np.sum(one_hot_votes, axis=0)))

    graph = GraphsTuple(
        n_node=jnp.asarray([n_voters], dtype=jnp.int32),
        n_edge=jnp.asarray([0], dtype=jnp.int32),
        nodes=jnp.asarray(one_hot_votes),
        edges=None,
        globals=jnp.zeros(
            (1, NUM_CANDIDATES),
            dtype=jnp.float32,
        ),
        senders=jnp.asarray([], dtype=jnp.int32),
        receivers=jnp.asarray([], dtype=jnp.int32),
    )

    # Add one padding node and one padding graph for a static node shape.
    graph = pad_with_graphs_as_jax(
        graph,
        n_node=max_n_voters + 1,
        n_edge=0,
    )

    mask = get_graph_padding_mask(graph)
    labels = (
        jnp.zeros(mask.shape, dtype=jnp.int32)
        .at[0]
        .set(winner)
    )
    return Problem(graph=graph, labels=labels, mask=mask)


def get_voting_problem(
    min_n_voters: int,
    max_n_voters: int,
    *,
    rng: np.random.Generator,
) -> Problem:
    """Create a randomly generated election using an explicit RNG."""

    if min_n_voters < 1:
        raise ValueError("min_n_voters must be positive")
    if max_n_voters < min_n_voters:
        raise ValueError("max_n_voters must be at least min_n_voters")

    n_voters = int(rng.integers(min_n_voters, max_n_voters + 1))
    votes = rng.integers(
        0,
        NUM_CANDIDATES,
        size=n_voters,
        dtype=np.int32,
    )
    return build_voting_problem(
        votes,
        max_n_voters=max_n_voters,
    )


def build_model(
    *,
    seed: int = 42,
    num_message_passing_steps: int = 1,
) -> VotingModel:
    """Build the NNX DeepSets voting model."""

    return VotingModel(
        num_message_passing_steps=num_message_passing_steps,
        rngs=nnx.Rngs(seed),
    )


def evaluate(
    model: VotingModel,
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
    num_message_passing_steps: int = 1,
) -> TrainResult:
    """Train on fresh elections and evaluate on fixed problem sets."""

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
        get_voting_problem(
            *TRAIN_DATASET,
            rng=streams.in_distribution_evaluation,
        )
        for _ in range(num_eval_problems)
    )
    extrapolation_problems = tuple(
        get_voting_problem(
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
        problem = get_voting_problem(
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
                    "step %d in-distribution accuracy %.4f "
                    "extrapolation accuracy %.4f"
                ),
                step,
                float(in_distribution_metrics.accuracy),
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
    result = train(num_steps=100_000)
    logging.info(
        (
            "final in-distribution accuracy %.4f "
            "extrapolation accuracy %.4f"
        ),
        float(result.in_distribution.accuracy),
        float(result.extrapolation.accuracy),
    )


if __name__ == "__main__":
    main()
