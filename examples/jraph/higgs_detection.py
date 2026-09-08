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
r"""Higgs Boson detection with an NNX relation network.

One decay channel of the Higgs Boson is Higgs to two photons. The two photons
must have a combined invariant mass of approximately 125 GeV.

Each problem contains either:
  * two photons from a Higgs decay plus uncorrelated background photons; or
  * only uncorrelated background photons.

The graph is fully connected and a relation network classifies the whole graph.
"""

import logging
from dataclasses import dataclass
from typing import NamedTuple, cast

import jax
import jax.numpy as jnp
import numpy as np
import optax
import scipy.stats
from flax import nnx

import jraph
from examples.jraph._random import make_random_streams
from examples.jraph._train import (
    ClassificationMetrics,
    eval_step,
    train_step,
)

logger = logging.getLogger(__name__)


HIGGS_MASS_GEV = 125.18
HIGGS_LABEL = 0
BACKGROUND_LABEL = 1

TRAIN_DATASET = (2, 15)
TEST_DATASET = (16, 20)

PHOTON_FEATURES = 4
EDGE_FEATURES = 30
NUM_CLASSES = 2


class Problem(NamedTuple):
    """A padded photon graph and graph-level classification targets."""

    graph: jraph.GraphsTuple
    labels: jax.Array
    mask: jax.Array


@dataclass(frozen=True, slots=True)
class TrainResult:
    """Metrics for fixed in-distribution and extrapolation problems."""

    in_distribution: ClassificationMetrics
    extrapolation: ClassificationMetrics


class EdgeMLP(nnx.Module):
    """Three-layer relation MLP matching the original Haiku network."""

    def __init__(self, *, rngs: nnx.Rngs) -> None:
        self.layers = nnx.List(
            [
                nnx.Linear(
                    2 * PHOTON_FEATURES,
                    EDGE_FEATURES,
                    rngs=rngs,
                ),
                nnx.Linear(
                    EDGE_FEATURES,
                    EDGE_FEATURES,
                    rngs=rngs,
                ),
                nnx.Linear(
                    EDGE_FEATURES,
                    EDGE_FEATURES,
                    rngs=rngs,
                ),
            ]
        )

    def __call__(self, features: jax.Array) -> jax.Array:
        for layer in self.layers[:-1]:
            features = jax.nn.relu(layer(features))
        return self.layers[-1](features)


# The correct solution for the edge update function is the invariant mass
# of the photon pair.
# The simple MLP we use here seems to fail to find the correct solution.
# You can ensure that the example works in principle by replacing the
# update_edge_fn below with the following analytical solution.
@jax.vmap
def unused_update_edge_fn_solution(
    s: jax.Array,
    r: jax.Array,
) -> jax.Array:
    """Calculates invariant mass of photon pair and compares to Higgs mass."""

    t = (s + r) ** 2
    return jnp.array(
        jnp.abs(t[0] - t[1] - t[2] - t[3] - HIGGS_MASS_GEV**2) < 1,
        dtype=jnp.float32,
    )[None]


class HiggsModel(nnx.Module):
    """Relation network that produces one pair of logits per graph."""

    def __init__(self, *, rngs: nnx.Rngs) -> None:
        self.edge_mlp = EdgeMLP(rngs=rngs)
        self.decoder = nnx.Linear(
            EDGE_FEATURES,
            NUM_CLASSES,
            rngs=rngs,
        )

    def __call__(self, graph: jraph.GraphsTuple) -> jax.Array:
        def update_edges(
            sender_nodes: jax.Array,
            receiver_nodes: jax.Array,
        ) -> jax.Array:
            features = jnp.concatenate(
                (sender_nodes, receiver_nodes),
                axis=-1,
            )
            return self.edge_mlp(features)

        relation_network = jraph.RelationNetwork(
            update_edge_fn=update_edges,
            update_global_fn=self.decoder,
            aggregate_edges_for_globals_fn=jraph.segment_sum,
        )
        output_graph = relation_network(graph)
        return cast(jax.Array, output_graph.globals)


def get_random_rotation_matrix(
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample a spatial orthogonal transform embedded in four dimensions."""

    rotation = np.eye(4)
    rotation[1:, 1:] = scipy.stats.ortho_group.rvs(
        3,
        random_state=rng,
    )
    return rotation


def get_random_boost_matrix(
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample a Lorentz boost in a random spatial direction."""

    eta = float(rng.uniform(-1.0, 1.0))
    boost = np.eye(4)
    boost[:2, :2] = np.asarray(
        [
            [np.cosh(eta), -np.sinh(eta)],
            [-np.sinh(eta), np.cosh(eta)],
        ]
    )
    rotation = get_random_rotation_matrix(rng)
    return rotation.T @ boost @ rotation


def get_random_higgs_photons(
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a boosted and rotated two-photon Higgs decay."""

    boost = get_random_boost_matrix(rng)
    rotation = get_random_rotation_matrix(rng)
    photon1 = (
        boost
        @ rotation
        @ np.asarray(
            [
                HIGGS_MASS_GEV / 2,
                HIGGS_MASS_GEV / 2,
                0.0,
                0.0,
            ]
        )
    )
    photon2 = (
        boost
        @ rotation
        @ np.asarray(
            [
                HIGGS_MASS_GEV / 2,
                -HIGGS_MASS_GEV / 2,
                0.0,
                0.0,
            ]
        )
    )
    return photon1, photon2


def get_random_background_photon(
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate one boosted and rotated massless background photon."""

    boost = get_random_boost_matrix(rng)
    rotation = get_random_rotation_matrix(rng)
    energy = float(rng.uniform(20.0, 120.0))
    return boost @ rotation @ np.asarray([energy, energy, 0.0, 0.0])


def invariant_mass_squared(four_momentum: np.ndarray) -> float:
    """Return E^2 - |p|^2 for a four-momentum."""

    return float(four_momentum[0] ** 2 - np.sum(four_momentum[1:] ** 2))


def build_higgs_problem(
    photons: np.ndarray,
    label: int,
    *,
    max_n_photons: int,
) -> Problem:
    """Build a padded fully connected problem from explicit photon data."""

    photons = np.asarray(photons)
    if photons.ndim != 2 or photons.shape[1] != PHOTON_FEATURES:
        raise ValueError("photons must have shape (n_photons, PHOTON_FEATURES)")

    n_photons = photons.shape[0]
    if n_photons < 2:
        raise ValueError("at least two photons are required")
    if max_n_photons < n_photons:
        raise ValueError("max_n_photons must be at least the number of photons")
    if label not in (HIGGS_LABEL, BACKGROUND_LABEL):
        raise ValueError("label must be HIGGS_LABEL or BACKGROUND_LABEL")

    senders = np.repeat(
        np.arange(n_photons, dtype=np.int32),
        n_photons,
    )
    receivers = np.tile(
        np.arange(n_photons, dtype=np.int32),
        n_photons,
    )

    graph = jraph.GraphsTuple(
        n_node=jnp.asarray([n_photons], dtype=jnp.int32),
        n_edge=jnp.asarray([senders.size], dtype=jnp.int32),
        nodes=jnp.asarray(photons, dtype=jnp.float32),
        edges=None,
        globals=None,
        senders=jnp.asarray(senders),
        receivers=jnp.asarray(receivers),
    )

    graph = jraph.pad_with_graphs(
        graph,
        n_node=max_n_photons + 1,
        n_edge=max_n_photons * max_n_photons,
    )

    mask = jraph.get_graph_padding_mask(graph)
    labels = jnp.zeros(mask.shape, dtype=jnp.int32)
    labels = labels.at[0].set(label)

    return Problem(
        graph=graph,
        labels=labels,
        mask=mask,
    )


def get_higgs_problem(
    min_n_photons: int,
    max_n_photons: int,
    *,
    rng: np.random.Generator,
) -> Problem:
    """Randomly generate and then build a Higgs-classification problem."""

    if min_n_photons < 2:
        raise ValueError("min_n_photons must be at least 2")
    if max_n_photons < min_n_photons:
        raise ValueError("max_n_photons must be at least min_n_photons")

    n_photons = int(rng.integers(min_n_photons, max_n_photons + 1))
    photons = np.stack([get_random_background_photon(rng) for _ in range(n_photons)])

    if rng.random() > 0.5:
        label = HIGGS_LABEL
        photons[:2] = np.stack(get_random_higgs_photons(rng))
    else:
        label = BACKGROUND_LABEL

    return build_higgs_problem(
        photons,
        label,
        max_n_photons=max_n_photons,
    )


def build_model(*, seed: int = 42) -> HiggsModel:
    """Build the NNX relation-network classifier."""

    return HiggsModel(rngs=nnx.Rngs(seed))


def evaluate(
    model: HiggsModel,
    problems: tuple[Problem, ...],
) -> ClassificationMetrics:
    """Evaluate a model against a fixed collection of problems."""

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
        loss=jnp.mean(jnp.asarray([metric.loss for metric in metrics])),
        accuracy=jnp.mean(jnp.asarray([metric.accuracy for metric in metrics])),
    )


def train(
    num_steps: int,
    *,
    seed: int = 42,
    learning_rate: float = 2e-4,
    log_every: int | None = 1000,
    num_eval_problems: int = 100,
) -> TrainResult:
    """Train on fresh problems and evaluate on fixed problem sets."""

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
        get_higgs_problem(
            *TRAIN_DATASET,
            rng=streams.in_distribution_evaluation,
        )
        for _ in range(num_eval_problems)
    )
    extrapolation_problems = tuple(
        get_higgs_problem(
            *TEST_DATASET,
            rng=streams.extrapolation_evaluation,
        )
        for _ in range(num_eval_problems)
    )

    model = build_model(seed=seed)
    optimizer = nnx.Optimizer(
        model,
        optax.adam(learning_rate),
        wrt=nnx.Param,
    )

    for step in range(num_steps):
        problem = get_higgs_problem(
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
            logger.info(
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

    logger.basicConfig(level=logger.INFO, format="%(message)s")
    result = train(num_steps=10_000)
    logger.info(
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
