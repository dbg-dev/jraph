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

"""Shared training utilities for the OGB examples."""

import logging
from collections.abc import Callable, Iterable, Iterator, Mapping
from typing import TypeVar, cast

import jax
import jax.numpy as jnp

import jraph

logger = logging.getLogger(__name__)

def _nearest_bigger_power_of_two(x: int) -> int:
    """Computes the nearest power of two greater than x."""
    y = 2
    while y < x:
        y *= 2
    return y


def pad_graph_to_nearest_power_of_two(
    graphs_tuple: jraph.GraphsTuple,
) -> jraph.GraphsTuple:
    """Pads a batched `GraphsTuple` to the nearest power of two.

    For example, if a `GraphsTuple` has 7 nodes, 5 edges and 3 graphs, this method
    would pad the `GraphsTuple` nodes and edges:
      7 nodes --> 8 nodes (2^3)
      5 edges --> 8 edges (2^3)

    And since padding is accomplished using `jraph.pad_with_graphs`, an extra
    graph and node is added:
      8 nodes --> 9 nodes
      3 graphs --> 4 graphs

    Args:
      graphs_tuple: a batched `GraphsTuple` (can be batch size 1).

    Returns:
      A graphs_tuple batched to the nearest power of two.
    """

    # Add one because pad_with_graphs requires at least one padding node.
    pad_nodes_to = _nearest_bigger_power_of_two(jnp.sum(graphs_tuple.n_node)) + 1
    pad_edges_to = _nearest_bigger_power_of_two(jnp.sum(graphs_tuple.n_edge))

    # Add one padding graph. The batch size itself remains fixed.
    pad_graphs_to = graphs_tuple.n_node.shape[0] + 1

    return jraph.pad_with_graphs(
        graphs_tuple,
        pad_nodes_to,
        pad_edges_to,
        pad_graphs_to,
    )


def prepare_graph(
    graph: jraph.GraphsTuple,
) -> tuple[jraph.GraphsTuple, jax.Array]:
    """Pad a graph batch, extract its labels, and clear its globals."""

    graph = pad_graph_to_nearest_power_of_two(graph)

    graph_globals = cast(Mapping[str, jax.Array], graph.globals)
    labels = graph_globals["label"]

    return graph._replace(globals={}), labels


def loss_and_accuracy(
    predicted_graph: jraph.GraphsTuple,
    labels: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Compute masked graph-classification loss and accuracy."""

    logits = cast(jax.Array, predicted_graph.globals)
    log_probabilities = jax.nn.log_softmax(logits)
    targets = jax.nn.one_hot(labels, 2)

    # Ignore the dummy graph added by pad_with_graphs.
    mask = jraph.get_graph_padding_mask(predicted_graph)

    loss = -jnp.mean(log_probabilities * targets * mask[:, None])

    accuracy = jnp.sum((jnp.argmax(logits, axis=1) == labels) * mask) / jnp.sum(mask)

    return loss, accuracy


StateT = TypeVar("StateT")

type StepMetrics = tuple[jax.Array, jax.Array]


def run_training(
    reader: Iterator[jraph.GraphsTuple],
    state: StateT,
    train_step: Callable[
        [StateT, jraph.GraphsTuple, jax.Array],
        tuple[StateT, StepMetrics],
    ],
    *,
    num_training_steps: int,
) -> StateT:
    """Run a single-device graph training loop."""

    for step in range(num_training_steps):
        graph, labels = prepare_graph(next(reader))
        state, (loss, accuracy) = train_step(state, graph, labels)

        if step % 100 == 0:
            logger.info(
                "step: %s, loss: %s, acc: %s",
                step,
                loss,
                accuracy,
            )

    return state


def run_evaluation(
    reader: Iterable[jraph.GraphsTuple],
    state: StateT,
    eval_step: Callable[
        [StateT, jraph.GraphsTuple, jax.Array],
        StepMetrics,
    ],
) -> StepMetrics:
    """Run a single-device graph evaluation loop."""

    accumulated_loss = jnp.asarray(0.0)
    accumulated_accuracy = jnp.asarray(0.0)
    num_batches = 0

    for graph in reader:
        graph, labels = prepare_graph(graph)
        loss, accuracy = eval_step(state, graph, labels)

        accumulated_loss += loss
        accumulated_accuracy += accuracy
        num_batches += 1

        if num_batches % 100 == 0:
            logger.info("Evaluated %s graph batches", num_batches)

    if num_batches == 0:
        raise ValueError("Cannot evaluate an empty dataset.")

    return (
        accumulated_loss / num_batches,
        accumulated_accuracy / num_batches,
    )
