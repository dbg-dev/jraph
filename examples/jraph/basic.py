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
"""A basic GraphNetwork example.

This example demonstrates the core mechanics of Jraph:

* constructing individual, nested, implicitly batched, and explicitly batched
  ``GraphsTuple`` objects;
* batching, unbatching, padding, and unpadding graphs;
* applying a ``GraphNetwork`` in eager and JIT-compiled modes.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax
import numpy as np

from jraph import (
    GraphNetwork,
    GraphsTuple,
    batch,
    pad_with_graphs,
    segment_sum,
    unbatch,
    unpad_with_graphs,
)


@dataclass(frozen=True)
class BasicExampleResult:
    """Values produced by :func:`run` for inspection and smoke testing."""

    single_graph: GraphsTuple
    nested_graph: GraphsTuple
    implicitly_batched_graph: GraphsTuple
    unbatched_graphs: tuple[GraphsTuple, ...]
    padded_graph: GraphsTuple
    unpadded_graph: GraphsTuple
    explicitly_batched_graph: GraphsTuple
    updated_single_graph: GraphsTuple
    updated_nested_graph: GraphsTuple
    updated_batched_graph: GraphsTuple
    updated_padded_graph: GraphsTuple
    jitted_updated_padded_graph: GraphsTuple


def _build_identity_graph_network() -> Callable[
    [GraphsTuple], GraphsTuple
]:
    """Builds a ``GraphNetwork`` whose update functions preserve all features."""

    def update_edge_fn(
        edge_features: Any,
        sender_node_features: Any,
        receiver_node_features: Any,
        globals_: Any,
    ) -> Any:
        del sender_node_features, receiver_node_features, globals_
        return edge_features

    def update_node_fn(
        node_features: Any,
        aggregated_sender_edge_features: Any,
        aggregated_receiver_edge_features: Any,
        globals_: Any,
    ) -> Any:
        del (
            aggregated_sender_edge_features,
            aggregated_receiver_edge_features,
            globals_,
        )
        return node_features

    def update_global_fn(
        aggregated_node_features: Any,
        aggregated_edge_features: Any,
        globals_: Any,
    ) -> Any:
        del aggregated_node_features, aggregated_edge_features
        return globals_

    return GraphNetwork(
        update_edge_fn=update_edge_fn,
        update_node_fn=update_node_fn,
        update_global_fn=update_global_fn,
        aggregate_edges_for_nodes_fn=segment_sum,
        aggregate_nodes_for_globals_fn=segment_sum,
        aggregate_edges_for_globals_fn=segment_sum,
    )


def run(*, log_output: bool = True) -> BasicExampleResult:
    """Runs the example and returns all significant intermediate values."""

    # A single graph with three nodes, two edges, and graph-level features.
    single_graph = GraphsTuple(
        n_node=np.asarray([3]),
        n_edge=np.asarray([2]),
        nodes=np.ones((3, 4)),
        edges=np.ones((2, 5)),
        globals=np.ones((1, 6)),
        senders=np.asarray([0, 1]),
        receivers=np.asarray([2, 2]),
    )

    # Features may be arbitrary JAX pytrees.
    nested_graph = GraphsTuple(
        n_node=np.asarray([3]),
        n_edge=np.asarray([2]),
        nodes={"a": np.ones((3, 4))},
        edges={"b": np.ones((2, 5))},
        globals={"c": np.ones((1, 6))},
        senders=np.asarray([0, 1]),
        receivers=np.asarray([2, 2]),
    )

    # Two graphs represented using Jraph's implicit batching convention.
    two_graph_batch = GraphsTuple(
        n_node=np.asarray([3, 1]),
        n_edge=np.asarray([2, 1]),
        nodes=np.ones((4, 4)),
        edges=np.ones((3, 5)),
        globals=np.ones((2, 6)),
        senders=np.asarray([0, 1, 3]),
        receivers=np.asarray([2, 2, 3]),
    )

    implicitly_batched_graph = batch([single_graph, two_graph_batch])
    unbatched_graphs = tuple(unbatch(implicitly_batched_graph))

    # Padding gives a static total number of nodes, edges, and graphs.
    padded_graph = pad_with_graphs(
        single_graph,
        n_node=10,
        n_edge=5,
        n_graph=4,
    )
    unpadded_graph = unpad_with_graphs(padded_graph)

    # Explicit batching adds a leading batch axis. Functions operating on this
    # representation are normally transformed with ``jax.vmap``.
    explicitly_batched_graph = GraphsTuple(
        n_node=np.asarray([[3], [1]]),
        n_edge=np.asarray([[2], [1]]),
        nodes=np.ones((2, 3, 4)),
        edges=np.ones((2, 2, 5)),
        globals=np.ones((2, 1, 6)),
        senders=np.asarray([[0, 1], [0, -1]]),
        receivers=np.asarray([[2, 2], [0, -1]]),
    )

    network = _build_identity_graph_network()

    updated_single_graph = network(single_graph)
    updated_nested_graph = network(nested_graph)
    updated_batched_graph = network(implicitly_batched_graph)
    updated_padded_graph = network(padded_graph)
    jitted_updated_padded_graph = jax.jit(network)(padded_graph)

    result = BasicExampleResult(
        single_graph=single_graph,
        nested_graph=nested_graph,
        implicitly_batched_graph=implicitly_batched_graph,
        unbatched_graphs=unbatched_graphs,
        padded_graph=padded_graph,
        unpadded_graph=unpadded_graph,
        explicitly_batched_graph=explicitly_batched_graph,
        updated_single_graph=updated_single_graph,
        updated_nested_graph=updated_nested_graph,
        updated_batched_graph=updated_batched_graph,
        updated_padded_graph=updated_padded_graph,
        jitted_updated_padded_graph=jitted_updated_padded_graph,
    )

    if log_output:
        logging.info("Single graph: %r", result.single_graph)
        logging.info("Nested graph: %r", result.nested_graph)
        logging.info("Implicitly batched graph: %r", result.implicitly_batched_graph)
        logging.info("Unbatched graphs: %r", result.unbatched_graphs)
        logging.info("Padded graph: %r", result.padded_graph)
        logging.info("Unpadded graph: %r", result.unpadded_graph)
        logging.info("Explicitly batched graph: %r", result.explicitly_batched_graph)
        logging.info("Updated single graph: %r", result.updated_single_graph)
        logging.info("Updated nested graph: %r", result.updated_nested_graph)
        logging.info("Updated batched graph: %r", result.updated_batched_graph)
        logging.info("Updated padded graph: %r", result.updated_padded_graph)
        logging.info(
            "JIT-updated padded graph: %r",
            result.jitted_updated_padded_graph,
        )
        logging.info("basic.py complete")

    return result


def main() -> None:
    """Runs the example from the command line."""

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run()


if __name__ == "__main__":
    main()
