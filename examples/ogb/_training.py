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

from collections.abc import Mapping
from typing import cast

import jax
import jax.numpy as jnp

import jraph


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
