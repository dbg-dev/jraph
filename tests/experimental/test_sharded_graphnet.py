# Copyright 2020 DeepMind Technologies Limited.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for sharded graphnet."""



import functools
import jax
import jraph
import numpy as np
import pytest

from jraph.graph import GraphsTuple
from jraph.models import GraphNetwork
from jraph.experimental.sharded_graphnet import (
    ShardedEdgesGraphsTuple, 
    ShardedEdgesGraphNetwork, 
    graphs_tuple_to_broadcasted_sharded_graphs_tuple, 
    broadcasted_sharded_graphs_tuple_to_graphs_tuple
)
from jraph.utils import batch_np


def test_expected_device_count():
    assert jax.local_device_count() >= 3


def _get_graphs_from_n_edge(n_edge: list[int]) -> GraphsTuple:
  """Get a graphs tuple from n_edge."""
  graphs = []
  for el in n_edge:
    graphs.append(
        jraph.GraphsTuple(
            nodes=np.random.uniform(size=(128, 2)),
            edges=np.random.uniform(size=(el, 2)),
            senders=np.random.choice(128, el),
            receivers=np.random.choice(128, el),
            n_edge=np.array([el]),
            n_node=np.array([128]),
            globals=np.array([[el]]),
        ))
  graphs = batch_np(graphs)
  return graphs


def get_graphs_tuples(n_edge, sharded_n_edge, device_graph_idx) -> tuple[GraphsTuple, ShardedEdgesGraphsTuple]:
  sharded_n_edge = np.array(sharded_n_edge)
  device_graph_idx = np.array(device_graph_idx)
  devices = len(sharded_n_edge)
  graphs = _get_graphs_from_n_edge(n_edge)
  sharded_senders = np.reshape(graphs.senders, [devices, -1])
  sharded_receivers = np.reshape(graphs.receivers, [devices, -1])
  sharded_edges = np.reshape(graphs.edges,
                             [devices, -1, graphs.edges.shape[-1]])
  # Broadcast replicated features to have a devices leading axis.
  broadcast = lambda x: np.broadcast_to(x[None, :], [devices] + list(x.shape))

  sharded_graphs = ShardedEdgesGraphsTuple(
      device_senders=sharded_senders,
      device_receivers=sharded_receivers,
      device_edges=sharded_edges,
      device_n_edge=sharded_n_edge,
      nodes=broadcast(graphs.nodes),
      senders=broadcast(graphs.senders),
      receivers=broadcast(graphs.receivers),
      device_graph_idx=device_graph_idx,
      globals=broadcast(graphs.globals),
      n_node=broadcast(graphs.n_node),
      n_edge=broadcast(graphs.n_edge))
  return graphs, sharded_graphs





@pytest.mark.parametrize(
    "n_edge",
    [
        pytest.param(
            [3, 5, 4, 3, 3],
            id="split-intermediate",
        ),
        pytest.param(
            [1, 2, 5, 4, 6],
            id="split-zero-last-edge",
        ),
        pytest.param(
            [1, 11],
            id="split-one-over-multiple",
        ),
    ],
)
def test_sharded_same_as_non_sharded(
    n_edge: list[int],
) -> None:
    in_tuple = _get_graphs_from_n_edge(n_edge)
    devices = 3

    sharded_tuple = graphs_tuple_to_broadcasted_sharded_graphs_tuple(in_tuple, devices)

    update_fn = jraph.concatenated_args(lambda x: x)

    sharded_gn = ShardedEdgesGraphNetwork(
        update_fn,
        update_fn,
        update_fn,
        num_shards=devices,
    )
    gn = GraphNetwork(
        update_fn,
        update_fn,
        update_fn,
    )

    sharded_out = jax.pmap(sharded_gn, axis_name="i",)(sharded_tuple)
    expected_out = gn(in_tuple)
    reduced_out = broadcasted_sharded_graphs_tuple_to_graphs_tuple(sharded_out)

    jax.tree.map(
        functools.partial(
            np.testing.assert_allclose,
            atol=1e-5,
            rtol=1e-5,
        ),
        expected_out,
        reduced_out,
    )