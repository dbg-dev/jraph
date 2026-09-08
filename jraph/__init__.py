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
"""Jraph."""


# move to uv build and versioning
from importlib.metadata import version as _package_version

from jraph._src.graph import GraphsTuple
from jraph._src.models import (
  GAT,
  DeepSets,
  EmbedEdgeFn,
  EmbedGlobalFn,
  EmbedNodeFn,
  GATAttentionLogitFn,
  GATAttentionQueryFn,
  GATNodeUpdateFn,
  GraphConvolution,
  GraphMapFeatures,
  GraphNetGAT,
  GraphNetwork,
  InteractionNetwork,
  InteractionUpdateEdgeFn,
  InteractionUpdateNodeFn,
  RelationNetwork,
)
from jraph._src.types import (
  AggregateEdgesToGlobalsFn,
  AggregateEdgesToNodesFn,
  AggregateNodesToGlobalsFn,
  ArrayTree,
  AttentionLogitFn,
  AttentionReduceFn,
  GNUpdateEdgeFn,
  GNUpdateGlobalFn,
  GNUpdateNodeFn,
  NodeFeatures,
)
from jraph._src.utils import (
  batch,
  batch_np,
  concatenated_args,
  dynamically_batch,
  get_edge_padding_mask,
  get_fully_connected_graph,
  get_graph_padding_mask,
  get_node_padding_mask,
  get_number_of_padding_with_graphs_edges,
  get_number_of_padding_with_graphs_graphs,
  get_number_of_padding_with_graphs_nodes,
  pad_with_graphs,
  partition_softmax,
  segment_max,
  segment_max_or_constant,
  segment_mean,
  segment_min,
  segment_min_or_constant,
  segment_normalize,
  segment_softmax,
  segment_sum,
  segment_variance,
  sparse_matrix_to_graphs_tuple,
  unbatch,
  unbatch_np,
  unpad_with_graphs,
  with_zero_out_padding_outputs,
  zero_out_padding,
)

__version__ = _package_version("jraph")

__all__ = (
  "GAT",
  "AggregateEdgesToGlobalsFn",
  "AggregateEdgesToNodesFn",
  "AggregateNodesToGlobalsFn",
  "ArrayTree",
  "AttentionLogitFn",
  "AttentionReduceFn",
  "DeepSets",
  "EmbedEdgeFn",
  "EmbedGlobalFn",
  "EmbedNodeFn",
  "GATAttentionLogitFn",
  "GATAttentionQueryFn",
  "GATNodeUpdateFn",
  "GNUpdateEdgeFn",
  "GNUpdateGlobalFn",
  "GNUpdateNodeFn",
  "GraphConvolution",
  "GraphMapFeatures",
  "GraphNetGAT",
  "GraphNetwork",
  "GraphsTuple",
  "InteractionNetwork",
  "InteractionUpdateEdgeFn",
  "InteractionUpdateNodeFn",
  "NodeFeatures",
  "RelationNetwork",
  "batch",
  "batch_np",
  "concatenated_args",
  "dynamically_batch",
  "get_edge_padding_mask",
  "get_fully_connected_graph",
  "get_graph_padding_mask",
  "get_node_padding_mask",
  "get_number_of_padding_with_graphs_edges",
  "get_number_of_padding_with_graphs_graphs",
  "get_number_of_padding_with_graphs_nodes",
  "pad_with_graphs",
  "partition_softmax",
  "segment_max",
  "segment_max_or_constant",
  "segment_mean",
  "segment_min",
  "segment_min_or_constant",
  "segment_normalize",
  "segment_softmax",
  "segment_sum",
  "segment_variance",
  "sparse_matrix_to_graphs_tuple",
  "unbatch",
  "unbatch_np",
  "unpad_with_graphs",
  "with_zero_out_padding_outputs",
  "zero_out_padding",
)

#  _________________________________________
# / Please don't use symbols in `_src` they \
# \ are not part of the Jraph public API.    /
#  -----------------------------------------
#         \   ^__^
#          \  (oo)\_______
#             (__)\       )\/\
#                 ||----w |
#                 ||     ||
#
try:
  del _src  # pylint: disable=undefined-variable
except NameError:
  pass
