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

from .graph import GraphsTuple
from .types import AggregateEdgesToGlobalsFn
from .types import AggregateEdgesToNodesFn
from .types import AggregateNodesToGlobalsFn
from .types import AttentionLogitFn
from .types import AttentionReduceFn
from .models import DeepSets
from .models import EmbedEdgeFn
from .models import EmbedGlobalFn
from .models import EmbedNodeFn
from .models import GAT
from .models import GATAttentionLogitFn
from .models import GATAttentionQueryFn
from .models import GATNodeUpdateFn
from .models import GNUpdateEdgeFn
from .models import GNUpdateGlobalFn
from .models import GNUpdateNodeFn
from .models import GraphConvolution
from .models import GraphMapFeatures
from .models import GraphNetGAT
from .models import GraphNetwork
from .models import InteractionNetwork
from .models import InteractionUpdateEdgeFn
from .models import InteractionUpdateNodeFn
from .models import NodeFeatures
from .models import RelationNetwork
from .types import ArrayTree
from .utils import batch
from .utils import batch_np
from .utils import concatenated_args
from .utils import dynamically_batch
from .utils import get_edge_padding_mask
from .utils import get_fully_connected_graph
from .utils import get_graph_padding_mask
from .utils import get_node_padding_mask
from .utils import get_number_of_padding_with_graphs_edges
from .utils import get_number_of_padding_with_graphs_graphs
from .utils import get_number_of_padding_with_graphs_nodes
from .utils import pad_with_graphs
from .utils import partition_softmax
from .utils import segment_max
from .utils import segment_max_or_constant
from .utils import segment_mean
from .utils import segment_min
from .utils import segment_min_or_constant
from .utils import segment_normalize
from .utils import segment_softmax
from .utils import segment_sum
from .utils import segment_variance
from .utils import sparse_matrix_to_graphs_tuple
from .utils import unbatch
from .utils import unbatch_np
from .utils import unpad_with_graphs
from .utils import with_zero_out_padding_outputs
from .utils import zero_out_padding

# move to uv build and versioning
from importlib.metadata import version as _package_version

__version__ = _package_version("jraph")


__all__ = (
    "ArrayTree",
    "DeepSets",
    "GraphConvolution",
    "GraphMapFeatures",
    "InteractionNetwork",
    "RelationNetwork",
    "GraphNetGAT",
    "GAT",
    "GraphsTuple",
    "GraphNetwork",
    "NodeFeatures",
    "AggregateEdgesToNodesFn",
    "AggregateNodesToGlobalsFn",
    "AggregateEdgesToGlobalsFn",
    "AttentionLogitFn",
    "AttentionReduceFn",
    "GNUpdateEdgeFn",
    "GNUpdateNodeFn",
    "GNUpdateGlobalFn",
    "InteractionUpdateNodeFn",
    "InteractionUpdateEdgeFn",
    "EmbedEdgeFn",
    "EmbedNodeFn",
    "EmbedGlobalFn",
    "GATAttentionQueryFn",
    "GATAttentionLogitFn",
    "GATNodeUpdateFn",
    "batch",
    "batch_np",
    "unbatch",
    "unbatch_np",
    "pad_with_graphs",
    "get_number_of_padding_with_graphs_graphs",
    "get_number_of_padding_with_graphs_nodes",
    "get_number_of_padding_with_graphs_edges",
    "unpad_with_graphs",
    "get_node_padding_mask",
    "get_edge_padding_mask",
    "get_graph_padding_mask",
    "segment_max",
    "segment_max_or_constant",
    "segment_min_or_constant",
    "segment_softmax",
    "segment_sum",
    "partition_softmax",
    "concatenated_args",
    "get_fully_connected_graph",
    "dynamically_batch",
    "with_zero_out_padding_outputs",
    "zero_out_padding",
    "sparse_matrix_to_graphs_tuple",
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
