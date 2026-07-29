import jax
from typing import Callable, Hashable
from collections.abc import Iterable, Mapping


# As of 04/2020 pytype doesn't support recursive types.
# pytype: disable=not-supported-yet
type ArrayTree = jax.Array | Iterable[ArrayTree] | Mapping[Hashable, ArrayTree]

# All features will be an ArrayTree.
NodeFeatures = EdgeFeatures = SenderFeatures = ReceiverFeatures = Globals = ArrayTree

# Signature:
# (edges of each node to be aggregated, segment ids, number of segments) ->
# aggregated edges
AggregateEdgesToNodesFn = Callable[
    [EdgeFeatures, jax.Array, int], NodeFeatures]


# Signature:
# (nodes of each graph to be aggregated, segment ids, number of segments) ->
# aggregated nodes
AggregateNodesToGlobalsFn = Callable[
  [NodeFeatures, jax.Array, int],
  Globals
]

# Signature:
# (edges of each graph to be aggregated, segment ids, number of segments) ->
# aggregated edges
AggregateEdgesToGlobalsFn = Callable[
  [EdgeFeatures, jax.Array, int],
  Globals
]

# Signature:
# (edge features, sender node features, receiver node features, globals) ->
# attention weights
AttentionLogitFn = Callable[
    [EdgeFeatures, SenderFeatures, ReceiverFeatures, Globals], ArrayTree]

# Signature:
# (edge features, weights) -> edge features for node update
AttentionReduceFn = Callable[[EdgeFeatures, ArrayTree], EdgeFeatures]

# Signature:
# (edges to be normalized, segment ids, number of segments) ->
# normalized edges
AttentionNormalizeFn = Callable[
  [EdgeFeatures, jax.Array, int], 
  EdgeFeatures
]

# Signature:
# (edge features, sender node features, receiver node features, globals) ->
# updated edge features
GNUpdateEdgeFn = Callable[
    [EdgeFeatures, SenderFeatures, ReceiverFeatures, Globals], 
    EdgeFeatures
]

# Signature:
# (node features, outgoing edge features, incoming edge features,
#  globals) -> updated node features
GNUpdateNodeFn = Callable[
    [NodeFeatures, SenderFeatures, ReceiverFeatures, Globals], 
    NodeFeatures
]

GNUpdateGlobalFn = Callable[
  [NodeFeatures, EdgeFeatures, Globals], 
  Globals
]

