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
"""Tests for jraph.models."""

from collections.abc import Callable
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jraph import (
    GAT,
    DeepSets,
    GraphMapFeatures,
    GraphNetGAT,
    GraphNetwork,
    GraphsTuple,
    InteractionNetwork,
    RelationNetwork,
    segment_sum,
)

type GraphApplyFn = Callable[[GraphsTuple], GraphsTuple]
type GraphExpectedFn = Callable[[GraphsTuple], tuple[GraphsTuple, GraphsTuple]]
type GraphTransform = Callable[[GraphsTuple], GraphsTuple]


def _identity_node_update(
    nodes: Any,
    _sent_edges: Any,
    _received_edges: Any,
    _globals: Any,
) -> Any:
    return nodes


def _identity_edge_update(
    edges: Any,
    _sender_nodes: Any,
    _receiver_nodes: Any,
    _globals: Any,
) -> Any:
    return edges


def _identity_global_update(
    _aggregated_nodes: Any,
    _aggregated_edges: Any,
    globals_: Any,
) -> Any:
    return globals_


def _constant_attention_logit(
    _edges: Any,
    _sender_nodes: Any,
    _receiver_nodes: Any,
    _globals: Any,
) -> jax.Array:
    return jnp.array(1.0)


def _identity_attention_reduce(edges: Any, _weights: Any) -> Any:
    return edges


def _assert_trees_allclose(
    actual: GraphsTuple,
    expected: GraphsTuple,
) -> None:
    jax.tree.map(
        np.testing.assert_allclose,
        actual,
        expected,
    )


def _apply(
    function: GraphApplyFn,
    graph: GraphsTuple,
    *,
    use_jit: bool,
) -> GraphsTuple:
    return jax.jit(function)(graph) if use_jit else function(graph)


def _apply_with_expected(
    function: GraphExpectedFn,
    graph: GraphsTuple,
    *,
    use_jit: bool,
) -> tuple[GraphsTuple, GraphsTuple]:
    return jax.jit(function)(graph) if use_jit else function(graph)


def _apply_graph_network(graph: GraphsTuple) -> GraphsTuple:
    network = GraphNetwork(
        update_edge_fn=_identity_edge_update,
        update_node_fn=_identity_node_update,
        update_global_fn=_identity_global_update,
    )
    return network(graph)


def _apply_graph_network_without_global_update(
    graph: GraphsTuple,
) -> GraphsTuple:
    network = GraphNetwork(
        update_edge_fn=_identity_edge_update,
        update_node_fn=_identity_node_update,
        update_global_fn=None,
    )
    return network(graph)


def _apply_graph_network_without_node_update(
    graph: GraphsTuple,
) -> GraphsTuple:
    network = GraphNetwork(
        update_edge_fn=_identity_edge_update,
        update_node_fn=None,
        update_global_fn=_identity_global_update,
    )
    return network(graph)


def _apply_graph_network_without_edge_update(
    graph: GraphsTuple,
) -> GraphsTuple:
    network = GraphNetwork(
        update_edge_fn=None,
        update_node_fn=_identity_node_update,
        update_global_fn=_identity_global_update,
    )
    return network(graph)


def _apply_attention_graph_network(graph: GraphsTuple) -> GraphsTuple:
    network = GraphNetwork(
        update_edge_fn=_identity_edge_update,
        update_node_fn=_identity_node_update,
        update_global_fn=_identity_global_update,
        attention_logit_fn=_constant_attention_logit,
        attention_reduce_fn=_identity_attention_reduce,
    )
    return network(graph)


def _apply_graph_net_gat(graph: GraphsTuple) -> GraphsTuple:
    network = GraphNetGAT(
        update_edge_fn=_identity_edge_update,
        update_node_fn=_identity_node_update,
        attention_logit_fn=_constant_attention_logit,
        attention_reduce_fn=_identity_attention_reduce,
        update_global_fn=_identity_global_update,
    )
    return network(graph)


def _apply_multi_head_attention_graph_network(
    graph: GraphsTuple,
) -> GraphsTuple:
    def update_edge_fn(
        edges: Any,
        _sender_nodes: Any,
        _receiver_nodes: Any,
        _globals: Any,
    ) -> Any:
        return jax.tree.map(
            lambda leaf: jnp.stack([leaf, leaf, leaf]),
            edges,
        )

    def attention_logit_fn(
        edges: Any,
        _sender_nodes: Any,
        _receiver_nodes: Any,
        _globals: Any,
    ) -> Any:
        return jax.tree.map(
            lambda leaf: jnp.sum(leaf, axis=-1),
            edges,
        )

    def attention_reduce_fn(edges: Any, _weights: Any) -> Any:
        return jax.tree.map(lambda leaf: leaf[0], edges)

    network = GraphNetwork(
        update_edge_fn=jax.vmap(update_edge_fn),
        update_node_fn=jax.vmap(_identity_node_update),
        update_global_fn=_identity_global_update,
        attention_logit_fn=jax.vmap(attention_logit_fn),
        attention_reduce_fn=jax.vmap(attention_reduce_fn),
    )
    return network(graph)


def _apply_interaction_network(
    graph: GraphsTuple,
) -> tuple[GraphsTuple, GraphsTuple]:
    def update_node_fn(nodes: jax.Array, received_edges: jax.Array) -> jax.Array:
        return jnp.concatenate((nodes, received_edges), axis=-1)

    def update_edge_fn(
        edges: jax.Array,
        sender_nodes: jax.Array,
        receiver_nodes: jax.Array,
    ) -> jax.Array:
        return jnp.concatenate(
            (edges, sender_nodes, receiver_nodes),
            axis=-1,
        )

    actual = InteractionNetwork(
        update_edge_fn,
        update_node_fn,
    )(graph)

    nodes = cast(jax.Array, graph.nodes)
    edges = cast(jax.Array, graph.edges)
    senders = graph.senders
    receivers = graph.receivers

    expected_edges = jnp.concatenate(
        (edges, nodes[senders], nodes[receivers]),
        axis=-1,
    )
    aggregated_nodes = segment_sum(
        expected_edges,
        receivers,
        num_segments=len(nodes),
    )
    expected_nodes = jnp.concatenate(
        (nodes, aggregated_nodes),
        axis=-1,
    )
    expected = graph._replace(
        edges=expected_edges,
        nodes=expected_nodes,
    )
    return actual, expected


def _apply_graph_map_features(
    graph: GraphsTuple,
) -> tuple[GraphsTuple, GraphsTuple]:
    def double(value: jax.Array) -> jax.Array:
        return value * 2

    actual = GraphMapFeatures(
        double,
        double,
        double,
    )(graph)

    nodes = cast(jax.Array, graph.nodes)
    edges = cast(jax.Array, graph.edges)
    globals_ = cast(jax.Array, graph.globals)
    expected = graph._replace(
        nodes=nodes * 2,
        edges=edges * 2,
        globals=globals_ * 2,
    )
    return actual, expected


def _apply_relation_network(
    graph: GraphsTuple,
) -> tuple[GraphsTuple, GraphsTuple]:
    def edge_fn(
        sender_nodes: jax.Array,
        receiver_nodes: jax.Array,
    ) -> jax.Array:
        return jnp.concatenate((sender_nodes, receiver_nodes), axis=-1)

    def global_fn(edges: jax.Array) -> jax.Array:
        return edges * 2

    actual = RelationNetwork(edge_fn, global_fn)(graph)

    nodes = cast(jax.Array, graph.nodes)
    edges = cast(jax.Array, graph.edges)
    expected_edges = jnp.concatenate(
        (nodes[graph.senders], nodes[graph.receivers]),
        axis=-1,
    )
    num_graphs = len(graph.n_edge)
    edge_graph_indices = jnp.repeat(
        jnp.arange(num_graphs),
        graph.n_edge,
        total_repeat_length=edges.shape[0],
    )
    aggregated_edges = segment_sum(
        expected_edges,
        edge_graph_indices,
        num_segments=num_graphs,
    )
    expected = graph._replace(
        edges=expected_edges,
        globals=aggregated_edges * 2,
    )
    return actual, expected


def _apply_deep_sets(
    graph: GraphsTuple,
) -> tuple[GraphsTuple, GraphsTuple]:
    def node_fn(nodes: jax.Array, globals_: jax.Array) -> jax.Array:
        return jnp.concatenate((nodes, globals_), axis=-1)

    def global_fn(nodes: jax.Array) -> jax.Array:
        return nodes * 2

    actual = DeepSets(node_fn, global_fn)(graph)

    nodes = cast(jax.Array, graph.nodes)
    globals_ = cast(jax.Array, graph.globals)
    num_graphs = len(graph.n_node)
    num_nodes = len(nodes)
    broadcasted_globals = jnp.repeat(
        globals_,
        graph.n_node,
        total_repeat_length=num_nodes,
        axis=0,
    )
    expected_nodes = jnp.concatenate(
        (nodes, broadcasted_globals),
        axis=-1,
    )
    node_graph_indices = jnp.repeat(
        jnp.arange(num_graphs),
        graph.n_node,
        total_repeat_length=num_nodes,
    )
    expected = graph._replace(
        nodes=expected_nodes,
        globals=segment_sum(
            expected_nodes,
            node_graph_indices,
            num_segments=num_graphs,
        )
        * 2,
    )
    return actual, expected


def _apply_gat(
    graph: GraphsTuple,
) -> tuple[GraphsTuple, GraphsTuple]:
    def attention_query_fn(nodes: Any) -> Any:
        return jax.tree.map(
            lambda leaf: jnp.stack([leaf, leaf, leaf], axis=2),
            nodes,
        )

    def attention_logit_fn(
        sender_attributes: jax.Array,
        receiver_attributes: jax.Array,
        edge_attributes: Any,
    ) -> jax.Array:
        del edge_attributes
        return (sender_attributes == receiver_attributes) + (
            sender_attributes != receiver_attributes
        ) * -1e10

    def node_update_fn(nodes: jax.Array) -> jax.Array:
        return jnp.mean(nodes, axis=2)

    network = GAT(
        attention_query_fn,
        attention_logit_fn,
        node_update_fn,
    )

    nodes = cast(jax.Array, graph.nodes)
    expected = graph._replace(nodes=jnp.asarray(nodes, dtype=jnp.float32))
    return network(expected), expected


def _make_nested_array_tree(array: jax.Array) -> dict[str, Any]:
    return {
        "a": array,
        "b": [
            jnp.ones_like(array),
            {"c": jnp.zeros_like(array)},
        ],
    }


def _make_nested_batched_graph() -> GraphsTuple:
    return GraphsTuple(
        n_node=jnp.array([1, 3, 1, 0, 2, 0, 0]),
        n_edge=jnp.array([2, 5, 0, 0, 1, 0, 0]),
        nodes=_make_nested_array_tree(jnp.arange(14).reshape(7, 2)),
        edges=_make_nested_array_tree(jnp.arange(24).reshape(8, 3)),
        globals=_make_nested_array_tree(jnp.arange(14).reshape(7, 2)),
        senders=jnp.array([0, 0, 1, 1, 2, 3, 3, 6]),
        receivers=jnp.array([0, 0, 2, 1, 3, 2, 1, 5]),
    )


def _make_array_batched_graph() -> GraphsTuple:
    return GraphsTuple(
        n_node=jnp.array([1, 3, 1, 0, 2, 0, 0]),
        n_edge=jnp.array([1, 7, 1, 0, 3, 0, 0]),
        nodes=jnp.arange(14).reshape(7, 2),
        edges=jnp.arange(36).reshape(12, 3),
        globals=jnp.arange(14).reshape(7, 2),
        senders=jnp.array([0, 1, 2, 3, 4, 5, 6, 1, 2, 3, 3, 6]),
        receivers=jnp.array([0, 1, 2, 3, 4, 5, 6, 2, 3, 2, 1, 5]),
    )


def _without_globals(graph: GraphsTuple) -> GraphsTuple:
    return graph._replace(globals=None)


def _with_empty_globals(graph: GraphsTuple) -> GraphsTuple:
    return graph._replace(globals=[])


def _without_edges(graph: GraphsTuple) -> GraphsTuple:
    return graph._replace(edges=None)


def _with_empty_edges(graph: GraphsTuple) -> GraphsTuple:
    return graph._replace(edges=[])


GRAPH_NETWORK_CASES = [
    pytest.param(_apply_graph_network, id="graph-network"),
    pytest.param(
        _apply_graph_network_without_node_update,
        id="without-node-update",
    ),
    pytest.param(
        _apply_graph_network_without_edge_update,
        id="without-edge-update",
    ),
    pytest.param(
        _apply_graph_network_without_global_update,
        id="without-global-update",
    ),
    pytest.param(_apply_attention_graph_network, id="attention"),
    pytest.param(
        _apply_multi_head_attention_graph_network,
        id="multi-head-attention",
    ),
    pytest.param(_apply_graph_net_gat, id="graph-net-gat"),
]

GRAPH_NETWORK_NONE_CASES = GRAPH_NETWORK_CASES[:4]

GRAPH_TRANSFORM_CASES = [
    pytest.param(_without_globals, id="no-globals"),
    pytest.param(_with_empty_globals, id="empty-globals"),
    pytest.param(_without_edges, id="no-edges"),
    pytest.param(_with_empty_edges, id="empty-edges"),
]

GNN_CASES = [
    pytest.param(_apply_interaction_network, id="interaction-network"),
    pytest.param(_apply_graph_map_features, id="graph-map-features"),
    pytest.param(_apply_gat, id="gat"),
    pytest.param(_apply_relation_network, id="relation-network"),
    pytest.param(_apply_deep_sets, id="deep-sets"),
]

JIT_CASES = [
    pytest.param(False, id="eager"),
    pytest.param(True, id="jit"),
]


@pytest.mark.parametrize("network_fn", GRAPH_NETWORK_CASES)
@pytest.mark.parametrize("use_jit", JIT_CASES)
def test_graph_network_preserves_features(
    network_fn: GraphApplyFn,
    use_jit: bool,
) -> None:
    graph = _make_nested_batched_graph()
    actual = _apply(network_fn, graph, use_jit=use_jit)
    _assert_trees_allclose(actual, graph)


@pytest.mark.parametrize("transform", GRAPH_TRANSFORM_CASES)
@pytest.mark.parametrize("network_fn", GRAPH_NETWORK_NONE_CASES)
@pytest.mark.parametrize("use_jit", JIT_CASES)
def test_graph_network_handles_missing_or_empty_features(
    transform: GraphTransform,
    network_fn: GraphApplyFn,
    use_jit: bool,
) -> None:
    graph = transform(_make_nested_batched_graph())
    actual = _apply(network_fn, graph, use_jit=use_jit)
    _assert_trees_allclose(actual, graph)


@pytest.mark.parametrize("network_fn", GNN_CASES)
@pytest.mark.parametrize("use_jit", JIT_CASES)
def test_gnn_model_matches_expected_output(
    network_fn: GraphExpectedFn,
    use_jit: bool,
) -> None:
    graph = _make_array_batched_graph()
    actual, expected = _apply_with_expected(
        network_fn,
        graph,
        use_jit=use_jit,
    )
    _assert_trees_allclose(actual, expected)


def _return_first(value: Any, *_args: Any) -> Any:
    return value


@pytest.mark.parametrize(
    ("attention_logit_fn", "attention_reduce_fn"),
    [
        pytest.param(_return_first, None, id="missing-reduce-fn"),
        pytest.param(None, _return_first, id="missing-logit-fn"),
    ],
)
def test_graph_network_requires_complete_attention_configuration(
    attention_logit_fn: Callable[..., Any] | None,
    attention_reduce_fn: Callable[..., Any] | None,
) -> None:
    with pytest.raises(
        ValueError,
        match=(
            r"attention_logit_fn and attention_reduce_fn "
            r"must both be supplied\."
        ),
    ):
        GraphNetwork(
            update_edge_fn=None,
            update_node_fn=None,
            attention_logit_fn=attention_logit_fn,
            attention_reduce_fn=attention_reduce_fn,
        )
