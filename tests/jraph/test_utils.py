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
"""Tests for jraph.utils."""

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jraph import graph
from jraph import utils


Tree = Any
SegmentFunction = Callable[..., jax.Array]


# ---------------------------------------------------------------------------
# Shared assertions and graph fixtures.
# ---------------------------------------------------------------------------


def _assert_tree_allclose(
    actual: Tree,
    expected: Tree,
    *,
    atol: float = 1e-7,
    rtol: float = 1e-7,
) -> None:
    jax.tree.map(
        lambda actual_leaf, expected_leaf: np.testing.assert_allclose(
            actual_leaf,
            expected_leaf,
            atol=atol,
            rtol=rtol,
        ),
        actual,
        expected,
    )


def _assert_tree_array_equal(actual: Tree, expected: Tree) -> None:
    jax.tree.map(np.testing.assert_array_equal, actual, expected)


def _make_nest(array: Any) -> dict[str, Any]:
    """Return a small representative pytree containing ``array``."""
    return {
        "a": array,
        "b": [jnp.ones_like(array), {"c": jnp.zeros_like(array)}],
    }


def _get_random_graph(
    rng: np.random.Generator,
    *,
    max_n_graph: int = 10,
    include_node_features: bool = True,
    include_edge_features: bool = True,
    include_globals: bool = True,
) -> graph.GraphsTuple:
    n_graph = int(rng.integers(1, max_n_graph + 1))
    n_node = rng.integers(0, 10, size=n_graph, dtype=np.int32)
    n_edge = rng.integers(0, 20, size=n_graph, dtype=np.int32)

    # A graph with no nodes cannot contain edges.
    n_edge[n_node == 0] = 0

    senders: list[int] = []
    receivers: list[int] = []
    offset = 0

    for nodes_in_graph, edges_in_graph in zip(n_node, n_edge, strict=True):
        if edges_in_graph:
            senders.extend(
                rng.integers(0, nodes_in_graph, size=edges_in_graph) + offset
            )
            receivers.extend(
                rng.integers(0, nodes_in_graph, size=edges_in_graph) + offset
            )
        offset += int(nodes_in_graph)

    nodes = (
        jnp.asarray(rng.random((int(n_node.sum()), 4)))
        if include_node_features
        else None
    )
    edges = (
        jnp.asarray(rng.random((int(n_edge.sum()), 3)))
        if include_edge_features
        else None
    )
    globals_ = (
        jnp.asarray(rng.random((n_graph, 5))) if include_globals else None
    )

    return graph.GraphsTuple(
        n_node=jnp.asarray(n_node),
        n_edge=jnp.asarray(n_edge),
        nodes=nodes,
        edges=edges,
        globals=globals_,
        senders=jnp.asarray(senders, dtype=jnp.int32),
        receivers=jnp.asarray(receivers, dtype=jnp.int32),
    )


def _get_list_and_batched_graph(
) -> tuple[list[graph.GraphsTuple], graph.GraphsTuple]:
    """Return individual graphs and their expected batched representation."""
    batched_graph = graph.GraphsTuple(
        n_node=jnp.array([1, 3, 1, 0, 2, 0, 0]),
        n_edge=jnp.array([2, 5, 0, 0, 1, 0, 0]),
        nodes=_make_nest(jnp.arange(14).reshape(7, 2)),
        edges=_make_nest(jnp.arange(24).reshape(8, 3)),
        globals=_make_nest(jnp.arange(14).reshape(7, 2)),
        senders=jnp.array([0, 0, 1, 1, 2, 3, 3, 6]),
        receivers=jnp.array([0, 0, 2, 1, 3, 2, 1, 5]),
    )

    graphs = [
        graph.GraphsTuple(
            n_node=jnp.array([1]),
            n_edge=jnp.array([2]),
            nodes=_make_nest(jnp.array([[0, 1]])),
            edges=_make_nest(jnp.array([[0, 1, 2], [3, 4, 5]])),
            globals=_make_nest(jnp.array([[0, 1]])),
            senders=jnp.array([0, 0]),
            receivers=jnp.array([0, 0]),
        ),
        graph.GraphsTuple(
            n_node=jnp.array([3]),
            n_edge=jnp.array([5]),
            nodes=_make_nest(jnp.array([[2, 3], [4, 5], [6, 7]])),
            edges=_make_nest(
                jnp.array(
                    [
                        [6, 7, 8],
                        [9, 10, 11],
                        [12, 13, 14],
                        [15, 16, 17],
                        [18, 19, 20],
                    ]
                )
            ),
            globals=_make_nest(jnp.array([[2, 3]])),
            senders=jnp.array([0, 0, 1, 2, 2]),
            receivers=jnp.array([1, 0, 2, 1, 0]),
        ),
        graph.GraphsTuple(
            n_node=jnp.array([1]),
            n_edge=jnp.array([0]),
            nodes=_make_nest(jnp.array([[8, 9]])),
            edges=_make_nest(jnp.zeros((0, 3))),
            globals=_make_nest(jnp.array([[4, 5]])),
            senders=jnp.array([], dtype=jnp.int32),
            receivers=jnp.array([], dtype=jnp.int32),
        ),
        graph.GraphsTuple(
            n_node=jnp.array([0]),
            n_edge=jnp.array([0]),
            nodes=_make_nest(jnp.zeros((0, 2))),
            edges=_make_nest(jnp.zeros((0, 3))),
            globals=_make_nest(jnp.array([[6, 7]])),
            senders=jnp.array([], dtype=jnp.int32),
            receivers=jnp.array([], dtype=jnp.int32),
        ),
        graph.GraphsTuple(
            n_node=jnp.array([2]),
            n_edge=jnp.array([1]),
            nodes=_make_nest(jnp.array([[10, 11], [12, 13]])),
            edges=_make_nest(jnp.array([[21, 22, 23]])),
            globals=_make_nest(jnp.array([[8, 9]])),
            senders=jnp.array([1]),
            receivers=jnp.array([0]),
        ),
        graph.GraphsTuple(
            n_node=jnp.array([0]),
            n_edge=jnp.array([0]),
            nodes=_make_nest(jnp.zeros((0, 2))),
            edges=_make_nest(jnp.zeros((0, 3))),
            globals=_make_nest(jnp.array([[10, 11]])),
            senders=jnp.array([], dtype=jnp.int32),
            receivers=jnp.array([], dtype=jnp.int32),
        ),
        graph.GraphsTuple(
            n_node=jnp.array([0]),
            n_edge=jnp.array([0]),
            nodes=_make_nest(jnp.zeros((0, 2))),
            edges=_make_nest(jnp.zeros((0, 3))),
            globals=_make_nest(jnp.array([[12, 13]])),
            senders=jnp.array([], dtype=jnp.int32),
            receivers=jnp.array([], dtype=jnp.int32),
        ),
        # An entirely empty GraphsTuple is accepted by batch(), but unbatch()
        # deliberately omits it because it contains no graph.
        graph.GraphsTuple(
            n_node=jnp.array([], dtype=jnp.int32),
            n_edge=jnp.array([], dtype=jnp.int32),
            nodes=_make_nest(jnp.zeros((0, 2))),
            edges=_make_nest(jnp.zeros((0, 3))),
            globals=_make_nest(jnp.zeros((0, 2))),
            senders=jnp.array([], dtype=jnp.int32),
            receivers=jnp.array([], dtype=jnp.int32),
        ),
    ]

    return graphs, batched_graph


# ---------------------------------------------------------------------------
# Batching and padding.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("batch_fn", "unbatch_fn"),
    [
        pytest.param(utils.batch, utils.unbatch, id="jax"),
        pytest.param(utils.batch_np, utils.unbatch_np, id="numpy"),
    ],
)
def test_batch_matches_expected(
    batch_fn: Callable[[Sequence[graph.GraphsTuple]], graph.GraphsTuple],
    unbatch_fn: Callable[[graph.GraphsTuple], list[graph.GraphsTuple]],
) -> None:
    del unbatch_fn
    graphs, expected = _get_list_and_batched_graph()
    _assert_tree_allclose(batch_fn(graphs), expected)


@pytest.mark.parametrize(
    ("batch_fn", "unbatch_fn"),
    [
        pytest.param(utils.batch, utils.unbatch, id="jax"),
        pytest.param(utils.batch_np, utils.unbatch_np, id="numpy"),
    ],
)
def test_unbatch_matches_expected(
    batch_fn: Callable[[Sequence[graph.GraphsTuple]], graph.GraphsTuple],
    unbatch_fn: Callable[[graph.GraphsTuple], list[graph.GraphsTuple]],
) -> None:
    del batch_fn
    expected, batched = _get_list_and_batched_graph()
    _assert_tree_allclose(unbatch_fn(batched), expected[:-1])


_FEATURE_CASES = [
    pytest.param(True, True, False, id="globals-nodes"),
    pytest.param(True, False, True, id="globals-edges"),
    pytest.param(False, True, True, id="nodes-edges"),
]


@pytest.mark.parametrize(
    ("include_globals", "include_nodes", "include_edges"),
    _FEATURE_CASES,
)
def test_batch_unbatch_round_trip_for_batched_graph(
    include_globals: bool,
    include_nodes: bool,
    include_edges: bool,
) -> None:
    rng = np.random.default_rng(42)

    for _ in range(100):
        original = _get_random_graph(
            rng,
            include_globals=include_globals,
            include_node_features=include_nodes,
            include_edge_features=include_edges,
        )
        recovered = utils.batch(utils.unbatch(original))
        _assert_tree_allclose(recovered, original)


@pytest.mark.parametrize(
    ("include_globals", "include_nodes", "include_edges"),
    _FEATURE_CASES,
)
def test_unbatch_batch_round_trip_for_graph_list(
    include_globals: bool,
    include_nodes: bool,
    include_edges: bool,
) -> None:
    rng = np.random.default_rng(42)

    for _ in range(10):
        graphs = [
            _get_random_graph(
                rng,
                max_n_graph=1,
                include_globals=include_globals,
                include_node_features=include_nodes,
                include_edge_features=include_edges,
            )
            for _ in range(int(rng.integers(1, 10)))
        ]
        recovered = utils.unbatch(utils.batch(graphs))
        _assert_tree_allclose(recovered, graphs)


def test_pad_with_graphs_matches_expected() -> None:
    _, graphs = _get_list_and_batched_graph()

    actual = utils.pad_with_graphs(graphs, 10, 12, 9)
    expected = graph.GraphsTuple(
        n_node=jnp.concatenate([graphs.n_node, jnp.array([3, 0])]),
        n_edge=jnp.concatenate([graphs.n_edge, jnp.array([4, 0])]),
        nodes=jax.tree.map(
            lambda leaf: jnp.concatenate(
                [leaf, jnp.zeros((3, 2), dtype=leaf.dtype)]
            ),
            graphs.nodes,
        ),
        edges=jax.tree.map(
            lambda leaf: jnp.concatenate(
                [leaf, jnp.zeros((4, 3), dtype=leaf.dtype)]
            ),
            graphs.edges,
        ),
        globals=jax.tree.map(
            lambda leaf: jnp.concatenate(
                [leaf, jnp.zeros((2, 2), dtype=leaf.dtype)]
            ),
            graphs.globals,
        ),
        senders=jnp.concatenate([graphs.senders, jnp.array([7, 7, 7, 7])]),
        receivers=jnp.concatenate(
            [graphs.receivers, jnp.array([7, 7, 7, 7])]
        ),
    )

    _assert_tree_allclose(actual, expected)


def test_unpad_with_graphs_matches_expected() -> None:
    _, graphs = _get_list_and_batched_graph()

    actual = utils.unpad_with_graphs(graphs)
    expected = graph.GraphsTuple(
        n_node=jnp.array([1, 3, 1, 0]),
        n_edge=jnp.array([2, 5, 0, 0]),
        nodes=_make_nest(jnp.arange(10).reshape(5, 2)),
        edges=_make_nest(jnp.arange(21).reshape(7, 3)),
        globals=_make_nest(jnp.arange(8).reshape(4, 2)),
        senders=jnp.array([0, 0, 1, 1, 2, 3, 3]),
        receivers=jnp.array([0, 0, 2, 1, 3, 2, 1]),
    )

    _assert_tree_allclose(actual, expected)


@pytest.mark.parametrize(
    ("include_globals", "include_nodes", "include_edges"),
    _FEATURE_CASES,
)
def test_pad_unpad_round_trip(
    include_globals: bool,
    include_nodes: bool,
    include_edges: bool,
) -> None:
    rng = np.random.default_rng(42)

    for _ in range(100):
        original = _get_random_graph(
            rng,
            include_globals=include_globals,
            include_node_features=include_nodes,
            include_edge_features=include_edges,
        )
        padded = utils.pad_with_graphs(original, 101, 200, 11)
        _assert_tree_allclose(utils.unpad_with_graphs(padded), original)


def test_pad_unpad_round_trip_with_exact_edge_budget() -> None:
    rng = np.random.default_rng(42)
    original = _get_random_graph(rng)

    padded = utils.pad_with_graphs(
        original,
        n_node=int(np.asarray(original.n_node).sum()) + 1,
        n_edge=int(np.asarray(original.n_edge).sum()),
        n_graph=original.n_node.shape[0] + 1,
    )

    _assert_tree_allclose(utils.unpad_with_graphs(padded), original)


_PADDING_METADATA_CASES = [
    pytest.param(
        utils.get_number_of_padding_with_graphs_graphs,
        3,
        id="graph-count",
    ),
    pytest.param(
        utils.get_number_of_padding_with_graphs_nodes,
        2,
        id="node-count",
    ),
    pytest.param(
        utils.get_number_of_padding_with_graphs_edges,
        1,
        id="edge-count",
    ),
    pytest.param(
        utils.get_node_padding_mask,
        jnp.array([True, True, True, True, True, False, False]),
        id="node-mask",
    ),
    pytest.param(
        utils.get_edge_padding_mask,
        jnp.array([True, True, True, True, True, True, True, False]),
        id="edge-mask",
    ),
    pytest.param(
        utils.get_graph_padding_mask,
        jnp.array([True, True, True, True, False, False, False]),
        id="graph-mask",
    ),
]


@pytest.mark.parametrize("use_jit", [False, True], ids=["eager", "jit"])
@pytest.mark.parametrize(("metadata_fn", "expected"), _PADDING_METADATA_CASES)
def test_padding_metadata(
    metadata_fn: Callable[[graph.GraphsTuple], jax.Array],
    expected: Any,
    use_jit: bool,
) -> None:
    _, graphs = _get_list_and_batched_graph()
    apply_fn = jax.jit(metadata_fn) if use_jit else metadata_fn
    np.testing.assert_array_equal(apply_fn(graphs), expected)


# ---------------------------------------------------------------------------
# Segment operations.
# ---------------------------------------------------------------------------


def test_segment_sum() -> None:
    result = utils.segment_sum(
        jnp.arange(9),
        jnp.array([0, 1, 2, 0, 4, 0, 1, 1, 0]),
        6,
    )
    np.testing.assert_allclose(result, jnp.array([16, 14, 2, 0, 4, 0]))


def test_segment_sum_infers_num_segments() -> None:
    result = utils.segment_sum(
        jnp.arange(9),
        jnp.array([0, 1, 2, 0, 4, 0, 1, 1, 0]),
    )
    np.testing.assert_allclose(result, jnp.array([16, 14, 2, 0, 4]))


@pytest.mark.parametrize("nan_data", [False, True], ids=["finite", "nan"])
def test_segment_mean(nan_data: bool) -> None:
    data = jnp.arange(9, dtype=jnp.float32)
    segment_ids = jnp.array([0, 1, 2, 0, 4, 0, 1, 1, 0])
    expected = jnp.array([4, 14 / 3.0, 2, 0, 4, 0])

    if nan_data:
        data = data.at[0].set(jnp.nan)
        expected = expected.at[0].set(jnp.nan)

    np.testing.assert_allclose(utils.segment_mean(data, segment_ids, 6), expected)


@pytest.mark.parametrize("nan_data", [False, True], ids=["finite", "nan"])
def test_segment_variance(nan_data: bool) -> None:
    data = jnp.arange(8, dtype=jnp.float32)
    segment_ids = jnp.array([0, 0, 0, 1, 1, 2, 2, 2])
    expected = jnp.stack(
        [
            jnp.var(jnp.arange(3)),
            jnp.var(jnp.arange(3, 5)),
            jnp.var(jnp.arange(5, 8)),
        ]
    )

    if nan_data:
        data = data.at[0].set(jnp.nan)
        expected = expected.at[0].set(jnp.nan)

    np.testing.assert_allclose(
        utils.segment_variance(data, segment_ids, 3),
        expected,
    )


@pytest.mark.parametrize("nan_data", [False, True], ids=["finite", "nan"])
def test_segment_normalize(nan_data: bool) -> None:
    def normalize(values: jax.Array) -> jax.Array:
        return (values - jnp.mean(values)) * jax.lax.rsqrt(jnp.var(values))

    data = jnp.arange(8, dtype=jnp.float32)
    segment_ids = jnp.array([0, 0, 0, 1, 1, 2, 2, 2])
    expected = jnp.concatenate(
        [
            normalize(jnp.arange(3, dtype=jnp.float32)),
            normalize(jnp.arange(3, 5, dtype=jnp.float32)),
            normalize(jnp.arange(5, 8, dtype=jnp.float32)),
        ]
    )

    if nan_data:
        data = data.at[0].set(jnp.nan)
        expected = expected.at[:3].set(jnp.nan)

    np.testing.assert_allclose(
        utils.segment_normalize(data, segment_ids, 3),
        expected,
    )


_SEGMENT_LAYOUTS = [
    pytest.param(False, False, id="unsorted-repeated"),
    pytest.param(True, False, id="sorted-repeated"),
    pytest.param(True, True, id="sorted-unique"),
    pytest.param(False, True, id="unsorted-unique"),
]


def _segment_max_case(
    indices_are_sorted: bool,
    unique_indices: bool,
) -> tuple[jax.Array, jax.Array, jax.Array, int]:
    negative_infinity = jnp.iinfo(jnp.int32).min

    if unique_indices:
        data = jnp.arange(6)
        if indices_are_sorted:
            return data, jnp.arange(6), jnp.arange(6), 6
        return (
            data,
            jnp.array([1, 0, 2, 4, 3, -5]),
            jnp.array([1, 0, 2, 4, 3]),
            5,
        )

    data = jnp.arange(9)
    if indices_are_sorted:
        return (
            data,
            jnp.array([0, 0, 0, 1, 1, 1, 2, 3, 4]),
            jnp.array([2, 5, 6, 7, 8, negative_infinity]),
            6,
        )
    return (
        data,
        jnp.array([0, 1, 2, 0, 4, 0, 1, 1, -6]),
        jnp.array([5, 7, 2, negative_infinity, 4, negative_infinity]),
        6,
    )


def _segment_min_case(
    indices_are_sorted: bool,
    unique_indices: bool,
) -> tuple[jax.Array, jax.Array, jax.Array, int]:
    positive_infinity = jnp.iinfo(jnp.int32).max

    if unique_indices:
        data = jnp.arange(6)
        if indices_are_sorted:
            return data, jnp.arange(6), jnp.arange(6), 6
        return (
            data,
            jnp.array([1, 0, 2, 4, 3, -5]),
            jnp.array([1, 0, 2, 4, 3]),
            5,
        )

    data = jnp.arange(9)
    if indices_are_sorted:
        return (
            data,
            jnp.array([0, 0, 0, 1, 1, 1, 2, 3, 4]),
            jnp.array([0, 3, 6, 7, 8, positive_infinity]),
            6,
        )
    return (
        data,
        jnp.array([0, 1, 2, 0, 4, 0, 1, 1, -6]),
        jnp.array([0, 1, 2, positive_infinity, 4, positive_infinity]),
        6,
    )


def _segment_max_constant_case(
    indices_are_sorted: bool,
    unique_indices: bool,
    *,
    two_dimensional: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array, int]:
    if unique_indices:
        base = jnp.arange(6)
        data = (
            jnp.stack([base, jnp.arange(6, 0, -1)], axis=1)
            if two_dimensional
            else base.astype(jnp.float32)
        )
        if indices_are_sorted:
            expected = (
                jnp.array([[0, 6], [1, 5], [2, 4], [3, 3], [4, 2], [5, 1]])
                if two_dimensional
                else jnp.array([0, 1, 2, 3, 4, 5, 0], dtype=jnp.float32)
            )
            return data, jnp.arange(6), expected, 6 if two_dimensional else 7

        expected = (
            jnp.array([[1, 5], [0, 6], [2, 4], [4, 2], [3, 3]])
            if two_dimensional
            else jnp.array([1, 0, 2, 4, 3], dtype=jnp.float32)
        )
        return data, jnp.array([1, 0, 2, 4, 3, -5]), expected, 5

    base = jnp.arange(9)
    data = (
        jnp.stack([base, jnp.arange(9, 0, -1)], axis=1)
        if two_dimensional
        else base.astype(jnp.float32)
    )
    if indices_are_sorted:
        expected = (
            jnp.array([[2, 9], [5, 6], [6, 3], [7, 2], [8, 1], [0, 0]])
            if two_dimensional
            else jnp.array([2, 5, 6, 7, 8, 0], dtype=jnp.float32)
        )
        return (
            data,
            jnp.array([0, 0, 0, 1, 1, 1, 2, 3, 4]),
            expected,
            6,
        )

    expected = (
        jnp.array([[5, 9], [7, 8], [2, 7], [0, 0], [4, 5], [0, 0]])
        if two_dimensional
        else jnp.array([5, 7, 2, 0, 4, 0], dtype=jnp.float32)
    )
    return (
        data,
        jnp.array([0, 1, 2, 0, 4, 0, 1, 1, -6]),
        expected,
        6,
    )


def _segment_min_constant_case(
    indices_are_sorted: bool,
    unique_indices: bool,
    *,
    two_dimensional: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array, int]:
    if unique_indices:
        base = jnp.arange(6)
        data = (
            jnp.stack([base, jnp.arange(6, 0, -1)], axis=1)
            if two_dimensional
            else base.astype(jnp.float32)
        )
        if indices_are_sorted:
            expected = (
                jnp.array([[0, 6], [1, 5], [2, 4], [3, 3], [4, 2], [5, 1]])
                if two_dimensional
                else jnp.array([0, 1, 2, 3, 4, 5], dtype=jnp.float32)
            )
            return data, jnp.arange(6), expected, 6

        expected = (
            jnp.array([[1, 5], [0, 6], [2, 4], [4, 2], [3, 3]])
            if two_dimensional
            else jnp.array([1, 0, 2, 4, 3], dtype=jnp.float32)
        )
        return data, jnp.array([1, 0, 2, 4, 3, -5]), expected, 5

    base = jnp.arange(9)
    data = (
        jnp.stack([base, jnp.arange(9, 0, -1)], axis=1)
        if two_dimensional
        else base.astype(jnp.float32)
    )
    if indices_are_sorted:
        expected = (
            jnp.array([[0, 7], [3, 4], [6, 3], [7, 2], [8, 1], [0, 0]])
            if two_dimensional
            else jnp.array([0, 3, 6, 7, 8, 0], dtype=jnp.float32)
        )
        return (
            data,
            jnp.array([0, 0, 0, 1, 1, 1, 2, 3, 4]),
            expected,
            6,
        )

    expected = (
        jnp.array([[0, 4], [1, 2], [2, 7], [0, 0], [4, 5], [0, 0]])
        if two_dimensional
        else jnp.array([0, 1, 2, 0, 4, 0], dtype=jnp.float32)
    )
    return (
        data,
        jnp.array([0, 1, 2, 0, 4, 0, 1, 1, -6]),
        expected,
        6,
    )


def _apply_segment_function(
    function: SegmentFunction,
    data: jax.Array,
    segment_ids: jax.Array,
    num_segments: int,
    indices_are_sorted: bool,
    unique_indices: bool,
    *,
    use_jit: bool,
) -> jax.Array:
    apply_fn = (
        jax.jit(function, static_argnums=(2, 3, 4)) if use_jit else function
    )
    return apply_fn(
        data,
        segment_ids,
        num_segments,
        indices_are_sorted,
        unique_indices,
    )


@pytest.mark.parametrize("use_jit", [False, True], ids=["eager", "jit"])
@pytest.mark.parametrize(
    ("indices_are_sorted", "unique_indices"),
    _SEGMENT_LAYOUTS,
)
def test_segment_max(
    indices_are_sorted: bool,
    unique_indices: bool,
    use_jit: bool,
) -> None:
    data, segment_ids, expected, num_segments = _segment_max_case(
        indices_are_sorted,
        unique_indices,
    )
    actual = _apply_segment_function(
        utils.segment_max,
        data,
        segment_ids,
        num_segments,
        indices_are_sorted,
        unique_indices,
        use_jit=use_jit,
    )
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize(
    ("indices_are_sorted", "unique_indices"),
    _SEGMENT_LAYOUTS,
)
def test_segment_max_infers_num_segments(
    indices_are_sorted: bool,
    unique_indices: bool,
) -> None:
    data, segment_ids, expected, _ = _segment_max_case(
        indices_are_sorted,
        unique_indices,
    )
    actual = utils.segment_max(
        data,
        segment_ids,
        indices_are_sorted=indices_are_sorted,
        unique_indices=unique_indices,
    )
    inferred_segments = int(np.asarray(segment_ids).max()) + 1
    np.testing.assert_allclose(actual, expected[:inferred_segments])


@pytest.mark.parametrize("use_jit", [False, True], ids=["eager", "jit"])
@pytest.mark.parametrize(
    ("indices_are_sorted", "unique_indices"),
    _SEGMENT_LAYOUTS,
)
def test_segment_min(
    indices_are_sorted: bool,
    unique_indices: bool,
    use_jit: bool,
) -> None:
    data, segment_ids, expected, num_segments = _segment_min_case(
        indices_are_sorted,
        unique_indices,
    )
    actual = _apply_segment_function(
        utils.segment_min,
        data,
        segment_ids,
        num_segments,
        indices_are_sorted,
        unique_indices,
        use_jit=use_jit,
    )
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize(
    ("indices_are_sorted", "unique_indices"),
    _SEGMENT_LAYOUTS,
)
def test_segment_min_infers_num_segments(
    indices_are_sorted: bool,
    unique_indices: bool,
) -> None:
    data, segment_ids, expected, _ = _segment_min_case(
        indices_are_sorted,
        unique_indices,
    )
    actual = utils.segment_min(
        data,
        segment_ids,
        indices_are_sorted=indices_are_sorted,
        unique_indices=unique_indices,
    )
    inferred_segments = int(np.asarray(segment_ids).max()) + 1
    np.testing.assert_allclose(actual, expected[:inferred_segments])


@pytest.mark.parametrize("use_jit", [False, True], ids=["eager", "jit"])
@pytest.mark.parametrize(
    ("indices_are_sorted", "unique_indices"),
    _SEGMENT_LAYOUTS,
)
def test_segment_max_or_constant(
    indices_are_sorted: bool,
    unique_indices: bool,
    use_jit: bool,
) -> None:
    data, segment_ids, expected, num_segments = _segment_max_constant_case(
        indices_are_sorted,
        unique_indices,
    )
    actual = _apply_segment_function(
        utils.segment_max_or_constant,
        data,
        segment_ids,
        num_segments,
        indices_are_sorted,
        unique_indices,
        use_jit=use_jit,
    )
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize(
    ("indices_are_sorted", "unique_indices"),
    _SEGMENT_LAYOUTS,
)
def test_segment_max_or_constant_infers_num_segments(
    indices_are_sorted: bool,
    unique_indices: bool,
) -> None:
    data, segment_ids, expected, _ = _segment_max_constant_case(
        indices_are_sorted,
        unique_indices,
    )
    actual = utils.segment_max_or_constant(
        data,
        segment_ids,
        indices_are_sorted=indices_are_sorted,
        unique_indices=unique_indices,
    )
    inferred_segments = int(np.asarray(segment_ids).max()) + 1
    np.testing.assert_allclose(actual, expected[:inferred_segments])


@pytest.mark.parametrize("use_jit", [False, True], ids=["eager", "jit"])
@pytest.mark.parametrize(
    ("indices_are_sorted", "unique_indices"),
    _SEGMENT_LAYOUTS,
)
def test_segment_max_or_constant_has_finite_gradients(
    indices_are_sorted: bool,
    unique_indices: bool,
    use_jit: bool,
) -> None:
    data, segment_ids, _, num_segments = _segment_max_constant_case(
        indices_are_sorted,
        unique_indices,
    )

    def objective(values: jax.Array) -> jax.Array:
        result = utils.segment_max_or_constant(
            values,
            segment_ids,
            num_segments,
            indices_are_sorted,
            unique_indices,
        )
        return jnp.sum(result)

    grad_fn = jax.jit(jax.grad(objective)) if use_jit else jax.grad(objective)
    assert bool(jnp.all(jnp.isfinite(grad_fn(data))))


@pytest.mark.parametrize("use_jit", [False, True], ids=["eager", "jit"])
@pytest.mark.parametrize(
    ("indices_are_sorted", "unique_indices"),
    _SEGMENT_LAYOUTS,
)
def test_segment_max_or_constant_2d(
    indices_are_sorted: bool,
    unique_indices: bool,
    use_jit: bool,
) -> None:
    data, segment_ids, expected, num_segments = _segment_max_constant_case(
        indices_are_sorted,
        unique_indices,
        two_dimensional=True,
    )
    actual = _apply_segment_function(
        utils.segment_max_or_constant,
        data,
        segment_ids,
        num_segments,
        indices_are_sorted,
        unique_indices,
        use_jit=use_jit,
    )
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize("use_jit", [False, True], ids=["eager", "jit"])
@pytest.mark.parametrize(
    ("indices_are_sorted", "unique_indices"),
    _SEGMENT_LAYOUTS,
)
def test_segment_min_or_constant(
    indices_are_sorted: bool,
    unique_indices: bool,
    use_jit: bool,
) -> None:
    data, segment_ids, expected, num_segments = _segment_min_constant_case(
        indices_are_sorted,
        unique_indices,
    )
    actual = _apply_segment_function(
        utils.segment_min_or_constant,
        data,
        segment_ids,
        num_segments,
        indices_are_sorted,
        unique_indices,
        use_jit=use_jit,
    )
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize(
    ("indices_are_sorted", "unique_indices"),
    _SEGMENT_LAYOUTS,
)
def test_segment_min_or_constant_infers_num_segments(
    indices_are_sorted: bool,
    unique_indices: bool,
) -> None:
    data, segment_ids, expected, _ = _segment_min_constant_case(
        indices_are_sorted,
        unique_indices,
    )
    actual = utils.segment_min_or_constant(
        data,
        segment_ids,
        indices_are_sorted=indices_are_sorted,
        unique_indices=unique_indices,
    )
    inferred_segments = int(np.asarray(segment_ids).max()) + 1
    np.testing.assert_allclose(actual, expected[:inferred_segments])


@pytest.mark.parametrize("use_jit", [False, True], ids=["eager", "jit"])
@pytest.mark.parametrize(
    ("indices_are_sorted", "unique_indices"),
    _SEGMENT_LAYOUTS,
)
def test_segment_min_or_constant_has_finite_gradients(
    indices_are_sorted: bool,
    unique_indices: bool,
    use_jit: bool,
) -> None:
    data, segment_ids, _, num_segments = _segment_min_constant_case(
        indices_are_sorted,
        unique_indices,
    )

    def objective(values: jax.Array) -> jax.Array:
        result = utils.segment_min_or_constant(
            values,
            segment_ids,
            num_segments,
            indices_are_sorted,
            unique_indices,
        )
        return jnp.sum(result)

    grad_fn = jax.jit(jax.grad(objective)) if use_jit else jax.grad(objective)
    assert bool(jnp.all(jnp.isfinite(grad_fn(data))))


@pytest.mark.parametrize("use_jit", [False, True], ids=["eager", "jit"])
@pytest.mark.parametrize(
    ("indices_are_sorted", "unique_indices"),
    _SEGMENT_LAYOUTS,
)
def test_segment_min_or_constant_2d(
    indices_are_sorted: bool,
    unique_indices: bool,
    use_jit: bool,
) -> None:
    data, segment_ids, expected, num_segments = _segment_min_constant_case(
        indices_are_sorted,
        unique_indices,
        two_dimensional=True,
    )
    actual = _apply_segment_function(
        utils.segment_min_or_constant,
        data,
        segment_ids,
        num_segments,
        indices_are_sorted,
        unique_indices,
        use_jit=use_jit,
    )
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize("use_jit", [False, True], ids=["eager", "jit"])
@pytest.mark.parametrize("nan_data", [False, True], ids=["finite", "nan"])
def test_segment_softmax_with_explicit_num_segments(
    nan_data: bool,
    use_jit: bool,
) -> None:
    data = jnp.arange(9, dtype=jnp.float32)
    segment_ids = jnp.array([0, 1, 2, 0, 4, 0, 1, 1, 0])
    expected = jnp.array(
        [
            3.1741429e-04,
            1.8088353e-03,
            1.0,
            6.3754367e-03,
            1.0,
            4.7108460e-02,
            2.6845494e-01,
            7.2973621e-01,
            9.4619870e-01,
        ]
    )

    if nan_data:
        data = data.at[0].set(jnp.nan)
        expected = expected.at[jnp.array([0, 3, 5, 8])].set(jnp.nan)

    apply_fn = (
        jax.jit(utils.segment_softmax, static_argnums=2)
        if use_jit
        else utils.segment_softmax
    )
    np.testing.assert_allclose(apply_fn(data, segment_ids, 6), expected)


@pytest.mark.parametrize("nan_data", [False, True], ids=["finite", "nan"])
def test_segment_softmax_infers_num_segments(nan_data: bool) -> None:
    data = jnp.arange(9, dtype=jnp.float32)
    segment_ids = jnp.array([0, 1, 2, 0, 4, 0, 1, 1, 0])
    expected = jnp.array(
        [
            3.1741429e-04,
            1.8088353e-03,
            1.0,
            6.3754367e-03,
            1.0,
            4.7108460e-02,
            2.6845494e-01,
            7.2973621e-01,
            9.4619870e-01,
        ]
    )

    if nan_data:
        data = data.at[0].set(jnp.nan)
        expected = expected.at[jnp.array([0, 3, 5, 8])].set(jnp.nan)

    np.testing.assert_allclose(utils.segment_softmax(data, segment_ids), expected)


@pytest.mark.parametrize("use_jit", [False, True], ids=["eager", "jit"])
def test_partition_softmax_with_explicit_partition_sum(use_jit: bool) -> None:
    data = jnp.arange(9)
    partitions = jnp.array([3, 2, 4])
    expected = np.array(
        [
            0.090031,
            0.244728,
            0.665241,
            0.268941,
            0.731059,
            0.032059,
            0.087144,
            0.236883,
            0.643914,
        ]
    )

    apply_fn = (
        jax.jit(utils.partition_softmax, static_argnums=2)
        if use_jit
        else utils.partition_softmax
    )
    np.testing.assert_allclose(
        apply_fn(data, partitions, 9),
        expected,
        atol=1e-5,
        rtol=1e-5,
    )


def test_partition_softmax_infers_partition_sum() -> None:
    data = jnp.arange(9)
    partitions = jnp.array([3, 2, 4])
    expected = np.array(
        [
            0.090031,
            0.244728,
            0.665241,
            0.268941,
            0.731059,
            0.032059,
            0.087144,
            0.236883,
            0.643914,
        ]
    )

    np.testing.assert_allclose(
        utils.partition_softmax(data, partitions),
        expected,
        atol=1e-5,
        rtol=1e-5,
    )


# ---------------------------------------------------------------------------
# Graph construction and argument wrappers.
# ---------------------------------------------------------------------------


_FULLY_CONNECTED_CASES = [
    pytest.param(1, 1, False, False, id="one-no-features"),
    pytest.param(5, 5, False, False, id="five-no-features"),
    pytest.param(1, 1, True, False, id="one-node-features"),
    pytest.param(5, 5, False, True, id="five-global-features"),
    pytest.param(5, 5, True, True, id="five-all-features"),
    pytest.param(0, 1, False, False, id="zero-nodes"),
    pytest.param(1, 0, False, False, id="zero-graphs"),
]


@pytest.mark.parametrize("use_jit", [False, True], ids=["eager", "jit"])
@pytest.mark.parametrize(
    ("n_node", "n_graph", "include_nodes", "include_globals"),
    _FULLY_CONNECTED_CASES,
)
def test_get_fully_connected_graph_shapes(
    n_node: int,
    n_graph: int,
    include_nodes: bool,
    include_globals: bool,
    use_jit: bool,
) -> None:
    rng = np.random.default_rng(42)
    node_features = (
        rng.random((n_node * n_graph, 32)) if include_nodes else None
    )
    global_features = rng.random((n_graph, 32)) if include_globals else None

    apply_fn = (
        jax.jit(utils.get_fully_connected_graph, static_argnums=(0, 1))
        if use_jit
        else utils.get_fully_connected_graph
    )
    result = apply_fn(n_node, n_graph, node_features, global_features)

    if include_nodes:
        assert result.nodes is not None
        assert len(result.nodes) == n_node * n_graph
    if include_globals:
        assert result.globals is not None
        assert len(result.globals) == n_graph

    assert len(result.senders) == n_node**2 * n_graph
    assert len(result.receivers) == n_node**2 * n_graph
    np.testing.assert_allclose(result.n_node, jnp.array([n_node] * n_graph))


@pytest.mark.parametrize(
    ("n_node", "n_graph"),
    [
        pytest.param(1, 1, id="one"),
        pytest.param(5, 5, id="five"),
        pytest.param(0, 1, id="zero-nodes"),
        pytest.param(1, 0, id="zero-graphs"),
    ],
)
def test_get_fully_connected_graph_sender_receiver_indices(
    n_node: int,
    n_graph: int,
) -> None:
    result = utils.get_fully_connected_graph(n_node, n_graph)

    if n_node:
        node_indices = np.arange(n_node)
        expected_senders = np.concatenate([node_indices] * n_node)
        expected_receivers = np.stack([node_indices] * n_node, axis=-1).reshape(-1)
    else:
        expected_senders = np.array([], dtype=np.int32)
        expected_receivers = np.array([], dtype=np.int32)

    for individual_graph in utils.unbatch(result):
        np.testing.assert_array_equal(individual_graph.senders, expected_senders)
        np.testing.assert_array_equal(
            individual_graph.receivers,
            expected_receivers,
        )


@pytest.mark.parametrize(
    ("n_node", "n_graph"),
    [
        pytest.param(1, 1, id="one"),
        pytest.param(5, 5, id="five"),
        pytest.param(0, 1, id="zero-nodes"),
        pytest.param(1, 0, id="zero-graphs"),
    ],
)
def test_get_fully_connected_graph_without_self_edges(
    n_node: int,
    n_graph: int,
) -> None:
    with_self_edges = utils.get_fully_connected_graph(
        n_node,
        n_graph,
        add_self_edges=True,
    )
    without_self_edges = utils.get_fully_connected_graph(
        n_node,
        n_graph,
        add_self_edges=False,
    )

    actual = set(
        zip(
            np.asarray(without_self_edges.senders).tolist(),
            np.asarray(without_self_edges.receivers).tolist(),
            strict=True,
        )
    )

    non_self_mask = with_self_edges.senders != with_self_edges.receivers
    expected = set(
        zip(
            np.asarray(with_self_edges.senders[non_self_mask]).tolist(),
            np.asarray(with_self_edges.receivers[non_self_mask]).tolist(),
            strict=True,
        )
    )

    assert actual == expected


@pytest.mark.parametrize(
    "add_self_edges",
    [
        pytest.param(True, id="with-self-edges"),
        pytest.param(False, id="without-self-edges"),
    ],
)
def test_get_fully_connected_graph_edge_order(add_self_edges: bool) -> None:
    result = utils.get_fully_connected_graph(
        n_node_per_graph=3,
        n_graph=1,
        add_self_edges=add_self_edges,
    )

    if add_self_edges:
        np.testing.assert_array_equal(result.senders, [0, 1, 2] * 3)
        np.testing.assert_array_equal(
            result.receivers,
            [0] * 3 + [1] * 3 + [2] * 3,
        )
    else:
        np.testing.assert_array_equal(result.senders, [1, 2, 2, 0, 0, 1])
        np.testing.assert_array_equal(result.receivers, [0, 0, 1, 1, 2, 2])


_CONCATENATED_ARGUMENT_CASES = [
    pytest.param([], {"a": np.array([10, 2])}, -1, id="kwargs-only"),
    pytest.param(
        [np.array([10, 5])],
        {"a": np.array([10, 2])},
        -1,
        id="one-arg",
    ),
    pytest.param(
        [np.array([10, 5]), np.array([10, 3])],
        {"a": np.array([10, 2])},
        -1,
        id="two-args",
    ),
    pytest.param(
        [np.array([10, 5]), np.array([10, 3])],
        {},
        -1,
        id="args-only",
    ),
    pytest.param(
        [{"a": np.array([10, 2]), "b": np.array([10, 4])}],
        {"c": np.array([10, 3])},
        1,
        id="nested-axis-one",
    ),
    pytest.param(
        [{"a": np.array([2, 10]), "b": np.array([4, 10])}],
        {"c": np.array([3, 10])},
        0,
        id="nested-axis-zero",
    ),
]


@pytest.mark.parametrize(
    ("args_shapes", "kwargs_shapes", "axis"),
    _CONCATENATED_ARGUMENT_CASES,
)
def test_concatenated_args(
    args_shapes: list[Any],
    kwargs_shapes: Mapping[str, np.ndarray],
    axis: int,
) -> None:
    rng = np.random.default_rng(42)
    args = jax.tree.map(
        lambda shape: rng.normal(size=tuple(np.asarray(shape))),
        args_shapes,
    )
    kwargs = {
        name: rng.normal(size=tuple(shape))
        for name, shape in kwargs_shapes.items()
    }

    @utils.concatenated_args(axis=axis)
    def identity(features: jax.Array) -> jax.Array:
        return features

    actual = identity(*args, **kwargs)
    expected = jnp.concatenate(
        [*jax.tree.leaves(args), *jax.tree.leaves(kwargs)],
        axis=axis,
    )
    np.testing.assert_allclose(actual, expected)


# ---------------------------------------------------------------------------
# Dynamic batching.
# ---------------------------------------------------------------------------


_DB_NUM_NODES = (10, 15)
_DB_NODE_SHAPE = (3, 4, 1)
_DB_NUM_EDGES = (12, 17)
_DB_EDGE_SHAPE = (4, 3)
_DB_GLOBAL_SHAPE = (2, 3, 1, 4)


def _make_dynamic_batch_graph(
    rng: np.random.Generator,
    *,
    add_globals: bool,
    num_nodes: tuple[int, ...] = _DB_NUM_NODES,
    num_edges: tuple[int, ...] = _DB_NUM_EDGES,
) -> graph.GraphsTuple:
    total_num_nodes = sum(num_nodes)
    total_num_edges = sum(num_edges)
    globals_ = (
        _make_nest(rng.normal(size=_DB_GLOBAL_SHAPE)) if add_globals else {}
    )

    return graph.GraphsTuple(
        nodes=_make_nest(
            rng.normal(size=(total_num_nodes, *_DB_NODE_SHAPE))
        ),
        edges=_make_nest(
            rng.normal(size=(total_num_edges, *_DB_EDGE_SHAPE))
        ),
        n_edge=np.array(num_edges),
        n_node=np.array(num_nodes),
        senders=rng.integers(
            0,
            total_num_nodes,
            size=total_num_edges,
            dtype=np.int32,
        ),
        receivers=rng.integers(
            0,
            total_num_nodes,
            size=total_num_edges,
            dtype=np.int32,
        ),
        globals=globals_,
    )


_DYNAMIC_BATCH_CASES = [
    pytest.param(
        True,
        {
            "n_node": sum(_DB_NUM_NODES) + 1,
            "n_edge": sum(_DB_NUM_EDGES) + 100,
            "n_graph": len(_DB_NUM_EDGES) + 100,
        },
        id="globals-node-budget",
    ),
    pytest.param(
        True,
        {
            "n_node": sum(_DB_NUM_NODES) + 100,
            "n_edge": sum(_DB_NUM_EDGES),
            "n_graph": len(_DB_NUM_EDGES) + 100,
        },
        id="globals-edge-budget",
    ),
    pytest.param(
        True,
        {
            "n_node": sum(_DB_NUM_NODES) + 100,
            "n_edge": sum(_DB_NUM_EDGES) + 100,
            "n_graph": len(_DB_NUM_EDGES) + 1,
        },
        id="globals-graph-budget",
    ),
    pytest.param(
        True,
        {
            "n_node": sum(_DB_NUM_NODES) + 5,
            "n_edge": sum(_DB_NUM_EDGES) + 5,
            "n_graph": len(_DB_NUM_EDGES) + 5,
        },
        id="globals-tight-budget",
    ),
    pytest.param(
        False,
        {
            "n_node": sum(_DB_NUM_NODES) + 5,
            "n_edge": sum(_DB_NUM_EDGES) + 5,
            "n_graph": len(_DB_NUM_EDGES) + 5,
        },
        id="no-globals-tight-budget",
    ),
]


@pytest.mark.parametrize(("use_globals", "batch_kwargs"), _DYNAMIC_BATCH_CASES)
def test_dynamically_batch(
    use_globals: bool,
    batch_kwargs: dict[str, int],
) -> None:
    rng = np.random.default_rng(42)
    graphs = [
        _make_dynamic_batch_graph(rng, add_globals=use_globals)
        for _ in range(4)
    ]
    input_graphs = [*graphs, *utils.unbatch_np(graphs[-1])]

    batches = list(utils.dynamically_batch(iter(input_graphs), **batch_kwargs))

    assert len(batches) == 5
    for batch in batches:
        for nodes in jax.tree.leaves(batch.nodes):
            assert nodes.shape[0] == batch_kwargs["n_node"]
        for edges in jax.tree.leaves(batch.edges):
            assert edges.shape[0] == batch_kwargs["n_edge"]

        assert len(batch.n_node) == batch_kwargs["n_graph"]
        assert int(utils.get_number_of_padding_with_graphs_nodes(batch)) == (
            batch_kwargs["n_node"] - sum(_DB_NUM_NODES)
        )
        assert int(utils.get_number_of_padding_with_graphs_edges(batch)) == (
            batch_kwargs["n_edge"] - sum(_DB_NUM_EDGES)
        )


@pytest.mark.parametrize(
    ("batch_kwargs", "error_type", "match"),
    [
        pytest.param(
            {"n_node": 15, "n_edge": 50, "n_graph": 10},
            RuntimeError,
            r"Found graph bigger than batch size.*",
            id="too-many-nodes",
        ),
        pytest.param(
            {"n_node": 26, "n_edge": 15, "n_graph": 10},
            RuntimeError,
            r"Found graph bigger than batch size.*",
            id="too-many-edges",
        ),
        pytest.param(
            {"n_node": 50, "n_edge": 50, "n_graph": 1},
            ValueError,
            r"The number of graphs.*",
            id="too-many-graphs",
        ),
    ],
)
def test_dynamically_batch_rejects_graph_larger_than_budget(
    batch_kwargs: dict[str, int],
    error_type: type[Exception],
    match: str,
) -> None:
    rng = np.random.default_rng(42)
    graph_ = _make_dynamic_batch_graph(rng, add_globals=True)
    iterator = utils.dynamically_batch(iter([graph_]), **batch_kwargs)

    with pytest.raises(error_type, match=match):
        next(iterator)


def test_dynamically_batch_yields_accumulated_batch_before_error() -> None:
    rng = np.random.default_rng(42)
    small_graph = _make_dynamic_batch_graph(
        rng,
        add_globals=True,
        num_nodes=(5, 7),
        num_edges=(6, 8),
    )
    oversized_graph = _make_dynamic_batch_graph(rng, add_globals=True)
    iterator = utils.dynamically_batch(
        iter([small_graph, oversized_graph]),
        n_node=15,
        n_edge=15,
        n_graph=10,
    )

    next(iterator)
    with pytest.raises(RuntimeError, match=r"Found graph bigger than batch size.*"):
        next(iterator)


def test_dynamically_batch_requires_room_for_padding_graph() -> None:
    rng = np.random.default_rng(42)
    graph_ = _make_dynamic_batch_graph(rng, add_globals=True)
    iterator = utils.dynamically_batch(
        iter([graph_]),
        n_node=5,
        n_edge=5,
        n_graph=1,
    )

    with pytest.raises(ValueError, match=r"The number of graphs.*"):
        next(iterator)


# ---------------------------------------------------------------------------
# Padding-output masking and sparse adjacency conversion.
# ---------------------------------------------------------------------------


def _assert_zeroed_padding(
    padded_graph: graph.GraphsTuple,
    *,
    use_wrapper: bool,
) -> None:
    # Make every padded feature non-zero before zeroing it.
    padded_graph = padded_graph._replace(
        nodes=jax.tree.map(lambda value: value - 1.0, padded_graph.nodes),
        edges=jax.tree.map(lambda value: value - 1.0, padded_graph.edges),
        globals=jax.tree.map(lambda value: value - 1.0, padded_graph.globals),
    )
    expected_valid_graph = utils.unbatch(padded_graph)[0]

    if use_wrapper:
        zeroing_fn = utils.with_zero_out_padding_outputs(lambda value: value)
        zeroed_graph = zeroing_fn(padded_graph)
    else:
        zeroed_graph = utils.zero_out_padding(padded_graph)

    valid_graph, *padding_graphs = utils.unbatch(zeroed_graph)
    _assert_tree_array_equal(valid_graph, expected_valid_graph)

    for padding_graph in padding_graphs:
        for feature_tree in (
            padding_graph.nodes,
            padding_graph.edges,
            padding_graph.globals,
        ):
            for value in jax.tree.leaves(feature_tree):
                np.testing.assert_array_equal(value, jnp.zeros_like(value))


@pytest.mark.parametrize(
    "use_wrapper",
    [pytest.param(False, id="direct"), pytest.param(True, id="wrapper")],
)
@pytest.mark.parametrize(
    "padding_case",
    [
        pytest.param("all", id="all-features"),
        pytest.param("no-edge-padding", id="no-edge-padding"),
        pytest.param("minimal-graph-padding", id="minimal-graph-padding"),
    ],
)
def test_zero_out_padding_values(
    use_wrapper: bool,
    padding_case: str,
) -> None:
    rng = np.random.default_rng(42)
    original = _get_random_graph(rng, max_n_graph=1)
    current_nodes = int(np.asarray(original.n_node).sum())
    current_edges = int(np.asarray(original.n_edge).sum())

    if padding_case == "all":
        padded = utils.pad_with_graphs(
            original,
            n_node=20,
            n_edge=20,
            n_graph=3,
        )
    elif padding_case == "no-edge-padding":
        padded = utils.pad_with_graphs(
            original,
            n_node=current_nodes + 1,
            n_edge=current_edges,
            n_graph=3,
        )
    else:
        padded = utils.pad_with_graphs(
            original,
            n_node=current_nodes + 1,
            n_edge=current_edges,
            n_graph=2,
        )

    _assert_zeroed_padding(padded, use_wrapper=use_wrapper)


def _sparse_graph_cases() -> list[Any]:
    return [
        pytest.param(
            (np.array([0]), np.array([0]), np.array([2]), np.array([1])),
            graph.GraphsTuple(
                n_node=jnp.array([1]),
                n_edge=jnp.array([2]),
                nodes=None,
                edges=None,
                globals=None,
                senders=jnp.array([0, 0]),
                receivers=jnp.array([0, 0]),
            ),
            id="single-node-double-edge",
        ),
        pytest.param(
            (
                np.array([0, 0, 1, 2, 2]),
                np.array([0, 1, 2, 0, 1]),
                np.array([1, 1, 1, 1, 1]),
                np.array(3),
            ),
            graph.GraphsTuple(
                n_node=jnp.array([3]),
                n_edge=jnp.array([5]),
                nodes=None,
                edges=None,
                globals=None,
                senders=jnp.array([0, 0, 1, 2, 2]),
                receivers=jnp.array([0, 1, 2, 0, 1]),
            ),
            id="three-nodes",
        ),
        pytest.param(
            (np.array([]), np.array([]), np.array([]), np.array(1)),
            graph.GraphsTuple(
                n_node=jnp.array([1]),
                n_edge=jnp.array([0]),
                nodes=None,
                edges=None,
                globals=None,
                senders=jnp.array([]),
                receivers=jnp.array([]),
            ),
            id="one-node-no-edges",
        ),
        pytest.param(
            (np.array([]), np.array([]), np.array([]), np.array(0)),
            graph.GraphsTuple(
                n_node=jnp.array([0]),
                n_edge=jnp.array([0]),
                nodes=None,
                edges=None,
                globals=None,
                senders=jnp.array([]),
                receivers=jnp.array([]),
            ),
            id="empty-graph",
        ),
        pytest.param(
            (np.array([1]), np.array([0]), np.array([1]), np.array(2)),
            graph.GraphsTuple(
                n_node=jnp.array([2]),
                n_edge=jnp.array([1]),
                nodes=None,
                edges=None,
                globals=None,
                senders=jnp.array([1]),
                receivers=jnp.array([0]),
            ),
            id="two-nodes-one-edge",
        ),
    ]


@pytest.mark.parametrize(("sparse_matrix", "expected"), _sparse_graph_cases())
def test_sparse_matrix_to_graphs_tuple(
    sparse_matrix: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    expected: graph.GraphsTuple,
) -> None:
    senders, receivers, values, n_node = sparse_matrix
    actual = utils.sparse_matrix_to_graphs_tuple(
        senders,
        receivers,
        values,
        n_node,
    )
    _assert_tree_allclose(actual, expected)
