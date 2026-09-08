"""Tests for examples._graphs."""

import jax
import jax.numpy as jnp
import numpy as np

import jraph


def test_pad_with_graphs_as_jax_normalizes_array_leaves() -> None:
    graph = jraph.GraphsTuple(
        n_node=jnp.asarray([2], dtype=jnp.int32),
        n_edge=jnp.asarray([0], dtype=jnp.int32),
        nodes=jnp.ones((2, 3), dtype=jnp.float32),
        edges=None,
        globals=jnp.zeros((1, 3), dtype=jnp.float32),
        senders=jnp.asarray([], dtype=jnp.int32),
        receivers=jnp.asarray([], dtype=jnp.int32),
    )

    padded = jraph.pad_with_graphs(
        graph,
        n_node=4,
        n_edge=0,
    )

    leaves = jax.tree.leaves(padded)
    assert leaves
    assert all(isinstance(leaf, jax.Array) for leaf in leaves)

    nodes = padded.nodes
    assert isinstance(nodes, jax.Array)
    updated_nodes = nodes.at[0].set(np.asarray([2.0, 2.0, 2.0], dtype=np.float32))
    np.testing.assert_array_equal(
        updated_nodes[0],
        np.asarray([2.0, 2.0, 2.0], dtype=np.float32),
    )
