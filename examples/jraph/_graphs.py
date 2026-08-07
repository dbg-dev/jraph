"""Graph helpers shared by executable examples."""

from typing import cast

import jax
import jax.numpy as jnp
import jraph


def pad_with_graphs_as_jax(
    graph: jraph.GraphsTuple,
    *,
    n_node: int,
    n_edge: int,
    n_graph: int = 2,
) -> jraph.GraphsTuple:
    """Pad a graph and normalize every array leaf back to a JAX array.

    ``jraph.pad_with_graphs`` may construct its result with NumPy arrays.
    Example models expect JAX arrays for immutable ``.at`` updates, JIT, and
    consistent array semantics, so this helper establishes that boundary
    explicitly.
    """

    padded_graph = jraph.pad_with_graphs(
        graph,
        n_node=n_node,
        n_edge=n_edge,
        n_graph=n_graph,
    )
    return cast(
        jraph.GraphsTuple,
        jax.tree.map(jnp.asarray, padded_graph),
    )
