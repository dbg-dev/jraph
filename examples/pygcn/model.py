from typing import cast

import jax
from flax import nnx

import jraph


class TwoLayerGCN(nnx.Module):
    """Two-layer GCN for node classification."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        *,
        dropout_rate: float = 0.5,
        rngs: nnx.Rngs,
    ) -> None:
        if not 0.0 <= dropout_rate < 1.0:
            raise ValueError(f"dropout_rate must be in [0, 1), got {dropout_rate}")

        self.linear1: nnx.Linear = nnx.Linear(
            in_features=in_features,
            out_features=hidden_features,
            rngs=rngs,
        )
        self.linear2: nnx.Linear = nnx.Linear(
            in_features=hidden_features,
            out_features=out_features,
            rngs=rngs,
        )
        self.dropout: nnx.Dropout = nnx.Dropout(
            rate=dropout_rate,
            rngs=rngs,
        )

    def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        """Return the graph with node features replaced by class logits."""

        graph = jraph.GraphConvolution(
            update_node_fn=self.linear1,
            add_self_edges=True,
            symmetric_normalization=True,
        )(graph)

        nodes = cast(jax.Array, graph.nodes)
        nodes = jax.nn.relu(nodes)
        nodes = self.dropout(nodes)

        graph = graph._replace(nodes=nodes)

        return jraph.GraphConvolution(
            update_node_fn=self.linear2,
            add_self_edges=True,
            symmetric_normalization=True,
        )(graph)
