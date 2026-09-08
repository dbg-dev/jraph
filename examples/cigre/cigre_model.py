"""Two-block NNX GraphNetwork for the CIGRE MV example.

Architecture
------------

raw standardized graph
        |
        +-- node encoder
        +-- edge encoder
        +-- global encoder
                |
                v
        GraphNetwork block 1
          edge -> node -> global
                |
                v
        GraphNetwork block 2
          edge -> node -> global
          ^                 |
          |_________________|
          block-1 global state is available to the
          block-2 edge and node update functions
                |
                v
          global decoder
                |
                v
        three standardized targets

The model returns one prediction vector per graph, shape [n_graph, 3].

Usage:
    uv run python cigre_model.py cigre_mv_multiday.npz
    uv run python cigre_model.py cigre_mv_multiday.npz --index 123
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from cigre_dataset import CigreGraphDataset
from flax import nnx

import jraph


class MLP(nnx.Module):
    """Small two-layer MLP."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        *,
        rngs: nnx.Rngs,
    ):
        self.linear1 = nnx.Linear(
            input_size,
            hidden_size,
            rngs=rngs,
        )
        self.linear2 = nnx.Linear(
            hidden_size,
            output_size,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        x = nnx.gelu(self.linear1(x))
        return self.linear2(x)


class Encoder(nnx.Module):
    """Map raw feature vectors into a common latent space."""

    def __init__(
        self,
        input_size: int,
        latent_size: int,
        *,
        rngs: nnx.Rngs,
    ):
        self.linear = nnx.Linear(
            input_size,
            latent_size,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        return nnx.gelu(self.linear(x))


class GraphNetworkBlock(nnx.Module):
    """One fully learned Jraph GraphNetwork block."""

    def __init__(
        self,
        latent_size: int,
        *,
        rngs: nnx.Rngs,
    ):
        # edge + sender node + receiver node + global
        self.edge_mlp = MLP(
            4 * latent_size,
            latent_size,
            latent_size,
            rngs=rngs,
        )

        # node + aggregated sent edges + aggregated received edges + global
        self.node_mlp = MLP(
            4 * latent_size,
            latent_size,
            latent_size,
            rngs=rngs,
        )

        # aggregated nodes + aggregated edges + previous global
        self.global_mlp = MLP(
            3 * latent_size,
            latent_size,
            latent_size,
            rngs=rngs,
        )

    def _update_edge(
        self,
        edges: jax.Array,
        sender_nodes: jax.Array,
        receiver_nodes: jax.Array,
        globals_: jax.Array,
    ) -> jax.Array:
        features = jnp.concatenate(
            [edges, sender_nodes, receiver_nodes, globals_],
            axis=-1,
        )
        return self.edge_mlp(features)

    def _update_node(
        self,
        nodes: jax.Array,
        sent_edges: jax.Array,
        received_edges: jax.Array,
        globals_: jax.Array,
    ) -> jax.Array:
        features = jnp.concatenate(
            [nodes, sent_edges, received_edges, globals_],
            axis=-1,
        )
        return self.node_mlp(features)

    def _update_global(
        self,
        aggregated_nodes: jax.Array,
        aggregated_edges: jax.Array,
        globals_: jax.Array,
    ) -> jax.Array:
        features = jnp.concatenate(
            [aggregated_nodes, aggregated_edges, globals_],
            axis=-1,
        )
        return self.global_mlp(features)

    def __call__(
        self,
        graph: jraph.GraphsTuple,
    ) -> jraph.GraphsTuple:
        graph_network = jraph.GraphNetwork(
            update_edge_fn=self._update_edge,
            update_node_fn=self._update_node,
            update_global_fn=self._update_global,
        )
        return graph_network(graph)


class CigreGraphNetwork(nnx.Module):
    """Predict CIGRE grid-level operating quantities from a Jraph graph."""

    def __init__(
        self,
        *,
        node_input_size: int = 4,
        edge_input_size: int = 4,
        global_input_size: int = 1,
        latent_size: int = 32,
        output_size: int = 3,
        rngs: nnx.Rngs,
    ):
        self.node_encoder = Encoder(
            node_input_size,
            latent_size,
            rngs=rngs,
        )
        self.edge_encoder = Encoder(
            edge_input_size,
            latent_size,
            rngs=rngs,
        )
        self.global_encoder = Encoder(
            global_input_size,
            latent_size,
            rngs=rngs,
        )

        # Deliberately independent parameter sets.
        self.block1 = GraphNetworkBlock(
            latent_size,
            rngs=rngs,
        )
        self.block2 = GraphNetworkBlock(
            latent_size,
            rngs=rngs,
        )

        self.decoder = MLP(
            latent_size,
            latent_size,
            output_size,
            rngs=rngs,
        )

    def encode(
        self,
        graph: jraph.GraphsTuple,
    ) -> jraph.GraphsTuple:
        return graph._replace(
            nodes=self.node_encoder(graph.nodes),
            edges=self.edge_encoder(graph.edges),
            globals=self.global_encoder(graph.globals),
        )

    def __call__(
        self,
        graph: jraph.GraphsTuple,
    ) -> jax.Array:
        graph = self.encode(graph)
        graph = self.block1(graph)
        graph = self.block2(graph)
        return self.decoder(graph.globals)


@nnx.jit
def predict(
    model: CigreGraphNetwork,
    graph: jraph.GraphsTuple,
) -> jax.Array:
    return model(graph)


def loss_fn(
    model: CigreGraphNetwork,
    graph: jraph.GraphsTuple,
    target: jax.Array,
) -> jax.Array:
    prediction = model(graph)
    target = jnp.reshape(target, prediction.shape)
    return jnp.mean((prediction - target) ** 2)


def count_parameters(model: nnx.Module) -> int:
    state = nnx.state(model, nnx.Param)
    leaves = jax.tree.leaves(state)
    return int(
        sum(
            np.prod(np.asarray(leaf).shape)
            for leaf in leaves
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset",
        nargs="?",
        type=Path,
        default=Path("cigre_mv_multiday.npz"),
    )
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--latent-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset = CigreGraphDataset(args.dataset)
    sample = dataset.sample(args.index)

    model = CigreGraphNetwork(
        latent_size=args.latent_size,
        rngs=nnx.Rngs(args.seed),
    )

    eager_prediction = model(sample.graph)
    compiled_prediction = predict(model, sample.graph)

    eager_physical = dataset.inverse_target(
        eager_prediction[0]
    )
    compiled_physical = dataset.inverse_target(
        compiled_prediction[0]
    )

    loss, grads = nnx.value_and_grad(loss_fn)(
        model,
        sample.graph,
        sample.target,
    )

    grad_leaves = jax.tree.leaves(grads)
    gradients_finite = all(
        bool(jnp.all(jnp.isfinite(leaf)))
        for leaf in grad_leaves
    )

    print(f"Sample index:       {args.index}")
    print(f"Latent size:        {args.latent_size}")
    print(f"Parameter count:    {count_parameters(model):,}")

    print("\nInput graph")
    print("  nodes:   ", sample.graph.nodes.shape)
    print("  edges:   ", sample.graph.edges.shape)
    print("  globals: ", sample.graph.globals.shape)
    print("  n_node:  ", np.asarray(sample.graph.n_node))
    print("  n_edge:  ", np.asarray(sample.graph.n_edge))

    print("\nModel output")
    print("  shape:           ", eager_prediction.shape)
    print("  eager z-score:   ", np.asarray(eager_prediction))
    print("  compiled z-score:", np.asarray(compiled_prediction))
    print(
        "  eager/JIT close: ",
        bool(
            np.allclose(
                np.asarray(eager_prediction),
                np.asarray(compiled_prediction),
                rtol=1e-5,
                atol=1e-6,
            )
        ),
    )

    print("\nUntrained physical prediction")
    print("  eager:   ", np.asarray(eager_physical))
    print("  compiled:", np.asarray(compiled_physical))
    print("  target:  ", np.asarray(sample.target_physical))

    print("\nGradient smoke test")
    print("  loss:             ", float(loss))
    print("  gradient leaves:  ", len(grad_leaves))
    print("  gradients finite: ", gradients_finite)


if __name__ == "__main__":
    main()
