from pathlib import Path
from typing import cast

import jax
import numpy as np
from flax import nnx

from examples.pygcn.cora import load_cora
from examples.pygcn.model import TwoLayerGCN


DATA_PATH = (
    Path(__file__).parents[2]
    / "examples"
    / "pygcn"
    / "data"
    / "cora"
)


def test_two_layer_gcn_produces_node_logits() -> None:
    dataset = load_cora(DATA_PATH)

    nodes = cast(jax.Array, dataset.graph.nodes)

    model = TwoLayerGCN(
        in_features=nodes.shape[-1],
        hidden_features=16,
        out_features=len(dataset.class_names),
        rngs=nnx.Rngs(0),
    )
    model.eval()

    output = model(dataset.graph)
    logits = cast(jax.Array, output.nodes)

    assert logits.shape == (2708, 7)

    np.testing.assert_array_equal(
        output.senders,
        dataset.graph.senders,
    )
    np.testing.assert_array_equal(
        output.receivers,
        dataset.graph.receivers,
    )
    np.testing.assert_array_equal(
        output.n_node,
        dataset.graph.n_node,
    )
    np.testing.assert_array_equal(
        output.n_edge,
        dataset.graph.n_edge,
    )


def test_two_layer_gcn_parameter_shapes() -> None:
    model = TwoLayerGCN(
        in_features=1433,
        hidden_features=16,
        out_features=7,
        rngs=nnx.Rngs(0),
    )

    assert model.linear1.kernel.shape == (1433, 16)
    assert model.linear1.bias.shape == (16,)
    assert model.linear2.kernel.shape == (16, 7)
    assert model.linear2.bias.shape == (7,)


def test_two_layer_gcn_is_deterministic_in_eval_mode() -> None:
    dataset = load_cora(DATA_PATH)

    model = TwoLayerGCN(
        in_features=1433,
        hidden_features=16,
        out_features=7,
        rngs=nnx.Rngs(0),
    )
    model.eval()

    first = cast(jax.Array, model(dataset.graph).nodes)
    second = cast(jax.Array, model(dataset.graph).nodes)

    np.testing.assert_array_equal(first, second)