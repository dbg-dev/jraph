from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from examples.pygcn.cora import load_cora
from examples.pygcn.model import TwoLayerGCN
from examples.pygcn.training import (
    eval_step,
    loss_fn,
    masked_accuracy,
    masked_cross_entropy,
    train_step,
)

DATA_PATH = (
    Path(__file__).parents[3]
    / "examples"
    / "pygcn"
    / "data"
    / "cora"
)

print(DATA_PATH)

def make_model() -> TwoLayerGCN:
    return TwoLayerGCN(
        in_features=1433,
        hidden_features=16,
        out_features=7,
        rngs=nnx.Rngs(0),
    )


def test_masked_cross_entropy_uses_only_selected_nodes() -> None:
    logits = jnp.asarray(
        [
            [10.0, -10.0],
            [-10.0, 10.0],
            [10.0, -10.0],
        ]
    )
    labels = jnp.asarray([0, 1, 1])
    mask = jnp.asarray([True, True, False])

    loss = masked_cross_entropy(logits, labels, mask)

    assert float(loss) < 1e-6


def test_masked_accuracy_uses_only_selected_nodes() -> None:
    logits = jnp.asarray(
        [
            [10.0, -10.0],
            [-10.0, 10.0],
            [10.0, -10.0],
        ]
    )
    labels = jnp.asarray([0, 1, 1])
    mask = jnp.asarray([True, True, False])

    accuracy = masked_accuracy(logits, labels, mask)

    assert float(accuracy) == 1.0


def test_loss_produces_finite_parameter_gradients() -> None:
    dataset = load_cora(DATA_PATH)
    model = make_model()
    model.eval()

    loss, grads = nnx.value_and_grad(loss_fn)(
        model,
        dataset.graph,
        dataset.labels,
        dataset.train_mask,
    )

    assert loss.shape == ()
    assert bool(jnp.isfinite(loss))

    grad_state = nnx.state(grads)
    grad_leaves = jax.tree.leaves(grad_state)

    assert grad_leaves
    assert all(
        bool(jnp.all(jnp.isfinite(gradient)))
        for gradient in grad_leaves
    )
    assert any(
        bool(jnp.any(gradient != 0))
        for gradient in grad_leaves
    )


def test_compiled_loss_and_grad_match_eager_execution() -> None:
    dataset = load_cora(DATA_PATH)

    eager_model = make_model()
    eager_model.eval()

    compiled_model = make_model()
    compiled_model.eval()

    eager_loss, eager_grads = nnx.value_and_grad(loss_fn)(
        eager_model,
        dataset.graph,
        dataset.labels,
        dataset.train_mask,
    )

    @nnx.jit
    def compiled_loss_and_grad(
        model: TwoLayerGCN,
        graph,
        labels: jax.Array,
        mask: jax.Array,
    ):
        return nnx.value_and_grad(loss_fn)(
            model,
            graph,
            labels,
            mask,
        )

    compiled_loss, compiled_grads = compiled_loss_and_grad(
        compiled_model,
        dataset.graph,
        dataset.labels,
        dataset.train_mask,
    )

    np.testing.assert_allclose(
        compiled_loss,
        eager_loss,
        rtol=1e-6,
        atol=1e-6,
    )

    eager_state = nnx.state(eager_grads)
    compiled_state = nnx.state(compiled_grads)

    jax.tree.map(
        lambda actual, expected: np.testing.assert_allclose(
            actual,
            expected,
            rtol=1e-5,
            atol=1e-6,
        ),
        compiled_state,
        eager_state,
    )


def test_training_reduces_cora_loss() -> None:
    dataset = load_cora(DATA_PATH)

    model = TwoLayerGCN(
        in_features=1433,
        hidden_features=16,
        out_features=7,
        # Disable dropout to make this unit test deterministic.
        dropout_rate=0.0,
        rngs=nnx.Rngs(0),
    )

    train_model = nnx.view(
        model,
        deterministic=False,
    )
    eval_model = nnx.view(
        model,
        deterministic=True,
    )

    optimizer = nnx.Optimizer(
        model,
        optax.adam(learning_rate=0.01),
        wrt=nnx.Param,
    )

    before = eval_step(
        eval_model,
        dataset.graph,
        dataset.labels,
        dataset.train_mask,
    )

    for _ in range(25):
        train_step(
            train_model,
            optimizer,
            dataset.graph,
            dataset.labels,
            dataset.train_mask,
        )

    after = eval_step(
        eval_model,
        dataset.graph,
        dataset.labels,
        dataset.train_mask,
    )

    assert float(after.loss) < float(before.loss)
    assert int(optimizer.step[...]) == 25