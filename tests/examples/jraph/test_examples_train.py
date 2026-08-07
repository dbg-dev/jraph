"""Tests for examples._training."""

from flax import nnx
import jax
import jax.numpy as jnp
import jraph
import numpy as np
import optax
import pytest

from examples.jraph import _train


class LinearNodeClassifier(nnx.Module):
    """Small classifier used to exercise the shared NNX steps."""

    def __init__(self, *, rngs: nnx.Rngs) -> None:
        self.linear = nnx.Linear(2, 2, rngs=rngs)

    def __call__(self, graph: jraph.GraphsTuple) -> jax.Array:
        return self.linear(graph.nodes)


def _graph() -> jraph.GraphsTuple:
    return jraph.GraphsTuple(
        n_node=jnp.asarray([4], dtype=jnp.int32),
        n_edge=jnp.asarray([0], dtype=jnp.int32),
        nodes=jnp.asarray(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]
        ),
        edges=None,
        globals=None,
        senders=jnp.asarray([], dtype=jnp.int32),
        receivers=jnp.asarray([], dtype=jnp.int32),
    )


def test_masked_classification_metrics() -> None:
    logits = jnp.asarray(
        [[3.0, 0.0], [0.0, 3.0], [0.0, 3.0]]
    )
    labels = jnp.asarray([0, 1, 0], dtype=jnp.int32)
    mask = jnp.asarray([True, True, False])

    metrics = _train.masked_classification_metrics(
        logits,
        labels,
        mask,
    )

    expected_loss = np.mean(
        np.asarray(
            optax.softmax_cross_entropy_with_integer_labels(
                logits[:2],
                labels[:2],
            )
        )
    )
    np.testing.assert_allclose(metrics.loss, expected_loss)
    np.testing.assert_allclose(metrics.accuracy, 1.0)


def test_zero_mask_returns_finite_zero_metrics() -> None:
    metrics = _train.masked_classification_metrics(
        jnp.ones((3, 2)),
        jnp.zeros(3, dtype=jnp.int32),
        jnp.zeros(3, dtype=jnp.bool_),
    )

    np.testing.assert_array_equal(metrics.loss, 0.0)
    np.testing.assert_array_equal(metrics.accuracy, 0.0)


@pytest.mark.parametrize(
    ("labels_shape", "mask_shape", "message"),
    [
        ((2,), (3,), "labels must have the same leading shape"),
        ((3,), (2,), "mask must have the same shape"),
    ],
)
def test_invalid_shapes(
    labels_shape: tuple[int, ...],
    mask_shape: tuple[int, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _train.masked_classification_metrics(
            jnp.ones((3, 2)),
            jnp.zeros(labels_shape, dtype=jnp.int32),
            jnp.ones(mask_shape, dtype=jnp.bool_),
        )


def test_shared_nnx_train_step_reduces_loss() -> None:
    graph = _graph()
    labels = jnp.asarray([0, 1, 0, 1], dtype=jnp.int32)
    mask = jnp.ones(4, dtype=jnp.bool_)
    model = LinearNodeClassifier(rngs=nnx.Rngs(42))
    optimizer = nnx.Optimizer(
        model,
        optax.adam(1e-1),
        wrt=nnx.Param,
    )

    initial = _train.eval_step(model, graph, labels, mask)
    for _ in range(20):
        _train.train_step(model, optimizer, graph, labels, mask)
    final = _train.eval_step(model, graph, labels, mask)

    assert float(final.loss) < float(initial.loss)
    assert float(final.accuracy) >= 0.75
