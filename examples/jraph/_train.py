"""Shared classification helpers for executable examples."""

from typing import NamedTuple, Protocol, cast

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from jraph import GraphsTuple


class ClassificationMetrics(NamedTuple):
    """Masked classification loss and accuracy."""

    loss: jax.Array
    accuracy: jax.Array


class GraphClassifier(Protocol):
    """Callable interface required by the shared classification steps."""

    def __call__(
        self,
        graph: GraphsTuple,
        /,
    ) -> jax.Array:
        """Return classification logits for a graph."""


def masked_classification_metrics(
    logits: jax.Array,
    labels: jax.Array,
    mask: jax.Array,
) -> ClassificationMetrics:
    """Compute mean cross-entropy and accuracy over selected examples."""

    if logits.shape[:-1] != labels.shape:
        raise ValueError(
            "labels must have the same leading shape as logits"
        )
    if mask.shape != labels.shape:
        raise ValueError("mask must have the same shape as labels")

    weights = mask.astype(logits.dtype)
    denominator = jnp.maximum(jnp.sum(weights), 1.0)

    per_item_loss = optax.softmax_cross_entropy_with_integer_labels(
        logits,
        labels,
    )
    loss = jnp.sum(per_item_loss * weights) / denominator

    correct = jnp.argmax(logits, axis=-1) == labels
    accuracy = (
        jnp.sum(correct.astype(logits.dtype) * weights) / denominator
    )
    return ClassificationMetrics(loss=loss, accuracy=accuracy)


@nnx.jit
def train_step(
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    graph: GraphsTuple,
    labels: jax.Array,
    mask: jax.Array,
) -> ClassificationMetrics:
    """Update an NNX classifier whose call returns logits."""

    def loss_fn(
        differentiable_model: nnx.Module,
    ) -> tuple[jax.Array, ClassificationMetrics]:
        classifier = cast(GraphClassifier, differentiable_model)
        logits = classifier(graph)
        metrics = masked_classification_metrics(logits, labels, mask)
        return metrics.loss, metrics

    (_, metrics), gradients = nnx.value_and_grad(
        loss_fn,
        has_aux=True,
    )(model)
    optimizer.update(model, gradients)
    return metrics


@nnx.jit
def eval_step(
    model: nnx.Module,
    graph: GraphsTuple,
    labels: jax.Array,
    mask: jax.Array,
) -> ClassificationMetrics:
    """Evaluate an NNX classifier whose call returns logits."""

    classifier = cast(GraphClassifier, model)
    return masked_classification_metrics(
        classifier(graph),
        labels,
        mask,
    )
