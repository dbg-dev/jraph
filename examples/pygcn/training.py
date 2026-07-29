from typing import NamedTuple, cast

import jax
import jax.numpy as jnp
import optax
from flax import nnx

import jraph

from .model import TwoLayerGCN


class StepMetrics(NamedTuple):
    loss: jax.Array
    accuracy: jax.Array


def masked_cross_entropy(
    logits: jax.Array,
    labels: jax.Array,
    mask: jax.Array,
) -> jax.Array:
    """Mean cross-entropy over selected nodes."""

    losses = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits,
        labels=labels,
    )

    weights = mask.astype(losses.dtype)
    total_weight = jnp.maximum(weights.sum(), 1.0)

    return jnp.sum(losses * weights) / total_weight


def masked_accuracy(
    logits: jax.Array,
    labels: jax.Array,
    mask: jax.Array,
) -> jax.Array:
    """Classification accuracy over selected nodes."""

    predictions = jnp.argmax(logits, axis=-1)
    correct = predictions == labels

    weights = mask.astype(jnp.float32)
    total_weight = jnp.maximum(weights.sum(), 1.0)

    return jnp.sum(correct * weights) / total_weight


def loss_and_logits(
    model: TwoLayerGCN,
    graph: jraph.GraphsTuple,
    labels: jax.Array,
    mask: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Return masked classification loss and node logits."""

    output = model(graph)
    logits = cast(jax.Array, output.nodes)
    loss = masked_cross_entropy(logits, labels, mask)

    return loss, logits


def loss_fn(
    model: TwoLayerGCN,
    graph: jraph.GraphsTuple,
    labels: jax.Array,
    mask: jax.Array,
) -> jax.Array:
    """Return masked node-classification loss."""

    loss, _ = loss_and_logits(model, graph, labels, mask)
    return loss


@nnx.jit
def train_step(
    model: TwoLayerGCN,
    optimizer: nnx.Optimizer,
    graph: jraph.GraphsTuple,
    labels: jax.Array,
    mask: jax.Array,
) -> StepMetrics:
    """Apply one full-graph optimization step."""

    grad_fn = nnx.value_and_grad(
        loss_and_logits,
        has_aux=True,
    )

    (loss, logits), gradients = grad_fn(
        model,
        graph,
        labels,
        mask,
    )

    optimizer.update(model, gradients)

    return StepMetrics(
        loss=loss,
        accuracy=masked_accuracy(logits, labels, mask),
    )


@nnx.jit
def eval_step(
    model: TwoLayerGCN,
    graph: jraph.GraphsTuple,
    labels: jax.Array,
    mask: jax.Array,
) -> StepMetrics:
    """Evaluate the model without updating its parameters."""

    loss, logits = loss_and_logits(
        model,
        graph,
        labels,
        mask,
    )

    return StepMetrics(
        loss=loss,
        accuracy=masked_accuracy(logits, labels, mask),
    )