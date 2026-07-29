from typing import cast

import jax
import jax.numpy as jnp
import optax
from flax import nnx

import jraph

from .model import TwoLayerGCN


def masked_cross_entropy(
    logits: jax.Array,
    labels: jax.Array,
    mask: jax.Array,
) -> jax.Array:
    """Mean cross-entropy over the selected nodes."""

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
    """Classification accuracy over the selected nodes."""

    predictions = jnp.argmax(logits, axis=-1)
    correct = predictions == labels

    weights = mask.astype(jnp.float32)
    total_weight = jnp.maximum(weights.sum(), 1.0)

    return jnp.sum(correct * weights) / total_weight


def loss_fn(
    model: TwoLayerGCN,
    graph: jraph.GraphsTuple,
    labels: jax.Array,
    mask: jax.Array,
) -> jax.Array:
    """Evaluate the masked node-classification loss."""

    output = model(graph)
    logits = cast(jax.Array, output.nodes)

    return masked_cross_entropy(logits, labels, mask)