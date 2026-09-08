"""Train the CIGRE MV two-block NNX GraphNetwork.

This script uses the existing modules:

    cigre_graph.py
    cigre_dataset.py
    cigre_model.py

Training setup
--------------
* complete-day 40/10/10 train/validation/test split from CigreGraphDataset
* fixed-size Jraph graph batches
* Adam optimizer
* MSE on standardized targets
* validation once per epoch
* early stopping on validation MSE
* final metrics reported in physical units

Usage:
    uv run python cigre_train.py cigre_mv_multiday.npz

Useful options:
    uv run python cigre_train.py cigre_mv_multiday.npz \
        --epochs 50 \
        --batch-size 32 \
        --learning-rate 1e-3 \
        --latent-size 32
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from cigre_dataset import CigreGraphDataset
from cigre_model import CigreGraphNetwork, count_parameters
from flax import nnx

import jraph


@dataclass(frozen=True, slots=True)
class Metrics:
    mae: float
    rmse: float
    r2: float


@dataclass(frozen=True, slots=True)
class Batch:
    graph: jraph.GraphsTuple
    target: jax.Array


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset",
        nargs="?",
        type=Path,
        default=Path("cigre_mv_multiday.npz"),
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--latent-size", type=int, default=32)
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Stop after this many epochs without validation improvement",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def cache_graphs(
    dataset: CigreGraphDataset,
) -> tuple[list[jraph.GraphsTuple], np.ndarray]:
    """Build each standardized graph once to keep Python preprocessing cheap."""
    graphs = [dataset.graph(i) for i in range(len(dataset))]
    targets = np.stack(
        [
            np.asarray(dataset.sample(i).target, dtype=np.float32)
            for i in range(len(dataset))
        ],
        axis=0,
    )
    return graphs, targets


def make_batch(
    graphs: Sequence[jraph.GraphsTuple],
    targets: np.ndarray,
    indices: np.ndarray,
) -> Batch:
    graph = jraph.batch([graphs[int(i)] for i in indices])
    target = jnp.asarray(targets[indices], dtype=jnp.float32)
    return Batch(graph=graph, target=target)


def iter_batches(
    indices: np.ndarray,
    *,
    batch_size: int,
) -> Sequence[np.ndarray]:
    """Yield only full batches to keep compiled shapes constant."""
    n_full = len(indices) // batch_size
    return [
        indices[i * batch_size : (i + 1) * batch_size]
        for i in range(n_full)
    ]


@nnx.jit
def train_step(
    model: CigreGraphNetwork,
    optimizer: nnx.Optimizer,
    graph: jraph.GraphsTuple,
    target: jax.Array,
) -> jax.Array:
    def loss_fn(model: CigreGraphNetwork) -> jax.Array:
        prediction = model(graph)
        return jnp.mean((prediction - target) ** 2)

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return loss


@nnx.jit
def eval_step(
    model: CigreGraphNetwork,
    graph: jraph.GraphsTuple,
    target: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    prediction = model(graph)
    loss = jnp.mean((prediction - target) ** 2)
    return loss, prediction


def evaluate_standardized(
    model: CigreGraphNetwork,
    graphs: Sequence[jraph.GraphsTuple],
    targets: np.ndarray,
    indices: np.ndarray,
    *,
    batch_size: int,
) -> tuple[float, np.ndarray, np.ndarray]:
    losses: list[float] = []
    predictions: list[np.ndarray] = []
    expected: list[np.ndarray] = []

    for batch_indices in iter_batches(indices, batch_size=batch_size):
        batch = make_batch(graphs, targets, batch_indices)
        loss, prediction = eval_step(
            model,
            batch.graph,
            batch.target,
        )
        losses.append(float(loss))
        predictions.append(np.asarray(prediction))
        expected.append(np.asarray(batch.target))

    if not losses:
        raise ValueError(
            f"Split has fewer than one full batch of size {batch_size}"
        )

    return (
        float(np.mean(losses)),
        np.concatenate(predictions, axis=0),
        np.concatenate(expected, axis=0),
    )


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Metrics:
    error = y_pred - y_true
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error**2)))

    denominator = float(
        np.sum((y_true - np.mean(y_true)) ** 2)
    )
    numerator = float(np.sum(error**2))
    r2 = (
        float("nan")
        if denominator == 0.0
        else 1.0 - numerator / denominator
    )

    return Metrics(mae=mae, rmse=rmse, r2=r2)


def print_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    labels = (
        "Minimum MV voltage (pu)",
        "Maximum line loading (%)",
        "Maximum transformer loading (%)",
    )

    print("\nTest-set performance in physical units")
    print(
        f"{'target':33s}"
        f"{'MAE':>12s}"
        f"{'RMSE':>12s}"
        f"{'R^2':>12s}"
    )
    print("-" * 69)

    for i, label in enumerate(labels):
        metrics = regression_metrics(y_true[:, i], y_pred[:, i])
        print(
            f"{label:33s}"
            f"{metrics.mae:12.5f}"
            f"{metrics.rmse:12.5f}"
            f"{metrics.r2:12.5f}"
        )


def print_stressed_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    cases = (
        ("Low voltage < 0.95 pu", 0, y_true[:, 0] < 0.95),
        ("Line overload > 100%", 1, y_true[:, 1] > 100.0),
        (
            "Transformer overload > 100%",
            2,
            y_true[:, 2] > 100.0,
        ),
    )

    print("\nTest-set stressed cases")

    for label, target_index, mask in cases:
        count = int(np.sum(mask))
        print(f"\n{label}: {count} cases")

        if count == 0:
            continue

        metrics = regression_metrics(
            y_true[mask, target_index],
            y_pred[mask, target_index],
        )
        print(
            f"  MAE={metrics.mae:.5f}  "
            f"RMSE={metrics.rmse:.5f}  "
            f"R^2={metrics.r2:.5f}"
        )


def copy_params(model: CigreGraphNetwork) -> nnx.State:
    """Take an immutable snapshot of trainable model parameters."""
    params = nnx.state(model, nnx.Param)
    return jax.tree.map(lambda x: x.copy(), params)


def main() -> None:
    args = parse_args()

    dataset = CigreGraphDataset(args.dataset)

    train_indices = dataset.indices("train")
    validation_indices = dataset.indices("validation")
    test_indices = dataset.indices("test")

    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")

    print("Caching standardized graphs...")
    graphs, targets = cache_graphs(dataset)

    model = CigreGraphNetwork(
        latent_size=args.latent_size,
        rngs=nnx.Rngs(args.seed),
    )

    optimizer = nnx.Optimizer(
        model,
        optax.adam(args.learning_rate),
        wrt=nnx.Param,
    )

    print(f"Train scenarios:      {len(train_indices)}")
    print(f"Validation scenarios: {len(validation_indices)}")
    print(f"Test scenarios:       {len(test_indices)}")
    print(f"Batch size:           {args.batch_size}")
    print(f"Latent size:          {args.latent_size}")
    print(f"Parameters:           {count_parameters(model):,}")
    print(f"Learning rate:        {args.learning_rate:g}")

    rng = np.random.default_rng(args.seed)

    best_validation_loss = float("inf")
    best_epoch = 0
    best_params = copy_params(model)
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        shuffled = rng.permutation(train_indices)

        train_losses: list[float] = []
        for batch_indices in iter_batches(
            shuffled,
            batch_size=args.batch_size,
        ):
            batch = make_batch(graphs, targets, batch_indices)
            loss = train_step(
                model,
                optimizer,
                batch.graph,
                batch.target,
            )
            train_losses.append(float(loss))

        train_loss = float(np.mean(train_losses))

        (
            validation_loss,
            _,
            _,
        ) = evaluate_standardized(
            model,
            graphs,
            targets,
            validation_indices,
            batch_size=args.batch_size,
        )

        improved = validation_loss < best_validation_loss - 1e-6

        if improved:
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_params = copy_params(model)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        marker = " *" if improved else ""
        print(
            f"Epoch {epoch:3d}: "
            f"train MSE={train_loss:.6f}  "
            f"validation MSE={validation_loss:.6f}"
            f"{marker}"
        )

        if epochs_without_improvement >= args.patience:
            print(
                f"Early stopping after {epoch} epochs "
                f"({args.patience} without improvement)"
            )
            break

    nnx.update(model, best_params)

    print(
        f"\nRestored best model from epoch {best_epoch} "
        f"(validation MSE={best_validation_loss:.6f})"
    )

    (
        test_loss,
        test_prediction_z,
        test_target_z,
    ) = evaluate_standardized(
        model,
        graphs,
        targets,
        test_indices,
        batch_size=args.batch_size,
    )

    test_prediction = np.asarray(
        dataset.inverse_target(test_prediction_z)
    )
    test_target = np.asarray(
        dataset.inverse_target(test_target_z)
    )

    print(f"Test standardized MSE: {test_loss:.6f}")

    print_metrics(
        test_target,
        test_prediction,
    )
    print_stressed_metrics(
        test_target,
        test_prediction,
    )


if __name__ == "__main__":
    main()
