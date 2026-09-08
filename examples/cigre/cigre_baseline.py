"""Fit topology-blind baselines for the CIGRE MV multiday dataset.

The baselines deliberately ignore graph structure. They use only system-wide
aggregate quantities:

* total active load P
* total reactive load Q
* total active generation P

and predict three graph-level physical targets:

* minimum MV-bus voltage (pu)
* maximum line loading (%)
* maximum transformer loading (%)

The dataset is split by complete days:
    days  0-39 -> train
    days 40-49 -> validation
    days 50-59 -> test

All input and target standardization statistics are fitted on the training split
only. Two models are evaluated:

1. ordinary linear regression
2. a small JAX/Optax MLP

Usage:
    uv run python cigre_baseline.py cigre_mv_multiday.npz
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from pandapower.networks import create_cigre_network_mv

FEATURE_NAMES = (
    "total_load_p_mw",
    "total_load_q_mvar",
    "total_gen_p_mw",
)

TARGET_NAMES = (
    "min_voltage_pu",
    "max_line_loading_percent",
    "max_trafo_loading_percent",
)

TARGET_LABELS = (
    "Minimum MV voltage (pu)",
    "Maximum line loading (%)",
    "Maximum transformer loading (%)",
)


@dataclass(frozen=True)
class Standardizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> Standardizer:
        mean = np.mean(x, axis=0)
        scale = np.std(x, axis=0)
        scale = np.where(scale > 0.0, scale, 1.0)
        return cls(mean=mean, scale=scale)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.scale

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return x * self.scale + self.mean


@dataclass(frozen=True)
class Metrics:
    mae: float
    rmse: float
    r2: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset",
        nargs="?",
        type=Path,
        default=Path("cigre_mv_multiday.npz"),
        help="Dataset produced by cigre_multiday.py",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=3000,
        help="Maximum MLP optimization steps",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=32,
        help="Width of each MLP hidden layer",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Adam learning rate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="MLP initialization seed",
    )
    return parser.parse_args()


def reconstruct_aggregate_features(
    load_scaling: np.ndarray,
    sgen_scaling: np.ndarray,
) -> np.ndarray:
    """Reconstruct topology-blind aggregate inputs from benchmark base values."""
    net = create_cigre_network_mv(with_der="pv_wind")

    base_load_p = net.load["p_mw"].to_numpy(dtype=np.float64)
    base_load_q = net.load["q_mvar"].to_numpy(dtype=np.float64)
    base_gen_p = net.sgen["p_mw"].to_numpy(dtype=np.float64)

    if load_scaling.shape[1] != len(base_load_p):
        raise ValueError("Dataset load layout does not match CIGRE benchmark")
    if sgen_scaling.shape[1] != len(base_gen_p):
        raise ValueError("Dataset generator layout does not match CIGRE benchmark")

    total_load_p = load_scaling @ base_load_p
    total_load_q = load_scaling @ base_load_q
    total_gen_p = sgen_scaling @ base_gen_p

    return np.column_stack(
        [
            total_load_p,
            total_load_q,
            total_gen_p,
        ]
    )


def load_dataset(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path) as data:
        day = data["day"].astype(np.int64)
        converged = data["converged"].astype(bool)
        load_scaling = data["load_scaling"].astype(np.float64)
        sgen_scaling = data["sgen_scaling"].astype(np.float64)
        y = np.column_stack([data[name].astype(np.float64) for name in TARGET_NAMES])

    valid = converged & np.all(np.isfinite(y), axis=1)
    x = reconstruct_aggregate_features(
        load_scaling[valid],
        sgen_scaling[valid],
    )

    return day[valid], x, y[valid]


def split_by_day(
    day: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
]:
    train = day < 40
    validation = (day >= 40) & (day < 50)
    test = day >= 50

    if not (np.any(train) and np.any(validation) and np.any(test)):
        raise ValueError(
            "Expected at least 60 days so the default 40/10/10 split is non-empty"
        )

    return (
        (x[train], y[train]),
        (x[validation], y[validation]),
        (x[test], y[test]),
    )


def fit_linear_regression(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fit y = [x, 1] @ beta with ordinary least squares."""
    design = np.column_stack([x, np.ones(len(x), dtype=x.dtype)])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    return beta


def predict_linear(beta: np.ndarray, x: np.ndarray) -> np.ndarray:
    design = np.column_stack([x, np.ones(len(x), dtype=x.dtype)])
    return design @ beta


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Metrics:
    error = y_pred - y_true
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error**2)))

    denominator = float(np.sum((y_true - np.mean(y_true)) ** 2))
    numerator = float(np.sum(error**2))
    r2 = float("nan") if denominator == 0.0 else 1.0 - numerator / denominator

    return Metrics(mae=mae, rmse=rmse, r2=r2)


def init_mlp(
    key: jax.Array,
    *,
    input_size: int,
    hidden_size: int,
    output_size: int,
) -> dict[str, jax.Array]:
    k1, k2, k3 = jax.random.split(key, 3)

    return {
        "w1": jax.random.normal(k1, (input_size, hidden_size))
        * np.sqrt(2.0 / input_size),
        "b1": jnp.zeros((hidden_size,)),
        "w2": jax.random.normal(k2, (hidden_size, hidden_size))
        * np.sqrt(2.0 / hidden_size),
        "b2": jnp.zeros((hidden_size,)),
        "w3": jax.random.normal(k3, (hidden_size, output_size))
        * np.sqrt(1.0 / hidden_size),
        "b3": jnp.zeros((output_size,)),
    }


def mlp_predict(
    params: dict[str, jax.Array],
    x: jax.Array,
) -> jax.Array:
    x = jax.nn.relu(x @ params["w1"] + params["b1"])
    x = jax.nn.relu(x @ params["w2"] + params["b2"])
    return x @ params["w3"] + params["b3"]


def train_mlp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    *,
    hidden_size: int,
    learning_rate: float,
    max_steps: int,
    seed: int,
) -> dict[str, jax.Array]:
    """Train a small full-batch MLP with validation-based early stopping."""
    x_train_jax = jnp.asarray(x_train, dtype=jnp.float32)
    y_train_jax = jnp.asarray(y_train, dtype=jnp.float32)
    x_validation_jax = jnp.asarray(x_validation, dtype=jnp.float32)
    y_validation_jax = jnp.asarray(y_validation, dtype=jnp.float32)

    params = init_mlp(
        jax.random.key(seed),
        input_size=x_train.shape[1],
        hidden_size=hidden_size,
        output_size=y_train.shape[1],
    )

    optimizer = optax.adam(learning_rate)
    optimizer_state = optimizer.init(params)

    @jax.jit
    def train_step(
        current_params: dict[str, jax.Array],
        current_optimizer_state: optax.OptState,
    ) -> tuple[dict[str, jax.Array], optax.OptState, jax.Array]:
        def loss_fn(candidate_params: dict[str, jax.Array]) -> jax.Array:
            prediction = mlp_predict(candidate_params, x_train_jax)
            return jnp.mean((prediction - y_train_jax) ** 2)

        loss, gradients = jax.value_and_grad(loss_fn)(current_params)
        updates, new_optimizer_state = optimizer.update(
            gradients,
            current_optimizer_state,
            current_params,
        )
        new_params = optax.apply_updates(current_params, updates)
        return new_params, new_optimizer_state, loss

    @jax.jit
    def validation_loss(current_params: dict[str, jax.Array]) -> jax.Array:
        prediction = mlp_predict(current_params, x_validation_jax)
        return jnp.mean((prediction - y_validation_jax) ** 2)

    eval_every = 50
    patience_checks = 20
    checks_without_improvement = 0
    best_validation_loss = float("inf")
    best_params = params
    best_step = 0

    for step in range(1, max_steps + 1):
        params, optimizer_state, train_loss = train_step(
            params,
            optimizer_state,
        )

        if step % eval_every != 0 and step != max_steps:
            continue

        val_loss = float(validation_loss(params))

        if val_loss < best_validation_loss - 1e-7:
            best_validation_loss = val_loss
            best_params = jax.tree.map(lambda value: value.copy(), params)
            best_step = step
            checks_without_improvement = 0
        else:
            checks_without_improvement += 1

        if step == eval_every or step % 500 == 0 or step == max_steps:
            print(
                f"MLP step {step:4d}: "
                f"train MSE={float(train_loss):.6f}, "
                f"validation MSE={val_loss:.6f}"
            )

        if checks_without_improvement >= patience_checks:
            print(f"MLP early stopping at step {step}")
            break

    print(f"Best validation MSE={best_validation_loss:.6f} at step {best_step}")
    return best_params


def print_metrics_table(
    *,
    title: str,
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
) -> None:
    print(f"\n{title}")
    print(f"{'target':33s}{'model':10s}{'MAE':>12s}{'RMSE':>12s}{'R^2':>12s}")
    print("-" * 79)

    for index, label in enumerate(TARGET_LABELS):
        for model_name, y_pred in predictions.items():
            metrics = regression_metrics(
                y_true[:, index],
                y_pred[:, index],
            )
            print(
                f"{label:33s}"
                f"{model_name:10s}"
                f"{metrics.mae:12.5f}"
                f"{metrics.rmse:12.5f}"
                f"{metrics.r2:12.5f}"
            )


def print_stressed_metrics(
    *,
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
) -> None:
    stress_cases = (
        ("Low voltage < 0.95 pu", 0, y_true[:, 0] < 0.95),
        ("Line overload > 100%", 1, y_true[:, 1] > 100.0),
        ("Transformer overload > 100%", 2, y_true[:, 2] > 100.0),
    )

    print("\nTest-set stressed cases")
    for label, target_index, mask in stress_cases:
        count = int(np.sum(mask))
        print(f"\n{label}: {count} cases")
        if count == 0:
            continue

        for model_name, y_pred in predictions.items():
            metrics = regression_metrics(
                y_true[mask, target_index],
                y_pred[mask, target_index],
            )
            print(
                f"  {model_name:8s} "
                f"MAE={metrics.mae:.5f}  "
                f"RMSE={metrics.rmse:.5f}  "
                f"R^2={metrics.r2:.5f}"
            )


def main() -> None:
    args = parse_args()

    day, x, y = load_dataset(args.dataset)
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = split_by_day(
        day,
        x,
        y,
    )

    print(f"Train:      {len(x_train)} scenarios")
    print(f"Validation: {len(x_val)} scenarios")
    print(f"Test:       {len(x_test)} scenarios")

    print("\nTopology-blind inputs")
    for name in FEATURE_NAMES:
        print(f"  {name}")

    print("\nTargets")
    for label in TARGET_LABELS:
        print(f"  {label}")

    x_scaler = Standardizer.fit(x_train)
    y_scaler = Standardizer.fit(y_train)

    x_train_z = x_scaler.transform(x_train)
    x_val_z = x_scaler.transform(x_val)
    x_test_z = x_scaler.transform(x_test)

    y_train_z = y_scaler.transform(y_train)
    y_val_z = y_scaler.transform(y_val)

    # Linear baseline.
    beta = fit_linear_regression(x_train_z, y_train_z)
    linear_test_z = predict_linear(beta, x_test_z)
    linear_test = y_scaler.inverse_transform(linear_test_z)

    # Nonlinear topology-blind baseline.
    mlp_params = train_mlp(
        x_train_z,
        y_train_z,
        x_val_z,
        y_val_z,
        hidden_size=args.hidden_size,
        learning_rate=args.learning_rate,
        max_steps=args.steps,
        seed=args.seed,
    )
    mlp_test_z = np.asarray(
        mlp_predict(
            mlp_params,
            jnp.asarray(x_test_z, dtype=jnp.float32),
        )
    )
    mlp_test = y_scaler.inverse_transform(mlp_test_z)

    predictions = {
        "Linear": linear_test,
        "MLP": mlp_test,
    }

    print_metrics_table(
        title="Test-set performance in physical units",
        y_true=y_test,
        predictions=predictions,
    )

    print_stressed_metrics(
        y_true=y_test,
        predictions=predictions,
    )


if __name__ == "__main__":
    main()
