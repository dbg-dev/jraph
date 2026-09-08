from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import optax
from flax import nnx

from .cora import CoraDataset, load_cora
from .model import TwoLayerGCN
from .training import StepMetrics, eval_step, train_step

DEFAULT_DATA_PATH = Path(__file__).parent / "data" / "cora"


@dataclass(frozen=True, slots=True)
class Metrics:
    loss: float
    accuracy: float


@dataclass(frozen=True, slots=True)
class TrainingResult:
    train: Metrics
    validation: Metrics
    test: Metrics


def _to_metrics(metrics: StepMetrics) -> Metrics:
    return Metrics(
        loss=float(metrics.loss),
        accuracy=float(metrics.accuracy),
    )


def train_cora(
    dataset: CoraDataset,
    *,
    hidden_features: int = 16,
    dropout_rate: float = 0.5,
    learning_rate: float = 0.01,
    weight_decay: float = 5e-4,
    epochs: int = 200,
    seed: int = 0,
    log_every: int = 10,
) -> tuple[TwoLayerGCN, TrainingResult]:
    """Train a two-layer GCN on the full Cora graph."""

    if epochs < 1:
        raise ValueError(f"epochs must be positive, got {epochs}")

    if log_every < 1:
        raise ValueError(f"log_every must be positive, got {log_every}")

    graph = dataset.graph

    if graph.nodes is None:
        raise ValueError("Cora graph must contain node features")

    model = TwoLayerGCN(
        in_features=graph.nodes.shape[-1],
        hidden_features=hidden_features,
        out_features=len(dataset.class_names),
        dropout_rate=dropout_rate,
        rngs=nnx.Rngs(seed),
    )

    # Both views share the same parameters.
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
        optax.adamw(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        ),
        wrt=nnx.Param,
    )

    for epoch in range(1, epochs + 1):
        train_step(
            train_model,
            optimizer,
            graph,
            dataset.labels,
            dataset.train_mask,
        )

        should_report = epoch == 1 or epoch == epochs or epoch % log_every == 0

        if should_report:
            train_metrics = _to_metrics(
                eval_step(
                    eval_model,
                    graph,
                    dataset.labels,
                    dataset.train_mask,
                )
            )
            validation_metrics = _to_metrics(
                eval_step(
                    eval_model,
                    graph,
                    dataset.labels,
                    dataset.validation_mask,
                )
            )

            print(
                f"epoch={epoch:03d} "
                f"loss={train_metrics.loss:.4f} "
                f"train_accuracy={train_metrics.accuracy:.4f} "
                f"validation_accuracy="
                f"{validation_metrics.accuracy:.4f}"
            )

    train_metrics = _to_metrics(
        eval_step(
            eval_model,
            graph,
            dataset.labels,
            dataset.train_mask,
        )
    )
    validation_metrics = _to_metrics(
        eval_step(
            eval_model,
            graph,
            dataset.labels,
            dataset.validation_mask,
        )
    )
    test_metrics = _to_metrics(
        eval_step(
            eval_model,
            graph,
            dataset.labels,
            dataset.test_mask,
        )
    )

    return model, TrainingResult(
        train=train_metrics,
        validation=validation_metrics,
        test=test_metrics,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a two-layer NNX GCN on Cora.")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--hidden-features", type=int, default=16)
    parser.add_argument("--dropout-rate", type=float, default=0.5)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=10)

    args = parser.parse_args()

    dataset = load_cora(args.data_path)

    _, result = train_cora(
        dataset,
        hidden_features=args.hidden_features,
        dropout_rate=args.dropout_rate,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        seed=args.seed,
        log_every=args.log_every,
    )

    print(f"test_loss={result.test.loss:.4f} test_accuracy={result.test.accuracy:.4f}")


if __name__ == "__main__":
    main()
