"""Dataset and train-only standardization for the CIGRE MV graph example.

This module sits between the raw physical dataset and the neural network.

It deliberately keeps:
  * the .npz dataset in physical units
  * cigre_graph.py as the raw physical graph builder
  * ML standardization here

Continuous features standardized:
  node: p_net_pu, q_net_pu
  edge: r_pu, x_pu, nominal_capacity_pu
  target: all three regression targets

Features left untouched:
  node: is_slack, vm_setpoint_pu
  edge: is_transformer
  global: [0.0]

The default split is by complete days:
  days  0-39 -> train
  days 40-49 -> validation
  days 50-59 -> test

Usage:
    uv run python cigre_dataset.py cigre_mv_multiday.npz
    uv run python cigre_dataset.py cigre_mv_multiday.npz --index 123
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from cigre_graph import build_node_features, build_static_edges
from pandapower.networks import create_cigre_network_mv

import jraph

TARGET_NAMES = (
    "min_voltage_pu",
    "max_line_loading_percent",
    "max_trafo_loading_percent",
)

NODE_CONTINUOUS_COLUMNS = (0, 1)
EDGE_CONTINUOUS_COLUMNS = (0, 1, 2)

Split = Literal["train", "validation", "test"]


@dataclass(frozen=True, slots=True)
class Standardizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(
        cls,
        x: np.ndarray,
        *,
        axis: int | tuple[int, ...] = 0,
    ) -> Standardizer:
        mean = np.mean(x, axis=axis)
        scale = np.std(x, axis=axis)
        scale = np.where(scale > 0.0, scale, 1.0)
        return cls(
            mean=np.asarray(mean, dtype=np.float32),
            scale=np.asarray(scale, dtype=np.float32),
        )

    def transform(self, x: np.ndarray | jax.Array) -> jax.Array:
        return (
            jnp.asarray(x, dtype=jnp.float32) - jnp.asarray(self.mean)
        ) / jnp.asarray(self.scale)

    def inverse_transform(self, x: np.ndarray | jax.Array) -> jax.Array:
        return jnp.asarray(x, dtype=jnp.float32) * jnp.asarray(
            self.scale
        ) + jnp.asarray(self.mean)


@dataclass(frozen=True, slots=True)
class DatasetScalers:
    node_power: Standardizer
    edge_physical: Standardizer
    target: Standardizer


@dataclass(frozen=True, slots=True)
class GraphSample:
    graph: jraph.GraphsTuple
    target: jax.Array
    target_physical: jax.Array
    day: int
    source_index: int


class CigreGraphDataset:
    """CIGRE MV operating states exposed as standardized Jraph graphs."""

    def __init__(self, path: Path):
        self.path = path
        self.net = create_cigre_network_mv(with_der="pv_wind")

        with np.load(path) as data:
            required = {
                "day",
                "converged",
                "p_net_mw",
                "q_net_mvar",
                *TARGET_NAMES,
            }
            missing = required.difference(data.files)
            if missing:
                raise KeyError(
                    "Dataset is missing required arrays: " + ", ".join(sorted(missing))
                )

            day = np.asarray(data["day"], dtype=np.int64)
            converged = np.asarray(data["converged"], dtype=bool)
            p_net_mw = np.asarray(data["p_net_mw"], dtype=np.float32)
            q_net_mvar = np.asarray(data["q_net_mvar"], dtype=np.float32)
            targets = np.column_stack(
                [np.asarray(data[name], dtype=np.float32) for name in TARGET_NAMES]
            )

        valid = (
            converged
            & np.all(np.isfinite(p_net_mw), axis=1)
            & np.all(np.isfinite(q_net_mvar), axis=1)
            & np.all(np.isfinite(targets), axis=1)
        )

        self.source_indices = np.flatnonzero(valid)
        self.day = day[valid]
        self.p_net_mw = p_net_mw[valid]
        self.q_net_mvar = q_net_mvar[valid]
        self.targets = targets[valid]

        (
            self.senders,
            self.receivers,
            self.raw_edges,
            self.physical_branches,
        ) = build_static_edges(self.net)

        self._split_indices = {
            "train": np.flatnonzero(self.day < 40),
            "validation": np.flatnonzero((self.day >= 40) & (self.day < 50)),
            "test": np.flatnonzero(self.day >= 50),
        }

        for name, indices in self._split_indices.items():
            if len(indices) == 0:
                raise ValueError(f"{name} split is empty")

        self.scalers = self._fit_scalers()

    def __len__(self) -> int:
        return len(self.day)

    def indices(self, split: Split) -> np.ndarray:
        return self._split_indices[split].copy()

    def _fit_scalers(self) -> DatasetScalers:
        train = self._split_indices["train"]

        s_base_mva = float(self.net.sn_mva)
        train_node_power = np.stack(
            [
                self.p_net_mw[train] / s_base_mva,
                self.q_net_mvar[train] / s_base_mva,
            ],
            axis=-1,
        )
        node_power_scaler = Standardizer.fit(
            train_node_power,
            axis=(0, 1),
        )

        raw_edges = np.asarray(self.raw_edges, dtype=np.float32)
        edge_physical_scaler = Standardizer.fit(
            raw_edges[:, EDGE_CONTINUOUS_COLUMNS],
            axis=0,
        )

        target_scaler = Standardizer.fit(
            self.targets[train],
            axis=0,
        )

        return DatasetScalers(
            node_power=node_power_scaler,
            edge_physical=edge_physical_scaler,
            target=target_scaler,
        )

    def raw_graph(self, index: int) -> jraph.GraphsTuple:
        self._check_index(index)

        nodes = build_node_features(
            self.net,
            self.p_net_mw[index],
            self.q_net_mvar[index],
        )

        return jraph.GraphsTuple(
            nodes=nodes,
            edges=self.raw_edges,
            senders=self.senders,
            receivers=self.receivers,
            globals=jnp.zeros((1, 1), dtype=jnp.float32),
            n_node=jnp.asarray([nodes.shape[0]], dtype=jnp.int32),
            n_edge=jnp.asarray(
                [self.raw_edges.shape[0]],
                dtype=jnp.int32,
            ),
        )

    def graph(self, index: int) -> jraph.GraphsTuple:
        raw = self.raw_graph(index)

        nodes = jnp.asarray(raw.nodes, dtype=jnp.float32)
        scaled_power = self.scalers.node_power.transform(
            nodes[:, NODE_CONTINUOUS_COLUMNS]
        )
        nodes = nodes.at[:, NODE_CONTINUOUS_COLUMNS].set(scaled_power)

        edges = jnp.asarray(raw.edges, dtype=jnp.float32)
        scaled_physical_edges = self.scalers.edge_physical.transform(
            edges[:, EDGE_CONTINUOUS_COLUMNS]
        )
        edges = edges.at[:, EDGE_CONTINUOUS_COLUMNS].set(scaled_physical_edges)

        return raw._replace(
            nodes=nodes,
            edges=edges,
        )

    def sample(self, index: int) -> GraphSample:
        self._check_index(index)

        target_physical = jnp.asarray(
            self.targets[index],
            dtype=jnp.float32,
        )
        target = self.scalers.target.transform(target_physical)

        return GraphSample(
            graph=self.graph(index),
            target=target,
            target_physical=target_physical,
            day=int(self.day[index]),
            source_index=int(self.source_indices[index]),
        )

    def inverse_target(
        self,
        target_standardized: np.ndarray | jax.Array,
    ) -> jax.Array:
        return self.scalers.target.inverse_transform(target_standardized)

    def _check_index(self, index: int) -> None:
        if not 0 <= index < len(self):
            raise IndexError(f"index {index} outside [0, {len(self)})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset",
        nargs="?",
        type=Path,
        default=Path("cigre_mv_multiday.npz"),
    )
    parser.add_argument("--index", type=int, default=0)
    return parser.parse_args()


def _describe_scaler(
    name: str,
    scaler: Standardizer,
) -> None:
    print(f"\n{name}")
    print("  mean :", np.asarray(scaler.mean))
    print("  scale:", np.asarray(scaler.scale))


def _training_standardization_check(
    dataset: CigreGraphDataset,
) -> None:
    train = dataset.indices("train")

    node_power = []
    for index in train:
        graph = dataset.graph(int(index))
        node_power.append(np.asarray(graph.nodes)[:, NODE_CONTINUOUS_COLUMNS])
    node_power_array = np.concatenate(node_power, axis=0)

    edge_array = np.asarray(dataset.graph(int(train[0])).edges)
    target_array = np.asarray([dataset.sample(int(index)).target for index in train])

    print("\nStandardization checks")
    print(
        "  train node P/Q mean:",
        np.mean(node_power_array, axis=0),
    )
    print(
        "  train node P/Q std: ",
        np.std(node_power_array, axis=0),
    )
    print(
        "  edge r/x/cap mean:  ",
        np.mean(
            edge_array[:, EDGE_CONTINUOUS_COLUMNS],
            axis=0,
        ),
    )
    print(
        "  edge r/x/cap std:   ",
        np.std(
            edge_array[:, EDGE_CONTINUOUS_COLUMNS],
            axis=0,
        ),
    )
    print(
        "  train target mean:  ",
        np.mean(target_array, axis=0),
    )
    print(
        "  train target std:   ",
        np.std(target_array, axis=0),
    )

    node_binary = np.unique(np.asarray(dataset.graph(int(train[0])).nodes)[:, 2])
    edge_binary = np.unique(np.asarray(dataset.graph(int(train[0])).edges)[:, 3])
    print("  is_slack values:    ", node_binary)
    print("  is_transformer vals:", edge_binary)


def main() -> None:
    args = parse_args()

    dataset = CigreGraphDataset(args.dataset)

    print(f"Valid scenarios: {len(dataset)}")
    print(
        "Split sizes:",
        f"train={len(dataset.indices('train'))},",
        f"validation={len(dataset.indices('validation'))},",
        f"test={len(dataset.indices('test'))}",
    )

    _describe_scaler(
        "Node P/Q scaler",
        dataset.scalers.node_power,
    )
    _describe_scaler(
        "Edge r/x/nominal-capacity scaler",
        dataset.scalers.edge_physical,
    )
    _describe_scaler(
        "Target scaler",
        dataset.scalers.target,
    )

    _training_standardization_check(dataset)

    sample = dataset.sample(args.index)
    raw = dataset.raw_graph(args.index)

    print(f"\nSample {args.index}")
    print(f"  source index={sample.source_index}, day={sample.day}")
    print("  raw node[0]:   ", np.asarray(raw.nodes[0]))
    print("  scaled node[0]:", np.asarray(sample.graph.nodes[0]))
    print("  raw edge[0]:   ", np.asarray(raw.edges[0]))
    print("  scaled edge[0]:", np.asarray(sample.graph.edges[0]))
    print("  physical target:", np.asarray(sample.target_physical))
    print("  scaled target:  ", np.asarray(sample.target))
    print(
        "  inverse target: ",
        np.asarray(dataset.inverse_target(sample.target)),
    )


if __name__ == "__main__":
    main()
