"""Build a Jraph graph for one CIGRE MV operating state.

This is intentionally the *raw physical* graph representation. It does not
perform ML standardization; that should be fitted later using the training
split only.

Node features
-------------
0  p_net_pu         net active injection, generation positive
1  q_net_pu         net reactive injection, generation positive
2  is_slack         1 for the external-grid bus
3  vm_setpoint_pu   external-grid voltage setpoint, 0 elsewhere

Edge features
-------------
0  r_pu             branch series resistance on net.sn_mva base
1  x_pu             branch series reactance on net.sn_mva base
2  rating_pu        thermal/rated capacity on net.sn_mva base
3  is_transformer   1 for transformer, 0 for line

Each active physical branch is represented by two directed Jraph edges so
messages can propagate in both directions.

Global features
---------------
[0.0]

Targets
-------
0  minimum MV voltage (pu)
1  maximum line loading (%)
2  maximum transformer loading (%)

Usage:
    uv run python cigre_graph.py cigre_mv_multiday.npz
    uv run python cigre_graph.py cigre_mv_multiday.npz --index 123
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from pandapower.networks import create_cigre_network_mv

import jraph

NODE_FEATURE_NAMES = (
    "p_net_pu",
    "q_net_pu",
    "is_slack",
    "vm_setpoint_pu",
)

EDGE_FEATURE_NAMES = (
    "r_pu",
    "x_pu",
    "rating_pu",
    "is_transformer",
)

TARGET_NAMES = (
    "min_voltage_pu",
    "max_line_loading_percent",
    "max_trafo_loading_percent",
)


@dataclass(frozen=True, slots=True)
class GraphSample:
    graph: jraph.GraphsTuple
    target: jax.Array


@dataclass(frozen=True, slots=True)
class PhysicalBranch:
    sender: int
    receiver: int
    features: tuple[float, float, float, float]
    name: str
    kind: str


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


def _element_closed(net, *, element_type: str, element_index: int) -> bool:
    """True when no open switch disconnects this line/transformer."""
    switches = net.switch[
        (net.switch["et"] == element_type) & (net.switch["element"] == element_index)
    ]
    return bool(switches.empty or switches["closed"].all())


def _line_branch(net, index: int) -> PhysicalBranch:
    line = net.line.loc[index]

    sender = int(line["from_bus"])
    receiver = int(line["to_bus"])
    parallel = float(line["parallel"])
    vn_kv = float(net.bus.at[sender, "vn_kv"])

    # Balanced power-flow per-unit impedance:
    #   Z_base = V_base^2 / S_base
    #   R_pu = R_ohm / Z_base
    z_base_ohm = vn_kv**2 / float(net.sn_mva)
    r_pu = (
        float(line["r_ohm_per_km"]) * float(line["length_km"]) / parallel / z_base_ohm
    )
    x_pu = (
        float(line["x_ohm_per_km"]) * float(line["length_km"]) / parallel / z_base_ohm
    )

    # Equivalent current rating expressed on the common apparent-power base.
    # This is the line current limit at 1.0 pu bus voltage.
    rating_mva = (
        np.sqrt(3.0) * vn_kv * float(line["max_i_ka"]) * float(line["df"]) * parallel
    )
    rating_pu = rating_mva / float(net.sn_mva)

    return PhysicalBranch(
        sender=sender,
        receiver=receiver,
        features=(r_pu, x_pu, rating_pu, 0.0),
        name=str(line["name"]),
        kind="line",
    )


def _transformer_branch(net, index: int) -> PhysicalBranch:
    trafo = net.trafo.loc[index]

    sender = int(trafo["hv_bus"])
    receiver = int(trafo["lv_bus"])
    parallel = float(trafo["parallel"])

    # The CIGRE benchmark uses fixed transformers. A non-neutral tap would
    # require using pandapower's tap-adjusted transformer voltage here.
    tap_pos = trafo.get("tap_pos", np.nan)
    tap_neutral = trafo.get("tap_neutral", np.nan)
    if (
        np.isfinite(tap_pos)
        and np.isfinite(tap_neutral)
        and not np.isclose(float(tap_pos), float(tap_neutral))
    ):
        raise ValueError(
            f"Transformer {index} has a non-neutral tap; "
            "this v1 graph builder assumes fixed/neutral taps"
        )

    vn_lv_bus_kv = float(net.bus.at[receiver, "vn_kv"])
    vn_trafo_lv_kv = float(trafo["vn_lv_kv"])
    sn_trafo_mva = float(trafo["sn_mva"])

    # Matches pandapower's balanced transformer branch conversion:
    # tap_lv = (V_trafo_lv / V_bus_lv)^2 * S_net
    # z_sc   = vk%  / 100 / S_trafo * tap_lv
    # r_sc   = vkr% / 100 / S_trafo * tap_lv
    tap_lv = (vn_trafo_lv_kv / vn_lv_bus_kv) ** 2 * float(net.sn_mva)
    z_pu = float(trafo["vk_percent"]) / 100.0 / sn_trafo_mva * tap_lv / parallel
    r_pu = float(trafo["vkr_percent"]) / 100.0 / sn_trafo_mva * tap_lv / parallel
    x_squared = z_pu**2 - r_pu**2
    if x_squared < -1e-12:
        raise ValueError(f"Transformer {index} has impossible vk/vkr values")
    x_pu = float(np.sqrt(max(x_squared, 0.0)))

    rating_mva = sn_trafo_mva * float(trafo["df"]) * parallel
    rating_pu = rating_mva / float(net.sn_mva)

    return PhysicalBranch(
        sender=sender,
        receiver=receiver,
        features=(r_pu, x_pu, rating_pu, 1.0),
        name=str(trafo["name"]),
        kind="transformer",
    )


def build_physical_branches(net) -> tuple[PhysicalBranch, ...]:
    branches: list[PhysicalBranch] = []

    for index, line in net.line.iterrows():
        if not bool(line["in_service"]):
            continue
        if not _element_closed(
            net,
            element_type="l",
            element_index=int(index),
        ):
            continue
        branches.append(_line_branch(net, int(index)))

    for index, trafo in net.trafo.iterrows():
        if not bool(trafo["in_service"]):
            continue
        if not _element_closed(
            net,
            element_type="t",
            element_index=int(index),
        ):
            continue
        branches.append(_transformer_branch(net, int(index)))

    return tuple(branches)


def build_static_edges(
    net,
) -> tuple[jax.Array, jax.Array, jax.Array, tuple[PhysicalBranch, ...]]:
    """Build bidirectional message-passing edges from active equipment."""
    branches = build_physical_branches(net)

    senders: list[int] = []
    receivers: list[int] = []
    edge_features: list[tuple[float, float, float, float]] = []

    for branch in branches:
        # forward
        senders.append(branch.sender)
        receivers.append(branch.receiver)
        edge_features.append(branch.features)

        # reverse: same physical branch parameters
        senders.append(branch.receiver)
        receivers.append(branch.sender)
        edge_features.append(branch.features)

    return (
        jnp.asarray(senders, dtype=jnp.int32),
        jnp.asarray(receivers, dtype=jnp.int32),
        jnp.asarray(edge_features, dtype=jnp.float32),
        branches,
    )


def build_node_features(
    net,
    p_net_mw: np.ndarray,
    q_net_mvar: np.ndarray,
) -> jax.Array:
    n_bus = len(net.bus)
    if p_net_mw.shape != (n_bus,) or q_net_mvar.shape != (n_bus,):
        raise ValueError(
            f"Expected one P/Q value for each of {n_bus} buses; "
            f"got {p_net_mw.shape=} and {q_net_mvar.shape=}"
        )

    is_slack = np.zeros(n_bus, dtype=np.float64)
    vm_setpoint = np.zeros(n_bus, dtype=np.float64)

    for _, ext_grid in net.ext_grid.iterrows():
        bus = int(ext_grid["bus"])
        is_slack[bus] = 1.0
        vm_setpoint[bus] = float(ext_grid["vm_pu"])

    s_base_mva = float(net.sn_mva)

    nodes = np.column_stack(
        [
            p_net_mw / s_base_mva,
            q_net_mvar / s_base_mva,
            is_slack,
            vm_setpoint,
        ]
    )
    return jnp.asarray(nodes, dtype=jnp.float32)


def make_graph(
    net,
    *,
    p_net_mw: np.ndarray,
    q_net_mvar: np.ndarray,
) -> jraph.GraphsTuple:
    senders, receivers, edges, _ = build_static_edges(net)
    nodes = build_node_features(net, p_net_mw, q_net_mvar)

    return jraph.GraphsTuple(
        nodes=nodes,
        edges=edges,
        senders=senders,
        receivers=receivers,
        globals=jnp.zeros((1, 1), dtype=jnp.float32),
        n_node=jnp.asarray([nodes.shape[0]], dtype=jnp.int32),
        n_edge=jnp.asarray([edges.shape[0]], dtype=jnp.int32),
    )


def load_sample(path: Path, index: int) -> GraphSample:
    net = create_cigre_network_mv(with_der="pv_wind")

    with np.load(path) as data:
        required = {
            "p_net_mw",
            "q_net_mvar",
            *TARGET_NAMES,
        }
        missing = required.difference(data.files)
        if missing:
            raise KeyError(
                "Dataset is missing required arrays: " + ", ".join(sorted(missing))
            )

        n_samples = len(data["p_net_mw"])
        if not 0 <= index < n_samples:
            raise IndexError(f"Sample index {index} outside [0, {n_samples})")

        p_net_mw = np.asarray(data["p_net_mw"][index], dtype=np.float64)
        q_net_mvar = np.asarray(data["q_net_mvar"][index], dtype=np.float64)
        target = np.asarray(
            [data[name][index] for name in TARGET_NAMES],
            dtype=np.float32,
        )

    graph = make_graph(
        net,
        p_net_mw=p_net_mw,
        q_net_mvar=q_net_mvar,
    )
    return GraphSample(
        graph=graph,
        target=jnp.asarray(target),
    )


def _print_array(name: str, array: jax.Array) -> None:
    print(f"\n{name}: shape={array.shape}, dtype={array.dtype}")
    print(np.asarray(array))


def main() -> None:
    args = parse_args()
    net = create_cigre_network_mv(with_der="pv_wind")
    sample = load_sample(args.dataset, args.index)
    _, _, _, branches = build_static_edges(net)

    print(f"pandapower base power: {float(net.sn_mva):g} MVA")
    print(f"buses:                    {len(net.bus)}")
    print(f"active physical branches: {len(branches)}")
    print(f"directed Jraph edges:      {int(sample.graph.n_edge[0])}")

    print("\nPhysical branches")
    for branch in branches:
        print(
            f"  {branch.kind:11s} "
            f"{branch.sender:2d} <-> {branch.receiver:2d}  "
            f"{branch.name}"
        )

    print("\nNode columns:")
    print("  " + ", ".join(NODE_FEATURE_NAMES))
    _print_array("nodes", sample.graph.nodes)

    print("\nEdge columns:")
    print("  " + ", ".join(EDGE_FEATURE_NAMES))
    _print_array("edges", sample.graph.edges)

    _print_array("senders", sample.graph.senders)
    _print_array("receivers", sample.graph.receivers)

    print("\nTarget columns:")
    print("  " + ", ".join(TARGET_NAMES))
    _print_array("target", sample.target)


if __name__ == "__main__":
    main()
