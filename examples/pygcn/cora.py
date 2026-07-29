from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import jraph


@dataclass(frozen=True, slots=True)
class CoraDataset:
    """Cora citation graph and node-classification targets."""

    graph: jraph.GraphsTuple
    labels: jax.Array
    train_mask: jax.Array
    validation_mask: jax.Array
    test_mask: jax.Array
    class_names: tuple[str, ...]


def load_cora(path: Path) -> CoraDataset:
    """Load the PyGCN Cora data as a single Jraph GraphsTuple."""

    content_path = path / "cora.content"
    citations_path = path / "cora.cites"

    if not content_path.is_file():
        raise FileNotFoundError(content_path)

    if not citations_path.is_file():
        raise FileNotFoundError(citations_path)

    content = np.loadtxt(content_path, dtype=str)
    citations = np.loadtxt(citations_path, dtype=np.int64)

    node_ids = content[:, 0].astype(np.int64)
    features = content[:, 1:-1].astype(np.float32)
    label_names = content[:, -1]

    features = _row_normalize(features)
    labels, class_names = _encode_labels(label_names)
    senders, receivers = _convert_edges(node_ids, citations)

    num_nodes = features.shape[0]
    num_edges = senders.shape[0]

    graph = jraph.GraphsTuple(
        nodes=jnp.asarray(features),
        edges=None,
        senders=jnp.asarray(senders),
        receivers=jnp.asarray(receivers),
        globals=None,
        n_node=jnp.asarray([num_nodes], dtype=jnp.int32),
        n_edge=jnp.asarray([num_edges], dtype=jnp.int32),
    )

    return CoraDataset(
        graph=graph,
        labels=jnp.asarray(labels),
        train_mask=_range_mask(num_nodes, 0, 140),
        validation_mask=_range_mask(num_nodes, 200, 500),
        test_mask=_range_mask(num_nodes, 500, 1500),
        class_names=class_names,
    )


def _row_normalize(features: np.ndarray) -> np.ndarray:
    row_sums = features.sum(axis=1, keepdims=True)

    return np.divide(
        features,
        row_sums,
        out=np.zeros_like(features),
        where=row_sums != 0,
    )


def _encode_labels(
    labels: np.ndarray,
) -> tuple[np.ndarray, tuple[str, ...]]:
    class_names = tuple(sorted(np.unique(labels).tolist()))
    class_indices = {
        class_name: index
        for index, class_name in enumerate(class_names)
    }

    encoded = np.fromiter(
        (class_indices[label] for label in labels),
        dtype=np.int32,
        count=len(labels),
    )

    return encoded, class_names


def _convert_edges(
    node_ids: np.ndarray,
    citations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    id_to_index = {
        int(node_id): index
        for index, node_id in enumerate(node_ids)
    }

    unknown_ids = set(citations.ravel()) - set(id_to_index)

    if unknown_ids:
        sample = sorted(unknown_ids)[:5]
        raise ValueError(f"Citations reference unknown node IDs: {sample}")

    edges = np.asarray(
        [
            (id_to_index[int(sender)], id_to_index[int(receiver)])
            for sender, receiver in citations
        ],
        dtype=np.int32,
    )

    # PyGCN treats Cora as an undirected graph. Represent each undirected
    # connection as two directed Jraph edges and remove duplicates.
    edges = np.unique(
        np.concatenate([edges, edges[:, ::-1]], axis=0),
        axis=0,
    )

    return edges[:, 0], edges[:, 1]


def _range_mask(
    size: int,
    start: int,
    stop: int,
) -> jax.Array:
    mask = np.zeros(size, dtype=np.bool_)
    mask[start:stop] = True
    return jnp.asarray(mask)