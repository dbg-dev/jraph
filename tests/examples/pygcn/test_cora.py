from pathlib import Path

import numpy as np

from examples.pygcn.cora import load_cora

DATA_PATH = Path(__file__).parents[3] / "examples" / "pygcn" / "data" / "cora"


def test_load_cora() -> None:
    dataset = load_cora(DATA_PATH)
    graph = dataset.graph

    assert graph.nodes.shape == (2708, 1433)
    assert graph.senders.shape == (10556,)
    assert graph.receivers.shape == (10556,)

    np.testing.assert_array_equal(graph.n_node, [2708])
    np.testing.assert_array_equal(graph.n_edge, [10556])

    assert dataset.labels.shape == (2708,)
    assert len(dataset.class_names) == 7

    assert int(dataset.train_mask.sum()) == 140
    assert int(dataset.validation_mask.sum()) == 300
    assert int(dataset.test_mask.sum()) == 1000


def test_features_are_row_normalized() -> None:
    dataset = load_cora(DATA_PATH)

    row_sums = np.asarray(dataset.graph.nodes).sum(axis=1)

    np.testing.assert_allclose(
        row_sums,
        np.ones(2708),
        rtol=1e-6,
        atol=1e-6,
    )


def test_splits_do_not_overlap() -> None:
    dataset = load_cora(DATA_PATH)

    assert not np.any(dataset.train_mask & dataset.validation_mask)
    assert not np.any(dataset.train_mask & dataset.test_mask)
    assert not np.any(dataset.validation_mask & dataset.test_mask)


def test_edges_are_symmetric() -> None:
    dataset = load_cora(DATA_PATH)

    edges = set(
        zip(
            np.asarray(dataset.graph.senders).tolist(),
            np.asarray(dataset.graph.receivers).tolist(),
            strict=True,
        )
    )

    assert all((receiver, sender) in edges for sender, receiver in edges)
