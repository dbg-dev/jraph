"""Tests for examples.zacharys_karate_club."""

import jax
import numpy as np
import pytest
from flax import nnx

from examples.jraph import zacharys_karate_club
from examples.pygcn.model import TwoLayerGCN
from examples.pygcn.training import eval_step


def test_graph_structure() -> None:
    graph = zacharys_karate_club.get_zacharys_karate_club()

    np.testing.assert_array_equal(
        graph.n_node,
        np.asarray([zacharys_karate_club.NUM_CLUB_MEMBERS]),
    )
    np.testing.assert_array_equal(graph.n_edge, np.asarray([156]))
    assert graph.nodes.shape == (
        zacharys_karate_club.NUM_CLUB_MEMBERS,
        zacharys_karate_club.NUM_CLUB_MEMBERS,
    )
    assert graph.edges is None
    assert graph.globals is None
    assert graph.senders.shape == (156,)
    assert graph.receivers.shape == (156,)

    edges = set(
        zip(
            np.asarray(graph.senders).tolist(),
            np.asarray(graph.receivers).tolist(),
            strict=True,
        )
    )
    assert all((receiver, sender) in edges for sender, receiver in edges)


def test_ground_truth_and_supervision_mask() -> None:
    labels = (
        zacharys_karate_club
        .get_ground_truth_assignments_for_zacharys_karate_club()
    )
    mask = zacharys_karate_club.get_supervision_mask()

    assert labels.shape == (zacharys_karate_club.NUM_CLUB_MEMBERS,)
    assert set(np.asarray(labels).tolist()) == {0, 1}
    assert int(labels[0]) == 0
    assert int(labels[33]) == 1

    assert mask.dtype == jax.numpy.bool_
    assert int(mask.sum()) == 2
    assert bool(mask[0])
    assert bool(mask[33])


def test_uses_shared_pygcn_model() -> None:
    model = zacharys_karate_club.build_model(seed=42)

    assert isinstance(model, TwoLayerGCN)


def test_model_output_shape_and_finite_loss() -> None:
    graph = zacharys_karate_club.get_zacharys_karate_club()
    labels = (
        zacharys_karate_club
        .get_ground_truth_assignments_for_zacharys_karate_club()
    )
    mask = zacharys_karate_club.get_supervision_mask()
    model = zacharys_karate_club.build_model(seed=42)
    eval_model = nnx.view(model, deterministic=True)

    logits = zacharys_karate_club.node_logits(eval_model, graph)
    metrics = eval_step(eval_model, graph, labels, mask)

    assert logits.shape == (
        zacharys_karate_club.NUM_CLUB_MEMBERS,
        zacharys_karate_club.NUM_CLASSES,
    )
    assert np.isfinite(float(metrics.loss))
    assert np.isfinite(float(metrics.accuracy))


def test_training_reduces_supervised_loss() -> None:
    result = zacharys_karate_club.train(
        num_steps=30,
        seed=42,
        log_every=None,
    )

    assert np.isfinite(result.initial_loss)
    assert np.isfinite(result.final_loss)
    assert np.isfinite(result.initial_accuracy)
    assert np.isfinite(result.final_accuracy)
    assert result.final_loss < result.initial_loss


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"num_steps": -1}, "num_steps must be non-negative"),
        ({"learning_rate": 0.0}, "learning_rate must be positive"),
        ({"dropout_rate": 1.0}, r"dropout_rate must be in \[0, 1\)"),
        ({"log_every": 0}, "log_every must be positive or None"),
    ],
)
def test_invalid_training_arguments(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        zacharys_karate_club.train(**kwargs)
