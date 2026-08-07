"""Tests for the NNX Higgs-detection example."""

from typing import cast

from flax import nnx
import jax
import jax.numpy as jnp
import jraph
import numpy as np
import optax
import pytest

from examples.jraph import higgs_detection
from examples.jraph._train import eval_step, train_step


def _background_photons(
    n_photons: int,
    *,
    seed: int = 42,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.stack(
        [
            higgs_detection.get_random_background_photon(rng)
            for _ in range(n_photons)
        ]
    )


def _problem(
    *,
    seed: int = 42,
    photons: int = 6,
    max_photons: int | None = None,
) -> higgs_detection.Problem:
    if max_photons is None:
        max_photons = photons
    return higgs_detection.get_higgs_problem(
        photons,
        max_photons,
        rng=np.random.default_rng(seed),
    )


def _assert_trees_equal(actual: object, expected: object) -> None:
    jax.tree.map(
        np.testing.assert_array_equal,
        actual,
        expected,
    )


def test_problem_generation_is_reproducible() -> None:
    first = _problem(seed=17)
    second = _problem(seed=17)

    _assert_trees_equal(first, second)


def test_builder_preserves_explicit_label() -> None:
    problem = higgs_detection.build_higgs_problem(
        _background_photons(4),
        higgs_detection.BACKGROUND_LABEL,
        max_n_photons=6,
    )

    assert int(problem.labels[0]) == (
        higgs_detection.BACKGROUND_LABEL
    )
    np.testing.assert_array_equal(
        problem.mask,
        jraph.get_graph_padding_mask(problem.graph),
    )


def test_padded_problem_contains_jax_arrays() -> None:
    problem = _problem()

    leaves = jax.tree.leaves(problem.graph)

    assert leaves
    assert all(isinstance(leaf, jax.Array) for leaf in leaves)
    assert isinstance(problem.labels, jax.Array)
    assert isinstance(problem.mask, jax.Array)


def test_problem_is_fully_connected() -> None:
    problem = _problem(photons=5)

    assert int(problem.graph.n_node[0]) == 5
    assert int(problem.graph.n_edge[0]) == 25


def test_higgs_pair_has_expected_invariant_mass() -> None:
    photon1, photon2 = higgs_detection.get_random_higgs_photons(
        np.random.default_rng(42)
    )

    mass_squared = higgs_detection.invariant_mass_squared(
        photon1 + photon2
    )

    np.testing.assert_allclose(
        mass_squared,
        higgs_detection.HIGGS_MASS_GEV**2,
        rtol=1e-10,
        atol=1e-8,
    )


def test_background_photon_is_massless() -> None:
    photon = higgs_detection.get_random_background_photon(
        np.random.default_rng(42)
    )

    mass_squared = higgs_detection.invariant_mass_squared(photon)

    np.testing.assert_allclose(
        mass_squared,
        0.0,
        atol=1e-9,
    )


def test_original_analytical_solution_recognizes_higgs_pair() -> None:
    photon1, photon2 = higgs_detection.get_random_higgs_photons(
        np.random.default_rng(42)
    )

    indicator = higgs_detection.unused_update_edge_fn_solution(
        jnp.asarray([photon1]),
        jnp.asarray([photon2]),
    )

    np.testing.assert_array_equal(
        indicator,
        jnp.asarray([[1.0]], dtype=jnp.float32),
    )


def test_model_output_shape() -> None:
    problem = _problem()
    model = higgs_detection.build_model(seed=42)

    logits = model(problem.graph)

    assert logits.shape == (
        *problem.labels.shape,
        higgs_detection.NUM_CLASSES,
    )


def test_eager_and_nnx_jit_match() -> None:
    problem = _problem()
    model = higgs_detection.build_model(seed=42)

    eager = model(problem.graph)
    jitted_call = nnx.jit(
        lambda current_model, graph: current_model(graph)
    )
    jitted = jitted_call(model, problem.graph)

    np.testing.assert_allclose(
        jitted,
        eager,
        rtol=2e-5,
    )


def test_graph_isomorphism_does_not_change_real_graph_logits() -> None:
    photons = _background_photons(6)
    problem = higgs_detection.build_higgs_problem(
        photons,
        higgs_detection.BACKGROUND_LABEL,
        max_n_photons=6,
    )
    model = higgs_detection.build_model(seed=42)

    nodes = cast(jax.Array, problem.graph.nodes)
    senders = cast(jax.Array, problem.graph.senders)
    receivers = cast(jax.Array, problem.graph.receivers)

    permutation = jnp.asarray([4, 1, 5, 0, 3, 2])
    inverse_permutation = jnp.argsort(permutation)
    n_real_edges = int(problem.graph.n_edge[0])

    permuted_nodes = nodes.at[:6].set(nodes[:6][permutation])
    permuted_senders = senders.at[:n_real_edges].set(
        inverse_permutation[senders[:n_real_edges]]
    )
    permuted_receivers = receivers.at[:n_real_edges].set(
        inverse_permutation[receivers[:n_real_edges]]
    )
    permuted_graph = problem.graph._replace(
        nodes=permuted_nodes,
        senders=permuted_senders,
        receivers=permuted_receivers,
    )

    original_logits = model(problem.graph)
    permuted_logits = model(permuted_graph)

    np.testing.assert_allclose(
        permuted_logits[0],
        original_logits[0],
        rtol=2e-5,
        atol=1e-5,
    )


def test_padding_graph_does_not_affect_real_graph_logits() -> None:
    problem = higgs_detection.build_higgs_problem(
        _background_photons(4),
        higgs_detection.BACKGROUND_LABEL,
        max_n_photons=6,
    )
    model = higgs_detection.build_model(seed=42)

    nodes = cast(jax.Array, problem.graph.nodes)
    n_real_nodes = int(problem.graph.n_node[0])
    changed_graph = problem.graph._replace(
        nodes=nodes.at[n_real_nodes:].set(
            jnp.full_like(nodes[n_real_nodes:], 1_000.0)
        )
    )

    original_logits = model(problem.graph)
    changed_logits = model(changed_graph)

    np.testing.assert_allclose(
        changed_logits[0],
        original_logits[0],
        rtol=2e-5,
    )


def test_metrics_are_finite() -> None:
    problem = _problem()
    model = higgs_detection.build_model(seed=42)

    metrics = eval_step(
        model,
        problem.graph,
        problem.labels,
        problem.mask,
    )

    assert np.isfinite(float(metrics.loss))
    assert np.isfinite(float(metrics.accuracy))


def test_train_step_returns_finite_metrics() -> None:
    problem = _problem()
    model = higgs_detection.build_model(seed=42)
    optimizer = nnx.Optimizer(
        model,
        optax.adam(1e-3),
        wrt=nnx.Param,
    )

    metrics = train_step(
        model,
        optimizer,
        problem.graph,
        problem.labels,
        problem.mask,
    )

    assert np.isfinite(float(metrics.loss))
    assert np.isfinite(float(metrics.accuracy))


def test_short_training_run_returns_finite_metrics() -> None:
    result = higgs_detection.train(
        num_steps=2,
        seed=42,
        learning_rate=1e-3,
        log_every=None,
        num_eval_problems=2,
    )

    assert np.isfinite(float(result.in_distribution.loss))
    assert np.isfinite(float(result.in_distribution.accuracy))
    assert np.isfinite(float(result.extrapolation.loss))
    assert np.isfinite(float(result.extrapolation.accuracy))


def test_logging_frequency_does_not_change_training() -> None:
    without_logging = higgs_detection.train(
        num_steps=3,
        seed=42,
        learning_rate=1e-3,
        log_every=None,
        num_eval_problems=2,
    )
    with_logging = higgs_detection.train(
        num_steps=3,
        seed=42,
        learning_rate=1e-3,
        log_every=1,
        num_eval_problems=2,
    )

    for actual, expected in (
        (
            with_logging.in_distribution,
            without_logging.in_distribution,
        ),
        (
            with_logging.extrapolation,
            without_logging.extrapolation,
        ),
    ):
        np.testing.assert_allclose(actual.loss, expected.loss)
        np.testing.assert_allclose(
            actual.accuracy,
            expected.accuracy,
        )


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (
            lambda: higgs_detection.get_higgs_problem(
                1,
                5,
                rng=np.random.default_rng(42),
            ),
            "min_n_photons must be at least 2",
        ),
        (
            lambda: higgs_detection.get_higgs_problem(
                5,
                4,
                rng=np.random.default_rng(42),
            ),
            "max_n_photons must be at least min_n_photons",
        ),
        (
            lambda: higgs_detection.build_higgs_problem(
                np.ones((1, 4)),
                higgs_detection.HIGGS_LABEL,
                max_n_photons=5,
            ),
            "at least two photons are required",
        ),
        (
            lambda: higgs_detection.train(
                num_steps=-1,
                num_eval_problems=1,
            ),
            "num_steps must be non-negative",
        ),
        (
            lambda: higgs_detection.train(
                num_steps=0,
                num_eval_problems=0,
            ),
            "num_eval_problems must be positive",
        ),
    ],
)
def test_invalid_arguments(call, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        call()
