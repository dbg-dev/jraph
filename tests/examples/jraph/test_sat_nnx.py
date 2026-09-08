"""Tests for the NNX planted-2-SAT example."""

from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx

from examples.jraph import sat
from examples.jraph._train import eval_step, train_step


def _problem(
    *,
    seed: int = 42,
    literals: int = 6,
) -> sat.Problem:
    return sat.get_2sat_problem(
        literals,
        literals,
        rng=np.random.default_rng(seed),
    )


def test_model_output_shape() -> None:
    problem = _problem()
    model = sat.build_model(seed=42)

    logits = model(problem.graph)

    assert logits.shape == (
        *problem.labels.shape,
        sat.NUM_CLASSES,
    )


def test_model_uses_independent_interaction_blocks() -> None:
    model = sat.build_model(
        seed=42,
        num_message_passing_steps=3,
    )

    assert isinstance(model, sat.SATModel)
    assert len(model.blocks) == 3
    assert model.blocks[0] is not model.blocks[1]
    assert model.blocks[0].edge_mlp is not model.blocks[0].node_mlp


def test_first_and_later_blocks_accept_expected_feature_sizes() -> None:
    problem = _problem()
    model = sat.build_model(
        seed=42,
        num_message_passing_steps=2,
    )

    nodes = cast(jax.Array, problem.graph.nodes)
    edges = cast(jax.Array, problem.graph.edges)
    embedded_graph = problem.graph._replace(
        nodes=model.node_embedder(nodes),
        edges=model.edge_embedder(edges),
    )

    first_output = model.blocks[0](embedded_graph)
    second_output = model.blocks[1](first_output)

    assert cast(jax.Array, first_output.nodes).shape[-1] == (sat.MESSAGE_FEATURES)
    assert cast(jax.Array, first_output.edges).shape[-1] == (sat.MESSAGE_FEATURES)
    assert cast(jax.Array, second_output.nodes).shape[-1] == (sat.MESSAGE_FEATURES)
    assert cast(jax.Array, second_output.edges).shape[-1] == (sat.MESSAGE_FEATURES)


def test_eager_and_nnx_jit_match() -> None:
    problem = _problem()
    model = sat.build_model(seed=42)

    eager = model(problem.graph)
    jitted_call = nnx.jit(lambda current_model, graph: current_model(graph))
    jitted = jitted_call(model, problem.graph)

    np.testing.assert_allclose(
        jitted,
        eager,
        rtol=2e-5,
    )


def test_metrics_are_finite() -> None:
    problem = _problem()
    model = sat.build_model(seed=42)

    metrics = eval_step(
        model,
        problem.graph,
        problem.labels,
        problem.mask,
    )

    assert np.isfinite(float(metrics.loss))
    assert np.isfinite(float(metrics.accuracy))


def test_padding_graph_does_not_affect_literal_logits() -> None:
    problem = _problem()
    model = sat.build_model(seed=42)

    nodes = cast(jax.Array, problem.graph.nodes)
    edges = cast(jax.Array, problem.graph.edges)
    n_real_nodes = int(np.asarray(problem.graph.n_node)[0])
    n_real_edges = int(np.asarray(problem.graph.n_edge)[0])

    changed_graph = problem.graph._replace(
        nodes=nodes.at[n_real_nodes:].set(jnp.full_like(nodes[n_real_nodes:], 7.0)),
        edges=edges.at[n_real_edges:].set(jnp.full_like(edges[n_real_edges:], -3.0)),
    )

    original_logits = model(problem.graph)
    changed_logits = model(changed_graph)

    np.testing.assert_allclose(
        changed_logits[problem.mask],
        original_logits[problem.mask],
        rtol=2e-5,
    )


def test_repeated_training_on_one_problem_reduces_loss() -> None:
    problem = _problem(literals=6)
    model = sat.build_model(seed=42)
    optimizer = nnx.Optimizer(
        model,
        optax.adam(1e-2),
        wrt=nnx.Param,
    )

    initial = eval_step(
        model,
        problem.graph,
        problem.labels,
        problem.mask,
    )
    for _ in range(200):
        train_step(
            model,
            optimizer,
            problem.graph,
            problem.labels,
            problem.mask,
        )
    final = eval_step(
        model,
        problem.graph,
        problem.labels,
        problem.mask,
    )

    assert float(final.loss) < float(initial.loss)
    assert float(final.accuracy) == 1.0


def test_short_training_run_returns_finite_metrics() -> None:
    result = sat.train(
        num_steps=2,
        seed=42,
        learning_rate=1e-3,
        log_every=None,
        num_eval_problems=2,
        num_message_passing_steps=2,
    )

    assert np.isfinite(float(result.in_distribution.loss))
    assert np.isfinite(float(result.in_distribution.accuracy))
    assert np.isfinite(float(result.extrapolation.loss))
    assert np.isfinite(float(result.extrapolation.accuracy))


def test_logging_frequency_does_not_change_training() -> None:
    without_logging = sat.train(
        num_steps=3,
        seed=42,
        learning_rate=1e-3,
        log_every=None,
        num_eval_problems=2,
        num_message_passing_steps=2,
    )
    with_logging = sat.train(
        num_steps=3,
        seed=42,
        learning_rate=1e-3,
        log_every=1,
        num_eval_problems=2,
        num_message_passing_steps=2,
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
            lambda: sat.build_model(num_message_passing_steps=0),
            "num_message_passing_steps must be positive",
        ),
        (
            lambda: sat.get_2sat_problem(
                1,
                5,
                rng=np.random.default_rng(42),
            ),
            "min_n_literals must be at least 2",
        ),
        (
            lambda: sat.train(
                num_steps=-1,
                num_eval_problems=1,
            ),
            "num_steps must be non-negative",
        ),
        (
            lambda: sat.train(
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
