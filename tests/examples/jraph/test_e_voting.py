"""Tests for the NNX electronic-voting example."""

from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx

from examples.jraph import e_voting
from examples.jraph._train import eval_step, train_step


def _problem(
    *,
    seed: int = 42,
    voters: int = 8,
) -> e_voting.Problem:
    return e_voting.get_voting_problem(
        voters,
        voters,
        rng=np.random.default_rng(seed),
    )


def _clear_winner_problems() -> tuple[e_voting.Problem, ...]:
    return (
        e_voting.build_voting_problem(
            [0, 0, 0, 1, 2],
            max_n_voters=5,
        ),
        e_voting.build_voting_problem(
            [1, 1, 1, 0, 2],
            max_n_voters=5,
        ),
        e_voting.build_voting_problem(
            [2, 2, 2, 0, 1],
            max_n_voters=5,
        ),
    )



def test_padded_problem_contains_jax_arrays() -> None:
    problem = _problem()

    leaves = jax.tree.leaves(problem.graph)

    assert leaves
    assert all(isinstance(leaf, jax.Array) for leaf in leaves)
    assert isinstance(problem.labels, jax.Array)
    assert isinstance(problem.mask, jax.Array)

def test_model_output_shape() -> None:
    problem = _problem()
    model = e_voting.build_model(seed=42)

    logits = model(problem.graph)

    assert logits.shape == (
        *problem.labels.shape,
        e_voting.NUM_CANDIDATES,
    )


def test_model_uses_independent_deepsets_blocks() -> None:
    model = e_voting.build_model(
        seed=42,
        num_message_passing_steps=2,
    )

    assert isinstance(model, e_voting.VotingModel)
    assert len(model.blocks) == 2
    assert model.blocks[0] is not model.blocks[1]
    assert (
        model.blocks[0].node_mlp
        is not model.blocks[0].global_mlp
    )


def test_voter_permutation_does_not_change_prediction() -> None:
    problem = e_voting.build_voting_problem(
        [0, 1, 0, 2, 0, 1],
        max_n_voters=6,
    )
    model = e_voting.build_model(seed=42)

    n_voters = int(np.asarray(problem.graph.n_node)[0])
    permutation = jnp.asarray([4, 1, 5, 0, 3, 2])
    nodes = cast(jax.Array, problem.graph.nodes)
    permuted_nodes = nodes.at[:n_voters].set(
        nodes[:n_voters][permutation]
    )
    permuted_graph = problem.graph._replace(nodes=permuted_nodes)

    original_logits = model(problem.graph)[0]
    permuted_logits = model(permuted_graph)[0]

    np.testing.assert_allclose(
        permuted_logits,
        original_logits,
        rtol=2e-5,
    )


def test_padding_graph_is_ignored() -> None:
    problem = e_voting.build_voting_problem(
        [0, 0, 1, 2],
        max_n_voters=4,
    )
    model = e_voting.build_model(seed=42)

    original_logits = model(problem.graph)
    original_metrics = eval_step(
        model,
        problem.graph,
        problem.labels,
        problem.mask,
    )

    padding_node = int(np.asarray(problem.graph.n_node)[0])
    nodes = cast(jax.Array, problem.graph.nodes)
    changed_nodes = nodes.at[padding_node].set(
        jax.nn.one_hot(7, e_voting.NUM_CANDIDATES)
    )
    changed_graph = problem.graph._replace(nodes=changed_nodes)
    changed_labels = problem.labels.at[1].set(13)
    changed_metrics = eval_step(
        model,
        changed_graph,
        changed_labels,
        problem.mask,
    )

    np.testing.assert_allclose(
        model(changed_graph)[0],
        original_logits[0],
        rtol=2e-5,
    )
    np.testing.assert_allclose(
        changed_metrics.loss,
        original_metrics.loss,
    )
    np.testing.assert_allclose(
        changed_metrics.accuracy,
        original_metrics.accuracy,
    )


def test_eager_and_nnx_jit_match() -> None:
    problem = _problem()
    model = e_voting.build_model(seed=42)

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


def test_metrics_are_finite() -> None:
    problem = _problem()
    model = e_voting.build_model(seed=42)

    metrics = eval_step(
        model,
        problem.graph,
        problem.labels,
        problem.mask,
    )

    assert np.isfinite(float(metrics.loss))
    assert np.isfinite(float(metrics.accuracy))


def test_model_learns_multiple_elections() -> None:
    problems = _clear_winner_problems()
    model = e_voting.build_model(seed=42)
    optimizer = nnx.Optimizer(
        model,
        optax.adam(1e-2),
        wrt=nnx.Param,
    )

    initial = e_voting.evaluate(model, problems)
    for step in range(400):
        problem = problems[step % len(problems)]
        train_step(
            model,
            optimizer,
            problem.graph,
            problem.labels,
            problem.mask,
        )
    final = e_voting.evaluate(model, problems)

    assert float(final.loss) < float(initial.loss)
    assert float(final.accuracy) == 1.0


def test_short_training_run_returns_finite_metrics() -> None:
    result = e_voting.train(
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
    without_logging = e_voting.train(
        num_steps=3,
        seed=42,
        learning_rate=1e-3,
        log_every=None,
        num_eval_problems=2,
    )
    with_logging = e_voting.train(
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
            lambda: e_voting.build_model(
                num_message_passing_steps=0
            ),
            "num_message_passing_steps must be positive",
        ),
        (
            lambda: e_voting.build_voting_problem(
                [],
                max_n_voters=1,
            ),
            "votes must not be empty",
        ),
        (
            lambda: e_voting.build_voting_problem(
                [e_voting.NUM_CANDIDATES],
                max_n_voters=1,
            ),
            "votes must be in",
        ),
        (
            lambda: e_voting.train(
                num_steps=-1,
                num_eval_problems=1,
            ),
            "num_steps must be non-negative",
        ),
        (
            lambda: e_voting.train(
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
