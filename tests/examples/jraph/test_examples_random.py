"""Tests for examples._random."""

import numpy as np

from examples.jraph import _random


def test_random_streams_are_reproducible() -> None:
    first = _random.make_random_streams(42)
    second = _random.make_random_streams(42)

    for first_stream, second_stream in (
        (first.train, second.train),
        (
            first.in_distribution_evaluation,
            second.in_distribution_evaluation,
        ),
        (
            first.extrapolation_evaluation,
            second.extrapolation_evaluation,
        ),
    ):
        np.testing.assert_array_equal(
            first_stream.integers(0, 100, size=20),
            second_stream.integers(0, 100, size=20),
        )


def test_random_streams_are_mutually_independent() -> None:
    streams = _random.make_random_streams(42)

    values = (
        streams.train.integers(0, 2**31, size=20),
        streams.in_distribution_evaluation.integers(
            0,
            2**31,
            size=20,
        ),
        streams.extrapolation_evaluation.integers(
            0,
            2**31,
            size=20,
        ),
    )

    assert not np.array_equal(values[0], values[1])
    assert not np.array_equal(values[0], values[2])
    assert not np.array_equal(values[1], values[2])


def test_evaluation_alias_is_transitionally_supported() -> None:
    streams = _random.make_random_streams(42)

    assert streams.evaluation is streams.in_distribution_evaluation
