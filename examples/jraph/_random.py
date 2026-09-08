"""Independent random streams for executable examples."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class RandomStreams:
    """Independent generators for training and two evaluation regimes."""

    train: np.random.Generator
    in_distribution_evaluation: np.random.Generator
    extrapolation_evaluation: np.random.Generator

    @property
    def evaluation(self) -> np.random.Generator:
        """Return the in-distribution stream for transitional compatibility."""

        return self.in_distribution_evaluation


def make_random_streams(seed: int) -> RandomStreams:
    """Create deterministic and mutually independent random streams."""

    train_seed, in_distribution_seed, extrapolation_seed = np.random.SeedSequence(
        seed
    ).spawn(3)
    return RandomStreams(
        train=np.random.default_rng(train_seed),
        in_distribution_evaluation=np.random.default_rng(in_distribution_seed),
        extrapolation_evaluation=np.random.default_rng(extrapolation_seed),
    )
