# Copyright 2020 DeepMind Technologies Limited.


# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for jraph.ogb_examples.train_pmap."""

import pytest

pytest.skip(
    "Legacy OGB pmap example requires an obsolete JAX/Flax stack, including "
    "the removed jax.device_put_replicated API. Re-enable when the example "
    "is migrated to current JAX and Flax NNX.",
    allow_module_level=True,
)

from pathlib import Path
from examples.ogb import train_pmap


def test_train_and_eval_overfit(tmp_path: Path) -> None:
    test_dir = Path(__file__).parent
    test_data = test_dir / "test_data"

    master_csv_path = test_data / "master.csv"
    split_path = test_data / "train.csv.gz"

    train_pmap.train(
        test_data,
        master_csv_path,
        split_path,
        1,
        101,
        str(tmp_path),
    )

    _, accuracy = train_pmap.evaluate(
        test_data,
        master_csv_path,
        split_path,
        str(tmp_path),
    )

    assert float(accuracy) == 1.0