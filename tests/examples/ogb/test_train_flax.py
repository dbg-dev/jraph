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
"""Tests for examples.ogb.train_flax."""

from pathlib import Path

from examples.ogb import train_flax


def test_train_and_eval_overfit(tmp_path: Path) -> None:
    data_path = Path(__file__).parent / "test_data"
    master_csv_path = data_path / "master.csv"
    split_path = data_path / "train.csv.gz"

    train_flax.train(
        data_path,
        master_csv_path,
        split_path,
        1,
        101,
        tmp_path,
    )

    _, accuracy = train_flax.evaluate(
        data_path,
        master_csv_path,
        split_path,
        tmp_path,
    )

    assert float(accuracy) == 1.0
