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
"""Tests for examples.ogb.train."""

from pathlib import Path

from examples.ogb.train import evaluate, train


def test_train_and_eval_overfit(tmp_path: Path) -> None:
    test_data = Path(__file__).parent / "test_data"
    master_csv_path = test_data / "master.csv"
    split_path = test_data / "train.csv.gz"

    train(
        test_data,
        master_csv_path,
        split_path,
        1,
        101,
        tmp_path,
    )

    _, accuracy = evaluate(
        test_data,
        master_csv_path,
        split_path,
        tmp_path,
    )

    assert float(accuracy) == 1.0
