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
"""Tests for graph.ogb_examples.data_utils."""

from pathlib import Path

import jax
import numpy as np
import pytest

from examples.ogb import data_utils
from jraph import GraphsTuple


@pytest.fixture
def expected_graph() -> GraphsTuple:
    nodes = np.broadcast_to(
        np.arange(10, dtype=np.float32)[:, None],
        (10, 10),
    )

    edge_features = np.broadcast_to(
        np.arange(20, dtype=np.float32)[:, None],
        (20, 4),
    )

    endpoints = np.arange(20)

    return GraphsTuple(
        nodes=nodes,
        edges=np.concatenate((edge_features, edge_features)),
        receivers=np.concatenate((endpoints, endpoints)),
        senders=np.concatenate((endpoints, endpoints)),
        globals={"label": np.array([1], dtype=np.int32)},
        n_node=np.array([10], dtype=np.int32),
        n_edge=np.array([40], dtype=np.int32),
    )


@pytest.fixture
def reader() -> data_utils.DataReader:
    test_data = Path(__file__).resolve().parent / "test_data"

    return data_utils.DataReader(
        data_path=test_data,
        master_csv_path=test_data / "master.csv",
        split_path=test_data / "train.csv.gz",
    )


def test_total_num_graphs(
    reader: data_utils.DataReader,
) -> None:
    assert reader.total_num_graphs == 1


def test_reader_returns_expected_graph(
    reader: data_utils.DataReader,
    expected_graph: GraphsTuple,
) -> None:
    graph = next(reader)

    jax.tree.map(
        np.testing.assert_almost_equal,
        graph,
        expected_graph,
    )


def test_reader_stops_after_last_graph(
    reader: data_utils.DataReader,
) -> None:
    next(reader)

    with pytest.raises(StopIteration):
        next(reader)


def test_reader_repeat(
    reader: data_utils.DataReader,
    expected_graph: GraphsTuple,
) -> None:
    reader.repeat()

    next(reader)
    graph = next(reader)

    # There is one graph in the test dataset, so iteration wraps around.
    jax.tree.map(
        np.testing.assert_almost_equal,
        graph,
        expected_graph,
    )