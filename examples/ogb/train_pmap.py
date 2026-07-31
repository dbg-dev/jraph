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
r"""Example training script for training OGB molhiv with jax graph-nets.

The ogbg-molhiv dataset is a molecular property prediction dataset.
It is adopted from the MoleculeNet [1]. All the molecules are pre-processed
using RDKit [2].

Each graph represents a molecule, where nodes are atoms, and edges are chemical
bonds. Input node features are 9-dimensional, containing atomic number and
chirality, as well as other additional atom features such as formal charge and
whether the atom is in the ring or not.

The goal is to predict whether a molecule inhibits HIV virus replication or not.
Performance is measured in ROC-AUC.

This script uses a GraphNet to learn the prediction task.

[1] Zhenqin Wu, Bharath Ramsundar, Evan N Feinberg, Joseph Gomes,
Caleb Geniesse, Aneesh SPappu, Karl Leswing, and Vijay Pande.
Moleculenet: a benchmark for molecular machine learning.
Chemical Science, 9(2):513–530, 2018.

[2] Greg Landrum et al. RDKit: Open-source cheminformatics, 2006.

Example usage:

python3 train.py --data_path={DATA_PATH} --master_csv_path={MASTER_CSV_PATH} \
--save_dir={SAVE_DIR} --split_path={SPLIT_PATH}
"""

import functools
import logging
import pathlib
import pickle
from collections.abc import Iterable, Iterator

from absl import app
from absl import flags
import haiku as hk
import jax
import jax.numpy as jnp
import jraph
import numpy as np
import optax

from examples.ogb import data_utils
from examples.ogb._training import loss_and_accuracy


FLAGS = flags.FLAGS


def _define_flags() -> None:
    flags.DEFINE_string("data_path", None, "Directory of the data.")
    flags.DEFINE_string("split_path", None, "Path to the data split indices.")
    flags.DEFINE_string("master_csv_path", None, "Path to OGB master.csv.")
    flags.DEFINE_string("save_dir", None, "Directory to save parameters to.")
    flags.DEFINE_integer("batch_size", 1, "Number of graphs in batch.")
    flags.DEFINE_integer("num_training_steps", 1000, "Number of training steps.")
    flags.DEFINE_enum("mode", "train", ["train", "evaluate"], "Train or evaluate.")


@jraph.concatenated_args
def edge_update_fn(feats: jnp.ndarray) -> jnp.ndarray:
    """Edge update function for graph net."""
    net = hk.Sequential([hk.Linear(128), jax.nn.relu, hk.Linear(128)])
    return net(feats)


@jraph.concatenated_args
def node_update_fn(feats: jnp.ndarray) -> jnp.ndarray:
    """Node update function for graph net."""
    net = hk.Sequential([hk.Linear(128), jax.nn.relu, hk.Linear(128)])
    return net(feats)


@jraph.concatenated_args
def update_global_fn(feats: jnp.ndarray) -> jnp.ndarray:
    """Global update function for graph net."""
    # Molhiv is a binary classification task, so output pos neg logits.
    net = hk.Sequential([hk.Linear(128), jax.nn.relu, hk.Linear(2)])
    return net(feats)


def net_fn(graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
    """Graph net function."""
    # Add a global paramater for graph classification.
    graph = graph._replace(globals=jnp.zeros([graph.n_node.shape[0], 1]))
    embedder = jraph.GraphMapFeatures(hk.Linear(128), hk.Linear(128), hk.Linear(128))
    net = jraph.GraphNetwork(
        update_node_fn=node_update_fn,
        update_edge_fn=edge_update_fn,
        update_global_fn=update_global_fn,
    )
    return net(embedder(graph))


def device_batch(
    graphs: Iterable[jraph.GraphsTuple],
    *,
    num_devices: int | None = None,
) -> Iterator[jraph.GraphsTuple]:
    """Stack graph batches along a leading device axis."""
    batch_size = jax.local_device_count() if num_devices is None else num_devices

    if batch_size < 1:
        raise ValueError(f"num_devices must be positive, got {batch_size}")
    batch = []

    for graph in graphs:
        batch.append(graph)

        if len(batch) == batch_size:
            yield jax.tree.map(
                lambda *values: jnp.stack(values, axis=0),
                *batch,
            )
            batch = []

    # Evaluation datasets may not divide evenly across devices.
    if batch:
        yield jax.tree.map(
            lambda *values: jnp.stack(values, axis=0),
            *batch,
        )


def compute_loss(params, graph, labels, net):
    """Compute graph-classification loss and accuracy."""

    predicted_graph = net.apply(params, graph)
    return loss_and_accuracy(predicted_graph, labels)


def train(
    data_path, master_csv_path, split_path, batch_size, num_training_steps, save_dir
):
    """OGB Training Script."""
    # Initialize the dataset reader.
    reader = data_utils.DataReader(
        data_path=data_path,
        master_csv_path=master_csv_path,
        split_path=split_path,
        batch_size=batch_size,
        dynamically_batch=True,
    )
    # Repeat the dataset forever for training.
    reader.repeat()

    # Transform impure `net_fn` to pure functions with hk.transform.
    net = hk.without_apply_rng(hk.transform(net_fn))
    # Get a candidate graph and label to initialize the network.
    graph = reader.get_graph_by_idx(0)

    # Initialize the network.
    logging.info("Initializing network.")
    params = net.init(jax.random.PRNGKey(42), graph)
    # Initialize the optimizer.
    opt_init, opt_update = optax.adam(1e-4)
    opt_state = opt_init(params)

    compute_loss_fn = functools.partial(
        compute_loss,
        net=net,
    )

    @functools.partial(
        jax.pmap,
        axis_name="device",
        in_axes=(None, 0, 0, None),
        out_axes=(0, 0, None, None),
    )
    def update_fn(params, graph, labels, opt_state):
        (loss, accuracy), gradients = jax.value_and_grad(
            compute_loss_fn,
            has_aux=True,
        )(
            params,
            graph,
            labels,
        )

        gradients = jax.lax.pmean(
            gradients,
            axis_name="device",
        )

        updates, opt_state = opt_update(
            gradients,
            opt_state,
            params,
        )
        params = optax.apply_updates(params, updates)

        return loss, accuracy, opt_state, params

    graph_batches = device_batch(reader)

    for step in range(num_training_steps):
        graph_batch = next(graph_batches)

        labels = graph_batch.globals["label"]
        graph_batch = graph_batch._replace(globals={})

        loss, accuracy, opt_state, params = update_fn(
            params,
            graph_batch,
            labels,
            opt_state,
        )

        if step % 100 == 0:
            logging.info(
                "step: %s, loss: %s, acc: %s",
                step,
                loss,
                accuracy,
            )

    if save_dir is not None:
        with pathlib.Path(save_dir, "molhiv.pkl").open("wb") as fp:
            logging.info("Saving model to %s", save_dir)
            pickle.dump(params, fp)
    logging.info("Training finished")


def evaluate(data_path, master_csv_path, split_path, save_dir):
    """Evaluation Script."""
    logging.info("Evaluating OGB molviv")
    logging.info("Dataset split: %s", split_path)
    # Initialize the dataset reader.
    reader = data_utils.DataReader(
        data_path=data_path,
        master_csv_path=master_csv_path,
        split_path=split_path,
        batch_size=1,
        dynamically_batch=True,
    )
    # Transform impure `net_fn` to pure functions with hk.transform.
    net = hk.without_apply_rng(hk.transform(net_fn))
    with pathlib.Path(save_dir, "molhiv.pkl").open("rb") as fp:
        params = pickle.load(fp)

    compute_loss_fn = jax.pmap(
        functools.partial(
            compute_loss,
            net=net,
        ),
        in_axes=(None, 0, 0),
        out_axes=(0, 0),
    )

    accumulated_loss = 0.0
    accumulated_accuracy = 0.0
    num_graphs = 0

    for graph_batch in device_batch(reader):
        labels = graph_batch.globals["label"]
        graph_batch = graph_batch._replace(globals={})

        batch_loss, batch_accuracy = compute_loss_fn(
            params,
            graph_batch,
            labels,
        )

        batch_loss = np.asarray(jax.device_get(batch_loss))
        batch_accuracy = np.asarray(jax.device_get(batch_accuracy))

        accumulated_loss += float(batch_loss.sum())
        accumulated_accuracy += float(batch_accuracy.sum())

        padding_graphs = jax.vmap(jraph.get_number_of_padding_with_graphs_graphs)(
            graph_batch
        )

        num_padding_graphs = int(np.asarray(jax.device_get(padding_graphs)).sum())

        num_graphs += graph_batch.n_node.size - num_padding_graphs

        if num_graphs % 100 == 0:
            logging.info("Evaluated %s graphs", num_graphs)

    if num_graphs == 0:
        raise ValueError("Cannot evaluate an empty dataset.")

    loss = accumulated_loss / num_graphs
    accuracy = accumulated_accuracy / num_graphs

    logging.info("Completed evaluation.")
    logging.info(
        "Eval loss: %s, accuracy %s",
        loss,
        accuracy,
    )

    return loss, accuracy


def main(_):
    if FLAGS.mode == "train":
        train(
            FLAGS.data_path,
            FLAGS.master_csv_path,
            FLAGS.split_path,
            FLAGS.batch_size,
            FLAGS.num_training_steps,
            FLAGS.save_dir,
        )
    elif FLAGS.mode == "evaluate":
        evaluate(
            FLAGS.data_path, FLAGS.master_csv_path, FLAGS.split_path, FLAGS.save_dir
        )


if __name__ == "__main__":
    _define_flags()
    app.run(main)
