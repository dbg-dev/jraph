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
r"""Example training script for training OGB molhiv with jax graph-nets & flax.

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

uv run python train_flax.py train \
  --data_path {DATA_PATH} \
  --master_csv_path {MASTER_CSV_PATH} \
  --split_path {SPLIT_PATH} \
  --batch_size 32 \
  --num_training_steps 1000 \
  --save_dir {SAVE_DIR}

or

uv run python train_flax.py evaluate \
  --data_path {DATA_PATH} \
  --master_csv_path {MASTER_CSV_PATH} \
  --split_path {SPLIT_PATH} \
  --save_dir {SAVE_DIR}
"""

import argparse
import logging
import pathlib
import pickle
from collections.abc import Sequence

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn

import jraph
from examples.ogb import data_utils
from examples.ogb._training import (
  loss_and_accuracy,
  run_evaluation,
  run_training,
)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()

  common_parser = argparse.ArgumentParser(add_help=False)
  common_parser.add_argument("--data_path", help="Directory of the data.")
  common_parser.add_argument("--split_path", help="Path to the data split indices.")
  common_parser.add_argument("--master_csv_path", help="Path to OGB master.csv.")
  common_parser.add_argument("--save_dir", help="Directory to save parameters to.")

  subparsers = parser.add_subparsers(dest="mode", required=True)

  train_parser = subparsers.add_parser(
      "train",
      parents=[common_parser],
      help="Train the model.",
  )
  train_parser.add_argument(
      "--batch_size",
      type=int,
      default=1,
      help="Number of graphs in batch.",
  )
  train_parser.add_argument(
      "--num_training_steps",
      type=int,
      default=1000,
      help="Number of training steps.",
  )

  subparsers.add_parser(
      "evaluate",
      parents=[common_parser],
      help="Evaluate the model.",
  )

  return parser.parse_args()



class ExplicitMLP(nn.Module):
    """A flax MLP."""

    features: Sequence[int]

    @nn.compact
    def __call__(self, inputs):
        x = inputs
        for i, lyr in enumerate([nn.Dense(feat) for feat in self.features]):
            x = lyr(x)
            if i != len(self.features) - 1:
                x = nn.relu(x)
        return x


# Functions must be passed to jraph GNNs, but pytype does not recognise
# linen Modules as callables to here we wrap in a function.
def make_embed_fn(latent_size):
    def embed(inputs):
        return nn.Dense(latent_size)(inputs)

    return embed


def make_mlp(features):
    @jraph.concatenated_args
    def update_fn(inputs):
        return ExplicitMLP(features)(inputs)

    return update_fn


class GraphNetwork(nn.Module):
    """A flax GraphNetwork."""

    mlp_features: Sequence[int]
    latent_size: int

    @nn.compact
    def __call__(self, graph):
        # Add a global parameter for graph classification.
        graph = graph._replace(globals=jnp.zeros([graph.n_node.shape[0], 1]))

        embedder = jraph.GraphMapFeatures(
            embed_node_fn=make_embed_fn(self.latent_size),
            embed_edge_fn=make_embed_fn(self.latent_size),
            embed_global_fn=make_embed_fn(self.latent_size),
        )
        net = jraph.GraphNetwork(
            update_node_fn=make_mlp(self.mlp_features),
            update_edge_fn=make_mlp(self.mlp_features),
            # The global update outputs size 2 for binary classification.
            update_global_fn=make_mlp(self.mlp_features + (2,)),
        )  # pytype: disable=unsupported-operands
        return net(embedder(graph))


def compute_loss(params, graph, labels, net):
    """Compute loss and accuracy."""

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
    )
    # Repeat the dataset forever for training.
    reader.repeat()

    net = GraphNetwork(mlp_features=(128, 128), latent_size=128)

    # Get a candidate graph and label to initialize the network.
    graph = reader.get_graph_by_idx(0)

    # Initialize the network.
    logging.info("Initializing network.")
    params = net.init(jax.random.PRNGKey(42), graph)

    optimizer = optax.adam(learning_rate=1e-4)
    opt_state = optimizer.init(params)

    def loss_fn(
        params,
        graph: jraph.GraphsTuple,
        labels: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        return compute_loss(params, graph, labels, net)

    loss_and_grad_fn = jax.jit(
        jax.value_and_grad(
            loss_fn,
            has_aux=True,
        )
    )

    def train_step(state, graph, labels):
        params, opt_state = state

        (metrics, gradients) = loss_and_grad_fn(
            params,
            graph,
            labels,
        )

        updates, opt_state = optimizer.update(
            gradients,
            opt_state,
            params,
        )
        params = optax.apply_updates(params, updates)

        return (params, opt_state), metrics

    params, _ = run_training(
        reader,
        (params, opt_state),
        train_step,
        num_training_steps=num_training_steps,
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
    )

    with pathlib.Path(save_dir, "molhiv.pkl").open("rb") as fp:
        params = pickle.load(fp)

    accumulated_loss = 0
    accumulated_accuracy = 0
    idx = 0

    # We jit the computation of our loss, since this is the main computation.
    # Using jax.jit means that we will use a single accelerator. If you want
    # to use more than 1 accelerator, use jax.pmap. More information can be
    # found in the jax documentation.
    net = GraphNetwork(mlp_features=[128, 128], latent_size=128)

    @jax.jit
    def eval_step(params, graph, labels):
        return compute_loss(
            params,
            graph,
            labels,
            net,
        )

    loss, accuracy = run_evaluation(
        reader,
        params,
        eval_step,
    )

    logging.info("Completed evaluation.")
    logging.info(
        "Eval loss: %s, accuracy %s",
        loss,
        accuracy,
    )

    return loss, accuracy


def main() -> None:
  args = parse_args()

  if args.mode == "train":
    train(
        args.data_path,
        args.master_csv_path,
        args.split_path,
        args.batch_size,
        args.num_training_steps,
        args.save_dir,
    )
  elif args.mode == "evaluate":
    _ = evaluate(
        args.data_path,
        args.master_csv_path,
        args.split_path,
        args.save_dir,
    ) # logged already


if __name__ == "__main__":
  main()