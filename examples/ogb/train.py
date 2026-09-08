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

uv run python train.py -h 

to get CLI help
"""

import argparse
import functools
import logging
import pathlib
import pickle

import haiku as hk
import jax
import jax.numpy as jnp
import optax

from examples.ogb import data_utils
from examples.ogb._training import (
    StepMetrics,
    loss_and_accuracy,
    run_evaluation,
    run_training,
)
from jraph import GraphMapFeatures, GraphNetwork, GraphsTuple, concatenated_args


@concatenated_args
def edge_update_fn(feats: jnp.ndarray) -> jnp.ndarray:
    """Edge update function for graph net."""
    net = hk.Sequential([hk.Linear(128), jax.nn.relu, hk.Linear(128)])
    return net(feats)


@concatenated_args
def node_update_fn(feats: jnp.ndarray) -> jnp.ndarray:
    """Node update function for graph net."""
    net = hk.Sequential([hk.Linear(128), jax.nn.relu, hk.Linear(128)])
    return net(feats)


@concatenated_args
def update_global_fn(feats: jnp.ndarray) -> jnp.ndarray:
    """Global update function for graph net."""
    # Molhiv is a binary classification task, so output pos neg logits.
    net = hk.Sequential([hk.Linear(128), jax.nn.relu, hk.Linear(2)])
    return net(feats)


def net_fn(graph: GraphsTuple) -> GraphsTuple:
    """Graph net function."""
    # Add a global paramater for graph classification.
    graph = graph._replace(globals=jnp.zeros([graph.n_node.shape[0], 1]))
    embedder = GraphMapFeatures(hk.Linear(128), hk.Linear(128), hk.Linear(128))
    net = GraphNetwork(
        update_node_fn=node_update_fn,
        update_edge_fn=edge_update_fn,
        update_global_fn=update_global_fn,
    )
    return net(embedder(graph))


def compute_loss(params, graph, label, net):
    """Computes loss."""
    predicted_graph = net.apply(params, graph)
    return loss_and_accuracy(predicted_graph, label)


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

    compute_loss_fn = functools.partial(compute_loss, net=net)
    # We jit the computation of our loss, since this is the main computation.
    # Using jax.jit means that we will use a single accelerator. If you want
    # to use more than 1 accelerator, use jax.pmap. More information can be
    # found in the jax documentation.
    compute_loss_fn = jax.jit(jax.value_and_grad(compute_loss_fn, has_aux=True))

    def train_step(
        state,
        graph: GraphsTuple,
        labels: jax.Array,
    ) -> tuple[object, StepMetrics]:
        params, opt_state = state

        (metrics, gradients) = compute_loss_fn(
            params,
            graph,
            labels,
        )

        updates, opt_state = opt_update(
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
    # Transform impure `net_fn` to pure functions with hk.transform.
    net = hk.without_apply_rng(hk.transform(net_fn))
    with pathlib.Path(save_dir, "molhiv.pkl").open("rb") as fp:
        params = pickle.load(fp)
    
    # We jit the computation of our loss, since this is the main computation.
    # Using jax.jit means that we will use a single accelerator. If you want
    # to use more than 1 accelerator, use jax.pmap. More information can be
    # found in the jax documentation.
    compute_loss_fn = jax.jit(functools.partial(compute_loss, net=net))

    def eval_step(params, graph: GraphsTuple, labels):
        return compute_loss_fn(params, graph, labels)

    loss, accuracy = run_evaluation(
        reader,
        params,
        eval_step,
    )

    logging.info("Completed evaluation.")
    logging.info("Eval loss: %s, accuracy %s", loss, accuracy)
    return loss, accuracy



def parse_args() -> argparse.Namespace:
  common_parser = argparse.ArgumentParser(add_help=False)
  common_parser.add_argument("--data_path", help="Directory of the data.")
  common_parser.add_argument(
      "--split_path",
      help="Path to the data split indices.",
  )
  common_parser.add_argument(
      "--master_csv_path",
      help="Path to OGB master.csv.",
  )
  common_parser.add_argument(
      "--save_dir",
      help="Directory to save parameters to.",
  )

  parser = argparse.ArgumentParser()
  subparsers = parser.add_subparsers(dest="command", required=True)

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


def main() -> None:
  args = parse_args()

  if args.command == "train":
    train(
        args.data_path,
        args.master_csv_path,
        args.split_path,
        args.batch_size,
        args.num_training_steps,
        args.save_dir,
    )
  elif args.command == "evaluate":
    evaluate(
        args.data_path,
        args.master_csv_path,
        args.split_path,
        args.save_dir,
    )


if __name__ == "__main__":
  main()

