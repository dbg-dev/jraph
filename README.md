![Jraph logo](images/logo.png)

# Jraph

Jraph (pronounced “giraffe”) is a lightweight library for building graph neural
networks in JAX. It provides a graph data structure, utilities for batching,
padding and segment operations, and a small collection of composable
message-passing models.

## Maintenance status

This repository is an independent continuation of the original
[DeepMind Jraph project](https://github.com/google-deepmind/jraph).

The project preserves Jraph's lightweight `GraphsTuple` data model and
functional graph utilities while updating the repository for Python 3.14,
current JAX, pytest, uv, Optax and Flax NNX. The retained examples and
experimental sharding implementation are covered by the test suite.

The public Jraph API remains intentionally small and close to the original
project.

## Installation

Jraph requires Python 3.14.

To install this development version directly from GitHub:

```bash
python -m pip install \
  "jraph @ git+https://github.com/dbg-dev/jraph.git@nnx-modernisation"
```

To work on the repository and install the development and example dependencies:

```bash
git clone --branch nnx-modernisation https://github.com/dbg-dev/jraph.git
cd jraph
make sync
```

## Quick start

Jraph takes inspiration from the TensorFlow
[`graph_nets` library](https://github.com/deepmind/graph_nets). Its central
data structure is `GraphsTuple`, a `NamedTuple` representing one or more
directed graphs.

### Representing graphs with `GraphsTuple`

```python
import jax.numpy as jnp

import jraph

# Define a three-node graph with one feature per node.
node_features = jnp.array([[0.0], [1.0], [2.0]])

# Each edge runs from senders[i] to receivers[i].
senders = jnp.array([0, 1, 2])
receivers = jnp.array([1, 2, 0])
edge_features = jnp.array([[5.0], [6.0], [7.0]])

# n_node and n_edge contain the sizes of the graphs in the tuple.
n_node = jnp.array([3])
n_edge = jnp.array([3])
global_context = jnp.array([[1.0]])

graph = jraph.GraphsTuple(
    nodes=node_features,
    edges=edge_features,
    senders=senders,
    receivers=receivers,
    globals=global_context,
    n_node=n_node,
    n_edge=n_edge,
)
```

A `GraphsTuple` can contain more than one graph. Batching concatenates node,
edge and global features along their leading axes and adjusts the sender and
receiver indices:

```python
batched_graph = jraph.batch([graph, graph])

assert batched_graph.nodes.shape == (6, 1)
assert batched_graph.edges.shape == (6, 1)
assert batched_graph.n_node.tolist() == [3, 3]
assert batched_graph.n_edge.tolist() == [3, 3]
```

The `nodes`, `edges` and `globals` fields may contain nested feature
trees. Every leaf within a field must have a common leading dimension
corresponding to the total number of nodes, edges or graphs:

```python
node_targets = jnp.array([[True], [False], [True]])
graph_with_targets = graph._replace(
    nodes={"inputs": graph.nodes, "targets": node_targets},
)
```

## Overview

Jraph provides graph data structures and transformations without prescribing a
neural-network framework:

- `jraph/_src/graph.py` defines `GraphsTuple`, the central representation
  for one or more directed graphs.
- `jraph/_src/utils.py` provides batching, padding, masking, segment
  operations and graph-construction utilities.
- `jraph/_src/models.py` provides composable reference implementations of
  common message-passing architectures.
- `jraph/experimental/` contains the distributed graph-network
  implementation.
- `examples/jraph/` demonstrates Jraph with current JAX, Flax NNX and Optax.
- `examples/ogb/` contains single-device, Flax and multi-device OGB training
  paths.
- `examples/pygcn/` provides a graph-convolution example using the Cora
  dataset.

## Models and examples

Jraph's model functions define how information moves between edges, nodes and
global graph attributes. User-provided update functions perform the feature
transformations and may be ordinary JAX functions or learned neural networks.

For example, a `GraphNetwork` can update edge features using the current edge,
sender-node, receiver-node and global features:

```python
def update_edge_fn(edge, sender, receiver, globals_):
    del globals_
    return edge + sender + receiver


network = jraph.GraphNetwork(
    update_edge_fn=update_edge_fn,
    update_node_fn=None,
)
updated_graph = network(graph)
```

The runnable examples provide complete workflows:

- [`examples/jraph/basic.py`](examples/jraph/basic.py) introduces the graph
  representation, batching, padding and models.
- [`examples/jraph/zacharys_karate_club.py`](examples/jraph/zacharys_karate_club.py)
  trains an NNX graph model on Zachary's karate club.
- [`examples/jraph/e_voting.py`](examples/jraph/e_voting.py),
  [`sat.py`](examples/jraph/sat.py) and
  [`higgs_detection.py`](examples/jraph/higgs_detection.py) demonstrate
  additional NNX graph-learning problems.
- [`examples/pygcn/`](examples/pygcn/) contains a graph convolutional network
  for the Cora citation dataset.
- [`examples/ogb/`](examples/ogb/) contains Haiku, Flax and multi-device
  Open Graph Benchmark training paths.
- [CIGRE GNN](https://github.com/dbg-dev/cigre-gnn) is a standalone applied
  example using Jraph and NNX for electrical-network regression.

The tests under [`tests/jraph/`](tests/jraph/) provide focused examples of
individual models and utilities, including their expected inputs and outputs.
The tests under [`tests/examples/`](tests/examples/) show how the executable
examples are configured for small, deterministic runs.

## Development

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for development setup and contribution
guidance.

Run all required repository checks with:

```bash
make check
```

This checks formatting, runs Ruff, executes the full test suite and builds the
source and wheel distributions. Static type checking is currently available as
an advisory check through `make typecheck`.

## Resources

- [Educational introduction to graph neural networks and Jraph](https://github.com/deepmind/educational/blob/master/colabs/summer_schools/intro_to_graph_nets_tutorial_with_jraph.ipynb)
- [OGBG-MOLPCBA example](https://github.com/google/flax/tree/main/examples/ogbg_molpcba)
- [PyTorch data-loading guide for Jraph](https://colab.research.google.com/drive/1_X2su92_nS52RNl4m-WYvmkvUSrFE4xQ)
- [Distributed graph-network implementation](jraph/experimental/sharded_graphnet.py)
- [Multi-device OGB example](examples/ogb/train_pmap.py)

## Citing Jraph

To cite this repository:

```
@software{jraph2020github,
  author = {Jonathan Godwin* and Thomas Keck* and Peter Battaglia and Victor Bapst and Thomas Kipf and Yujia Li and Kimberly Stachenfeld and Petar Veli\v{c}kovi\'{c} and Alvaro Sanchez-Gonzalez},
  title = {{J}raph: {A} library for graph neural networks in jax.},
  url = {http://github.com/deepmind/jraph},
  version = {0.0.1.dev},
  year = {2020},
}
```

## Licence

Jraph is distributed under the [Apache License 2.0](LICENSE).
