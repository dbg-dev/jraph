"""Smoke tests for examples.hamiltonian_graph_network."""

import numpy as np

from examples.jraph import hamiltonian_graph_network


def test_build_hookes_particle_state_graph() -> None:
    np.random.seed(42)

    graph = hamiltonian_graph_network.build_hookes_particle_state_graph(4)

    np.testing.assert_array_equal(graph.n_node, np.asarray([4]))
    np.testing.assert_array_equal(graph.n_edge, np.asarray([12]))
    assert graph.nodes["mass"].shape == (4,)
    assert graph.nodes["position"].shape == (4, 2)
    assert graph.nodes["momentum"].shape == (4, 2)
    assert graph.edges["spring_constant"].shape == (12,)
    assert graph.senders.shape == (12,)
    assert graph.receivers.shape == (12,)


def test_hookes_hamiltonian_is_finite() -> None:
    np.random.seed(42)
    graph = hamiltonian_graph_network.build_hookes_particle_state_graph(4)

    output = hamiltonian_graph_network.hookes_hamiltonian_from_graph_fn(graph)
    energy = output.globals["hamiltonian"]

    assert energy.shape == (1,)
    assert np.all(np.isfinite(np.asarray(energy)))


def test_verlet_integration_step_preserves_state_structure() -> None:
    np.random.seed(42)
    graph = hamiltonian_graph_network.build_hookes_particle_state_graph(4)

    energy, next_graph = hamiltonian_graph_network.single_integration_step(
        graph,
        time_step=0.002,
        integrator_fn=hamiltonian_graph_network.verlet_integrator,
        hamiltonian_from_graph_fn=(
            hamiltonian_graph_network.hookes_hamiltonian_from_graph_fn
        ),
    )

    assert np.all(np.isfinite(np.asarray(energy)))
    assert next_graph.nodes["mass"].shape == (4,)
    assert next_graph.nodes["position"].shape == (4, 2)
    assert next_graph.nodes["momentum"].shape == (4, 2)
    assert next_graph.edges["spring_constant"].shape == (12,)
    np.testing.assert_array_equal(next_graph.senders, graph.senders)
    np.testing.assert_array_equal(next_graph.receivers, graph.receivers)
