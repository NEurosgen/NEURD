"""The public skeleton API retains soma stitches and ordered limb geometry."""
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest

from tests.unit import skip_if_datasci_tools_unusable

skip_if_datasci_tools_unusable()

from neurd.neuron import Limb, Neuron
from neurd import neuron_utils as nru


def segment(start, end):
    return np.asarray([[start, end]], dtype=np.float64)


def limb(branches, origins):
    obj = object.__new__(Limb)
    obj.concept_network = nx.Graph()
    for node, axis in branches:
        obj.concept_network.add_node(node, data=SimpleNamespace(skeleton=axis))
    obj.all_concept_network_data = [dict(starting_soma=soma, starting_coordinate=np.array(point))
                                   for soma, point in origins]
    obj._index = 17
    return obj


def neuron(graph):
    obj = object.__new__(Neuron)
    obj.concept_network = graph
    obj._index = 23
    obj.pipeline_products = SimpleNamespace()
    return obj


def ordered_neuron():
    left = limb([(9, segment([1, 0, 0], [2, 0, 0])),
                 (2, segment([2, 0, 0], [3, 0, 0]))],
                [(1, [1, 0, 0]), (0, [3, 0, 0]), (1, [2, 0, 0])])
    right = limb([(8, segment([0, 1, 0], [0, 2, 0]))], [(1, [0, 1, 0])])
    graph = nx.Graph()
    graph.add_node("S1", data=SimpleNamespace(mesh_center=np.array([0, 0, 0])))
    graph.add_node("L9", data=left)
    graph.add_node("S0", data=SimpleNamespace(mesh_center=np.array([4, 0, 0])))
    graph.add_node("L2", data=right)
    graph.add_edges_from([("S1", "L2"), ("S1", "L9"), ("S0", "L9")])
    return neuron(graph)


def test_whole_neuron_keeps_branch_order_and_all_soma_stitches():
    obj = ordered_neuron()
    before = {name: {node: data["data"].skeleton.copy()
                     for node, data in obj[name].concept_network.nodes(data=True)}
              for name in obj.get_limb_node_names()}
    expected = np.asarray([
        [[2, 0, 0], [3, 0, 0]],  # L9's sorted branch node 2 before node 9.
        [[1, 0, 0], [2, 0, 0]],
        [[0, 1, 0], [0, 2, 0]],  # L2 follows L9's graph insertion order.
        [[0, 0, 0], [0, 1, 0]],  # S1's neighbor L2 was attached first.
        [[0, 0, 0], [1, 0, 0]],
        [[0, 0, 0], [2, 0, 0]],  # Preserve the second attachment to S1.
        [[4, 0, 0], [3, 0, 0]],  # S0 follows S1 in neuron node order.
    ], dtype=np.float64)
    np.testing.assert_array_equal(obj.get_skeleton(), expected)
    np.testing.assert_array_equal(obj.skeleton, expected)
    assert obj.get_skeleton().dtype == expected.dtype
    assert obj._index == 23
    for name, branches in before.items():
        assert obj[name]._index == 17
        for node, axis in branches.items():
            np.testing.assert_array_equal(obj[name].concept_network.nodes[node]["data"].skeleton, axis)


def test_soma_helper_filters_origins_and_keeps_repeated_attachments():
    obj = ordered_neuron()
    np.testing.assert_array_equal(nru.get_soma_skeleton(obj, "S1"), [
        [[0, 0, 0], [0, 1, 0]], [[0, 0, 0], [1, 0, 0]], [[0, 0, 0], [2, 0, 0]]])
    np.testing.assert_array_equal(nru.get_soma_skeleton(obj, "S0"), [
        [[4, 0, 0], [3, 0, 0]]])


def test_disconnected_neuron_can_skip_only_the_whole_connectivity_check():
    graph = nx.Graph()
    graph.add_node("L4", data=limb([(0, segment([1, 0, 0], [2, 0, 0]))], []))
    graph.add_node("L7", data=limb([(0, segment([8, 0, 0], [9, 0, 0]))], []))
    obj = neuron(graph)
    with pytest.raises(Exception, match="Skeleton is not one component: n_components = 2"):
        obj.get_skeleton()
    np.testing.assert_array_equal(obj.get_skeleton(check_connected_component=False), [
        [[1, 0, 0], [2, 0, 0]], [[8, 0, 0], [9, 0, 0]]])
    np.testing.assert_array_equal(obj.skeleton, obj.get_skeleton(check_connected_component=False))


def test_disconnected_limb_is_checked_even_if_neuron_check_is_disabled():
    graph = nx.Graph()
    graph.add_node("L4", data=limb([(0, segment([1, 0, 0], [2, 0, 0])),
                                   (1, segment([8, 0, 0], [9, 0, 0]))], []))
    with pytest.raises(Exception, match="Skeleton is not one component: n_components = 2"):
        neuron(graph).get_skeleton(check_connected_component=False)


def test_soma_without_attachments_keeps_empty_stack_behavior():
    graph = nx.Graph()
    graph.add_node("S0", data=SimpleNamespace(mesh_center=np.zeros(3)))
    assert nru.get_soma_skeleton(neuron(graph), "S0") == []


def test_print_flag_reports_ordered_limbs_and_somas(capsys):
    nru.get_whole_neuron_skeleton(ordered_neuron(), print_flag=True)
    output = capsys.readouterr().out
    markers = ["Working on limb: L9", "Working on limb: L2", "Working on soma: S1", "Working on soma: S0"]
    positions = [output.index(marker) for marker in markers]
    assert positions == sorted(positions)
    assert "curr_limb_skeleton.shape = (2, 2, 3)" in output
    assert "for soma S1, curr_soma_skeleton.shape = (3, 2, 3)" in output
