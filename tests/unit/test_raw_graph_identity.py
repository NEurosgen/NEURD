"""Export branch positions using graph object identity and explicit availability."""
import json
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest
import trimesh


import process_all_neurons as batch


def limb_with(graph, branches):
    return SimpleNamespace(concept_network=graph, branches=branches,
                           current_starting_node=next(iter(graph), None))


def read_map(limb, tmp_path):
    batch._save_limb_connectivity(limb, tmp_path)
    return json.loads((tmp_path / "branch_index_map.json").read_text())


def test_sparse_nodes_use_actual_branch_identity(tmp_path):
    left, right = SimpleNamespace(), SimpleNamespace()
    graph = nx.Graph()
    graph.add_node(9, data=right)
    graph.add_node(2, data=left)
    graph.add_edge(9, 2)
    record = read_map(limb_with(graph, [left, right]), tmp_path)
    assert record["branch_index_to_node_id"] == [2, 9]
    assert record["node_id_to_branch_index"] == {"2": 0, "9": 1}
    assert np.load(tmp_path / "branch_nodes.npy").tolist() == [2, 9]
    assert np.load(tmp_path / "branch_edges.npy").tolist() == [[9, 2]]


@pytest.mark.parametrize("branches_reversed", [False, True])
def test_nodes_inserted_two_then_zero_keep_export_order(tmp_path, branches_reversed):
    zero, two = SimpleNamespace(), SimpleNamespace()
    graph = nx.Graph()
    graph.add_node(2, data=two)
    graph.add_node(0, data=zero)
    branches = [two, zero] if branches_reversed else [zero, two]
    record = read_map(limb_with(graph, branches), tmp_path)
    assert record["branch_index_to_node_id"] == ([2, 0] if branches_reversed else [0, 2])


def test_ordinary_contiguous_identity_is_preserved(tmp_path):
    branches = [SimpleNamespace(), SimpleNamespace()]
    graph = nx.Graph()
    for node, branch in enumerate(branches):
        graph.add_node(node, data=branch)
    graph.add_edge(0, 1)
    record = read_map(limb_with(graph, branches), tmp_path)
    assert record["branch_index_to_node_id"] == [0, 1]
    assert all(row == {"status": "available", "source": "graph_object_identity"}
               for row in record["branch_index_resolution"])


def test_same_object_at_two_nodes_is_explicitly_unavailable(tmp_path):
    branch = SimpleNamespace()
    graph = nx.Graph()
    graph.add_node(2, data=branch)
    graph.add_node(9, data=branch)
    record = read_map(limb_with(graph, [branch]), tmp_path)
    assert record["branch_index_to_node_id"] == [None]
    assert record["node_id_to_branch_index"] == {}
    assert record["branch_index_resolution"] == [
        {"status": "unavailable", "reason": "ambiguous_graph_object_identity"}]


def test_ambiguous_identity_is_not_hidden_by_explicit_id(tmp_path):
    branch = SimpleNamespace(node_id=9)
    graph = nx.Graph()
    graph.add_node(2, data=branch)
    graph.add_node(9, data=branch)
    assert read_map(limb_with(graph, [branch]), tmp_path)["branch_index_to_node_id"] == [None]


def test_equal_counts_and_equal_object_values_do_not_establish_identity(tmp_path):
    branches = [SimpleNamespace(value=1), SimpleNamespace(value=2)]
    graph = nx.Graph()
    graph.add_node(0, data=SimpleNamespace(value=1))
    graph.add_node(1, data=SimpleNamespace(value=2))
    record = read_map(limb_with(graph, branches), tmp_path)
    assert record["branch_index_to_node_id"] == [None, None]
    assert all(row["reason"] == "graph_object_identity_missing"
               for row in record["branch_index_resolution"])


@pytest.mark.parametrize("attribute", ["node_id", "branch_id"])
def test_stale_explicit_identifier_fails_before_metadata(tmp_path, attribute):
    branch = SimpleNamespace(**{attribute: 77})
    graph = nx.Graph()
    graph.add_node(2, data=branch)
    with pytest.raises(ValueError, match="absent from the graph"):
        read_map(limb_with(graph, [branch]), tmp_path)
    assert not list(tmp_path.iterdir())


def test_explicit_identifier_conflicting_with_identity_fails(tmp_path):
    branch = SimpleNamespace(node_id=0)
    graph = nx.Graph()
    graph.add_node(2, data=branch)
    graph.add_node(0, data=SimpleNamespace())
    with pytest.raises(ValueError, match="disagrees with graph object identity"):
        read_map(limb_with(graph, [branch]), tmp_path)


def test_two_explicit_identifiers_must_agree(tmp_path):
    branch = SimpleNamespace(node_id=2, branch_id=0)
    graph = nx.Graph()
    graph.add_node(2, data=branch)
    graph.add_node(0)
    with pytest.raises(ValueError, match="node_id and branch_id disagree"):
        read_map(limb_with(graph, [branch]), tmp_path)


def test_explicit_identifier_can_resolve_graph_without_data(tmp_path):
    branch = SimpleNamespace(branch_id=7)
    graph = nx.Graph()
    graph.add_node(7)
    record = read_map(limb_with(graph, [branch]), tmp_path)
    assert record["branch_index_to_node_id"] == [7]
    assert record["branch_index_resolution"] == [
        {"status": "available", "source": "validated_explicit_node_id"}]


def test_explicit_identifier_cannot_steal_other_branch_node(tmp_path):
    branch = SimpleNamespace(node_id=7)
    graph = nx.Graph()
    graph.add_node(7, data=SimpleNamespace())
    with pytest.raises(ValueError, match="belongs to different graph data"):
        read_map(limb_with(graph, [branch]), tmp_path)


def test_repeated_export_object_cannot_share_unique_graph_node(tmp_path):
    branch = SimpleNamespace()
    graph = nx.Graph()
    graph.add_node(2, data=branch)
    with pytest.raises(ValueError, match="assigned more than once"):
        read_map(limb_with(graph, [branch, branch]), tmp_path)


def test_no_graph_keeps_export_without_connectivity_files(tmp_path):
    assert batch._save_limb_connectivity(SimpleNamespace(branches=[]), tmp_path) is None
    assert not list(tmp_path.iterdir())


def test_empty_graph_has_explicit_empty_mapping(tmp_path):
    record = read_map(limb_with(nx.Graph(), []), tmp_path)
    assert record["branch_index_to_node_id"] == []
    assert record["node_id_to_branch_index"] == {}


def test_mesh_and_skeleton_files_follow_same_branch_object_as_identity(tmp_path):
    zero = SimpleNamespace(mesh=trimesh.creation.box(),
        skeleton=np.array([[[0., 0., 0.], [1., 0., 0.]]]), spines=None)
    two = SimpleNamespace(mesh=trimesh.Trimesh(
        vertices=[[10, 0, 0], [11, 0, 0], [10, 1, 0], [10, 0, 1]],
        faces=[[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], process=False),
        skeleton=np.array([[[10., 0., 0.], [11., 0., 0.]]]), spines=None)
    graph = nx.Graph()
    graph.add_node(2, data=two)
    graph.add_node(0, data=zero)
    graph.add_edge(2, 0)
    limb = limb_with(graph, [zero, two])
    limb.mesh, limb.skeleton = None, None
    batch.save_segmentation(SimpleNamespace(limbs=[limb], concept_network=None), tmp_path)
    folder = tmp_path / "limb_000"
    record = json.loads((folder / "branch_index_map.json").read_text())
    assert record["branch_index_to_node_id"] == [0, 2]
    for index, branch in enumerate([zero, two]):
        np.testing.assert_array_equal(np.load(folder / f"branch_{index:03d}/branch_skeleton.npy"),
                                      branch.skeleton)
        exported = trimesh.load_mesh(folder / f"branch_{index:03d}/branch_mesh.off", process=False)
        np.testing.assert_array_equal(exported.faces, branch.mesh.faces)
        np.testing.assert_array_equal(exported.vertices, branch.mesh.vertices)
