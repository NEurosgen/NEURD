import json
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest
import trimesh

import process_all_neurons as batch
from neurd import face_provenance


def fixture():
    parent = trimesh.creation.box()
    branches = [SimpleNamespace(mesh=parent.submesh([[index]], append=True, repair=False),
        mesh_face_idx=np.array([index]), skeleton=np.array([[[index, 0., 0.], [index, 1., 0.]]]),
        spines=None) for index in [3, 1]]
    graph = nx.Graph()
    for node, branch in zip([9, 2], branches):
        graph.add_node(node, data=branch)
    graph.add_edge(9, 2)
    limb = SimpleNamespace(mesh=parent, skeleton=None, branches=branches,
                           concept_network=graph, current_starting_node=9)
    return SimpleNamespace(limbs=[limb], concept_network=None), parent, branches


def test_export_graph_and_ordered_faces_describe_the_same_branch_objects(tmp_path, monkeypatch):
    obj, parent, branches = fixture()
    hashed = []
    original = face_provenance.file_sha256
    def observe(path):
        hashed.append(Path(path))
        return original(path)
    monkeypatch.setattr(face_provenance, 'file_sha256', observe)
    batch.save_segmentation(obj, tmp_path)
    folder = tmp_path / 'limb_000'
    graph_map = json.loads((folder / 'branch_index_map.json').read_text())
    assert graph_map['branch_index_to_node_id'] == [9, 2]
    assert hashed.count(folder / 'limb_mesh.off') == 1
    for index, expected in enumerate([3, 1]):
        directory = folder / f'branch_{index:03d}'
        record = json.loads((directory / 'face_provenance.json').read_text())
        assert record['status'] == 'available' and record['raw_input_map_available'] is False
        assert np.load(directory / 'parent_face_indices.npy').tolist() == [expected]
        assert record['parent_mesh_sha256'] == original(folder / 'limb_mesh.off')
        assert np.array_equal(np.load(directory / 'branch_skeleton.npy'), branches[index].skeleton)


def test_export_fails_on_contradictory_declared_face_relationship(tmp_path):
    obj, parent, branches = fixture()
    branches[0].mesh_face_idx = np.array([0])
    with pytest.raises(ValueError, match='triangle rows'):
        batch.save_segmentation(obj, tmp_path)
    assert not (tmp_path / 'limb_000/branch_000/face_provenance.json').exists()


def test_export_explicitly_marks_missing_maps_unavailable(tmp_path):
    obj, _, branches = fixture()
    branches[0].mesh_face_idx = None
    batch.save_segmentation(obj, tmp_path)
    first = tmp_path / 'limb_000/branch_000'
    assert json.loads((first / 'face_provenance.json').read_text())['status'] == 'unavailable'
    assert not (first / 'parent_face_indices.npy').exists()
    assert json.loads((tmp_path / 'limb_000/branch_001/face_provenance.json').read_text())['status'] == 'available'
