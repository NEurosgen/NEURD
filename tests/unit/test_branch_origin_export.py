"""Local origin records stay distinct from validated supplied face maps."""
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import trimesh

from tests.unit.test_mesh_axis_origin import helpers


def fixture(origin):
    parent = trimesh.creation.box()
    skeleton = np.array([[[1., 0., 0.], [0., 0., 0.]]])
    branch = SimpleNamespace(mesh=parent.submesh([[3, 1]], append=True, repair=False),
                             mesh_face_idx=np.array([3, 1]), skeleton=skeleton,
                             spines=None, mesh_axis_origin=origin)
    limb = SimpleNamespace(mesh=parent, branches=[branch], skeleton=None, concept_network=None)
    return SimpleNamespace(limbs=[limb], concept_network=None), branch


def test_export_keeps_observed_axis_origin_separate_from_file_and_face_validation(tmp_path, raw_exporter):
    origin = helpers()['_new_axis_origin']('test_pre_normalization',
                                         np.array([[[0., 0., 0.], [1., 0., 0.]]]))
    obj, branch = fixture(origin)
    original_axis = branch.skeleton.copy()
    raw_exporter.save_segmentation(obj, tmp_path)
    folder = tmp_path / 'limb_000/branch_000'
    journal = json.loads((folder / 'branch_origin.json').read_text())
    face = json.loads((folder / 'face_provenance.json').read_text())
    assert journal['status'] == face['status'] == 'available'
    assert journal['raw_input_map_available'] is False
    assert journal['observation']['raw_input_map_available'] is False
    assert journal['observation']['axis'] == origin['axis']
    assert journal['scope'] == 'local_map_mp_selection_and_axis_pairing'
    assert journal['exported_skeleton_file_sha256'] == hashlib.sha256((folder / 'branch_skeleton.npy').read_bytes()).hexdigest()
    np.testing.assert_array_equal(np.load(folder / 'branch_skeleton.npy'), original_axis)
    np.testing.assert_array_equal(branch.skeleton, original_axis)
    assert np.load(folder / 'parent_face_indices.npy').tolist() == [3, 1]
    assert face['mapping_origin'] == 'provided_indices_not_recovered_from_geometry'


@pytest.mark.parametrize('origin,reason', [
    (None, 'metadata_missing'),
    ({'schema_version': 8}, 'metadata_schema_unavailable'),
    ({'schema_version': True}, 'metadata_schema_unavailable'),
    ({'schema_version': 1, 'scope': 'raw_recovered'}, 'metadata_schema_unavailable'),
    ({'schema_version': 1, 'scope': 'local_map_mp_selection_and_axis_pairing',
      'raw_input_map_available': True}, 'metadata_schema_unavailable'),
    ({'schema_version': 1, 'scope': 'local_map_mp_selection_and_axis_pairing',
      'raw_input_map_available': False, 'invalid_array': np.array([1])}, 'metadata_not_json_safe'),
])
def test_missing_or_unsupported_origin_is_unavailable_even_when_supplied_map_passes(tmp_path, origin, reason, raw_exporter):
    obj, _ = fixture(origin)
    raw_exporter.save_segmentation(obj, tmp_path)
    folder = tmp_path / 'limb_000/branch_000'
    journal = json.loads((folder / 'branch_origin.json').read_text())
    assert journal['status'] == 'unavailable'
    assert journal['reason'] == reason
    assert journal['observation'] is None and journal['raw_input_map_available'] is False
    assert json.loads((folder / 'face_provenance.json').read_text())['status'] == 'available'


def test_origin_without_skeleton_has_explicitly_missing_export_hash(tmp_path, raw_exporter):
    origin = helpers()['_new_axis_origin']('test', np.zeros((1, 2, 3)))
    branch = SimpleNamespace(mesh_axis_origin=origin)
    raw_exporter._save_branch_origin(branch, tmp_path)
    journal = json.loads((tmp_path / 'branch_origin.json').read_text())
    assert journal['status'] == 'available'
    assert journal['exported_skeleton_file_sha256'] is None
