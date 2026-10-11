from hashlib import sha256
import json
from types import SimpleNamespace

import numpy as np
import pytest
import trimesh

from neurd.face_provenance import validate_ordered_face_indices, write_branch_face_provenance


@pytest.fixture
def surfaces():
    parent = trimesh.Trimesh(vertices=[[0, 0, 0], [10, 0, 0], [0, 10, 0], [0, 0, 10]],
        faces=[[0, 1, 2], [0, 1, 2], [0, 2, 3], [0, 3, 1]], process=False)
    ids = np.array([3, 1, 2], dtype=np.int64)
    child = parent.submesh([ids], append=True, repair=False)
    return parent, child, ids


def test_declared_duplicate_geometry_identity_and_nonmonotonic_order_are_retained(surfaces):
    parent, child, ids = surfaces
    parent._cache.clear()
    child._cache.clear()
    result = validate_ordered_face_indices(parent, child, ids, chunk_size=1)
    assert result.tolist() == [3, 1, 2]  # duplicate geometry belongs to supplied face 1
    assert np.array_equal(parent.vertices[parent.faces[result]], child.vertices[child.faces])
    assert 'triangles' not in parent._cache.cache
    assert 'triangles' not in child._cache.cache


@pytest.mark.parametrize('kind', ['float', 'bool', 'object', 'two_dimensions', 'negative',
                                  'bounds', 'uint64_overflow', 'repeated', 'count', 'row_order'])
def test_bad_declared_maps_are_rejected(surfaces, kind):
    parent, child, ids = surfaces
    bad = dict(float=ids.astype(float), bool=ids.astype(bool), object=ids.astype(object),
        two_dimensions=ids[:, None], negative=np.array([3, -1, 2]),
        bounds=np.array([3, 4, 2]), uint64_overflow=np.array([3, 2**64 - 1, 2], dtype=np.uint64),
        repeated=np.array([3, 3, 2]), count=ids[:-1], row_order=ids[::-1])[kind]
    with pytest.raises(ValueError, match='[Oo]rdered face'):
        validate_ordered_face_indices(parent, child, bad)


def test_exact_coordinates_required_and_input_geometry_not_modified(surfaces):
    parent, child, ids = surfaces
    before = parent.vertices.copy(), parent.faces.copy(), ids.copy()
    child.vertices[0, 0] += 1e-8
    with pytest.raises(ValueError, match='triangle rows'):
        validate_ordered_face_indices(parent, child, ids)
    assert np.array_equal(parent.vertices, before[0])
    assert np.array_equal(parent.faces, before[1])
    assert np.array_equal(ids, before[2])


def test_corner_permutations_do_not_change_face_identity(surfaces):
    parent, child, ids = surfaces
    child.faces = child.faces[:, [2, 1, 0]]
    assert validate_ordered_face_indices(parent, child, ids).tolist() == [3, 1, 2]


@pytest.mark.parametrize('side', ['parent', 'child'])
@pytest.mark.parametrize('bad_vertex', [-1, 4])
def test_malformed_vertex_references_are_rejected(side, bad_vertex):
    vertices = [[0, 0, 0], [10, 0, 0], [0, 10, 0], [0, 0, 0]]
    parent = trimesh.Trimesh(vertices=vertices, faces=[[0, 1, 2]], process=False)
    child = trimesh.Trimesh(vertices=vertices, faces=[[0, 1, 2]], process=False)
    (parent if side == 'parent' else child).faces[0, 0] = bad_vertex
    with pytest.raises(ValueError, match='vertex index is outside'):
        validate_ordered_face_indices(parent, child, np.array([0], dtype=np.int64))


@pytest.mark.parametrize('side', ['parent', 'child'])
def test_nonfinite_related_geometry_rejected(surfaces, side):
    parent, child, ids = surfaces
    (parent if side == 'parent' else child).vertices[0, 0] = np.nan
    with pytest.raises(ValueError, match='nonfinite'):
        validate_ordered_face_indices(parent, child, ids)


@pytest.mark.parametrize('size', [0, -1, 2.5, True])
def test_invalid_chunk_budget_rejected(surfaces, size):
    with pytest.raises(ValueError, match='chunk size'):
        validate_ordered_face_indices(*surfaces[:2], surfaces[2], chunk_size=size)


def test_journal_links_exported_files_and_provided_indices(surfaces, tmp_path):
    parent, child, ids = surfaces
    folder = tmp_path / 'limb_000/branch_000'
    folder.mkdir(parents=True)
    parent.export(folder.parent / 'limb_mesh.off')
    child.export(folder / 'branch_mesh.off')
    result = write_branch_face_provenance(SimpleNamespace(mesh=child, mesh_face_idx=ids), parent, folder)
    assert result['status'] == 'available'
    assert result['raw_input_map_available'] is False
    assert result['index_scope'] == 'limb_mesh.face_index'
    assert np.load(folder / result['indices_file'], allow_pickle=False).tolist() == [3, 1, 2]
    for field, path in [('parent_mesh_sha256', folder.parent / 'limb_mesh.off'),
                        ('child_mesh_sha256', folder / 'branch_mesh.off'),
                        ('indices_sha256', folder / 'parent_face_indices.npy')]:
        assert result[field] == sha256(path.read_bytes()).hexdigest()
    assert json.loads((folder / 'face_provenance.json').read_text()) == result


def test_missing_map_explicitly_unavailable_without_inference(surfaces, tmp_path):
    parent, child, _ = surfaces
    result = write_branch_face_provenance(SimpleNamespace(mesh=child), parent, tmp_path)
    assert result['status'] == 'unavailable'
    assert result['reason'] == 'ordered_parent_indices_not_supplied'
    assert not (tmp_path / 'parent_face_indices.npy').exists()


def test_bad_map_has_no_published_journal_or_indices(surfaces, tmp_path):
    parent, child, ids = surfaces
    with pytest.raises(ValueError, match='triangle rows'):
        write_branch_face_provenance(SimpleNamespace(mesh=child, mesh_face_idx=ids[::-1]), parent, tmp_path)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('missing', ['parent', 'child'])
def test_declared_map_requires_both_surfaces(surfaces, tmp_path, missing):
    parent, child, ids = surfaces
    with pytest.raises(ValueError, match='both branch and limb'):
        write_branch_face_provenance(SimpleNamespace(mesh=None if missing == 'child' else child,
                                                    mesh_face_idx=ids),
                                    None if missing == 'parent' else parent, tmp_path)


def test_empty_declared_child_has_valid_empty_map(surfaces):
    parent, _, _ = surfaces
    child = trimesh.Trimesh(vertices=np.empty((0, 3)), faces=np.empty((0, 3), dtype=int), process=False)
    result = validate_ordered_face_indices(parent, child, np.empty(0, dtype=np.int64))
    assert result.dtype == np.int64 and result.shape == (0,)
