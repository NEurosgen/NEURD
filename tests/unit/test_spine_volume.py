"""Geometry regressions without importing NEURD's unrelated optional stack."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
import trimesh

path = Path(__file__).resolve().parents[2] / "neurd" / "spine_volume.py"
spec = importlib.util.spec_from_file_location("spine_volume_under_test", path)
volume = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = volume
spec.loader.exec_module(volume)


def fan_for_box(mesh):
    result = mesh.copy()
    trimesh.repair.fill_holes(result)
    return result


def open_box():
    mesh = trimesh.creation.box(extents=[100, 200, 300])
    mesh.update_faces(np.arange(len(mesh.faces)) != 0)
    return mesh


def test_closed_volume_and_input_unchanged():
    mesh = trimesh.creation.box(extents=[100, 200, 300])
    mesh.apply_translation([2e6, 1e6, 3e4])
    vertices, faces = mesh.vertices.copy(), mesh.faces.copy()
    result = volume.estimate_mesh_volume(mesh)
    assert result['method'] == 'input_closed'
    assert result['volume_nm3'] == pytest.approx(6e6)
    assert np.array_equal(mesh.vertices, vertices)
    assert np.array_equal(mesh.faces, faces)


def test_open_input_can_be_closed_without_hull():
    result = volume.estimate_mesh_volume(open_box(), close_holes=fan_for_box)
    assert result['valid'] and not result['input_watertight']
    assert result['method'] == 'fan_closed'
    assert result['volume_nm3'] == pytest.approx(6e6)


def test_no_border_does_not_make_nonmanifold_closure_valid():
    def malformed_closure(mesh):
        closed = fan_for_box(mesh)
        # All edges have at least two incident faces; one duplicated triangle
        # makes three incident faces on its edges, defeating the old gate.
        return trimesh.Trimesh(vertices=closed.vertices,
                               faces=np.vstack([closed.faces, closed.faces[0]]),
                               process=False)
    result = volume.estimate_mesh_volume(open_box(), close_holes=malformed_closure)
    assert result['method'] == 'convex_hull'
    assert result['fallback_reason'] == 'not_watertight'
    assert result['volume_nm3'] == pytest.approx(6e6)


@pytest.mark.parametrize('offset', [[0, 0, 0], [2e6, -3e6, 4e6]])
@pytest.mark.parametrize('scale', [[1, 1, 1], [2, 2, 2], [8, 8, 33]])
def test_translation_and_voxel_scale(offset, scale):
    mesh = open_box()
    mesh.vertices = mesh.vertices * np.asarray(scale) + np.asarray(offset)
    result = volume.estimate_mesh_volume(mesh, close_holes=fan_for_box)
    assert result['volume_nm3'] == pytest.approx(6e6 * np.prod(scale), rel=1e-8)


def test_invalid_geometry_is_missing_not_zero():
    flat = trimesh.Trimesh(vertices=[[0,0,0],[1,0,0],[1,1,0],[0,1,0]],
                           faces=[[0,1,2],[0,2,3]], process=False)
    result = volume.estimate_mesh_volume(flat)
    assert not result['valid'] and result['volume_nm3'] is None
    assert np.isnan(volume.mesh_volume(flat))


def test_bad_closer_cannot_inflate_bounds():
    result = volume.estimate_mesh_volume(open_box(), close_holes=lambda mesh: trimesh.creation.box(extents=[1000]*3))
    assert result['method'] == 'convex_hull'
    assert result['fallback_reason'] == 'closure_outside_input_bounds'
    assert result['volume_nm3'] == pytest.approx(6e6)


def test_missing_dependency_is_not_a_geometric_fallback():
    def unavailable(mesh):
        raise ModuleNotFoundError('missing closure dependency')
    with pytest.raises(ModuleNotFoundError):
        volume.estimate_mesh_volume(open_box(), close_holes=unavailable)


def test_descriptor_resume_does_not_silently_mix_legacy_volumes(tmp_path):
    import importlib.util
    source = Path(__file__).resolve().parents[2] / 'scripts' / 'extract_spine_descriptors.py'
    spec = importlib.util.spec_from_file_location('spine_extractor_resume', source)
    extractor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(extractor)
    incoming, outgoing = tmp_path / 'meshes', tmp_path / 'descriptors'
    incoming.mkdir(); outgoing.mkdir()
    (incoming / 'spine.off').write_text('placeholder')
    np.save(outgoing / 'spine.npy', np.array([123.0]))
    np.save(outgoing / 'spine_centroid.npy', np.zeros(3))
    jobs, skipped, total = extractor._gather_jobs(
        incoming, outgoing, '*.off', False, False, configuration={'data_type': 'microns'})
    assert len(jobs) == 1 and skipped == 0 and total == 1
    assert np.load(outgoing / 'spine.npy')[0] == 123.0
    jobs, skipped, total = extractor._gather_jobs(
        incoming, outgoing, '*.off', False, True, configuration={'data_type': 'microns'})
    assert len(jobs) == 1 and skipped == 0 and total == 1
