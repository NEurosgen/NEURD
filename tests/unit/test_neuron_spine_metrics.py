"""Geometry checks used by the optional spine-enabled whole-neuron gate."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import trimesh

spec = importlib.util.spec_from_file_location(
    "neuron_metrics_under_test", Path(__file__).resolve().parents[1] / "tools/neuron_metrics.py")
metrics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metrics)


def report(parent, spines, volumes=None, masks=None):
    branch = SimpleNamespace(mesh=parent, spines=spines, spines_volume=volumes,
                             spines_face_idx=masks)
    return metrics.extract_spine_metrics(SimpleNamespace(
        limbs=[SimpleNamespace(branches=[branch])]))


def test_disjoint_submeshes_retain_exact_parent_face_provenance():
    parent = trimesh.creation.box()
    first = parent.submesh([[0, 3, 5]], append=True)
    second = parent.submesh([[7, 9]], append=True)
    a = report(parent, [first, second], [1, 2])
    first.faces = first.faces[::-1, ::-1]
    b = report(parent, [first, second], [1, 2])
    assert a == b
    assert a["n_spines_total"] == 2
    assert a["spine_mesh_faces_total"] == 5
    assert all(a[k] == 0 for k in a if k.endswith("faces_total") and k != "spine_mesh_faces_total")
    assert a["invalid_spine_volumes_total"] == 0


def test_changed_geometry_cannot_pass_as_a_parent_submesh():
    parent = trimesh.creation.box()
    spine = parent.submesh([[0, 1]], append=True)
    spine.vertices += [0.25, 0, 0]
    result = report(parent, [spine])
    assert result["invalid_parent_faces_total"] == 2


def test_overlapping_spines_are_reported():
    parent = trimesh.creation.box()
    result = report(parent, [parent.submesh([[0, 1]], append=True),
                             parent.submesh([[1, 2]], append=True)])
    assert result["overlapping_parent_faces_total"] == 1


def test_duplicate_parent_triangles_are_ambiguous():
    parent = trimesh.creation.box()
    parent.faces = np.concatenate([parent.faces, parent.faces[:1]])
    result = report(parent, [parent.submesh([[0]], append=True)])
    assert result["ambiguous_parent_faces_total"] == 1


def test_volume_alignment_and_nonfinite_values_are_reported():
    parent = trimesh.creation.box()
    spine = parent.submesh([[0]], append=True)
    assert report(parent, [spine], None)['invalid_spine_volumes_total'] == 1
    assert report(parent, [spine], [])['invalid_spine_volumes_total'] == 1
    assert report(parent, [spine], [float('nan')])['invalid_spine_volumes_total'] == 1
    assert report(parent, [spine], [0])['invalid_spine_volumes_total'] == 1


def test_explicit_indices_resolve_duplicate_triangle_occurrences():
    parent = trimesh.creation.box()
    parent.faces = np.concatenate([parent.faces, parent.faces[:1]])
    spine = parent.submesh([[0, 12]], append=True)
    result = report(parent, [spine], [1], masks=[[0, 12]])
    assert result['spines_with_face_indices'] == 1
    assert result['ambiguous_parent_faces_total'] == 0
    assert result['overlapping_parent_faces_total'] == 0
    assert result['invalid_parent_faces_total'] == 0
    assert result['overlapping_spine_surface_faces_total'] == 0


def test_distinct_parent_indices_do_not_hide_spine_surface_overlap():
    parent = trimesh.creation.box()
    parent.faces = np.concatenate([parent.faces, parent.faces[:1]])
    result = report(parent, [parent.submesh([[0]], append=True),
                             parent.submesh([[12]], append=True)], masks=[[0], [12]])
    assert result['overlapping_parent_faces_total'] == 0
    assert result['overlapping_spine_surface_faces_total'] == 1


def test_explicit_mask_must_match_mesh_geometry_and_face_count():
    parent = trimesh.creation.box()
    spine = parent.submesh([[0]], append=True)
    assert report(parent, [spine], masks=[[1]])['invalid_parent_faces_total'] == 1
    assert report(parent, [spine], masks=[[-1]])['invalid_spine_face_indices_total'] == 1
    assert report(parent, [spine], masks=[[]])['invalid_spine_face_indices_total'] == 1
