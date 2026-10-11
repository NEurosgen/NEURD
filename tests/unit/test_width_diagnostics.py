"""Support samples remain bounded and cannot establish branch eligibility."""
import json

import numpy as np
import pytest
import trimesh

from neurd import width_diagnostics as diagnostics, width_utils, skeletal_distance_ops as sdo


def _axis(count=1, offset=0):
    points = np.column_stack((np.arange(count + 1) * 1000,
                              np.full(count + 1, offset), np.zeros(count + 1)))
    return np.stack((points[:-1], points[1:]), axis=1)


def test_empty_slice_support_is_zero_and_does_not_build_adjacency(monkeypatch):
    mesh = trimesh.creation.icosphere(subdivisions=1, radius=100)
    edges = _axis(100, offset=10000)
    monkeypatch.setattr(width_utils, 'skeleton_resized_ordered', lambda *args, **kwargs: edges)
    monkeypatch.setattr(sdo, 'skeletal_distance',
                        lambda *args, **kwargs: pytest.fail('unsupported slices invoked kernel'))
    result = diagnostics.sample_width_support(mesh, edges)
    assert result['total_resized_edges'] == 100
    assert result['sampled_edges'] == 16
    assert not result['full_edge_coverage']
    assert [o['resized_edge_index'] for o in result['slices']] == np.linspace(0, 99, 16, dtype=int).tolist()
    assert result['sampled_empty_slice_count'] == 16
    assert result['sampled_positive_raw_distance_count'] == 0
    assert all(o['within_distance_candidate_faces'] == 0 and o['raw_distance_nm'] == 0 for o in result['slices'])
    assert result['diagnostic_only'] and not result['active_raw_distances_observed']
    json.dumps(result, allow_nan=False)


def test_supported_sample_matches_owned_unfilled_distance(monkeypatch):
    mesh = trimesh.creation.cylinder(radius=200, height=1000, sections=32)
    edges = np.array([[[0, 0, -500], [0, 0, 500]]], dtype=float)
    monkeypatch.setattr(width_utils, 'skeleton_resized_ordered', lambda *args, **kwargs: edges)
    expected, _, _, _ = sdo.skeletal_distance(mesh, edges, buffer=0.01,
        distance_threshold=3000, distance_by_mesh_center=True,
        connectivity='edges', keep_empty_placeholder=True)
    result = diagnostics.sample_width_support(mesh, edges)
    assert result['full_edge_coverage'] and result['sampled_edges'] == 1
    assert result['sampled_positive_raw_distance_count'] == 1
    assert result['slices'][0]['raw_distance_nm'] == expected[0]
    assert result['slices'][0]['within_distance_candidate_faces'] > 0
    assert 'eligible' not in result


def test_support_counts_survive_distance_rerun_failure(monkeypatch):
    mesh = trimesh.creation.cylinder(radius=200, height=1000, sections=32)
    edges = np.array([[[0, 0, -500], [0, 0, 500]]], dtype=float)
    monkeypatch.setattr(width_utils, 'skeleton_resized_ordered', lambda *args, **kwargs: edges)
    def broken(*args, **kwargs):
        raise RuntimeError('diagnostic kernel failed')
    monkeypatch.setattr(sdo, 'skeletal_distance', broken)
    result = diagnostics.sample_width_support(mesh, edges)
    assert result['status'] == 'partial'
    assert result['sampled_supported_slice_count'] == 1
    assert result['slices'][0]['within_distance_candidate_faces'] > 0
    assert result['slices'][0]['raw_distance_nm'] is None
    assert 'diagnostic kernel failed' in result['raw_distance_error']
    json.dumps(result, allow_nan=False)


def test_face_budget_precedes_axis_or_surface_work(monkeypatch):
    mesh = trimesh.creation.box()
    monkeypatch.setattr(width_utils, 'skeleton_resized_ordered',
                        lambda *args, **kwargs: pytest.fail('face budget not respected'))
    result = diagnostics.sample_width_support(mesh, _axis(), max_faces=1)
    assert result['status'] == 'not_sampled'
    assert result['reason'] == 'diagnostic_face_budget_exceeded'


def test_nonfinite_diagnostic_distances_are_json_safe(monkeypatch):
    mesh = trimesh.creation.cylinder(radius=200, height=1000, sections=16)
    edges = np.array([[[0, 0, -500], [0, 0, 500]]], dtype=float)
    monkeypatch.setattr(width_utils, 'skeleton_resized_ordered', lambda *args, **kwargs: edges)
    monkeypatch.setattr(sdo, 'skeletal_distance', lambda *args, **kwargs: ([float('nan')], [], None, []))
    result = diagnostics.sample_width_support(mesh, edges)
    assert result['sampled_nonfinite_raw_distance_count'] == 1
    assert result['slices'][0]['raw_distance_nm'] is None
    json.dumps(result, allow_nan=False)
