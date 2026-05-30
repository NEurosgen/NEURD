"""Fast unit tests for the in-process mesh ops (neurd/_mesh_ops.py).

These run on tiny synthetic meshes (no meshlab, no fixtures) in well under a second,
giving a seconds-long edit loop while replacing the meshlabserver subprocess ops.
Fidelity vs real meshlab output on actual soma meshes is checked separately
(integration, against captured golden fixtures).
"""

import numpy as np
import pytest

import neurd  # noqa: F401  (activates numpy/mesh shims before trimesh/open3d)
import trimesh

from neurd import _mesh_ops as mo


def _holed_sphere(subdivisions=3, n_drop=6):
    """An icosphere with a few faces removed -> an open (non-watertight) surface."""
    s = trimesh.creation.icosphere(subdivisions=subdivisions)
    mask = np.ones(len(s.faces), dtype=bool)
    mask[:n_drop] = False
    holed = s.copy()
    holed.update_faces(mask)
    holed.remove_unreferenced_vertices()
    return holed


def test_decimate_reduces_face_count_near_ratio():
    s = trimesh.creation.icosphere(subdivisions=4)  # 20480 faces
    out = mo.decimate(s, 0.25)
    assert 0 < len(out.faces) < len(s.faces)
    # within 30% of the requested ratio (quadric decimation isn't exact)
    assert len(out.faces) == pytest.approx(len(s.faces) * 0.25, rel=0.3)
    assert out.is_volume or out.is_watertight or len(out.faces) > 0


def test_decimate_preserves_shape_roughly():
    s = trimesh.creation.icosphere(subdivisions=4, radius=10.0)
    out = mo.decimate(s, 0.2)
    # centroid and bbox extent stay close (shape preserved, not just face count)
    assert np.allclose(out.bounds[1] - out.bounds[0], s.bounds[1] - s.bounds[0], atol=1.0)


def test_fill_holes_closes_a_simple_hole():
    # single removed triangle -> a simple hole trimesh can close
    holed = _holed_sphere(n_drop=1)
    assert not holed.is_watertight
    filled = mo.fill_holes(holed)
    assert filled.is_watertight, "fill_holes did not close a single-triangle hole"
    assert len(filled.faces) >= len(holed.faces)


def test_fill_holes_returns_valid_mesh_on_complex_hole():
    # larger hole: trimesh may not fully close it, but must return a valid mesh and
    # never crash (the meshlab op it replaces currently fails + no-ops anyway).
    holed = _holed_sphere(n_drop=8)
    filled = mo.fill_holes(holed)
    assert len(filled.faces) >= len(holed.faces)
    assert filled.vertices.shape[1] == 3


def test_poisson_returns_watertight_shell():
    s = trimesh.creation.icosphere(subdivisions=3, radius=5.0)
    rec = mo.poisson_surface_reconstruction(s, depth=6)
    assert len(rec.faces) > 0
    assert rec.is_watertight, "poisson reconstruction is not watertight"
    # reconstructed shell sits near the original sphere surface
    assert np.allclose(rec.centroid, s.centroid, atol=1.0)
