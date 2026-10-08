"""Geometry and partition invariants for opt-in spine refinement."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
import trimesh

path = Path(__file__).resolve().parents[2] / "neurd" / "spine_refinement.py"
spec = importlib.util.spec_from_file_location("refinement_under_test", path)
sr = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = sr
spec.loader.exec_module(sr)


def branch_surface(curved=False, bumps=(0,)):
    """Continuous shaft and protrusions, with known parent face membership."""
    z = np.arange(-6000, 6001, 50.0)
    angle = np.linspace(-np.pi, np.pi, 96, endpoint=False)
    centers = np.column_stack((0.00007 * z ** 2 if curved else z * 0, z * 0, z))
    tangent = np.column_stack((0.00014 * z if curved else z * 0, z * 0, np.ones(len(z))))
    tangent /= np.linalg.norm(tangent, axis=1)[:, None]
    u = np.column_stack((np.ones(len(z)), z * 0, -tangent[:, 0] / tangent[:, 2]))
    u /= np.linalg.norm(u, axis=1)[:, None]
    v = np.cross(tangent, u)
    radius = np.full((len(z), len(angle)), 300.0)
    for position in bumps:
        radius += 1400 * np.exp(-((z[:, None] - position) / 200) ** 4
                                - (angle[None, :] / 0.35) ** 4)
    vertices = (centers[:, None] + radius[:, :, None]
                * (np.cos(angle)[None, :, None] * u[:, None]
                   + np.sin(angle)[None, :, None] * v[:, None])).reshape(-1, 3)
    faces = []
    for j in range(len(z) - 1):
        for k in range(len(angle)):
            a, b = j * len(angle) + k, j * len(angle) + (k + 1) % len(angle)
            faces.extend(([a, b, a + len(angle)], [b, b + len(angle), a + len(angle)]))
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    skeleton = np.stack((centers[:-1], centers[1:]), axis=1)
    vertex_radius = radius.ravel()
    face_radius = vertex_radius[mesh.faces].min(axis=1)
    return mesh, skeleton, face_radius


def test_surface_scrap_removed_but_input_unchanged():
    mesh, skeleton, _ = branch_surface(bumps=())
    points = mesh.triangles_center
    patch = np.flatnonzero((abs(points[:, 2]) < 1500) & (points[:, 0] > 290))
    before_vertices, before_faces = mesh.vertices.copy(), mesh.faces.copy()
    result, diagnostics = sr.refine_spine_face_indices(mesh, skeleton, [patch])
    assert result == []
    assert diagnostics["counts"] == {"remove_surface_patch": 1}
    assert np.array_equal(mesh.vertices, before_vertices)
    assert np.array_equal(mesh.faces, before_faces)


@pytest.mark.parametrize("curved", [False, True])
def test_truncated_head_grows_on_original_surface(curved):
    mesh, skeleton, radius = branch_surface(curved=curved)
    head = np.flatnonzero(radius > 1100)
    result, diagnostics = sr.refine_spine_face_indices(mesh, skeleton, [head])
    record = diagnostics["candidates"][0]
    assert record["action"] == "grow", record
    assert len(result[0]) > len(head)
    assert np.isin(head, result[0]).all()
    assert np.unique(result[0]).size == len(result[0])
    assert record["geometry_proposed"]["gap_q10_nm"] <= 200
    assert radius[result[0]].min() > 300


def test_intact_short_protrusion_kept():
    mesh, skeleton, radius = branch_surface()
    intact = np.flatnonzero(radius > 340)
    result, diagnostics = sr.refine_spine_face_indices(mesh, skeleton, [intact])
    assert diagnostics["candidates"][0]["action"] in ("keep", "screened_keep")
    assert np.array_equal(result[0], intact)


def test_neighbour_seeds_and_prior_additions_cannot_be_stolen():
    mesh, skeleton, radius = branch_surface(curved=True, bumps=(-850, 850))
    z = mesh.triangles_center[:, 2]
    seeds = [np.flatnonzero((radius > 1000) & (z < 0)),
             np.flatnonzero((radius > 1000) & (z > 0))]
    results, diagnostics = sr.refine_spine_face_indices(mesh, skeleton, seeds)
    assert len(results) == 2
    assert not np.intersect1d(*results).size
    for old, new in zip(seeds, results):
        assert np.isin(old, new).all()
    assert any(r["action"] == "grow" for r in diagnostics["candidates"])


def test_unknown_context_keeps_original():
    mesh, _, radius = branch_surface()
    head = np.flatnonzero(radius > 1100)
    result, diagnostics = sr.refine_spine_face_indices(mesh, np.empty((0, 2, 3)), [head])
    assert np.array_equal(result[0], head)
    assert diagnostics["candidates"][0]["reason"] == "empty_skeleton"


def test_growth_limit_keeps_original():
    mesh, skeleton, radius = branch_surface()
    head = np.flatnonzero(radius > 1100)
    result, diagnostics = sr.refine_spine_face_indices(
        mesh, skeleton, [head], config={"maximum_added_faces": 0})
    assert np.array_equal(result[0], head)
    assert diagnostics["candidates"][0]["reason"] == "growth_face_limit"


def test_duplicate_parent_faces_do_not_duplicate_growth():
    mesh, skeleton, radius = branch_surface()
    head = np.flatnonzero(radius > 1100)
    duplicate = np.flatnonzero((radius > 400) & (radius < 900))[:5]
    parent = trimesh.Trimesh(vertices=mesh.vertices, faces=np.vstack((mesh.faces, mesh.faces[duplicate])), process=False)
    result, diagnostics = sr.refine_spine_face_indices(parent, skeleton, [head])
    assert diagnostics["candidates"][0]["action"] == "grow"
    triangles = np.sort(parent.faces[result[0]], axis=1)
    assert len(np.unique(triangles, axis=0)) == len(triangles)


def test_empty_candidates_do_not_require_skeleton():
    result, diagnostics = sr.refine_spine_face_indices(trimesh.Trimesh(), [], [])
    assert result == [] and diagnostics["candidates"] == []


def test_invalid_face_indices_are_an_api_error():
    with pytest.raises(ValueError, match="outside parent"):
        sr.refine_spine_face_indices(trimesh.creation.box(), [], [[1000]])


def test_polygon_distance_translation_and_inside_sign():
    polygon = np.array([[-2, -1], [2, -1], [2, 1], [-2, 1]], dtype=float)
    points = np.array([[0, 0], [3, 0], [0, 2], [2, 0]], dtype=float)
    expected = np.array([-1, 1, 1, 0])
    assert np.allclose(sr._polygon_distance(points, polygon), expected)
    assert np.allclose(sr._polygon_distance(points + 1e6, polygon + 1e6), expected)


@pytest.mark.parametrize("vertices", [31, 451])
def test_indexed_contour_distance_matches_all_segments(vertices):
    rng = np.random.default_rng(80)
    angle = np.linspace(0, 2 * np.pi, vertices, endpoint=False)
    radius = rng.uniform(300, 1100, len(angle))
    polygon = np.column_stack((np.cos(angle), np.sin(angle))) * radius[:, None]
    points = rng.uniform(-5000, 5000, (300, 2))
    edges = np.roll(polygon, -1, axis=0) - polygon
    delta = points[:, None] - polygon
    fractions = np.clip(np.einsum("nkj,kj->nk", delta, edges)
                        / np.sum(edges * edges, axis=1), 0, 1)
    expected = np.linalg.norm(delta - fractions[:, :, None] * edges, axis=2).min(axis=1)
    assert np.allclose(abs(sr._polygon_distance(points, polygon)), expected, atol=1e-9)


def test_upstream_rejected_neighbour_remains_protected():
    mesh, skeleton, radius = branch_surface(bumps=(-850, 850))
    z = mesh.triangles_center[:, 2]
    head = np.flatnonzero((radius > 1000) & (z < 0))
    neighbour = np.flatnonzero((radius > 500) & (z > 0))
    results, _ = sr.refine_spine_face_indices(mesh, skeleton, [head], protected_face_groups=[head, neighbour])
    assert np.isin(head, results[0]).all()
    assert not np.intersect1d(results[0], neighbour).size


def test_filter_only_mode_does_not_build_head_profiles():
    mesh, skeleton, radius = branch_surface()
    head = np.flatnonzero(radius > 1100)
    result, diagnostics = sr.refine_spine_face_indices(
        mesh, skeleton, [head], config={"maximum_repairs_per_branch": 0})
    assert np.array_equal(result[0], head)
    assert diagnostics["statistics"]["section_models"] == 0
    assert diagnostics["statistics"]["repairs_attempted"] == 0
