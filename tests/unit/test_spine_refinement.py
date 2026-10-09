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


def branch_surface(curved=False, bumps=(0,), bump_width=200):
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
        radius += 1400 * np.exp(-((z[:, None] - position) / bump_width) ** 4
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


def test_low_relief_filter_handles_missing_closed_sections(monkeypatch):
    mesh, skeleton, _ = branch_surface(bumps=())
    points = mesh.triangles_center
    patch = np.flatnonzero((abs(points[:, 2]) < 1500) & (points[:, 0] > 290))
    before = mesh.vertices.copy(), mesh.faces.copy(), patch.copy()

    def missing_sections(*args, **kwargs):
        raise ValueError('insufficient_closed_shaft_sections')

    monkeypatch.setattr(sr._Context, 'model', missing_sections)
    baseline, _ = sr.refine_spine_face_indices(mesh, skeleton, [patch])
    assert len(baseline) == 1 and np.array_equal(baseline[0], patch)
    result, diagnostics = sr.refine_spine_face_indices(
        mesh, skeleton, [patch], {'filter_low_relief_sheets': True})
    assert result == []
    decision = diagnostics['candidates'][0]
    assert decision['action'] == 'remove_low_relief_sheet'
    assert decision['surface_shape']['direction_agreement'] >= .9
    assert len(decision['surface_shape']['local_estimates']) == 2
    assert np.array_equal(mesh.vertices, before[0])
    assert np.array_equal(mesh.faces, before[1]) and np.array_equal(patch, before[2])


@pytest.mark.parametrize('axis_shift', [0, 100_000])
def test_low_relief_filter_preserves_complete_spine_without_sections(monkeypatch, axis_shift):
    mesh, skeleton, radius = branch_surface()
    intact = np.flatnonzero(radius > 340)

    def missing_sections(*args, **kwargs):
        raise ValueError('insufficient_closed_shaft_sections')

    monkeypatch.setattr(sr._Context, 'model', missing_sections)
    result, diagnostics = sr.refine_spine_face_indices(
        mesh, skeleton+axis_shift, [intact], {'filter_low_relief_sheets': True})
    assert len(result) == 1 and np.array_equal(result[0], intact), diagnostics


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


def branched_rim(mesh, indices):
    """Remove one triangle whose hole touches the cut rim at one vertex."""
    faces = mesh.faces[indices]
    edges = np.sort(np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1)
    edges, counts = np.unique(edges, axis=0, return_counts=True)
    vertices = np.unique(edges[counts == 1])
    candidates = np.flatnonzero(np.isin(faces, vertices).sum(axis=1) == 1)
    assert len(candidates)
    return np.delete(indices, candidates[len(candidates) // 2])


@pytest.mark.parametrize("ragged", ["branched", "duplicate"])
def test_ragged_shaft_patch_is_rejected_without_repairing_input(ragged):
    mesh, skeleton, _ = branch_surface(bumps=())
    points = mesh.triangles_center
    patch = np.flatnonzero((abs(points[:, 2]) < 1500) & (points[:, 0] > 290))
    if ragged == "branched":
        patch = branched_rim(mesh, patch)
        expected = "branched_open_boundary"
    else:
        duplicate = mesh.faces[patch[len(patch) // 2]]
        mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=np.vstack((mesh.faces, duplicate)), process=False)
        patch = np.r_[patch, len(mesh.faces) - 1]
        expected = "nonmanifold_candidate"
    before = mesh.faces.copy()
    with pytest.raises(ValueError, match=expected):
        sr._boundary(mesh, patch)
    result, diagnostics = sr.refine_spine_face_indices(mesh, skeleton, [patch])
    record = diagnostics["candidates"][0]
    assert result == []
    assert record["action"] == "remove_surface_patch"
    assert record["surface_boundary_status"] == expected
    assert np.array_equal(mesh.faces, before)
    original_mode, old = sr.refine_spine_face_indices(
        mesh, skeleton, [patch], config={"filter_ragged_surface_patches": False})
    assert np.array_equal(original_mode[0], patch)
    assert old["candidates"][0]["reason"] == expected


def test_ragged_true_protrusion_kept_and_not_grown():
    mesh, skeleton, radius = branch_surface()
    head = branched_rim(mesh, np.flatnonzero(radius > 1100))
    result, diagnostics = sr.refine_spine_face_indices(mesh, skeleton, [head])
    record = diagnostics["candidates"][0]
    assert np.array_equal(result[0], head)
    assert record["action"] == "unknown_keep"
    assert record["reason"] == "branched_open_boundary"
    assert record["geometry_before"]["surface_q95_nm"] > 100
    assert diagnostics["statistics"]["repairs_attempted"] == 0


def test_opt_in_ragged_head_completion_preserves_seed_and_parent():
    mesh, skeleton, radius = branch_surface()
    head = branched_rim(mesh, np.flatnonzero(radius > 1100))
    original = mesh.faces.copy()
    fallback = {}
    result, diagnostics = sr.refine_spine_face_indices(
        mesh, skeleton, [head], config={"grow_ragged_heads": True,
            "ragged_minimum_added_clearance_nm": 100},
        pre_growth_face_groups=fallback)
    record = diagnostics["candidates"][0]
    assert record["action"] == "grow_ragged_head", record
    assert len(result[0]) > len(head)
    assert np.isin(head, result[0]).all()
    assert record["geometry_proposed"]["gap_q10_nm"] <= 200
    assert radius[result[0]].min() > 300
    assert np.array_equal(fallback[0], head)
    assert np.array_equal(mesh.faces, original)


def test_ragged_completion_failure_preserves_original_seed():
    mesh, skeleton, radius = branch_surface()
    head = branched_rim(mesh, np.flatnonzero(radius > 1100))
    fallback = {}
    result, diagnostics = sr.refine_spine_face_indices(
        mesh, skeleton, [head], config={"grow_ragged_heads": True, "maximum_added_faces": 0},
        pre_growth_face_groups=fallback)
    assert np.array_equal(result[0], head)
    assert diagnostics["candidates"][0]["reason"] == "growth_face_limit"
    assert fallback == {}


def test_ragged_completion_does_not_expand_a_complete_short_spine():
    mesh, skeleton, radius = branch_surface()
    intact = branched_rim(mesh, np.flatnonzero(radius > 340))
    result, _ = sr.refine_spine_face_indices(
        mesh, skeleton, [intact], config={"grow_ragged_heads": True})
    assert np.array_equal(result[0], intact)


def test_low_clearance_additions_keep_the_original_head():
    mesh, skeleton, radius = branch_surface()
    head = branched_rim(mesh, np.flatnonzero(radius > 1100))
    fallback = {}
    result, diagnostics = sr.refine_spine_face_indices(
        mesh, skeleton, [head], config={"grow_ragged_heads": True,
            "ragged_minimum_added_clearance_nm": 5000}, pre_growth_face_groups=fallback)
    assert np.array_equal(result[0], head)
    assert diagnostics["candidates"][0]["reason"] == "ragged_growth_mostly_near_shaft"
    assert fallback == {}


def test_ragged_attempt_does_not_spend_the_existing_cap_repair_budget():
    mesh, skeleton, radius = branch_surface(bumps=(-1600, 1600))
    z = mesh.triangles_center[:, 2]
    ragged = branched_rim(mesh, np.flatnonzero((radius > 1100) & (z < 0)))
    cap = np.flatnonzero((radius > 1100) & (z > 0))
    baseline, _ = sr.refine_spine_face_indices(
        mesh, skeleton, [ragged, cap], config={"maximum_repairs_per_branch": 1})
    result, diagnostics = sr.refine_spine_face_indices(
        mesh, skeleton, [ragged, cap], config={
            "grow_ragged_heads": True, "maximum_repairs_per_branch": 1,
            "maximum_ragged_repairs_per_branch": 1})
    assert diagnostics["candidates"][0]["reason"] == "ragged_growth_mostly_near_shaft", diagnostics
    assert diagnostics["candidates"][1]["action"] == "grow", diagnostics
    assert diagnostics["statistics"]["repairs_attempted"] == 1
    assert diagnostics["statistics"]["ragged_repairs_attempted"] == 1
    assert not np.intersect1d(*result).size
    assert np.array_equal(result[1], baseline[1])
    assert diagnostics["candidates"][0]["output_index"] == 0
    assert diagnostics["candidates"][1]["output_index"] == 1


def test_ragged_short_protrusion_is_not_a_sheet_by_boundary_alone():
    mesh, skeleton, radius = branch_surface()
    intact = branched_rim(mesh, np.flatnonzero(radius > 340))
    result, diagnostics = sr.refine_spine_face_indices(mesh, skeleton, [intact])
    assert np.array_equal(result[0], intact)
    assert diagnostics["candidates"][0]["action"] == "unknown_keep"


def test_ragged_head_kept_when_sections_also_include_the_protrusion():
    mesh, skeleton, radius = branch_surface(bump_width=400)
    # A perforated cap has a long cut rim despite lying on a real protrusion.
    cap = np.flatnonzero((radius > 1100) & (abs(mesh.triangles_center[:, 2]) < 200))[::2]
    aggressive, without_guard = sr.refine_spine_face_indices(
        mesh, skeleton, [cap], config={"maximum_section_surface_spread_nm": 1e9})
    assert aggressive == []
    assert without_guard["candidates"][0]["action"] == "remove_surface_patch"
    result, diagnostics = sr.refine_spine_face_indices(mesh, skeleton, [cap])
    record = diagnostics["candidates"][0]
    assert np.array_equal(result[0], cap)
    assert record["reason"] == "unstable_shaft_surface_estimate"
    assert diagnostics["statistics"]["repairs_attempted"] == 0


def test_shaft_bridge_splits_into_distinct_protrusions_on_parent_faces():
    mesh, skeleton, radius = branch_surface(bumps=(-850,850))
    points = mesh.triangles_center
    # Two actual protrusions joined only by a strip of the shaft surface.
    merged = np.flatnonzero((radius > 340) | ((abs(points[:,2]) < 1600) & (points[:,0] > 100)))
    original, _ = sr.refine_spine_face_indices(mesh,skeleton,[merged])
    assert len(original)==1 and np.array_equal(original[0],merged)
    before = mesh.vertices.copy(), mesh.faces.copy()
    refined, d = sr.refine_spine_face_indices(mesh,skeleton,[merged],config={'trim_shaft_connections':True})
    assert len(refined)==2,d
    assert d['candidates'][0]['action']=='split_shaft_connection'
    assert d['candidates'][0]['output_indices']==[0,1]
    assert not np.intersect1d(*refined).size
    assert all(np.isin(part,merged).all() for part in refined)
    assert np.isin(np.flatnonzero(radius>1100),np.concatenate(refined)).all()
    assert np.array_equal(mesh.vertices,before[0]) and np.array_equal(mesh.faces,before[1])


def test_intact_short_spine_is_preserved_with_shaft_trimming():
    mesh,skeleton,radius=branch_surface()
    intact=np.flatnonzero(radius>340)
    refined,d=sr.refine_spine_face_indices(mesh,skeleton,[intact],config={'trim_shaft_connections':True})
    assert len(refined)==1 and np.array_equal(refined[0],intact),d


def test_trim_requires_attachment_coverage_and_handles_nonmanifold_edges():
    mesh,skeleton,radius=branch_surface()
    indices=np.flatnonzero(radius>340)
    ctx=sr._Context(mesh,skeleton+100_000,[indices],sr.RefinementConfig(trim_shaft_connections=True,trim_allow_mesh_direction=False))
    with pytest.raises(ValueError,match='axis_does_not_cover'):
        ctx.trim(indices)
    # Three triangles meeting at an edge remain one physical component.
    pairs=sr._edge_pairs(np.array([[0,1,2],[1,0,3],[0,1,4]]))
    assert len(pairs)==2


def test_invalid_ragged_head_not_trimmed_by_an_inconsistent_contour():
    mesh,skeleton,radius=branch_surface(bump_width=400)
    cap=np.flatnonzero((radius>1100)&(abs(mesh.triangles_center[:,2])<200))[::2]
    refined,d=sr.refine_spine_face_indices(mesh,skeleton,[cap],config={'trim_shaft_connections':True})
    assert len(refined)==1 and np.array_equal(refined[0],cap),d
    assert d['candidates'][0]['shaft_trim']['reason']=='shaft_trim_sections_disagree'


def test_missing_candidate_arcs_do_not_use_a_head_as_shaft_support():
    polygon=np.array([[0,0],[100,0],[200,1000],[300,0],[400,0],[400,-200],[0,-200]],float)
    support=np.array([True,True,False,True,True,True,True])
    result=sr._shaft_contour((polygon,1,support),maximum_bridge_nm=250)
    assert result is not None
    assert result[0][:,1].max()==0
    assert sr._shaft_contour((polygon,1,support),maximum_bridge_nm=150) is None


def test_local_mesh_sections_can_classify_a_patch_with_a_mismatched_axis():
    mesh,skeleton,_=branch_surface(bumps=())
    p=mesh.triangles_center
    patch=np.flatnonzero((abs(p[:,2])<1500)&(p[:,0]>290))
    ctx=sr._Context(mesh,skeleton+100_000,[patch],sr.RefinementConfig(trim_shaft_connections=True))
    result,d=ctx.trim(patch)
    assert result==[] and d['action']=='remove_surface_patch'
    assert d['model_direction']=='mesh'


def test_missing_arc_cannot_be_closed_across_another_shaft_edge():
    # A concave section with an external arc below a narrow indentation.
    # Removing that arc would make the replacement chord cross both bay walls.
    polygon=np.array([[0,0],[4,-4],[8,0],[8,8],[6,8],[6,-1],[2,-1],[2,8],[0,8]],float)*100
    supported=np.ones(len(polygon),bool);supported[1]=False
    assert sr._shaft_contour((polygon,1,supported),maximum_bridge_nm=1000) is None


def test_split_child_cannot_be_a_patch_elevated_by_a_biased_shaft_model():
    mesh = trimesh.creation.cylinder(radius=500, height=2400, sections=64)
    centers = mesh.triangles_center
    patch = np.flatnonzero((centers[:, 0] > 100) & (abs(centers[:, 2]) < 1000))
    skeleton = np.array([[[0, 0, -1200], [0, 0, 1200]]], dtype=float)
    ctx = sr._Context(mesh, skeleton, [patch], sr.RefinementConfig(trim_shaft_connections=True))
    angles = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    # A bad local estimate makes the entire real shaft patch appear >300 nm
    # above its surface. The patch still has no elevation above its own rim.
    polygon = 100 * np.column_stack((np.cos(angles), np.sin(angles)))
    model = (np.zeros(3), np.eye(3), [sr._Contour(polygon)] * 3)
    boundary, measured = sr._surface_boundary(mesh, patch)
    geometry, _ = ctx.assess(measured, boundary, model=model)
    assert geometry['surface_q95_nm'] > ctx.config.trim_minimum_protrusion_nm
    with pytest.raises(ValueError, match='part_has_no_distinct_protrusion'):
        ctx._validate_trim_part(patch, model)


def test_auxiliary_axis_completes_a_missing_head_and_preserves_primary_repair(monkeypatch):
    mesh, full_axis, radius = branch_surface(bumps=(-3000, 3000))
    z = mesh.triangles_center[:, 2]
    primary_cap = np.flatnonzero((radius > 1100) & (z < 0))
    missing_head = branched_rim(mesh, np.flatnonzero((radius > 1100) & (z > 0)))
    partial_axis = full_axis[full_axis.mean(axis=1)[:, 2] < -1000]
    config = dict(grow_ragged_heads=True, ragged_minimum_added_clearance_nm=100,
                  axis_coverage_distance_nm=1500)
    before, baseline = sr.refine_spine_face_indices(mesh, partial_axis, [primary_cap, missing_head], config)
    assert baseline['candidates'][0]['action'] == 'grow'
    assert baseline['candidates'][1]['reason'] == 'ragged_head_axis_does_not_cover_seed'
    built = []

    def build(parent, excluded):
        built.append(excluded.copy())
        assert parent is mesh and excluded[np.r_[primary_cap, missing_head]].all()
        return full_axis

    monkeypatch.setattr(sr, '_build_mesh_axis', build)
    fallback = {}
    after, diag = sr.refine_spine_face_indices(
        mesh, partial_axis, [primary_cap, missing_head], {**config, 'retry_with_mesh_axis': True},
        pre_growth_face_groups=fallback)
    assert len(built) == 1
    assert np.array_equal(after[0], before[0])  # Exact successful primary mask.
    assert len(after[1]) > len(missing_head) and np.isin(missing_head, after[1]).all()
    assert not np.intersect1d(*after).size
    assert np.array_equal(fallback[1], missing_head)
    assert diag['auxiliary_axis']['changed_candidates'] == 1
    assert diag['candidates'][1]['action'] == 'grow_ragged_head'
    assert diag['candidates'][1]['before_mesh_axis_retry']['reason'] == 'ragged_head_axis_does_not_cover_seed'


def test_covered_axis_and_default_mode_do_not_build_an_auxiliary_axis(monkeypatch):
    mesh, skeleton, radius = branch_surface()
    head = np.flatnonzero(radius > 1100)

    def unexpected(*args):
        raise AssertionError('Extra skeletonization must be lazy')

    monkeypatch.setattr(sr, '_build_mesh_axis', unexpected)
    # A forced growth failure gives the coverage guard an eligible candidate.
    config = dict(maximum_added_faces=0, retry_with_mesh_axis=True)
    after, diag = sr.refine_spine_face_indices(mesh, skeleton, [head], config)
    assert np.array_equal(after[0], head)
    assert diag['auxiliary_axis']['status'] == 'covered'
    sr.refine_spine_face_indices(mesh, skeleton + 100_000, [head])


@pytest.mark.parametrize('failure', ['exception', 'bad_coverage'])
def test_failed_auxiliary_axis_preserves_primary_masks(monkeypatch, failure):
    mesh, skeleton, radius = branch_surface()
    head = np.flatnonzero(radius > 1100)
    wrong_axis = skeleton + 100_000
    before, _ = sr.refine_spine_face_indices(mesh, wrong_axis, [head])

    def build(*args):
        if failure == 'exception':
            raise RuntimeError('Unavailable backend')
        return wrong_axis

    monkeypatch.setattr(sr, '_build_mesh_axis', build)
    after, diag = sr.refine_spine_face_indices(mesh, wrong_axis, [head], {'retry_with_mesh_axis': True})
    assert np.array_equal(after[0], before[0])
    assert diag['auxiliary_axis']['status'] == ('build_failed' if failure == 'exception' else 'rejected')


def test_auxiliary_retry_protects_upstream_rejected_neighbour(monkeypatch):
    mesh, skeleton, radius = branch_surface()
    head = branched_rim(mesh, np.flatnonzero(radius > 1100))
    neighbour = np.flatnonzero((radius > 340) & (radius <= 1100))
    monkeypatch.setattr(sr, '_build_mesh_axis', lambda *args: skeleton)
    after, diag = sr.refine_spine_face_indices(
        mesh, skeleton + 100_000, [head],
        dict(retry_with_mesh_axis=True, grow_ragged_heads=True, ragged_minimum_added_clearance_nm=100),
        protected_face_groups=[neighbour])
    assert not np.intersect1d(after[0], neighbour).size
    assert diag['auxiliary_axis']['status'] == 'retried'


def test_auxiliary_axis_cannot_delete_a_head_using_conflicting_sections(monkeypatch):
    mesh, skeleton, radius = branch_surface(bump_width=400)
    cap = np.flatnonzero((radius > 1100) & (abs(mesh.triangles_center[:, 2]) < 200))[::2]
    monkeypatch.setattr(sr, '_build_mesh_axis', lambda *args: skeleton)
    result, diag = sr.refine_spine_face_indices(mesh, skeleton + 100_000, [cap],
        dict(retry_with_mesh_axis=True, maximum_section_surface_spread_nm=1e9))
    assert len(result) == 1 and np.array_equal(result[0], cap)
    assert diag['candidates'][0]['mesh_axis_retry']['reason'] == 'auxiliary_axis_surface_estimate_uncertain'


@pytest.mark.parametrize('minimum_clearance', [100, 5000])
def test_cap_growth_requires_distal_additions_when_requested(minimum_clearance):
    mesh, skeleton, radius = branch_surface()
    head = np.flatnonzero(radius > 1100)
    result, diag = sr.refine_spine_face_indices(mesh, skeleton, [head],
        dict(require_distal_growth=True, ragged_minimum_added_clearance_nm=minimum_clearance))
    record = diag['candidates'][0]
    if minimum_clearance == 100:
        assert record['action'] == 'grow'
        assert len(result[0]) > len(head) and np.isin(head, result[0]).all()
        assert record['geometry_proposed']['added_surface_q25_nm'] > minimum_clearance
    else:
        assert record['reason'] == 'cap_growth_mostly_near_shaft'
        assert np.array_equal(result[0], head)


def test_head_completeness_detects_an_attached_half_head():
    mesh, skeleton, radius = branch_surface()
    full = np.flatnonzero(radius > 340)
    # Keep its entire proximal root, but remove one distal side of the head.
    cut = (radius[full] > 1100) & (mesh.triangles_center[full, 1] > 0)
    partial = full[~cut]
    context = sr._Context(mesh,skeleton,[partial],sr.RefinementConfig())
    angle = np.linspace(-np.pi,np.pi,96,endpoint=False)
    contour = sr._Contour(300*np.column_stack((np.cos(angle),np.sin(angle))))
    model = (np.zeros(3),np.eye(3),[contour,contour,contour])
    boundary, measured = sr._surface_boundary(mesh,partial)
    quality = context.head_boundary_quality(measured,boundary,model)
    assert quality['gap_q10_nm'] < 200
    assert quality['incomplete'] and quality['distal_cut_ratio'] > 1
    intact_boundary, intact = sr._surface_boundary(mesh,full)
    assert not context.head_boundary_quality(intact,intact_boundary,model)['incomplete']


def test_rejected_raw_fragments_can_be_recovered_but_accepted_neighbours_stay_reserved():
    mesh, skeleton, radius = branch_surface()
    cap = np.flatnonzero(radius > 1100)
    neighbour = np.setdiff1d(np.flatnonzero(radius > 340),cap)
    original = mesh.vertices.copy(),mesh.faces.copy(),cap.copy(),neighbour.copy()
    config = dict(refine_remaining_heads=True,ragged_minimum_added_clearance_nm=100)
    recovered, diag = sr.refine_remaining_spine_heads(mesh,skeleton,[cap],[cap,neighbour],[0],config)
    assert diag['candidates'][0]['action'] == 'grow_remaining_head'
    assert np.intersect1d(recovered[0],neighbour).size
    reserved, _ = sr.refine_remaining_spine_heads(mesh,skeleton,[cap,neighbour],[cap,neighbour],[0,1],config)
    assert np.array_equal(reserved[0],cap) and np.array_equal(reserved[1],neighbour)
    assert not np.intersect1d(*reserved).size
    assert np.array_equal(mesh.vertices,original[0]) and np.array_equal(mesh.faces,original[1])
    assert np.array_equal(cap,original[2]) and np.array_equal(neighbour,original[3])


def test_remaining_head_filter_failure_preserves_the_accepted_seed():
    mesh,skeleton,radius = branch_surface()
    cap = np.flatnonzero(radius > 1100)
    calls = []
    def reject(index,proposal):
        calls.append((index,proposal.copy()))
        return False
    result,diag = sr.refine_remaining_spine_heads(mesh,skeleton,[cap],[cap],[0],
        dict(refine_remaining_heads=True,filter_incomplete_heads=True,
             ragged_minimum_added_clearance_nm=100),validate_proposal=reject)
    assert calls and all(len(g) > len(cap) for _,g in calls)
    assert np.array_equal(result[0],cap)
    assert diag['candidates'][0]['reason'] == 'remaining_head_rejected_by_final_filters'


def test_smooth_head_screen_recovers_a_cap_but_preserves_a_complete_small_root():
    mesh,skeleton,radius = branch_surface()
    cap = np.flatnonzero(radius > 1100)
    whole = np.flatnonzero(radius > 340)
    config = dict(refine_remaining_heads=True,remaining_head_minimum_boundary_ratio=5,
                  ragged_minimum_added_clearance_nm=100,smooth_head_minimum_added_clearance_nm=100)
    previous,diag = sr.refine_remaining_spine_heads(mesh,skeleton,[cap],[cap],[0],config)
    assert np.array_equal(previous[0],cap) and diag['candidates'][0]['action']=='screened_keep'
    config['screen_smooth_incomplete_heads'] = True
    recovered,diag = sr.refine_remaining_spine_heads(mesh,skeleton,[cap],[cap],[0],config)
    assert diag['candidates'][0]['action']=='grow_remaining_head'
    assert len(recovered[0])>len(cap) and np.isin(cap,recovered[0]).all()
    preserved,diag = sr.refine_remaining_spine_heads(mesh,skeleton,[whole],[whole],[0],config)
    assert np.array_equal(preserved[0],whole) and diag['candidates'][0]['action']=='screened_keep'


def test_smooth_head_screen_with_collapsed_axis_keeps_seed_instead_of_crashing():
    mesh,_,radius = branch_surface()
    cap = np.flatnonzero(radius>1100)
    collapsed = np.array([[[0,0,0],[0,0,100]]],dtype=float)
    output,diag = sr.refine_remaining_spine_heads(mesh,collapsed,[cap],[cap],[0],
        dict(refine_remaining_heads=True,screen_smooth_incomplete_heads=True,
             remaining_head_minimum_boundary_ratio=5,filter_incomplete_heads=True))
    assert len(output)==1 and np.array_equal(output[0],cap)
    assert diag['candidates'][0]['action']=='unknown_keep'


def test_head_completion_cannot_introduce_a_perforated_proximal_surface(monkeypatch):
    mesh,skeleton,radius = branch_surface()
    cap = np.flatnonzero(radius>1100)
    whole = np.flatnonzero(radius>340)
    proximal = np.flatnonzero((radius>340)&(radius<700))
    perforated = np.setdiff1d(whole,proximal[::2])
    assert np.isin(cap,perforated).all()
    original = mesh.vertices.copy(),mesh.faces.copy(),cap.copy()
    monkeypatch.setattr(sr._Context,'grow',lambda *args:perforated)
    config = dict(refine_remaining_heads=True,screen_smooth_incomplete_heads=True,
                  remaining_head_minimum_boundary_ratio=6,smooth_head_minimum_added_clearance_nm=0,
                  filter_incomplete_heads=True,smooth_head_maximum_boundary_increase=100)
    unchecked,diag = sr.refine_remaining_spine_heads(mesh,skeleton,[cap],[cap],[0],config)
    assert np.array_equal(unchecked[0],perforated)
    assert not diag['candidates'][0]['geometry_after']['incomplete']
    config['smooth_head_maximum_boundary_increase'] = 2
    guarded,diag = sr.refine_remaining_spine_heads(mesh,skeleton,[cap],[cap],[0],config)
    assert guarded==[]
    assert all(a.get('reason')=='remaining_head_growth_creates_ragged_root'
               for a in diag['candidates'][0]['attempts'])
    assert np.array_equal(mesh.vertices,original[0]) and np.array_equal(mesh.faces,original[1])
    assert np.array_equal(cap,original[2])


def test_unrecoverable_head_removal_is_opt_in_and_does_not_use_existing_source_holes(monkeypatch):
    mesh,skeleton,radius = branch_surface()
    cap = np.flatnonzero(radius > 1100)
    def fail(*args):
        raise ValueError('no_safe_growth')
    monkeypatch.setattr(sr._Context,'grow',fail)
    kept,_ = sr.refine_remaining_spine_heads(mesh,skeleton,[cap],[cap],[0],dict(refine_remaining_heads=True))
    assert np.array_equal(kept[0],cap)
    removed,diag = sr.refine_remaining_spine_heads(mesh,skeleton,[cap],[cap],[0],
        dict(refine_remaining_heads=True,filter_incomplete_heads=True))
    assert removed == [] and diag['candidates'][0]['action'] == 'remove_incomplete_head'
    # A detached source patch has no missing neighbouring source surface to
    # recover. Its source-open rim must not become a segmentation rejection.
    source = mesh.submesh([cap],append=True)
    source_ids = np.arange(len(source.faces))
    result,diag = sr.refine_remaining_spine_heads(source,skeleton,[source_ids],[source_ids],[0],
        dict(refine_remaining_heads=True,filter_incomplete_heads=True))
    assert np.array_equal(result[0],source_ids)
    assert diag['candidates'][0]['action'] != 'remove_incomplete_head'
