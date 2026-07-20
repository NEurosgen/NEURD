"""Phase-0 equivalence + property tests for neurd/submesh_ops.py (the L2 partition layer).

The point of these tests is to PROVE the clean reimplementation behaves identically to the
`tu.*` originals it will later replace -- as a partition (set of face-index groups) and as
face-index provenance -- on both a synthetic mesh and the real small-h01 fixture, for both
connectivity modes. No production code is touched, so nothing in the pipeline can break.

Strict ordering is asserted on the synthetic mesh (its 3 components have DISTINCT sizes, so
largest-first order is unambiguous); on the real mesh (where tiny fragments can tie in size)
we assert the partition/kept-set and that the largest piece matches -- which is what downstream
`[0]` consumers rely on.
"""
from pathlib import Path

import neurd  # noqa: F401  -- activates numpy/mesh shims before trimesh
import numpy as np
import pytest
import trimesh

from tests.unit import skip_if_datasci_tools_unusable

skip_if_datasci_tools_unusable()

from mesh_tools import trimesh_utils as tu  # noqa: E402  (after the skip guard)
from neurd import submesh_ops as so  # noqa: E402

FIXTURE = Path(__file__).resolve().parents[2] / "Applications/Tutorials/Auto_Proof_Pipeline/neuron_2530864375.off"


def _fsets(arrays):
    """List of face-index arrays -> list of frozensets (order preserved)."""
    return [frozenset(int(x) for x in np.asarray(a).ravel()) for a in arrays]


@pytest.fixture(scope="module")
def synth():
    """3 disjoint icospheres of DISTINCT size (80 / 320 / 1280 faces) -> 3 known components,
    no size ties, so every ordered comparison is unambiguous."""
    s0 = trimesh.creation.icosphere(subdivisions=1)  # 80 faces
    s1 = trimesh.creation.icosphere(subdivisions=2)  # 320 faces
    s2 = trimesh.creation.icosphere(subdivisions=3)  # 1280 faces
    s1.apply_translation([5.0, 0.0, 0.0])
    s2.apply_translation([10.0, 0.0, 0.0])
    return tu.combine_meshes([s0, s1, s2])


@pytest.fixture(scope="module")
def real():
    if not FIXTURE.exists():
        pytest.skip(f"fixture missing: {FIXTURE}")
    return tu.load_mesh_no_processing(str(FIXTURE))


# --------------------------------------------------------------------------- #
# connected_face_components  ==  tu.split partition                            #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("connectivity", ["vertices", "edges"])
def test_components_partition_matches_tu_synth(synth, connectivity):
    mine = so.connected_face_components(synth, connectivity)
    _, tu_comps = tu.split(synth, connectivity=connectivity)
    assert set(_fsets(mine)) == set(_fsets(tu_comps))  # same partition (order-independent)
    assert len(mine) == len(tu_comps) == 3


@pytest.mark.parametrize("connectivity", ["vertices", "edges"])
def test_components_partition_matches_tu_real(real, connectivity):
    mine = so.connected_face_components(real, connectivity)
    _, tu_comps = tu.split(real, connectivity=connectivity)
    assert set(_fsets(mine)) == set(_fsets(tu_comps))
    assert len(mine) == len(tu_comps)


# --------------------------------------------------------------------------- #
# split  ==  tu.split  (SubMeshes, largest-first)                             #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("connectivity", ["vertices", "edges"])
def test_split_matches_tu_ordered_synth(synth, connectivity):
    mine = so.split(synth, connectivity)
    _, tu_comps = tu.split(synth, connectivity=connectivity)
    # same components, same largest-first order, same n_faces (distinct sizes -> strict)
    assert _fsets([s.face_idx for s in mine]) == _fsets(tu_comps)
    assert [s.n_faces for s in mine] == [len(c) for c in tu_comps]
    assert [s.n_faces for s in mine] == sorted([s.n_faces for s in mine], reverse=True)


@pytest.mark.parametrize("connectivity", ["vertices", "edges"])
def test_split_matches_tu_real(real, connectivity):
    mine = so.split(real, connectivity)
    _, tu_comps = tu.split(real, connectivity=connectivity)
    assert set(_fsets([s.face_idx for s in mine])) == set(_fsets(tu_comps))  # same partition
    assert [s.n_faces for s in mine] == [len(c) for c in tu_comps]  # same largest-first sizes
    # largest piece (what [0] consumers use) is identical
    assert frozenset(mine[0].face_idx.tolist()) == frozenset(int(x) for x in tu_comps[0])


# --------------------------------------------------------------------------- #
# split_significant  ==  tu.split_significant_pieces                           #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("thr", [0, 100, 400])
def test_split_significant_matches_tu_synth(synth, thr):
    mine = so.split_significant(synth, thr)
    _, tu_idx = tu.split_significant_pieces(synth, significance_threshold=thr, return_face_indices=True)
    assert _fsets([s.root_face_idx() for s in mine]) == _fsets(tu_idx)  # same kept set + order


def test_split_significant_matches_tu_real(real):
    thr = 100
    mine = so.split_significant(real, thr)
    _, tu_idx = tu.split_significant_pieces(real, significance_threshold=thr, return_face_indices=True)
    assert set(_fsets([s.root_face_idx() for s in mine])) == set(_fsets(tu_idx))  # same kept set


def test_split_significant_tie_order_matches_tu():
    """Two identical pieces tie on vertex count -> order must match tu EXACTLY (sorted[::-1]).

    Guards the faithful-drop-in tiebreak: with a plain `sort(reverse=True)` the tie would order
    the opposite way to tu, breaking byte-identity for the Phase-1 migration.
    """
    a = trimesh.creation.icosphere(subdivisions=2)  # 320 faces / 162 verts
    b = trimesh.creation.icosphere(subdivisions=2)  # identical -> ties with `a`
    c = trimesh.creation.icosphere(subdivisions=1)  # 80 faces (distinct)
    b.apply_translation([5.0, 0.0, 0.0])
    c.apply_translation([10.0, 0.0, 0.0])
    mesh = tu.combine_meshes([a, b, c])
    mine = so.split_significant(mesh, 1)
    _, tu_idx = tu.split_significant_pieces(mesh, significance_threshold=1, return_face_indices=True)
    assert _fsets([s.root_face_idx() for s in mine]) == _fsets(tu_idx)  # exact order, incl. the a/b tie


def test_split_significant_return_insignificant_matches_tu(synth):
    """return_insignificant=True -> (sig, insig), both vertex-ordered, matching tu's 4-tuple."""
    thr = 100  # synth 80/320/1280 faces -> sig={1280,320}, insig={80}
    sig, insig = so.split_significant(synth, thr, return_insignificant=True)
    tu_sig_m, tu_sig_idx, tu_insig_m, tu_insig_idx = tu.split_significant_pieces(
        synth, significance_threshold=thr, return_insignificant_pieces=True, return_face_indices=True
    )
    assert _fsets([s.root_face_idx() for s in sig]) == _fsets(tu_sig_idx)      # same sig set + order
    assert _fsets([s.root_face_idx() for s in insig]) == _fsets(tu_insig_idx)  # same insig set + order
    assert [s.n_faces for s in sig] == [1280, 320]
    assert [s.n_faces for s in insig] == [80]


# --------------------------------------------------------------------------- #
# largest_component  ==  tu.largest_conn_comp                                  #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("connectivity", ["vertices", "edges"])
def test_largest_component_matches_tu_synth(synth, connectivity):
    mine = so.largest_component(synth, connectivity)
    _, tu_idx = tu.largest_conn_comp(synth, connectivity=connectivity, return_face_indices=True)
    assert frozenset(mine.root_face_idx().tolist()) == frozenset(int(x) for x in tu_idx)


def test_largest_component_matches_tu_real(real):
    for connectivity in ("vertices", "edges"):
        mine = so.largest_component(real, connectivity)
        _, tu_idx = tu.largest_conn_comp(real, connectivity=connectivity, return_face_indices=True)
        assert frozenset(mine.root_face_idx().tolist()) == frozenset(int(x) for x in tu_idx)


# --------------------------------------------------------------------------- #
# components_from_face_idx  ==  tu.connected_components_from_face_idx           #
#   (proves the provenance composition: sub-of-sub remaps back to the parent)  #
# --------------------------------------------------------------------------- #
def test_components_from_face_idx_matches_tu_synth(synth):
    comps = so.connected_face_components(synth, "vertices")
    idx = np.concatenate([comps[0], comps[2]])  # two disjoint pieces -> 2 sub-components
    mine = so.components_from_face_idx(so.SubMesh.from_mesh(synth).sub(idx))
    tu_idx = tu.connected_components_from_face_idx(synth, idx, return_meshes=False)
    assert set(_fsets([s.root_face_idx() for s in mine])) == set(_fsets(tu_idx))
    assert len(mine) == len(tu_idx) == 2


def test_components_from_face_idx_matches_tu_real(real):
    comps = so.connected_face_components(real, "vertices")
    if len(comps) < 2:
        pytest.skip("real mesh has a single component; nothing to remap")
    idx = np.concatenate([comps[0], comps[-1]])
    mine = so.components_from_face_idx(so.SubMesh.from_mesh(real).sub(idx))
    tu_idx = tu.connected_components_from_face_idx(real, idx, return_meshes=False)
    assert set(_fsets([s.root_face_idx() for s in mine])) == set(_fsets(tu_idx))


# --------------------------------------------------------------------------- #
# original_faces  ==  tu.original_mesh_faces_map   (O(1) map == KDTree map)    #
# --------------------------------------------------------------------------- #
def test_original_faces_matches_tu_map_synth(synth):
    s = so.split(synth)[0]
    mine = so.original_faces(s)
    tu_map = tu.original_mesh_faces_map(synth, s.mesh)
    assert frozenset(int(x) for x in mine) == frozenset(int(x) for x in tu_map)


def test_original_faces_matches_tu_map_real(real):
    s = so.split(real)[0]
    mine = so.original_faces(s)
    tu_map = tu.original_mesh_faces_map(real, s.mesh)
    assert frozenset(int(x) for x in mine) == frozenset(int(x) for x in tu_map)


# --------------------------------------------------------------------------- #
# Property tests -- the composition law and round-trip (no tu.* needed)        #
# --------------------------------------------------------------------------- #
def test_composition_law(synth):
    """SubMesh.from_mesh(m).sub(A).sub(B).root_face_idx() == A[B].

    This is the core provenance-composition law. (It also guards the subtle bug in the plan's
    original snippet, where `sub` used `self.face_idx[faces]` AND `root_face_idx` folded the
    chain -- double-applying indices to give A[A[B]] instead of A[B].)
    """
    rng = np.random.default_rng(0)
    n = len(synth.faces)
    A = rng.permutation(n)[: n // 2]
    B = rng.permutation(len(A))[: max(1, len(A) // 2)]
    s2 = so.SubMesh.from_mesh(synth).sub(A).sub(B)
    assert np.array_equal(s2.root_face_idx(), A[B])
    assert s2.n_faces == len(B)


def test_from_mesh_identity(synth):
    s = so.SubMesh.from_mesh(synth)
    assert np.array_equal(s.root_face_idx(), np.arange(len(synth.faces)))
    assert s.n_faces == len(synth.faces)


def test_root_face_idx_round_trip(real):
    """submesh(real, s.root_face_idx()) reproduces s.mesh's faces (same faces, same order)."""
    s = so.split(real)[0]
    rebuilt = real.submesh([s.root_face_idx()], append=True, repair=False)
    assert len(rebuilt.faces) == s.n_faces
    assert np.allclose(rebuilt.triangles_center, s.mesh.triangles_center)


def test_empty_mesh_does_not_crash():
    empty = trimesh.Trimesh()
    assert so.connected_face_components(empty) == []
    assert so.split(empty) == []


def test_single_component_mesh():
    one = trimesh.creation.icosphere(subdivisions=2)
    for connectivity in ("vertices", "edges"):
        comps = so.connected_face_components(one, connectivity)
        assert len(comps) == 1
    pieces = so.split(one)
    assert len(pieces) == 1
    assert pieces[0].n_faces == len(one.faces)


def test_faces_by_match_matches_tu(synth, real):
    """faces_by_match == tu.original_mesh_faces_map (default exact_match=False), incl. inverse + return_mesh."""
    for mesh in (synth, real):
        idx = np.arange(0, len(mesh.faces), 3)                     # a genuine subset of faces
        child = mesh.submesh([idx], append=True, repair=False)
        assert frozenset(so.faces_by_match(mesh, child).tolist()) == \
               frozenset(int(x) for x in tu.original_mesh_faces_map(mesh, child))              # matching
        assert frozenset(so.faces_by_match(mesh, child, matching=False).tolist()) == \
               frozenset(int(x) for x in tu.original_mesh_faces_map(mesh, child, matching=False))  # inverse
        mm = so.faces_by_match(mesh, child, return_mesh=True)
        tm = tu.original_mesh_faces_map(mesh, child, return_mesh=True)
        assert len(mm.faces) == len(tm.faces)                                                  # return_mesh


def test_faces_by_match_empty_submesh():
    m = trimesh.creation.icosphere(subdivisions=2)
    assert len(so.faces_by_match(m, trimesh.Trimesh())) == 0
    assert len(so.faces_by_match(m, trimesh.Trimesh(), matching=False)) == len(m.faces)
