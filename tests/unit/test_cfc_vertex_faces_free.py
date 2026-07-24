"""Gate for the vertex_faces-free ``connected_face_components`` vertices-path.

The old implementation materialised ``mesh.vertex_faces`` -- an (n_vertices x MAX vertex degree)
-1-padded array whose *build* was measured as the single largest contributor to the big-H01
pipeline's RAM peak (+1565 MB of a +2127 MB rise). The replacement labels faces from the same
scipy union-find the vertex partition already uses.

These tests pin the replacement to the ORIGINAL algorithm byte-for-byte: same number of
components, same order, same face indices, same dtype. The reference below is a verbatim copy of
the pre-change body -- if it and the production function ever disagree, the partition changed.
"""
import numpy as np
import pytest
import trimesh

from neurd import submesh_ops as so


def reference_cfc_vertices(mesh):
    """Verbatim pre-change implementation, kept as the equivalence oracle."""
    n_faces = len(mesh.faces)
    if n_faces == 0:
        return []
    vertex_faces = mesh.vertex_faces
    comps = []
    for vgroup in so.vertex_components(mesh):
        faces = np.unique(np.concatenate(vertex_faces[vgroup]))
        comps.append(faces[faces != -1].astype(np.int64))
    return comps


def assert_identical(mesh):
    expected = reference_cfc_vertices(mesh)
    actual = so.connected_face_components(mesh, connectivity="vertices")
    assert len(actual) == len(expected), (
        f"component count differs: {len(actual)} != {len(expected)}")
    for i, (a, e) in enumerate(zip(actual, expected)):
        assert a.dtype == e.dtype == np.int64, f"comp {i} dtype {a.dtype} vs {e.dtype}"
        np.testing.assert_array_equal(a, e, err_msg=f"component {i} differs")
    return actual


# --------------------------------------------------------------------------- #
# structured meshes                                                           #
# --------------------------------------------------------------------------- #
def _disjoint(*meshes):
    return trimesh.util.concatenate(list(meshes))


def test_single_component():
    assert len(assert_identical(trimesh.creation.icosphere(subdivisions=3))) == 1


def test_two_components_different_sizes():
    m = _disjoint(
        trimesh.creation.icosphere(subdivisions=3),
        trimesh.creation.icosphere(subdivisions=1).apply_translation([10, 0, 0]),
    )
    assert len(assert_identical(m)) == 2


def test_many_components_including_equal_sizes():
    """Equal-sized pieces are the tie case: component ORDER must match, since split() re-sorts
    with an unstable argsort whose tiebreak depends on this order."""
    parts = [trimesh.creation.box().apply_translation([3 * i, 0, 0]) for i in range(12)]
    parts.append(trimesh.creation.icosphere(subdivisions=2).apply_translation([0, 20, 0]))
    assert len(assert_identical(_disjoint(*parts))) == 13


def test_single_triangle():
    m = trimesh.Trimesh(vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]], faces=[[0, 1, 2]],
                        process=False)
    assert_identical(m)


def test_unreferenced_vertices_present():
    """Vertices touched by no edge are not graph nodes; they must not create components."""
    base = trimesh.creation.box()
    verts = np.vstack([base.vertices, [[99, 99, 99], [98, 98, 98]]])
    m = trimesh.Trimesh(vertices=verts, faces=base.faces, process=False)
    assert_identical(m)


def test_degenerate_faces():
    """A face with repeated vertices still contributes its (self-loop) edges."""
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=float)
    faces = np.array([[0, 1, 2], [1, 3, 2], [0, 0, 0], [3, 3, 3]], dtype=np.int64)
    m = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    assert_identical(m)


def test_high_degree_vertex_is_the_padding_case():
    """A fan around one hub is exactly what blows up vertex_faces' padded width."""
    n = 200
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    verts = np.vstack([[0, 0, 0], np.c_[np.cos(ang), np.sin(ang), np.zeros(n)]])
    faces = np.array([[0, i + 1, (i + 1) % n + 1] for i in range(n)], dtype=np.int64)
    m = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    assert m.vertex_faces.shape[1] == n          # the pathological padding width
    assert_identical(m)


def test_empty_faces_returns_empty():
    m = trimesh.Trimesh(vertices=np.zeros((3, 3)), faces=np.zeros((0, 3), dtype=np.int64),
                        process=False)
    assert so.connected_face_components(m, connectivity="vertices") == []


# --------------------------------------------------------------------------- #
# randomised sweep                                                            #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", range(25))
def test_random_multicomponent_meshes(seed):
    rng = np.random.default_rng(seed)
    parts = []
    for _ in range(rng.integers(1, 7)):
        kind = rng.integers(0, 3)
        if kind == 0:
            p = trimesh.creation.icosphere(subdivisions=int(rng.integers(0, 3)))
        elif kind == 1:
            p = trimesh.creation.box()
        else:
            p = trimesh.creation.cylinder(radius=1, height=2, sections=int(rng.integers(4, 12)))
        parts.append(p.apply_translation(rng.uniform(-50, 50, 3)))
    m = _disjoint(*parts)
    # shuffle faces so first-appearance order (and thus component rank) varies
    perm = rng.permutation(len(m.faces))
    m = trimesh.Trimesh(vertices=m.vertices, faces=np.asarray(m.faces)[perm], process=False)
    assert_identical(m)


# --------------------------------------------------------------------------- #
# the property the replacement relies on, asserted directly                   #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", range(10))
def test_all_three_face_vertices_share_a_component(seed):
    """The correctness premise: labelling a face by vertex 0 is safe because a face's three
    vertices are always in one component (the face contributes all three of its edges)."""
    rng = np.random.default_rng(100 + seed)
    m = trimesh.util.concatenate([
        trimesh.creation.icosphere(subdivisions=int(rng.integers(0, 3))
                                   ).apply_translation(rng.uniform(-50, 50, 3))
        for _ in range(int(rng.integers(2, 5)))])
    labels, rank = so._vertex_component_labels(m)[:2]
    faces = np.asarray(m.faces, dtype=np.int64)
    lab = labels[faces]
    assert (lab[:, 0] == lab[:, 1]).all() and (lab[:, 0] == lab[:, 2]).all()
    assert (rank[lab[:, 0]] >= 0).all(), "every face must map to a yielded component"


def test_split_order_unchanged():
    """split() re-sorts by size with an UNSTABLE argsort, so it is the real consumer of order."""
    parts = [trimesh.creation.box().apply_translation([4 * i, 0, 0]) for i in range(8)]
    m = _disjoint(*parts)
    ref_order = [np.sort(c) for c in reference_cfc_vertices(m)]
    sizes_ref = [len(c) for c in ref_order]
    pieces = so.split(m, connectivity="vertices")
    assert [p.n_faces for p in pieces] == sorted(sizes_ref, reverse=True)
    for p in pieces:
        np.testing.assert_array_equal(np.sort(p.face_idx), p.face_idx)
