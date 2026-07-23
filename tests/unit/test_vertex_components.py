"""Equivalence gate for submesh_ops.vertex_components (the networkx vertex-graph replacement).

`tu.vertex_components` is `[list(k) for k in nx.connected_components(mesh.vertex_adjacency_graph)]`
-- building that networkx graph (`add_edges_from` over `edges_unique`) is ~205 s / 10% of a big-H01
build's wall (py-spy). `submesh_ops.vertex_components` computes the same partition with one scipy
connected-components pass, and NEURD monkeypatches it over `tu.vertex_components` (see
`neurd/__init__.py`), so it must be indistinguishable from the original -- including COMPONENT
ORDER, because the sole consumer `tu.split_by_vertices` re-sorts pieces with the unstable
`np.flip(np.argsort(sizes))`, whose tie order depends on the input order.

We compare against the REAL networkx implementation (inlined here, so the test still checks the
original even once the monkeypatch is live) on synthetic meshes with a controlled number of
components + the small real mesh fixture, and additionally assert that `tu.split_by_vertices`
returns identical pieces with the owned backend patched in.
"""
from pathlib import Path

import neurd  # noqa: F401  -- activates numpy/mesh shims before trimesh
import networkx as nx
import numpy as np
import pytest
import trimesh

from tests.unit import skip_if_datasci_tools_unusable

skip_if_datasci_tools_unusable()

from mesh_tools import trimesh_utils as tu  # noqa: E402  (after the skip guard)
from neurd import submesh_ops as so  # noqa: E402

REAL_MESH = Path(__file__).resolve().parents[1] / "990_mesh.off"


def _nx_vertex_components(mesh):
    """The ORIGINAL mesh_tools implementation, inlined (tu.vertex_components is monkeypatched)."""
    return [list(k) for k in nx.connected_components(mesh.vertex_adjacency_graph)]


def _assert_same_components(mesh):
    expected = _nx_vertex_components(mesh)
    got = so.vertex_components(mesh)
    assert len(got) == len(expected), f"{len(got)} components, expected {len(expected)}"
    for i, (g, e) in enumerate(zip(got, expected)):
        # same component at the same POSITION (order matters downstream), same members
        assert set(np.asarray(g).tolist()) == set(e), f"component {i} differs"
    return got


def _translated_copies(mesh, n, spacing=10.0):
    """n disjoint copies of `mesh`, far enough apart to stay separate components."""
    pieces = []
    for i in range(n):
        m = mesh.copy()
        m.apply_translation([i * spacing, 0, 0])
        pieces.append(m)
    return trimesh.util.concatenate(pieces)


def test_single_component_box():
    _assert_same_components(trimesh.creation.box())


def test_two_disjoint_boxes():
    comps = _assert_same_components(_translated_copies(trimesh.creation.box(), 2))
    assert len(comps) == 2


def test_many_equal_sized_components_order():
    """Equal-sized pieces: the ORDER is the only thing distinguishing the two backends."""
    mesh = _translated_copies(trimesh.creation.box(), 6)
    comps = _assert_same_components(mesh)
    assert len(comps) == 6
    # the components must come back in first-appearance order, not e.g. label order
    expected_first = [np.min(c) for c in _nx_vertex_components(mesh)]
    assert [int(np.min(np.asarray(c))) for c in comps] == [int(f) for f in expected_first]


def test_mixed_sizes_components():
    mesh = trimesh.util.concatenate([
        trimesh.creation.icosphere(subdivisions=2),
        _translated_copies(trimesh.creation.box(), 3, spacing=20.0),
    ])
    comps = _assert_same_components(mesh)
    assert len(comps) == 4


def test_unreferenced_vertices_excluded():
    """A vertex on no edge is not a networkx node -- it must not become a component."""
    box = trimesh.creation.box()
    verts = np.vstack([box.vertices, [[99.0, 99.0, 99.0], [98.0, 98.0, 98.0]]])
    mesh = trimesh.Trimesh(vertices=verts, faces=box.faces, process=False)
    comps = _assert_same_components(mesh)
    assert len(comps) == 1
    assert len(np.concatenate([np.asarray(c) for c in comps])) == len(box.vertices)


def test_empty_mesh():
    mesh = trimesh.Trimesh(vertices=np.zeros((0, 3)), faces=np.zeros((0, 3), dtype=int))
    assert so.vertex_components(mesh) == []


@pytest.mark.skipif(not REAL_MESH.exists(), reason="990_mesh.off fixture missing")
def test_real_mesh():
    _assert_same_components(trimesh.load_mesh(str(REAL_MESH)))


@pytest.mark.skipif(not REAL_MESH.exists(), reason="990_mesh.off fixture missing")
def test_real_mesh_fragmented():
    """A real EM mesh cut into many pieces -- the shape vertex_components actually sees."""
    mesh = trimesh.load_mesh(str(REAL_MESH))
    # keep every 3rd face: shatters the mesh into many components of assorted sizes
    keep = np.arange(0, len(mesh.faces), 3)
    frag = mesh.submesh([keep], append=True, repair=False)
    comps = _assert_same_components(frag)
    assert len(comps) > 1
    # the fixture must be ORDER-DISCRIMINATING: networkx's first-appearance order differs here
    # from the naive "sorted by lowest vertex id", so matching it is a real assertion
    mins = [min(c) for c in _nx_vertex_components(frag)]
    assert mins != sorted(mins), "fixture no longer distinguishes component ORDER"


def _split_pieces(mesh, backend):
    """(faces, vertices) of every piece tu.split_by_vertices returns under `backend`."""
    orig = tu.vertex_components
    tu.vertex_components = backend
    try:
        meshes, comps = tu.split_by_vertices(mesh, return_components=True)
    finally:
        tu.vertex_components = orig
    return [(len(m.faces), len(m.vertices)) for m in meshes], [np.sort(np.asarray(c)) for c in comps]


@pytest.mark.skipif(not REAL_MESH.exists(), reason="990_mesh.off fixture missing")
def test_split_by_vertices_identical_end_to_end():
    """The real consumer: identical pieces, in identical order, with identical face indices."""
    mesh = trimesh.load_mesh(str(REAL_MESH))
    # every 3rd face -> hundreds of pieces, most of them tied on face count: the exact input
    # that makes components_to_submeshes' unstable np.flip(np.argsort(sizes)) order-sensitive
    frag = mesh.submesh([np.arange(0, len(mesh.faces), 3)], append=True, repair=False)

    shapes_nx, comps_nx = _split_pieces(frag, _nx_vertex_components)
    shapes_owned, comps_owned = _split_pieces(frag, so.vertex_components)

    assert len(shapes_nx) > 10, "fixture must produce many (tied) pieces to be a real order test"
    assert shapes_owned == shapes_nx
    assert len(comps_owned) == len(comps_nx)
    for a, b in zip(comps_owned, comps_nx):
        assert np.array_equal(a, b)
