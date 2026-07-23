"""Equivalence gate for submesh_ops.pieces_adjacency (the O(N^2) spine-connectivity replacement).

get_spine_meshes_unfiltered_from_mesh built a segment adjacency graph by calling
`tu.mesh_pieces_connectivity` once per segment with periphery = ALL segments (O(N^2), ~36k
calls/neuron). `submesh_ops.pieces_adjacency` computes the same undirected "share a vertex" edge
set in one vectorized pass. This test asserts the two produce an IDENTICAL edge set -- so the
spine_graph, its shortest paths, and its connected components (hence every spine) are unchanged.

We compare against the REAL tu function (the thing being replaced), replicating the exact caller
loop from spine_utils.py, on synthetic partitions + the small real mesh fixture. No neuron build.
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

REAL_MESH = Path(__file__).resolve().parents[1] / "990_mesh.off"


def _tu_caller_edges(mesh, pieces_idx, connectivity="vertices"):
    """Edge set exactly as spine_utils.py:1129-1148 built it, via the real tu backend."""
    edges = set()
    for j, curr_idx in enumerate(pieces_idx):
        touching = tu.mesh_pieces_connectivity(
            main_mesh=mesh,
            central_piece=curr_idx,
            periphery_pieces=pieces_idx,
            connectivity=connectivity,
        )
        for h in touching:
            if h != j:                       # the caller's touching_meshes.remove(j)
                edges.add(frozenset((j, h)))
    return edges


def _prim_edges(mesh, pieces_idx, connectivity="vertices"):
    return {frozenset(pair) for pair in so.pieces_adjacency(mesh, pieces_idx, connectivity)}


def _label_partition(mesh, labels):
    """Per-face labels -> list of face-index arrays (one piece per label), like CGAL segments."""
    labels = np.asarray(labels)
    return [np.where(labels == lab)[0] for lab in np.unique(labels)]


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def sphere():
    """One connected icosphere (1280 faces) -- a single mesh whose face partitions share vertices
    along their boundaries, exactly like the spine/shaft segments of a branch."""
    return trimesh.creation.icosphere(subdivisions=3)


# --------------------------------------------------------------------------- #
# equivalence vs the real tu.mesh_pieces_connectivity (the replaced function)
# --------------------------------------------------------------------------- #
def test_equiv_contiguous_bands(sphere):
    """Contiguous spatial bands (adjacent bands touch) -- realistic segment shapes."""
    x = sphere.triangles_center[:, 0]
    labels = np.digitize(x, np.quantile(x, [0.2, 0.4, 0.6, 0.8]))  # 5 bands
    pieces = _label_partition(sphere, labels)
    assert _prim_edges(sphere, pieces) == _tu_caller_edges(sphere, pieces)


def test_equiv_random_labels(sphere):
    """Random labels -> scattered pieces sharing many vertices: stresses the pairing/dedup."""
    rng = np.random.default_rng(0)
    for k in (2, 3, 8, 20):
        labels = rng.integers(0, k, size=len(sphere.faces))
        pieces = _label_partition(sphere, labels)
        assert _prim_edges(sphere, pieces) == _tu_caller_edges(sphere, pieces), f"k={k}"


@pytest.mark.skipif(not REAL_MESH.exists(), reason="990_mesh.off fixture missing")
def test_equiv_real_mesh():
    """Real EM mesh geometry (990_mesh.off, ~7k faces) partitioned into segments."""
    mesh = trimesh.load(str(REAL_MESH), process=False)
    rng = np.random.default_rng(1)
    for k in (4, 12, 30):
        labels = rng.integers(0, k, size=len(mesh.faces))
        pieces = _label_partition(mesh, labels)
        assert _prim_edges(mesh, pieces) == _tu_caller_edges(mesh, pieces), f"k={k}"


# --------------------------------------------------------------------------- #
# shape / degenerate contracts
# --------------------------------------------------------------------------- #
def test_two_touching_pieces_one_edge(sphere):
    """A binary split of one connected mesh -> the two halves share their cut boundary -> 1 edge."""
    x = sphere.triangles_center[:, 0]
    pieces = _label_partition(sphere, (x > np.median(x)).astype(int))
    assert so.pieces_adjacency(sphere, pieces) == [(0, 1)]


def test_disjoint_pieces_no_edge():
    """Two spatially separated meshes combined -> pieces share no vertex -> no edge."""
    a = trimesh.creation.icosphere(subdivisions=1)
    b = trimesh.creation.icosphere(subdivisions=1)
    b.apply_translation([5.0, 0.0, 0.0])
    combined = tu.combine_meshes([a, b])
    pieces = [np.arange(0, len(a.faces)),
              np.arange(len(a.faces), len(a.faces) + len(b.faces))]
    assert so.pieces_adjacency(combined, pieces) == []


def test_single_and_empty(sphere):
    assert so.pieces_adjacency(sphere, [np.arange(len(sphere.faces))]) == []
    assert so.pieces_adjacency(sphere, []) == []


def test_output_shape(sphere):
    """Pairs are (i, j) with i < j, sorted, unique, no self-loops."""
    rng = np.random.default_rng(2)
    labels = rng.integers(0, 10, size=len(sphere.faces))
    pieces = _label_partition(sphere, labels)
    edges = so.pieces_adjacency(sphere, pieces)
    assert edges == sorted(set(edges))
    assert all(i < j for i, j in edges)


def test_edges_mode_not_implemented(sphere):
    with pytest.raises(NotImplementedError):
        so.pieces_adjacency(sphere, [np.arange(10), np.arange(10, 20)], connectivity="edges")
