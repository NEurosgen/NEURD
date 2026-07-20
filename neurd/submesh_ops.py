"""NEURD-owned mesh partition + face-index layer (bottom-up rebuild, Phase 0).

Why this module exists
----------------------
NEURD's essence is: take meshes -> segment them. That vocabulary is today ~63 scattered
``trimesh_utils`` (``tu.*``) calls into the fragile ``mesh_tools`` sibling dependency. This
module rebuilds the proven **L2 "partition + face-index"** cluster
(``split`` / ``split_significant_pieces`` / ``largest_conn_comp`` /
``connected_components_from_face_idx`` / ``original_mesh_faces_map`` -- ~420 mesh_tools lines)
as a small, clean, owned layer: 2 primitives + a ``SubMesh`` value type + ~5 tiny composites.

The linchpin
------------
``trimesh.submesh()`` returns a standalone mesh with NO link back to its parent's face
indices, so the framework perpetually re-derives that link -- either by KDTree face-midpoint
matching (``tu.original_mesh_faces_map``, 10x uses, ~103 lines, threshold-fragile) or by hand
(``[face_idx[k] for k in comps]``). ``SubMesh`` carries ``face_idx`` **by construction** and
COMPOSES (submesh-of-submesh remaps indices), so ``root_face_idx()`` folds the whole chain back
to the original mesh with pure index algebra -- no geometric re-matching.

Scope
-----
Phase 0: this module + its equivalence tests (``tests/unit/test_submesh_ops.py``) prove
behavioural equivalence to the ``tu.*`` originals on synthetic + real meshes. NO production code
is migrated yet, and this module imports **no** ``mesh_tools`` -- it is built directly on
trimesh / networkx primitives (``face_adjacency``, ``vertex_adjacency_graph``, ``vertex_faces``).

Companion module ``neurd/_mesh_ops.py`` holds the separate L6 "repair/reconstruct" ops
(decimate / poisson / fill_holes); this file is the L2 "partition" concern.

See ``MESH_OPERATIONS_LAYER.md`` / ``MESH_OPS_REBUILD_PLAN.md`` for the full layer map and plan.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import networkx as nx
from trimesh.graph import connected_components as _trimesh_connected_components


def _submesh(mesh, faces):
    """One face-group -> a standalone Trimesh.

    ``repair=False`` keeps the face count exact (no boundary hole-filling); ``append=True``
    returns a single mesh rather than a length-1 list. Matches how ``tu`` builds sub-pieces.
    """
    return mesh.submesh([np.asarray(faces, dtype=np.int64)], append=True, repair=False)


# --------------------------------------------------------------------------- #
# Primitive 1 -- the composing value type                                     #
# --------------------------------------------------------------------------- #
@dataclass(eq=False)
class SubMesh:
    """A mesh that remembers its provenance and composes.

    ``face_idx`` holds indices into the *immediate* parent's faces (so
    ``len(face_idx) == len(mesh.faces)``); ``root_face_idx()`` folds the parent chain to
    recover indices into the ORIGINAL (root) mesh. ``parent`` is either a raw
    ``trimesh.Trimesh`` (the root) or another ``SubMesh``.
    """

    mesh: object                 # trimesh.Trimesh -- the extracted geometry
    face_idx: np.ndarray         # indices into the immediate parent's faces
    parent: object               # trimesh.Trimesh (root) or SubMesh

    def __post_init__(self):
        self.face_idx = np.asarray(self.face_idx, dtype=np.int64)

    @classmethod
    def from_mesh(cls, mesh) -> "SubMesh":
        """Identity SubMesh over a raw mesh: ``face_idx = arange(n_faces)``, parent = the mesh."""
        return cls(mesh, np.arange(len(mesh.faces), dtype=np.int64), mesh)

    def sub(self, faces) -> "SubMesh":
        """Submesh-of-submesh. ``faces`` index into THIS mesh; the child stores them as its own
        (immediate-parent) ``face_idx`` and links back to ``self`` -- so ``root_face_idx()``
        remaps through the chain automatically.
        """
        faces = np.asarray(faces, dtype=np.int64)
        return SubMesh(_submesh(self.mesh, faces), faces, self)

    def root_face_idx(self) -> np.ndarray:
        """Indices of THIS mesh's faces in the ORIGINAL (root) mesh -- folds the parent chain."""
        if isinstance(self.parent, SubMesh):
            return self.parent.root_face_idx()[self.face_idx]
        return self.face_idx     # parent is the raw root mesh; face_idx already indexes it

    @property
    def n_faces(self) -> int:
        return len(self.face_idx)


# --------------------------------------------------------------------------- #
# Primitive 2 -- connected components as face-index groups                     #
# --------------------------------------------------------------------------- #
def connected_face_components(mesh, connectivity="vertices"):
    """Face-index groups of the mesh's connected components (raw graph order, not size-sorted).

    connectivity="vertices" (default): faces sharing a VERTEX are grouped (looser) -- mirrors
        ``tu.split_by_vertices`` (``vertex_adjacency_graph`` components -> the faces on them).
    connectivity="edges": faces sharing an EDGE are grouped -- trimesh ``face_adjacency`` graph
        (isolated faces included as singletons, matching ``tu.split(only_watertight=False)``).
    """
    n_faces = len(mesh.faces)
    if n_faces == 0:
        return []
    if connectivity == "vertices":
        vertex_faces = mesh.vertex_faces          # (n_vertices, max_deg), -1 padded
        comps = []
        for vgroup in nx.connected_components(mesh.vertex_adjacency_graph):
            faces = np.unique(np.concatenate(vertex_faces[list(vgroup)]))
            comps.append(faces[faces != -1].astype(np.int64))
        return comps
    if connectivity == "edges":
        comps = _trimesh_connected_components(
            edges=mesh.face_adjacency,
            nodes=np.arange(n_faces),
            min_len=1,
        )
        return [np.asarray(c, dtype=np.int64) for c in comps]
    raise ValueError(f"connectivity must be 'vertices' or 'edges', got {connectivity!r}")


# --------------------------------------------------------------------------- #
# Composites -- each a few lines over the two primitives                       #
# --------------------------------------------------------------------------- #
def split(mesh, connectivity="vertices"):
    """Connected components as SubMeshes, largest-first by face count.

    Mirrors ``tu.split``: order via ``np.flip(np.argsort(sizes))`` (identical tiebreak, since
    the underlying component partition is produced by the same trimesh primitive).
    """
    comps = connected_face_components(mesh, connectivity)
    order = np.flip(np.argsort([len(c) for c in comps]))
    return [SubMesh(_submesh(mesh, comps[i]), comps[i], mesh) for i in order]


def _order_by_vertices_desc(pieces):
    """Largest-first by VERTEX count, matching tu's stable-ascending-then-reversed tiebreak
    (``sorted(...)[::-1]``) so the order is byte-identical even when pieces tie on vertex count."""
    order = sorted(range(len(pieces)), key=lambda i: len(pieces[i].mesh.vertices))[::-1]
    return [pieces[i] for i in order]


def split_significant(mesh, min_faces, connectivity="vertices", return_insignificant=False):
    """Components split by a ``min_faces`` face-count threshold, each group largest-first by VERTEX count.

    Mirrors ``tu.split_significant_pieces``: it filters on face count but orders survivors by
    ``len(vertices)`` (not faces). With ``return_insignificant=True`` returns ``(significant,
    insignificant)`` -- both lists of ``SubMesh``, each vertex-ordered -- mirroring
    ``tu.split_significant_pieces(return_insignificant_pieces=True)``.
    """
    pieces = split(mesh, connectivity)
    significant = _order_by_vertices_desc([s for s in pieces if s.n_faces >= min_faces])
    if not return_insignificant:
        return significant
    insignificant = _order_by_vertices_desc([s for s in pieces if s.n_faces < min_faces])
    return significant, insignificant


def largest_component(mesh, connectivity="vertices"):
    """The single largest connected component (by vertex count).

    Mirrors ``tu.largest_conn_comp`` == ``split_significant_pieces(threshold=-1)[0]``.
    """
    return split_significant(mesh, -1, connectivity)[0]


def components_from_face_idx(sub: SubMesh):
    """Split a ``SubMesh`` into its connected sub-components (each a further ``SubMesh``).

    Mirrors ``tu.connected_components_from_face_idx``: split then remap indices back to the
    parent -- here the remap is automatic via ``SubMesh.sub`` / ``root_face_idx``.
    """
    return [sub.sub(c) for c in connected_face_components(sub.mesh, "vertices")]


def original_faces(sub: SubMesh) -> np.ndarray:
    """Face indices of ``sub`` in the original (root) mesh -- the O(depth) index-algebra
    replacement for ``tu.original_mesh_faces_map``'s KDTree face-midpoint matching.
    """
    return sub.root_face_idx()


def faces_by_match(original_mesh, submesh, matching=True, match_threshold=1e-3, return_mesh=False):
    """Face indices of ``original_mesh`` whose face-midpoint (does/doesn't) match a face of ``submesh``,
    by KDTree on face midpoints.

    The owned reimplementation of ``tu.original_mesh_faces_map``'s default (``exact_match=False``) path,
    for pieces that have NO ``SubMesh`` provenance to the original -- e.g. poisson/SDF-reconstructed soma
    meshes, whose geometry carries no index link back. (Where provenance DOES exist, use
    ``root_face_idx`` / ``original_faces`` for the O(1) map instead of this geometric match.)

    Uses the same ``pykdtree`` KDTree as ``tu`` and mirrors its exact behaviour: strict ``< threshold``,
    ascending face indices, and ``return_mesh`` via ``submesh([...], append=True)`` (trimesh's default
    ``repair=True``) -- so the result is byte-identical to ``tu.original_mesh_faces_map``.
    """
    from pykdtree.kdtree import KDTree
    orig_mid = original_mesh.triangles_center
    if submesh is None or len(getattr(submesh, "faces", [])) == 0:
        faces = np.array([], dtype=np.int64) if matching else np.arange(len(orig_mid), dtype=np.int64)
    else:
        dist, _ = KDTree(submesh.triangles_center).query(orig_mid)
        mask = dist < match_threshold if matching else dist >= match_threshold
        faces = np.where(mask)[0].astype(np.int64)
    if return_mesh:
        return original_mesh.submesh([faces], append=True)   # match tu: repair defaults to True
    return faces
