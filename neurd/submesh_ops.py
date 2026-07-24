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
trimesh / scipy primitives (``face_adjacency``, ``edges_unique``, ``vertex_faces``).

Companion module ``neurd/_mesh_ops.py`` holds the separate L6 "repair/reconstruct" ops
(decimate / poisson / fill_holes); this file is the L2 "partition" concern.

See ``MESH_OPERATIONS_LAYER.md`` / ``MESH_OPS_REBUILD_PLAN.md`` for the full layer map and plan.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components as _sp_connected_components
from trimesh.graph import connected_components as _trimesh_connected_components
from trimesh.util import concatenate as _trimesh_concatenate


def _submesh(mesh, faces):
    """One face-group -> a standalone Trimesh.

    ``repair=False`` keeps the face count exact (no boundary hole-filling); ``append=True``
    returns a single mesh rather than a length-1 list. Matches how ``tu`` builds sub-pieces.
    """
    return mesh.submesh([np.asarray(faces, dtype=np.int64)], append=True, repair=False)


def _submesh_single(mesh, faces):
    """One face-group -> a standalone Trimesh, built the *non-appended* way.

    ``tu.split_mesh_into_face_groups``'s default path uses
    ``submesh([faces], append=False, only_watertight=False, repair=False)[0]`` -- trimesh builds
    one mesh per entry of ``faces_sequence``, so a single entry always yields a length-1 list and
    ``[0]`` is that mesh. Kept distinct from :func:`_submesh` (which appends/concatenates) so the
    label-split composite reproduces ``tu``'s construction path exactly.
    """
    return mesh.submesh([np.asarray(faces, dtype=np.int64)],
                        append=False, only_watertight=False, repair=False)[0]


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
def vertex_components(mesh):
    """Vertex-index groups of the mesh's vertex-connected components (scipy union-find).

    The owned replacement for ``tu.vertex_components`` ==
    ``[list(k) for k in nx.connected_components(mesh.vertex_adjacency_graph)]``, whose networkx
    graph build (``add_edges_from`` over ``edges_unique``) is ~10% of a big-H01 build's wall
    (py-spy). Same input (``mesh.edges_unique``), same partition, computed as one sparse
    connected-components pass instead of a Python-level graph.

    **Order is preserved**: networkx yields components in the order their first node was
    inserted, i.e. first appearance scanning ``edges_unique`` row-major -- reproduced here by
    ranking each component on the lowest position its members occupy in the flattened edge list.
    That matters because the sole consumer (``tu.split_by_vertices``) re-sorts pieces with
    ``np.flip(np.argsort(sizes))``, an UNSTABLE sort whose tie order depends on input order.

    Vertices touched by no edge are excluded (they are not nodes of the networkx graph either).
    Within a component the vertices come back ascending rather than in networkx's set-iteration
    order; every consumer feeds them straight into ``np.unique`` / fancy-indexing, where order is
    irrelevant.
    """
    parts = _vertex_component_labels(mesh)
    if parts is None:
        return []
    labels, rank, _n_seen, flat, n_vertices = parts

    present = np.zeros(n_vertices, dtype=bool)
    present[flat] = True
    nodes = np.flatnonzero(present)               # graph nodes only -- drops unreferenced vertices
    node_rank = rank[labels[nodes]]
    order = np.argsort(node_rank, kind="stable")
    nodes, node_rank = nodes[order], node_rank[order]
    return np.split(nodes, np.flatnonzero(np.diff(node_rank)) + 1)


def _vertex_component_labels(mesh):
    """Shared union-find core: ``(labels, rank, n_seen, flat, n_vertices)``, or ``None`` if edgeless.

    ``labels[v]`` is scipy's raw component id of vertex ``v``; ``rank[label]`` is that component's
    position in :func:`vertex_components`' yield order (``-1`` for components with no vertex in the
    edge list); ``n_seen`` is how many components that order contains. Split out of
    :func:`vertex_components` so :func:`connected_face_components` can label faces from the same
    partition without materialising ``mesh.vertex_faces``.
    """
    edges = np.asarray(mesh.edges_unique, dtype=np.int64)
    if len(edges) == 0:
        return None
    # a self-inconsistent mesh (faces indexing past the vertex block -- the MeshLabServer OFF
    # exporter bug guarded in neurd/__init__.py) must not fail HERE: size the graph to the edges,
    # which are derived FROM the faces, so every face index is addressable in `labels`.
    n_vertices = max(len(mesh.vertices), int(edges.max()) + 1)
    adjacency = coo_matrix(
        (np.ones(len(edges), dtype=bool), (edges[:, 0], edges[:, 1])),
        shape=(n_vertices, n_vertices))
    n_comp, labels = _sp_connected_components(adjacency, directed=False)

    flat = edges.ravel()
    # networkx yield order == ascending first-appearance position of the component's first node.
    # Scatter-assign in REVERSE so the last (= lowest) write wins -> first_pos[label] = min index.
    first_pos = np.full(n_comp, len(flat), dtype=np.int64)
    first_pos[labels[flat][::-1]] = np.arange(len(flat) - 1, -1, -1)
    seen_labels = np.flatnonzero(first_pos < len(flat))
    rank = np.full(n_comp, -1, dtype=np.int64)
    rank[seen_labels[np.argsort(first_pos[seen_labels], kind="stable")]] = np.arange(len(seen_labels))
    return labels, rank, len(seen_labels), flat, n_vertices


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
        # Label faces straight from the vertex partition instead of going through
        # ``mesh.vertex_faces``. That array is (n_vertices x MAX vertex degree), -1 padded, so a
        # handful of high-degree vertices size the whole thing: on the big H01 neuron it is
        # (1.63M x 65) int64 = 847 MB of which 90.7% is filler, and *building* it spikes RSS by
        # ~1.6 GB -- measured as the single largest contributor to the pipeline's RAM peak. The
        # partition is already known from the union-find, and a face's three vertices are always
        # in one component (the face contributes all three of its edges to the graph), so vertex 0
        # alone identifies a face's component.
        parts = _vertex_component_labels(mesh)
        if parts is None:
            return []                             # no edges -> vertex_components() is empty too
        labels, rank, n_seen = parts[0], parts[1], parts[2]
        face_rank = rank[labels[np.asarray(mesh.faces, dtype=np.int64)[:, 0]]]
        # stable sort by component rank -> groups appear in vertex_components() order and each
        # group's face indices stay ascending, matching the old np.unique() output exactly.
        order = np.argsort(face_rank, kind="stable")
        sorted_rank = face_rank[order]
        boundaries = np.flatnonzero(np.diff(sorted_rank)) + 1
        starts = np.concatenate(([0], boundaries))    # POSITION of each group in sorted_rank
        comps = [np.empty(0, dtype=np.int64) for _ in range(n_seen)]
        for grp, start in zip(np.split(order, boundaries), starts):
            comps[sorted_rank[start]] = grp.astype(np.int64, copy=False)
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
# Primitive 2b -- adjacency AMONG pieces (inter-piece touch graph)             #
# --------------------------------------------------------------------------- #
def pieces_adjacency(parent_mesh, pieces_face_idx, connectivity="vertices"):
    """Undirected adjacency among mesh PIECES of ``parent_mesh``, each a face-index array.

    Edge ``(i, j)`` iff pieces ``i`` and ``j`` share a vertex -- the exact touch criterion of
    ``tu.mesh_pieces_connectivity``'s vertex path (``len(np.intersect1d(verts_i, verts_j)) > 0``),
    but computed in ONE vertex->pieces inversion pass instead of that function's O(N^2) per-piece
    re-scan (which rebuilds every piece's vertex set on every one of the N outer calls). Returns a
    sorted list of ``(i, j)`` index pairs with ``i < j`` and no self-loops; a piece that touches
    nothing yields no pair -- matching ``nx.from_edgelist``, which only creates nodes seen in an edge.

    ``pieces_face_idx[i]`` indexes into ``parent_mesh.faces`` (the pieces are a face partition of
    the parent, e.g. the CGAL spine/shaft segments of one branch). ``connectivity="edges"`` is not
    implemented -- the sole hot caller (spine detection) uses the ``"vertices"`` default.
    """
    if connectivity != "vertices":
        raise NotImplementedError(
            f"pieces_adjacency supports connectivity='vertices' only, got {connectivity!r}")
    n = len(pieces_face_idx)
    if n < 2:
        return []
    faces = parent_mesh.faces
    # one vertex set per piece (O(N) total), then invert: vertex id -> pieces sharing it
    vert_ids, piece_ids = [], []
    for i, fidx in enumerate(pieces_face_idx):
        v = np.unique(faces[np.asarray(fidx, dtype=np.int64)].ravel())
        vert_ids.append(v)
        piece_ids.append(np.full(len(v), i, dtype=np.int64))
    vert_ids = np.concatenate(vert_ids)
    piece_ids = np.concatenate(piece_ids)

    order = np.argsort(vert_ids, kind="stable")
    vert_ids = vert_ids[order]
    piece_ids = piece_ids[order]
    # split into runs of equal vertex id; distinct pieces in a run co-occur on that vertex -> edges.
    # Boundary vertices are shared by only a handful of pieces, so the inner pairing stays cheap.
    boundaries = np.flatnonzero(np.diff(vert_ids)) + 1
    edges = set()
    for grp in np.split(piece_ids, boundaries):
        members = np.unique(grp)
        if len(members) < 2:
            continue
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                edges.add((int(members[a]), int(members[b])))
    return sorted(edges)


# --------------------------------------------------------------------------- #
# Primitive 3 -- face groups from a per-face labeling (pure index algebra)      #
# --------------------------------------------------------------------------- #
def face_groups_by_label(labels):
    """Partition face indices by a per-face label -> ``{label: face_idx}``, ascending label order.

    ``labels`` is either a per-face sequence (``labels[i]`` = label of face ``i``) or a
    ``{face_idx: label}`` dict (normalised by sorting on the key, matching ``tu``). No mesh is
    involved -- this is pure index algebra, so it serves ANY face-labeling producer: the waterfill
    colourings (``resolve_empty_conflicting_face_labels``), the CGAL/SDF ``mesh_segmentation``
    labels used by spine detection, and one-off ``np.where(mapping == k)[0]`` lookups.

    Mirrors the grouping half of ``tu.split_mesh_into_face_groups``: labels come back in
    ``np.sort(np.unique(labels))`` order and each group is ``np.where(labels == lab)[0]``.
    """
    if isinstance(labels, dict):
        labels = list(dict(sorted(labels.items())).values())
    labels = np.asarray(labels)
    return {lab: np.where(labels == lab)[0] for lab in np.sort(np.unique(labels))}


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


def split_into_face_groups(parent, labels, return_dict=True, check_connected=False):
    """Split a mesh by a per-face labeling into one ``SubMesh`` per label (provenance carried).

    The owned replacement for ``tu.split_mesh_into_face_groups``: composes
    :func:`face_groups_by_label` (which faces carry each label) with :func:`_submesh_single`
    (``tu``'s exact construction path), returning ``SubMesh`` values instead of the parallel
    ``(submeshes, submeshes_idx)`` dicts -- so the face_idx travels WITH the geometry.

    ``parent`` may be a raw ``Trimesh`` or a ``SubMesh``; the labels index the parent's OWN faces,
    and passing the parent through keeps the chain -- with a ``SubMesh`` parent the children
    compose, so ``root_face_idx()`` folds straight to the original mesh.

    ``check_connected=True`` runs a REAL single-component check per group via
    :func:`connected_face_components`. Note ``tu``'s ``check_connect_comp`` flag is vestigial --
    it inspects ``submesh([faces], append=False)``, which trimesh builds as one mesh per
    ``faces_sequence`` entry, so its length is always 1 and its ``raise`` can never fire. This
    check therefore defaults to OFF: turning it on can legitimately raise where ``tu`` stayed
    silent, which is a behaviour change, not a bug fix.
    """
    base = parent.mesh if isinstance(parent, SubMesh) else parent
    groups = face_groups_by_label(labels)
    n_labelled = sum(len(f) for f in groups.values())
    if n_labelled != len(base.faces):
        raise ValueError(f"face labeling covers {n_labelled} faces but mesh has {len(base.faces)}")

    out = {}
    for lab, faces in groups.items():
        if check_connected and len(connected_face_components(_submesh_single(base, faces))) != 1:
            raise ValueError(f"label {lab} is not a single connected component")
        # NOTE: built via _submesh_single (not SubMesh.sub, which appends) so the geometry matches
        # tu exactly. Passing `parent` through keeps the chain: a SubMesh parent composes, a raw
        # mesh parent terminates it.
        out[lab] = SubMesh(_submesh_single(base, faces), faces, parent)
    return out if return_dict else list(out.values())


def combine(pieces, merge_vertices=True, return_pieces=False):
    """Concatenate meshes into one -- optionally reporting WHERE each piece landed.

    The owned replacement for ``tu.combine_meshes``, which returns only the merged mesh and throws
    the provenance away. Concatenation preserves face order and count, so piece ``i`` occupies the
    contiguous face range ``[offset, offset + len(piece.faces))`` of the result -- exactly the
    bookkeeping callers otherwise redo by hand (rebuilding a stitched limb frame, assembling
    sublimbs). With ``return_pieces=True`` those ranges come back as ``SubMesh`` values parented on
    the combined mesh, making ``combine`` the inverse of :func:`split_into_face_groups`.

    Mirrors ``tu.combine_meshes`` exactly: ``concatenate(list(pieces) + [empty])`` followed by
    ``merge_vertices()`` when asked. Merging touches vertices only -- face order and count are
    untouched, so the reported ranges stay valid.
    """
    pieces = list(pieces)
    combined = _trimesh_concatenate(
        pieces + [trimesh.Trimesh(vertices=np.array([]), faces=np.array([]))])
    if merge_vertices:
        combined.merge_vertices()
    if not return_pieces:
        return combined

    subs, offset = [], 0
    for p in pieces:
        n = len(p.faces)
        subs.append(SubMesh(p, np.arange(offset, offset + n, dtype=np.int64), combined))
        offset += n
    return combined, subs


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
