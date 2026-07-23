"""NEURD-owned skeletal-distance / mesh-correspondence kernel (byte-exact relocation).

Why this module exists
----------------------
The dominant remaining cost of `neuron._refine_neuron` lives in the LOCKED `mesh_tools`
functions `compartment_utils.get_skeletal_distance` / `..._no_skipping` (per-skeleton-edge
mesh subtraction) and their adaptive wrapper `mesh_correspondence_adaptive_distance`. This
module relocates that algorithm into owned `neurd` code so it can be gated and then optimized.

Step 1 (this file) is a **behaviour-preserving relocation**: the orchestration + the
`intersecting_array_components` union-find are owned here; the heavy mesh primitives
(`tu.split`, `sk.change_basis_matrix`, `check_coordinates_inside_bounding_box`,
`mesh_center_weighted_face_midpoints`, `trimesh.submesh`) are still reused verbatim, so the
output is byte-identical to the mesh_tools originals. Those primitive calls are marked
`# PRIM:` — each becomes a later, gate-protected swap to `submesh_ops`/numpy, and the per-edge
loop is the target of the real vectorisation win. Correctness is pinned by
`tests/unit/test_skeletal_distance.py` against captured real I/O.

The two mesh_tools variants collapse to ONE kernel here (`skeletal_distance`) plus a flag:
  * `keep_empty_placeholder=True`  == `get_skeletal_distance_no_skipping` (widths): an empty-slice
     edge appends a 0 distance, keeping the distance array aligned 1:1 to `edges`.
  * `keep_empty_placeholder=False` == `get_skeletal_distance` (correspondence): empty edges are
     skipped (shorter distance array).
Default-off branches (`stitch_patches>0`, `not fast_mesh_split`) are NOT reproduced — no neurd
caller enables them (see plan). Deterministic (the only `np.random` in the original is debug-dead).
"""
from __future__ import annotations

import numpy as np
import networkx as nx
import trimesh
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components as _sp_connected_components

# PRIM: reused mesh_tools primitives. sk.change_basis_matrix (linalg) + tu.split for the single
# post-loop correspondence split are kept; the per-edge submesh+split (the ~165k/40k hot calls) is
# gone -- replaced by a precomputed face-adjacency + subgraph connected-components (below).
from mesh_tools import trimesh_utils as tu
from mesh_tools import skeleton_utils as sk


def _face_adjacency_csr(mesh, connectivity):
    """Symmetric face x face adjacency (CSR, 0/1) for the whole mesh, built ONCE per call.

    connectivity="edges": faces sharing an edge (trimesh.face_adjacency).
    connectivity="vertices": faces sharing a vertex (face-vertex incidence M, faces adjacent where
    (M @ M.T) > 0). Matches what `tu.split(submesh, connectivity=...)` computes on a slice, but
    without building a per-edge submesh.
    """
    nf = len(mesh.faces)
    if connectivity == "edges":
        fa = np.asarray(mesh.face_adjacency)
        if len(fa) == 0:
            return csr_matrix((nf, nf), dtype=bool)
        data = np.ones(len(fa), dtype=bool)
        A = csr_matrix((data, (fa[:, 0], fa[:, 1])), shape=(nf, nf))
        A = A + A.T
        return A
    if connectivity == "vertices":
        nv = len(mesh.vertices)
        rows = np.repeat(np.arange(nf), 3)
        cols = np.asarray(mesh.faces).ravel()
        M = csr_matrix((np.ones(nf * 3, dtype=np.int32), (rows, cols)), shape=(nf, nv))
        A = (M @ M.T).tocsr()
        A.setdiag(0)
        A.eliminate_zeros()
        return A
    raise ValueError(f"connectivity must be 'edges' or 'vertices', got {connectivity!r}")


def _slice_components(adj, face_list):
    """Connected components of the subgraph induced on `face_list` -> list of ORIGINAL-face-index
    arrays. Replaces `tu.split(main_mesh.submesh([face_list]))` (same partition, no submesh)."""
    sub = adj[face_list][:, face_list]
    n_comp, labels = _sp_connected_components(sub, directed=False)
    if n_comp == 1:
        return [np.asarray(face_list)]
    order = np.argsort(labels, kind="stable")
    labels_sorted = labels[order]
    faces_sorted = np.asarray(face_list)[order]
    bounds = np.flatnonzero(np.diff(labels_sorted)) + 1
    return np.split(faces_sorted, bounds)


def intersecting_array_components(arrays, sort_components=True):
    """Connected components of a list of index-arrays that share ≥1 element.

    Owned reimplementation of `datasci_tools.numpy_utils.intersecting_array_components`
    (the `fast_mesh_split` grouping step). Byte-exact, incl. the size-sort tie-order:
    `np.flip(np.argsort(lengths))` REVERSES equal-length groups — preserved deliberately.
    """
    array_edges = []
    for i, arr1 in enumerate(arrays):
        for j, arr2 in enumerate(arrays):
            if i < j:
                if len(np.intersect1d(arr1, arr2)) > 0:
                    array_edges.append([i, j])
    G = nx.Graph()
    G.add_nodes_from(np.arange(len(arrays)))
    G.add_edges_from(array_edges)
    conn_comps = [list(k) for k in nx.connected_components(G)]
    if sort_components:
        lengths = [len(k) for k in conn_comps]
        conn_comps = [conn_comps[k] for k in np.flip(np.argsort(lengths))]
    return np.array(conn_comps, dtype=object)


def skeletal_distance(main_mesh, edges, *,
                      buffer,
                      distance_threshold,
                      distance_by_mesh_center,
                      connectivity,
                      keep_empty_placeholder,
                      significant_sub_components=20):
    """Per-edge mesh subtraction: which faces of `main_mesh` correspond to each skeleton edge,
    and the local skeleton-to-surface distance (width) per edge.

    Returns `(total_distances, total_distances_std, new_submesh, unique_removed_faces)` — the same
    4-tuple as the mesh_tools kernels. `new_submesh`/`unique_removed_faces` are the largest connected
    correspondence component (fast_mesh_split path). See module docstring for the two variants.
    """
    n_faces = len(main_mesh.faces)
    # precomputed ONCE per call (was rebuilt per edge inside tu.split's submesh)
    adj = _face_adjacency_csr(main_mesh, connectivity)
    face_centroids = np.asarray(main_mesh.triangles_center)   # trimesh cache, evaluated once
    verts = np.asarray(main_mesh.vertices)
    faces = np.asarray(main_mesh.faces)
    area_faces = np.asarray(main_mesh.area_faces)

    face_subtract_indices = []
    total_distances = []
    total_distances_std = []

    for i, ex_edge in enumerate(edges):
        edge_line = ex_edge[1] - ex_edge[0]
        if np.sum(np.abs(edge_line)) < 0.001:          # degenerate edge -> skip (no placeholder)
            continue

        cob_edge = sk.change_basis_matrix(edge_line)   # PRIM: 3x3 basis with edge as z-axis
        edge_trans = cob_edge @ ex_edge.T
        slice_range = np.sort(edge_trans[2, :])
        slice_range_buffer = slice_range + np.array([-buffer, buffer])

        fac_midpoints_trans = cob_edge @ face_centroids.T
        slice_mask_pre = ((fac_midpoints_trans[2, :] > slice_range_buffer[0]) &
                          (fac_midpoints_trans[2, :] < slice_range_buffer[1]))
        edge_midpoint = np.mean(edge_trans.T, axis=0)
        distance_check = np.linalg.norm((fac_midpoints_trans.T)[:, :2] - edge_midpoint[:2], axis=1) < distance_threshold
        slice_mask = slice_mask_pre & distance_check
        face_list = np.arange(0, n_faces)[slice_mask]

        if len(face_list) == 0:
            if keep_empty_placeholder:
                total_distances.append(0)
                total_distances_std.append(0)
            continue

        # connected components of the slice faces (subgraph op, ORIGINAL face indices) -- replaces
        # main_mesh.submesh([face_list]) + tu.split. Same partition, no per-edge submesh.
        comps = _slice_components(adj, face_list)
        if significant_sub_components > 0:
            comps_sig = [c for c in comps if len(c) >= significant_sub_components]
            if len(comps_sig) > 0:
                comps = comps_sig

        # component bounding boxes + which contain the whole edge (reimpl of
        # check_coordinates_inside_bounding_box: bbox = component vertex min/max, inclusive)
        bmins, bmaxs = [], []
        containing = []
        for c in comps:
            cv = verts[faces[c]].reshape(-1, 3)
            bmin = cv.min(axis=0); bmax = cv.max(axis=0)
            bmins.append(bmin); bmaxs.append(bmax)
            inside = np.all((ex_edge <= bmax) & (ex_edge >= bmin), axis=1)
            containing.append(int(np.sum(inside)) >= len(ex_edge))
        containing_indices = np.where(containing)[0]

        if len(containing_indices) != 1:
            inner = containing_indices if len(containing_indices) > 1 else np.arange(len(comps))
            edge_center = np.mean(ex_edge, axis=0)
            bbox_centers = [(bmins[k] + bmaxs[k]) / 2 for k in inner]   # == np.mean(mesh.bounds, axis=0)
            closest = np.argmin([np.linalg.norm(edge_center - bc) for bc in bbox_centers])
            edge_skeleton_faces = np.asarray(comps[inner[closest]])
        else:
            edge_skeleton_faces = np.asarray(comps[containing_indices[0]])

        face_subtract_indices.append(edge_skeleton_faces)

        # ---- local distance for this edge ----
        fac_midpoints_trans = cob_edge @ face_centroids[edge_skeleton_faces].T
        if distance_by_mesh_center:
            # area-weighted centroid of the edge's faces (reimpl of mesh_center_weighted_face_midpoints
            # over the face subset -- identical since submesh geometry == main_mesh subset geometry)
            a = area_faces[edge_skeleton_faces]
            center = np.sum(face_centroids[edge_skeleton_faces] * (a / a.sum()).reshape(-1, 1), axis=0)
            edge_midpoint = (cob_edge @ center.reshape(3, 1)).reshape(-1)
        mesh_slice_distances = np.linalg.norm((fac_midpoints_trans.T)[:, :2] - edge_midpoint[:2], axis=1)
        total_distances.append(np.mean(mesh_slice_distances))
        total_distances_std.append(np.std(mesh_slice_distances))

    # ---- fast_mesh_split: pick the largest connected correspondence component ----
    if len(face_subtract_indices) > 0:
        all_removed_faces = np.concatenate(face_subtract_indices)
        unique_removed_faces = np.array(list(set(all_removed_faces)))
        if len(unique_removed_faces) < 1:
            raise Exception(f"unique_removed_faces = {unique_removed_faces}")

        conn_face_components = intersecting_array_components(face_subtract_indices, sort_components=True)
        conn_comps_lengths = np.array([len(k) for k in conn_face_components])
        max_len_components_idx = np.where(conn_comps_lengths == np.max(conn_comps_lengths))[0]
        max_len_components_unique_faces = [
            np.unique(np.concatenate([face_subtract_indices[k] for k in conn_face_components[cmp_idx]]))
            for cmp_idx in max_len_components_idx]
        unique_removed_faces_pre = max_len_components_unique_faces[
            np.argmax([len(k) for k in max_len_components_unique_faces])]

        new_submesh = main_mesh.submesh([unique_removed_faces_pre], only_watertight=False, append=True)  # PRIM
        split_meshes, components_faces = tu.split(                                                        # PRIM
            new_submesh, return_components=True, connectivity=connectivity)
        new_submesh = split_meshes[0]
        unique_removed_faces = unique_removed_faces_pre[components_faces[0]]
    else:
        unique_removed_faces = np.array([])
        new_submesh = trimesh.Trimesh()

    return total_distances, total_distances_std, new_submesh, np.array(unique_removed_faces)


def adaptive_distance(curr_branch_skeleton, curr_branch_mesh, *,
                      skeleton_segment_width=1000,
                      distance_by_mesh_center=True,
                      distance_threshold=3000,
                      buffer=100,
                      connectivity="vertices",
                      return_closest_face_on_empty=False,
                      return_mesh_perc_drop=False):
    """Two-pass adaptive correspondence (owned reimplementation of
    `cu.mesh_correspondence_adaptive_distance`). Returns `(face_indices, width)`, or `[]` on empty
    (or `([closest_face], 0)` when `return_closest_face_on_empty`)."""
    new_skeleton = sk.resize_skeleton_branch(curr_branch_skeleton, segment_width=skeleton_segment_width)  # PRIM

    mean1, std1, corr1, idx1 = skeletal_distance(
        curr_branch_mesh, new_skeleton, buffer=buffer, distance_threshold=distance_threshold,
        distance_by_mesh_center=distance_by_mesh_center, connectivity=connectivity,
        keep_empty_placeholder=False)

    if len(idx1) == 0:
        if return_closest_face_on_empty:
            curr_nodes, _ = sk.convert_skeleton_to_nodes_edges(curr_branch_skeleton)                 # PRIM
            closest_face = tu.find_closest_face_to_coordinates(curr_branch_mesh, curr_nodes, verbose=True)  # PRIM
            return np.array([closest_face]), 0
        return []

    # adaptive threshold from pass-1 (drop endpoints, drop >1.5x median outliers)
    if len(mean1) > 4:
        fm = np.array(mean1[1:-1]); fms = np.array(std1[1:-1])
    else:
        fm = np.array(mean1); fms = np.array(std1)
    median_value = np.median(fm)
    outlier_mask = fm <= median_value * 1.5
    fm = fm[outlier_mask]; fms = fms[outlier_mask]

    buffer = 100
    total_threshold = np.max(fm) + 2 * np.max(fms)

    mean2, fms2, corr2, idx2 = skeletal_distance(
        corr1, new_skeleton, buffer=buffer, distance_threshold=total_threshold,
        distance_by_mesh_center=distance_by_mesh_center, connectivity=connectivity,
        keep_empty_placeholder=False)

    mesh_perc_drop = 1 - len(idx2) / len(idx1)

    if len(idx2) == 0:                                  # pass-2 emptied -> keep pass-1 result
        total_threshol = np.mean(mean1) + 2 * np.max(std1)
        if return_mesh_perc_drop:
            return idx1, total_threshold, mesh_perc_drop
        return idx1, total_threshol

    if return_mesh_perc_drop:
        return idx1[idx2], total_threshold, mesh_perc_drop
    return idx1[idx2], total_threshold
