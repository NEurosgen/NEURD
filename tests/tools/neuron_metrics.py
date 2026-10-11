"""Extract a compact, comparable set of FINAL-output metrics from a decomposed Neuron.

Used by the Neuron-output baseline: a golden snapshot of these metrics (on the fixture
mesh, with the current pipeline) is the reference any OUTPUT-CHANGING optimization
(e.g. Decimator -> open3d, a different SDF/skeletonizer) must stay close to. Unlike the
characterization test (which only checks the structural contract: >=1 soma, limbs,
branches with mesh+skeleton), this captures *how much / where* — so a "different but
fine" change can be told apart from a silent degradation.

Metrics are aggregates chosen to be meaningful yet robust to the pipeline's minor
run-to-run nondeterminism. Compare with tolerances, never exact equality.
"""

import hashlib

import numpy as np


def _skeleton_length(skel):
    """Total length of a branch/limb skeleton stored as edges, shape (n_edges, 2, 3)."""
    if skel is None:
        return 0.0
    skel = np.asarray(skel)
    if skel.ndim != 3 or skel.shape[0] == 0:
        return 0.0
    return float(np.linalg.norm(skel[:, 1, :] - skel[:, 0, :], axis=1).sum())


def _triangle_rows(mesh):
    """Canonical exact float64 triangles, retaining their parent face order."""
    triangles = np.asarray(mesh.triangles, dtype=np.float64).copy()
    vertices = triangles.view(np.dtype((np.void, 24))).reshape(-1, 3)
    vertices.sort(axis=1)
    return vertices.copy().view(np.dtype((np.void, 72))).reshape(-1)


def extract_spine_metrics(neuron_obj):
    """Check that detected spines are disjoint pieces of their branch surface.

    The digest records ordered branch-local face masks, not rounded geometry.
    Duplicate parent triangles are reported as ambiguous rather than treated as
    reliable provenance. These checks concern mesh consistency, not anatomy.
    """
    counts = dict(n_spines_total=0, branches_with_spines=0,
                  spine_mesh_faces_total=0, invalid_parent_faces_total=0,
                  ambiguous_parent_faces_total=0, overlapping_parent_faces_total=0,
                  overlapping_spine_surface_faces_total=0,
                  invalid_spine_face_indices_total=0, spines_with_face_indices=0,
                  invalid_spine_volumes_total=0)
    digest = hashlib.sha256()
    for limb_index, limb in enumerate(neuron_obj.limbs):
        for branch_index, branch in enumerate(limb.branches):
            spines = getattr(branch, "spines", None)
            spines = [] if spines is None else list(spines)
            if not spines:
                continue
            counts["branches_with_spines"] += 1
            counts["n_spines_total"] += len(spines)
            parent = _triangle_rows(branch.mesh)
            order = np.argsort(parent)
            ordered = parent[order]
            used = set()
            used_surface = set()
            masks = getattr(branch, "spines_face_idx", None)
            for spine_index, spine in enumerate(spines):
                mesh = getattr(spine, "mesh", spine)
                keys = _triangle_rows(mesh)
                counts["spine_mesh_faces_total"] += len(keys)
                surface = {key.tobytes() for key in keys}
                counts["overlapping_spine_surface_faces_total"] += len(used_surface & surface)
                used_surface.update(surface)
                if masks is not None:
                    if len(masks) != len(spines):
                        counts["invalid_spine_face_indices_total"] += 1
                        continue
                    indices = np.asarray(masks[spine_index])
                    if (indices.ndim != 1 or indices.dtype.kind not in "iu"
                            or len(indices) != len(keys) or np.any(indices < 0)
                            or np.any(indices >= len(parent))):
                        counts["invalid_spine_face_indices_total"] += 1
                        continue
                    counts["spines_with_face_indices"] += 1
                    faces = np.sort(indices)
                    if not np.array_equal(np.sort(parent[faces]), np.sort(keys)):
                        counts["invalid_parent_faces_total"] += len(keys)
                else:
                    left = np.searchsorted(ordered, keys, side="left")
                    right = np.searchsorted(ordered, keys, side="right")
                    present = right > left
                    counts["invalid_parent_faces_total"] += int((~present).sum())
                    counts["ambiguous_parent_faces_total"] += int(((right - left) > 1).sum())
                    faces = np.sort(order[left[present]])
                unique = set(int(i) for i in faces)
                counts["overlapping_parent_faces_total"] += len(used & unique)
                # Repeated faces inside one spine are also invalid provenance.
                counts["overlapping_parent_faces_total"] += len(faces) - len(unique)
                used.update(unique)
                digest.update(f"{limb_index}:{branch_index}:{spine_index}:{len(faces)}:".encode())
                digest.update(faces.astype("<i8").tobytes())
            volumes = getattr(branch, "spines_volume", None)
            if volumes is None:
                counts["invalid_spine_volumes_total"] += len(spines)
            else:
                volumes = np.asarray(volumes, dtype=float)
                counts["invalid_spine_volumes_total"] += abs(len(volumes) - len(spines))
                counts["invalid_spine_volumes_total"] += int((~np.isfinite(volumes) | (volumes <= 0)).sum())
    counts["spine_face_masks_sha256"] = digest.hexdigest()
    return counts


def extract_metrics(neuron_obj, *, include_spines=False):
    """Return a JSON-serializable dict of final-output metrics for `neuron_obj`."""
    # --- somas ---
    soma_names = sorted(neuron_obj.get_soma_node_names())
    soma_data = [neuron_obj.concept_network.nodes[k]["data"] for k in soma_names]
    soma_faces = sorted(int(len(s.mesh.faces)) for s in soma_data)
    soma_centers = sorted(
        [round(float(c), 3) for c in np.asarray(s.mesh_center).tolist()] for s in soma_data
    )
    soma_volume_ratios = sorted(round(float(getattr(s, "volume_ratio", float("nan"))), 4) for s in soma_data)

    # --- limbs / branches / skeleton ---
    limbs = list(getattr(neuron_obj, "limbs", []))
    branches_per_limb, skel_len_total, branch_faces_total, n_branches = [], 0.0, 0, 0
    for lb in limbs:
        brs = list(getattr(lb, "branches", []))
        branches_per_limb.append(len(brs))
        for br in brs:
            n_branches += 1
            m = getattr(br, "mesh", None)
            if m is not None and hasattr(m, "faces"):
                branch_faces_total += int(len(m.faces))
            skel_len_total += _skeleton_length(getattr(br, "skeleton", None))

    metrics = {
        "n_somas": len(soma_names),
        "soma_total_faces": int(sum(soma_faces)),
        "soma_faces": soma_faces,
        "soma_centers": soma_centers,
        "soma_volume_ratios": soma_volume_ratios,
        "n_limbs": len(limbs),
        "n_branches_total": int(n_branches),
        "branches_per_limb": sorted(branches_per_limb),
        "skeleton_length_total": round(skel_len_total, 2),
        "branch_mesh_faces_total": int(branch_faces_total),
    }
    if include_spines:
        metrics.update(extract_spine_metrics(neuron_obj))
    return metrics
