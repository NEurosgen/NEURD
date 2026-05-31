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

import numpy as np


def _skeleton_length(skel):
    """Total length of a branch/limb skeleton stored as edges, shape (n_edges, 2, 3)."""
    if skel is None:
        return 0.0
    skel = np.asarray(skel)
    if skel.ndim != 3 or skel.shape[0] == 0:
        return 0.0
    return float(np.linalg.norm(skel[:, 1, :] - skel[:, 0, :], axis=1).sum())


def extract_metrics(neuron_obj):
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

    return {
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
