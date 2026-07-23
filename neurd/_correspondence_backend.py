"""The single seam to the fragile `mesh_tools` correspondence/skeletonization backend.

Every call into `mesh_tools.compartment_utils` (mesh<->skeleton correspondence, waterfill
label expansion) and `mesh_tools.meshparty_skeletonize` (the skeletonizer) goes through
here. Two reasons this isolation is worth its own file:

1. These ~7 functions are the fragile, swap-worthy interactions -- to move the pipeline
   onto a different backend (skeletor, a CGAL correspondence, a meshparty-native map) you
   rewrite THIS file and nothing else. The stable `sk`/`tu` mesh+skeleton primitives are
   NOT here; they are not the fragile surface (see LOGIC_INVESTIGATION.md, Area E).

2. The pipeline's ONLY nondeterminism lives in one of these -- the `np.random` tie-break
   inside `waterfill_starting_label_to_soma_border` (`compartment_utils:1187`). Routing it
   through `waterfill_to_soma_border(..., rng=...)` makes that the single controllable
   point.

Wrappers are pass-throughs (exact args/returns preserved) except
`adaptive_distance_correspondence`, which normalizes the underlying variadic-tuple return
(`()` on failure / `(indices, width)` on success) to `None`-on-failure so call sites stop
branching on `len(result)`.
"""

import os as _os

from mesh_tools import compartment_utils as _cu
from mesh_tools import meshparty_skeletonize as _m_sk

from . import skeletal_distance_ops as _sdo


def _use_owned_skeletal_distance():
    """Feature flag: route the two skeletal-distance seams to the NEURD-owned kernel
    (`neurd/skeletal_distance_ops.py`) instead of the locked mesh_tools originals. Default OFF —
    production is unchanged; set NEURD_OWNED_SKELETAL_DISTANCE=1 for A/B. The owned kernel is
    gate-verified byte-exact (tests/unit/test_skeletal_distance.py)."""
    return _os.environ.get("NEURD_OWNED_SKELETAL_DISTANCE") == "1"


_ADAPT_OWNED_KEYS = ("skeleton_segment_width", "distance_by_mesh_center", "distance_threshold",
                     "buffer", "connectivity", "return_closest_face_on_empty", "return_mesh_perc_drop")


# --------------------------------------------------------------- compartment_utils (cu)

def adaptive_distance_correspondence(*args, **kwargs):
    """Assign limb-mesh faces to a branch skeleton by adaptive distance.

    Wraps `cu.mesh_correspondence_adaptive_distance` (or the owned `sdo.adaptive_distance` under
    the feature flag). Returns `(remaining_indices, width)` on success, or `None` when
    correspondence could not be found (the underlying function returns an empty tuple there).
    """
    if _use_owned_skeletal_distance():
        skeleton = kwargs.get("curr_branch_skeleton", args[0] if len(args) > 0 else None)
        mesh = kwargs.get("curr_branch_mesh", args[1] if len(args) > 1 else None)
        owned_kw = {k: v for k, v in kwargs.items() if k in _ADAPT_OWNED_KEYS}
        result = _sdo.adaptive_distance(skeleton, mesh, **owned_kw)
    else:
        result = _cu.mesh_correspondence_adaptive_distance(*args, **kwargs)
    if len(result) == 0:
        return None
    return result


def resolve_face_labels(*args, **kwargs):
    """Resolve empty/conflicting per-face branch labels (closest-skeleton partition).

    Wraps `cu.resolve_empty_conflicting_face_labels`. Pass-through, including its raise
    contract (it raises when a skeleton segment cannot claim a connected face patch).
    """
    return _cu.resolve_empty_conflicting_face_labels(*args, **kwargs)


def waterfill_to_soma_border(*args, **kwargs):
    """Grow a starting label out to the soma border by waterfilling.

    Wraps `cu.waterfill_starting_label_to_soma_border`. This is the pipeline's ONLY source
    of nondeterminism (a `np.random` tie-break inside); it is centralized here so a seed
    can later be injected at this one point.
    """
    return _cu.waterfill_starting_label_to_soma_border(*args, **kwargs)


def skeletal_distance_no_skipping(*args, **kwargs):
    """Per-face skeletal distance along a branch. Wraps `cu.get_skeletal_distance_no_skipping`
    (or the owned `sdo.skeletal_distance(keep_empty_placeholder=True)` under the feature flag)."""
    if _use_owned_skeletal_distance():
        mesh = kwargs.get("main_mesh", args[0] if len(args) > 0 else None)
        edges = kwargs.get("edges", args[1] if len(args) > 1 else None)
        return _sdo.skeletal_distance(
            mesh, edges,
            buffer=kwargs.get("buffer", 0.01),
            distance_threshold=kwargs.get("distance_threshold", 3000),
            distance_by_mesh_center=kwargs.get("distance_by_mesh_center", False),
            connectivity=kwargs.get("connectivity", "edges"),
            keep_empty_placeholder=True)
    return _cu.get_skeletal_distance_no_skipping(*args, **kwargs)


# ------------------------------------------------------- meshparty_skeletonize (m_sk)

def skeletonize_largest_component(*args, **kwargs):
    """Skeletonize a limb mesh's largest component. Wraps `m_sk.skeletonize_mesh_largest_component`."""
    return _m_sk.skeletonize_mesh_largest_component(*args, **kwargs)


def skeleton_to_branches(*args, **kwargs):
    """Split a skeleton object into branch skeletons + submeshes. Wraps `m_sk.skeleton_obj_to_branches`."""
    return _m_sk.skeleton_obj_to_branches(*args, **kwargs)


def weighted_width_median(*args, **kwargs):
    """Branch-length-weighted median width. Wraps `m_sk.width_median_weighted`."""
    return _m_sk.width_median_weighted(*args, **kwargs)
