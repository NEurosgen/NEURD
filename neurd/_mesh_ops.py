"""In-process replacements for the meshlabserver subprocess mesh operations.

The slim soma stage shells out to `xvfb-run meshlabserver` ~13× per identification
(decimation / Poisson / fill-holes), each spawn paying X-server startup + meshlab plugin
load + ASCII OFF round-trip to disk — on tiny meshes. That subprocess overhead, not the
geometry, dominates the ~283 s soma stage (the SDF rays themselves are <0.5 s).

These functions do the same operations in-process via open3d / trimesh: `mesh -> mesh`,
no disk, no subprocess. They are validated two ways:
  - fast unit tests on synthetic meshes (tests/unit/test_mesh_ops.py),
  - fidelity vs captured meshlab output on real soma meshes (tests/integration/...).

Kept deliberately decoupled from the meshlab wrappers' `(mesh, file_obj)` return contract:
call sites that thread temp filenames are refactored separately when swapped in.
"""

import numpy as np


def _to_o3d(mesh):
    """trimesh.Trimesh -> open3d.geometry.TriangleMesh (vertices + faces only)."""
    import open3d as o3d

    return o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(mesh.vertices, dtype=np.float64)),
        o3d.utility.Vector3iVector(np.asarray(mesh.faces, dtype=np.int32)),
    )


def _to_trimesh(o3d_mesh):
    """open3d.geometry.TriangleMesh -> trimesh.Trimesh (no reprocessing)."""
    import trimesh

    return trimesh.Trimesh(
        vertices=np.asarray(o3d_mesh.vertices),
        faces=np.asarray(o3d_mesh.triangles),
        process=False,
    )


def decimate(mesh, decimation_ratio=0.25, step_ratio=0.5):
    """Quadric decimation to ~`decimation_ratio` of the face count, in-process.

    Replaces `meshlab.Decimator` / `tu.decimate`. open3d's quadric decimation targets
    an absolute triangle count, so we scale by the current face count.

    **Cascaded**: open3d's cost is strongly SUPERLINEAR in how far one call collapses (its
    error-priority queue is rebuilt/re-heapified over the whole mesh), so reaching the target in
    halving steps is much cheaper than one big jump -- measured on the 963k-face component of a
    187 MB H01 neuron: 49.5 s in one call vs 18.1 s in two, same 240,754-face output. The soma
    stage's outer decimation (the full ~3.6M-face mesh at ratio 0.25) is ~100 s of a build's wall.
    `step_ratio` is the per-call reduction; set `NEURD_DECIMATE_SINGLE_STAGE=1` to restore the
    single-call behaviour (A/B, fidelity checks). Quadric decimation is greedy either way, so the
    cascaded surface is not bit-identical to the single-call one -- it is the same algorithm
    applied in stages, and is checked against the meshlab golden in
    tests/integration/test_mesh_ops_fidelity.py.
    """
    import os

    n = len(mesh.faces)
    target = max(4, int(round(n * decimation_ratio)))
    out = _to_o3d(mesh)
    if os.environ.get("NEURD_DECIMATE_SINGLE_STAGE") != "1" and 0 < step_ratio < 1:
        # intermediate rungs; the loop stops while one more `step_ratio` would overshoot `target`
        while n * step_ratio > target:
            n = max(target, int(round(n * step_ratio)))
            out = out.simplify_quadric_decimation(target_number_of_triangles=n)
    out = out.simplify_quadric_decimation(target_number_of_triangles=target)
    out.remove_unreferenced_vertices()
    return _to_trimesh(out)


def fill_holes(mesh, max_hole_size=2000):
    """Close holes in-process via trimesh.

    Replaces `meshlab.FillHoles` / `tu.fill_holes` — which currently FAILS on this
    meshlab build (returncode 255, filter not found) and silently no-ops, so any
    real hole-filling here is a strict improvement. `max_hole_size` is accepted for
    signature parity; trimesh fills triangular/quad boundary holes regardless.
    """
    m = mesh.copy()
    m.merge_vertices()
    m.fill_holes()
    return m


def poisson_surface_reconstruction(mesh, depth=9, n_points=None):
    """Screened Poisson surface reconstruction in-process via open3d.

    Replaces `meshlab.Poisson` / `tu.poisson_surface_reconstruction`. Samples an
    oriented point cloud from the input surface, then reconstructs a watertight shell.
    Callers downstream already pick the largest connected component, so we return the
    raw reconstruction.
    """
    import open3d as o3d

    o = _to_o3d(mesh)
    o.compute_vertex_normals()
    n = n_points or max(2000, len(mesh.vertices))
    pcd = o.sample_points_poisson_disk(number_of_points=n)
    if not pcd.has_normals():
        pcd.estimate_normals()
    pcd.orient_normals_consistent_tangent_plane(k=10)
    rec, _density = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth
    )
    rec.remove_unreferenced_vertices()
    return _to_trimesh(rec)


def poisson_surface_reconstruction_meshlab(
    mesh,
    depth=11, full_depth=6, cg_depth=0, scale=1.1,
    samples_per_node=1.5, point_weight=4.0, iters=8,
    confidence=False, pre_clean=False,
):
    """Screened Poisson via pymeshlab — the SAME MeshLab filter + parameters the
    original Docker pipeline used (the values the fork's __init__ deleted when it
    replaced Poisson with a no-op). In-process, no subprocess / Docker / xvfb.

    Why this matters: H01 EM meshes have gaps/self-contacts; the original pipeline
    relied on this reconstruction to produce a clean, connected surface. Without it the
    mesh-correspondence step can hand `correspondence_1_to_1` a disconnected limb mesh
    and the pipeline raises ("not just one mesh") on the more complex neurons. open3d's
    Poisson (above) sub-samples points and loses thin neurites (over-segments); this
    uses MeshLab's actual algorithm/params, so it matches the Docker output.
    """
    import numpy as _np
    import trimesh as _tm
    import pymeshlab as _ml

    ms = _ml.MeshSet()
    ms.add_mesh(_ml.Mesh(
        _np.asarray(mesh.vertices, dtype=_np.float64),
        _np.asarray(mesh.faces, dtype=_np.int32),
    ))
    # Screened Poisson requires every vertex to carry a proper (non-null) normal.
    # Real EM meshes have unreferenced/degenerate vertices, so clean + recompute first;
    # pre_clean lets MeshLab drop any that remain (rather than aborting the filter).
    for _f in ("meshing_remove_unreferenced_vertices",
               "meshing_remove_duplicate_vertices",
               "compute_normal_per_vertex"):
        try:
            getattr(ms, _f)()
        except Exception:
            pass
    ms.generate_surface_reconstruction_screened_poisson(
        depth=int(depth), fulldepth=int(full_depth), cgdepth=int(cg_depth),
        scale=float(scale), samplespernode=float(samples_per_node),
        pointweight=float(point_weight), iters=int(iters),
        confidence=bool(confidence), preclean=True,
    )
    rec = ms.current_mesh()
    return _tm.Trimesh(
        vertices=_np.asarray(rec.vertex_matrix()),
        faces=_np.asarray(rec.face_matrix()),
        process=False,
    )
