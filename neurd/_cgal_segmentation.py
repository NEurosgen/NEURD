"""Pure-Python stand-in for the compiled CGAL `cgal_Segmentation_Module`.

The original was a C++/CGAL extension built inside the (now removed) Docker image;
rebuilding it locally would reintroduce the heavy platform-specific toolchain this
fork is shedding. This module reproduces its on-disk contract in pure Python so the
segmentation pipeline runs on a plain conda/venv env.

Contract (mirrors `mesh_tools.trimesh_utils.mesh_segmentation`, which constructs the
expected filenames itself and reads them back):
  {filepath_no_ext}-cgal_{round(clusters,2)}_{smoothness:.2f}.csv      int cluster id / face
  {filepath_no_ext}-cgal_{round(clusters,2)}_{smoothness:.2f}_sdf.csv  float SDF / face

SDF — inward ray-traced thickness per face via `tu.ray_trace_distance` (same method
as CGAL::sdf_values), then **linearly normalized to [0,1]** to match CGAL's output:
soma detection compares SDF medians against a hardcoded threshold (soma_width_threshold
= 0.32 in soma_extraction_utils) tuned for that [0,1] range, so raw mesh-unit distances
would silently break it.

Clustering — KMeans on log(SDF). The `smoothness` arg (CGAL's MRF lambda) is accepted
for signature compatibility but unused: there is no spatial/MRF smoothing, so adjacent
faces may land in different clusters. Acceptable for soma extraction, where the SDF
contrast between the round soma and thin processes is large. Downstream the seam splits
each cluster into connected components anyway, so the exact cluster count is not load-bearing.
"""

import numpy as np


def cgal_segmentation(filepath_no_ext, clusters=2, smoothness=0.2, *args, **kwargs):
    # Lazy imports: trimesh_utils imports this module's name at load time, and
    # sklearn is only needed when segmentation actually runs.
    from mesh_tools import trimesh_utils as tu
    from sklearn.cluster import KMeans

    mesh = tu.load_mesh_no_processing(str(filepath_no_ext) + ".off")

    sdf = np.asarray(
        tu.ray_trace_distance(mesh, replace_zero_values_with_center_distance=True),
        dtype=float,
    )

    # Match CGAL: linearly normalize SDF to [0, 1], but anchor on the 2nd/98th
    # percentiles rather than raw min/max. A single ray that passes clean through
    # the mesh yields a huge outlier distance that, under plain min/max, inflates
    # the denominator and crushes every value downward — the soma segment's median
    # then lands near the 0.32 soma_width_threshold and detection becomes fragile.
    # Robust anchoring restores the soma's margin (median ~0.6 vs CGAL ~0.77).
    lo, hi = np.percentile(sdf, [2, 98])
    span = hi - lo
    sdf_norm = np.clip((sdf - lo) / span, 0.0, 1.0) if span > 0 else np.zeros_like(sdf)

    n_clusters = max(1, int(clusters))
    feats = np.log(sdf + 1e-9).reshape(-1, 1)
    if n_clusters == 1 or len(np.unique(feats)) < n_clusters:
        labels = np.zeros(len(sdf), dtype=int)
    else:
        # n_init=1: features are 1-D (log SDF), where k-means++ from a fixed seed
        # reaches the same optimum as multi-restart — so 10 restarts were ~10x wasted
        # work (the dominant self-time in the spine/soma segmentation profile). Cluster
        # exactness is not load-bearing here (see module docstring); random_state keeps
        # it deterministic.
        labels = KMeans(n_clusters=n_clusters, n_init=1, random_state=0).fit_predict(feats)

    base = f"{filepath_no_ext}-cgal_{np.round(clusters, 2)}_{smoothness:.2f}"
    np.savetxt(base + ".csv", labels.astype(int), fmt="%d")
    np.savetxt(base + "_sdf.csv", sdf_norm, fmt="%.6f")
