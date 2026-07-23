#!/usr/bin/env python
"""Capture real I/O of the skeletal-distance / correspondence seam for a byte-exact gate.

Ahead of building a NEURD-owned skeletal-distance kernel (plan: replace the locked mesh_tools
`compartment_utils.get_skeletal_distance*` behind `_correspondence_backend`), this records the
actual `(mesh, skeleton/edges, kwargs) -> return` tuples that the two seam functions see during a
real segmentation, so the owned kernel can be asserted byte-identical without a full neuron build.

It monkeypatches the two seam entry points on `neurd._correspondence_backend`:
  * adaptive_distance_correspondence  (correspondence; sites A/B/C)
  * skeletal_distance_no_skipping     (widths; site D)
records each call's inputs + the seam's return, then delegates to the real implementation. One h01
anchor build exercises all four call sites. A curated subset (first N of each, to bound size) is
pickled to tests/fixtures/skeletal_distance_cases.pbz2 as plain numpy arrays + scalars (no trimesh
objects), so the gate can reconstruct inputs with trimesh.Trimesh(vertices=, faces=).

    python tests/tools/capture_skeletal_distance_fixtures.py [--neuron <off>] [--max-per-fn 15]
"""
import argparse
import bz2
import os
import pickle
import sys
import tempfile
from pathlib import Path

for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

ANCHOR = REPO / "Applications/Tutorials/Auto_Proof_Pipeline/neuron_2530864375.off"
OUT = REPO / "tests" / "fixtures" / "skeletal_distance_cases.pbz2"


def _mesh_rec(m):
    """Trimesh -> picklable {vertices, faces}; None/empty tolerated."""
    if m is None:
        return None
    import numpy as np
    try:
        return dict(vertices=np.asarray(m.vertices, dtype=np.float64),
                    faces=np.asarray(m.faces, dtype=np.int64))
    except Exception:
        return None


def _scalar_kwargs(kwargs, drop):
    """Keep only picklable scalar/str/None kwargs (drop the mesh/skeleton/edges entries)."""
    out = {}
    for k, v in kwargs.items():
        if k in drop:
            continue
        if isinstance(v, (int, float, bool, str, type(None))):
            out[k] = v
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--neuron", default=str(ANCHOR))
    ap.add_argument("--data-type", default="h01")
    ap.add_argument("--max-per-fn", type=int, default=15,
                    help="cap captured cases per function (bounds fixture size)")
    args = ap.parse_args()

    os.chdir(tempfile.mkdtemp(prefix="skel_capture_"))

    import numpy as np  # noqa
    import neurd  # noqa: F401  activate shims
    from neurd import parameters
    from neurd import _correspondence_backend as cb
    from neurd.segmentation_pipeline import segmentation_pipeline
    from mesh_tools import trimesh_utils as tu

    parameters.params.use(args.data_type)

    cases = {"adaptive": [], "no_skipping": []}
    counts = {"adaptive": 0, "no_skipping": 0}

    _real_adaptive = cb.adaptive_distance_correspondence
    _real_no_skip = cb.skeletal_distance_no_skipping

    def cap_adaptive(*a, **kw):
        out = _real_adaptive(*a, **kw)
        counts["adaptive"] += 1
        if len(cases["adaptive"]) < args.max_per_fn:
            skeleton = kw.get("curr_branch_skeleton", a[0] if len(a) > 0 else None)
            mesh = kw.get("curr_branch_mesh", a[1] if len(a) > 1 else None)
            # normalize output: None, or (indices, width)
            if out is None:
                out_rec = None
            else:
                try:
                    idx, width = out
                    out_rec = (np.asarray(idx), float(width) if np.ndim(width) == 0 else np.asarray(width))
                except Exception:
                    out_rec = ("RAW", repr(out)[:200])
            cases["adaptive"].append(dict(
                skeleton=np.asarray(skeleton) if skeleton is not None else None,
                mesh=_mesh_rec(mesh),
                kwargs=_scalar_kwargs(kw, drop=("curr_branch_skeleton", "curr_branch_mesh")),
                n_pos=len(a), out=out_rec))
        return out

    def cap_no_skip(*a, **kw):
        out = _real_no_skip(*a, **kw)
        counts["no_skipping"] += 1
        if len(cases["no_skipping"]) < args.max_per_fn:
            mesh = kw.get("main_mesh", a[0] if len(a) > 0 else None)
            edges = kw.get("edges", a[1] if len(a) > 1 else None)
            try:
                distances, std, _submesh, indices = out
                out_rec = dict(distances=np.asarray(distances, dtype=np.float64),
                               std=np.asarray(std, dtype=np.float64),
                               indices=np.asarray(indices))
            except Exception:
                out_rec = dict(raw=repr(out)[:200])
            cases["no_skipping"].append(dict(
                edges=np.asarray(edges) if edges is not None else None,
                mesh=_mesh_rec(mesh),
                kwargs=_scalar_kwargs(kw, drop=("main_mesh", "edges")),
                out=out_rec))
        return out

    cb.adaptive_distance_correspondence = cap_adaptive
    cb.skeletal_distance_no_skipping = cap_no_skip

    print(f"[capture] building {Path(args.neuron).name} (data_type={args.data_type}) ...", flush=True)
    mesh = tu.load_mesh_no_processing(str(args.neuron))
    segmentation_pipeline(mesh, verbose=False)

    cb.adaptive_distance_correspondence = _real_adaptive
    cb.skeletal_distance_no_skipping = _real_no_skip

    OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(neuron=Path(args.neuron).stem, data_type=args.data_type,
                   total_calls=counts, cases=cases)
    with bz2.open(OUT, "wb") as f:
        pickle.dump(payload, f)

    size_mb = OUT.stat().st_size / 1e6
    print(f"\n[capture] total calls: adaptive={counts['adaptive']} no_skipping={counts['no_skipping']}")
    print(f"[capture] saved {len(cases['adaptive'])} adaptive + {len(cases['no_skipping'])} no_skipping "
          f"cases -> {OUT}  ({size_mb:.1f} MB)")
    # quick variety report
    for fn in ("adaptive", "no_skipping"):
        nfaces = [c["mesh"]["faces"].shape[0] if c["mesh"] else 0 for c in cases[fn]]
        none_out = sum(1 for c in cases[fn] if c["out"] is None)
        print(f"  {fn}: mesh_faces min/max = {min(nfaces) if nfaces else 0}/{max(nfaces) if nfaces else 0}"
              f"  None-outputs={none_out}")


if __name__ == "__main__":
    main()
