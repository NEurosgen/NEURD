"""Golden gate for the SOMA STAGE ONLY — the cheap harness for refactoring
`neurd/soma_extraction_utils.extract_soma_center`.

Why a dedicated gate: the full `tests/integration/test_segmentation_pipeline.py` run
decomposes the whole neuron (skeletonization, correspondence, ...) and takes minutes to
an hour. Every step of the `extract_soma_center` decomposition only needs to know that
the SOMAS came out identical, and that stage is a small fraction of the total. This runs
just it, dumps a comparable JSON, and diffs against a stored baseline.

Two entry points, because the function has two callers with DIFFERENT argument shapes:

  --entry preprocess  (default)  extract_soma_center(segment_id, mesh.vertices, mesh.faces)
                                 -> exactly what `preprocess_neuron._extract_single_soma`
                                    does, i.e. the production hot path. Note it passes
                                    verts/faces, so `extract_soma_center` rebuilds a
                                    trimesh internally (mesh is None branch).
  --entry stage                  soma_indentification(mesh) -> the StageProducts wrapper
                                 (only used by tests/tools/capture_mesh_op_fixtures.py).
                                 Cover this too when changing the RETURN SHAPE.

Usage:

    # record a baseline (run twice first and diff, to prove the stage is deterministic)
    python tests/tools/soma_gate.py --neuron <mesh.off> --data-type h01 \
        --out tests/fixtures/soma_baseline_<name>.json

    # check a refactor against it — exit code 1 on any difference
    python tests/tools/soma_gate.py --neuron <mesh.off> --data-type h01 \
        --baseline tests/fixtures/soma_baseline_<name>.json

`run_time` is deliberately NOT part of the compared metrics.
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

import neurd  # noqa: F401,E402  activates the numpy-2 / cgal / meshlab shims
from neurd import parameters  # noqa: E402
from neurd import soma_extraction_utils as sm  # noqa: E402
from mesh_tools import trimesh_utils as tu  # noqa: E402


def _mesh_metrics(mesh):
    """Identity of one soma mesh: size + geometry, rounded to survive float noise."""
    return {
        "n_faces": int(len(mesh.faces)),
        "n_vertices": int(len(mesh.vertices)),
        "center": [round(float(c), 3) for c in np.asarray(mesh.vertices).mean(axis=0)],
        "bounds": [[round(float(c), 3) for c in row] for row in np.asarray(mesh.bounds)],
    }


def _unpack(result):
    """Accept the 5-tuple (with glia/nuclei) and the 3-tuple it becomes after step 1a."""
    if len(result) == 5:
        somas, run_time, sdfs, glia, nuclei = result
        return somas, run_time, sdfs, len(glia), len(nuclei)
    somas, run_time, sdfs = result
    return somas, run_time, sdfs, None, None


def extract_soma_metrics(mesh, segment_id=12345, entry="preprocess", verbose=False):
    """Run the soma stage and return a JSON-serializable metrics dict."""
    if entry == "preprocess":
        # mirrors preprocess_neuron._extract_single_soma
        result = sm.extract_soma_center(segment_id, mesh.vertices, mesh.faces)
        somas, _, sdfs, n_glia, n_nuclei = _unpack(result)
    elif entry == "stage":
        products = sm.soma_indentification(mesh, verbose=verbose)
        somas = products["soma_meshes"]
        sdfs = products["soma_sdfs"]
        n_glia = len(products["glia_meshes"]) if "glia_meshes" in products else None
        n_nuclei = len(products["nuclei_meshes"]) if "nuclei_meshes" in products else None
    else:
        raise ValueError(f"unknown entry {entry!r}")

    metrics = {
        "entry": entry,
        "n_somas": int(len(somas)),
        "somas": [_mesh_metrics(m) for m in somas],
        "sdfs": [round(float(s), 6) for s in np.asarray(sdfs).tolist()],
    }
    # Informational only (see _INFORMATIONAL): recorded while the glia/nuclei layer
    # still exists, absent once it is removed.
    if n_glia is not None:
        metrics["n_glia_pieces"] = n_glia
        metrics["n_nuclei_pieces"] = n_nuclei
    return metrics


# Not part of the soma deliverable and read by no caller (`preprocess_neuron` discards
# both; the StageProducts entries have no consumer), so they are printed but NOT
# compared — otherwise removing the glia/nuclei layer would trip the gate on a value
# nothing depends on. `n_nuclei_pieces` is genuinely non-zero: the glia/nuclei REMOVAL
# call is dead (NameError), but `original_mesh_soma`'s inside-pieces still accumulate
# into `nuclei_pieces` during soma backtracking.
_INFORMATIONAL = ("entry", "n_glia_pieces", "n_nuclei_pieces")


def _compare(baseline, current):
    """Return a list of human-readable differences (empty == identical)."""
    diffs = []
    for key in sorted(set(baseline) | set(current)):
        if key in _INFORMATIONAL:
            continue
        b, c = baseline.get(key, "<missing>"), current.get(key, "<missing>")
        if b != c:
            diffs.append(f"  {key}:\n    baseline = {b}\n    current  = {c}")
    return diffs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--neuron", required=True, help="path to the .off mesh anchor")
    ap.add_argument("--data-type", default="h01", choices=["h01", "microns"])
    ap.add_argument("--segment-id", type=int, default=12345)
    ap.add_argument("--entry", default="preprocess", choices=["preprocess", "stage"])
    ap.add_argument("--out", help="write the metrics JSON here")
    ap.add_argument("--baseline", help="compare against this JSON; exit 1 on any diff")
    ap.add_argument("--repeat", type=int, default=1,
                    help="run N times and require all runs identical (determinism check)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    mesh_path = Path(args.neuron).expanduser().resolve()
    if not mesh_path.exists():
        sys.exit(f"missing mesh {mesh_path}")

    # meshlab shells out and litters ./<segment_id>/ and ./temp/ in the CWD.
    os.chdir(tempfile.mkdtemp(prefix="soma_gate_"))
    parameters.params.use(args.data_type)

    print(f"[gate] mesh={mesh_path.name} data_type={args.data_type} entry={args.entry}")
    mesh = tu.load_mesh_no_processing(str(mesh_path))
    print(f"[gate] faces={len(mesh.faces)} verts={len(mesh.vertices)}")

    runs = []
    for i in range(args.repeat):
        metrics = extract_soma_metrics(
            mesh, segment_id=args.segment_id, entry=args.entry, verbose=args.verbose
        )
        runs.append(metrics)
        print(f"[gate] run {i}: n_somas={metrics['n_somas']} "
              f"faces={[s['n_faces'] for s in metrics['somas']]} sdfs={metrics['sdfs']}")

    metrics = runs[0]
    if args.repeat > 1:
        unstable = [i for i, r in enumerate(runs[1:], 1) if _compare(runs[0], r)]
        if unstable:
            print(f"[gate] NON-DETERMINISTIC: runs {unstable} differ from run 0")
            for i in unstable:
                print("\n".join(_compare(runs[0], runs[i])))
            sys.exit(2)
        print(f"[gate] deterministic across {args.repeat} runs")

    print("\n" + json.dumps(metrics, indent=2))

    if args.out:
        out = Path(args.out)
        if not out.is_absolute():
            out = REPO / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(metrics, indent=2))
        print(f"\n[gate] written -> {out}")

    if args.baseline:
        base = Path(args.baseline)
        if not base.is_absolute():
            base = REPO / base
        diffs = _compare(json.loads(base.read_text()), metrics)
        if diffs:
            print(f"\n[gate] FAIL — differs from {base}:")
            print("\n".join(diffs))
            sys.exit(1)
        print(f"\n[gate] PASS — identical to {base}")


if __name__ == "__main__":
    main()
