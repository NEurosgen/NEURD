"""Golden gate for the WHOLE decomposition — the harness for refactoring `neurd/neuron.py`.

The soma gate covers only the soma stage and the concept-network fixtures only one
function; changes to the Neuron/Limb/Branch/Soma classes can move anything downstream.
Luckily the full pipeline on the small h01 anchor is ~3.5 min, so an end-to-end check
after every refactor step is affordable.

Reuses `neuron_metrics.extract_metrics` (the same metric set the older, now-disabled
neuron_baseline test used) rather than inventing another one.

    # record a baseline (run twice first and compare, to see how stable it is)
    python tests/tools/neuron_gate.py --out tests/fixtures/neuron_baseline_2530864375.json

    # check a refactor against it -- exit 1 on any difference outside tolerance
    python tests/tools/neuron_gate.py --baseline tests/fixtures/neuron_baseline_2530864375.json

Note the pipeline has one known nondeterminism (`np.random.choice` in the waterfill),
so `--tolerance` switches from exact comparison to the per-metric tolerances below.
Default is exact: prefer to find out that a step WAS exact rather than assume it wasn't.
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
sys.path.insert(0, str(REPO / "tests" / "tools"))

import neurd  # noqa: F401,E402  activates the numpy-2 / cgal / meshlab shims
from neurd import parameters  # noqa: E402
from neurd.segmentation_pipeline import segmentation_pipeline  # noqa: E402
from mesh_tools import trimesh_utils as tu  # noqa: E402

from neuron_metrics import extract_metrics  # noqa: E402

_ANCHOR = REPO / "Applications" / "Tutorials" / "Auto_Proof_Pipeline" / "neuron_2530864375.off"

# metric -> (kind, tol) used only with --tolerance. "exact" | "abs" | "rel"
_TOL = {
    "n_somas": ("exact", None),
    "n_limbs": ("abs", 1),
    "n_branches_total": ("rel", 0.05),
    "branches_per_limb": ("skip", None),
    "skeleton_length_total": ("rel", 0.05),
    "branch_mesh_faces_total": ("rel", 0.05),
    "soma_total_faces": ("rel", 0.02),
    "soma_faces": ("skip", None),
    "soma_centers": ("skip", None),
    "soma_volume_ratios": ("skip", None),
}


def _compare_exact(baseline, current):
    return [f"  {k}:\n    baseline = {baseline.get(k)}\n    current  = {current.get(k)}"
            for k in sorted(set(baseline) | set(current))
            if baseline.get(k) != current.get(k)]


def _compare_tolerant(baseline, current):
    problems = []
    for key, (kind, tol) in _TOL.items():
        if kind == "skip":
            continue
        b, c = baseline.get(key), current.get(key)
        if kind == "exact":
            ok = b == c
        elif kind == "abs":
            ok = abs(c - b) <= tol
        else:
            ok = abs(c - b) <= tol * abs(b) if b else c == b
        if not ok:
            problems.append(f"  {key}: baseline={b} current={c} (tolerance {kind} {tol})")
    return problems


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--neuron", default=str(_ANCHOR), help="anchor mesh (.off)")
    ap.add_argument("--data-type", default="h01", choices=["h01", "microns"])
    ap.add_argument("--segment-id", type=int, default=2530864375)
    ap.add_argument("--out", help="write the metrics JSON here")
    ap.add_argument("--baseline", help="compare against this JSON; exit 1 on any diff")
    ap.add_argument("--tolerance", action="store_true",
                    help="compare within per-metric tolerances instead of exactly")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    mesh_path = Path(args.neuron).expanduser().resolve()
    if not mesh_path.exists():
        sys.exit(f"missing mesh {mesh_path}")

    # the pipeline shells out to meshlab, which litters ./<segment_id>/ and ./temp/
    os.chdir(tempfile.mkdtemp(prefix="neuron_gate_"))
    parameters.params.use(args.data_type)

    print(f"[gate] mesh={mesh_path.name} data_type={args.data_type}")
    mesh = tu.load_mesh_no_processing(str(mesh_path))
    print(f"[gate] faces={len(mesh.faces)} verts={len(mesh.vertices)}; decomposing ...")

    neuron_obj = segmentation_pipeline(mesh, segment_id=args.segment_id,
                                       verbose=args.verbose)
    metrics = extract_metrics(neuron_obj)
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
        golden = json.loads(base.read_text())
        problems = (_compare_tolerant if args.tolerance else _compare_exact)(golden, metrics)
        if problems:
            print(f"\n[gate] FAIL — differs from {base}"
                  f"{' (within-tolerance check)' if args.tolerance else ''}:")
            print("\n".join(problems))
            sys.exit(1)
        print(f"\n[gate] PASS — matches {base}")


if __name__ == "__main__":
    main()
