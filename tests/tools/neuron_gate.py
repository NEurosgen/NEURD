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
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests" / "tools"))

import neurd  # noqa: F401,E402  activates the numpy-2 / cgal / meshlab shims
from neurd.segmentation_pipeline import segmentation_pipeline  # noqa: E402
from neurd.run_config import add_run_arguments, config_from_args  # noqa: E402
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
    if ("n_spines_total" in baseline) != ("n_spines_total" in current):
        problems.append("  baseline and current spine modes differ; compare matching spine-enabled or disabled runs")
    elif "n_spines_total" in current:
        b, c = baseline["n_spines_total"], current["n_spines_total"]
        if abs(c - b) > max(1, .1 * b):
            problems.append(f"  n_spines_total: baseline={b} current={c} (10% or 1 spine)")
    return problems


def _spine_problems(metrics, require_spines=False):
    problems = []
    for key in ("invalid_parent_faces_total", "ambiguous_parent_faces_total",
                "overlapping_parent_faces_total", "overlapping_spine_surface_faces_total",
                "invalid_spine_face_indices_total", "invalid_spine_volumes_total"):
        if metrics.get(key, 0):
            problems.append(f"  {key}={metrics[key]}")
    if require_spines and metrics.get("n_spines_total", 0) == 0:
        problems.append("  this spine-positive fixture produced no spines")
    if metrics.get("spines_with_face_indices", 0) != metrics.get("n_spines_total", 0):
        problems.append("  detected spines lack explicit parent face indices")
    return problems


def _provenance(mesh_path, config, elapsed):
    import cgal_Segmentation_Module as provider
    from neurd.h01_batch import digest_json, sha
    from process_all_neurons import _run_signature

    signature = _run_signature(config, 0.0, None)
    for path in (Path(__file__).resolve(), REPO / "tests/tools/neuron_metrics.py"):
        signature["code_sha256"][str(path.relative_to(REPO))] = sha(path)
    result = dict(input=dict(path=str(mesh_path), sha256=sha(mesh_path)),
                  configuration=config.as_dict(), elapsed_seconds=elapsed,
                  platform=platform.platform(), cpu=platform.processor(),
                  dependencies=signature["dependencies"],
                  provider=getattr(provider, "__file__", None),
                  source_sha256=signature["code_sha256"],
                  runtime_environment=signature["runtime_environment"],
                  provenance=signature, run_hash=digest_json(signature),
                  thread_environment={k: os.environ.get(k) for k in
                                      ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS")})
    try:
        result["git_sha"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
        result["tracked_dirty"] = bool(subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO, text=True))
    except (OSError, subprocess.CalledProcessError):
        result["git_sha"] = None
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--neuron", default=str(_ANCHOR), help="anchor mesh (.off)")
    add_run_arguments(ap, default_spines=False)
    ap.set_defaults(dataset="h01")
    ap.add_argument("--data-type", dest="dataset", choices=["h01", "microns"],
                    default=argparse.SUPPRESS, help="alias for --dataset")
    ap.add_argument("--calculate-spines", dest="spines", action="store_true",
                    default=argparse.SUPPRESS, help="alias for --spines")
    ap.add_argument("--require-spines", action="store_true", help="fail if a spine-positive fixture produces none")
    ap.add_argument("--segment-id", type=int, default=2530864375)
    ap.add_argument("--out", help="write the metrics JSON here")
    ap.add_argument("--baseline", help="compare against this JSON; exit 1 on any diff")
    ap.add_argument("--tolerance", action="store_true",
                    help="compare within per-metric tolerances instead of exactly")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    try:
        config = config_from_args(args)
    except ValueError as exc:
        ap.error(str(exc))
    if args.require_spines and not config.calculate_spines:
        ap.error("--require-spines requires --spines")

    mesh_path = Path(args.neuron).expanduser().resolve()
    if not mesh_path.exists():
        sys.exit(f"missing mesh {mesh_path}")

    # the pipeline shells out to meshlab, which litters ./<segment_id>/ and ./temp/
    os.chdir(tempfile.mkdtemp(prefix="neuron_gate_"))
    print(f"[gate] mesh={mesh_path.name} config={config.as_dict()}")
    mesh = tu.load_mesh_no_processing(str(mesh_path))
    print(f"[gate] faces={len(mesh.faces)} verts={len(mesh.vertices)}; decomposing ...")

    started = time.perf_counter()
    neuron_obj = segmentation_pipeline(mesh, segment_id=args.segment_id,
                                       verbose=args.verbose, config=config)
    metrics = extract_metrics(neuron_obj, include_spines=config.calculate_spines)
    print("\n" + json.dumps(metrics, indent=2))

    if args.out:
        out = Path(args.out)
        if not out.is_absolute():
            out = REPO / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(metrics, indent=2))
        out.with_suffix(out.suffix + ".run.json").write_text(
            json.dumps(_provenance(mesh_path, config, time.perf_counter() - started), indent=2) + "\n")
        print(f"\n[gate] written -> {out}")

    problems = _spine_problems(metrics, args.require_spines) if config.calculate_spines else []
    if problems:
        print("\n[gate] FAIL — spine surface/volume consistency:")
        print("\n".join(problems))
        sys.exit(1)

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
