#!/usr/bin/env python
"""Run benchmark_neuron.py across a curated corpus of real meshes.

Purpose: see WHICH functions actually fire in WHICH cases, WHERE time goes and
WHEN memory is allocated -- across meshes that differ by dataset (microns/h01),
size (faces) and limb count. Each mesh is benchmarked in its OWN subprocess so a
NEURD crash or OOM on one mesh never poisons the others (same isolation strategy
as process_all_neurons._worker_process).

Two passes per mesh (both cheap to ask for, expensive to run -- see below):
  * clean : benchmark_neuron.py --no-profile   -> true wall-clock + RSS timeline
  * prof  : benchmark_neuron.py (cProfile on)   -> profile.prof for the aggregator

cProfile inflates wall ~25-30%, so the 'clean' pass owns the timing numbers and
the 'prof' pass owns the function attribution. Run --pass clean and --pass prof
separately (or --pass both) depending on what you need.

Layout produced:
    <out>/
      corpus.json                         # what ran, resolved
      <stem>/clean/  {summary.txt, rss_timeline.csv, ...}
      <stem>/prof/   {summary.txt, profile.prof, profile_*.txt, ...}

Then aggregate with:  python tests/tools/aggregate_profiles.py --root <out>

Cost realism: microns (~15MB) are minutes; h01 scales with faces AND limb count
(a 477MB mesh can be 20+ min). Launch under nohup / a background task and walk away.
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BENCH = REPO / "tests" / "tools" / "benchmark_neuron.py"

# Real full-resolution meshes. NOTE: tests/temp/*.off are 15 MB decimated CROPS whose
# soma does not survive the crop -> "No soma found"; do NOT use them. Full microns meshes
# live on the external drive and build on DEFAULTS (compact soma, no real-Poisson needed).
MICRONS_DIR = Path("/mnt/wwn-0x50014ee26c2ca7b0-part1/minnie65_meshes")
H01_DIR = Path("/home/eugen/Desktop/CodeWork/Projects/Diplom/notebooks/notebooks/H01")

# Curated corpus: 4 microns + 4 h01, balanced so cross-dataset diffs in the function
# matrix are real, not a sample-count artifact. Each entry is (mesh_path, data_type).
# Reality check: microns are NOT cheaper -- 864691136107964377 (122 MB) took ~53 min.
# So both halves lean to the SMALLER end to keep a full batch feasible overnight; add
# bigger meshes later once the small run is understood. 864691136107964377 is the
# user-confirmed known-good microns anchor.
DEFAULT_CORPUS = [
    (MICRONS_DIR / "864691135689048288.off", "microns"),  # ~56 MB  (smallest microns)
    (MICRONS_DIR / "864691135571154925.off", "microns"),  # ~60 MB
    (MICRONS_DIR / "864691135619541903.off", "microns"),  # ~64 MB
    (MICRONS_DIR / "864691136107964377.off", "microns"),  # ~122 MB (known-good anchor)
    (H01_DIR / "neuron_2530864375.off", "h01"),   # ~24.7 MB (validated gate anchor)
    (H01_DIR / "neuron_4476359994.off", "h01"),   # ~187 MB
    (H01_DIR / "neuron_4138580687.off", "h01"),   # ~294 MB
    (H01_DIR / "neuron_5175880292.off", "h01"),   # ~477 MB
]


def run_one(mesh_path, data_type, out_base, do_profile, spines, extra):
    """Invoke benchmark_neuron.py for a single mesh; return (ok, secs, msg)."""
    sub = "prof" if do_profile else "clean"
    out_dir = out_base / mesh_path.stem / sub
    cmd = [
        sys.executable, str(BENCH),
        "--neuron", str(mesh_path),
        "--out", str(out_dir),
        "--data-type", data_type,
    ]
    if not do_profile:
        cmd.append("--no-profile")
    if not spines:
        cmd.append("--no-spines")
    cmd += extra

    t0 = time.time()
    print(f"\n{'='*72}\n[batch] {mesh_path.stem} [{data_type}] pass={sub}\n"
          f"[batch] {' '.join(cmd)}\n{'='*72}", flush=True)
    proc = subprocess.run(cmd, cwd=str(REPO))
    secs = time.time() - t0
    ok = proc.returncode == 0
    print(f"[batch] {mesh_path.stem} pass={sub}: "
          f"{'OK' if ok else f'FAILED rc={proc.returncode}'} in {secs:.0f}s", flush=True)
    return ok, secs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None, help="report root (default: ./batch_<ts>)")
    ap.add_argument("--pass", dest="which", choices=["clean", "prof", "both"],
                    default="both", help="clean=true wall, prof=cProfile attribution")
    ap.add_argument("--no-spines", dest="spines", action="store_false",
                    help="build Neuron with calculate_spines=False")
    ap.add_argument("--only", nargs="*", default=None,
                    help="restrict to these mesh stems (e.g. --only neuron_11080 neuron_2530864375)")
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                    help="extra args passed through to benchmark_neuron.py (e.g. --tracemalloc)")
    args = ap.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_base = Path(args.out).resolve() if args.out else (REPO / f"batch_{ts}").resolve()
    out_base.mkdir(parents=True, exist_ok=True)

    corpus = [(p, dt) for (p, dt) in DEFAULT_CORPUS
              if args.only is None or p.stem in args.only]

    missing = [str(p) for p, _ in corpus if not p.exists()]
    if missing:
        print("[batch] WARNING: missing meshes (skipped):\n  " + "\n  ".join(missing), flush=True)
    corpus = [(p, dt) for (p, dt) in corpus if p.exists()]
    if not corpus:
        ap.error("no meshes in corpus exist on disk")

    passes = ["clean", "prof"] if args.which == "both" else [args.which]

    (out_base / "corpus.json").write_text(json.dumps(
        {"timestamp": ts, "passes": passes, "spines": args.spines,
         "meshes": [{"stem": p.stem, "path": str(p), "data_type": dt,
                     "size_mb": round(p.stat().st_size / 1e6, 1)} for p, dt in corpus]},
        indent=2))

    results = []
    for mesh_path, data_type in corpus:
        for do_profile in [p == "prof" for p in passes]:
            ok, secs = run_one(mesh_path, data_type, out_base, do_profile,
                               args.spines, args.extra)
            results.append((mesh_path.stem, data_type,
                            "prof" if do_profile else "clean", ok, secs))

    print(f"\n{'='*72}\n[batch] DONE -> {out_base}\n{'='*72}")
    print(f"{'mesh':<22}{'dtype':<9}{'pass':<7}{'ok':<5}{'secs':>8}")
    for stem, dt, sub, ok, secs in results:
        print(f"{stem:<22}{dt:<9}{sub:<7}{str(ok):<5}{secs:>8.0f}")
    print(f"\nnext: python tests/tools/aggregate_profiles.py --root {out_base}")


if __name__ == "__main__":
    main()
