"""Interleave separately built CGAL providers; require byte-identical CSVs.

Example:
    python tests/tools/benchmark_cgal_io.py --baseline /tmp/cgal-old \
        --candidate /tmp/cgal-new --mesh /path/to/fixture.off --out /tmp/cgal-ab.json

Each invocation imports one native provider in a fresh process. Source meshes
are copied to a temporary directory. Nothing is installed or overwritten.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time

CHILD = """
import cgal_Segmentation_Module as csm
import json, resource, sys, time
started = time.perf_counter()
code = csm.cgal_segmentation(sys.argv[1], int(sys.argv[2]), float(sys.argv[3]))
print('NEURD_BENCH_RESULT ' + json.dumps(dict(code=code, provider=csm.__file__,
    seconds=time.perf_counter()-started, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)))
sys.exit(0 if code == 1 else 1)
"""


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True, help="directory containing the old native module")
    parser.add_argument("--candidate", type=Path, required=True, help="directory containing the new native module")
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--rays", type=int, default=25)
    parser.add_argument("--clusters", type=int, default=3)
    parser.add_argument("--smoothness", type=float, default=.2)
    args = parser.parse_args()
    if min(args.repeat, args.rays, args.clusters) <= 0:
        parser.error("repeat, rays and clusters must be positive")
    mesh = args.mesh.resolve()
    initial_hash = sha(mesh)
    env = dict(os.environ, NEURD_SDF_RAYS=str(args.rays), NEURD_CGAL_TIMINGS="1",
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    rows = []
    for trial in range(args.repeat):
        for name in (("baseline", "candidate") if trial % 2 == 0 else ("candidate", "baseline")):
            library = getattr(args, name).resolve()
            with tempfile.TemporaryDirectory(prefix="neurd_cgal_ab_") as directory:
                work = Path(directory)
                shutil.copyfile(mesh, work / "mesh.off")
                started = time.perf_counter()
                process = subprocess.run([sys.executable, "-c", CHILD, str(work / "mesh"),
                                          str(args.clusters), str(args.smoothness)],
                                         env=dict(env, PYTHONPATH=str(library)), cwd=work,
                                         capture_output=True, text=True, check=True)
                elapsed = time.perf_counter() - started
                line = next(line for line in process.stdout.splitlines() if line.startswith("NEURD_BENCH_RESULT "))
                row = json.loads(line.removeprefix("NEURD_BENCH_RESULT "))
                if Path(row["provider"]).resolve().parent != library:
                    raise RuntimeError("Imported a provider outside the requested build directory")
                row.update(trial=trial, build=name, wall_seconds=elapsed,
                           provider_sha256=sha(row["provider"]),
                           outputs_sha256={p.name:sha(p) for p in sorted(work.glob("*.csv"))},
                           stages={stage:float(value) for stage,value in re.findall(
                               r"\[neurd\.cgal\] (\w+)_seconds=([0-9.eE+-]+)", process.stderr)})
                if len(row["outputs_sha256"]) != 2:
                    raise RuntimeError("CGAL did not produce both per-face CSVs")
                rows.append(row)
                print(f"{name} trial={trial} provider_seconds={row['seconds']:.4f}", flush=True)
    identical = all(row["outputs_sha256"] == rows[0]["outputs_sha256"] for row in rows)
    receipt = dict(input=dict(path=str(mesh), sha256=initial_hash),
                   python=sys.version, platform=platform.platform(),
                   configuration=dict(rays=args.rays, clusters=args.clusters, smoothness=args.smoothness,
                                      threads=1),
                   byte_identical=identical, input_unchanged=sha(mesh) == initial_hash, trials=rows,
                   summary={name:dict(median_seconds=statistics.median(
                       r["seconds"] for r in rows if r["build"] == name),
                       median_wall_seconds=statistics.median(r["wall_seconds"] for r in rows if r["build"] == name),
                       peak_rss_kib=max(r["peak_rss_kib"] for r in rows if r["build"] == name))
                       for name in ("baseline", "candidate")})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2) + "\n")
    if not identical or not receipt["input_unchanged"]:
        raise SystemExit("FAIL: CSV bytes differed or source mesh changed; see receipt")
    print(json.dumps(receipt["summary"], indent=2))


if __name__ == "__main__":
    main()
