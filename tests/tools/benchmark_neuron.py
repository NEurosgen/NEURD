#!/usr/bin/env python
"""Reusable time + memory benchmark for the NEURD H01 single-neuron pipeline.

Replicates the per-neuron work of process_all_neurons.py exactly:

    ensure_neurd_defaults()  (data_type="h01", OPENBLAS/OMP threads = 1)
    mesh = tu.load_mesh_no_processing(off_path)
    neuron_obj = neuron.Neuron(mesh=mesh)
    save_segmentation(neuron_obj, out_base)

...and instruments it with:

  * per-stage wall-clock (load / construct / save),
  * a background RSS sampler -> timeline CSV + per-stage peak/delta,
  * process peak RSS via getrusage(ru_maxrss),
  * optional cProfile  -> function attribution: top cumulative, top self-time,
    and a "junk function" view (very high call counts) + a reusable .prof dump,
  * optional tracemalloc -> Python-side allocation attribution at peak RSS.

cProfile inflates wall-clock (proportions stay meaningful); tracemalloc inflates
both time and memory. So for clean numbers run more than once:

  # where the time goes (function attribution) + real memory peak/timeline
  python tests/tools/benchmark_neuron.py --neuron <path.off> --out <dir>

  # true wall-clock (no profiler skew), still with memory sampling
  python tests/tools/benchmark_neuron.py --neuron <path.off> --out <dir> --no-profile

  # python memory attribution (what holds RAM); pair with --no-profile
  python tests/tools/benchmark_neuron.py --neuron <path.off> --out <dir> --no-profile --tracemalloc

The pipeline writes scratch (meshlab/cgal temp) into the CWD, so the script chdirs
into <out>/_work for the duration. All artifacts land under <out>/.
"""

# Thread caps MUST be set before numpy is imported anywhere (matches process_all_neurons.py).
import os
for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import cProfile
import gc
import io
import pstats
import resource
import sys
import threading
import time
import tracemalloc
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


# --------------------------- memory helpers ---------------------------

def _rss_mb(pid=None):
    """Resident set size in MB. psutil if present, else /proc fallback."""
    pid = pid or os.getpid()
    try:
        import psutil
        return psutil.Process(pid).memory_info().rss / (1024 * 1024)
    except Exception:
        try:
            with open(f"/proc/{pid}/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1]) / 1024.0
        except Exception:
            return float("nan")
    return float("nan")


def _ru_maxrss_mb():
    """Process lifetime peak RSS from the kernel (Linux: ru_maxrss is in KB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


class RSSSampler(threading.Thread):
    """Samples whole-process RSS on a fixed interval in a daemon thread."""

    def __init__(self, interval=0.5):
        super().__init__(daemon=True)
        self.interval = interval
        self.samples = []  # (t_rel_seconds, rss_mb)
        # NB: must NOT be named _stop — threading.Thread uses self._stop internally.
        self._stop_event = threading.Event()
        self.t0 = time.time()

    def run(self):
        while not self._stop_event.is_set():
            self.samples.append((time.time() - self.t0, _rss_mb()))
            self._stop_event.wait(self.interval)

    def stop(self):
        self._stop_event.set()
        self.join(timeout=3)

    def peak(self):
        return max((r for _, r in self.samples), default=float("nan"))

    def peak_between(self, t_lo, t_hi):
        vals = [r for t, r in self.samples if t_lo <= t <= t_hi]
        return max(vals, default=float("nan"))


# --------------------------- stage timing ---------------------------

class Stage:
    """Context manager: records wall-clock + RSS delta + in-stage RSS peak."""

    def __init__(self, name, sampler, record):
        self.name = name
        self.sampler = sampler
        self.record = record

    def __enter__(self):
        gc.collect()
        self.rss0 = _rss_mb()
        self.t_rel0 = time.time() - self.sampler.t0
        self.t0 = time.time()
        print(f"[stage] >>> {self.name}", flush=True)
        return self

    def __exit__(self, *exc):
        secs = time.time() - self.t0
        t_rel1 = time.time() - self.sampler.t0
        rss1 = _rss_mb()
        self.record.append(dict(
            name=self.name,
            secs=secs,
            rss_start_mb=self.rss0,
            rss_end_mb=rss1,
            rss_delta_mb=rss1 - self.rss0,
            rss_peak_mb=self.sampler.peak_between(self.t_rel0, t_rel1),
        ))
        print(f"[stage] <<< {self.name}: {secs:.1f}s  "
              f"RSS {self.rss0:.0f}->{rss1:.0f}MB (Δ{rss1 - self.rss0:+.0f}, "
              f"peak {self.sampler.peak_between(self.t_rel0, t_rel1):.0f})", flush=True)
        return False  # never swallow exceptions


# --------------------------- profile reporting ---------------------------

def _write_pstats(pr, out_dir, top=70, ncalls_floor=20000):
    """Dump cProfile views: cumulative, self-time, junk (high ncalls), and .prof."""
    pr.dump_stats(str(out_dir / "profile.prof"))

    for sort_key, fname in (("cumulative", "profile_cumulative.txt"),
                            ("tottime", "profile_tottime.txt")):
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats(sort_key)
        ps.print_stats(top)
        (out_dir / fname).write_text(s.getvalue())

    # "junk function" view: functions called an enormous number of times, ranked by
    # cumulative time. These are the vectorize/cache candidates (e.g. a helper hit
    # hundreds of thousands of times inside a Python loop).
    ps = pstats.Stats(pr)
    rows = []
    for func, (cc, nc, tt, ct, callers) in ps.stats.items():
        if nc >= ncalls_floor:
            rows.append((nc, ct, tt, func))
    rows.sort(key=lambda r: r[1], reverse=True)
    lines = [f"functions with ncalls >= {ncalls_floor}, ranked by cumulative time",
             f"{'ncalls':>12} {'cumtime':>9} {'tottime':>9}  function", "-" * 90]
    for nc, ct, tt, func in rows[:60]:
        fn, lineno, name = func
        short = f"{Path(fn).name}:{lineno}({name})"
        lines.append(f"{nc:>12} {ct:>9.2f} {tt:>9.2f}  {short}")
    (out_dir / "profile_junk_highncalls.txt").write_text("\n".join(lines))

    # compact filtered cumulative view for the summary (project + key libs only)
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(120)
    keep = ("neurd/", "mesh_tools/", "skeleton", "meshparty", "trimesh/", "networkx",
            "sklearn", "scipy", "seconds", "Ordered", "ncalls")
    filt = [ln for ln in s.getvalue().splitlines() if any(k in ln for k in keep)]
    return filt[:60]


def _write_tracemalloc(snapshot, out_dir, top=40):
    stats = snapshot.statistics("lineno")
    lines = [f"top {top} python allocations at peak RSS (tracemalloc, by size)",
             f"{'size_MB':>10} {'count':>10}  location", "-" * 90]
    for st in stats[:top]:
        fr = st.traceback[0]
        lines.append(f"{st.size / 1024 / 1024:>10.2f} {st.count:>10}  "
                     f"{Path(fr.filename).name}:{fr.lineno}")
    total = sum(st.size for st in stats) / 1024 / 1024
    lines.append("-" * 90)
    lines.append(f"tracked total: {total:.1f} MB")
    (out_dir / "tracemalloc_top.txt").write_text("\n".join(lines))


# --------------------------- main ---------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--neuron", required=True, help="path to the .off mesh")
    ap.add_argument("--out", default=None, help="report dir (default: ./bench_<stem>_<ts>)")
    ap.add_argument("--no-profile", dest="profile", action="store_false",
                    help="disable cProfile (true wall-clock; memory still sampled)")
    ap.add_argument("--tracemalloc", action="store_true",
                    help="enable tracemalloc python-allocation attribution (skews time/mem)")
    ap.add_argument("--no-save", dest="save", action="store_false",
                    help="skip save_segmentation (benchmark compute only)")
    ap.add_argument("--sample-interval", type=float, default=0.5, help="RSS sample seconds")
    ap.add_argument("--no-spines", dest="spines", action="store_false",
                    help="build Neuron with calculate_spines=False (as process_all_neurons now does)")
    ap.add_argument("--data-type", default=None, choices=["microns", "h01"],
                    help="dataset parameter set to activate (default: process_all_neurons.DATA_TYPE). "
                         "Required to mix microns + h01 meshes in one batch.")
    args = ap.parse_args()

    off_path = Path(args.neuron).resolve()
    if not off_path.exists():
        ap.error(f"neuron mesh not found: {off_path}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out).resolve() if args.out else Path(f"./bench_{off_path.stem}_{ts}").resolve()
    work_dir = out_dir / "_work"
    work_dir.mkdir(parents=True, exist_ok=True)

    # The pipeline drops scratch in the CWD; isolate it.
    os.chdir(work_dir)

    # Import after thread caps + path setup. neurd MUST be imported before mesh_tools:
    # importing neurd activates the numpy-2 / cgal / meshlab compat shims that
    # datasci_tools and mesh_tools rely on (else `np.float_` blows up on import).
    # Reuse the exact h01 setup + saver from process_all_neurons so the benchmark
    # cannot drift from production.
    from neurd import neuron
    from neurd import parameters
    from mesh_tools import trimesh_utils as tu
    import process_all_neurons as pan

    if args.data_type is not None:
        parameters.params.use(args.data_type)
    else:
        pan.ensure_neurd_defaults()
    active_data_type = args.data_type or pan.DATA_TYPE

    sampler = RSSSampler(interval=args.sample_interval)
    sampler.start()
    rss_import = _rss_mb()

    if args.tracemalloc:
        tracemalloc.start(25)

    pr = cProfile.Profile() if args.profile else None
    stages = []
    peak_snapshot = None
    save_error = None
    pipeline_error = None
    neuron_obj = None
    n_faces = 0

    # The whole pipeline runs inside try/except so that a NEURD failure (these long
    # runs do sometimes die deep in preprocessing) still yields a written report:
    # the cProfile data + memory timeline up to the crash are exactly what we want.
    wall0 = time.time()
    if pr:
        pr.enable()
    try:
        with Stage("load_mesh", sampler, stages):
            mesh = tu.load_mesh_no_processing(str(off_path))
        n_faces = int(len(mesh.faces))

        with Stage("neuron_construction", sampler, stages):
            neuron_obj = neuron.Neuron(mesh=mesh, calculate_spines=args.spines)

        if args.tracemalloc:
            peak_snapshot = tracemalloc.take_snapshot()

        if args.save:
            with Stage("save_segmentation", sampler, stages):
                try:
                    pan.save_segmentation(neuron_obj, out_dir / off_path.stem)
                except Exception as e:
                    save_error = f"{type(e).__name__}: {e}"
                    print(f"[warn] save_segmentation failed: {save_error}", flush=True)
    except Exception as e:
        import traceback as _tb
        pipeline_error = f"{type(e).__name__}: {e}"
        print(f"[error] pipeline failed: {pipeline_error}\n{_tb.format_exc()}", flush=True)
    finally:
        if pr:
            pr.disable()
        wall = time.time() - wall0
        sampler.stop()
        if args.tracemalloc and peak_snapshot is None:
            try:
                peak_snapshot = tracemalloc.take_snapshot()
            except Exception:
                pass

    # ----------------- summarise neuron output -----------------
    if neuron_obj is not None:
        try:
            info = pan._summarize_neuron(neuron_obj)
        except Exception as e:
            info = {"summary_error": f"{type(e).__name__}: {e}"}
    else:
        info = {"summary_error": "neuron construction did not complete"}

    # ----------------- write artifacts -----------------
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "rss_timeline.csv", "w") as f:
        f.write("t_seconds,rss_mb\n")
        for t, r in sampler.samples:
            f.write(f"{t:.3f},{r:.1f}\n")

    filt_cum = []
    if pr:
        filt_cum = _write_pstats(pr, out_dir)
    tm_top = None
    if peak_snapshot is not None:
        _write_tracemalloc(peak_snapshot, out_dir)
        try:
            st = peak_snapshot.statistics("lineno")[0]
            fr = st.traceback[0]
            tm_top = {"loc": f"{Path(fr.filename).name}:{fr.lineno}",
                      "size_mb": round(st.size / 1024 / 1024, 1)}
        except Exception:
            pass

    # Machine-readable sibling of summary.txt so aggregate_profiles.py never has to
    # scrape the text report. One row of the cross-mesh time/mem table lives here.
    limbs = info.get("limbs", []) if isinstance(info, dict) else []
    import json as _json
    (out_dir / "summary.json").write_text(_json.dumps({
        "neuron": off_path.stem,
        "path": str(off_path),
        "data_type": active_data_type,
        "spines": args.spines,
        "profile": bool(pr),
        "size_mb": round(off_path.stat().st_size / 1e6, 1),
        "faces": n_faces,
        "wall_s": round(wall, 1),
        "stages": [{"name": s["name"], "secs": round(s["secs"], 1),
                    "rss_peak_mb": round(s["rss_peak_mb"], 0),
                    "rss_delta_mb": round(s["rss_delta_mb"], 0)} for s in stages],
        "rss_import_mb": round(rss_import, 0),
        "rss_peak_mb": round(sampler.peak(), 0),
        "ru_maxrss_mb": round(_ru_maxrss_mb(), 0),
        "n_limbs": info.get("n_limbs") if isinstance(info, dict) else None,
        "n_branches_total": sum(l.get("n_branches", 0) for l in limbs),
        "total_spines": sum(l.get("total_spines", 0) for l in limbs),
        "branches_per_limb": [l.get("n_branches", 0) for l in limbs],
        "tracemalloc_top": tm_top,
        "pipeline_error": pipeline_error,
        "save_error": save_error,
    }, indent=2))

    lines = []
    lines.append("=" * 72)
    lines.append("NEURD single-neuron benchmark")
    lines.append("=" * 72)
    lines.append(f"neuron        : {off_path}")
    lines.append(f"file size     : {off_path.stat().st_size / 1e6:.1f} MB")
    lines.append(f"mesh faces    : {n_faces:,}")
    lines.append(f"timestamp     : {ts}")
    lines.append(f"profile       : {'cProfile ON (wall inflated)' if pr else 'off (clean wall)'}"
                 f"{' + tracemalloc' if args.tracemalloc else ''}")
    lines.append(f"threads       : OPENBLAS/OMP/MKL/NUMEXPR = "
                 f"{os.environ.get('OPENBLAS_NUM_THREADS')}")
    lines.append(f"data_type     : {active_data_type}")
    lines.append(f"spines        : {args.spines}")
    lines.append("")
    lines.append("--- TIME (per stage) ---")
    lines.append(f"{'stage':<22}{'secs':>9}{'%wall':>8}   RSS start->end  (Δ / peak) MB")
    for s in stages:
        pct = 100 * s["secs"] / wall if wall else 0
        lines.append(f"{s['name']:<22}{s['secs']:>9.1f}{pct:>7.0f}%   "
                     f"{s['rss_start_mb']:>6.0f}->{s['rss_end_mb']:<6.0f} "
                     f"(Δ{s['rss_delta_mb']:+.0f} / {s['rss_peak_mb']:.0f})")
    lines.append(f"{'TOTAL wall':<22}{wall:>9.1f}{100:>7.0f}%")
    lines.append("")
    lines.append("--- MEMORY ---")
    lines.append(f"RSS after imports     : {rss_import:.0f} MB")
    lines.append(f"RSS sampled peak      : {sampler.peak():.0f} MB")
    lines.append(f"process ru_maxrss peak: {_ru_maxrss_mb():.0f} MB  (kernel lifetime peak)")
    lines.append("")
    lines.append("--- NEURON OUTPUT ---")
    limbs = info.get("limbs", []) if isinstance(info, dict) else []
    n_branches_total = sum(l.get("n_branches", 0) for l in limbs)
    total_spines = sum(l.get("total_spines", 0) for l in limbs)
    lines.append(f"{'data_type':<22}: {info.get('data_type')}")
    lines.append(f"{'n_limbs':<22}: {info.get('n_limbs')}")
    lines.append(f"{'n_branches_total':<22}: {n_branches_total}")
    lines.append(f"{'total_spines':<22}: {total_spines}")
    lines.append(f"{'soma_present':<22}: {info.get('soma_present')}")
    if pipeline_error:
        lines.append(f"PIPELINE FAILED       : {pipeline_error}")
        lines.append("  (timings/memory/profile below are valid UP TO the failure point)")
    if save_error:
        lines.append(f"save_segmentation     : FAILED ({save_error})")
    if "summary_error" in info:
        lines.append(f"summary               : FAILED ({info['summary_error']})")
    if filt_cum:
        lines.append("")
        lines.append("--- TOP CUMULATIVE (filtered to project + key libs) ---")
        lines.extend(filt_cum)
    lines.append("")
    lines.append(f"artifacts: {out_dir}")
    lines.append("  summary.txt, rss_timeline.csv" +
                 (", profile.prof, profile_cumulative.txt, profile_tottime.txt, "
                  "profile_junk_highncalls.txt" if pr else "") +
                 (", tracemalloc_top.txt" if peak_snapshot is not None else ""))

    report = "\n".join(lines)
    (out_dir / "summary.txt").write_text(report)
    print("\n" + report, flush=True)


if __name__ == "__main__":
    main()
