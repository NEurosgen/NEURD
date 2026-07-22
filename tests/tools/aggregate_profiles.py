#!/usr/bin/env python
"""Aggregate a batch_bench.py run into cross-mesh tables.

Answers the three questions the batch was run to answer:
  * WHERE time goes  -> function matrix (cumtime/tottime per mesh) + per-stage table
  * WHICH functions fire in WHICH cases -> presence matrix, incl. microns-only /
    h01-only functions and the ones whose cost scales hardest with mesh size
  * WHEN/where memory goes -> time_mem_table (peak RSS, per-stage delta, top holder)

Consumes the layout batch_bench.py writes:
    <root>/<stem>/clean/summary.json      (true wall-clock + memory)
    <root>/<stem>/prof/summary.json       (same fields, cProfile-inflated wall)
    <root>/<stem>/prof/profile.prof       (function attribution)

Timing/memory numbers are taken from the CLEAN pass when present (unskewed);
function attribution always comes from the PROF pass. Writes CSVs + a readable
AGGREGATE_REPORT.md into <root>.
"""

import argparse
import json
import pstats
from collections import defaultdict
from pathlib import Path

# Project = the repo's own `neurd` package + the sibling mesh backends it drives
# (mesh_tools/datasci_tools/meshparty). NOTE: the conda env is literally named
# "neurd", so a naive "neurd/" substring matches the ENTIRE site-packages tree
# (numpy, psutil, threading, ...) and the filter becomes a no-op. Anchor on the
# real package directory instead. The raw function matrix still keeps everything.
_REPO = Path(__file__).resolve().parents[2]
_NEURD_PKG = str(_REPO / "neurd") + "/"
_LIB_MARKERS = ("/mesh_tools/", "/datasci_tools/", "/meshparty/")


def _short(func):
    fn, lineno, name = func
    return f"{Path(fn).name}:{lineno}({name})"


def _is_project(func):
    p = func[0]
    return p.startswith(_NEURD_PKG) or any(m in p for m in _LIB_MARKERS)


def load_prof(path):
    """Return {func_key: (ncalls, cumtime, tottime)} for one .prof file."""
    ps = pstats.Stats(str(path))
    out = {}
    for func, (cc, nc, tt, ct, callers) in ps.stats.items():
        out[func] = (nc, ct, tt)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="a batch_bench.py output root")
    ap.add_argument("--top", type=int, default=45,
                    help="per-mesh top-N by cumtime to union into the function matrix")
    ap.add_argument("--min-cumtime", type=float, default=0.5,
                    help="ignore functions below this cumtime (s) in presence analysis")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    corpus = json.loads((root / "corpus.json").read_text()) if (root / "corpus.json").exists() else None

    stems = sorted(p.parent.parent.name for p in root.glob("*/clean/summary.json"))
    if not stems:
        stems = sorted(p.parent.parent.name for p in root.glob("*/prof/summary.json"))
    if not stems:
        ap.error(f"no <stem>/{{clean,prof}}/summary.json under {root}")

    # ---- per-mesh scalars (prefer clean pass for time/mem) ----
    meta = {}
    for stem in stems:
        clean = root / stem / "clean" / "summary.json"
        prof = root / stem / "prof" / "summary.json"
        src = clean if clean.exists() else prof
        meta[stem] = json.loads(src.read_text())
        meta[stem]["_time_src"] = "clean" if clean.exists() else "prof"

    # order meshes by data_type then faces so the tables read microns-then-h01, small-first
    order = sorted(stems, key=lambda s: (meta[s].get("data_type", ""), meta[s].get("faces", 0)))

    # ---- function profiles from the prof pass ----
    profs = {}
    for stem in order:
        pf = root / stem / "prof" / "profile.prof"
        if pf.exists():
            profs[stem] = load_prof(pf)
    prof_stems = [s for s in order if s in profs]

    lines = ["# NEURD batch aggregate", ""]
    if corpus:
        lines.append(f"batch: `{root.name}` — {len(order)} meshes, passes={corpus.get('passes')}")
    lines.append("")

    # ---------------- TIME + MEMORY TABLE ----------------
    tm_rows = []
    tm_header = ["mesh", "dtype", "size_mb", "faces", "n_limbs", "n_branch",
                 "spines", "wall_s", "load_s", "construct_s", "save_s",
                 "rss_peak_mb", "ru_maxrss_mb", "top_ram_holder", "time_src", "error"]
    for stem in order:
        m = meta[stem]
        st = {s["name"]: s for s in m.get("stages", [])}
        def stage_s(n):
            return f"{st[n]['secs']:.1f}" if n in st else "-"
        tm = m.get("tracemalloc_top")
        holder = f"{tm['loc']}({tm['size_mb']}MB)" if tm else "-"
        err = m.get("pipeline_error") or m.get("save_error") or ""
        tm_rows.append([
            stem, m.get("data_type"), m.get("size_mb"), m.get("faces") or "-",
            m.get("n_limbs"), m.get("n_branches_total"), m.get("total_spines"),
            f"{m.get('wall_s'):.0f}" if m.get("wall_s") is not None else "-",
            stage_s("load_mesh"), stage_s("neuron_construction"), stage_s("save_segmentation"),
            m.get("rss_peak_mb"), m.get("ru_maxrss_mb"), holder, m.get("_time_src"), err[:40],
        ])

    (root / "time_mem_table.csv").write_text(
        ",".join(tm_header) + "\n" +
        "\n".join(",".join(str(c) for c in r) for r in tm_rows) + "\n")

    lines.append("## Time + memory per mesh")
    lines.append("")
    lines.append("| " + " | ".join(tm_header) + " |")
    lines.append("|" + "|".join(["---"] * len(tm_header)) + "|")
    for r in tm_rows:
        lines.append("| " + " | ".join(str(c) for c in r) + " |")
    lines.append("")
    lines.append("_wall/RSS from the clean pass where available; cProfile pass inflates wall._")
    lines.append("")

    if not prof_stems:
        lines.append("> No prof pass found — run `batch_bench.py --pass prof` for the function matrix.")
        (root / "AGGREGATE_REPORT.md").write_text("\n".join(lines))
        print("\n".join(lines))
        return

    # ---------------- FUNCTION MATRIX (cumtime) ----------------
    # Union of each mesh's top-N by cumtime = the hot set that matters somewhere.
    hot = set()
    for stem in prof_stems:
        ranked = sorted(profs[stem].items(), key=lambda kv: kv[1][1], reverse=True)
        hot.update(f for f, _ in ranked[:args.top])

    def total_ct(func):
        return sum(profs[s].get(func, (0, 0, 0))[1] for s in prof_stems)
    hot_sorted = sorted(hot, key=total_ct, reverse=True)

    for metric_name, idx in (("cumtime", 1), ("ncalls", 0)):
        header = ["function"] + prof_stems
        out = [",".join(header)]
        for func in hot_sorted:
            row = [_short(func)]
            for s in prof_stems:
                v = profs[s].get(func, (0, 0, 0))[idx]
                row.append(f"{v:.1f}" if idx == 1 else str(int(v)))
            out.append(",".join(row))
        (root / f"function_matrix_{metric_name}.csv").write_text("\n".join(out) + "\n")

    # compact cumtime matrix into the report (top 30 rows)
    lines.append("## Function cumtime (s) by mesh — hot set")
    lines.append("")
    lines.append("| function | " + " | ".join(prof_stems) + " |")
    lines.append("|" + "|".join(["---"] * (len(prof_stems) + 1)) + "|")
    for func in hot_sorted[:30]:
        cells = [f"{profs[s].get(func, (0,0,0))[1]:.0f}" for s in prof_stems]
        lines.append(f"| {_short(func)} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("_full matrix (cumtime + ncalls, all hot functions): "
                 "`function_matrix_cumtime.csv`, `function_matrix_ncalls.csv`._")
    lines.append("")

    # ---------------- PRESENCE: which functions in which cases ----------------
    by_dtype = defaultdict(list)
    for stem in prof_stems:
        by_dtype[meta[stem].get("data_type")].append(stem)

    def fires_on(func, stems_):
        return [s for s in stems_ if profs[s].get(func, (0, 0, 0))[1] >= args.min_cumtime]

    all_funcs = {f for s in prof_stems for f in profs[s] if _is_project(f)}
    microns_stems = by_dtype.get("microns", [])
    h01_stems = by_dtype.get("h01", [])

    only_h01, only_microns = [], []
    for func in all_funcs:
        m_hit = fires_on(func, microns_stems)
        h_hit = fires_on(func, h01_stems)
        if h_hit and not m_hit:
            only_h01.append((func, total_ct(func)))
        elif m_hit and not h_hit:
            only_microns.append((func, total_ct(func)))
    only_h01.sort(key=lambda x: x[1], reverse=True)
    only_microns.sort(key=lambda x: x[1], reverse=True)

    lines.append("## Which functions fire in which cases (project fns, cumtime ≥ "
                 f"{args.min_cumtime}s)")
    lines.append("")
    lines.append(f"**h01-only** (fire on h01, never on microns) — top 20 by total cumtime:")
    lines.append("")
    for func, ct in only_h01[:20]:
        lines.append(f"- `{_short(func)}` — {ct:.0f}s")
    if not only_h01:
        lines.append("- (none)")
    lines.append("")
    lines.append(f"**microns-only** — top 20:")
    lines.append("")
    for func, ct in only_microns[:20]:
        lines.append(f"- `{_short(func)}` — {ct:.0f}s")
    if not only_microns:
        lines.append("- (none)")
    lines.append("")

    # scaling: cumtime on the biggest mesh vs the smallest (by faces) within prof_stems
    by_faces = sorted(prof_stems, key=lambda s: meta[s].get("faces", 0))
    small, big = by_faces[0], by_faces[-1]
    scale = []
    for func in all_funcs:
        cs = profs[small].get(func, (0, 0, 0))[1]
        cb = profs[big].get(func, (0, 0, 0))[1]
        if cb >= args.min_cumtime:
            ratio = cb / cs if cs > 1e-6 else float("inf")
            scale.append((func, cs, cb, ratio))
    scale.sort(key=lambda x: (x[2] - x[1]), reverse=True)
    lines.append(f"## Cost that scales with size ({small} {meta[small].get('faces'):,}f "
                 f"→ {big} {meta[big].get('faces'):,}f) — top 20 by absolute growth")
    lines.append("")
    lines.append("| function | small_s | big_s | ×growth |")
    lines.append("|---|---|---|---|")
    for func, cs, cb, ratio in scale[:20]:
        r = "∞" if ratio == float("inf") else f"{ratio:.1f}"
        lines.append(f"| `{_short(func)}` | {cs:.1f} | {cb:.1f} | {r} |")
    lines.append("")

    (root / "AGGREGATE_REPORT.md").write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\n[aggregate] wrote: AGGREGATE_REPORT.md, time_mem_table.csv, "
          f"function_matrix_cumtime.csv, function_matrix_ncalls.csv  -> {root}")


if __name__ == "__main__":
    main()
