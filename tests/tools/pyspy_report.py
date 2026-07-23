#!/usr/bin/env python
"""Aggregate py-spy RAW folded stacks into a wall-time report (and A/B diff two runs).

Why: on this GIL-heavy, native-heavy pipeline cProfile's cumtime LIES (it charges blocked
`_thread.lock` waits to whatever Python frame is on top). py-spy samples the real stack at a
fixed rate, so `samples / rate` is honest wall time. Record with:

    py-spy record --format raw --rate 50 -o stacks.txt -- \
        python tests/tools/benchmark_neuron.py --neuron X.off --data-type h01 --no-profile --out D

Then:

    python tests/tools/pyspy_report.py stacks.txt                  # top functions
    python tests/tools/pyspy_report.py stacks.txt --callers split  # who calls `split`
    python tests/tools/pyspy_report.py before.txt --diff after.txt # A/B a change

Definitions
-----------
exclusive (self) = samples where the function is the LEAF frame -- the wall it burns itself.
inclusive        = samples where it appears ANYWHERE in the stack -- the wall spent under it.
Optimizations move the exclusive number; stage totals are inclusive.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path


def parse(path, rate=50.0):
    """Raw folded stacks -> (exclusive Counter, inclusive Counter, rows, total_seconds).

    `rows` keeps the parsed lines as (frame_names, raw_frames, n_samples) for caller queries.
    """
    exclusive, inclusive, rows, total = Counter(), Counter(), [], 0
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        stack, _, count = line.rpartition(" ")
        try:
            n = int(count)
        except ValueError:
            continue
        frames = stack.split(";")
        names = [f.split(" (")[0] for f in frames]
        rows.append((names, frames, n))
        total += n
        exclusive[names[-1]] += n
        for name in set(names):          # a recursive frame counts once per sample
            inclusive[name] += n
    return exclusive, inclusive, rows, total / rate


def _table(counter, total_s, rate, top, title):
    print(f"\n--- {title} ---")
    print(f"{'seconds':>9} {'%wall':>7}  function")
    for name, n in counter.most_common(top):
        print(f"{n / rate:9.1f} {n / rate / total_s:7.1%}  {name}")


def callers(rows, target, rate, top, only=None):
    """Wall attributed to `target`, split by the call chain above it (optionally filtered)."""
    by_chain = Counter()
    for names, frames, n in rows:
        if target not in names:
            continue
        i = names.index(target)
        chain = [f for f in frames[:i] if only is None or only in f]
        by_chain[" < ".join(chain[-3:][::-1])] += n
    print(f"\n--- callers of {target} (inclusive, nearest 3 frames, innermost first) ---")
    for chain, n in by_chain.most_common(top):
        print(f"{n / rate:9.1f}s  {chain}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stacks", help="py-spy --format raw output")
    ap.add_argument("--rate", type=float, default=50.0, help="the --rate py-spy recorded at")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--callers", metavar="FUNC", help="show what calls FUNC")
    ap.add_argument("--only", default="neurd/", help="--callers: keep only frames matching this")
    ap.add_argument("--diff", metavar="OTHER", help="A/B: report OTHER minus `stacks` per function")
    args = ap.parse_args()

    exclusive, inclusive, rows, total_s = parse(args.stacks, args.rate)
    print(f"{args.stacks}: {total_s:.0f}s of samples @ {args.rate} Hz")

    if args.callers:
        callers(rows, args.callers, args.rate, args.top, args.only)
        return

    if args.diff:
        excl_b, _, _, total_b = parse(args.diff, args.rate)
        print(f"{args.diff}: {total_b:.0f}s of samples  (delta {total_b - total_s:+.0f}s)")
        delta = Counter()
        for name in set(exclusive) | set(excl_b):
            delta[name] = (excl_b[name] - exclusive[name]) / args.rate
        moved = sorted(delta.items(), key=lambda kv: kv[1])
        print(f"\n--- exclusive wall: {args.diff} vs {args.stacks} (negative = faster) ---")
        for name, d in moved[:args.top]:
            print(f"{d:+9.1f}s  {name}")
        print("   ...")
        for name, d in moved[-args.top:]:
            print(f"{d:+9.1f}s  {name}")
        return

    _table(exclusive, total_s, args.rate, args.top, "exclusive (self) wall")
    _table(inclusive, total_s, args.rate, args.top, "inclusive wall")


if __name__ == "__main__":
    main()
