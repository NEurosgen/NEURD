# process_all_neurons_parallel.py
# -*- coding: utf-8 -*-
#
# Parallel version of process_all_neurons.py.
#
# The per-mesh work (NEURD segmentation, spines via CGAL, the on-disk layout, the
# manifest) is reused ENTIRELY from process_all_neurons.py -- this file is only the
# scheduler that runs several workers at once.
#
# The constraint that motivates it: the combined size of the input meshes being processed
# AT THE SAME TIME is kept under a memory budget (1 GB by default). The input .off size
# serves as a proxy for peak RAM -- the algorithm expands considerably on large meshes, so
# without this bound a few big meshes running together exhaust system memory.
#
# Scheduler rules:
#   * a new worker starts only if (sum of the sizes already running + the candidate's
#     size) <= budget;
#   * a mesh larger than the whole budget is not dropped -- it waits until every slot is
#     free and then runs on its own;
#   * concurrency is additionally capped by the worker count (--max-workers, default =
#     number of cores);
#   * the queue is sorted by size, ascending, so try_launch() immediately fills a pool of
#     small meshes up to the budget (greedy first-fit) and they run in parallel, while the
#     large ones settle at the tail and are processed one at a time. Unsorted, a large
#     mesh interleaved with small ones would be picked first, take almost the whole
#     budget, and leave no room for the small ones to run alongside it.

import os
import gc
import time
import multiprocessing as mp
from pathlib import Path
from typing import List, Dict, Any

# The same numeric-backend limits as the sequential version -- set BEFORE numpy/neurd is
# imported (process_all_neurons sets them too).
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

# Reuse the whole working part of the sequential script.
from process_all_neurons import (
    _worker_process,
    ensure_neurd_defaults,
    read_manifest,
    append_manifest_atomic,
    _rss_mb,
    EXPORT_EXT,
    DATA_TYPE,
)

# Default memory budget: combined size of concurrent meshes < 1 GB.
DEFAULT_MEM_BUDGET_MB = 1024


def _mb(n_bytes: int) -> float:
    return n_bytes / (1024 * 1024)


def process_folder_parallel(
    input_dir: str,
    output_dir: str,
    manifest: str = "processed.txt",
    pattern: str = "*.off",
    no_save: bool = False,
    no_decimate: bool = False,
    max_workers: int = None,
    mem_budget_mb: int = DEFAULT_MEM_BUDGET_MB,
    poll_interval: float = 0.2,
) -> None:
    in_dir = Path(input_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(manifest) if os.path.isabs(manifest) else (out_dir / manifest)

    ensure_neurd_defaults()

    files = sorted(in_dir.glob(pattern))
    if not files:
        print(f"[!] no files matched: {in_dir}/{pattern}")
        return

    done = read_manifest(manifest_path)
    done_failed = read_manifest(out_dir / "failed.txt")

    # The to-do list with file sizes, already excluding done/failed entries.
    pending: List[Dict[str, Any]] = []
    skipped = 0
    for f in files:
        sid = f.stem
        if sid in done or sid in done_failed:
            skipped += 1
            continue
        try:
            size = f.stat().st_size
        except OSError:
            size = 0
        pending.append({"path": f, "spine_id": sid, "size": size})

    # Sort the queue by size, ascending: try_launch() then fills a pool of small meshes
    # up to the budget and runs them in parallel, while large ones settle at the tail and
    # go one at a time. Otherwise a large mesh interleaved with small ones would be picked
    # first, take almost the whole budget, and the small ones would not fit alongside it --
    # degenerating into effectively single-threaded processing.
    pending.sort(key=lambda j: j["size"])

    budget_bytes = int(mem_budget_mb) * 1024 * 1024
    if not max_workers or max_workers <= 0:
        max_workers = os.cpu_count() or 1

    rss_limit_mb = int(os.environ.get("SEG_RSS_LIMIT_MB", "0"))
    per_item_timeout = int(os.environ.get("SEG_TIMEOUT_SEC", "0"))
    failed_log = out_dir / "failed.txt"

    total = len(pending)
    print(f"[i] data_type={DATA_TYPE}")
    print(f"[i] manifest: {manifest_path} (loaded {len(done)} ids, skipping {skipped})")
    print(f"[i] found {len(files)} mesh(es), {total} to process in: {in_dir}")
    print(f"[i] parallel: max_workers={max_workers}, mem_budget={mem_budget_mb} MB "
          f"({_mb(budget_bytes):.0f} MB)")
    if rss_limit_mb:
        print(f"[i] per-worker RSS limit: {rss_limit_mb} MB")
    if per_item_timeout:
        print(f"[i] per-item timeout: {per_item_timeout} s")

    oversized = [j for j in pending if j["size"] > budget_bytes]
    if oversized:
        print(f"[!] {len(oversized)} mesh(es) exceed the budget by themselves — "
              f"each will run ALONE (no co-tenants):")
        for j in oversized:
            print(f"      {j['path'].name}: {_mb(j['size']):.0f} MB")

    running: List[Dict[str, Any]] = []
    processed = failed = started = 0

    def in_flight_bytes() -> int:
        return sum(j["size"] for j in running)

    def try_launch() -> None:
        """Start as many workers as the budget and the free slots allow."""
        nonlocal started
        while pending and len(running) < max_workers:
            cur = in_flight_bytes()
            pick = None
            for idx, j in enumerate(pending):
                if not running:
                    # Nothing is running -- start the smallest of the remaining meshes
                    # (the queue is sorted ascending); if it exceeds the budget it runs
                    # on its own.
                    pick = idx
                    break
                if cur + j["size"] <= budget_bytes:
                    # The queue is sorted, so the first that fits is the smallest that
                    # fits, which maximizes the number of tasks in the pool.
                    pick = idx
                    break
            if pick is None:
                # None of the remaining meshes fits in what is left of the budget --
                # wait for slots to free up.
                break

            job = pending.pop(pick)
            parent_conn, child_conn = mp.Pipe(duplex=False)
            p = mp.Process(
                target=_worker_process,
                args=(str(job["path"]), str(out_dir), (not no_decimate),
                      EXPORT_EXT, child_conn),
                daemon=False,
            )
            p.start()
            # Close the writing end in the parent: otherwise an fd leaks per task and
            # EOF on the pipe is never detected.
            child_conn.close()

            started += 1
            job.update(proc=p, conn=parent_conn, start=time.time())
            running.append(job)
            print(f"[start {started}/{total}] {job['path'].name} "
                  f"({_mb(job['size']):.0f} MB) | in-flight: "
                  f"{_mb(in_flight_bytes()):.0f}/{mem_budget_mb} MB, "
                  f"workers: {len(running)}/{max_workers}")

    def finalize(job: Dict[str, Any]) -> None:
        nonlocal processed, failed
        p = job["proc"]
        conn = job["conn"]
        if "result" in job:
            ok, msg = job["result"]
        else:
            # The process finished on its own -- collect its message from the pipe.
            try:
                if conn.poll(1.0):
                    ok, msg = conn.recv()
                else:
                    ok, msg = (p.exitcode == 0), f"exitcode={p.exitcode}"
            except EOFError:
                ok, msg = (p.exitcode == 0), f"exitcode={p.exitcode}"
        try:
            conn.close()
        except Exception:
            pass
        p.join()

        if ok:
            if not no_save:
                append_manifest_atomic(manifest_path, job["spine_id"])
            processed += 1
            print(f"    [ok] {job['spine_id']}")
        else:
            failed += 1
            with open(failed_log, "a", encoding="utf-8") as fl:
                fl.write(f"{job['spine_id']}\t{msg}\n")
            print(f"    [x] failed: {job['spine_id']}: {msg}")
        gc.collect()

    try:
        while pending or running:
            try_launch()

            finished: List[Dict[str, Any]] = []
            for job in running:
                p = job["proc"]
                conn = job["conn"]

                # Drain the pipe while the process is still alive: a large traceback may
                # not fit in the buffer and would block the worker in send().
                try:
                    if "result" not in job and conn.poll(0):
                        job["result"] = conn.recv()
                except (EOFError, OSError):
                    pass

                if not p.is_alive():
                    finished.append(job)
                    continue

                now = time.time()
                if per_item_timeout and (now - job["start"] > per_item_timeout):
                    p.terminate(); p.join(5)
                    job["result"] = (False, f"Timeout after {per_item_timeout}s")
                    finished.append(job)
                    continue
                if rss_limit_mb:
                    r = _rss_mb(p.pid)
                    if r is not None and r > rss_limit_mb:
                        p.terminate(); p.join(5)
                        job["result"] = (False,
                                         f"RSS limit exceeded: {r} MB > {rss_limit_mb} MB")
                        finished.append(job)
                        continue

            for job in finished:
                running.remove(job)
                finalize(job)

            # Sleep only if nothing finished (otherwise refill the slots immediately).
            if running and not finished:
                time.sleep(poll_interval)

    except KeyboardInterrupt:
        print("\n[!] interrupted — terminating running workers...")
        for job in running:
            try:
                job["proc"].terminate()
                job["proc"].join(5)
            except Exception:
                pass
        raise

    print(f"\nDone. processed={processed}, skipped={skipped}, failed={failed}")
    print(f"[i] manifest now has {len(read_manifest(manifest_path))} ids at {manifest_path}")


# ---------------------- CLI ----------------------

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Parallel NEURD segmentation of neuron meshes, under a memory budget.")
    ap.add_argument("input_dir", type=str, help="directory of meshes (.off/.ply/...)")
    ap.add_argument("output_dir", type=str,
                    help="output directory (one subdir per input basename)")
    ap.add_argument("--pattern", type=str, default="*.off",
                    help="glob for the input files")
    ap.add_argument("--manifest", type=str, default="processed.txt",
                    help="manifest path (default: <output_dir>/processed.txt)")
    ap.add_argument("--no-save", action="store_true",
                    help="do not record entries in the manifest (manifest dry run)")
    ap.add_argument("--no-decimate", action="store_true", help="disable decimation")
    ap.add_argument("--max-workers", type=int, default=int(os.environ.get("SEG_MAX_WORKERS", "0")),
                    help="maximum concurrent workers (0 = number of cores)")
    ap.add_argument("--mem-budget-mb", type=int,
                    default=int(os.environ.get("SEG_MEM_BUDGET_MB", str(DEFAULT_MEM_BUDGET_MB))),
                    help="budget for the combined size of concurrent meshes, MB (default 1024)")
    args = ap.parse_args()

    process_folder_parallel(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        manifest=args.manifest,
        pattern=args.pattern,
        no_save=args.no_save,
        no_decimate=args.no_decimate,
        max_workers=args.max_workers,
        mem_budget_mb=args.mem_budget_mb,
    )

# Example (1 GB budget, up to 8 workers):
#   python process_all_neurons_parallel.py path/to/meshes path/to/output_segmentations \
#       --max-workers 8 --mem-budget-mb 1024
#
# NOTE: the 1024 MB default predates the -33% / -18% peak-RAM work, so it is now
# conservative -- a larger budget will usually keep more workers busy.