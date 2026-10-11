# process_all_neurons.py
# -*- coding: utf-8 -*-
#
# Batch driver for the slim NEURD segmentation pipeline: walks a directory of neuron
# meshes, decomposes each one, and writes the segmentation to disk. Progress is tracked
# in a manifest so an interrupted run resumes where it stopped.
#
# For every limb it records the branch connectivity: alongside branch_nodes.npy /
# branch_edges.npy it writes connectivity.json, holding the directed edges
# (parent_branch_idx -> child_branch_idx) from the soma-attached root branch outward,
# plus a parent map for each branch.
#
# Dataset, spine mode and SDF profile are explicit shared RunConfig settings.

import os
import json
import time
import traceback
import multiprocessing as mp
import signal
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Set

# Keep the numeric backends single-threaded (we parallelize across meshes, not inside them)
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

from neurd import parameters
from neurd.segmentation_pipeline import segmentation_pipeline
from neurd.mesh_io import load_mesh
from neurd.run_config import RunConfig, add_run_arguments, config_from_args, require_compiled_cgal
from neurd.batch_utils import (
    sha, digest_json, save, complete_valid, separate_roots, dependency_signature,
    triangle_keys, raw_run_signature as _run_signature,
)

# Compatibility aliases for callers that imported the original runner helpers.
from neurd.neuron_export import (
    EXPORT_EXT, save_segmentation,
    _get_limb_concept_network, _get_limb_starting_node, _to_int_safe,
    _build_parent_map, _branch_graph_mapping, _save_limb_connectivity,
    _safe_export_mesh, _safe_save_array, _iter_spines, _get_soma_mesh,
    _summarize_neuron, _save_spine_masks, _save_branch_origin,
)

# Start a fresh interpreter rather than inheriting numerical-library threads.
mp = mp.get_context("spawn")

# ---------------------- PARAMETERS ----------------------

DATA_TYPE = "microns"  # Legacy default; use --dataset or RunConfig for new callers.


# ---------------------- MEMORY / PROCESS UTILITIES ----------------------

def _try_import_psutil():
    try:
        import psutil
        return psutil
    except Exception:
        return None

_PSUTIL = _try_import_psutil()

def _rss_mb(pid=None):
    """Observed RSS in MiB, including a worker's external subprocesses."""
    if _PSUTIL is None:
        return None
    try:
        process = _PSUTIL.Process(pid or os.getpid())
        processes = [process] + (process.children(recursive=True) if pid else [])
        total = 0
        for item in processes:
            try:
                total += item.memory_info().rss
            except _PSUTIL.NoSuchProcess:
                pass
        return total / (1024 * 1024)
    except (_PSUTIL.NoSuchProcess, _PSUTIL.AccessDenied):
        return None


def _worker_process(off_path_str, out_dir_str, do_decimate, export_ext, conn, *,
                    config=None, no_save=False, decimation_ratio=0.0,
                    run_hash=None, input_record=None, max_somas=None,
                    private_work_dir=None, staging_dir=None):
    """Process one mesh in a private CWD and publish only completed exports."""
    original_cwd = os.getcwd()
    work_dir = staging = None
    try:
        # The process group also contains MeshLab/Xvfb children for limit cleanup.
        if hasattr(os, "setsid"):
            os.setsid()
        off_path = Path(off_path_str).resolve()
        out_dir = Path(out_dir_str).resolve()
        config = config or RunConfig(dataset=DATA_TYPE, calculate_spines=True)
        if input_record is not None and sha(off_path) != input_record["mesh"]["sha256"]:
            raise ValueError("Input mesh changed after batch planning")
        if run_hash is not None and _run_hash(config, decimation_ratio, max_somas) != run_hash:
            raise ValueError("Runtime changed after batch planning")
        work_dir = private_work_dir or tempfile.mkdtemp(prefix="neurd_work_")
        os.chdir(work_dir)
        mesh = load_mesh(off_path)
        if do_decimate and decimation_ratio:
            from neurd import _mesh_ops
            mesh = _mesh_ops.decimate(mesh, decimation_ratio=decimation_ratio)
        neuron_obj = segmentation_pipeline(mesh, config=config, max_somas=max_somas)
        if input_record is not None and sha(off_path) != input_record["mesh"]["sha256"]:
            raise ValueError("Input mesh changed during processing")
        if run_hash is not None and _run_hash(config, decimation_ratio, max_somas) != run_hash:
            raise ValueError("Runtime changed during processing")
        if not no_save:
            destination = out_dir / off_path.stem
            if destination.exists():
                raise ValueError(f"Refusing to overwrite an unverified output: {destination}")
            if staging_dir is None:
                scratch = out_dir / ".work"
                scratch.mkdir(parents=True, exist_ok=True)
                staging = Path(tempfile.mkdtemp(prefix=off_path.stem + "_", dir=scratch))
            else:
                staging = Path(staging_dir)
            save_segmentation(neuron_obj, staging, export_ext=export_ext)
            outputs = {str(path.relative_to(staging)): sha(path)
                       for path in sorted(staging.rglob("*")) if path.is_file()}
            provenance = _run_signature(config, decimation_ratio, max_somas)
            if run_hash is not None and digest_json(provenance) != run_hash:
                raise ValueError("Runtime changed while exporting results")
            save(staging / "result.json", dict(
                schema_version=1, status="complete", run_hash=run_hash,
                inputs=input_record, run_config=config.as_dict(),
                provenance=provenance,
                outputs_sha256=outputs,
            ))
            os.replace(staging, destination)
            staging = None
        conn.send((True, f"ok:{off_path.stem}"))
    except Exception as exc:
        conn.send((False, f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"))
    finally:
        os.chdir(original_cwd)
        if work_dir is not None:
            shutil.rmtree(work_dir, ignore_errors=True)
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)
        conn.close()


# ---------------------- NEURD SETUP ----------------------

def ensure_neurd_defaults():
    """
    Apply the NEURD parameter set selected by DATA_TYPE.
    """
    parameters.params.use(DATA_TYPE)


# ---------------------- MANIFEST ----------------------

def read_manifest(manifest_path: Path) -> Set[str]:
    if not manifest_path.exists():
        return set()
    with open(manifest_path, "r", encoding="utf-8") as f:
        return {line.strip().split()[0] for line in f if line.strip()}

def append_manifest_atomic(manifest_path: Path, entry: str) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    old = ""
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            old = f.read()
    tmp = manifest_path.with_suffix(manifest_path.suffix + f".tmp.{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as f:
        if old:
            f.write(old)
            if not old.endswith("\n"):
                f.write("\n")
        f.write(entry + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, manifest_path)


# ---------------------- PROCESSING / SHARED SCHEDULER ----------------------

def process_one_mesh(off_path: Path, out_dir: Path, do_decimate: bool = True, *,
                     config=None, no_save=False, decimation_ratio=0.0,
                     max_somas=None) -> Path:
    """Compatibility in-process entry; use the batch driver for isolated workers."""
    _validate_decimation(decimation_ratio, not do_decimate)
    config = config or RunConfig(dataset=DATA_TYPE, calculate_spines=True)
    mesh = load_mesh(off_path)
    if do_decimate and decimation_ratio:
        from neurd import _mesh_ops
        mesh = _mesh_ops.decimate(mesh, decimation_ratio=decimation_ratio)
    neuron_obj = segmentation_pipeline(mesh, config=config, max_somas=max_somas)
    destination = Path(out_dir) / off_path.stem
    if not no_save:
        save_segmentation(neuron_obj, destination)
    return destination


@dataclass(frozen=True)
class BatchResult:
    processed: int = 0
    skipped: int = 0
    failed: int = 0
    skipped_failed: int = 0

    @property
    def exit_code(self):
        return 1 if self.failed or self.skipped_failed else 0


def _validate_decimation(ratio, no_decimate):
    if not np.isfinite(ratio) or not 0 <= ratio <= 1:
        raise ValueError("decimation_ratio must be between 0 and 1 (0 means disabled)")
    if no_decimate and ratio:
        raise ValueError("--no-decimate cannot be combined with a positive decimation ratio")


def _stop_process(process):
    """Terminate the worker and its external helpers, then reap it."""
    owns_group = False
    if hasattr(os, "killpg"):
        try:
            owns_group = os.getpgid(process.pid) == process.pid
            if owns_group:
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
        except ProcessLookupError:
            pass
    else:
        process.terminate()
    process.join(5)
    if owns_group:
        # A helper may outlive the terminated Python worker. Reap the whole
        # process group before removing its scratch files.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    if process.is_alive():
        if not owns_group:
            process.kill()
        process.join()


def _cleanup_job(job):
    for key in ("private_work_dir", "staging_dir"):
        if job.get(key) is not None:
            shutil.rmtree(job[key], ignore_errors=True)


def _drain_result(job):
    # Drain while the child is alive: send() of a large traceback can fill a Pipe.
    if "result" not in job:
        try:
            if job["conn"].poll(0):
                job["result"] = job["conn"].recv()
        except (EOFError, OSError):
            pass


def _run_hash(config, decimation_ratio, max_somas):
    return digest_json(_run_signature(config, decimation_ratio, max_somas))


def _validate_manifest_path(manifest, source, output, files):
    path = Path(manifest)
    path = (path if path.is_absolute() else output / path).resolve()
    if path == source or source in path.parents:
        raise ValueError("Manifest must not be inside the raw input directory")
    if any(path == mesh.resolve() for mesh in files):
        raise ValueError("Manifest must not overwrite a raw input mesh")
    reserved = [output / "failed.txt", output / ".work"]
    reserved.extend(output / mesh.stem for mesh in files)
    if output.is_dir():
        reserved.extend(child for child in output.iterdir() if child.is_dir()
                        and ((child / "result.json").is_file() or (child / "summary.json").is_file()))
    if path == output or any(path == root.resolve() or root.resolve() in path.parents for root in reserved):
        raise ValueError("Manifest conflicts with a reserved output path")
    return path


def process_folder_with_manifest(
    input_dir: str,
    output_dir: str,
    manifest: str = "processed.txt",
    pattern: str = "*.off",
    no_save: bool = False,
    no_decimate: bool = False,
    *,
    config=None,
    retry_failed=False,
    decimation_ratio=0.0,
    max_somas=None,
    max_workers=1,
    input_size_budget_mb=1024,
    poll_interval=0.2,
) -> BatchResult:
    """Run raw-neuron decomposition with verified resume and isolated processes.

    The input-size budget limits concurrent *file bytes*, not RAM. Set
    SEG_RSS_LIMIT_MB for observed per-worker RSS enforcement (requires psutil).
    --no-save computes without creating exports, manifests or failure logs.
    Legacy outputs without result.json must be handled in a new output directory.
    """
    config = config or RunConfig(dataset=DATA_TYPE, calculate_spines=True)
    if not isinstance(config, RunConfig):
        raise TypeError("config must be a RunConfig")
    _validate_decimation(decimation_ratio, no_decimate)
    if max_somas is not None and (isinstance(max_somas, bool) or not isinstance(max_somas, int) or max_somas < 1):
        raise ValueError("max_somas must be a positive integer")
    if max_workers is None or max_workers == 0:
        max_workers = os.cpu_count() or 1
    if max_workers < 1 or input_size_budget_mb <= 0 or poll_interval <= 0:
        raise ValueError("Worker count, input-size budget and poll interval must be positive")
    rss_limit_mb = int(os.environ.get("SEG_RSS_LIMIT_MB", "0"))
    timeout = int(os.environ.get("SEG_TIMEOUT_SEC", "0"))
    if rss_limit_mb < 0 or timeout < 0:
        raise ValueError("RSS and timeout limits must be nonnegative")
    if rss_limit_mb and _PSUTIL is None:
        raise RuntimeError("SEG_RSS_LIMIT_MB requires psutil; install it or unset the RSS limit")
    if config.calculate_spines:
        require_compiled_cgal()
    in_dir, out_dir = separate_roots(input_dir, output_dir)
    if not in_dir.is_dir():
        raise ValueError(f"Missing input directory: {in_dir}")
    files = sorted(path for path in in_dir.glob(pattern) if path.is_file())
    if len({path.stem for path in files}) != len(files):
        raise ValueError("Input mesh basenames must be unique")
    if not files:
        raise ValueError(f"No files matched: {in_dir}/{pattern}")
    manifest_path = _validate_manifest_path(manifest, in_dir, out_dir, files)
    if not no_save:
        out_dir.mkdir(parents=True, exist_ok=True)
    failed_log = out_dir / "failed.txt"
    previously_failed = set() if no_save else read_manifest(failed_log)
    run_hash = _run_hash(config, decimation_ratio, max_somas)
    pending = []
    skipped = skipped_failed = 0
    for mesh in files:
        inputs = {"mesh": {"path": str(mesh), "sha256": sha(mesh)}}
        destination = out_dir / mesh.stem
        if not no_save and complete_valid(destination, run_hash, inputs):
            skipped += 1
            continue
        if not no_save and destination.exists():
            raise ValueError(f"Existing output lacks verified completion: {destination}; use a new output directory")
        if mesh.stem in previously_failed and not retry_failed:
            skipped_failed += 1
            print(f"[=] previous failure: {mesh.stem}; use --retry-failed")
            continue
        pending.append(dict(path=mesh, size=mesh.stat().st_size, inputs=inputs))
    pending.sort(key=lambda item: item["size"])
    budget = int(input_size_budget_mb * 1024 * 1024)
    print(f"[i] dataset={config.dataset} spines={config.calculate_spines} "
          f"refinement={config.refinement} sdf_rays={config.sdf_rays}")
    print(f"[i] workers={max_workers}, concurrent input-size budget={input_size_budget_mb} MiB (file bytes)")
    running = []
    processed = failed = 0
    try:
        while pending or running:
            while pending and len(running) < max_workers:
                inflight = sum(job["size"] for job in running)
                pick = next((i for i, item in enumerate(pending)
                             if not running or inflight + item["size"] <= budget), None)
                if pick is None:
                    break
                job = pending.pop(pick)
                parent, child = mp.Pipe(duplex=False)
                process = None
                try:
                    job["private_work_dir"] = tempfile.mkdtemp(prefix="neurd_work_")
                    job["staging_dir"] = None
                    if not no_save:
                        scratch = out_dir / ".work"
                        scratch.mkdir(parents=True, exist_ok=True)
                        job["staging_dir"] = tempfile.mkdtemp(prefix=job["path"].stem + "_", dir=scratch)
                    process = mp.Process(target=_worker_process, args=(
                        str(job["path"]), str(out_dir), not no_decimate, EXPORT_EXT, child,
                    ), kwargs=dict(config=config, no_save=no_save, decimation_ratio=decimation_ratio,
                                   run_hash=run_hash, input_record=job["inputs"], max_somas=max_somas,
                                   private_work_dir=job["private_work_dir"], staging_dir=job["staging_dir"]))
                    process.start()
                except BaseException:
                    if process is not None and process.pid is not None and process.is_alive():
                        _stop_process(process)
                    parent.close()
                    child.close()
                    _cleanup_job(job)
                    raise
                child.close()
                job.update(proc=process, conn=parent, start=time.monotonic())
                running.append(job)
                print(f"[start] {job['path'].name}")
            finished = []
            for job in running:
                process = job["proc"]
                _drain_result(job)
                if process.is_alive():
                    reason = None
                    if timeout and time.monotonic() - job["start"] > timeout:
                        reason = f"Timeout after {timeout}s"
                    if rss_limit_mb and reason is None:
                        observed = _rss_mb(process.pid)
                        if observed is None:
                            if process.is_alive():
                                reason = "Unable to observe RSS while SEG_RSS_LIMIT_MB is required"
                        elif observed > rss_limit_mb:
                            reason = f"RSS limit exceeded: {observed:.0f} MiB > {rss_limit_mb} MiB"
                    if reason:
                        _stop_process(process)
                        job["result"] = False, reason
                if not process.is_alive():
                    finished.append(job)
            for job in finished:
                running.remove(job)
                process = job["proc"]
                process.join()
                _drain_result(job)
                ok, message = job.get("result", (False, f"No worker result; exitcode={process.exitcode}"))
                job["conn"].close()
                _cleanup_job(job)
                ok = bool(ok) and process.exitcode == 0
                if ok:
                    if not no_save:
                        # Completion is independently validated before a manifest entry.
                        try:
                            if not complete_valid(out_dir / job["path"].stem, run_hash, job["inputs"]):
                                ok, message = False, "Worker did not publish a verified result"
                        except (ValueError, OSError) as exc:
                            ok, message = False, f"Worker completion validation failed: {exc}"
                if ok:
                    processed += 1
                    if not no_save:
                        append_manifest_atomic(manifest_path, job["path"].stem)
                else:
                    failed += 1
                    if not no_save:
                        with failed_log.open("a", encoding="utf-8") as stream:
                            stream.write(job["path"].stem + "\t" + json.dumps(message) + "\n")
                    print(f"[x] {job['path'].stem}: {message}")
            if running and not finished:
                time.sleep(poll_interval)
    finally:
        for job in running:
            if job["proc"].is_alive():
                _stop_process(job["proc"])
            job["conn"].close()
            _cleanup_job(job)
    result = BatchResult(processed, skipped, failed, skipped_failed)
    print(f"Done. processed={processed}, skipped={skipped}, failed={failed}, skipped_failed={skipped_failed}")
    return result


# ---------------------- CLI ----------------------

def add_batch_arguments(parser):
    parser.add_argument("input_dir", help="directory of raw neuron meshes")
    parser.add_argument("output_dir", help="separate output directory")
    parser.add_argument("--pattern", default="*.off")
    parser.add_argument("--manifest", default="processed.txt")
    parser.add_argument("--no-save", action="store_true", help="compute without exports, manifests or failure logs")
    parser.add_argument("--no-decimate", action="store_true", help="disable input decimation (already the default)")
    parser.add_argument("--decimation-ratio", type=float, default=0.0,
                        help="optional retained input-face fraction, 0 disables decimation")
    parser.add_argument("--max-somas", type=int, help="positive cap for the soma search")
    parser.add_argument("--retry-failed", action="store_true", help="retry inputs recorded in failed.txt")
    add_run_arguments(parser)


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description="Raw-neuron NEURD segmentation with verified resume.")
    add_batch_arguments(parser)
    args = parser.parse_args(argv)
    try:
        result = process_folder_with_manifest(
            args.input_dir, args.output_dir, manifest=args.manifest, pattern=args.pattern,
            no_save=args.no_save, no_decimate=args.no_decimate, config=config_from_args(args),
            retry_failed=args.retry_failed, decimation_ratio=args.decimation_ratio,
            max_somas=args.max_somas,
        )
        return result.exit_code
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"[x] {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
