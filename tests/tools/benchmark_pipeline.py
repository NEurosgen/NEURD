"""Run sequential, fresh-process whole-neuron A/B trials with exact output checks.

Use the same Python and compiled extensions for both source trees. Each variant
gets one excluded setup run and a separate persistent Numba cache; measured runs
alternate A/B and B/A. Kernel peak RSS is captured immediately after export.
Parent sampling also records RSS of the worker and its descendants (an estimate
which counts shared pages more than once). Quality checks run after timing.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import re
import resource
import signal
import statistics
import subprocess
import sys
import time
import traceback


THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS")
PROVIDERS = ("cgal_Segmentation_Module", "calcification_param_Module")
ADDED_EXPORTS = {"parent_face_indices.npy", "face_provenance.json", "branch_origin.json"}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def run_order(repeat):
    return [("setup", 0, name) for name in ("baseline", "candidate")] + [
        ("measured", index + 1, name) for index in range(repeat)
        for name in (("baseline", "candidate") if index % 2 == 0 else ("candidate", "baseline"))]


def source_manifest(root):
    """Code/configuration only; exclude generated files and dataset copies."""
    paths = [path for path in root.rglob("*") if path.is_file()
             and path.suffix in (".py", ".cpp", ".hpp", ".toml", ".json", ".md", ".txt", ".yaml", ".yml")
             and not any(part in {".git", "__pycache__", "build", "artifacts", ".venv"}
                         for part in path.relative_to(root).parts)]
    paths += [root / name for name in ("requirements.txt", "constraints/linux-py312-20261012.txt")
              if (root / name).is_file()]
    return {str(path.relative_to(root)): sha(path) for path in sorted(set(paths))}


def environment_manifest():
    """Hash installed distribution payloads, excluding bytecode/JIT caches."""
    distributions, files = {}, {}
    for dist in importlib.metadata.distributions():
        name = dist.metadata["Name"]
        if name.lower() == "neurd":  # Editable source ownership is guarded separately.
            continue
        distributions[name] = dist.version
        for entry in dist.files or ():
            path = Path(dist.locate_file(entry)).resolve()
            if path.suffix in (".pyc", ".nbi", ".nbc") or "__pycache__" in path.parts:
                continue
            if path.is_file() and str(path) not in files:
                files[str(path)] = sha(path)
    providers = {}
    for name in PROVIDERS:
        spec = importlib.util.find_spec(name)
        if spec is None or not spec.origin or not spec.origin.endswith(".so"):
            raise RuntimeError(f"Missing compiled provider: {name}")
        path = Path(spec.origin).resolve()
        providers[name] = {"path": str(path), "sha256": sha(path)}
    return {"python": sys.version, "executable": sys.executable,
            "platform": platform.platform(), "distributions": distributions,
            "files_sha256": files, "providers": providers}


def array_signature(value):
    """Exact ordered values with original dtype recorded separately.

    Numeric storage widths may change between compatible implementations. Compare
    canonical float64/int64 bytes, never rounded values, sorted faces or triangles.
    """
    import numpy as np

    if value is None:
        return None
    array = np.asarray(value)
    if array.dtype.hasobject:
        raise TypeError("Object arrays are not a scientific comparison format")
    digest = hashlib.sha256()
    if array.dtype.kind in "SU":
        canonical = "text"
        digest.update(json.dumps(array.tolist(), ensure_ascii=False).encode())
    else:
        dtype = {"f": "<f8", "i": "<i8", "u": "<u8", "b": "|b1"}.get(array.dtype.kind)
        if dtype is None:
            raise TypeError(f"Unsupported scientific array dtype: {array.dtype}")
        canonical = dtype
        # Chunk flat views instead of making a second copy of a whole mesh.
        flat = array.ravel(order="C")
        for start in range(0, flat.size, 262144):
            digest.update(np.ascontiguousarray(flat[start:start + 262144], dtype=dtype).tobytes())
    return {"shape": list(array.shape), "dtype": array.dtype.str,
            "canonical_dtype": canonical, "sha256": digest.hexdigest()}


def mesh_signature(mesh):
    if mesh is None:
        return None
    return {name: array_signature(getattr(mesh, name)) for name in ("vertices", "faces")}


def neuron_signature(neuron):
    somas = {name: mesh_signature(neuron.concept_network.nodes[name]["data"].mesh)
             for name in sorted(neuron.get_soma_node_names())}
    limbs = []
    for limb in neuron.limbs:
        branches = []
        for branch in limb.branches:
            masks = [] if branch.spines_face_idx is None else list(branch.spines_face_idx)
            spines = [] if branch.spines is None else list(branch.spines)
            branches.append(dict(mesh=mesh_signature(branch.mesh), skeleton=array_signature(branch.skeleton),
                masks=[array_signature(mask) for mask in masks],
                spines=[mesh_signature(getattr(spine, "mesh", spine)) for spine in spines],
                volumes=array_signature(branch.spines_volume), width=array_signature(branch.width),
                widths={key: array_signature(value) for key, value in sorted(branch.width_new.items())},
                width_arrays={key: array_signature(value) for key, value in sorted(branch.width_array.items())}))
        limbs.append(dict(mesh=mesh_signature(limb.mesh), skeleton=array_signature(limb.skeleton), branches=branches))
    return dict(somas=somas, limbs=limbs)


def export_manifest(root):
    import numpy as np

    result = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        entry = {"size_bytes": path.stat().st_size, "sha256": sha(path)}
        if path.suffix == ".npy":
            entry["arrays"] = array_signature(np.load(path, allow_pickle=False))
        elif path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as archive:
                entry["arrays"] = {name: array_signature(archive[name]) for name in archive.files}
        elif path.suffix == ".json":
            entry["json"] = json.loads(path.read_text())
        result[str(path.relative_to(root))] = entry
    return result


def values_only(value):
    if isinstance(value, dict):
        return {key: values_only(item) for key, item in value.items() if key != "dtype"}
    if isinstance(value, list):
        return [values_only(item) for item in value]
    return value


def compare_results(baseline, candidate):
    problems = []
    for key in ("config", "input", "metrics", "scientific", "providers", "patches"):
        if values_only(baseline[key]) != values_only(candidate[key]):
            problems.append(f"{key} differs")
    a, b = baseline["exports"], candidate["exports"]
    missing, added = sorted(a.keys() - b.keys()), sorted(b.keys() - a.keys())
    if missing:
        problems.append(f"Missing candidate exports: {missing}")
    if any(Path(name).name not in ADDED_EXPORTS or not re.fullmatch(
            r"limb_\d+/branch_\d+/[^/]+", name) for name in added):
        problems.append(f"Unexpected additional candidate exports: {added}")
    for name in sorted(a.keys() & b.keys()):
        left, right = a[name], b[name]
        if "arrays" in left:
            equal = values_only(left["arrays"]) == values_only(right.get("arrays"))
        elif "json" in left:
            left, right = dict(left["json"]), dict(right.get("json", {}))
            if Path(name).name == "branch_index_map.json":
                left.pop("branch_index_resolution", None)
                right.pop("branch_index_resolution", None)
            equal = left == right
        else:
            equal = left["sha256"] == right["sha256"]
        if not equal:
            problems.append(f"Export differs: {name}")
    return {"passed": not problems, "problems": problems, "common_files": len(a.keys() & b.keys()),
            "baseline_files": len(a), "candidate_files": len(b), "additional_candidate_files": added,
            "export_workloads_equal": not added and not missing,
            "dtype_policy": "Original dtype recorded; canonical exact numeric values compared"}


def worker(args):
    if args.cpu is not None:
        os.sched_setaffinity(0, {args.cpu})
    source, output = args.worker_source.resolve(), args.out.resolve()
    sys.path.insert(0, str(source))
    sys.path.insert(1, str(source / "tests/tools"))
    phases = []
    def mark(name, when=None):
        phases.append({"phase": name, "monotonic_seconds": time.perf_counter() if when is None else when})
        write_json(output / "timeline.json", phases)
    mark("import_setup")
    started = time.perf_counter()
    import random
    import numpy as np
    import neurd
    from neurd.run_config import RunConfig
    from neurd.segmentation_pipeline import segmentation_pipeline
    from neuron_metrics import extract_metrics
    if Path(neurd.__file__).resolve().parent != source / "neurd":
        raise RuntimeError("Imported a different NEURD source tree")
    if args.variant == "baseline":
        from mesh_tools.trimesh_utils import load_mesh_no_processing as load_mesh
        from process_all_neurons import save_segmentation
    else:
        from neurd.mesh_io import load_mesh
        from neurd.neuron_export import save_segmentation
    config = RunConfig(dataset=args.dataset, calculate_spines=True,
                       refinement="off", sdf_profile=args.sdf_profile)
    config.apply()
    random.seed(args.seed)
    np.random.seed(args.seed)
    durations = {"import_setup": time.perf_counter() - started}
    mark("load")
    started = time.perf_counter()
    mesh = load_mesh(str(args.neuron))
    durations["load"] = time.perf_counter() - started
    mark("construction")
    started = time.perf_counter()
    neuron = segmentation_pipeline(mesh, segment_id=args.segment_id, verbose=False, config=config)
    durations["construction"] = time.perf_counter() - started
    mark("export")
    started = time.perf_counter()
    save_segmentation(neuron, output / "export")
    finished = time.perf_counter()
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_bytes = int(rss if sys.platform == "darwin" else rss * 1024)
    durations["export"] = finished - started
    mark("quality_checks", finished)
    durations["pipeline_total"] = sum(durations[name] for name in ("load", "construction", "export"))
    durations["import_and_pipeline_total"] = durations["import_setup"] + durations["pipeline_total"]
    providers = {name: {"path": importlib.util.find_spec(name).origin,
                       "sha256": sha(importlib.util.find_spec(name).origin)} for name in PROVIDERS}
    metrics = extract_metrics(neuron, include_spines=True)
    bad = {key: value for key, value in metrics.items()
           if (key.startswith("invalid_") or key.startswith("overlapping_") or key.startswith("ambiguous_")) and value}
    if bad or metrics["n_spines_total"] <= 0:
        raise RuntimeError(f"Scientific quality checks failed: {bad}")
    receipt = dict(status="passed", variant=args.variant, config=config.as_dict(),
        input={"vertices": len(mesh.vertices), "faces": len(mesh.faces), "mesh": mesh_signature(mesh)},
        stages_seconds=durations, kernel_peak_rss_bytes=rss_bytes, metrics=metrics,
        scientific=neuron_signature(neuron), exports=export_manifest(output / "export"),
        providers=providers, patches=dict(neurd.PATCH_STATUS),
        numba_cache_dir=os.environ["NUMBA_CACHE_DIR"], thread_limits={key: os.environ[key] for key in THREAD_VARS},
        cpu_affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None)
    write_json(output / "result.json", receipt)


def run_child(command, directory, environment, timeout):
    import psutil

    samples, started = [], time.monotonic()
    with (directory / "output.log").open("w") as log:
        child = subprocess.Popen(command, cwd=directory / "work", env=environment,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        process = psutil.Process(child.pid)
        while child.poll() is None:
            if time.monotonic() - started > timeout:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
                write_json(directory / "rss_samples.json", samples)
                raise TimeoutError(f"Worker exceeded {timeout} seconds")
            try:
                own = process.memory_info().rss
                children = process.children(recursive=True)
                tree = own
                for descendant in children:
                    try:
                        tree += descendant.memory_info().rss
                    except psutil.NoSuchProcess:
                        pass
                samples.append(dict(monotonic_seconds=time.perf_counter(), worker_rss_bytes=own,
                                    tree_rss_bytes=tree, descendants=len(children)))
            except psutil.NoSuchProcess:
                pass
            time.sleep(0.05)
    write_json(directory / "rss_samples.json", samples)
    if child.returncode:
        raise RuntimeError(f"Worker exited {child.returncode}; see {directory / 'output.log'}")
    result = json.loads((directory / "result.json").read_text())
    timeline = json.loads((directory / "timeline.json").read_text())
    cutoff = timeline[-1]["monotonic_seconds"]
    timed = [sample for sample in samples if sample["monotonic_seconds"] <= cutoff]
    result["sampled_memory"] = {"interval_seconds": 0.05,
        "scope": "Through export; tree RSS sums shared pages and may miss short peaks",
        "peak_worker_rss_bytes": max((s["worker_rss_bytes"] for s in timed), default=0),
        "peak_tree_rss_bytes": max((s["tree_rss_bytes"] for s in timed), default=0),
        "stages": {point["phase"]: {
            "peak_worker_rss_bytes": max((s["worker_rss_bytes"] for s in timed
                if point["monotonic_seconds"] <= s["monotonic_seconds"] < timeline[i + 1]["monotonic_seconds"]), default=0),
            "peak_tree_rss_bytes": max((s["tree_rss_bytes"] for s in timed
                if point["monotonic_seconds"] <= s["monotonic_seconds"] < timeline[i + 1]["monotonic_seconds"]), default=0)}
            for i, point in enumerate(timeline[:-1])}}
    write_json(directory / "result.json", result)
    return result


def summarize(runs):
    result = {}
    for variant in ("baseline", "candidate"):
        selected = [run["result"] for run in runs if run["kind"] == "measured" and run["variant"] == variant]
        values = {name: [run["stages_seconds"][name] for run in selected] for name in selected[0]["stages_seconds"]}
        values["kernel_peak_rss_bytes"] = [run["kernel_peak_rss_bytes"] for run in selected]
        values["sampled_tree_peak_rss_bytes"] = [run["sampled_memory"]["peak_tree_rss_bytes"] for run in selected]
        result[variant] = {name: {"median": statistics.median(trials), "min": min(trials), "max": max(trials),
                                  "raw_trials": trials} for name, trials in values.items()}
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-source", type=Path)
    parser.add_argument("--candidate-source", type=Path)
    parser.add_argument("--neuron", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=1200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--segment-id", type=int, default=2530864375)
    parser.add_argument("--cpu", type=int, help="Pin workers to one allowed Linux CPU; supervisor stays unpinned")
    parser.add_argument("--dataset", choices=("h01", "microns"), default="h01")
    parser.add_argument("--sdf-profile", choices=("fast", "reference"), default="fast")
    parser.add_argument("--reference-metrics", type=Path)
    parser.add_argument("--worker-source", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--variant", choices=("baseline", "candidate"), help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.repeat <= 0 or not math.isfinite(args.timeout) or args.timeout <= 0 or not 0 <= args.seed <= 4294967295:
        parser.error("repeat/timeout must be positive; seed must be an unsigned 32-bit integer")
    if args.cpu is not None and (not hasattr(os, "sched_getaffinity") or args.cpu not in os.sched_getaffinity(0)):
        parser.error("--cpu must select a CPU in this process's allowed Linux affinity")
    if args.worker_source:
        try:
            worker(args)
        except Exception as exc:
            write_json(args.out.resolve() / "result.json", dict(status="failed", error=str(exc), traceback=traceback.format_exc()))
            raise
        return 0
    if not args.baseline_source or not args.candidate_source:
        parser.error("Both source trees are required")
    sources = {name: path.resolve() for name, path in (("baseline", args.baseline_source), ("candidate", args.candidate_source))}
    for source in sources.values():
        if not (source / "neurd/segmentation_pipeline.py").is_file():
            parser.error(f"Missing NEURD source tree: {source}")
    args.neuron = args.neuron.expanduser().resolve()
    args.out = args.out.expanduser().resolve()
    if not args.neuron.is_file() or args.out.exists():
        parser.error("Input must exist and output directory must be new")
    if any(args.out.is_relative_to(source) for source in sources.values()):
        parser.error("Output must be outside the guarded source trees")
    if sha(sources["baseline"] / "tests/tools/neuron_metrics.py") != sha(sources["candidate"] / "tests/tools/neuron_metrics.py"):
        parser.error("Both source trees must use the same frozen metrics helper")
    args.out.mkdir(parents=True)
    before = dict(sources={name: source_manifest(path) for name, path in sources.items()},
                  environment=environment_manifest(), input_sha256=sha(args.neuron), harness_sha256=sha(__file__))
    write_json(args.out / "guard_before.json", before)
    report = dict(status="running", input_identifier=args.neuron.name, order=run_order(args.repeat), runs=[],
                  cpu=args.cpu, supervisor_affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
                  cpu_governors={str(cpu): path.read_text().strip() for cpu in os.sched_getaffinity(0)
                    if (path := Path(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_governor")).is_file()}
                    if hasattr(os, "sched_getaffinity") else {},
                  memory_scope="Kernel worker lifetime peak through export; sampled process-tree RSS estimate",
                  timing_scope="Function-stage wall durations; totals are their sum; timeline-write bookkeeping and QC excluded",
                  setup_scope="One excluded setup run per source; persistent variant-specific Numba caches; filesystem caches uncontrolled")
    anchors = {}
    try:
        for kind, pair, variant in run_order(args.repeat):
            directory = args.out / f"{kind}_{pair:02d}_{variant}"
            (directory / "work").mkdir(parents=True)
            env = {key: value for key, value in os.environ.items() if not key.startswith("NEURD_")}
            env.update(PYTHONPATH=str(sources[variant]), PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1", PYTHONHASHSEED="0",
                       NUMBA_CACHE_DIR=str(args.out / f"numba_cache_{variant}"), MPLCONFIGDIR=str(directory / "mplconfig"))
            env.update({name: "1" for name in THREAD_VARS})
            command = [sys.executable, str(Path(__file__).resolve()), "--worker-source", str(sources[variant]), "--variant", variant,
                       "--neuron", str(args.neuron), "--out", str(directory), "--seed", str(args.seed),
                       "--segment-id", str(args.segment_id), "--dataset", args.dataset, "--sdf-profile", args.sdf_profile]
            if args.cpu is not None:
                command += ["--cpu", str(args.cpu)]
            report["active_run"] = dict(kind=kind, pair=pair, variant=variant, directory=directory.name)
            write_json(args.out / "report.json", report)
            print(f"Starting {kind} {pair}: {variant}", flush=True)
            result = run_child(command, directory, env, args.timeout)
            comparison = compare_results(anchors.get("baseline", result), result)
            if variant in anchors:
                repeat_comparison = compare_results(anchors[variant], result)
                if not repeat_comparison["passed"]:
                    comparison["problems"] += [f"Repeatability: {problem}" for problem in repeat_comparison["problems"]]
                    comparison["passed"] = False
            anchors.setdefault(variant, result)
            if args.reference_metrics and variant == "candidate":
                expected = json.loads(args.reference_metrics.read_text())
                if expected.get("metrics", expected) != result["metrics"]:
                    comparison["passed"] = False
                    comparison["problems"].append("Candidate differs from reference metrics")
            report["runs"].append(dict(kind=kind, pair=pair, variant=variant, directory=directory.name,
                                       comparison=comparison, result=result))
            write_json(args.out / "report.json", report)
            print(f"Finished {kind} {pair}: {variant}; pipeline={result['stages_seconds']['pipeline_total']:.3f}s; exact={comparison['passed']}", flush=True)
            if not comparison["passed"]:
                raise RuntimeError("; ".join(comparison["problems"]))
        report["summary"] = summarize(report["runs"])
        report["status"] = "passed"
    except Exception as exc:
        report.update(status="failed", error=str(exc), traceback=traceback.format_exc())
    finally:
        after = dict(sources={name: source_manifest(path) for name, path in sources.items()},
                     environment=environment_manifest(), input_sha256=sha(args.neuron), harness_sha256=sha(__file__))
        write_json(args.out / "guard_after.json", after)
        report["guard_before_after_equal"] = before == after
        if before != after:
            report.update(status="failed", error="Sources, environment, input or harness changed during trials")
        write_json(args.out / "report.json", report)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
