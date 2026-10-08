#!/usr/bin/env python3
"""Compare final spine meshes on preselected H01 branches with a shared fresh CGAL run.

The two modes share a fresh CGAL run and its initial size/length filter. Counts
refer to the FINAL exported spines, not the unfiltered CGAL pieces. A separate
code snapshot can incorporate existing workspace changes into both modes.

python scripts/validate_spine_refinement.py --selection selection.json \
    --output /tmp/h01-validation --code-snapshot /path/to/code_snapshot --workers 2
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter
import traceback
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def classify_change(before, after):
    if before is None:
        return "newly_passing_filters"
    if after is None:
        return "removed"
    if set(before) == set(after):
        return "unchanged"
    if not set(before).issubset(after):
        raise ValueError("Final refinement lost original seed faces")
    return "grown"


def code_provenance(code_root):
    files = ["spine_utils.py", "spine_refinement.py", "parameters.py"]
    if (code_root / "neurd/spine_volume.py").exists():
        files.append("spine_volume.py")
    result = {name: sha(code_root / "neurd" / name) for name in files}
    snapshot = code_root / "provenance.json"
    if snapshot.exists():
        recorded = json.loads(snapshot.read_text())
        for name, value in result.items():
            if recorded["source_sha256"]["neurd/" + name] != value:
                raise ValueError("Code snapshot differs from its provenance")
        return dict(files=result, snapshot=recorded)
    return dict(files=result)


def same_prefix_code(before, after):
    """Only the optional refinement module may change when replaying a prefix.

    Baseline extraction and the initial length filter never call this module.
    Compare every snapshotted upstream source, not just spine_utils.py.
    """
    def upstream(provenance):
        files = provenance.get("snapshot", {}).get("source_sha256")
        if files is None:
            files = {"neurd/" + name: digest for name, digest in provenance["files"].items()}
        return {name: digest for name, digest in files.items() if name != "neurd/spine_refinement.py"}
    return upstream(before) == upstream(after)


def run_job(row, destination, args, manifest):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/neurd-spine-refinement-mpl")
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "1"
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(args.code_snapshot))
    import neurd  # Apply compatibility patches before mesh_tools.
    import numpy as np
    import trimesh
    from mesh_tools import trimesh_utils as tu
    import cgal_Segmentation_Module as csm
    from neurd import parameters, spine_utils as spu, spine_refinement as sr

    assert Path(neurd.__file__).resolve().parent.parent == args.code_snapshot
    if not str(csm.__file__).endswith(".so"):
        raise RuntimeError("Validation requires the real compiled CGAL extension")
    parameters.params.use("h01")
    os.environ["NEURD_SDF_RAYS"] = str(args.rays)
    destination.mkdir(parents=True, exist_ok=True)
    started = perf_counter()

    def progress(phase):
        write_json(destination / "progress.json", dict(key=row["key"], phase=phase,
                   elapsed_seconds=perf_counter() - started))

    def load(path):
        return trimesh.load(path, process=False, force="mesh")

    def pack(groups, raw_ids, path):
        offsets = np.r_[0, np.cumsum([len(ids) for ids in groups])]
        flat = np.concatenate(groups) if groups else np.empty(0, dtype=np.int64)
        np.savez_compressed(path, faces=flat, offsets=offsets, raw_ids=raw_ids)

    def verify(mesh, ids, parent):
        def keys(vertices, faces):
            triangles = np.asarray(vertices[faces], dtype=np.float64)
            vertex = triangles.view(np.dtype((np.void, 24))).reshape(-1, 3)
            vertex.sort(axis=1)
            return np.sort(vertex.copy().view(np.dtype((np.void, 72))).reshape(-1))
        assert len(mesh.faces) == len(ids)
        assert np.array_equal(keys(mesh.vertices, mesh.faces), keys(parent.vertices, parent.faces[ids]))

    progress("load")
    source = Path(row["source"])
    parent, skeleton = load(source), np.load(row["skeleton"])
    soma_path = source.parents[2] / "soma/soma_mesh.off"
    soma = load(soma_path).vertices if soma_path.exists() else None
    branch = SimpleNamespace(mesh=parent, skeleton=skeleton)
    kwargs = dict(soma_verts=soma, soma_vertex_nullification=soma is not None,
                  cgal_folder=destination / "cgal", print_flag=False)
    captured = {}
    original_raw, original_segmentation = spu.get_spine_meshes_unfiltered_from_mesh, tu.mesh_segmentation
    original_length_filter = spu.filter_spine_meshes

    def raw(*a, **kw):
        raw_started = perf_counter()
        requested = kw.get("return_mesh_idx", False)
        meshes, indices = original_raw(*a, **{**kw, "return_mesh_idx": True})
        captured.update(meshes=list(meshes), indices=[np.asarray(ids, dtype=np.int64) for ids in indices],
                        seconds=perf_counter() - raw_started)
        pack(captured["indices"], np.arange(len(indices)), destination / "raw_faces.npz")
        return (meshes, indices) if requested else meshes

    def segmentation(*a, **kw):
        result = original_segmentation(*a, **kw)
        if not kw.get("return_meshes", True):
            captured["labels"], captured["sdf"] = np.asarray(result[0]), np.asarray(result[1])
            np.savez_compressed(destination / "cgal.npz", labels=captured["labels"], sdf=captured["sdf"])
        return result

    def length_filter(meshes, *a, **kw):
        call_started = perf_counter()
        result = original_length_filter(meshes, *a, **kw)
        captured.update(length_meshes=list(result), length_seconds=perf_counter() - call_started)
        identity = {id(mesh): i for i, mesh in enumerate(captured["meshes"])}
        write_json(destination / "shared_prefix.json", dict(
            source_sha256=sha(source), skeleton_sha256=sha(row["skeleton"]),
            raw_seconds=captured["seconds"], length_seconds=captured["length_seconds"],
            length_raw_ids=[identity[id(mesh)] for mesh in result], manifest=manifest))
        return result

    reused_result = None
    prior_root = args.reuse_completed or args.reuse_prefix_from
    if prior_root:
        prior_directory = prior_root / "branches" / row["key"]
        prior_path = prior_directory / "result.json"
        if prior_path.exists():
            prior = json.loads(prior_path.read_text())
            code_matches = (same_prefix_code(prior["manifest"]["code"], manifest["code"])
                            if args.reuse_prefix_from else prior["manifest"]["code"] == manifest["code"])
            if (prior.get("status") != "passed" or not code_matches
                    or prior["manifest"]["rays"] != args.rays
                    or prior["parent_sha256"] != sha(source)
                    or prior["skeleton_sha256"] != sha(row["skeleton"])):
                raise ValueError("Previously completed branch has incompatible provenance")
            reused_result = prior
            with np.load(prior_directory / "raw_faces.npz") as cache:
                indices = np.split(cache["faces"], cache["offsets"][1:-1])
            captured.update(indices=indices, meshes=[parent.submesh([ids], append=True) for ids in indices],
                            seconds=prior["shared_raw_extraction_seconds"], length_seconds=None)
            saved_prefix = prior_directory / "shared_prefix.json"
            passed_ids = (json.loads(saved_prefix.read_text())["length_raw_ids"] if saved_prefix.exists()
                          else [c["raw_input_index"] for c in prior["refinement"]["candidates"]])
            captured["length_meshes"] = [captured["meshes"][i] for i in passed_ids]
            with np.load(prior_directory / "cgal.npz") as cache:
                captured["labels"], captured["sdf"] = cache["labels"], cache["sdf"]
            write_json(destination / "shared_prefix.json", dict(
                source_sha256=sha(source), skeleton_sha256=sha(row["skeleton"]),
                raw_seconds=captured["seconds"], length_seconds=None,
                length_raw_ids=passed_ids, reused_completed=str(prior_directory), manifest=manifest))

    def cached_length(meshes, *a, **kw):
        assert len(meshes) == len(captured["meshes"])
        assert all(a is b for a, b in zip(meshes, captured["meshes"]))
        return captured["length_meshes"]

    def cached_raw(*a, **kw):
        meshes, indices = captured["meshes"], captured["indices"]
        return (meshes, indices) if kw.get("return_mesh_idx", False) else meshes

    progress("baseline_from_saved_prefix" if reused_result else "fresh_cgal_and_baseline")
    np.random.seed(0)
    spu.get_spine_meshes_unfiltered_from_mesh = cached_raw if reused_result else raw
    tu.mesh_segmentation = original_segmentation if reused_result else segmentation
    spu.filter_spine_meshes = cached_length if reused_result else length_filter
    baseline_started = perf_counter()
    try:
        baseline, baseline_volumes = spu.calculate_spines_on_branch(branch, geometric_refinement=False, **kwargs)
    finally:
        spu.get_spine_meshes_unfiltered_from_mesh, tu.mesh_segmentation = original_raw, original_segmentation
        spu.filter_spine_meshes = original_length_filter
    baseline_seconds = reused_result["fresh_baseline_seconds"] if reused_result else perf_counter() - baseline_started
    raw_meshes, raw_indices = captured["meshes"], captured["indices"]
    raw_identity = {id(mesh): i for i, mesh in enumerate(raw_meshes)}
    baseline_ids = [raw_identity[id(mesh)] for mesh in baseline]
    baseline_groups = [np.unique(raw_indices[i]) for i in baseline_ids]
    pack(raw_indices, np.arange(len(raw_indices)), destination / "raw_faces.npz")
    np.savez_compressed(destination / "cgal.npz", labels=captured["labels"], sdf=captured["sdf"])
    refined_capture, stage = {}, {}
    original_refine = sr.refine_spine_face_indices

    def refinement(*a, **kw):
        groups, diagnostics = original_refine(*a, **kw)
        refined_capture.update(groups=groups, diagnostics=diagnostics)
        return groups, diagnostics

    @contextmanager
    def reuse_raw():
        spu.get_spine_meshes_unfiltered_from_mesh = cached_raw
        sr.refine_spine_face_indices = refinement
        spu.filter_spine_meshes = cached_length
        try:
            yield
        finally:
            spu.get_spine_meshes_unfiltered_from_mesh = original_raw
            sr.refine_spine_face_indices = original_refine
            spu.filter_spine_meshes = original_length_filter

    timings = []
    reference = None
    with reuse_raw():
        for repetition in range(args.repeat):
            timing = {}
            for enabled in ((False, True) if repetition % 2 == 0 else (True, False)):
                progress(("refinement" if enabled else "baseline_replay") + f"_{repetition}")
                call_started = perf_counter()
                result, volumes = spu.calculate_spines_on_branch(
                    branch, geometric_refinement=enabled,
                    geometric_refinement_diagnostics=stage if enabled else None, **kwargs)
                timing[("refined" if enabled else "baseline") + "_post_seconds"] = perf_counter() - call_started
                if enabled:
                    final_records = sorted((c for c in stage["candidates"] if c["final_output_index"] is not None),
                                           key=lambda c: c["final_output_index"])
                    assert [c["final_output_index"] for c in final_records] == list(range(len(result)))
                    new_ids = [c["raw_input_index"] for c in final_records]
                    new_groups = [refined_capture["groups"][c["output_index"]] for c in final_records]
                    if reference is not None:
                        assert reference[0] == new_ids
                        assert all(np.array_equal(a, b) for a, b in zip(reference[1], new_groups))
                    reference = new_ids, new_groups
                    refined, refined_volumes = result, volumes
                    timing["geometry_seconds"] = stage["seconds"]
                else:
                    assert [raw_identity[id(mesh)] for mesh in result] == baseline_ids
            timings.append(timing)

    progress("export_and_audit")
    if reused_result:
        modes = [("baseline", baseline_ids, baseline_groups)]
        if not args.reuse_prefix_from:
            modes.append(("refined", new_ids, new_groups))
        for name, ids, groups in modes:
            with np.load(prior_directory / (name + "_faces.npz")) as cache:
                previous = np.split(cache["faces"], cache["offsets"][1:-1])
                assert np.array_equal(ids, cache["raw_ids"])
                assert all(np.array_equal(np.sort(a), np.sort(b)) for a, b in zip(previous, groups))
    by_before = {i: (j, ids) for j, (i, ids) in enumerate(zip(baseline_ids, baseline_groups))}
    by_after = {i: (j, ids) for j, (i, ids) in enumerate(zip(new_ids, new_groups))}
    by_record = {c["raw_input_index"]: c for c in stage["candidates"]}
    for name, meshes, groups, ids in (("baseline", baseline, baseline_groups, baseline_ids),
                                    ("refined", refined, new_groups, new_ids)):
        folder = destination / name
        folder.mkdir(exist_ok=True)
        for i, (mesh, faces) in enumerate(zip(meshes, groups)):
            verify(mesh, faces, parent)
            mesh.export(folder / f"spine_{i:03d}.off")
        pack(groups, ids, destination / (name + "_faces.npz"))
    changes = []
    _, canonical = np.unique(np.sort(parent.faces, axis=1), axis=0, return_inverse=True)
    raw_owner = np.full(int(canonical.max()) + 1, -1, dtype=int)
    for i, ids in enumerate(raw_indices):
        raw_owner[canonical[ids]] = i
    added_owner = np.full_like(raw_owner, -1)
    for raw_id in sorted(set(baseline_ids) | set(new_ids)):
        before = by_before.get(raw_id)
        after = by_after.get(raw_id)
        action = classify_change(None if before is None else before[1], None if after is None else after[1])
        original = np.unique(raw_indices[raw_id])
        if after is not None:
            assert np.isin(original, after[1]).all()
            added = np.setdiff1d(after[1], original)
            physical_added = np.unique(canonical[added])
            assert np.all((raw_owner[physical_added] == -1) | (raw_owner[physical_added] == raw_id))
            assert np.all(added_owner[physical_added] == -1)
            added_owner[physical_added] = raw_id
        if action != "unchanged":
            record = by_record.get(raw_id, {})
            change = dict(raw_id=raw_id, action=action,
                          baseline_index=None if before is None else before[0],
                          refined_index=None if after is None else after[0],
                          original_faces=len(original), new_faces=0 if after is None else len(after[1]),
                          refinement_action=record.get("action"), reason=record.get("reason"),
                          geometry_before=record.get("geometry_before"),
                          geometry_proposed=record.get("geometry_proposed"))
            changes.append(change)
            folder = destination / "changes"
            folder.mkdir(exist_ok=True)
            before_mesh = parent.submesh([original], append=True)
            before_mesh.export(folder / f"raw_{raw_id:04d}_before.off")
            np.save(folder / f"raw_{raw_id:04d}_before_faces.npy", original)
            if after is not None:
                parent.submesh([after[1]], append=True).export(folder / f"raw_{raw_id:04d}_after.off")
                np.save(folder / f"raw_{raw_id:04d}_after_faces.npy", after[1])
            elif record.get("action") == "grow":
                proposed = refined_capture["groups"][record["output_index"]]
                parent.submesh([proposed], append=True).export(folder / f"raw_{raw_id:04d}_proposed.off")
                np.save(folder / f"raw_{raw_id:04d}_proposed_faces.npy", proposed)
    median_extra = float(np.median([t["refined_post_seconds"] - t["baseline_post_seconds"] for t in timings]))
    result = dict(status="passed", key=row["key"], selection=row, manifest=manifest,
                  parent_sha256=sha(source), skeleton_sha256=sha(row["skeleton"]),
                  raw_candidates=len(raw_indices), archived_spines=row["archived_spines"],
                  baseline_spines=len(baseline), refined_spines=len(refined),
                  change_counts=dict(Counter(c["action"] for c in changes)), changes=changes,
                  refinement=stage, fresh_baseline_seconds=baseline_seconds,
                  shared_raw_extraction_seconds=captured["seconds"], timings=timings,
                  shared_length_filter_seconds=captured["length_seconds"],
                  reused_completed=None if reused_result is None else str(prior_directory),
                  reused_prefix_only=bool(args.reuse_prefix_from),
                  previous_refined_spines=None if reused_result is None else reused_result["refined_spines"],
                  median_extra_seconds=median_extra,
                  estimated_extra_fraction=median_extra / baseline_seconds,
                  cgal_shared=True, cgal_extension=str(csm.__file__),
                  parameters=parameters.params._data,
                  geometry_checks=dict(seed_faces_preserved=True, no_neighbor_seed_capture=True,
                                       no_shared_added_faces=True, parent_triangles_only=True,
                                       repeated_outputs_identical=True),
                  anatomical_ground_truth_available=False,
                  elapsed_seconds=perf_counter() - started)
    write_json(destination / "result.json", result)
    progress("done")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--code-snapshot", type=Path, default=REPO)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--rays", type=int, default=25)
    parser.add_argument("--job-key")
    parser.add_argument("--reuse-completed", type=Path)
    parser.add_argument("--reuse-prefix-from", type=Path,
                        help="Reuse verified CGAL/length results when only spine_refinement.py has changed")
    args = parser.parse_args()
    args.selection, args.output, args.code_snapshot = (p.resolve() for p in (args.selection, args.output, args.code_snapshot))
    if args.reuse_completed:
        args.reuse_completed = args.reuse_completed.resolve()
    if args.reuse_prefix_from:
        args.reuse_prefix_from = args.reuse_prefix_from.resolve()
    if args.reuse_completed and args.reuse_prefix_from:
        parser.error("Choose either --reuse-completed or --reuse-prefix-from")
    if args.repeat < 1 or args.workers < 1:
        parser.error("--repeat and --workers must be positive")
    rows = json.loads(args.selection.read_text())["rows"]
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = dict(selection_sha256=sha(args.selection), script_sha256=sha(__file__),
                    code=code_provenance(args.code_snapshot), rays=args.rays,
                    repetitions=args.repeat, dataset="h01", shared_fresh_cgal=True,
                    shared_length_filter=True, reuse_prefix_from=None if args.reuse_prefix_from is None
                    else str(args.reuse_prefix_from))
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("Resume manifest differs; choose a new output folder")
    else:
        write_json(manifest_path, manifest)
    if args.job_key:
        row = next(r for r in rows if r["key"] == args.job_key)
        destination = args.output / "branches" / row["key"]
        result_path = destination / "result.json"
        if result_path.exists() and json.loads(result_path.read_text()).get("status") == "passed":
            return
        destination.mkdir(parents=True, exist_ok=True)
        os.chdir(destination)
        try:
            run_job(row, destination, args, manifest)
        except Exception:
            write_json(destination / "failure.json", dict(key=row["key"], traceback=traceback.format_exc()))
            raise
        return

    def execute(row):
        directory = args.output / "branches" / row["key"]
        directory.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, str(Path(__file__).resolve()), "--selection", str(args.selection),
                   "--output", str(args.output), "--code-snapshot", str(args.code_snapshot),
                   "--repeat", str(args.repeat), "--rays", str(args.rays), "--job-key", row["key"]]
        if args.reuse_completed:
            command.extend(["--reuse-completed", str(args.reuse_completed)])
        if args.reuse_prefix_from:
            command.extend(["--reuse-prefix-from", str(args.reuse_prefix_from)])
        with (directory / "run.log").open("w") as log:
            process = subprocess.run(command, cwd=directory, stdout=log, stderr=subprocess.STDOUT)
        if process.returncode:
            return dict(status="failed", key=row["key"], log=str(directory / "run.log"))
        return json.loads((directory / "result.json").read_text())

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(execute, row): row for row in rows}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            write_json(args.output / "results.json", sorted(results, key=lambda r: r["key"]))
            print(json.dumps(dict(completed=len(results), total=len(rows), key=result["key"],
                                  status=result["status"], old=result.get("baseline_spines"),
                                  new=result.get("refined_spines"), changes=result.get("change_counts"))), flush=True)
    fields = ["key", "archived_spines", "baseline_spines", "refined_spines", "raw_candidates",
              "previous_refined_spines", "fresh_baseline_seconds", "median_extra_seconds", "estimated_extra_fraction"]
    with (args.output / "counts.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for result in sorted(results, key=lambda r: r["key"]):
            if result["status"] == "passed":
                writer.writerow({k: result[k] for k in fields})
    if any(r["status"] != "passed" for r in results):
        raise RuntimeError("Some branches failed; inspect their logs")


if __name__ == "__main__":
    main()
