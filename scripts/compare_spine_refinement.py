#!/usr/bin/env python3
"""Compare archived or freshly extracted spine partitions and time refinement.

Examples (run in the NEURD environment):
  python scripts/compare_spine_refinement.py --output /tmp/spine-comparison \
      --examples-bundle /path/to/segmentation_debug_examples_20261008
  python scripts/compare_spine_refinement.py --output /tmp/spine-comparison \
      --branch /path/to/branch_mesh.off --resegment --dataset h01 --rays 25

Writes machine-readable measurements, original face indices and changed OFF
meshes. Source meshes and segmentation settings are never overwritten.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/neurd-spine-refinement-mpl")
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(name, "1")

import neurd  # Compatibility patches before mesh_tools imports.
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from neurd.spine_refinement import refine_spine_face_indices

REFINEMENT_SHA256 = hashlib.sha256(
    (Path(__file__).resolve().parents[1] / "neurd" / "spine_refinement.py").read_bytes()).hexdigest()


def load(path):
    return trimesh.load(path, process=False, force="mesh")


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False,
                                    allow_nan=False) + "\n")


def archive_indices(parent, directory):
    tree = cKDTree(parent.vertices)
    lookup = {tuple(face): i for i, face in enumerate(np.sort(parent.faces, axis=1))}
    names, groups = [], []
    for path in sorted(directory.glob("*.off")):
        mesh = load(path)
        distance, vertex = tree.query(mesh.vertices)
        if np.any(distance > 1e-6):
            raise ValueError(f"{path} has vertices outside the parent mesh")
        groups.append(np.unique([lookup[tuple(face)] for face in np.sort(vertex[mesh.faces], axis=1)]))
        names.append(path.name)
    return names, groups


@contextmanager
def trace_raw(spu):
    original = spu.get_spine_meshes_unfiltered_from_mesh
    captured = {}

    def extract(*args, **kwargs):
        requested_indices = kwargs.get("return_mesh_idx", False)
        started = perf_counter()
        meshes, indices = original(*args, **{**kwargs, "return_mesh_idx": True})
        captured.update(indices=[np.unique(ids) for ids in indices], meshes=list(meshes),
                        seconds=perf_counter() - started)
        return (meshes, indices) if requested_indices else meshes

    spu.get_spine_meshes_unfiltered_from_mesh = extract
    try:
        yield captured
    finally:
        spu.get_spine_meshes_unfiltered_from_mesh = original


def extract_pair(parent, skeleton, destination, dataset, rays, source, config, repeat):
    from neurd import parameters, spine_utils as spu
    import cgal_Segmentation_Module as csm

    if not str(csm.__file__).endswith(".so"):
        raise RuntimeError("Comparison requires the real compiled CGAL extension")
    parameters.params.use("h01" if dataset == "h01" else "microns")
    os.environ["NEURD_SDF_RAYS"] = str(rays)
    branch = SimpleNamespace(mesh=parent, skeleton=skeleton)
    soma_path = source.parents[2] / "soma" / "soma_mesh.off"
    soma = load(soma_path).vertices if soma_path.exists() else None
    stage = {}
    records, partitions = [], []
    for enabled in (False, True):
        np.random.seed(0)
        started = perf_counter()
        with trace_raw(spu) as raw:
            meshes, _ = spu.calculate_spines_on_branch(
                branch, soma_verts=soma, soma_vertex_nullification=soma is not None,
                cgal_folder=destination / "cgal", geometric_refinement=enabled,
                geometric_refinement_config=config,
                geometric_refinement_diagnostics=stage if enabled else None)
        elapsed = perf_counter() - started
        partitions.append(raw["indices"])
        records.append(dict(enabled=enabled, seconds=elapsed, final_spines=len(meshes)))
        records[-1]["raw_extraction_seconds"] = raw["seconds"]
        folder = destination / ("refined_final" if enabled else "baseline_final")
        folder.mkdir(exist_ok=True)
        for i, mesh in enumerate(meshes):
            mesh.export(folder / f"spine_{i:03d}.off")
    equivalent = (len(partitions[0]) == len(partitions[1])
                  and all(np.array_equal(a, b) for a, b in zip(*partitions)))
    if not equivalent:
        raise RuntimeError("CGAL partitions differ; this is not a controlled comparison")
    # Reuse exactly the same raw partition to separate postprocessing cost from
    # CGAL runtime variation. These timings include all existing branch filters.
    original_extract = spu.get_spine_meshes_unfiltered_from_mesh
    cached_meshes, cached_indices = raw["meshes"], partitions[0]

    def cached(*args, **kwargs):
        return (cached_meshes, cached_indices) if kwargs.get("return_mesh_idx") else cached_meshes

    spu.get_spine_meshes_unfiltered_from_mesh = cached
    replays = []
    try:
        for i in range(repeat):
            row = {}
            for enabled in ((False, True) if i % 2 == 0 else (True, False)):
                timing = {}
                started = perf_counter()
                meshes, _ = spu.calculate_spines_on_branch(
                    branch, soma_verts=soma, soma_vertex_nullification=soma is not None,
                    geometric_refinement=enabled, geometric_refinement_config=config,
                    geometric_refinement_diagnostics=timing if enabled else None)
                name = "refined" if enabled else "baseline"
                row[name + "_seconds"] = perf_counter() - started
                row[name + "_spines"] = len(meshes)
                if enabled:
                    row["geometry_seconds"] = timing["seconds"]
            replays.append(row)
    finally:
        spu.get_spine_meshes_unfiltered_from_mesh = original_extract
    return partitions[0], dict(runs=records, raw_partitions_identical=equivalent,
                              refinement=stage, cgal_extension=csm.__file__, rays=rays,
                              cached_partition_replays=replays,
                              median_postprocessing_extra_seconds=float(np.median([
                                  r["refined_seconds"] - r["baseline_seconds"] for r in replays])))


def compare(source, skeleton_path, destination, args, cases):
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / "comparison.json").exists():
        raise FileExistsError(f"Existing result: {destination}; choose a new output")
    parent, skeleton = load(source), np.load(skeleton_path)
    config = {"maximum_repairs_per_branch": 0} if args.no_growth else None
    if args.resegment:
        groups, full = extract_pair(parent, skeleton, destination, args.dataset, args.rays, source, config, args.repeat)
        names = [f"raw_spine_{i:03d}.off" for i in range(len(groups))]
    else:
        names, groups = archive_indices(parent, source.parent / "spines")
        full = None
    repetitions, diagnostics = [], None
    first_result = None
    for _ in range(args.repeat):
        refined, diagnostics = refine_spine_face_indices(
            parent, skeleton, groups, config=config)
        if first_result is None:
            first_result = refined
        else:
            assert len(first_result) == len(refined)
            assert all(np.array_equal(a, b) for a, b in zip(first_result, refined))
        repetitions.append(diagnostics["seconds"])
    diagnostics["repeat_seconds"] = repetitions
    diagnostics["median_seconds"] = float(np.median(repetitions))
    for record in diagnostics["candidates"]:
        record["source_spine"] = names[record["input_index"]]
        before = groups[record["input_index"]]
        after = None if record["output_index"] is None else refined[record["output_index"]]
        if after is None or not np.array_equal(before, after):
            stem = Path(record["source_spine"]).stem
            parent.submesh([before], append=True).export(destination / f"{stem}_before.off")
            np.save(destination / f"{stem}_before_faces.npy", before)
            if after is not None:
                parent.submesh([after], append=True).export(destination / f"{stem}_after.off")
                np.save(destination / f"{stem}_after_faces.npy", after)
        if after is not None:
            assert np.isin(before, after).all(), "Refinement lost original seed faces"
            assert len(after) == len(np.unique(after))
            for j, neighbor in enumerate(groups):
                if j != record["input_index"]:
                    assert not np.setdiff1d(np.intersect1d(after, neighbor), before).size
    example_results = []
    for case in cases:
        old = np.unique(np.load(args.examples_bundle / case["case_folder"] / "parent_triangle_indices.npy"))
        overlap = [len(np.intersect1d(old, ids)) for ids in groups]
        best = int(np.argmax(overlap)) if overlap and max(overlap) else None
        example_results.append(dict(key=case["canonical_key"], group=case["group"],
                                    matched_input=best,
                                    decision=None if best is None else diagnostics["candidates"][best]))
    result = dict(source=str(source), dataset=args.dataset, parent_faces=len(parent.faces),
                  parent_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  skeleton_sha256=hashlib.sha256(skeleton_path.read_bytes()).hexdigest(),
                  refinement_code_sha256=REFINEMENT_SHA256,
                  versions=dict(numpy=np.__version__, trimesh=trimesh.__version__),
                  before_spines=len(groups), after_spines=len(refined), refinement=diagnostics,
                  full_pipeline=full, examples=example_results,
                  seed_faces_preserved=True, neighboring_seeds_preserved=True,
                  anatomical_ground_truth_available=False)
    write_json(destination / "comparison.json", result)
    print(json.dumps(dict(branch=str(source), before=len(groups), after=len(refined),
                          median_seconds=diagnostics["median_seconds"], counts=diagnostics["counts"])), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--branch", type=Path, action="append", default=[])
    parser.add_argument("--examples-bundle", type=Path)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--dataset", choices=("h01", "minnie65"), default="h01")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--resegment", action="store_true")
    parser.add_argument("--no-growth", action="store_true", help="Only reject surface scraps; retain truncated candidates")
    parser.add_argument("--rays", type=int, default=25)
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be positive")
    cases = []
    if args.examples_bundle:
        args.examples_bundle = args.examples_bundle.resolve()
        cases = [c for c in json.loads((args.examples_bundle / "cases.json").read_text())
                 if c["dataset"] == args.dataset and (not args.case or any(x in c["canonical_key"] for x in args.case))]
    sources = {Path(c["source_branch_path"]): Path(c["source_skeleton_path"]) for c in cases}
    sources.update({p.resolve(): p.resolve().parent / "branch_skeleton.npy" for p in args.branch})
    if not sources:
        parser.error("Select --branch or --examples-bundle")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    for source, skeleton in sources.items():
        key = "_".join(source.parts[-4:-1])
        selected = [c for c in cases if Path(c["source_branch_path"]) == source]
        results.append(compare(source, skeleton, args.output / key, args, selected))
        write_json(args.output / "measurements.json", results)


if __name__ == "__main__":
    main()
