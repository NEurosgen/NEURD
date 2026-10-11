#!/usr/bin/env python3
"""Read-only diagnosis of unresolved branches in an archived H01 batch run.

This tool computes no spine segmentation and writes only a separate diagnostic
directory. Width probes use the archived NEURD modules with currently installed
mesh_tools dependencies; the receipt distinguishes fingerprinted dependencies
from historical dependencies for which no fingerprint was recorded.
"""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import sys
import time
from types import SimpleNamespace

import numpy as np

SCHEMA_VERSION = 1
WIDTH_THRESHOLD_NM = 115.0
DISTANCE_THRESHOLD_NM = 3000.0
COMPLETED_STATUSES = {"complete", "resumed"}
BEHAVIOR_ENVIRONMENT_NAMES = (
    "NEURD_OWNED_SKELETAL_DISTANCE", "NEURD_LEGACY_VERTEX_COMPONENTS", "NEURD_LEGACY_MATCHING_VERTICES",
    "NEURD_DECIMATE_SINGLE_STAGE", "NEURD_MESHLAB_DECIMATE", "NEURD_ENABLE_COMPRESSED_PICKLE",
    "NEURD_REAL_POISSON", "NEURD_POISSON_DEPTH", "NEURD_POISSON_BACKEND", "NEURD_POISSON_ITERS",
    "NEURD_SDF_RAYS", "NEURD_SKIP_SOMA_END_SPLIT", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "MKL_DYNAMIC", "OMP_NUM_THREADS",
    "OMP_DYNAMIC", "OMP_THREAD_LIMIT", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_receipt(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha256(path), bytes=path.stat().st_size)


def off_counts(path):
    """Read only the OFF header for branches outside the geometry probe budget."""
    with Path(path).open() as stream:
        lines = (line.split("#", 1)[0].strip() for line in stream)
        header = next(line for line in lines if line)
        if header != "OFF":
            raise ValueError("Expected the exported OFF header")
        counts = next(line for line in lines if line).split()
    vertices, faces = int(counts[0]), int(counts[1])
    if vertices < 1 or faces < 1:
        raise ValueError("Empty exported parent surface")
    return vertices, faces


def unresolved_rows(plan, last_run):
    """Use final run state, not stale failure directories or progress counts."""
    by_key = {row["key"]: row for row in plan}
    if len(by_key) != len(plan) or len({row["key"] for row in last_run}) != len(last_run):
        raise ValueError("Duplicate branch keys in historical inventory/state")
    selected = []
    for state in last_run:
        if state["status"] in COMPLETED_STATUSES:
            continue
        if state["key"] not in by_key:
            raise ValueError("Unresolved branch is absent from historical plan")
        selected.append((by_key[state["key"]], state))
    return sorted(selected, key=lambda pair: pair[0]["key"])


def segment_distances(points, segments, sample=2048):
    """Exact point-to-segment distance for bounded deterministic vertex samples."""
    points, segments = np.asarray(points, float), np.asarray(segments, float)
    if segments.ndim != 3 or segments.shape[1:] != (2, 3) or not len(segments):
        raise ValueError("Invalid or empty original skeleton")
    if not np.isfinite(points).all() or not np.isfinite(segments).all() or not len(points):
        raise ValueError("Empty or nonfinite geometry")
    ids = np.unique(np.linspace(0, len(points) - 1, min(sample, len(points)), dtype=int))
    a, vector = segments[:, 0], segments[:, 1] - segments[:, 0]
    squared = np.einsum("ij,ij->i", vector, vector)
    distances = []
    for start in range(0, len(ids), 32):
        delta = points[ids[start:start + 32], None, :] - a[None, :, :]
        projection = np.divide(np.einsum("ijk,jk->ij", delta, vector), squared,
                               out=np.zeros(delta.shape[:2]), where=squared > 0)
        residual = delta - np.clip(projection, 0, 1)[..., None] * vector
        distances.extend(np.linalg.norm(residual, axis=2).min(axis=1))
    distances = np.asarray(distances)
    return dict(sampled_vertices=len(ids), minimum_nm=float(distances.min()),
                median_nm=float(np.median(distances)), q95_nm=float(np.percentile(distances, 95)),
                maximum_nm=float(distances.max()), fraction_over_3um=float(np.mean(distances > 3000)),
                fraction_over_10um=float(np.mean(distances > 10000)),
                method="exact_point_to_segment_on_deterministic_vertex_sample")


def slice_support(mesh, edges, basis, split=None):
    """Count the face-centroid slices used by the historical width kernel.

    This screens support only. It does not change correspondence, choose a new
    axis, or infer an anatomical diameter from unsupported edges.
    """
    centers = np.asarray(mesh.triangles_center)
    axial, radial, both, component_sizes, degenerate = [], [], [], [], 0
    for edge in np.asarray(edges):
        vector = edge[1] - edge[0]
        if np.abs(vector).sum() < 0.001:
            degenerate += 1
            continue
        transform = basis(vector)
        endpoints = transform @ edge.T
        faces = (transform @ centers.T).T
        low, high = np.sort(endpoints[2]) + [-0.01, 0.01]
        in_slice = (faces[:, 2] > low) & (faces[:, 2] < high)
        in_radius = np.linalg.norm(faces[:, :2] - endpoints[:2].mean(axis=1), axis=1) < DISTANCE_THRESHOLD_NM
        axial.append(int(in_slice.sum()))
        radial.append(int(in_radius.sum()))
        supported = np.flatnonzero(in_slice & in_radius)
        both.append(len(supported))
        if split is not None and len(supported):
            _, components = split(mesh.submesh([supported], append=True),
                                  only_watertight=False, connectivity="edges")
            component_sizes.append([len(component) for component in components])
        else:
            component_sizes.append([])
    return dict(resized_segments=len(edges), skipped_degenerate_segments=degenerate,
                axial_faces_per_segment=axial, radial_faces_per_segment=radial,
                supported_faces_per_segment=both, supported_segments=sum(n > 0 for n in both),
                slice_component_face_counts=component_sizes,
                distance_threshold_nm=DISTANCE_THRESHOLD_NM, axial_buffer_nm=0.01)


def width_probe(mesh, skeleton, width_utils):
    """Capture raw distances before the width helper imputes zero placeholders."""
    captured = {}
    original_call = width_utils.cb.skeletal_distance_no_skipping
    backend = getattr(width_utils.cb, "_cu", None)
    original_center = backend.tu.mesh_center_weighted_face_midpoints if backend is not None else None
    selected_slice_face_counts = []

    def observed_center(selected_mesh, *args, **kwargs):
        selected_slice_face_counts.append(len(selected_mesh.faces))
        return original_center(selected_mesh, *args, **kwargs)

    def observed(*args, **kwargs):
        result = original_call(*args, **kwargs)
        values = np.asarray(result[0], dtype=float)
        captured.update(raw_distances_nm=values.tolist(), raw_positive_distances=int(np.sum(values > 0)),
                        raw_zero_distances=int(np.sum(values == 0)),
                        raw_nonfinite_distances=int(np.sum(~np.isfinite(values))),
                        assigned_parent_faces=len(result[3]))
        return result

    edges = width_utils.skeleton_resized_ordered(skeleton)
    started = time.perf_counter()
    width_utils.cb.skeletal_distance_no_skipping = observed
    if original_center is not None:
        backend.tu.mesh_center_weighted_face_midpoints = observed_center
    try:
        values, median = width_utils.calculate_new_width(
            SimpleNamespace(mesh=mesh, skeleton=skeleton, width=0), no_spines=False,
            summary_measure="median", return_average=True)
    finally:
        width_utils.cb.skeletal_distance_no_skipping = original_call
        if original_center is not None:
            backend.tu.mesh_center_weighted_face_midpoints = original_center
    split = backend.tu.split if backend is not None else None
    captured.update(median_mesh_center_nm=float(median), final_distances_nm=np.asarray(values).tolist(),
                    selected_slice_face_counts=selected_slice_face_counts,
                    old_width_fallback_nm=0, skeleton_segment_size_nm=1000,
                    seconds=time.perf_counter() - started,
                    slice_support=slice_support(mesh, edges, width_utils.sk.change_basis_matrix, split))
    return captured


def metadata_inventory(branch):
    """Inspect only directly associated exported JSON metadata, preserving scope."""
    branch = Path(branch)
    records = []
    width_names = {"width", "width_new", "width_array", "median_mesh_center", "median_mesh_center_nm"}

    def fields(value, prefix=""):
        found = []
        if isinstance(value, dict):
            for key, child in value.items():
                path = f"{prefix}.{key}" if prefix else key
                if key in width_names:
                    found.append(path)
                found.extend(fields(child, path))
        elif isinstance(value, list):
            for index, child in enumerate(value):
                found.extend(fields(child, f"{prefix}[{index}]"))
        return found

    for scope, directory in [("branch", branch), ("limb", branch.parent), ("neuron", branch.parent.parent)]:
        for path in sorted(directory.glob("*.json")):
            content = json.loads(path.read_text())
            records.append(dict(scope=scope, **file_receipt(path), width_field_paths=fields(content)))
    # Named width fields still require schema/source validation before use. In
    # particular, a derived cache's caliber_um is not median_mesh_center.
    return dict(files=records, original_width_eligibility_metadata="not_available" if not any(
        record["scope"] == "branch" and record["width_field_paths"] for record in records
    ) else "unverified_named_fields", metadata_used_for_eligibility=False)


def failure_evidence(root, key):
    attempts = []
    for directory in sorted((root / "failures").glob(key + "_*")):
        if directory.name.rsplit("_", 1)[0] != key:
            continue
        log = directory / "run.log"
        text = log.read_text(errors="replace") if log.is_file() else ""
        segments = re.findall(r"Number of segments:\s*(\d+)", text)
        attempts.append(dict(directory=str(directory.resolve()),
                             files=[file_receipt(p) for p in sorted(directory.rglob("*")) if p.is_file()],
                             terminal_error=text.splitlines()[-1] if text else None,
                             reported_cgal_segments=int(segments[-1]) if segments else None,
                             saved_axis_files=[p.name for p in directory.glob("*.npy")],
                             mentions_zero_volume_warning="divide by zero encountered in divide" in text))
    return attempts


def classification(state, probe, coverage, attempts=None):
    """Keep failures unresolved; diagnostic evidence cannot manufacture completion."""
    if "TimeoutExpired" in state.get("reason", ""):
        cgal_finished = any(attempt.get("reported_cgal_segments") is not None for attempt in (attempts or []))
        return dict(status="processing_unresolved", cause="post_cgal_timeout_stage_not_fully_instrumented" if cgal_finished else "timeout_stage_not_identified")
    if probe is None:
        return dict(status="eligibility_unresolved", cause="width_probe_not_run")
    median = probe["median_mesh_center_nm"]
    if not np.isfinite(median) or median <= 0:
        support = probe["slice_support"]
        if support["supported_segments"] == 0:
            cause = "axis_surface_mismatch_no_supported_width_slices" if coverage["minimum_nm"] > 3000 else "no_supported_width_slices"
        else:
            selected_sizes = probe.get("selected_slice_face_counts", [])
            cause = "selected_single_face_slices_zero_mesh_center_width" if selected_sizes and max(selected_sizes) == 1 else "supported_slices_have_no_positive_width"
        return dict(status="eligibility_unresolved", cause=cause)
    return dict(status="eligibility_unresolved", cause="current_probe_has_positive_width_historical_failure_needs_reconciliation",
                current_probe_above_115nm=bool(median > WIDTH_THRESHOLD_NM))


def frozen_providers(root, manifest):
    """Import the pinned NEURD source; distinguish absent historical fingerprints."""
    sys.dont_write_bytecode = True
    runtime = root / "runtime"
    if "neurd" in sys.modules:
        raise RuntimeError("Run diagnosis in a fresh process to select the archived NEURD package")
    sys.path.insert(0, str(runtime))
    from neurd import width_utils
    if runtime not in Path(width_utils.__file__).resolve().parents:
        raise RuntimeError("Width helper did not load from archived runtime")
    source_hashes = {name: sha256(runtime / name) for name in manifest["code_sha256"]}
    if source_hashes != manifest["code_sha256"]:
        raise ValueError("Archived NEURD source does not match historical manifest")
    installed = {}
    for filename, expected in manifest["dependencies"]["source_files"].items():
        path = Path(filename)
        actual = sha256(path) if path.is_file() else None
        installed[str(path)] = dict(sha256=actual, historical_sha256=expected,
                                    historical_fingerprint_available=True, matches_historical=actual == expected)
        if actual != expected:
            raise ValueError(f"Historically fingerprinted provider changed: {path}")
    for module in [width_utils.sk, width_utils.cb._cu, width_utils.cb._cu.tu]:
        path = Path(module.__file__).resolve()
        expected = manifest["dependencies"]["source_files"].get(str(path))
        actual = sha256(path)
        installed[str(path)] = dict(sha256=actual, historical_sha256=expected,
                                    historical_fingerprint_available=expected is not None,
                                    matches_historical=actual == expected if expected is not None else None)
        if expected is not None and expected != actual:
            raise ValueError(f"Historically fingerprinted provider changed: {path}")
    versions = {}
    for name, expected in manifest["dependencies"]["versions"].items():
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            actual = None
        versions[name] = dict(current=actual, historical=expected,
                              matches_historical=actual == expected if expected is not None else None)
        if expected is not None and actual != expected:
            raise ValueError(f"Historically recorded dependency version changed: {name}")
    return width_utils, dict(archived_neurd_sha256=source_hashes, installed_provider_fingerprints=installed,
                            dependency_versions=versions,
                            historical_limitations=["compartment_utils.py was not fingerprinted by the historical manifest",
                                                    "historical ray backend fingerprints were not recorded; this diagnosis casts no rays",
                                                    "historical correspondence feature-flag environment was not recorded; current probe environment is recorded"],
                            owned_skeletal_distance=os.environ.get("NEURD_OWNED_SKELETAL_DISTANCE"),
                            current_behavior_environment={name: os.environ.get(name) for name in BEHAVIOR_ENVIRONMENT_NAMES},
                            archived_neurd_patch_status=dict(sys.modules["neurd"].PATCH_STATUS),
                            python=sys.version, executable=sys.executable)


def diagnose(root, output, maximum_width_faces=25000, supplemental_metadata=None):
    import trimesh

    root, output = Path(root).resolve(), Path(output).resolve()
    if root == output or root in output.parents or output in root.parents:
        raise ValueError("Diagnostic output must be separate from the archived run")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new diagnostic output directory")
    evidence_names = ["manifest.json", "plan.json", "last_run.json", "last_run_summary.json", "profile.json", "run_full_h01.sh"]
    evidence_names += [name for name in ("last_run.jsonl", "progress.json", "launch.json", "launch_history.jsonl", "release.json") if (root / name).is_file()]
    before = {name: file_receipt(root / name) for name in evidence_names}
    diagnostic_code_before = file_receipt(__file__)
    manifest = json.loads((root / "manifest.json").read_text())
    plan = json.loads((root / "plan.json").read_text())
    last_run = json.loads((root / "last_run.json").read_text())
    selected = unresolved_rows(plan, last_run)
    summary = json.loads((root / "last_run_summary.json").read_text())
    if "failed" in summary and len(selected) != sum(summary.get(name, 0) for name in ("failed", "interrupted", "unattempted")):
        raise ValueError("Historical summary and final branch states disagree")
    width_utils, providers = frozen_providers(root, manifest)
    records = []
    for row, state in selected:
        source, skpath = Path(row["source"]), Path(row["skeleton"])
        inputs = dict(mesh=file_receipt(source), skeleton=file_receipt(skpath))
        vertices, faces = off_counts(source)
        skeleton = np.load(skpath, allow_pickle=False)
        coverage, bounds = None, None
        probe = None
        geometry_loading = "header_only_probe_budget_or_historical_timeout"
        if faces <= maximum_width_faces and "TimeoutExpired" not in state.get("reason", ""):
            mesh = trimesh.load(source, process=False, force="mesh")
            if (len(mesh.vertices), len(mesh.faces)) != (vertices, faces):
                raise ValueError("Loaded parent surface differs from OFF header")
            bounds = mesh.bounds.tolist()
            coverage = segment_distances(mesh.vertices, skeleton)
            probe = width_probe(mesh, skeleton, width_utils)
            geometry_loading = "trimesh.load(process=False)"
        attempts = failure_evidence(root, row["key"])
        record = dict(key=row["key"], relative=row["relative"], historical_state=state,
                      inputs=inputs, archived_spines=row["archived_spines"], parent_faces=faces,
                      parent_vertices=vertices, mesh_bounds_nm=bounds, geometry_loading=geometry_loading,
                      original_skeleton=dict(segments=len(skeleton),
                          length_nm=float(np.linalg.norm(skeleton[:, 1] - skeleton[:, 0], axis=1).sum()),
                          bounds_nm=[skeleton.reshape(-1, 3).min(0).tolist(), skeleton.reshape(-1, 3).max(0).tolist()]),
                      original_axis_coverage=coverage, original_width_probe=probe,
                      metadata=metadata_inventory(source.parent), failure_attempts=attempts,
                      diagnosis=classification(state, probe, coverage, attempts), accepted_result=False)
        if inputs != dict(mesh=file_receipt(source), skeleton=file_receipt(skpath)):
            raise ValueError(f"Source changed during diagnosis: {row['key']}")
        records.append(record)
        print(f"{row['key']}: {record['diagnosis']['cause']}", flush=True)
    after = {name: file_receipt(root / name) for name in evidence_names}
    if before != after or any(sha256(root / "runtime" / name) != expected for name, expected in manifest["code_sha256"].items()):
        raise ValueError("Archived evidence changed during diagnosis")
    if diagnostic_code_before != file_receipt(__file__):
        raise ValueError("Diagnostic script changed during diagnosis")
    associated_files = {}
    for record in records:
        receipts = list(record["inputs"].values()) + record["metadata"]["files"]
        receipts += [receipt for attempt in record["failure_attempts"] for receipt in attempt["files"]]
        for receipt in receipts:
            associated_files[receipt["path"]] = receipt["sha256"]
    associated_files.update({path: receipt["sha256"] for path, receipt in providers.get("installed_provider_fingerprints", {}).items()})
    if any(sha256(path) != expected for path, expected in associated_files.items()):
        raise ValueError("Associated input, metadata, failure evidence or provider changed during diagnosis")
    supplemental = None
    if supplemental_metadata is not None:
        path = Path(supplemental_metadata)
        supplemental = dict(receipt=file_receipt(path), evidence=json.loads(path.read_text()),
                            used_for_eligibility=False)
        source_record = supplemental["evidence"].get("source")
        if source_record is not None and sha256(source_record["path"]) != source_record["sha256"]:
            raise ValueError("Supplemental metadata source changed since inspection")
    report = dict(schema_version=SCHEMA_VERSION, historical_run_hash=manifest["run_hash"],
                  historical_evidence=before, providers=providers, diagnostic_code=file_receipt(__file__),
                  historical_input_hash_limitations="Failed historical plan rows contain no source content hashes; current input hashes prove stability during this diagnosis, not byte identity to the historical attempt",
                  historical_axis_limitations="Historical workers measured the axis returned by repair_skeleton; failed attempts saved no accepted/candidate axis. These probes measure the exported original axis without rebuilding repair and cannot prove the identity of the historical width axis",
                  configuration=dict(width_threshold_nm=WIDTH_THRESHOLD_NM, maximum_width_probe_faces=maximum_width_faces,
                                     mesh_loading="trimesh.load(process=False)", segmentation_run=False, skeleton_repair_run=False,
                                     axis_sample_maximum_vertices=2048),
                  summary=dict(unresolved=len(records), width_probed=sum(r["original_width_probe"] is not None for r in records),
                               historical_run_unmodified=True, original_sources_unmodified=True,
                               accepted_results=0,
                               unchanged_hash_scope=["historical manifest/plan/final state/profile/launch",
                                                     "manifest-listed archived NEURD source files",
                                                     "selected source mesh/skeleton inputs and directly associated JSON metadata",
                                                     "selected failure evidence files", "recorded installed providers"]),
                  supplemental_metadata=supplemental, branches=records)
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    with (output / "branches.csv").open("w", newline="") as stream:
        fields = ["key", "status", "cause", "parent_faces", "archived_spines", "axis_q95_nm", "axis_minimum_nm", "median_mesh_center_nm", "supported_width_segments", "mesh_sha256", "skeleton_sha256"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in records:
            probe = row["original_width_probe"] or {}
            writer.writerow(dict(key=row["key"], status=row["diagnosis"]["status"], cause=row["diagnosis"]["cause"], parent_faces=row["parent_faces"],
                                 archived_spines=row["archived_spines"], axis_q95_nm=(row["original_axis_coverage"] or {}).get("q95_nm"),
                                 axis_minimum_nm=(row["original_axis_coverage"] or {}).get("minimum_nm"),
                                 median_mesh_center_nm=probe.get("median_mesh_center_nm"),
                                 supported_width_segments=probe.get("slice_support", {}).get("supported_segments"),
                                 mesh_sha256=row["inputs"]["mesh"]["sha256"], skeleton_sha256=row["inputs"]["skeleton"]["sha256"]))
    for name, status in [("eligibility_unresolved_keys.json", "eligibility_unresolved"),
                         ("timeout_keys.json", "processing_unresolved")]:
        (output / name).write_text(json.dumps([r["key"] for r in records if r["diagnosis"]["status"] == status], indent=2) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-width-faces", type=int, default=25000)
    parser.add_argument("--supplemental-metadata", type=Path, help="Optional independently recorded metadata inspection JSON; never used for eligibility")
    args = parser.parse_args(argv)
    if args.maximum_width_faces < 1:
        parser.error("Width face budget must be positive")
    diagnose(args.historical_run, args.output, args.maximum_width_faces, args.supplemental_metadata)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
