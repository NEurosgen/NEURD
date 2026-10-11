# extract_spine_descriptors.py
# -*- coding: utf-8 -*-
#
# Walks a directory of spine meshes (.off/.ply/...) and for EACH mesh saves:
#   * <stem>.npy          -- the descriptor vector (float64, np.save)
#   * <stem>_volume_qc.json -- volume algorithm, closure method and fallback status
#   * <stem>_centroid.npy -- the mesh centroid in ABSOLUTE coordinates (3 float, nm)
# The descriptor .npy repeats the mesh filename (stem); the centroid adds "_centroid".
# feature_names.json is written once alongside them, giving the column order.
#
# The descriptors are computed by the NEURD engine on already-cut spine meshes:
# base attributes + head/neck segmentation (the real CGAL SDF).
# head/neck is called directly (Spine.calculate_head_neck), BYPASSING the
# head_mesh_splits attribute, which raises IndexError on headless spines.
#
# Parallelism: ProcessPoolExecutor. Each worker handles one mesh in its OWN private CWD
# (tempfile.mkdtemp), because CGAL writes temporary .off/.csv files into "./" with a
# random 10..1000 prefix -- a shared CWD across workers means races and name collisions.
# A private CWD per task isolates them completely.
#
# Example:
#   python scripts/extract_spine_descriptors.py path/to/spine_meshes path/to/descriptors \
#       --workers 8
#   # recursively, mirroring the sub-directory structure into the output:
#   python scripts/extract_spine_descriptors.py path/to/datasets path/to/descriptors \
#       --recursive --workers 8

import os

# Cap the numeric backends BEFORE importing numpy/neurd: otherwise each of the N workers
# spins up its own BLAS thread pool and they fight over the cores.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import sys
import json
import time
import shutil
import tempfile
import argparse
import hashlib
import importlib.metadata
import importlib.util
import warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

warnings.filterwarnings("ignore")

# Column order of the descriptor vector. FIXED -- do not reorder, or previously saved
# .npy files become incomparable. Append new features at the end only.
# head_* features are NaN when the spine has no head (head_exist == 0).
FEATURE_NAMES = [
    # --- whole spine ---
    "n_faces",                    # number of faces in the spine mesh
    "n_vertices",                 # number of vertices
    "volume",                     # volume (nm^3), validated closure; see *_volume_qc.json
    "area",                       # surface area (nm^2)
    "skeletal_length",            # length of the spine surface skeleton (nm)
    "spine_volume_to_spine_area", # volume/area ratio
    "bbox_side_min",              # oriented bbox: shortest side (nm)
    "bbox_side_mid",              # middle side (nm)
    "bbox_side_max",              # longest side (nm)
    # --- head flags ---
    "head_exist",                 # 1.0 if a head was found, else 0.0
    "n_heads",                    # number of heads (0.0 if there is none)
    # --- head (NaN when head_exist == 0) ---
    "head_volume",                # head volume (nm^3)
    "head_area",                  # head area (nm^2)
    "head_volume_to_head_area",   # head volume/area
    "head_skeletal_length",       # head skeleton length (nm)
    "head_n_faces",               # faces in the head
    "head_width",                 # head width (SDF estimate, nm)
    "head_width_ray",             # head width (ray-trace, 50th percentile, nm)
    "head_width_ray_80_perc",     # head width (ray-trace, 80th percentile, nm)
    "head_sdf",                   # mean head SDF
    "head_bbox_side_min",         # head bbox: shortest side (nm)
    "head_bbox_side_mid",         # middle side (nm)
    "head_bbox_side_max",         # longest side (nm)
    # --- neck (always computed) ---
    "neck_volume",                # neck volume (nm^3)
    "neck_area",                  # neck area (nm^2)
    "neck_volume_to_neck_area",   # neck volume/area
    "neck_skeletal_length",       # neck length = neck skeleton length (nm)
    "neck_n_faces",               # faces in the neck
    "neck_width",                 # neck width (SDF estimate, nm)
    "neck_width_ray",             # neck width (ray-trace, 50th percentile, nm)
    "neck_width_ray_80_perc",     # neck width (ray-trace, 80th percentile, nm)
    "neck_sdf",                   # mean neck SDF
    "neck_bbox_side_min",         # neck bbox: shortest side (nm)
    "neck_bbox_side_mid",         # middle side (nm)
    "neck_bbox_side_max",         # longest side (nm)
]

# NEURD sentinel for "no head" -> stored as NaN in the .npy (cleaner for ML).
_NO_HEAD_SENTINEL = -1

# QC doubles as the completion marker. Older unversioned sidecars are rebuilt,
# because file existence cannot establish which algorithm produced a vector.
QUALITY_SCHEMA_VERSION = 1


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _quality_path(out_desc):
    path = Path(out_desc)
    return path.with_name(path.stem + "_volume_qc.json")


def _descriptor_configuration(data_type):
    """Snapshot settings and implementations that can affect descriptors.

    Use a separate parameter object so inventory does not change the process's
    active dataset. Worker processes apply the requested dataset themselves.
    """
    from neurd import parameters

    params = parameters._Params()
    params.use(data_type)
    root = Path(__file__).resolve().parents[1]
    source_files = [Path(__file__).resolve(), *sorted((root / "neurd").glob("*.py"))]
    versions = {}
    for name in ("numpy", "scipy", "trimesh", "networkx", "meshparty",
                 "mesh-processing-tools", "datasci-stdlib-tools",
                 "embreex", "pyembree", "rtree"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    provider = {}
    # Head selection and width descriptors use the available ray intersector.
    # Fingerprint its native modules as well as distribution versions, since a
    # local rebuild need not change a package version. Calcification is recorded
    # conservatively as an installed mesh-tools backend; the current descriptor
    # skeletal-length path itself uses meshparty surface skeletons.
    for name in ("cgal_Segmentation_Module", "calcification_param_Module", "mesh_tools",
                 "embreex.rtcore", "embreex.rtcore_scene", "embreex.mesh_construction",
                 "pyembree.rtcore", "pyembree.rtcore_scene", "pyembree.mesh_construction",
                 "rtree.core"):
        try:
            spec = importlib.util.find_spec(name)
        except (ImportError, ValueError, OSError):
            spec = None
        origin = getattr(spec, "origin", None) or getattr(sys.modules.get(name), "__file__", None)
        provider[name] = {"path": origin, "sha256": _sha256(origin) if origin and Path(origin).is_file() else None}
        if name == "mesh_tools" and origin:
            for basename in ("mesh_utils.py", "trimesh_utils.py", "skeleton_utils.py", "meshparty_skeletonize.py"):
                path = Path(origin).parent / basename
                if path.is_file():
                    provider[f"mesh_tools/{basename}"] = {"path": str(path), "sha256": _sha256(path)}
    return {
        "data_type": data_type,
        "parameters": dict(params._data),
        "feature_names": list(FEATURE_NAMES),
        "environment": {name: os.environ.get(name) for name in (
            "NEURD_SDF_RAYS", "NEURD_LEGACY_MESH_LOADER", "NEURD_LEGACY_CGAL_ASSEMBLY",
            "NEURD_REAL_POISSON", "NEURD_POISSON_DEPTH",
            "NEURD_POISSON_BACKEND", "NEURD_POISSON_ITERS", "NEURD_OWNED_SKELETAL_DISTANCE",
            "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "OMP_NUM_THREADS")},
        "python": sys.version,
        "code_sha256": {str(path.relative_to(root)): _sha256(path) for path in source_files},
        "dependencies": versions,
        "providers": provider,
    }


def _completed_valid(path, out_desc, out_centroid, data_type, configuration):
    """Only skip a complete, matching record whose files still match hashes.

    Missing, malformed and legacy records are queued for regeneration. In
    particular, NaN volumes are valid scientific outputs when QC records the
    failed closure; validity here concerns provenance, not a positive volume.
    """
    from neurd.spine_volume import VOLUME_ALGORITHM

    try:
        def invalid_constant(value):
            raise ValueError(f"Nonfinite JSON constant: {value}")

        quality = json.loads(_quality_path(out_desc).read_text(encoding="utf-8"), parse_constant=invalid_constant)
        if not isinstance(quality, dict):
            return False
        if (type(quality.get("schema_version")) is not int
                or quality.get("schema_version") != QUALITY_SCHEMA_VERSION
                or quality.get("status") != "complete"
                or quality.get("algorithm") != VOLUME_ALGORITHM
                or quality.get("data_type") != data_type
                or quality.get("configuration") != configuration
                or quality.get("input_sha256") != _sha256(path)):
            return False
        outputs = quality.get("outputs_sha256")
        if not isinstance(outputs, dict):
            return False
        expected_names = {Path(out_desc).name, Path(out_centroid).name}
        if set(outputs) != expected_names:
            return False
        for output in (out_desc, out_centroid):
            if outputs[Path(output).name] != _sha256(output):
                return False
        # A truncated sidecar must not turn an arbitrary vector into a result.
        if not _quality_records_valid(quality, VOLUME_ALGORITHM):
            return False
        return True
    except (OSError, ValueError, TypeError, KeyError):
        return False


def _quality_records_valid(quality, algorithm):
    required = {"algorithm", "method", "valid", "volume_nm3", "input_watertight", "fallback_reason"}
    for name in ("spine", "head", "neck"):
        if name not in quality:
            return False
        report = quality[name]
        if name == "head" and report is None:
            continue
        if (not isinstance(report, dict) or not required.issubset(report)
                or report["algorithm"] != algorithm
                or not isinstance(report["valid"], bool)):
            return False
    return True


def _publish_result(path, out_desc, out_centroid, vector, centroid, quality,
                    data_type, configuration, input_sha256):
    """Stage both arrays, then atomically publish the QC completion marker.

    Arrays have to keep their historical filenames. Their replacement is not
    one atomic filesystem operation, so remove the previous marker first and
    publish the new marker only after both arrays were replaced successfully.
    """
    import numpy as np
    from neurd.spine_volume import VOLUME_ALGORITHM

    if not _quality_records_valid(quality, VOLUME_ALGORITHM):
        raise ValueError("Missing or incompatible spine/head/neck volume QC")
    out_desc, out_centroid = Path(out_desc), Path(out_centroid)
    out_desc.parent.mkdir(parents=True, exist_ok=True)
    quality_path = _quality_path(out_desc)
    staging = Path(tempfile.mkdtemp(prefix=".spine_result_", dir=out_desc.parent))
    try:
        staged_desc, staged_centroid = staging / out_desc.name, staging / out_centroid.name
        np.save(staged_desc, vector)
        np.save(staged_centroid, centroid)
        record = dict(quality)
        record.update(
            schema_version=QUALITY_SCHEMA_VERSION, status="complete",
            algorithm=VOLUME_ALGORITHM, data_type=data_type,
            configuration=configuration, input_sha256=input_sha256,
            outputs_sha256={out_desc.name: _sha256(staged_desc), out_centroid.name: _sha256(staged_centroid)},
        )
        staged_quality = staging / quality_path.name
        staged_quality.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        if _sha256(path) != input_sha256:
            raise ValueError(f"Input mesh changed while descriptors were computed: {path}")
        quality_path.unlink(missing_ok=True)
        staged_desc.replace(out_desc)
        staged_centroid.replace(out_centroid)
        staged_quality.replace(quality_path)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _f(x):
    """None/sentinel -> NaN, otherwise float."""
    import math
    if x is None:
        return math.nan
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return math.nan
    return xf


def _compute_descriptors(path: str, *, with_quality=False):
    """Compute (descriptor_vector, centroid) for one mesh.

    The NEURD/mesh_tools imports are inside the function and STRICTLY in this order:
    neurd patches datasci_tools.numpy_dep, without which a bare mesh_tools import fails
    on np.float_ (numpy>=2.0).
    """
    from neurd import spine_utils as spu      # MUST BE FIRST
    from neurd import parameters              # noqa: F401 (mode is set by the caller)
    from neurd.mesh_io import load_mesh
    import numpy as np

    mesh = load_mesh(path)

    # Centroid in absolute coordinates -- taken from the RAW mesh, before any processing
    # that could recompute or shift it.
    centroid = np.asarray(mesh.centroid, dtype=np.float64).reshape(3)

    sp = spu.Spine(mesh)
    # Base attributes; volume may be NaN if no valid closed estimate is available.
    spu.calculate_spine_attributes(
        sp, branch_obj=None,
        calculate_coordinates=False,
        calculate_head_neck=False,
    )
    # head/neck directly, bypassing head_mesh_splits (which fails on headless spines).
    sp = spu.calculate_spine_obj_mesh_skeleton_coordinates(spine_obj=sp, mesh=sp.mesh)
    sp.calculate_head_neck()

    nan = float("nan")

    def _bbox3(lengths):
        """The 3 oriented-bbox sides in ascending order; NaN when unavailable."""
        vals = [_f(x) for x in lengths]
        if len(vals) < 3 or any(v != v for v in vals):  # a NaN among the values
            return [nan, nan, nan]
        return sorted(vals)

    def _ratio(num, den):
        """num/den, guarded against zero/NaN."""
        n_, d_ = _f(num), _f(den)
        if n_ != n_ or d_ != d_ or d_ == 0:
            return nan
        return n_ / d_

    # --- whole spine (always) ---
    bbox = _bbox3(sp.bbox_oriented_side_lengths)
    spine_vals = [
        _f(sp.n_faces),
        _f(sp.n_vertices),
        _f(sp.volume),
        _f(sp.area),
        _f(sp.skeletal_length),
        _f(spu.spine_volume_to_spine_area(sp)),
        bbox[0], bbox[1], bbox[2],
    ]

    head_exist = spu.head_exist(sp)

    # --- head: only when found (otherwise head_* fail or are meaningless -> NaN) ---
    if head_exist:
        hbbox = _bbox3(sp.head_bbox_oriented_side_lengths)
        head_vals = [
            _f(sp.head_volume),
            _f(sp.head_area),
            _ratio(sp.head_volume, sp.head_area),
            _f(sp.head_skeletal_length),
            _f(sp.head_n_faces),
            _f(sp.head_width),
            _f(sp.head_width_ray),
            _f(sp.head_width_ray_80_perc),
            _f(sp.head_sdf),
            hbbox[0], hbbox[1], hbbox[2],
        ]
        n_heads = _f(sp.n_heads)
    else:
        head_vals = [nan] * 12
        n_heads = 0.0

    # --- neck (always computed) ---
    nbbox = _bbox3(sp.neck_bbox_oriented_side_lengths)
    neck_vals = [
        _f(sp.neck_volume),
        _f(sp.neck_area),
        _ratio(sp.neck_volume, sp.neck_area),
        _f(sp.neck_skeletal_length),
        _f(sp.neck_n_faces),
        _f(sp.neck_width),
        _f(sp.neck_width_ray),
        _f(sp.neck_width_ray_80_perc),
        _f(sp.neck_sdf),
        nbbox[0], nbbox[1], nbbox[2],
    ]

    values = spine_vals + [1.0 if head_exist else 0.0, n_heads] + head_vals + neck_vals
    assert len(values) == len(FEATURE_NAMES), (len(values), len(FEATURE_NAMES))
    vector = np.asarray(values, dtype=np.float64)
    if not with_quality:
        return vector, centroid
    from neurd.spine_volume import estimate_mesh_volume, VOLUME_ALGORITHM
    quality = {
        "algorithm": VOLUME_ALGORITHM,
        "data_type": None,  # filled by the worker from the actual CLI argument
        "spine": sp.volume_quality,
        "head": estimate_mesh_volume(spu.mesh_from_compartment(sp, compartment="head", index=0)) if head_exist else None,
        "neck": estimate_mesh_volume(sp.neck_mesh),
    }
    return vector, centroid, quality


def _worker(args):
    """Runs in a separate process, in a private CWD for CGAL's sake."""
    path, out_desc, out_centroid, data_type, *supplied_configuration = args
    from neurd import parameters

    parameters.params.use(data_type)
    configuration = supplied_configuration[0] if supplied_configuration else _descriptor_configuration(data_type)
    path = str(Path(path).resolve())

    prev_cwd = os.getcwd()
    tmp = tempfile.mkdtemp(prefix="spine_cgal_")
    try:
        os.chdir(tmp)  # CGAL writes temp files into "./" -> isolate them
        t0 = time.time()
        if _descriptor_configuration(data_type) != configuration:
            raise ValueError("Descriptor runtime changed after job planning")
        input_sha256 = _sha256(path)
        vec, centroid, quality = _compute_descriptors(path, with_quality=True)
        if _descriptor_configuration(data_type) != configuration:
            raise ValueError("Descriptor runtime changed during computation")
        os.chdir(prev_cwd)  # restore BEFORE writing the result (out paths are absolute)
        _publish_result(path, out_desc, out_centroid, vec, centroid, quality,
                        data_type, configuration, input_sha256)
        return (True, path, f"{time.time() - t0:.1f}s")
    except Exception as e:
        import traceback
        return (False, path, f"{type(e).__name__}: {e}\n{traceback.format_exc()}")
    finally:
        try:
            os.chdir(prev_cwd)
        except Exception:
            pass
        shutil.rmtree(tmp, ignore_errors=True)


def _gather_jobs(input_dir: Path, output_dir: Path, pattern: str,
                 recursive: bool, overwrite: bool, *, data_type="microns", configuration=None):
    if configuration is None:
        configuration = _descriptor_configuration(data_type)
    files = sorted(input_dir.rglob(pattern) if recursive else input_dir.glob(pattern))
    jobs, skipped = [], 0
    for f in files:
        if not f.is_file():
            continue
        # In recursive mode, mirror the sub-directories to avoid name collisions.
        rel_parent = f.parent.relative_to(input_dir) if recursive else Path(".")
        out_sub = output_dir / rel_parent
        out_desc = out_sub / f"{f.stem}.npy"
        out_centroid = out_sub / f"{f.stem}_centroid.npy"
        if not overwrite and _completed_valid(f, out_desc, out_centroid, data_type, configuration):
            skipped += 1
            continue
        out_sub.mkdir(parents=True, exist_ok=True)
        jobs.append((str(f), str(out_desc), str(out_centroid)))
    return jobs, skipped, len(files)


def main():
    ap = argparse.ArgumentParser(
        description="Extract spine descriptors (+centroid) from meshes into .npy, in parallel.")
    ap.add_argument("input_dir", type=str, help="directory of spine meshes")
    ap.add_argument("output_dir", type=str, help="directory for the .npy results")
    ap.add_argument("--pattern", type=str, default="*.off", help="glob (default *.off)")
    ap.add_argument("--recursive", action="store_true",
                    help="walk sub-directories (structure mirrored into output)")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1,
                    help="number of processes (default = number of cores)")
    ap.add_argument("--data-type", choices=["microns", "h01"], default="microns",
                    help="NEURD parameter set (default microns)")
    ap.add_argument("--overwrite", action="store_true",
                    help="recompute even verified completed results; incomplete or mismatched results always recompute")
    args = ap.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    if not input_dir.is_dir():
        print(f"[!] not a directory: {input_dir}", file=sys.stderr)
        sys.exit(1)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Record the feature order alongside the results.
    with open(output_dir / "feature_names.json", "w", encoding="utf-8") as fh:
        json.dump(FEATURE_NAMES, fh, ensure_ascii=False, indent=2)

    configuration = _descriptor_configuration(args.data_type)
    jobs, skipped, n_found = _gather_jobs(
        input_dir, output_dir, args.pattern, args.recursive, args.overwrite,
        data_type=args.data_type, configuration=configuration)

    print(f"[i] input: {input_dir} (pattern={args.pattern}, recursive={args.recursive})")
    print(f"[i] meshes found: {n_found}, to process: {len(jobs)}, skipped (already done): {skipped}")
    print(f"[i] workers={args.workers}, data_type={args.data_type}")
    print(f"[i] descriptors per vector: {len(FEATURE_NAMES)} -> {output_dir/'feature_names.json'}")
    if not jobs:
        print("[i] nothing to do.")
        return 0

    task_args = [(p, od, oc, args.data_type, configuration) for (p, od, oc) in jobs]
    failed_log = output_dir / "failed.txt"
    ok = fail = 0
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futures = [ex.submit(_worker, ta) for ta in task_args]
        for i, fut in enumerate(as_completed(futures), 1):
            success, path, msg = fut.result()
            name = Path(path).name
            if success:
                ok += 1
                print(f"[{i}/{len(futures)}] ok  {name}  ({msg})")
            else:
                fail += 1
                short = msg.splitlines()[0] if msg else ""
                print(f"[{i}/{len(futures)}] FAIL {name}  -> {short}")
                with open(failed_log, "a", encoding="utf-8") as fl:
                    fl.write(f"{path}\t{msg}\n\n")

    print(f"\nDone in {time.time() - t0:.1f}s. ok={ok}, fail={fail}, skipped={skipped}")
    if fail:
        print(f"[i] error details: {failed_log}")
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
