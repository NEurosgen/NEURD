"""A real exported branch must produce disjoint, volume-valid parent pieces.

Set NEURD_SPINE_BRANCH_MANIFEST to a JSON fixture manifest containing
schema_version=1, units="nm", data_type="microns" or "h01", and mesh/skeleton
records with path and sha256. A soma record is optional. Paths can be relative
to the manifest; maximum_parent_faces defaults to 50000, minimum_spines to 1.
This gate uses the original skeleton, ordinary soma/endpoint filters, 25 SDF
rays and geometric refinement disabled. It checks geometry, not anatomy.
"""
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fixture_inputs(manifest_path, manifest):
    paths = {}
    for name in ("mesh", "skeleton", "soma"):
        source = manifest.get(name)
        if source is None and name == "soma":
            continue
        if not isinstance(source, dict) or not source.get("path") or not source.get("sha256"):
            raise ValueError(f"Fixture {name} requires a path and SHA-256")
        path = Path(source["path"])
        if not path.is_absolute():
            path = Path(manifest_path).parent / path
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Missing branch fixture {name}: {path}")
        if sha256(path) != source["sha256"]:
            raise ValueError(f"Branch fixture checksum changed: {path}")
        paths[name] = path
    return paths


def run_branch_fixture(manifest_path, workdir):
    """Run the spine-positive branch check and return a JSON-safe receipt."""
    import numpy as np
    import neurd
    from neurd import h01_batch, parameters, spine_utils as spu
    import cgal_Segmentation_Module as csm
    from neurd.mesh_io import load_mesh
    from neurd.run_config import RunConfig, runtime_environment
    from tests.tools.neuron_metrics import extract_spine_metrics

    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != 1 or manifest.get("units") != "nm":
        raise ValueError("Branch fixture requires schema_version=1 and units=nm")
    data_type = manifest["data_type"]
    if data_type not in ("microns", "h01"):
        raise ValueError(f"Unsupported branch dataset: {data_type}")
    if not getattr(csm, "__file__", "").endswith(".so"):
        raise RuntimeError("The branch spine gate requires compiled CGAL segmentation")
    paths = fixture_inputs(manifest_path, manifest)
    parent = load_mesh(paths["mesh"])
    skeleton = np.load(paths["skeleton"], allow_pickle=False)
    assert 0 < len(parent.faces) <= manifest.get("maximum_parent_faces", 50000)
    assert np.isfinite(parent.vertices).all()
    assert skeleton.ndim == 3 and skeleton.shape[1:] == (2, 3) and len(skeleton)
    assert np.isfinite(skeleton).all()
    soma = load_mesh(paths["soma"]) if "soma" in paths else None
    branch = SimpleNamespace(mesh=parent, skeleton=skeleton)
    root = Path(neurd.__file__).resolve().parent.parent
    code_paths = [*sorted((root / "neurd").glob("*.py")), Path(__file__).resolve(),
                  root / "tests/tools/neuron_metrics.py"]
    code_hashes = {str(path.relative_to(root)): sha256(path) for path in code_paths}
    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        git_status = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"],
                                             cwd=root, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        git_sha, git_status = None, None
    previous_params = dict(parameters.params._data), parameters.params.data_type
    previous_rays = os.environ.get("NEURD_SDF_RAYS")
    previous_rng = np.random.get_state()
    previous_cwd = Path.cwd()
    workdir = Path(workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    face_masks = []
    try:
        parameters.params.use(data_type)
        os.environ["NEURD_SDF_RAYS"] = "25"
        np.random.seed(0)
        configuration = dict(data_type=data_type, parameters=dict(parameters.params._data),
                             sdf_rays=25, seed=0, geometric_refinement=False,
                             original_skeleton=True, soma_nullification=soma is not None)
        os.chdir(workdir)
        meshes, volumes = spu.calculate_spines_on_branch(
            branch, soma_verts=None if soma is None else soma.vertices,
            soma_vertex_nullification=soma is not None, geometric_refinement=False,
            spine_face_indices=face_masks,
            cgal_folder=workdir / "cgal", print_flag=True)
    finally:
        os.chdir(previous_cwd)
        parameters.params._data, parameters.params.data_type = previous_params
        np.random.set_state(previous_rng)
        if previous_rays is None:
            os.environ.pop("NEURD_SDF_RAYS", None)
        else:
            os.environ["NEURD_SDF_RAYS"] = previous_rays
    elapsed = time.perf_counter() - started
    volume_array = np.asarray(volumes, dtype=float)
    assert volume_array.shape == (len(meshes),)
    assert np.isfinite(volume_array).all() and (volume_array > 0).all()
    branch.spines, branch.spines_volume = meshes, volumes
    branch.spines_face_idx = face_masks
    metrics = extract_spine_metrics(SimpleNamespace(limbs=[SimpleNamespace(branches=[branch])]))
    assert metrics["n_spines_total"] >= max(1, int(manifest.get("minimum_spines", 1))), metrics
    assert metrics["spines_with_face_indices"] == metrics["n_spines_total"], metrics
    for name in ("invalid_parent_faces_total", "ambiguous_parent_faces_total",
                 "overlapping_parent_faces_total", "invalid_spine_face_indices_total",
                 "overlapping_spine_surface_faces_total", "invalid_spine_volumes_total"):
        assert metrics[name] == 0, metrics
    # Verify that read-only fixtures remained unchanged during processing.
    fixture_inputs(manifest_path, manifest)
    current_hashes = {str(path.relative_to(root)): sha256(path) for path in code_paths}
    return dict(
        status="passed", branch_key=manifest.get("branch_key"),
        fixture_manifest=dict(path=str(manifest_path), sha256=sha256(manifest_path)),
        inputs={name: dict(path=str(path), sha256=sha256(path)) for name, path in paths.items()},
        configuration=configuration,
        configuration_sha256=hashlib.sha256(json.dumps(configuration, sort_keys=True).encode()).hexdigest(),
        code_sha256=code_hashes, code_changed_during_run=current_hashes != code_hashes,
        git_sha=git_sha, tracked_git_status=git_status,
        compiled_cgal=dict(path=csm.__file__, sha256=sha256(csm.__file__)),
        dependencies=h01_batch.dependency_signature(),
        runtime_environment=runtime_environment(
            RunConfig(dataset=data_type, calculate_spines=True, sdf_rays=25)),
        environment={name: os.environ.get(name) for name in (
            "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "OMP_NUM_THREADS")},
        python=sys.version, platform=platform.platform(),
        parent_faces=len(parent.faces), parent_vertices=len(parent.vertices),
        archived_spines=manifest.get("archived_spines"), metrics=metrics,
        volumes_nm3=volume_array.tolist(), elapsed_seconds=elapsed,
    )


def test_real_spine_positive_branch(tmp_path):
    value = os.environ.get("NEURD_SPINE_BRANCH_MANIFEST")
    if not value:
        pytest.skip("Set NEURD_SPINE_BRANCH_MANIFEST to a checksummed spine-positive branch fixture")
    manifest_path = Path(value)
    if not manifest_path.is_file():
        pytest.skip(f"Missing branch fixture manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    try:
        fixture_inputs(manifest_path, manifest)
    except FileNotFoundError as error:
        pytest.skip(str(error))
    receipt = run_branch_fixture(manifest_path, tmp_path)
    (tmp_path / "receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
