"""Descriptor resume requires matching provenance and a complete publication."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "spine_descriptors_under_test",
    Path(__file__).resolve().parents[2] / "scripts/extract_spine_descriptors.py",
)
extractor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(extractor)


def quality_record():
    from neurd.spine_volume import VOLUME_ALGORITHM

    compartment = dict(algorithm=VOLUME_ALGORITHM, method="invalid", valid=False,
                       volume_nm3=None, input_watertight=False, fallback_reason="invalid_geometry")
    return dict(algorithm=VOLUME_ALGORITHM, spine=dict(compartment), head=None, neck=dict(compartment))


@pytest.fixture
def completed(tmp_path):
    incoming, outgoing = tmp_path / "meshes", tmp_path / "descriptors"
    incoming.mkdir()
    outgoing.mkdir()
    mesh = incoming / "spine.off"
    mesh.write_text("OFF\n0 0 0\n")
    desc, centroid = outgoing / "spine.npy", outgoing / "spine_centroid.npy"
    configuration = {"data_type": "h01", "sdf_rays": "25"}
    vector = np.arange(len(extractor.FEATURE_NAMES), dtype=float)
    vector[2] = np.nan  # Invalid closure is a completed result with explicit QC.
    extractor._publish_result(mesh, desc, centroid, vector, np.zeros(3), quality_record(),
                              "h01", configuration, extractor._sha256(mesh))
    return mesh, desc, centroid, configuration


def gather(completed, **kwargs):
    mesh, desc, _, configuration = completed
    return extractor._gather_jobs(mesh.parent, desc.parent, "*.off", False,
                                  kwargs.pop("overwrite", False), data_type="h01",
                                  configuration=configuration, **kwargs)


def test_matching_completed_outputs_are_skipped_even_with_missing_volume(completed):
    jobs, skipped, total = gather(completed)
    assert jobs == [] and skipped == 1 and total == 1
    mesh, desc, centroid, configuration = completed
    assert extractor._completed_valid(mesh, desc, centroid, "h01", configuration)
    assert np.isnan(np.load(desc)[2])
    assert not list(desc.parent.glob(".spine_result_*"))


@pytest.mark.parametrize("field,value", [
    ("schema_version", 0), ("schema_version", True), ("status", "running"),
    ("algorithm", "old_volume"), ("data_type", "microns"),
    ("configuration", {"data_type": "h01", "sdf_rays": "12"}),
    ("input_sha256", "old_input"), ("outputs_sha256", {}),
    ("outputs_sha256", {"other.npy": "arbitrary"}), ("spine", {}),
    ("head", []), ("neck", None),
])
def test_mismatched_or_truncated_metadata_requeues_without_replacing_arrays(completed, field, value):
    _, desc, centroid, _ = completed
    before = desc.read_bytes(), centroid.read_bytes()
    qc = extractor._quality_path(desc)
    record = json.loads(qc.read_text())
    record[field] = value
    qc.write_text(json.dumps(record))
    jobs, skipped, total = gather(completed)
    assert len(jobs) == 1 and skipped == 0 and total == 1
    assert (desc.read_bytes(), centroid.read_bytes()) == before


@pytest.mark.parametrize("contents", ["{", "[]", "null", '{"spine": NaN}'])
def test_malformed_qc_is_requeued(completed, contents):
    extractor._quality_path(completed[1]).write_text(contents)
    jobs, skipped, _ = gather(completed)
    assert len(jobs) == 1 and skipped == 0


@pytest.mark.parametrize("changed", ["mesh", "descriptor", "centroid", "qc"])
def test_missing_or_changed_file_is_requeued(completed, changed):
    mesh, desc, centroid, _ = completed
    if changed == "mesh":
        mesh.write_text("OFF\n1 0 0\n0 0 0\n")
    elif changed == "descriptor":
        desc.write_bytes(b"changed")
    elif changed == "centroid":
        centroid.unlink()
    else:
        extractor._quality_path(desc).unlink()
    jobs, skipped, _ = gather(completed)
    assert len(jobs) == 1 and skipped == 0


def test_compartment_algorithm_is_part_of_qc_schema(completed):
    qc = extractor._quality_path(completed[1])
    record = json.loads(qc.read_text())
    record["neck"]["algorithm"] = "legacy_closure"
    qc.write_text(json.dumps(record))
    assert gather(completed)[1] == 0


def test_overwrite_requeues_verified_completed_result(completed):
    jobs, skipped, _ = gather(completed, overwrite=True)
    assert len(jobs) == 1 and skipped == 0


def test_dataset_and_effective_configuration_change_requeues(completed):
    mesh, desc, _, configuration = completed
    jobs, skipped, _ = extractor._gather_jobs(mesh.parent, desc.parent, "*.off", False, False,
                                             data_type="microns", configuration=configuration)
    assert len(jobs) == 1 and skipped == 0
    changed = dict(configuration, sdf_rays="12")
    jobs, skipped, _ = extractor._gather_jobs(mesh.parent, desc.parent, "*.off", False, False,
                                             data_type="h01", configuration=changed)
    assert len(jobs) == 1 and skipped == 0


def test_input_change_before_publication_preserves_previous_files(completed):
    mesh, desc, centroid, configuration = completed
    qc = extractor._quality_path(desc)
    before = desc.read_bytes(), centroid.read_bytes(), qc.read_bytes()
    original_hash = extractor._sha256(mesh)
    mesh.write_text("changed during computation")
    with pytest.raises(ValueError, match="Input mesh changed"):
        extractor._publish_result(mesh, desc, centroid, np.ones(len(extractor.FEATURE_NAMES)),
                                  np.ones(3), quality_record(), "h01", configuration, original_hash)
    assert (desc.read_bytes(), centroid.read_bytes(), qc.read_bytes()) == before
    assert not list(desc.parent.glob(".spine_result_*"))


def test_staging_failure_preserves_previous_completed_files(completed):
    mesh, desc, centroid, configuration = completed
    qc = extractor._quality_path(desc)
    before = desc.read_bytes(), centroid.read_bytes(), qc.read_bytes()
    quality = quality_record()
    quality["diagnostic"] = float("nan")
    with pytest.raises(ValueError):
        extractor._publish_result(mesh, desc, centroid, np.ones(len(extractor.FEATURE_NAMES)),
                                  np.ones(3), quality, "h01", configuration, extractor._sha256(mesh))
    assert (desc.read_bytes(), centroid.read_bytes(), qc.read_bytes()) == before
    assert extractor._completed_valid(mesh, desc, centroid, "h01", configuration)
    assert not list(desc.parent.glob(".spine_result_*"))


def test_interrupted_publication_cannot_be_resumed_as_complete(completed, monkeypatch):
    mesh, desc, centroid, configuration = completed
    old_centroid = centroid.read_bytes()
    replace = Path.replace

    def fail_second_array(self, target):
        if Path(target) == centroid:
            raise OSError("simulated interrupted publication")
        return replace(self, target)

    monkeypatch.setattr(Path, "replace", fail_second_array)
    with pytest.raises(OSError, match="interrupted publication"):
        extractor._publish_result(mesh, desc, centroid, np.ones(len(extractor.FEATURE_NAMES)),
                                  np.ones(3), quality_record(), "h01", configuration, extractor._sha256(mesh))
    assert centroid.read_bytes() == old_centroid
    assert not extractor._quality_path(desc).exists()
    assert not extractor._completed_valid(mesh, desc, centroid, "h01", configuration)
    assert len(gather(completed)[0]) == 1
    assert not list(desc.parent.glob(".spine_result_*"))


def test_failed_computation_keeps_existing_completed_files(completed, monkeypatch):
    from neurd import parameters

    # _worker normally runs in an isolated process. Undo its dataset selection
    # here so this in-process failure test cannot alter later unit tests.
    monkeypatch.setattr(parameters.params, "_data", dict(parameters.params._data))
    monkeypatch.setattr(parameters.params, "data_type", parameters.params.data_type)
    mesh, desc, centroid, configuration = completed
    qc = extractor._quality_path(desc)
    before = desc.read_bytes(), centroid.read_bytes(), qc.read_bytes()

    def fail(*args, **kwargs):
        raise RuntimeError("simulated descriptor failure")

    monkeypatch.setattr(extractor, "_descriptor_configuration", lambda _: configuration)
    monkeypatch.setattr(extractor, "_compute_descriptors", fail)
    success, _, message = extractor._worker((str(mesh), str(desc), str(centroid), "h01", configuration))
    assert not success and "simulated descriptor failure" in message
    assert (desc.read_bytes(), centroid.read_bytes(), qc.read_bytes()) == before


def test_configuration_snapshot_tracks_dataset_rays_and_feature_order(monkeypatch):
    from neurd import parameters

    previous = parameters.params.data_type
    monkeypatch.setenv("NEURD_SDF_RAYS", "12")
    first = extractor._descriptor_configuration("h01")
    assert parameters.params.data_type == previous
    assert first["data_type"] == "h01"
    assert first["environment"]["NEURD_SDF_RAYS"] == "12"
    assert first["feature_names"] == extractor.FEATURE_NAMES
    assert first["parameters"]["filter_by_volume_threshold"] == 50_000_000
    assert "mesh-processing-tools" in first["dependencies"]
    monkeypatch.setenv("NEURD_SDF_RAYS", "25")
    assert first != extractor._descriptor_configuration("h01")


def test_changing_mesh_loader_invalidates_descriptor_resume(completed, monkeypatch):
    mesh, desc, centroid, _ = completed
    monkeypatch.delenv("NEURD_LEGACY_MESH_LOADER", raising=False)
    configuration = extractor._descriptor_configuration("h01")
    assert configuration["environment"]["NEURD_LEGACY_MESH_LOADER"] is None
    extractor._publish_result(mesh, desc, centroid, np.load(desc), np.load(centroid),
                              quality_record(), "h01", configuration, extractor._sha256(mesh))
    assert extractor._completed_valid(mesh, desc, centroid, "h01", configuration)
    monkeypatch.setenv("NEURD_LEGACY_MESH_LOADER", "1")
    changed = extractor._descriptor_configuration("h01")
    assert changed != configuration
    assert not extractor._completed_valid(mesh, desc, centroid, "h01", changed)


def test_recursive_inventory_mirrors_directories(tmp_path):
    incoming, outgoing = tmp_path / "input", tmp_path / "output"
    mesh = incoming / "neuron_1/spines/one.off"
    mesh.parent.mkdir(parents=True)
    mesh.write_text("mesh")
    jobs, skipped, total = extractor._gather_jobs(incoming, outgoing, "*.off", True, False,
                                                 configuration={"data_type": "microns"})
    assert skipped == 0 and total == 1
    assert jobs == [(str(mesh), str(outgoing / "neuron_1/spines/one.npy"),
                     str(outgoing / "neuron_1/spines/one_centroid.npy"))]


@pytest.mark.parametrize("provider", ["calcification_param_Module", "embreex.rtcore_scene"])
def test_changed_native_provider_requeues_without_replacing_completed_arrays(completed, monkeypatch, provider):
    mesh, desc, centroid, _ = completed
    binary = desc.parent / "test_native_backend.so"
    binary.write_bytes(b"native build one")
    find_spec = extractor.importlib.util.find_spec

    def native_spec(name, *args, **kwargs):
        return SimpleNamespace(origin=str(binary)) if name == provider else find_spec(name, *args, **kwargs)

    monkeypatch.setattr(extractor.importlib.util, "find_spec", native_spec)
    configuration = extractor._descriptor_configuration("h01")
    extractor._publish_result(mesh, desc, centroid, np.zeros(len(extractor.FEATURE_NAMES)),
                              np.zeros(3), quality_record(), "h01", configuration, extractor._sha256(mesh))
    assert extractor._completed_valid(mesh, desc, centroid, "h01", configuration)
    before = desc.read_bytes(), centroid.read_bytes()
    binary.write_bytes(b"native build two")
    changed = extractor._descriptor_configuration("h01")
    assert configuration["providers"][provider]["sha256"] != changed["providers"][provider]["sha256"]
    jobs, skipped, _ = extractor._gather_jobs(mesh.parent, desc.parent, "*.off", False, False,
                                             data_type="h01", configuration=changed)
    assert len(jobs) == 1 and skipped == 0
    assert (desc.read_bytes(), centroid.read_bytes()) == before


@pytest.mark.parametrize("backend", ["embreex", "pyembree", "rtree"])
def test_changed_ray_backend_version_requeues(completed, monkeypatch, backend):
    mesh, desc, centroid, _ = completed
    version = extractor.importlib.metadata.version
    backend_version = ["1.0-test"]
    monkeypatch.setattr(extractor.importlib.metadata, "version", lambda name:
                        backend_version[0] if name == backend else version(name))
    configuration = extractor._descriptor_configuration("h01")
    extractor._publish_result(mesh, desc, centroid, np.zeros(len(extractor.FEATURE_NAMES)),
                              np.zeros(3), quality_record(), "h01", configuration, extractor._sha256(mesh))
    backend_version[0] = "2.0-test"
    changed = extractor._descriptor_configuration("h01")
    assert configuration["dependencies"][backend] != changed["dependencies"][backend]
    jobs, skipped, _ = extractor._gather_jobs(mesh.parent, desc.parent, "*.off", False, False,
                                             data_type="h01", configuration=changed)
    assert len(jobs) == 1 and skipped == 0


@pytest.mark.parametrize("when", ["before", "during"])
def test_worker_runtime_change_cannot_replace_existing_arrays_or_qc(completed, monkeypatch, when):
    from neurd import parameters

    monkeypatch.setattr(parameters.params, "_data", dict(parameters.params._data))
    monkeypatch.setattr(parameters.params, "data_type", parameters.params.data_type)
    mesh, desc, centroid, configuration = completed
    qc = extractor._quality_path(desc)
    before = desc.read_bytes(), centroid.read_bytes(), qc.read_bytes()
    actual = dict(configuration)
    if when == "before":
        actual["source_sha256"] = "changed_source"
    monkeypatch.setattr(extractor, "_descriptor_configuration", lambda _: dict(actual))

    def compute(*args, **kwargs):
        if when == "before":
            pytest.fail("Changed runtime should fail before descriptor calculation")
        actual["provider_sha256"] = "changed_native_backend"
        return np.ones(len(extractor.FEATURE_NAMES)), np.ones(3), quality_record()

    monkeypatch.setattr(extractor, "_compute_descriptors", compute)
    success, _, message = extractor._worker((str(mesh), str(desc), str(centroid), "h01", configuration))
    assert not success and "Descriptor runtime changed" in message
    assert (desc.read_bytes(), centroid.read_bytes(), qc.read_bytes()) == before


@pytest.mark.parametrize("failed", [False, True])
def test_cli_failure_status_keeps_partial_success_and_logs_failure(tmp_path, monkeypatch, failed):
    incoming, outgoing = tmp_path / "meshes", tmp_path / "descriptors"
    incoming.mkdir()
    (incoming / "good.off").write_text("good mesh")
    if failed:
        (incoming / "bad.off").write_text("bad mesh")
    configuration = {"data_type": "h01", "sdf_rays": "25"}
    monkeypatch.setattr(extractor, "_descriptor_configuration", lambda _: configuration)
    monkeypatch.setattr(extractor.sys, "argv", ["extract_spine_descriptors.py", str(incoming),
                                               str(outgoing), "--data-type", "h01", "--workers", "2"])

    def worker(args):
        path, desc, centroid, data_type, config = args
        if Path(path).stem == "bad":
            return False, path, "RuntimeError: controlled descriptor failure"
        extractor._publish_result(path, desc, centroid, np.ones(len(extractor.FEATURE_NAMES)),
                                  np.zeros(3), quality_record(), data_type, config, extractor._sha256(path))
        return True, path, "0.0s"

    class Executor:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def submit(self, function, args):
            result = function(args)
            return SimpleNamespace(result=lambda: result)

    monkeypatch.setattr(extractor, "_worker", worker)
    monkeypatch.setattr(extractor, "ProcessPoolExecutor", Executor)
    monkeypatch.setattr(extractor, "as_completed", lambda futures: iter(futures))
    assert extractor.main() == (1 if failed else 0)
    assert extractor._completed_valid(incoming / "good.off", outgoing / "good.npy",
                                      outgoing / "good_centroid.npy", "h01", configuration)
    assert json.loads((outgoing / "feature_names.json").read_text()) == extractor.FEATURE_NAMES
    failed_log = outgoing / "failed.txt"
    if failed:
        assert "bad.off" in failed_log.read_text() and "controlled descriptor failure" in failed_log.read_text()
        assert not (outgoing / "bad.npy").exists()
        assert not (outgoing / "bad_volume_qc.json").exists()
    else:
        assert not failed_log.exists()
        # A resumed complete selection also has a successful CLI status.
        assert extractor.main() == 0
