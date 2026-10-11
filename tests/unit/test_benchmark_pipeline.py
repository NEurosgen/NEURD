"""A/B guards must catch changed science and keep QC out of timed memory."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest


SPEC = importlib.util.spec_from_file_location("benchmark_pipeline", Path(__file__).parents[1] / "tools/benchmark_pipeline.py")
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


def receipt():
    array = benchmark.array_signature(np.array([[1, 2], [3, 4]], dtype=np.int64))
    return dict(config={"sdf_rays": 12}, input={"faces": 2}, metrics={"n_spines_total": 1},
                scientific={"faces": array}, providers={"native": {"sha256": "compiled"}},
                patches={"decimator_open3d": True}, exports={
                    "limb_000/branch_000/branch_mesh.off": {"sha256": "mesh", "size_bytes": 20},
                    "limb_000/branch_000/branch_skeleton.npy": {"arrays": array},
                    "limb_000/branch_index_map.json": {"json": {
                        "branch_index_to_node_id": [0], "node_id_to_branch_index": {"0": 0}}}})


def test_run_order_excludes_setup_and_alternates_pairs():
    assert benchmark.run_order(3) == [("setup", 0, "baseline"), ("setup", 0, "candidate"),
        ("measured", 1, "baseline"), ("measured", 1, "candidate"),
        ("measured", 2, "candidate"), ("measured", 2, "baseline"),
        ("measured", 3, "baseline"), ("measured", 3, "candidate")]


def test_compatible_storage_widths_preserve_exact_values_but_order_matters():
    old = benchmark.array_signature(np.array([[1, 2], [3, 4]], dtype=np.int32))
    new = benchmark.array_signature(np.array([[1, 2], [3, 4]], dtype=np.int64))
    assert old["dtype"] != new["dtype"]
    assert benchmark.values_only(old) == benchmark.values_only(new)
    reversed_rows = benchmark.array_signature(np.array([[3, 4], [1, 2]], dtype=np.int64))
    assert old["sha256"] != reversed_rows["sha256"]
    with pytest.raises(TypeError, match="Object arrays"):
        benchmark.array_signature(np.array([object()]))


def test_exact_float_values_are_not_rounded():
    value = np.array([1.0], dtype=np.float64)
    changed = np.nextafter(value, np.inf)
    assert benchmark.array_signature(value)["sha256"] != benchmark.array_signature(changed)["sha256"]


def test_added_provenance_is_reported_and_common_mapping_checked():
    old, new = receipt(), receipt()
    new["exports"]["limb_000/branch_index_map.json"]["json"]["branch_index_resolution"] = [{"source": "identity"}]
    for name in benchmark.ADDED_EXPORTS:
        new["exports"][f"limb_000/branch_000/{name}"] = {"sha256": name}
    comparison = benchmark.compare_results(old, new)
    assert comparison["passed"] and comparison["common_files"] == 3
    assert comparison["candidate_files"] == 6 and not comparison["export_workloads_equal"]
    new["exports"]["limb_000/branch_index_map.json"]["json"]["branch_index_to_node_id"] = [7]
    assert "Export differs: limb_000/branch_index_map.json" in benchmark.compare_results(old, new)["problems"]


@pytest.mark.parametrize("changed", ["science", "missing", "unexpected", "wrong_provenance_location"])
def test_changed_or_missing_science_fails_closed(changed):
    old, new = receipt(), receipt()
    if changed == "science":
        new["scientific"]["faces"]["sha256"] = "reordered"
    elif changed == "missing":
        del new["exports"]["limb_000/branch_000/branch_mesh.off"]
    elif changed == "unexpected":
        new["exports"]["unrelated.npy"] = {"sha256": "data"}
    else:
        new["exports"]["face_provenance.json"] = {"sha256": "unrelated"}
    assert not benchmark.compare_results(old, new)["passed"]


def test_supervisor_filters_quality_check_rss_and_reports_stage_tree(tmp_path):
    (tmp_path / "work").mkdir()
    code = """
import json, pathlib, time
p = pathlib.Path('..')
start = time.perf_counter()
time.sleep(.14)
finished = time.perf_counter()
(p/'timeline.json').write_text(json.dumps([
    {'phase':'import_setup','monotonic_seconds':start},
    {'phase':'quality_checks','monotonic_seconds':finished}]))
large_qc = bytearray(60 * 1024 * 1024)
time.sleep(.18)
(p/'result.json').write_text(json.dumps({'status':'passed'}))
"""
    result = benchmark.run_child([sys.executable, "-c", code], tmp_path, os.environ.copy(), 5)
    samples = json.loads((tmp_path / "rss_samples.json").read_text())
    memory = result["sampled_memory"]
    assert max(sample["worker_rss_bytes"] for sample in samples) > memory["peak_worker_rss_bytes"] + 40 * 1024 * 1024
    assert "import_setup" in memory["stages"]
    assert memory["peak_tree_rss_bytes"] >= memory["peak_worker_rss_bytes"]


def test_timeout_is_bounded_and_persists_samples(tmp_path):
    (tmp_path / "work").mkdir()
    with pytest.raises(TimeoutError, match="exceeded"):
        benchmark.run_child([sys.executable, "-c", "import time; time.sleep(10)"], tmp_path, os.environ.copy(), .1)
    assert (tmp_path / "rss_samples.json").is_file()


def test_source_guard_ignores_generated_bytecode_but_catches_code_and_configuration(tmp_path):
    (tmp_path / "neurd").mkdir()
    (tmp_path / "neurd/module.py").write_text("value = 1\n")
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/run.json").write_text('{"profile": "fast"}')
    before = benchmark.source_manifest(tmp_path)
    (tmp_path / "neurd/__pycache__").mkdir()
    (tmp_path / "neurd/__pycache__/module.pyc").write_bytes(b"cache")
    assert benchmark.source_manifest(tmp_path) == before
    (tmp_path / "configs/run.json").write_text('{"profile": "reference"}')
    assert benchmark.source_manifest(tmp_path) != before


def test_summary_omits_setup_and_keeps_all_measured_trials():
    runs = []
    for kind, pair, variant in benchmark.run_order(3):
        runs.append(dict(kind=kind, variant=variant, result=dict(stages_seconds={"load": 1000 if kind == "setup" else pair},
            kernel_peak_rss_bytes=pair, sampled_memory={"peak_tree_rss_bytes": pair + 1})))
    summary = benchmark.summarize(runs)
    assert summary["baseline"]["load"] == {"median": 2, "min": 1, "max": 3, "raw_trials": [1, 2, 3]}


def test_existing_output_is_preserved(tmp_path):
    for name in ("old", "new"):
        (tmp_path / name / "neurd").mkdir(parents=True)
        (tmp_path / name / "neurd/segmentation_pipeline.py").write_text("# source\n")
    mesh = tmp_path / "neuron.off"
    mesh.write_text("OFF\n0 0 0\n")
    output = tmp_path / "results"
    output.mkdir()
    (output / "sentinel").write_text("preserve")
    with pytest.raises(SystemExit) as exc:
        benchmark.main(["--baseline-source", str(tmp_path / "old"), "--candidate-source", str(tmp_path / "new"),
                        "--neuron", str(mesh), "--out", str(output)])
    assert exc.value.code == 2
    assert (output / "sentinel").read_text() == "preserve"
