"""Unsupported historical width measurements remain unresolved diagnostics."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import trimesh

spec = importlib.util.spec_from_file_location(
    "h01_failure_diagnostics_under_test",
    Path(__file__).resolve().parents[2] / "scripts/diagnose_h01_failures.py",
)
diagnostics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostics)


def test_final_state_selects_failed_keys_and_rejects_duplicate_inventory():
    plan = [dict(key=key) for key in ["old_failure_now_resumed", "failed", "complete"]]
    state = [dict(key=key, status=status) for key, status in [
        ("old_failure_now_resumed", "resumed"), ("failed", "failed"), ("complete", "complete")]]
    assert [row[0]["key"] for row in diagnostics.unresolved_rows(plan, state)] == ["failed"]
    with pytest.raises(ValueError, match="Duplicate"):
        diagnostics.unresolved_rows(plan + [plan[0]], state)


def test_exact_coverage_distinguishes_axis_mismatch_and_covered_short_branch():
    segments = np.array([[[0, 0, 0], [100, 0, 0]]], float)
    covered = diagnostics.segment_distances([[50, 30, 40], [150, 0, 0]], segments)
    assert covered["minimum_nm"] == covered["maximum_nm"] == 50
    detached = diagnostics.segment_distances([[50, 0, 40000], [100, 0, 45000]], segments)
    assert detached["minimum_nm"] == 40000 and detached["fraction_over_10um"] == 1


def test_slice_support_identifies_axial_and_radial_exclusion_without_inventing_width():
    # Basis identity represents a skeleton along z. First face is axially in
    # range but radially far; second is radially near but axially out of range.
    mesh = SimpleNamespace(triangles_center=np.array([[10000, 0, 50], [0, 0, 200]]))
    support = diagnostics.slice_support(mesh, np.array([[[0, 0, 0], [0, 0, 100]]]), lambda _: np.eye(3))
    assert support["axial_faces_per_segment"] == [1]
    assert support["radial_faces_per_segment"] == [1]
    assert support["supported_segments"] == 0


@pytest.mark.parametrize("median,positive,minimum,cause", [
    (0, 0, 40000, "axis_surface_mismatch_no_supported_width_slices"),
    (0, 0, 50, "no_supported_width_slices"),
    (0, 1, 50, "supported_slices_have_no_positive_width"),
    (116, 1, 50, "current_probe_has_positive_width_historical_failure_needs_reconciliation"),
])
def test_failure_classification_preserves_unresolved_status_and_115nm(median, positive, minimum, cause):
    probe = dict(median_mesh_center_nm=median, slice_support=dict(supported_segments=positive))
    outcome = diagnostics.classification(dict(reason="Worker exited 1"), probe, dict(minimum_nm=minimum))
    assert outcome["status"] == "eligibility_unresolved" and outcome["cause"] == cause
    assert diagnostics.WIDTH_THRESHOLD_NM == 115
    if median > 0:
        assert outcome["current_probe_above_115nm"] is True


def test_timeout_stays_processing_unresolved_without_width_inference():
    outcome = diagnostics.classification(dict(reason="TimeoutExpired: 2700s"), None, dict(minimum_nm=1))
    assert outcome["status"] == "processing_unresolved"
    assert outcome["cause"] == "timeout_stage_not_identified"
    with_log = diagnostics.classification(dict(reason="TimeoutExpired: 2700s"), None, None,
                                         [dict(reported_cgal_segments=22167)])
    assert with_log["cause"] == "post_cgal_timeout_stage_not_fully_instrumented"


def test_selected_single_triangle_is_measurement_degeneracy_not_thin_classification():
    probe = dict(median_mesh_center_nm=0, slice_support=dict(supported_segments=1),
                 selected_slice_face_counts=[1])
    outcome = diagnostics.classification(dict(reason="Worker exited 1"), probe, dict(minimum_nm=50))
    assert outcome == dict(status="eligibility_unresolved", cause="selected_single_face_slices_zero_mesh_center_width")


def test_off_header_only_probe_does_not_require_loading_large_geometry(tmp_path):
    path = tmp_path / "large.off"
    path.write_text("# exported\nOFF\n779393 1564481 0\nbody intentionally not read\n")
    assert diagnostics.off_counts(path) == (779393, 1564481)


def test_export_counts_and_derived_caliber_are_not_original_width_metadata(tmp_path):
    branch = tmp_path / "neuron_1/limb_000/branch_000"
    branch.mkdir(parents=True)
    (branch.parent.parent / "summary.json").write_text(json.dumps(dict(
        limbs=[dict(n_branches=1, spines_per_branch=[0])], caliber_um=0.2)))
    (branch.parent / "branch_index_map.json").write_text('{"branch_index_to_node_id": [0]}')
    result = diagnostics.metadata_inventory(branch)
    assert result["original_width_eligibility_metadata"] == "not_available"
    assert result["metadata_used_for_eligibility"] is False
    assert all(not record["width_field_paths"] for record in result["files"])


def test_named_width_field_requires_validation_before_use(tmp_path):
    branch = tmp_path / "neuron_1/limb_000/branch_000"
    branch.mkdir(parents=True)
    (branch / "width.json").write_text('{"median_mesh_center": 116}')
    result = diagnostics.metadata_inventory(branch)
    assert result["original_width_eligibility_metadata"] == "unverified_named_fields"
    assert result["metadata_used_for_eligibility"] is False


def fake_width_helper():
    edges = np.array([[[0, 0, 0], [0, 0, 1000]], [[0, 0, 1000], [0, 0, 2000]]], float)
    backend = SimpleNamespace(skeletal_distance_no_skipping=lambda **kwargs: ([0, 0], [0, 0], None, []))

    def calculate(branch, **kwargs):
        backend.skeletal_distance_no_skipping(main_mesh=branch.mesh, edges=edges)
        return np.zeros(2), 0

    return SimpleNamespace(cb=backend, skeleton_resized_ordered=lambda _: edges,
                           calculate_new_width=calculate, sk=SimpleNamespace(change_basis_matrix=lambda _: np.eye(3)))


def test_probe_captures_raw_placeholders_and_restores_backend():
    helper = fake_width_helper()
    original_call = helper.cb.skeletal_distance_no_skipping
    mesh = SimpleNamespace(triangles_center=np.array([[40000, 0, 500]]))
    probe = diagnostics.width_probe(mesh, np.array([[[0, 0, 0], [0, 0, 2000]]]), helper)
    assert probe["raw_distances_nm"] == [0, 0]
    assert probe["raw_positive_distances"] == 0 and probe["old_width_fallback_nm"] == 0
    assert helper.cb.skeletal_distance_no_skipping is original_call


def test_report_hashes_sources_keeps_failures_unresolved_and_preserves_frozen_run(tmp_path, monkeypatch):
    run, output = tmp_path / "historical", tmp_path / "diagnostic"
    source = tmp_path / "source/neuron_1/limb_000/branch_000"
    source.mkdir(parents=True)
    mesh_path, skeleton_path = source / "branch_mesh.off", source / "branch_skeleton.npy"
    trimesh.creation.box().apply_translation([40000, 0, 0]).export(mesh_path)
    np.save(skeleton_path, np.array([[[0, 0, 0], [0, 0, 2000]]]))
    row = dict(key="neuron_1_limb_000_branch_000", relative="neuron_1/limb_000/branch_000",
               source=str(mesh_path), skeleton=str(skeleton_path), archived_spines=0)
    run.mkdir()
    content = {"manifest.json": dict(run_hash="frozen", code_sha256={}), "plan.json": [row],
               "last_run.json": [dict(key=row["key"], status="failed", reason="Worker exited 1")],
               "last_run_summary.json": dict(failed=1), "profile.json": {}}
    for name, value in content.items():
        (run / name).write_text(json.dumps(value))
    (run / "run_full_h01.sh").write_text("frozen launch")
    failure = run / "failures" / (row["key"] + "_attempt")
    failure.mkdir(parents=True)
    (failure / "run.log").write_text("ValueError: Cannot establish original NEURD branch-width eligibility\n")
    before = {p: p.read_bytes() for p in run.rglob("*") if p.is_file()}
    monkeypatch.setattr(diagnostics, "frozen_providers", lambda *args: (fake_width_helper(), {}))
    report = diagnostics.diagnose(run, output)
    branch = report["branches"][0]
    assert branch["diagnosis"]["status"] == "eligibility_unresolved"
    assert branch["inputs"]["mesh"]["sha256"] == diagnostics.sha256(mesh_path)
    assert branch["accepted_result"] is False
    assert report["summary"]["accepted_results"] == 0
    assert json.loads((output / "eligibility_unresolved_keys.json").read_text()) == [row["key"]]
    assert "eligibility_unresolved" in (output / "branches.csv").read_text()
    assert {p: p.read_bytes() for p in run.rglob("*") if p.is_file()} == before
    with pytest.raises(ValueError, match="new diagnostic"):
        diagnostics.diagnose(run, output)
    changed_output = tmp_path / "changed_diagnostic"
    original_probe = diagnostics.width_probe

    def changing_probe(*args, **kwargs):
        result = original_probe(*args, **kwargs)
        mesh_path.write_text(mesh_path.read_text() + "\n")
        return result

    monkeypatch.setattr(diagnostics, "width_probe", changing_probe)
    with pytest.raises(ValueError, match="Source changed"):
        diagnostics.diagnose(run, changed_output)
    assert not changed_output.exists()
