"""Raw worker resume, no-save, failures, pipe draining and shared scheduling."""
import importlib
import json
import multiprocessing as mp
from types import SimpleNamespace

import pytest

from tests.unit import skip_if_datasci_tools_unusable

skip_if_datasci_tools_unusable()

import process_all_neurons as batch
import process_all_neurons_parallel as parallel
from neurd.run_config import RunConfig
from neurd import batch_utils


@pytest.fixture
def raw_case(tmp_path, monkeypatch):
    # Real isolated processes inherit mocks with fork; this is the Linux runner.
    if 'fork' not in mp.get_all_start_methods():
        pytest.skip('worker mock needs fork')
    monkeypatch.setattr(batch, 'mp', mp.get_context('fork'))
    monkeypatch.delenv('SEG_RSS_LIMIT_MB', raising=False)
    monkeypatch.delenv('SEG_TIMEOUT_SEC', raising=False)
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'neuron.off').write_text('fixture')
    output = tmp_path / 'output'
    config = RunConfig(dataset='h01', calculate_spines=False, sdf_profile='fast')
    monkeypatch.setattr(batch, '_run_signature', lambda config, ratio, max_somas: dict(config=config.as_dict()))
    monkeypatch.setattr(batch, 'load_mesh', lambda path: 'loaded_mesh')
    def build(mesh, *, config, max_somas):
        assert mesh in ('loaded_mesh', 'decimated_mesh')
        assert config.dataset == 'h01' and config.sdf_rays == 12
        return SimpleNamespace(limbs=[], concept_network=None, run_config=config.as_dict())
    monkeypatch.setattr(batch, 'segmentation_pipeline', build)
    return source, output, config


def run(case, **kwargs):
    source, output, config = case
    return batch.process_folder_with_manifest(str(source), str(output), config=config, poll_interval=0.01, **kwargs)


def test_no_save_computes_without_outputs_or_manifest(raw_case):
    result = run(raw_case, no_save=True)
    assert result.processed == 1 and result.exit_code == 0
    assert not raw_case[1].exists()


def test_saved_result_is_verified_before_resume_and_tampering_fails(raw_case):
    source, output, config = raw_case
    assert run(raw_case).processed == 1
    record = json.loads((output / 'neuron/result.json').read_text())
    assert record['run_config']['sdf_rays'] == 12
    assert record['run_config']['calculate_spines'] is False
    assert batch.digest_json(record['provenance']) == record['run_hash']
    assert 'summary.json' in record['outputs_sha256']
    assert batch.read_manifest(output / 'processed.txt') == {'neuron'}
    assert run(raw_case).skipped == 1
    (output / 'neuron/summary.json').write_text('changed')
    with pytest.raises(ValueError, match='missing or changed'):
        run(raw_case)


def test_changed_input_and_config_do_not_resume(raw_case):
    source, output, config = raw_case
    run(raw_case)
    changed_config = RunConfig(dataset='h01', sdf_profile='reference')
    with pytest.raises(ValueError, match='changed inputs or settings'):
        run((source, output, changed_config))
    (source / 'neuron.off').write_text('different source bytes')
    with pytest.raises(ValueError, match='changed inputs or settings'):
        run(raw_case)


def test_legacy_output_is_preserved_and_requires_new_destination(raw_case):
    output = raw_case[1]
    (output / 'neuron').mkdir(parents=True)
    sentinel = output / 'neuron/user_file'
    sentinel.write_text('preserve me')
    with pytest.raises(ValueError, match='new output directory'):
        run(raw_case)
    assert sentinel.read_text() == 'preserve me'


def test_large_failure_message_drains_pipe_and_retry_is_explicit(raw_case, monkeypatch, capsys):
    monkeypatch.setenv('SEG_TIMEOUT_SEC', '2')
    def fail(*args, **kwargs):
        raise ValueError('controlled failure ' + 'x' * 200000)
    original = batch.segmentation_pipeline
    monkeypatch.setattr(batch, 'segmentation_pipeline', fail)
    result = run(raw_case)
    assert result.failed == 1 and result.exit_code == 1
    failed_log = raw_case[1] / 'failed.txt'
    contents = failed_log.read_text()
    assert 'controlled failure' in contents and 'Timeout' not in contents
    assert len(contents.splitlines()) == 1
    assert not (raw_case[1] / 'neuron').exists()
    assert run(raw_case).skipped_failed == 1
    monkeypatch.setattr(batch, 'segmentation_pipeline', original)
    assert run(raw_case, retry_failed=True).processed == 1
    capsys.readouterr()


def test_no_save_failure_does_not_create_failure_log(raw_case, monkeypatch, capsys):
    monkeypatch.setattr(batch, 'segmentation_pipeline', lambda *a, **kw: (_ for _ in ()).throw(ValueError('failure')))
    assert run(raw_case, no_save=True).exit_code == 1
    assert not raw_case[1].exists()
    capsys.readouterr()


def test_explicit_decimation_is_applied_in_worker(raw_case, monkeypatch):
    mesh_ops = importlib.import_module('neurd._mesh_ops')
    def decimate(mesh, *, decimation_ratio):
        assert mesh == 'loaded_mesh' and decimation_ratio == 0.5
        return 'decimated_mesh'
    monkeypatch.setattr(mesh_ops, 'decimate', decimate)
    assert run(raw_case, no_save=True, decimation_ratio=0.5).processed == 1
    with pytest.raises(ValueError, match='cannot be combined'):
        run(raw_case, no_decimate=True, decimation_ratio=0.5)


def test_required_rss_limit_cannot_silently_disappear(raw_case, monkeypatch):
    monkeypatch.setattr(batch, '_PSUTIL', None)
    monkeypatch.setenv('SEG_RSS_LIMIT_MB', '512')
    with pytest.raises(RuntimeError, match='requires psutil'):
        run(raw_case)


def test_cli_failure_status_and_parallel_settings_are_shared(monkeypatch):
    captured = []
    def process(*args, **kwargs):
        captured.append(kwargs)
        return batch.BatchResult(failed=1)
    monkeypatch.setattr(batch, 'process_folder_with_manifest', process)
    assert batch.main(['input', 'output', '--dataset', 'h01', '--no-spines', '--sdf-profile', 'fast']) == 1
    assert captured[0]['config'] == RunConfig(dataset='h01', sdf_profile='fast')
    monkeypatch.setattr(parallel, 'process_folder_with_manifest', process)
    assert parallel.main(['input', 'output', '--no-spines', '--input-size-budget-mb', '256', '--max-workers', '2']) == 1
    assert captured[1]['input_size_budget_mb'] == 256 and captured[1]['max_workers'] == 2
    assert parallel.main(['input', 'output', '--no-spines', '--mem-budget-mb', '128']) == 1
    assert captured[2]['input_size_budget_mb'] == 128


@pytest.mark.parametrize('location', ['source', 'neuron/summary.json', 'failed.txt', '.work/manifest'])
def test_manifest_cannot_modify_source_or_reserved_outputs(raw_case, location):
    source, output, config = raw_case
    manifest = str(source / 'neuron.off') if location == 'source' else location
    before = (source / 'neuron.off').read_bytes()
    with pytest.raises(ValueError, match='Manifest'):
        run(raw_case, manifest=manifest)
    assert (source / 'neuron.off').read_bytes() == before
    assert not output.exists()


def test_skeletonizer_binary_changes_raw_resume_fingerprint(monkeypatch, tmp_path):
    binary = tmp_path / 'calcification_param_Module.so'
    binary.write_bytes(b'original native skeletonizer')
    monkeypatch.setattr(batch_utils, 'dependency_signature', lambda: {})
    original_find_spec = batch_utils.importlib.util.find_spec
    def find_spec(name):
        if name == 'calcification_param_Module':
            return SimpleNamespace(origin=str(binary))
        return original_find_spec(name)
    monkeypatch.setattr(batch_utils.importlib.util, 'find_spec', find_spec)
    config = RunConfig()
    original = batch._run_hash(config, 0.0, None)
    binary.write_bytes(b'rebuilt native skeletonizer')
    assert batch._run_hash(config, 0.0, None) != original


def test_manifest_cannot_overwrite_symlink_target_or_other_completed_neuron(raw_case, tmp_path):
    source, output, _ = raw_case
    real_mesh = tmp_path / 'outside.off'
    (source / 'neuron.off').rename(real_mesh)
    (source / 'neuron.off').symlink_to(real_mesh)
    with pytest.raises(ValueError, match='Manifest'):
        run(raw_case, manifest=str(real_mesh))
    old_neuron = output / 'other_neuron'
    old_neuron.mkdir(parents=True)
    (old_neuron / 'result.json').write_text('{}')
    with pytest.raises(ValueError, match='reserved'):
        run(raw_case, manifest='other_neuron/result.json')
    assert real_mesh.read_text() == 'fixture'


@pytest.mark.parametrize('no_save', [False, True])
def test_timeout_removes_supervised_scratch_and_staging(raw_case, monkeypatch, no_save):
    import time
    from pathlib import Path
    created = []
    mkdtemp = batch.tempfile.mkdtemp
    def track_directory(*args, **kwargs):
        path = mkdtemp(*args, **kwargs)
        created.append(Path(path))
        return path
    monkeypatch.setattr(batch.tempfile, 'mkdtemp', track_directory)
    monkeypatch.setattr(batch, 'segmentation_pipeline', lambda *a, **kw: time.sleep(5))
    monkeypatch.setenv('SEG_TIMEOUT_SEC', '1')
    assert run(raw_case, no_save=no_save).failed == 1
    assert created and all(not path.exists() for path in created)
    assert not (raw_case[1] / 'neuron').exists()
    if no_save:
        assert not raw_case[1].exists()


@pytest.mark.parametrize('flag', ['NEURD_OWNED_SKELETAL_DISTANCE', 'NEURD_LEGACY_MESH_LOADER',
                                  'NEURD_LEGACY_CGAL_ASSEMBLY'])
def test_behavior_flags_change_resume_hash_but_ambient_sdf_rays_do_not(monkeypatch, flag):
    monkeypatch.setattr(batch_utils, 'dependency_signature', lambda: {})
    monkeypatch.delenv(flag, raising=False)
    config = RunConfig(sdf_profile='fast')
    monkeypatch.setenv('NEURD_SDF_RAYS', '999')
    original = batch._run_hash(config, 0.0, None)
    monkeypatch.setenv('NEURD_SDF_RAYS', '12')
    assert batch._run_hash(config, 0.0, None) == original
    monkeypatch.setenv(flag, '1')
    assert batch._run_hash(config, 0.0, None) != original


def test_exported_masks_preserve_actual_parent_face_identity(tmp_path):
    import numpy as np
    import trimesh
    # The first two parent faces are geometrically identical, but only face 1
    # belongs to this spine. Geometry-based reverse mapping would be ambiguous.
    parent = trimesh.Trimesh(vertices=[[0,0,0],[1,0,0],[0,1,0],[0,0,1]],
                             faces=[[0,1,2],[0,1,2],[0,2,3]], process=False)
    spine = parent.submesh([[1]], append=True, repair=False)
    branch = SimpleNamespace(mesh=parent, skeleton=None, spines=[spine],
                             spines_face_idx=[np.array([1], dtype=np.int64)])
    limb = SimpleNamespace(mesh=parent, skeleton=None, branches=[branch])
    obj = SimpleNamespace(limbs=[limb], concept_network=None)
    batch.save_segmentation(obj, tmp_path)
    with np.load(tmp_path / 'limb_000/branch_000/spine_faces.npz', allow_pickle=False) as masks:
        assert masks['faces'].tolist() == [1]
        assert masks['offsets'].tolist() == [0, 1]
        assert masks['spine_files'].tolist() == ['spines/spine_000.off']
        assert str(masks['scope']) == 'branch_mesh.face_index'
    summary = json.loads((tmp_path / 'summary.json').read_text())
    assert summary['limbs'][0]['spine_face_provenance_per_branch'] == ['available']
    assert (tmp_path / 'limb_000/branch_000/spines/spine_000.off').is_file()
    branch.spines_face_idx = None
    legacy = tmp_path / 'legacy'
    batch.save_segmentation(obj, legacy)
    assert not (legacy / 'limb_000/branch_000/spine_faces.npz').exists()
    assert json.loads((legacy / 'summary.json').read_text())['limbs'][0]['spine_face_provenance_per_branch'] == ['unavailable']


@pytest.mark.parametrize('bad_mask', ['count', 'bounds', 'geometry', 'duplicate', 'float'])
def test_supplied_spine_masks_are_validated_before_export(tmp_path, bad_mask):
    import numpy as np
    import trimesh
    parent = trimesh.creation.box()
    spine = parent.submesh([[0]], append=True, repair=False)
    masks = dict(count=[], bounds=[np.array([len(parent.faces)])], geometry=[np.array([1])],
                 duplicate=[np.array([0,0])], float=[np.array([0.0])])
    branch = SimpleNamespace(mesh=parent, spines_face_idx=masks[bad_mask])
    with pytest.raises(ValueError, match='[Ss]pine'):
        batch._save_spine_masks(branch, [spine], tmp_path / 'masks.npz')
    assert not (tmp_path / 'masks.npz').exists()


def test_computed_empty_spine_masks_have_an_explicit_empty_export(tmp_path):
    import numpy as np
    import trimesh
    branch = SimpleNamespace(mesh=trimesh.creation.box(), spines_face_idx=[])
    batch._save_spine_masks(branch, [], tmp_path / 'masks.npz')
    with np.load(tmp_path / 'masks.npz', allow_pickle=False) as masks:
        assert masks['faces'].tolist() == []
        assert masks['offsets'].tolist() == [0]


def test_distinct_parent_ids_cannot_hide_cross_spine_surface_overlap(tmp_path):
    import numpy as np
    import trimesh
    parent = trimesh.Trimesh(vertices=[[0,0,0],[1,0,0],[0,1,0]],
                             faces=[[0,1,2],[0,1,2]], process=False)
    meshes = [parent.submesh([[index]], append=True, repair=False) for index in (0,1)]
    branch = SimpleNamespace(mesh=parent, spines_face_idx=[np.array([0]),np.array([1])])
    with pytest.raises(ValueError, match='same triangle geometry'):
        batch._save_spine_masks(branch, meshes, tmp_path / 'overlap.npz')
    assert not (tmp_path / 'overlap.npz').exists()
    # Original duplicated faces inside one object retain their distinct parent
    # identities; this check prevents sharing between different objects only.
    branch.spines_face_idx = [np.array([0,1])]
    combined = parent.submesh([[0,1]], append=True, repair=False)
    batch._save_spine_masks(branch, [combined], tmp_path / 'single_spine.npz')
    with np.load(tmp_path / 'single_spine.npz', allow_pickle=False) as masks:
        assert masks['faces'].tolist() == [0,1]
