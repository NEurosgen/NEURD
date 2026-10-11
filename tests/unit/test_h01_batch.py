"""Protect source ownership, resumability and skeleton graph connections."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('h01_batch_under_test',Path(__file__).resolve().parents[2]/'neurd/h01_batch.py')
batch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(batch)


def segments(points,edges):
    return np.asarray(points,dtype=float)[np.asarray(edges)]


def test_skeleton_anchoring_retains_connections_and_prunes_new_twigs():
    original = segments([(0,0,0),(5000,0,0)],[(0,1)])
    fresh = segments([(-1000,0,0),(2000,0,0),(6000,0,0),(2000,1000,0)],[(0,1),(1,2),(1,3)])
    repaired,reason = batch.anchor_skeleton(original,fresh)
    assert repaired is not None,reason
    vertices,edges,degree = batch.graph_data(repaired)
    assert {tuple(p) for p in vertices[degree==1]} == {(0.,0.,0.),(5000.,0.,0.)}
    assert not (vertices[:,1] != 0).any()
    assert np.linalg.norm(repaired[:,1]-repaired[:,0],axis=1).sum() == pytest.approx(5000)


def test_far_old_graph_connections_cannot_be_replaced_by_local_axis():
    original = segments([(0,0,0),(5000,0,0)],[(0,1)])
    fresh = original + [0,5000,0]
    repaired,reason = batch.anchor_skeleton(original,fresh)
    assert repaired is None and 'far_from' in reason


def test_new_cycle_cannot_be_exported_as_a_tree():
    original = segments([(0,0,0),(5000,0,0)],[(0,1)])
    fresh = segments([(0,0,0),(2500,500,0),(5000,0,0)],[(0,1),(1,2),(2,0)])
    assert batch.anchor_skeleton(original,fresh)[0] is None


def test_matching_landmark_degrees_do_not_allow_leaf_attachment_changes():
    points = [(0,0,0),(1000,0,0),(-1000,-1000,0),(-1000,1000,0),
              (2000,-1000,0),(2000,1000,0)]
    original = segments(points,[(0,1),(0,2),(0,3),(1,4),(1,5)])
    changed = segments(points,[(0,1),(0,2),(0,4),(1,3),(1,5)])
    old_vertices,_,old_degree = batch.graph_data(original)
    new_vertices,_,new_degree = batch.graph_data(changed)
    assert np.array_equal(old_vertices,new_vertices)
    assert np.array_equal(old_degree,new_degree)
    repaired,reason = batch.anchor_skeleton(original,changed)
    assert repaired is None
    assert reason == 'anchoring_changes_original_landmark_topology'


def test_changed_degree_two_sampling_preserves_labelled_branch_connections():
    points = [(0,0,0),(1000,0,0),(-1000,-1000,0),(-1000,1000,0),
              (2000,-1000,0),(2000,1000,0)]
    original = segments(points,[(0,1),(0,2),(0,3),(1,4),(1,5)])
    sampled = []
    for a,b in original:
        middle = a + .4*(b-a)
        sampled.extend([[b,middle],[middle,a]])
    fresh = np.asarray(sampled)[::-1]
    repaired,reason = batch.anchor_skeleton(original,fresh)
    assert repaired is not None,reason
    assert reason == 'exact_original_graph_landmark_topology_preserved'
    assert batch._landmark_topology(repaired) == batch._landmark_topology(original)
    assert len(repaired) == 2*len(original)


@pytest.mark.parametrize('original',[
    segments([(0,0,0),(1000,0,0),(500,1000,0)],[(0,1),(1,2),(2,0)]),
    segments([(0,0,0),(1000,0,0),(2000,0,0),(3000,0,0)],[(0,1),(2,3)]),
    segments([(0,0,0),(1000,0,0)],[(0,1),(0,1)]),
])
def test_original_cycles_and_disconnected_graphs_are_retained(original):
    fresh = segments([(0,0,0),(3000,0,0)],[(0,1)])
    repaired,reason = batch.anchor_skeleton(original,fresh)
    assert repaired is None
    assert reason == 'original_graph_is_not_a_connected_tree'


def test_contracted_topology_retains_parallel_paths_and_unlabelled_cycles():
    theta = segments([(0,0,0),(4000,0,0),(2000,-1000,0),(2000,0,0),(2000,1000,0)],
                     [(0,2),(2,1),(0,3),(3,1),(0,4),(4,1)])
    topology = batch._landmark_topology(theta)
    assert topology['landmarks'] == (((0.,0.,0.),3),((4000.,0.,0.),3))
    assert topology['paths'] == (((0.,0.,0.),(4000.,0.,0.)),)*3
    assert topology['components'] == 1 and topology['cycle_rank'] == 2
    cycle = segments([(10000,0,0),(11000,0,0),(10500,1000,0)],[(0,1),(1,2),(2,0)])
    combined = batch._landmark_topology(np.concatenate([theta,cycle]))
    assert combined['components'] == 2 and combined['cycle_rank'] == 3
    assert combined['unlabelled_cycles'] == 1
    assert combined['paths'] == topology['paths']


def test_contracted_topology_retains_a_cycle_returning_to_its_landmark():
    original = segments([(0,0,0),(-1000,0,0),(1000,-1000,0),(1000,1000,0)],
                        [(0,1),(0,2),(2,3),(3,0)])
    topology = batch._landmark_topology(original)
    assert topology['paths'] == (((-1000.,0.,0.),(0.,0.,0.)),((0.,0.,0.),(0.,0.,0.)))
    assert topology['cycle_rank'] == 1 and topology['unlabelled_cycles'] == 0


def test_contracted_topology_rejects_new_landmarks():
    original = segments([(0,0,0),(2000,0,0)],[(0,1)])
    branched = segments([(0,0,0),(1000,0,0),(2000,0,0),(1000,1000,0)],[(0,1),(1,2),(1,3)])
    assert batch._landmark_topology(original) != batch._landmark_topology(branched)


@pytest.mark.parametrize('allow_replacement',[False,True])
def test_repair_preserves_exported_skeleton_until_explicitly_enabled(monkeypatch,allow_replacement):
    import sys
    from types import SimpleNamespace
    original = segments([(0,0,0),(1000,0,0)],[(0,1)])
    fresh_vertices = np.array([(0,0,0),(500,100,0),(1000,0,0)],float)
    fresh_edges = np.array([(0,1),(1,2)])
    rebuilt = SimpleNamespace(vertices=fresh_vertices,edges=fresh_edges)
    monkeypatch.setitem(sys.modules,'mesh_tools',SimpleNamespace(
        meshparty_skeletonize=SimpleNamespace(skeletonize_mesh_largest_component=lambda *a,**k:rebuilt)))
    qualities = iter([dict(q95_nm=30000),dict(q95_nm=100),dict(q95_nm=100)])
    monkeypatch.setattr(batch,'coverage',lambda *a,**k:next(qualities))
    settings = dict(repair=True,coverage_q95_nm=10000,invalidation_d_nm=12000,maximum_connector_nm=2000)
    if allow_replacement:
        settings['allow_topological_replacement'] = True
    repaired,auxiliary,record = batch.repair_skeleton(SimpleNamespace(vertices=fresh_vertices),original,settings)
    assert np.array_equal(auxiliary,fresh_vertices[fresh_edges])
    assert record['topological_replacement'] is allow_replacement
    if allow_replacement:
        assert record['status'] == 'repaired_with_original_connections'
        assert batch._landmark_topology(repaired) == batch._landmark_topology(original)
    else:
        assert repaired is original
        assert record['status'] == 'auxiliary_axis_only'
        assert record['reason'] == 'topological_replacement_disabled'


def test_explicit_replacement_rejects_a_reconnected_axis_but_keeps_auxiliary(monkeypatch):
    import sys
    from types import SimpleNamespace
    points = np.array([(0,0,0),(1000,0,0),(-1000,-1000,0),(-1000,1000,0),
                       (2000,-1000,0),(2000,1000,0)],float)
    original = segments(points,[(0,1),(0,2),(0,3),(1,4),(1,5)])
    fresh_edges = np.array([(0,1),(0,2),(0,4),(1,3),(1,5)])
    rebuilt = SimpleNamespace(vertices=points,edges=fresh_edges)
    monkeypatch.setitem(sys.modules,'mesh_tools',SimpleNamespace(
        meshparty_skeletonize=SimpleNamespace(skeletonize_mesh_largest_component=lambda *a,**k:rebuilt)))
    qualities = iter([dict(q95_nm=30000),dict(q95_nm=100)])
    monkeypatch.setattr(batch,'coverage',lambda *a,**k:next(qualities))
    settings = dict(repair=True,allow_topological_replacement=True,coverage_q95_nm=10000,
                    invalidation_d_nm=12000,maximum_connector_nm=2000)
    repaired,auxiliary,record = batch.repair_skeleton(SimpleNamespace(vertices=points),original,settings)
    assert repaired is original
    assert np.array_equal(auxiliary,points[fresh_edges])
    assert record['topological_replacement'] is False
    assert record['status'] == 'auxiliary_axis_only'
    assert record['topology_reason'] == 'anchoring_changes_original_landmark_topology'


@pytest.mark.parametrize('opt_in',[False,True])
def test_cli_skeleton_replacement_requires_explicit_flag(tmp_path,monkeypatch,opt_in):
    monkeypatch.setattr(batch,'run',lambda args,*roots:args)
    argv = ['--source',str(tmp_path/'source'),'--output',str(tmp_path/'out'),'--config',str(tmp_path/'config.json')]
    if opt_in:
        argv.append('--experimental-skeleton-replacement')
    args = batch.main(argv)
    assert args.experimental_skeleton_replacement is opt_in


def test_cli_rejects_replacement_when_repair_is_disabled(tmp_path):
    with pytest.raises(SystemExit):
        batch.main(['--source',str(tmp_path/'source'),'--output',str(tmp_path/'out'),
                    '--config',str(tmp_path/'config.json'),'--no-skeleton-repair','--experimental-skeleton-replacement'])


def test_changed_completed_output_or_input_prevents_resume(tmp_path):
    artifact = tmp_path/'branch_skeleton.npy'
    np.save(artifact,np.zeros((1,2,3)))
    source = {'mesh':{'path':'/source.off','sha256':'original'}}
    batch.save(tmp_path/'result.json',dict(status='complete',run_hash='version',inputs=source,
               outputs_sha256={artifact.name:batch.sha(artifact)}))
    assert batch.complete_valid(tmp_path,'version',source)
    with pytest.raises(ValueError,match='changed inputs or settings'):
        batch.complete_valid(tmp_path,'version',{'mesh':{'path':'/source.off','sha256':'new'}})
    artifact.write_bytes(b'damaged')
    with pytest.raises(ValueError,match='missing or changed'):
        batch.complete_valid(tmp_path,'version',source)


@pytest.mark.parametrize('out',['source','source/new','parent'])
def test_output_must_not_contain_or_be_inside_source(tmp_path,out):
    source = tmp_path/'source'
    output = tmp_path if out=='parent' else tmp_path/out
    with pytest.raises(ValueError,match='non-nested'):
        batch.separate_roots(source,output)


def test_inventory_includes_branches_without_old_spines(tmp_path):
    branch = tmp_path/'neuron_1/limb_000/branch_000'
    branch.mkdir(parents=True)
    (branch/'branch_mesh.off').write_text('OFF\n0 0 0\n')
    np.save(branch/'branch_skeleton.npy',np.zeros((1,2,3)))
    rows = batch.inventory(tmp_path)
    assert len(rows)==1 and rows[0]['archived_spines']==0 and rows[0]['skeleton_exists']


def test_short_original_axis_is_not_covered_by_large_parent():
    vertices = np.array([(0,0,0),(30000,0,0),(60000,0,0)],float)
    axis = segments([(0,0,0),(100,0,0)],[(0,1)])
    assert batch.coverage(vertices,axis)['q95_nm'] > 50000


def test_triangle_provenance_accepts_reordering_but_rejects_different_surface():
    import trimesh
    mesh = trimesh.creation.box()
    reordered = trimesh.Trimesh(mesh.vertices.copy(),mesh.faces[::-1,::-1].copy(),process=False)
    assert np.array_equal(batch.triangle_keys(mesh),batch.triangle_keys(reordered))
    reordered.vertices[0] += [0.1,0,0]
    assert not np.array_equal(batch.triangle_keys(mesh),batch.triangle_keys(reordered))


def test_interrupt_stops_a_large_queue_and_keeps_completed_results():
    from threading import Event
    stop = Event()
    started = []
    def job(row):
        started.append(row)
        stop.set()
        return dict(key=row,status='complete')
    observed = []
    result = batch.dispatch(range(10000),job,2,stop,lambda r,n:observed.append(r))
    assert len(started)<=2 and result==observed
    assert {r['key'] for r in result}==set(started)


@pytest.mark.parametrize('outputs', [None, {}, [], {'branch_skeleton.npy': 'not-a-sha256'},
                                     {'../source.off': 'a'*64}, {'/source.off': 'a'*64}])
def test_empty_or_malformed_completion_hash_map_is_rejected(tmp_path, outputs):
    source = {'mesh': {'path': '/source.off', 'sha256': 'input'}}
    batch.save(tmp_path/'result.json', dict(status='complete', run_hash='version',
                                          inputs=source, outputs_sha256=outputs))
    with pytest.raises(ValueError, match='output hash'):
        batch.complete_valid(tmp_path, 'version', source)


def test_truncated_hash_map_cannot_hide_existing_unverified_output(tmp_path):
    first, second = tmp_path/'branch_skeleton.npy', tmp_path/'spine_faces.npz'
    first.write_bytes(b'original skeleton')
    second.write_bytes(b'actual spine masks')
    source = {'mesh': {'path': '/source.off', 'sha256': 'input'}}
    record = dict(status='complete', run_hash='version', inputs=source,
                  outputs_sha256={first.name: batch.sha(first), second.name: batch.sha(second)})
    batch.save(tmp_path/'result.json', record)
    assert batch.complete_valid(tmp_path, 'version', source)
    record['outputs_sha256'].pop(second.name)
    batch.save(tmp_path/'result.json', record)
    with pytest.raises(ValueError, match='inventory is incomplete'):
        batch.complete_valid(tmp_path, 'version', source)


def test_fresh_worker_imports_pinned_package_before_external_package(tmp_path):
    output = tmp_path/'out'
    pinned = output/'runtime/neurd'
    pinned.mkdir(parents=True)
    worker = pinned/'h01_batch.py'
    shutil.copyfile(batch.__file__, worker)
    (pinned/'__init__.py').write_text(
        "print('PINNED_RUNTIME_IMPORTED', flush=True)\nraise SystemExit(37)\n")
    external = tmp_path/'external/neurd'
    external.mkdir(parents=True)
    (external/'__init__.py').write_text(
        "print('EXTERNAL_PACKAGE_IMPORTED', flush=True)\nraise SystemExit(38)\n")
    staging = output/'.work/branch'
    staging.mkdir(parents=True)
    result = subprocess.run(
        [sys.executable, str(worker), '--output', str(output), '--worker', 'branch',
         '--staging', str(staging)], cwd=staging,
        env=dict(os.environ, PYTHONPATH=str(external.parent), PYTHONDONTWRITEBYTECODE='1'),
        capture_output=True, text=True, timeout=30)
    assert result.returncode == 37, result.stdout + result.stderr
    assert 'PINNED_RUNTIME_IMPORTED' in result.stdout
    assert 'EXTERNAL_PACKAGE_IMPORTED' not in result.stdout


def _plan_for_runtime_test(tmp_path, monkeypatch):
    source = tmp_path/'source'
    branch = source/'neuron_1/limb_000/branch_000'
    branch.mkdir(parents=True)
    (branch/'branch_mesh.off').write_text('OFF\n0 0 0\n')
    np.save(branch/'branch_skeleton.npy', np.zeros((1,2,3)))
    config = tmp_path/'config.json'
    config.write_text('{}\n')
    output = tmp_path/'out'
    # Planning does not need native dependencies to verify the runtime contract.
    monkeypatch.setattr(batch, 'dependency_signature', lambda: {'test': 'dependencies'})
    argv = ['--source', str(source), '--output', str(output), '--config', str(config),
            '--plan-only', '--rays', '12']
    batch.main(argv)
    return output, argv


def test_plan_fingerprints_effective_threads_and_matches_fresh_child(tmp_path, monkeypatch):
    parent_threads = dict(OMP_NUM_THREADS='7', OPENBLAS_NUM_THREADS='3',
                          MKL_NUM_THREADS='2', NUMBA_NUM_THREADS='2')
    for name, value in parent_threads.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv('NEURD_LEGACY_MESH_LOADER', raising=False)
    output, _ = _plan_for_runtime_test(tmp_path, monkeypatch)
    expected = json.loads((output/'manifest.json').read_text())['runtime_environment']
    assert {name: expected['environment'][name] for name in parent_threads} == dict(
        OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
        NUMBA_NUM_THREADS='4')
    assert {name: os.environ[name] for name in parent_threads} == parent_threads
    child = """
import json
from neurd.run_config import RunConfig, runtime_environment
print('RUNTIME_RESULT ' + json.dumps(runtime_environment(
    RunConfig(dataset='h01', calculate_spines=True, sdf_rays=12)), sort_keys=True))
"""
    env = batch.worker_environment(output)
    env['PYTHONPATH'] = str(output/'runtime')
    result = subprocess.run([sys.executable, '-c', child], cwd=tmp_path, env=env,
                            capture_output=True, text=True, timeout=45, check=True)
    marker = 'RUNTIME_RESULT '
    line = next(line for line in result.stdout.splitlines() if line.startswith(marker))
    assert json.loads(line[len(marker):]) == expected


def test_changing_loader_behavior_flag_prevents_h01_resume(tmp_path, monkeypatch):
    monkeypatch.delenv('NEURD_LEGACY_MESH_LOADER', raising=False)
    output, argv = _plan_for_runtime_test(tmp_path, monkeypatch)
    manifest = json.loads((output/'manifest.json').read_text())
    assert manifest['runtime_environment']['environment']['NEURD_LEGACY_MESH_LOADER'] is None
    monkeypatch.setenv('NEURD_LEGACY_MESH_LOADER', '1')
    with pytest.raises(ValueError, match='Settings or code changed'):
        batch.main(argv)


def _width_worker_run(tmp_path, monkeypatch, width, *, faces=320, width_error=None,
                      sampler_error=None, broken_json=False):
    """Run the actual worker and parent's failure publication with controlled width."""
    from contextlib import redirect_stdout, redirect_stderr
    from types import SimpleNamespace
    import traceback
    import trimesh
    from neurd import mesh_io, parameters, spine_utils, width_utils, width_diagnostics, run_config

    source = tmp_path/'source'
    branch = source/'neuron_1/limb_000/branch_000'
    branch.mkdir(parents=True)
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=200)
    mesh = trimesh.Trimesh(mesh.vertices, mesh.faces[:faces], process=False)
    mesh.export(branch/'branch_mesh.off')
    original = segments([(0,0,0),(1000,0,0)], [(0,1)])
    effective, auxiliary = original + [0,200,0], original + [0,300,0]
    np.save(branch/'branch_skeleton.npy', original)
    source_before = {p.name: (batch.sha(p), p.stat().st_mtime_ns) for p in branch.iterdir()}
    config = tmp_path/'config.json'
    config.write_text('{}\n')
    output = tmp_path/'output'
    native = tmp_path/'fake_cgal.so'
    native.write_bytes(b'native guard fixture')
    monkeypatch.setitem(sys.modules, 'cgal_Segmentation_Module', SimpleNamespace(__file__=str(native)))
    monkeypatch.setattr(batch, 'dependency_signature', lambda: {'fixture': 'dependencies'})
    monkeypatch.setattr(run_config, 'runtime_environment', lambda *args, **kwargs: {})
    monkeypatch.setattr(parameters.params, 'use', lambda *args, **kwargs: None)
    qc = dict(status='repaired_with_original_connections', topological_replacement=True,
              before=dict(q95_nm=20000), after=dict(q95_nm=100))
    monkeypatch.setattr(batch, 'repair_skeleton', lambda *args: (effective, auxiliary, qc))
    state = dict(width_calls=[], spine_calls=[], sampler_calls=[], worker_errors=[])
    def calculate(branch, **kwargs):
        state['width_calls'].append((branch, kwargs))
        if width_error is not None:
            raise width_error
        return np.array([width, width]), width
    monkeypatch.setattr(width_utils, 'calculate_new_width', calculate)
    def spines(branch, **kwargs):
        state['spine_calls'].append(branch)
        kwargs['geometric_refinement_diagnostics'].update(candidates=[], counts={})
        return [], np.empty(0)
    monkeypatch.setattr(spine_utils, 'calculate_spines_on_branch', spines)
    def sample(mesh, axis):
        state['sampler_calls'].append(axis.copy())
        if sampler_error is not None:
            raise sampler_error
        return dict(status='sampled', diagnostic_only=True, sampled_edges=16,
                    full_edge_coverage=False, sampled_positive_raw_distance_count=0)
    monkeypatch.setattr(width_diagnostics, 'sample_width_support', sample)
    if broken_json:
        original_save = batch.save
        def failing_save(path, data):
            if Path(path).name in ('skeleton_repair.json', 'width_evidence.json', 'result.json'):
                raise TypeError('diagnostic serialization failed')
            original_save(path, data)
        monkeypatch.setattr(batch, 'save', failing_save)
    class Worker:
        pid = 999999999
        returncode = None
        def __init__(self, argv, **kwargs):
            self.output = Path(argv[argv.index('--output') + 1])
            self.key = argv[argv.index('--worker') + 1]
            self.staging = Path(argv[argv.index('--staging') + 1])
            self.log = kwargs['stdout']
        def wait(self, timeout=None):
            old_cwd, old_path = Path.cwd(), list(sys.path)
            try:
                with redirect_stdout(self.log), redirect_stderr(self.log):
                    try:
                        batch.branch_worker(self.output, self.key, self.staging)
                        self.returncode = 0
                    except Exception as exc:
                        state['worker_errors'].append(exc)
                        traceback.print_exc()
                        self.returncode = 1
            finally:
                os.chdir(old_cwd)
                sys.path[:] = old_path
            return self.returncode
    monkeypatch.setattr(batch.subprocess, 'Popen', Worker)
    argv = ['--source', str(source), '--output', str(output), '--config', str(config), '--workers', '1']
    failed = width_error is not None or (faces > 100 and (not np.isfinite(width) or width <= 0))
    if failed:
        with pytest.raises(RuntimeError, match='1 branches failed'):
            batch.main(argv)
    else:
        batch.main(argv)
    assert {p.name: (batch.sha(p), p.stat().st_mtime_ns) for p in branch.iterdir()} == source_before
    return output, state, original, effective, auxiliary, qc


@pytest.mark.parametrize('width,state_name', [(0.,'zero'), (-1.,'nonpositive'),
                                           (float('nan'),'nonfinite'), (float('inf'),'nonfinite')])
def test_unresolved_width_publishes_axes_and_failure_evidence(tmp_path, monkeypatch, width, state_name):
    output, state, original, effective, auxiliary, qc = _width_worker_run(tmp_path, monkeypatch, width)
    failures = list((output/'failures').iterdir())
    assert len(failures) == 1 and not list((output/'.work').iterdir())
    failed = failures[0]
    assert not (output/'dataset/neuron_1/limb_000/branch_000').exists()
    np.testing.assert_array_equal(np.load(failed/'branch_skeleton.npy'), effective)
    np.testing.assert_array_equal(np.load(failed/'branch_skeleton_original.npy'), original)
    np.testing.assert_array_equal(np.load(failed/'segmentation_axis.npy'), auxiliary)
    assert json.loads((failed/'skeleton_repair.json').read_text()) == qc
    evidence = json.loads((failed/'width_evidence.json').read_text())
    assert evidence['observed_width']['average_state'] == state_name
    assert evidence['observed_width']['values_source'] == 'exact_postprocessed_calculate_new_width_return'
    np.testing.assert_array_equal(np.load(failed/'width_distances.npy'), [width, width])
    assert evidence['branch_eligibility']['eligible'] is None
    assert evidence['branch_eligibility']['reason'] == 'eligibility_unresolved'
    assert evidence['axes']['width_eligibility']['sha256'] == batch.sha(failed/'branch_skeleton.npy')
    assert evidence['support_sampler']['diagnostic_only']
    assert not evidence['support_sampler']['full_edge_coverage']
    assert len(state['sampler_calls']) == 1 and not state['spine_calls']
    np.testing.assert_array_equal(state['sampler_calls'][0], effective)
    result = json.loads((failed/'result.json').read_text())
    assert result['status'] == 'failed' and result['reason'] == 'eligibility_unresolved'
    for relative, expected in result['outputs_sha256'].items():
        assert batch.sha(failed/relative) == expected
    with pytest.raises(ValueError):
        batch.complete_valid(failed, result['run_hash'], result['inputs'])
    summary = json.loads((output/'last_run_summary.json').read_text())
    assert summary['failed'] == 1 and summary['completed'] == 0


@pytest.mark.parametrize('width,eligible', [(114.9,False), (115.,False),
                                         (np.nextafter(115., np.inf),True)])
def test_existing_width_boundary_and_successful_path_remain_unchanged(tmp_path, monkeypatch, width, eligible):
    output, state, original, effective, _, _ = _width_worker_run(tmp_path, monkeypatch, width)
    dest = output/'dataset/neuron_1/limb_000/branch_000'
    result = json.loads((dest/'result.json').read_text())
    assert result['status'] == 'complete' and result['branch_eligibility']['eligible'] is eligible
    assert result['branch_eligibility']['median_mesh_center_nm'] == width
    assert len(state['width_calls']) == 1 and not state['sampler_calls']
    assert state['width_calls'][0][1] == dict(no_spines=False, summary_measure='median', return_average=True)
    np.testing.assert_array_equal(state['width_calls'][0][0].skeleton, effective)
    assert len(state['spine_calls']) == int(eligible)
    if eligible:
        np.testing.assert_array_equal(state['spine_calls'][0].skeleton, original)
    assert not (dest/'width_evidence.json').exists()
    assert not (dest/'skeleton_repair.json').exists()
    assert batch.complete_valid(dest, result['run_hash'], result['inputs'])


def test_original_width_error_survives_diagnostic_serialization_failure(tmp_path, monkeypatch):
    original_error = RuntimeError('original width calculation failed')
    output, state, _, effective, _, _ = _width_worker_run(
        tmp_path, monkeypatch, 0, width_error=original_error, broken_json=True)
    assert state['worker_errors'] == [original_error]
    failed, = (output/'failures').iterdir()
    np.testing.assert_array_equal(np.load(failed/'branch_skeleton.npy'), effective)
    assert not (failed/'result.json').exists()
    assert not batch.complete_valid(failed, 'version', {})


def test_diagnostic_sampler_failure_does_not_hide_unresolved_width(tmp_path, monkeypatch):
    output, state, _, _, _, _ = _width_worker_run(
        tmp_path, monkeypatch, 0, sampler_error=RuntimeError('support sampler failed'))
    assert len(state['worker_errors']) == 1
    assert str(state['worker_errors'][0]) == 'Cannot establish original NEURD branch-width eligibility'
    failed, = (output/'failures').iterdir()
    evidence = json.loads((failed/'width_evidence.json').read_text())
    assert evidence['support_sampler']['status'] == 'failed'
    assert any('support sampler failed' in item['error'] for item in evidence['diagnostic_errors'])


def test_face_minimum_still_bypasses_width_calculation(tmp_path, monkeypatch):
    output, state, _, _, _, _ = _width_worker_run(tmp_path, monkeypatch, 0, faces=100)
    result = json.loads((output/'dataset/neuron_1/limb_000/branch_000/result.json').read_text())
    assert result['branch_eligibility'] == dict(eligible=False, reason='neurd_minimum_branch_faces')
    assert not state['width_calls'] and not state['sampler_calls'] and not state['spine_calls']
