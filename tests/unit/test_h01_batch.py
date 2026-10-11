"""Protect source ownership, resumability and skeleton graph connections."""
import importlib.util
import json
from pathlib import Path

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
