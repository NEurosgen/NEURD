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
