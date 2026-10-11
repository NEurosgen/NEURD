"""Artifact-only proof of aligned reports without acceptance memoization."""
import ast
from pathlib import Path
from types import SimpleNamespace

import neurd
import numpy as np
import pytest
import trimesh

from tests.unit import skip_if_datasci_tools_unusable

skip_if_datasci_tools_unusable()
from neurd import (h01_batch, spine_refinement as refinement, spine_utils as spu,
                   spine_volume as volume)


def calculate(**overrides):
    """Execute the actual calculation with controlled leaf dependencies."""
    tree = ast.parse(Path(spu.__file__).read_text())
    calculation_names = {'calculate_spines_on_branch', '_refine_branch_spines',
                         '_apply_final_spine_filters', '_select_spine_results',
                         '_refresh_refinement_counts'}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)
             and n.name in calculation_names]
    env = dict(spu.__dict__)
    env.update(overrides)
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), '<actual volume calculation>', 'exec'), env)
    return env['calculate_spines_on_branch']


def options():
    return dict(spine_n_face_threshold=0, spine_sk_length_threshold=0,
                filter_by_bounding_box_longest_side_length=False,
                filter_out_border_spines=False, skeleton_endpoint_nullification=False,
                soma_vertex_nullification=False, calculate_spine_volume=False,
                filter_by_volume=False, filter_by_face_area_mean=False,
                filter_by_boundary_to_area_ratio=False)


def same_rng(actual, expected):
    assert all(np.array_equal(a, b) for a, b in zip(actual, expected))


def closed_box():
    return trimesh.creation.box(extents=[100, 200, 300])


def open_box():
    mesh = closed_box()
    mesh.update_faces(np.arange(len(mesh.faces)) != 0)
    return mesh


def fan(mesh):
    output = mesh.copy()
    trimesh.repair.fill_holes(output)
    return output


@pytest.mark.parametrize('kind', ['closed', 'fan', 'hull', 'flat', 'empty', 'none'])
def test_quality_keeps_exact_scalar_policy_and_reports(monkeypatch, kind):
    estimator = volume.estimate_mesh_volume
    mesh = closed_box()
    if kind in ('fan', 'hull'):
        mesh = open_box()
        closer = fan if kind == 'fan' else lambda mesh: trimesh.creation.box(extents=[1000]*3)
        monkeypatch.setattr(volume, 'estimate_mesh_volume', lambda mesh: estimator(mesh, close_holes=closer))
    elif kind == 'flat':
        mesh = trimesh.Trimesh(vertices=[[0,0,0],[1,0,0],[1,1,0],[0,1,0]],
                               faces=[[0,1,2],[0,2,3]], process=False)
    elif kind == 'empty':
        mesh = trimesh.Trimesh(vertices=np.empty((0,3)), faces=np.empty((0,3), int), process=False)
    elif kind == 'none':
        mesh = None
    state = np.random.get_state()
    expected_report = volume.estimate_mesh_volume(mesh)
    expected_scalar = volume.mesh_volume(mesh)
    scalar, record = volume.mesh_volume_with_quality(mesh)
    assert type(scalar) is type(expected_scalar) is float
    if expected_report['valid']:
        assert scalar == expected_scalar == expected_report['volume_nm3']
    else:
        assert np.isnan(scalar) and np.isnan(expected_scalar) and expected_report['volume_nm3'] is None
    assert volume.reuse_volume_quality(record, mesh) == expected_report
    assert record.mesh is mesh
    same_rng(np.random.get_state(), state)
    with pytest.raises(TypeError):
        record.quality['valid'] = False
    independent = volume.reuse_volume_quality(record, mesh)
    independent['valid'] = False
    assert record.quality['valid'] == expected_report['valid']


@pytest.mark.parametrize('exception', [ModuleNotFoundError, ImportError, RuntimeError])
def test_estimator_exception_is_not_converted_to_fallback(monkeypatch, exception):
    def fail(mesh):
        raise exception('controlled original estimator error')
    monkeypatch.setattr(volume, 'estimate_mesh_volume', fail)
    for estimate in (volume.mesh_volume, volume.mesh_volume_with_quality):
        with pytest.raises(exception, match='controlled original estimator error'):
            estimate(closed_box())


@pytest.mark.parametrize('change', ['vertices', 'faces', 'face_order', 'vertex_dtype', 'face_dtype', 'copy', 'missing'])
def test_reuse_rejects_stale_or_different_geometry(change):
    mesh = closed_box()
    _, record = volume.mesh_volume_with_quality(mesh)
    target = mesh
    if change == 'vertices':
        mesh.vertices[0,0] += 1
    elif change == 'faces':
        mesh.faces[0,0] = mesh.faces[0,1]
    elif change == 'face_order':
        mesh.faces = mesh.faces[::-1]
    elif change == 'vertex_dtype':
        target = SimpleNamespace(vertices=mesh.vertices.astype(np.float32), faces=mesh.faces)
        record = volume.VolumeQualityRecord(target, record.geometry_sha256, record.quality)
    elif change == 'face_dtype':
        target = SimpleNamespace(vertices=mesh.vertices, faces=mesh.faces.astype(np.int32))
        record = volume.VolumeQualityRecord(target, record.geometry_sha256, record.quality)
    elif change == 'copy':
        target = mesh.copy()
    else:
        record = None
    with pytest.raises(ValueError):
        volume.reuse_volume_quality(record, target)


def test_digest_is_bounded_and_storage_layout_independent(monkeypatch):
    mesh = closed_box()
    original = SimpleNamespace(vertices=np.tile(mesh.vertices, (2050,1)), faces=np.tile(mesh.faces,(700,1)))
    noncontiguous = SimpleNamespace(vertices=np.asfortranarray(original.vertices), faces=np.asfortranarray(original.faces))
    expected = volume.mesh_geometry_digest(original)
    copies = []
    contiguous = np.ascontiguousarray
    def track(array):
        copies.append(array.shape[0])
        assert array.shape[0] <= 8192
        return contiguous(array)
    monkeypatch.setattr(volume.np, 'ascontiguousarray', track)
    assert volume.mesh_geometry_digest(original) == expected and not copies
    assert volume.mesh_geometry_digest(noncontiguous) == expected
    assert copies and max(copies) == 8192


def test_default_and_explicit_none_do_not_collect_or_hash(monkeypatch):
    parent = closed_box()
    groups = [np.arange(6), np.arange(6,12)]
    raw = [parent.submesh([g], append=True) for g in groups]
    monkeypatch.setattr(spu, 'get_spine_meshes_unfiltered_from_mesh', lambda *a, **kw: (raw, groups))
    calls = []
    def scalar(mesh):
        calls.append(('scalar', len(mesh.faces))); np.random.random()
        return float(len(mesh.faces))
    def forbidden(mesh):
        pytest.fail('Collector=None invoked a quality record or geometry hash')
    outcomes = []
    for explicit_none in (False, True):
        calls.clear(); masks = []
        np.random.seed(45)
        kwargs = {"spine_volume_quality": None} if explicit_none else {}
        meshes, volumes = calculate(validated_spine_volume=scalar,
                                    mesh_volume_with_quality=forbidden)(
            SimpleNamespace(mesh=parent), geometric_refinement=False, spine_face_indices=masks,
            **{**options(), 'calculate_spine_volume': True}, **kwargs)
        outcomes.append(([id(m) for m in meshes], volumes.tolist(), [g.tolist() for g in masks],
                         list(calls), np.random.get_state()))
    assert outcomes[0][:4] == outcomes[1][:4]
    same_rng(outcomes[0][4], outcomes[1][4])


@pytest.mark.parametrize('with_volume', [False, True])
@pytest.mark.parametrize('morphology', [False, True])
def test_ordinary_filter_order_and_final_record_alignment(monkeypatch, with_volume, morphology):
    parent = closed_box()
    groups = [np.array([i]) for i in range(10)]
    raw = [parent.submesh([g], append=True) for g in groups]
    identities = {id(m):i for i,m in enumerate(raw)}
    monkeypatch.setattr(spu, 'get_spine_meshes_unfiltered_from_mesh', lambda *a, **kw: (raw,groups))
    calls = []
    def drop(name):
        def filter(meshes, *a, **kw):
            calls.append((name,[identities[id(m)] for m in meshes])); np.random.random()
            return meshes[1:]
        return filter
    monkeypatch.setattr(spu, 'filter_spine_meshes', drop('length'))
    monkeypatch.setattr(spu.tu, 'filter_meshes_by_bounding_box_longest_side', drop('bbox'))
    monkeypatch.setattr(spu, 'filter_out_border_spines', lambda parent,meshes,**kw:drop('border')(meshes))
    monkeypatch.setattr(spu.sk, 'find_skeleton_endpoint_coordinates', lambda _:np.zeros((1,3)))
    monkeypatch.setattr(spu.tu, 'filter_meshes_by_containing_coordinates', drop('endpoint'))
    monkeypatch.setattr(spu, 'filter_out_soma_touching_spines', drop('soma'))
    def report(mesh):
        index=identities[id(mesh)];calls.append(('volume',index));np.random.random()
        return dict(algorithm=volume.VOLUME_ALGORITHM, valid=index!=5, method='controlled',
                    volume_nm3=float(index+1) if index!=5 else None)
    monkeypatch.setattr(volume,'estimate_mesh_volume',report)
    def area(mesh):
        index=identities[id(mesh)];calls.append(('area',index));np.random.random()
        return 0 if index==6 else 1
    def boundary(mesh):
        index=identities[id(mesh)];calls.append(('boundary',index));np.random.random()
        return 0 if index==7 else 1
    monkeypatch.setattr(spu.tu,'face_area_mean',area,raising=False)
    monkeypatch.setattr(spu.tu,'area_to_boundary_length_ratio',boundary,raising=False)
    opts={**options(),'spine_n_face_threshold':1,'filter_by_bounding_box_longest_side_length':True,
        'filter_out_border_spines':True,'skeleton_endpoint_nullification':True,
        'soma_vertex_nullification':True,'soma_verts':np.zeros((1,3)),
        'calculate_spine_volume':with_volume,'filter_by_volume':with_volume,'filter_by_volume_threshold':.5,
        'filter_by_face_area_mean':morphology,'filter_by_face_area_mean_min':.5,
        'filter_by_boundary_to_area_ratio':morphology,'filter_by_boundary_to_area_ratio_min':.5}
    outcomes=[]
    for collect in (False,True):
        calls.clear();np.random.seed(47);masks=[];collector=['old']
        kwargs={'spine_volume_quality':collector} if collect else {}
        result=calculate(validated_spine_volume=volume.mesh_volume)(SimpleNamespace(mesh=parent,skeleton=np.zeros((1,2,3))),
            geometric_refinement=False,spine_face_indices=masks,**opts,**kwargs)
        meshes,volumes=result if with_volume else (result,None)
        outcomes.append(([identities[id(m)] for m in meshes],None if volumes is None else np.asarray(volumes).tolist(),
                         [g.tolist() for g in masks],list(calls),np.random.get_state()))
        if collect:
            assert len(collector)==len(meshes)
            if with_volume:
                for mesh,record,scalar in zip(meshes,collector,volumes):
                    assert record.mesh is mesh and record.quality['volume_nm3']==scalar
                    assert volume.reuse_volume_quality(record,mesh)==dict(record.quality)
            else:
                assert collector==[None]*len(meshes)
        else:
            assert collector==['old']
    assert outcomes[0][:4]==outcomes[1][:4]
    same_rng(outcomes[0][4],outcomes[1][4])


@pytest.mark.parametrize('mode', ['trim_reject','growth_reject','head_reject','head_accept','head_remove'])
def test_refinement_fallback_and_proposal_alignment_without_memoizing(monkeypatch,mode):
    parent=closed_box()
    original=np.arange(6 if mode=='growth_reject' else 12)
    raw=parent.submesh([original],append=True)
    monkeypatch.setattr(spu,'get_spine_meshes_unfiltered_from_mesh',lambda *a,**kw:([raw],[original]))
    calls=[]
    def refine(*a,**kw):
        if mode=='trim_reject':
            kw['pre_trim_face_groups'][0]=original
            mask,action,extra=original[:6],'trim_shaft_surface',{'shaft_trim':{'action':'trim'}}
        elif mode=='growth_reject':
            kw['pre_growth_face_groups'][0]=original
            mask,action,extra=np.arange(12),'grow_head',{'head_growth':{'action':'grow'}}
        else:
            mask,action,extra=original,'keep',{}
        return [mask],dict(candidates=[dict(input_index=0,output_index=0,output_indices=[0],
            output_faces=len(mask),action=action,**extra)],counts={action:1},seconds=0)
    monkeypatch.setattr(refinement,'refine_spine_face_indices',refine)
    def heads(parent,skeleton,groups,raw_groups,owners,config,validate_proposal):
        proposal=original[:6]
        accepted=validate_proposal(0,proposal)
        assert accepted==(mode=='head_accept')
        if mode=='head_accept':
            return [proposal],dict(candidates=[dict(input_index=0,output_index=0,action='grow_remaining_head')],
                seconds=0,validation_seconds=0)
        if mode=='head_remove':
            return [],dict(candidates=[dict(input_index=0,output_index=None,action='remove_incomplete_head')],
                seconds=0,validation_seconds=0)
        return groups,dict(candidates=[dict(input_index=0,output_index=0,action='keep')],seconds=0,validation_seconds=0)
    monkeypatch.setattr(refinement,'refine_remaining_spine_heads',heads)
    def report(mesh):
        faces=len(mesh.faces);calls.append(('volume',faces));np.random.random()
        valid=faces==len(original) or mode=='head_accept'
        return dict(algorithm=volume.VOLUME_ALGORITHM,valid=valid,volume_nm3=float(faces) if valid else None,
                    method='controlled', input_watertight=False)
    monkeypatch.setattr(volume,'estimate_mesh_volume',report)
    outcomes=[]
    for collect in (False,True):
        calls.clear();np.random.seed(51);masks=[];diagnostics={};collector=['old']
        kwargs={'spine_volume_quality':collector} if collect else {}
        meshes,volumes=calculate(validated_spine_volume=volume.mesh_volume)(
            SimpleNamespace(mesh=parent,skeleton=np.zeros((1,2,3))),geometric_refinement=True,
            geometric_refinement_config={'refine_remaining_heads':mode.startswith('head')},
            geometric_refinement_diagnostics=diagnostics,spine_face_indices=masks,
            **{**options(),'calculate_spine_volume':True,'filter_by_volume':True,'filter_by_volume_threshold':.5},**kwargs)
        outcomes.append(([m.triangles.tolist() for m in meshes],np.asarray(volumes).tolist(),
                         [g.tolist() for g in masks],diagnostics,list(calls),np.random.get_state()))
        if collect:
            assert len(collector)==len(meshes)
            for mesh,record,scalar in zip(meshes,collector,volumes):
                assert record.mesh is mesh and record.quality['volume_nm3']==scalar
                assert volume.reuse_volume_quality(record,mesh)==dict(record.quality)
    assert outcomes[0][:5]==outcomes[1][:5]
    same_rng(outcomes[0][5],outcomes[1][5])
    assert len(outcomes[0][4])==2 # Every original fallback/proposal validation still estimates once.


def test_collector_is_transactional_when_estimator_raises(monkeypatch):
    parent=closed_box();groups=[np.arange(6),np.arange(6,12)]
    raw=[parent.submesh([g],append=True) for g in groups]
    monkeypatch.setattr(spu,'get_spine_meshes_unfiltered_from_mesh',lambda *a,**kw:(raw,groups) if kw['return_mesh_idx'] else raw)
    calls=[]
    def fail_second(mesh):
        calls.append(mesh)
        if len(calls)==2:
            raise ModuleNotFoundError('controlled closure dependency')
        return dict(algorithm=volume.VOLUME_ALGORITHM,valid=True,volume_nm3=1.,method='controlled')
    monkeypatch.setattr(volume,'estimate_mesh_volume',fail_second)
    collector=['prior successful handoff']
    with pytest.raises(ModuleNotFoundError,match='controlled closure dependency'):
        calculate()(SimpleNamespace(mesh=parent),geometric_refinement=False,spine_volume_quality=collector,
            **{**options(),'calculate_spine_volume':True})
    assert collector==['prior successful handoff'] and len(calls)==2


def test_empty_detection_clears_collector_on_success(monkeypatch):
    parent=closed_box()
    monkeypatch.setattr(spu,'get_spine_meshes_unfiltered_from_mesh',lambda *a,**kw:([],[]) if kw['return_mesh_idx'] else [])
    collector=['old']
    meshes,volumes=calculate()(SimpleNamespace(mesh=parent),geometric_refinement=False,spine_volume_quality=collector,
                              **{**options(),'calculate_spine_volume':True})
    assert meshes==[] and len(volumes)==0 and collector==[]


def test_invalid_nan_scalar_survives_when_volume_filter_disabled(monkeypatch):
    parent=closed_box();raw=parent.submesh([np.array([0])],append=True)
    monkeypatch.setattr(spu,'get_spine_meshes_unfiltered_from_mesh',lambda *a,**kw:([raw],[np.array([0])]) if kw['return_mesh_idx'] else [raw])
    collector=[]
    meshes,volumes=calculate()(SimpleNamespace(mesh=parent),geometric_refinement=False,spine_volume_quality=collector,
        **{**options(),'calculate_spine_volume':True})
    assert meshes[0] is raw and np.isnan(volumes[0])
    assert collector[0].quality['valid'] is False and collector[0].quality['volume_nm3'] is None


def publication():
    tree=ast.parse(Path(h01_batch.__file__).read_text())
    worker=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='branch_worker')
    start=next(i for i,n in enumerate(worker.body) if isinstance(n,ast.With)
        and n.items[0].context_expr.args[0].value=='h01_surface_provenance_check')
    return compile(ast.fix_missing_locations(ast.Module(body=worker.body[start:],type_ignores=[])),
                   '<actual H01 publication>','exec')


@pytest.mark.parametrize('mutation', [None,'vertices','export_mutation','record_identity','records_length'])
def test_publisher_keeps_output_qc_and_rejects_stale_handoffs(monkeypatch,tmp_path,mutation):
    parent=closed_box();mesh=parent.copy();mask=np.arange(len(mesh.faces))
    scalar,record=volume.mesh_volume_with_quality(mesh)
    source=tmp_path/'source.off';parent.export(source)
    provider=tmp_path/'provider.so';provider.write_bytes(b'publication-only fixture')
    source_inputs={};original=np.zeros((1,2,3));calls=[]
    quality=volume.estimate_mesh_volume
    def observe_quality(mesh):
        calls.append('publication estimate')
        return quality(mesh)
    monkeypatch.setattr(spu,'estimate_mesh_volume',observe_quality)
    if mutation=='vertices':
        mesh.vertices[0,0]+=1
    if mutation=='export_mutation':
        export=mesh.export
        def mutate_after_export(*a,**kw):
            result=export(*a,**kw);mesh.vertices[0,0]+=1;return result
        monkeypatch.setattr(mesh,'export',mutate_after_export)
    if mutation=='record_identity':
        record=volume.mesh_volume_with_quality(mesh.copy())[1]
    results=[]
    for candidate in ((False,True) if mutation is None else (True,)):
        staging=tmp_path/('candidate' if candidate else 'baseline');staging.mkdir()
        env=dict(np=np,cgal_phase=spu.cgal_phase,meshes=[mesh],masks=[mask],volumes=np.array([scalar]),
            volume_quality=[] if mutation=='records_length' else [record],
            reuse_volume_quality=volume.reuse_volume_quality if candidate else lambda record,mesh:spu.estimate_mesh_volume(mesh),
            parent=parent,diagnostics={'candidates':[{'raw_input_index':0,'action':'keep',
                'final_output_mapping':[{'final_output_index':0,'part_index':0}]}]},
            triangle_keys=h01_batch.triangle_keys,staging=staging,spu=spu,skeleton=original,original=original,
            auxiliary=None,row={'source':str(source),'archived_spines':1},Path=Path,save=h01_batch.save,
            inputs=lambda _:source_inputs,source_inputs=source_inputs,shutil=h01_batch.shutil,
            key='test',manifest={'run_hash':'test','skeleton':{'use_auxiliary_for_heads':False}},
            skeleton_qc={},elapsed=1,eligibility={'eligible':True},csm=SimpleNamespace(__file__=str(provider)),sha=h01_batch.sha)
        if mutation is not None:
            with pytest.raises(ValueError):exec(publication(),env)
            assert not (staging/'result.json').exists()
        else:
            exec(publication(),env)
            results.append((staging/'result.json').read_bytes())
            reloaded=trimesh.load(staging/'spines/spine_00000.off',process=False,force='mesh')
            assert volume.estimate_mesh_volume(reloaded)==dict(record.quality) # Independent OFF QC stays independent.
    if mutation is None:
        assert results[0]==results[1]
        assert calls==['publication estimate'] # Candidate publication reused exactly one accepted record.


@pytest.mark.parametrize('mismatch', ['scope', 'algorithm'])
def test_reuse_rejects_wrong_report_scope_or_algorithm(mismatch):
    mesh=closed_box();_,record=volume.mesh_volume_with_quality(mesh)
    if mismatch=='scope':
        record=volume.VolumeQualityRecord(mesh,record.geometry_sha256,record.quality,'rounded_off')
    else:
        report=dict(record.quality,algorithm='other_algorithm')
        record=volume.VolumeQualityRecord(mesh,record.geometry_sha256,report)
    with pytest.raises(ValueError):volume.reuse_volume_quality(record,mesh)


def test_whole_calculation_uses_exact_actual_quality_reports(monkeypatch):
    raw=[closed_box(),open_box(),trimesh.Trimesh(vertices=[[0,0,0],[1,0,0],[1,1,0],[0,1,0]],
            faces=[[0,1,2],[0,2,3]],process=False)]
    parent=trimesh.util.concatenate(raw)
    monkeypatch.setattr(spu,'get_spine_meshes_unfiltered_from_mesh',lambda *a,**kw:raw)
    collector=[]
    kept,values=calculate()(SimpleNamespace(mesh=parent),geometric_refinement=False,spine_volume_quality=collector,
                            **{**options(),'calculate_spine_volume':True})
    assert len(kept)==len(values)==len(collector)==3
    for mesh,value,record in zip(kept,values,collector):
        fresh=volume.estimate_mesh_volume(mesh)
        assert record.mesh is mesh and volume.reuse_volume_quality(record,mesh)==fresh
        if fresh['valid']:assert value==fresh['volume_nm3']
        else:assert np.isnan(value) and fresh['volume_nm3'] is None
