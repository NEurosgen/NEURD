"""The opt-in stage must preserve the disabled branch API and filtering order."""
from types import SimpleNamespace

import neurd  # Apply compatibility patches before importing mesh_tools.
import numpy as np
import pytest
import trimesh

from tests.unit import skip_if_datasci_tools_unusable

skip_if_datasci_tools_unusable()
from neurd import parameters, spine_utils as spu
from neurd import spine_refinement as sr


def options():
    return dict(spine_n_face_threshold=0, spine_sk_length_threshold=0,
                filter_by_bounding_box_longest_side_length=False,
                filter_out_border_spines=False, skeleton_endpoint_nullification=False,
                soma_vertex_nullification=False, calculate_spine_volume=False,
                filter_by_volume=False, filter_by_face_area_mean=False,
                filter_by_boundary_to_area_ratio=False)


def test_default_does_not_call_geometry(monkeypatch):
    mesh = trimesh.creation.box()
    calls = []

    def extract(*args, **kwargs):
        calls.append(kwargs)
        return [mesh]

    def forbidden(*args, **kwargs):
        pytest.fail("Disabled refinement performed geometric work")

    monkeypatch.setattr(spu, "get_spine_meshes_unfiltered_from_mesh", extract)
    monkeypatch.setattr(sr, "refine_spine_face_indices", forbidden)
    result = spu.calculate_spines_on_branch(SimpleNamespace(mesh=mesh), **options())
    assert result[0] is mesh
    assert calls[0]["return_mesh_idx"] is False
    assert parameters.params.spine_geometric_refinement is False


def test_refinement_runs_after_length_and_before_volume(monkeypatch):
    mesh = trimesh.creation.box()
    indices = np.arange(len(mesh.faces))
    events = []

    def extract(*args, **kwargs):
        assert kwargs["return_mesh_idx"] is True
        return [mesh], [indices]

    def length_filter(meshes, **kwargs):
        events.append("length")
        return meshes

    def refine(parent, skeleton, groups, **kwargs):
        events.append("refine")
        assert np.array_equal(groups[0], indices)
        assert np.array_equal(kwargs["protected_face_groups"][0], indices)
        return groups, dict(candidates=[dict(input_index=0, output_index=0, action="keep")],
                            counts={"keep": 1}, seconds=0)

    def volume(mesh, **kwargs):
        events.append("volume")
        return 1

    monkeypatch.setattr(spu, "get_spine_meshes_unfiltered_from_mesh", extract)
    monkeypatch.setattr(spu, "filter_spine_meshes", length_filter)
    monkeypatch.setattr(sr, "refine_spine_face_indices", refine)
    monkeypatch.setattr(spu.tu, "mesh_volume", volume)
    diagnostics = {}
    opts = {**options(), "spine_n_face_threshold": 1, "calculate_spine_volume": True}
    meshes, _ = spu.calculate_spines_on_branch(
        SimpleNamespace(mesh=mesh, skeleton=np.zeros((1, 2, 3))),
        geometric_refinement=True, geometric_refinement_diagnostics=diagnostics, **opts)
    assert meshes[0] is mesh
    assert events == ["length", "refine", "volume"]
    assert diagnostics["candidates"][0]["final_output_index"] == 0


def test_split_parts_repeat_length_filter_and_restore_a_rejected_part(monkeypatch):
    parent=trimesh.creation.box()
    original=np.arange(len(parent.faces))
    events=[]

    def extract(*args,**kwargs):
        return [parent],[original]

    def length(meshes,**kwargs):
        events.append(('length',len(meshes)))
        return meshes if len(meshes)==1 else meshes[1:]

    def refine(*args,**kwargs):
        events.append(('refine',1))
        kwargs['pre_trim_face_groups'][0]=original
        return [original[:6],original[6:]],dict(candidates=[dict(input_index=0,output_index=0,
                   output_indices=[0,1],output_faces=len(original),action='split_shaft_connection',shaft_trim={'action':'split'})],
                   counts={'split_shaft_connection':1},seconds=0)

    def volume(mesh,**kwargs):
        events.append(('volume',len(mesh.faces)))
        return 1

    monkeypatch.setattr(spu,'get_spine_meshes_unfiltered_from_mesh',extract)
    monkeypatch.setattr(spu,'filter_spine_meshes',length)
    monkeypatch.setattr(sr,'refine_spine_face_indices',refine)
    monkeypatch.setattr(spu.tu,'mesh_volume',volume)
    d={};opts={**options(),'spine_n_face_threshold':1,'calculate_spine_volume':True}
    result,_=spu.calculate_spines_on_branch(SimpleNamespace(mesh=parent,skeleton=np.zeros((1,2,3))),
                          geometric_refinement=True,geometric_refinement_diagnostics=d,**opts)
    assert len(result)==1 and len(result[0].faces)==12
    assert events==[('length',1),('refine',1),('length',2),('volume',6),('volume',12)]
    assert d['candidates'][0]['raw_input_index']==0
    assert d['candidates'][0]['final_output_index']==0
    assert d['candidates'][0]['final_output_indices']==[0]
    assert d['candidates'][0]['final_output_mapping']==[
        dict(output_index=2,part_index=0,final_output_index=0)]
    assert d['candidates'][0]['proposed_output_indices']==[0,1]
    assert d['candidates'][0]['shaft_trim']['reason']=='split_part_rejected_by_final_filters'
    assert d['counts']=={'keep_after_split_filter_rejection':1}
    assert d['statistics']['splits_reverted_by_final_filters']==1


def test_rejected_single_trim_restores_original_through_final_filters(monkeypatch):
    parent = trimesh.creation.box()
    original = np.arange(len(parent.faces))
    volumes = []

    def extract(*args, **kwargs):
        return [parent], [original]

    def refine(*args, **kwargs):
        kwargs['pre_trim_face_groups'][0] = original
        return [original[:6]], dict(candidates=[dict(
            input_index=0, output_index=0, output_indices=[0], output_faces=6,
            action='trim_shaft_surface', shaft_trim={'action': 'trim'})],
            counts={'trim_shaft_surface': 1}, seconds=0)

    def volume(mesh, **kwargs):
        volumes.append(len(mesh.faces))
        return 1 if len(mesh.faces) == 12 else 0

    monkeypatch.setattr(spu, 'get_spine_meshes_unfiltered_from_mesh', extract)
    monkeypatch.setattr(sr, 'refine_spine_face_indices', refine)
    monkeypatch.setattr(spu.tu, 'mesh_volume', volume)
    diagnostics = {}
    result, _ = spu.calculate_spines_on_branch(
        SimpleNamespace(mesh=parent, skeleton=np.zeros((1, 2, 3))),
        geometric_refinement=True, geometric_refinement_diagnostics=diagnostics,
        **{**options(), 'calculate_spine_volume': True,
           'filter_by_volume': True, 'filter_by_volume_threshold': .5})
    assert len(result) == 1 and len(result[0].faces) == 12
    assert volumes == [6, 12]
    record = diagnostics['candidates'][0]
    assert record['action'] == 'keep_after_trim_filter_rejection'
    assert record['final_output_indices'] == [0]
    assert record['shaft_trim']['proposed_action'] == 'trim'
    assert diagnostics['statistics']['trims_reverted_by_final_filters'] == 1


def test_sheet_rejection_after_growth_is_not_restored(monkeypatch):
    parent = trimesh.creation.box()
    original = np.arange(len(parent.faces))

    def extract(*args, **kwargs):
        return [parent], [original]

    def refine(*args, **kwargs):
        kwargs['pre_growth_face_groups'][0] = original
        return [], dict(candidates=[dict(
            input_index=0, output_index=None, output_indices=[], output_faces=0,
            action='remove_low_relief_sheet')], counts={'remove_low_relief_sheet': 1}, seconds=0)

    monkeypatch.setattr(spu, 'get_spine_meshes_unfiltered_from_mesh', extract)
    monkeypatch.setattr(sr, 'refine_spine_face_indices', refine)
    diagnostics = {}
    result = spu.calculate_spines_on_branch(
        SimpleNamespace(mesh=parent, skeleton=np.zeros((1, 2, 3))),
        geometric_refinement=True, geometric_refinement_diagnostics=diagnostics, **options())
    assert result == []
    assert diagnostics['candidates'][0]['final_output_indices'] == []
    assert diagnostics['counts'] == {'remove_low_relief_sheet': 1}


def test_successful_split_maps_all_parts_without_fallback(monkeypatch):
    parent=trimesh.creation.box();ids=np.arange(len(parent.faces))

    def extract(*a,**kw):return [parent],[ids]

    def refine(*a,**kw):
        kw['pre_trim_face_groups'][0]=ids
        return [ids[:6],ids[6:]],dict(candidates=[dict(input_index=0,output_index=0,output_indices=[0,1],
                      action='split_shaft_connection',shaft_trim={'action':'split'})],counts={},seconds=0)

    monkeypatch.setattr(spu,'get_spine_meshes_unfiltered_from_mesh',extract)
    monkeypatch.setattr(sr,'refine_spine_face_indices',refine)
    d={};result=spu.calculate_spines_on_branch(SimpleNamespace(mesh=parent,skeleton=np.zeros((1,2,3))),
                 geometric_refinement=True,geometric_refinement_diagnostics=d,**options())
    assert len(result)==2
    assert d['candidates'][0]['final_output_indices']==[0,1]
    assert d['candidates'][0]['final_output_mapping']==[
        dict(output_index=0,part_index=0,final_output_index=0),
        dict(output_index=1,part_index=1,final_output_index=1)]


def test_rejected_head_completion_restores_seed_through_final_filters(monkeypatch):
    parent = trimesh.creation.box()
    original = np.arange(6)
    seed = parent.submesh([original], append=True)
    volume_inputs = []

    def extract(*args, **kwargs):
        return [seed], [original]

    def refine(*args, **kwargs):
        kwargs['pre_growth_face_groups'][0] = original
        return [np.arange(12)], dict(candidates=[dict(
            input_index=0, output_index=0, output_indices=[0],
            action='grow_ragged_head', head_growth={'action': 'grow'},
            added_faces=6, removed_faces=0)], counts={}, seconds=0)

    def volume(mesh, **kwargs):
        volume_inputs.append(len(mesh.faces))
        return 1 if len(mesh.faces) == 6 else 0

    monkeypatch.setattr(spu, 'get_spine_meshes_unfiltered_from_mesh', extract)
    monkeypatch.setattr(sr, 'refine_spine_face_indices', refine)
    monkeypatch.setattr(spu.tu, 'mesh_volume', volume)
    diagnostics = {}
    opts = {**options(), 'calculate_spine_volume': True,
            'filter_by_volume': True, 'filter_by_volume_threshold': .5}
    meshes, volumes = spu.calculate_spines_on_branch(
        SimpleNamespace(mesh=parent, skeleton=np.zeros((1, 2, 3))),
        geometric_refinement=True, geometric_refinement_diagnostics=diagnostics, **opts)
    assert volume_inputs == [12, 6]
    assert len(meshes) == 1 and len(meshes[0].faces) == 6
    assert np.array_equal(meshes[0].triangles, seed.triangles)
    assert volumes.tolist() == [1]
    record = diagnostics['candidates'][0]
    assert record['head_growth']['reason'] == 'grown_head_rejected_by_final_filters'
    assert record['added_faces'] == record['removed_faces'] == 0
    assert record['final_output_mapping'] == [dict(output_index=1, part_index=0, final_output_index=0)]
    assert diagnostics['statistics']['head_growth_reverted_by_final_filters'] == 1
