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
