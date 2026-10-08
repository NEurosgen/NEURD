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
