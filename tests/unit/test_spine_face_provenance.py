"""Actual parent face occurrences survive both spine processing modes."""
from types import SimpleNamespace

import neurd
import numpy as np
import pytest
import trimesh

from tests.unit import skip_if_datasci_tools_unusable

skip_if_datasci_tools_unusable()
from neurd import neuron, spine_refinement as refinement, spine_utils as spu


def options():
    return dict(spine_n_face_threshold=0, spine_sk_length_threshold=0,
                filter_by_bounding_box_longest_side_length=False,
                filter_out_border_spines=False, skeleton_endpoint_nullification=False,
                soma_vertex_nullification=False, calculate_spine_volume=False,
                filter_by_volume=False, filter_by_face_area_mean=False,
                filter_by_boundary_to_area_ratio=False)


def extraction(monkeypatch, meshes, masks):
    def extract(*args, **kwargs):
        assert kwargs["return_mesh_idx"] is True
        return meshes, masks

    monkeypatch.setattr(spu, "get_spine_meshes_unfiltered_from_mesh", extract)


def test_duplicate_triangles_keep_distinct_actual_parent_faces(monkeypatch):
    parent = trimesh.creation.box()
    parent.faces = np.vstack([parent.faces, parent.faces[:1]])
    raw_masks = [np.array([0]), np.array([12])]
    raw_meshes = [parent.submesh([mask], append=True) for mask in raw_masks]
    assert np.array_equal(raw_meshes[0].triangles, raw_meshes[1].triangles)
    extraction(monkeypatch, raw_meshes, raw_masks)
    masks = []
    result = spu.calculate_spines_on_branch(SimpleNamespace(mesh=parent),
                                            spine_face_indices=masks, geometric_refinement=False,
                                            **options())
    assert result[0] is raw_meshes[0] and result[1] is raw_meshes[1]
    assert [mask.tolist() for mask in masks] == [[0], [12]]
    assert all(mask.dtype == np.int64 for mask in masks)
    masks[0][0] = 7
    assert raw_masks[0][0] == 0  # Caller mutation cannot alter source ownership.


def test_all_ordinary_filters_preserve_final_mask_and_volume_alignment(monkeypatch):
    parent = trimesh.creation.box()
    raw_masks = [np.array([i]) for i in range(10)]
    raw_meshes = [parent.submesh([mask], append=True) for mask in raw_masks]
    raw_ids = {id(mesh): i for i, mesh in enumerate(raw_meshes)}
    extraction(monkeypatch, raw_meshes, raw_masks)
    events = []

    def drop_first(label):
        def filter_meshes(meshes, *args, **kwargs):
            events.append(label)
            return meshes[1:]
        return filter_meshes

    monkeypatch.setattr(spu, "filter_spine_meshes", drop_first("length"))
    monkeypatch.setattr(spu.tu, "filter_meshes_by_bounding_box_longest_side", drop_first("bbox"))
    monkeypatch.setattr(spu, "filter_out_border_spines",
                        lambda parent, meshes, **kw: drop_first("border")(meshes))
    monkeypatch.setattr(spu.sk, "find_skeleton_endpoint_coordinates", lambda _: np.zeros((1, 3)))
    monkeypatch.setattr(spu.tu, "filter_meshes_by_containing_coordinates", drop_first("endpoint"))
    monkeypatch.setattr(spu, "filter_out_soma_touching_spines", drop_first("soma"))
    monkeypatch.setattr(spu, "validated_spine_volume",
                        lambda mesh: 0 if raw_ids[id(mesh)] == 5 else raw_ids[id(mesh)] + 1)
    monkeypatch.setattr(spu.tu, "face_area_mean", lambda mesh: 0 if raw_ids[id(mesh)] == 6 else 1,
                        raising=False)
    monkeypatch.setattr(spu.tu, "area_to_boundary_length_ratio",
                        lambda mesh: 0 if raw_ids[id(mesh)] == 7 else 1, raising=False)
    masks = []
    meshes, volumes = spu.calculate_spines_on_branch(
        SimpleNamespace(mesh=parent, skeleton=np.zeros((1, 2, 3))),
        geometric_refinement=False, spine_face_indices=masks,
        **{**options(), "spine_n_face_threshold": 1,
           "filter_by_bounding_box_longest_side_length": True,
           "filter_out_border_spines": True, "skeleton_endpoint_nullification": True,
           "soma_vertex_nullification": True, "soma_verts": np.zeros((1, 3)),
           "calculate_spine_volume": True, "filter_by_volume": True,
           "filter_by_volume_threshold": .5, "filter_by_face_area_mean": True,
           "filter_by_face_area_mean_min": .5, "filter_by_boundary_to_area_ratio": True,
           "filter_by_boundary_to_area_ratio_min": .5})
    assert events == ["length", "bbox", "border", "endpoint", "soma"]
    assert [raw_ids[id(mesh)] for mesh in meshes] == [8, 9]
    assert [mask.tolist() for mask in masks] == [[8], [9]]
    assert list(volumes) == [9, 10]


@pytest.mark.parametrize("enabled", [False, True])
def test_empty_detection_clears_requested_masks(monkeypatch, enabled):
    parent = trimesh.creation.box()
    extraction(monkeypatch, [], [])
    monkeypatch.setattr(refinement, "refine_spine_face_indices", lambda *a, **kw:
                        ([], dict(candidates=[], counts={}, seconds=0)))
    masks = [np.array([99])]
    legacy = [np.array([99])] if enabled else None
    result = spu.calculate_spines_on_branch(
        SimpleNamespace(mesh=parent, skeleton=np.zeros((1, 2, 3))),
        geometric_refinement=enabled, spine_face_indices=masks,
        geometric_refinement_face_indices=legacy, **options())
    assert result == [] and masks == []
    if enabled:
        assert legacy == []


def test_refined_generic_masks_match_legacy_masks_after_split(monkeypatch):
    parent = trimesh.creation.box()
    original = np.arange(len(parent.faces))
    extraction(monkeypatch, [parent], [original])

    def split(*args, **kwargs):
        kwargs["pre_trim_face_groups"][0] = original
        return [original[:6], original[6:]], dict(candidates=[dict(
            input_index=0, output_index=0, output_indices=[0, 1],
            action="split_shaft_connection", shaft_trim={"action": "split"})],
            counts={}, seconds=0)

    monkeypatch.setattr(refinement, "refine_spine_face_indices", split)
    generic, legacy = [], []
    meshes = spu.calculate_spines_on_branch(
        SimpleNamespace(mesh=parent, skeleton=np.zeros((1, 2, 3))),
        geometric_refinement=True, spine_face_indices=generic,
        geometric_refinement_face_indices=legacy, **options())
    assert len(meshes) == len(generic) == len(legacy) == 2
    assert all(np.array_equal(a, b) for a, b in zip(generic, legacy))
    assert [mask.tolist() for mask in generic] == [list(range(6)), list(range(6, 12))]
    generic[0][0] = 100
    assert legacy[0][0] == 0


def test_legacy_mask_argument_still_requires_refinement(monkeypatch):
    parent = trimesh.creation.box()
    monkeypatch.setattr(spu, "get_spine_meshes_unfiltered_from_mesh", lambda *a, **kw: [])
    with pytest.raises(ValueError, match="geometric_refinement=True"):
        spu.calculate_spines_on_branch(SimpleNamespace(mesh=parent),
                                      geometric_refinement=False,
                                      geometric_refinement_face_indices=[], **options())


@pytest.mark.parametrize("with_volumes", [False, True])
def test_neuron_stores_masks_and_distinguishes_unselected_and_empty(monkeypatch, with_volumes):
    parent = trimesh.creation.box()
    branches = [SimpleNamespace(mesh=parent, spines=None, spines_face_idx=None)
                for _ in range(4)]
    selected_mesh = parent.submesh([[3]], append=True)

    def branch_spines(branch, *, spine_face_indices, calculate_spine_volume, **kwargs):
        assert calculate_spine_volume is with_volumes
        if branch is branches[0]:
            spine_face_indices[:] = [np.array([3], dtype=np.int64)]
            meshes, volumes = [selected_mesh], [1.0]
        else:
            spine_face_indices[:] = []
            meshes, volumes = [], []
        return (meshes, volumes) if calculate_spine_volume else meshes

    monkeypatch.setattr(spu, "calculate_spines_on_branch", branch_spines)
    spu.calculate_spines_on_neuron(
        {0: branches}, limb_branch_dict={0: [0, 1, 2]}, limb_branch_dict_exclude={0: [2]},
        soma_vertex_nullification=False, calculate_spine_volume=with_volumes)
    assert branches[0].spines == [selected_mesh]
    assert branches[0].spines_face_idx[0].tolist() == [3]
    assert branches[1].spines == [] and branches[1].spines_face_idx == []
    assert branches[2].spines is None and branches[2].spines_face_idx is None  # Excluded.
    assert branches[3].spines is None and branches[3].spines_face_idx is None  # Unselected.


def test_new_branch_has_unavailable_spine_provenance():
    branch = neuron.Branch(skeleton=np.array([[[0., 0., 0.], [1., 0., 0.]]]))
    assert branch.spines is None and branch.spines_face_idx is None
