"""Retained raw meshes keep their provided face order through refinement."""
from types import SimpleNamespace

import neurd
import numpy as np
import pytest
import trimesh

from tests.unit import skip_if_datasci_tools_unusable

skip_if_datasci_tools_unusable()
from neurd import spine_refinement as refinement, spine_utils as spu
from neurd.face_provenance import validate_ordered_face_indices


def options():
    return dict(spine_n_face_threshold=0, spine_sk_length_threshold=0,
                filter_by_bounding_box_longest_side_length=False,
                filter_out_border_spines=False, skeleton_endpoint_nullification=False,
                soma_vertex_nullification=False, calculate_spine_volume=True,
                filter_by_volume=False, filter_by_face_area_mean=False,
                filter_by_boundary_to_area_ratio=False)


def fixture_mesh(duplicate):
    parent = trimesh.creation.box()
    if duplicate:
        parent.faces = np.vstack([parent.faces, parent.faces[0]])
    # Nonmonotonic order models CGAL groups joined in graph component order.
    indices = np.roll(np.arange(len(parent.faces)), 3)
    raw = parent.submesh([indices], append=True)
    validate_ordered_face_indices(parent, raw, indices)
    return parent, indices, raw


def setup(monkeypatch, parent, original, raw, primary='keep', heads='disabled'):
    events = []
    def extract(*args, **kwargs):
        assert kwargs['return_mesh_idx'] is True
        return [raw], [original]
    monkeypatch.setattr(spu, 'get_spine_meshes_unfiltered_from_mesh', extract)
    def refine(parent, skeleton, groups, **kwargs):
        events.append('primary'); np.random.random()
        assert np.array_equal(groups[0], original)
        canonical = np.unique(groups[0])
        if primary == 'keep':
            masks, extra = [canonical], {}
        elif primary == 'trim':
            masks = [canonical[1:]]
            kwargs['pre_trim_face_groups'][0] = canonical
            extra = {'shaft_trim': {'action': 'trim'}}
        else:
            masks = [canonical[:6], canonical[6:]]
            kwargs['pre_trim_face_groups'][0] = canonical
            extra = {'shaft_trim': {'action': 'split'}}
        outputs = list(range(len(masks)))
        return masks, dict(candidates=[dict(input_index=0, output_index=0, output_indices=outputs,
            output_faces=sum(len(mask) for mask in masks), action=primary, **extra)], counts={primary: 1}, seconds=0)
    monkeypatch.setattr(refinement, 'refine_spine_face_indices', refine)
    def remaining(parent, skeleton, groups, raw_groups, owners, config, validate_proposal):
        events.append('remaining')
        # The actual remaining-head routine sorts masks upon entry. Retaining
        # its current mesh must still retain the provided row map at the API.
        canonical = [np.unique(group) for group in groups]
        if heads == 'grow':
            proposal = canonical[0][1:]
            assert validate_proposal(0, proposal)
            return [proposal], dict(candidates=[dict(input_index=0, output_index=0,
                action='grow_remaining_head')], seconds=0, validation_seconds=0)
        if heads == 'remove':
            return [], dict(candidates=[dict(input_index=0, output_index=None,
                action='remove_incomplete_head')], seconds=0, validation_seconds=0)
        return canonical, dict(candidates=[dict(input_index=0, output_index=0,
            action='keep')], seconds=0, validation_seconds=0)
    monkeypatch.setattr(refinement, 'refine_remaining_spine_heads', remaining)
    def volume(mesh):
        events.append(('volume', len(mesh.faces))); np.random.random()
        return float(len(mesh.faces))
    monkeypatch.setattr(spu, 'validated_spine_volume', volume)
    return events


@pytest.mark.parametrize('duplicate', [False, True])
@pytest.mark.parametrize('heads', ['disabled', 'keep', 'grow', 'remove'])
def test_unsorted_raw_ids_keep_mesh_geometry_scalar_and_rng(monkeypatch, duplicate, heads):
    parent, original, raw = fixture_mesh(duplicate)
    vertices, faces, input_ids = raw.vertices.copy(), raw.faces.copy(), original.copy()
    events = setup(monkeypatch, parent, original, raw, heads=heads)
    masks, legacy = [], []
    np.random.seed(31)
    meshes, volumes = spu.calculate_spines_on_branch(
        SimpleNamespace(mesh=parent, skeleton=np.zeros((1, 2, 3))),
        geometric_refinement=True, geometric_refinement_config={'refine_remaining_heads': heads != 'disabled'},
        spine_face_indices=masks, geometric_refinement_face_indices=legacy, **options())
    final_rng = np.random.get_state()
    assert np.array_equal(raw.vertices, vertices) and np.array_equal(raw.faces, faces)
    assert np.array_equal(original, input_ids)
    assert len(meshes) == len(masks) == len(legacy) == len(volumes)
    for mesh, mask, old_mask, scalar in zip(meshes, masks, legacy, volumes):
        validate_ordered_face_indices(parent, mesh, mask)
        assert np.array_equal(mask, old_mask) and scalar == float(len(mesh.faces))
    if heads in ('disabled', 'keep'):
        assert meshes[0] is raw and np.array_equal(masks[0], original)
        if duplicate:
            assert 0 in masks[0] and len(parent.faces)-1 in masks[0]
        expected_events = ['primary', ('volume', len(original))]
        if heads == 'keep':
            expected_events.append('remaining')
        draws = 2
    elif heads == 'grow':
        assert meshes[0] is not raw
        assert np.array_equal(masks[0], np.sort(original)[1:])
        if duplicate:
            assert 0 not in masks[0] and len(parent.faces)-1 in masks[0]
        expected_events = ['primary', ('volume', len(original)), 'remaining', ('volume', len(original)-1)]
        draws = 3
    else:
        assert meshes == [] and masks == legacy == []
        expected_events = ['primary', ('volume', len(original)), 'remaining']
        draws = 2
    assert events == expected_events
    np.random.seed(31)
    for _ in range(draws):
        np.random.random()
    assert all(np.array_equal(a, b) for a, b in zip(final_rng, np.random.get_state()))
    if masks:
        masks[0][0] = 99
        assert np.array_equal(original, input_ids) and not np.array_equal(masks[0], legacy[0])


@pytest.mark.parametrize('duplicate', [False, True])
@pytest.mark.parametrize('primary', ['trim', 'split'])
def test_changed_cut_proposals_keep_their_constructed_row_maps(monkeypatch, duplicate, primary):
    parent, original, raw = fixture_mesh(duplicate)
    events = setup(monkeypatch, parent, original, raw, primary=primary)
    masks, legacy = [], []
    meshes, volumes = spu.calculate_spines_on_branch(
        SimpleNamespace(mesh=parent, skeleton=np.zeros((1, 2, 3))),
        geometric_refinement=True, spine_face_indices=masks,
        geometric_refinement_face_indices=legacy, **options())
    expected = [np.sort(original)[1:]] if primary == 'trim' else [np.sort(original)[:6], np.sort(original)[6:]]
    assert len(meshes) == len(masks) == len(expected)
    for mesh, mask, old_mask, ids, scalar in zip(meshes, masks, legacy, expected, volumes):
        assert mesh is not raw and np.array_equal(mask, ids) and np.array_equal(mask, old_mask)
        validate_ordered_face_indices(parent, mesh, mask)
        assert scalar == float(len(mask))
    assert events == ['primary', *[('volume', len(mask)) for mask in expected]]
