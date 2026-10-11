"""Public filter results retain legacy types, order and transactional outputs."""
from types import SimpleNamespace

import neurd
import numpy as np
import pytest
import trimesh

from tests.unit import skip_if_datasci_tools_unusable

skip_if_datasci_tools_unusable()
from neurd import spine_utils as spu


def options():
    return dict(spine_n_face_threshold=0, spine_sk_length_threshold=0,
                filter_by_bounding_box_longest_side_length=False,
                filter_out_border_spines=False, skeleton_endpoint_nullification=False,
                soma_vertex_nullification=False, calculate_spine_volume=True,
                filter_by_volume=False, filter_by_face_area_mean=False,
                filter_by_boundary_to_area_ratio=False, geometric_refinement=False)


def candidates(monkeypatch):
    parent = trimesh.creation.box()
    masks = [np.array([2, 0, 1]), np.array([6, 4, 5]), np.array([10, 8, 9])]
    raw = [parent.submesh([mask], append=True) for mask in masks]
    monkeypatch.setattr(spu, 'get_spine_meshes_unfiltered_from_mesh',
                        lambda *a, **kw: (raw, masks) if kw['return_mesh_idx'] else raw)
    values = {id(mesh): np.float32(index + 1) for index, mesh in enumerate(raw)}
    return parent, masks, raw, values


@pytest.mark.parametrize('filter_kind, expected_ids', [
    ('none', [0, 1, 2]), ('volume', [1, 2]), ('area', [0, 2]),
    ('boundary', [0, 1]), ('all', [1]),
])
@pytest.mark.parametrize('collect', [False, True])
def test_public_volume_container_dtype_and_aligned_outputs(monkeypatch, filter_kind, expected_ids, collect):
    parent, original_masks, raw, values = candidates(monkeypatch)
    records = {id(mesh): object() for mesh in raw}
    measured = []

    def scalar(mesh):
        measured.append(id(mesh))
        return values[id(mesh)]

    def with_quality(mesh):
        return scalar(mesh), records[id(mesh)]

    monkeypatch.setattr(spu, 'validated_spine_volume', scalar)
    monkeypatch.setattr(spu, 'mesh_volume_with_quality', with_quality)
    monkeypatch.setattr(spu.tu, 'face_area_mean',
                        lambda mesh: 0 if mesh is raw[1] and filter_kind == 'area' else 1, raising=False)
    monkeypatch.setattr(spu.tu, 'area_to_boundary_length_ratio',
                        lambda mesh: 0 if mesh is raw[2] else 1, raising=False)
    opts = options()
    opts.update(filter_by_volume=filter_kind in ('volume', 'all'), filter_by_volume_threshold=1,
                filter_by_face_area_mean=filter_kind in ('area', 'all'), filter_by_face_area_mean_min=.5,
                filter_by_boundary_to_area_ratio=filter_kind in ('boundary', 'all'),
                filter_by_boundary_to_area_ratio_min=.5)
    masks = ['prior mask']
    quality = ['prior quality']
    kwargs = {'spine_volume_quality': quality} if collect else {}
    meshes, volumes = spu.calculate_spines_on_branch(
        SimpleNamespace(mesh=parent), spine_face_indices=masks, **opts, **kwargs)
    assert [id(mesh) for mesh in meshes] == [id(raw[index]) for index in expected_ids]
    assert measured == [id(mesh) for mesh in raw]
    assert np.array_equal(np.asarray(volumes), np.array([index + 1 for index in expected_ids], np.float32))
    if filter_kind in ('none', 'volume'):
        assert isinstance(volumes, np.ndarray) and volumes.dtype == np.float32
    else:
        assert type(volumes) is list and all(type(value) is np.float32 for value in volumes)
    for mask, index in zip(masks, expected_ids):
        assert np.array_equal(mask, original_masks[index]) and mask.dtype == np.int64
    if collect:
        assert quality == [records[id(raw[index])] for index in expected_ids]
    else:
        assert quality == ['prior quality']


def test_late_filter_exception_preserves_output_collectors(monkeypatch):
    parent, _, raw, values = candidates(monkeypatch)
    measurements = []

    def measure(mesh):
        measurements.append(id(mesh))
        return values[id(mesh)], object()

    def fail(mesh):
        raise ValueError('controlled late filter failure')

    monkeypatch.setattr(spu, 'mesh_volume_with_quality', measure)
    monkeypatch.setattr(spu.tu, 'face_area_mean', fail, raising=False)
    masks = ['prior mask']
    quality = ['prior quality']
    with pytest.raises(ValueError, match='controlled late filter failure'):
        spu.calculate_spines_on_branch(
            SimpleNamespace(mesh=parent), spine_face_indices=masks, spine_volume_quality=quality,
            **{**options(), 'filter_by_face_area_mean': True, 'filter_by_face_area_mean_min': .5})
    assert measurements == [id(mesh) for mesh in raw]
    assert masks == ['prior mask'] and quality == ['prior quality']
