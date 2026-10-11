"""Exact characterization of the owned, labels-only spine segmentation seam."""
import json
from pathlib import Path

import numpy as np
import pytest
import trimesh

from tests.unit import skip_if_datasci_tools_unusable

skip_if_datasci_tools_unusable()

from neurd import cgal_labels as seam, spine_utils as spu
from neurd.run_config import RunConfig, runtime_environment


def grid(side=7):
    x, y = np.meshgrid(np.arange(side + 1), np.arange(side + 1))
    lower = (np.arange(side * side) // side * (side + 1) + np.arange(side * side) % side)
    faces = np.r_[np.c_[lower, lower + 1, lower + side + 1],
                  np.c_[lower + 1, lower + side + 2, lower + side + 1]]
    return trimesh.Trimesh(vertices=np.c_[x.ravel(), y.ravel(), np.zeros(x.size)] * 100,
                           faces=faces, process=False)


@pytest.fixture
def provider(monkeypatch):
    calls = []

    def controlled(stem, clusters, smoothness):
        mesh = trimesh.load(str(stem) + '.off', process=False, force='mesh')
        calls.append((Path(stem).name, clusters, smoothness, mesh.triangles.copy()))
        prefix = f'{stem}-cgal_{np.round(clusters, 2)}_{smoothness:.2f}'
        np.savetxt(prefix + '.csv', np.arange(len(mesh.faces)) // 2, fmt='%d')
        np.savetxt(prefix + '_sdf.csv', np.linspace(.1, .9, len(mesh.faces)), fmt='%.17g')

    monkeypatch.setattr(seam.tu.csm, 'cgal_segmentation', controlled)
    monkeypatch.setattr(seam.tu, 'ray_trace_distance', lambda mesh: np.ones(len(mesh.faces)))
    monkeypatch.delenv('NEURD_LEGACY_CGAL_ASSEMBLY', raising=False)
    monkeypatch.delenv('NEURD_CGAL_TIMINGS', raising=False)
    return calls


def characterized_mesh(kind):
    if kind == 'empty':
        return trimesh.Trimesh(vertices=np.empty((0, 3)), faces=np.empty((0, 3), dtype=int), process=False)
    mesh = grid()
    if kind == 'degenerate_only':
        return trimesh.Trimesh(vertices=mesh.vertices, faces=np.tile([0, 0, 1], (5, 1)), process=False)
    if kind == 'small':
        return grid(2)
    if kind == 'disconnected':
        second = grid()
        second.apply_translation([2000, 0, 0])
        return trimesh.util.concatenate([mesh, second, grid(2)])
    if kind == 'degenerate':
        return trimesh.Trimesh(vertices=mesh.vertices,
            faces=np.insert(mesh.faces, [0, 40, 97], [0, 0, 1], axis=0), process=False)
    return mesh


@pytest.mark.parametrize('kind', ['ordinary', 'empty', 'small', 'degenerate_only', 'degenerate', 'disconnected', 'sheet'])
@pytest.mark.parametrize('delete', [True, False])
def test_labels_are_exact_legacy_values_and_original_face_mapping(tmp_path, monkeypatch, provider, kind, delete):
    mesh = characterized_mesh(kind)
    before = mesh.vertices.copy(), mesh.faces.copy()
    if kind == 'sheet':
        monkeypatch.setattr(seam.tu, 'ray_trace_distance', lambda mesh: np.zeros(len(mesh.faces)))
    np.random.seed(31)
    old = seam.tu.mesh_segmentation(mesh, cgal_folder=tmp_path / 'old', clusters=3, smoothness=.17,
        return_sdf=True, delete_temp_files=delete, return_meshes=False, return_ordered_by_size=False)
    old_rng = np.random.get_state()
    old_calls = list(provider)
    provider.clear()
    np.random.seed(31)
    new = seam.mesh_segmentation_labels(mesh, cgal_folder=tmp_path / 'new', clusters=3,
                                         smoothness=.17, delete_temp_files=delete)
    assert all(np.array_equal(actual, expected) for actual, expected in zip(np.random.get_state(), old_rng))
    assert isinstance(new, tuple) and len(new) == 2
    for actual, expected in zip(new, old):
        assert actual.dtype == expected.dtype == np.float64
        assert np.array_equal(actual, expected)
    assert len(provider) == len(old_calls)
    for actual, expected in zip(provider, old_calls):
        assert actual[:3] == expected[:3]  # same random basename and native arguments
        assert np.array_equal(actual[3], expected[3])
    for directory in ('old', 'new'):
        paths = list((tmp_path / directory).glob('*'))
        assert len(paths) == (0 if delete else 3 * len(provider))
    assert np.array_equal(mesh.vertices, before[0]) and np.array_equal(mesh.faces, before[1])
    if kind in ('empty', 'small', 'degenerate_only', 'sheet'):
        assert not provider
    if kind == 'degenerate':
        mask = seam.tu.remove_degenerate_faces(mesh, return_face_idxs=True)[1]
        assert np.array_equal(new[0][np.setdiff1d(np.arange(len(mesh.faces)), mask)], np.zeros(3))
    if kind == 'disconnected':
        assert len(provider) == 2  # insignificant piece is ignored, not segmented


def test_without_sdf_returns_same_plain_array(tmp_path, provider):
    mesh = grid()
    np.random.seed(1)
    expected = seam.tu.mesh_segmentation(mesh, cgal_folder=tmp_path/'old', return_sdf=False,
                                         return_meshes=False)
    np.random.seed(1)
    actual = seam.mesh_segmentation_labels(mesh, cgal_folder=tmp_path/'new', return_sdf=False)
    assert isinstance(actual, np.ndarray)
    assert np.array_equal(expected, actual)


@pytest.mark.parametrize('failed_stage', ['native', 'csv'])
def test_failure_keeps_same_debug_files_as_legacy(tmp_path, monkeypatch, provider, failed_stage):
    if failed_stage == 'native':
        def failed_native(stem, clusters, smoothness):
            Path(f'{stem}-cgal_{np.round(clusters, 2)}_{smoothness:.2f}.csv').write_text('partial\n')
            raise RuntimeError('controlled native failure')
        monkeypatch.setattr(seam.tu.csm, 'cgal_segmentation', failed_native)
    else:
        def failed_parse(*args, **kwargs):
            raise RuntimeError('controlled CSV failure')
        monkeypatch.setattr(seam.np, 'genfromtxt', failed_parse)
    mesh = grid()
    for mode in ('old', 'new'):
        np.random.seed(8)
        with pytest.raises(RuntimeError, match='controlled'):
            if mode == 'old':
                seam.tu.mesh_segmentation(mesh, cgal_folder=tmp_path/mode, return_meshes=False)
            else:
                seam.mesh_segmentation_labels(mesh, cgal_folder=tmp_path/mode)
    retained = [{path.name: path.read_bytes() for path in (tmp_path/mode).iterdir()}
                for mode in ('old', 'new')]
    assert retained[0] == retained[1]
    assert len(retained[0]) == (2 if failed_stage == 'native' else 3)


@pytest.mark.parametrize('legacy,plot', [('1', False), (None, True)])
def test_legacy_and_plot_paths_delegate_with_installed_tuple_arguments(monkeypatch, legacy, plot):
    if legacy:
        monkeypatch.setenv('NEURD_LEGACY_CGAL_ASSEMBLY', legacy)
    else:
        monkeypatch.delenv('NEURD_LEGACY_CGAL_ASSEMBLY', raising=False)
    calls = []
    marker = object()
    monkeypatch.setattr(seam.tu, 'mesh_segmentation', lambda *args, **kwargs: calls.append((args, kwargs)) or marker)
    mesh = grid()
    assert seam.mesh_segmentation_labels(mesh, clusters=3, smoothness=.13, cgal_folder=Path('work'),
        delete_temp_files=False, plot_segmentation=plot) is marker
    assert calls == [((mesh,), dict(cgal_folder=Path('work'), clusters=3, smoothness=.13,
        return_sdf=True, delete_temp_files=False, return_meshes=False,
        return_ordered_by_size=False, plot_segmentation=plot))]


@pytest.mark.parametrize('kind', ['interleaved', 'disconnected_label', 'duplicates', 'empty'])
def test_assembly_preserves_exact_mesh_geometry_masks_and_label_order(monkeypatch, kind):
    monkeypatch.delenv('NEURD_LEGACY_CGAL_ASSEMBLY', raising=False)
    mesh = grid()
    if kind == 'duplicates':
        mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=np.r_[mesh.faces, mesh.faces], process=False)
    if kind == 'empty':
        mesh = characterized_mesh('empty')
    labels = np.arange(len(mesh.faces)) % 7 if kind != 'disconnected_label' else np.zeros(len(mesh.faces))
    old_meshes, old_indices = seam.tu.split_mesh_into_face_groups(mesh, labels, check_connect_comp=False)
    meshes, indices = seam.assemble_face_groups(mesh, labels)
    assert list(meshes) == list(old_meshes) == list(indices) == list(old_indices)
    for label in meshes:
        assert np.array_equal(indices[label], old_indices[label])
        assert np.array_equal(meshes[label].vertices, old_meshes[label].vertices)
        assert np.array_equal(meshes[label].faces, old_meshes[label].faces)
        assert np.array_equal(meshes[label].triangles, mesh.triangles[indices[label]])


def test_actual_spine_split_removes_hidden_assembly_without_changing_outputs(tmp_path, monkeypatch, provider):
    original_groups = seam.tu.split_mesh_into_face_groups
    calls = []

    def observed(*args, **kwargs):
        calls.append(kwargs['check_connect_comp'])
        return original_groups(*args, **kwargs)

    monkeypatch.setattr(seam.tu, 'split_mesh_into_face_groups', observed)
    # Deterministic ordinary shaft selection preserves both result categories.
    monkeypatch.setattr(spu, 'restrict_meshes_to_shaft_meshes_without_coordinates',
                        lambda meshes, **kwargs: np.arange(0, len(meshes), 2))
    mesh = grid()
    monkeypatch.setenv('NEURD_LEGACY_CGAL_ASSEMBLY', '1')
    np.random.seed(2)
    old = spu.split_mesh_into_spines_shaft(mesh, cgal_folder=tmp_path/'old', clusters=3,
                                          smoothness=.17, shaft_threshold=10)
    assert calls == [True, False]
    calls.clear()
    monkeypatch.delenv('NEURD_LEGACY_CGAL_ASSEMBLY')
    np.random.seed(2)
    new = spu.split_mesh_into_spines_shaft(mesh, cgal_folder=tmp_path/'new', clusters=3,
                                          smoothness=.17, shaft_threshold=10)
    assert calls == []  # one owned explicit assembly; no installed hidden assembly
    for index in (1, 3):
        assert len(new[index]) == len(old[index])
        for actual, expected in zip(new[index], old[index]):
            assert np.array_equal(actual, expected)
    for index in (0, 2):
        for actual, expected in zip(new[index], old[index]):
            assert np.array_equal(actual.triangles, expected.triangles)
    assert np.array_equal(new[4], old[4])


def test_phase_records_survive_worker_exception_and_are_opt_in(monkeypatch, capsys):
    monkeypatch.delenv('NEURD_CGAL_TIMINGS', raising=False)
    with seam.phase('quiet'):
        pass
    assert capsys.readouterr().out == ''
    monkeypatch.setenv('NEURD_CGAL_TIMINGS', '1')
    with pytest.raises(RuntimeError):
        with seam.phase('native', faces=123):
            assert json.loads(capsys.readouterr().out)['state'] == 'begin'
            raise RuntimeError('controlled provider failure')
    ended = json.loads(capsys.readouterr().out)
    assert ended['state'] == 'end' and ended['status'] == 'error'
    assert ended['phase'] == 'native' and ended['faces'] == 123 and ended['seconds'] >= 0


def test_actual_phase_sequence_and_configuration_flag(tmp_path, monkeypatch, capsys, provider):
    monkeypatch.setenv('NEURD_CGAL_TIMINGS', '1')
    monkeypatch.setattr(spu, 'restrict_meshes_to_shaft_meshes_without_coordinates', lambda meshes, **kw: [0])
    spu.split_mesh_into_spines_shaft(grid(), cgal_folder=tmp_path/'cgal', clusters=3,
                                    smoothness=.17, shaft_threshold=10)
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith('{')]
    assert [(record['phase'], record['state']) for record in records] == [
        ('labels', 'begin'), ('preparation', 'begin'), ('preparation', 'end'),
        ('component_preparation', 'begin'), ('component_preparation', 'end'),
        ('native', 'begin'), ('native', 'end'), ('csv', 'begin'), ('csv', 'end'),
        ('labels', 'end'), ('explicit_face_assembly', 'begin'), ('explicit_face_assembly', 'end'),
        ('shaft_filter', 'begin'), ('shaft_filter', 'end')]
    assert all(record.get('status', 'complete') == 'complete' for record in records)
    receipt = runtime_environment(RunConfig(), environment={'NEURD_LEGACY_CGAL_ASSEMBLY': '1'})
    assert receipt['environment']['NEURD_LEGACY_CGAL_ASSEMBLY'] == '1'


