"""Tracing observes ordered filters, RNG and publication without changing them."""
import ast
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from threading import Barrier
from types import SimpleNamespace

import neurd
import numpy as np
import pytest
import trimesh

from tests.unit import skip_if_datasci_tools_unusable

skip_if_datasci_tools_unusable()
from neurd import (cgal_labels as seam, h01_batch, spine_refinement as refinement,
                   spine_utils as spu, spine_volume as volume)


def records(capsys):
    return [json.loads(line) for line in capsys.readouterr().out.splitlines()
            if line.startswith('{')]


def options():
    return dict(spine_n_face_threshold=0, spine_sk_length_threshold=0,
                filter_by_bounding_box_longest_side_length=False,
                filter_out_border_spines=False, skeleton_endpoint_nullification=False,
                soma_vertex_nullification=False, calculate_spine_volume=False,
                filter_by_volume=False, filter_by_face_area_mean=False,
                filter_by_boundary_to_area_ratio=False)


def same_rng(actual, expected):
    assert all(np.array_equal(a, b) for a, b in zip(actual, expected))


@pytest.mark.parametrize('setting', [None, '', '0', 'true'])
def test_disabled_phase_is_silent_and_never_evaluates_lazy_fields(monkeypatch, capsys, setting):
    if setting is None:
        monkeypatch.delenv('NEURD_CGAL_TIMINGS', raising=False)
    else:
        monkeypatch.setenv('NEURD_CGAL_TIMINGS', setting)
    def forbidden():
        pytest.fail('The silent path evaluated a diagnostic field')
    with seam.phase('quiet', input_faces=forbidden) as metadata:
        assert metadata is None
    assert capsys.readouterr().out == ''


@pytest.mark.parametrize('exception', [RuntimeError, KeyboardInterrupt, SystemExit])
def test_nested_phases_flush_errors_and_restore_parent(monkeypatch, exception):
    monkeypatch.setenv('NEURD_CGAL_TIMINGS', '1')
    output = []
    monkeypatch.setattr(seam, 'print', lambda line, **kw: output.append((json.loads(line), kw)), raising=False)
    with pytest.raises(exception):
        with seam.phase('outer', call_role='trim_fallback'):
            with seam.phase('inner', input_meshes=lambda: 3) as metadata:
                metadata['output_meshes'] = 2
                raise exception('controlled failure')
    with seam.phase('after'):
        pass
    assert all(kw == {'flush': True} for _, kw in output)
    rows = [row for row, _ in output]
    assert [(r['phase'], r['state']) for r in rows] == [
        ('outer', 'begin'), ('inner', 'begin'), ('inner', 'end'),
        ('outer', 'end'), ('after', 'begin'), ('after', 'end')]
    assert rows[1]['parent_phase'] == 'outer' and rows[1]['depth'] == 1
    assert rows[2]['status'] == rows[3]['status'] == 'error'
    assert rows[2]['output_meshes'] == 2
    assert rows[1]['call_role'] == 'trim_fallback'
    assert all(r['duration_scope'] == 'inclusive' for r in rows)
    assert rows[4]['parent_phase'] is None and rows[4]['depth'] == 0
    assert 'call_role' not in rows[4]


def test_phase_nesting_is_thread_local(monkeypatch):
    monkeypatch.setenv('NEURD_CGAL_TIMINGS', '1')
    output, barrier = [], Barrier(2)
    monkeypatch.setattr(seam, 'print', lambda line, **kw: output.append(json.loads(line)), raising=False)
    def worker(name):
        with seam.phase(name, call_role=name):
            barrier.wait(timeout=5)
            with seam.phase(name + '_child'):
                pass
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(worker, ['left', 'right']))
    for name in ('left', 'right'):
        child = next(r for r in output if r['phase'] == name + '_child' and r['state'] == 'begin')
        assert child['parent_phase'] == name and child['call_role'] == name and child['depth'] == 1


def test_length_trace_preserves_rejected_candidate_calls_and_rng(monkeypatch, capsys):
    meshes = [SimpleNamespace(faces=np.zeros((n, 3)), vertices=np.zeros((v, 3)))
              for n, v in [(2, 999), (4, 1000), (6, 1001)]]
    ids = {id(mesh): i for i, mesh in enumerate(meshes)}
    calls = []
    def length(mesh):
        calls.append(ids[id(mesh)])
        return np.random.random()
    monkeypatch.setattr(spu, 'spine_length', length)
    results = []
    for enabled in (False, True):
        monkeypatch.setenv('NEURD_CGAL_TIMINGS', '1' if enabled else '0')
        calls.clear()
        np.random.seed(27)
        result = spu.filter_spine_meshes(meshes, spine_n_face_threshold=4,
                                        spine_sk_length_threshold=.5)
        results.append(([id(m) for m in result], list(calls), np.random.get_state()))
        traced = records(capsys)
        if enabled:
            assert [(r['phase'], r['state']) for r in traced] == [
                ('spine_length_calculation', 'begin'), ('spine_length_calculation', 'end')]
            assert traced[0]['vertices_at_most_1000'] == 2
            assert traced[0]['vertices_above_1000'] == 1 and traced[0]['input_faces'] == 12
        else:
            assert not traced
    assert results[0][:2] == results[1][:2]
    assert results[0][1] == [0, 1, 2]  # Face reject still consumes its existing length call.
    same_rng(results[0][2], results[1][2])


def test_disabled_length_phase_does_not_read_vertices(monkeypatch, capsys):
    class Mesh:
        faces = np.zeros((4, 3))
        @property
        def vertices(self):
            pytest.fail('Default tracing inspected vertex counts')
    mesh = Mesh()
    monkeypatch.delenv('NEURD_CGAL_TIMINGS', raising=False)
    monkeypatch.setattr(spu, 'spine_length', lambda _: 2)
    assert spu.filter_spine_meshes([mesh], spine_n_face_threshold=1,
                                   spine_sk_length_threshold=1) == [mesh]
    assert capsys.readouterr().out == ''


def test_ordinary_filters_keep_exact_order_masks_volumes_and_rng(monkeypatch, capsys):
    parent = trimesh.creation.box()
    masks = [np.array([i]) for i in range(10)]
    meshes = [parent.submesh([mask], append=True) for mask in masks]
    ids = {id(mesh): i for i, mesh in enumerate(meshes)}
    calls = []
    monkeypatch.setattr(spu, 'get_spine_meshes_unfiltered_from_mesh', lambda *a, **kw: (meshes, masks))
    def drop_first(name):
        def apply(meshes, *args, **kw):
            calls.append((name, [ids[id(mesh)] for mesh in meshes]))
            np.random.random()
            return meshes[1:]
        return apply
    monkeypatch.setattr(spu, 'filter_spine_meshes', drop_first('length'))
    monkeypatch.setattr(spu.tu, 'filter_meshes_by_bounding_box_longest_side', drop_first('bbox'))
    monkeypatch.setattr(spu, 'filter_out_border_spines', lambda parent, meshes, **kw: drop_first('border')(meshes))
    monkeypatch.setattr(spu.sk, 'find_skeleton_endpoint_coordinates', lambda _: np.zeros((1, 3)))
    monkeypatch.setattr(spu.tu, 'filter_meshes_by_containing_coordinates', drop_first('endpoint'))
    monkeypatch.setattr(spu, 'filter_out_soma_touching_spines', drop_first('soma'))
    def metric(name, reject, mesh):
        calls.append((name, ids[id(mesh)]))
        np.random.random()
        return 0 if ids[id(mesh)] == reject else ids[id(mesh)] + 1
    monkeypatch.setattr(spu, 'validated_spine_volume', lambda mesh: metric('volume', 5, mesh))
    monkeypatch.setattr(spu.tu, 'face_area_mean', lambda mesh: metric('area', 6, mesh), raising=False)
    monkeypatch.setattr(spu.tu, 'area_to_boundary_length_ratio', lambda mesh: metric('boundary', 7, mesh), raising=False)
    opts = {**options(), 'spine_n_face_threshold': 1,
            'filter_by_bounding_box_longest_side_length': True, 'filter_out_border_spines': True,
            'skeleton_endpoint_nullification': True, 'soma_vertex_nullification': True,
            'soma_verts': np.zeros((1, 3)), 'calculate_spine_volume': True,
            'filter_by_volume': True, 'filter_by_volume_threshold': .5,
            'filter_by_face_area_mean': True, 'filter_by_face_area_mean_min': .5,
            'filter_by_boundary_to_area_ratio': True, 'filter_by_boundary_to_area_ratio_min': .5}
    outcomes = []
    for enabled in (False, True):
        monkeypatch.setenv('NEURD_CGAL_TIMINGS', '1' if enabled else '0')
        np.random.seed(17)
        calls.clear(); output_masks = []
        kept, volumes = spu.calculate_spines_on_branch(
            SimpleNamespace(mesh=parent, skeleton=np.zeros((1, 2, 3))),
            geometric_refinement=False, spine_face_indices=output_masks, **opts)
        outcomes.append(([ids[id(mesh)] for mesh in kept], [m.tolist() for m in output_masks],
                         list(volumes), list(calls), np.random.get_state()))
        traced = records(capsys)
        if enabled:
            initial = next(r for r in traced if r['phase'] == 'final_filters' and r['state'] == 'end')
            assert initial['call_role'] == 'initial'
            assert initial['input_meshes'] == 9 and initial['output_meshes'] == 2
            subphases = [r['phase'] for r in traced if r['state'] == 'begin' and r['parent_phase'] == 'final_filters']
            assert subphases == ['final_filter_bbox', 'final_filter_border', 'final_filter_endpoint',
                'final_filter_soma', 'final_filter_volume', 'final_filter_volume_threshold',
                'final_filter_face_area', 'final_filter_boundary_area']
        else:
            assert not traced
    assert outcomes[0][:4] == outcomes[1][:4] == ([8, 9], [[8], [9]], [9, 10], outcomes[0][3])
    same_rng(outcomes[0][4], outcomes[1][4])


@pytest.mark.parametrize('mode', ['trim', 'growth', 'remaining_head'])
def test_fallback_and_head_proposal_roles_preserve_exact_science(monkeypatch, capsys, mode):
    parent = trimesh.creation.box()
    original = np.arange(6 if mode == 'growth' else 12)
    raw = parent.submesh([original], append=True)
    monkeypatch.setattr(spu, 'get_spine_meshes_unfiltered_from_mesh', lambda *a, **kw: ([raw], [original]))
    def refine(*args, **kwargs):
        if mode == 'trim':
            kwargs['pre_trim_face_groups'][0] = original
            mask, action, extra = original[:6], 'trim_shaft_surface', {'shaft_trim': {'action': 'trim'}}
        elif mode == 'growth':
            kwargs['pre_growth_face_groups'][0] = original
            mask, action, extra = np.arange(12), 'grow_head', {'head_growth': {'action': 'grow'}}
        else:
            mask, action, extra = original, 'keep', {}
        return [mask], dict(candidates=[dict(input_index=0, output_index=0, output_indices=[0],
            output_faces=len(mask), action=action, **extra)], counts={action: 1}, seconds=0)
    monkeypatch.setattr(refinement, 'refine_spine_face_indices', refine)
    def remaining(parent, skeleton, groups, raw_groups, owners, config, validate_proposal):
        assert validate_proposal(0, original[:6]) is False
        return groups, dict(candidates=[dict(input_index=0, output_index=0, action='keep')],
                            seconds=0, validation_seconds=0)
    monkeypatch.setattr(refinement, 'refine_remaining_spine_heads', remaining)
    calls = []
    def volume(mesh):
        calls.append(len(mesh.faces)); np.random.random()
        return 1 if len(mesh.faces) == len(original) else 0
    monkeypatch.setattr(spu, 'validated_spine_volume', volume)
    outcomes = []
    for enabled in (False, True):
        monkeypatch.setenv('NEURD_CGAL_TIMINGS', '1' if enabled else '0')
        calls.clear(); masks, diagnostics = [], {}
        np.random.seed(91)
        kept, volumes = spu.calculate_spines_on_branch(
            SimpleNamespace(mesh=parent, skeleton=np.zeros((1, 2, 3))),
            geometric_refinement=True, geometric_refinement_config={'refine_remaining_heads': mode == 'remaining_head'},
            geometric_refinement_diagnostics=diagnostics, spine_face_indices=masks,
            **{**options(), 'calculate_spine_volume': True, 'filter_by_volume': True, 'filter_by_volume_threshold': .5})
        outcomes.append(([mesh.triangles.tolist() for mesh in kept], [g.tolist() for g in masks],
                         np.asarray(volumes).tolist(), diagnostics, list(calls), np.random.get_state()))
        traced = records(capsys)
        if enabled:
            roles = [r['call_role'] for r in traced if r['phase'] == 'final_filters' and r['state'] == 'begin']
            role = {'trim': 'trim_fallback', 'growth': 'growth_fallback', 'remaining_head': 'remaining_head'}[mode]
            assert roles == ['initial', role]
            nested = next(r for r in traced if r['phase'] == 'final_filters' and r['call_role'] == role)
            assert nested['parent_phase'] == {'trim': 'trim_fallback_validation',
                'growth': 'growth_fallback_validation', 'remaining_head': 'remaining_head_proposal_validation'}[mode]
        else:
            assert not traced
    assert outcomes[0][:5] == outcomes[1][:5]
    assert outcomes[0][1] == [original.tolist()] and outcomes[0][2] == [1]
    same_rng(outcomes[0][5], outcomes[1][5])


def test_final_filter_exception_closes_inner_and_outer_phases(monkeypatch, capsys):
    parent = trimesh.creation.box()
    monkeypatch.setenv('NEURD_CGAL_TIMINGS', '1')
    monkeypatch.setattr(spu, 'get_spine_meshes_unfiltered_from_mesh', lambda *a, **kw: [parent])
    def fail(mesh):
        raise RuntimeError('controlled volume failure')
    monkeypatch.setattr(spu, 'validated_spine_volume', fail)
    with pytest.raises(RuntimeError, match='controlled volume failure'):
        spu.calculate_spines_on_branch(SimpleNamespace(mesh=parent), geometric_refinement=False,
            **{**options(), 'calculate_spine_volume': True})
    ended = [r for r in records(capsys) if r['state'] == 'end']
    assert [(r['phase'], r['status']) for r in ended] == [
        ('final_filter_volume', 'error'), ('final_filters', 'error')]


def publication_body():
    """Execute the actual worker's publication statements without native setup."""
    tree = ast.parse(Path(h01_batch.__file__).read_text())
    worker = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'branch_worker')
    start = next(i for i, n in enumerate(worker.body) if isinstance(n, ast.With)
                 and n.items[0].context_expr.args[0].value == 'h01_surface_provenance_check')
    return compile(ast.fix_missing_locations(ast.Module(body=worker.body[start:], type_ignores=[])),
                   '<actual h01 worker publication>', 'exec')


@pytest.mark.parametrize('fail', [None, 'mask', 'volume', 'output'])
def test_h01_publication_phases_preserve_outputs_and_close_errors(monkeypatch, capsys, tmp_path, fail):
    monkeypatch.setenv('NEURD_CGAL_TIMINGS', '1')
    parent = trimesh.creation.box()
    mask = np.arange(len(parent.faces))
    quality = spu.estimate_mesh_volume(parent)
    source = tmp_path/'source.off'; parent.export(source)
    provider = tmp_path/'provider.so'; provider.write_bytes(b'publication test only')
    staging = tmp_path/'staging'; staging.mkdir()
    exported_mesh = parent.copy()
    _, volume_record = volume.mesh_volume_with_quality(exported_mesh)
    if fail == 'mask':
        exported_mesh.vertices[0] += 1
    if fail == 'output':
        def failed_export(*args, **kwargs):
            raise RuntimeError('controlled export failure')
        monkeypatch.setattr(exported_mesh, 'export', failed_export)
    if fail == 'volume':
        monkeypatch.setattr(spu, 'estimate_mesh_volume', lambda _: dict(valid=False, volume_nm3=0))
    env = dict(np=np, cgal_phase=seam.phase, meshes=[exported_mesh], masks=[mask],
        volume_quality=[volume_record],
        reuse_volume_quality=(volume.reuse_volume_quality if fail != 'volume'
                              else lambda record, mesh: spu.estimate_mesh_volume(mesh)),
        volumes=np.array([quality['volume_nm3']]), parent=parent,
        diagnostics={'candidates': [{'raw_input_index': 0, 'action': 'keep',
            'final_output_mapping': [{'final_output_index': 0, 'part_index': 0}]}]},
        triangle_keys=h01_batch.triangle_keys, staging=staging, spu=spu,
        skeleton=np.zeros((1, 2, 3)), original=np.zeros((1, 2, 3)), auxiliary=None,
        row={'source': str(source), 'archived_spines': 1}, Path=Path,
        save=h01_batch.save, inputs=lambda _: {}, source_inputs={}, shutil=h01_batch.shutil,
        key='test', manifest={'run_hash': 'test', 'skeleton': {'use_auxiliary_for_heads': False}},
        skeleton_qc={}, elapsed=1, eligibility={'eligible': True},
        csm=SimpleNamespace(__file__=str(provider)), sha=h01_batch.sha)
    if fail:
        with pytest.raises((RuntimeError, ValueError)):
            exec(publication_body(), env)
        ended = [r for r in records(capsys) if r['state'] == 'end']
        assert ended[-1]['status'] == 'error'
        assert ended[-1]['phase'] == {'mask': 'h01_mesh_mask_validation',
            'volume': 'h01_volume_quality', 'output': 'h01_output_write'}[fail]
        assert not (staging/'result.json').exists()
    else:
        exec(publication_body(), env)
        traced = records(capsys)
        assert all(r.get('status', 'complete') == 'complete' for r in traced)
        assert [r['phase'] for r in traced if r['state'] == 'begin'] == [
            'h01_surface_provenance_check', 'h01_mesh_mask_validation', 'h01_output_write',
            'h01_volume_quality', 'h01_output_write']
        result = json.loads((staging/'result.json').read_text())
        assert result['status'] == 'complete' and result['spine_count'] == 1
        assert result['spines'][0]['volume_quality'] == quality
        assert np.array_equal(np.load(staging/'spine_faces.npz')['faces'], mask)
        rng = np.random.get_state()
        silent_staging = tmp_path/'silent_staging'; silent_staging.mkdir()
        env['staging'] = silent_staging
        monkeypatch.setenv('NEURD_CGAL_TIMINGS', '0')
        exec(publication_body(), env)
        assert capsys.readouterr().out == ''
        assert json.loads((silent_staging/'result.json').read_text()) == result
        same_rng(np.random.get_state(), rng)


def test_group_adjacency_paths_and_assembly_preserve_ordered_masks(monkeypatch, capsys):
    parent = trimesh.creation.box()
    groups = [np.array([0, 1]), np.array([2, 3]), np.array([4, 5]),
              np.array([6, 7, 8]), np.array([9, 10, 11])]
    pieces = [parent.submesh([group], append=True) for group in groups]
    sdf = np.arange(len(parent.faces), dtype=float)
    monkeypatch.setattr(spu, 'split_mesh_into_spines_shaft', lambda *a, **kw:
                        (pieces[:3], groups[:3], pieces[3:], groups[3:], sdf))
    events = []
    def adjacency(parent, groups, **kw):
        events.append(('adjacency', [g.tolist() for g in groups])); np.random.random()
        return np.array([[0, 3], [0, 4], [1, 3], [2, 4]])
    monkeypatch.setattr(spu.submesh_ops, 'pieces_adjacency', adjacency)
    path = spu.nx.shortest_path
    def shortest(*a, **kw):
        events.append(('path', kw['source'], kw['target'])); np.random.random()
        return path(*a, **kw)
    monkeypatch.setattr(spu.nx, 'shortest_path', shortest)
    outcomes = []
    for enabled in (False, True):
        monkeypatch.setenv('NEURD_CGAL_TIMINGS', '1' if enabled else '0')
        events.clear(); np.random.seed(57)
        meshes, values, masks = spu.get_spine_meshes_unfiltered_from_mesh(parent,
            segment_name='controlled', return_sdf=True, return_mesh_idx=True,
            connectivity_type_for_shaft='vertices')
        outcomes.append(([m.triangles.tolist() for m in meshes], [v.tolist() for v in values],
                         [g.tolist() for g in masks], list(events), np.random.get_state()))
        traced = records(capsys)
        if enabled:
            assert [(r['phase'], r['state']) for r in traced] == [
                ('spine_group_adjacency', 'begin'), ('spine_group_adjacency', 'end'),
                ('shaft_path_expansion', 'begin'), ('shaft_path_expansion', 'end'),
                ('spine_group_assembly', 'begin'), ('spine_group_assembly', 'end')]
            assert traced[-1]['output_meshes'] == 2 and traced[-1]['output_faces'] == 4
        else:
            assert not traced
    assert outcomes[0][:4] == outcomes[1][:4]
    assert outcomes[0][2] == [[4, 5], [2, 3]]
    same_rng(outcomes[0][4], outcomes[1][4])
