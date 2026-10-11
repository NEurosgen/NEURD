"""Whole-helper regression tests; no neuron or installed provider is imported.

Load the production helper by default; NEURD_STITCH_TEST_SOURCE can select
a captured baseline for the retained counterexample. Only `_recorrespond_stitch` is compiled from that file;
mesh extraction and the correspondence provider are controlled boundaries.
"""
import ast
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]


def load_helper(path):
    tree = ast.parse(Path(path).read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                and n.name == '_recorrespond_stitch')
    namespace = {'np': np, 'nu': SimpleNamespace(intersect1d=np.intersect1d)}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['_recorrespond_stitch'], namespace


def axis(label):
    return np.asarray([[[label * 100., 0., 0.], [label * 100. + 10., 0., 0.]]])


def context(mp_order, conn):
    entries = {label: dict(branch_face_idx=np.array([10 * label + 7, 10 * label + 8]),
                          branch_mesh=object(), width_from_skeleton=label + 0.5,
                          branch_skeleton=axis(label)) for label in mp_order}
    modified = [axis(label + 10) for label in mp_order]
    map_entry = dict(branch_face_idx=np.array([4, 5]), branch_mesh=object(),
                     width_from_skeleton=8.5, branch_skeleton=axis(8))
    extracts = []
    def submesh(groups, **kwargs):
        extracts.append((np.array(groups[0], copy=True), kwargs))
        return object()
    ctx = SimpleNamespace(MAP_branches_with_stitch_point=np.array([8]),
          MAP_pieces_idx_touching_border=np.array([8]), limb_correspondence_MAP={0: {8: map_entry}},
          MAP_idx=0, curr_MAP_sk=[axis(18)], cut_flag=False,
          limb_mesh_mparty=SimpleNamespace(submesh=submesh), conn=np.array(conn),
          MP_branches_with_stitch_point=np.array(mp_order), limb_correspondence_MP={0: entries},
          MP_idx=0, curr_MP_sk=modified)
    return ctx, entries, modified, extracts


def candidate():
    return load_helper(os.environ.get('NEURD_STITCH_TEST_SOURCE',
                       str(REPO / 'neurd/preprocess_neuron.py')))


@pytest.mark.parametrize('mp_order,conn,expected_positions', [
    ([0, 1, 2], [2], [2]),
    ([2, 1, 0], [2, 0], [2, 0]),
])
def test_exception_fallback_keeps_selected_branch_axis_pair(mp_order, conn, expected_positions):
    helper, namespace = candidate()
    ctx, entries, modified, extracts = context(mp_order, conn)
    original = {label: (entry['branch_mesh'], entry['branch_face_idx'].copy(),
                       entry['width_from_skeleton'], entry['branch_skeleton'])
                for label, entry in entries.items()}
    selected, _, positions = np.intersect1d(conn, mp_order, return_indices=True)
    assert positions.tolist() == expected_positions
    calls = []
    def fail(mesh, skeletons, **kwargs):
        calls.append((mesh, skeletons, kwargs))
        raise RuntimeError('controlled correspondence failure')
    namespace['_run_mesh_correspondence'] = fail
    helper(ctx)
    assert len(calls) == len(extracts) == 1
    assert ctx.MP_branches_for_correspondence.tolist() == selected.tolist()
    for output, label, position in zip(range(1, len(selected) + 1), selected, positions):
        result = ctx.local_correspondence_stitch_revised[output]
        assert result is entries[label]  # retain existing alias/state contract
        assert result['branch_skeleton'] is modified[position]
        assert result['branch_mesh'] is original[label][0]
        np.testing.assert_array_equal(result['branch_face_idx'], original[label][1])
        assert result['width_from_skeleton'] == original[label][2]
    for label in set(mp_order) - set(selected):
        assert entries[label]['branch_skeleton'] is original[label][3]


def test_prefix_fallback_retains_existing_aliases_and_face_frame():
    helper, namespace = candidate()
    ctx, entries, modified, _ = context([0, 1, 2], [0, 1])
    def fail(*args, **kwargs):
        raise ValueError('controlled prefix failure')
    namespace['_run_mesh_correspondence'] = fail
    helper(ctx)
    for output, label in enumerate([0, 1], 1):
        assert ctx.local_correspondence_stitch_revised[output] is entries[label]
        assert entries[label]['branch_skeleton'] is modified[label]
        np.testing.assert_array_equal(entries[label]['branch_face_idx'], [label * 10 + 7, label * 10 + 8])


@pytest.mark.parametrize('mp_order,conn', [([0, 1, 2], [2]), ([2, 1, 0], [2, 0])])
def test_success_route_preserves_selected_axes_provider_results_and_remap(mp_order, conn):
    helper, namespace = candidate()
    ctx, entries, modified, extracts = context(mp_order, conn)
    selected, _, positions = np.intersect1d(conn, mp_order, return_indices=True)
    original_axes = {label: entry['branch_skeleton'] for label, entry in entries.items()}
    captured = []
    returned = {}
    def succeed(mesh, skeletons, **kwargs):
        captured.append((mesh, skeletons, kwargs))
        for i, skeleton in enumerate(skeletons):
            returned[i] = dict(branch_skeleton=skeleton, branch_face_idx=np.array([2 * i, 2 * i + 1]),
                               branch_mesh=object(), width_from_skeleton=100. + i)
        return returned
    namespace['_run_mesh_correspondence'] = succeed
    helper(ctx)
    assert ctx.local_correspondence_stitch_revised is returned
    assert len(captured) == len(extracts) == 1
    for output, label, position in zip(range(1, len(selected) + 1), selected, positions):
        assert captured[0][1][output] is modified[position]
        assert returned[output]['branch_skeleton'] is modified[position]
        np.testing.assert_array_equal(returned[output]['branch_face_idx'], entries[label]['branch_face_idx'])
        assert returned[output]['width_from_skeleton'] == 100. + output
    assert all(entries[label]['branch_skeleton'] is original_axes[label] for label in mp_order)

