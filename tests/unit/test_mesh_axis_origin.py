"""Origin observations never infer ownership or change stitching decisions."""
import ast
from copy import deepcopy
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import networkx as nx
import numpy as np
import pytest

from tests.unit.test_stitch_axis_pairing import axis, context

REPO = Path(__file__).resolve().parents[2]


def helpers():
    path = REPO / 'neurd/preprocess_neuron.py'
    names = {'_axis_origin_fingerprint', '_new_axis_origin', '_origin_local_id', '_cleaned_axis_origin',
             '_recorrespond_stitch', '_merge_map_mp_correspondence', '_decompose_mp_sublimbs',
             '_decompose_map_sublimbs', 'filter_limb_correspondence_for_end_nodes',
             '_SkeletonizationResult', '_SublimbDecompositionResult'}
    nodes = [node for node in ast.parse(path.read_text()).body
             if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    namespace = dict(np=np, deepcopy=deepcopy, time=time, NamedTuple=NamedTuple,
                     nu=SimpleNamespace(intersect1d=np.intersect1d))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


def test_mp_boundary_records_supplied_origin_without_changing_arrays_or_widths():
    ns = helpers()
    axes, meshes, indices, widths = [axis(1), axis(2)], [object(), object()], [np.array([2]), np.array([0, 1])], [10., 20.]
    result, endpoints, touching = ns['_decompose_mp_sublimbs'](
        [object()], [axes], [meshes], [indices], [widths], False, None, object(),
        print_fusion_steps=False, fusion_time=0.)
    assert endpoints == touching == []
    for index, entry in result[0].items():
        np.testing.assert_array_equal(entry['branch_skeleton'], axes[index])
        assert entry['branch_mesh'] is meshes[index]
        assert entry['branch_face_idx'] is indices[index]
        assert entry['width_from_skeleton'] == widths[index]
        assert entry['mesh_axis_origin']['stage'] == 'mp_decomposition'
        assert entry['mesh_axis_origin']['face_map_origin']['status'] == 'supplied_provider_indices_geometry_unverified'
        assert entry['mesh_axis_origin']['raw_input_map_available'] is False
    cleaned = ns['_cleaned_axis_origin'](result[0], np.array([1]), axis(3))
    source = cleaned['sources'][0]
    assert source['producer_sublimb_id'] == 0 and source['producer_branch_id'] == 1
    assert source['axis_provider'] == result[0][1]['mesh_axis_origin']['axis_provider']
    assert source['face_map_origin'] == result[0][1]['mesh_axis_origin']['face_map_origin']


def test_map_boundary_records_declared_callsite_without_guessing_actual_provider():
    ns = helpers()
    entry = dict(branch_skeleton=axis(1), branch_mesh=object(), branch_face_idx=np.array([4]), width_from_skeleton=10.)
    ns['_decompose_map_piece'] = lambda **kwargs: ({7: entry}, None, None)
    result, _, _ = ns['_decompose_map_sublimbs']([object()], None,
        filter_end_node_length=0., perform_cleaning_checks=False, combine_close_skeleton_nodes=False,
        combine_close_skeleton_nodes_threshold=0., use_surface_after_CGAL=False,
        surface_reconstruction_size=0., remove_mesh_interior_face_threshold=0.,
        error_on_bad_cgal_return=False, max_stitch_distance_CGAL=0., distance_by_mesh_center=True)
    assert result[0][7] is entry
    origin = entry['mesh_axis_origin']
    assert origin['axis_provider']['declared_callsite'] == 'skeletonize_and_clean_connected_branch_CGAL'
    assert origin['axis_provider']['actual_provider'] == 'unavailable_internal_provider_fallback'
    assert origin['face_map_origin']['operation'] == 'SubMesh.root_face_idx_after_correspondence'


def test_full_end_node_cleaning_preserves_width_mesh_and_maps_and_marks_recomputation():
    ns = helpers()
    source = {i: dict(branch_skeleton=axis(i), branch_mesh=object(), branch_face_idx=np.array([i]),
                      width_from_skeleton=10. * (i + 1),
                      mesh_axis_origin=ns['_new_axis_origin']('mp_decomposition', axis(i)))
              for i in range(3)}
    cleaned_axes = [axis(20), axis(30)]
    outputs = [SimpleNamespace(mesh=object(), root_face_idx=lambda: np.array([0, 1])),
               SimpleNamespace(mesh=object(), root_face_idx=lambda: np.array([2]))]
    ns['sk'] = SimpleNamespace(stack_skeletons=lambda value: value,
        skeletal_distance=lambda value: 1., clean_skeleton=lambda *args, **kwargs: (object(), []),
        decompose_skeleton_to_branches=lambda value: cleaned_axes,
        map_between_branches_lists=lambda *args: np.array([0, 0, 1]),
        calculate_skeleton_distance=lambda value: 10.)
    ns['gu'] = SimpleNamespace(invert_mapping=lambda groups: {int(face): [i] for i, faces in enumerate(groups) for face in faces})
    ns['cb'] = SimpleNamespace(resolve_face_labels=lambda **kwargs: np.array([0, 0, 1]))
    ns['submesh_ops'] = SimpleNamespace(split_into_face_groups=lambda *args, **kwargs: outputs)
    mesh = SimpleNamespace(faces=np.zeros((3, 3), dtype=np.int64))
    result = ns['filter_limb_correspondence_for_end_nodes'](source, mesh,
        error_on_no_starting_coordinates=False, verbose=False)
    assert result[0]['width_from_skeleton'] == 15.
    assert result[1]['width_from_skeleton'] == 30.
    for index in result:
        assert result[index]['branch_mesh'] is outputs[index].mesh
        assert result[index]['branch_skeleton'] is cleaned_axes[index]
        np.testing.assert_array_equal(result[index]['branch_face_idx'], outputs[index].root_face_idx())
        assert result[index]['mesh_axis_origin']['axis_identity'] == 'unavailable_recomputed_axis'
        assert result[index]['mesh_axis_origin']['raw_input_map_available'] is False


def test_noninteger_local_labels_remain_explicitly_unavailable_after_cleaning():
    ns = helpers()
    value = ns['_cleaned_axis_origin']({'legacy_label': dict(branch_skeleton=axis(1))}, np.array([0]), axis(2))
    assert value['sources'][0]['source_branch_id'] == dict(status='unavailable', reason='noninteger_local_id', label='legacy_label')


def assert_rng_equal(before, after):
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]


@pytest.mark.parametrize('value', [None, np.zeros((2, 3)), np.full((1, 2, 3), np.nan), object()])
def test_unsupported_axis_fingerprint_is_explicitly_unavailable(value):
    result = helpers()['_axis_origin_fingerprint'](value)
    assert result['status'] == 'unavailable'
    assert 'sha256' not in result


def test_axis_fingerprint_retains_order_without_mutation_or_rng_use():
    fingerprint = helpers()['_axis_origin_fingerprint']
    value = np.concatenate([axis(1), axis(2)])
    original = value.copy()
    state = np.random.get_state()
    result = fingerprint(value)
    expected = hashlib.sha256(str(tuple(value.shape)).encode('ascii'))
    expected.update(value.astype('<f8').tobytes(order='C'))
    assert result['sha256'] == expected.hexdigest()
    assert fingerprint(value.astype('>f8')) == result
    assert fingerprint(value[::-1])['sha256'] != result['sha256']
    np.testing.assert_array_equal(value, original)
    assert_rng_equal(state, np.random.get_state())


@pytest.mark.parametrize('fail', [False, True])
@pytest.mark.parametrize('order,conn', [([0, 1, 2], [2]), ([2, 1, 0], [2, 0])])
def test_stitch_journal_records_actual_ids_positions_and_axis_hashes(fail, order, conn):
    ns = helpers()
    ctx, entries, modified, extracts = context(order, conn)
    selected, conn_positions, positions = np.intersect1d(conn, order, return_indices=True)
    original = {key: ns['_axis_origin_fingerprint'](entry['branch_skeleton'])
                for key, entry in entries.items()}
    state = np.random.get_state()
    def correspond(mesh, skeletons, **kwargs):
        if fail:
            raise ValueError('controlled retained partition')
        return {index: dict(branch_skeleton=skeleton, branch_face_idx=np.array([2 * index, 2 * index + 1]),
                            branch_mesh=object(), width_from_skeleton=11. + index)
                for index, skeleton in enumerate(skeletons)}
    ns['_run_mesh_correspondence'] = correspond
    ns['_recorrespond_stitch'](ctx)
    for index, entry in ctx.local_correspondence_stitch_revised.items():
        journal = entry['mesh_axis_origin']
        assert journal['raw_input_map_available'] is False
        assert journal['face_map_origin']['status'] == 'operation_recorded_geometry_unverified'
        record = journal['stitch']
        assert record['output_role'] == ('MAP' if index == 0 else 'MP')
        assert record['selected_mp_count'] == len(selected)
        assert record['omitted_selected_mp_count'] == 0
        assert record['outcome'] == ('retained_partition' if fail else 'resolved')
        if fail:
            assert record['fallback'] == dict(exception_type='ValueError', message='controlled retained partition')
        else:
            assert record['fallback'] is None
        for source, key, position, conn_position in zip(record['selected_mp'], selected, positions, conn_positions):
            assert source['branch_id'] == key
            assert source['candidate_position'] == position
            assert source['conn_position'] == conn_position
            assert source['correspondence_position'] == order.index(key)
            assert source['original_axis'] == original[key]
            assert source['used_axis'] == ns['_axis_origin_fingerprint'](modified[position])
        json.dumps(journal, allow_nan=False)
    assert_rng_equal(state, np.random.get_state())
    assert len(extracts) == 1


def test_large_selection_has_explicit_bounded_observation_coverage():
    ns = helpers()
    order = list(range(70))
    ctx, _, _, _ = context(order, order)
    def fail(*args, **kwargs):
        raise RuntimeError('x' * 2000)
    ns['_run_mesh_correspondence'] = fail
    ns['_recorrespond_stitch'](ctx)
    record = ctx.local_correspondence_stitch_revised[1]['mesh_axis_origin']['stitch']
    assert len(record['selected_mp']) == 64
    assert record['selected_mp_count'] == 70
    assert record['omitted_selected_mp_count'] == 6
    assert len(record['fallback']['message']) == 1000


def test_flattening_and_cleaning_retain_bounded_flat_decisions():
    ns = helpers()
    ctx, _, _, _ = context([2, 1, 0], [2, 0])
    def fail(*args, **kwargs):
        raise RuntimeError('pairing retained')
    ns['_run_mesh_correspondence'] = fail
    ns['_recorrespond_stitch'](ctx)
    entries = ctx.local_correspondence_stitch_revised
    flat = ns['_merge_map_mp_correspondence']({4: {8: entries[0]}}, {9: {2: entries[1], 0: entries[2]}})
    assert flat[1] is entries[1]
    record = ns['_cleaned_axis_origin'](flat, np.array([1, 2]), axis(99))
    assert record['axis_identity'] == 'unavailable_recomputed_axis'
    assert record['source_count'] == 2
    assert [source['source_branch_id'] for source in record['sources']] == [1, 2]
    assert all(source['stitch']['fallback']['exception_type'] == 'RuntimeError' for source in record['sources'])
    assert not any('sources' in source for source in record['sources'])
    record['sources'][0]['stitch']['fallback']['message'] = 'changed only in copy'
    assert flat[1]['mesh_axis_origin']['stitch']['fallback']['message'] == 'pairing retained'
    bounded = ns['_cleaned_axis_origin']({i: flat[1] for i in range(30)}, np.arange(30), axis(99))
    assert len(bounded['sources']) == 16
    assert bounded['source_count'] == 30 and bounded['omitted_source_count'] == 14


def test_branch_constructor_and_correspondence_roundtrip_isolated_metadata():
    from neurd.neuron import Branch, Limb, _attach_branches
    origin = helpers()['_new_axis_origin']('test_decision', axis(1))
    data = {7: dict(branch_skeleton=axis(1), width_from_skeleton=8., branch_mesh=None,
                    branch_face_idx=np.array([], dtype=np.int64), mesh_axis_origin=origin)}
    graph = nx.Graph()
    graph.add_node(7)
    assert _attach_branches(graph, data) is False
    branch = graph.nodes[7]['data']
    assert branch.mesh_axis_origin == origin and branch.mesh_axis_origin is not origin
    limb = Limb.__new__(Limb)
    limb.concept_network = graph
    result = limb.limb_correspondence
    assert result[7]['mesh_axis_origin'] == origin
    result[7]['mesh_axis_origin']['stage'] = 'mutated copy'
    assert branch.mesh_axis_origin['stage'] == 'test_decision'
    assert origin['stage'] == 'test_decision'
    assert Branch(axis(2)).mesh_axis_origin is None
    del branch.mesh_axis_origin  # Older serialized Branch objects remain supported.
    assert 'mesh_axis_origin' not in limb.limb_correspondence[7]
