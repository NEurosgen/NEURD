"""The query engine preserves feature dispatch, row order and public results."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import trimesh

from tests.unit import skip_if_datasci_tools_unusable

skip_if_datasci_tools_unusable()

from datasci_tools import networkx_utils as xu
from neurd import neuron_searching as ns
from neurd.neuron import Branch, Limb, Neuron


def branch(width, coordinate):
    obj = Branch(np.array([[[coordinate, 0, 0], [coordinate + 10, 0, 0]]], dtype=float),
                 width=width, mesh=trimesh.creation.box())
    obj.width_new['median_mesh_center'] = width
    obj.custom_score = width + 1
    return obj


@pytest.fixture
def neuron():
    obj = object.__new__(Neuron)
    obj.concept_network = xu.GraphOrderedEdges()
    obj.concept_network.add_node('S0', data=SimpleNamespace())
    for name, nodes in [('L9', [(9, 120), (2, 80)]), ('L2', [(8, 150)])]:
        limb = object.__new__(Limb)
        limb.concept_network = xu.GraphOrderedEdges()
        for node, width in nodes:
            limb.concept_network.add_node(node, data=branch(width, node * 10))
        obj.concept_network.add_node(name, data=limb)
    return obj


def assert_grouped(actual, expected):
    assert list(actual) == list(expected)
    for name, nodes in expected.items():
        np.testing.assert_array_equal(actual[name], nodes)
        assert np.issubdtype(actual[name].dtype, np.integer)


def test_default_inferred_query_and_group_order(neuron):
    result = ns.query_neuron(neuron, '(width > 100) and (n_faces_branch >= 12)')
    assert_grouped(result, {'L2': [8], 'L9': [9]})


def test_configured_spine_query_uses_explicit_string_features(neuron):
    from neurd import parameters

    for limb_name in neuron.get_limb_node_names():
        for node in neuron[limb_name].concept_network:
            neuron[limb_name][node].mesh = trimesh.creation.icosphere(subdivisions=2)
    result = ns.query_neuron(neuron, parameters.params.query,
                            functions_list=['median_mesh_center', 'n_faces_branch'])
    assert_grouped(result, {'L2': [8], 'L9': [9]})


def test_concept_network_input_has_same_grouped_result(neuron):
    result = ns.query_neuron(neuron.concept_network, '(width > 100)')
    assert_grouped(result, {'L2': [8], 'L9': [9]})


def test_real_branch_features_and_attribute_fallback(neuron):
    features = [ns.width, ns.skeletal_length, ns.n_faces_branch, ns.n_spines,
                ns.median_mesh_center, ns.area, 'custom_score']
    frame = ns.query_neuron(neuron, 'width > 100', functions_list=features,
                            return_dataframe_before_filtering=True)
    assert list(frame) == ['limb', 'node', 'width', 'skeletal_length', 'n_faces_branch',
                           'n_spines', 'median_mesh_center', 'area', 'custom_score']
    assert frame[['limb', 'node']].values.tolist() == [['L9', 9], ['L9', 2], ['L2', 8]]
    assert frame.width.tolist() == [120, 80, 150]
    assert frame.skeletal_length.tolist() == [10, 10, 10]
    assert frame.n_faces_branch.tolist() == [12, 12, 12]
    assert frame.n_spines.tolist() == [0, 0, 0]
    assert frame.custom_score.tolist() == [121, 81, 151]
    assert frame.area.tolist() == [6e-6, 6e-6, 6e-6]


def test_custom_callable_forwards_context_in_graph_order(neuron):
    calls = []

    def scaled(curr_branch, limb_name, branch_name, scale):
        calls.append((limb_name, branch_name, scale))
        return curr_branch.width * scale

    result = ns.query_neuron(neuron, 'scaled > @threshold', functions_list=[scaled],
                            function_kwargs={'scale': 2}, query_variables_dict={'threshold': 200})
    assert_grouped(result, {'L2': [8], 'L9': [9]})
    assert calls == [('L9', 9, 2), ('L9', 2, 2), ('L2', 8, 2)]


@pytest.mark.parametrize('inferred', [False, True])
@pytest.mark.parametrize('mapping', [False, True])
def test_limb_scalar_and_complete_mapping_dispatch(neuron, inferred, mapping):
    def limb_score(curr_limb, limb_name, **kwargs):
        return ({node: node for node in curr_limb.concept_network} if mapping else
                11 if limb_name == 'L9' else 3)

    if not inferred:
        limb_score = ns.run_options(run_type='Limb')(limb_score)
    frame = ns.query_neuron(neuron, 'limb_score >= 0', functions_list=[limb_score],
                            return_dataframe=True)
    assert frame.limb_score.tolist() == ([9, 2, 8] if mapping else [11, 11, 3])


def test_module_string_dispatch_supports_custom_limb_function(neuron, monkeypatch):
    @ns.run_options(run_type='Limb')
    def custom_limb_ns(curr_limb, **kwargs):
        return 42

    monkeypatch.setattr(ns, 'custom_limb_ns', custom_limb_ns, raising=False)
    frame = ns.query_neuron(neuron, 'custom == 42', functions_list=['custom_limb_ns'],
                            return_dataframe=True)
    assert 'custom' in frame and 'custom_limb_ns' not in frame
    assert frame.custom.tolist() == [42, 42, 42]


def test_limb_mapping_requires_all_nodes_even_for_restriction(neuron):
    @ns.run_options(run_type='Limb')
    def incomplete(curr_limb, **kwargs):
        return {9: 1}

    with pytest.raises(Exception, match='not a scalar nor did it match the keys of the limb branches'):
        ns.query_neuron(neuron, 'incomplete > 0', functions_list=[incomplete],
                        limb_branch_dict_restriction={'L9': [9]})


def test_limb_callback_mapping_is_validated_against_replaced_graph_node(neuron):
    returns = {}

    @ns.run_options(run_type='Limb')
    def replace_limb(curr_limb, limb_name, **kwargs):
        assert isinstance(curr_limb, Limb)
        replacement = object.__new__(Limb)
        replacement.concept_network = xu.GraphOrderedEdges()
        replacement.concept_network.add_node(99, data=SimpleNamespace())
        neuron.concept_network.nodes[limb_name]['data'] = replacement
        returns[limb_name] = {99: 42}
        return returns[limb_name]

    result = ns.apply_function_to_neuron(neuron, replace_limb)
    assert list(result) == ['L9', 'L2']
    assert result == {'L9': {99: 42}, 'L2': {99: 42}}
    assert all(result[name] is returns[name] for name in result)


def test_restrictions_limit_evaluation_and_preserve_requested_row_order(neuron):
    seen = []

    def score(curr_branch, limb_name, branch_name, **kwargs):
        seen.append((limb_name, branch_name, kwargs.copy()))
        return curr_branch.width

    kwargs = {'marker': 17}
    restriction = {'L9': [2, 9, 2]}
    frame = ns.query_neuron(neuron, 'score >= 0', functions_list=[score],
                            limbs_to_process=[9], limb_branch_dict_restriction=restriction,
                            function_kwargs=kwargs, return_dataframe=True)
    assert frame[['limb', 'node']].values.tolist() == [['L9', 2], ['L9', 9], ['L9', 2]]
    assert [(limb, node) for limb, node, _ in seen] == [('L9', 9), ('L9', 2)]
    assert kwargs == {'marker': 17, 'limbs_to_process': ['L9'],
                      'limb_branch_dict_restriction': restriction}
    assert all(context == kwargs for _, _, context in seen)
    assert kwargs['limb_branch_dict_restriction'] is restriction


def test_grouped_restriction_intersection_deduplicates_but_dataframe_keeps_duplicates(neuron):
    restriction = {'L9': [9, 2, 9]}
    result = ns.query_neuron(neuron, 'width >= 0', functions_list=['width'],
                            limb_branch_dict_restriction=restriction)
    assert_grouped(result, {'L9': [2, 9]})


def test_function_kwargs_restriction_keeps_existing_grouped_duplicates(neuron):
    result = ns.query_neuron(neuron, 'width >= 0', functions_list=['width'],
                            function_kwargs={'limb_branch_dict_restriction': {'L9': [9, 2, 9]}})
    assert_grouped(result, {'L9': [2, 9, 9]})


@pytest.mark.parametrize('flags, expected_nodes', [
    ({'return_dataframe_before_filtering': True}, [9, 2, 8]),
    ({'return_dataframe': True}, [9, 8]),
    ({'return_dataframe_before_filtering': True, 'return_dataframe': True,
      'return_limbs': True, 'return_limb_grouped_branches': False}, [9, 2, 8]),
    ({'return_dataframe': True, 'return_limbs': True,
      'return_limb_grouped_branches': False}, [9, 8]),
])
def test_dataframe_flags_and_priority(neuron, flags, expected_nodes):
    result = ns.query_neuron(neuron, 'width > 100', functions_list=['width'], **flags)
    assert isinstance(result, pd.DataFrame)
    assert result.node.tolist() == expected_nodes


def test_before_filtering_does_not_parse_invalid_query(neuron):
    frame = ns.query_neuron(neuron, 'invalid query syntax !!!', functions_list=['width'],
                            return_dataframe_before_filtering=True)
    assert len(frame) == 3


@pytest.mark.parametrize('matches', [True, False])
def test_return_limbs_takes_priority_over_pair_mode(neuron, matches):
    result = ns.query_neuron(neuron, 'width > 100' if matches else 'width < 0',
                            functions_list=['width'], return_limbs=True,
                            return_limb_grouped_branches=False)
    assert result == (['L2', 'L9'] if matches else [])


def test_pair_mode_returns_ordered_limb_node_array(neuron):
    result = ns.query_neuron(neuron, 'width > 100', functions_list=['width'],
                            return_limb_grouped_branches=False)
    assert isinstance(result, np.ndarray) and result.shape == (2, 2)
    np.testing.assert_array_equal(result, np.array([['L9', 9], ['L2', 8]], dtype=object))
    assert result.dtype == object


def test_pair_mode_keeps_restriction_row_order_and_duplicates(neuron):
    result = ns.query_neuron(neuron, 'width >= 0', functions_list=['width'],
                            limb_branch_dict_restriction={'L9': [2, 9, 2]},
                            return_limb_grouped_branches=False)
    assert result.tolist() == [['L9', 2], ['L9', 9], ['L9', 2]]


@pytest.mark.parametrize('empty_input', [False, True])
@pytest.mark.parametrize('mode', ['grouped', 'limbs', 'pairs', 'dataframe', 'before'])
def test_empty_results_have_coherent_public_types(neuron, empty_input, mode):
    if empty_input:
        neuron.concept_network = xu.GraphOrderedEdges()
    flags = {'grouped': {}, 'limbs': {'return_limbs': True},
             'pairs': {'return_limb_grouped_branches': False},
             'dataframe': {'return_dataframe': True},
             'before': {'return_dataframe_before_filtering': True}}[mode]
    result = ns.query_neuron(neuron, 'width < 0', functions_list=['width'], **flags)
    if mode in ('dataframe', 'before'):
        assert isinstance(result, pd.DataFrame)
        assert list(result) == (['limb', 'node'] if empty_input else ['limb', 'node', 'width'])
        assert len(result) == (3 if mode == 'before' and not empty_input else 0)
    elif mode == 'pairs':
        assert isinstance(result, np.ndarray) and result.shape == (0, 2)
    else:
        assert result == ([] if mode == 'limbs' else {})


@pytest.mark.parametrize('value', [None, np.nan])
def test_nan_check_preserves_per_column_error_message_and_can_be_disabled(neuron, value):
    def missing(curr_branch, **kwargs):
        return value

    frame = ns.generate_neuron_dataframe(neuron, [missing], check_nans=False)
    expected = ('Some of the data in the dataframe were incomplete; '
                f'nans per column = {frame.isnull().sum()}')
    with pytest.raises(Exception) as error:
        ns.generate_neuron_dataframe(neuron, [missing])
    assert str(error.value) == expected
    unchecked = ns.query_neuron(neuron, 'node >= 0', functions_list=[missing],
                                check_nans=False, return_dataframe=True)
    assert unchecked.missing.isnull().tolist() == [True, True, True]


def test_branch_exception_is_propagated(neuron):
    marker = ValueError('controlled feature failure')

    def failure(curr_branch, **kwargs):
        raise marker

    with pytest.raises(ValueError) as error:
        ns.query_neuron(neuron, 'failure > 0', functions_list=[failure])
    assert error.value is marker


def test_unknown_attribute_and_invalid_feature_have_established_errors(neuron):
    with pytest.raises(AttributeError, match='not_a_feature'):
        ns.query_neuron(neuron, 'not_a_feature > 0', functions_list=['not_a_feature'])
    with pytest.raises(Exception, match='Function item 17 was not a string or callable function'):
        ns.query_neuron(neuron, 'width > 0', functions_list=[17])


def test_invalid_run_type_raises_even_for_empty_neuron(neuron):
    neuron.concept_network = xu.GraphOrderedEdges()

    @ns.run_options(run_type='invalid')
    def bad(curr_branch, **kwargs):
        return 1

    with pytest.raises(Exception, match='Function recieved was neither a Branch nor a Limb'):
        ns.apply_function_to_neuron(neuron, bad)


@pytest.mark.parametrize('name, expected', [('curr_branch', 'Branch'), ('curr_limb', 'Limb'),
                                          ('unknown', 'Branch')])
def test_run_type_inference_keeps_branch_default(name, expected):
    namespace = {}
    exec(f'def feature({name}, **kwargs): return 1', namespace)
    assert ns.get_run_type(namespace['feature']) == expected


def test_limb_conversion_keeps_complete_mapping_identity(neuron):
    graph = neuron['L9'].concept_network
    mapping = {2: 4, 9: 7}
    assert ns.convert_limb_function_return_to_dict(mapping, graph) is mapping
    assert ns.convert_limb_function_return_to_dict(3, graph) == {9: 3, 2: 3}


def test_map_values_preserves_dataframe_identity_index_and_object_values():
    frame = pd.DataFrame({'limb': ['L9', 'L9', 'L2'], 'node': [2, 9, 8]}, index=[7, 7, 3])
    values = {'L9': {2: [1, 2], 9: {'score': 4}}, 'L2': {8: np.array([5, 6])}}
    result = ns.map_new_limb_node_value(frame, values, 'values')
    assert result is frame and frame.index.tolist() == [7, 7, 3]
    assert frame['values'].iloc[0] is values['L9'][2]
    assert frame['values'].iloc[1] is values['L9'][9]
    assert frame['values'].iloc[2] is values['L2'][8]


def test_series_valued_feature_keeps_pandas_column_expansion_error():
    frame = pd.DataFrame({'limb': ['L9', 'L9'], 'node': [2, 9]})
    values = {'L9': {2: pd.Series([1, 2]), 9: pd.Series([3, 4])}}
    with pytest.raises(ValueError, match='Cannot set a DataFrame with multiple columns'):
        ns.map_new_limb_node_value(frame, values, 'score')
    assert list(frame) == ['limb', 'node']


def test_empty_restriction_skips_feature_calls_and_query_parsing(neuron):
    def forbidden(curr_branch, **kwargs):
        pytest.fail('an empty restriction should never evaluate a branch feature')

    result = ns.query_neuron(neuron, 'invalid query !!!', functions_list=[forbidden],
                            limb_branch_dict_restriction={}, return_dataframe=True)
    assert result.empty and list(result) == ['limb', 'node']


def test_direct_dataframe_generation_keeps_scalar_function_support(neuron):
    frame = ns.generate_neuron_dataframe(neuron, ns.width)
    assert frame.width.tolist() == [120, 80, 150]


def test_query_parser_keeps_existing_parenthesized_feature_set():
    names = ns.functions_list_from_query('(width > @minimum) and (n_faces_branch >= 12) and (width < 200)')
    assert set(names) == {'width', 'n_faces_branch'} and len(names) == 2


def test_empty_plot_message_preserved(neuron, capsys):
    assert ns.query_neuron(neuron, 'width < 0', functions_list=['width'],
                           plot_limb_branch_dict=True) == {}
    assert capsys.readouterr().out == 'No limb branch to plot\n'
