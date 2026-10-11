"""Exercise real stage orchestration with controlled geometry-provider boundaries.

No native skeletonizer is needed: the selected production helpers are compiled
from their source and given deterministic providers. The fixtures also support
the separate captured-source oracle without a committed artifact dependency.
"""
import ast
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import networkx as nx
import numpy as np
import pytest

SOURCE = Path(__file__).resolve().parents[2] / "neurd/preprocess_neuron.py"
HELPERS = {
    "_axis_origin_fingerprint", "_new_axis_origin", "_origin_local_id",
    "_SkeletonizationResult", "_SublimbDecompositionResult", "_SublimbGrouping",
    "_new_invalidation_d", "_run_skeletonization_pass", "_decide_next_limb_cfg",
    "_skeletonize_with_adaptive_invalidation", "_group_into_map_mp_sublimbs",
    "_decompose_map_sublimbs", "_decompose_mp_sublimbs", "_merge_map_mp_correspondence",
    "preprocess_limb",
}
CONSTANTS = {
    "meshparty_segment_size", "prevent_MP_starter_branch_stitches",
    "check_correspondence_branches",
}


def load_helpers(source=SOURCE):
    nodes = []
    for node in ast.parse(Path(source).read_text()).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in HELPERS:
            nodes.append(node)
        elif isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id in CONSTANTS for target in node.targets):
            nodes.append(node)
    namespace = dict(np=np, nx=nx, dataclass=dataclass, NamedTuple=NamedTuple,
                     deepcopy=deepcopy, time=SimpleNamespace(time=lambda: 0.))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
    return namespace


def result(namespace, name, values):
    """Supply the stage contract used by either the production or captured source."""
    factory = namespace.get(name, tuple)
    return tuple(values) if factory is tuple else factory(*values)


def axis(label):
    return np.asarray([[[label * 100., 0., 0.], [label * 100. + 10., 0., 0.]]])


class Mesh:
    def __init__(self, label, calls=None):
        self.label = label
        self.calls = [] if calls is None else calls

    def submesh(self, groups, **kwargs):
        indices = np.asarray(groups[0])
        self.calls.append(("submesh", indices.copy(), kwargs.copy()))
        return Mesh(tuple(indices.tolist()))


def grouping_case(source, route):
    namespace = load_helpers(source)
    meshes = np.asarray([Mesh(index) for index in range(4)], dtype=object)
    indices = np.empty(4, dtype=object)
    for index in range(4):
        indices[index] = np.asarray([10 * index + 2, 10 * index + 1], dtype=np.int64)
    axes = np.asarray([axis(index) for index in range(4)])
    widths = np.asarray([100., 200., 300., 400.])
    mesh = Mesh("limb")
    calls = mesh.calls
    enabled, width_threshold, size_threshold = True, 250., 0
    if route == "disabled":
        enabled = False
    elif route == "width_equal":
        width_threshold = 400.
    elif route == "size_equal":
        size_threshold = 4
    elif route == "small_map":
        size_threshold = 5
    elif route == "map_only":
        width_threshold = 0.

    def mesh_connectivity(**kwargs):
        calls.append(("mesh_connectivity", kwargs))
        return [(index, index + 1) for index in range(len(kwargs["meshes"]) - 1)]

    def skeleton_connectivity(**kwargs):
        calls.append(("skeleton_connectivity", kwargs))
        return []  # Retain two disconnected MP sublimbs in the mixed route.

    namespace["tu"] = SimpleNamespace(mesh_list_connectivity=mesh_connectivity)
    namespace["sk"] = SimpleNamespace(skeleton_list_connectivity=skeleton_connectivity)
    namespace["submesh_ops"] = SimpleNamespace(SubMesh=lambda mesh, idx, parent:
                                               SimpleNamespace(mesh=mesh, face_idx=idx, parent=parent))
    output = namespace["_group_into_map_mp_sublimbs"](
        axes, meshes, indices, widths, mesh, enabled, width_threshold, size_threshold)
    return dict(namespace=namespace, output=output, mesh=mesh, meshes=meshes,
                indices=indices, axes=axes, widths=widths, calls=calls)


@pytest.mark.parametrize("route", ["disabled", "width_equal", "size_equal", "small_map"])
def test_no_map_fallback_retains_original_payload_even_for_disconnected_axes(route):
    case = grouping_case(SOURCE, route)
    output = case["output"]
    assert output.MAP_flag is False and output.map_subs == []
    assert output.sublimb_meshes_MP == [case["mesh"]]
    assert output.sublimb_mesh_branches_MP[0] is case["meshes"]
    assert output.sublimb_mesh_idx_branches_MP[0] is case["indices"]
    assert output.sublimb_skeleton_branches[0] is case["axes"]
    assert output.widths_MP[0] is case["widths"]
    assert all(call[0] != "skeleton_connectivity" for call in case["calls"])


@pytest.mark.parametrize("route", ["mixed", "map_only"])
def test_map_grouping_keeps_component_order_limb_indices_and_axis_mesh_pairs(route):
    case = grouping_case(SOURCE, route)
    output = case["output"]
    assert output.MAP_flag is True
    assert len(output.map_subs) == 1
    expected_map = [22, 21, 32, 31] if route == "mixed" else [2, 1, 12, 11, 22, 21, 32, 31]
    np.testing.assert_array_equal(output.map_subs[0].face_idx, expected_map)
    assert output.map_subs[0].parent is case["mesh"]
    assert output.map_subs[0].mesh.label == tuple(expected_map)
    expected_mp = [0, 1] if route == "mixed" else []
    assert len(output.sublimb_meshes_MP) == len(expected_mp)
    for position, branch in enumerate(expected_mp):
        np.testing.assert_array_equal(output.sublimb_skeleton_branches[position], case["axes"][[branch]])
        np.testing.assert_array_equal(output.widths_MP[position], case["widths"][[branch]])
        assert output.sublimb_mesh_idx_branches_MP[position][0] is case["indices"][branch]
        assert output.sublimb_mesh_branches_MP[position][0].label == tuple(case["indices"][branch])
        assert output.sublimb_meshes_MP[position].label == tuple(case["indices"][branch])
    for kind, _, kwargs in [call for call in case["calls"] if call[0] == "submesh"]:
        assert kwargs == dict(append=True, repair=False)


def adaptive_case(source, route, late_change=False):
    namespace = load_helpers(source)
    params = SimpleNamespace(invalidation_d_axon=150., smooth_neighborhood_axon=5,
        combine_close_skeleton_nodes_threshold_meshparty_axon=25,
        filter_end_node_length_meshparty_axon=35, filter_end_node_length_axon=77)
    namespace["parameters"] = SimpleNamespace(params=params)
    limb = dict(invalidation_d=1200., smooth_neighborhood=3,
        combine_close_skeleton_nodes_threshold_meshparty=10,
        filter_end_node_length_meshparty=20, filter_end_node_length=999,
        use_meshafterparty=True)
    neuron = dict(verbose=False, use_adaptive_invalidation_d=route != "disabled",
        width_threshold_MAP=1000., axon_width_preprocess_limb_max=100.,
        mp_only_revised_invalidation_d=True, mp_only_invalidation_d_axon_buffer=20.,
        mp_only_revised_width_reference=600., mp_only_revised_invalidation_d_reference=900.)
    widths = np.asarray([50., 60.] if route == "axon" else
                        [200., 400.] if route == "revised_mp" else [900., 1100.])
    median = 55. if route == "axon" else 300. if route == "revised_mp" else 1200.
    mesh, root = Mesh("limb"), np.asarray([1., 2., 3.])
    calls, outputs = [], []

    def skeletonize(actual_mesh, **kwargs):
        assert actual_mesh is mesh and kwargs["root"] is root
        calls.append(("skeletonize", kwargs.copy()))
        return f"skeleton-{len(outputs)}"

    def branches(skeleton, **kwargs):
        calls.append(("branches", skeleton, kwargs.copy()))
        values = (np.asarray([axis(10 * len(outputs) + index) for index in range(2)]),
                  np.asarray([Mesh(10 * len(outputs) + index) for index in range(2)], dtype=object),
                  np.asarray([[2, 1], [4, 3]]), widths)
        outputs.append(values)
        if late_change and len(outputs) == 1:
            params.invalidation_d_axon = 175.
            params.smooth_neighborhood_axon = 8
        return values

    def weighted(actual_widths, actual_axes):
        assert actual_widths is outputs[0][3] and actual_axes is outputs[0][0]
        calls.append(("weighted_width",))
        return median

    namespace["cb"] = SimpleNamespace(skeletonize_largest_component=skeletonize,
        skeleton_to_branches=branches, weighted_width_median=weighted)
    original_limb, original_neuron = deepcopy(limb), deepcopy(neuron)
    output = namespace["_skeletonize_with_adaptive_invalidation"](mesh, root, 100, neuron, limb)
    assert limb == original_limb and neuron == original_neuron
    return dict(output=output, outputs=outputs, calls=calls, params=params, limb=limb)


@pytest.mark.parametrize("route,passes", [("disabled", 1), ("final", 1), ("axon", 2), ("revised_mp", 2)])
def test_adaptive_skeletonization_preserves_provider_payload_and_pass_limit(route, passes):
    case = adaptive_case(SOURCE, route)
    assert len(case["outputs"]) == passes
    assert isinstance(case["output"], tuple)
    assert case["output"]._fields == ("segment_branches", "divided_submeshes", "divided_submeshes_idx", "segment_widths_median")
    assert all(actual is expected for actual, expected in zip(case["output"], case["outputs"][-1]))
    skeleton_calls = [call[1] for call in case["calls"] if call[0] == "skeletonize"]
    assert skeleton_calls[0]["invalidation_d"] == 1200.
    assert skeleton_calls[0]["smooth_neighborhood"] == 3
    assert skeleton_calls[0]["filter_mesh"] is False
    assert sum(call[0] == "weighted_width" for call in case["calls"]) == (route != "disabled")
    if route == "axon":
        assert skeleton_calls[1]["invalidation_d"] == 150.
        assert skeleton_calls[1]["smooth_neighborhood"] == 5
        branch_calls = [call[2] for call in case["calls"] if call[0] == "branches"]
        assert branch_calls[1]["combine_close_skeleton_nodes_threshold"] == 25
        assert branch_calls[1]["filter_end_node_length"] == 35
    elif route == "revised_mp":
        assert skeleton_calls[1]["invalidation_d"] == 462.
        assert skeleton_calls[1]["smooth_neighborhood"] == 3


def test_adaptive_decision_reads_current_global_axon_settings_after_first_pass():
    case = adaptive_case(SOURCE, "axon", late_change=True)
    second = [call[1] for call in case["calls"] if call[0] == "skeletonize"][1]
    assert second["invalidation_d"] == 175.
    assert second["smooth_neighborhood"] == 8


def map_case(source, endpoints):
    namespace = load_helpers(source)
    mesh = Mesh("map")
    entry = dict(branch_skeleton=axis(1), branch_mesh=mesh,
                 branch_face_idx=np.asarray([4, 1]), width_from_skeleton=np.float64(10.))
    soma_touching = {3: [np.asarray([[1., 2., 3.]])]}
    calls = []

    def decompose(**kwargs):
        calls.append(kwargs)
        return {7: entry}, endpoints, soma_touching

    namespace["_decompose_map_piece"] = decompose
    options = dict(filter_end_node_length=123, perform_cleaning_checks=True,
        combine_close_skeleton_nodes=True, combine_close_skeleton_nodes_threshold=700,
        use_surface_after_CGAL=True, surface_reconstruction_size=99,
        remove_mesh_interior_face_threshold=20, error_on_bad_cgal_return=False,
        max_stitch_distance_CGAL=55, distance_by_mesh_center=True)
    output = namespace["_decompose_map_sublimbs"]([mesh], soma_touching, **options)
    return dict(output=output, entry=entry, endpoints=endpoints, touching=soma_touching,
                calls=calls, options=options, mesh=mesh)


@pytest.mark.parametrize("endpoints", [None, {}, {3: [np.asarray([1., 2., 3.])]}])
def test_map_collector_distinguishes_missing_and_empty_endpoint_additions(endpoints):
    case = map_case(SOURCE, endpoints)
    correspondence, added_endpoints, added_touching = case["output"]
    assert case["output"]._fields == ("correspondence", "endpoints_additions", "soma_touching_additions")
    assert correspondence[0][7] is case["entry"]
    assert case["calls"][0]["map_sub"] is case["mesh"]
    assert case["calls"][0]["soma_touching_vertices_dict"] is case["touching"]
    for key, value in case["options"].items():
        assert case["calls"][0][key] == value
    if endpoints is None:
        assert added_endpoints == added_touching == []
    else:
        assert added_endpoints[0] is endpoints
        assert added_touching[0] is case["touching"]
    assert case["entry"]["mesh_axis_origin"]["face_map_origin"]["index_scope"] == "preprocess_limb.input_mesh.face_index"


def mp_case(source, touching, map_flag=False, count=1):
    namespace = load_helpers(source)
    meshes = [Mesh(index) for index in range(count)]
    axes = [[axis(1), axis(2)] for _ in meshes]
    branch_meshes = [[Mesh(10), Mesh(20)] for _ in meshes]
    indices = [[np.asarray([2]), np.asarray([0, 1])] for _ in meshes]
    widths = [[10., 20.] for _ in meshes]
    limb = Mesh("limb")
    calls, fixed_payloads = [], []

    def filter_touching(**kwargs):
        calls.append(("filter", kwargs))
        return touching

    def fix(*args):
        calls.append(("fix", args))
        assert args[4] is touching and args[5] is limb
        fixed_payloads.append(args[:4])
        return (*args[:4], {}, {}, True)

    namespace["filter_soma_touching_vertices_dict_by_mesh"] = filter_touching
    namespace["_fix_mp_soma_extension"] = fix
    output = namespace["_decompose_mp_sublimbs"](
        meshes, axes, branch_meshes, indices, widths, map_flag, touching, limb,
        print_fusion_steps=False, fusion_time=0.)
    return dict(output=output, calls=calls, indices=indices, branch_meshes=branch_meshes,
                axes=axes, widths=widths, fixed_payloads=fixed_payloads)


@pytest.mark.parametrize("touching", [None, {}])
@pytest.mark.parametrize("map_flag,count", [(False, 1), (True, 1), (False, 2)])
def test_mp_collector_keeps_fast_path_and_none_empty_soma_distinction(touching, map_flag, count):
    case = mp_case(SOURCE, touching, map_flag, count)
    correspondence, endpoints, added_touching = case["output"]
    assert isinstance(case["output"], tuple)
    assert case["output"].correspondence is correspondence
    expected_filters = count if touching is not None and (map_flag or count != 1) else 0
    assert sum(call[0] == "filter" for call in case["calls"]) == expected_filters
    assert sum(call[0] == "fix" for call in case["calls"]) == (0 if touching is None else count)
    assert len(endpoints) == len(added_touching) == (0 if touching is None else count)
    for sublimb, branches in correspondence.items():
        for branch, entry in branches.items():
            assert entry["branch_mesh"] is case["branch_meshes"][sublimb][branch]
            assert entry["branch_face_idx"] is case["indices"][sublimb][branch]
            assert entry["width_from_skeleton"] == case["widths"][sublimb][branch]
            np.testing.assert_array_equal(entry["branch_skeleton"], case["axes"][sublimb][branch])
            assert entry["mesh_axis_origin"]["face_map_origin"]["index_scope"] == "preprocess_limb.input_mesh.face_index"


def public_case(source, route, return_network, return_starts, late_change=False):
    namespace = load_helpers(source)
    params = SimpleNamespace(filter_end_node_length=123, surface_reconstruction_size=99,
        remove_mesh_interior_face_threshold=20, max_stitch_distance_CGAL=55)
    namespace["parameters"] = SimpleNamespace(params=params)
    mesh = Mesh("limb")
    soma_touching = {3: [np.asarray([[1., 2., 3.]])]}
    neuron_cfg = dict(width_threshold_MAP=250., size_threshold_MAP=10)
    limb_cfg = dict(use_meshafterparty=True)
    axes, branch_meshes, indices, widths = [axis(0)], [Mesh(0)], [np.asarray([4, 1])], [10.]
    map_entry, mp_entry = {"label": "map"}, {"label": "mp"}
    map_correspondence = {2: {7: map_entry}} if route != "mp_only" else {}
    mp_correspondence = {8: {3: mp_entry}} if route != "map_only" else {}
    group = namespace["_SublimbGrouping"](
        [Mesh("map")] if map_correspondence else [], [Mesh("mp")] if mp_correspondence else [],
        [branch_meshes], [indices], [axes], [widths], bool(map_correspondence))
    map_endpoints = {3: [np.asarray([1., 2., 3.])]}
    mp_endpoints = {3: [np.asarray([4., 5., 6.])]}
    map_touching, mp_touching = {3: [np.asarray([[7., 8., 9.]])]}, {3: [np.asarray([[10., 11., 12.]])]}
    raw_starts, clean_starts, networks = {"label": "raw"}, {"label": "clean"}, {"label": "networks"}
    events = []

    def skeletonize(*args, **kwargs):
        events.append(("skeletonize", args, kwargs))
        if late_change:
            params.filter_end_node_length = 999
            params.surface_reconstruction_size = 888
            params.remove_mesh_interior_face_threshold = 777
            params.max_stitch_distance_CGAL = 666
        return result(namespace, "_SkeletonizationResult", (axes, branch_meshes, indices, widths))

    def grouping(*args):
        events.append(("group", args))
        return group

    def decompose_map(*args, **kwargs):
        events.append(("map", args, kwargs))
        return result(namespace, "_SublimbDecompositionResult", (
            map_correspondence, [map_endpoints] if map_correspondence else [],
            [map_touching] if map_correspondence else []))

    def decompose_mp(*args, **kwargs):
        events.append(("mp", args, kwargs))
        return result(namespace, "_SublimbDecompositionResult", (
            mp_correspondence, [mp_endpoints] if mp_correspondence else [],
            [mp_touching] if mp_correspondence else []))

    def stitch(*args):
        events.append(("stitch", args))
        return args[:2]

    def rearrange(*args):
        events.append(("rearrange", args))
        return raw_starts

    def clean(*args):
        events.append(("clean", args))
        return clean_starts

    def filter_end_nodes(**kwargs):
        events.append(("filter_end_nodes", kwargs))
        return kwargs["limb_correspondence"]

    def concept_networks(*args, **kwargs):
        events.append(("networks", args, kwargs))
        return networks

    namespace.update(_skeletonize_with_adaptive_invalidation=skeletonize,
        _group_into_map_mp_sublimbs=grouping, _decompose_map_sublimbs=decompose_map,
        _decompose_mp_sublimbs=decompose_mp, _stitch_map_and_mp=stitch,
        _rearrange_network_starting_info=rearrange, _clean_network_starting_info=clean,
        filter_limb_correspondence_for_end_nodes=filter_end_nodes,
        calculate_limb_concept_networks=concept_networks,
        sk=SimpleNamespace(check_correspondence_branches_have_2_endpoints=lambda value:
                           events.append(("check", value))))
    output = namespace["preprocess_limb"](mesh, neuron_cfg, limb_cfg, soma_touching,
        return_concept_network=return_network, return_concept_network_starting_info=return_starts,
        error_on_no_starting_coordinates=False)
    assert neuron_cfg == dict(width_threshold_MAP=250., size_threshold_MAP=10)
    assert limb_cfg == dict(use_meshafterparty=True)
    return dict(output=output, events=events, group=group, axes=axes,
        branch_meshes=branch_meshes, indices=indices, widths=widths, mesh=mesh,
        neuron_cfg=neuron_cfg, limb_cfg=limb_cfg, soma_touching=soma_touching,
        map_entry=map_entry, mp_entry=mp_entry, map_endpoints=map_endpoints,
        mp_endpoints=mp_endpoints, map_touching=map_touching, mp_touching=mp_touching,
        clean_starts=clean_starts, raw_starts=raw_starts, networks=networks)


@pytest.mark.parametrize("route", ["map_only", "mp_only", "mixed"])
@pytest.mark.parametrize("return_network,return_starts", [(False, False), (False, True), (True, False), (True, True)])
def test_public_limb_returns_preserve_flags_stage_order_payloads_and_filter_options(route, return_network, return_starts):
    case = public_case(SOURCE, route, return_network, return_starts)
    output = case["output"]
    if return_network or return_starts:
        assert type(output) is tuple and len(output) == 2
        correspondence, extra = output
        assert extra is (case["networks"] if return_network else case["clean_starts"])
    else:
        assert type(output) is dict
        correspondence = output
    assert list(correspondence) == list(range(len(correspondence)))
    expected_entries = ([case["map_entry"]] if route != "mp_only" else []) + ([case["mp_entry"]] if route != "map_only" else [])
    assert len(correspondence) == len(expected_entries)
    assert all(actual is expected for actual, expected in zip(correspondence.values(), expected_entries))
    events = case["events"]
    names = [event[0] for event in events]
    expected_names = ["skeletonize", "group", "map", "mp", "check", "check"]
    if route == "mixed":
        expected_names.append("stitch")
    expected_names += ["rearrange", "clean", "filter_end_nodes"]
    if return_network:
        expected_names.append("networks")
    assert names == expected_names
    by_name = {event[0]: event[1:] for event in events}
    args, kwargs = by_name["skeletonize"]
    assert args[0] is case["mesh"]
    np.testing.assert_array_equal(args[1], [1., 2., 3.])
    assert kwargs["neuron_cfg"] is case["neuron_cfg"] and kwargs["limb_cfg"] is case["limb_cfg"]
    assert kwargs["meshparty_segment_size"] == 100
    assert len(by_name["group"][0]) == 8
    assert all(actual is expected for actual, expected in zip(by_name["group"][0][:4],
                                                            (case["axes"], case["branch_meshes"], case["indices"], case["widths"])))
    assert by_name["group"][0][4] is case["mesh"]
    assert by_name["group"][0][5:] == (True, 250., 10)
    assert by_name["map"][0][0] is case["group"].map_subs
    mp_args, mp_options = by_name["mp"]
    assert len(mp_args) == 8
    expected_mp_args = (case["group"].sublimb_meshes_MP,
        case["group"].sublimb_skeleton_branches, case["group"].sublimb_mesh_branches_MP,
        case["group"].sublimb_mesh_idx_branches_MP, case["group"].widths_MP,
        case["group"].MAP_flag, case["soma_touching"], case["mesh"])
    assert all(actual is expected for actual, expected in zip(mp_args, expected_mp_args))
    assert mp_options == dict(print_fusion_steps=True, fusion_time=0.)
    expected_endpoints = ([case["map_endpoints"]] if route != "mp_only" else []) + ([case["mp_endpoints"]] if route != "map_only" else [])
    expected_touching = ([case["map_touching"]] if route != "mp_only" else []) + ([case["mp_touching"]] if route != "map_only" else [])
    touching, endpoints, source_touching = by_name["rearrange"][0]
    assert len(endpoints) == len(expected_endpoints)
    assert len(touching) == len(expected_touching)
    assert all(actual is expected for actual, expected in zip(endpoints, expected_endpoints))
    assert all(actual is expected for actual, expected in zip(touching, expected_touching))
    assert source_touching is case["soma_touching"]
    options = by_name["filter_end_nodes"][0]
    assert options["limb_correspondence"] is correspondence
    assert options["mesh"] is case["mesh"]
    assert options["starting_info"] is case["clean_starts"]
    assert options["filter_end_node_length"] == 123
    assert options["error_on_no_starting_coordinates"] is False
    assert options["error_on_starting_coordinates_not_endnodes"] is False
    if route == "mixed":
        np.testing.assert_array_equal(by_name["stitch"][0][3], [[1., 2., 3.], [4., 5., 6.]])
    if return_network:
        assert by_name["networks"][0][0] is correspondence
        assert by_name["networks"][0][1] is case["clean_starts"]
        assert by_name["networks"][1] == dict(run_concept_network_checks=True)


def test_limb_map_and_final_filter_use_settings_captured_before_skeletonization():
    case = public_case(SOURCE, "mixed", False, True, late_change=True)
    events = {event[0]: event[1:] for event in case["events"]}
    map_options = events["map"][1]
    assert map_options["filter_end_node_length"] == 123
    assert map_options["surface_reconstruction_size"] == 99
    assert map_options["remove_mesh_interior_face_threshold"] == 20
    assert map_options["max_stitch_distance_CGAL"] == 55
    assert events["filter_end_nodes"][0]["filter_end_node_length"] == 123
