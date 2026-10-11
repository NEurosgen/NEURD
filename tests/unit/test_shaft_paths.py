"""Full ordered path and exception comparisons; no native NEURD invocation."""
import itertools
import random

import networkx as nx
import numpy as np
import pytest

def ExactBidirectionalPaths(graph):
    # Numba is optional: kernel comparisons skip cleanly in an environment
    # without it; the production import/fallback guard tests below still run.
    pytest.importorskip("numba")
    from neurd.shaft_paths import ExactBidirectionalPaths as implementation
    return implementation(graph)


def outcome(function, source, target):
    try:
        return ("path", function(source, target))
    except (nx.NodeNotFound, nx.NetworkXNoPath) as error:
        return (type(error).__name__, str(error))


def compare_queries(graph, pairs):
    prototype = ExactBidirectionalPaths(graph)
    expected = lambda source, target: nx.shortest_path(graph, source, target)
    for source, target in pairs:
        assert outcome(prototype.shortest_path, source, target) == outcome(expected, source, target)


@pytest.mark.parametrize("node_count", range(6))
def test_exhaustive_simple_graphs_all_ordered_pairs(node_count):
    edges = list(itertools.combinations(range(node_count), 2))
    pairs = list(itertools.product(range(node_count), repeat=2))
    pairs += [("missing", 0), (0, "missing"), ("missing", "missing")]
    for mask in range(1 << len(edges)):
        present = [edge for idx, edge in enumerate(edges) if mask & (1 << idx)]
        # Insertion order determines tie paths; test both deterministic orders.
        for reverse in (False, True):
            graph = nx.Graph()
            graph.add_nodes_from(reversed(range(node_count)) if reverse else range(node_count))
            graph.add_edges_from(reversed(present) if reverse else present)
            compare_queries(graph, pairs)


@pytest.mark.parametrize("seed", range(30))
def test_random_insertion_orders_full_paths(seed):
    rng = random.Random(seed)
    for case in range(8):
        count = rng.randrange(2, 76)
        labels = [(f"group-{seed}-{case}", idx) for idx in range(count)]
        nodes = labels.copy()
        rng.shuffle(nodes)
        probability = rng.choice([0.02, 0.1, 0.25, 0.7])
        edges = [pair for pair in itertools.combinations(labels, 2) if rng.random() < probability]
        # Self loops are allowed in a simple undirected NetworkX Graph.
        edges += [(node, node) for node in rng.sample(labels, rng.randrange(min(count, 6)))]
        rng.shuffle(edges)
        graph = nx.Graph()
        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges)
        pairs = [(rng.choice(labels), rng.choice(labels)) for _ in range(80)]
        compare_queries(graph, pairs)


@pytest.mark.parametrize("reverse_edges", [False, True])
def test_diamond_cycle_and_frontier_size_ties(reverse_edges):
    edges = [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t"),
             ("t", "c"), ("c", "d"), ("d", "s"), ("b", "b")]
    graph = nx.Graph()
    graph.add_edges_from(reversed(edges) if reverse_edges else edges)
    compare_queries(graph, itertools.product(graph, repeat=2))


def test_sparse_disconnected_missing_and_unhashable_nodes():
    graph = nx.Graph()
    graph.add_nodes_from([9001, -72, "isolated", ("tuple", 2)])
    graph.add_edge(9001, -72)
    pairs = list(itertools.product(graph, repeat=2))
    pairs += [("absent", 9001), (9001, "absent"), ([], 9001), (9001, [])]
    compare_queries(graph, pairs)


def test_missing_none_endpoint_matches_underlying_bidirectional_algorithm():
    # nx.shortest_path uses None as its all-sources/all-targets dispatch sentinel;
    # this endpoint-only object implements its bidirectional branch, not that API.
    graph = nx.path_graph(3)
    prototype = ExactBidirectionalPaths(graph)
    expected = lambda source, target: nx.bidirectional_shortest_path(graph, source, target)
    for source, target in [(None, 0), (0, None), (None, None)]:
        assert outcome(prototype.shortest_path, source, target) == outcome(expected, source, target)


def test_edge_weight_attributes_are_ignored_when_weight_is_none():
    graph = nx.Graph()
    graph.add_edge("s", "t", weight=500)
    graph.add_edge("s", "a", weight=0)
    graph.add_edge("a", "t", weight=0)
    prototype = ExactBidirectionalPaths(graph)
    assert prototype.shortest_path("s", "t") == nx.shortest_path(graph, "s", "t") == ["s", "t"]
    with pytest.raises(TypeError, match="weight=None"):
        prototype.shortest_path("s", "t", weight="weight")


@pytest.mark.parametrize("graph", [nx.DiGraph(), nx.MultiGraph(), nx.MultiDiGraph(), object()])
def test_unsupported_graphs_fail_explicitly(graph):
    with pytest.raises(TypeError):
        ExactBidirectionalPaths(graph)


def test_adjacency_order_csr_snapshot_and_no_graph_mutation():
    graph = nx.Graph()
    graph.add_edges_from([("s", "b"), ("s", "a"), ("a", "t"), ("b", "t")])
    graph_snapshot = {node: tuple(graph.adj[node]) for node in graph}
    prototype = ExactBidirectionalPaths(graph)
    for idx, node in enumerate(prototype.nodes):
        ids = prototype.neighbors[prototype.indptr[idx]:prototype.indptr[idx + 1]]
        assert tuple(prototype.nodes[int(value)] for value in ids) == graph_snapshot[node]
    compare_queries(graph, itertools.product(graph, repeat=2))
    assert graph_snapshot == {node: tuple(graph.adj[node]) for node in graph}
    before = prototype.shortest_path("s", "t")
    graph.add_edge("s", "t")
    assert prototype.shortest_path("s", "t") == before
    assert nx.shortest_path(graph, "s", "t") != before
    assert not prototype.indptr.flags.writeable
    assert not prototype.neighbors.flags.writeable


def test_epoch_rollover_preserves_exact_paths():
    graph = nx.cycle_graph(12)
    prototype = ExactBidirectionalPaths(graph)
    for source, target in itertools.product(graph, repeat=2):
        assert prototype.shortest_path(source, target) == nx.shortest_path(graph, source, target)
    prototype._epoch = np.iinfo(np.int64).max
    for source, target in itertools.product(graph, repeat=2):
        assert prototype.shortest_path(source, target) == nx.shortest_path(graph, source, target)


def test_python_and_numpy_random_states_are_unchanged():
    graph = nx.cycle_graph(15)
    numpy_before = np.random.get_state()
    python_before = random.getstate()
    prototype = ExactBidirectionalPaths(graph)
    compare_queries(graph, itertools.product(graph, repeat=2))
    for source, target in itertools.product(graph, repeat=2):
        prototype.shortest_path(source, target)
    numpy_after = np.random.get_state()
    assert numpy_before[0] == numpy_after[0]
    assert np.array_equal(numpy_before[1], numpy_after[1])
    assert numpy_before[2:] == numpy_after[2:]
    assert python_before == random.getstate()


def production_guard():
    """Compile the actual selection seam without importing the large detector."""
    import ast
    from pathlib import Path

    source = Path(__file__).resolve().parents[2] / "neurd" / "spine_utils.py"
    function = next(node for node in ast.parse(source.read_text()).body
                    if isinstance(node, ast.FunctionDef)
                    and node.name == "get_spine_meshes_unfiltered_from_mesh")
    phase = next(node for node in ast.walk(function) if isinstance(node, ast.With)
                 and isinstance(node.items[0].context_expr, ast.Call)
                 and node.items[0].context_expr.args
                 and isinstance(node.items[0].context_expr.args[0], ast.Constant)
                 and node.items[0].context_expr.args[0].value == "shaft_path_expansion")
    start = next(idx for idx, node in enumerate(phase.body)
                 if isinstance(node, ast.Import) and node.names[0].name == "os")
    end = next(idx for idx, node in enumerate(phase.body)
               if isinstance(node, ast.Assign)
               and any(isinstance(target, ast.Name) and target.id == "final_shaft_mesh_names"
                       for target in node.targets))
    return compile(ast.Module(body=phase.body[start:end], type_ignores=[]), str(source), "exec")


def run_guard(graph, shaft_count, method="path_to_all_shaft_mesh", timing=None):
    names = [f"b{idx}" for idx in range(shaft_count)]
    namespace = dict(nx=nx, spine_graph=graph, shaft_mesh_names=names,
                     non_biggest_shaft=names[1:], shaft_expansion_method=method,
                     path_timing=timing, __package__="neurd")
    exec(production_guard(), namespace)
    return namespace


def forbid_helper_import(monkeypatch):
    import builtins

    original_import = builtins.__import__

    def checked_import(name, *args, **kwargs):
        if name == "shaft_paths" or name == "neurd.shaft_paths":
            raise AssertionError("The optional helper must not be imported on this path")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", checked_import)


@pytest.mark.parametrize("shaft_count, method, expected_queries", [
    (1, "path_to_all_shaft_mesh", 0),
    (32, "path_to_all_shaft_mesh", 992),
    (1024, "path_to_largest_shaft_mesh", 1023),
    (33, "unknown_method", 0),
])
def test_small_query_batches_never_import_helper(monkeypatch, shaft_count, method, expected_queries):
    monkeypatch.delenv("NEURD_LEGACY_SHAFT_PATHS", raising=False)
    monkeypatch.setattr(nx, "__version__", "3.6.1")
    forbid_helper_import(monkeypatch)
    timing = {}
    selected = run_guard(nx.path_graph(["b0", "b1"]), shaft_count, method, timing)
    assert timing == dict(path_backend="networkx", path_backend_reason="small_query_batch",
                          path_queries=expected_queries)
    assert selected["shortest_path"]("b0", "b1") == ["b0", "b1"]


def test_legacy_flag_never_imports_helper(monkeypatch):
    monkeypatch.setenv("NEURD_LEGACY_SHAFT_PATHS", "1")
    forbid_helper_import(monkeypatch)
    selected = run_guard(nx.path_graph(["b0", "b1"]), 149)
    assert selected["path_backend_reason"] == "legacy_requested"
    assert selected["shortest_path"]("b0", "b1") == ["b0", "b1"]


@pytest.mark.parametrize("version", ["3.5", "3.6.2", "4.0", None])
def test_unverified_networkx_version_never_imports_helper(monkeypatch, version):
    monkeypatch.delenv("NEURD_LEGACY_SHAFT_PATHS", raising=False)
    monkeypatch.setattr(nx, "__version__", version)
    forbid_helper_import(monkeypatch)
    selected = run_guard(nx.path_graph(["b0", "b1"]), 149)
    assert selected["path_backend_reason"] == "unverified_networkx_version"
    assert selected["shortest_path"]("b0", "b1") == ["b0", "b1"]


@pytest.mark.parametrize("kind", [nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph])
def test_unsupported_graph_kind_preserves_networkx_fallback(monkeypatch, kind):
    monkeypatch.delenv("NEURD_LEGACY_SHAFT_PATHS", raising=False)
    monkeypatch.setattr(nx, "__version__", "3.6.1")
    forbid_helper_import(monkeypatch)
    graph = kind()
    graph.add_edge("b0", "b1")
    selected = run_guard(graph, 149)
    assert selected["path_backend_reason"] == "unsupported_graph"
    assert selected["shortest_path"]("b0", "b1") == ["b0", "b1"]


def test_missing_optional_numba_preserves_networkx_fallback(monkeypatch):
    import builtins

    monkeypatch.delenv("NEURD_LEGACY_SHAFT_PATHS", raising=False)
    monkeypatch.setattr(nx, "__version__", "3.6.1")
    original_import = builtins.__import__

    def absent_helper(name, *args, **kwargs):
        if name == "shaft_paths":
            raise ImportError("No module named 'numba'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", absent_helper)
    selected = run_guard(nx.path_graph(["b0", "b1"]), 149)
    assert selected["path_backend_reason"] == "optional_numba_unavailable"
    assert selected["shortest_path"]("b0", "b1") == ["b0", "b1"]


def test_constructor_unsupported_graph_falls_back(monkeypatch):
    pytest.importorskip("numba")
    import neurd.shaft_paths as implementation

    monkeypatch.delenv("NEURD_LEGACY_SHAFT_PATHS", raising=False)
    monkeypatch.setattr(nx, "__version__", "3.6.1")

    def unsupported(graph):
        raise TypeError("unsupported constructor domain")

    monkeypatch.setattr(implementation, "ExactBidirectionalPaths", unsupported)
    selected = run_guard(nx.path_graph(["b0", "b1"]), 149)
    assert selected["path_backend_reason"] == "unsupported_graph"
    assert selected["shortest_path"]("b0", "b1") == ["b0", "b1"]


@pytest.mark.parametrize("shaft_count, method, expected_queries", [
    (33, "path_to_all_shaft_mesh", 1056),
    (1025, "path_to_largest_shaft_mesh", 1024),
])
def test_eligible_backend_counts_only_actual_queries_and_does_not_query_early(
    monkeypatch, shaft_count, method, expected_queries,
):
    pytest.importorskip("numba")
    monkeypatch.delenv("NEURD_LEGACY_SHAFT_PATHS", raising=False)
    monkeypatch.setattr(nx, "__version__", "3.6.1")
    calls = []
    original_shortest_path = nx.shortest_path

    def observed_networkx(*args, **kwargs):
        calls.append((args, kwargs))
        return original_shortest_path(*args, **kwargs)

    monkeypatch.setattr(nx, "shortest_path", observed_networkx)
    graph = nx.path_graph(["b0", "b1", "s0", "b2"])
    timing = {}
    selected = run_guard(graph, shaft_count, method, timing)
    assert calls == []
    assert timing == dict(path_backend="numba_csr", path_backend_reason=None, path_queries=expected_queries)
    assert selected["shortest_path"]("b0", "b2") == ["b0", "b1", "s0", "b2"]
    assert calls == []


def test_silent_phase_does_not_require_metadata(monkeypatch):
    monkeypatch.delenv("NEURD_LEGACY_SHAFT_PATHS", raising=False)
    selected = run_guard(nx.path_graph(["b0", "b1"]), 2, timing=None)
    assert selected["shortest_path"]("b0", "b1") == ["b0", "b1"]


def test_runtime_environment_fingerprints_legacy_path_flag():
    from neurd.run_config import RunConfig, runtime_environment

    off = runtime_environment(RunConfig(), environment={})
    on = runtime_environment(RunConfig(), environment={"NEURD_LEGACY_SHAFT_PATHS": "1"})
    assert off["environment"]["NEURD_LEGACY_SHAFT_PATHS"] is None
    assert on["environment"]["NEURD_LEGACY_SHAFT_PATHS"] == "1"
