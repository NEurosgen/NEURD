"""Golden capture/replay for `neuron_utils.branches_to_concept_network`.

Why: that function builds each limb's concept network -- the framework's central
topology structure -- but it runs in the LIMB stage, which the soma gate
(tests/tools/soma_gate.py) stops short of. Gating it through the full pipeline would
cost tens of minutes per refactor step.

It is, however, almost pure and deterministic: skeleton arrays + coordinates in, a
`GraphOrderedEdges` out, with no RNG and no mesh access. So one real pipeline run can
capture every (input, output) pair, and afterwards a refactor is checked in seconds.
Same trick as tests/tools/capture_mesh_op_fixtures.py does for the meshlab ops.

    # once: run the pipeline and record every call (tens of minutes)
    python tests/tools/concept_network_fixtures.py capture

    # once more: add characterization cases for paths the real run never took
    python tests/tools/concept_network_fixtures.py synthesize

    # after every refactor step: replay them all (seconds)
    python tests/tools/concept_network_fixtures.py replay

`capture` also reports how many calls exercised the duplicate-branch path, because
~70 lines of the function are reachable only when it fires -- if the count is 0, those
lines are NOT covered by this gate and must be refactored conservatively.
"""

import bz2
import os
import pickle
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

import neurd  # noqa: F401,E402  activates the numpy-2 / cgal / meshlab shims
from neurd import neuron_utils as nru  # noqa: E402
from neurd import parameters  # noqa: E402

OUT = REPO / "tests" / "fixtures" / "concept_network"
_ANCHOR = REPO / "Applications" / "Tutorials" / "Auto_Proof_Pipeline" / "neuron_2530864375.off"


# --------------------------------------------------------------------------- signature

def _norm(value):
    """Make an attribute value comparable and picklable-plain."""
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return ("ndarray", value.shape, [round(float(x), 6) for x in value.ravel().tolist()])
    if isinstance(value, (list, tuple)):
        return [_norm(v) for v in value]
    if isinstance(value, (np.integer, np.floating)):
        return round(float(value), 6)
    return value


def graph_signature(graph):
    """A comparable snapshot of a concept network.

    Includes each edge's `order` attribute: GraphOrderedEdges stamps it with the edge
    count at insertion time, and `branches_to_concept_network` reads it back as the
    BRANCH INDEX. Edge insertion order is therefore semantic here, not cosmetic, and a
    signature that ignored it would miss the most dangerous class of regression.
    """
    edges = sorted(
        (int(u), int(v), int(data.get("order", -1)))
        for u, v, data in graph.edges(data=True)
    )
    nodes = {
        int(n): {k: _norm(v) for k, v in sorted(data.items())}
        for n, data in graph.nodes(data=True)
    }
    return {"edges": edges, "nodes": nodes}


def _diff(expected, actual):
    """Human-readable first differences between two signatures."""
    problems = []
    if expected["edges"] != actual["edges"]:
        exp, act = set(map(tuple, expected["edges"])), set(map(tuple, actual["edges"]))
        problems.append(f"edges differ: only-in-baseline={sorted(exp - act)[:5]} "
                        f"only-in-current={sorted(act - exp)[:5]}")
    exp_nodes, act_nodes = expected["nodes"], actual["nodes"]
    if set(exp_nodes) != set(act_nodes):
        problems.append(f"node sets differ: baseline-only={sorted(set(exp_nodes)-set(act_nodes))[:5]} "
                        f"current-only={sorted(set(act_nodes)-set(exp_nodes))[:5]}")
    for n in sorted(set(exp_nodes) & set(act_nodes)):
        if exp_nodes[n] != act_nodes[n]:
            for k in sorted(set(exp_nodes[n]) | set(act_nodes[n])):
                if exp_nodes[n].get(k) != act_nodes[n].get(k):
                    problems.append(f"node {n} attr {k!r}: baseline={exp_nodes[n].get(k)} "
                                    f"current={act_nodes[n].get(k)}")
            if len(problems) > 6:
                break
    return problems


# --------------------------------------------------------------------------- capture

def capture():
    from neurd.segmentation_pipeline import segmentation_pipeline
    from mesh_tools import trimesh_utils as tu

    OUT.mkdir(parents=True, exist_ok=True)
    for stale in OUT.glob("call_*.pbz2"):
        stale.unlink()

    original = nru.branches_to_concept_network
    calls = {"n": 0, "with_duplicates": 0}

    def wrapped(*args, **kwargs):
        result = original(*args, **kwargs)
        i = calls["n"]
        calls["n"] = i + 1
        try:
            # both live call sites pass keywords only; positionals are folded in for safety
            names = ("curr_branch_skeletons", "starting_coordinate", "starting_edge",
                     "touching_soma_vertices", "soma_group_idx", "starting_soma")
            recorded = dict(zip(names, args))
            recorded.update({k: v for k, v in kwargs.items() if k != "verbose"})
            n_branches = len(recorded["curr_branch_skeletons"])
            if _has_duplicate_branches(recorded["curr_branch_skeletons"]):
                calls["with_duplicates"] += 1
            with bz2.open(OUT / f"call_{i:04d}.pbz2", "wb") as fh:
                pickle.dump({"kwargs": recorded, "signature": graph_signature(result)}, fh)
            print(f"  captured call_{i:04d}: {n_branches} branches -> "
                  f"{result.number_of_nodes()} nodes / {result.number_of_edges()} edges")
        except Exception as e:  # never let capture break the run
            print(f"  capture call_{i:04d} failed: {e}")
        return result

    nru.branches_to_concept_network = wrapped

    os.chdir(tempfile.mkdtemp(prefix="capture_concept_network_"))
    parameters.params.use("h01")
    print(f"[capture] anchor={_ANCHOR.name} data_type=h01")
    mesh = tu.load_mesh_no_processing(str(_ANCHOR))
    segmentation_pipeline(mesh, segment_id=2530864375)

    print(f"\nDONE. {calls['n']} calls captured -> {OUT}")
    print(f"Calls exercising the duplicate-branch path: {calls['with_duplicates']}")
    if calls["with_duplicates"] == 0:
        print("  !! the dedupe / restore-duplicates code is NOT covered by these fixtures;\n"
              "     refactor those two blocks conservatively (move as-is, do not rewrite).")


def synthesize():
    """Add characterization fixtures for the paths the real run never took.

    The captured calls cover only 3..12-branch limbs with no duplicate branches, leaving
    two whole blocks untested: the single-branch early return and the dedupe /
    restore-duplicates pair (~70 lines). The function is pure, so inputs DERIVED from a
    captured call are just as valid as captured ones -- and the current output is
    recorded as the baseline, which is exactly what a refactor needs to preserve.
    """
    import copy

    with bz2.open(OUT / "call_0000.pbz2", "rb") as fh:
        base = pickle.load(fh)["kwargs"]

    cases = {}

    # 1) single branch -> the early-return path. Its starting_edge must be that branch's
    #    own endpoints, which is what the real caller passes.
    one = copy.deepcopy(base)
    one["curr_branch_skeletons"] = [base["curr_branch_skeletons"][0]]
    one["starting_edge"] = np.array([base["curr_branch_skeletons"][0][0][0],
                                     base["curr_branch_skeletons"][0][-1][-1]])
    cases["single_branch"] = one

    # 2) a duplicated branch -> two branches downsample to the same pair of endpoints,
    #    which is exactly what the dedupe block keys on.
    dup = copy.deepcopy(base)
    dup["curr_branch_skeletons"] = list(base["curr_branch_skeletons"]) + \
                                   [copy.deepcopy(base["curr_branch_skeletons"][1])]
    cases["duplicate_branch"] = dup

    # 3) two duplicates of two different branches -> more than one domination group
    dup2 = copy.deepcopy(base)
    dup2["curr_branch_skeletons"] = list(base["curr_branch_skeletons"]) + [
        copy.deepcopy(base["curr_branch_skeletons"][1]),
        copy.deepcopy(base["curr_branch_skeletons"][2]),
    ]
    cases["duplicate_branch_pair"] = dup2

    os.chdir(tempfile.mkdtemp(prefix="synth_concept_network_"))
    parameters.params.use("h01")

    for name, kwargs in cases.items():
        try:
            signature = graph_signature(nru.branches_to_concept_network(**kwargs))
        except Exception as e:
            print(f"  SKIP synth_{name}: current implementation raises "
                  f"{type(e).__name__}: {e}")
            continue
        with bz2.open(OUT / f"synth_{name}.pbz2", "wb") as fh:
            pickle.dump({"kwargs": kwargs, "signature": signature}, fh)
        print(f"  wrote synth_{name}: {len(kwargs['curr_branch_skeletons'])} branches -> "
              f"{len(signature['nodes'])} nodes / {len(signature['edges'])} edges "
              f"(duplicates: {_has_duplicate_branches(kwargs['curr_branch_skeletons'])})")


def _has_duplicate_branches(skeletons):
    """Mirror of the dedupe test inside branches_to_concept_network, for coverage stats."""
    from collections import Counter
    from neurd.neuron_utils import sk
    try:
        downsampled = [sk.resize_skeleton_branch(b, n_segments=1) for b in skeletons]
        stacked = sk.stack_skeletons(downsampled)
        _, indices = np.unique(stacked.reshape(-1, 3), return_inverse=True, axis=0)
        _, edge_ids = np.unique(np.sort(indices.reshape(-1, 2), axis=1), axis=0,
                                return_inverse=True)
        return any(v > 1 for v in Counter(edge_ids).values())
    except Exception:
        return False


# --------------------------------------------------------------------------- replay

def replay():
    fixtures = sorted(OUT.glob("*.pbz2"))
    if not fixtures:
        sys.exit(f"no fixtures in {OUT} -- run `capture` first")

    os.chdir(tempfile.mkdtemp(prefix="replay_concept_network_"))
    parameters.params.use("h01")

    failures = 0
    for path in fixtures:
        with bz2.open(path, "rb") as fh:
            saved = pickle.load(fh)
        try:
            current = graph_signature(nru.branches_to_concept_network(**saved["kwargs"]))
        except Exception as e:
            print(f"FAIL {path.name}: raised {type(e).__name__}: {e}")
            failures += 1
            continue
        problems = _diff(saved["signature"], current)
        if problems:
            failures += 1
            print(f"FAIL {path.name}:")
            for p in problems[:6]:
                print(f"    {p}")

    print(f"\n{len(fixtures) - failures}/{len(fixtures)} fixtures identical")
    if failures:
        sys.exit(1)
    print("[replay] PASS")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "replay"
    if mode == "capture":
        capture()
    elif mode == "synthesize":
        synthesize()
    elif mode == "replay":
        replay()
    else:
        sys.exit(f"unknown mode {mode!r} (expected 'capture', 'synthesize' or 'replay')")
