"""Byte-exact gate for the owned skeletal-distance kernel (neurd/skeletal_distance_ops.py).

Asserts the owned reimplementation reproduces the LOCKED mesh_tools kernels on real captured I/O:
  * skeletal_distance(keep_empty_placeholder=True)  == cu.get_skeletal_distance_no_skipping (widths)
  * adaptive_distance(...)                           == cu.mesh_correspondence_adaptive_distance (corr.)

Fixtures (tests/fixtures/skeletal_distance_cases.pbz2) were captured from a real h01 build by
tests/tools/capture_skeletal_distance_fixtures.py. The kernel is deterministic, so equality is the
bar: exact for the integer face-index arrays, tight tolerance for the float distances/width (same
ops, so they come out bit-identical in practice).
"""
import bz2
import pickle
from pathlib import Path

import neurd  # noqa: F401  activate numpy/mesh shims before trimesh
import numpy as np
import pytest
import trimesh

from tests.unit import skip_if_datasci_tools_unusable

skip_if_datasci_tools_unusable()

from neurd import skeletal_distance_ops as sdo  # noqa: E402

FIX = Path(__file__).resolve().parents[1] / "fixtures" / "skeletal_distance_cases.pbz2"
pytestmark = pytest.mark.skipif(not FIX.exists(), reason="capture fixture missing (run capture_skeletal_distance_fixtures.py)")


def _load():
    return pickle.load(bz2.open(FIX, "rb"))["cases"]


def _mesh(rec):
    return trimesh.Trimesh(vertices=rec["vertices"], faces=rec["faces"], process=False)


def _idx_equal(a, b):
    return np.array_equal(np.asarray(a).astype(np.int64), np.asarray(b).astype(np.int64))


def _float_equal(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return a.shape == b.shape and np.allclose(a, b, rtol=0, atol=1e-6, equal_nan=True)


# ------------------------- widths: no_skipping -------------------------
_NOSKIP = _load()["no_skipping"] if FIX.exists() else []


@pytest.mark.parametrize("k", range(len(_NOSKIP)), ids=[f"noskip{i}" for i in range(len(_NOSKIP))])
def test_no_skipping_matches_capture(k):
    c = _NOSKIP[k]
    kw = c["kwargs"]
    dist, std, submesh, idx = sdo.skeletal_distance(
        _mesh(c["mesh"]), c["edges"],
        buffer=kw["buffer"],
        distance_threshold=kw["distance_threshold"],
        distance_by_mesh_center=kw["distance_by_mesh_center"],
        connectivity=kw.get("connectivity", "edges"),   # width caller leaves the mesh_tools default
        keep_empty_placeholder=True,
    )
    out = c["out"]
    assert _float_equal(dist, out["distances"]), "distance array (the width input) diverged"
    assert _idx_equal(idx, out["indices"]), "unique_removed_faces diverged"


# ------------------------- correspondence: adaptive -------------------------
_ADAPT = _load()["adaptive"] if FIX.exists() else []


@pytest.mark.parametrize("k", range(len(_ADAPT)), ids=[f"adapt{i}" for i in range(len(_ADAPT))])
def test_adaptive_matches_capture(k):
    c = _ADAPT[k]
    kw = c["kwargs"]
    out = sdo.adaptive_distance(
        c["skeleton"], _mesh(c["mesh"]),
        skeleton_segment_width=kw.get("skeleton_segment_width", 1000),
        distance_by_mesh_center=kw.get("distance_by_mesh_center", True),
        distance_threshold=kw.get("distance_threshold", 3000),
        buffer=kw.get("buffer", 100),
        connectivity=kw.get("connectivity", "vertices"),   # wrapper default is "vertices"
        return_closest_face_on_empty=kw.get("return_closest_face_on_empty", False),
    )
    exp = c["out"]
    assert exp is not None, "captured None output (empty-correspondence case)"
    if isinstance(exp, tuple) and len(exp) == 2 and isinstance(exp[0], str) and exp[0] == "RAW":
        pytest.skip("captured RAW output shape")
    exp_idx, exp_width = exp
    assert isinstance(out, tuple) and len(out) == 2, f"owned returned {type(out)}"
    got_idx, got_width = out
    assert _idx_equal(got_idx, exp_idx), "correspondence face indices diverged"
    assert _float_equal(got_width, exp_width), "correspondence width diverged"
