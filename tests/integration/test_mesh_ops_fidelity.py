"""Fidelity of in-process mesh ops vs real meshlabserver output.

Compares neurd/_mesh_ops.py against golden (input, meshlab-output) pairs captured from a
real soma identification (tests/tools/capture_mesh_op_fixtures.py). Runs in seconds on the
small captured meshes — the fast loop for swapping the meshlab subprocess ops.

Fixtures live under tests/fixtures/mesh_ops/ and are gitignored (regenerable, some large);
this test SKIPS when they are absent.

NOTE on Poisson: capture proved MeshLab's Poisson is a silent no-op on this build (output
byte-identical to input — see neurd/__init__.py, where it is replaced by an in-process
no-op). So there is nothing to match for Poisson; only the real op (Decimator) is checked
here. _mesh_ops.poisson_surface_reconstruction is kept for the separate "real watertight
reconstruction" quality experiment, not for drop-in parity.
"""

from pathlib import Path

import numpy as np
import pytest

import neurd  # noqa: F401  (shims first)
import trimesh

from neurd import _mesh_ops as mo

_FIX = Path(__file__).resolve().parents[1] / "fixtures" / "mesh_ops"
_HAS_FIX = _FIX.exists() and any(_FIX.glob("decimate_*_in.off"))
pytestmark = pytest.mark.skipif(not _HAS_FIX, reason="no captured mesh_ops fixtures")


def _load(name):
    return trimesh.load(str(_FIX / name), process=False)


def _decimate_pairs():
    out = []
    for p in sorted(_FIX.glob("decimate_*_in.off")):
        o = p.with_name(p.name.replace("_in.off", "_out.off"))
        if o.exists():
            out.append((p.name, o.name))
    return out


def _mean_surface_dist(a, b, n=3000):
    """Mean distance from points sampled on `a` to surface of `b`, normalized by b's bbox diag."""
    pts = a.sample(min(n, max(100, len(a.faces))))
    _, dist, _ = b.nearest.on_surface(pts)
    diag = float(np.linalg.norm(b.bounds[1] - b.bounds[0]))
    return float(np.mean(dist)) / diag if diag > 0 else float("inf")


@pytest.mark.parametrize("in_name,out_name", _decimate_pairs())
def test_decimate_matches_meshlab(in_name, out_name):
    mesh_in, mesh_out = _load(in_name), _load(out_name)
    ratio = len(mesh_out.faces) / len(mesh_in.faces)
    mine = mo.decimate(mesh_in, ratio)
    # similar face budget (quadric decimation isn't bit-exact across libs)
    assert len(mine.faces) == pytest.approx(len(mesh_out.faces), rel=0.35)
    # and geometrically close to meshlab's decimated surface
    d = _mean_surface_dist(mine, mesh_out)
    assert d < 0.02, f"decimate surface dist {d:.4f} of bbox-diag too large"
