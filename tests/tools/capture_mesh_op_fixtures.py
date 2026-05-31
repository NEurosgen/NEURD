"""Capture real meshlab op (input, output) pairs as golden fixtures.

Wraps mesh_tools.meshlab Poisson/Decimator/FillHoles to dump, on every call during a
real soma identification, the input mesh and the meshlab output mesh to
tests/fixtures/mesh_ops/<op>_<i>_{in,out}.off. These golden pairs let the in-process
replacements in neurd/_mesh_ops.py be validated for *fidelity* against the actual
meshlabserver output (not just synthetic meshes), in seconds, without re-running the
~5-min soma stage.

Run once (no pytest): python tests/tools/capture_mesh_op_fixtures.py
"""

import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import neurd  # noqa: F401  (shims first)
import trimesh
from mesh_tools import meshlab as ml
from mesh_tools import trimesh_utils as tu

OUT = REPO / "tests" / "fixtures" / "mesh_ops"
OUT.mkdir(parents=True, exist_ok=True)
_counters = {}


def _wrap(cls, name):
    orig_call = cls.__call__

    def wrapped(self, *args, **kwargs):
        result = orig_call(self, *args, **kwargs)
        mesh_out = result[0] if isinstance(result, tuple) else result
        i = _counters.get(name, 0)
        _counters[name] = i + 1
        v, f = kwargs.get("vertices"), kwargs.get("faces")
        try:
            if v is not None and f is not None:
                trimesh.Trimesh(vertices=v, faces=f, process=False).export(
                    OUT / f"{name}_{i}_in.off"
                )
            if mesh_out is not None and hasattr(mesh_out, "faces"):
                mesh_out.export(OUT / f"{name}_{i}_out.off")
            print(f"  captured {name}_{i}: in={None if v is None else len(f)} faces")
        except Exception as e:  # never let capture break the run
            print(f"  capture {name}_{i} failed: {e}")
        return result

    cls.__call__ = wrapped


def main():
    from neurd import soma_extraction_utils as sm
    from neurd import parameters

    for cls, nm in [(ml.Poisson, "poisson"), (ml.Decimator, "decimate"), (ml.FillHoles, "fill_holes")]:
        _wrap(cls, nm)

    os.chdir(tempfile.mkdtemp(prefix="capture_mesh_ops_"))
    parameters.params.use("microns")
    mesh = tu.load_mesh_no_processing(str(REPO / "tests" / "fixtures" / "864691135510518224.off"))
    sm.soma_indentification(mesh, verbose=False)
    print(f"\nDONE. Captured: { {k: v for k, v in _counters.items()} } -> {OUT}")


if __name__ == "__main__":
    main()
