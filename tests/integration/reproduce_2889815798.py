#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Reproduce + diagnose the "class A" failure on neuron_2889815798:

    IndexError: index 169855 is out of bounds for axis 0 with size 169855

raised from neuron_utils.apply_adaptive_mesh_correspondence_to_neuron (neuron.py:2377).

This run is EXPENSIVE (soma extraction + full decomposition; can exceed 2h on a big H01
mesh), so it is built to extract the maximum forensic signal in ONE pass:

  * data_type = h01 (this mesh is H01; microns would die earlier with NameError calcification_param)
  * calculate_spines = False -> class A happens BEFORE spine calc, so we skip ~100s+ of work
  * the instrumented guard in neuron_utils._dump_adaptive_correspondence_failure writes
        /tmp/neurd_diag/classA_<id>_<limb>_b<branch>_<ts>.{json,npz,_limbmesh.off}
    and prints a VERDICT distinguishing "off-by-one fencepost" vs "wrong reference mesh".
  * a heartbeat thread prints elapsed time + RSS every 2 min so the long run is observable.

Usage:
    python tests/integration/reproduce_2889815798.py [path/to/mesh.off]
"""
import os

# Pin numerical backends to 1 thread (match the production worker) BEFORE importing numpy.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

# Enable the IDX-TRACE frame probes for this validation run (default OFF in production).
# After the fix, N2 (pre-rebuild) may show desync but N3 (post-rebuild) must be all clean.
os.environ.setdefault("NEURD_IDX_TRACE", "1")

import sys
import time
import threading
import traceback

DEFAULT_MESH = "/home/eugen/Desktop/CodeWork/Projects/Diplom/notebooks/notebooks/H01/neuron_2889815798.off"


def _start_heartbeat(start_time):
    """Daemon thread: print elapsed + RSS every 120s so the >2h run is observable."""
    try:
        import psutil
        proc = psutil.Process(os.getpid())
    except Exception:
        proc = None

    def _beat():
        while True:
            time.sleep(120)
            elapsed = time.time() - start_time
            rss = ""
            if proc is not None:
                try:
                    rss = f", RSS={proc.memory_info().rss / (1024 ** 2):.0f} MB"
                except Exception:
                    pass
            print(f"[heartbeat] elapsed={elapsed / 60:.1f} min{rss}", flush=True)

    t = threading.Thread(target=_beat, daemon=True)
    t.start()


def main():
    mesh_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MESH
    if not os.path.exists(mesh_path):
        print(f"[FATAL] mesh not found: {mesh_path}")
        return 2

    from neurd import neuron
    from neurd import parameters
    from mesh_tools import trimesh_utils as tu

    parameters.params.use("h01")  # this mesh is H01

    print(f"[i] loading mesh: {mesh_path}", flush=True)
    mesh = tu.load_mesh_no_processing(mesh_path)
    print(f"[i] mesh: faces={len(mesh.faces)}, vertices={len(mesh.vertices)}, "
          f"watertight={tu.is_watertight(mesh)}", flush=True)

    start = time.time()
    _start_heartbeat(start)

    print(f"[i] building Neuron (calculate_spines=False) ...", flush=True)
    try:
        neuron_obj = neuron.Neuron(mesh=mesh, calculate_spines=False)
    except Exception as e:
        elapsed = time.time() - start
        print("\n" + "#" * 80)
        print(f"[REPRODUCED] {type(e).__name__}: {e}")
        print(f"[REPRODUCED] after {elapsed / 60:.1f} min")
        print("#" * 80)
        traceback.print_exc()
        print("\n[i] forensic dumps (if class A): /tmp/neurd_diag/")
        try:
            for f in sorted(os.listdir("/tmp/neurd_diag")):
                print(f"      /tmp/neurd_diag/{f}")
        except Exception:
            pass
        return 1

    elapsed = time.time() - start
    print(f"\n[UNEXPECTED] Neuron built WITHOUT the class-A failure in {elapsed / 60:.1f} min "
          f"(n_limbs={neuron_obj.n_limbs}). The bug did not reproduce on this run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
