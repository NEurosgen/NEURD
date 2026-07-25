#!/usr/bin/env python
"""Wall-clock attribution for the segmentation build (finds the REAL bottleneck).

cProfile cumtime is unreliable here: native ops (embree, meshparty C, meshlab, scipy) release the
GIL, so their wall shows up as `threading.wait` or an inflated caller cumtime. This tool instead
monkeypatches a curated set of the major pipeline operations with an EXCLUSIVE wall accumulator
(time in the function minus time spent in nested patched functions), so the totals sum ~= build wall
and the biggest self-wall is the true hot spot.

    python tests/tools/wall_profile.py --neuron <off> --data-type h01 [--spines]

Prints a sorted self-wall table + call counts. Add targets by editing TARGETS below.
"""
import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

# (module_path, attribute) targets spanning the major operations. Missing ones are skipped.
TARGETS = [
    # skeletonization
    ("mesh_tools.meshparty_skeletonize", "skeletonize_mesh_largest_component"),
    ("mesh_tools.meshparty_skeletonize", "skeleton_obj_to_branches"),
    ("mesh_tools.skeleton_utils", "skeletonize_and_clean_connected_branch_CGAL"),
    # correspondence / width seams
    ("neurd._correspondence_backend", "adaptive_distance_correspondence"),
    ("neurd._correspondence_backend", "resolve_face_labels"),
    ("neurd._correspondence_backend", "skeletal_distance_no_skipping"),
    ("neurd._correspondence_backend", "waterfill_to_soma_border"),
    # decomposition
    ("neurd.preprocess_neuron", "preprocess_limb"),
    # spines / widths
    ("neurd.spine_utils", "calculate_spines_on_neuron"),
    ("neurd.spine_utils", "calculate_spines_on_branch"),
    ("neurd.width_utils", "calculate_new_width_for_neuron_obj"),
    # mesh partition primitives
    ("mesh_tools.trimesh_utils", "split"),
    ("mesh_tools.trimesh_utils", "split_by_vertices"),
    ("mesh_tools.trimesh_utils", "mesh_pieces_connectivity"),
    ("mesh_tools.trimesh_utils", "decimate"),
    ("mesh_tools.trimesh_utils", "remove_mesh_interior"),
    # soma
    ("neurd.soma_extraction_utils", "extract_soma_center"),
    # reconstruction ops (in-process replacements)
    ("neurd._mesh_ops", "decimate"),
    ("neurd._mesh_ops", "poisson_surface_reconstruction_meshlab"),
]

_self = {}     # name -> exclusive wall seconds
_calls = {}    # name -> call count
_stack = []    # list of [name, child_wall_accum]


def _wrap(name, fn):
    def w(*a, **k):
        _stack.append([name, 0.0])
        t0 = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            elapsed = time.perf_counter() - t0
            frame = _stack.pop()
            _self[name] = _self.get(name, 0.0) + (elapsed - frame[1])
            _calls[name] = _calls.get(name, 0) + 1
            if _stack:
                _stack[-1][1] += elapsed
    return w


def _patch():
    import importlib
    patched = []
    for mod_path, attr in TARGETS:
        try:
            mod = importlib.import_module(mod_path)
            fn = getattr(mod, attr, None)
            if fn is None or not callable(fn):
                continue
            setattr(mod, attr, _wrap(f"{mod_path.split('.')[-1]}.{attr}", fn))
            patched.append(f"{mod_path.split('.')[-1]}.{attr}")
        except Exception as e:
            print(f"[wall] skip {mod_path}.{attr}: {e}")
    return patched


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--neuron", required=True)
    ap.add_argument("--data-type", default="h01")
    ap.add_argument("--spines", action="store_true")
    args = ap.parse_args()

    os.chdir(tempfile.mkdtemp(prefix="wall_prof_"))

    import neurd  # noqa
    from neurd import parameters, neuron
    from mesh_tools import trimesh_utils as tu

    parameters.params.use(args.data_type)
    patched = _patch()
    print(f"[wall] patched {len(patched)} targets", flush=True)

    mesh = tu.load_mesh_no_processing(str(args.neuron))
    t0 = time.perf_counter()
    neuron.Neuron(mesh=mesh, calculate_spines=args.spines)
    wall = time.perf_counter() - t0

    rows = sorted(_self.items(), key=lambda kv: kv[1], reverse=True)
    attributed = sum(_self.values())
    print(f"\n{'='*74}\nWALL ATTRIBUTION  ({Path(args.neuron).name}, {args.data_type}, spines={args.spines})")
    print(f"total build wall = {wall:.0f}s | attributed = {attributed:.0f}s ({100*attributed/wall:.0f}%) | "
          f"unattributed = {wall-attributed:.0f}s\n{'='*74}")
    print(f"{'self_wall_s':>11} {'%wall':>6} {'ncalls':>8}  function")
    print("-" * 74)
    for name, s in rows:
        print(f"{s:>11.1f} {100*s/wall:>5.0f}% {_calls[name]:>8}  {name}")


if __name__ == "__main__":
    main()
