"""One-off inspection run: decompose the dense H01 neuron_1830470325 and save fragments.

Exercises the dense-limb fidelity path (apply_adaptive_mesh_correspondence_to_neuron ->
resolve_empty_conflicting_face_labels) that used to crash with
"missing labels was not resolved". Saves soma/limb/branch .off + connectivity via the
production save_segmentation so the partition can be eyeballed.

    python tests/integration/run_big_neuron_inspect.py
"""
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import neurd  # noqa: F401  activates numpy-2 / cgal / meshlab shims
from neurd import neuron
from neurd import parameters
from mesh_tools import trimesh_utils as tu
from process_all_neurons import save_segmentation

MESH = Path("/home/eugen/Desktop/CodeWork/Projects/Diplom/notebooks/notebooks/H01/neuron_1830470325.off")
OUT = REPO / "tests" / "integration" / "_seg_output_big" / MESH.stem


def main():
    assert MESH.exists(), f"missing mesh {MESH}"
    parameters.params.use("h01")
    t0 = time.time()
    print(f"[run] loading {MESH} ({MESH.stat().st_size/1e6:.0f} MB)")
    mesh = tu.load_mesh_no_processing(str(MESH))
    print(f"[run] faces={len(mesh.faces)} verts={len(mesh.vertices)}; decomposing ...")
    neuron_obj = neuron.Neuron(mesh=mesh, calculate_spines=False)
    print(f"[run] decomposed in {time.time()-t0:.0f}s; n_limbs={neuron_obj.n_limbs}")
    save_segmentation(neuron_obj, OUT)
    print(f"[run] saved -> {OUT}")
    for lb in neuron_obj.limbs:
        print(f"   {getattr(lb,'name','?')}: {len(list(lb.branches))} branches")


if __name__ == "__main__":
    main()
