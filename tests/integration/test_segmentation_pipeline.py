"""Characterization test for the slim mesh-segmentation pipeline.

Pins the segmentation deliverable / contract:

    mesh in -> Neuron with somas + limbs -> branches (each with .mesh and .skeleton)

This is the output that the Phase 5-7 cluster removal (DEPS_PLAN.md) must NOT break.
It exercises `neurd.segmentation_pipeline.segmentation_pipeline`, i.e. the same
`neuron.Neuron(mesh=...)` decomposition path that `process_all_neurons.py` runs.

Runtime: the real decomposition of the committed fixture takes several minutes and
shells out to `xvfb-run meshlabserver` (wrapped internally by mesh_tools). The test
SKIPS cleanly when MeshLab/xvfb are not on PATH, so it is safe to collect anywhere.
It lives under tests/integration/ and is therefore outside the fast `tests/unit/` gate.
"""

import json
import os
import shutil
import sys
from pathlib import Path

import pytest

# extract_metrics lives in tests/tools (not a package)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from neuron_metrics import extract_metrics  # noqa: E402

# Importing neurd first activates the numpy-2 / cgal / meshlab compat shims that
# datasci_tools and mesh_tools rely on (see neurd/__init__.py).
import neurd  # noqa: F401,E402
from neurd import preprocess_neuron as pre
from neurd import soma_extraction_utils as sm
from neurd import spine_utils as spu
from neurd import neuron_utils as nru
from neurd import neuron
from neurd.segmentation_pipeline import segmentation_pipeline

from datasci_tools import module_utils as modu
from mesh_tools import trimesh_utils as tu

_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "864691135510518224.off"

_MESH_TOOLS = all(shutil.which(b) for b in ("xvfb-run", "meshlabserver"))
_requires_mesh_tools = pytest.mark.skipif(
    not _MESH_TOOLS, reason="requires `xvfb-run` + `meshlabserver` on PATH"
)
_requires_fixture = pytest.mark.skipif(
    not _FIXTURE.exists(), reason=f"missing fixture mesh {_FIXTURE}"
)


@pytest.fixture(scope="module")
def decomposed_neuron(tmp_path_factory):
    """Run the slim pipeline once; share the decomposed Neuron across assertions.

    The decomposition shells out to meshlabserver, which writes ~tens of MB of
    working files to `./<segment_id>/` in the CWD. Run inside a pytest tmp dir so
    that scratch never lands in the repo.
    """
    work_dir = tmp_path_factory.mktemp("segmentation_pipeline")
    prev_cwd = os.getcwd()
    os.chdir(work_dir)
    try:
        modu.set_global_parameters_and_attributes_by_data_type(
            module=[pre, sm, spu, nru, neuron],
            data_type="microns",
            set_default_first=True,
        )
        mesh = tu.load_mesh_no_processing(str(_FIXTURE))
        assert len(mesh.faces) > 0, "fixture mesh has no faces"
        return segmentation_pipeline(mesh, verbose=False)
    finally:
        os.chdir(prev_cwd)


@_requires_mesh_tools
@_requires_fixture
class TestSegmentationContract:
    def test_returns_neuron(self, decomposed_neuron):
        assert decomposed_neuron is not None
        assert isinstance(decomposed_neuron, neuron.Neuron)

    def test_has_soma(self, decomposed_neuron):
        # soma(s) are recorded as "S<i>" nodes in the top-level concept network
        cn = decomposed_neuron.concept_network
        soma_nodes = [n for n in cn.nodes() if isinstance(n, str) and n.startswith("S")]
        assert len(soma_nodes) >= 1, "decomposition produced no soma"

    def test_has_limbs_and_branches(self, decomposed_neuron):
        limbs = list(getattr(decomposed_neuron, "limbs", []))
        assert len(limbs) > 0, "decomposition produced no limbs"
        total_branches = sum(len(list(getattr(lb, "branches", []))) for lb in limbs)
        assert total_branches > 0, "limbs contain no branches"
        # every limb has at least one branch
        assert all(len(list(getattr(lb, "branches", []))) > 0 for lb in limbs)

    def test_branches_have_mesh_and_skeleton(self, decomposed_neuron):
        for lb in decomposed_neuron.limbs:
            assert getattr(lb, "skeleton", None) is not None, "limb missing skeleton"
            for br in lb.branches:
                mesh = getattr(br, "mesh", None)
                skel = getattr(br, "skeleton", None)
                assert mesh is not None and len(mesh.faces) > 0, "branch mesh empty"
                # skeleton is an array of edges: shape (n_edges, 2, 3)
                assert skel is not None and skel.ndim == 3 and skel.shape[1:] == (2, 3)


_BASELINE = Path(__file__).resolve().parents[1] / "fixtures" / "neuron_baseline.json"


@_requires_mesh_tools
@_requires_fixture
@pytest.mark.skipif(not _BASELINE.exists(), reason="no neuron_baseline.json (generate it)")
class TestNeuronBaseline:
    """Numerical regression vs a golden snapshot of the decomposed Neuron.

    Guards against SILENT degradation from output-changing optimizations (e.g.
    Decimator -> open3d, a different SDF/skeletonizer): the structural contract above
    can pass while the actual segmentation drifts. Tolerances are generous — they catch
    real regressions, not the pipeline's minor run-to-run nondeterminism. Regenerate the
    golden file (tests/tools/generate_neuron_baseline.py) only when an output change is
    deliberately accepted as the new reference.
    """

    # metric -> (kind, tol): "exact" | "abs" (max |diff|) | "rel" (max fractional diff)
    _TOL = {
        "n_somas": ("exact", None),
        "n_limbs": ("abs", 1),
        "n_branches_total": ("rel", 0.20),
        "skeleton_length_total": ("rel", 0.12),
        "branch_mesh_faces_total": ("rel", 0.20),
        "soma_total_faces": ("rel", 0.20),
    }

    def test_metrics_within_tolerance(self, decomposed_neuron):
        golden = json.loads(_BASELINE.read_text())
        current = extract_metrics(decomposed_neuron)

        failures = []
        for key, (kind, tol) in self._TOL.items():
            g, c = golden[key], current[key]
            if kind == "exact":
                ok = c == g
            elif kind == "abs":
                ok = abs(c - g) <= tol
            else:  # rel
                ok = abs(c - g) <= tol * abs(g) if g else c == g
            if not ok:
                failures.append(f"{key}: current={c} vs golden={g} ({kind} tol={tol})")

        # soma center drift: nearest-golden-center distance, soma scale ~thousands nm
        import numpy as np
        if current["n_somas"] == golden["n_somas"] and golden["soma_centers"]:
            gc = np.asarray(golden["soma_centers"])
            for c in np.asarray(current["soma_centers"]):
                d = float(np.linalg.norm(gc - c, axis=1).min())
                if d > 3000.0:
                    failures.append(f"soma center drift {d:.0f}nm (>3000) for {c.tolist()}")

        assert not failures, "Neuron output drifted from baseline:\n  " + "\n  ".join(failures)
