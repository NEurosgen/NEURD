"""Oracle / characterization test for the CGAL mesh-segmentation seam.

`mesh_tools.trimesh_utils.mesh_segmentation` is the single entry point that calls
`cgal_Segmentation_Module.cgal_segmentation(filepath_no_ext, clusters, smoothness)`,
which writes two CSVs next to the input: one int cluster-id per face and one float
SDF per face (linearly normalized to [0,1]).

The compiled CGAL extension was built inside the (removed) Docker image; this fork
ships a pure-Python stand-in instead (neurd/_cgal_segmentation.py, registered in
sys.modules by neurd/__init__.py). These tests pin the seam's *contract* and
characterize the Python stand-in's *fidelity* against golden output produced by the
original CGAL binary.

Fixtures (committed under tests/):
  - 990_mesh.off                  : input mesh (7240 faces)
  - 990_mesh-cgal_3_0.20.csv      : golden cluster id per face (clusters=3, smoothness=0.20)
  - 990_mesh-cgal_3_0.20_sdf.csv  : golden SDF per face, normalized [0,1]

The provider tests SKIP if no `cgal_Segmentation_Module` is registered.
"""

import shutil
from pathlib import Path

import numpy as np
import pytest

# Importing neurd registers the pure-Python cgal_Segmentation_Module stand-in
# in sys.modules (see neurd/__init__.py).
import neurd  # noqa: F401

# --- fixture locations ---
_TESTS_DIR = Path(__file__).resolve().parent.parent
_MESH_OFF = _TESTS_DIR / "990_mesh.off"
_GOLDEN_SEG = _TESTS_DIR / "990_mesh-cgal_3_0.20.csv"
_GOLDEN_SDF = _TESTS_DIR / "990_mesh-cgal_3_0.20_sdf.csv"

# Parameters that produced the golden files (encoded in their names).
_CLUSTERS = 3
_SMOOTHNESS = 0.20
_N_FACES = 7240

# Fidelity floor for the SDF. The Python stand-in ray-traces a single jittered ray
# per face (vs CGAL's per-face ray cone) so values differ in detail, but the
# thickness *ordering* — all that soma detection needs — is preserved. Measured
# Pearson correlation of the stand-in vs golden is ~0.95 (robust percentile
# normalization); a faithful C++ rebuild would score ~1.0, so this floor passes
# for both providers.
_SDF_CORR_MIN = 0.90


def _read_per_face_csv(path):
    """One value per line (genfromtxt with newline delimiter, as the seam reads it)."""
    return np.genfromtxt(str(path), delimiter="\n")


def _run_provider(tmp_path):
    """Run the registered provider on a tmp copy; return (cluster_ids, sdf) arrays.

    Skips if no provider is registered. Outputs land beside the tmp copy, never
    clobbering the committed golden CSVs (the seam writes {filepath_no_ext}-cgal_*.csv).
    """
    csm = pytest.importorskip(
        "cgal_Segmentation_Module",
        reason="no cgal_Segmentation_Module registered",
    )
    work_mesh = tmp_path / "990_mesh.off"
    shutil.copy(_MESH_OFF, work_mesh)
    filepath_no_ext = str(tmp_path / "990_mesh")

    csm.cgal_segmentation(filepath_no_ext, _CLUSTERS, _SMOOTHNESS)

    suffix = f"-cgal_{int(_CLUSTERS)}_{_SMOOTHNESS:.2f}"
    out_seg = tmp_path / f"990_mesh{suffix}.csv"
    out_sdf = tmp_path / f"990_mesh{suffix}_sdf.csv"
    assert out_seg.exists(), f"provider did not write {out_seg.name}"
    assert out_sdf.exists(), f"provider did not write {out_sdf.name}"
    return _read_per_face_csv(out_seg), _read_per_face_csv(out_sdf)


def test_golden_fixtures_well_formed():
    """Always-on: the committed oracle fixtures match the mesh and each other.

    No provider needed — guards against the fixtures rotting (wrong length, NaNs,
    cluster ids drifting to floats, SDF leaving [0,1]).
    """
    assert _MESH_OFF.exists(), f"missing oracle mesh {_MESH_OFF}"
    assert _GOLDEN_SEG.exists(), f"missing golden segmentation {_GOLDEN_SEG}"
    assert _GOLDEN_SDF.exists(), f"missing golden sdf {_GOLDEN_SDF}"

    seg = _read_per_face_csv(_GOLDEN_SEG)
    sdf = _read_per_face_csv(_GOLDEN_SDF)

    assert seg.shape == (_N_FACES,), f"seg fixture has {seg.shape}, expected ({_N_FACES},)"
    assert sdf.shape == (_N_FACES,), f"sdf fixture has {sdf.shape}, expected ({_N_FACES},)"

    assert np.all(seg >= 0)
    assert np.allclose(seg, np.round(seg)), "cluster ids must be integral"

    # CGAL normalizes SDF to [0,1]; one degenerate face is 0 in this fixture.
    assert np.all(np.isfinite(sdf))
    assert sdf.min() >= 0.0 and sdf.max() <= 1.0
    assert np.mean(sdf > 0) > 0.99


def test_provider_contract(tmp_path):
    """Drop-in contract any cgal_Segmentation_Module provider must satisfy.

    The seam reads these CSVs straight back and feeds the SDF medians to a hardcoded
    soma_width_threshold of 0.32 — so the SDF *must* be normalized to [0,1].
    """
    seg, sdf = _run_provider(tmp_path)

    assert seg.shape == (_N_FACES,), f"provider seg has {seg.shape}"
    assert sdf.shape == (_N_FACES,), f"provider sdf has {sdf.shape}"

    assert np.all(seg >= 0)
    assert np.allclose(seg, np.round(seg)), "cluster ids must be integral"

    assert np.all(np.isfinite(sdf))
    assert sdf.min() >= 0.0 and sdf.max() <= 1.0 + 1e-6, "SDF must be normalized to [0,1]"


def test_provider_sdf_fidelity_vs_golden(tmp_path):
    """Characterization: the provider's SDF tracks the original CGAL SDF.

    Correlation, not exact match — the stand-in's ray sampling differs from CGAL's,
    but soma detection only relies on the thickness ordering being preserved.
    """
    _, sdf = _run_provider(tmp_path)
    gold_sdf = _read_per_face_csv(_GOLDEN_SDF)

    corr = float(np.corrcoef(sdf, gold_sdf)[0, 1])
    assert corr >= _SDF_CORR_MIN, f"SDF correlation {corr:.4f} < {_SDF_CORR_MIN}"
