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

import os
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


# --- NEURD_SDF_RAYS override (real-CGAL provider only) ---
#
# The compiled provider reads NEURD_SDF_RAYS to lower CGAL's per-face ray count
# (default 25). Ray casting is ~100% of the segmentation cost and ~linear in the
# ray count, so this is the main speed lever. These tests pin the two properties
# the lever must keep: (1) unset behaves exactly like 25 (no default regression),
# (2) a reduced ray count still tracks the 25-ray SDF closely (thickness ordering,
# which spine/soma detection keys off, is preserved).

# Correlation floor for reduced-ray SDF vs the 25-ray SDF. Measured 0.996 at 16 rays
# and 0.989 at 8 on this fixture; 0.97 leaves margin for run-to-run jitter.
_REDUCED_RAY_SDF_CORR_MIN = 0.97


def _run_provider_rays(tmp_path, rays):
    """Run the provider with NEURD_SDF_RAYS=`rays` (or unset if None). Real provider only.

    Skips unless the compiled cgal_Segmentation_Module is registered — the pure-Python
    stand-in ignores the env var, so the override contract only applies to real CGAL.
    """
    csm = pytest.importorskip(
        "cgal_Segmentation_Module",
        reason="no cgal_Segmentation_Module registered",
    )
    if getattr(csm, "__file__", "").endswith(".py"):
        pytest.skip("pure-Python stand-in does not honor NEURD_SDF_RAYS")

    tmp_path.mkdir(parents=True, exist_ok=True)
    work_mesh = tmp_path / "990_mesh.off"
    shutil.copy(_MESH_OFF, work_mesh)
    filepath_no_ext = str(tmp_path / "990_mesh")

    prev = os.environ.get("NEURD_SDF_RAYS")
    if rays is None:
        os.environ.pop("NEURD_SDF_RAYS", None)
    else:
        os.environ["NEURD_SDF_RAYS"] = str(rays)
    try:
        csm.cgal_segmentation(filepath_no_ext, _CLUSTERS, _SMOOTHNESS)
    finally:
        if prev is None:
            os.environ.pop("NEURD_SDF_RAYS", None)
        else:
            os.environ["NEURD_SDF_RAYS"] = prev

    suffix = f"-cgal_{int(_CLUSTERS)}_{_SMOOTHNESS:.2f}"
    return (
        _read_per_face_csv(tmp_path / f"990_mesh{suffix}.csv"),
        _read_per_face_csv(tmp_path / f"990_mesh{suffix}_sdf.csv"),
    )


def test_sdf_rays_unset_equals_25(tmp_path):
    """Unset NEURD_SDF_RAYS is byte-identical to the explicit default of 25.

    This is the no-regression guarantee for the committed default: the env-var
    mechanism must not perturb output unless a caller opts in.
    """
    seg_unset, sdf_unset = _run_provider_rays(tmp_path / "unset", None)
    seg_25, sdf_25 = _run_provider_rays(tmp_path / "r25", 25)

    assert np.array_equal(seg_unset, seg_25), "unset != 25 for segmentation ids"
    assert np.array_equal(sdf_unset, sdf_25), "unset != 25 for SDF values"


@pytest.mark.parametrize("rays", [16, 12, 8])
def test_sdf_rays_reduced_tracks_baseline(tmp_path, rays):
    """Fewer rays still track the 25-ray SDF closely (ordering preserved)."""
    _, sdf_base = _run_provider_rays(tmp_path / "base", 25)
    _, sdf_red = _run_provider_rays(tmp_path / f"r{rays}", rays)

    corr = float(np.corrcoef(sdf_red, sdf_base)[0, 1])
    assert corr >= _REDUCED_RAY_SDF_CORR_MIN, (
        f"rays={rays}: SDF correlation {corr:.4f} < {_REDUCED_RAY_SDF_CORR_MIN}"
    )
