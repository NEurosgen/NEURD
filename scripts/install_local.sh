#!/usr/bin/env bash
# Local (non-Docker) install for NEURD on Python 3.12.
#
# Usage:
#   bash scripts/install_local.sh                # creates ./.venv with system python3
#   PYTHON=python3.12 bash scripts/install_local.sh
#
# After install:
#   source .venv/bin/activate
#   pytest tests/unit/
set -euo pipefail

PYTHON="${PYTHON:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"

py_version="$("$PYTHON" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
case "$py_version" in
    3.10|3.11|3.12)
        ;;
    *)
        echo "ERROR: NEURD local install requires Python 3.10-3.12 (got $py_version)."
        echo "       Python 3.13+ has no open3d wheel; 3.9 lacks numpy 2 wheels for some deps."
        echo "       Set PYTHON=python3.12 or use a conda env: conda create -n neurd python=3.12"
        exit 1
        ;;
esac

if [[ ! -d "$VENV_DIR" ]]; then
    "$PYTHON" -m venv "$VENV_DIR"
fi
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

pip install --upgrade pip

# mesh_processing_tools is pure-Python but pins open3d==0.11.2 in its metadata.
# Install with --no-deps so we can use modern open3d (see requirements-local.txt).
pip install --no-deps mesh_processing_tools==1.0.4

pip install -r requirements-local.txt

# NEURD itself in editable mode.
pip install -e .

# CGAL mean-curvature-flow skeletonizer (calcification_param_Module). Required for the
# MAP decomposition path (thick branches); without it mesh_tools raises
# `NameError: calcification_param` the moment a neuron has a thick limb. Best-effort:
# it needs CGAL/Eigen/gmp/mpfr C++ headers in the env prefix (present in conda envs, or
# `apt install libcgal-dev libeigen3-dev libgmp-dev libmpfr-dev`). If they are missing the
# build is skipped with a warning — the pipeline still works for thin-branch neurons.
echo
echo "Building CGAL skeletonizer extension (calcification_param_Module)..."
if [[ -f "$(python -c 'import sys; print(sys.prefix)')/include/CGAL/version.h" ]]; then
    if pip install ./cgal/cgal_skeleton_param; then
        echo "  CGAL skeletonizer installed."
    else
        echo "  WARNING: CGAL skeletonizer build FAILED — MAP path (thick branches) will crash."
        echo "           See cgal/cgal_skeleton_param/ and PIPELINE.md."
    fi
else
    echo "  WARNING: CGAL headers not found in env prefix — skipping calcification_param_Module."
    echo "           MAP path (thick branches) will raise NameError. To enable: install CGAL 6"
    echo "           (conda: 'conda install -c conda-forge cgal eigen gmp mpfr', or apt libcgal-dev),"
    echo "           then 'pip install ./cgal/cgal_skeleton_param'."
fi

# Second CGAL extension: SDF mesh segmentation (cgal_Segmentation_Module). Without it
# mesh_tools falls back to neurd/_cgal_segmentation.py (KMeans stand-in) which produces
# 0 spines, so spine detection is effectively off. With it built, real CGAL segmentation
# runs and spines work as in Docker. Same build prerequisites as the skeletonizer.
echo
echo "Building CGAL segmentation extension (cgal_Segmentation_Module)..."
if [[ -f "$(python -c 'import sys; print(sys.prefix)')/include/CGAL/version.h" ]]; then
    if pip install ./cgal/cgal_segmentation; then
        echo "  CGAL segmentation installed (spine detection enabled)."
    else
        echo "  WARNING: CGAL segmentation build FAILED — spines fall back to the KMeans stub (0 spines)."
        echo "           See cgal/cgal_segmentation/ and PIPELINE.md."
    fi
else
    echo "  WARNING: CGAL headers not found — skipping cgal_Segmentation_Module."
    echo "           Spine detection will use the KMeans stub (0 spines). To enable: install CGAL 6,"
    echo "           then 'pip install ./cgal/cgal_segmentation'."
fi

echo
echo "Install complete. Activate with:"
echo "  source $VENV_DIR/bin/activate"
echo "Smoke check:"
echo "  pytest tests/unit/test_env_compat.py tests/unit/test_numpy_compat.py tests/unit/test_mesh_tools_compat.py"
