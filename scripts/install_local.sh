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

echo
echo "Install complete. Activate with:"
echo "  source $VENV_DIR/bin/activate"
echo "Smoke check:"
echo "  pytest tests/unit/test_env_compat.py tests/unit/test_numpy_compat.py tests/unit/test_mesh_tools_compat.py"
