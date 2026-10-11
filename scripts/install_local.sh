#!/usr/bin/env bash
# Local (non-Docker) install for NEURD on Python 3.12.
#
# Usage:
#   bash scripts/install_local.sh                # creates the repository's .venv
#   PYTHON=python3.12 bash scripts/install_local.sh
#   NEURD_ALLOW_MISSING_CGAL=1 bash scripts/install_local.sh  # development only
# Relative VENV_DIR paths are resolved from the repository, regardless of cwd.
#
# After install:
#   source .venv/bin/activate
#   pytest tests/unit/
set -euo pipefail

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
case "$VENV_DIR" in
    /*) ;;
    *) VENV_DIR="$REPO_DIR/$VENV_DIR" ;;
esac
NEURD_ALLOW_MISSING_CGAL="${NEURD_ALLOW_MISSING_CGAL:-0}"
case "$NEURD_ALLOW_MISSING_CGAL" in
    0|1) ;;
    *) echo "ERROR: NEURD_ALLOW_MISSING_CGAL must be 0 or 1." >&2; exit 1 ;;
esac

py_version="$("$PYTHON" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
case "$py_version" in
    3.10|3.11|3.12)
        ;;
    *)
        echo "ERROR: NEURD local install requires Python 3.10-3.12 (got $py_version)."
        echo "       This installer currently supports and validates only Python 3.10-3.12."
        echo "       Set PYTHON=python3.12 or use a conda env: conda create -n neurd python=3.12"
        exit 1
        ;;
esac

if [[ ! -d "$VENV_DIR" ]]; then
    "$PYTHON" -m venv "$VENV_DIR"
fi
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

active_py_version="$(python -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
case "$active_py_version" in
    3.10|3.11|3.12) ;;
    *)
        echo "ERROR: Activated environment requires Python 3.10-3.12 (got $active_py_version)." >&2
        echo "       Choose a new VENV_DIR instead of reusing $VENV_DIR." >&2
        exit 1
        ;;
esac

cd -- "$REPO_DIR"

# Check before installing Python packages. A venv may obtain native headers from
# its base interpreter (for example conda) or system development packages.
native_headers_available=1
if ! python - <<'PY'
import os
import sys

roots = list(dict.fromkeys([os.path.join(p, "include") for p in (sys.prefix, sys.base_prefix)]
                          + ["/usr/local/include", "/usr/include"]))
headers = {"CGAL/version.h": roots,
           "Eigen/Core": [path for root in roots for path in (root, os.path.join(root, "eigen3"))]}
missing = False
for header, paths in headers.items():
    if not any(os.path.isfile(os.path.join(path, header)) for path in paths):
        print(f"ERROR: Missing {header}; searched: {', '.join(paths)}.", file=sys.stderr)
        missing = True
if missing:
    print("Install native development dependencies first: conda-forge cgal eigen gmp mpfr, "
          "or Debian/Ubuntu libcgal-dev libeigen3-dev libgmp-dev libmpfr-dev.", file=sys.stderr)
    sys.exit(1)
PY
then
    native_headers_available=0
    if [[ "$NEURD_ALLOW_MISSING_CGAL" != 1 ]]; then
        echo "ERROR: Both native CGAL backends are required for a complete installation." >&2
        echo "       NEURD_ALLOW_MISSING_CGAL=1 permits a development-only installation." >&2
        exit 1
    fi
    echo "WARNING: Explicit development opt-out: native CGAL builds will be skipped." >&2
fi

python -m pip install --upgrade pip

# mesh_processing_tools is pure-Python but pins open3d==0.11.2 in its metadata.
# Install with --no-deps so we can use modern open3d (see requirements.txt).
python -m pip install --no-deps mesh_processing_tools==1.0.4

python -m pip install -r requirements.txt

# NEURD itself in editable mode, plus the test runner for the tests/unit gate.
python -m pip install -e ".[dev]"

# CGAL mean-curvature-flow skeletonizer (calcification_param_Module). Required for the
# MAP decomposition path (thick branches). Segmentation is required for spine
# detection. Build and import both binaries; Python fallbacks do not satisfy this.
native_complete="$native_headers_available"
build_native() {
    local package="$1" module="$2" entry_point="$3"
    echo "Building native extension $module..."
    if python -m pip install "$package" && python - "$module" "$entry_point" <<'PY'
import importlib
from importlib.machinery import EXTENSION_SUFFIXES
import sys

module = importlib.import_module(sys.argv[1])
origin = getattr(module, "__file__", "")
if not origin or not any(origin.endswith(suffix) for suffix in EXTENSION_SUFFIXES):
    raise RuntimeError(f"{sys.argv[1]} is not a compiled extension: {origin}")
if not callable(getattr(module, sys.argv[2], None)):
    raise RuntimeError(f"{sys.argv[1]} is missing native entry point {sys.argv[2]}")
print(f"  Verified native backend: {origin}")
PY
    then
        return 0
    fi
    native_complete=0
    echo "ERROR: Native backend $module failed to build or import." >&2
    echo "       Check the compiler output above and CGAL/Eigen/GMP/MPFR development dependencies." >&2
    if [[ "$NEURD_ALLOW_MISSING_CGAL" != 1 ]]; then
        return 1
    fi
    echo "WARNING: Continuing only because NEURD_ALLOW_MISSING_CGAL=1 was explicitly set." >&2
}

if [[ "$native_headers_available" == 1 ]]; then
    build_native ./cgal/cgal_skeleton_param calcification_param_Module calcification_param
    build_native ./cgal/cgal_segmentation cgal_Segmentation_Module cgal_segmentation
fi

echo
if [[ "$native_complete" == 1 ]]; then
    echo "Install complete; both native CGAL backends verified. Activate with:"
else
    echo "Development install complete with missing native CGAL backends." >&2
    echo "MAP skeletonization or spine segmentation may be unavailable. Activate with:" >&2
fi
echo "  source $VENV_DIR/bin/activate"
echo "Smoke check:"
echo "  pytest tests/unit/test_env_compat.py tests/unit/test_numpy_compat.py tests/unit/test_mesh_tools_compat.py"
