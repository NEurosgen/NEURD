# CGAL C++ extensions

`calcification_param_Module` — the CGAL **Mean Curvature Flow** mesh skeletonizer used by
`mesh_tools.skeleton_utils` for the **MAP decomposition path** (thick branches,
`width > width_threshold_MAP`). It contracts a closed mesh to a medially-centered 1D
skeleton and writes a `.cgal` polyline file.

## Why this lives in the repo

It used to be built inside the (now-removed) Docker image. After Docker was dropped, the
extension was no longer compiled, so `mesh_tools/skeleton_utils.py` (which does
`try: from calcification_param_Module import calcification_param / except: pass`) left the
name unbound. Any neuron with a thick limb then crashed the MAP decomposition with
`NameError: name 'calcification_param' is not defined`. Thin-branch / single-soma neurons
never hit the MAP path, so the breakage was invisible on small test neurons.

Sources recovered from git history (`docker/CGAL/cgal_skeleton_param/`) and **ported to
CGAL 6**: C++17; `CGAL/IO/OFF_reader.h` → `CGAL/IO/OFF.h`; `CGAL::read_OFF` →
`CGAL::IO::read_OFF`; OFF points typed to the Polyhedron kernel (`Kernel2::Point_3`).

## Build

`scripts/install_local.sh` builds it automatically (best-effort) when CGAL headers are
present in the env prefix. To build manually:

```bash
# needs CGAL 6 + Eigen + gmp + mpfr headers in the active env:
#   conda install -c conda-forge cgal eigen gmp mpfr
#   (or: apt install libcgal-dev libeigen3-dev libgmp-dev libmpfr-dev)
pip install ./cgal/cgal_skeleton_param
```

`setup.py` is env-relative (derives include/lib dirs from `sys.prefix`), so it picks up
the conda env's CGAL automatically.

## Verify

```python
import neurd
from mesh_tools import skeleton_utils as sk
assert "calcification_param" in sk.__dict__   # bound -> MAP path will work
```
