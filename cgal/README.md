# CGAL C++ extensions

`cgal_Segmentation_Module` provides the actual CGAL SDF and graph-based mesh
segmentation used by soma detection and spine extraction. Spine-enabled public
API/CLI calls require this compiled module; the compatibility Python provider
does not provide equivalent spine segmentation.

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

`scripts/install_local.sh` builds and verifies both native modules. Missing headers,
build failures and failed native imports stop installation by default. An explicit
`NEURD_ALLOW_MISSING_CGAL=1` permits an incomplete development installation.
To build manually:

```bash
# needs CGAL 6 + Eigen + gmp + mpfr headers in the active env:
#   conda install -c conda-forge cgal eigen gmp mpfr
#   (or: apt install libcgal-dev libeigen3-dev libgmp-dev libmpfr-dev)
pip install ./cgal/cgal_skeleton_param
pip install ./cgal/cgal_segmentation
```

Both build scripts search the active interpreter prefix, `sys.base_prefix` (useful
for a venv made from a conda interpreter), `/usr/local/include` and `/usr/include`.
Eigen is detected in either the include root or its `eigen3` subdirectory. Library
paths and runtime search paths include both interpreter prefixes; the compiler also
searches its normal system library directories. The maintained build was validated
with CGAL 6 and Python 3.12.

## Verify

```python
import neurd
from neurd.run_config import require_compiled_cgal
from mesh_tools import skeleton_utils as sk
require_compiled_cgal()
assert "calcification_param" in sk.__dict__   # bound -> MAP path will work
```

## SDF profiles and stage timing

The public `RunConfig` and raw-neuron CLI use 25 rays for `reference`, or 12 for
`fast`. Lower-level native callers can set `NEURD_SDF_RAYS`; the native default
when unset is 25. These profiles are separate processing configurations, because
ray count can change segmentation.

CSV output is buffered, preserving the original decimal formatting, row order and
newline bytes. `NEURD_CGAL_TIMINGS=1` prints OFF-read, SDF, SDF-write, clustering
and label-write durations to stderr. Timings are disabled by default.

Compare separately built old/new providers without changing the installed environment:

```bash
python setup.py build_ext --build-lib /tmp/cgal-new --build-temp /tmp/cgal-new-build
# Run this from the repository root, with an old build saved separately:
python tests/tools/benchmark_cgal_io.py --baseline /tmp/cgal-old \
    --candidate /tmp/cgal-new --mesh /path/to/fixture.off --repeat 5 \
    --out /tmp/cgal-ab.json
```

Run the build command from `cgal/cgal_segmentation/`. The benchmark uses fresh processes,
interleaved order, fixed ray/cluster settings and copied inputs, and rejects differing
CSV hashes. Whole-neuron time and memory require a separate end-to-end measurement.
