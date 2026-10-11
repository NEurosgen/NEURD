"""Build the CGAL Mean-Curvature-Flow skeletonizer Python extension.

Produces `calcification_param_Module`, which mesh_tools.skeleton_utils imports for
MAP-path skeletonization of thick branches (without it: NameError calcification_param,
so any neuron with a thick limb crashes the MAP decomposition).

Originally built inside the (now-removed) Docker image. This setup finds headers
in the active interpreter's prefix, its base prefix, or system include paths.
GMP/MPFR libraries can come from the environment or the system. Ported to CGAL 6
(C++17; OFF_reader.h -> IO/OFF.h; CGAL::read_OFF -> CGAL::IO::read_OFF).

Build + install into the active env's site-packages:
    python setup.py build_ext --inplace      # local .so next to sources, or
    pip install .                            # into site-packages (preferred)
"""
import os
import sys
from setuptools import setup, Extension

_PREFIXES = list(dict.fromkeys([sys.prefix, sys.base_prefix]))
_INCLUDES = list(dict.fromkeys([os.path.join(p, "include") for p in _PREFIXES]
                             + ["/usr/local/include", "/usr/include"]))
_EIGEN_INCLUDES = [path for root in _INCLUDES for path in (root, os.path.join(root, "eigen3"))]


def header_root(header, roots):
    for root in roots:
        if os.path.isfile(os.path.join(root, header)):
            return root
    raise RuntimeError(f"Missing {header}; searched: {', '.join(roots)}. "
                       "Install CGAL and Eigen development headers in the active environment "
                       "or system (conda-forge: cgal eigen gmp mpfr; "
                       "Debian/Ubuntu: libcgal-dev libeigen3-dev libgmp-dev libmpfr-dev).")


_CGAL_INCLUDE = header_root("CGAL/version.h", _INCLUDES)
_EIGEN_INCLUDE = header_root("Eigen/Core", _EIGEN_INCLUDES)
_LIBRARIES = [os.path.join(prefix, "lib") for prefix in _PREFIXES]

module = Extension(
    "calcification_param_Module",
    sources=["skeleton_param_module.cpp"],
    include_dirs=list(dict.fromkeys([_CGAL_INCLUDE, _EIGEN_INCLUDE] + _INCLUDES)),
    library_dirs=_LIBRARIES,
    libraries=["mpfr", "gmp"],  # CGAL 6 is header-only; only mpfr/gmp need linking
    extra_compile_args=["-std=c++17", "-O2"],
    extra_link_args=["-Wl,-rpath," + path for path in _LIBRARIES],
)

setup(
    name="calcification_param",
    version="1.0",
    description="CGAL mean-curvature-flow skeletonizer (calcification_param_Module)",
    ext_modules=[module],
)
