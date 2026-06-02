"""Build the CGAL Mean-Curvature-Flow skeletonizer Python extension.

Produces `calcification_param_Module`, which mesh_tools.skeleton_utils imports for
MAP-path skeletonization of thick branches (without it: NameError calcification_param,
so any neuron with a thick limb crashes the MAP decomposition).

Originally built inside the (now-removed) Docker image. This setup is env-relative:
it picks up CGAL / Eigen / gmp / mpfr from the active interpreter's prefix
(`sys.prefix`), e.g. the conda env, which already ships all of them. Ported to CGAL 6
(C++17; OFF_reader.h -> IO/OFF.h; CGAL::read_OFF -> CGAL::IO::read_OFF).

Build + install into the active env's site-packages:
    python setup.py build_ext --inplace      # local .so next to sources, or
    pip install .                            # into site-packages (preferred)
"""
import os
import sys
from setuptools import setup, Extension

_PREFIX = sys.prefix  # active env (conda/venv) — has CGAL headers + gmp/mpfr libs

module = Extension(
    "calcification_param_Module",
    sources=["skeleton_param_module.cpp"],
    include_dirs=[
        os.path.join(_PREFIX, "include"),
        os.path.join(_PREFIX, "include", "eigen3"),
    ],
    library_dirs=[os.path.join(_PREFIX, "lib")],
    libraries=["mpfr", "gmp"],  # CGAL 6 is header-only; only mpfr/gmp need linking
    extra_compile_args=["-std=c++17", "-O2"],
    extra_link_args=["-Wl,-rpath," + os.path.join(_PREFIX, "lib")],
)

setup(
    name="calcification_param",
    version="1.0",
    description="CGAL mean-curvature-flow skeletonizer (calcification_param_Module)",
    ext_modules=[module],
)
