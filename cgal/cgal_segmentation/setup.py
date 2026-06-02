"""Build the CGAL SDF mesh-segmentation Python extension.

Produces `cgal_Segmentation_Module`, which mesh_tools.trimesh_utils imports
(`import cgal_Segmentation_Module as csm`) for SDF-based mesh segmentation — the
real CGAL `sdf_values` + `segmentation_from_sdf_values`. Spine detection needs the
fine over-segmentation this gives; the pure-Python KMeans stand-in
(neurd/_cgal_segmentation.py) produces 0 spines, which is why spines were disabled.
With this module built, the stub is bypassed and spine detection works as in Docker.

Originally built inside the (now-removed) Docker image. This setup is env-relative:
it picks up CGAL / Eigen / gmp / mpfr from the active interpreter's prefix
(`sys.prefix`), e.g. the conda env, which already ships all of them. Ported to CGAL 6
(C++17; OFF_reader.h -> IO/OFF.h; CGAL::read_OFF -> CGAL::IO::read_OFF). Same recipe
as cgal/cgal_skeleton_param.

Build + install into the active env's site-packages:
    python setup.py build_ext --inplace      # local .so next to sources, or
    pip install .                            # into site-packages (preferred)
"""
import os
import sys
from setuptools import setup, Extension

_PREFIX = sys.prefix  # active env (conda/venv) — has CGAL headers + gmp/mpfr libs

module = Extension(
    "cgal_Segmentation_Module",
    sources=["neuron_cgal_segmentation.cpp"],
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
    name="cgal_Segmentation",
    version="1.0",
    description="CGAL SDF mesh segmentation (cgal_Segmentation_Module)",
    ext_modules=[module],
)
