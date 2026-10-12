# Validated local installation

Use `scripts/install_local.sh` from a checkout. It installs
`mesh_processing_tools==1.0.4` without its historical dependency pins, installs
NEURD with test dependencies, and builds both CGAL extensions. Installing the
NEURD wheel alone does not supply the mesh provider or native extensions.

The installer accepts Python 3.10–3.12. The measured installation below covers
**Linux x86_64 / Python 3.12.13**, with a fresh venv and existing native development
tools. Other interpreters and operating systems have not received this gate.

## Reproduce the dependency versions

Prepare C++17, Python development headers, CGAL, Boost, Eigen, GMP and MPFR.
The native setups search the active and base interpreter prefixes and system
include directories, including Eigen's `eigen3` layout. Debian/Ubuntu development
packages are `libcgal-dev libeigen3-dev libgmp-dev libmpfr-dev`; a conda-forge
toolchain can provide `cgal eigen gmp mpfr`. MeshLab and Xvfb are separate system
requirements for raw soma processing.

From the repository, with an interpreter whose native prerequisites are available:

```bash
PYTHON=/path/to/python3.12 \
VENV_DIR=.venv \
PIP_CONSTRAINT="$PWD/constraints/linux-py312-20261012.txt" \
bash scripts/install_local.sh
source .venv/bin/activate
pytest tests/unit/
```

The [constraints file](../constraints/linux-py312-20261012.txt) records 207 version
pins. Versions from the previously successful reference environment are used where
available; remaining pins come from the fresh installation observations. A constraint
does not require the corresponding package to be installed. This is not a standalone install
list: the installer supplies NEURD and builds the two native packages from this
repository. It includes test and packaging tools. Native headers, shared libraries,
compiler, system tools and wheel hashes are not locked by this file.

The installer can be called by absolute path from another directory. Default and
relative `VENV_DIR` paths resolve from the repository. A reused venv must itself
run a supported Python, even when `PYTHON` names a supported base interpreter.

## What the fresh installation establishes

The venv began with only pip, `include-system-site-packages = false`, empty
`PYTHONPATH`, and no NumPy, SciPy, trimesh, Open3D, mesh_tools or NEURD import.
Both native modules were compiled from repository sources and loaded from the
fresh venv. All 198 installed distributions and checked Python dependency
providers belong to that venv; the package itself is an editable source checkout.

Selected versions: NumPy 2.4.6, SciPy 1.17.1, pandas 2.3.3, trimesh 4.12.2,
Open3D 0.19.0, NetworkX 3.6.1 and meshparty 2.0.3. The unchanged runtime ranges in
`requirements.txt` permit other versions; the constraints identify the observed
profile for these results.

The native toolchain was inherited through `sys.base_prefix`: CGAL 6.1.1,
Eigen 5.0.1, Boost 1.88.0, GMP 6.3.0, MPFR 4.2.2 and GCC/G++ 13.3.0.
The generated extension RPATH depends on this base prefix. This is a fresh
Python-package installation, not a bare-OS installation or a portable binary
wheel. The existing conda Python packages were not reused.

Installed metadata was checked for missing requirements, version conflicts and
the `dev` extra. The only exception is
`mesh-processing-tools 1.0.4 -> open3d==0.11.2`; consequently ordinary `pip check`
returns 1 with that single conflict. No installed metadata was rewritten.
The retained author/meshparty dependencies still pull visualization and cloud
packages: 198 distributions is an observed environment count, not a dependency
reduction measurement.

## Validation and publication boundary

The complete unit suite in the populated local checkout passes
**1,063 tests / 1 expected skip**. Some real-mesh cases read the ignored
`tests/990_mesh*` files and local `Applications/Tutorials/Auto_Proof_Pipeline/neuron_2530864375.off`.
These inputs were copied and fingerprinted in the fresh checkout; the count does
not establish that an unpopulated Git clone can run every case. The skip
exercises an optional stub only when real ipyvolume is unavailable; real ipyvolume
is installed here. The three removed import roots remain blocked during validation.
All 30 NEURD modules import with all eight default compatibility patches active.
Actual patched Open3D decimation and Embree ray intersections pass.

Fresh native SDF runs cover the unset, reference-25 and fast-12 ray profiles.
Unset and reference-25 arrays match; reference SDF matches the local golden SDF,
and fast/reference correlation is 0.99595. A generated closed 1,280-face prolate
mesh exercises native MAP skeletonization and produces 54 positive edges.
The Minnie gate processes one 29,495-face branch: nine spines / 2,854 spine faces,
nine scalar volumes and 27 saved arrays match the prior native result exactly.

The complete raw H01 gate processes `neuron_2530864375.off` (507,737 faces,
253,307 vertices), with seed 0, H01 parameters, spines enabled, refinement off
and fast-12 SDF rays. It retains one soma, five limbs, 28 branches and
18 spines / 8,145 spine faces. All 218 exported files and 239 NumPy arrays match
the prior reference exactly. A separate reload verifies all 28 ordered branch
maps and 18 spine masks. This raw case takes the MP route (22 calls); the separate
generated-mesh smoke exercises native MAP.

Observed raw processing took 389.67 seconds with a process peak RSS of
1,269,178,368 bytes (1.18 GiB). Unit and other gates ran concurrently, so these
numbers are recorded observations for this correctness run. They do not support
a comparison with the earlier 160.55-second run or a speed/memory improvement claim.

The NEURD wheel was also built, inspected and installed with `--no-deps --target`
into a separate directory. All 30 module files import from that installation with
exact source bytes, eight active patches and the fresh native providers. Existing
fresh dependencies are shared for this check; it does not establish a standalone
wheel installation with native providers or batch console commands.

The first unconstrained fresh installation passed imports, unit tests, native SDF/MAP
checks and the Minnie branch comparison, but failed the full raw H01 equivalence
check: 28 spines instead of 18, with eight branch-local face counts already changed
before spine detection. Native provider A/Bs agreed exactly; a trimesh-only A/B
also agreed on two identical archived branch inputs. These bounded diagnostics
do not identify the particular dependency responsible for the earlier geometry
change. The first attempt is preserved separately under
`artifacts/clean_install_validation_20261012/v1/`; use the constrained profile for
the reference processing result rather than assuming every permitted version is equivalent.

The constrained fresh installation started with 203 pins. Its resolved environment
introduced four additional transitive packages; the final profile adds their exact
already-installed versions. All 195 external distributions in the observed
environment are now pinned; NEURD and the two native packages are supplied from
source. Twelve unused constraints are retained. This final addition changes no
installed package or runtime source, and separate resolver checks verify the profile.

Constrained-install evidence is under `artifacts/clean_install_validation_20261012/v2/`.
These artifacts and dataset paths are local. In particular, the SDF oracle files
`tests/990_mesh*` are ignored by Git, and the real branch/raw meshes are not a
downloadable reproduction bundle. A [controlled whole-neuron comparison](BENCHMARKING.md)
and an [executed publication example](../examples/README.md) are now available.
Portable fixtures and required native CI remain release work. The installation
checks establish computational behavior; anatomical accuracy and performance
claims require their own evidence and measurement scope.
