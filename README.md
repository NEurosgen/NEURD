# NEURD — local single-neuron segmentation

A research fork of [NEURD](https://github.com/reimerlab/NEURD) for processing a
single-soma EM reconstruction locally: raw mesh → soma, limbs, branches,
skeletons and spine candidates, with geometry exports and resumable batch runs.

**Research preview:** [v0.2.0-preview.1](https://github.com/NEurosgen/NEURD/releases/tag/v0.2.0-preview.1).
The [release notes](docs/releases/v0.2.0-preview.1.md) describe its validation
scope. See [NOTICE.md](NOTICE.md) for attribution and the unresolved inherited-code
licensing status.

The workflow assumes a single neuron with one soma and no glia or nuclei.
Spine candidates can include mesh defects; counts and volumes do not establish
anatomical accuracy. Synapse assignment, axon classification, multi-soma splitting
and connectome analysis are outside the supported workflow.

## Install

The validated profile is **Linux x86_64 / Python 3.12.13** with existing native
development tools. Prepare CGAL, Eigen, Boost, GMP and MPFR headers, MeshLab and
Xvfb as described in the [installation guide](docs/INSTALLATION.md).

```bash
PYTHON=python3.12 \
PIP_CONSTRAINT="$PWD/constraints/linux-py312-20261012.txt" \
bash scripts/install_local.sh
source .venv/bin/activate
```

Run from the repository root. The installer supplies the mesh provider and builds
both required [CGAL extensions](cgal/README.md). The constraints select the observed
dependency profile; an unconstrained installation changed the reference geometry.
Native binaries, a bare-OS installation and other platforms require further
validation. Detailed prerequisites and the historical dependency-pin exception
are documented in the installation guide.

## Run

Load a locally supplied mesh, select an explicit configuration and export into a
fresh directory:

```python
from pathlib import Path
import neurd
from neurd.run_config import RunConfig
from neurd.mesh_io import load_mesh
from neurd.segmentation_pipeline import segmentation_pipeline
from neurd.neuron_export import save_segmentation

mesh = load_mesh("neuron.off")
config = RunConfig(dataset="h01", calculate_spines=True, sdf_profile="fast")
neuron = segmentation_pipeline(mesh, segment_id=123, max_somas=1, config=config)
save_segmentation(neuron, Path("output/neuron_123"))

for limb in neuron.limbs:
    for branch in limb:
        branch.mesh, branch.skeleton, branch.spines
```

Import `neurd` before its geometry providers to install the required compatibility
patches. Request spines explicitly: the Python API defaults to spines disabled.
The `reference` SDF profile uses 25 rays and `fast` uses 12; ray counts and optional
refinement select different processing configurations.

For a directory of meshes:

```bash
python process_all_neurons.py path/to/meshes path/to/output \
    --dataset h01 --spines --sdf-profile fast

python process_all_neurons_parallel.py path/to/meshes path/to/output \
    --dataset h01 --spines --sdf-profile fast \
    --max-workers 2 --input-size-budget-mb 1024
```

Batch processing uses isolated workers, records failures and verifies input,
configuration, runtime and output fingerprints before resuming completed results.
Use a new output directory when those inputs change. Input and output directories
must be separate and non-nested. `--retry-failed` retries failures; `--no-save`
computes without exports. Input decimation is disabled unless explicitly requested.
Use `--help` for resource limits and other options.

Exports include soma, limb, branch and spine meshes, skeleton arrays, graph
mappings and supplied face-index/provenance records. Missing original raw-face
ownership stays unavailable. The library saver writes directly to the supplied
directory; the batch runners add staging and completion verification.

## Example

The [H01 notebook](examples/h01_single_neuron.ipynb) contains saved results from
complete raw-neuron processing: geometry, exports, time, memory and environment
fingerprints. Its figure includes a separate manually reviewed C03 spine with
bundled geometry and explicit source attribution.

Download the [standalone HTML](https://github.com/NEurosgen/NEURD/releases/download/v0.2.0-preview.1/h01_single_neuron.html)
to view the saved example without installing the scientific environment.
The [example guide](examples/README.md) explains viewing, kernel setup and rerunning.
The complete raw input must be supplied locally; a verified remote acquisition
recipe for a byte-identical copy is not provided.

## Documentation

- [Installation and validation scope](docs/INSTALLATION.md).
- [Notebook viewing and rerun guide](examples/README.md).
- [Optional spine refinement and exported-branch processing](docs/SPINE_REFINEMENT.md).
- [Release notes](docs/releases/v0.2.0-preview.1.md) and [attribution](NOTICE.md).

[Developer documentation](docs/development/README.md) covers architecture,
compatibility contracts, diagnostics and verification tools.

## Credits

The original framework and scientific algorithms are by Brendan Celii and
collaborators. See [upstream NEURD](https://github.com/reimerlab/NEURD) and
[the publication](https://doi.org/10.1038/s41586-025-08660-5).
This fork focuses on local geometry processing and adds optional spine refinement.
