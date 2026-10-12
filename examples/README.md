# Single-neuron example

[The H01 notebook](h01_single_neuron.ipynb) demonstrates the public API from a
raw neuron mesh through decomposition, spine detection and export. It includes
saved tables, measured time and memory, and a skeleton overview from an actual
run. The figure also shows a separate manually reviewed H01 spine with its
parent-branch context. Detected candidates can include mesh defects; their
counts do not establish anatomical quality.

## View the saved example

Open the notebook in GitHub, Jupyter or your editor to read the saved outputs.
No dataset or scientific environment is needed to inspect them.

For a browser view, download [the HTML companion](h01_single_neuron.html) and
open it locally. GitHub may show its source rather than render the page. The HTML
contains the notebook's saved results and figure and does not execute Python.

After saving updated notebook outputs, rebuild the HTML from the repository root:

```bash
python scripts/export_example_notebook.py examples/h01_single_neuron.ipynb --overwrite
```

The renderer requires `nbformat`, `Markdown` and `Pygments`. It embeds the saved
PNG and tables without rerunning the processing or fetching remote assets.

The accompanying files have different roles:

| File | Role |
|---|---|
| [`h01_single_neuron.ipynb`](h01_single_neuron.ipynb) | Explanation, runnable public API calls and saved outputs |
| [`_single_neuron_example.py`](_single_neuron_example.py) | Measurement, environment fingerprints and plotting helpers |
| [`h01_reference_run.json`](h01_reference_run.json) | The saved observation shown in this notebook edition: input checksum, configuration, geometry, measurements and environment |
| [`data/h01_reviewed_spine/`](data/h01_reviewed_spine/README.md) | Small, manually reviewed C03 spine and parent crop, face membership and source/review fingerprints |

The reference-run JSON records one notebook observation. It retains the measured
stages and their memory scope, configuration and input/environment fingerprints.

## Run the notebook

Use the [validated installation](../docs/INSTALLATION.md) with its constrained
scientific environment, compiled CGAL providers, MeshLab and Xvfb. Follow the
notebook-tool setup below and select that scientific environment as the kernel
in Jupyter or your editor.

Notebook tooling was checked in a separate seven-package overlay, preserving
the scientific environment. Activate the validated environment, run from the
repository root, and choose an overlay directory outside the checkout. The
`--no-deps` option relies on support packages already in the validated profile.
Registration below creates a named user kernel for Jupyter or your IDE:

```bash
NEURD_NOTEBOOK_TOOLS=/path/to/neurd-notebook-tools
PIP_CONSTRAINT="$PWD/constraints/linux-py312-20261012.txt" \
python -m pip install --no-deps --target "$NEURD_NOTEBOOK_TOOLS" \
  ipykernel==7.4.0 nbclient==0.11.0 jupyter-client==8.10.0 \
  tornado==6.5.10 pyzmq==27.2.0 debugpy==1.8.22 nest-asyncio2==1.7.4
PYTHONPATH="$NEURD_NOTEBOOK_TOOLS" python -m ipykernel install --user \
  --name neurd-h01-example --display-name "NEURD H01 example" \
  --env PYTHONPATH "$NEURD_NOTEBOOK_TOOLS" --env PYTHONHASHSEED 0 \
  --env NEURD_EXAMPLE_MESH /path/to/neuron_2530864375.off
```

Select the `NEURD H01 example` kernel. Jupyter startup contributes to kernel
lifetime RSS, and first-use Numba compilation contributes to processing time.
The saved observation records both scopes.

Keep the notebook, `_single_neuron_example.py` and `data/` together in `examples/`.
Launch your notebook application from the repository
or its `examples/` directory. Set the input before starting a fresh kernel, for
example through the application's environment:

```bash
export NEURD_EXAMPLE_MESH=/path/to/neuron_2530864375.off
export PYTHONHASHSEED=0
```

You can also edit `MESH_PATH` in the setup cell. For another H01 reconstruction,
change `SEGMENT_ID` to its identifier as well. Run all cells in order in a fresh
kernel. The notebook checks the named reference input's checksum and records the
effective configuration; another input or parameter profile is a new experiment.

Optional `NEURD_EXAMPLE_OUTPUT` selects a new output directory. Otherwise the
notebook creates a unique temporary directory, available as `OUTPUT_ROOT`.
Evaluate `OUTPUT_ROOT` in an extra cell to locate the exports, logs, measurement
summary, RSS samples and figure files. Existing exports are not overwritten.
Optional `NEURD_EXAMPLE_CPU` selects an allowed Linux CPU for affinity-controlled
measurements.

The reference mesh is not bundled. The [official H01 data
page](https://h01-release.storage.googleapis.com/data.html) describes access to
released reconstructions, but the release layer and conversion procedure for
this particular local OFF were not recovered. Its checksum identifies the
measured input; the portal does not provide a verified recipe for retrieving
a byte-identical copy. The notebook states the input units and the limits of
the geometry and performance observations. The bundled spine illustration comes
from neuron `5319188363`, limb `000`, branch `150`; the timed input is neuron
`2530864375`. The notebook identifies these two sources separately and describes
the engineering optimizations without a version-to-version comparison.
