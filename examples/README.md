# Single-neuron example

[The H01 notebook](h01_single_neuron.ipynb) demonstrates the public API from a
raw neuron mesh through decomposition, spine detection and export. It includes
saved tables, measured time and memory, and a geometry figure from an actual run.

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
| [`h01_benchmark_summary.json`](h01_benchmark_summary.json) | A separate repeated before/after experiment, including source versions, individual trials and measurement scope |

The reference-run JSON records one notebook observation. The benchmark JSON
records the separate comparison described in the final notebook section;
their cache and memory scopes differ.

## Run the notebook on your mesh

Use the [validated installation](../docs/INSTALLATION.md) with its constrained
scientific environment, compiled CGAL providers, MeshLab and Xvfb. Follow the
[notebook-tool instructions](../docs/BENCHMARKING.md#run-the-notebook) and select
that scientific environment as the kernel in Jupyter or your editor.

Keep the notebook, `_single_neuron_example.py` and `h01_benchmark_summary.json`
together in `examples/`. Launch your notebook application from the repository
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
the geometry and performance observations.
