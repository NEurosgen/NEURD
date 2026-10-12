# Whole-neuron benchmark and notebook

The controlled comparison processes the complete raw H01 mesh
`neuron_2530864375.off`: **253,307 vertices / 507,737 faces**, SHA256
`ed914a7b70f370300b6ece631de9b1420eb5f0f876f01fb97f775ef63e203c3e`.
The dataset is supplied locally and is not bundled with the repository.
Portable data acquisition and native CI remain release work.

## Recorded result, 2026-10-12

Three measured runs per version retain one soma, five limbs, 28 branches and
18 spines / 8,145 spine faces. Independent reload checks preserve original array
dtypes, shapes, bytes and NPZ member order, as well as the supplied branch maps,
spine masks and new provenance records.

| Measurement | Baseline `e2fb11026` | Candidate `4b7fa78d9` |
|---|---:|---:|
| Input loading, median | 2.135 s | 1.397 s |
| Full construction, median | 210.153 s | 224.298 s |
| Export, median | 1.203 s / 134 files | 1.434 s / 218 files |
| Load + construction + export, median | 213.383 s | 227.159 s |
| Processing range | 207.306–221.653 s | 219.456–228.367 s |
| Worker kernel peak RSS, median | 1.139 GiB | 1.155 GiB |
| Worker peak RSS range | 1.129–1.154 GiB | 1.153–1.170 GiB |
| Sampled worker + descendants peak, median | 1.302 GiB | 1.331 GiB |

Loading is **34.6% faster** in this experiment. The full workflow is **6.5% slower**,
and median worker peak RSS is **1.4% higher**. The candidate exports 84 additional
provenance records; their individual cost was not measured. The larger measured
difference is in construction. Its cause has not been isolated. Three pairs on one neuron, with
overlapping timing ranges and an unchanged dynamic CPU governor, do not establish
a general performance change. These results do not support advertising a new
whole-pipeline speed or memory improvement.

The excluded setup observations were 196.229 s / 1.149 GiB for baseline and
218.599 s / 1.194 GiB for candidate. The candidate creates a Numba cache during
setup; measured runs reuse it in fresh processes. Profiling construction under
the same protocol is the next performance task, before selecting another
optimization or making broader speed claims.

## Comparison protocol

The baseline is `e2fb110263d1707089f44dbdc08d6eb9d9fc6521`, the integration
candidate before the owned OFF loader, label grouping/assembly and subsequent
post-CGAL optimizations and readability work. The runtime candidate is
`4b7fa78d993444530444b3c440d0e451823a5b58`. Both use the same
[constrained Python environment](INSTALLATION.md), with 198 installed distributions,
and the same compiled segmentation and skeletonization providers. Neither source
snapshot is installed over that environment.

The effective settings are H01, spines enabled, refinement disabled, fast-12 SDF
rays, `max_somas=None`, segment ID `2530864375`, Python/NumPy seed 0 and
`PYTHONHASHSEED=0`. The recorded runs pin workers to Linux CPU 2 and set
OMP, OpenBLAS, MKL, NumExpr and Numba threads to one. CPU 2 identifies this host's
measurement choice; choose an allowed CPU when reproducing elsewhere.

Each source gets one excluded setup run and a separate persistent Numba cache.
Three measured pairs follow, ordered baseline/candidate, candidate/baseline,
baseline/candidate. Every run starts a new interpreter in a private working
directory. No benchmark jobs overlap. Filesystem caches are not flushed, so the
setup runs describe first use of the selected Numba cache, not cold OS caches.

`tests/tools/benchmark_pipeline.py` measures imports/setup separately, then
loading, complete construction and export with `perf_counter`. Construction
includes soma extraction, limbs, branches, skeletons, widths and detected spines.
The processing total sums those three stages; source/input hashing, timeline
bookkeeping and subsequent QC are excluded. No profiler or forced garbage
collection is used.

The Linux kernel `ru_maxrss` peak is captured immediately after export, before
QC. It covers the worker's lifetime, including imports, and excludes its separate
MeshLab/Xvfb processes. A common 50 ms sampler also records worker and descendant
RSS through export. The latter is an estimate: shared pages count in each process,
and brief child peaks can be missed. Median, minimum, maximum and every measured
trial are retained for timings and memory.

Checks compare ordered numeric values of meshes, skeletons, parent-face masks,
widths and scalar spine volumes. Original dtypes are recorded; numeric storage
widths may differ while canonical values remain exact. Common export arrays and
geometry are compared, and the new graph-identity resolution field is treated as
an added schema field. Source/configuration, installed package payloads, native
providers and input hashes are guarded before and after the sequence. Worker
errors, timeouts, changed outputs or changed guarded inputs fail the comparison.

The exporter workload grows from 134 to 218 files: 28 branches each gain
`parent_face_indices.npy`, `face_provenance.json` and `branch_origin.json`, or
**84 additional provenance files**. Export and total timings include each
version's actual saver. They therefore compare complete workflows with different
export obligations. The separate release gates validate the new ownership and
provenance records.

Recorded results and scope are in
[the public summary](../examples/h01_benchmark_summary.json). This comparison
excludes earlier optimizations of the original framework and the buffered native
CSV change already present in the baseline. Computational equivalence does not
establish anatomical accuracy. This H01 anchor exercises MP processing; native
MAP is covered separately by the generated-mesh installation smoke check.

## Reproduce the comparison

Activate the validated Python environment, provide the checksummed mesh, and use
separate source/output directories. Run from the current repository checkout:

```bash
BENCH_ROOT=$(mktemp -d)
mkdir -p "$BENCH_ROOT/baseline" "$BENCH_ROOT/candidate"
git archive e2fb110263d1707089f44dbdc08d6eb9d9fc6521 \
  neurd process_all_neurons.py tests/tools/neuron_metrics.py configs \
  requirements.txt setup.py README.md | tar -xf - -C "$BENCH_ROOT/baseline"
git archive 4b7fa78d993444530444b3c440d0e451823a5b58 \
  neurd process_all_neurons.py tests/tools/neuron_metrics.py configs \
  requirements.txt setup.py README.md constraints | tar -xf - -C "$BENCH_ROOT/candidate"
python tests/tools/benchmark_pipeline.py \
  --baseline-source "$BENCH_ROOT/baseline" \
  --candidate-source "$BENCH_ROOT/candidate" \
  --neuron /path/to/neuron_2530864375.off --out "$BENCH_ROOT/results" \
  --dataset h01 --sdf-profile fast --seed 0 --segment-id 2530864375 \
  --cpu 2 --repeat 3 --timeout 1200
```

Keep the output outside both source trees and use a new destination on every run.
The result directory contains `report.json`, guards, per-run logs, sampled RSS,
exports and per-run receipts. Use the summary's source/provider/input
fingerprints when comparing another machine or installation.

## Run the notebook

[The H01 example](../examples/h01_single_neuron.ipynb) uses the public API and
[`_single_neuron_example.py`](../examples/_single_neuron_example.py). Keep the
helper beside the notebook, select the constrained environment's kernel and set
`NEURD_EXAMPLE_MESH` to your input. The notebook creates a new output directory,
records metrics and fingerprints, and renders the neuron and a detected spine.
Its observation uses a fresh Jupyter kernel and a new Numba cache. Jupyter startup
contributes to kernel lifetime RSS; first-use compilation contributes to
processing time. That observation is separate from the repeated benchmark.

Notebook tooling was checked in a separate seven-package overlay, preserving the
scientific environment. To reproduce that overlay, choose a directory outside the
checkout; `--no-deps` relies on support packages already in the validated profile:

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

Select that kernel in Jupyter or your IDE. Optional `NEURD_EXAMPLE_CPU` selects
an allowed Linux CPU; the measurement used 2. The saved notebook and accompanying
summary contain basename/hash identifiers, rather than local absolute data paths.
