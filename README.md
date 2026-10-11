# NEURD — slim single-soma segmentation fork

A fork of [reimerlab/NEURD](https://github.com/reimerlab/NEURD) narrowed to **one job**: turn a
single-neuron EM mesh into a structured decomposition, as fast and as lean as possible.

```
mesh (.off)  ->  Neuron
                  ├── somas:  S0                       (cell body: mesh + center)
                  └── limbs:  L0, L1, ...              (processes left after cutting the soma out)
                        ├── branches: 0, 1, 2, ...     (skeleton segments between branch points)
                        │     ├── mesh, mesh_face_idx  (branch submesh + its faces in the parent)
                        │     ├── skeleton             (centreline)
                        │     ├── width_array          (from SDF)
                        │     └── spines               (raw spine submeshes)
                        └── concept_network            (branch graph, rooted at the soma contact)
```

Everything downstream of that — automated proofreading, multi-soma splitting, synapse assignment,
cell typing, axon labelling, SWC export, connectome/motif/proximity analysis, GNN, the cloud and
dataset adapters — **is removed**. The stable geometry refactor reduced the package from
~55k to **15.4k LOC across 18 files**. The integration branch adds opt-in spine refinement,
validated run configuration, branch-batch tools and bounded OFF loading. The candidate contains
**18,350 physical Python lines across 23 core files**; the earlier count describes the
pre-refinement baseline.

## Why a fork

Upstream NEURD is a full analysis framework built around its Docker image. This fork answers a
narrower question — *segment one neuron mesh, quickly, on a plain local env* — which makes tradeoffs
available that the general framework cannot take: exactly one soma with no glia or nuclei, no
downstream stages, and permission to replace expensive-but-inconsequential steps with in-process
equivalents. Measured on a 166 MB / 3.27 M-face H01 neuron, peak RAM went **4842 → 3226 MB (−33%)**
and several independent wall-time levers landed — details and numbers in [PERF.md](PERF.md).

It also runs without Docker on numpy 2 / trimesh 4, which upstream does not.

Upstream publication: https://www.nature.com/articles/s41586-025-08660-5

## Install

The installer supports **Python 3.10–3.12**; the current validation uses Python 3.12.
It also needs `meshlabserver` and `xvfb` for the
mesh operations that still shell out (`apt install meshlab xvfb`). On a bare OS you may also need
`apt install libgl1 libglib2.0-0 libgomp1` for open3d.

```bash
bash scripts/install_local.sh          # creates ./.venv
PYTHON=python3.12 bash scripts/install_local.sh
source .venv/bin/activate
pytest                                 # unit gate; about one minute in the tested environment
```

The script handles the one fiddly bit: `mesh_processing_tools` pins `open3d==0.11.2` in its metadata,
so it is installed with `--no-deps` before the rest of [requirements.txt](requirements.txt). It then
builds and verifies both required CGAL C++ extensions (see [cgal/README.md](cgal/README.md)):

| extension | what it unblocks | without it |
|---|---|---|
| `calcification_param_Module` | the MAP skeletonization path (thick branches) | neurons with thick limbs raise `NameError: calcification_param` |
| `cgal_Segmentation_Module` | real SDF mesh segmentation | the public spine-enabled API fails before processing; the lower-level compatibility fallback cannot provide equivalent spines |

Both builds were validated with CGAL 6 headers (`apt install libcgal-dev libeigen3-dev libgmp-dev libmpfr-dev`, or a conda
env). Headers are searched in the active/base interpreter prefixes and system include
directories, including Eigen's `eigen3` layout. Missing headers, failed builds or failed
native imports make installation fail. `NEURD_ALLOW_MISSING_CGAL=1` explicitly permits
an incomplete development installation.

## Run

```bash
# a directory of meshes -> one output directory per mesh, resumable via a manifest
python process_all_neurons.py path/to/meshes path/to/output \
    --dataset microns --spines --sdf-profile reference

# several meshes at a time, limiting concurrent input-file bytes
python process_all_neurons_parallel.py path/to/meshes path/to/output \
    --dataset h01 --spines --sdf-profile fast \
    --max-workers 8 --input-size-budget-mb 1024
```

Both runners share worker supervision and verified resume. Raw CLI defaults are
`microns`, spines enabled, refinement off and the 25-ray reference profile. `fast`
selects 12 rays; an explicit `--sdf-rays` overrides the profile. The input-size budget
is an admission heuristic; `SEG_RSS_LIMIT_MB` enforces observed per-worker RSS, including
external children, and requires `psutil`. `--mem-budget-mb` remains a compatibility alias.

`--no-save` computes without exports or manifests. Input decimation is disabled unless
`--decimation-ratio` is positive. Failures return a nonzero exit code; `--retry-failed`
requeues failures. Existing successful exports are reused only when input, runtime,
configuration and output hashes match. Use a new output directory for old unverified
exports or results from another runtime. Input/output directories must be separate
and non-nested.

Programmatically, use the same configuration:

```python
import neurd                                     # activates the compat + perf patches
from neurd.run_config import RunConfig
from neurd.segmentation_pipeline import segmentation_pipeline
from neurd.mesh_io import load_mesh

mesh = load_mesh("neuron.off")
config = RunConfig(dataset="h01", calculate_spines=True, sdf_profile="reference")
neuron = segmentation_pipeline(mesh, segment_id=123, max_somas=1, config=config)

for limb in neuron.limbs:
    for branch in limb:
        branch.mesh, branch.skeleton, branch.spines
```

`import neurd` must come first: it installs the numpy-2 / meshlab / CGAL shims that `datasci_tools`
and `mesh_tools` need, plus the perf patches.

The API defaults to `calculate_spines=False`; request spines explicitly as above.
Without an explicit dataset it retains the selected parameter dataset. Dataset parameters
and SDF rays are process globals, so concurrent runs use separate processes.

The raw runners, descriptor extractor and geometry gates use `neurd.mesh_io.load_mesh`.
It reads plain triangular ASCII OFF files in bounded chunks, preserving float64
coordinates, duplicate/unused vertices and ordered integer faces. Other mesh formats
and unsupported OFF variants use trimesh's existing loader. Pass the actual filename;
the API does not append `.off`. `NEURD_LEGACY_MESH_LOADER=1` selects the previous reader
and is included in processing and resume fingerprints. Internal MeshLab index repair
continues to use its existing reader.

For optional H01 geometric refinement, exported-branch resegmentation, skeleton policies
and spine-quality checks, see [SPINE_REFINEMENT.md](docs/SPINE_REFINEMENT.md).

The whole-neuron gate's default H01 input and the small CGAL oracle mesh are local
fixtures outside tracked source. Supply `--neuron /path/to/neuron.off` for a different
whole-neuron input, and configure the checksummed branch manifest described in the
refinement guide. A portable dataset-download recipe remains part of release preparation.

## Environment flags

Most of these are **inverted kill-switches** — the fast path is the default and the flag restores the
slow original. Nothing needs to be set for normal use.

| flag | default | effect when set |
|---|---|---|
| `NEURD_LEGACY_MESH_LOADER` | bounded triangular ASCII OFF reader at supported entry points | `1` restores the previous reader |
| `NEURD_LEGACY_VERTEX_COMPONENTS` | scipy | restore the networkx vertex graph (+~205 s) |
| `NEURD_LEGACY_MATCHING_VERTICES` | cKDTree | restore the O(N²) distance matrix |
| `NEURD_MESHLAB_DECIMATE` | open3d, in-process | restore the meshlab subprocess (+~184 s) |
| `NEURD_DECIMATE_SINGLE_STAGE` | cascaded halving | decimate in one step |
| `NEURD_ENABLE_COMPRESSED_PICKLE` | no-op'd | restore the debug pickle dumps (+~98 s) |
| `NEURD_SDF_RAYS` | low-level compatibility 12 | CGAL SDF ray count; the public API/CLI sets it from the explicit reference25/fast12 configuration |
| `NEURD_CGAL_TIMINGS` | off | `1` writes OFF-read, SDF, clustering and CSV-write stage timings to stderr |
| `NEURD_SKIP_SOMA_END_SPLIT` | off | skip `_split_soma_at_end`'s CGAL trim (−~225 s) |
| `NEURD_OWNED_SKELETAL_DISTANCE` | off | route the correspondence/width kernel through `neurd/skeletal_distance_ops.py` |
| `NEURD_REAL_POISSON` | off (no-op) | run a real Poisson reconstruction — see PERF.md |
| `NEURD_POISSON_DEPTH` / `_BACKEND` / `_ITERS` | 11 / meshlab / 8 | sub-flags of the above |
| `NEURD_SOMA_OUTER_DECIM` / `_INNER_DECIM` | from `parameters.py` | override the soma decimation ratios |

The batch runners also read `SEG_RSS_LIMIT_MB` and `SEG_TIMEOUT_SEC` (per-mesh kill switches), plus
`SEG_MAX_WORKERS` and `SEG_INPUT_SIZE_BUDGET_MB` (defaults for the parallel runner's flags;
`SEG_MEM_BUDGET_MB` remains a legacy alias for input-file bytes).

Every perf patch registers itself in `neurd.PATCH_STATUS` and warns loudly if it fails to install.
These patches rewrite functions inside `mesh_tools`/`datasci_tools`, so an upstream rename would
otherwise silently revert the perf work — the build would still succeed, just ~15 min slower.
`tests/unit/test_perf_patches.py` guards that.

## Tests

```bash
pytest                                                   # tests/unit only (pytest.ini)
pytest tests/integration/test_segmentation_pipeline.py   # end-to-end, ~11 min, needs xvfb + meshlabserver
python tests/tools/neuron_gate.py \
    --sdf-profile fast --baseline tests/fixtures/neuron_baseline_2530864375.json --tolerance
# exercise spine detection and record surface/volume checks and runtime provenance
python tests/tools/neuron_gate.py --dataset h01 --spines --sdf-profile fast \
    --seed 0 --out /tmp/h01_spine_metrics.json
# a checksummed small branch containing spines (manifest schema in the refinement guide)
NEURD_SPINE_BRANCH_MANIFEST=/path/to/fixture.json \
    pytest tests/integration/test_spine_branch_gate.py
```

The last one is the real backstop for any change to the decomposition: a full build of the h01 anchor
(~3.5 min) compared against a recorded baseline. **Use `--tolerance`** — the pipeline has one
documented nondeterminism (`np.random.choice` in the waterfill), so exact comparison fails even on
unchanged code. `tests/tools/` also holds the profiling harnesses (`benchmark_neuron.py`,
`wall_profile.py`, `batch_bench.py`).

For paired reader comparisons, set the same `--seed` and numerical thread limits in
both runs. The gate applies the optional NumPy seed after loading and records it in
the receipt and run signature. This controls the known waterfill randomness; native
or decomposition variance still has to pass the existing tolerances and spine checks.

## Documentation

| doc | what's in it |
|---|---|
| [NEURD_STRUCTURE.md](NEURD_STRUCTURE.md) | module map, LOC, import DAG, working conventions |
| [PIPELINE.md](PIPELINE.md) | the geometry of `mesh → Neuron`, the compat patches, known failure modes |
| [PERF.md](PERF.md) | measured time + RAM state, shipped levers, what's next, measurement methodology |
| [MESH_OPS.md](MESH_OPS.md) | the NEURD-owned mesh-operations layer (`submesh_ops`) and its seams |
| [docs/SPINE_REFINEMENT.md](docs/SPINE_REFINEMENT.md) | opt-in policies, branch batch, skeleton safeguards and spine gates |
| [docs/notes/](docs/notes/) | finished investigations and open refactor plans |

## Credits

Upstream NEURD is by Brendan Celii and the Reimer lab
([reimerlab/NEURD](https://github.com/reimerlab/NEURD)) — the science, the algorithms and the original
implementation are theirs. This fork narrows the pipeline, improves runtime and adds
experimental spine-geometry postprocessing.
