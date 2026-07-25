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
dataset adapters — **is removed**. The package went from ~55k to **15.4k LOC across 18 files**.

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

Requires **Python 3.10–3.12** (3.13+ has no open3d wheel), plus `meshlabserver` and `xvfb` for the
mesh operations that still shell out (`apt install meshlab xvfb`). On a bare OS you may also need
`apt install libgl1 libglib2.0-0 libgomp1` for open3d.

```bash
bash scripts/install_local.sh          # creates ./.venv
PYTHON=python3.12 bash scripts/install_local.sh
source .venv/bin/activate
pytest                                 # fast gate, ~25 s
```

The script handles the one fiddly bit: `mesh_processing_tools` pins `open3d==0.11.2` in its metadata,
so it is installed with `--no-deps` before the rest of [requirements.txt](requirements.txt). It then
builds two optional CGAL C++ extensions (see [cgal/README.md](cgal/README.md)):

| extension | what it unblocks | without it |
|---|---|---|
| `calcification_param_Module` | the MAP skeletonization path (thick branches) | neurons with thick limbs raise `NameError: calcification_param` |
| `cgal_Segmentation_Module` | real SDF mesh segmentation | falls back to a KMeans stand-in that detects **0 spines**, and `NEURD_SDF_RAYS` becomes inert |

Both need CGAL 6 headers (`apt install libcgal-dev libeigen3-dev libgmp-dev libmpfr-dev`, or a conda
env). The build is best-effort — the install succeeds without them.

## Run

```bash
# a directory of meshes -> one output directory per mesh, resumable via a manifest
python process_all_neurons.py path/to/meshes path/to/output

# the same, several meshes at a time under a memory budget
python process_all_neurons_parallel.py path/to/meshes path/to/output \
    --max-workers 8 --mem-budget-mb 1024
```

The dataset parameter set is chosen by `DATA_TYPE` at the top of `process_all_neurons.py`
(`"microns"` or `"h01"`). Programmatically:

```python
import neurd                                     # activates the compat + perf patches
from neurd import parameters
from neurd.segmentation_pipeline import segmentation_pipeline
from mesh_tools import trimesh_utils as tu

parameters.params.use("h01")                     # or "microns"
mesh = tu.load_mesh_no_processing("neuron.off")
neuron = segmentation_pipeline(mesh, segment_id=123, max_somas=1)

for limb in neuron.limbs:
    for branch in limb:
        branch.mesh, branch.skeleton, branch.spines
```

`import neurd` must come first: it installs the numpy-2 / meshlab / CGAL shims that `datasci_tools`
and `mesh_tools` need, plus the perf patches.

## Environment flags

Most of these are **inverted kill-switches** — the fast path is the default and the flag restores the
slow original. Nothing needs to be set for normal use.

| flag | default | effect when set |
|---|---|---|
| `NEURD_LEGACY_VERTEX_COMPONENTS` | scipy | restore the networkx vertex graph (+~205 s) |
| `NEURD_LEGACY_MATCHING_VERTICES` | cKDTree | restore the O(N²) distance matrix |
| `NEURD_MESHLAB_DECIMATE` | open3d, in-process | restore the meshlab subprocess (+~184 s) |
| `NEURD_DECIMATE_SINGLE_STAGE` | cascaded halving | decimate in one step |
| `NEURD_ENABLE_COMPRESSED_PICKLE` | no-op'd | restore the debug pickle dumps (+~98 s) |
| `NEURD_SDF_RAYS` | 12 (25 = CGAL's own) | CGAL SDF ray count; **only the compiled ext reads it** |
| `NEURD_SKIP_SOMA_END_SPLIT` | off | skip `_split_soma_at_end`'s CGAL trim (−~225 s) |
| `NEURD_OWNED_SKELETAL_DISTANCE` | off | route the correspondence/width kernel through `neurd/skeletal_distance_ops.py` |
| `NEURD_REAL_POISSON` | off (no-op) | run a real Poisson reconstruction — see PERF.md |
| `NEURD_POISSON_DEPTH` / `_BACKEND` / `_ITERS` | 11 / meshlab / 8 | sub-flags of the above |
| `NEURD_SOMA_OUTER_DECIM` / `_INNER_DECIM` | from `parameters.py` | override the soma decimation ratios |

The batch runners also read `SEG_RSS_LIMIT_MB` and `SEG_TIMEOUT_SEC` (per-mesh kill switches), plus
`SEG_MAX_WORKERS` and `SEG_MEM_BUDGET_MB` (defaults for the parallel runner's flags).

Every perf patch registers itself in `neurd.PATCH_STATUS` and warns loudly if it fails to install.
These patches rewrite functions inside `mesh_tools`/`datasci_tools`, so an upstream rename would
otherwise silently revert the perf work — the build would still succeed, just ~15 min slower.
`tests/unit/test_perf_patches.py` guards that.

## Tests

```bash
pytest                                                   # tests/unit only (pytest.ini), 180 tests, ~25 s
pytest tests/integration/test_segmentation_pipeline.py   # end-to-end, ~11 min, needs xvfb + meshlabserver
python tests/tools/neuron_gate.py \
    --baseline tests/fixtures/neuron_baseline_2530864375.json --tolerance
```

The last one is the real backstop for any change to the decomposition: a full build of the h01 anchor
(~3.5 min) compared against a recorded baseline. **Use `--tolerance`** — the pipeline has one
documented nondeterminism (`np.random.choice` in the waterfill), so exact comparison fails even on
unchanged code. `tests/tools/` also holds the profiling harnesses (`benchmark_neuron.py`,
`wall_profile.py`, `batch_bench.py`).

## Documentation

| doc | what's in it |
|---|---|
| [NEURD_STRUCTURE.md](NEURD_STRUCTURE.md) | module map, LOC, import DAG, working conventions |
| [PIPELINE.md](PIPELINE.md) | the geometry of `mesh → Neuron`, the compat patches, known failure modes |
| [PERF.md](PERF.md) | measured time + RAM state, shipped levers, what's next, measurement methodology |
| [MESH_OPS.md](MESH_OPS.md) | the NEURD-owned mesh-operations layer (`submesh_ops`) and its seams |
| [docs/notes/](docs/notes/) | finished investigations and open refactor plans |

## Credits

Upstream NEURD is by Brendan Celii and the Reimer lab
([reimerlab/NEURD](https://github.com/reimerlab/NEURD)) — the science, the algorithms and the original
implementation are theirs. This fork only strips and optimizes.
