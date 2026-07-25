# Structure and state of the fork

Navigation map for the slim fork. The code is the source of truth; this file collects what you need
at the start of a session. Companions: [PIPELINE.md](PIPELINE.md) (how the pipeline works + the
compat patches), [PERF.md](PERF.md) (measured time/RAM state and levers),
[MESH_OPS.md](MESH_OPS.md) (the owned mesh-ops layer), [docs/notes/](docs/notes/) (finished
investigations + open plans).

**Goal of the fork:** keep only the segmentation path `mesh → Neuron` (somas + limbs/branches with
mesh + skeleton + raw spines). Everything downstream — autoproofreading, cell typing, synapses, axon,
connectome/motif/proximity, GNN, visualization, cloud/dataset adapters — is deleted.

**Size:** `neurd/` is **18 files, 15.4k LOC** (from ~55k at the start). The internal import graph is a
**DAG with 0 cycles**.

## Environment and tests

```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate neurd   # Python 3.12, numpy 2
pytest                                                   # fast gate (tests/unit, 180 tests, ~25 s)
python tests/tools/neuron_gate.py \
    --baseline tests/fixtures/neuron_baseline_2530864375.json --tolerance   # ~3.5 min
pytest tests/integration/test_segmentation_pipeline.py -s                   # ~11 min
```

- **Fast gate** (`tests/unit/`): import smoke for the core modules, `submesh_ops` `tu`-equivalence,
  the perf-patch registry, and the numpy-2 / mesh_tools compat shims. `tests/unit/__init__.py` imports
  `neurd` first, which activates the shims in [neurd/__init__.py](neurd/__init__.py). It catches
  syntax, import and equivalence regressions — **not** behaviour changes in the decomposition.
- **Full-decomposition gate** (`tests/tools/neuron_gate.py`) is the backstop for any core change: a
  real build of the single-soma h01 anchor
  (`Applications/Tutorials/Auto_Proof_Pipeline/neuron_2530864375.off`) compared to a recorded
  baseline. **Always pass `--tolerance`**: the waterfill's `np.random.choice` means the exact
  comparison fails even on unchanged code (observed diffs ≤0.64% against 2–5% tolerances).
- **Characterization test** (`tests/integration/test_segmentation_pipeline.py`) pins the
  `mesh → Neuron(somas + limbs/branches[.mesh + .skeleton])` contract and saves the result under
  `tests/integration/_seg_output/<stem>/` for visual inspection (`SEG_OUTPUT_DIR=` to relocate;
  gitignored, ~30 MB of .off). It shells out to `xvfb-run meshlabserver`, so it **skips silently**
  without both binaries.
- **CGAL oracle** (`tests/integration/test_cgal_segmentation_oracle.py`) pins the Python CGAL stub
  against a reference (SDF correlation ≥0.85). Skips without a provider.

## Module map (18 files)

| module | LOC | purpose |
|---|---|---|
| [preprocess_neuron.py](neurd/preprocess_neuron.py) | 3174 | mesh → limbs/branches. `preprocess_neuron`/`preprocess_limb` are decomposed into named phases (PIPELINE.md §2) |
| [spine_utils.py](neurd/spine_utils.py) | 3136 | spine detection (off the `segmentation_pipeline` path; ~51 `tu.*` calls, the largest holdout) |
| [neuron_utils.py](neurd/neuron_utils.py) | 2599 | neuron queries/utilities. Most-imported module; an intra-package **sink** |
| [neuron.py](neurd/neuron.py) | 1825 | the `Neuron`, `Limb`, `Branch`, `Soma` classes |
| [neuron_searching.py](neurd/neuron_searching.py) | 1081 | branch query system (strings → `ns.<fn>` via `@run_options`). Reached by exactly one call, from `spine_utils` |
| [soma_extraction_utils.py](neurd/soma_extraction_utils.py) | 903 | soma identification |
| [__init__.py](neurd/__init__.py) | 503 | compat shims + the 8 perf patches + `PATCH_STATUS` — see PIPELINE.md §3 |
| [submesh_ops.py](neurd/submesh_ops.py) | 446 | **owned** partition/provenance layer (`SubMesh`) — see MESH_OPS.md |
| [branch_utils.py](neurd/branch_utils.py) | 355 | branch geometry helpers |
| [neuron_statistics.py](neurd/neuron_statistics.py) | 347 | skeleton statistics/distances. Reachable only via `Neuron.neuron_stats` → `calculate_decomposition_products`, which the slim path does not call |
| [skeletal_distance_ops.py](neurd/skeletal_distance_ops.py) | 281 | owned correspondence/width kernel, behind `NEURD_OWNED_SKELETAL_DISTANCE` (off by default) |
| [width_utils.py](neurd/width_utils.py) | 183 | branch widths from SDF |
| [_mesh_ops.py](neurd/_mesh_ops.py) | 158 | L6 mesh primitives (decimate/poisson/fill_holes, in-process open3d) |
| [parameters.py](neurd/parameters.py) | 155 | the explicit microns/h01 parameter object (replaced global-config monkey-patching) |
| [_correspondence_backend.py](neurd/_correspondence_backend.py) | 113 | **seam** for the fragile `cu`/`m_sk` calls + the pipeline's only nondeterminism |
| [_cgal_segmentation.py](neurd/_cgal_segmentation.py) | 66 | pure-Python CGAL-segmentation fallback |
| [segmentation_pipeline.py](neurd/segmentation_pipeline.py) | 60 | the slim orchestrator |
| version.py | 3 | — |

**Seam family** (`submesh_ops.py` / `_mesh_ops.py` / `_correspondence_backend.py` /
`skeletal_distance_ops.py`): the NEURD-owned layer that isolates or replaces the fragile `mesh_tools`
surface. Details in [MESH_OPS.md](MESH_OPS.md). Note that `neurd/__init__.py` is *also* a seam in
practice — most of its lines are cross-library surgery — and it is the one a reader will not think to
look in.

**Entry points** (nothing in core imports them): `segmentation_pipeline.py`,
`process_all_neurons.py` (the real user entry — `neuron.Neuron(mesh=)` per worker process),
`process_all_neurons_parallel.py` (a scheduler reusing the sequential script's internals),
`scripts/extract_spine_descriptors.py` (standalone; descriptors from pre-cut spine meshes).

### The import graph is a DAG

It used to be a god-hub (`neuron_utils`) plus 6 two-way cycles, band-aided with bottom-of-file
imports. Now there are **0 cycles of any length**: `neuron_utils` was made an intra-package sink (its
references to `neuron` / `neuron_statistics` are local imports at the call sites), the self-imports
were removed, and dead or one-shot cross-imports were dropped or made lazy. Bottom-of-file imports
still exist in five modules (`spine_utils` has ~20 in its last 25 lines) as circular-import
workarounds — hoisting them is open work, see [docs/notes/REFACTOR_PLAN.md](docs/notes/REFACTOR_PLAN.md).

## Conventions

- **Don't change behaviour without a gate.** The fast gate only catches syntax/import; the backstop
  for core changes is `tests/tools/neuron_gate.py --tolerance` (~3.5 min).
- **Never gate on spine count** — it is nondeterministic run to run (22/16/20 observed on unchanged
  code). Gate spine work on pure-function equivalence instead.
- **Self-import (`from . import X as X`) is an antipattern** — replace with direct calls, or
  `sys.modules[__name__]` if the module object itself is needed.
- **Logic changes and cosmetics go in separate commits.** Commit messages say *why*.
- **Before deleting a function, grep for live callers** in `neurd/` + `tests/` + the entry scripts,
  and grep loosely (without a trailing word boundary) so dynamically built names like
  `<name>_limb_ns` are caught. Notebooks under `Applications/` are upstream tutorials — ignore them.
- **Don't edit the upstream packages** (`datasci_tools`, `mesh_tools`, `meshparty`): they are plain
  site-packages, so edits vanish on reinstall. Fix by wrapping or patching inside `neurd/` — see
  PIPELINE.md §3.
- **`neurd/__init__.py` is load-bearing.** Every patch block has its own reason, documented inline. If
  you touch it, `tests/unit/test_perf_patches.py` must stay green — it asserts each patch installed.

## Dependencies

- **Base** ([requirements.txt](requirements.txt)): `numpy>=2,<3`, scipy, `pandas>=2`, `networkx>=3`,
  `matplotlib>=3.7`, h5py, tqdm, `scikit-learn>=1.3`, `trimesh>=4`, `meshparty>=2`, `open3d>=0.19`,
  `pykdtree>=1.4`.
- **Author packages** (PyPI): `datasci-stdlib-tools` → `datasci_tools` (a god-package, ~22 submodules,
  numpy via `numpy_dep` — cannot be removed), `mesh_processing_tools` → `mesh_tools`, and others.
  25 distinct upstream modules are imported across 85 import statements; `datasci_tools` (16
  submodules) and `mesh_tools` (6) dominate.
- **Runtime binaries:** `meshlabserver` + `xvfb-run` for the mesh ops that still shell out.
- **Optional:** `pymeshlab` (`[poisson]` extra), `seaborn`/`ipyvolume` (`[viz]`), pytest (`[dev]`).

## History (condensed)

~55k LOC → 15.4k. Phases: Docker → local numpy2/trimesh4 with 8 compat bug-fixes; deletion of the
whole downstream cluster (synapse/axon/apical/visualization, autoproof/typing/graph/dataset adapters,
neuron_simplification — 44 → 16 files); zero-ref dead-code sweeps (−~10.7k) and elimination of the
import cycles; a readability refactor (decomposing the giant functions and classes); the mesh-ops and
correspondence seams (`submesh_ops`, `_correspondence_backend`); the wall-time and RAM-peak campaigns
(PERF.md); and a final cleanup pass (dead-module removal, the perf-patch registry, English-only
source). Details are in `git log`.
