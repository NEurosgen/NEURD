# Structure and state of the fork

Navigation map for the slim fork. The code is the source of truth; this file collects what you need
at the start of a session. Companions: [PIPELINE.md](PIPELINE.md) (how the pipeline works + the
compat patches), [PERF.md](PERF.md) (measured time/RAM state and levers),
[MESH_OPS.md](MESH_OPS.md) (the owned mesh-ops layer), [docs/notes/](docs/notes/) (finished
investigations + open plans).

**Goal of the fork:** keep only the segmentation path `mesh → Neuron` (somas + limbs/branches with
mesh + skeleton + raw spines). Downstream processing subsystems — autoproofreading, cell typing, synapse assignment,
connectome/motif/proximity, GNN and cloud/dataset adapters — are removed. Some legacy
object properties and string-dispatched query helpers remain for compatibility; their
query signatures and existing features are retained. Statistics and decomposition
products support slim objects with explicit diagnostic/centroid controls;
axon/dendrite classification and synapse enrichment still require a migration decision.

**Size:** the current integration candidate has **29 Python files / 19,343 physical lines**
(2026-10-11). The stable geometry refactor contained 18 files / 15,389 lines, reduced from
about 55k; refinement, volume QC, configuration, H01 batch and mesh loading add supported
behavior. Raw export now lives in the package rather than the runner, which shrinks
from 982 to 500 lines. The package and runner together grow by 28 lines for imports,
compatibility aliases and documentation. Counts include comments and blank lines.
The subsequent MAP/MP cleanup names stage results and removes intermediate aliases:
`preprocess_limb` shrinks from 206 to 197 lines; its module shrinks by two lines.
Its validation passes 1,031 unit tests / one expected skip and a 33-case prior-code
oracle. Full raw H01 preserves all 218 exported files; native coverage is MP,
with MAP/mixed routes checked under controlled providers. The Minnie branch
retains nine spines and 27 geometry/mask files. Evidence:
`artifacts/map_mp_boundary_cleanup_20261011/v1/`.

## Environment and tests

```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate neurd   # Python 3.12, numpy 2
pytest                                                   # portable unit suite
python tests/tools/neuron_gate.py \
    --baseline tests/fixtures/neuron_baseline_2530864375.json --tolerance   # ~3.5 min
pytest tests/integration/test_segmentation_pipeline.py -s                   # ~11 min
```

- **Fast gate** (`tests/unit/`): portable geometry, query/object, configuration and export contracts,
  core import smoke, `submesh_ops` `tu`-equivalence, the perf-patch registry, and the
  numpy-2 / mesh_tools compat shims. `tests/unit/__init__.py` imports
  `neurd` first, which activates the shims in [neurd/__init__.py](neurd/__init__.py). It catches
  syntax, import and isolated contract/equivalence regressions. A complete native
  decomposition is checked by the separate full-neuron gate.
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
- **CGAL oracle** (`tests/integration/test_cgal_segmentation_oracle.py`) checks provider CSV
  contracts and reference SDF correlation. Release validation requires the compiled provider;
  the local OFF oracle fixture is not tracked. See [README.md](README.md) for fixture limits.
- **Spine-positive branch gate** (`tests/integration/test_spine_branch_gate.py`) uses a checksummed
  H01 or Minnie65 branch with compiled CGAL, actual parent-face masks and positive-volume checks.
  Configure `NEURD_SPINE_BRANCH_MANIFEST`; anatomical quality still needs independent review.

## Module map (29 files)

| module | LOC | purpose |
|---|---|---|
| [preprocess_neuron.py](neurd/preprocess_neuron.py) | 3276 | mesh → limbs/branches with named skeletonization/MAP/MP results and local mesh/axis origin observations. `preprocess_neuron`/`preprocess_limb` are decomposed into named phases (PIPELINE.md §2) |
| [spine_utils.py](neurd/spine_utils.py) | 3367 | ordinary spine detection with named primary-refinement/final-filter stages, aligned result selection and scoped shaft paths; enabled by `RunConfig(calculate_spines=True)` |
| [neuron_utils.py](neurd/neuron_utils.py) | 2665 | neuron queries/utilities, explicit-unit soma centers, soma-stitched whole skeleton and controlled decomposition-product storage; an intra-package **sink** |
| [neuron.py](neurd/neuron.py) | 1845 | the `Neuron`, `Limb`, `Branch`, `Soma` classes; actual soma counts, explicit decomposition controls and nonrecursive optional stage-product delegation |
| [spine_refinement.py](neurd/spine_refinement.py) | 1498 | experimental parent-face growth, trimming and head completion with diagnostics |
| [neuron_searching.py](neurd/neuron_searching.py) | 696 | branch query system with string/callable features, grouped IDs, ordered pairs and dataframe results. One production call from `spine_utils`; legacy string-dispatched features retained |
| [soma_extraction_utils.py](neurd/soma_extraction_utils.py) | 899 | soma identification |
| [__init__.py](neurd/__init__.py) | 503 | compat shims + the 8 perf patches + `PATCH_STATUS` — see PIPELINE.md §3 |
| [h01_batch.py](neurd/h01_batch.py) | 752 | isolated exported-branch resegmentation, pinned runtime, topology safeguards, verified resume and failed-width evidence |
| [batch_utils.py](neurd/batch_utils.py) | 135 | shared completion hashes/inventory, runtime/source fingerprints and exact geometry keys; stdlib-only module body, pinned standalone workers load their own sibling |
| [neuron_export.py](neurd/neuron_export.py) | 510 | direct raw-neuron geometry, graph, masks and local provenance export; no worker supervision or completion publication |
| [submesh_ops.py](neurd/submesh_ops.py) | 458 | **owned** partition/provenance layer (`SubMesh`); stable label grouping without per-label full scans — see MESH_OPS.md |
| [branch_utils.py](neurd/branch_utils.py) | 355 | branch geometry helpers |
| [neuron_statistics.py](neurd/neuron_statistics.py) | 344 | ordered statistics with lazy exclusions and explicit upstream-diagnostic control; skeleton features and explicitly scaled voxel centroids |
| [skeletal_distance_ops.py](neurd/skeletal_distance_ops.py) | 281 | owned correspondence/width kernel, behind `NEURD_OWNED_SKELETAL_DISTANCE` (off by default) |
| [mesh_io.py](neurd/mesh_io.py) | 218 | bounded triangular ASCII OFF loading; exact coordinates and ordered faces |
| [width_utils.py](neurd/width_utils.py) | 183 | branch surface-to-skeleton width measurements |
| [shaft_paths.py](neurd/shaft_paths.py) | 164 | optional ordered bidirectional shaft-path queries on an immutable graph snapshot; scoped NetworkX tie equivalence |
| [_mesh_ops.py](neurd/_mesh_ops.py) | 158 | L6 mesh primitives (decimate/poisson/fill_holes, in-process open3d) |
| [parameters.py](neurd/parameters.py) | 156 | the explicit microns/h01 parameter object (replaced global-config monkey-patching) |
| [run_config.py](neurd/run_config.py) | 147 | shared dataset/spines/refinement/ray configuration and runtime environment receipts |
| [cgal_labels.py](neurd/cgal_labels.py) | 139 | narrow labels-only CGAL spine seam, exact parent-face assembly and nested opt-in phase logging |
| [width_diagnostics.py](neurd/width_diagnostics.py) | 91 | bounded diagnostic support sampling for unresolved width; never establishes eligibility |
| [_correspondence_backend.py](neurd/_correspondence_backend.py) | 113 | **seam** for the fragile `cu`/`m_sk` calls + the pipeline's only nondeterminism |
| [_cgal_segmentation.py](neurd/_cgal_segmentation.py) | 66 | pure-Python CGAL-segmentation fallback |
| [spine_volume.py](neurd/spine_volume.py) | 162 | strict centered closure; immutable same-mesh quality handoff with mutation checks; separate OFF QC required |
| [face_provenance.py](neurd/face_provenance.py) | 99 | exact chunked validation and export journal for supplied ordered branch-to-limb maps; raw ownership stays unavailable |
| [segmentation_pipeline.py](neurd/segmentation_pipeline.py) | 60 | the slim orchestrator |
| version.py | 3 | — |

**Seam family** (`submesh_ops.py` / `_mesh_ops.py` / `_correspondence_backend.py` /
`skeletal_distance_ops.py`): the NEURD-owned layer that isolates or replaces the fragile `mesh_tools`
surface. Details in [MESH_OPS.md](MESH_OPS.md). Note that `neurd/__init__.py` is *also* a seam in
practice — most of its lines are cross-library surgery — and it is the one a reader will not think to
look in.

**Entry points:** `segmentation_pipeline.py` is the public raw-neuron API.
`neuron_export.save_segmentation` saves its result from Python or an installed package.
`process_all_neurons.py` supervises isolated spawn workers using that API;
`process_all_neurons_parallel.py` is a small adapter to the same executor.
The raw runner keeps aliases for its historical export functions. It owns staging,
limits, completion records and resume; the saver writes directly into its destination.
Shared hashing, completion and runtime fingerprints live in `batch_utils.py`;
the gate and raw runner import them directly, and legacy H01 imports remain compatible.
`scripts/resegment_h01.py` launches the separate exported-branch batch, and
`scripts/extract_spine_descriptors.py` extracts descriptors from pre-cut spine meshes.

**Local origin records:** correspondence entries carry optional version-1
`mesh_axis_origin` metadata through `Branch` and `Limb.limb_correspondence` into
`branch_origin.json`. Its scope is `local_map_mp_selection_and_axis_pairing`, with
`raw_input_map_available=false`. It records producer operations and local index domains,
not a recovered raw ownership map. `face_provenance.json` separately checks consistency
of supplied branch-to-limb indices; availability of that map does not supply missing origin.

The latest stitch decision records MP IDs, positions in the candidate, connection and
correspondence lists, original/used axis fingerprints, and any retained-partition exception.
At most 64 selected MP summaries and 16 flat cleaning-source summaries are retained;
omitted counts are explicit and histories do not nest recursively. End-node cleaning marks
exact axis identity unavailable because it recomputes the axis. Fingerprints describe the
ordered axis observed at the decision, before `Branch` normalizes it or later code changes
it; the ready export has a separate `exported_skeleton_file_sha256`. Missing/unsupported
metadata and opaque internal provider choices remain unavailable. These records do not
recover the 46 historical width failures.

**Post-CGAL tracing and shaft paths:** `NEURD_CGAL_TIMINGS=1` emits flushed begin/end
records for group adjacency/assembly, shaft expansion, skeletal length, refinement,
rechecks, individual final filters and H01 mask/volume/output publication. Parent/depth and
call-role fields identify nested work; durations are inclusive and must not be summed.
The lazy [shaft_paths.py](neurd/shaft_paths.py) module preserves ordered bidirectional
shortest-path ties in a serial immutable graph snapshot. The caller selects it for supported
simple graphs under NetworkX 3.6.1 with at least 1,024 queries; other cases retain NetworkX.
`NEURD_LEGACY_SHAFT_PATHS=1` also selects NetworkX. This is a path-query optimization,
separate from axis repair, volume policy or anatomical validation.

### Import-cycle cleanup

It used to be a god-hub (`neuron_utils`) plus 6 two-way cycles, band-aided with bottom-of-file
imports. The stable geometry refactor removed the audited eager import cycles:
`neuron_utils` was made an intra-package sink (its
references to `neuron` / `neuron_statistics` are local imports at the call sites), the self-imports
were removed, and dead or one-shot cross-imports were dropped or made lazy. Bottom-of-file imports
still exist in five modules (`spine_utils` has ~20 in its last 25 lines) as circular-import
workarounds — hoisting them is open work, see [docs/notes/REFACTOR_PLAN.md](docs/notes/REFACTOR_PLAN.md).

## Conventions

- **Don't change behaviour without a gate.** The fast gate checks imports and portable behavior
  contracts; core geometry changes also need `tests/tools/neuron_gate.py --tolerance` and the
  applicable native data gates.
- **Compare spine counts with controlled sampling and explicit tolerances.** Use the same seed,
  configuration and thread limits for paired whole-neuron gates. Count alone cannot establish
  equivalence; also check branch-local face masks, surface/volume invariants and pure functions.
- **Face-map scopes and origin are explicit.** Validation of supplied branch-to-limb IDs
  establishes exact row correspondence, not direct raw ownership or provider history.
  Preserve missing/generated maps as unavailable and graph labels by Branch object identity.
  The [final production gate](artifacts/post_cgal_followup_20261011/v1/native_raw_final_gate/receipt.json)
  passed all 28 supplied maps, 28 local origin journals and graph identity with exact prior H01 metrics;
  its observed RSS/HWM is not a controlled performance measurement.
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
