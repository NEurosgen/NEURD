# NEURD — slim single-soma segmentation fork

**Research preview: `v0.2.0-preview.1`.** See the
[release notes](docs/releases/v0.2.0-preview.1.md) for the supported workflow,
examples and validation limits, and [NOTICE.md](NOTICE.md) for attribution and
the unresolved inherited-code licensing status.

A fork of [reimerlab/NEURD](https://github.com/reimerlab/NEURD) narrowed to **one job**: turn a
single-neuron EM mesh into a structured decomposition in a local Python environment.

```
mesh (.off)  ->  Neuron
                  ├── somas:  S0                       (cell body: mesh + center)
                  └── limbs:  L0, L1, ...              (processes left after cutting the soma out)
                        ├── branches: 0, 1, 2, ...     (skeleton segments between branch points)
                        │     ├── mesh, mesh_face_idx  (branch submesh + its faces in the parent)
                        │     ├── skeleton             (centreline)
                        │     ├── width_array          (from SDF)
                        │     └── spines               (detected candidate submeshes)
                        └── concept_network            (branch graph, rooted at the soma contact)
```

The downstream subsystems — automated proofreading, multi-soma splitting, synapse assignment,
cell typing, axon labelling, SWC export, connectome/motif/proximity analysis, GNN, the cloud and
dataset adapters — **are removed**. Some legacy object properties and query helpers,
including string-dispatched axon queries, remain for historical callers. Their
[external input requirements and current limits](docs/ENRICHMENT_CONTRACT.md)
are explicit: raw segmentation supplies neither compartment classification nor
synapse assignment. The four upstream autoproof notebooks remain historical
examples of the wider framework.
The package contains 30 core Python modules for geometry processing, validated
run configuration, mesh input, raw export and compatibility installation. The
batch runners supervise separate worker processes, verify completed exports and
resume matching runs. Geometric spine refinement is optional.

The detector returns spine candidates. Mesh defects can be captured as candidates;
their presence or a positive volume does not establish anatomical quality.

## Why a fork

This fork focuses on a single-soma reconstruction with no glia or nuclei and no
downstream connectome-analysis stages. Its supported local environment uses
NumPy 2 and trimesh 4 without Docker.

The processing mechanisms include bounded OFF parsing, sparse connected-component
graphs, cKDTree vertex matching, in-process Open3D decimation and grouped CGAL face
labels. Configuration exposes the SDF ray count and optional refinement. Worker
limits and verified resume control batch execution. These choices describe the
implementation; the publication example records time and memory for one actual run.

Upstream publication: https://www.nature.com/articles/s41586-025-08660-5

## Install

The installer supports **Python 3.10–3.12**; clean Python-package installation was
validated on Linux x86_64 / Python 3.12.13 with the existing native toolchain.
See [installation scope and checks](docs/INSTALLATION.md).
It also needs `meshlabserver` and `xvfb` for the
mesh operations that still shell out (`apt install meshlab xvfb`). On a bare OS you may also need
`apt install libgl1 libglib2.0-0 libgomp1` for open3d.

```bash
PYTHON=python3.12 \
PIP_CONSTRAINT="$PWD/constraints/linux-py312-20261012.txt" \
bash scripts/install_local.sh          # creates the repository's .venv
source .venv/bin/activate
pytest tests/unit/
```

Run this command from the repository. The constraints select the reference
dependency profile for Linux / Python 3.12. An unconstrained fresh installation
changed branch geometry and spine counts in the whole-H01 comparison; successful
imports and unit tests alone do not establish equivalent processing. Other
Python versions need their own validated dependency profile.

The script can also be invoked by its absolute path from another directory.
Relative `VENV_DIR` paths resolve from the repository; reused environments are
checked for supported Python before installation.

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

Use the repository installer: `pip install neurd` alone does not supply the mesh
provider or build these extensions. `pip check` still reports the historical
Open3D pin; fresh validation found no other missing or conflicting requirements.

The runtime requirements omit `graph-nx-tools`, `neuron_morphology_tools` and
`code_structure_tools`. Their import roots are blocked during validation of the
supported segmentation paths. `machine-learning-tools` stays because the mesh
provider uses its PCA helpers; upstream imports also still require seaborn.

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
from pathlib import Path
import neurd                                     # activates the compat + perf patches
from neurd.run_config import RunConfig
from neurd.segmentation_pipeline import segmentation_pipeline
from neurd.mesh_io import load_mesh
from neurd.neuron_export import save_segmentation

mesh = load_mesh("neuron.off")
config = RunConfig(dataset="h01", calculate_spines=True, sdf_profile="reference")
neuron = segmentation_pipeline(mesh, segment_id=123, max_somas=1, config=config)

for limb in neuron.limbs:
    for branch in limb:
        branch.mesh, branch.skeleton, branch.spines

save_segmentation(neuron, Path("output/neuron_123"))
```

The [whole-H01 notebook](examples/h01_single_neuron.ipynb) and its
[HTML companion](examples/h01_single_neuron.html) demonstrate raw mesh processing,
export, measured time and memory, a skeleton overview and a separate manually
reviewed H01 spine with bundled geometry. Candidate counts report detector output;
the notebook explains the engineering optimizations and the scope of its figures.
See the [viewing and rerun instructions](examples/README.md).

`import neurd` must come first: it installs the numpy-2 / meshlab / CGAL shims that `datasci_tools`
and `mesh_tools` need, plus the perf patches.

The API defaults to `calculate_spines=False`; request spines explicitly as above.
Without an explicit dataset it retains the selected parameter dataset. Dataset parameters
and SDF rays are process globals, so concurrent runs use separate processes.

`neurd.neuron_export.save_segmentation` writes soma, limb, branch and spine meshes,
skeleton arrays, graph mappings, supplied face indices and provenance journals.
It can be used from an installed package or notebook without importing a batch
runner. The legacy `process_all_neurons.save_segmentation` import remains available.
The saver writes directly into the supplied directory; use a fresh directory for
each export. The raw batch runners provide worker isolation, limits, staging,
completion records and verified resume.

`neuron_searching.query_neuron` retains string and callable features. Its default
result groups branch IDs by limb; `return_limb_grouped_branches=False` returns
ordered `(limb, node)` rows, including shape `(0, 2)` for an empty result.
`neuron_utils.soma_centers` returns centers in nm without a scale. To convert to
voxels, pass `voxel_adjustment=True` and an explicit three-value
`voxel_adjustment_vector` in nm per voxel. Centroid statistics containing both nm
and voxel coordinates require this vector; a dataset-specific scale is not inferred.
For statistics on a slim neuron, explicitly omit upstream preprocessing diagnostics:

```python
summary = neuron.neuron_stats(include_upstream_diagnostics=False)
products = neuron.calculate_decomposition_products(
    include_upstream_diagnostics=False,
    include_centroids=False,
    store_in_obj=True,
)
```

`stats_to_ignore` skips getters before evaluating them. The legacy default
`include_upstream_diagnostics=True` requires the four original preprocessing
diagnostic fields and preserves enriched-object statistics. `n_somas` counts the
actual soma nodes. Decomposition keeps the `StageProducts` return type and merges
into the decomposition stage when storage is requested, creating a missing
`PipelineProducts` container after calculation succeeds. Skeleton features use S0
as their soma reference, including on multi-soma objects. Centroid output contains
nm and voxel coordinates: keep `include_centroids=True` only with an explicit
`voxel_adjustment_vector`; otherwise use False as above. Each selected getter still
requires its inputs, such as computed width entries; omit unavailable fields with
`stats_to_ignore`. Decomposition requires S0 even with centroids disabled, so
zero-soma decomposition is unsupported. Fields retain their mixed historical units
and scalings. Volume-density helpers check their own length divisor before division.

The raw runners, descriptor extractor and geometry gates use `neurd.mesh_io.load_mesh`.
It reads plain triangular ASCII OFF files in bounded chunks, preserving float64
coordinates, duplicate/unused vertices and ordered integer faces. Other mesh formats
and unsupported OFF variants use trimesh's existing loader. Pass the actual filename;
the API does not append `.off`. `NEURD_LEGACY_MESH_LOADER=1` selects the previous reader
and is included in processing and resume fingerprints. Internal MeshLab index repair
continues to use its existing reader.

Spine segmentation reads CGAL face labels without constructing meshes that would
be discarded, then groups those faces once while retaining parent indices and
face order. `NEURD_LEGACY_CGAL_ASSEMBLY=1` selects the alternative two-pass assembly.
Large shaft-path batches use an ordered bidirectional BFS when
Numba is available and NetworkX is the verified 3.6.1 version; smaller batches and
other versions retain NetworkX. `NEURD_LEGACY_SHAFT_PATHS=1` selects the previous
path search and is included in processing fingerprints. H01 batch export reuses
volume quality from the final filter only for the identical, unchanged in-memory
mesh. Scientific validation of saved OFF files requires a separate reload check;
the batch completion/resume check verifies hashes and inventory.

Failed H01 width eligibility preserves the actual axes and
measurement evidence; see [H01_FAILURE_DIAGNOSTICS.md](docs/H01_FAILURE_DIAGNOSTICS.md).
Raw export records validated supplied branch-to-limb face maps and graph labels
by Branch object identity. Missing maps remain unavailable; direct raw ownership
and the causes of the historical axis mismatches are still unproved. Local origin
journals record the observed MAP/MP route without inferring original raw face
ownership.

Mesh loading, statistics/decomposition, raw export and compatibility installation
have separate module boundaries. MAP/MP preprocessing uses named internal stage
results. Compatibility installation lives in `neurd._bootstrap`; set behavior
flags before importing and use a fresh process to change them. The eight default
patches expose their installation status through `neurd.PATCH_STATUS`.

The [validated installation](docs/INSTALLATION.md) records the unit, native-provider
and raw-mesh correctness checks. See the [refactoring plan](docs/notes/REFACTOR_PLAN.md)
for remaining work.

| Export | Verified scope |
|---|---|
| `parent_face_indices.npy` + `face_provenance.json` | Supplied ordered branch-to-limb IDs; exact in-memory triangles before OFF rounding, exported-file hashes; absent maps explicitly unavailable |
| `branch_index_map.json` | Export position to graph node by Branch object identity; missing/ambiguous identity recorded, contradictory IDs rejected |
| `branch_origin.json` | Observed local MAP/MP selection, axis fingerprints and fallback cause; actual exported skeleton hash recorded separately; missing origin explicitly unavailable |
| Original raw OFF face ownership | Unavailable; never inferred from geometric matches or branch numbers |

For optional H01 geometric refinement, exported-branch resegmentation, skeleton policies
and spine-quality checks, see [SPINE_REFINEMENT.md](docs/SPINE_REFINEMENT.md).

The whole-neuron gate's default H01 input and the small CGAL oracle mesh are local
fixtures outside tracked source. Supply `--neuron /path/to/neuron.off` for a different
whole-neuron input, and configure the checksummed branch manifest described in the
refinement guide. A portable dataset-download recipe remains part of release preparation.

## Environment flags

Most flags select an alternative compatibility path. The supported defaults are
active without setting them.

| flag | default | effect when set |
|---|---|---|
| `NEURD_LEGACY_MESH_LOADER` | bounded triangular ASCII OFF reader at supported entry points | `1` restores the previous reader |
| `NEURD_LEGACY_CGAL_ASSEMBLY` | labels-only CGAL spine seam and one face assembly | `1` restores the previous two-pass assembly |
| `NEURD_LEGACY_VERTEX_COMPONENTS` | scipy | use the NetworkX vertex graph |
| `NEURD_LEGACY_MATCHING_VERTICES` | cKDTree | restore the O(N²) distance matrix |
| `NEURD_MESHLAB_DECIMATE` | open3d, in-process | use the MeshLab subprocess |
| `NEURD_DECIMATE_SINGLE_STAGE` | cascaded halving | decimate in one step |
| `NEURD_ENABLE_COMPRESSED_PICKLE` | no-op'd | enable the debug pickle dumps |
| `NEURD_SDF_RAYS` | low-level compatibility 12 | CGAL SDF ray count; the public API/CLI sets it from the explicit reference25/fast12 configuration |
| `NEURD_CGAL_TIMINGS` | off | `1` writes native timings to stderr and nested Python phase records for assembly, shaft paths, lengths, refinement, final filters and H01 publication to stdout |
| `NEURD_LEGACY_SHAFT_PATHS` | off | `1` retains the previous NetworkX shaft-path search; fingerprinted for resume |
| `NEURD_SKIP_SOMA_END_SPLIT` | off | skip `_split_soma_at_end`'s CGAL trim |
| `NEURD_OWNED_SKELETAL_DISTANCE` | off | route the correspondence/width kernel through `neurd/skeletal_distance_ops.py` |
| `NEURD_REAL_POISSON` | off (no-op) | run a real Poisson reconstruction |
| `NEURD_POISSON_DEPTH` / `_BACKEND` / `_ITERS` | 11 / meshlab / 8 | sub-flags of the above |
| `NEURD_SOMA_OUTER_DECIM` / `_INNER_DECIM` | from `parameters.py` | override the soma decimation ratios |

The batch runners also read `SEG_RSS_LIMIT_MB` and `SEG_TIMEOUT_SEC` (per-mesh kill switches), plus
`SEG_MAX_WORKERS` and `SEG_INPUT_SIZE_BUDGET_MB` (defaults for the parallel runner's flags;
`SEG_MEM_BUDGET_MB` remains a legacy alias for input-file bytes).

Every performance patch registers itself in `neurd.PATCH_STATUS` and reports an
installation failure. These patches replace functions inside
`mesh_tools`/`datasci_tools`; dependency changes therefore need import and patch
checks. `tests/unit/test_perf_patches.py` verifies the supported installation.

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

`tests/tools/neuron_gate.py` is the backstop for decomposition changes: a full build of the h01 anchor
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
| [PERF.md](PERF.md) | historical measurements, investigations and measurement methodology |
| [MESH_OPS.md](MESH_OPS.md) | the NEURD-owned mesh-operations layer (`submesh_ops`) and its seams |
| [docs/SPINE_REFINEMENT.md](docs/SPINE_REFINEMENT.md) | opt-in policies, branch batch, skeleton safeguards and spine gates |
| [docs/notes/](docs/notes/) | finished investigations and open refactor plans |

## Credits

Upstream NEURD is by Brendan Celii and the Reimer lab
([reimerlab/NEURD](https://github.com/reimerlab/NEURD)) — the science, the algorithms and the original
implementation are theirs. This fork narrows the pipeline, streamlines local processing and adds
experimental spine-geometry postprocessing.
