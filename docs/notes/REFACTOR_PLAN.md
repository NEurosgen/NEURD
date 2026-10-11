# Readability refactor: current state and next steps

Keep the supported raw-neuron and exported-branch paths easy to follow while
preserving geometry, ordered masks, filter decisions and configuration behavior.
The current module map and physical line counts live in
[NEURD_STRUCTURE.md](../../NEURD_STRUCTURE.md); historical measurements live in
[PERF.md](../../PERF.md).

## Completed cleanup

The earlier fork removed downstream analysis subsystems, audited import cycles
and decomposed the neuron constructors and preprocessing orchestration.
The 2026-10-11 readability pass additionally:

- Removes unused shaft-face arrays, local values, duplicate imports/properties
  and obsolete example/comment blocks.
- Extracts primary spine refinement and ordered final filters from
  `calculate_spines_on_branch`. A shared selection helper keeps meshes, volumes
  and quality reports aligned without changing legacy list/ndarray returns.
- Moves shared completion/provenance helpers into `neurd.batch_utils`; the raw
  runner retains compatibility imports, and the gate no longer imports a private
  function from the CLI. Completion hashing remains separate from scientific QC.
- Restores `Neuron.get_skeleton()` and `Neuron.skeleton` using the historical
  limb-then-soma ordering and virtual soma attachments.
- Names the adaptive skeletonization helper for its actual bounded two-pass role,
  removes unused private arguments and avoids assembling unused skeletons.
- Deduplicates soma component ordering without changing NumPy's tie policy.
- Restricts distribution discovery to `neurd`, removes unused cell-typing model
  assets and reads the Markdown README directly during packaging.

Validation receipts and frozen baseline source are in the local ignored directory
`artifacts/readability_cleanup_20261011/v1/`. This is cleanup and API repair;
there is no new whole-process speed or memory claim.

The following query/object API pass reduces the core from **18,903 to 18,800
physical lines**, and `neuron_searching` from **810 to 696**. It retains all query
signatures and scientific feature bodies, fixes ordered-pair/empty-dataframe returns,
and computes NaN counts directly with the same pandas operation. Custom limb
callbacks retain the historical post-callback graph lookup.

Soma centers now work without the deleted global scale; all-soma indexing and empty
array shape are fixed. Voxel conversion and centroid statistics require an explicit
positive finite three-value nm-per-voxel vector. Skeleton statistics without centroid
conversion need no scale. Missing `Neuron` attributes raise normal `AttributeError`
without recursion; optional stage-product delegation remains compatible.

The import/provider and installed-metadata audit removes three direct runtime roots:
`graph-nx-tools`, `neuron_morphology_tools`, and `code_structure_tools`.
Their import roots are blocked during unit/native validation. Under the recorded
Linux CPython 3.12 metadata/marker/extras profile, the reachable distribution graph
falls from **210 to 192**. This is a potential installation graph reduction, not an
uninstall, a byte measurement or a clean installation test. Keep the ML provider,
mesh stack and seaborn transitive imports. Audit evidence and baseline copies live
separately in `artifacts/legacy_api_cleanup_20261011/v1/`.
The frozen-source portable suite passes **881 tests / 1 expected skip** with the
removed import roots blocked; the native Minnie65 branch also preserves all nine
spines and 27 geometry/mask files exactly. Native timings are correctness
observations, not comparative performance results.
The full raw H01 native gate retains exact prior metrics (18 spines / 8,145 faces),
28 supplied ordered branch maps and 28 origin journals. Old/new query selection and
feature rows agree on the actual 28-branch object, with RNG unchanged. The built
wheel matches frozen module bytes and current requirements and imports from an
isolated unpacked directory with the three removed roots blocked.

The supported statistics/decomposition pass now restores graph-derived soma counts,
lazy statistic exclusions and explicit `include_upstream_diagnostics` controls.
Enriched defaults preserve their ordered values and types. Decomposition accepts
an explicit voxel scale or `include_centroids=False`, and creates a missing stage
container only after successful calculation when storage is requested. Historical
callback argument order and strict default keyword sets remain compatible. The two
volume-density guards now check their own actual length divisor; all positive-length
formulas and counting/volume policies remain unchanged. This bounded API repair adds
35 core lines (**18,800 → 18,835**); it does not claim a size reduction or performance
gain. Evidence lives in `artifacts/statistics_api_cleanup_20261011/v1/`.
The portable suite passes **958 tests / 1 expected skip** with the removed import
roots blocked. A 66-case historical oracle preserves enriched summaries; 77 new
portable cases cover exclusions, diagnostic controls, decomposition/storage and
the density edge cases. The native Minnie branch retains all nine spines and
27 geometry/mask files exactly against the preceding package.
The full raw H01 control retains exact prior metrics (18 spines / 8,145 faces)
and all 28 branch maps/origin journals. Actual-object statistics omit unavailable
diagnostics, and returned/stored decomposition products and skeleton bytes agree
without inferring voxel calibration. Geometry metrics remain exact after the audit.

The raw export pass separates geometry writing from supervision.
`neurd.neuron_export.save_segmentation` is available without the repository runner;
the runner retains compatibility aliases and shrinks from **982 to 500 lines**.
All 14 moved export functions and 16 retained runner functions preserve their
baseline AST. Staging, completion publication, limits and verified resume stay
with the worker/scheduler. The core now has **29 files / 19,345 lines**, because
the 510-line export module moved into the package; package plus runner grows by
28 lines for imports, aliases and documentation. This is a separation of duties,
without a size or computational-performance claim. Evidence is retained in
`artifacts/raw_export_cleanup_20261011/v1/`.
The frozen portable suite passes **998 tests / 1 expected skip** on 106 source/test
files and three configuration files with removed import roots blocked. Full raw
H01 processing preserves prior metrics (18 spines / 8,145 faces); all 218 exported
files are byte-identical between the frozen prior saver, the library and CLI alias.
All 28 supplied branch maps, graph identities, origin journals and spine-mask
arrays are checked. The native Minnie control also retains all nine spines,
nine volumes and 27 geometry/mask files exactly, with independent reload checks.

The MAP/MP boundary pass introduces two private named tuple results for
skeletonization and sublimb decomposition. Callers use named payload fields;
seven redundant grouping aliases are removed. Payload arrays, mesh references
and ordered additions are retained. Public return types and function signatures
remain unchanged; private results support tuple indexing/unpacking, while their
exact type is now a named tuple. MAP/cleaning scalars are still captured before
skeletonization, and adaptive invalidation still rereads its separate globals
after the first pass. The orchestrator shrinks **206 → 197 lines**, its module
**3,278 → 3,276**, and the core **19,345 → 19,343**. Grouping, soma extension,
axis-origin and stitch algorithms retain their AST; this is a readability change.
Evidence is separate in `artifacts/map_mp_boundary_cleanup_20261011/v1/`.
The 33 new portable boundary cases cover grouping thresholds, adaptive passes,
configuration sampling, soma additions and public return flags. A 33-case oracle
against frozen `3993b5bee` preserves ordered payloads, provider arguments,
aliases, stdout and RNG state under controlled geometry providers.
The frozen-source suite passes **1,031 tests / one expected skip** on 107
source/test files and three configuration files with removed import roots blocked.
Full raw H01 processing preserves all prior metrics and **218 exported files
byte for byte**, including 239 ordered NumPy arrays. Its 22 preprocessing calls
all use MP; four take a second skeletonization pass. MAP/mixed native processing
is not exercised by this anchor. The separate Minnie branch retains nine spines,
nine volumes and all 27 geometry/mask files, with independent reload checks.

## Remaining work, in order

1. Isolate package bootstrap/compatibility patches while preserving their exact
   initialization order and `PATCH_STATUS`. Existing native and mesh-tools patches
   are required behavior, not unused code.
2. Define the input schema for retained axon/dendrite and synapse hooks before
   restoring aggregates or removing wrappers. Classification maps are unavailable
   on slim objects; empty branch labels do not establish anatomical compartments.
   The four shipped decomposition notebooks use historical enriched constructors
   and require wider notebook migration. String dispatch and external notebooks
   can still call retained features. Review the legacy spine count versus volume
   population policy and mixed-unit fields as separate scientific changes. Keep
   these optional enrichments outside the raw-path cleanup until their inputs and
   migration are defined.
3. Validate a clean dependency/native-provider installation before publishing.
   The blocked-import checks cover current supported paths; they do not exercise
   every dormant upstream helper or establish resolver behavior on another platform.
   Run a controlled complete raw-neuron benchmark and build an executed release
   notebook with checksummed input, configuration, stage timings and peak memory.

## Working method

Separate mechanical removal/extraction from algorithm changes. Preserve public
parameters, even unused compatibility flags, unless an explicit migration is part
of the change. Use a few named stages with explicit inputs; avoid registries or
automatic property generation that hide definitions from readers.

Before deleting code, search core, scripts and tests, including string names used
by dynamic query dispatch. Check import-time references in decorators and class
bodies before moving an import. Preserve comments explaining geometry ownership,
threshold boundaries, RNG order and fallback decisions; shorten historical notes
and examples that describe behavior the fork no longer provides.

For production changes, run the relevant portable contracts and the full unit
suite. A spine-enabled raw-neuron gate and native positive-branch gate establish
integration behavior. Conditional fixed-label comparisons can additionally check
exact geometry, scalar values, masks, refinement decisions and RNG without
pretending to replay native sampling. Record the actual input, configuration,
source/provider hashes and validation scope.
