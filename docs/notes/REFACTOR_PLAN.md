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

## Remaining work, in order

1. Define the input schema for retained axon/dendrite and synapse hooks before
   restoring aggregates or removing wrappers. Classification maps are unavailable
   on slim objects; empty branch labels do not establish anatomical compartments.
   The four shipped decomposition notebooks use historical enriched constructors
   and require wider notebook migration. String dispatch and external notebooks
   can still call retained features. Review the legacy spine count versus volume
   population policy and mixed-unit fields as separate scientific changes.
2. Separate raw export from process supervision if the current shared helpers
   are insufficient. Keep the CLI compatible; avoid another generic executor.
3. Clarify the MAP/MP stage result structures and resolve settings at clear stage
   boundaries. Do not change axis pairing, fallback choices or generated-map scopes.
4. Isolate package bootstrap/compatibility patches while preserving their exact
   initialization order and `PATCH_STATUS`. Existing native and mesh-tools patches
   are required behavior, not unused code.
5. Validate a clean dependency/native-provider installation before publishing.
   The blocked-import checks cover current supported paths; they do not exercise
   every dormant upstream helper or establish resolver behavior on another platform.

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
