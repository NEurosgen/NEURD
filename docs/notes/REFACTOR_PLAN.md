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

## Remaining work, in order

1. Continue the production reachability audit of `neuron_searching`,
   `neuron_utils` and legacy object properties. String dispatch and user notebooks
   can call functions with no ordinary internal callers. Remove an API only after
   deciding its supported replacement or migration.
2. Separate raw export from process supervision if the current shared helpers
   are insufficient. Keep the CLI compatible; avoid another generic executor.
3. Clarify the MAP/MP stage result structures and resolve settings at clear stage
   boundaries. Do not change axis pairing, fallback choices or generated-map scopes.
4. Isolate package bootstrap/compatibility patches while preserving their exact
   initialization order and `PATCH_STATUS`. Existing native and mesh-tools patches
   are required behavior, not unused code.
5. Test candidate dependency removals in an isolated environment, including the
   imports of upstream providers. A search limited to `neurd` cannot prove removal.

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
