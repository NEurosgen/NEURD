# NEURD branch integration and public release plan

Prepare a reproducible local single-soma NEURD fork, retain the useful spine-refinement work,
and improve runtime and the public interface through separately validated changes. Start from
the existing experimental continuation, establish the supported processing paths and their
checks, then optimize and reduce code. No additional large rewrite is required for a first release.

This plan records the repository state inspected on **2026-10-11**, the first implementation
package and subsequent mesh-loading, H01 correspondence and segment-assembly work. Branch counts below describe the initial audit and use local Git
refs, including the locally recorded `origin/*` refs; they do not assert the current live state
of GitHub. The implementation uses a new local branch; nothing has been published.

## Implementation status, 2026-10-11

Candidate branch: **`release/integrate-spine-refinement`**, based directly on `c4ce054`.
All 15 experimental commits and their stable-base ancestry are preserved. Existing local volume
work was reconciled by content rather than reapplied, and the four local research scripts remain
unstaged. The original dirty patch, file copies and checksums are retained in ignored local
`artifacts/release_candidate_validation_20261011/workspace_backup/`.
The candidate core has **28 Python files / 18,800 physical lines**, compared with 21 / 17,859
at the experiment tip. The parallel raw runner is now a 59-line adapter to the common executor;
the new configuration/provenance contracts add maintained behavior. Further dead-code removal
still requires the production reachability work in Stage 7.

| Package | Implemented | Remaining |
|---|---|---|
| Candidate and topology | Contracted labelled-graph comparison; connected-tree-only export replacement behind `--experimental-skeleton-replacement`; auxiliary axis remains separate | Independent anatomical audit of axis-assisted repairs |
| Readability cleanup | Named spine stages, shared completion/provenance, historical whole-skeleton API, dead allocation/model removal and Markdown packaging; query/pair/empty-result and soma-unit contracts repaired; recursive missing-attribute lookup fixed; three unused runtime roots removed | Migrate legacy analysis/axon/synapse APIs; clean installation; separate raw export/bootstrap if justified |
| Public API and raw batch | Validated `RunConfig`; explicit dataset/spines/refinement/rays; shared isolated spawn workers; working no-save/decimation/retry; nonzero failure status; source/backend/environment/output fingerprints; RSS enforcement and termination cleanup | Installed console commands and clean native-provider installation |
| Face and descriptor provenance | Actual ordered parent-face masks for ordinary and refined spines, exported with filename alignment; strict volume QC; descriptor resume validates input/runtime/algorithm/output hashes and publishes completion last | Versioned migration of existing descriptor collections; fallback-rate quality review |
| Geometry gates | Real positive-spine branch gate, mask ownership/surface/volume checks, whole-neuron spine metrics and runtime receipts | Portable downloadable fixtures and required native CI job; a clean installation run |
| Native I/O and build | Buffered CSV writes, opt-in stage timers, isolated native A/B tool, system/conda header discovery and fail-fast native installation | No consistent native wall-time gain established |
| OFF loading | Bounded float64/int64 reader at explicit raw/branch/descriptor/gate call sites, legacy fallback and fingerprinted switch; exact large-mesh A/B | Re-profile complete construction on large raw neurons; internal MeshLab reads retain their index-repair reader |
| Spine label assembly and post-CGAL work | Narrow labels/SDF seam and stable grouping; nested post-CGAL traces; guarded exact shaft paths; accepted-mesh volume-quality handoff; separate timeout retry strictly complete with 300 spines / 286,839 faces | Measure remaining length/refinement/native costs on a bounded workload; no total-speed claim or anatomical validation |
| Historical H01 failures | Read-only diagnosis of 47 archived keys; five-case surface/axis audits; failed-width journal; one timeout completed in a separate patched run | 46 current width problems remain; recover branch identity/support before changing measurement policy; historical archive unchanged |
| Export lineage | Selected-MP fallback pairing corrected; graph labels by Branch object identity; ordered maps journaled with vertex bounds; bounded local MAP/MP decisions, axis fingerprints and fallback cause exported separately | Ordered raw ownership and historical attribution remain unavailable |

The subsequent readability package reduces the core from **19,275 to 18,903
physical lines** while adding a shared batch module and restoring supported skeleton
behavior. The branch-spine orchestrator falls from **688 to 454 lines**, keeping
its public parameters and legacy volume container types. Seven moved batch helper
bodies are AST-identical; 404 untouched core functions were checked mechanically.
Current-source validation and baseline copies are retained separately in
`artifacts/readability_cleanup_20261011/v1/`; earlier archives remain unchanged.
See [the current readability plan](REFACTOR_PLAN.md) for remaining legacy API,
query-dispatch and dependency work. The release notebook with a checksummed raw
neuron and recorded metrics is scheduled for the publication preparation stage.

The following query/object pass reduces the core further to **18,800 lines**,
retaining all query signatures and scientific features. Ordered pair/empty outputs,
soma indexing and explicit voxel conversion now have contracts; optional stage-product
delegation no longer recurses. The installed-provider audit removes three direct
dependency roots, with a **210 → 192** reachable distribution graph under the recorded
Linux CPython 3.12 metadata profile. Blocked imports exercise supported paths;
this is not a clean installation or a measured installed-size reduction.
Evidence lives separately in `artifacts/legacy_api_cleanup_20261011/v1/`.
Its frozen-source portable suite passes **881 tests / 1 expected skip** with all
three removed import roots blocked. The current Minnie65 native branch preserves
all nine spines / 2,854 faces, nine scalar volumes and 27 saved geometry/mask arrays;
independent reload verification checks every ordered mask. These checks use the
existing provider environment and establish no comparative performance gain.
The final full raw H01 gate also retains 1 soma, 5 limbs, 28 branches and
**18 spines / 8,145 faces**, with all 28 supplied branch maps and local origin
journals valid. On its actual object, old/new query feature rows, dtypes and selected
ordered branch IDs match exactly, with RNG unchanged. Soma centers match the original
mesh centers in nm. Removed import roots are neither requested nor loaded. The final
wheel contains all 28 frozen modules and the reduced 15 direct runtime requirements;
isolated unpacked-wheel import also passes with those roots blocked.

The preceding readability cleanup's frozen-source checks pass **802 unit tests / 1 expected skip**.
Conditional real-label replays preserve all **460 arrays** across 61 plain and
54 refined spines. Native checks preserve the full raw H01 result (1 soma,
5 limbs, 28 branches, 18 spines / 8,145 faces), the independently reloaded
52-spine H01 branch and the nine-spine Minnie65 branch. H01 resume leaves all
59 branch-file hashes and mtimes unchanged. The wheel contains the 28 checked
core modules, excludes tests and unused model assets, and imports from an isolated
unpacked directory without root runner files. It uses the existing native/dependency
environment; a clean dependency installation remains pending. These checks support
the cleanup's correctness, without a comparative whole-pipeline performance claim.

The fifth package instruments post-CGAL work and integrates two measured local
changes. On a captured real 1,724-node shaft graph, three fresh interleaved pairs
preserved all 22,052 ordered paths and measured **10.321 → 0.456 s** for queries
including CSR preparation (22.63×; JIT warmup recorded separately). On 61 accepted
meshes, one volume estimate plus protected report handoff measured **3.481 →
2.232 s**, compared with two estimates (1.56× for that operation). Neither result
establishes a whole-pipeline gain. Prepared filter KDTrees remain a prototype.

The refined-mask audit exposed sorted IDs paired with retained original meshes.
The fix retains their original row order. A frozen-baseline real comparison of
**54 meshes / 42,546 faces** preserves ordered geometry, volumes, RNG, refinement
decisions and occurrence-ID sets exactly; it corrects 40 previously row-invalid
masks. All 54 final ordered maps pass. The fifth-package unit suite passes **771 tests,
with 1 expected skip**, in **48.39 s** (10 warnings), with all 69 recorded
source/test hashes unchanged. Independent review additionally verified optional
Numba absence and the unchanged scientific estimator.

The final-source native raw H01 gate retains **18 spines / 8,145 faces**, exact
prior metrics and mask digest, all 28 supplied branch maps and 28 local origin
journals. A separate native H01 batch completes **52 spines / 42,050 faces**;
explicit OFF-reload QC verifies ordered masks, owners, disjoint surfaces and
recomputed positive strict volumes (34 fan closures, 18 hull approximations).
Its reference-25/full-refinement profile differs from the conditional replay and
the raw fast-12 gate. Completion/resume validation itself checks hashes/inventory;
independent scientific QC is a separate check. Eligibility remains
`previously_spine_positive`, without a recovered width. These are computational
gates, not anatomical validation or performance comparisons. Evidence is retained
in `artifacts/post_cgal_followup_20261011/v1/`.
The final Minnie65 positive-branch native gate also reproduces all prior metrics
and nine volumes exactly: **9 spines / 2,854 faces**, with nine checked ordered
maps. Its 8,372-query shaft batch actually uses the new CSR backend.

First-package validation on the local Python 3.12 environment:

| Check | Result and scope |
|---|---|
| Final unit suite | **409 passed, 1 skipped** in 59.59 s; skip is the unavailable stub-only path because real `ipyvolume` is installed |
| Compiled CGAL oracle | 7 passed with the separately built candidate extension |
| Minnie65 positive branch | `864691134886335738/limb_000/branch_003`: 29,495 parent faces, 9 spines / 2,854 faces; actual indices, geometry, overlap and volumes passed |
| H01 exported-branch batch | One positive and one zero-spine branch: 5 and 0 outputs, original graph retained; verified resume reused both results |
| Full raw H01 worker | `2530864375`, refinement off, fast/12 rays: 1 soma, 5 limbs, 28 branches, 21 spines / 7,138 faces; 182.32 s including worker/export; peak RSS 1,282,500 KiB |
| Raw export and resume audit | 137 hashed outputs valid; all masks exactly match OFF round-trip geometry; no cross-spine index/surface overlap; unchanged hashes and mtimes on resume, processed=0/skipped=1 |
| Real descriptor extraction/resume | One exported H01 spine: float64 vector with 35 columns, centroid and versioned volume QC; repeat skips it with descriptor/centroid/QC hashes and mtimes unchanged |
| Native old/new A/B | Five trials per build on each of two meshes; both CSVs byte-identical and source meshes unchanged; no consistent wall-time improvement |
| Native I/O syscall check | 7,240-face mesh: 14,483 to 39 write/writev calls, including diagnostics; identical CSV bytes |

The raw worker does not serialize its original volume arrays. An independent audit recomputed
21 positive strict volumes from the exported meshes: **5 fan closures and 16 convex-hull
fallbacks**. Hull results remain approximations; positive volume is a computational check,
not proof of anatomical correctness. Whole-neuron spine counts varied between 20 and 21
in local runs, as expected from the existing decomposition variance.

Local receipts, logs, source/runtime manifests and input fixture checksums are copied to
`artifacts/release_candidate_validation_20261011/`. They contain machine-local data paths
and are not a public reproduction bundle. Native test builds stay separate from the installed
environment, preserving the frozen runtime of the historical H01 resegmentation.
The recorded raw worker/resume used the source snapshot in its receipt. The subsequent stricter
export and completion checks were validated against those saved results and regression tests;
their source changes correctly require a new run signature for future processing.

The follow-up package measured plain triangular ASCII OFF loading on the large H01
anchor with three fresh, interleaved trials per reader. Median reading time fell from
**11.838 to 6.251 s**, and lifetime peak RSS, including common imports, from
**2,723,964 to 587,304 KiB**. Float64 vertices, int64 faces, metadata and visual state
matched exactly in every trial; input and measured source hashes were unchanged.
The 63 reader tests cover precision, ordering, unsupported variants, strict invalid-input
checks, bounded conversions and temporary-array release before fallback. A separate
one-trial small-anchor check also matched exactly. These results establish load-stage
improvement, not a whole-pipeline speed or peak-memory reduction.

The final common unit suite passed **492 tests with 1 expected skip** in 47.08 s.
A recorded optional NumPy seed is applied after loading, before decomposition, and
included in the gate fingerprint. Source and runtime receipts are retained under
`artifacts/off_loader_validation_20261011/v1/`.

| Follow-up integration check | Result |
|---|---|
| Paired full H01 gate, fast/12 rays, seed=0 | Legacy and owned readers produced **exactly identical complete metrics**, including 1 soma, 5 limbs, 28 branches, 18 spines / 8,145 faces and the same actual-mask digest; all surface/volume checks passed |
| Unseeded comparison | 24 versus earlier 20 spines exceeded the unchanged 10% count tolerance; it remains recorded as a failed comparison, motivating the controlled pair |
| Minnie65 positive branch | 9 spines; all metrics, including actual-mask digest, exactly equal to the previous validated branch run |
| H01 branch batch and resume | 5 and 0 outputs; effective child thread fingerprint works despite different parent limits; both branches reused, with 19 branch-file hashes and mtimes unchanged |
| Descriptor extraction and resume | Real 35-column float64 vector, centroid and positive volume QC; repeat skips processing and preserves all three output hashes/mtimes |

The controlled full-neuron pair does not establish a whole-pipeline speed improvement:
the recorded post-load gate times were about 191 and 200 s in these single runs. The
load-stage improvement remains the three-trial measurement above.

The independent [H01 diagnosis](../H01_FAILURE_DIAGNOSTICS.md) preserves the 47-key
historical failure record. Current exported-original-axis probes produce zero width in 46 cases:
38 have no supported slices and separated axis/surface samples, 5 have empty axial slices,
and 3 select a single triangle whose centroid width is zero. The remaining attempt timed out
after reporting 22,167 CGAL segments. Missing historical input hashes and failed-attempt
repaired axes prevent an exact historical replay claim. Zero width supplies no eligibility
decision; the strict **width > 115 nm** criterion and original graph remain intact.

The user confirmed that original H01 Neuron pickles are unavailable; complete raw OFF
meshes remain available. The five diagnostic pilot parents contain about **6.9–12.1 million
faces each**. Bounded branch/limb/raw inclusion audits are now complete; require evidence
of branch identity before a full parent reconstruction. Reconstructed branches require
validated surface matching to old exports;
their numeric branch labels are not a provenance map. Future exports should persist
validated, ordered branch-to-limb and limb/soma-to-pipeline-input face maps with parent
hashes, index scopes and unavailable/generated-face status. Compose maps to raw OFF
indices only when undecimated input identity and exact per-face geometry are verified.
Existing geometric set maps must not be presented as ordered source ownership.

The third package completed the bounded five-case checks. Every target branch
surface is an exact multiset subset of its full raw parent, with zero unmatched
faces and retained 16/114 ambiguous target faces. The raw scan kept vertices and
32,768-face chunks, not a full face array; maximum observed RSS was **611.1 MiB**
and maximum worker wall time **71.6 s**. All raw content hashes were checked before
and after. Each own axis is present in its limb skeleton, with no whole segment
duplicated by another branch. The detached case's **113 vertices** are all at
least **50.88 µm** from their own axis. Nearby nonadjacent axes are hypotheses,
not grounds for an automatic label/axis swap.

Failed eligibility now publishes actual original/used/returned auxiliary axes,
repair QC and exact postprocessed width values before raising. A separate support
sampler uses at most 16 resized edges and never determines eligibility. A fresh
five-pilot run preserved these records in `failures/`, with five failures and no
completed outputs. The width rule, graph-retention policy and historical run
remain unchanged; the recorded fresh auxiliary axis does not reconstruct a
missing historical repaired axis.

The timeout review confirmed one discarded mesh assembly before the required
spine assembly. That extra construction is removed at the narrow spine call site;
the installed tuple API is untouched. Stable grouping reduces repeated whole-face
scans: three fresh interleaved trials on a synthetic 1,564,481-face / 22,167-label
vector measured **9.569 to 0.155 s**, with exactly equal ordered index groups.
This is a grouping-only result. Two compiled native cases matched per-face
labels/SDF; a paired real Minnie65 branch matched all nine spines, masks and volumes.
The complete raw H01 neuron `2530864375` also matched every metric exactly across
the old and new assembly paths: **18 spines / 8,145 spine faces**, the same actual
face-mask digest and zero invalid, ambiguous or overlapping indices/geometry and
invalid volumes. Input, source, provider, configuration and seed 0 matched; the
only environment difference was the assembly flag (Numba 4; recorded BLAS/OpenMP/
NumExpr limits 1). This is a correctness gate, not a repeated performance trial.
The initial sandboxed MeshLab failure is retained; both successful runs used the
same elevated execution profile, with unchanged acceptance tolerances.
The third-package frozen unit suite passed **547 tests, with 1 expected skip** in 200.70 s
during concurrent local validation. Phase begin/end logs are available through
`NEURD_CGAL_TIMINGS=1`; the prior assembly path remains available through
`NEURD_LEGACY_CGAL_ASSEMBLY=1`. Local evidence is retained in
`artifacts/h01_correspondence_followup_20261011/v1/`.

The subsequent [source audit](../../artifacts/h01_export_lineage_20261011/v1/findings.json)
proved the selected-MP exception fallback could pair the wrong axis with a retained
mesh. The correction preserves the successful path and passes five whole-helper
cases. The graph exporter now uses Branch object identity rather than enumeration
for sparse IDs. Neither defect is established as the cause of the three historical
detached/junction cases; their original runtime and intermediate choices are absent.
The new 99-line `face_provenance` module validates supplied maps before publication,
including referenced vertex bounds. Four new bounds regressions bring its isolated
validator/export checks to 29 passing cases.
The earlier [candidate probe](../../artifacts/h01_export_lineage_20261011/v1/raw_candidate_gate/receipt.json)
validated all 28 supplied branch-to-limb maps and matched prior metrics (18 spines /
8,145 spine faces), before the final bounds correction. It is retained with its
measured source. The [final production gate](../../artifacts/h01_export_lineage_20261011/v1/raw_final_gate/receipt.json)
now passes on the actual promoted source, including bounds checks: all 28 maps
are available, graph identity is verified and every prior metric/mask digest
matches exactly, with zero geometry/volume error or overlap counters. The real
non-soma stage was captured for the separate A/B below; capture affects later RSS/HWM, so
that run supplies no clean memory or speed comparison. Maps validate declared
consistency before OFF rounding; they do not recover provider origin or raw ownership. The [final package unit suite](../../artifacts/h01_export_lineage_20261011/v1/validation/final_unit_result.json)
passed **601 tests, with 1 expected skip**, in **52.27 s** (10 warnings); all 60
recorded source/test hashes matched before and after.

The separate [timeout retry](../../artifacts/h01_timeout_retry_20261011/v1/receipt.json)
passed strict completion validation: **1 complete / 0 failed**, **300 spines /
286,839 faces** in 2,273.321 s segmentation (2,336.936 s observer total).
All mask/geometry/volume error and overlap counters are zero; volume closure used
209 convex-hull and 91 fan-closed estimates. The original graph and endpoint axis
were retained, with auxiliary head completion. Eligibility retained the existing
`previously_spine_positive` exception: no positive width was recovered and the
115 nm width gate was not applied. The patched `91d511b` run is not a historical
replay, anatomical validation or a speed comparison. The original 47-key archive
is unchanged; **46 current width failures remain**.

Next work should be bounded as follows:

1. Use the new nested post-CGAL traces to select the next bounded experiment.
   Ordered shaft paths and accepted-mesh volume handoff are now implemented; their
   measured gains apply only to the captured stages. Preserve filter order, masks,
   threshold ties, RNG state, volume policy and independent growth/ownership decisions.
   Surface-skeleton sampling must retain random-state behavior. The large retry's
   native SDF remains the largest measured stage; investigate it with per-face gates.
   Prepared final-filter KDTrees remain a prototype because the tested H01 border
   filter is disabled and actual survivor filtering was already small.
2. Extend the new local origin journal at the actual provider seam when available.
   Current MAP/MP decisions and axis fingerprints do not supply ordered raw ownership
   or recover missing historical evidence. Compose maps only with exact geometry, parent identity/hashes and
   explicit index domains; generated/missing maps stay unavailable, and decimated input
   remains separate from raw OFF. Preserve old labels and the width policy until identity
   and usable support are established; require anatomical review of axis-assisted repairs
   and the retry's volume fallbacks.
3. Evaluate integration of the [non-soma prototype](../../artifacts/h01_export_lineage_20261011/v1/memory_review/proposal.json)
   if whole-run memory measurements justify it. Beyond its 43 synthetic cases,
   [three real-input fresh-process pairs](../../artifacts/h01_export_lineage_20261011/v1/memory_review/real_stage_ab/receipt.json)
   now preserve all 424 ordered components, source masks, normals, routing/touching
   and RNG state. Component-end sampled HWM falls 661.99 to 607.92 MiB, while
   median extraction takes 0.238 versus 0.558 s. Common parent geometry warmup
   excludes matching/selection, connectivity and validation; original matching
   allocator history and whole-stage/pipeline peak are not reproduced. Keep this
   memory/time tradeoff candidate separate until a controlled full seeded gate
   demonstrates a useful whole-run memory reduction; it is not integrated.
4. Add portable fixtures and required native CI, test a clean supported installation,
   resolve inherited licensing, and prepare a bounded public demonstration before
   publication or upstream contact. Keep experimental H01 policies opt-in.

Stages below retain the detailed acceptance criteria and the initial audit evidence.

## Branch state and integration decision

Checkout at the initial audit: `refactor/skeletal-coords-pure`, `1437232`. All older local branches are
ancestors of this commit. Their fixes already entered the current history.

| Local branch | Tip | Commits only in branch versus current HEAD | Commits current HEAD is ahead | Disposition |
|---|---|---:|---:|---|
| `refactor/skeletal-coords-pure` | `1437232` | 0 | 0 | Existing stable base |
| `main` | `c9d3f7f` | 0 | 111 | Older release reference |
| `optimize_segmentation` | `ead3181` | 0 | 138 | Already integrated |
| `refactor_dependens` | `573fd57` | 0 | 162 | Already integrated |
| `refactor_readability` | `143aa99` | 0 | 119 | Already integrated |
| `fix/dense-limb-correspondence-and-cgal-fallback` | `a01f35d` | 0 | 82 | Already integrated |
| `refactor/simplify-preprocess-logic` | `42604cc` | 0 | 28 | Already integrated |
| `experiment/spine-geometric-refinement` | `c4ce054` | **15** | **0** | Candidate continuation to integrate |

The experimental branch is a direct descendant of that stable-base HEAD. Its diff contains
**37 files, 8,584 added lines and 75 removed lines**:

| Area | Changed files | Added lines | Removed lines |
|---|---:|---:|---:|
| `neurd/` | 5 | 2,540 | 70 |
| `scripts/` | 26 | 4,736 | 5 |
| `tests/` | 4 | 1,179 | 0 |
| `configs/` | 2 | 129 | 0 |

Tracked `neurd/*.py` grows from **18 files / 15,389 lines** at `1437232` to
**21 files / 17,859 lines** at `c4ce054`. These are physical line counts, including comments
and blank lines. Most experimental additions are evaluation tools and tests. Integration
adds supported behavior; subsequent cleanup should reduce duplication and the dependency
surface without treating a lower total line count as evidence of better software.

Recorded `origin/refactor/skeletal-coords-pure` equals the audited stable-base HEAD. Recorded `origin/main`
is 118 commits behind it; `origin/optimize_segmentation` and `origin/refactor_dependens`
point to `573fd57`. There is no second unmerged older fix branch in these refs.

Preserve ancestry of all 15 experimental commits when preparing an integration branch.
Cherry-picking only `c4ce054` would omit the refinement API that its final change depends on.
Do not restore historical disabled-spine defaults, the reverted floating-piece-count rejection,
or old KMeans/Poisson benchmark recommendations from ancestor branches.

## Existing work to reuse

The experimental branch already supplies:

- `neurd/spine_refinement.py`: parent-face-based growth, shaft trimming/splitting, surface-patch
  rejection, guarded axis retries, head completion and per-candidate diagnostics.
- Integration in `spine_utils.py`: repeated ordinary filters, rollback of rejected repairs,
  original-owner mappings and final parent-face masks.
- `neurd/h01_batch.py` and `scripts/resegment_h01.py`: exported-branch resegmentation with
  compiled-CGAL checks, isolated workers, runtime snapshots, input/code/dependency/output
  hashes, staged results, verified resume and explicit partial-result reporting.
- Named H01 configurations, synthetic tests, branch-comparison scripts, sample audits,
  human-review tooling and optional encoder diagnostics.

At the initial audit, the working-tree `neurd/spine_volume.py`, `tests/unit/test_spine_volume.py` and
`scripts/extract_spine_descriptors.py` are **byte-identical** to their experimental versions.
The stable-base `spine_utils.py` contained the volume correction but lacked the additional refinement
integration. Reconcile it by diff, rather than independently implementing the volume change again.
Preserve the local [volume investigation](SPINE_VOLUME_QC.md) and review the four untracked
debug/report/geometry/partition scripts separately. The multi-gigabyte `artifacts/` directory
is evidence storage, not an automatic part of the source integration.

There was no tracked spine-refinement guide in `c4ce054`. README, PERF and the old working
plans were not updated by the experimental commits. Their older scope statements and task
lists cannot describe the candidate release as they stand.

## Checks already completed

| Check | Code and scope | Result |
|---|---|---|
| Current unit suite | `1437232` with existing dirty volume edits | 192 passed, 1 skipped |
| CGAL oracle | Current checkout, compiled provider, 12-ray invocation plus ray-override tests | 7 passed |
| H01 neuron gate | Current checkout, 12 rays, tolerance comparison, **spines disabled** | Passed; 1 soma, 5 limbs, 28 branches; 202.30 s; peak RSS 1,233,176 KiB |
| Experimental unit suite | Isolated tracked-source snapshot of `c4ce054` under `/tmp` | 245 passed, 21 skipped |
| Skeleton topology counterexample | `c4ce054` function, synthetic graph, no neuron processing | Incorrectly accepts changed landmark connectivity |

The snapshot omits `Applications/` legacy assets and untracked local test data, so tests that
require those real-mesh fixtures were skipped. Passing unit tests and the non-spine neuron gate
do not establish anatomical correctness of refined spines. Record skipped tests and their fixture
requirements in the future release job.

## Stage 0 Prepare one candidate branch

**Deliverable:** a reviewed integration branch based on `c4ce054`, preserving the stable base
and the experimental ancestry, with an explicit list of included local changes.

1. Preserve the current dirty diff and selected untracked source/notes before manipulating Git.
   Inventory artifacts separately; avoid a blanket `git add .`.
2. Reconstruct a real checkout of the experiment. Its registered old `/tmp` worktree is prunable
   and its directory is gone; the branch commits remain available.
3. Reconcile the duplicate volume files, current notes and research scripts. Record which scripts
   become maintained commands and which remain archived development tools.
4. Keep `1437232` as the comparison reference. Prepare correctness, interface, performance and
   readability changes as separate reviewable commits after the experiment's history.

**Exit check:** the candidate imports from its own checkout, the unit suite passes, every
expected skip is explained, and the original workspace changes have been preserved.

## Stage 1 Close the experimental correctness gaps

**Deliverable:** safely includable opt-in refinement and an explicit skeleton-repair policy.

1. Fix `h01_batch.anchor_skeleton`: matching endpoint/junction coordinates and degrees is
   insufficient. Contract degree-2 paths in old and new graphs and compare the labelled
   landmark adjacency. Preserve components, cycles and the complete landmark set; reject
   new endpoints/junctions. Retain connector-distance and coverage checks.
2. Add the multi-junction counterexample below as a regression, plus an accepted example that
   changes only degree-2 sampling. Check this invariant for every exported replacement.
3. Make topological replacement explicitly experimental until this contract passes. Keep an
   auxiliary geometric axis separate from the exported branch graph and record which axis
   influenced head repair. The current H01 batch enables skeleton repair by default.
4. Retain geometric refinement off by default for both `microns` and `h01`. Keep current and
   smooth-head policies as named opt-in H01 configurations. Do not transfer H01 thresholds
   into Minnie65 without independent validation.
5. Preserve the strict volume helper and version its consequences separately from refinement.
   Replace descriptor resume's QC-file existence check with validation of schema, algorithm,
   dataset/configuration, input hash and completed output hashes.
6. Correct distribution names in H01 dependency metadata: `mesh-processing-tools` and
   `datasci-stdlib-tools`, rather than `mesh-tools` and `datasci-tools`.

**Exit check:** topology regressions pass; refinement disabled preserves the agreed volume
baseline and branch API; missing backend or mismatched resume metadata fails clearly.

## Stage 2 Unify the public API and the processing commands

**Deliverable:** one documented configuration for raw-neuron segmentation, with explicit
spine behavior and consistent use by the API, batch worker and neuron gate.

1. Extend `segmentation_pipeline` to expose and forward `calculate_spines`, dataset selection
   and the chosen refinement configuration. Make the documented default and the actual
   behavior agree. Preserve lower-level compatibility where callers need it.
2. Distinguish raw-neuron `mesh -> Neuron` processing from resegmentation of exported branches.
   Reuse H01 batch's validation/resume design without turning branch resegmentation into a
   replacement for full soma/limb preprocessing.
3. Use a validated run configuration shared by API and CLI. Apply it once in each isolated
   worker; do not add threaded mixed-dataset processing while helpers read global params.
4. Make SDF rays explicit and record them. The old runner uses 12; the H01 branch batch and
   its recorded full run use 25. Treat 25-ray reference and validated 12-ray fast processing
   as different named configurations, with identical values across entry points for each run.
5. Fix old-runner `--no-save`, implement or remove `--no-decimate`, expose dataset selection,
   propagate failure exit status, add explicit retry and create output directories before
   writing failure logs. Drain worker pipes while children run.
6. Share process supervision and completion/provenance utilities where behavior is genuinely
   common. Keep task-specific segmentation and exported-branch selection separate.
7. Replace the file-byte memory-budget label with an honest input-size limit, then add a
   measured RSS estimate or actual memory admission policy. Required RSS enforcement must
   not silently disappear when `psutil` is missing.

**Exit check:** the README API example and worker execute the same spine mode; a tiny real
input runs through both; no-save, controlled failure, retry and resume behave as documented.

## Stage 3 Establish the gates for future changes

**Deliverable:** portable representative fixtures and separate correctness/performance reports.

1. Supply a download recipe with dataset revision, units and checksums for the existing H01
   neuron and CGAL fixtures. Add small **spine-positive** exported branches from H01 and
   Minnie65. The small neuron anchor can have zero spines and cannot alone prove detection.
2. Extend `neuron_gate.py` and `neuron_metrics.py` to exercise the production spine path.
   Add parent-face membership, face-index digests, output mask/mesh alignment, overlap,
   skeleton coverage and connectivity alongside the existing decomposition metrics.
3. Reuse `compare_spine_refinement.py`, `validate_spine_refinement.py` and existing refinement
   integration tests. Compare off/current/smooth policies with one shared raw-CGAL and
   initial-filter prefix, then apply the real final filters independently.
4. For behavior-preserving changes, compare optimized pure functions with the previous
   implementation; compare fixed branch masks and order exactly where deterministic.
   Use documented tolerances and matched seeds for whole-neuron nondeterminism.
5. Add a small H01 batch smoke selection containing an ordinary branch, a zero-spine branch,
   an open/complex parent, a poorly covered skeleton and a controlled worker failure.
   Verify interruption, resume and changed input/code/dependency/output detection.
6. Record Git SHA and dirty state, source hashes, backend/binary, dependency versions, CPU,
   thread limits, seeds, active flags and configuration in benchmark reports. Run repeated
   interleaved A/B on the same meshes, with no profiler for the final wall/RSS comparison.

**Exit check:** a fresh supported installation can run the fixtures; the spine-positive gate
checks actual geometry; performance and semantic quality claims have distinct evidence.

## Stage 4 Resolve the incomplete H01 result without restarting the whole dataset

**Deliverable:** a versioned failure report and a traceable result for each unresolved branch.

The recorded run already attempted **21,750 branches from 88 exported neurons**. It completed
**21,703**, failed **47**, and produced **37,925 spines**. This was resegmentation of existing
branch exports, not a fresh full-neuron decomposition. Its manifest used 25 rays and enabled
the auxiliary axis for head refinement.

- **46 failures:** originally zero-spine branches where width eligibility could not be
  established. The calculation object uses `width=0`, which can also become the fallback
  when no usable distances exist. Add width/coverage diagnostics; recover a valid original
  width only where recorded metadata supports it. Otherwise retain an explicit
  `eligibility_unresolved` status. Do not adjust the 115 nm criterion or silently mark the
  branches non-spiny merely to finish the run.
- **1 timeout:** `neuron_7052969349_limb_000_branch_000`, 2,700 seconds, 356 archived spines.
  Preserve its historical frozen runtime. Retry this key in a separate versioned patch run
  using the current labels-only assembly, unchanged input hashes and declared parameters,
  one worker, the same 2,700-second limit and phase/resource diagnostics.

All 21,703 saved success records were inspected for the new anchoring concern:

| Skeleton status | Branches |
|---|---:|
| `original_covered` | 21,359 |
| `auxiliary_axis_only` | 292 |
| `repair_failed` | 51 |
| `repair_rejected` | 1 |
| Accepted topological replacement | **0** |

Consequently the anchoring bug does not invalidate an accepted graph replacement in this
existing run. Review the 292 auxiliary-axis cases as a separate geometry stratum.

Changing code changes the run signature. An eligibility fix therefore needs a new versioned
patch run. Reuse prior verified results only with explicit lineage and compatibility checks;
do not edit old manifests or snapshots to bypass their guards. Audit affected keys and adopt
verified unaffected outputs explicitly, instead of recomputing all branches by default.

**Exit check:** every selected key has a documented completed, adopted, unresolved or failed
status; the aggregate never claims full completion while unresolved cases remain.

## Stage 5 Validate refinement quality before promoting defaults

**Deliverable:** a frozen independent validation set and a compact quality report.

1. Freeze a neuron-held-out selection before further tuning. Keep historical controls and
   encoder-island diagnostics as development evidence; they are not anatomical ground truth.
2. Audit random unchanged and changed objects, plus removals, splits, head completion,
   auxiliary-axis branches and formerly spiny branches that now have no outputs. Existing
   count screening identifies **1,509** such branches with **2,637** archived spines.
3. Use the existing human-review workflow. Distinguish correct, incorrect and uncertain
   decisions; review small real spines removed by filters and capture of neighboring seeds.
   Use available EM evidence where it can settle ambiguous geometry.
4. Report mask changes, fallback/missing-volume rates and effect on morphology descriptors
   separately. A reduction in object count or a cleaner encoder cluster does not by itself
   establish improved segmentation.
5. Keep the latest smooth-head policy experimental unless this independent evaluation
   supports it. The final development replay explicitly makes no held-out claim.

**Exit check:** the report supports a bounded statement about the tested population. Default
promotion is a separate decision from making the opt-in feature available in a release.

## Stage 6 Optimize the measured boundaries

**Deliverable:** small isolated performance changes with unchanged geometric output.

| Order | Change | Implementation target | Acceptance |
|---:|---|---|---|
| 1 | Buffer CGAL CSV writes | Replace per-face `std::endl` with `\n` in segmentation C++; time OFF read, SDF, clustering and write separately | Same output CSVs/masks; measured I/O benefit |
| 2 | Reduce ASCII mesh-load transients | Streaming OFF loader or cache of float64 vertices and ordered integer faces, keyed by source hash | Same vertices/faces/provenance; loader and end-to-end RSS/time measured |
| 3 | Evaluate owned skeletal kernel | Existing `NEURD_OWNED_SKELETAL_DISTANCE` path | Equivalence plus useful wall/RSS change on supported anchors |
| 4 | Reduce remaining per-edge scans | Inverted face/edge index or spatial preselection in the owned kernel | Exact boundary/tie/component behavior; no excessive RAM growth |
| 5 | Remove full non-soma copy if worthwhile | Face-mask partition with provenance in `_segment_limbs_from_soma` | Exact component order and root face maps; demonstrated peak reduction |

Do not use ordinary float32 PLY export as a supposedly exact cache. Do not replace CGAL's
segmentation algorithm while optimizing its I/O boundary. Replacing OFF/CSV exchange with
an array-return native API is a later option if stage timers establish a meaningful remaining
cost. Keep existing skeletonization until its actual computational kernel is a measured limit.

**Exit check:** each adopted optimization has an interleaved comparison against the same
candidate code/configuration. Historical percentages are not added together into a new claim.

## Stage 7 Reduce code and dependencies with production coverage

**Deliverable:** a smaller maintained runtime surface and clearer module responsibilities.

1. Commit a conservative reachability analyzer and a runtime coverage probe. Roots must cover
   segmentation with spines, public object accessors, branch resegmentation and descriptor
   extraction. Account for string dispatch, decorators, aliases and default-argument references.
   Tests are evidence about behavior, not automatic production roots.
2. Re-run the [spine migration plan](SPINE_UTILS_PLAN.md) under the real spine driver. It predates
   the experimental branch; its first phase is still useful, but it must include refinement.
3. Separate extraction/filtering, geometry refinement, volume/descriptor calculations and
   diagnostics in `spine_utils`. Keep compatibility wrappers initially. Factor duplicated
   policies or traversal instead of rewriting every `mesh_tools` primitive.
4. Move compat installation and backend choice out of the large `__init__.py`, retaining
   ordered initialization and `PATCH_STATUS`. Prevent unrelated packages from being silently
   replaced by global stubs where they are genuinely installed.
5. Audit duplicated upstream/downstream skeleton helpers and factor their implementation
   behind existing properties. Do not redesign all Neuron/Limb/Branch storage for the release.
6. Consolidate overlapping research scripts around shared comparison/report utilities and
   argument-supplied paths. Keep Torch/encoder/EM-viewer dependencies optional.
7. Report removed reachable-dependency edges and verified dead functions alongside line counts.
   Delete in small batches, and rerun the corresponding geometry gates after each batch.

Preserve `_rebuild_limb_frames`: it incorporates newly stitched geometry as well as remapping
indices. Preserve dense-limb fallback, floating-piece guards, provenance and stable ordering.
The optional Branch-to-SubMesh storage migration and low-value stitch-remap cleanup can wait.

**Exit check:** supported routes still pass; deleted code has an explicit reachability reason;
the package no longer depends on an optional research tool merely to import its core.

## Stage 8 Verify installation and prepare the public release

**Deliverable:** a reproducible tagged source release with one working example and clear scope.

1. In parallel with the code stages, clarify the upstream license and the recovered C++ source
   terms. Preserve credits and citations; do not assign a new license to inherited code without
   establishing the applicable rights.
2. Fix system/conda CGAL and Eigen discovery. Validate one supported clean Linux/Python
   installation, build both extensions, and provide a backend/capability check. Freeze tested
   dependency constraints; document the mesh-processing-tools metadata workaround honestly.
3. Add a CGAL/integration CI job using portable fixtures. Keep fast unit checks per change and
   heavier geometry checks for candidate releases. Missing required release data is a failure,
   not a successful job consisting of skips.
4. Update README, PIPELINE, PERF, module counts and old working plans. Add a short refinement
   guide covering nm input, presets versus repair policy, raw-neuron versus exported-branch
   workflows, original graph versus auxiliary axis, resume/provenance and partial results.
5. Make a portable demonstration bundle. H01 exports use absolute source symlinks; a shareable
   subset needs copied context meshes or a documented source-download recipe. Publish selected
   figures/tables and checksums rather than the whole 4.8 GB artifact directory.
6. Archive obsolete tutorials for removed workflows. Prepare installed console commands and
   modern packaging after the unified entry points are stable.
7. Publish bounded benchmark/quality claims, then send upstream authors a concise description
   and propose small transferable changes: KDTree/components, buffered native I/O and the volume
   defect correction. Describe Poisson/FillHoles passthrough as a local processing policy.

**Exit check:** a new user can reproduce a representative result from the release instructions;
license/credits, input units, configuration, output limitations and evidence are explicit.

## First work packages

Proceed in this order: **candidate reconciliation -> topology/provenance fixes -> unified API
and spine gates -> low-risk optimizations -> measured code reduction -> release validation**.
Installation and license clarification can proceed in parallel; the targeted H01 failure work
and independent refinement-quality audit do not require waiting for every performance change.

The first four reviewable implementation packages should be:

1. Candidate integration with preserved ancestry, local volume note, explicit opt-in policy and
   a correct labelled-topology guard.
2. Public spine/configuration contract, truthfully behaving runners and validated completion metadata.
3. Portable positive-spine fixtures, production-path gates and benchmark provenance.
4. Buffered CGAL I/O and mesh-load optimization, each with its own measured comparison.

Large algorithm replacement, GPU work, universal dataset support and wholesale class redesign
remain outside this release plan unless later evidence establishes their value.

## Skeleton counterexample

Two junctions and four leaves have the same coordinates and degrees in both graphs, but leaves
3 and 4 change which junction they connect to. The audited `c4ce054` version of
`anchor_skeleton` accepts this change; the integration candidate rejects it.

```python
points = [
    [0, 0, 0], [1000, 0, 0],
    [-1000, -1000, 0], [-1000, 1000, 0],
    [2000, -1000, 0], [2000, 1000, 0],
]
old_edges = [[0, 1], [0, 2], [0, 3], [1, 4], [1, 5]]
new_edges = [[0, 1], [0, 2], [0, 4], [1, 3], [1, 5]]
```

The corrected guard must reject this case even though the coordinate and degree checks pass.

## Evidence references

- Repository history: `git rev-list --left-right --count HEAD...<branch>` and
  `git diff HEAD...experiment/spine-geometric-refinement` at the recorded tips above.
- Existing guidance: [MESH_OPS](../../MESH_OPS.md), [performance](../../PERF.md),
  [reachability method](DEAD_CODE_REACHABILITY.md), [readability plan](REFACTOR_PLAN.md),
  [class architecture](NEURON_CLASSES_ARCHITECTURE.md), [spine plan](SPINE_UTILS_PLAN.md),
  [volume correction](SPINE_VOLUME_QC.md). Older snapshots inside these notes need updating.
- Experimental code: `experiment/spine-geometric-refinement:neurd/h01_batch.py`,
  `neurd/spine_refinement.py`, `tests/unit/test_h01_batch.py` and both refinement test modules.
- Recorded H01 run: `artifacts/h01_full_resegmentation_20261009/v1/manifest.json`,
  `last_run_summary.json`, `last_run.jsonl`, failure logs and saved branch `result.json` records.
- Scope-qualified development validation: `artifacts/h01_heldout_validation_20261009/v1/`,
  `artifacts/h01_fresh_validation_20261009/v1/` and the experimental evaluation/review scripts.
