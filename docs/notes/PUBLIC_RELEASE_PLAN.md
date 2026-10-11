# NEURD branch integration and public release plan

Prepare a reproducible local single-soma NEURD fork, retain the useful spine-refinement work,
and improve runtime and the public interface through separately validated changes. Start from
the existing experimental continuation, establish the supported processing paths and their
checks, then optimize and reduce code. No additional large rewrite is required for a first release.

This plan records the repository state inspected on **2026-10-11** and the first implementation
package completed that day. Branch counts below describe the initial audit and use local Git
refs, including the locally recorded `origin/*` refs; they do not assert the current live state
of GitHub. The implementation uses a new local branch; nothing has been published.

## Implementation status, 2026-10-11

Candidate branch: **`release/integrate-spine-refinement`**, based directly on `c4ce054`.
All 15 experimental commits and their stable-base ancestry are preserved. Existing local volume
work was reconciled by content rather than reapplied, and the four local research scripts remain
unstaged. The original dirty patch, file copies and checksums are retained in ignored local
`artifacts/release_candidate_validation_20261011/workspace_backup/`.
The candidate core has **22 Python files / 18,118 physical lines**, compared with 21 / 17,859
at the experiment tip. The parallel raw runner is now a 59-line adapter to the common executor;
the new configuration/provenance contracts add maintained behavior. Further dead-code removal
still requires the production reachability work in Stage 7.

| Package | Implemented | Remaining |
|---|---|---|
| Candidate and topology | Contracted labelled-graph comparison; connected-tree-only export replacement behind `--experimental-skeleton-replacement`; auxiliary axis remains separate | Independent anatomical audit of axis-assisted repairs |
| Public API and raw batch | Validated `RunConfig`; explicit dataset/spines/refinement/rays; shared isolated spawn workers; working no-save/decimation/retry; nonzero failure status; source/backend/environment/output fingerprints; RSS enforcement and termination cleanup | Installed console commands and packaging cleanup |
| Face and descriptor provenance | Actual ordered parent-face masks for ordinary and refined spines, exported with filename alignment; strict volume QC; descriptor resume validates input/runtime/algorithm/output hashes and publishes completion last | Versioned migration of existing descriptor collections; fallback-rate quality review |
| Geometry gates | Real positive-spine branch gate, mask ownership/surface/volume checks, whole-neuron spine metrics and runtime receipts | Portable downloadable fixtures and required native CI job; a clean installation run |
| Native I/O and build | Buffered CSV writes, opt-in stage timers, isolated native A/B tool, system/conda header discovery and fail-fast native installation | No consistent wall-time gain established; mesh-load transient is the next performance target |

Validation on the local Python 3.12 environment:

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

Next work should be bounded as follows:

1. Measure the ASCII-OFF load on the large H01 anchor, implement a float64/order-preserving
   loader or checksum-keyed cache, and validate loader equality before another whole-neuron A/B.
2. Diagnose the 47 unresolved historical H01 branches in a separate versioned patch run, preserving
   their original source/runtime lineage. Do not change eligibility thresholds to complete counts.
3. Freeze independent anatomical review cases before further refinement tuning. Keep both H01
   policies opt-in while removals, head repairs and auxiliary-axis cases remain unaudited.
4. Add a portable fixture recipe and native CI, then test a clean supported installation.
5. Resolve inherited source licensing and prepare a bounded demonstration/benchmark report before
   public publication or contact with upstream authors.

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
  Retry this key with the unchanged frozen runtime and adequate resource/time diagnostics.

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
