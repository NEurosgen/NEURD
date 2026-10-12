# Historical H01 failure diagnosis

The frozen `artifacts/h01_full_resegmentation_20261009/v1` run remains partial:
21,703 of 21,750 branches completed, with 37,925 exported spines and 47
unresolved branches. The October 11 diagnosis preserves that run, its runtime,
its input meshes and skeletons, and the original width threshold of **115 nm**.
The diagnostic audits create no accepted segmentation results. A separate current-code
timeout retry, described below, produced one strictly validated completion.

The local evidence is in
`report.json` (local evidence: `artifacts/h01_failure_diagnostics_20261011/v1/report.json`) and
`branches.csv` (local evidence: `artifacts/h01_failure_diagnostics_20261011/v1/branches.csv`).
These ignored artifacts contain local source paths and hashes. The reusable
tool is [`diagnose_h01_failures.py`](../../scripts/diagnose_h01_failures.py).

## What the 47 failures mean

Current original-axis probes give a zero `median_mesh_center` width on all 46
historical width-failure keys, using the archived NEURD helper and recorded
current dependencies. Their
meshes contain 113–5,416 faces. Zero is an unavailable measurement in these
cases; it cannot justify classifying the branch as thinner than 115 nm.

| Branches | Evidence | Diagnosis |
| --- | --- | --- |
| 38 | No face-centroid slice meets the historical axial and 3,000 nm radial limits; sampled vertex distances, computed exactly to original axis segments, exceed 3,000 nm | Axis and branch surface do not provide usable correspondence |
| 5 | All axial slices are empty despite nearby surface samples | The exported axis does not span any usable face-centroid slice |
| 3 | One supported slice is present, but the actual selected component is exactly one triangle | Distance around that triangle's own area-weighted face centroid is zero by construction |
| 1 | The historical worker exceeded 2,700 seconds after reporting 22,167 CGAL segments | That archived attempt remains unresolved; a separate patched retry completed, without establishing the original slow operation |

The three single-triangle cases are
`neuron_4039221194_limb_001_branch_006`,
`neuron_5061320552_limb_000_branch_004`, and
`neuron_7052969349_limb_000_branch_033`. Their supported slices contain
multiple disconnected components. Observing the existing centroid function
confirms the chosen component has one face; the diagnosis does not change
selection or width calculation.

The timeout key is `neuron_7052969349_limb_000_branch_000`. Its OFF header
contains 779,393 vertices and 1,564,481 faces, the original skeleton is a single
segment about 99.97 nm long, and the old export contains 356 spines. CGAL
completed far enough to print its segment count. No segmentation CSV, accepted
axis, or completed result survives in this attempt. The initial diagnosis inspected only its header, skeleton,
logs and file hashes; it did not repeat the large mesh load or segmentation. Trimesh zero-volume warnings in its log do not identify the
timeout's root cause.

## Metadata and provenance limits

The directly associated branch, limb and neuron JSON exports contain no
validated original `median_mesh_center` width for the 46 affected branches.
The explicit derived H01 branch cache contains none of these 46 keys. It
contains the timeout key, but its `caliber_um` measures a different quantity
and cannot supply the original width eligibility decision.

The historical failure directories contain no saved skeleton or auxiliary
axis for these 46 attempts. Some logs show that the axis builder ran; the
actual candidate axis and its acceptance diagnostics were not persisted
before the eligibility exception. They cannot be audited from those logs.
No new axis was generated or silently substituted during this diagnosis.
The historical worker measured the axis returned by `repair_skeleton`, which
could retain the original or accept a repaired axis. With no saved failed
axis, these original-axis measurements establish concrete failure mechanisms
and meet the observed eligibility-error condition, but cannot prove the
identity of the axis used for the historical width calculation.

The receipt verifies all manifest-listed archived NEURD source hashes and
historically fingerprinted installed provider files. It also records the
current width-provider hashes, versions, behavior flags, numerical thread
settings and the archived package's `PATCH_STATUS`.
The historical manifest omitted `compartment_utils.py`, ray backend
fingerprints, and the correspondence feature-flag environment. The width
diagnosis casts no rays. It reproduces the observed failures, but the missing
historical fingerprints prevent a claim of complete historical environment
reconstruction.

The failed historical plan rows do not contain source mesh/skeleton content
hashes. Current input hashes establish stability during this inspection;
they do not prove byte identity to the inputs at the October 9–10 attempt.

Selected input meshes/skeletons, directly associated JSON metadata, failure
files, archived control files/runtime source and recorded provider files are
hashed before and after the inspection. The report states this verification
scope; it does not hash every completed branch in the dataset.

## Next patch and retries

The first bounded surface audit is complete for the five pilot cases. Canonical
float64 triangle occurrences show that all five old branch surfaces are exact
multiset subsets of their own exported limb. No target face is unmatched.

| Old branch | Target faces | Unambiguous faces | Ambiguous faces |
| --- | ---: | ---: | ---: |
| `4039221194/limb_001/branch_006` | 193 | 193 | 0 |
| `5061320552/limb_000/branch_004` | 876 | 860 | 16 |
| `7052969349/limb_000/branch_033` | 5,012 | 4,898 | 114 |
| `2047644309/limb_001/branch_000` | 243 | 243 | 0 |
| `1304599410/limb_002/branch_002` | 165 | 165 | 0 |

The audit receipt (local evidence: `artifacts/h01_failure_diagnostics_20261011/v1/pilot_correspondence/receipt.json`)
links the reproducible one-off code and all candidate occurrence arrays. Duplicate
candidates are retained without assigning owners. Peak observed worker RSS was
586.1 MiB and the longest worker took 21.3 s, below the 1.5 GiB / 180 s limits.
Processing used 32,768-face chunks and avoided a full triangle cache. All source
hashes were unchanged; raw-neuron geometry was not loaded.

This rules out absence of those branch surfaces from their old limb. It does not
validate their skeleton labels or recover ordered raw face ownership. The old
graph/index JSON files contain topology labels, not face maps. The user confirmed
that no original Neuron pickles remain; five full raw parent OFF files are available,
with approximately 6.9–12.1 million faces each. Their
inventory (local evidence: `artifacts/h01_failure_diagnostics_20261011/v1/raw_pilot_inventory.json`)
records headers and graph linkage; four raw content hashes are explicitly deferred.
Continue with skeleton-label correspondence and bounded raw membership checks
before attempting a full reconstruction.

The subsequent October 11 follow-up completed both bounded checks. Exact triangle
occurrences place all five pilot branch surfaces in their original raw OFF parent,
with **zero unmatched faces** and the same 16/114 ambiguous target faces as the
limb audit. It retained all raw vertices, streamed faces in 32,768-record chunks
and kept no complete raw face array. Maximum observed RSS was **611.1 MiB** and
the longest worker took **71.6 s**. All five raw inputs now have before/after
content hashes; this supplements the earlier inventory without changing it.
The raw membership receipt (local evidence: `artifacts/h01_correspondence_followup_20261011/v1/raw_membership/receipt.json`)
and candidate arrays record zero-based original raw OFF face positions. These
are geometric candidates, not a recovered segmentation ownership map.

The axis audit (local evidence: `artifacts/h01_correspondence_followup_20261011/v1/axis_audit/receipt.json`)
finds each pilot axis's exact segments in its limb skeleton and no whole segment
duplicated in a sibling branch. Graph neighbors share endpoint coordinates.
The strongest mismatch is `1304599410/limb_002/branch_002`: **all 113 vertices**
are at least **50.88 µm** from their own axis. Nonadjacent axes 106/21/107 have
much closer coverage, but proximity cannot establish a correct replacement label.
Two other cases have 10.45/14.24 nm axes at graph junctions; another has a
27.8 µm axis with poor reverse coverage of its small surface fragment. These
observations support investigating partition/export correspondence rather than
automatically swapping axes. Axis-to-nearest-vertex values in the report are
upper bounds to surface distance, with explicit sampling limits.

The bounded export-source audit (local evidence: `artifacts/h01_export_lineage_20261011/v1/findings.json`)
confirms a pairing defect in `_recorrespond_stitch`: its exception fallback
previously indexed the full MP axis list after selecting a subset. Nonprefix or
reordered selections could retain a branch mesh with another modified axis.
The one-line fix uses `curr_MP_sk_for_correspondence[mp_idx]`, preserving retained
meshes, face indices, widths and fallback aliases. Five whole-helper regressions
cover nonprefix/reordered selection, successful correspondence and prefix
fallback; the isolated witness (local evidence: `artifacts/h01_export_lineage_20261011/v1/candidate_validation.json`)
records two failures on the previous source and five passes on the correction.

The graph exporter also previously guessed node IDs from enumeration when counts
matched, which is unsafe for sparse labels. It now resolves the actual Branch
object's graph identity, records unavailable identity and rejects contradictions.
The 17 isolated graph cases (local evidence: `artifacts/h01_export_lineage_20261011/v1/api_audit/candidate/validation.json`)
cover that contract. The three inspected old limb maps have consecutive identity
labels; this sparse-ID defect was not observed in their saved metadata.

Neither defect proves the origin of the detached
`1304599410/limb_002/branch_002` surface or short junction axes at
`4039221194/limb_001/branch_006` and `2047644309/limb_001/branch_000`.
The original exporter runtime, selected MP IDs and intermediate maps are missing.
Retained exporters save mesh and skeleton from the same Branch object without an
export-stage coordinate transform; they do not prove how that object was built.
The source search was bounded to known local roots.

Raw export now validates supplied `Branch.mesh_face_idx` maps in ordered,
8,192-face chunks and records `parent_face_indices.npy` plus
`face_provenance.json`. It rejects malformed parent IDs, referenced vertex IDs
and nonfinite or mismatching triangle geometry. Exact in-memory rows are checked
before OFF rounding; exported parent/child meshes and index arrays are bound by
hashes. Missing maps remain explicitly unavailable. This proves consistency of
the declared branch-to-limb map, not upstream provider history or raw ownership.
All journals explicitly say `raw_input_map_available=false`; no geometric
candidate set is promoted to an ordered source map.

The current additive package records local origin independently of that consistency
check. Version-1 `mesh_axis_origin` metadata travels from correspondence entries
through `Branch` and `Limb.limb_correspondence` into a separate `branch_origin.json`.
Its scope is `local_map_mp_selection_and_axis_pairing`, and
`raw_input_map_available=false` remains explicit even when a supplied face map passes.
Producer operations and index domains describe known construction steps; opaque
internal provider fallback choices are unavailable rather than inferred from a
function named CGAL.

The latest stitch record includes selected MP IDs and their positions in the
candidate list, connection list and branch dictionary, original/used ordered axis
fingerprints, MAP context and the fallback exception type/message when the old
partition is retained. It keeps at most **64 selected MP summaries** and end-node
cleaning carries at most **16 flat source summaries**, with explicit total/omitted
counts and no recursive parent history. Recomputed cleaning axes have unavailable
exact identity. An `original_axis` fingerprint refers to the correspondence entry
before that current attempt, not an original raw-neuron axis.

Axis fingerprints bind the observation before `Branch` normalizes axis order or
later code mutates it. The exporter separately hashes the actual ready
`branch_skeleton.npy` as `exported_skeleton_file_sha256`; the journal does not assert
that its earlier ordered fingerprint matches that file. Missing, unsupported or
non-JSON-safe metadata is explicitly unavailable. The
focused origin receipt (local evidence: `artifacts/post_cgal_followup_20261011/v1/origin_review/receipt.json`)
records this additive implementation. It supplies decisions for future runs, not
the absent historical runtime or axes: **none of the 46 width failures is restored**.
The original 47-key archive and the separate successful 300-spine timeout retry
remain frozen evidence for their recorded source.

The final fifth-package raw gate (local evidence: `artifacts/post_cgal_followup_20261011/v1/native_raw_final_gate/receipt.json`)
verifies all 28 supplied maps and 28 local origin journals on the promoted code,
including actual exported skeleton hashes and graph identity. It retains exact
prior 18-spine / 8,145-face metrics and mask digest. This validates the forward
journal, without resolving historical attribution or supplying raw face ownership.

The earlier candidate probe (local evidence: `artifacts/h01_export_lineage_20261011/v1/raw_candidate_gate/receipt.json`)
validated all **28 supplied branch-to-limb maps** for neuron `2530864375` and
preserved prior exact metrics: **18 spines / 8,145 spine faces**, with the same
actual-mask digest. Independent review then added parent/child vertex-bounds
checks and four regressions; the isolated validator/export suite passed 29 cases.
That earlier probe remains evidence for its recorded source and phase profile.
The final production gate (local evidence: `artifacts/h01_export_lineage_20261011/v1/raw_final_gate/receipt.json`)
now passes on the promoted source, including the bounds checks: all 28 supplied
maps are available, graph identity is verified, every prior metric and the mask
digest match exactly, and geometry/volume error and overlap counters are zero.
It also captured the real non-soma stage; its later RSS/HWM includes capture
writes and is not a clean benchmark. The separate timeout retry has completed
as described below. The frozen historical run retains all 47 unresolved keys.

The worker now persists evidence before raising an eligibility error:
`branch_skeleton_original.npy`, the actual width axis `branch_skeleton.npy`, any
returned auxiliary `segmentation_axis.npy`, `skeleton_repair.json`, exact
postprocessed `width_distances.npy` and `width_evidence.json`. Its failure
`result.json` records `status="failed"` and `eligible=null`. These files survive
staging publication to `failures/` and are rejected by the completion validator.
The original width calculation and strict **width > 115 nm** rule are unchanged.

Support observations rerun the owned kernel on at most **16 uniformly selected
resized edges**, only after a failure. They report axial/radial face support and
separate raw diagnostic distances; they are not the original kernel's raw values
or a complete slice census. Diagnostic errors cannot replace the original error.
Basic axis/width records are written before sampling, so they remain available
if that extra inspection times out. Rejected repair candidates that were not
returned are described by repair QC rather than an invented axis file.

A fresh five-case failure run (local evidence: `artifacts/h01_correspondence_followup_20261011/v1/pilot_failure_validation.json`)
verified the journal on all pilot keys: five unresolved failures, no accepted
branches, source/output hashes checked and no completed dataset publication.
The detached case saved a newly built auxiliary axis while retaining its original
graph and width axis. This is current-policy validation, not recovery of the
missing axes from the historical attempts.

A fallback needs a separate, explicit eligibility policy and validation. It
can use a validated original width if one is recovered, or evaluate a
geometric auxiliary axis while retaining the original graph skeleton. Any
new rule for selecting a sufficiently supported component changes the
measurement policy and must be versioned and checked against the existing
criterion. Audit the original mesh-to-skeleton branch correspondence and
exported graph/index mapping for the detached and out-of-span cases before
replacing geometry. The diagnosis identifies the measurement failure, but
does not establish which earlier partition/export step created the mismatch.
Keep the strict `width > 115 nm` threshold. A derived cache caliber,
old zero spine count, or missing measurement cannot manufacture eligibility
or a completed empty result.

The five-case follow-up covers the three single-triangle keys,
`neuron_2047644309_limb_001_branch_000` (a 14.24 nm original segment without
an axial slice), and the detached `neuron_1304599410_limb_002_branch_002`.
All five remain unresolved; geometry and source findings alone cannot supply a
new eligibility policy for the 46 width failures.

The timeout review found a concrete redundant operation: installed
`mesh_segmentation(return_meshes=False)` assembled every label mesh before
discarding them, then the spine caller assembled them again. The owned spine
seam now reads labels/SDF without that first assembly and performs the required
assembly once. Stable grouping also avoids scanning every face for every label.
Synthetic timings and exact native/Minnie checks are in the
timeout review (local evidence: `artifacts/h01_correspondence_followup_20261011/v1/timeout_review/review.json`);
they do not establish that the historical timeout is solved. Its first `0/22167`
progress display may have belonged to the discarded assembly, so it cannot
identify which later filters were reached.

The separate timeout retry receipt (local evidence: `artifacts/h01_timeout_retry_20261011/v1/receipt.json`)
now records **1 complete / 0 failed** within the retained 2,700-second budget:
**300 spines / 286,839 faces**, with strict completion validation and zero
invalid, ambiguous or overlapping masks/surfaces or invalid volumes. Segmentation
took **2,273.321 s**; the observer recorded **2,336.936 s** overall. Volume QC used
**209 convex-hull estimates and 91 fan-closed estimates**; computational validity
does not establish anatomical accuracy of those closures.

The original graph and endpoint-filter axis were retained; head completion used
an auxiliary axis. Eligibility used the inherited `previously_spine_positive`
exception: **no positive width was recovered and the 115 nm width gate was not
applied**. This does not resolve any of the 46 width failures. The frozen October
9 archive remains unchanged with 47 failed keys; the new retry leaves 46 current
width problems. Its 300 spines versus 356 in the old export is not an anatomical
quality improvement or an exact historical-output comparison.

This was patched source recorded at `91d511b`, not a historical replay or a
controlled performance comparison; inputs, runtime, providers and selected old
controls were checked unchanged across the retry. Native SDF took **1,233.340 s**,
clustering **18.092 s**, explicit label-mesh assembly **9.295 s** and shaft filtering
**84.211 s**. Processing after shaft filtering completed but was not individually
tagged, so the remaining time cannot be assigned to one function. No speedup
claim follows from this successful attempt.

The current package adds flushed post-CGAL phase traces under
`NEURD_CGAL_TIMINGS=1`: group adjacency/assembly, shaft paths, skeletal length,
refinement and rechecks, final filters, and H01 mask/volume/output publication.
Parent/depth and call-role fields identify nesting; durations are inclusive, so
adding parent and child values would double-count work. The earlier retry above
did not execute these new traces and its untagged remainder remains unassigned.

The shaft-path caller now selects a lazy, serial ordered-bidirectional implementation
for at least 1,024 queries on supported simple graphs under NetworkX 3.6.1. Its
snapshot preserves adjacency insertion order and shortest-path ties; unsupported
or small batches retain NetworkX, as does `NEURD_LEGACY_SHAFT_PATHS=1`. This changes
the implementation of path queries while preserving filter order and ownership
decisions; it supplies no recovered width, historical replay or anatomical evidence.

H01 publication now reuses the final filter's immutable volume-quality report
only for the same unchanged in-memory mesh. Mesh identity and ordered geometry
digests protect the handoff; independent OFF-reload QC remains required. Three
fresh-process pairs on 61 accepted meshes measured 3.481 to 2.232 s for that local
operation, with exact scalars/reports/RNG. Rejected-candidate work and whole-run
performance are outside this comparison.

Next, use the traces and separately controlled real calls to evaluate the remaining
geometry and length work. Prepared KDTree support remains an artifact-only prototype:
H01's border filter is disabled and actual survivor filtering was already small. The
source proposal (local evidence: `artifacts/h01_export_lineage_20261011/v1/post_cgal_review/proposal.json`)
preserves filter order, threshold ties, RNG state and independent ownership state.
The non-soma prototype (local evidence: `artifacts/h01_export_lineage_20261011/v1/memory_review/proposal.json`)
has 43 artifact-only cases and a completed
real component A/B (local evidence: `artifacts/h01_export_lineage_20261011/v1/memory_review/real_stage_ab/receipt.json`):
three fresh-process pairs preserved all 424 components, ordered face IDs and soma
connectivity. Sampled component-end HWM fell 54.07 MiB, with 0.320 s additional
median extraction time, under common geometry-cache warmup. This bounded tradeoff
does not measure full-stage/pipeline peak and is not integrated. Require a
controlled full gate before changing the pipeline. All future
retries require separate outputs and signatures; do not resume into the frozen
historical directory.

The full unresolved sets are recorded in
`eligibility_unresolved_keys.json` (local evidence: `artifacts/h01_failure_diagnostics_20261011/v1/eligibility_unresolved_keys.json`)
and `timeout_keys.json` (local evidence: `artifacts/h01_failure_diagnostics_20261011/v1/timeout_keys.json`).
The prioritized five-key diagnostic follow-up is
`pilot_keys.json` (local evidence: `artifacts/h01_failure_diagnostics_20261011/v1/pilot_keys.json`).
None of these lists implies a branch is already eligible or complete.

For a fresh diagnosis, use a new, separate output directory:

```bash
python scripts/diagnose_h01_failures.py \
  --historical-run artifacts/h01_full_resegmentation_20261009/v1 \
  --output artifacts/h01_failure_diagnostics_<date>/v1
```

The tool runs sequentially, limits geometry/width probes to 25,000 faces,
reads only the OFF header for the historical timeout, and verifies input and
provider hashes. The optional supplemental metadata JSON records an
independent inspection; it never drives an eligibility decision.
