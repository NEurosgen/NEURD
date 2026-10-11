# Historical H01 failure diagnosis

The frozen `artifacts/h01_full_resegmentation_20261009/v1` run remains partial:
21,703 of 21,750 branches completed, with 37,925 exported spines and 47
unresolved branches. The October 11 diagnosis preserves that run, its runtime,
its input meshes and skeletons, and the original width threshold of **115 nm**.
It creates no accepted segmentation results.

The local evidence is in
[`report.json`](../artifacts/h01_failure_diagnostics_20261011/v1/report.json) and
[`branches.csv`](../artifacts/h01_failure_diagnostics_20261011/v1/branches.csv).
These ignored artifacts contain local source paths and hashes. The reusable
tool is [`diagnose_h01_failures.py`](../scripts/diagnose_h01_failures.py).

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
| 1 | The worker exceeded 2,700 seconds after reporting 22,167 CGAL segments | Subsequent processing remains unresolved; the log does not identify the exact slow operation |

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
axis, or completed result survives in this attempt. Only its header, skeleton,
logs and file hashes were inspected; a large mesh load or segmentation was
not repeated. Trimesh zero-volume warnings in its log do not identify the
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

The [audit receipt](../artifacts/h01_failure_diagnostics_20261011/v1/pilot_correspondence/receipt.json)
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
[inventory](../artifacts/h01_failure_diagnostics_20261011/v1/raw_pilot_inventory.json)
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
The [raw membership receipt](../artifacts/h01_correspondence_followup_20261011/v1/raw_membership/receipt.json)
and candidate arrays record zero-based original raw OFF face positions. These
are geometric candidates, not a recovered segmentation ownership map.

The [axis audit](../artifacts/h01_correspondence_followup_20261011/v1/axis_audit/receipt.json)
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

A fresh [five-case failure run](../artifacts/h01_correspondence_followup_20261011/v1/pilot_failure_validation.json)
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

Start the bounded follow-up with the three single-triangle keys above and
`neuron_2047644309_limb_001_branch_000` (a 14.24 nm original segment without
any axial slice). Add `neuron_1304599410_limb_002_branch_002` as a clearly
detached-axis case. These five small cases cover the failure mechanisms
before proposing a policy for all 46.

The timeout review found a concrete redundant operation: installed
`mesh_segmentation(return_meshes=False)` assembled every label mesh before
discarding them, then the spine caller assembled them again. The owned spine
seam now reads labels/SDF without that first assembly and performs the required
assembly once. Stable grouping also avoids scanning every face for every label.
Synthetic timings and exact native/Minnie checks are in the
[timeout review](../artifacts/h01_correspondence_followup_20261011/v1/timeout_review/review.json);
they do not establish that the historical timeout is solved. Its first `0/22167`
progress display may have belonged to the discarded assembly, so it cannot
identify which later filters were reached.

Retry the timeout separately after reviewing the new phase traces. Use one
worker, `NEURD_CGAL_TIMINGS=1`, the retained 2,700-second budget,
and the same source hashes and declared segmentation parameters. Loader
improvements alone do not establish that its later bottleneck is solved.
Every patched retry requires a new output directory and provenance signature;
do not resume into the frozen historical directory.

The full unresolved sets are recorded in
[`eligibility_unresolved_keys.json`](../artifacts/h01_failure_diagnostics_20261011/v1/eligibility_unresolved_keys.json)
and [`timeout_keys.json`](../artifacts/h01_failure_diagnostics_20261011/v1/timeout_keys.json).
The prioritized five-key diagnostic follow-up is
[`pilot_keys.json`](../artifacts/h01_failure_diagnostics_20261011/v1/pilot_keys.json).
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
