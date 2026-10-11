# Performance — measured state, shipped levers, what's next

Goal: cut wall time and peak RAM on `mesh → Neuron`, **by measuring**, never by intuition. Pipeline
geometry is in [PIPELINE.md](PIPELINE.md), the module map in [NEURD_STRUCTURE.md](NEURD_STRUCTURE.md).

**Target case:** single-neuron meshes with **exactly one soma** (microns and h01). Every optimization
below assumes that. Validation meshes:

| anchor | size | role |
|---|---|---|
| small h01 `neuron_2530864375.off` | 507k faces | the fast gate anchor, ~3.5 min end to end |
| big h01 `1830470325.off` | 166 MB, 3.27 M faces | historical pipeline wall/RAM anchor |

⚠️ The committed fixture `864691…` contains **two** neurons and is not representative.

**Legacy measurement tools:** [tests/tools/benchmark_neuron.py](tests/tools/benchmark_neuron.py) (calls
the lower-level Neuron constructor, gives per-stage wall + sampled RSS, writes its report even on a crash),
[tests/tools/wall_profile.py](tests/tools/wall_profile.py) (exclusive wall attribution over a curated
set of pipeline ops), [tests/tools/batch_bench.py](tests/tools/batch_bench.py). They do not yet use
the public `RunConfig`; explicitly match their dataset, spine switch, `NEURD_SDF_RAYS` and
thread settings when comparing historical measurements. The production fidelity gate
`tests/tools/neuron_gate.py --tolerance` uses the shared configuration and records its fingerprint.

---

## Current defaults

| node | default | switch |
|---|---|---|
| Input mesh loading at supported entry points | bounded triangular ASCII OFF reader; other formats/variants delegate | `NEURD_LEGACY_MESH_LOADER=1` |
| **Spines** | API **off**; raw batch CLI **on**, using compiled `cgal_Segmentation_Module` | `RunConfig(calculate_spines=True)`; CLI `--no-spines` |
| CGAL SDF rays | Public API/CLI **25**, reference profile; low-level compatibility **12** | `sdf_profile="fast"` / `--sdf-profile fast` selects 12; explicit `sdf_rays` / `--sdf-rays` |
| Geometric spine refinement | off; named H01 policies are experimental | `refinement="h01_current"` or `"h01_smooth_heads"`; CLI `--refinement` |
| CGAL mean-curvature-flow skeleton (MAP, thick branches) | real C++ ext (`calcification_param_Module`) | falls back to the meshparty path on a non-watertight mesh |
| Poisson (soma) | **no-op** — the meshlab filter was already broken | `NEURD_REAL_POISSON=1` |
| Decimator (soma) | **open3d**, in-process, cascaded halving | `NEURD_MESHLAB_DECIMATE=1`, `NEURD_DECIMATE_SINGLE_STAGE=1` |
| FillHoles (soma) | no-op | — |
| `tu.vertex_components` | scipy union-find | `NEURD_LEGACY_VERTEX_COMPONENTS=1` |
| `nu.get_matching_vertices` | `cKDTree.query_pairs` | `NEURD_LEGACY_MATCHING_VERTICES=1` |
| `su.compressed_pickle` | no-op | `NEURD_ENABLE_COMPRESSED_PICKLE=1` |
| trimesh `_cache` | dropped after segmentation and on branches | — |
| memory reclaim at the soma→segment boundary | on (`_reclaim_memory`) | — |

The public spine/refinement/ray settings are applied by `RunConfig`. Compatibility patches
are installed by `neurd/_bootstrap.py` on the first package import and registered
in `neurd.PATCH_STATUS`; failed patches warn. The runner fingerprints effective
settings, behavior flags and native binaries for resume.

---

## Shipped levers

**Historical wall time** — measured before the experimental integration. These effects
are observations from their respective configurations, not additive release speed claims.

| lever | effect | commit |
|---|---|---|
| Poisson → in-process no-op (the meshlab filter was a byte-identical no-op that still spawned xvfb) | small mesh 688 → 151 s (−4.5×) | `469f492` |
| Decimator → open3d in-process | big −184 s (−14%) | `fcc72ce` |
| CGAL SDF ray count 25 → 12 | big **−17.8%**; now the explicit fast profile, with a different sampling budget | `c4197ab`, `a06ef87` |
| `tu.vertex_components` → owned scipy union-find | −205..219 s (−10%) | `a83d7c3` |
| spine segment connectivity O(N²) → `submesh_ops.pieces_adjacency` | **−23%** on the 187 MB h01 | `cc81169` |
| `get_matching_vertices` O(N²) distance matrix → `cKDTree.query_pairs` | −~250 s, ~25 GB less churn | patch in `_bootstrap.py` |
| `get_matching_vertices` skeleton path → `cKDTree.query_pairs` | O(N²) → O(N log N) | `a6d4f98` |
| debug `compressed_pickle` dump-before-raise → no-op | −98 s (−4.6%) | `19de5c2` |
| cascaded open3d decimation in halving steps | −25 s of 107 | `8409365` |
| soma volume → convex hull (was `fill_mesh_holes_with_fan`) | small −35 s | `7918ffc` |
| opt-in skip of `_split_soma_at_end`'s CGAL trim | −225 s when enabled | `d4e1149` |

**Peak RAM** — two levers, measured with `ru_maxrss` (kernel lifetime peak, sampler-independent):

| step | peak | note |
|---|---|---|
| before | 5282 MB | sampled at 0.02 s |
| **R1** — `connected_face_components` no longer materializes `vertex_faces` | 4330 MB (**−18%**) | also made CFC 3.5× faster |
| **R2** — `_reclaim_memory` at the soma→segment boundary | **3226 MB** (from an `ru_maxrss` baseline of 4842, **−33%**) | structure identical |

R1: `mesh.vertex_faces` is an `(n_vertices × max_vertex_degree)` int64 array, `-1` padded to the
maximum degree in the mesh. On the big h01 that is `(1,628,854 × 65) = 847 MB` while the *median*
degree is 6 — **90.7% of the array is filler**, sized by a handful of pathological vertices. Faces are
now labelled from the same scipy union-find the vertex partition already computes
(`labels[faces[:,0]]`), grouped in `vertex_components` order by a stable argsort. It was the only
`vertex_faces` consumer in the codebase. Gated by `tests/unit/test_cfc_vertex_faces_free.py` (44
tests) which keeps a verbatim copy of the old implementation as the equivalence oracle — that gate
caught a real order bug during development.

R2: probing the instant `_extract_single_soma` returns showed its +1904 MB resident baseline was
**92% reclaimable garbage**, not live data:

```
RSS at return        = 2746 MB
after gc.collect()   = 1838 MB   (−908 MB — reference cycles in the Poisson/CGAL intermediates)
after malloc_trim(0) =  646 MB   (−1192 MB — glibc arena fragmentation)
truly still resident =  188 MB above the pre-extract baseline
```

So `_reclaim_memory()` = `gc.collect()` + `ctypes` `malloc_trim(0)`, called at the stage boundary so
the peak-bearing stage stacks on 646 MB instead of 2746. Byte-exact by construction: `gc.collect()`
frees only unreachable objects and `malloc_trim` only returns unused pages.

---

## Where time and RAM actually go

**Wall (py-spy leaf attribution, big h01).** The real driver is **CGAL `mesh_segmentation`, ~34%** —
about 490 s in spines and 250 s in the soma path. Skeletonization plus its networkx graph assembly is
next (the teasar core itself is cheap at 0.31 s; building the skeleton graphs is what costs). Roughly
60% of the remaining wall sits **inside `mesh_tools`**, where durable edits are impossible —
`split_by_vertices`, `resolve_empty`/`filter_face_coloring`, and ~1.2 M `np.unique` calls.

⛔ **`cProfile` cumtime lies here** and cost real time in this project: the native ops (embree,
meshparty C, meshlab, scipy, CGAL) release the GIL, so their wall shows up as `threading.wait` or as
inflated caller cumtime. It reported skeletal-distance as the hot spot at "0.5%" reality. Use py-spy
or `tests/tools/wall_profile.py`, never cProfile cumtime.

**Earlier anchor observations:** Python SDF ray tracing measured 0.46 s, `deepcopy`
about 2 s and teasar 0.31 s in the earlier profiled setup. These do not bound native
CGAL SDF on larger branches: the October 11 retry measured 1,233.340 s there.
`max_somas` is a correctness guardrail on a true single-soma mesh. Earlier local
spine-parallel experiments encountered native-pool deadlocks after `fork` and
roughly tripled RAM with `forkserver`/`spawn`; those results are scoped to that
setup, and raw batch uses isolated spawn workers.

**Historical RAM ranking after reclaim**, before the owned OFF loader:

1. **⭐ The `load_mesh` ASCII-OFF transient, 2571 MB.** `tu.load_mesh_no_processing` is
   `trimesh.load_mesh(off, process=False)` (in `mesh_tools`, locked); the ASCII OFF parser balloons to
   ~2.57 GB on the 166 MB file, then releases. Everything in construction is now squeezed to within
   +655 MB of it. The owned chunked reader now removes this transient at the raw-neuron,
   descriptor and gate entry points; see the measurements below. Internal upstream/MeshLab
   reads still use their existing reader. Re-profile the whole construction before claiming
   its new peak. Ordinary float32 PLY would not be an exact geometry cache.
2. **`connected_face_components` residual, +636 MB** — `edges_unique` plus coo/union-find transients
   on 3 M faces.
3. **The `non_soma_mesh` full copy, +551 MB** — `main_mesh.submesh([non_soma_faces])` exists only to
   remove the soma faces before splitting, costing one whole extra copy alive through the split. Would
   need a face mask + provenance remap so the level-2 face maps stay exact
   (`root_face_idx` / `branch_meshes_orig_idx`).

Note the decomposition in 2/3 was measured *before* the reclaim landed — re-profile on the trimmed
   base before sizing them; they may now largely fit under the load transient.

⚠️ Wall time correlates with the **number of limbs**, and stitching is nondeterministic run to run
(±100–150 s). Don't mistake that for a regression from an optimization.

---

## The remaining big lever: CGAL spine segmentation

At ~34% of wall (490 s spines + 250 s soma), CGAL `mesh_segmentation` is the largest single cost and
the only one big enough to change the overall picture. The ray-count lever (`NEURD_SDF_RAYS`, −17.8%)
took the cheap part of it. Going further means touching the segmentation itself, which is **high risk
and hard to gate**: spine *count* is nondeterministic (22/16/20 observed on unchanged code), so a
count alone cannot establish equivalence. Use the compiled-provider SDF oracle in
`tests/integration/test_cgal_segmentation_oracle.py`, branch-local face masks and mesh/volume
invariants. Whole-neuron comparisons still need explicit tolerances for decomposition variance.

## Buffered native output, 2026-10-11

The integration candidate replaces the two per-face `std::endl` flushes with newline writes.
`NEURD_CGAL_TIMINGS=1` reports OFF read, SDF, clustering and both CSV writes separately.
The algorithm, sampling budget and CSV formatting stay the same. Candidate and baseline
modules were built into separate temporary directories; the installed environment was preserved.

Five interleaved trials per build used 25 rays and one numerical thread:

| Mesh | Baseline median provider time | Candidate median provider time | Output |
|---|---:|---:|---|
| `tests/990_mesh.off`, 7,240 faces, 3 clusters / 0.20 | 0.431 s | 0.417 s | Both CSVs byte-identical |
| H01 `1304599410/limb_000/branch_003`, 5 clusters / 0.08 | 2.877 s | 3.590 s | Both CSVs byte-identical |

These timings do **not** demonstrate a consistent wall-time improvement. The H01 trial
times varied substantially; median candidate SDF time alone was 3.508 s, while its two
CSV writes totalled about 0.0033 s. Treat the change as a measured reduction in I/O work,
and use the timers before changing any numerical algorithm.

A separate `strace` check on the 7,240-face mesh counted **14,483 → 39** write/writev
calls, including diagnostic output. CSV hashes still matched. Traced times are excluded
from the wall-time comparison. Peak RSS was unchanged at 20,180 KiB on that small mesh;
the H01 build peaks were 26,844 and 26,744 KiB.

Reproduce the A/B check with separately built provider directories:

```bash
python tests/tools/benchmark_cgal_io.py \
    --baseline /path/to/old-build --candidate /path/to/new-build \
    --mesh /path/to/fixture.off --repeat 5 --rays 25 --out /tmp/cgal-ab.json
```

The report records input and native-binary hashes, configuration, CSV hashes, timings
and peak RSS. Local receipts are retained under
`artifacts/release_candidate_validation_20261011/`; these private data paths are not
a portable public fixture recipe. The measured `tests/990_mesh.off` is an existing
local fixture and is not tracked in Git; supply your own checksummed OFF input.

## Bounded ASCII OFF loading, 2026-10-11

`neurd.mesh_io.load_mesh` allocates the final float64 vertex and int64 triangle arrays
once, then parses at most roughly 1 MiB of text per conversion. It preserves vertex
and face order, winding, duplicate/unused vertices and duplicate triangles. It performs
no geometry cleaning, persistent caching or source conversion. The raw runners,
descriptor extractor and geometry gates call it explicitly.

Three fresh-process trials per reader, interleaved with identical imports and one
numerical thread, used H01 `neuron_1830470325.off`: 166,526,436 bytes,
1,628,854 vertices and 3,267,142 faces.

| Reader | Median reading time | Maximum kernel peak RSS |
|---|---:|---:|
| Previous `tu.load_mesh_no_processing` | 11.838 s | 2,723,964 KiB (2.60 GiB) |
| Owned reader | 6.251 s | 587,304 KiB (0.56 GiB) |

Reading was **47.2% faster**, with **78.4% lower peak RSS**. These peaks include the
same NEURD/mesh_tools imports; they are lifetime process peaks, not allocation deltas.
All six outputs had byte-identical float64 vertices, int64 faces, metadata and visual
state. Input hashes/stat metadata and benchmark/loader source hashes were unchanged.
This establishes a load-stage improvement; it does not establish the same reduction
for the complete soma/limb/spine pipeline. Initial reader trials cut memory but were
slower; compiled checks for ordinary records removed that parser overhead before the
final measurements.

```bash
python tests/tools/benchmark_mesh_load.py \
    --mesh /path/to/neuron.off --repeat 3 --out /tmp/mesh-load-ab.json
```

Other formats, polygon/color/binary OFF variants, oversized records and unusual
comment/separator layouts delegate to trimesh after temporary fast-reader arrays
have been released. Later first comments delegate because the tested trimesh reader
has legacy prefix handling that must be preserved. Malformed counts, nonfinite
coordinates and out-of-range triangular indices fail clearly on the fast path.
MeshLab's index-repair path intentionally reads invalid intermediate indices and
retains its previous loader. `NEURD_LEGACY_MESH_LOADER=1` restores legacy loading at
supported entry points; the flag is included in raw, branch and descriptor fingerprints.

Receipts and logs are retained in the ignored local directory
`artifacts/off_loader_validation_20261011/v1/`; source data stays outside Git.

---

## Principle: never edit `mesh_tools` directly

`mesh_tools` is a plain site-packages install (not in the repo, not editable): edits are lost on the
next `pip install` — which has already happened here, to the C++ CGAL `.so` and to skeleton_utils
patches. It is 116 functions / 16.7k lines of core algorithms; replacing it means rewriting NEURD and
losing fidelity.

Use an owned wrapper at explicit call sites when the contract can be isolated, as
with `mesh_io.load_mesh`. Shared upstream seams can be intercepted in-process through
`neurd/_bootstrap.py`, with a reversible flag and an equivalence gate. That is how
Poisson, FillHoles, Decimator, `vertex_components` and `get_matching_vertices` are
handled. Keep durable changes in this repository rather than editing site-packages.

---

## Measurement methodology — do not repeat these mistakes

- **`cProfile` cumtime is wrong for this pipeline** (GIL-releasing native ops) — see above.
- **`tracemalloc` is the wrong tool for RAM**: ~10× slowdown, it inflates the metric (load peak 5.2 GB
  vs a true 2.6 GB), and it **misses numpy/trimesh native memory** entirely, since that goes through
  malloc rather than the CPython allocator. Same class of error as cProfile-cumtime for wall.
- **`memray` is right** (it intercepts malloc, so it sees native memory) but on the big h01 the capture
  had 960 M allocations: `memray stats` works (~2 min, though it reports *cumulative churn*, not
  resident-at-peak) while `memray tree`/`flamegraph` — the resident high-watermark snapshot you
  actually want — is effectively hung. Get that snapshot from the smaller microns mesh (same split
  transient, far fewer allocations) or with `--aggregate`. Invoke as `memray run -o out.bin SCRIPT.py`,
  **not** `memray run … python SCRIPT` (memray runs the interpreter itself; a literal `python` argument
  makes it try to open a file called `python`).
- **Use `ru_maxrss`, not a sampled peak.** Sampler interval is itself a confound: 0.5 s sampling
  under-reports the peak by ~350 MB versus 0.02 s. If you must A/B sampled numbers, use an identical
  interval on both sides.
- **Run variance is large** — 4.8–5.6 GB peak and 840–1050 s wall for the *same* config. Judge a lever
  by a clear drop (>0.5 GB) plus structure identity, and interleave off/on runs.
- **"Attributed rise" is an upper bound on the win, not the win.** Removing the top holder does not
  drop the peak by its size — the next holder underneath becomes the new ceiling. Here +1565 MB
  attributed yielded −952 MB realised.
- **Do not predict a fix from a diagnosis.** Two levers were designed from a correct diagnosis and
  then falsified by A/B: (a) `del non_soma_mesh` after the split moved nothing, because the peak
  happens *during* the split; (b) clearing the parent's trimesh cache inside `split` between the
  partition and the copies moved nothing (4893 vs 4894 MB), because the peak is reached *while*
  `vertex_faces` is being built, and freeing memory after a high-watermark has passed cannot lower it.
  The corrected rule: **to move a peak you must not allocate the object, or reduce what coexists
  during its construction — freeing afterwards is useless.**

## Spine label assembly, October 11 follow-up

The installed labels-only `mesh_segmentation` path still constructed all segment
meshes, discarded them, and let the spine caller construct them again. On a bounded
synthetic 625-label case, the actual caller performed two assemblies taking about
0.063 and 0.057 s. The owned `neurd/cgal_labels.py` seam preserves preparation,
native arguments, CSV parsing, original-face remapping and temporary-file behavior,
while constructing the required label meshes once. Its explicit assembly uses the
same trimesh call and face order as the previous spine caller.

`submesh_ops.face_groups_by_label` now uses one stable sort and group slices rather
than `np.where(labels == label)` for each label. Three fresh interleaved trials on
the same **synthetic 1,564,481-face / 22,167-label vector** gave:

| Index grouping | Median seconds |
|---|---:|
| Previous per-label scan | 9.569 |
| Stable grouping | 0.155 |

All label groups, ordered face indices and input hashes matched exactly. This
approximately **62×** result measures index grouping only. It excludes native
segmentation, mesh construction and shaft/head filters, and does not demonstrate
resolution of the historical 2,700-second timeout. The archived measured source
and final source have identical executable ASTs; only the module scope docstring
was updated to describe production use.

Two compiled-provider per-face comparisons and the real Minnie65 positive branch
matched labels/SDF, masks and volumes. Its sequential wall times were affected by
concurrent work and are not performance evidence. The full raw H01 neuron
`2530864375` also matched every metric exactly between the previous and owned
assembly paths: 18 spines, 8,145 spine faces and the same actual-face-mask digest,
with all invalid/ambiguous/overlap/volume counters zero. Both runs used seed 0,
fast 12 rays, the same compiled provider and pinned threads (Numba 4; the recorded
BLAS/OpenMP/NumExpr limits 1). Their single observed times, 394.844 and 235.075 s,
are retained for traceability, not a whole-pipeline speedup claim; the first run
overlapped unit validation. The initial sandboxed MeshLab failure is retained;
both successful comparison runs used the same elevated execution profile. Use
`NEURD_LEGACY_CGAL_ASSEMBLY=1` for the previous path. `NEURD_CGAL_TIMINGS=1` adds
flushed begin/end records for Python preparation, native/CSV, explicit assembly
and shaft filtering, alongside native timings. An unfinished begin record can
identify the phase active when a worker is killed.

Local receipts and reproducible bounded probes are in
`artifacts/h01_correspondence_followup_20261011/v1/grouping_benchmark/` and
`artifacts/h01_correspondence_followup_20261011/v1/timeout_review/`; the full-neuron
comparison is in `artifacts/h01_correspondence_followup_20261011/v1/full_gate/`.

The earlier [lineage candidate probe](artifacts/h01_export_lineage_20261011/v1/raw_candidate_gate/receipt.json)
also matched prior metrics (28 branches, 18 spines / 8,145 spine faces) and checked
28 supplied branch-to-limb maps. RSS rose **375.18 MiB** across the broad
`segment_limbs_from_soma` boundary; the same process's kernel HWM was **1,255.8 MiB**.
These are one recorded source/run's stage delta and self high-water mark, not a
process-group peak, allocation attribution or demonstrated memory/speed gain.
Independent review subsequently added referenced vertex-bounds checks. The
[final production gate](artifacts/h01_export_lineage_20261011/v1/raw_final_gate/receipt.json)
then passed on the actual promoted source: all 28 supplied maps available,
graph identity verified and every prior metric exactly equal (18 spines /
8,145 faces, identical mask digest, zero error/overlap counters). This run also
captured the real stage; capture writes affect subsequent RSS/HWM, which must
not be treated as a clean memory benchmark.

The separate [patched timeout retry](artifacts/h01_timeout_retry_20261011/v1/receipt.json)
completed within the 2,700-second limit: **300 spines / 286,839 faces**, strict
completion/QC passed. Segmentation took **2,273.321 s**, observer total **2,336.936 s**;
native SDF **1,233.340 s**, clustering **18.092 s**, explicit assembly **9.295 s** and
shaft filtering **84.211 s**. The post-shaft remainder was not individually tagged.
This one current-code attempt is not an isolated speed trial or a historical replay;
300 outputs versus 356 old spines is not an anatomical improvement. The original
graph was retained, auxiliary geometry served heads, and the inherited
`previously_spine_positive` exception applied without a recovered width or the
115 nm width gate. All computational error/overlap counters are zero, but 209
convex-hull and 91 fan-closed volume estimates still require quality review.

The post-CGAL follow-up below implements observation and two bounded changes
from the [post-CGAL proposal](artifacts/h01_export_lineage_20261011/v1/post_cgal_review/proposal.json).
Mask order, filter/RNG order, threshold ties and independent ownership checks
remain acceptance criteria.
The [non-soma prototype](artifacts/h01_export_lineage_20261011/v1/memory_review/proposal.json)
has 43 artifact-only cases and no implemented production RSS saving. Its
[captured real-input A/B](artifacts/h01_export_lineage_20261011/v1/memory_review/real_stage_ab/receipt.json)
passed three interleaved pairs of fresh children: all 424 ordered components,
source-face IDs, normals, stored colors/metadata, soma routing/touching and RNG
state matched exactly on 301,315 selected faces.

| Component extraction, median of three trials | Current stage | Direct prototype |
|---|---:|---:|
| Wall time | 0.238 s | 0.558 s |
| Kernel HWM sampled immediately after components | 661.99 MiB | 607.92 MiB |

This saves **54.07 MiB** of sampled HWM and **75.55 MiB** of retained
post-component RSS, adding **0.320 s** median extraction time. Both routes warm
the same parent triangles/centers before timing. Selection/matching, soma-edge
connectivity and validation are excluded; HWM includes common imports/loading/
warmup and is sampled before validation. The original matching allocator/KDTree
history and full-stage/pipeline peak were not reproduced. This is a measured
component-level memory/time tradeoff, not a production speedup or whole-pipeline
peak reduction. Require a controlled full gate before integration. Ordered raw
ownership remains unavailable.

## Post-CGAL observation and local optimizations

`NEURD_CGAL_TIMINGS=1` now observes adjacency construction, shaft paths,
size/surface-skeleton length, primary refinement, trim rechecks, fallback and
remaining-head validation, final-filter subphases and H01 publication. Fields
include nesting depth, parent phase and inherited call role. Counts are computed
only when tracing is enabled; tracing defaults to silent. A failed phase closes
with an error status; a worker killed during a phase leaves an unfinished begin.

Summarize a **single worker, single thread** log with:

```bash
python tests/tools/aggregate_cgal_phases.py worker.log --output phases.json
```

Inclusive durations overlap. The tool subtracts timed direct children to report
self time, rejects malformed nesting/durations and retains unfinished/error
phases. Untimed work is absent; concatenated or interleaved workers cannot be
treated as one stack. Native stderr remains a separate timing source.

A checksummed real H01 branch (`1072605926/limb_004/branch_001`, 189,940 faces)
supplied a captured shaft graph with 1,724 nodes, 3,175 edges and **22,052 ordered
queries**. Three fresh, interleaved process pairs compared NetworkX 3.6.1 against
an owned Numba CSR search preserving node/neighbor insertion order and the same
bidirectional BFS tie policy. Every full ordered path and exception matched.

| Captured shaft query stage | Median seconds |
|---|---:|
| NetworkX | 10.321 |
| Ordered CSR, including preparation | 0.456 |

The **22.63×** ratio applies to this query stage. Common kernel warmup was outside
both timed regions: first cold compilation took 0.379 s, later fresh-process
cache loads 0.146–0.292 s. Owned CSR/scratch arrays occupied 147,352 bytes; RSS/HWM
include common imports/JIT. Production selects this path only for at least 1,024
queries, an ordinary undirected simple Graph, verified NetworkX 3.6.1 and available
Numba. Other cases retain NetworkX; `NEURD_LEGACY_SHAFT_PATHS=1` restores it
explicitly. The runtime fingerprint records that flag. The kernel's executable
AST matches the measured prototype. See the [query A/B receipt](artifacts/post_cgal_followup_20261011/v1/shaft_path_review/real_stage_ab/receipt.json).

The H01 worker previously estimated the same accepted in-memory volume once in
the filter and again in export. An optional branch-local collector now carries
the exact immutable quality report, strong mesh identity and ordered geometry
digest. Export rejects a changed mesh and copies the report. Rejected candidates
still undergo the original independent acceptance calculations. Scientific
validation of saved OFF meshes requires a separate independent reload QC;
completion/resume validation checks hashes and inventory. Default calls without a collector retain
the previous scalar volume path and return structure.

Three fresh, interleaved pairs on **61 accepted meshes / 40,455 faces** compared
two estimates against one estimate plus collection/reuse digests. Reports,
scalars, input geometry and RNG matched exactly. Median time was **3.481 →
2.232 s** (1.56×, 1.249 s saved) for this accepted-mesh operation. Collection costs
for rejected candidates, selection/filtering, OFF writes and reload QC were
outside that measurement. Concurrent machine activity and trial variance limit
the comparison; it supplies no whole-pipeline or memory-saving claim. See the
[volume A/B receipt](artifacts/post_cgal_followup_20261011/v1/volume_reuse_review/real_volume_pairs/receipt.json).

Prepared border/endpoint/soma filter support remains an artifact-only prototype.
H01's default border filter is disabled; three actual survivor-filter calls totalled
0.782 s versus 0.725 s in one noisy pair. This does not justify promotion. The
non-soma prototype above also remains separate pending whole-run memory evidence.

Fifth-package production validation passed **771 unit tests / 1 expected skip** in
48.39 s, with 69 source/test hashes unchanged. A real cached-label refinement
pair preserved all 54 mesh geometries, volumes, RNG states, decisions and parent-ID
sets, while correcting the original retained-mesh row order for 40 masks. All
54 candidate ordered maps pass. The final native raw H01 gate reproduces exact
prior metrics (18 spines / 8,145 faces), with 28 validated supplied branch maps
and 28 local origin journals. The separate final reference-25 H01 batch completes
52 spines / 42,050 faces; an explicit independent OFF-reload audit verifies
ordered maps, owners, surface disjointness and recomputed strict positive volumes.
These runs establish correctness for their recorded policies, with no overall
speed/memory or anatomical claim. Final receipts are in
`artifacts/post_cgal_followup_20261011/v1/`.
The final native Minnie65 branch reproduces all nine spines / 2,854 faces, metrics
and scalar volumes exactly; all nine ordered maps pass. Its 8,372-query batch
uses the optimized shaft backend, establishing cross-dataset correctness for
that recorded fixture.

The subsequent readability cleanup passes **802 unit tests / 1 expected skip**
in 59.76 s. Conditional plain/refined replays preserve 460 ordered arrays,
refinement decisions and RNG state. Native checks preserve the full raw H01
result (18 spines / 8,145 faces), the 52-spine H01 branch with independent OFF-reload
QC, and all nine Minnie65 spines. Receipts are retained separately in
`artifacts/readability_cleanup_20261011/v1/`.

One cleanup raw-H01 observation records loading **0.604 s**, decomposition
**174.176 s** and export **0.690 s**; observed total **175.510 s** includes
metrics/export and excludes imports and subsequent audits. Process lifetime
peak RSS is **1,309,884,416 bytes**. This is a single correctness run with the
recorded fast/12-ray, refinement-off policy, not a controlled before/after
performance comparison. The cleanup removes unused whole-mesh allocations,
but no additional overall speed or memory reduction has been measured.

The subsequent query/object API pass is a correctness/readability and dependency
change. Its **881 unit tests / 1 expected skip**, full raw H01 gate and native
Minnie65 gate pass with three removed import roots blocked. The raw observer also
compares old/new query rows and branch selections on the same actual object.
These runs use the existing environment; the raw run overlaps portable testing
and includes additional query audits. Their elapsed times provide no controlled
before/after speed or memory result. The installed-metadata graph reduction
**210 → 192** is a potential dependency closure change, not a measured installation
size. Evidence is in `artifacts/legacy_api_cleanup_20261011/v1/`.

The following statistics/decomposition API repair passes **958 unit tests / 1
expected skip** on 104 frozen source/test files. It adds explicit input/storage
controls and corrects two zero-length density guards; the 66-case historical
summary oracle preserves ordinary enriched results and scalar types. Its native
Minnie control retains nine spines and all 27 geometry/mask files exactly.
The core grows by 35 physical lines (**18,800 → 18,835**) to support the repaired
API. These are correctness checks; no new time, memory or installation-size
improvement is measured. Evidence is in
`artifacts/statistics_api_cleanup_20261011/v1/`.
Its full raw H01 control also preserves all prior metrics and 28 supplied maps,
and exercises returned/stored products on the actual object. The first sandboxed
attempt failed in external MeshLab filters; the same frozen source passed after
external-process escalation, using the authenticated prior CGAL binary. The final
run includes extra summary/storage audits, so its recorded durations and RSS are
correctness observations without a controlled performance comparison.

The raw export separation passes **998 unit tests / 1 expected skip**. It moves
14 unchanged export functions into `neurd.neuron_export` and reduces the raw runner
from 982 to 500 lines; package plus runner grows by 28 lines for documentation,
imports and compatibility aliases. Full raw H01 processing retains prior metrics
and produces 218 byte-identical files with the old saver, public API and CLI alias.
The Minnie native control preserves nine spines, nine volumes and all 27 saved
geometry/mask files. The raw observation includes three exports and additional
audits; its timings and RSS establish no controlled speed or memory improvement.
Evidence is in `artifacts/raw_export_cleanup_20261011/v1/`.

The MAP/MP boundary cleanup passes **1,031 tests / one expected skip** and a
33-case frozen-source comparison under controlled geometry providers. It names
private stage results and removes aliases; the orchestrator shrinks 206 → 197
lines and the core by two. Full raw H01 processing retains all prior metrics and
218 byte-identical exports; native coverage is MP, with MAP/mixed checked by the
controlled-provider oracle. Minnie retains nine spines, nine volumes and 27 saved
geometry/mask files. Evidence is in `artifacts/map_mp_boundary_cleanup_20261011/v1/`.
The raw correctness observation records loading **0.612 s**, decomposition
**737.974 s**, one public export **8.916 s** and observed total **752.022 s**;
process lifetime peak RSS is **1,391,616,000 bytes**. It includes private stage
observation, metrics/export audits and overlaps other validation work. Unchanged
native SDF stages also take longer than prior observations; the cause has not
been established. These runs provide no controlled before/after speed or memory
comparison, and this cleanup establishes no measured performance improvement.

The bootstrap extraction passes **1,053 unit tests / one expected skip** and a
19-case initial-import comparison. It retains the scientific patch bodies and
installation order while reducing `__init__.py` from 503 to 48 lines; the core
grows by 36 lines. Full raw H01 processing with 507,737 faces, seed zero, fast
12-ray profile, spines enabled and refinement off retains 28 branches, 18 spines
and 8,145 spine faces. All 218 exports match the preceding package byte for byte.
This correctness observation records loading **0.588 s**, decomposition
**157.799 s**, one public export **1.346 s**, observed total **160.548 s** and
process lifetime peak RSS **1,324,535,808 bytes**. The run includes stage observation
and metrics/export audits; it is not a repeated, isolated before/after benchmark.
Its difference from earlier observations cannot be attributed to this extraction,
which establishes no measured speed or memory improvement. Evidence:
`artifacts/bootstrap_cleanup_20261011/v1/`.
The separate Minnie native branch retains nine spines, nine volumes and all
27 geometry/mask files; this is a branch control, not a full Minnie-neuron run.

The large retry's 1,233.340-second native SDF is still the largest observed stage.
The new traces should guide the next CPU change; surface-skeleton length work
must preserve sampling/RNG behavior, and native changes need per-face gates.

## GPU — honest assessment

No GPU implementation or speed gain has been validated for this pipeline.
The current large-branch retry measured native CPU SDF at 1,233.340 s and clustering
at 18.092 s; earlier Python SDF timings describe a different workload. These
measurements identify stages to investigate, not expected GPU gains. Any GPU
experiment needs its own measured latency/throughput and exact per-face scientific
gates; post-CGAL CPU work and geometry/provenance checks still require profiling.
