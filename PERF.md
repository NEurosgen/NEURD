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
are installed by `neurd/__init__.py` and registered in `neurd.PATCH_STATUS`; failed patches
warn. The runner fingerprints effective settings, behavior flags and native binaries for resume.

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
| `get_matching_vertices` O(N²) distance matrix → `cKDTree.query_pairs` | −~250 s, ~25 GB less churn | patch in `__init__.py` |
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

**Not bottlenecks** (measure-first ruled these out): SDF ray tracing in Python (0.46 s), `deepcopy`
(~2 s), swapping the skeleton algorithm (teasar is 0.31 s — Kimimaro/Skeletor cannot help), GPU for a
single neuron, and `max_somas` short-circuiting (on a genuine single-soma mesh there is nothing extra
to skip — it is a correctness guardrail, not a speed lever). Spine parallelization via `fork` is
shelved: fork-after-threads deadlocks on the native BLAS/CGAL pools, and `forkserver`/`spawn` triples
RAM.

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
`neurd/__init__.py`, with a reversible flag and an equivalence gate. That is how
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

## GPU — honest assessment

For a **single neuron** GPU offers little: the bottleneck is skeleton graphs and topology, not number
crunching, and SDF ray tracing is already 0.46 s. The real potential is **throughput** —
`process_all_neurons` runs many meshes, and now that per-neuron RAM is down, more can run in parallel.
GPU-able in parts: ray casting/SDF (open3d RaycastingScene, warp, OptiX) and KMeans (cuML).
Skeletonization and stitching are not. The CUDA-Python stack is not installed; this is worth revisiting
only once it is clear whether latency or throughput is the binding constraint.
