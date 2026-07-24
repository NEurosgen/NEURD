# RAM peak scout — how to work on it next session

Scouting notes for optimizing the build's **peak RAM** (item "C"). Measured 2026-07-24 on the big
h01 `neuron_1830470325.off` (166 MB, 3.27 M faces). Companion memory: `ram-peak-decomposition`,
`cgal-spine-segmentation-lever`. This doc is the starting point — read it before touching anything.

## 1. The measured RAM map (honest RSS, phase markers)

| phase | window | RSS |
|---|---|---|
| import | — | ~460 MB |
| load_mesh | 0–20 s | **transient 2.57 GB** (ASCII OFF parse), settles to ~890 MB |
| **soma + `_segment_limbs_from_soma`** | 20–509 s | **PEAK ~4.9–5.6 GB** ← the ceiling |
| per-limb `preprocess_limb` ×N | 509–1487 s | flat plateau **~2.25 GB** |
| correspondence / spines | tail | ~2.8 GB |

- Ray count (`NEURD_SDF_RAYS`) is **RAM-neutral** — the CGAL time lever does nothing for RAM.
- Spines do NOT set the ceiling (a `calculate_spines=False` run still peaked 5.6 GB).
- The peak is a **transient**: >4.5 GB for only ~1 % of wall. Steady state is ~2.25 GB.
- Tools used: `scratchpad/ram_phases.py` (phase-boundary monkeypatch + RSS sampler),
  `tests/tools/benchmark_neuron.py` (per-stage RSS in summary.json).

## 2. Root cause — the mesh-copy transient in `_segment_limbs_from_soma`

`neurd/preprocess_neuron.py:2831` `_segment_limbs_from_soma(main_mesh, soma_mesh, params)`:

```
soma_faces_idx = submesh_ops.faces_by_match(main_mesh, soma_mesh)      # builds main_mesh.triangles_center (~78MB) + KDTree over 3.27M midpoints
non_soma_mesh  = main_mesh.submesh([non_soma_faces], append=True)      # (1) a NEAR-FULL COPY of the neuron mesh
sig_pieces_sm  = submesh_ops.split_significant(non_soma_mesh, ...)     # (2) split -> materializes ALL component submeshes at once
...
connected_pieces = tu.mesh_pieces_connectivity(main_mesh=main_mesh, ...) # touches main_mesh again
```

At the peak these are **all alive simultaneously** (≈ 3× the mesh + caches):

- `main_mesh` (166 MB) + its caches: `triangles_center`, `triangles`, `edges`, a KDTree.
- `non_soma_mesh` (~166 MB copy) + its caches: **`vertex_faces` ~0.9 GB** + `edges_unique`, built by
  `connected_face_components` (`submesh_ops.py:176`) inside the split.
- all component submeshes from `split()`'s list-comp `[SubMesh(_submesh(mesh, comps[i]), …) for i]`
  (`submesh_ops.py:273`) — their vertices sum to ~1× the mesh, and they're **held all at once**.

memray (cumulative churn, big h01) corroborates: `submesh` 37 GB of allocation churn (mesh copying),
the O(N²) distance-matrix family ~25 GB (that was a *separate time* lever, already fixed — see §5),
`mesh_segmentation` 22 GB, `_unique1d` 22 GB.

**Why the "free it after" reflex fails:** a speculative `del non_soma_mesh` placed AFTER the split (before
`mesh_pieces_connectivity`) did NOT move the peak (A/B 4971 vs 4878) — the peak happens *during* the
split, before any post-split del. `_drop_trimesh_caches(mesh)` at `preprocess_neuron.py:3095` likewise
runs after `_segment_limbs_from_soma` returns → lowers the plateau, not the peak. **To move the peak you
must reduce what's live DURING the split, not after.**

## 3. Ranked levers (for next session)

> ⛔ **C1 was IMPLEMENTED, gated, A/B-tested, and FALSIFIED on 2026-07-24 — see §6.** It does NOT
> move the peak (4893 MB off vs 4894 MB on). Reverted. Read §6 before re-attempting anything here.
>
> ✅ **The corrected lever (don't materialise `vertex_faces`) SHIPPED later that day: −952 MB.
> The peak is now localized and this ranking is superseded — read §6b, not §3/§7.**

**C1 — drop the parent's trimesh cache INSIDE `submesh_ops.split`, between the partition and the
copy materialization (cheapest, lowest risk).**
In `split()` / `split_significant()` the parent's heavy caches (`vertex_faces` ~0.9 GB, `edges_unique`)
are built by `connected_face_components` and are NOT needed to build the piece copies (`_submesh` only
reads `mesh.vertices` / `mesh.faces`). Clearing `mesh._cache` right after `comps =
connected_face_components(mesh, …)` frees ~0.9 GB during the N-copy build. Cache recomputes on demand →
safe. Caveat: `split` is a shared primitive — gate on structure identity + `tests/unit/test_submesh_ops.py`
and `test_vertex_components.py` (byte-exact split-by-vertices) so the cache-drop doesn't perturb output.
Expected: shave ~0.5–1 GB off the ceiling. **Do this one first and A/B the peak.**

**C2 — avoid the `non_soma_mesh` full copy (medium effort, medium risk).**
`non_soma_mesh = main_mesh.submesh([non_soma_faces])` exists only to remove the soma faces before
splitting. That's one whole extra copy of the mesh (~166 MB + its caches) alive through the split. If the
split could run on `main_mesh` restricted to `non_soma_faces` (a face mask + provenance remap) without
first materializing the copy, the peak drops by ~1×. Needs careful provenance (`root_face_idx` /
`branch_meshes_orig_idx` at `preprocess_neuron.py:2860`) so the level-2 face maps stay exact. Gate: full
structure identity (3 limbs / 129 branches on the big h01) + Phase-B provenance tests.

**C3 — the 2.57 GB load transient (isolated, low risk, does NOT touch the ceiling).**
`tu.load_mesh_no_processing` is just `trimesh.load_mesh(off, process=False)` (`mesh_tools`, LOCKED); the
ASCII OFF parser balloons to 2.57 GB on the 166 MB file, then releases. Below the construction ceiling,
but it matters when N parallel workers all load at once (`process_all_neurons_parallel.py`
`--mem-budget-mb`). Fix: convert meshes to binary PLY once, or wrap a numpy/streaming OFF reader in
`neurd`. Independent of C1/C2.

**Why it matters:** the peak gates per-worker RAM in the parallel runner. Dropping the ceiling from ~5 GB
toward the ~2.25 GB plateau roughly doubles the safe worker count in the same `--mem-budget-mb`.

## 4. Methodology — do NOT repeat these mistakes

- **tracemalloc is the wrong tool** — ~10× slowdown, inflates the metric (load peak 5.2 GB vs true 2.6 GB),
  and MISSES numpy/trimesh native memory (allocated via malloc, not the CPython allocator). Same class of
  error as cProfile-cumtime for wall.
- **memray is right** (intercepts malloc → sees native) BUT the big-h01 bin had **960 M allocations** →
  `memray stats` works (~2 min, but it's CUMULATIVE churn, not resident-at-peak) while `memray tree` /
  `flamegraph` (the high-watermark *resident* snapshot we actually want) is effectively hung (>40 min).
  **Next time: get the resident-at-peak snapshot by either (a) running memray on the microns 56 MB mesh
  (peaks ~2.8 GB, same split transient, far fewer allocations → tree/flamegraph tractable), or (b)
  `memray run --aggregate` for a smaller capture.** Invoke as `memray run -o out.bin SCRIPT.py …` — NOT
  `memray run … python SCRIPT` (memray runs the interpreter itself; a literal `python` arg makes it try to
  open a file named `python`). ptrace_scope=1 here, so py-spy must launch the child (`py-spy record -- …`),
  never `--pid`.
- **Honest RSS attribution without memray:** `scratchpad/ram_phases.py` (phase-boundary monkeypatch +
  0.25 s RSS sampler) localizes the peak to a phase cheaply (~1 build). It's the fastest way to A/B a lever.
- **Run variance is large** (~4.8–5.6 GB peak, ~840–1050 s wall for the same config) — judge a RAM lever by
  a clear peak drop (>0.5 GB) + structure identity, not a single noisy delta.

## 5. Context: what already shipped adjacent to this

- `get_matching_vertices` O(N²) → `cKDTree.query_pairs` (monkeypatch in `neurd/__init__.py`, gate
  `tests/unit/test_matching_vertices.py`). memray surfaced it while hunting the peak; it turned out to be a
  **time** lever (~-250 s / -25 %, pending py-spy confirm), NOT the peak (it's on the skeleton plateau).
  Peak unchanged by it (4877 MB).
- `NEURD_SDF_RAYS` CGAL ray lever (committed) — time only, RAM-neutral.
- `_drop_trimesh_caches(mesh)` at `preprocess_neuron.py:3095` already drops main_mesh caches for the
  plateau; C1 is the same idea moved *inside* the split to catch the peak.

## 6. 2026-07-24 session — C1 SHIPPED→REVERTED (falsified), and where the peak really isn't

**What was done.** Implemented C1 exactly as specified: added `free_parent_cache=False` to
`submesh_ops.split` / `split_significant` (default off → all other callers untouched), cleared
`mesh._cache` after `connected_face_components` but before the copy list-comp, opted in only at the
`_segment_limbs_from_soma` call site (env kill-switch `NEURD_FREE_SPLIT_CACHE`). Gated it with 2 new
byte-exact unit tests (output-identical on/off, both connectivities; heavy holders `vertex_faces`/
`edges_unique` proven dropped and not rebuilt by the copies). All 48 submesh/vertex-component tests green.

**A/B result — NO peak movement.** Big h01, `benchmark_neuron.py --no-profile --no-spines --data-type
h01`, interleaved off/on/off/on (each ~11 min):

| run | flag | peak RSS |
|---|---|---|
| off_a | OFF | 4863 MB |
| on_a  | ON  | 4881 MB |
| off_b | OFF | 4923 MB |
| on_b  | ON  | 4908 MB |

OFF mean 4893 vs ON mean 4894 → **+1 MB, inside the 60-MB off-run spread.** C1 does nothing.

**Why C1 fails (the corrected mental model).** The peak is reached WHILE `vertex_faces` (~0.9 GB) is
being *built* inside `connected_face_components`, i.e. it coexists with `main_mesh` + the
`non_soma_mesh` copy at that instant. C1's `.clear()` runs AFTER `connected_face_components` returns —
the high-watermark has already passed, so freeing 0.9 GB afterward can't lower a max that already
happened. (Numpy mmaps a 0.9-GB array; `free()`→`munmap()` WOULD drop RSS immediately if the peak were
after the clear — it isn't, so no drop.) **Corollary: to move this peak you must NOT MATERIALIZE
`vertex_faces` at all, or reduce what coexists DURING its build — not free anything after.**

**All P1 changes were reverted** (working tree clean; nothing committed). The env var, the two tests,
and the `import os` are gone.

**Localization attempt (inconclusive — tooling bug).** Recreated a phase-stack + RSS-sampler localizer
(`scratchpad/ram_localize.py`) to pin the exact peak phase. BUG: monkeypatching
`pn._extract_single_soma` / `_segment_limbs_from_soma` / `_decompose_limbs` and `so.split` did NOT fire
(only `so.connected_face_components` did) — so the phase labels were unreliable and the run was killed
before the final summary. An isolated 30-s patch-mechanism test was interrupted. NOT yet debugged.
- One real data point survived: during the observed per-limb decompose window the many small
  `connected_face_components` calls sat at only **~1731 MB** — i.e. the ~4900 peak is NOT in the small
  per-limb CFCs. It's earlier: the single big CFC on the full `non_soma_mesh`, and/or soma extraction.
- Fresh suspicion: the heavy legacy **soma-extraction** path prints "adjusted for decimation" from
  `soma_extraction_utils.py:900` (inside `extract_soma_center`, called by `_extract_single_soma`), and
  `soma_extraction_utils` ALSO calls `submesh_ops.split_significant` (lines 413, 549) on big meshes.
  The peak may live in soma extraction, NOT in `_segment_limbs_from_soma` at all — this was never
  cleanly separated (the doc's original markers also missed `extract_soma_center`).

## 6b. ✅ 2026-07-24 (later still) — PEAK LOCALIZED, and the `vertex_faces` lever SHIPPED

**R2 — the peak instant, measured.** Rebuilt the localizer as *direct in-function instrumentation*
(`neurd/_ram_probe.py`, env `NEURD_RAM_PROBE=1`, 0.02 s RSS sampler + a label stack, so the global
high-watermark is attributed to a scope instead of guessed). Note the §6 "patches didn't fire"
puzzle was never reproduced — `benchmark_neuron.py` runs **in-process, no subprocess**, so
monkeypatching should have worked; direct instrumentation sidesteps the question.

Result — unambiguous, it is **(b)**, and sharper than §7 hoped: not "the CFC" but ONE attribute
access inside it.

```
GLOBAL PEAK 5282 MB at t=421s
PEAK PHASE: _segment_limbs_from_soma > split_significant(non_soma_mesh)
            > CFC[verts] n_faces=3,001,397 > .vertex_faces
  +2127 MB  PHASE _segment_limbs_from_soma      (30.4s)
  +1565 MB    .vertex_faces                     ( 1.4s)   <- 74% of the rise
   +522 MB    submesh -> non_soma_mesh
     +0 MB    .loop / mesh_pieces_connectivity / faces_by_match
     +0 MB  every per-limb CFC                            <- confirms §6's surviving datapoint
```

**Why `vertex_faces` is so big** (the root cause §2/§3 never named): it is
`(n_vertices x MAX vertex degree)` int64, **-1 padded to the maximum degree in the mesh**. On the
big h01 that is `(1,628,854 x 65) = 847 MB` while the median vertex degree is **6** — so
**90.7 % of the array (769 MB) is filler**, sized by a handful of pathological vertices.

**R1 — SHIPPED.** `submesh_ops.connected_face_components` vertices-path no longer touches
`mesh.vertex_faces`. Faces are labelled from the same scipy union-find the vertex partition
already computes (`labels[faces[:,0]]`), grouped in `vertex_components` order by a stable argsort.
Shared core factored out as `_vertex_component_labels`. **It was the only `vertex_faces` consumer
in the codebase.**

| run (same 0.02 s probe sampler) | peak |
|---|---|
| baseline | 5282 MB |
| R1 | **4330 MB** |

**−952 MB / −18 %**, and CFC on the real mesh is **3.5x faster** (6.4 s -> 1.8 s).

Gating: `tests/unit/test_cfc_vertex_faces_free.py` (44 tests) keeps a **verbatim copy of the old
implementation as the equivalence oracle** — degenerate faces, a degree-200 hub vertex, unreferenced
vertices, randomised sweep with shuffled face order. Plus **byte-exact on the real 3.27 M-face h01**
(1678 components identical) and structure identity (3 limbs). Full suite 199 passed.
The gate earned its keep: it caught a real order bug (indexing `sorted_rank` by face index instead
of group start position) — exactly the failure the `vertex_components` docstring warns about, since
`split()` re-sorts with an unstable argsort.

### ⚠️ Methodology corrections (both cost time this session)

- **`peak_rose` is an UPPER BOUND on the win, not the win.** Removing the top holder does not drop
  the peak by its size — the next holder underneath becomes the new ceiling. Here `+1565 MB`
  attributed yielded `−952 MB` realised. Same shape of error as C1's, one level up.
- **Run variance is FAR worse than §4's "~60 MB"**: R1 measured 4753 / 4330 / 4072 MB across runs
  and samplers. **Sampler interval is itself a confound** — 0.5 s sampling under-reports the peak
  by ~350 MB vs 0.02 s. **Always A/B with an identical sampler interval**, and prefer the probe's
  number over `summary.json:rss_peak_mb` (default 0.5 s).

### Where the ceiling is NOW (post-R1) — the new ranking

```
  +2103 MB  PHASE _extract_single_soma   705 -> 2808 resident, over 340s (peak_rose only +331)
  +1232 MB  PHASE _segment_limbs_from_soma
     +636 MB   split_significant -> CFC   (edges_unique / union-find transients, no vertex_faces)
     +551 MB   submesh -> non_soma_mesh   (= C2)
      +38 MB   faces_by_match
```

1. **⭐ NEW #1 — soma extraction's resident baseline.** `_extract_single_soma` *leaves* +2103 MB
   resident (705 -> 2808) that everything else then stacks on. It barely moves the watermark itself,
   but it is the floor under the peak — the single biggest remaining lever. Profile its
   intermediates (decimation copies, poisson, its own `split_significant` at
   `soma_extraction_utils.py:413/549`) and free what it no longer needs before returning.
2. **C2 — the `non_soma_mesh` full copy** (+551 MB), unchanged from §3.
3. **CFC's residual +636 MB** — `edges_unique` + coo/union-find transients on 3 M faces.
4. **C3 — the 2.57 GB ASCII-OFF load transient**, still below the ceiling, still matters per-worker.

## 6c. ✅ 2026-07-24 (later still) — the soma floor was 92% GARBAGE, `_reclaim_memory` SHIPPED

§6b called `_extract_single_soma`'s +2103 MB resident baseline the new #1 lever and guessed it was
live intermediates to free. **That guess was wrong in a good way.** Probed the instant `_extract`
returns (`scratchpad/soma_floor_probe.py`: RSS at return → after `gc.collect()` → after
`malloc_trim(0)`):

```
  RSS at return             = 2746 MB   (rose +1904 during extract)
  after gc.collect()        = 1838 MB   (freed  908 MB — 78639 objs: reference cycles)
  after malloc_trim(0)      =  646 MB   (freed 1192 MB — glibc arena fragmentation)
  truly still resident      =  188 MB above the pre-extract baseline
```

The floor is **not live data**. It is dead objects the refcounter can't reclaim (reference cycles
in `extract_soma_center`'s Poisson/CGAL intermediates → `gc.collect()`) plus freed pages glibc
never returned to the OS (`malloc_trim(0)`). Only 188 MB actually survives. **92% of the floor is
reclaimable with two standard calls, touching no data path.**

**SHIPPED.** New `_reclaim_memory()` (helper by `_drop_trimesh_caches`) = `gc.collect()` +
`ctypes` `malloc_trim(0)`, called at the `_extract`→`_segment` boundary so the peak-bearing stage
stacks on 646 MB, not 2746. `import gc` added.

**Result — same benchmark harness, sequential runs, `ru_maxrss` (kernel lifetime peak, sampler-
independent):**

```
  R1-only (HEAD)          ru_maxrss 4842 MB   (construct-stage peak 4828)
  R1 + _reclaim_memory    ru_maxrss 3226 MB   (construct-stage peak 3222)
  -->  -1616 MB / -33%    structure identical: 3 limbs, 129 branches (54/74/1)
```

Byte-exact by construction — `gc.collect()` frees only unreachable objects, `malloc_trim` only
returns unused pages; neither can alter a result. Confirmed §6b's model: the `_segment` peak was
stacking on the un-trimmed floor; drop the floor and the whole ceiling falls.

### Where the ceiling is NOW (post-reclaim) — updated ranking

```
  3226 MB  new global peak (inside neuron_construction, on the trimmed 646 MB base)
  2571 MB  load_mesh transient  <-- now the #1 remaining floor (= C3, ASCII-OFF parse)
           headroom in construction above the load transient is only +655 MB
```

1. **⭐ NEW #1 — the load_mesh / ASCII-OFF transient (C3, 2571 MB).** Everything in construction is
   now squeezed to within +655 MB of it, so the load transient is the dominant remaining floor. The
   166 MB ASCII `.off` balloons to ~2.57 GB during parse. Lever: stream/chunk the OFF parse (or a
   binary pre-convert). Isolated, independent of the neuron algorithm, low risk. See §3's C3 note.
2. **CFC residual +636 MB** — `edges_unique` + coo/union-find transients on 3 M faces (in construction).
3. **C2 — the `non_soma_mesh` full copy** (+551 MB, in construction).
   NB the raw §6b decomposition (+2103/+1232) predates reclaim; re-profile the phases on the trimmed
   base before sizing 2/3 — they may now largely fit under the load transient.

## 7. ⭐ NEXT-SESSION PLAN (2026-07-24, task carried over)

> Superseded by §6b + §6c: localization DONE, the `vertex_faces` lever (item 2) SHIPPED, and the
> soma floor (item 3) turned out to be reclaimable garbage — `_reclaim_memory` SHIPPED (−33%).
> Start from §6c's "Where the ceiling is NOW" ranking; the new #1 is the C3 ASCII-OFF load transient.

Goal unchanged: cut the ~5 GB transient ceiling (gates parallel-worker count) toward the ~2.25 GB
plateau. C1 is dead; do NOT re-try "free after the split". New plan:

1. **LOCALIZE THE PEAK INSTANT FIRST — this is the gating unknown.** Fix the localizer
   (`scratchpad/ram_localize.py` is gone with the scratchpad; rebuild it). First debug WHY the
   top-level `preprocess_neuron` function patches didn't stick (note: `neuron.py:1280` does a *local*
   `from neurd import preprocess_neuron as pre` at call time — same module object, so patching
   `sys.modules['neurd.preprocess_neuron'].<fn>` *should* work; verify with a 30-s isolated test
   before burning an 11-min build). Or just instrument by temporarily adding RSS prints directly
   inside `_extract_single_soma`, the big `connected_face_components`, and the copy list-comp. Need ONE
   clean answer: is the 4900 peak in **(a) soma extraction** (extract_soma_center / decimation /
   poisson / its own split_significant) or **(b) the first `connected_face_components` on the full
   `non_soma_mesh`** (the `vertex_faces` 0.9-GB build)?

2. **If (b) — `vertex_faces`-free partition (the corrected lever).** In
   `submesh_ops.connected_face_components` vertices-path, stop materializing the `(n_verts × max_deg)`
   `vertex_faces` array (the 0.9 GB). We already have per-vertex component labels from the scipy
   union-find in `vertex_components`; a face's 3 vertices are always in one component, so label faces by
   `labels[faces[:, 0]]` and group in the SAME order `vertex_components` yields — O(n_faces), no padded
   array. This *doesn't build* the peak's dominant object → should actually move the ceiling, and is a
   time win too. Gate with the existing byte-exact `test_submesh_ops.py` / `test_vertex_components.py`
   (partition + order identical) + one structure-identity build (3 limbs / 129 branches).

3. **If (a) — attack soma extraction.** Profile `extract_soma_center` intermediates
   (`soma_extraction_utils.py`): decimation copies, poisson, the whole-mesh `split_significant`.
   Different lever family; re-scope from there.

4. **Still on the shelf:** C2 (avoid the `non_soma_mesh` full copy — §3), C3 (2.57-GB ASCII-OFF load
   transient — §3, independent, helps parallel workers).

**Tooling reminders for next session:**
- Big h01 mesh: `/home/eugen/Desktop/CodeWork/Projects/Diplom/notebooks/notebooks/H01/neuron_1830470325.off`.
- Peak in one number: `python tests/tools/benchmark_neuron.py --neuron <off> --out <dir> --no-profile
  --no-spines --data-type h01` → `summary.json:rss_peak_mb`. `--no-spines` cuts ~8 min/run and the peak
  is unchanged by spines (4863–4923 vs spines-on 4877), so a peak A/B run is ~11 min, not ~19.
- Run-variance is ~60 MB between identical runs → judge a lever by a >0.5 GB drop, interleave off/on.
- Do NOT use tracemalloc (§4). RSS-sampler + phase markers is the cheap honest tool.
