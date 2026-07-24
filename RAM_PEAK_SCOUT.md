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
