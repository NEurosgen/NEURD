# The slim segmentation pipeline — how it works, and the patches that hold it together

What happens geometrically on the path `mesh → Neuron`, where it lives in the code, what may and may
not be touched, and which compat patches keep it running on numpy 2 / trimesh 4 without Docker.
Module map: [NEURD_STRUCTURE.md](NEURD_STRUCTURE.md). Performance: [PERF.md](PERF.md).

> ⚠️ The downstream stages (multi-soma split, cell typing, axon, autoproofreading, after-proof stats)
> and their modules are **deleted** from the fork. Only the live slim path is described here.

---

## 0. The whole picture

The input is a **mesh of one neuron** from EM: a huge triangular surface (hundreds of thousands of
faces) — a rounded **cell body (soma)** with branching **processes** (dendrites/axon) covered in
**spines**. The slim pipeline's job is to turn that bag of triangles into structure: find the soma,
decompose the processes into a **skeleton** (centrelines) and **branches**, build the **connectivity
graph (concept network)**, and extract raw spines. Three primitives carry everything: **SDF**, the
**skeleton**, and the **concept network**.

The canonical path is [segmentation_pipeline.py](neurd/segmentation_pipeline.py) → `neuron.Neuron(mesh=)`
→ `preprocess_neuron` (soma identification, skeletonization, branches, raw spines, concept network).
Soma identification is not a separate stage: it runs inline inside the decomposition, which is where
`max_somas` and `verbose` are honoured.

`process_all_neurons.py` (the user's entry point) is shorter still — `neuron.Neuron(mesh=...)` directly
in a worker process per OFF mesh. The stats step `calculate_decomposition_products` is **off** the slim
path (its only caller was removed).

---

## 1. Geometric primitives

- **SDF (Shape Diameter Function, "thickness").** For each face, cast a ray **inward** along the
  inverted normal; the distance to exit is the local thickness. The soma is thick → high SDF;
  dendrites/axon are thin → low. This is the main "soma vs process" signal and the basis of branch
  width estimates. **Code:** `mesh_tools.trimesh_utils.ray_trace_distance(mesh)` (embree if available).
  ⚠️ The pipeline expects SDF **normalized to [0,1]** — the thresholds (`soma_width_threshold=0.32`)
  are on that scale.
- **Skeleton.** A 1-D graph running down the middle of a tubular process (its axis). Its length is the
  process length; its points carry width (from SDF); it is split into **branches** at branch points.
  **Code:** skeletonization in `preprocess_neuron.py` (meshafterparty/MAP,
  `mesh_tools.skeleton_utils`).
- **Mesh segmentation by SDF.** Clustering faces by SDF via a CGAL MRF graph cut. The real C++
  `cgal_Segmentation_Module` (.so) is **restored** (`cgal/cgal_segmentation/`, see
  [cgal/README.md](cgal/README.md)); the KMeans stand-in `neurd/_cgal_segmentation.py` remains only as
  a **fallback** when the .so is not built (`__init__.py` registers the stub only if the real module is
  absent). KMeans gave a coarse segmentation — acceptable for the soma, but **0 spines** — which is why
  real CGAL was brought back. **Code:** `tu.mesh_segmentation(mesh, clusters, smoothness)` → submeshes
  + a **per-segment SDF median**.
- **Concept network.** A directed graph of a limb's branches: nodes are branches, edges mean "A
  continues into B", the root is the soma contact point, and direction is upstream→downstream (outward
  from the soma). One graph per (limb, soma). **Code:** `nru.branches_to_concept_network(...)`, stored
  as `Limb.concept_network`.

### The Neuron object hierarchy

```
Neuron
 ├── somas:  S0, S1, ...                    (cell bodies: mesh + center)
 └── limbs:  L0, L1, ...                    (processes = connected components after cutting the soma out)
       ├── branches: 0,1,2,...              (skeleton segments between branch points)
       │     ├── mesh, mesh_face_idx        (the branch submesh)
       │     ├── skeleton                   (centreline)
       │     ├── width_array                (from SDF)
       │     ├── endpoint_upstream/downstream
       │     └── web                        (the mesh "web" at a branch point)
       └── concept_network                  (branch graph, rooted at the soma contact)
```

Files: `neuron.py` (the classes), `neuron_utils.py` (`nru.*` — graph queries), `branch_utils.py`.

---

## 2. The live stages — geometry and code

### Stage 1 — soma identification (`soma_extraction_utils`)

The most geometrically dense part. To find the soma mesh(es):

1. Decimation (coarser) → a fast candidate (`meshlab.Decimator`, now open3d in-process).
2. Isolate the large mesh pieces.
3. **Poisson surface reconstruction** (`meshlab.Poisson`) → a watertight shell.
4. **Remove interior** (`meshlab.Interior`, via Ambient Occlusion).
5. **Mesh segmentation by SDF** (`tu.mesh_segmentation`, clusters ≈ 3) → segments + SDF medians.
6. **Selection:** a segment passes if `SDF_median > soma_width_threshold` (0.32) **and** its size is
   within `[soma_size_threshold, _max]` — i.e. "a thick, large, roundish piece".
7. **Sphere validator** (bbox ≈ a ball) + **backtrack** of the coarse shell onto the original mesh's
   faces.

⚠️ The `0.32` threshold and the selection logic depend on SDF being normalized to [0,1] — any new SDF
provider **must** normalize (the stub does, on the [2,98] percentiles). The replacement bottleneck is
steps 3–4 (MeshLab).

`max_somas` caps this search. Only the first soma is ever used, so on a genuine single-neuron mesh it
bounds work without changing the result; if more than one soma is found the extras are dropped with a
warning.

### Stage 2 — decomposition (`neuron.Neuron(...)` → `preprocess_neuron.preprocess_neuron`)

`preprocess_neuron` is rewritten for **exactly one soma with no glia or nuclei**, and decomposed into a
thin orchestrator over named phases (each a separate function in
[neurd/preprocess_neuron.py](neurd/preprocess_neuron.py)):

1. `_extract_single_soma(mesh, segment_id, max_somas)` — finds exactly one soma (raises if none).
2. `_reclaim_memory()` — `gc.collect()` + `malloc_trim(0)` at the stage boundary. Not cosmetic: the
   soma stage leaves ~1.9 GB resident of which ~92% is reclaimable garbage, and the peak-bearing stage
   would otherwise stack on top of it (−33% peak RAM, see PERF.md).
3. `_segment_limbs_from_soma(mesh, soma, params)` — cuts the soma out; the remainder falls apart into
   **limbs** (connected components touching the soma) plus floating pieces. Returns
   `soma_to_piece_connectivity` with **positional** limb indices `0..N-1`. This matters: the concept
   network's limb nodes are named `L{j}` by that order, and a desync gives `KeyError: 'data'`.
4. `_decompose_limbs(branch_meshes, touching, params)` — calls `preprocess_limb` per limb:
   skeletonization (meshafterparty/MAP) → branches → `limb_correspondence` (branch_mesh,
   branch_skeleton, width_from_skeleton).
5. `_stitch_floating_pieces(...)` — stitches significant floating pieces onto the skeleton.
   ⚠️ **Carries the frame-desync bug — see §2.1.**
6. `_rebuild_limb_frames(...)` — the **fix for §2.1**: rebuilds a self-consistent frame for limbs whose
   partition stitching broke (`limb_mesh = combine_meshes(branches)`, contiguous `branch_face_idx`).
7. `_build_concept_networks(...)` — a concept network per (limb, soma).

**`preprocess_limb` — the parameter contract.** Signature
`preprocess_limb(mesh, neuron_params, limb_params, soma_touching_vertices_dict=None, ...)`:

- `neuron_params` — neuron-level, shared across limbs (width/size_threshold_MAP, axon_width_*,
  adaptive-invalidation, mp_only_*). Read by the helpers `_decide_next_limb_cfg` / `_cycle_for_something`.
- `limb_params` — parameters of one skeletonization pass (invalidation_d, smooth_neighborhood,
  combine/filter meshparty, use_meshafterparty).
- Other scalars come from `parameters.params` (config-driven) or are literals (former hardcoded
  defaults).

Its body is decomposed by phase: `_cycle_for_something` (MP skeletonization + adaptive invalidation_d),
`_decompose_map_piece` (the MAP piece, CGAL), `_fix_mp_soma_extension` (building the soma-extending
branches), and Parts 17–18 (`_merge_map_mp_correspondence`,
`_rearrange`/`_clean_network_starting_info`).

⚠️ **The MAP/stitching path is ACTIVE again** after the CGAL teasar skeletonizer was rebuilt
(`c9d3f7f`): thick branches (`width > width_threshold_MAP`) go through `_decompose_map_piece`, and
floating pieces through `_stitch_floating_pieces`. The path used to be dead (`NameError
calcification_param`) and was only covered statically — now it executes, and it is what exposed the
§2.1 bug. The function does **not** override the dataset config: the caller must set
`parameters.params.use("microns"|"h01")` beforehand.

⚠️ This is the heart of the core clump. `Branch.__init__` deepcopies the submesh/skeleton — expensive in
RAM, but risky to change. Skeletonization (meshparty) is a separate dependency, not meshlab.

### Invariants — do not change without an end-to-end gate

- The soma selection thresholds (`soma_width_threshold=0.32`, size thresholds) — tied to [0,1] SDF.
- `Branch`/`Limb`/`Neuron` and the concept network — everything rests on the graph structure and those
  attributes.
- The direction semantics of the concept network (upstream/downstream from the soma).

### 2.1 ⚠️ The stitching bug: `branch_face_idx` frame desync (class A) — partially fixed

**Symptom (widespread).** Many H01 neurons crashed with `IndexError: index N is out of bounds for axis
0 with size N` in `nru.apply_adaptive_mesh_correspondence_to_neuron`
([neuron_utils.py](neurd/neuron_utils.py), at `ex_limb.mesh.submesh([surround_mesh_faces])`).

**Root cause.** The invariant is that every branch's `branch_face_idx` must address the **stored limb
mesh** (`limb_meshes[limb_idx]`) as a clean partition. **Decomposition (`_decompose_limbs`) holds that
invariant.** **`_stitch_floating_pieces` breaks it:** it appends floating branches (from
`preprocess_limb(mesh=floating_piece)`) and cut branches (from
`correspondence_1_to_1(mesh=stitch_mesh)`) with `branch_face_idx` **in the frame of their own mesh**,
not remapped into the limb-mesh frame → indices go out of bounds and/or alias other faces. It is **not**
a watertightness issue, not an off-by-one, and not a MAP/MP combination — all three were checked and
rejected.

**The current fix (`50b769f`, step 6 above).** `_rebuild_limb_frames` runs after stitching: for a limb
whose partition is no longer clean, it rebuilds the frame — `limb_mesh = tu.combine_meshes(branches)`
with `branch_face_idx` as contiguous ranges (`combine_meshes` preserves order and face count even on
duplicates — verified). Only broken limbs are touched, so unstitched neurons are unaffected. **The crash
is gone** and those neurons segment (validated on `neuron_2889815798`, 7 limbs, no crash).

**🔧 What remains (deeper).** The fix makes the output *self-consistent* but does not cure the cause:
stitching produces **overlapping** branches, so the rebuilt limb mesh inflates (×1.8–12.7 on the test
neuron) and adaptive refinement on it is **skipped** (a working guard in `apply_adaptive_*`, because
gluing branches yields a disconnected mesh). Not a crash, but a quality cost: duplicate geometry and no
2-hop refinement. The right place to fix it is
`attach_floating_pieces_to_limb_correspondence` — remap `branch_face_idx` into the limb frame **at
insertion time** and dedupe the overlap, so that `_rebuild_limb_frames` becomes unnecessary. Note it is
*currently* necessary and cannot simply be deleted: it also **grows** the limb mesh to hold
floating-piece geometry that is genuinely absent from the original limb mesh (see MESH_OPS.md C1).
Repro: `tests/integration/reproduce_2889815798.py`.

**Related mode — class B, `missing labels was not resolved`** (also stitching, now guarded). When a
floating piece's junction lands in the **middle** of a main-limb branch, that branch is cut and
`correspondence_1_to_1(mesh=stitch_mesh)` re-partitions it into 2 pieces. On a degenerate cut (one half
gets no connected face patch) `resolve_empty_conflicting_face_labels` raises `missing labels was not
resolved` — which used to kill the whole neuron. **Fix:** the call is wrapped in try/except (the cut
path in `attach_floating_pieces_to_limb_correspondence`); on failure the piece is marked processed and
**skipped** (floating is best-effort) and the neuron completes. At the crash point
`limb_correspondence_cp` is not yet mutated, so skipping is safe. ⚠️ **Deeper:** the cause is the
degenerate/mid-branch cut; fix it where class A is fixed.

**Related mode — class C, `too many indices for array: array is 1-dimensional`** (also stitching, now
guarded). While decomposing a floating piece (`preprocess_limb(mesh=k)` in attach step 1), its MAP
skeletonization (`_decompose_map_piece` → CGAL/meshparty → `mesh_subtraction_by_skeleton`) can leave a
**degenerate/empty** leftover submesh (`faces` of shape `(0,)`/1-D), and trimesh fails on `faces[:, …]`
(deep inside mesh_tools, which we cannot durably edit). **Fix:** the `preprocess_limb` call in the
floating-piece decomposition loop is wrapped in try/except; a piece that fails to decompose is
**skipped** (not added to `floating_limbs_correspondence` — everything below is derived from it, so
indices stay consistent). ⚠️ **Deeper:** the same leftover can surface on a MAIN limb (in
`_decompose_limbs`), where skipping is not an option — that is still unguarded.

**Mode "torn limb" — `concept graph nodes != branches`** (not duplicates). On a neuron with **very
many** floating pieces, stitching piles dozens or hundreds of fragments onto a limb and **does not join
their skeletons** (limb 0: decomposition gave 23 connected branches → 217 after stitching, ~148 of them
torn, gaps in the thousands, 88 fully isolated). `branches_to_concept_network` reaches only the
connected component from the soma → `len(nodes) != len(branches)` → a crash at the very end (~2 h in).
**The right fix (substantial, NOT done):** before building the concept network, keep only the
soma-connected component in the limb and discard torn fragments (which would also remove
`_rebuild_limb_frames`'s inflation). ⚠️ **A fail-fast on floating-piece count was tried and REVERTED:**
the number of detached pieces does **not** predict the error — there are neurons with many pieces that
pass. The issue is skeleton connectivity after stitching, not the count. No cheap early predictor was
found, so the error stays late until the substantial fix lands.

---

## 3. The compat patches (numpy 2 / trimesh 4 / a dead meshlabserver)

All of them live in `neurd/` (never upstream). Most are monkeypatches in
[neurd/__init__.py](neurd/__init__.py), applied on `import neurd` (before `mesh_tools` is imported).
**This is load-bearing — understand a patch's purpose before editing it.** Each registers into
`neurd.PATCH_STATUS` and warns loudly if it fails to install; `tests/unit/test_perf_patches.py` asserts
none of them silently regressed.

| # | symptom | cause | fix (where) |
|---|---|---|---|
| 1 | `TypeError ... scalar index` (soma split) | trimesh≥4 `mesh.split()` returns a `list`, not an `ndarray` | `soma_extraction_utils.py`: `list(...)` + list comprehension |
| 2 | Poisson does not run, no output file | `meshlab.Poisson` writes `<xmlfilter>` XML that MeshLabServer 2020.09 ignores | `__init__.py`: **superseded** — `Poisson.__call__` → in-process no-op (output = input), since the filter was a no-op on this build anyway; a real Poisson sits behind `NEURD_REAL_POISSON=1` |
| 3 | `NameError: csm` (CGAL not installed) | the C++ `cgal_Segmentation_Module` is missing | **the real ext was restored** (`cgal/cgal_segmentation/`). `__init__.py` installs the `_cgal_segmentation.py` KMeans stub into `sys.modules` **only if** the .so is absent |
| 4 | `scipy ValueError: axis 0 index ... exceeds` | the MeshLabServer 2020.09 OFF exporter writes compacted vertices but leaves faces in the original numbering | `__init__.py`: patch `Meshlab.fetch_mesh_from_off` (searchsorted renumbering) |
| 5 | `ValueError: data_pts ... 2 dimensions` | newer pykdtree requires 2-D data, upstream builds trees from 1-D arrays | `__init__.py`: wrap `skeleton_utils.KDTree` (1-D → (N,1)) |
| 6 | `numpy_dep has no attribute 'in1d'` | `np.in1d` was removed in numpy 2 | `__init__.py`: `numpy.in1d = isin` + `numpy_dep.in1d = isin` |
| 7 | `NameError: calcification_param` (MAP path, thick branches) | the CGAL teasar skeletonizer (`calcification_param_Module`) was built inside Docker and vanished with it; `mesh_tools.skeleton_utils` imports it in a `try/except`, so the name is simply unbound | **not a stub but a real rebuild:** [cgal/cgal_skeleton_param/](cgal/cgal_skeleton_param/), sources from git, ported to CGAL 6 (C++17, `IO/OFF.h`, `CGAL::IO::read_OFF`). Built best-effort by `install_local.sh`. Without it, only neurons with thick branches fail |

There was also a historical guard for `IndexError ... size 1` in the multi-soma split
(`proofreading_utils.py`), in a module that has since been deleted.

**Perf patches** (not bug fixes — in-process replacements of expensive `mesh_tools` nodes; numbers and
rationale in [PERF.md](PERF.md)):

| node | was | now (`__init__.py`) | effect |
|---|---|---|---|
| `meshlab.Poisson` | xvfb+meshlabserver subprocess (filter was a no-op) | in-process no-op; real behind `NEURD_REAL_POISSON=1` | −536 s |
| `meshlab.FillHoles` | subprocess (also broken → no-op) | in-process pass-through | one less fork |
| `meshlab.Decimator` | subprocess + OFF round-trip | open3d in-process (`_mesh_ops.decimate`), cascaded halving | −184 s (−14%) |
| `tu.vertex_components` | networkx vertex-adjacency graph | owned scipy union-find (`submesh_ops`) | −205 s (−10%) |
| `nu.get_matching_vertices` | full N×N distance matrix | `cKDTree.query_pairs` | −~250 s, ~25 GB less churn |
| `su.compressed_pickle` | bz2 dump before a swallowed raise | no-op (`NEURD_ENABLE_COMPRESSED_PICKLE=1` restores) | −98 s |
| trimesh `mesh._cache` | accumulates | `_drop_trimesh_caches` after segmentation + `_clear_mesh_caches` on branches | lowers the plateau |
| soma→segment boundary | ~1.9 GB of garbage left resident | `_reclaim_memory` (`gc.collect` + `malloc_trim`) | **peak −33%** |

> Bug roots are in `git log` for the relevant commits; this table is the what/where, so `__init__.py`
> is readable. Patch #7 (the CGAL skeletonizer) is a separate C++ extension, not a monkeypatch — see
> [cgal/README.md](cgal/README.md).
> **Why perf fixes are monkeypatches rather than edits to mesh_tools:** `mesh_tools` is plain
> site-packages, so edits are lost on reinstall (this already happened to the `.so` and to
> skeleton_utils patches); and it is 116 functions / 16.7k lines of core algorithms — replacing it means
> rewriting NEURD. Intercepting nodes in-process is durable, reversible via an env flag, and
> fidelity-neutral. Full rationale in [PERF.md](PERF.md).
