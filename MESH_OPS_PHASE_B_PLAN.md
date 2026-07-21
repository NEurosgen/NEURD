# Phase B — carry provenance through correspondence (status + remaining work)

Companions: `MESH_OPS_REBUILD_PLAN.md` (Phase 0), `MESH_OPERATIONS_LAYER.md`.
Branch: `refactor/simplify-preprocess-logic`.

> **This file was rewritten 2026-07-21 after executing most of it.** The original was a forward plan;
> several of its assumptions turned out to be wrong in ways that matter (see *Corrections* below). Line
> numbers from the original are gone — they shifted twice during execution. **Reference code by function
> name and grep for it**; line numbers here are hints only.

## What Phase B was for
Three coupled fragilities, all from `branch_face_idx` being carried as **raw index arrays** whose frame is
implicit:
1. **4 manual remaps** `parent_idx[branch_face_idx]` — rebasing a branch's faces from the
   correspondence-input mesh's frame to the limb frame.
2. **The frame-desync bug class** — `_stitch_floating_pieces` leaves some branches' `branch_face_idx` in a
   FOREIGN mesh frame; repaired post-hoc by `_rebuild_limb_frames` and hand-guarded in `neuron_utils`
   ("mesh_face_idx has desynced from the limb mesh").
3. **The limb-frame KDTree fallback** in `neuron.py` — regenerated limb face-idx geometrically because
   preprocess never supplied it.

## Status

| Fragility | State |
|---|---|
| #1 — the 4 manual remaps | ✅ **2 of 4 consolidated into one seam**; the 2 stitch ones remain (low value, see step 3) |
| #2 — frame desync | ⚠️ **only partially eliminable — see Correction C1** |
| #3 — limb/soma KDTree fallback | ✅ **DONE** (`63d2584`) — and it also killed the soma one |

**Done, each byte-identical on all 3 anchors:**
- `7291bf8` — first migration: MP soma-extension remap → `SubMesh` composition at the call site.
- `5555c1e` — bundled the MAP parallel arrays (`mesh_pieces_for_MAP` + `..._face_idx`) into one
  `list[SubMesh]` (`map_subs`); migrated the MAP-sublimb remap.
- `a9bb9af` — **the seam**: optional `input_sub` on `correspondence_1_to_1`; both non-stitch remaps moved
  INTO it.
- `b61d6b4` — DRY'd the MP/MAP sublimb-assembly twins into `_assemble_sublimbs`.
- `63d2584` — preprocess supplies `limb_mehses_face_idx` + `soma_meshes_face_idx` from provenance →
  both `neuron.py` KDTree fallbacks stop firing.

Later, outside Phase B proper but on the same provenance line: `ad02e00`/`b17bf5d`
(`face_groups_by_label` + `split_into_face_groups`), `32b01eb`/`37d8e03` (`combine` with provenance →
`_rebuild_limb_frames`'s manual offset loop gone), `f8ee2ed` (feed `mesh_pieces_connectivity` face indices).

## Corrections to the original plan (read these before continuing)

**C1 — `_rebuild_limb_frames` and the `neuron_utils` guard CANNOT be deleted.** The original step 5 was
wrong. `_rebuild_limb_frames` does *two* jobs: (a) reindex branch_face_idx to a clean frame, and (b) **grow
the limb mesh** (`limb_meshes[li] = combine(branch meshes)`) so it contains the floating-piece geometry that
is genuinely NOT in the original limb mesh. SubMesh provenance can express (a) but never (b) — there is no
index into the original limb for faces that were never in it. So fragility #2 is only partly removable, and
the guard stays as a cheap invariant.

**C2 — the dict readers never needed changing.** The original step 2 planned to store a `SubMesh` in the
correspondence dict and update ~25 readers (`branch_mesh` / `branch_face_idx`). What actually worked is
smaller: `correspondence_1_to_1` gained an optional `input_sub`, and when it is supplied the branch
`branch_face_idx` comes out **already in the limb frame**. It stays a plain array, so **no reader changed at
all**. Prefer this shape for anything remaining.

**C3 — "wrapping the remaps is cosmetic" was half right.** Wrapping at the *call site* is indeed cosmetic;
moving the remap *into the seam* is not — it deletes the frame special-case and is where the stitch-path
desync could also be fixed at source.

**C4 — coverage before design.** Two planned pieces turned out to target code that never runs (see step 3
and the *Dropped* section). **Probe what actually executes, and from where, before designing a replacement.**
The harness prints caller histograms for exactly this.

## Remaining work

**Step 3 — stitch remaps into the seam. ⏸ LOW VALUE, optional.**
The two stitch remaps live in `_recorrespond_stitch` (MP-stitch, and the MAP-stitch one behind `cut_flag`).
Analysis: their output is **overwritten by `_rebuild_limb_frames`** on most stitched limbs (it recomputes
branch_face_idx from `branch_mesh`, ignoring the old value, and it fires on nearly every stitched limb —
big 2/3, stitch 6/7), so migrating them is byte-identical but mostly cosmetic. Worse, the **MAP-stitch
remap fires on NO anchor** (`cut_flag` = the stitch cut a MAP branch mid-skeleton, rare geometry), so it
cannot be golden-gated at all — it would rely on the composition law plus the seam's precondition assert.
⚠️ Both sit inside a bare `except:` that silently reverts to the original assignments on ANY error; if this
is ever attempted, temporarily re-raise there or a threading bug will be diagnosed blind.

**Step 4 — `Branch` stores a `SubMesh`. ⏸ The last real Phase-B item.**
`mesh_face_idx` → `@property` returning `root_face_idx()`. Smaller than originally feared (C2: readers are
fine), but the **write side is the problem**: ~16 writers including reassignments in `neuron_utils`, and the
attribute name is **shared with `Soma`**. A bare `@property` breaks the writes; a setter that rebuilds a
`SubMesh` from a raw array loses the parent linkage and defeats the point. Decide up front: migrate the
writers, or add a transitional "detached" setter. **Scope it to `Branch`** — leave `Soma.mesh_face_idx` a
plain attribute (its idx is geometric by nature).

**~~Step 5~~ — retracted.** The `neuron.py` fallback half is DONE (`63d2584`); the scaffolding-deletion half
is impossible (C1).

## Dropped after probing (do not revive without a new driver)
`original_mesh_faces_map`, `subtract_mesh` and `compare_meshes_by_face_midpoints` have **zero neurd callers**
during `segmentation_pipeline` on either anchor. The spine sites are on a different entry point, and
`compare_meshes_by_face_midpoints` is only used in `__eq__`. Owning them would be work in the drawer, and
ungateable. Reviving this needs a **spine-path harness** first — which would also unblock `spine_utils`
(~51 `tu.*` calls, about a third of the remaining mesh_tools surface, and the single biggest holdout).

Note the geometric matching that *does* still run is one level down, inside `mesh_tools` functions called by
warm sites: `remove_mesh_interior` (the bulk) and `mesh_list_connectivity`. Neither is reachable by handing
in provenance — `remove_mesh_interior` remaps pieces it creates internally, and `mesh_list_connectivity`'s
pieces there are poisson/SDF soma candidates that have no provenance in principle. Removing those means
owning the operations outright (soma path = Plan A's domain).

## Nondeterminism (gating-critical, unchanged)
The ONLY pipeline nondeterminism is `np.random.choice` in waterfill
(`mesh_tools/compartment_utils.py`, inside `resolve_empty_conflicting_face_labels`). The harness seeds
`np.random.seed(0)`, so runs are deterministic **provided a change does not shift the RNG-draw sequence**.
Reordering branch/piece iteration CAN shift it → then RNG-capture replay is needed. None of the work above
required it (no iteration order changed).

## Verification
`scratchpad/gate_run.py` (NOT committed — **it has been wiped by the system once; consider committing it to
`tests/tools/`**). Seeded + thread-pinned → `segmentation_pipeline` → `tests/tools/neuron_metrics.extract_metrics`.

Anchors, byte-identical (skeleton_length_total / branch_mesh_faces_total):
- small-h01 `neuron_2530864375` — `366280.26` / `291893` (1 soma, 5 limbs, 28 br)
- big `1830470325` — `1648072.07` / `1323533` (3 limbs, 129 br)
- stitch `neuron_2889815798` — `6819002.31` / `6259362` (7 limbs, 186 br)

Two harness features added during execution, both worth keeping:
- **`face_idx_digests`** — per-limb/soma `mesh_face_idx` sorted-set digests. `extract_metrics` does NOT
  cover the level-2 (limb/soma ↔ original mesh) map, so without this the gate is **blind** to changes there.
- **caller histograms** (`_caller_probe(tu, name)`) — records which call sites actually fire, with live line
  numbers. Robust to line shifts, unlike a line-number coverage probe. Use it *before* designing (C4).

Plus `pytest tests/unit/test_submesh_ops.py` (36 tests: `tu`-equivalence for the whole owned layer).

## Key references (by name — grep, don't trust line numbers)
| what | where |
|---|---|
| owned layer | `neurd/submesh_ops.py` — `SubMesh`, `split*`, `face_groups_by_label`, `split_into_face_groups`, `combine`, `faces_by_match` |
| composition law | `tests/unit/test_submesh_ops.py::test_composition_law` |
| the seam | `correspondence_1_to_1(..., input_sub=)` in `preprocess_neuron.py`; wrapper `_run_mesh_correspondence` forwards it via `**kwargs` |
| remaining remaps | `_recorrespond_stitch` (MP-stitch + the `cut_flag` MAP-stitch branch) |
| stitch + reframe | `_stitch_floating_pieces` → `_rebuild_limb_frames` |
| level-2 provenance | `_segment_limbs_from_soma` returns `branch_meshes_orig_idx` + `soma_faces_idx`; `preprocess_neuron` builds `limb_mehses_face_idx` |
| `Branch.mesh_face_idx` | `neuron.py` `Branch.__init__`; ~42 refs / 16 writes across `neuron.py` + `neuron_utils.py` |
