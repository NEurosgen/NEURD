# Phase B — carry provenance through correspondence (✅ mostly DONE — status stub)

Companions: `MESH_OPS_REBUILD_PLAN.md` (Phase 0/1), `MESH_OPERATIONS_LAYER.md`.
Reference code **by function name — grep, don't trust line numbers** (they shifted twice during execution).

## What Phase B addressed
Three coupled fragilities, all from `branch_face_idx` being carried as **raw index arrays** whose frame is
implicit: (1) 4 manual remaps `parent_idx[branch_face_idx]`; (2) the **frame-desync bug class**
(`_stitch_floating_pieces` leaves branches in a FOREIGN mesh frame); (3) the limb-frame KDTree fallback in
`neuron.py` (regenerated because preprocess never supplied it).

## Status
| Fragility | State |
|---|---|
| #1 — 4 manual remaps | ✅ 2 of 4 consolidated into one seam; the 2 stitch ones remain (low value — Step 3) |
| #2 — frame desync | ⚠️ only **partially** eliminable (see C1) |
| #3 — limb/soma KDTree fallback | ✅ DONE (`63d2584`) — also killed the soma one |

**Done, each byte-identical on all 3 anchors:** `7291bf8` (MP soma-extension remap → `SubMesh`),
`5555c1e` (MAP parallel arrays → one `list[SubMesh]` `map_subs`), **`a9bb9af` — the seam** (optional
`input_sub` on `correspondence_1_to_1`; both non-stitch remaps moved into it), `b61d6b4` (`_assemble_sublimbs`
DRY), `63d2584` (preprocess supplies `limb/soma_meshes_face_idx` → both `neuron.py` KDTree fallbacks stop
firing). Same provenance line, later: `ad02e00`/`b17bf5d` (`face_groups_by_label` + `split_into_face_groups`),
`32b01eb`/`37d8e03` (`combine` with provenance → `_rebuild_limb_frames`'s manual offset loop gone),
`f8ee2ed` (feed `mesh_pieces_connectivity` face indices).

## Durable corrections (learned by executing)
- **C1 — `_rebuild_limb_frames` + the `neuron_utils` guard CANNOT be deleted.** It does *two* jobs:
  (a) reindex `branch_face_idx` to a clean frame, and (b) **grow the limb mesh** to contain floating-piece
  geometry that is genuinely NOT in the original limb mesh. SubMesh provenance can express (a) but never
  (b) — no index exists into the original limb for faces never in it. So #2 is only partly removable; the
  guard stays as a cheap invariant.
- **C2 — the preferred seam shape leaves readers untouched.** `correspondence_1_to_1(input_sub=...)`
  returns `branch_face_idx` **already in the limb frame**, but still a plain array → **no reader changed**
  (~25 `branch_mesh`/`branch_face_idx` readers). Prefer this over storing a `SubMesh` in the dict.
- **C4 — coverage before design.** Two originally-planned pieces targeted code that never runs. Probe what
  actually executes (the harness prints caller histograms) *before* designing a replacement.

## Remaining work (both optional, low value)
- **Step 3 — stitch remaps into the seam. ⏸** The two live in `_recorrespond_stitch` (MP-stitch + the
  `cut_flag` MAP-stitch). Migrating is byte-identical but mostly cosmetic: their output is **overwritten by
  `_rebuild_limb_frames`** on nearly every stitched limb, and the MAP-stitch remap fires on **NO anchor**
  (rare `cut_flag` geometry) so it can't be golden-gated. ⚠️ Both sit inside a bare `except:` that silently
  reverts on ANY error — if attempted, temporarily re-raise there or a threading bug diagnoses blind.
- **Step 4 — `Branch` stores a `SubMesh`. ⏸ the last real Phase-B item.** `mesh_face_idx` → `@property`
  returning `root_face_idx()`. Readers are fine (C2); the **write side** is the problem (~16 writers incl.
  `neuron_utils` reassignments, and the name is **shared with `Soma`**). Decide up front: migrate the
  writers, or add a transitional "detached" setter. **Scope to `Branch`** — leave `Soma.mesh_face_idx` a
  plain attribute (geometric by nature).

## Dropped after probing (don't revive without a new driver)
`original_mesh_faces_map`, `subtract_mesh`, `compare_meshes_by_face_midpoints` have **zero neurd callers**
during `segmentation_pipeline` (spine sites are a different entry point; the last is only used in `__eq__`).
The geometric matching that *does* still run is one level down inside `mesh_tools` (`remove_mesh_interior`,
`mesh_list_connectivity`) — not reachable by handing in provenance (they remap internally / operate on
poisson-SDF soma candidates with no provenance in principle). Reviving needs a **spine-path harness** first
(would also unblock `spine_utils`, ~51 `tu.*` calls — the biggest holdout; see `SPINE_UTILS_PLAN.md`).

## Nondeterminism (gating-critical)
The ONLY pipeline nondeterminism is `np.random.choice` in waterfill (`compartment_utils`, inside
`resolve_empty_conflicting_face_labels`) — now behind `_correspondence_backend.py`'s rng seam. The harness
seeds `np.random.seed(0)`; runs are deterministic **provided a change doesn't shift the RNG-draw sequence**
(reordering branch/piece iteration does → RNG-capture replay needed). None of the work above required it.

## Verification
Seeded + thread-pinned → `segmentation_pipeline` → `tests/tools/neuron_metrics.extract_metrics`. Two harness
features worth keeping: **`face_idx_digests`** (per-limb/soma `mesh_face_idx` set digests — `extract_metrics`
does NOT cover the L2 map, so without this the gate is blind there) and **caller histograms** (`_caller_probe`,
robust to line shifts — use before designing, per C4). Plus `pytest tests/unit/test_submesh_ops.py`.

Anchors, byte-identical (skeleton_length / branch_faces): small-h01 `2530864375` `366280.26`/`291893`;
big `1830470325` `1648072.07`/`1323533`; stitch `2889815798` `6819002.31`/`6259362`.

## Key references (grep by name)
| what | where |
|---|---|
| owned layer | `neurd/submesh_ops.py` — `SubMesh`, `split*`, `face_groups_by_label`, `split_into_face_groups`, `combine`, `faces_by_match` |
| composition law | `tests/unit/test_submesh_ops.py::test_composition_law` |
| the seam | `correspondence_1_to_1(..., input_sub=)` in `preprocess_neuron.py`; `_run_mesh_correspondence` forwards via `**kwargs` |
| remaining remaps | `_recorrespond_stitch` (MP-stitch + `cut_flag` MAP-stitch) |
| stitch + reframe | `_stitch_floating_pieces` → `_rebuild_limb_frames` |
| L2 provenance | `_segment_limbs_from_soma` returns `branch_meshes_orig_idx` + `soma_faces_idx`; `preprocess_neuron` builds `limb_mehses_face_idx` |
| `Branch.mesh_face_idx` | `neuron.py` `Branch.__init__`; ~42 refs / 16 writes across `neuron.py` + `neuron_utils.py` |
