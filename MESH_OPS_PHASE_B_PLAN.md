# Phase B — thread a carried `SubMesh` through correspondence (executable handoff)

Self-contained plan for a **future focused session**. Companion: `MESH_OPS_REBUILD_PLAN.md` (Phase 0),
`MESH_OPERATIONS_LAYER.md`, and the finished Phase-A plan `~/.claude/plans/unified-tinkering-ocean.md`.
Branch: `refactor/simplify-preprocess-logic`.

## Status going in
- **Phase 0 + Phase 1 (split cluster) + Plan A DONE** — `neurd/submesh_ops.py` exists with `SubMesh`,
  `connected_face_components`, `split`/`split_significant`/`largest_component`/`components_from_face_idx`/
  `original_faces`/`faces_by_match`, all `tu`-equivalent (unit-tested), and every gateable L2 call site in
  pipeline code migrated + golden-gated byte-identical. Commits: `e04fdb8 … 422710b`.
- **`SubMesh` composition law is proven** (`test_composition_law`: `from_mesh(m).sub(A).sub(B).root_face_idx()
  == A[B]`). That law is the entire engine of Phase B.
- **User allows committing directly in `preprocess_neuron.py`** (no stash-dance).

## Why B (the real payoff, no small slice)
B eliminates three coupled fragilities that all stem from `branch_face_idx` being carried as **raw index
arrays** whose frame is implicit:
1. **4 manual remaps** `parent_idx[branch_face_idx]` (rebasing a branch's faces from the correspondence-input
   mesh's frame to the limb frame) — `preprocess_neuron.py:1399, 1526, 2281, 2340`.
2. **The frame-desync bug class** — `_stitch_floating_pieces` leaves some branches' `branch_face_idx` in a
   FOREIGN mesh frame; repaired post-hoc by `_rebuild_limb_frames` (`preprocess_neuron.py:2969`) and
   hand-guarded by `neuron_utils.py:823-829` ("mesh_face_idx has desynced from the limb mesh").
3. **The limb-frame KDTree fallback** `neuron.py:2318` (`tu.original_mesh_faces_map(mesh, curr_limb)`) —
   regenerates limb face-idx geometrically because preprocess doesn't supply it. (Note: the SOMA fallback
   `neuron.py:2268` stays KDTree — soma comes from poisson, no provenance; that is Plan A's domain, not B.)

A carried `SubMesh(branch_mesh, face_idx, parent)` makes the frame **explicit and composing**: the remap is
`sub.root_face_idx()` by construction, a wrong-frame branch is a type/assert error instead of a silent
IndexError, and limb face-idx falls out of provenance. **Wrapping just the 4 remaps is cosmetic** (they are
already correct O(1) ops) — the value only lands when the `SubMesh` is *carried* through the dict and into
`Branch`.

## Current data flow (recon — verified 2026-07-20)

### The correspondence dict
`correspondence_1_to_1` (`preprocess_neuron.py:198`, returns at ~`:316`) and its wrapper
`_run_mesh_correspondence` (`:318`) return a dict `{key: entry}` where each `entry` holds:
`{"branch_mesh", "branch_face_idx", "branch_skeleton"}` (+ later `"width_from_skeleton"`).
- `branch_mesh = divided_submeshes[k]`, `branch_face_idx = divided_submeshes_idx[k]` (set at `:307-308`),
  where `divided_submeshes,divided_submeshes_idx = tu.split_mesh_into_face_groups(corr_input_mesh, coloring)`
  (`:301`, also `:1047`). So `branch_face_idx` indexes the **correspondence-input mesh**, not the limb.

### The 4 remaps (input-mesh frame → limb frame), each `parent_idx[branch_face_idx]`
| site | `parent_idx` origin | context |
|---|---|---|
| `:1399` | `mesh_idx` (fn param — sublimb faces in limb) | MAP sublimb, after `correspondence_1_to_1(mesh=mesh)` |
| `:1526` | `extend_soma_mesh_idx = np.concatenate(divided_submeshes_idx[match_sk_branches])` (`:1509-1510`) | MP soma-extension, after `_run_mesh_correspondence(extend_soma_mesh, …)` |
| `:2281` | `pre_stitch_mesh_idx = curr_MAP_meshes_idx[0]` (`:2272`); `pre_stitch_mesh = limb.submesh([pre_stitch_mesh_idx])` (`:2273`) — **literally a SubMesh** | MAP stitch |
| `:2340` | `stitching_mesh_idx = np.concatenate(curr_MAP_meshes_idx + curr_MP_meshes_idx)` (`:2306`) | MP stitch |
Downstream the remapped arrays feed `curr_MAP_meshes_idx`/`curr_MP_meshes_idx` (`:2264/2283/2303`) and
ultimately the per-limb `branch_meshes` → `Branch.mesh_face_idx`.

### `Branch.mesh_face_idx`
`neuron.py:86` (class), `__init__` stores `self.mesh_face_idx = mesh_face_idx` (`:200`). Surface: **42 refs**
(`neuron.py` 29 / `neuron_utils.py` 13), **16 writes**. Consumed as **limb-frame** indices, e.g.
`ex_limb.mesh.submesh([mesh_face_idx])` (`neuron_utils.py:815/831/861`). This is where the frame must be
correct.

### Stitch + the frame-desync repair
`_stitch_floating_pieces` (`:2934`) → `_rebuild_limb_frames` (`:2969`) at `:3072/3083`. `_rebuild_limb_frames`
rebuilds each inconsistent limb mesh as `tu.combine_meshes(branch_meshes)` with `branch_face_idx` = contiguous
`[offset, offset+n)` ranges. Only stitched limbs are rebuilt (clean limbs byte-unaffected).

### Nondeterminism (gating-critical)
The ONLY pipeline nondeterminism is `np.random.choice` in waterfill
(`mesh_tools/compartment_utils.py:1187`, inside `resolve_empty_conflicting_face_labels`). The gate harness
already `np.random.seed(0)`s, so it is deterministic **provided a change does not shift the RNG-draw
sequence**. Reordering branch/piece iteration CAN shift it → then use RNG-capture replay (recipe in memory
`preprocess-limb-refactor`: capture the draws at the stitch, replay them).

## The `SubMesh` threading design
Make the correspondence-input mesh a `SubMesh` of the limb, and have the correspondence build each branch by
`.sub(...)`:
```python
# where today:  divided_submeshes_idx[k]  (into corr_input_mesh)  +  parent_idx[...] remap
# instead:      corr_input = SubMesh(corr_input_mesh, parent_idx, limb_mesh)   # parent linkage explicit
#               entry["branch"] = corr_input.sub(divided_submeshes_idx[k])     # a SubMesh
#               entry["branch_face_idx"] = entry["branch"].root_face_idx()     # == parent_idx[idx], no manual remap
```
`root_face_idx()` folds the chain (proven), so the 4 remaps vanish. Endgame: the dict carries one `SubMesh`
per branch instead of separate `branch_mesh`+`branch_face_idx`; `Branch` stores it; `mesh_face_idx` becomes
`branch.root_face_idx()` → always limb-consistent → the guard (`neuron_utils:823`) and `_rebuild_limb_frames`
become dead/removable, and the limb KDTree fallback (`neuron.py:2318`) is fed from provenance.

**⚠️ Concatenated-parent contract (`:1526`, `:2340`).** Two of the four remaps have
`parent_idx = np.concatenate(divided_submeshes_idx[...])` — the parent frame is itself an *assembled*
index array, not a clean submesh extraction. `SubMesh(corr_input_mesh, parent_idx, limb_mesh)` is only
valid when `corr_input_mesh == limb_mesh.submesh([parent_idx])` and `len(parent_idx) == len(corr_input_mesh.faces)`.
For the concatenated frames that is NOT free — add an explicit assert (no overlap, no reorder) when the
`SubMesh` is built (step 1), or `root_face_idx()` will diverge from the old `parent_idx[branch_face_idx]`.

## Incremental task list (each its own golden-gated commit; STOP-and-review between)
0. **Coverage probe first.** Add the 4 remap sites + `_rebuild_limb_frames` + `neuron.py:2318` to a coverage
   probe (edit `scratchpad/gate_run.py` — it already monkeypatches; wrap the enclosing fns or add temporary
   counters). Run small-h01 + big `1830470325` to learn **which remaps actually fire**. ⚠️ MAP/stitch is
   partly CGAL-gated and the frame-desync only triggers on stitching neurons — the 2 current anchors may NOT
   exercise `:2281/:2340` or `_rebuild_limb_frames`. If so, add a **stitching** anchor (memory notes
   `neuron_2889815798` builds via the stitch path, ~66.7min) and its own baseline before touching stitch code.

   **The stitch/floating path IS in scope — full B (incl. steps 3/5) is the plan of record.** Even though
   the target is single-soma neurons, real target meshes have **torn / fragmented limbs** (some limbs arrive
   in several disconnected pieces), so `_stitch_floating_pieces` fires and the frame-desync class (fragility
   #2) is live in production — do NOT descope stitch. The probe's second purpose is therefore **not** a
   go/no-go on 3/5 but a **readiness check for them**: confirm which remaps fire on the current anchors, and
   confirm the **stitching anchor** (`neuron_2889815798`, ~66.7min) actually exercises `:2281/:2340` +
   `_rebuild_limb_frames`, then baseline it — before touching stitch code in step 3. (The fragilities do
   decompose — #1/#3 are killed without stitch — but that is a sequencing aid, not a reason to stop at a
   "minimal B"; steps 3/5 are required for this workload.)
1. **Non-stitch remap → composition (smallest real step):** pick a remap that fires on the current anchors
   (likely `:1399` MAP-sublimb and/or `:1526` MP-extension). Build a `SubMesh` at the correspondence-input
   mesh and replace `parent_idx[branch_face_idx]` with `sub.root_face_idx()`. `branch_face_idx` values are
   unchanged → **byte-identical**. Gate small+big. One commit per remap.
2. **Carry the `SubMesh` in the dict:** store `entry["branch"] = SubMesh(...)` alongside (then instead of)
   `branch_mesh`+`branch_face_idx`; update the readers (`branch_mesh` 11 sites, `branch_face_idx` 14 sites —
   grep `\["branch_mesh"\]` / `\["branch_face_idx"\]`) to `.mesh` / `.root_face_idx()`. Gate.
3. **Stitch remaps (`:2281/:2340`) via `SubMesh`** — needs the stitching anchor + RNG-replay gating. Replace
   `_rebuild_limb_frames`'s manual re-frame with `SubMesh` parent-relinking (the combined limb mesh becomes
   the new parent; branches are `.sub(range)` of it). Prove the guard `neuron_utils:823` can no longer fire.
   ⚠️ **The `:2340` remap sits inside a bare `except:` (`preprocess_neuron.py:2341`) that silently reverts to
   the original mesh assignments on ANY error.** A subtle SubMesh-threading exception is therefore swallowed
   and the run falls into the fallback — the gate catches the resulting diff, but diagnosis is blind. For the
   duration of step 3, temporarily re-raise / log inside that `except` so a threading bug surfaces at source.
4. **`Branch` stores `SubMesh`:** `mesh_face_idx` → `@property` returning `self._sub.root_face_idx()`; update
   the 16 writers + 42 refs incrementally (keep `mesh_face_idx` read-compatible). Gate at each sub-step.
   ⚠️ **Read-compat is not enough — resolve the write side first.** `mesh_face_idx` has **16 writers**,
   including reassignments (`neuron_utils.py:864, 920`), and the name is **shared with `Soma`**
   (`neuron.py:1954`), not just `Branch`. A bare `@property` breaks the writes; a setter that rebuilds a
   `SubMesh` from a raw array loses the parent linkage and defeats the whole point. Decide up front: migrate
   the 16 writers to write a `SubMesh`, or add a transitional "detached" setter (`parent = limb.mesh`) as a
   bridge. **Scope the property to `Branch` only** — leave `Soma.mesh_face_idx` a plain attribute (its idx
   is the soma KDTree, Plan A's domain, no provenance to carry).
5. **Delete the dead scaffolding:** remove `_rebuild_limb_frames` + the `neuron_utils:823` guard (now
   structurally impossible), and feed `neuron.py:2318` limb face-idx from provenance so its KDTree fallback
   dies. Gate; confirm `tu.original_mesh_faces_map` is gone from all non-soma sites.
   ⚠️ **`_rebuild_limb_frames` also skips refinement on rebuilt limbs as a side effect** (memory
   `class-a-frame-desync`). Once `SubMesh` makes frames correct by construction and this step deletes the
   rebuild, refinement **re-activates** on those limbs → a byte-diff even when the fix is correct. Separate
   the two concerns: (a) frame correctness (the goal), (b) refinement behavior (an accidental side effect) —
   otherwise the gate reddens for the right reason and blocks the deletion.

## Risks / notes
- **Highest blast radius in the repo.** `neuron.py`/`neuron_utils.py` are consumed pipeline-wide; do steps 2/4
  in small sub-commits, each gated.
- **Stitch is fragile + nondeterministic** — never touch `:2281/:2340`/`_rebuild_limb_frames` without a
  stitching anchor and RNG-replay; a reorder that shifts the waterfill RNG stream will diff even when logically
  equivalent.
- **`_rebuild_limb_frames` is load-bearing today** — do not delete (step 5) until step 3/4 make the frame
  correct by construction, verified on the stitching anchor.
- Steps 1–2 are byte-identical index-preserving refactors (low risk); steps 3–5 are the invasive core.
- **Half-migration risk.** Steps 1–2 are cosmetic *on their own* — the payoff only lands at steps 3–5. If
  3–5 stall, the carried `SubMesh` becomes pure overhead with no fragility removed. Mitigation: don't commit
  1–2 without a committed intent to reach step 3, **and** give 1–2 standalone value via the in-process
  invariant assert (see Verification) so they at least harden the existing code while 3–5 is pending.

## Verification (unchanged harness)
`scratchpad/gate_run.py` (seeded `np.random.seed(0)` + `OMP/MKL/OPENBLAS/VECLIB/NUMEXPR_NUM_THREADS=1`) →
`segmentation_pipeline` → `tests/tools/neuron_metrics.extract_metrics`, byte-identical vs baselines:
small-h01 `366280.26`/`291893`; big `1830470325` `1648072.07`/`1323533`; + a stitching anchor
(`neuron_2889815798`) baseline for steps 3+. Plus `pytest tests/unit/test_submesh_ops.py -v`. Stitch-touching
steps additionally gated by RNG-capture replay.

**Belt-and-suspenders — in-process invariant (steps 1–3).** Byte-identity of final metrics only catches a
frame error *downstream*, far from the cause. Cheaply catch it *at source*: while both paths coexist, keep a
debug-assert `np.array_equal(sub.root_face_idx(), parent_idx[branch_face_idx])` right where the old remap
runs. It fails exactly where a threading bug is introduced instead of as an opaque metric diff, and gives
steps 1–2 standalone value (they harden the current code, not just prepare for step 3). Remove the asserts
once the `SubMesh` path is the sole path and stable.

## Key references
| what | where |
|---|---|
| owned `SubMesh` + composition | `neurd/submesh_ops.py`; law `tests/unit/test_submesh_ops.py::test_composition_law` |
| correspondence dict build | `preprocess_neuron.py:198` (`correspondence_1_to_1`), `:318` (`_run_mesh_correspondence`), set `:307-308` |
| 4 manual remaps | `preprocess_neuron.py:1399, 1526, 2281, 2340` |
| stitch + reframe | `preprocess_neuron.py:2934` (`_stitch_floating_pieces`), `:2969` (`_rebuild_limb_frames`), `:3072/3083` |
| frame-desync guard | `neuron_utils.py:823-829` |
| `Branch.mesh_face_idx` | `neuron.py:86/200`; 42 refs / 16 writes across `neuron.py`+`neuron_utils.py` |
| limb/soma KDTree fallbacks | `neuron.py:2318` (limb — B kills it) / `neuron.py:2268` (soma — stays, Plan A domain) |
| nondeterminism | `mesh_tools/compartment_utils.py:1187` (`np.random.choice` waterfill) |
