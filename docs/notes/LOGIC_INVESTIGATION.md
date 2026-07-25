# preprocess_neuron.py — Logic Investigation (✅ SWEPT — status stub)

> **Status 2026-07-19→22:** Areas A–E swept with golden-verified instrumentation; the one
> actionable outcome (the mesh_tools adapter seam) is **BUILT** — see `neurd/_correspondence_backend.py`
> (commits `902d3be` + `42604cc`). No large safe *logic* rewrite was found; the value was ruling out
> tempting-but-unsafe changes **with measurement before code**. This file is now a findings record, not
> a forward plan. Detailed per-question logs (Q1–Q11) live in `git log` / the scratchpad harnesses named
> below; only durable conclusions are kept here.

## Two goals & verdict
1. **Minimize the unreliable `mesh_tools`** → the fragile surface is *small and isolable*: **~6 calls**
   in `cu`/`m_sk` (+1 `sk` CGAL); the 60+ `sk`/`tu` skeleton/mesh primitives are stable and NOT the
   problem. ✅ Isolated behind `_correspondence_backend.py`. (`mesh_tools` is a sibling package — we can
   only *isolate*, not edit it here.)
2. **Tame the re-call / branch / exception explosion** → measured: the suspected hotspots are mostly
   **non-problems** (below). No unbounded loops; most branching is load-bearing validation.

## Pipeline map (top-level call tree)
```
preprocess_neuron
├─ _extract_single_soma                (strict single-soma)
├─ _segment_limbs_from_soma            (soma-touching → candidate limb meshes)
├─ _decompose_limbs  → preprocess_limb  (PER LIMB — the hub, fans to 10 helpers)
├─ _stitch_floating_pieces → attach_floating_pieces_to_limb_correspondence
├─ _rebuild_limb_frames                (CLASS-A fix: contiguous mesh frame after stitching)
└─ _build_concept_networks → calculate_limb_concept_networks
```
`preprocess_limb` hub → `_cycle_for_something` (skeletonization ≤2 passes) · `_group_into_map_mp_sublimbs` ·
`_decompose_map_sublimbs` (CGAL) · `_decompose_mp_sublimbs` (meshparty) · `_merge_map_mp_correspondence` ·
`_stitch_map_and_mp` · `filter_limb_correspondence_for_end_nodes` · `calculate_limb_concept_networks`.

## Durable findings (measured on 1830470325 + 2451406889 + minnie65 864691135689048288)
- **The fragile mesh_tools surface (Area E contract):** `cu.mesh_correspondence_adaptive_distance` (×2),
  `cu.resolve_empty_conflicting_face_labels` (×2), `cu.waterfill_starting_label_to_soma_border`,
  `m_sk.skeletonize_mesh_largest_component`, `sk.skeletonize_and_clean_connected_branch_CGAL`. **The
  ONLY pipeline non-determinism** is the waterfill `np.random.choice` tie-break at
  `compartment_utils:1187` (inside `resolve_empty`). All now wrapped by `_correspondence_backend.py`
  (the rng seam lives there). Failure classes: **A** post-stitch frame desync (→ `_rebuild_limb_frames`),
  **B** cut-branch re-correspondence fails (rate ~1–3 %, real), **C** degenerate CGAL submesh (skip).
  *Addendum:* the seam has since grown a second job — it also gates the owned correspondence/width
  kernel (`NEURD_OWNED_SKELETAL_DISTANCE` → `neurd/skeletal_distance_ops.py`, off by default), and the
  `cu.skeletal_distance` algorithm was replaced with precomputed subgraph components.
- **Correspondence re-call (Area A):** ~25 runs/neuron, **~92 % from floating-piece stitching**, not the
  main MAP/MP decomposition. The inner adaptive re-run fires **100 %** (unavoidable, inside the dep); the
  **outer `first_pass` double-call fires 0/341 branches** across 3 neurons/2 datasets — dead **by
  geometry** (a corresponded skeleton is always ⊂ its mesh, so `get_skeletal_distance` never empties).
  Safe to simplify, but keep the net: prefer `base_kwargs` dedup over deletion (guards a future
  foreign-mesh caller). *Not yet done — golden-gated logic decision for later.*
- **Skeletonization "cycle" (Area B):** NOT a loop — `_cycle_for_something` is a fixed **≤2-pass**
  adaptive, always terminates. No re-call problem.
- **Concept network (Area D):** `calculate_limb_concept_networks`'s branch-density is a single build
  (`nru.branches_to_concept_network`) + a switchable **4-check validation battery** (the 4 `raise`s), not
  tangled control-flow. Construction complexity lives in `neuron_utils`, out of this file's scope.
- **Floating-stitch waste (Area C):** 128 pieces decomposed on big-H01, **~40 % dropped as too-far** =
  wasted `preprocess_limb`. A cheap **mesh-proximity connectivity pre-filter** was proven **safe on all 3
  neurons** (0 false-prune, surface-dist ≤ skeleton-dist superset guarantee) **but savings DON'T
  generalize** — 40 % on thin-piece H01, **0 % on fat-piece minnie65** (surface-vs-skeleton gap
  over-includes at the safe threshold). **SHELVED**; would need a cheap centerline proxy (PCA/medial-axis)
  to be worth revisiting. The O(N²·L) selection-loop smell is real but 1.5 % of time — ignore.

## Net
No large safe logic win exists to take blindly. The realized win is the **isolation seam**
(`_correspondence_backend.py`). Remaining *optional* leads, all golden-gated: collapse the outer
double-call (via `base_kwargs`), and — only with a new cheap-centerline idea — the connectivity pre-filter.

## Harnesses (in `git log` / scratchpad, recreate from memory)
`instrument_areaA[_generic].py` (correspondence call counts), `instrument_areaC[_Q7].py` +
`instrument_areaC_Q9_capture[_generic].py` + `q9_analyze.py` (floating funnel + pre-filter gating). All
golden-verified against small-h01 + `1830470325`.
