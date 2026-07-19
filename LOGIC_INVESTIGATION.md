# preprocess_neuron.py — Logic Investigation

Living notes on the **logic** of the decomposition pipeline (not code style). We are *not*
rewriting yet — this document records findings, open questions, and the plan for investigating
gradually. Working branch: `refactor/simplify-preprocess-logic`; safe checkpoint:
`fix/dense-limb-correspondence-and-cgal-fallback` @ `a01f35d`.

## Goals of the investigation
1. **Minimize reliance on the unreliable `mesh_tools` dependency** — understand exactly which calls
   are fragile, what their contract is, and where a thin seam could isolate (and later replace) them.
2. **Tame the exception / branching / re-call explosion** — the pipeline re-invokes the same heavy
   operations many times through nested retries and fallbacks; map where and why, and which branches
   are real vs. dead defensive code.

Non-goals (for now): rewriting logic, merging functions, performance. Behavior stays anchored by the
golden harness (small-h01 + `1830470325`) on every future change.

## Executive summary (2026-07-19 — Areas A–E swept, no code changed)

**Goal 1 — minimize the unreliable `mesh_tools`:** the fragile surface is small and mapped (Area E
contract table): **6 calls** in `cu`/`m_sk` (+1 `sk` CGAL). The **only non-determinism in the entire
pipeline** is one line — the waterfill `np.random` tie-break at `compartment_utils:1187` inside
`resolve_empty_conflicting_face_labels`. A thin adapter around those 6 (uniform empty/failure contract)
would isolate every fragile interaction in one file, and is the natural plug-in point for a library swap
(skeletor / meshparty-native map / CGAL correspondence). The 60+ `sk`/`tu` skeleton primitives are stable
and not the problem. (We can only *isolate* — `mesh_tools` is a sibling package, edited upstream.)

**Goal 2 — tame re-calls / branches / exceptions:** measured, several suspected hotspots are **non-problems**:
- Skeletonization "cycle" (Area B) is a bounded **≤2-pass** adaptive, not a loop → no unbounded re-calls.
- Concept-network branching (Area D) is a **build + 4-check validation battery**, not tangled logic.
- The floating selection loop's O(N²·L) rebuild (Area C) is **1.5 %** of time → ignore.
- The correspondence "double-call" retry (Area A) **never fires** (0/341 branches) — dead by geometry.

The **real** re-call cost is the floating-piece **decomposition**: `preprocess_limb` runs on every
above-threshold piece (128 on the big H01), and ~**40 % are decomposed then dropped** as too-far. But that
waste is **structural** (decomposition reveals chain connectivity) and the one safe way to cut it — a
cheap connectivity pre-filter — **saves 40 % on thin-piece H01 but 0 % on fat-piece minnie65** (Q7/Q9/Q10),
so it's shelved.

**Actionable takeaways:** (a) the outer correspondence double-call is safely removable (prefer `base_kwargs`
dedup, keep the net); (b) a `mesh_tools` adapter seam is well-specified if isolation/determinism is wanted;
(c) most of this file's branching is load-bearing validation, not cruft — consistent with the earlier
finding that deep branch-reduction has poor risk/reward. **Net: no large safe logic win found; the value
was ruling out tempting-but-unsafe changes with measurement before writing code.**

---

## Pipeline map (top-level call tree)

```
preprocess_neuron
├─ _extract_single_soma                (soma detection; strict single-soma)
├─ _segment_limbs_from_soma            (soma-touching → candidate limb meshes)
├─ _decompose_limbs  → preprocess_limb  (PER LIMB — the hub, see below)
├─ _stitch_floating_pieces → attach_floating_pieces_to_limb_correspondence
├─ _rebuild_limb_frames                (CLASS-A fix: contiguous mesh frame after stitching)
└─ _build_concept_networks → calculate_limb_concept_networks
```

`preprocess_limb` (the hub) fans out to 10 helpers:
```
preprocess_limb
├─ _cycle_for_something                (SKELETONIZATION RETRY LOOP → _run_skeletonization_pass ×2, _decide_next_limb_cfg → _new_invalidation_d)
├─ _group_into_map_mp_sublimbs         (split into MAP [CGAL] vs MP [meshparty] sublimbs)
├─ _decompose_map_sublimbs → _decompose_map_piece   (CGAL skeleton + correspondence)
├─ _decompose_mp_sublimbs  → _fix_mp_soma_extension (meshparty + soma-extension)
├─ _merge_map_mp_correspondence
├─ _rearrange_network_starting_info / _clean_network_starting_info
├─ _stitch_map_and_mp                  (MAP↔MP STITCH → finders, reroute, _recorrespond_stitch, _overwrite_stitched_entries)
├─ filter_limb_correspondence_for_end_nodes
└─ calculate_limb_concept_networks
```

---

## Finding 1 — The correspondence engine is the re-call epicenter (3 nested levels)

The mesh→skeleton "correspondence" (assign faces to skeleton branches) is invoked from **many** sites
and **re-calls itself at three levels**:

- **Level 3 (pipeline):** `mesh_correspondence_first_pass` + `correspondence_1_to_1` (bundled as
  `_run_mesh_correspondence`) run once per MAP piece (`_decompose_map_piece`), **twice** per stitch
  (`_recorrespond_stitch`), once per floating stitch (`_stitch_floating_piece_into_limb`), once per
  soma-extension (`_fix_mp_soma_extension`). → order of *(#limbs × #stitches)* invocations / neuron.
- **Level 2 (per branch):** inside `mesh_correspondence_first_pass`, the **double-call** — try tight
  params, and if it returns nothing, retry with looser `backup_*` params + `return_closest_face_on_empty`
  ([preprocess_neuron.py:97-113]).
- **Level 1 (inside the dep):** `cu.mesh_correspondence_adaptive_distance` itself runs `get_skeletal_distance`
  **twice** (first pass, then an adaptive threshold = max(mean)+2·max(std) with outlier rejection).

⇒ One logical "correspond this limb" can trigger correspondence work O(branches × 2 × 2) times, plus a
full re-correspondence per stitch. This is the "функции перевызываются по многу раз" the user senses.
**Open Q:** how many total correspondence calls per real neuron, and how often does each retry level
actually fire? (needs instrumentation — see Methodology.)

---

## Finding 2 — The *unreliable* mesh_tools surface is SMALL and isolable

Call counts into `mesh_tools` from this file: **sk (skeleton_utils) 62, tu (trimesh_utils) 26,
cu (compartment_utils) 5, m_sk (meshparty) 3.** But most `sk`/`tu` calls are *stable primitives*
(stack/convert/find skeleton, combine/split mesh). The **fragile, failure-prone, or non-deterministic**
surface is only ~6 call sites:

| call | sites | why fragile |
|---|---|---|
| `cu.mesh_correspondence_adaptive_distance` | 2 | heavy heuristic; returns empty → drives the double-call retry |
| `cu.resolve_empty_conflicting_face_labels` | 2 | source of **CLASS-B** failure ("missing labels not resolved") |
| `cu.waterfill_starting_label_to_soma_border` | 1 | soma-border label flood |
| `m_sk.skeletonize_mesh_largest_component` | 1 | meshparty skeletonization (slow, can degenerate) |
| `sk.skeletonize_and_clean_connected_branch_CGAL` | 1 | CGAL ext; **CLASS-C** degenerate-submesh crash |
| `np.random.choice` @ `compartment_utils.py:1187` | — | **non-determinism** in face-label resolution |

**Implication:** "minimize mesh_tools" ≠ removing the 60 skeleton primitives (foundational). It means
wrapping these ~6 unreliable calls behind a thin **adapter seam** so their failure modes / determinism /
replaceability are contained in one place. That seam is also where a library swap (meshparty / skeletor /
CGAL native correspondence — see the mesh_correspondence_adaptive_distance analysis) would plug in.
`mesh_tools` is a sibling reimerlab package (not vendored), so we cannot edit it here — only isolate it.

---

## Finding 3 — Exception / branch hotspots (by static count)

Functions carrying the most control-flow weight (try / while / if / raise):

| function | try | while | for | if | raise | note |
|---|---|---|---|---|---|---|
| `calculate_limb_concept_networks` | 0 | 0 | 3 | **9** | 4 | concept-graph build, very branchy |
| `correspondence_1_to_1` | 1 | 0 | 5 | 5 | 4 | the 1-to-1 resolver (public API) |
| `_decompose_map_piece` | **2** | 0 | 1 | 5 | 2 | CGAL decomp + fallback |
| `_clean_network_starting_info` | 0 | 0 | 4 | 7 | 2 | starting-info massaging |
| `_fix_mp_soma_extension` | 1 | 0 | 3 | 5 | 1 | soma-extension branches |
| `_recorrespond_stitch` | 1 | 0 | 6 | 3 | 1 | stitch re-correspondence |
| `_rearrange_network_starting_info` | 0 | 0 | 4 | 6 | 3 | |
| `attach_floating_..._correspondence` | 0 | **1** | 0 | 2 | 0 | iterative closest-piece `while` loop |
| `_cycle_for_something` | 0 | 0 | — | — | — | skeletonization retry driver |

**Known failure classes (control-flow-by-exception):**
- **CLASS-A** — post-stitch mesh-frame desync → mass `IndexError`; fixed by `_rebuild_limb_frames`.
- **CLASS-B** — cut-branch re-correspondence fails (`resolve_empty_conflicting_face_labels`); skipped
  best-effort in stitch / floating.
- **CLASS-C** — floating piece won't decompose (degenerate CGAL submesh); skipped best-effort.

**Open Q:** which of these `raise`/`except` arms actually fire on real neurons vs. dead defensive code?

---

## How else to investigate — methodology toolbox

The pipeline is **deterministic run-to-run** on a fixed mesh, and we already have a monkeypatch/capture
harness (used for the stitch RNG-replay). Reuse it to turn hunches into numbers:

1. **Runtime call-count profiling.** Monkeypatch the correspondence / skeletonization / retry entrypoints
   to count invocations + which retry/fallback branch was taken, on `1830470325`. Answers Finding-1's
   open question with hard numbers (calls/neuron, % of branches that hit the backup double-call, #
   iterations of the skeletonization cycle, # CLASS-B/C skips).
2. **Branch liveness (coverage).** Run a golden decomposition under `coverage.py`; list the `if`/`except`/
   `raise` arms that never execute → candidates for "dead defensive branch" (investigate, don't delete yet).
3. **Dependency contract mapping.** For each of the ~6 unreliable `mesh_tools` calls: document inputs,
   outputs, empty/failure semantics, and determinism. This is the spec any adapter/replacement must meet.
4. **Data-flow trace of the "limb entity."** Follow the tuple *(branch_skeleton, branch_mesh,
   branch_face_idx, width_from_skeleton)* through the pipeline — the parallel-args problem the user
   flagged. Note every place it is unpacked, re-bundled, or re-derived.
5. **Redundant-recompute detection.** Same primitive recomputed across retries (e.g. `sk.stack_skeletons`
   ×14, `find_branch_endpoints` ×6) — flag skeleton/mesh work repeated inside retry loops.
6. **Failure-mode catalog.** One row per `raise`/guarded `except`: trigger condition, real-error vs.
   control-flow, and what recovers it.

---

## Investigation areas (prioritized — fill in gradually)

### Area A — Correspondence subsystem  ⭐ (highest leverage)
`mesh_correspondence_first_pass` → `correspondence_1_to_1` → `cu.mesh_correspondence_adaptive_distance`.
The 3-level re-call cascade + CLASS-B failures + the unreliable `cu` calls all live here. This is both
the "re-call" pain and the "minimize mesh_tools" target. **Start here.**

#### ✅ Measured (2026-07-19, `1830470325`, golden-verified transparent instrumentation)

| level | metric | value |
|---|---|---|
| L3 | `mesh_correspondence_first_pass` calls | **25** |
| L3 | `correspondence_1_to_1` calls | **25** |
| L3 | first_pass **by stage** | `_stitch_floating_piece_into_limb`: **23**, `_decompose_map_piece`: 1, `_recorrespond_stitch`: 1 |
| L2 | branches processed (tight first calls) | 178 |
| L2 | first-pass returned empty → **backup double-call fires** | **0 / 178 = 0.00 %** |
| L1 | `get_skeletal_distance` calls | 356 |
| L1 | `adaptive_distance` calls | 178 |
| L1 | **adaptive re-run rate** (2nd distance pass) | **178 / 178 = 100 %** |
| C  | `resolve_empty_conflicting_face_labels` calls / raises | 291 / **2** (CLASS-B) |

**Interpretation (answers Q1, Q2):**
- **Q1:** correspondence is invoked 25× / neuron, and **~92 % of those (23/25) come from floating-piece
  stitching** (the cut-branch re-correspondence in `_stitch_floating_piece_into_limb`), *not* from the
  main MAP/MP decomposition. → The correspondence "re-call" cost concentrates in the **floating-piece
  path**, which is surprising and reframes where to focus. (MP limbs don't hit first_pass at all here;
  `_fix_mp_soma_extension` fired 0 correspondence calls this run.)
- **Q2:** the two retry mechanisms are **not redundant with each other, but wildly asymmetric.** The
  **inner** adaptive re-run (L1, inside `adaptive_distance`) is the real workhorse — **fires 100 %**.
  The **outer** double-call backup (L2, in `first_pass`, the block flagged as "стремно") **fired 0 %** —
  on this neuron it is pure dead defensive code. ⇒ The outer double-call is a **strong simplification
  candidate**, but it is an empty-correspondence safety net, so confirm on a few more (esp. pathological)
  neurons before touching it — a 0 % rate on one neuron ≠ never.
- Every `adaptive_distance` call does 2 `get_skeletal_distance` passes; that inner cost is unavoidable
  without changing the dep. The place we *control* is the outer double-call and the floating-stitch
  re-correspondence volume.

#### ✅ Q2b confirmed — outer double-call is dead across datasets (2026-07-19)

Same instrumentation on two more neurons, incl. a different dataset+config:

| neuron | dataset | config | time | first_pass (by stage) | branches | **backup fire** | rerun | CLASS-B raises |
|---|---|---|---|---|---|---|---|---|
| 1830470325 | H01 | h01 | 1091s | 25 (float 23, map 1, restitch 1) | 178 | **0 / 178 = 0 %** | 100 % | 2 |
| 2451406889 | H01 | h01 | 428s | 8 (soma-ext 4, float 4) | 96 | **0 / 96 = 0 %** | 100 % | 4 |
| 864691135689048288 | minnie65 | microns | 937s | 14 (map 3, restitch 2, float 9) | 67 | **0 / 67 = 0 %** | 100 % | 1 |

**Verdict:** the outer `first_pass` double-call (the "стремно" block, [preprocess_neuron.py:104-113])
fired **0 / 341 branches across 2 datasets × 2 configs** — the tight first pass *never* returned empty
(`first_empty = 0` everywhere). It is effectively **dead defensive code**. The real fallback lives one
level down: `adaptive_distance` runs its 2nd distance pass 100 % of the time and has its *own*
`return_closest_face_on_empty` path. ⇒ Strong candidate to **collapse the outer retry** once we confirm
*why* the tight pass never empties (is `distance_threshold=3000` always sufficient, or does the inner
pass already guarantee non-empty?). Not acting yet — this is a logic decision for later, golden-gated.

Also note (feeds Area C/E): the stage mix of correspondence work is **neuron-dependent** (float 23 vs 9
vs 4; map 0–3) — floating-stitch is usually the largest single contributor but not always. And CLASS-B
(`resolve_empty_conflicting_face_labels` raise) fires **1–4× / neuron** out of ~110–291 calls (~1–3 %),
i.e. it is real, low-rate error handling, not dead.

#### ✅ Q2c — *why* the tight pass never empties (traced through the dep, 2026-07-19)

Empty-return chain: outer double-call fires ⟺ `adaptive_distance` returns `[]` ⟺ its **first**
`get_skeletal_distance` returns `mesh_correspondence_indices` of length 0. Reading
`compartment_utils.get_skeletal_distance` (420-884):
- it collects, per skeleton edge, the mesh faces whose midpoints fall in a z-slice along the edge
  **and** within `distance_threshold` (=3000) laterally → appended to `face_subtract_indices`;
- the return is empty (`unique_removed_faces = np.array([])`, via the `else` at **:877-879**) **iff
  `face_subtract_indices` is empty — i.e. NO edge of the branch skeleton found any mesh face within
  3000**. (If some faces are found but resolve to <1 unique, it *raises* at :633 instead — a different,
  also-rare path.)

**So the outer double-call is a real fallback for "this skeleton branch has no mesh faces near it."**
It is dead **by geometry, not by contract**: `mesh_correspondence_first_pass` always corresponds a
branch skeleton against a mesh the branch is **embedded in** (it was `decompose_skeleton_to_branches`'d
from that very mesh, or is a `cut_skeleton_at_coordinate` of a branch whose own `branch_mesh` is passed).
Every non-degenerate edge therefore sits inside the mesh → finds faces well within 3000 → never empty.
The empty case needs a skeleton *detached from / degenerate w.r.t.* its mesh, which the current data
flow never produces (0/341 confirms).

**Recommendation (revised):** don't just delete the fallback — it guards a real (currently-unreachable)
case that *would* matter if some future caller ever corresponded a skeleton against a **foreign** mesh.
Two options, both golden-gated:
- **(preferred, lowest-risk)** keep behavior identical, kill only the visual duplication via the
  `base_kwargs` dedup (hoist the 3 shared kwargs; the retry stays but stops repeating 7 args). Removes
  the "стремно" without removing the net.
- **(more aggressive)** collapse to a single call + add an `assert`/comment documenting the invariant
  "corresponded skeleton ⊂ its mesh". Smaller code, but drops a safety net for a future misuse.

**Follow-ups:** measure whether the many floating-stitch correspondences recompute overlapping work
(Q6, Area A ∩ Area C).

### Area B — Skeletonization "retry cycle"  ✅ (investigated 2026-07-19 — reassuring negative result)
`_cycle_for_something` / `_decide_next_limb_cfg` / `_new_invalidation_d` / `_run_skeletonization_pass`.

**Q4 answered by static read — it is NOT a loop.** `_cycle_for_something` ([:1181]):
```
pass1 = _run_skeletonization_pass(limb_cfg)            # 1 m_sk skeletonization
if not use_adaptive_invalidation_d: return pass1        # → 1 pass
new_cfg = _decide_next_limb_cfg(pass1, ...)             # decides ONCE
if new_cfg is None: return pass1                        # → 1 pass
return _run_skeletonization_pass(new_cfg)               # → exactly 2 passes, NOT re-checked
```
⇒ **bounded at ≤ 2 skeletonization passes per limb/piece; always terminates; no `while`/`for`.** The
name "_cycle_for_something" is misleading — it's a fixed *adaptive second pass*, not a cycle. So the
user's "functions re-called many times" concern **does not apply here** — this hotspot is bounded.
Total `m_sk.skeletonize_mesh_largest_component` calls ≤ 2·(n_limbs + n_floating_pieces).

**What triggers the 2nd pass** (`_decide_next_limb_cfg` [:1121]): (1) limb is axon-thin
(`width_median ≤ axon_width_preprocess_limb_max`) → re-run with axon params; or (2) no MAP-wide pieces
+ `mp_only_revised_invalidation_d` + not-already-axon → re-run with a width-interpolated `invalidation_d`
(`_new_invalidation_d`, a bounded linear interpolation clamped to `[lowest, max_invalidation_d]`). If the
limb has wide (MAP) pieces → `None` (no 2nd pass; those go to the CGAL/MAP path). Monotone/terminating
trivially (the decision is made once, pass 2 is never re-evaluated).

**Unreliable-mesh_tools here:** each pass = `m_sk.skeletonize_mesh_largest_component` +
`m_sk.skeleton_obj_to_branches` (meshparty; slow, can degenerate) — but bounded ≤2, no error-retry
(failures propagate). The separate **CGAL** skeletonization + its meshparty fallback lives in
`_decompose_map_piece` (Area A/C, the `try=2`), not here.

**Verdict: no problem — bounded and terminating.** Optional color (not essential): the empirical 2nd-pass
fire-rate per limb (needs one instrumented run). Skipped — the bound is the point, and it's proven
statically.

### Area C — Stitching (MAP↔MP + floating)  ⭐ (dominant correspondence volume)
`_stitch_map_and_mp` (+ `_recorrespond_stitch`, finders, reroute) and `attach_floating_*` (`while` loop).
Already decomposed structurally; logic still re-runs correspondence heavily and has CLASS-A/B.

#### Static structure (read 2026-07-19)

**The floating-stitch funnel** (`attach_floating_pieces_to_limb_correspondence`, [:704]):
```
floating_meshes
  → filter len(faces) > floating_piece_face_threshold        (_preprocess_floating_pieces)
  → preprocess_limb PER surviving piece  (full decomposition!) — CLASS-C: skip pieces that won't decompose
  → while pieces remain:
        _find_closest_floating_piece  (rebuild + KDTree, see below)
        if closest dist > max_stitch_distance:  return   ← drops ALL remaining pieces at once
        _stitch_floating_piece_into_limb        ← may CLASS-B skip (cut-branch re-correspondence fails)
        mark winning piece processed
```

**Re-computation smell — the selection loop is O(N²·L).** `_find_closest_floating_piece` ([:429]) is
called **once per `while` iteration** and each call:
1. rebuilds **every** main limb's full skeleton (`sk.stack_skeletons`, [:436-438]),
2. builds a fresh **KDTree per limb** ([:446]),
3. re-queries **every** still-unprocessed floating piece against every limb ([:448-454]).
Only **one** limb changes per stitch (the one that received the winning piece), yet all L limbs'
skeletons+KDTrees are rebuilt and all remaining pieces re-queried every iteration ⇒ for N pieces and L
limbs the selection is ~O(N²·L) where an incremental version (rebuild only the changed limb, cache the
rest) would be ~O(N·L). **Whether this matters depends on cost split** — the per-piece `preprocess_limb`
(step 2) is the expensive part; the selection rebuilds are on small skeletons. → quantify (below).

**Two ways a piece leaves without being stitched:**
- **too-far** (`dist > max_stitch_distance`): a single `return` abandons **all** remaining pieces (not
  just the far one) — greedy-closest order means once the nearest is too far, the rest are too.
- **CLASS-B** (`_stitch_floating_piece_into_limb` → cut-branch `correspondence_1_to_1` raises): that one
  piece is skipped (early return, still marked processed); loop continues.

#### 📊 Measured funnel + time split (2026-07-19, `1830470325`, golden-verified)

**Funnel:**
| stage | count |
|---|---|
| floating meshes input | **1107** |
| pass face threshold → `preprocess_limb` attempted | **128** (≈12 %) |
| decomposed OK (CLASS-C skipped = 0) | 128 |
| `while` iterations (`_find_closest` calls) | 78 |
| **stitched** (passed `max_stitch_distance`) | **77** |
| **dropped as too-far** (single early `return`) | **51** |
| of the 77 stitches: split a main branch (cut+re-corr) | 23 |
| of the 77 stitches: onto existing end/branch node | 54 |
| CLASS-B raises **in floating** | **0** (of 277 `resolve_empty` calls) |

**Time split (of ~4.4 min total floating work):**
| phase | time | note |
|---|---|---|
| `_preprocess_floating_pieces` (`preprocess_limb` ×128) | **117 s** | full decomposition per piece |
| `_stitch_floating_piece_into_limb` (×77) | **146 s** | the 23 branch-cut re-correspondences live here |
| selection `_find_closest` (×78) | **1.8 s** | the O(N²·L) rebuild — **1.5 % of preprocess** |

**Findings:**
1. **~40 % of the expensive decomposition is wasted.** 128 pieces are fully `preprocess_limb`'d up front
   (**117 s**), but **51 of them (40 %) are then dropped as too-far** — the greedy loop stitches 77 by
   nearest-first, hits one beyond `max_stitch_distance`, and `return`s, abandoning the other 50. Those 51
   decompositions (~47 s) were pure waste. **Root cause: decomposition happens *before* the distance
   filter.** ⭐ Real logic-reordering opportunity — a cheap pre-filter (e.g. mesh-centroid distance to
   main-limb skeletons) *before* `preprocess_limb` would skip decomposing the obviously-far pieces.
   (Caveat: the exact stitch distance needs the piece's skeleton; a coarse proxy would only prune the
   clearly-hopeless ones — but at 40 % waste even a loose proxy pays off.)
2. **The O(N²·L) selection re-computation is a non-issue** — 1.8 s total, 1.5 % of preprocess. The
   "smell" is real but **not worth optimizing.** Don't touch it.
3. **Stitch (146 s) > preprocess (117 s)** — the stitching itself is the single largest floating cost;
   the 23 branch-cut re-correspondences (the Area-A "23 floating first_pass") are inside it.
4. **CLASS-B did NOT fire in floating this run (0/277).** The 2 CLASS-B raises seen in Area A were in the
   **core** decomposition / MAP↔MP stitch path, not floating — corrects the earlier assumption that
   "CLASS-B lives in floating." (It *can* fire there — the guard is real — just didn't here.)
5. Cheap face-threshold filter does the bulk of the culling well (1107 → 128) at ~zero cost; the waste is
   entirely in the 51 that pass the threshold, get decomposed, then dropped for distance.

#### 🔬 Q7 verdict — the simple pre-filter is INFEASIBLE; chaining dominates (2026-07-19, golden-verified)

Measured, per above-threshold piece: `proxy = min dist(mesh.vertices → INITIAL main-limb skeleton)`
(a would-be lower bound) vs. its actual stitch outcome. Threshold `max_stitch_distance = 13000`.

| set | n | proxy dist [min / med / max] |
|---|---|---|
| STITCHED | 77 | 216 / 13946 / **164228** |
| DROPPED  | 51 | 14348 / 121959 / 281275 |

- **Pre-filter (prune if proxy > 13000) would FALSE-PRUNE 41 of the 77 stitched pieces (53 %)** while
  saving all 51 dropped. **Unsafe** — it deletes real limb pieces.
- **Why:** the "lower bound" is violated for 64/77 stitched pieces (e.g. piece 3: proxy 72 095 but real
  stitch dist **1 331**). The proxy is measured against the **initial** main limbs; the real stitch is
  against the **grown** limbs. ⇒ **chaining is dominant: >half the stitched pieces are too far from the
  initial limbs and only reach them through chains of other already-stitched floating pieces.** Initial
  distance is **not predictive** of final stitchability (stitched proxies span 216–164 228, overlapping
  the dropped range entirely).

**Verdict:** the 40 % "waste" is **not** trivially avoidable. `preprocess_limb` does double duty —
producing each piece's skeleton *and* revealing the chain connectivity that decides reachability. You
can't know a piece is unreachable without (it and its potential chain-neighbours) being decomposed.

**Reframed opportunity (bigger, its own correctness proof needed):** eventual stitchability is a
**connected-component** property of the graph *{main limbs + floating pieces, edges = "within 13000"}*.
A *connectivity-aware* pre-filter could build that graph on **cheap mesh-to-mesh proximity** (centroid /
bbox / vertex KD-trees, no decomposition), keep only pieces in a main-limb's component, and decompose
only those. → tested in Q9.

#### ✅ Q9 verdict — cheap mesh-proximity connectivity reproduces the stitched set EXACTLY (2026-07-19)

Captured (golden-verified) the initial main-limb skeleton clouds + each above-threshold piece's
downsampled vertex cloud (≤3000 pts) + the real stitched set, then built offline the graph
*{3 mains + 128 pieces, edge iff min cloud-to-cloud dist ≤ thr}* and took the main-limb connected
component (`scratchpad/instrument_areaC_Q9_capture.py` + `q9_analyze.py`):

| threshold | cheap-reachable | false-prune (stitched missed) | savings (dropped pruned) | over-incl |
|---|---|---|---|---|
| **13000 (= max_stitch_distance)** | **77** | **0** ✓ | **51/51 = 100 %** | 0 |
| 19500 / 26000 / 39000 | 128 | 0 | 0 % | 51 |

**At the real threshold the cheap graph == the real stitch reachability, perfectly:** reachable = the 77
stitched, unreachable = the 51 dropped, with **0 false-prune and 0 over-inclusion.** ⇒ a connectivity
pre-filter would decompose **77 instead of 128 pieces (−40 %, ~47 s saved on this neuron)** with **no
behavior change to the stitched set.** The downsampling biases toward *missing* edges (i.e. toward
*false* false-prunes), so 0 false-prune under downsampling is a **conservative/strong** safety signal.
Larger thresholds are safe but over-connect (0 savings) — 13000 is exactly right, as expected since it
*is* the stitch threshold.

**Why it works (theory matches):** mesh **surface**-to-surface distance ≤ interior **skeleton**-endpoint
distance, so the cheap graph is a **superset** of the real skeleton-stitch graph ⇒ its component can
never *miss* a stitched piece (safety by construction). The 0 over-inclusion says the surface-vs-skeleton
gap doesn't create spurious links **on this neuron** — the open question for generalization.

**Status (after Q9, single neuron): looked FEASIBLE and high-value.** Gate before coding: Q10.

#### ⚠️ Q10 verdict — SAFE everywhere, but SAVINGS do NOT generalize (2026-07-19)

Ran the same capture+analysis on two more neurons (`instrument_areaC_Q9_capture_generic.py` + `q9_analyze.py`):

| neuron | dataset | config | thr | pieces | stitched | dropped | false-prune | **savings** |
|---|---|---|---|---|---|---|---|---|
| 1830470325 | H01 | h01 | 13000 | 128 | 77 | 51 | **0** ✓ | **51/51 = 40 % fewer** |
| 2451406889 | H01 | h01 | 13000 | 36 | 36 | 0 | **0** ✓ | n/a (no waste) |
| 864691135689048288 | minnie65 | microns | **8000** | 50 | 31 | 19 | **0** ✓ | **0/19 = 0 %** |

- **Safety holds universally: 0 false-prune on all 3** (as the surface ≤ skeleton superset argument
  guarantees). Good — the method never drops a real piece.
- **But savings are geometry-dependent and can be ZERO.** On **minnie65 all 19 dropped pieces are
  over-included** — their *surfaces* sit within the 8000 threshold of the reachable component even
  though their *skeleton* stitch distance exceeds it. The **surface-vs-skeleton gap** (large for
  minnie65's **fat** pieces) plus the **smaller threshold** (8000 vs H01's 13000) means mesh-proximity
  can't separate stitched from dropped. Tightening to 6000 immediately **false-prunes** a real piece
  (unsafe) — so there is **no safe threshold with savings** on minnie65.

**Conclusion: the connectivity pre-filter is NOT a robust win — shelve it.** It saves 40 % on thin-piece
H01 but **0 % on the fatter minnie65** — and minnie65 is the *larger, costlier* data, where saving would
matter most. The mesh-surface proxy is fundamentally too loose for fat pieces; a proxy tight enough to
save there would have to approximate the **skeleton centerline**, which is most of the decomposition work
we're trying to avoid. Idea not dead, but needs a cheap centerline proxy (PCA/medial-axis) to be worth
revisiting. **Value of Q10:** the tempting single-neuron Q9 result (a "40 % slam-dunk") was correctly
caught as non-general *before any code was written* — the whole point of this investigation phase.

### Area D — Concept network + starting-info  ✅ (static read, 2026-07-19)
`calculate_limb_concept_networks` (if=9, raise=4) + `_clean/_rearrange_network_starting_info`.

**Finding — the branch-density is a VALIDATION battery, not tangled logic.** `calculate_limb_concept_networks`
([:801]) is structurally simple: for each `soma_idx` → each soma-group starting point:
1. find the must-keep start branch (`sk.find_branch_skeleton_with_specific_coordinate`);
2. **build** the concept network — one call to `nru.branches_to_concept_network(...)` (the real work,
   in **neuron_utils**, NEURD-owned, *not* mesh_tools);
3. **validate** (gated by `run_concept_network_checks`): 3.1 start piece recovered, 3.2 #nodes == #branches,
   3.3 single connected component, 3.4 per-branch endpoints match — **4 of the 4 `raise`s are these
   defensive checks.** So the "huge branching" is an **assertion suite around a single build call**, not
   control-flow-by-exception. No re-calling loop; bounded by (n_somas × soma-groups) — tiny for the
   single-soma target. The whole check battery is switchable off (`run_concept_network_checks=False`).

⇒ Not a re-call / complexity hotspot. If one wanted fewer branches here, the lever is factoring the 4
checks into a `_validate_concept_network(...)` helper (cosmetic, low value) — the logic itself is a clean
build+verify. The concept-network *construction* complexity lives in `nru.branches_to_concept_network`
(a separate neuron_utils study, out of this file's scope).

**Secondary (not deep-dived):** `_clean/_rearrange_network_starting_info` (for=4, if=6-7 each) prepare
`network_starting_info` before this; open sub-Q (low priority): do the two massage it redundantly? — a
targeted read if starting-info ever becomes suspect.

### Area E — mesh_tools dependency seam (cross-cutting)  ✅ (contract table, 2026-07-19, static read)
The isolation target from Finding 2 — serves "minimize mesh_tools". Contract of the ~6 unreliable calls
(what a replacement/adapter must provide):

| call (mesh_tools) | in → out | empty / failure | determinism |
|---|---|---|---|
| `cu.get_skeletal_distance(mesh, edges, distance_threshold=3000, …)` | → `(mean_d, std_d, submesh, face_idx)`: faces within a per-edge tube of the skeleton | returns `face_idx=[]` (else @ :877) when **no edge finds a face** within threshold; raises if faces found but <1 unique (:633); raises on internal errors | **deterministic** (geometry) |
| `cu.mesh_correspondence_adaptive_distance(skel, mesh, distance_threshold, return_closest_face_on_empty)` | → `(face_idx, width)` or `[]` | `[]` when first pass empty (unless `return_closest_face_on_empty` → single closest face); wraps 2× `get_skeletal_distance` (adaptive thr = mean+2·std) | **deterministic** |
| `cu.resolve_empty_conflicting_face_labels(mesh, face_lookup, must_keep_labels, …)` | face labeling (may have empty/conflict) → **complete 1-to-1** labeling, each label 1 connected component | **raises if mesh not 1 connected component** (:1273); **CLASS-B raise** when a branch label ends with no faces / waterfill can't resolve | **NON-deterministic** — `np.random.choice` in the waterfill tie-break (`compartment_utils:1187`, propagation_type="random") ⚠️ |
| `cu.waterfill_starting_label_to_soma_border(…)` | soma-border label flood | (not fully read) | likely non-det (shared waterfill) — TODO |
| `m_sk.skeletonize_mesh_largest_component(mesh, root, invalidation_d, …)` | mesh → MCF skeleton obj | slow; can degenerate on bad meshes; failures propagate (no retry in `_run_skeletonization_pass`) | meshparty MCF (mostly deterministic, slow) |
| `sk.skeletonize_and_clean_connected_branch_CGAL(…)` | mesh → cleaned CGAL skeleton | **CLASS-C** degenerate-submesh crash; guarded by `error_on_bad_cgal_return` + meshparty fallback in `_decompose_map_piece` | CGAL (deterministic); the fallback path is the variable one |

**Key takeaways for "minimize mesh_tools":**
- The **only non-determinism** in the whole correspondence/skeletonization surface is the **waterfill
  random tie-break** in `resolve_empty_conflicting_face_labels` (:1187). Everything else is
  deterministic. ⇒ a single deterministic tie-break (seeded / lowest-label / majority) would make the
  pipeline deterministic-in-isolation (today it's only deterministic run-to-run because the global RNG
  advances identically from the seed — the reason the stitch RNG-capture harness exists).
- The unreliable surface is **6 calls in 2 dep modules** (`cu`, `m_sk`) + one `sk` CGAL entry. A thin
  adapter wrapping these (uniform empty/failure/determinism contract) would isolate every fragile
  interaction in one file — without touching the 60+ stable `sk`/`tu` skeleton primitives. That adapter
  is also where a library swap (skeletor / meshparty-native map / CGAL correspondence) would plug in
  (see the `mesh_correspondence_adaptive_distance` analysis in Area A discussion).
- `mesh_tools` is a **sibling reimerlab package** (not vendored) — we can only **isolate**, not edit it
  here; changes to it belong upstream (`reimerlab/mesh_tools`, `pip install -e`).

---

## Open questions log
- Q1 (Area A): ✅ ANSWERED — 25 correspondence runs/neuron, 23 from floating-stitch. See Area A table.
- Q2 (Area A): ✅ ANSWERED — outer double-call fires 0 %, inner adaptive re-run 100 %; NOT mutually
  redundant; outer double-call is a simplification candidate (confirm on more neurons). See Area A.
- Q2b: ✅ ANSWERED — outer double-call fired 0/341 branches across 3 neurons / 2 datasets / 2 configs
  (H01 h01 ×2 + minnie65 microns). Dead defensive code; collapse candidate (gated on Q2c).
- Q2c: ✅ ANSWERED — get_skeletal_distance returns empty (`else` @ compartment_utils:877) only when NO
  skeleton edge finds a face within 3000; the outer double-call is a real "no nearby faces" fallback,
  dead **by geometry** (corresponded skeleton always ⊂ its mesh). Prefer `base_kwargs` dedup over deletion.
- Q3 (Finding 3): which raise/except arms never fire on real neurons (dead defensive code)? → coverage run.
- Q4 (Area B): ✅ ANSWERED (static) — `_cycle_for_something` is NOT a loop; bounded ≤2 skeletonization
  passes/limb, always terminates. 2nd pass triggers on axon-thin OR no-MAP+invalidation-tune. No problem.
- Q5 (Area E): ✅ ANSWERED — contract table done (see Area E). Only non-determinism = waterfill random
  tie-break @ compartment_utils:1187 (in resolve_empty); rest deterministic. Adapter seam = 6 calls.
- Q6 (Area A∩C): ✅ partly — the 23 floating-stitch re-correspondences are branch-cut re-corr inside the
  77 stitches; they're the biggest floating cost (part of 146 s stitch). Not obviously overlapping.
- Q7 (Area C): ❌ ANSWERED — a simple initial-distance pre-filter is INFEASIBLE: chaining dominates
  (41/77 stitched pieces have proxy>threshold, chain-reached), so it would false-prune 53 % of real
  pieces. The 40 % waste is structural (decomposition reveals reachability).
- Q8 (Area C): does the too-far greedy `return` ever drop a piece that a *different* main limb could have
  stitched (i.e. is nearest-across-all-limbs the right global criterion)? [now clearly connectivity-shaped]
- Q9 (Area C ⭐⭐): ✅ ANSWERED — on `1830470325`, cheap mesh-proximity component at thr=13000 == the
  real stitched set EXACTLY (0 false-prune, 100 % savings, 40 % fewer decompositions). FEASIBLE.
- Q10 (gate): ⚠️ ANSWERED — SAFE on all 3 neurons (0 false-prune) but savings DON'T generalize: 40 % on
  H01-big, **0 % on fat-piece minnie65** (all 19 dropped over-included; surface-vs-skeleton gap + smaller
  8000 thr). Connectivity pre-filter SHELVED (not robust); would need a cheap centerline proxy.
- Q11 (NEW, parked): a cheap centerline proxy (PCA/medial-axis skeleton, far cheaper than preprocess_limb)
  to recover savings on fat pieces — only if this optimization is ever revisited.

## Findings log (dated)
- 2026-07-19: doc created; seeded Findings 1–3 from static analysis (call graph, mesh_tools surface,
  control-flow counts). No code changed.
- 2026-07-19: **Area A instrumented on `1830470325`** (golden-verified transparent). Answered Q1/Q2:
  correspondence dominated by floating-stitch (23/25); outer double-call backup **0 %**, inner adaptive
  re-run **100 %**; CLASS-B raised 2×. → outer double-call = simplification candidate pending multi-neuron
  confirmation. Harness: `scratchpad/instrument_areaA.py`.
- 2026-07-19: **Q2b confirmed** on 2 more neurons (H01 `2451406889` + minnie65 `864691135689048288`,
  microns config): outer double-call **0/341 branches across 2 datasets × 2 configs**, rerun 100 %,
  CLASS-B 1–4×/neuron. Generic harness: `scratchpad/instrument_areaA_generic.py`.
- 2026-07-19: **Q2c answered** by tracing `get_skeletal_distance` (compartment_utils). Empty return via
  `else` @ :877 only when no skeleton edge finds a face within `distance_threshold=3000`; outer
  double-call = real "no nearby faces" fallback, dead **by geometry** (corresponded skeleton ⊂ its mesh).
  → recommend `base_kwargs` dedup (keep the net, kill the duplication) over outright deletion. No code
  changed. Area A investigation complete.
- 2026-07-19: **Area C** static + instrumented (`scratchpad/instrument_areaC.py`, golden-verified) on
  `1830470325`. Funnel 1107→128→77 stitched, **51 (40 %) decomposed-then-dropped-too-far = wasted
  preprocess (~47 s)** → Q7. O(N²·L) selection is 1.5 % of time (ignore). Stitch (146 s) > preprocess
  (117 s). CLASS-B fired 0× in floating (the 2 Area-A raises were core-path). No code changed.
- 2026-07-19: **Q7 answered ❌** (`scratchpad/instrument_areaC_Q7.py`, golden-verified). Simple
  initial-distance pre-filter is UNSAFE: **chaining dominates** — 41/77 stitched pieces (53 %) are
  chain-reached (proxy vs initial limbs > 13000 but real stitch dist tiny, e.g. 72 095 vs 1 331). Initial
  distance not predictive; would false-prune 53 % of real pieces. The 40 % waste is structural
  (decomposition reveals chain connectivity). Real lead = Q9 (connectivity-aware pre-filter on cheap
  mesh-proximity), a bigger change needing its own correctness study. No code changed.
- 2026-07-19: **Q9 answered ✅ ⭐** (`instrument_areaC_Q9_capture.py` + `q9_analyze.py`, golden-verified
  capture). Cheap mesh-proximity connectivity component at thr=13000 == real stitched set **exactly**
  (77 reachable = 77 stitched; 0 false-prune; 51/51 dropped pruned = 40 % fewer `preprocess_limb`).
  Safe by construction (surface-dist ≤ skeleton-dist ⇒ cheap graph ⊇ real graph). Gate = Q10.
- 2026-07-19: **Q10 answered ⚠️ — SHELVED** (generic capture on minnie65 `864691135689048288` + H01
  `2451406889`). Safety universal (0 false-prune ×3), but savings geometry-dependent: 40 % H01-big,
  **0 % minnie65** (fat pieces → surface-vs-skeleton gap over-includes all 19 dropped at safe thr 8000;
  6000 false-prunes). Connectivity pre-filter NOT robust → shelved (parked as Q11: needs a cheap
  centerline proxy). Investigation correctly killed a tempting single-neuron win before coding.
- 2026-07-19: **Area B investigated (static) — reassuring negative result.** `_cycle_for_something` is a
  fixed ≤2-pass adaptive skeletonization, NOT a loop → always terminates; the "re-called many times"
  concern doesn't apply here. Q4 answered without a run.
- 2026-07-19: **Area E contract table (static).** Documented the 6 unreliable mesh_tools calls'
  in/out/empty/failure/determinism. Only non-determinism = waterfill `np.random` @ compartment_utils:1187
  (in `resolve_empty_conflicting_face_labels`); rest deterministic. Adapter seam = those 6 calls in
  cu/m_sk (+1 sk CGAL).
- 2026-07-19: **Area D (static).** `calculate_limb_concept_networks` branch-density is a build (one
  `nru.branches_to_concept_network`) + a 4-check validation battery (gated) — not tangled logic, no
  re-call loop. Concept-network construction complexity is in neuron_utils (out of file scope).
  **All areas A–E swept; see Executive summary. No code changed.**
