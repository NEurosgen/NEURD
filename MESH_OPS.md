# The owned mesh-operations layer

NEURD's essence is *take meshes → segment them*, so the mesh operations are the **foundation**, not a
peripheral API. This layer rebuilds the primitives as clean NEURD-owned code and composes upward. That
is also how the "minimize `mesh_tools`" goal is met: an owned layer replaces scattered `tu.*` calls and
isolates the fragile dependency behind one seam.

Merges what were three separate docs (`MESH_OPERATIONS_LAYER`, `MESH_OPS_REBUILD_PLAN`,
`MESH_OPS_PHASE_B_PLAN`). Companion: [PIPELINE.md](PIPELINE.md).

## The linchpin

`trimesh.submesh()` returns a standalone mesh with **no link to its parent's face indices**. So the
framework perpetually re-derived that link — either by geometric KDTree midpoint matching
(`original_mesh_faces_map`, a fragile 103-line function, threshold-dependent) or by hand
(`[face_idx[k] for k in comps]`). A **`SubMesh` value type that carries `face_idx` and composes**
removes that whole class of bookkeeping *structurally*.

Reading the actual `trimesh_utils` implementations showed the L2 cluster is a composition tower buried
in defensive cruft, reducing to ~3 primitives:

| mesh_tools fn | lines | what it really is |
|---|---|---|
| `split` | 138 | `face_adjacency → connected_components → submesh(each) → order-by-size` (+ duplicated type checks, a no-op try/except, commented code) |
| `split_significant_pieces` | 129 | `split` + filter `len(faces) ≥ threshold` + sort (+ a 4-way return-flag explosion) |
| `largest_conn_comp` | 19 | `split_significant_pieces(threshold=-1)[0]` (magic −1) |
| `connected_components_from_face_idx` | 28 | `split(mesh.submesh(face_idx))` then a **manual remap** |
| `original_mesh_faces_map` | 103 | recover submesh→parent face indices by KDTree-matching face midpoints — needed only because `submesh()` drops provenance |

**~420 mesh_tools lines → 3 primitives + a small value type + ~8 tiny composites**, and the
most-fragile op largely disappears, because provenance is carried by construction rather than
re-derived.

## What exists — `neurd/submesh_ops.py` (446 L)

Named `submesh_ops` because `neurd/_mesh_ops.py` already holds the L6 decimate/poisson/fill_holes
layer. The primitives are reimplemented **directly on trimesh**, not as thin `tu.*` wrappers.

```python
@dataclass(eq=False)
class SubMesh:                 # a mesh that REMEMBERS its provenance and COMPOSES
    mesh:     "Trimesh"
    face_idx: np.ndarray       # indices into the IMMEDIATE parent's faces
    parent:   "Trimesh | SubMesh"
    def sub(self, faces) -> "SubMesh"     # submesh-of-submesh; `faces` index into THIS mesh
    def root_face_idx(self) -> np.ndarray # fold the whole chain to the ORIGINAL mesh, O(1)
    def n_faces(self) -> int
```

| API | role |
|---|---|
| `connected_face_components(mesh, connectivity)` | the core primitive; `"edges"` and `"vertices"` modes |
| `vertex_components` / `_vertex_component_labels` | scipy union-find vertex partition (shared core; also the `tu.vertex_components` replacement) |
| `split`, `split_significant`, `largest_component` | partition composites (`split_significant` has a `return_insignificant` mode) |
| `components_from_face_idx`, `face_groups_by_label`, `split_into_face_groups` | face-index algebra |
| `combine` | merge with provenance preserved |
| `original_faces`, `faces_by_match` | provenance recovery (O(1) for in-pipeline meshes; geometric match only for foreign ones) |
| `pieces_adjacency` | piece-to-piece adjacency graph (replaced the O(N²) spine connectivity) |

The composition law `sub(A).sub(B) == A[B]` is the core invariant, tested as
`tests/unit/test_submesh_ops.py::test_composition_law`.

**Gating:** `tests/unit/test_submesh_ops.py` (29 tests) proves `tu.*` equivalence and the composition
laws on a synthetic 3-icosphere and the small h01, in both connectivity modes, including
`original_faces == tu.original_mesh_faces_map` on the real mesh. Plus
`test_cfc_vertex_faces_free.py` (44), `test_vertex_components.py`, `test_pieces_adjacency.py`.
Getting `connectivity="vertices"` semantics exactly right is the thing those tests guard.

## Status

**Phase 0/1 — done.** The layer plus the gateable split-based L2 call sites in
`soma_extraction_utils` / `preprocess_neuron`. The simple-swap queue is exhausted.

**Phase B (provenance through correspondence) — mostly done.** It addressed three coupled fragilities,
all stemming from `branch_face_idx` being carried as a raw index array whose frame is implicit:

| fragility | state |
|---|---|
| 4 manual remaps `parent_idx[branch_face_idx]` | ✅ 2 consolidated into one seam; the 2 stitch ones remain (low value) |
| the frame-desync bug class (`_stitch_floating_pieces` leaves branches in a foreign frame) | ⚠️ only **partially** eliminable — see C1 |
| limb/soma KDTree frame fallback in `neuron.py` | ✅ done — preprocess now supplies `limb/soma_meshes_face_idx`, so neither fallback fires |

The seam is `correspondence_1_to_1(..., input_sub=...)` in `preprocess_neuron.py`, forwarded by
`_run_mesh_correspondence` via `**kwargs`.

### Durable corrections (learned by executing, don't relitigate)

- **C1 — `_rebuild_limb_frames` and the `neuron_utils` guard CANNOT be deleted.** It does *two* jobs:
  (a) reindex `branch_face_idx` into a clean frame, and (b) **grow the limb mesh** to contain
  floating-piece geometry that genuinely is not in the original limb mesh. `SubMesh` provenance can
  express (a) but never (b) — no index exists into the original limb for faces that were never in it.
  So the frame-desync class is only partly removable and the guard stays as a cheap invariant.
  (PIPELINE.md §2.1 describes the bug; note it, not the older claim that the function became
  unnecessary.)
- **C2 — the preferred seam shape leaves readers untouched.** `input_sub=` makes
  `correspondence_1_to_1` return `branch_face_idx` already in the limb frame, but still as a plain
  array, so **no reader changed** (~25 `branch_mesh`/`branch_face_idx` readers). Prefer this over
  storing a `SubMesh` in the dict.
- **C4 — coverage before design.** Two originally-planned pieces targeted code that never runs. Probe
  what actually executes (the harness prints caller histograms) *before* designing a replacement.

### Remaining work (both optional, low value)

- **Step 3 — the two stitch remaps into the seam.** They live in `_recorrespond_stitch` (MP-stitch and
  the `cut_flag` MAP-stitch). Migration is byte-identical but mostly cosmetic: the output is
  **overwritten by `_rebuild_limb_frames`** on nearly every stitched limb, and the MAP-stitch remap
  fires on **no anchor** (rare `cut_flag` geometry), so it cannot be golden-gated. Both sit inside a
  broad `except` that reverts on any error — if attempted, temporarily re-raise there, or a threading
  bug diagnoses blind.
- **Step 4 — `Branch` stores a `SubMesh`.** `mesh_face_idx` becomes a `@property` returning
  `root_face_idx()`. Readers are fine (C2); the **write side** is the problem (~16 writers including
  `neuron_utils` reassignments, and the name is shared with `Soma`). Decide up front: migrate the
  writers, or add a transitional "detached" setter. **Scope to `Branch`** — leave `Soma.mesh_face_idx`
  a plain attribute, it is geometric by nature.

### Dropped after probing (don't revive without a new driver)

`original_mesh_faces_map`, `subtract_mesh` and `compare_meshes_by_face_midpoints` have **zero callers
during `segmentation_pipeline`**. They are still live on the *spine* path (`spine_utils`,
`neuron_utils`, `branch_utils`), which is a different entry point — so this is "not on the segmentation
path", not "dead". Note `submesh_ops` has **no `subtract_mesh` equivalent**, contrary to an earlier
claim; that would-be drop-in swap does not exist.

The geometric matching that *does* still run is one level down inside `mesh_tools`
(`remove_mesh_interior`, `mesh_list_connectivity`) and is not reachable by handing in provenance — they
remap internally, or operate on Poisson-SDF soma candidates that have no provenance in principle.

## The remaining `tu` surface

**59 distinct `tu.*` functions**, and the distribution is the whole story:

| module | `tu.*` call sites |
|---|---|
| `spine_utils.py` | 48 |
| `submesh_ops.py` | 19 (deliberate — this is the owned wrapper layer) |
| `preprocess_neuron.py` | 16 |
| `soma_extraction_utils.py` | 12 |
| `neuron_utils.py` | 12 |
| `skeletal_distance_ops.py` | 8 |
| `neuron.py` | 6 |
| others | ≤3 each |

`spine_utils` is the largest holdout and has its own plan
([docs/notes/SPINE_UTILS_PLAN.md](docs/notes/SPINE_UTILS_PLAN.md)) — with an honest caveat: the surface
is a long thin tail (~1–2 calls per function, mostly L1 trivia), so the call count overstates the
structural payoff, and spines are not part of the segmentation pipeline the rest of this work
optimizes. Confirm the priority before starting.

## Invariants

- The layer lives in **`neurd/`** and wraps or replaces `tu.*` usage. It does **not** edit the sibling
  `mesh_tools` — see the rationale in [PERF.md](PERF.md).
- Migrations are **one call site per commit, golden-gated**, never bulk: `neuron.py`, `spine_utils` and
  `soma_extraction_utils` are consumed pipeline-wide.
- Keep cosmetic/structural commits separate from logic changes.

## Verification

Seeded and thread-pinned (`OPENBLAS/MKL/OMP_NUM_THREADS=1`), `parameters.params.use("h01")`, private
CWD → `segmentation_pipeline` → `tests/tools/neuron_metrics.extract_metrics`. Two harness features are
worth keeping: **`face_idx_digests`** (per-limb/soma `mesh_face_idx` set digests — `extract_metrics`
does **not** cover the L2 map, so without this the gate is blind exactly where this layer works) and
**caller histograms** (robust to line shifts; use before designing, per C4).

The pipeline's only nondeterminism is `np.random.choice` in the waterfill (inside
`resolve_empty_conflicting_face_labels`), now behind `_correspondence_backend.py`'s rng seam. Seeding
makes runs deterministic **provided a change does not shift the RNG-draw sequence** — reordering
branch/piece iteration does, and then RNG-capture replay is needed.

Anchors, byte-identical (skeleton_length / branch_faces): small h01 `2530864375` `366280.26`/`291893`;
big `1830470325` `1648072.07`/`1323533`; stitch `2889815798` `6819002.31`/`6259362`. ⚠️ These are the
*byte-identical* anchors from the seeded harness; the committed
`tests/fixtures/neuron_baseline_2530864375.json` no longer reproduces exactly on a plain run — use
`neuron_gate.py --tolerance` for ordinary verification.
