# Mesh-ops rebuild — Phase 0/1 (✅ DONE — status stub)

Companions: `MESH_OPERATIONS_LAYER.md` (the full map + L2 worked example),
`MESH_OPS_PHASE_B_PLAN.md` (provenance-through-correspondence: status + remaining work).

## Why
NEURD's essence is *meshes → segment them*, but the mesh-op vocabulary was **63 scattered `tu.*`**
(trimesh_utils, a fragile sibling dep) calls. **Linchpin:** `trimesh.submesh()` drops face-index
provenance, so the framework perpetually re-derived face↔parent mapping (`original_mesh_faces_map`, a
fragile 103-line KDTree midpoint match). A **`SubMesh` value type that carries `face_idx` and composes**
removes that whole class of bookkeeping *structurally*.

## What was built — DONE
- **`neurd/submesh_ops.py`** (named thus because `neurd/_mesh_ops.py` already exists for L6
  decimate/poisson/fill_holes). The composing value type + partition layer:
  - `SubMesh(mesh, face_idx, parent)` — `face_idx` indexes the *immediate* parent; `root_face_idx()`
    folds the chain to the original mesh (O(1)); `.sub(faces)`, `.n_faces`.
  - `connected_face_components(mesh, connectivity)` (both `"edges"` and `"vertices"` modes) + composites
    `split`, `split_significant`, `largest_component`, `components_from_face_idx`, `faces_by_match`,
    `combine`, `face_groups_by_label`, `split_into_face_groups`.
- **`tests/unit/test_submesh_ops.py`** — `tu.*`-equivalence + composition-law tests (36 green), proven on
  a synthetic 3-icosphere + small-h01 (both connectivities), incl. `original_faces ==
  tu.original_mesh_faces_map` on the real mesh. Run: `pytest tests/unit/test_submesh_ops.py -v`.
- **Phase 1 migrations** (all golden-gated byte-identical): the gateable split-based L2 call sites in
  `soma_extraction_utils` / `preprocess_neuron` (commits `09573d7`, `fe1f3ad`, `87bd943`, `023e502`,
  `422710b`, …). The simple-swap queue is **exhausted**.

## Phase 2 = provenance through correspondence → see `MESH_OPS_PHASE_B_PLAN.md`
Carrying `SubMesh` through the `branch_face_idx` threading (where the manual remaps live) is mostly done
there; two low-value items remain (stitch remaps + `Branch.mesh_face_idx` as a property). Later: L1
accessors / L3 filters / neuron-class delegation.

## Invariants (don't relitigate)
- The layer lives in **neurd/**, wraps/replaces `tu.*` usage; it does **NOT** edit the sibling
  `mesh_tools`.
- Migrations are **one call site per commit, golden-gated**, never bulk — `neuron.py`/`spine_utils`/
  `soma_extraction` are pipeline-wide.
- Get `connectivity="vertices"` semantics exactly right (the equivalence tests are the guard).
- Keep cosmetic/structural commits separate from logic changes.

## Golden gate (recreate from memory `preprocess-limb-refactor`)
Deterministic with `OPENBLAS/MKL/OMP_NUM_THREADS=1`, `parameters.params.use("h01")`, private CWD, seeded.
Anchors, byte-identical via `tests/tools/neuron_metrics.extract_metrics`:
- **small-h01** `neuron_2530864375.off` (~160s): 1 soma / 5 limbs / 28 br [3,3,4,6,12] / skeleton
  366280.26 / branch_faces 291893.
- **big `1830470325`** (~18min): 3 limbs / 129 br [1,54,74] / skeleton 1648072.07 / faces 1323533.
Stitch-path changes also need RNG-capture replay (`replay_stitch.py`).
