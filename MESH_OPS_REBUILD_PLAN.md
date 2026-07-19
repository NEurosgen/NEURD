# Mesh-ops rebuild — executable plan (for a future session)

Self-contained handoff. Companion investigation: `MESH_OPERATIONS_LAYER.md` (the full map + the L2 worked
example) and `NEURON_CLASSES_ARCHITECTURE.md`. Branch: `refactor/simplify-preprocess-logic` (safe
checkpoint `fix/dense-limb-correspondence-and-cgal-fallback`).

> **STATUS — ✅ Phase 0 DONE (commit `e04fdb8`).** Two corrections vs the draft below:
> **(1) the module is `neurd/submesh_ops.py`, not `mesh_ops.py`** — `neurd/_mesh_ops.py` already exists
> (L6 decimate/poisson/fill_holes, prod-wired via `__init__.py`), so the L2 layer took a distinct name;
> tests live in `tests/unit/test_submesh_ops.py`. **(2) the `SubMesh` snippet below was buggy** — `sub()`
> used `self.face_idx[faces]` while `root_face_idx()` also folds the chain, double-applying indices
> (`A[A[B]]` not `A[B]`); the delivered code and the snippet here are corrected. 24 `tu.*`-equivalence
> tests are green on a synthetic 3-icosphere + small-h01 (both connectivities), incl.
> `original_faces == tu.original_mesh_faces_map` proven on the real mesh. **Next = Phase 1 migration
> (not started; stop-for-review).**

## Why (1-paragraph brief)
NEURD's essence is *take meshes → segment them*, but the mesh-op vocabulary is **63 scattered `tu.*`
(trimesh_utils, a fragile sibling dependency) calls** across `spine_utils`/`soma_extraction`/
`preprocess_neuron`/`neuron.py`. Rebuild it **bottom-up** as a clean NEURD-owned layer: a few elementary
primitives, then compose the narrow functions from them. The framework re-forms around the clean base,
and reliance on `mesh_tools` shrinks. **Linchpin:** `trimesh.submesh()` drops face-index provenance, so
the framework perpetually re-derives face↔parent mapping (`original_mesh_faces_map`, 10× uses, 103 lines
of fragile KDTree midpoint-matching). A **`SubMesh` value type that carries `face_idx` and composes**
removes that whole class of bookkeeping — structurally, not cosmetically.

## Scope of THIS plan — the L2 foundation ONLY (don't boil the ocean)
Rebuild the proven **partition + face-index** cluster first. Everything else (L1 accessors, L3 filters,
L7 verbs, neuron-class refactor) is deferred until this foundation is trusted.

Target replacements (see the L2 worked example in `MESH_OPERATIONS_LAYER.md`):
| `mesh_tools` fn (lines) | new `mesh_ops` fn | note |
|---|---|---|
| `tu.split` (138) | `split(mesh, connectivity)` | components→SubMesh, largest-first |
| `tu.split_significant_pieces` (129) | `split_significant(mesh, min_faces, connectivity)` | split + filter |
| `tu.largest_conn_comp` (19) | `largest_component(mesh, connectivity)` | `split(...)[0]` |
| `tu.connected_components_from_face_idx` (28) | `components_from_face_idx(sub)` | provenance auto-composes |
| `tu.original_mesh_faces_map` (103) | `SubMesh.root_face_idx()` (O(1)); KDTree fallback only for foreign meshes | the big win |
| (`connected_components(edges,…)` internal) | `connected_face_components(mesh, connectivity)` | primitive |

## Deliverable module: `neurd/submesh_ops.py` (delivered — see STATUS banner)

### Primitive 1 — `SubMesh` (the composing value type)
```python
@dataclass(eq=False)
class SubMesh:
    mesh: "trimesh.Trimesh"          # the extracted geometry
    face_idx: np.ndarray             # indices into the IMMEDIATE parent's faces (len == len(mesh.faces))
    parent: "trimesh.Trimesh | SubMesh"
    def sub(self, faces) -> "SubMesh":            # submesh-of-submesh; `faces` index into THIS mesh
        return SubMesh(self.mesh.submesh([faces], append=True, repair=False), np.asarray(faces), self)
    def root_face_idx(self) -> np.ndarray:        # fold the chain to the ORIGINAL (root) mesh
        if isinstance(self.parent, SubMesh):
            return self.parent.root_face_idx()[self.face_idx]
        return self.face_idx                      # parent is the raw root mesh; face_idx indexes it
    @property
    def n_faces(self) -> int: return len(self.face_idx)
```
Design notes: `face_idx` is always indices into the *immediate* parent; `root_face_idx()` folds the chain.
Handle empty (`face_idx=[]`) cleanly. Consider `from_mesh(mesh)` classmethod (identity SubMesh) and
equality by `root_face_idx` set.

### Primitive 2 — `connected_face_components(mesh, connectivity="vertices") -> list[np.ndarray]`
Face-index groups of connected components. **Two modes (must match `tu` exactly):**
- `"edges"`: faces sharing an **edge** → `mesh.face_adjacency` graph → connected components.
- `"vertices"`: faces sharing a **vertex** → replicates `tu.split_by_vertices` semantics (looser grouping).
  ⚠️ trimesh's `face_adjacency` is edge-based; the vertex mode needs a vertex→faces union-find. Get the
  semantics right by **equivalence-testing against `tu.split(..., connectivity="vertices")`**.

### Composites (each a few lines — see worked example)
`split`, `split_significant`, `largest_component`, `components_from_face_idx`, `original_faces(sub)`.

## Isolated testing — Phase 0 (ZERO pipeline risk, no meshlab, fast)
Test file: `tests/unit/test_submesh_ops.py`, run with `pytest tests/unit/test_submesh_ops.py -v`.

**Sample meshes (no pipeline needed):**
```python
import trimesh
from mesh_tools import trimesh_utils as tu
# synthetic: 3 disjoint boxes → known 3 components
boxes = [trimesh.creation.box().apply_translation([d,0,0]) for d in (0,50,100)]
synth = tu.combine_meshes(boxes)
# real: any neuron .off (small h01 fixture)
real = tu.load_mesh_no_processing("Applications/Tutorials/Auto_Proof_Pipeline/neuron_2530864375.off")
```

**Equivalence tests — the core of Phase 0 (prove `mesh_ops` == `tu.*` before ANY migration):**
For each function, assert identical output to the `tu.*` original on both meshes and both
connectivities:
- `connected_face_components` vs the component partition from `tu.split(...)` — equal **as sets of
  frozensets of face indices** (order-independent), and the count matches.
- `split` vs `tu.split` — same submeshes by `face_idx` set, **same largest-first order**, same n_faces.
- `split_significant` vs `tu.split_significant_pieces(significance_threshold=n)` — same kept set + order.
- `largest_component` vs `tu.largest_conn_comp` — same component.
- `components_from_face_idx(SubMesh(real, idx))` vs `tu.connected_components_from_face_idx(real, idx)` —
  same **parent-relative** index groups (this proves the provenance composition).
- `original_faces`: build `s = split(real)[0]` (a SubMesh) → `original_faces(s)` must equal
  `tu.original_mesh_faces_map(real, s.mesh)` (same parent face set). Proves the O(1) map == the KDTree map.

**Property tests (the composition law — no `tu` needed):**
- `SubMesh(SubMesh(m,A),B).root_face_idx() == A[B]` for random valid `A`,`B`.
- `submesh(m, s.root_face_idx()).faces` reproduces `s.mesh.faces` (round-trip).
- empty / single-component / all-one-piece edge cases don't crash.

Definition of done for Phase 0: `pytest` green; every composite proven equal to its `tu.*` twin on
synthetic + real meshes. **No production code touched yet** → nothing can break.

## Phased execution
- **Phase 0 (✅ DONE, commit `e04fdb8`, zero risk):** wrote `neurd/submesh_ops.py` +
  `tests/unit/test_submesh_ops.py`; `tu.*` equivalence proven (24 tests green). Committed.
- **Phase 1 (first migration, golden-gated):** swap the **safest, most self-contained** call sites first —
  e.g. a `tu.split_significant_pieces` / `tu.largest_conn_comp` call in `soma_extraction_utils` or
  `preprocess_neuron` → the `mesh_ops` equivalent. Do NOT start with the `branch_face_idx` data structure.
  Verify byte-identical goldens (below). One call site per commit.
- **Phase 2 (the real payoff):** migrate `preprocess_neuron`'s `branch_face_idx` threading (Areas A/C in
  `LOGIC_INVESTIGATION.md`) to carry `SubMesh` instead of raw index arrays — this is where
  `original_mesh_faces_map` / manual remaps get deleted. Invasive; do incrementally, golden-gated.
- **Later:** expand to L1 accessors / L3 filters / the neuron-class delegation.

## Golden gate (Phases 1+; recreate harness from memory)
The pipeline is deterministic run-to-run with `OPENBLAS/MKL/OMP_NUM_THREADS=1`, fixed `segment_id=12345`,
`parameters.params.use("h01")`, private CWD. Two anchors (recipe in memory `preprocess-limb-refactor`):
- **small-h01** (`neuron_2530864375.off`, ~160s): 1 soma / 5 limbs / 28 branches [3,3,4,6,12] /
  skeleton 366280.26 / branch_faces 291893.
- **big `1830470325`** (166 MB, ~18min): 3 limbs / 129 br [1,54,74] / skeleton 1648072.07 / faces 1323533.
Recreate `verify_refactor.py` (small) + `verify_big.py` (big) via `tests/tools/neuron_metrics.extract_metrics`;
each migration commit must reproduce these **byte-identical**. Stitch-path changes also need
`replay_stitch.py` (RNG-capture) — see memory.

## Concrete first tasks — ✅ ALL DONE (commit `e04fdb8`)
1. ✅ Created `neurd/submesh_ops.py` with `SubMesh` + `connected_face_components` (both connectivities).
2. ✅ Added composites: `split`, `split_significant`, `largest_component`, `components_from_face_idx`,
   `original_faces`.
3. ✅ Created `tests/unit/test_submesh_ops.py` with the equivalence + property tests above; `pytest` green.
4. ✅ Committed Phase 0 (module + tests only — zero pipeline risk).
5. ✅ STOPPED for review before Phase 1 (first migration).

## Risks / notes
- `mesh_tools` is a sibling dep — the new layer lives in **neurd/**, wraps/replaces `tu.*` usage; it does
  NOT edit `mesh_tools`.
- Get the `connectivity="vertices"` semantics exactly right (equivalence tests are the guard).
- `neuron.py`/`spine_utils`/`soma_extraction` are consumed pipeline-wide → migrations are one-call-site,
  golden-gated, never bulk.
- Keep cosmetic/structural commits separate from any logical change (repo convention).
