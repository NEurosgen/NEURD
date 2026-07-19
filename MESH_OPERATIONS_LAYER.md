# Mesh-operations layer — bottom-up foundation map

**Premise (user's framing):** NEURD's essence is *take meshes → segment them*. The mesh operations are
the **foundation**, not a peripheral API. Strategy = **bottom-up**: rebuild the primitive mesh ops as
clean, logical, NEURD-owned code; compose upward to the larger ones; the rest of the framework
(preprocess_neuron, soma_extraction, neuron classes, spine_utils) then re-forms around that clean base.
This also *is* the "minimize `mesh_tools`" goal — a clean owned layer replaces the scattered `tu.*`/`cu.*`
calls and isolates the fragile dependency behind one seam.

**Scale of the vocabulary:** the framework calls **63 distinct `tu.*` (trimesh_utils) functions** +
`cu`/`sk`/`m_sk`, spread across `spine_utils` (52 tu-calls), `soma_extraction_utils` (38),
`preprocess_neuron` (26), `neuron.py` (16), `neuron_utils` (13). That scattered surface is what we'd
consolidate.

## The mesh operations, ordered primitive → composite

### L0 — construct / IO  *(the base)*
`load_mesh_no_processing`, `empty_mesh`, `combine_meshes` (12×), `write_neuron_off`

### L1 — pure geometry accessors  *(one mesh → scalar/array; no side effects)*
`.faces` `.vertices` `.triangles_center` `.area` `.volume` `.bounds`;
`mesh_volume` (10×), `mesh_center_vertex_average` (7×), `face_area_mean/sum`, `mesh_size`,
`is_watertight`, `largest_hole_length`, `area_to_boundary_length_ratio`, `max_distance_betwee_mesh_vertices`

### L2 — partition / submesh  *(one mesh → sub-pieces + face-index bookkeeping)*  ⭐ the true "bottom"
`.submesh(face_idx)`, `split` (connected comps), `split_mesh_into_face_groups`, `largest_conn_comp`,
`split_significant_pieces` (7×);
**face-index algebra:** `original_mesh_faces_map` (10×), `connected_components_from_face_idx`,
`overlapping_vertices_from_face_lists`  ← this bookkeeping (submesh ↔ parent face indices) is the crux the
whole pipeline threads through; getting a clean abstraction here pays off everywhere.

### L3 — set ops / filtering  *(meshes → meshes)*
`subtract_mesh`, `remove_mesh_interior` (6×), `filter_meshes_by_containing_coordinates` (5×),
`filter_away_inside_meshes`, `filter_meshes_by_bounding_box_longest_side`, `filter_vertices_by_mesh`,
`bbox_mesh_restriction`, `filter_away_border_touching_submeshes_by_group`

### L4 — spatial / proximity  *(mesh ↔ points / other meshes)*
`ray_trace_distance` (4×), `ray_pyembree`, `width_ray_trace_perc`, `n_vertices_outside_mesh_bbox`,
`compare_meshes_by_face_midpoints` (3×, identity), `find_border_vertices`, `find_border_face_groups` (3×)

### L5 — connectivity  *(many meshes → graph)*
`mesh_list_connectivity`, `mesh_pieces_connectivity`

### L6 — repair / reconstruct
`fill_holes`, `poisson_surface_reconstruction`, `pymeshfix_clean`

### L7 — composite "verbs" *(the framework's core segmentation operations — built FROM L0–L6)*
`mesh_segmentation` (7×, CGAL/SDF), `split_mesh_by_closest_skeleton`, `skeletal_length_from_mesh`;
+ **out of `tu`:** `cu.mesh_correspondence_adaptive_distance` (skeleton↔faces),
`m_sk.skeletonize_mesh_largest_component` / `sk.…CGAL` (mesh→skeleton),
`sm.extract_soma_center` (mesh→soma). These are what `preprocess_neuron` orchestrates.

## Why bottom-up works here (and where the leverage is)

- **L2 (submesh + face-index algebra) is the true bottom.** Everything upward carries "which faces of the
  parent does this sub-piece own" — the `original_mesh_faces_map` / `branch_face_idx` threading that the
  preprocess/stitch code (and Areas A/C) wrestle with constantly. A clean, typed *SubMesh(parent,
  face_idx)* abstraction — that composes (submesh of submesh remaps indices for you) — would remove a
  whole class of manual index bookkeeping across the framework.
- Once L0–L2 are a clean owned module, L3–L5 become thin, and L7 (segmentation/correspondence) reads as
  composition of named ops instead of scattered `tu.*` calls — and the fragile `cu`/`m_sk` calls sit
  behind the L7 seam (the Area-E adapter idea, now as the *top* of a clean stack rather than a patch).

## Proposed first step (bottom-up)

Draft the **L1+L2 core** as a NEURD-owned module — the primitive geometry accessors + a composing
**SubMesh/face-index abstraction** — specified as a small clean API, with the current `tu.*` calls behind
it (thin wrappers at first, so nothing breaks). Then migrate one heavy consumer (e.g. the face-idx
threading in `preprocess_neuron` correspondence, which we already understand from Areas A/C) onto it,
golden-gated. That migration is the proof the foundation is right; the rest of the framework follows.

## Constraints / caveats
- `mesh_tools` is a sibling dep (can't edit here) — the clean layer lives in **neurd/** and wraps/replaces
  `tu.*` usage; it does not modify `mesh_tools`.
- `neuron.py` classes + `spine_utils` + `soma_extraction` are consumed pipeline-wide → every migration
  step needs a usage audit + the golden gate (small-h01 + `1830470325`). Big project; incremental.
- This supersedes the "isolate a small mesh API" framing in `NEURON_CLASSES_ARCHITECTURE.md` — the mesh
  layer is the *foundation to build up from*, not a leaf to extract.

## L2 worked example — proof the bottom-up rebuild is real (2026-07-19, read the mesh_tools impls)

Read the actual `trimesh_utils` implementations of the L2 cluster. They are a **composition tower buried
in defensive cruft**, and they all reduce to **~3 primitives**:

| mesh_tools fn | lines | what it really is |
|---|---|---|
| `split` | 138 | `face_adjacency → connected_components → submesh(each) → order-by-size` (+ duplicated type-checks, a no-op try/except, an assertion, commented code) |
| `split_significant_pieces` | 129 | `split` + filter `len(faces) ≥ threshold` + sort (+ a 4-way return-flag explosion) |
| `largest_conn_comp` | 19 | `split_significant_pieces(threshold=-1)[0]` (magic −1) |
| `connected_components_from_face_idx` | 28 | `split(mesh.submesh(face_idx))` then **manual remap** `[face_idx[k] for k in comps]` |
| `original_mesh_faces_map` | 103 | recover submesh→parent face indices by **KDTree-matching face midpoints** — needed only because `trimesh.submesh()` drops index provenance |

**Root cause the code is fat/fragile:** `trimesh.submesh()` returns a standalone mesh with **no link to
its parent's face indices**, so the framework perpetually re-derives that link — by geometric KDTree
matching (`original_mesh_faces_map`, 10× uses, threshold-fragile) or by hand
(`[face_idx[k] for k in …]`).

### Clean rebuild — ~3 primitives, everything else composes
```python
# ---- primitives ----
def connected_face_components(mesh, connectivity="vertices") -> list[np.ndarray]:
    """Face-index groups of the mesh's connected components (shared vertices or edges)."""
    # adjacency (vertex/edge) → connected_components → face-index arrays

@dataclass
class SubMesh:                      # a mesh that REMEMBERS its provenance and COMPOSES
    mesh: "Trimesh"                 # the geometry
    face_idx: np.ndarray           # indices into `parent`
    parent: "Trimesh | SubMesh"
    def sub(self, faces) -> "SubMesh":         # submesh-of-submesh: indices remap automatically
        return SubMesh(self.mesh.submesh([faces]), self.face_idx[faces], self)
    def root_face_idx(self) -> np.ndarray:     # collapse the whole chain to the ORIGINAL mesh
        return self.parent.root_face_idx()[self.face_idx] if isinstance(self.parent, SubMesh) else self.face_idx

# ---- composites (each a few lines) ----
def split(mesh, connectivity="vertices"):
    comps = connected_face_components(mesh, connectivity)
    comps.sort(key=len, reverse=True)                       # largest-first
    return [SubMesh(mesh.submesh([c]), c, mesh) for c in comps]

def split_significant(mesh, min_faces, **kw):   return [s for s in split(mesh, **kw) if len(s.face_idx) >= min_faces]
def largest_component(mesh, **kw):              return split(mesh, **kw)[0]
def components_from_face_idx(sub: SubMesh):     return [sub.sub(c) for c in connected_face_components(sub.mesh)]
def original_faces(sub: SubMesh):               return sub.root_face_idx()      # O(1); KDTree-match only for
                                                                                # meshes from OUTSIDE the pipeline
```

**Result:** ~420 mesh_tools lines → **3 primitives + a small `SubMesh` type + ~5 tiny composites**, and the
most-used, most-fragile op (`original_mesh_faces_map`'s geometric re-matching) **largely disappears** —
provenance is carried by construction, not re-derived. This is precisely "elementary functions → assemble
the narrow ones", and it *shrinks* `mesh_tools` reliance (Goal 1) at the same time.

**Migration shape (non-breaking):** put these in a NEURD-owned `mesh_ops` module; internally the composites
can call the existing `tu.*` at first (thin), then get replaced primitive-by-primitive. The real leverage
lands when `preprocess_neuron`'s `branch_face_idx` threading (Areas A/C) is switched to `SubMesh` — the
manual remapping there is exactly what the type removes. Each step golden-gated (small-h01 + `1830470325`).

## Open questions
- MO1: what's the right *SubMesh/face-index* abstraction (composition semantics: submesh-of-submesh index
  remap; equality; parent linkage)? — **sketched above; the `root_face_idx` composition law is the core.**
- MO2: which heavy consumer to migrate first as the proof (preprocess_neuron correspondence face-idx vs
  soma_extraction vs spine_utils)?
- MO3: how much of the 63-fn surface is genuinely needed vs incidental (some may be one-off / dead)?
