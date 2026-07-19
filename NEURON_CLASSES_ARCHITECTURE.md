# neuron.py class architecture — investigation

Companion to `LOGIC_INVESTIGATION.md`, for the **class-rethink** direction: `neuron.py` (3184 lines)
defines 4 fat classes and their concerns are tightly interwoven. Goal of THIS doc: map the tangle and
find where to start separating "what we do with **meshes**" from "what we want from **neurons/branches**".
No code changed yet. (`Spine` is a 5th class, in `spine_utils.py:152`, not here.)

## The 4 classes (all property-heavy "god objects")

| class | lines | methods | @property | role |
|---|---|---|---|---|
| `Branch` | 573 | 57 | **50** | one non-branching skeleton+mesh piece (leaf) |
| `Limb` | 1220 | 56 | 28 | a limb = many branches + a concept-network (connectivity) |
| `Soma` | 172 | 7 | 4 | soma mesh (thin) |
| `Neuron` | 1030 | 95 | **67** | whole neuron = limbs + somas + aggregates |

The bulk is **computed/cached properties**, not logic — that's the "fat".

## Each class mixes FOUR concerns

1. **Mesh geometry** — pure trimesh queries (volume, area, center, submesh, faces, kdtree).
2. **Skeleton geometry** — endpoints, vectors, length, graph, upstream/downstream ordering.
3. **Connectivity** — the concept-network machinery (almost all in `Limb`).
4. **Biology / annotation** — spines, boutons, synapses, webs, axon/dendrite compartment labels.

### Branch's 50 properties, categorized (the leaf, most illustrative)
- **Mesh (~7):** `mesh_shaft`, `mesh_shaft_idx`, `mesh_center_x/y/z`, `mesh_volume`, `area`.
- **Skeleton (~25 — the DOMINANT chunk):** `endpoint_upstream/downstream` (+`_with_offset`, +`_x/y/z`),
  `skeletal_coordinates_*`, `skeleton_vector_upstream/downstream` (+`_extra_offset`),
  `skeleton_smooth_vector_*` (×4), `skeleton_smooth`, `skeleton_graph`, `endpoints_nodes`,
  `skeletal_length(_eligible)`.
- **Width (~6):** `width_array_*`, `width_upstream/downstream` (+`_extra_offset`).
- **Annotation (~12):** `n_spines/n_boutons/n_web`, `spine_density`, `*spine_volume*`,
  `min_dist_synapses_pre/post_upstream/downstream` (×4), `axon_compartment`.

**Core data a Branch actually holds** (from `__init__`): `{skeleton, mesh, width, mesh_face_idx, labels}`.
Everything else is derived and **cached as ~30 instance attributes** (the giant `dc_check(...)` copy path):
spines, boutons, webs, spine/bouton volumes, width arrays, and the skeleton-vector variants.

## Two concrete findings that reshape "where to start"

**F1 — "What we want from meshes" is a SMALL, clean API (~10 ops).** The classes' entire dependence on
mesh mechanics is: volume, area, center, submesh, faces/vertices/triangles_center, face-idx→parent map
(`tu.original_mesh_faces_map`), combine, mesh-equality. That's it. So a **mesh layer** is easy to name
and cleanly separable — but it's only ~7 of Branch's 50 properties, so separating *just* meshes is
**low-risk but also low-impact on class size**.

**F2 — the real "fat" is skeleton-directional combinatorics + annotations, not meshes.** Branch's biggest
chunk (~25 props) is the **upstream/downstream × extra_offset × smooth** combinatorial explosion of
skeleton-vector / endpoint / width accessors (~16 near-duplicate variants), plus ~12 biology-annotation
props. These are what make the class fat and hard to read — and they're computed by *other* util modules
(spine_utils, width calc, skeleton_utils) but **cached onto the geometry object**.

## Proposed layering (target mental model, not a plan yet)

```
Mesh layer      : ~10 pure trimesh ops (volume/area/center/submesh/face-map/…)   ← "what we do with meshes"
Skeleton layer  : ~7 sk.* ops (endpoints/vectors/length/graph/order/smooth)
Morphology      : Branch / Limb / Neuron = COMPOSE {mesh, skeleton, width, correspondence, connectivity}
                  and expose domain queries — delegating geometry to the two layers above
Annotation      : spines / boutons / synapses / compartments — attached to morphology,
                  produced by *_utils, NOT baked in as 12+ cached properties
```
The win is making Branch/Limb/Neuron **thin composers** over these layers instead of god-objects that
own mesh math + skeleton combinatorics + biology all at once.

## Where to start — options (ranked by risk/impact)

1. **(lowest risk, documentation-first)** Freeze the **mesh-layer API** (the ~10 ops) as an explicit
   contract/protocol — the concrete answer to "what we want from meshes." No behavior change; it's the
   seam every later step leans on. Good first move; small.
2. **(medium) Collapse the skeleton-directional combinatorics.** The ~16 `upstream/downstream ×
   extra_offset × smooth` Branch accessors are near-duplicates of a few parameterized operations
   (`endpoint(side, offset, smooth)`, `skeleton_vector(side, …)`). Factoring them into a small helper +
   thin properties would cut the biggest chunk of Branch's fat — but it touches heavily-used public
   properties (needs a downstream-usage audit + golden gate). Highest impact on readability.
3. **(bigger) Extract the annotation concern** — move spines/boutons/synapses/webs off Branch/Limb into
   an attached annotations object built by the *_utils. Large surface (consumed everywhere downstream),
   so a real project, not a first step.
4. **(biggest, later) `Limb` connectivity** — the concept-network machinery (~half of Limb's methods) is
   its own subsystem; separable but large. (Related: `concept_network_utils.py`, which the user has open.)

**Recommendation:** start with **#1** (name the mesh API — it's the tractable "start from meshes" the
user asked for and de-risks everything after), and in parallel scope **#2** with a downstream-usage
audit, since that's where the readability payoff actually is. Every change stays golden-gated
(small-h01 + `1830470325`), and neuron.py's classes are consumed pipeline-wide, so each step needs a
usage audit first — much higher blast radius than the preprocess_neuron refactors.

## Open questions
- CA1: full downstream-usage audit of Branch's skeleton-directional props (which are actually consumed?).
- CA2: are Limb's concept-network methods duplicative with `concept_network_utils.py` / `neuron_utils`?
- CA3: how much of the annotation caching is load-bearing (perf) vs incidental?
