# `spine_utils` — plan for a future session

Companions: `MESH_OPS_PHASE_B_PLAN.md` (status + corrections), `MESH_OPERATIONS_LAYER.md`.
Branch: `refactor/simplify-preprocess-logic`.

## Why this target
`spine_utils.py` is the **largest remaining `mesh_tools` holdout**: **51 `tu.*` calls across 39 distinct
functions** — roughly a third of the framework's whole remaining `tu` surface. Everything else reachable
from the segmentation pipeline has been migrated or shown unreachable.

## The pivotal fact (measured, not assumed)
**Spine detection does not run in `segmentation_pipeline`.** `Neuron.__init__` takes
`calculate_spines=False` by default and only calls `spu.calculate_spines_on_neuron(self)` when it is True
(`neuron.py`, the `if calculate_spines:` branch). `segmentation_pipeline` constructs
`neuron.Neuron(mesh=mesh, segment_id=segment_id)` — **without the flag** — so the whole `spine_utils` surface
is cold under the current golden gate.

Evidence: caller-histogram probes for 10 `tu` ops (`original_mesh_faces_map`, `mesh_segmentation`,
`largest_conn_comp`, `connected_components_from_face_idx`, `split_mesh_into_face_groups`, `combine_meshes`,
`mesh_list_connectivity`, `remove_mesh_interior`, `filter_meshes_by_containing_coordinates`, `mesh_volume`)
on **both** anchors: **zero `spine_utils` callers**. `mesh_segmentation` fires only from
`soma_extraction_utils` (soma SDF), never from spine code.

⇒ Any migration here is ungateable until a spine-path driver exists. **That driver is step one.**

## The driver already exists — no new machinery needed
| piece | where |
|---|---|
| enable spines | `neuron.Neuron(mesh, calculate_spines=True)` — exactly what `process_all_neurons.py` does |
| entry into `spine_utils` | `spu.calculate_spines_on_neuron(neuron_obj)` (called from `Neuron.__init__`) |
| ready-made metrics | `process_all_neurons.py`: `_iter_spines`, `spines_per_branch`, `total_spines` |
| an existing baseline artifact | `tests/integration/_seg_output_spines/neuron_1830470325/summary.json` (real per-branch counts) |
| CGAL SDF segmentation (what spine detection rests on) | `neurd/_cgal_segmentation.py`; oracle test `tests/integration/test_cgal_segmentation_oracle.py` |

Memory also records that spines were restored on this branch (real CGAL SDF segmentation, 0 → 165 spines on
big H01), so the path is known to work.

## Plan

### Phase 0 — spine harness (do this first, nothing else is verifiable without it)
Extend the gate harness (`scratchpad/gate_run.py`; **consider committing it to `tests/tools/` first — it has
already been wiped once by the system**) with a spine mode:
- build the neuron with `calculate_spines=True`;
- extract spine metrics alongside the existing ones — reuse `process_all_neurons`'s `_iter_spines` /
  `spines_per_branch` / `total_spines` rather than inventing new ones;
- establish baselines on the anchors. **Expect small-h01 to yield 0 spines** (it produced none in the
  probes), so the working anchor is probably big `1830470325` — iteration cost goes up accordingly.
- **Check determinism explicitly.** The pipeline's only known nondeterminism is the waterfill
  `np.random.choice`; CGAL SDF segmentation may add its own. Run the spine baseline **twice** before
  trusting it.

### Phase 1 — coverage probe under the spine driver
Run the caller histogram over `spine_utils`'s `tu` surface with spines enabled, and record which of the 39
functions actually fire and how often. **Do not design anything before this** — this session burned real
effort designing owned replacements for `original_mesh_faces_map` / `subtract_mesh` /
`compare_meshes_by_face_midpoints` that turned out to have no live caller at all.

### Phase 2 — migrate what fires, one gated commit at a time
Only then pick targets. Prefer ops already owned in `neurd/submesh_ops.py` — a quick look at the surface
suggests several are already covered (`largest_conn_comp`, `connected_components_from_face_idx`,
`split_mesh_into_face_groups`, `subtract_mesh`) and would be near drop-in swaps.

## Honest caveats — decide after Phase 1, not before
- **The surface is a long thin tail**: 51 calls over 39 distinct functions, i.e. ~1–2 calls each. There is
  little here to "decompose into reusable primitives" the way `split` / `combine` were — expect mostly L1
  trivia (`mesh_volume`, `face_area_mean`, `empty_mesh`) plus a handful of L2/L3 ops. **The call count
  overstates the structural payoff.** Phase 1 should be followed by an explicit go/no-go.
- **Value question**: spines are not part of the single-soma segmentation pipeline that the rest of this
  work optimises. Owning their `tu` surface is a dependency-reduction goal, not a correctness or
  performance one for that pipeline. Worth confirming the priority before Phase 2.
- **The spine path may be fragile**: `neuron.py` carries a comment about a latent `NameError` in the
  `calculate_spines` branch that was only fixed recently. Treat the first spine run as a smoke test.
- **Cost**: with big as the anchor, every gate iteration is ~20+ min plus spine detection on top. Use the
  fast/full split that worked here — cheap checks per step, one full run at the end.

## Verification
Baselines to establish in Phase 0 (there are none for spines yet beyond the stale artifact):
`total_spines` and `spines_per_branch` per limb, on the spine anchor, reproduced twice. Then every migration
commit must reproduce them **byte-identical**, exactly as the mesh-ops work was gated.

Existing anchors and metrics (unchanged, still apply to the non-spine half):
small-h01 `366280.26`/`291893`; big `1830470325` `1648072.07`/`1323533`; stitch `neuron_2889815798`
`6819002.31`/`6259362`.
