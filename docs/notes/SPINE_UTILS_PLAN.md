# `spine_utils` — plan for a future session

Companion: [../../MESH_OPS.md](../../MESH_OPS.md) (the owned layer, its status and corrections).

## Why this target
`spine_utils.py` is the **largest remaining `mesh_tools` holdout**: ~48 `tu.*` call sites across ~39
distinct functions — roughly a third of the framework's whole remaining `tu` surface. Everything else
reachable from the segmentation pipeline has been migrated or shown unreachable.

> ⚠️ Two premises below have since changed. (1) Phase 2 was already *entered* without the Phase-0
> harness this plan calls mandatory: spine segment connectivity was migrated to
> `submesh_ops.pieces_adjacency` (commit `cc81169`, −23% wall on the 187 MB h01), gated on
> pure-function equivalence instead of a spine baseline. (2) The claim further down that `submesh_ops`
> already covers `subtract_mesh` is **wrong** — there is no such function in `submesh_ops`, so that
> "near drop-in swap" does not exist.
>
> ⚠️ Also note **spine count is nondeterministic** (22/16/20 observed on unchanged code), so the
> "reproduce the baseline byte-identical" verification below cannot work as written. Gate spine work on
> pure-function equivalence, or on a segmentation-level SDF-correlation oracle.

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
The harnesses are now committed under `tests/tools/` (`neuron_gate.py`, `soma_gate.py`,
`benchmark_neuron.py`, `batch_bench.py`, and `wall_profile.py`, which already has spine phase markers
and a `--spines` switch). Extend one of them with a spine mode:
- build the neuron with `calculate_spines=True`;
- extract spine metrics alongside the existing ones — reuse `process_all_neurons`'s `_iter_spines` /
  `spines_per_branch` / `total_spines` rather than inventing new ones;
- establish baselines on the anchors. **Expect small-h01 to yield 0 spines** (it produced none in the
  probes), so the working anchor is probably big `1830470325` — iteration cost goes up accordingly.
- **Determinism was checked and the answer is no:** spine count varies run to run on unchanged code
  (22/16/20 observed). So a spine-count baseline is not a usable gate. What *is* gateable: pure-function
  equivalence for whatever is migrated, and an SDF-correlation oracle at the segmentation level (the
  shape of `tests/integration/test_cgal_segmentation_oracle.py`). Build that instead of a count
  baseline.

### Phase 1 — coverage probe under the spine driver
Run the caller histogram over `spine_utils`'s `tu` surface with spines enabled, and record which of the 39
functions actually fire and how often. **Do not design anything before this** — this session burned real
effort designing owned replacements for `original_mesh_faces_map` / `subtract_mesh` /
`compare_meshes_by_face_midpoints` that turned out to have no live caller at all.

### Phase 2 — migrate what fires, one gated commit at a time
Only then pick targets. Prefer ops already owned in `neurd/submesh_ops.py`: `largest_conn_comp`,
`connected_components_from_face_idx` and `split_mesh_into_face_groups` are covered and would be near
drop-in swaps. **`subtract_mesh` is NOT** — `submesh_ops` has no equivalent, so that one needs writing
from scratch, not swapping. (Already migrated this way: spine segment connectivity →
`submesh_ops.pieces_adjacency`, `cc81169`.)

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
⚠️ The original plan here — establish `total_spines` / `spines_per_branch` baselines and require every
migration to reproduce them byte-identically — **does not work**: spine count is nondeterministic on
unchanged code. Gate instead on (a) pure-function equivalence for each migrated op, against the
pre-migration implementation kept as an oracle (the pattern in `test_cfc_vertex_faces_free.py`), and
(b) the non-spine structure metrics below, which *are* stable within tolerance.

Existing anchors and metrics (unchanged, still apply to the non-spine half):
small-h01 `366280.26`/`291893`; big `1830470325` `1648072.07`/`1323533`; stitch `neuron_2889815798`
`6819002.31`/`6259362`.
