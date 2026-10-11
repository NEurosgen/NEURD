# Spine processing: current boundaries and remaining cleanup

Spine detection is supported by the public
[segmentation_pipeline](../../neurd/segmentation_pipeline.py) through
`RunConfig(calculate_spines=True)`. It is off by default for API compatibility;
the raw batch CLI enables it by default. Named H01 refinement policies are opt-in.
Earlier notes describing this module as absent from the raw pipeline are obsolete.

## Current structure

`spine_utils` retains the legacy spine object/helpers and the branch/neuron
orchestration. Primary refinement and final filters are named private stages;
final filters are reused for initial candidates, trim/growth fallback and
remaining-head proposals. Result selection keeps volume/QC alignment and the
public volume container types. Mesh-mask and volume-quality outputs remain
transactional during filtering; refinement diagnostics can be populated before
the final filters complete, preserving the earlier behavior.

The focused owned modules have distinct responsibilities:

| Module | Responsibility |
|---|---|
| `cgal_labels` | Native labels/SDF seam, ordered mesh assembly, opt-in nested phase timing |
| `shaft_paths` | Guarded exact ordered path queries on an immutable graph snapshot |
| `spine_refinement` | Parent-face growth/trimming and head decisions |
| `spine_volume` | Strict volume estimates and protected in-memory quality handoff |
| `face_provenance` | Validation of supplied ordered child-to-parent maps |

These boundaries are useful and should remain. `submesh_ops` does not implement
`subtract_mesh`; migration to an owned layer must use an actual tested equivalent.

## Remaining work

- Audit legacy spine attributes, visualization hooks and downstream helpers against
  supported callers before removal. Keep explicit compatibility for user notebooks.
- Simplify the branch-selection query dependency after auditing string dispatch;
  the ordinary path needs width/face-count selection, but custom queries exist.
- Consider a small result structure only if it reduces synchronization beyond the
  current selection helper. Avoid a configurable filter framework.
- Migrate an upstream mesh operation only after observing its live callers and
  establishing ordered geometry/mask equivalence for that operation.

## Validation

Use the ordinary and refined native branch gates plus a spine-enabled raw gate.
Counts alone are insufficient: check ordered parent masks, exact geometry where
possible, overlap and volumes, and keep thresholds/filter/RNG order unchanged.
Pin configuration, seed, native provider and numerical thread settings for pairs.
Conditional cached-label replay isolates post-CGAL changes; it is not a replay
of native or historical RNG. Tests and evidence are described in
[the readability plan](REFACTOR_PLAN.md) and
[the public release plan](PUBLIC_RELEASE_PLAN.md).
