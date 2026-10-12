# Spine-volume closure defect and correction — 2026-09-08

## Finding and impact

Open input spine meshes are expected after cutting them from dendritic shafts.
They must not be rejected merely because they are open. The defect concerns the
surface accepted **after** attempted hole filling.

The descriptor extractor loads already-scaled meshes, constructs `Spine`, and
calls `calculate_spine_attributes` → `volume_from_spine` → `mesh_tools.mesh_volume`.
The descriptor-volume call does not apply an extra voxel/acquisition multiplier.
The dataset parameter sets govern segmentation/filtering thresholds; they do not
explain translation dependence of a fixed surface.

In mesh_processing_tools 1.0.4, the default `allow_if_no_border_verts=True`
permits a non-watertight mesh with positive volume if no edge has only one
incident face. It can therefore accept nonmanifold edges with more than two
incident faces. The resulting oriented volume can depend on absolute position.

Two H01 examples reproduce the original float64 descriptor volumes exactly:

| Spine | Original estimate, nm³ | Same processed faces translated to origin, nm³ |
|---|---:|---:|
| `neuron_3503540481_limb_000_branch_089_spine_009` | 2,197,020,746.458333 | 1,896,122.458333 |
| `neuron_5319188363_limb_000_branch_196_spine_018` | 1,939,998,946.041667 | 209,804.166667 |

No scale, shape, or face indices changed. Centering alone is not a repair of
nonmanifold topology; the translated numbers are not anatomical ground truth.

The existing 19-feature QC panels contain 72/41,631 H01 and 3/556,771 Minnie65
spines with stored volume above the axis-aligned bounding-box volume by >1%.
All 75 were confirmed against meshes and failed translation invariance after
upstream closure. They are rare gross failures, not an estimate of every volume
error. The 475-mesh verification also includes 200 randomly chosen nonflagged
objects per corpus; all 475 input meshes were open.

The H01 local raw V/A correlation excess changes from 0.1245 (95% neuron CI
[-0.1172, 0.5636]) to 0.0088 [-0.0083, 0.0235] after removing only the two
extreme objects, or 0.0127 [0.0003, 0.0259] after removing the 72 bbox flags.
All 64 neurons remain. These are diagnostic mask sensitivities, not replacement
primary results. Log-volume changes much less: 0.0122 → 0.0093 for the 72-object
mask. Thus the effect is important for a particular untransformed descriptor;
it does not invalidate every result or establish a zero biological effect.

## Implemented correction

`neurd/spine_volume.py` owns `centered_strict_closure_v1`:

1. Work on a centered copy; preserve the original input geometry and centroid.
2. Accept a closed input or fan-filled candidate only after strict watertightness,
   winding, positive finite volume, input-bounds and translation checks.
3. If fan closure fails, try the existing intended convex-hull fallback, also
   centered and validated. Record `method=convex_hull` and the rejection reason.
4. If no candidate passes, return a missing volume (NaN through the scalar API),
   never an arbitrary positive volume or an unlabelled zero.

Whole-spine, head/neck volume calls and spine/shaft volume filters now use
this helper. The extractor keeps the established 35-column order and adds
`<stem>_volume_qc.json` with algorithm and spine/head/neck quality records.
Missing dependencies still raise; they do not silently select a different method.

A convex-hull volume is an explicit geometric approximation. It can overestimate
concave anatomy and must not be called recovered true volume. On the random
nonflagged samples, this stricter algorithm changes volumes by >10% for 53/200
H01 and 20/200 Minnie65 objects. This is sensitivity to reconstruction method,
not an independently verified error rate. Overall prevalence cannot be inferred
from the bbox flag alone.

## Validation and migration

The focused tests cover open and closed inputs, nonmanifold closure without
border edges, invalid geometry, large translations, isotropic and anisotropic
scaling (including [8,8,33]), unchanged inputs, and protection against silently
resuming old descriptor outputs without QC. The proposed helper was also checked
on all 475 audit meshes: input geometry was unchanged and estimates were stable
under translation. Full historical head/neck segmentation was not rerun.

Run focused tests from this repository with the required research interpreter:

```bash
python -m pytest tests/unit/test_spine_volume.py tests/unit/test_spine_descriptor_resume.py -q
```

For new extraction, use a **new versioned output directory** and the intended
`--data-type`. Resume now requires versioned complete QC, the volume algorithm,
dataset/configuration and input/output SHA-256 hashes. Missing, malformed, legacy
or mismatched records are queued for regeneration; `--overwrite` forces it for all
selected inputs. Computation and staging preserve the old files until publication
starts. The completion marker is published last, so interrupted replacement is
never accepted as complete. Preserve a backup or use a new directory when keeping
the original descriptors matters. Current source changes do not repair existing
`.npy`/graph caches.
Rebuild affected volume-derived ratios and study-specific cohorts together;
changed volume thresholds can change which spines are retained.

Before promoting regenerated descriptors to an article result, compare closure
methods on the complete population (including below-threshold meshes), inspect
representative repairs, assess missingness/fallback selection, and rerun the
relevant analyses with frozen old/new provenance. Preserve the old scientific
bundle and report the migration as a subsequent version.

Audit and validation artifacts are in the research repository:
`GIT_Graph_refactor/artifacts/morphology/volume_geometry_audit_20260908/` and
`GIT_Graph_refactor/artifacts/morphology/volume_impact_20260908/`.
