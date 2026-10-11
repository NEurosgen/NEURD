# Spine processing and experimental refinement

NEURD supports two workflows: decomposing a raw single-neuron mesh, and
resegmenting spines on already exported H01 branches. Both use nanometre input
coordinates. The `microns` parameter-set name refers to Minnie65 processing;
it does **not** mean that mesh coordinates should be supplied in micrometres.
Areas and volumes are nm² and nm³.

Geometric refinement is opt-in. It changes which existing parent triangles
belong to a spine; it does not smooth the parent surface or create anatomical
geometry. The strict volume estimator is active independently of refinement.

## Raw-neuron API and runners

[`RunConfig`](../neurd/run_config.py) is shared by
[`segmentation_pipeline`](../neurd/segmentation_pipeline.py) and the raw mesh
runners. A spine-enabled configuration requires the compiled CGAL segmentation
extension and rejects the Python stand-in before neuron construction.

| Setting | Values | Behavior |
|---|---|---|
| `dataset` | `microns`, `h01` | Selects dataset parameters; input units remain nm |
| `calculate_spines` | `False`, `True` | The API defaults to `False`; raw CLI runners default to enabled |
| `refinement` | `off` | Ordinary CGAL extraction and existing branch filters |
| `refinement` | `h01_current` | H01-only experimental head growth, shaft trimming/splitting and fragment filtering |
| `refinement` | `h01_smooth_heads` | Adds screening and repair of smooth incomplete heads; also H01-only and experimental |
| `sdf_profile` | `reference`, `fast` | 25 and 12 SDF rays, respectively |
| `sdf_rays` | Positive integer | Explicit override; recorded as a custom profile when it differs from the selected profile |

The named H01 policies require `calculate_spines=True`. Their resolved settings
match [`h01_spines_current.json`](../configs/h01_spines_current.json) and
[`h01_spines_smooth_heads.json`](../configs/h01_spines_smooth_heads.json).
These names describe selectable development policies, not a quality ranking.
Minnie65 accepts the `off` policy; H01 refinement thresholds have not been
validated for Minnie65.

```python
import neurd  # Install compatibility patches before importing mesh_tools.
from neurd.run_config import RunConfig
from neurd.segmentation_pipeline import segmentation_pipeline
from mesh_tools import trimesh_utils as tu

mesh = tu.load_mesh_no_processing("neuron.off")
config = RunConfig(
    dataset="h01",
    calculate_spines=True,
    refinement="h01_current",
    sdf_profile="reference",
)
result = segmentation_pipeline(mesh, segment_id=123, max_somas=1, config=config)
print(result.run_config)  # Includes resolved rays and refinement parameters.
for limb in result.limbs:
    for branch in limb.branches:
        meshes, volumes = branch.spines, branch.spines_volume
        face_masks = branch.spines_face_idx  # Branch-local IDs, aligned with meshes.
```

Supply either `config` or individual configuration arguments. For new calls,
select the dataset explicitly. A compatibility call without a dataset retains
the currently selected `parameters.params` dataset, disables spines and uses
the reference ray profile. Dataset parameters and CGAL rays are process globals;
use separate processes when running different configurations concurrently.

The same settings are available in the raw mesh commands:

```bash
# Fresh whole-neuron decomposition, including ordinary Minnie65 spines.
python process_all_neurons.py raw_meshes output/minnie-reference \
    --dataset microns --spines --refinement off --sdf-profile reference

# Fresh H01 decomposition with an explicitly selected experimental policy.
python process_all_neurons_parallel.py raw_meshes output/h01-current \
    --dataset h01 --spines --refinement h01_current \
    --sdf-profile reference --max-workers 2
```

Use `--no-spines` for decomposition without spine detection. `--sdf-profile fast`
selects 12 rays. Keep 12-ray and 25-ray results as distinct configurations and
compare them on the intended inputs before adopting the fast profile for a
scientific analysis; the profile name does not establish equivalent detection.

## Exported H01 branch resegmentation

[`scripts/resegment_h01.py`](../scripts/resegment_h01.py) reads an existing
`neuron_*/limb_*/branch_*/` export containing `branch_mesh.off`,
`branch_skeleton.npy` and optional archived `spines/`. It keeps the existing
soma/limb/branch decomposition. It does not run raw-neuron soma extraction or
produce a new decomposition of the original full mesh.

The source and output roots must be separate, non-nested directories. Start
with a limited selection, then resume the same output with unchanged settings:

```bash
python scripts/resegment_h01.py \
    --source exported_h01 --output output/h01-current-v1 \
    --config configs/h01_spines_current.json \
    --rays 25 --workers 2 --limit 4 --plan-only

python scripts/resegment_h01.py \
    --source exported_h01 --output output/h01-current-v1 \
    --config configs/h01_spines_current.json \
    --rays 25 --workers 2 --limit 4
```

`--keys branches.json` selects a JSON list of exact branch keys, for example
`["neuron_123_limb_000_branch_003"]`. The planner inventories the source export
and records a frozen runtime snapshot. Each isolated worker validates its
inputs, stages its outputs, and publishes a branch only after completion checks.
Previously empty branches still need to pass NEURD's branch eligibility check;
an unresolved width is reported as a failure rather than converted to an
apparently successful empty result.

Consult `last_run_summary.json` and `last_run.json` for completed, failed,
interrupted and unattempted branches. A successful limited selection does not
mean that the full dataset is complete. Failed selections produce a nonzero
exit status, and successfully completed branches can be reused on resume.

## Skeleton policy

The **original branch skeleton** controls endpoint filtering even when an
auxiliary geometric axis is available. The H01 branch command checks skeleton
coverage and may build a better surface-based axis by default. It keeps the
original exported graph unless replacement is explicitly enabled.

| H01 branch option | Effect |
|---|---|
| Default | Check coverage/build an auxiliary axis when needed; retain the original exported skeleton |
| `--no-skeleton-repair` | Disable rebuilding the branch axis |
| `--auxiliary-heads` | Experimentally allow the auxiliary axis to guide head completion |
| `--experimental-skeleton-replacement` | Allow an anchored replacement only after topology and coverage checks |

The replacement guard requires both the original graph and rebuilt axis to be
connected trees; disconnected or cyclic inputs retain the original graph.
It preserves the labelled connections between original ends and junctions.
Changing only sampling along degree-2 paths is allowed. Matching
endpoint/junction coordinates and degrees alone would be insufficient.
Replacement also has connector-distance and coverage limits; enabling it does
not guarantee that a replacement occurs. It cannot be combined with
`--no-skeleton-repair`.

The export retains `branch_skeleton_original.npy`, writes the accepted graph
to `branch_skeleton.npy`, and writes `segmentation_axis.npy` when an auxiliary
axis exists. `result.json` records the endpoint/head axes actually used and
whether topological replacement occurred. Limb skeletons combine accepted
branch graphs and retain original axes for unprocessed branches; their
`skeleton_provenance.json` reports incomplete coverage.

## Face ownership, volumes and completion provenance

Refinement works with exact parent-face indices. Growth reserves accepted
neighbours and owner seeds; trimming may split a candidate into multiple
outputs. Changed candidates run through the ordinary final branch filters,
and rejected repairs retain or roll back to their allowed original candidates.

Whole-neuron processing stores `branch.spines_face_idx` for ordinary and refined
spines. It contains int64 branch-local masks in returned mesh order; `[]` means
a selected branch produced no final spines, while `None` means provenance is
unavailable because the branch was not processed. Direct branch callers can
pass `spine_face_indices=[]` to receive the same masks without changing the
legacy mesh/volume return value. These IDs retain distinct source occurrences
when a parent contains geometrically duplicate triangles. Within each mask,
IDs follow the returned mesh's face rows, including when refinement retains an
original mesh whose rows have nonmonotonic parent IDs. Raw-neuron exports
include the masks in `spine_faces.npz`; geometry consistency checks also detect
shared surface across different spines independently of their numeric IDs.
Raw masks additionally record their schema, branch-face scope, parent face count
and exact spine filenames. Raw mesh exports do not serialize the original
`branch.spines_volume` arrays; compute descriptors/QC separately when those values
are needed alongside exported meshes.

An H01 branch result contains:

| File | Meaning |
|---|---|
| `spines/spine_*.off` | Final spine surfaces in nm |
| `spine_faces.npz` | Concatenated parent `faces` and `offsets`, plus raw-owner `raw_ids` and split `part_ids` |
| `refinement.json` | Candidate actions, rejected proposals and final output mappings |
| `result.json` | Input/output hashes, resolved run signature, axes, owner records, volumes and checks |

The exporter checks mesh/mask/volume alignment, exact equality to the indicated
parent triangles and no surface overlap. Spine numbers are reassigned; matching
an old and new `spine_003` by filename is not reliable. Compare masks or geometry
on their common parent mesh.

H01 export carries volume quality from the final filter only for the identical,
unchanged in-memory mesh. The optional `spine_volume_quality=[]` collector leaves
the mesh/volume return value unchanged; records are process-local and reject
stale geometry. Scientific QC of saved OFF files must independently reload and
recompute volumes. Batch completion/resume validation verifies hashes and inventory.

Volumes use `centered_strict_closure_v1`: the estimator centres a copy, attempts
validated closure and records an explicit convex-hull fallback when needed.
The input surface stays unchanged. Convex-hull volume is a geometric
approximation and can overestimate concave anatomy. Unavailable estimates are
missing values, not unlabelled zeros. See the
[volume investigation and migration notes](notes/SPINE_VOLUME_QC.md).

Descriptor extraction preserves the 35-column order and adds a versioned
`*_volume_qc.json` completion record. Resume verifies its algorithm, dataset,
configuration, input hash and descriptor/centroid output hashes. Missing,
legacy, malformed or mismatched records are recomputed; `--overwrite` also
recomputes verified completed results. Arrays stage before publication and the
completion marker is published last. Use a new versioned output directory to
preserve historical scientific bundles when changing the volume algorithm.
Worker failures leave an explicit failure log and return a nonzero command status.

H01 branch resume is stricter: changing code, dependencies/provider, ray count,
refinement, skeleton policy or the source inventory requires a **new output
directory**. Changed completed inputs or outputs are rejected. Keep the
manifest/runtime snapshot with its results; editing hashes to make a changed
run resume destroys their provenance. Raw-neuron batches also validate input,
configuration, code/dependency and completed output signatures.

Context `branch_mesh.off`, limb meshes/topology and soma directories in the H01
export are source links. Copying the output root alone does not create a
portable bundle. For sharing, copy the referenced context into an explicitly
versioned bundle, or provide a reproducible source-download recipe with units
and checksums.

## A small spine-positive integration gate

[`test_spine_branch_gate.py`](../tests/integration/test_spine_branch_gate.py)
processes a real exported branch using its original skeleton, available soma,
25 rays and refinement disabled. It requires at least one detected spine and
checks finite positive volumes, exact parent membership and disjoint masks.
It requires actual supplied face indices for every spine and checks both index
overlap and shared geometric surface across distinct spines.

Create a fixture manifest with checksums of the actual files:

```json
{
  "schema_version": 1,
  "units": "nm",
  "data_type": "microns",
  "branch_key": "example/limb_000/branch_003",
  "mesh": {"path": "branch_mesh.off", "sha256": "REPLACE_WITH_SHA256"},
  "skeleton": {"path": "branch_skeleton.npy", "sha256": "REPLACE_WITH_SHA256"},
  "soma": {"path": "soma_mesh.off", "sha256": "REPLACE_WITH_SHA256"},
  "maximum_parent_faces": 50000,
  "minimum_spines": 1,
  "archived_spines": 9
}
```

Paths resolve relative to the manifest; absolute paths also work. The `soma`
record is optional. `archived_spines` and `branch_key` are receipt annotations;
`maximum_parent_faces` and `minimum_spines` default to 50000 and 1. Dataset
selection accepts `microns` and `h01`. Compute hashes with `sha256sum`, then run:

```bash
NEURD_SPINE_BRANCH_MANIFEST=fixtures/branch/manifest.json \
    python -m pytest tests/integration/test_spine_branch_gate.py -q
```

Missing manifests/data cause a clearly explained skip; mismatched checksums or
a missing compiled backend fail the configured check. The reusable
`run_branch_fixture(manifest_path, workdir)` function returns a receipt with
input, configuration, code, dependency and native-provider hashes, mask metrics
and extraction time.

On 2026-10-11, a local Minnie65 check on
`864691134886335738/limb_000/branch_003` processed 29,495 parent faces and found
9 spines with 2,854 faces. Parent membership, disjointness and all nine volumes
passed; inputs and code stayed unchanged during the run. Extraction took about
7.67 seconds in the final run with explicit face indices and the buffered candidate CGAL
extension (9.95 seconds in the earlier extraction-only smoke). This is one geometry
smoke check, not a benchmark across neurons or anatomical validation.

## Quality limits

Topology, ownership, finite volumes and valid resume metadata establish
computational consistency. They cannot establish that every retained object is
an anatomical spine, that removals are correct or that repaired heads are
complete. Equal old/new counts likewise do not establish equal masks or quality.

The latest smooth-head policy was tuned and replayed on development examples;
it has no independent held-out quality claim. Before promoting either H01
policy, freeze a neuron-held-out selection and review unchanged objects,
repairs, splits, removals, formerly positive branches with zero outputs and
auxiliary-axis cases. Encoder diagnostics and reduced object counts are useful
screens, but neither is anatomical ground truth. Keep policy/ray/volume versions
separate in comparisons and report unresolved or failed branches explicitly.
