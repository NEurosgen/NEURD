# Optional compartment and synapse inputs

The supported raw API produces soma/limb/branch geometry, skeletons and optional
spines. It does not classify axons/dendrites, assign synapses, or compute synapse
endpoint distances. This document defines the external inputs needed by retained
hooks and the boundary for a future annotation loader. It adds no loader,
classifier or runtime validation; historical behavior remains unchanged.

The audit covers [neuron.py](../neurd/neuron.py),
[neuron_utils.py](../neurd/neuron_utils.py),
[branch_utils.py](../neurd/branch_utils.py) and string feature dispatch in
[neuron_searching.py](../neurd/neuron_searching.py).

## Available data and retained hooks

| Surface | Raw object | Required external input / current behavior |
|---|---|---|
| `Branch.labels` | `[]` | Generic labels; no anatomical classification is supplied |
| `Branch.axon_compartment` | Returns `"dendrite"` when `"axon"` is absent | Historical binary fallback; an empty label list does not establish a dendrite |
| `Neuron.axon_skeleton`, `dendrite_skeleton` | Classification maps absent | Work from explicitly supplied `axon_limb_branch_dict` / `dendrite_limb_branch_dict` |
| `Neuron.axon_limb_name`, `axon_limb`, `axon_limb_idx` | Axon map absent | Use its first key; `{}` produces `None`, without proving absence of an axon |
| `Neuron.axon_mesh`, `dendrite_mesh`, `axon_length`, `axon_area` | Unavailable | Delegated functions were removed; attaching maps alone does not repair these properties |
| `Branch.synapses`, `Soma.synapses` | `[]` | Containers only; empty defaults do not distinguish unexamined from confirmed empty |
| `Limb.synapses`, `Neuron.synapses` | Empty branch collections | Flatten branch lists; the neuron aggregate excludes soma synapses |
| Four branch `min_dist_synapses_pre/post_upstream/downstream` properties | `synapses_pre` / `synapses_post` absent | Require separate supplied lists and distance fields; explicit empty lists return `inf` |
| Six neuron `n_mesh_errored_synapses*` / `n_distance_errored_synapses*` counters | Underlying lists absent | Return lengths of individually supplied error lists; no error detection/filtering is performed |

Absent inputs currently raise `AttributeError` when accessed; they are not
converted to zero or empty results. A generic `synapses=[]` is not a substitute
for the separate pre/post lists. String-dispatched queries can reach these
properties and inherit their limitations. A query example mentioning a feature
does not make that feature available on a raw object.

The missing mesh/length/area delegates follow the intentional removal of axon
classification helpers in commit `5e3203c9154eddadb0fa0fc9073fa4ecab414d7a`.
Restoring them would be a separate API change with explicit inputs and geometry
tests, rather than a mechanical extraction of the old implementation.

## Compartment selections

The existing skeleton hooks consume a mapping from limb names to branch IDs:

```python
# Illustrative IDs; annotations must refer to branches in this exact object.
neuron_obj.axon_limb_branch_dict = {"L0": [0, 2]}
axon_segments = neuron_obj.axon_skeleton
```

The input requirements for external producers are:

- Use canonical limb keys such as `"L0"` and ordered lists of existing integer
  branch IDs. These are object-graph IDs, not mesh face IDs or external segment IDs.
- Select each branch once. Do not mix integer and string aliases for the same
  limb; the current helpers accept legacy aliases but do not normalize them.
- Omit empty limb groups; use `{}` for an intentionally empty selection.
  `axon_limb_name` otherwise chooses the first key even when its list is empty.
- Supply dendrite selections independently. Unclassified branches remain
  unclassified; a dendrite map is not automatically the complement of the axon map.
  If both selections describe exclusive compartments, they must not overlap.
- Record the source, coverage and mapping to the current geometry separately.
  An empty selection alone does not establish complete classification or absence
  of that biological compartment.

These requirements are for producers; existing consumers do not enforce them.
They visit mapping insertion order and each supplied branch order, retaining
duplicates and alias selections. The skeleton wrapper stacks selected branch
segments without adding soma connections. It does not classify branches or
normalize the result. With the tested provider, an empty selection returns an
array of shape `(0,)`, rather than `(0, 2, 3)`; a future shape change needs its own
compatibility decision.

## Synapse records and endpoint distances

The endpoint helpers accept objects with scalar attributes, not CSV dictionaries.
The minimum required inputs for a selected branch are:

| Field | Meaning |
|---|---|
| `branch.synapses_pre` | Ordered sequence of externally assigned presynaptic records |
| `branch.synapses_post` | Ordered sequence of externally assigned postsynaptic records |
| `record.upstream_dist` | Precomputed distance from the declared upstream endpoint |
| `record.downstream_dist` | Precomputed distance from the declared downstream endpoint |
| `branch.synapses` | Independent aggregate list, if collection queries are also needed |

```python
from types import SimpleNamespace

# Distances are illustrative, already computed for this branch and orientation.
branch.synapses_pre = [SimpleNamespace(upstream_dist=120.0, downstream_dist=380.0)]
branch.synapses_post = []
nearest_pre = branch.min_dist_synapses_pre_upstream  # 120.0, supplied distance units
nearest_post = branch.min_dist_synapses_post_upstream  # inf: explicit empty list
```

The current consumer only takes a minimum. It does not compute projection,
Euclidean/geodesic distance, orientation, coordinate conversion, or pre/post
partitioning; it does not synchronize these lists with `branch.synapses`.
The producer must identify the distance method and endpoint orientation and
supply nonnegative finite distances in the same declared physical unit.
Current code does not validate numeric types, negative values or NaNs.
`branch_utils.min_dist_synapse_endpoint` also permits a custom `default_value`
for an explicitly empty collection.

For assignment or re-import, retain synapse IDs without floating-point loss,
source-neuron identity, pre/post direction, source coordinates and their units,
the verified transform to the current mesh, current branch/soma identity,
assignment method and any parent-face/projected-axis association. Keep unassigned
records and failure reasons distinguishable from an inspected empty population.
This is required producer metadata, not an implemented import format. A future
loader must define how it validates and stores it before producing annotations.

Raw constructors initialize neither the pre/post views nor these neuron error
lists: `mesh_errored_synapses`, `mesh_errored_synapses_pre`,
`mesh_errored_synapses_post`, `distance_errored_synapses`,
`distance_errored_synapses_pre`, `distance_errored_synapses_post`. The six counters
do not derive one list from another. Empty lists are appropriate only when a
producer has explicitly inspected the corresponding population.

## Units and geometry identity

Use verified physical coordinate units. The dataset setting `"microns"` names a
parameter profile; it does not convert input coordinates to micrometres.
The H01 path assumes nm geometry. Skeleton distances and supplied endpoint
distances otherwise retain their input units. `Branch.area` divides mesh area by
`1_000_000`, so it represents µm² only for nm coordinates.

Historical deleted `axon_length(units="um")` summed branch skeletal lengths and
divided by `1000`; historical `axon_area(units="um")` summed already-scaled branch
areas, multiplying by `1_000_000` for `units="nm"`. These are migration facts,
not supported properties in this fork. Mixed legacy statistics units and spine
count/volume population policy remain separate scientific decisions.

For every annotation source, preserve its hash and declared units/transform,
the neuron input hash and run configuration, and the mapping to current branch
meshes and ordered skeletons. Reprocessing or segment IDs alone do not establish
the same branch identities. Existing raw origin journals and parent-face maps
check their recorded local scope; they do not recover historical annotation
assignment or ordered raw ownership.

The shipped synapse CSVs contain `segment_id`, `synapse_id`, `prepost`
(`presyn`/`postsyn`), `synapse_x/y/z` and `synapse_size`; some also contain
`segment_id_secondary`. Their headers declare neither coordinate nor size units.
Stored notebook centroid pairs suggest a historical `[4, 4, 40]` nm-per-voxel
conversion, which does not verify these CSV coordinate or size semantics.
Do not infer H01 conversion or interpret `synapse_size` as a physical volume.
The H01 EM-review localizer uses a different source with `post_x/y/z_nm` and an
incomplete contact subset; it does not establish spine assignment.

## Historical notebook migration

All four shipped autoproof notebooks require a broader workflow migration.
Their sources and stored outputs remain archival; none was executed or rewritten
in this documentation pass. Cell references below are **1-based JSON cell indices**.
Paths are relative to `Applications/Tutorials/Auto_Proof_Pipeline/`.

| Notebook | Legacy constructor cell | Later unsupported stages |
|---|---:|---|
| `Single_Soma_Exc/neuron_pipeline_vp5_single_demo_exc.ipynb` | 32 | Multi-soma split, classification, autoproof, synapse visualization |
| `Single_Soma_Exc/Old_notebooks/neuron_pipeline_vp5_single_demo_exc.ipynb` | 30 | Same enriched workflow |
| `Single_Soma_Inh/neuron_pipeline_vp5_single_demo_inh.ipynb` | 29 | Same enriched workflow |
| `Double_Soma/neuron_pipeline_vp5_double_soma.ipynb` | 31 | Multi-soma processing plus the enriched workflow |

They pass `pipeline_products`, `suppress_preprocessing_print` and `suppress_output`
to `Neuron`; the current constructor does not accept those keywords. They also
import removed dataset, pipeline, visualization and synapse modules. Updating
only the constructor call cannot preserve their recorded analysis.

| Historical step | Supported migration / remaining work |
|---|---|
| Dataset adapter and enriched construction | Load a verified local OFF mesh; call `segmentation_pipeline(..., config=RunConfig(...))` for fresh raw processing |
| Implicit stage products | Explicit `calculate_decomposition_products(include_upstream_diagnostics=False, include_centroids=False)` |
| Voxel centroid statistics | Supply a verified `voxel_adjustment_vector` explicitly |
| `.pbz2` object recovery | `save_segmentation` exports geometry, arrays and provenance; it does not recreate an enriched object |
| Compartment/synapse assignment | Supply verified external inputs as described above; no slim classification/assignment pipeline exists |
| Multi-soma splitting, autoproof and legacy plots | No replacement in the supported raw API; retain historical notebooks as archives |

Before publication, create a separate executed raw-neuron notebook with input
hash, explicit configuration, complete processing, export checks, timings and
peak memory. Do not carry historical enriched outputs into it as new results.

## Evidence and next decisions

Local audit/probe receipts are retained in
`artifacts/enrichment_contract_cleanup_20261011/v1/`. They describe existing
consumers, supplied tiny fixtures and archived notebook/CSV inputs; they do not
validate anatomical classification or synapse assignment. No whole-neuron
processing or before/after performance comparison was run for this documentation
pass; runtime source bytes are unchanged from the bootstrap package.

The input requirements are now explicit. A future enrichment change must choose
an annotation source, verify units and current geometry identity, define missing
versus inspected-empty states, and validate any new importer or restored
aggregate. It should not reintroduce the removed classifier or derive dendrites
from empty labels as a side effect of preparing the raw fork for publication.
