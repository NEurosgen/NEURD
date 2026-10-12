# Advanced API and environment controls

Reference for statistics, legacy object contracts and low-level compatibility
settings. Start with the [public workflow](../../README.md#run) for mesh loading,
segmentation and export. Code paths in backticks are relative to the repository
root.

## Statistics, provenance and provider boundaries

`neuron_searching.query_neuron` retains string and callable features. Its default
result groups branch IDs by limb; `return_limb_grouped_branches=False` returns
ordered `(limb, node)` rows, including shape `(0, 2)` for an empty result.
`neuron_utils.soma_centers` returns centers in nm without a scale. To convert to
voxels, pass `voxel_adjustment=True` and an explicit three-value
`voxel_adjustment_vector` in nm per voxel. Centroid statistics containing both nm
and voxel coordinates require this vector; a dataset-specific scale is not inferred.
For statistics on a slim neuron, explicitly omit upstream preprocessing diagnostics:

```python
summary = neuron.neuron_stats(include_upstream_diagnostics=False)
products = neuron.calculate_decomposition_products(
    include_upstream_diagnostics=False,
    include_centroids=False,
    store_in_obj=True,
)
```

`stats_to_ignore` skips getters before evaluating them. The legacy default
`include_upstream_diagnostics=True` requires the four original preprocessing
diagnostic fields and preserves enriched-object statistics. `n_somas` counts the
actual soma nodes. Decomposition keeps the `StageProducts` return type and merges
into the decomposition stage when storage is requested, creating a missing
`PipelineProducts` container after calculation succeeds. Skeleton features use S0
as their soma reference, including on multi-soma objects. Centroid output contains
nm and voxel coordinates: keep `include_centroids=True` only with an explicit
`voxel_adjustment_vector`; otherwise use False as above. Each selected getter still
requires its inputs, such as computed width entries; omit unavailable fields with
`stats_to_ignore`. Decomposition requires S0 even with centroids disabled, so
zero-soma decomposition is unsupported. Fields retain their mixed historical units
and scalings. Volume-density helpers check their own length divisor before division.

The raw runners, descriptor extractor and geometry gates use `neurd.mesh_io.load_mesh`.
It reads plain triangular ASCII OFF files in bounded chunks, preserving float64
coordinates, duplicate/unused vertices and ordered integer faces. Other mesh formats
and unsupported OFF variants use trimesh's existing loader. Pass the actual filename;
the API does not append `.off`. `NEURD_LEGACY_MESH_LOADER=1` selects the previous reader
and is included in processing and resume fingerprints. Internal MeshLab index repair
continues to use its existing reader.

Spine segmentation reads CGAL face labels without constructing meshes that would
be discarded, then groups those faces once while retaining parent indices and
face order. `NEURD_LEGACY_CGAL_ASSEMBLY=1` selects the alternative two-pass assembly.
Large shaft-path batches use an ordered bidirectional BFS when
Numba is available and NetworkX is the verified 3.6.1 version; smaller batches and
other versions retain NetworkX. `NEURD_LEGACY_SHAFT_PATHS=1` selects the previous
path search and is included in processing fingerprints. H01 batch export reuses
volume quality from the final filter only for the identical, unchanged in-memory
mesh. Scientific validation of saved OFF files requires a separate reload check;
the batch completion/resume check verifies hashes and inventory.

Failed H01 width eligibility preserves the actual axes and
measurement evidence; see [H01_FAILURE_DIAGNOSTICS.md](H01_FAILURE_DIAGNOSTICS.md).
Raw export records validated supplied branch-to-limb face maps and graph labels
by Branch object identity. Missing maps remain unavailable; direct raw ownership
and the causes of the historical axis mismatches are still unproved. Local origin
journals record the observed MAP/MP route without inferring original raw face
ownership.

Mesh loading, statistics/decomposition, raw export and compatibility installation
have separate module boundaries. MAP/MP preprocessing uses named internal stage
results. Compatibility installation lives in `neurd._bootstrap`; set behavior
flags before importing and use a fresh process to change them. The eight default
patches expose their installation status through `neurd.PATCH_STATUS`.

The [validated installation](../INSTALLATION.md) records the unit, native-provider
and raw-mesh correctness checks.

| Export | Verified scope |
|---|---|
| `parent_face_indices.npy` + `face_provenance.json` | Supplied ordered branch-to-limb IDs; exact in-memory triangles before OFF rounding, exported-file hashes; absent maps explicitly unavailable |
| `branch_index_map.json` | Export position to graph node by Branch object identity; missing/ambiguous identity recorded, contradictory IDs rejected |
| `branch_origin.json` | Observed local MAP/MP selection, axis fingerprints and fallback cause; actual exported skeleton hash recorded separately; missing origin explicitly unavailable |
| Original raw OFF face ownership | Unavailable; never inferred from geometric matches or branch numbers |


## Environment flags

Most flags select an alternative compatibility path. The supported defaults are
active without setting them.

| flag | default | effect when set |
|---|---|---|
| `NEURD_LEGACY_MESH_LOADER` | bounded triangular ASCII OFF reader at supported entry points | `1` restores the previous reader |
| `NEURD_LEGACY_CGAL_ASSEMBLY` | labels-only CGAL spine seam and one face assembly | `1` restores the previous two-pass assembly |
| `NEURD_LEGACY_VERTEX_COMPONENTS` | scipy | use the NetworkX vertex graph |
| `NEURD_LEGACY_MATCHING_VERTICES` | cKDTree | restore the O(N²) distance matrix |
| `NEURD_MESHLAB_DECIMATE` | open3d, in-process | use the MeshLab subprocess |
| `NEURD_DECIMATE_SINGLE_STAGE` | cascaded halving | decimate in one step |
| `NEURD_ENABLE_COMPRESSED_PICKLE` | no-op'd | enable the debug pickle dumps |
| `NEURD_SDF_RAYS` | low-level compatibility 12 | CGAL SDF ray count; the public API/CLI sets it from the explicit reference25/fast12 configuration |
| `NEURD_CGAL_TIMINGS` | off | `1` writes native timings to stderr and nested Python phase records for assembly, shaft paths, lengths, refinement, final filters and H01 publication to stdout |
| `NEURD_LEGACY_SHAFT_PATHS` | off | `1` retains the previous NetworkX shaft-path search; fingerprinted for resume |
| `NEURD_SKIP_SOMA_END_SPLIT` | off | skip `_split_soma_at_end`'s CGAL trim |
| `NEURD_OWNED_SKELETAL_DISTANCE` | off | route the correspondence/width kernel through `neurd/skeletal_distance_ops.py` |
| `NEURD_REAL_POISSON` | off (no-op) | run a real Poisson reconstruction |
| `NEURD_POISSON_DEPTH` / `_BACKEND` / `_ITERS` | 11 / meshlab / 8 | sub-flags of the above |
| `NEURD_SOMA_OUTER_DECIM` / `_INNER_DECIM` | from `parameters.py` | override the soma decimation ratios |

The batch runners also read `SEG_RSS_LIMIT_MB` and `SEG_TIMEOUT_SEC` (per-mesh kill switches), plus
`SEG_MAX_WORKERS` and `SEG_INPUT_SIZE_BUDGET_MB` (defaults for the parallel runner's flags;
`SEG_MEM_BUDGET_MB` remains a legacy alias for input-file bytes).

Every performance patch registers itself in `neurd.PATCH_STATUS` and reports an
installation failure. These patches replace functions inside
`mesh_tools`/`datasci_tools`; dependency changes therefore need import and patch
checks. `tests/unit/test_perf_patches.py` verifies the supported installation.
