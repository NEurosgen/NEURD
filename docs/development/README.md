# Developer reference

Start with the [user README](../../README.md),
[installation guide](../INSTALLATION.md) and
[single-neuron example](../../examples/README.md). This directory documents
implementation boundaries, compatibility inputs and validation methods.

| Reference | Purpose |
|---|---|
| [NEURD_STRUCTURE.md](NEURD_STRUCTURE.md) | Module map, entry points and contribution conventions |
| [PIPELINE.md](PIPELINE.md) | Soma extraction, limb/branch decomposition and provider patches |
| [MESH_OPS.md](MESH_OPS.md) | Owned submesh operations, face-index scopes and invariants |
| [SPINE_VOLUME_QC.md](SPINE_VOLUME_QC.md) | Strict closure, convex-hull fallback and descriptor migration |
| [API_COMPATIBILITY.md](API_COMPATIBILITY.md) | Advanced statistics, query API and environment switches |
| [ENRICHMENT_CONTRACT.md](ENRICHMENT_CONTRACT.md) | Inputs required by retained compartment/synapse hooks |
| [H01_FAILURE_DIAGNOSTICS.md](H01_FAILURE_DIAGNOSTICS.md) | Historical branch-failure analysis and provenance limits |
| [BENCHMARKING.md](BENCHMARKING.md) | Historical controlled comparison and measurement protocol |

The [spine guide](../SPINE_REFINEMENT.md) covers optional refinement, exported-branch
processing and its geometry gates. Historical working plans and performance
logs remain in Git history; they are not required for using the fork.

## Checks from the repository root

Activate the validated environment, then run the unit suite:

```bash
source .venv/bin/activate
pytest
```

Real-data gates need external inputs. For the named H01 reference mesh:

```bash
NEURD_H01_MESH=/path/to/neuron_2530864375.off \
    pytest tests/unit/test_submesh_ops.py tests/integration/test_segmentation_pipeline.py
python tests/tools/neuron_gate.py --neuron /path/to/neuron_2530864375.off \
    --baseline tests/fixtures/neuron_baseline_2530864375.json --tolerance
python tests/tools/neuron_gate.py --neuron /path/to/neuron_2530864375.off \
    --dataset h01 --spines --sdf-profile fast --seed 0 --out /tmp/h01_spine_metrics.json
```

The baseline belongs to neuron `2530864375`. Another mesh needs its own reference.
Use matching seeds, configuration and numerical thread limits for paired checks;
native and decomposition variance still require explicit tolerances.

For a checksummed branch containing spines:

```bash
NEURD_SPINE_BRANCH_MANIFEST=/path/to/fixture.json \
    pytest tests/integration/test_spine_branch_gate.py
```

The [manifest format and requirements](../SPINE_REFINEMENT.md#a-small-spine-positive-integration-gate)
include native CGAL, masks and volume checks. CGAL oracle inputs `tests/990_mesh*`
and raw H01 meshes are local fixtures, not downloadable repository assets. A
passing portable unit suite does not replace the native gates, anatomical review
or a controlled performance measurement. The installation guide records the
scope of the previously populated validation checkout.

## Source and evidence scope

Preserve face-index ordering, local coordinate domains and missing-provenance
states. A branch-to-limb map does not establish original raw-face ownership.
Keep behavior changes separate from structural edits and check actual callers,
including string-dispatched features, before deleting a function.

Historical diagnostics mention ignored local `artifacts/` paths. Those receipts
are not included in the public source. Historical commit identifiers in saved
metadata refer to the original recording history and may not resolve after
sanitization; use the published tag and recorded source/input hashes.
