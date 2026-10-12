# Reviewed H01 spine: C03

This small display fixture contains a manually reviewed spine and its parent
surface crop, in **nanometres**. It comes from H01 neuron `5319188363`, limb `000`,
branch `150`, saved candidate `spine_076.off` (raw candidate 82, final candidate 76).

The project maintainer's review on 9 October 2026 recorded `one_spine`, `looks_ok`
and `high` confidence. The target surface has 358 triangular faces. This is a
separate illustration from the notebook's timed neuron `2530864375`.

| File | Meaning |
|---|---|
| `target_mesh.off` | Exact reviewed spine mesh; no surface repair or simplification |
| `parent_roi.off` | Local parent-branch crop with 2,127 faces |
| `parent_face_indices.npy` | Global parent-branch index for each face of the crop |
| `target_parent_face_membership.npy` | Unordered membership of the 358 target faces in the full parent branch |
| `provenance.json` | Identities, original mesh/review fingerprints, file hashes and view |

The target membership array is not a row-by-row correspondence for the target
OFF mesh. The display excludes these faces from the gray parent context and
renders the complete target separately in red. Mesh coordinates and triangle
geometry remain unchanged. The caption identifies both source neurons.

H01 data credit: Lichtman laboratory, Harvard University, and Google
Connectomics. The [official release](https://h01-release.storage.googleapis.com/data.html)
licenses its datasets under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The original remote acquisition/conversion of the local branch is not reconstructed;
these included crops and their fingerprints identify the exact displayed data.
This one reviewed example does not estimate detection accuracy across a dataset.
