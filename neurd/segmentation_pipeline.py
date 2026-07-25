"""Slim mesh-segmentation pipeline.

Input: a neuron mesh (trimesh). Output: a decomposed `Neuron` object exposing the
segmented pieces and their skeletons — `somas`, `limbs` → `branches` (each with
`.mesh` + `.skeleton`), plus raw per-branch spine detection from decomposition.

This runs ONLY the geometry needed for that output: soma identification (inline, inside
the decomposition) -> skeletonization -> branches -> raw spines.

It deliberately omits everything downstream of raw decomposition that the segmentation
use-case does not need: width refinement, branch simplification, spine head/neck/shaft
packaging, multi-soma split, synapse assignment, E/I cell typing, axon labeling,
auto-proofreading, and after-proof statistics.

Prerequisite: module global parameters must already be configured for the dataset
(`neurd.set_volume_params(...)` / the volume_data_interface), same as the full pipeline.
"""

from neurd import neuron


def segmentation_pipeline(
    mesh,
    segment_id=12345,
    verbose=False,
    max_somas=None,
):
    """Decompose `mesh` into a Neuron of segmented pieces (mesh + skeleton).

    Args:
        mesh: input neuron mesh (trimesh.Trimesh).
        segment_id: id to tag the neuron with (cosmetic; default 12345).
        verbose: print the preprocessing timing line. The pipeline's stage progress is
            printed unconditionally, so this does not silence the build.
        max_somas: if set (e.g. 1 for single-neuron files), stop the soma search once
            this many somas are found. Only the first soma is ever used, so on a genuine
            single-neuron mesh this bounds work without changing the result; on a mesh
            that splits into several large pieces it also skips the extra searches. If
            more than one soma is found the extras are dropped with a warning.

    Returns:
        neuron.Neuron — `.somas`, `.limbs[i].branches[j].mesh/.skeleton`.
    """
    # Soma identification is not a separate stage here: Neuron -> preprocess_neuron
    # -> _extract_single_soma -> extract_soma_center runs it inline, which is where
    # max_somas/verbose are honoured.
    #
    # Decomposition (skeletonization, branches, raw spine detection) mirrors
    # process_all_neurons: build the Neuron straight from the mesh. The segmented pieces
    # and their skeletons live on the object after this; the downstream
    # `calculate_decomposition_products` stats step is omitted (it aggregates
    # axon/synapse/centroid stats that the slim build removed).
    neuron_obj = neuron.Neuron(
        mesh=mesh,
        segment_id=segment_id,
        verbose=verbose,
        max_somas=max_somas,
    )

    return neuron_obj
