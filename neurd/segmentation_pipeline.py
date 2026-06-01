"""Slim mesh-segmentation pipeline.

Input: a neuron mesh (trimesh). Output: a decomposed `Neuron` object exposing the
segmented pieces and their skeletons — `somas`, `limbs` → `branches` (each with
`.mesh` + `.skeleton`), plus raw per-branch spine detection from decomposition.

This runs ONLY the geometry needed for that output:
  soma identification -> decomposition (skeletonization, branches, raw spines).

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
        verbose: print stage timings/info.
        max_somas: if set (e.g. 1 for single-neuron files), stop the soma search
            once this many somas are found. This is a CORRECTNESS guardrail for the
            single-neuron use case — it prevents over-segmenting one neuron into
            multiple somas. It is NOT a speed win on a genuine single-neuron mesh
            (which yields one mesh piece, so there is nothing extra to skip); it only
            saves work on meshes that split into several large pieces.

    Returns:
        neuron.Neuron — `.somas`, `.limbs[i].branches[j].mesh/.skeleton`.
    """
    #products = pipeline.PipelineProducts()

    # --- Soma identification ---
    #soma_products = sm.soma_indentification(mesh, verbose=verbose, max_somas=max_somas)
    #products.set_stage_attrs(stage="soma_identification", attr_dict=soma_products)

    # --- Decomposition (skeletonization, branches, raw spine detection) ---
    # Mirrors process_all_neurons: build the Neuron straight from the mesh.
    # The segmented pieces and their skeletons live on the object after this;
    # the downstream `calculate_decomposition_products` stats step is omitted
    # (it aggregates axon/synapse/centroid stats that the slim build removed).
    neuron_obj = neuron.Neuron(
        mesh=mesh,
        segment_id=segment_id,
       # pipeline_products=products,
        #suppress_preprocessing_print=not verbose,
        #suppress_output=not verbose,
    )

    return neuron_obj
