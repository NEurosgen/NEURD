"""Slim mesh-segmentation pipeline.

Input: a neuron mesh (trimesh). Output: a decomposed `Neuron` object exposing the
segmented pieces and their skeletons — `somas`, `limbs` → `branches` (each with
`.mesh` + `.skeleton`), and per-branch spine meshes (`.spines`, head/neck/shaft labeled).

This runs ONLY the geometry needed for that output:
  soma identification -> decomposition (skeletonization, branches, spines) ->
  width refinement -> (optional) branch simplification -> spine head/neck/shaft packaging.

It deliberately omits the downstream "neuron typing / graph" stages that the segmentation
use-case does not need: multi-soma split, synapse assignment, E/I cell typing, axon
labeling, auto-proofreading, and after-proof statistics. Spines are geometric and are
packaged with `add_synapse_labels=False`, so no synapse data is required.

Prerequisite: module global parameters must already be configured for the dataset
(`neurd.set_volume_params(...)` / the volume_data_interface), same as the full pipeline.
"""

from datasci_tools import pipeline

from neurd import soma_extraction_utils as sm
from neurd import neuron
from neurd import branch_utils as bu
from neurd import neuron_simplification as nsimp
from neurd import spine_utils as spu


def segmentation_pipeline(
    mesh,
    segment_id=12345,
    branch_simplification=True,
    verbose=False,
):
    """Decompose `mesh` into a Neuron of segmented pieces (mesh + skeleton + spines).

    Args:
        mesh: input neuron mesh (trimesh.Trimesh).
        segment_id: id to tag the neuron with (cosmetic; default 12345).
        branch_simplification: collapse trivial branch chains for cleaner pieces.
        verbose: print stage timings/info.

    Returns:
        neuron.Neuron — `.somas`, `.limbs[i].branches[j].mesh/.skeleton/.spines`.
    """
    products = pipeline.PipelineProducts()

    # --- Soma identification ---
    soma_products = sm.soma_indentification(mesh, verbose=verbose)
    products.set_stage_attrs(stage="soma_identification", attr_dict=soma_products)

    # --- Decomposition (skeletonization, branches, raw spine detection) ---
    neuron_obj = neuron.Neuron(
        mesh=mesh,
        segment_id=segment_id,
        pipeline_products=products,
        suppress_preprocessing_print=not verbose,
        suppress_output=not verbose,
    )
    neuron_obj.calculate_decomposition_products(store_in_obj=True)

    # --- Width refinement + (optional) branch simplification ---
    bu.refine_width_array_to_match_skeletal_coordinates(neuron_obj, verbose=False)
    if branch_simplification:
        neuron_obj = nsimp.branching_simplification(
            neuron_obj,
            return_copy=True,
            verbose=verbose,
        )

    # --- Spine head/neck/shaft packaging (geometric; no synapses) ---
    neuron_obj = spu.add_head_neck_shaft_spine_objs(
        neuron_obj,
        add_synapse_labels=False,
        verbose=verbose,
    )

    return neuron_obj
