
import numpy as np
from mesh_tools import compartment_utils as cu
from mesh_tools import skeleton_utils as sk

from . import neuron_utils as nru


default_skeleton_segment_size = 1000


def skeleton_resized_ordered(
    skeleton,
    skeleton_segment_size=default_skeleton_segment_size,
    width_segment_size=None,
    return_skeletal_points=False,
):
    ex_branch_skeleton_resized = sk.resize_skeleton_branch(
        skeleton, segment_width=skeleton_segment_size
    )
    if width_segment_size is not None:
        ex_branch_skeleton_resized = sk.resize_skeleton_branch(
            ex_branch_skeleton_resized, segment_width=width_segment_size
        )
    ex_branch_skeleton_resized = sk.order_skeleton(ex_branch_skeleton_resized)
    if return_skeletal_points:
        return sk.skeleton_coordinate_path_from_start(ex_branch_skeleton_resized)
    return ex_branch_skeleton_resized


def calculate_new_width(
    branch,
    skeleton_segment_size=1000,
    width_segment_size=None,
    return_average=False,
    distance_by_mesh_center=True,
    no_spines=True,
    summary_measure="mean",
    no_boutons=False,
    distance_threshold=3000,
    old_width_calculation=None,
):
    """Compute width array (and optionally its mean/median) along a branch skeleton."""
    f = getattr(np, summary_measure)

    if no_spines:
        ex_branch_no_spines_mesh = nru.branch_mesh_no_spines(branch)
    elif no_boutons:
        if hasattr(branch, "boutons"):
            ex_branch_no_spines_mesh = nru.mesh_without_boutons(branch)
        else:
            ex_branch_no_spines_mesh = branch.mesh
    else:
        ex_branch_no_spines_mesh = branch.mesh

    ex_branch_skeleton_resized = skeleton_resized_ordered(
        skeleton=branch.skeleton,
        skeleton_segment_size=skeleton_segment_size,
        width_segment_size=width_segment_size,
    )

    total_distances, _, _, _ = (
        cu.get_skeletal_distance_no_skipping(
            main_mesh=ex_branch_no_spines_mesh,
            edges=ex_branch_skeleton_resized,
            buffer=0.01,
            bbox_ratio=1.2,
            distance_threshold=distance_threshold,
            distance_by_mesh_center=distance_by_mesh_center,
            print_flag=False,
            edge_loop_print=False,
        )
    )

    total_distances = np.array(total_distances)

    if old_width_calculation is None:
        old_width_calculation = branch.width

    if len(total_distances[total_distances > 0]) > 0:
        total_distances[total_distances <= 0] = f(total_distances[total_distances > 0])

    branch_width_average = f(total_distances)
    if branch_width_average < 0.0001:
        branch_width_average = old_width_calculation
        total_distances = np.ones(len(ex_branch_skeleton_resized)) * branch_width_average
    else:
        total_distances[total_distances == 0] = branch_width_average

    if return_average:
        return total_distances, branch_width_average
    return total_distances




def calculate_new_width_for_neuron_obj(
    neuron_obj,
    skeleton_segment_size=1000,
    width_segment_size=None,
    width_name=None,
    distance_by_mesh_center=True,
    no_spines=True,
    summary_measure="mean",
    limb_branch_dict=None,
    verbose=True,
    skip_no_spine_width_if_no_spine=True,
    **kwargs,
):
    """Recalculate and store width arrays for all branches in a neuron."""
    if width_name is None:
        width_name = str(summary_measure)
    else:
        if "mesh_center" in width_name:
            distance_by_mesh_center = True
        else:
            distance_by_mesh_center = False

        if "no_spine" in width_name:
            no_spines = True
        else:
            no_spines = False

        if "mean" in width_name:
            summary_measure = "mean"
        elif "median" in width_name:
            summary_measure = "median"
        else:
            raise Exception("No summary statistic was specified in the name")

    if summary_measure != "mean":
        width_name = width_name.replace("mean", summary_measure)
        if summary_measure not in width_name:
            width_name = f"{width_name}_{summary_measure}"

    if ("no_spine" not in width_name) and no_spines:
        width_name = f"no_spine_{width_name}"
    if ("mesh_center" not in width_name) and distance_by_mesh_center:
        width_name = f"{width_name}_mesh_center"

    for limb_idx in neuron_obj.get_limb_node_names():
        if limb_branch_dict is not None and limb_idx not in limb_branch_dict:
            continue

        for branch_idx in neuron_obj.get_branch_node_names(limb_idx):
            if limb_branch_dict is not None and branch_idx not in limb_branch_dict[limb_idx]:
                continue

            curr_branch_obj = neuron_obj[limb_idx][branch_idx]
            already_computed = False

            if skip_no_spine_width_if_no_spine:
                if (
                    (curr_branch_obj.spines is None or len(curr_branch_obj.spines) == 0)
                    and no_spines
                    and "no_spine" in width_name
                ):
                    new_width_name = width_name.replace("no_spine_", "")
                    if new_width_name in curr_branch_obj.width_new:
                        curr_branch_obj.width_new[width_name] = curr_branch_obj.width_new[new_width_name]
                        curr_branch_obj.width_array[width_name] = curr_branch_obj.width_array[new_width_name]
                        if verbose:
                            print(
                                f"    No spines and using precomputed width: "
                                f"{curr_branch_obj.width_new[new_width_name]}"
                            )
                        already_computed = True

            if not already_computed:
                current_width_array, current_width = calculate_new_width(
                    curr_branch_obj,
                    skeleton_segment_size=skeleton_segment_size,
                    width_segment_size=width_segment_size,
                    distance_by_mesh_center=distance_by_mesh_center,
                    no_spines=no_spines,
                    summary_measure=summary_measure,
                    return_average=True,
                    **kwargs,
                )
                curr_branch_obj.width_new[width_name] = current_width
                curr_branch_obj.width_array[width_name] = current_width_array

            curr_branch_obj.width_array_skeletal_lengths = None
