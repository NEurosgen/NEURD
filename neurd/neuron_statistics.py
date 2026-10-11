

import numpy as np

from mesh_tools import skeleton_utils as sk
from . import neuron_utils as nru

top_of_layer_vector = np.array([0,-1,0])


    


# ---------- Functions over upstream and downstream branches ----------- #

def stats_dict_over_limb_branch(
    neuron_obj,
   limb_branch_dict=None,
   stats_to_compute = ("skeletal_length","area","mesh_volume","n_branches"),
    ):
    """
    Purpose: To get a statistics 
    over a limb branch dict

    Stats to retrieve:
    1) skeletal length
    2) surface area
    3) volume
    

    """
    if limb_branch_dict is None:
        limb_branch_dict= neuron_obj.limb_branch_dict
    
    s_dict = {k:nru.sum_feature_over_limb_branch_dict(neuron_obj,
                                                   limb_branch_dict,
                                                   feature=k) for k in stats_to_compute}

    return s_dict

def features_from_neuron_skeleton_and_soma_center(
    neuron_obj,
    limb_branch_dict = None,
    neuron_obj_aligned = None, 
    **kwargs
    ):
    
    if limb_branch_dict is not None:
        skeleton = nru.skeleton_over_limb_branch_dict(
            neuron_obj,
            limb_branch_dict,
        )
    else:
        skeleton = neuron_obj.skeleton
        
    if neuron_obj_aligned is not None:
        if limb_branch_dict is not None:
            skeleton_aligned = nru.skeleton_over_limb_branch_dict(
                neuron_obj_aligned,
                limb_branch_dict,
            )
        else:
            skeleton_aligned = neuron_obj_aligned.skeleton
            
        soma_center = neuron_obj_aligned["S0"].mesh_center
    else:
        skeleton_aligned = None 
        soma_center = neuron_obj["S0"].mesh_center
    
    return features_from_skeleton_and_soma_center(
    skeleton,
    soma_center = soma_center,#neuron_obj["S0"].mesh_center,
    skeleton_aligned = skeleton_aligned,
    **kwargs
    )
    
def features_from_skeleton_and_soma_center(
    skeleton,
    soma_center,
    short_threshold = 6000,
    long_threshold = 100000,
    volume_divisor = 1_000_000_000_000_000,#(10**14),
    
    name_prefix = None,
    features_to_exclude = None,
    skeleton_aligned = None,
    in_um = True,):
    """
    Purpose: To calculate features about a skeleton
    representing a subset of the neuron (
    features specifically in relation to soma)
    
    """
    if len(skeleton) == 0:
        axon_dict =  dict(

                    length = 0,
                    branch_length_median = 0,
                    branch_length_mean = 0,

                    n_branches = 0,
                    n_short_branches = 0,
                    n_long_branches = 0,
                    n_medium_branches = 0,

                    bbox_volume=0,
                    bbox_x_min=0,
                    bbox_y_min=0,
                    bbox_z_min=0,
                    bbox_x_max=0,
                    bbox_y_max=0,
                    bbox_z_max=0,

                    bbox_x_min_soma_relative=0,
                    bbox_y_min_soma_relative=0,
                    bbox_z_min_soma_relative=0,
                    bbox_x_max_soma_relative=0,
                    bbox_y_max_soma_relative=0,
                    bbox_z_max_soma_relative=0,

                    )
    else:

        # Calculating the boudning box
        sk_branches = sk.decompose_skeleton_to_branches(skeleton)

        sk_branches_dist = np.array([sk.calculate_skeleton_distance(k) for k in sk_branches])

        n_branches = len(sk_branches)
        n_short_branches = np.sum(sk_branches_dist<short_threshold)
        n_long_branches = np.sum(sk_branches_dist>long_threshold)
        n_medium_branches = np.sum((sk_branches_dist<=long_threshold) & 
                                  (sk_branches_dist>=short_threshold))

        # calculating the skeletal lengths
        if in_um:
            divisor = 1000
        else:
            divisor = 1

        axon_length = np.sum(sk_branches_dist)/divisor
        axon_branch_length_median = np.median(sk_branches_dist)/divisor
        axon_branch_length_mean = np.mean(sk_branches_dist)/divisor

        bbox_volume = sk.bbox_volume(skeleton)/volume_divisor
        bbox_corners = sk.bounding_box_corners(skeleton)
        bbox_corners_soma_relative = bbox_corners - soma_center
        if skeleton_aligned is not None:
            bbox_corners_aligned = sk.bounding_box_corners(skeleton_aligned)
            bbox_corners_soma_relative = bbox_corners_aligned - soma_center
            
        axon_dict = dict(

                        length = axon_length,
                        branch_length_median = axon_branch_length_median,
                        branch_length_mean = axon_branch_length_mean,

                        n_branches = n_branches,
                        n_short_branches = n_short_branches,
                        n_long_branches = n_long_branches,
                        n_medium_branches = n_medium_branches,

                        bbox_volume=bbox_volume,
                        bbox_x_min=bbox_corners[0][0],
                        bbox_y_min=bbox_corners[0][1],
                        bbox_z_min=bbox_corners[0][2],
                        bbox_x_max=bbox_corners[1][0],
                        bbox_y_max=bbox_corners[1][1],
                        bbox_z_max=bbox_corners[1][2],

                        bbox_x_min_soma_relative=bbox_corners_soma_relative[0][0],
                        bbox_y_min_soma_relative=bbox_corners_soma_relative[0][1],
                        bbox_z_min_soma_relative=bbox_corners_soma_relative[0][2],
                        bbox_x_max_soma_relative=bbox_corners_soma_relative[1][0],
                        bbox_y_max_soma_relative=bbox_corners_soma_relative[1][1],
                        bbox_z_max_soma_relative=bbox_corners_soma_relative[1][2],

                        )
    
    if features_to_exclude is not None:
        axon_dict = {k:v for k,v in axon_dict.items() if k not in features_to_exclude}
    if name_prefix is not None:
        axon_dict = {f"{name_prefix}_{k}":v for k,v in axon_dict.items()}
        
    return axon_dict

#-------------- 12/9 Developed for work with cell typing -----------------
            
def centroid_stats_from_neuron_obj(neuron_obj,
                                  voxel_adjustment_vector=None,
                                  include_volume=True):
    """Return the S0 centroid in nm and voxels using an explicit nm-per-voxel scale."""
    if voxel_adjustment_vector is None:
        raise ValueError("Centroid voxel coordinates require an explicit voxel_adjustment_vector "
                         "with three nm-per-voxel values")
        
    soma_x_nm,soma_y_nm,soma_z_nm = nru.soma_centers(neuron_obj,
                                       soma_name="S0",
                                       voxel_adjustment=False,
                                       return_int_form=False)
    soma_x,soma_y,soma_z = nru.soma_centers(neuron_obj,
                                           soma_name="S0",
                                            voxel_adjustment = True,
                                           voxel_adjustment_vector=voxel_adjustment_vector,
                                           return_int_form=True)
    return_dict = dict(
        centroid_x_nm=soma_x_nm,
        centroid_y_nm=soma_y_nm,
        centroid_z_nm=soma_z_nm,
        centroid_x=soma_x,
        centroid_y=soma_y,
        centroid_z=soma_z 
    )
    
    if include_volume:
        return_dict["centroid_volume"] = neuron_obj["S0"].volume
        
    return return_dict
    
def skeleton_stats_from_neuron_obj(neuron_obj,
                                  include_centroids = True,
                                  voxel_adjustment_vector=None,
                                  
                                   limb_branch_dict = None,
                                   neuron_obj_aligned=None,
                                  ):
    """
    Compute all the statistics for a neurons skeleton (should have only one soma)
    """
    sk_dict = stats_dict_over_limb_branch(
        neuron_obj,
        limb_branch_dict=limb_branch_dict,
        stats_to_compute=["skeletal_length","n_branches"])

    sk_dict_2 = features_from_neuron_skeleton_and_soma_center(
        neuron_obj,
        limb_branch_dict=limb_branch_dict,
        features_to_exclude=("length","n_branches"),
        neuron_obj_aligned=neuron_obj_aligned,
        )
    sk_dict.update(sk_dict_2)
    sk_dict["n_limbs"] = neuron_obj.n_limbs
    
    if include_centroids:
        cent_stats = centroid_stats_from_neuron_obj(neuron_obj,
                                                       voxel_adjustment_vector=voxel_adjustment_vector)
        sk_dict.update(cent_stats)
    
    return sk_dict




def neuron_stats(
    neuron_obj,
    stats_to_ignore=None,
    include_skeletal_stats = False,
    include_centroids= False,
    voxel_adjustment_vector = None,
    cell_type_mode = False,
    *,
    include_upstream_diagnostics=True,
    **kwargs):
    """
    Summarize neuron geometry, widths, and computed spines.

    ``stats_to_ignore`` skips each selected getter. Slim objects can omit the
    four preprocessing diagnostic counts with ``include_upstream_diagnostics``
    set to False; requested diagnostics require their original metadata.
    Centroid voxel coordinates require an explicit nm-per-voxel scale.
    """
    if cell_type_mode:
        stats_to_ignore = [
            "n_not_processed_soma_containing_meshes",
            "n_error_limbs",
            "n_same_soma_multi_touching_limbs",
            "n_multi_soma_touching_limbs",
            "n_somas",
            "spine_density"
            ]
        include_skeletal_stats = False
        include_centroids= True

    statistic_fields = (
        "n_vertices", "n_faces",
        "max_soma_volume", "max_soma_n_faces", "max_soma_area",
        "n_not_processed_soma_containing_meshes", "n_error_limbs",
        "n_same_soma_multi_touching_limbs", "n_multi_soma_touching_limbs",
        "n_somas", "n_limbs", "n_branches", "max_limb_n_branches",
        "skeletal_length", "max_limb_skeletal_length", "median_branch_length",
        "width_median", "width_no_spine_median", "width_90_perc", "width_no_spine_90_perc",
        "n_spines", "spine_density", "spines_per_branch",
        "skeletal_length_eligible", "n_spine_eligible_branches", "spine_density_eligible",
        "spines_per_branch_eligible", "total_spine_volume", "spine_volume_median",
        "spine_volume_density", "spine_volume_density_eligible", "spine_volume_per_branch_eligible",
    )
    diagnostic_count_attributes = {
        "n_not_processed_soma_containing_meshes": "not_processed_soma_containing_meshes",
        "n_same_soma_multi_touching_limbs": "same_soma_multi_touching_limbs",
        "n_multi_soma_touching_limbs": "multi_soma_touching_limbs",
    }
    ignored = set()
    if stats_to_ignore is not None:
        known_fields = set(statistic_fields)
        for name in stats_to_ignore:
            if name not in known_fields or name in ignored:
                raise KeyError(name)
            ignored.add(name)

    stats_dict = {}
    for name in statistic_fields:
        upstream_diagnostic = name in diagnostic_count_attributes or name == "n_error_limbs"
        if name in ignored or (upstream_diagnostic and not include_upstream_diagnostics):
            continue
        if upstream_diagnostic:
            attribute = diagnostic_count_attributes.get(name, name)
            try:
                value = getattr(neuron_obj, attribute)
            except AttributeError as error:
                raise AttributeError(
                    f"Cannot compute {name!r}: missing upstream diagnostic {attribute!r}. "
                    "Pass include_upstream_diagnostics=False or add the statistic "
                    "to stats_to_ignore."
                ) from error
            if name in diagnostic_count_attributes:
                value = len(value)
        else:
            value = getattr(neuron_obj, name)
        stats_dict[name] = value

    if include_skeletal_stats:
        sk_dict = features_from_neuron_skeleton_and_soma_center(neuron_obj,
                                              features_to_exclude=("length","n_branches"),
                                             )
        stats_dict.update(sk_dict)

    if include_centroids:
        cent_stats = centroid_stats_from_neuron_obj(neuron_obj,
                                                       voxel_adjustment_vector=voxel_adjustment_vector)
        stats_dict.update(cent_stats)

    return stats_dict

# -- 5/9 Addition for computing more statistics
