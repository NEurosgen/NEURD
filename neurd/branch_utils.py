
import copy
from scipy.spatial import KDTree
import time
from datasci_tools import numpy_dep as np
from datasci_tools import general_utils as gu

non_optional_features = [
    "mesh",
    "mesh_face_idx",
    
    "spines",
    'spines_obj',
    'spines_volume',
    
    'boutons', 
    'boutons_cdfs',
    'boutons_volume',
    'head_neck_shaft_idx',
    'synapses',
    
    
]

only_keep_upstream = [
    'web',
    'web_cdf',
]

non_optional_features_recalculate = [
    "_mesh_volume",
    "mesh_center",
]

optional_features = [
    "labels",
    
    "endpoints",
    "skeleton",
    'width',
    'width_array',
    'width_new'
]

optional_features_recalculate = [
    "_endpoints_nodes",
    "_skeleton_graph",
]








    
# --------------- setting branch attributes --------------
def set_branch_attr_on_limb(
    limb_obj,
    func,
    attr_name,
    branch_idxs=None,
    **kwargs):
    """
    Purpose: To set the upstream and downstream order of the
    endpoints of a branch in a limb
    """
    if branch_idxs is None:
        branch_idxs = limb_obj.get_branch_names()
    for branch_idx in branch_idxs:
        setattr(limb_obj[branch_idx],attr_name,func(limb_obj,branch_idx,**kwargs))
        
    return limb_obj

def set_branches_endpoints_upstream_downstream_idx_on_limb(
    limb_obj,
    **kwargs
    ):
    return set_branch_attr_on_limb(
        limb_obj,
        func = nru.upstream_downstream_endpoint_idx,
        attr_name = "endpoints_upstream_downstream_idx",
        **kwargs,
    )
    
def set_branch_attr_on_limb_on_neuron(
    neuron_obj,
    func,
    attr_name,
    verbose = False,
    **kwargs):
    """
    Purpose: To set the upstream and downstream order of the
    endpoints of a branch in a limb
    """
    for limb_idx in neuron_obj.get_limb_names():
        neuron_obj[limb_idx] = set_branch_attr_on_limb(neuron_obj[limb_idx],func,attr_name,**kwargs)
    return neuron_obj
        
def set_branches_endpoints_upstream_downstream_idx(
    neuron_obj,
    **kwargs
    ):
    
    neuron_obj = set_branch_attr_on_limb_on_neuron(
        neuron_obj,
        func = nru.upstream_downstream_endpoint_idx,
        attr_name = "endpoints_upstream_downstream_idx",
        **kwargs
    )
    
    #print(f"{neuron_obj[0][0].endpoints_upstream_downstream_idx}")
    return neuron_obj


def skeleton_vector_endpoint(
    branch_obj,
    endpoint_type,
    directional_flow = "downstream",
    endpoint_coordinate = None,
    verbose = False,
    plot_restricted_skeleton= False,
    offset=500,
    comparison_distance=3_000,#2_000,
    skeleton_attribute = "skeleton",
    **kwargs
    ):
    """
    Purpose: To restrict a skeleton
    to its upstream or downstream vector ()

    The vector is always in the direction of most
    upstream skeleton point to downstream skeletal point

    Example: 
    skeleton_vector_endpoint(
    branch_obj,
    endpoint_type = "downstream",
    #endpoint_coordinate = np.array([2504610. ,  480431. ,   33741.2])
    plot_restricted_skeleton = True,
    verbose = True,
    )
    """
    if offset is None:
        offset = offset_skeleton_vector_global
    
    if comparison_distance is None:
        comparison_distance = comparison_distance_skeleton_vector_global
    

    if endpoint_coordinate is None:
        endpoint_coordinate = getattr(branch_obj,f"endpoint_{endpoint_type}")
        if verbose:
            print(f"{endpoint_type} endpoint: {endpoint_coordinate}")

    skeleton = getattr(branch_obj,skeleton_attribute)
    curr_vec = sk.vector_away_from_endpoint(
        skeleton = skeleton,
        endpoint = endpoint_coordinate,
        verbose = False,
        plot_restricted_skeleton = plot_restricted_skeleton,
        offset=offset,
        comparison_distance=comparison_distance,#2_000,
        **kwargs
    )



    mult_number = 1
    if endpoint_type == directional_flow:
        mult_number = -1

    final_vec=  mult_number*curr_vec

    if verbose:
        print(f"curr_vec = {curr_vec}")
        print(f"mult_number = {mult_number}")
        print(f"final_vec = {final_vec}")
        
    return final_vec


def skeleton_vector_upstream(
    branch_obj,
    directional_flow = "downstream",
    endpoint_coordinate = None,
    verbose = False,
    plot_restricted_skeleton= False,
    skeleton_attribute = "skeleton",
    **kwargs
    
    ):
    return skeleton_vector_endpoint(
        branch_obj,
        endpoint_type = "upstream",
        directional_flow = directional_flow,
        endpoint_coordinate = endpoint_coordinate,
        verbose = verbose,
        plot_restricted_skeleton= plot_restricted_skeleton,
        skeleton_attribute=skeleton_attribute,
        **kwargs
    )

def skeleton_vector_downstream(
    branch_obj,
    directional_flow = "downstream",
    endpoint_coordinate = None,
    verbose = False,
    plot_restricted_skeleton= False,
    skeleton_attribute = "skeleton",
    **kwargs
    
    ):
    return skeleton_vector_endpoint(
        branch_obj,
        endpoint_type = "downstream",
        directional_flow = directional_flow,
        endpoint_coordinate = endpoint_coordinate,
        verbose = verbose,
        plot_restricted_skeleton= plot_restricted_skeleton,
        skeleton_attribute=skeleton_attribute,
        **kwargs
    )
    
def skeleton_vector_upstream_extra_offset(
    branch_obj,
    offset = None,
    skeleton_attribute = "skeleton",
    **kwargs
    ):
    if offset is None:
        offset = extra_offset_skeleton_vector_global
    return skeleton_vector_upstream(
        branch_obj,
        offset = offset,
        skeleton_attribute=skeleton_attribute,
        **kwargs
    )

def skeleton_vector_downstream_extra_offset(
    branch_obj,
    offset = None,
    skeleton_attribute = "skeleton",
    **kwargs
    ):
    if offset is None:
        offset = extra_offset_skeleton_vector_global
    return skeleton_vector_downstream(
        branch_obj,
        offset = offset,
        skeleton_attribute=skeleton_attribute,
        **kwargs
    )



def width_endpoint(
    branch_obj,
    endpoint, # if None then will select most upstream endpoint of branch
    #parameters for the restriction
    offset=None,
    comparison_distance=2000,
    skeleton_segment_size=1000,
    verbose = False,
    ):
    """
    Purpose: To compute the width of a branch
    around a comparison distance and offset of an endpoint
    on it's skeleton

    """
    if offset is None:
        offset = offset_width_endpoint_global

    if verbose:
        print(f"endpoint = {endpoint}")


    (base_final_skeleton,
    base_final_widths,
    base_final_seg_lengths) = nru.align_and_restrict_branch(branch_obj,
                              common_endpoint=endpoint,
                             offset=offset,
                             comparison_distance=comparison_distance,
                             skeleton_segment_size=skeleton_segment_size,
                              verbose=False,
                             )


    branch_width = np.mean(base_final_widths)
    overall_ais_width = branch_obj.width_new
    if verbose:
        print(f"base_final_widths = {base_final_widths}")
        print(f"overall_branch_width = {overall_ais_width}")
        print(f"branch_width = {branch_width}")
        
    return branch_width

def width_upstream(
    branch_obj,
    **kwargs
    ):
    
    return width_endpoint(
        branch_obj,
        endpoint = branch_obj.endpoint_upstream,
        **kwargs
    )

def width_downstream(
    branch_obj,
    **kwargs
    ):
    
    return width_endpoint(
        branch_obj,
        endpoint = branch_obj.endpoint_downstream,
        **kwargs
    )

def width_upstream_extra_offset(
    branch_obj,
    offset = None,
    **kwargs
    ):
    
    if offset is None:
        offset = extra_offset_skeleton_vector_global
    
    return width_upstream(
        branch_obj,
        offset = offset,
        **kwargs
    )
    
def width_downstream_extra_offset(
    branch_obj,
    offset = None,
    **kwargs
    ):
    
    if offset is None:
        offset = extra_offset_skeleton_vector_global
    
    return width_downstream(
        branch_obj,
        offset = offset,
        **kwargs
    )

# ---------- synapse dists ------------
def min_dist_synapse_endpoint(
    branch_obj,
    synapse_type,
    endpoint_type,
    verbose = False,
    default_value = np.inf,
    ):
    if synapse_type == "synapses":
        syns = branch_obj.synapses
    else:
        if "post" in synapse_type:
            syns = branch_obj.synapses_post
        elif "pre" in synapse_type:
            syns = branch_obj.synapses_pre
        else:
            raise Exception("")
        
    if len(syns) == 0:
        return default_value
    
    dists = [getattr(k,f"{endpoint_type}_dist") for k in syns]
    min_dist = np.min(dists)
    
    if verbose:
        print(f"For {synapse_type}, {endpoint_type}:min_dist={min_dist} \n   dists = {dists}")
    return min_dist


def min_dist_synapses_pre_upstream(
    branch_obj,
    **kwargs):
    return min_dist_synapse_endpoint(
    branch_obj,
    synapse_type="pre",
    endpoint_type="upstream",
    **kwargs
    )

def min_dist_synapses_post_upstream(
    branch_obj,
    **kwargs):
    return min_dist_synapse_endpoint(
    branch_obj,
    synapse_type="post",
    endpoint_type="upstream",
    **kwargs
    )

def min_dist_synapses_pre_downstream(
    branch_obj,
    **kwargs):
    return min_dist_synapse_endpoint(
    branch_obj,
    synapse_type="pre",
    endpoint_type="downstream",
    **kwargs
    )

def min_dist_synapses_post_downstream(
    branch_obj,
    **kwargs):
    return min_dist_synapse_endpoint(
    branch_obj,
    synapse_type="post",
    endpoint_type="downstream",
    **kwargs
    )


def closest_mesh_skeleton_dist(
    obj,
    verbose = False):
    """
    Purpose: To find the closest distance between mesh and 
    the skeleton of a branch
    """
    coordinates = np.array(obj.skeleton).reshape(-1,3)
    mesh_kd = KDTree(obj.mesh.triangles_center)
    dist,face_idx = mesh_kd.query(coordinates)
    min_dist = np.min(dist)
    
    if verbose:
        print(f"Closest face dist = {min_dist}")
        
    return min_dist

def mesh_shaft(
    obj,
    plot = False,
    return_mesh = True):
    """
    Purpose: To export the shaft mesh of the branch
    (aka the mesh without the spine meshes)
    """
    from mesh_tools import trimesh_utils as tu

    shaft_mesh = tu.subtract_mesh(
        obj.mesh,
        [k.mesh for k in obj.spines_obj],
        return_mesh = return_mesh)

        
    return shaft_mesh
    
def mesh_shaft_idx(obj,
    plot = False,):
    return mesh_shaft(obj,plot=plot,return_mesh=False)

def is_skeleton_upstream_to_downstream(branch_obj,verbose = False):
    upstream_endpoint = branch_obj.endpoint_upstream
    if verbose:
        print(f"endpoints = {branch_obj.endpoints}")
        print(f"upstream_endpoint= {upstream_endpoint}")
    if np.array_equal(branch_obj.endpoints[0],upstream_endpoint):
        up_flag = True
    elif np.array_equal(branch_obj.endpoints[1],upstream_endpoint):
        up_flag = False
    else:
        raise Exception("")
    
    if verbose:
        print(f"up_flag = {up_flag}")
    
    return up_flag


def width_array_upstream_to_downstream(branch_obj,verbose = False):
    is_upstream = is_skeleton_upstream_to_downstream(branch_obj,verbose)
    if is_upstream:
        return branch_obj.width_array
    else:
        if verbose:
            print(f"Applying Flip")
        return {k:np.flip(v) for k,v in branch_obj.width_array.items()}
    
def width_array_skeletal_lengths_upstream_to_downstream(branch_obj,verbose = False):
    if branch_obj.width_array_skeletal_lengths is None:
        return None
    
    is_upstream = is_skeleton_upstream_to_downstream(branch_obj,verbose)
    if is_upstream:
        return branch_obj.width_array_skeletal_lengths
    else:
        return np.flip(branch_obj.width_array_skeletal_lengths)
    
    
def skeletal_coordinates_upstream_to_downstream(
    branch_obj,
    verbose = False,
    skeleton = None,
    coordinate_dists = None,
    resize=True):
    
    if not resize:
        skeleton = branch_obj.skeleton
        
    if skeleton is None:
        array = wu.skeleton_resized_ordered(branch_obj.skeleton)
    else:
        skeleton = sk.order_skeleton(skeleton)
        array = skeleton
        
        
    is_upstream = is_skeleton_upstream_to_downstream(branch_obj,verbose)
    if is_upstream:
        pass
    else:
        if verbose:
            print(f"Applying Flip")
        array= sk.flip_skeleton(array)
        
    if branch_obj.width_array_skeletal_lengths_upstream_to_downstream is not None:
        coordinate_dists = np.concatenate([[0],branch_obj.width_array_skeletal_lengths_upstream_to_downstream])
        
    #coordinate_dists = branch_obj.width_array_skeletal_lengths_upstream_to_downstream
    
    
    if coordinate_dists is not None:
        coordinate_dists = np.cumsum(coordinate_dists)
        coordinates = sk.coordinates_from_downstream_dist(
            array,
            coordinate_dists,
            start_endpoint_coordinate = branch_obj.endpoint_upstream,
            verbose = False,
            segment_width=0,
            plot = False
        )
    else:
        coordinates = sk.skeleton_coordinate_path_from_start(array)
    
    return coordinates

def skeletal_coordinates_dist_upstream_to_downstream(
    branch_obj,
    verbose = False,
    cumsum = True,
    skeleton = None,
    **kwargs):
    
    if skeleton is None:
        array = skeletal_coordinates_upstream_to_downstream(branch_obj,**kwargs)
    else:
        array = skeleton
        
    dist_array = np.linalg.norm(array[1:] - array[:-1],axis=1)
    if cumsum:
        return np.cumsum(dist_array)
    else:
        return dist_array
    
    
def endpoint_upstream_idx(branch_obj,coordinate = None):
    if coordinate is None:
        coordinate = branch_obj.endpoint_upstream
    return nu.matching_row_index(branch_obj.endpoints,coordinate)

def endpoint_downstream_idx(branch_obj,coordinate = None):
    if coordinate is None:
        coordinate = branch_obj.endpoint_downstream
    return nu.matching_row_index(branch_obj.endpoints,coordinate)

    
    
    
synapse_dynamics_attrs = [
    "upstream_dist",
    "head_neck_shaft",
    "syn_id",
    "volume",
    "syn_type",  
    #"soma_distance"
]

spine_dynamics_attrs = [
    "upstream_dist",
    "volume",
    "area",
    "spine_id"
    #"soma_distance"
]

def width_array_upstream_to_dowstream_with_skeletal_points(
    branch_obj,
    width_name = "no_spine_median_mesh_center",
    ):
    """
    Purpose: Want to get the width at a certain
    point on the branch where that certain point
    is the closest distcretization to another coordinate
    """
    skeleton_coords = branch_obj.skeletal_coordinates_upstream_to_downstream
    skeleton_coords_mid = (skeleton_coords[:-1] + skeleton_coords[1:])/2
    
    return branch_obj.width_array_upstream_to_downstream[width_name],skeleton_coords_mid

def width_array_value_closest_to_coordinate(
    branch_obj,
    coordinate,
    verbose = False,):
    """
    Purpose: To find the width closest to certain coordinates
    on a branch obj
    """
    coordinate = np.array(coordinate)
    if coordinate.ndim == 1:
        single_flag = True
    else:
        single_flag = False
        
    coordinate = np.array(coordinate).reshape(-1,3)
    widths,sk_coordinates = width_array_upstream_to_dowstream_with_skeletal_points(branch_obj)
    closest_widths = widths[nu.closest_idx_for_each_coordinate(
        coordinate,
        sk_coordinates,
        closest_idx_algorithm = "linalg")
    ]
    
    if verbose:
        print(f"closest_widths= {closest_widths}")
        
    if single_flag:
        return closest_widths[0]
    else:
        return closest_widths




def endpoint_type_with_offset(
    branch_obj,
    endpoint_type="upstream",
    offset =1000,
    plot = False,
    verbose= False,
    ):
    """
    Purpose: To get the skeleton point
    a little offset from the current endpoint
    """
    endpoint_coordinate = getattr(branch_obj,f"endpoint_{endpoint_type}")
    coordinate = sk.skeleton_coordinate_offset_from_endpoint(
        branch_obj.skeleton,
        offset_distance = offset,
        endpoint_coordinate = endpoint_coordinate,
        plot_coordinate = plot,
    )
    
    if verbose:
        print(f"coordinate = {coordinate} (with enpoint coord = {endpoint_coordinate})")
        
    return coordinate
    
def endpoint_upstream_with_offset(
    branch_obj,
    offset =1000,
    plot = False,
    verbose= False,
    ):
    
    """
    Ex: 
    endpoint_upstream_with_offset(
        branch_obj = limb_obj[26],
        verbose = True,
        offset = 200,
        plot = True
    )
    
    """
    
    return endpoint_type_with_offset(
        branch_obj=branch_obj,
        endpoint_type="upstream",
        offset =offset,
        plot = plot,
        verbose= verbose,
        )

def endpoint_downstream_with_offset(
    branch_obj,
    offset =1000,
    plot = False,
    verbose= False,
    ):
    
    return endpoint_type_with_offset(
        branch_obj=branch_obj,
        endpoint_type="downstream",
        offset =offset,
        plot = plot,
        verbose= verbose,
        )



def skeleton_angle_from_top(
    branch_obj,
    top_of_layer_vector = None):
    if top_of_layer_vector is None:
        top_of_layer_vector = nst.top_of_layer_vector


    sk_vector = branch_obj.skeleton_vector_upstream
    angle_from_top = np.round(nu.angle_between_vectors(nst.top_of_layer_vector,sk_vector),4)
    return angle_from_top


    
    

        

            


    
def all_width_values(
    branch_obj,
    width_attributes = (
            "width_downstream",
            "width_downstream_extra_offset",
            "width_upstream",
            "width_upstream",
        ),
    width_functions = None,
    verbose = False,
    default_value = 0,
    return_dict = True,
    ):
    
    if width_functions is None:
        width_functions = []
    
    width_dict = {k:getattr(branch_obj,k,default_value) for k in width_attributes}
    for func in width_functions:
        try:
            width_dict[func.__name__] = func(branch_obj)
        except:
            width_dict[func.__name__] = 0
    
    if verbose:
        print(f"width_dict = {width_dict}")
        
    if return_dict:
        return width_dict
    else:
        return list(width_dict.values())
        
def width_extrema(
        branch_obj,
        verbose = False,
        extrema = "max",
        return_width_type = False,
        verbose_widths = False,
        **kwargs
        ):
    """
    Purpose: To find the maximum width from all the different width versions
    of a branch
    """
    width_dict = all_width_values(
        branch_obj,
        return_dict = True,
        verbose = verbose_widths,
        **kwargs
    )
    
    width_names = np.array(list(width_dict.keys()))
    width_values = np.array(list(width_dict.values()))
    
    arg_func = getattr(np,f"arg{extrema}")
    idx = arg_func(width_values)
    width_type = width_names[idx]
    width_value = width_values[idx]
    
    if verbose:
        print(f"{extrema} width value ({width_value}) from {width_type}")
    if return_width_type:
        return width_value,width_type
    else:
        return width_value
    
def width_max(
        branch_obj,
        verbose = False,
        return_width_type = False,
        **kwargs
    ):
    return width_extrema(
        branch_obj,
        verbose = verbose,
        extrema = "max",
        return_width_type = return_width_type,
        **kwargs
        )
    
def width_min(
        branch_obj,
        verbose = False,
        return_width_type = False,
        **kwargs
    ):
    return width_extrema(
        branch_obj,
        verbose = verbose,
        extrema = "min",
        return_width_type = return_width_type,
        **kwargs
        )
    
# `mesh_tools.skeleton_utils` transitively pulls cloudvolume, which can fail
# to import on some Python/cloudvolume combos (e.g. Python 3.8 + PEP 585).
# Treat it as a soft dependency: the module still imports, and functions that
# really need `sk.*` will surface a clear error only when called.
try:
    from mesh_tools import skeleton_utils as sk
except Exception:  # pragma: no cover
    sk = None




    
    

# -------------------------------------------------------


global_parameters_dict_default = dict(
    offset_skeleton_vector = 500,
    comparison_distance_skeleton_vector = 3000,
    extra_offset_skeleton_vector = 6000,
    offset_width_endpoint = 0,
)
attributes_dict_default = dict()    


# ------- microns -----------
global_parameters_dict_microns = {}
attributes_dict_microns = {}


# --------- h01 -------------
global_parameters_dict_h01 = dict()
attributes_dict_h01 = dict()





#--- from neurd_packages ---
from . import neuron_utils as nru
from . import spine_utils as spu

from . import width_utils as wu
from . import neuron_statistics as nst

 

#--- from mesh_tools ---
# `skeleton_utils as sk` is imported once above with a soft fallback.
from mesh_tools import trimesh_utils as tu

#--- from datasci_tools ---
from datasci_tools import general_utils as gu
from datasci_tools import ipyvolume_utils as ipvu
from datasci_tools import numpy_dep as np
from datasci_tools import numpy_utils as nu

