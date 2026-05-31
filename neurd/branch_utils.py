import numpy as np
from scipy.spatial import KDTree

from datasci_tools import numpy_utils as nu

try:
    from mesh_tools import skeleton_utils as sk
except Exception:  # pragma: no cover
    sk = None

from . import neuron_utils as nru
from . import width_utils as wu


# --------------- setting branch attributes --------------

def set_branch_attr_on_limb(limb_obj, func, attr_name, branch_idxs=None, **kwargs):
    """Set an attribute on every branch in a limb by calling func(limb, branch_idx)."""
    if branch_idxs is None:
        branch_idxs = limb_obj.get_branch_names()
    for branch_idx in branch_idxs:
        setattr(limb_obj[branch_idx], attr_name, func(limb_obj, branch_idx, **kwargs))
    return limb_obj


def set_branches_endpoints_upstream_downstream_idx_on_limb(limb_obj, **kwargs):
    return set_branch_attr_on_limb(
        limb_obj,
        func=nru.upstream_downstream_endpoint_idx,
        attr_name="endpoints_upstream_downstream_idx",
        **kwargs,
    )


def set_branch_attr_on_limb_on_neuron(neuron_obj, func, attr_name, verbose=False, **kwargs):
    """Set an attribute on every branch of every limb in a neuron."""
    for limb_idx in neuron_obj.get_limb_names():
        neuron_obj[limb_idx] = set_branch_attr_on_limb(
            neuron_obj[limb_idx], func, attr_name, **kwargs
        )
    return neuron_obj


def set_branches_endpoints_upstream_downstream_idx(neuron_obj, **kwargs):
    return set_branch_attr_on_limb_on_neuron(
        neuron_obj,
        func=nru.upstream_downstream_endpoint_idx,
        attr_name="endpoints_upstream_downstream_idx",
        **kwargs,
    )


# --------------- skeleton vectors --------------

def skeleton_vector_endpoint(
    branch_obj,
    endpoint_type,
    directional_flow="downstream",
    endpoint_coordinate=None,
    verbose=False,
    plot_restricted_skeleton=False,
    offset=500,
    comparison_distance=3_000,
    skeleton_attribute="skeleton",
    **kwargs,
):
    """Return the unit direction vector at an endpoint (upstream or downstream)."""
    if offset is None:
        offset = offset_skeleton_vector_global
    if comparison_distance is None:
        comparison_distance = comparison_distance_skeleton_vector_global

    if endpoint_coordinate is None:
        endpoint_coordinate = getattr(branch_obj, f"endpoint_{endpoint_type}")

    skeleton = getattr(branch_obj, skeleton_attribute)
    curr_vec = sk.vector_away_from_endpoint(
        skeleton=skeleton,
        endpoint=endpoint_coordinate,
        verbose=False,
        plot_restricted_skeleton=plot_restricted_skeleton,
        offset=offset,
        comparison_distance=comparison_distance,
        **kwargs,
    )

    mult_number = -1 if endpoint_type == directional_flow else 1
    return mult_number * curr_vec


def skeleton_vector_upstream(
    branch_obj,
    directional_flow="downstream",
    endpoint_coordinate=None,
    verbose=False,
    plot_restricted_skeleton=False,
    skeleton_attribute="skeleton",
    **kwargs,
):
    return skeleton_vector_endpoint(
        branch_obj,
        endpoint_type="upstream",
        directional_flow=directional_flow,
        endpoint_coordinate=endpoint_coordinate,
        verbose=verbose,
        plot_restricted_skeleton=plot_restricted_skeleton,
        skeleton_attribute=skeleton_attribute,
        **kwargs,
    )


def skeleton_vector_downstream(
    branch_obj,
    directional_flow="downstream",
    endpoint_coordinate=None,
    verbose=False,
    plot_restricted_skeleton=False,
    skeleton_attribute="skeleton",
    **kwargs,
):
    return skeleton_vector_endpoint(
        branch_obj,
        endpoint_type="downstream",
        directional_flow=directional_flow,
        endpoint_coordinate=endpoint_coordinate,
        verbose=verbose,
        plot_restricted_skeleton=plot_restricted_skeleton,
        skeleton_attribute=skeleton_attribute,
        **kwargs,
    )


def skeleton_vector_upstream_extra_offset(branch_obj, offset=None, skeleton_attribute="skeleton", **kwargs):
    if offset is None:
        offset = extra_offset_skeleton_vector_global
    return skeleton_vector_upstream(branch_obj, offset=offset, skeleton_attribute=skeleton_attribute, **kwargs)


def skeleton_vector_downstream_extra_offset(branch_obj, offset=None, skeleton_attribute="skeleton", **kwargs):
    if offset is None:
        offset = extra_offset_skeleton_vector_global
    return skeleton_vector_downstream(branch_obj, offset=offset, skeleton_attribute=skeleton_attribute, **kwargs)


# --------------- width at endpoints --------------

def width_endpoint(
    branch_obj,
    endpoint,
    offset=None,
    comparison_distance=2000,
    skeleton_segment_size=1000,
    verbose=False,
):
    """Compute mean branch width in a window around an endpoint."""
    if offset is None:
        offset = offset_width_endpoint_global

    _, base_final_widths, _ = nru.align_and_restrict_branch(
        branch_obj,
        common_endpoint=endpoint,
        offset=offset,
        comparison_distance=comparison_distance,
        skeleton_segment_size=skeleton_segment_size,
        verbose=False,
    )
    return np.mean(base_final_widths)


def width_upstream(branch_obj, **kwargs):
    return width_endpoint(branch_obj, endpoint=branch_obj.endpoint_upstream, **kwargs)


def width_downstream(branch_obj, **kwargs):
    return width_endpoint(branch_obj, endpoint=branch_obj.endpoint_downstream, **kwargs)


def width_upstream_extra_offset(branch_obj, offset=None, **kwargs):
    if offset is None:
        offset = extra_offset_skeleton_vector_global
    return width_upstream(branch_obj, offset=offset, **kwargs)


def width_downstream_extra_offset(branch_obj, offset=None, **kwargs):
    if offset is None:
        offset = extra_offset_skeleton_vector_global
    return width_downstream(branch_obj, offset=offset, **kwargs)


# --------------- synapse distances --------------

def min_dist_synapse_endpoint(
    branch_obj,
    synapse_type,
    endpoint_type,
    verbose=False,
    default_value=np.inf,
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

    dists = [getattr(k, f"{endpoint_type}_dist") for k in syns]
    return np.min(dists)


def min_dist_synapses_pre_upstream(branch_obj, **kwargs):
    return min_dist_synapse_endpoint(branch_obj, synapse_type="pre", endpoint_type="upstream", **kwargs)


def min_dist_synapses_post_upstream(branch_obj, **kwargs):
    return min_dist_synapse_endpoint(branch_obj, synapse_type="post", endpoint_type="upstream", **kwargs)


def min_dist_synapses_pre_downstream(branch_obj, **kwargs):
    return min_dist_synapse_endpoint(branch_obj, synapse_type="pre", endpoint_type="downstream", **kwargs)


def min_dist_synapses_post_downstream(branch_obj, **kwargs):
    return min_dist_synapse_endpoint(branch_obj, synapse_type="post", endpoint_type="downstream", **kwargs)


# --------------- mesh / skeleton geometry --------------

def closest_mesh_skeleton_dist(obj, verbose=False):
    """Minimum distance between any skeleton point and the nearest mesh face centre."""
    coordinates = np.array(obj.skeleton).reshape(-1, 3)
    mesh_kd = KDTree(obj.mesh.triangles_center)
    dist, _ = mesh_kd.query(coordinates)
    return np.min(dist)


def mesh_shaft(obj, plot=False, return_mesh=True):
    """Return the shaft mesh (branch mesh minus spine meshes)."""
    from mesh_tools import trimesh_utils as tu
    return tu.subtract_mesh(obj.mesh, [k.mesh for k in obj.spines_obj], return_mesh=return_mesh)


def mesh_shaft_idx(obj, plot=False):
    return mesh_shaft(obj, plot=plot, return_mesh=False)


def is_skeleton_upstream_to_downstream(branch_obj, verbose=False):
    upstream_endpoint = branch_obj.endpoint_upstream
    if np.array_equal(branch_obj.endpoints[0], upstream_endpoint):
        return True
    elif np.array_equal(branch_obj.endpoints[1], upstream_endpoint):
        return False
    else:
        raise Exception("")


def width_array_upstream_to_downstream(branch_obj, verbose=False):
    if is_skeleton_upstream_to_downstream(branch_obj, verbose):
        return branch_obj.width_array
    return {k: np.flip(v) for k, v in branch_obj.width_array.items()}


def width_array_skeletal_lengths_upstream_to_downstream(branch_obj, verbose=False):
    if branch_obj.width_array_skeletal_lengths is None:
        return None
    if is_skeleton_upstream_to_downstream(branch_obj, verbose):
        return branch_obj.width_array_skeletal_lengths
    return np.flip(branch_obj.width_array_skeletal_lengths)


def skeletal_coordinates_upstream_to_downstream(
    branch_obj,
    verbose=False,
    skeleton=None,
    coordinate_dists=None,
    resize=True,
):
    if not resize:
        skeleton = branch_obj.skeleton

    if skeleton is None:
        array = wu.skeleton_resized_ordered(branch_obj.skeleton)
    else:
        array = sk.order_skeleton(skeleton)

    if not is_skeleton_upstream_to_downstream(branch_obj, verbose):
        array = sk.flip_skeleton(array)

    if branch_obj.width_array_skeletal_lengths_upstream_to_downstream is not None:
        coordinate_dists = np.concatenate(
            [[0], branch_obj.width_array_skeletal_lengths_upstream_to_downstream]
        )

    if coordinate_dists is not None:
        coordinate_dists = np.cumsum(coordinate_dists)
        coordinates = sk.coordinates_from_downstream_dist(
            array,
            coordinate_dists,
            start_endpoint_coordinate=branch_obj.endpoint_upstream,
            verbose=False,
            segment_width=0,
            plot=False,
        )
    else:
        coordinates = sk.skeleton_coordinate_path_from_start(array)

    return coordinates


def skeletal_coordinates_dist_upstream_to_downstream(
    branch_obj,
    verbose=False,
    cumsum=True,
    skeleton=None,
    **kwargs,
):
    if skeleton is None:
        array = skeletal_coordinates_upstream_to_downstream(branch_obj, **kwargs)
    else:
        array = skeleton

    dist_array = np.linalg.norm(array[1:] - array[:-1], axis=1)
    return np.cumsum(dist_array) if cumsum else dist_array


# --------------- width array lookups --------------

def width_array_upstream_to_dowstream_with_skeletal_points(
    branch_obj,
    width_name="no_spine_median_mesh_center",
):
    """Return (width_array, midpoint coordinates) ordered upstream→downstream."""
    skeleton_coords = branch_obj.skeletal_coordinates_upstream_to_downstream
    skeleton_coords_mid = (skeleton_coords[:-1] + skeleton_coords[1:]) / 2
    return branch_obj.width_array_upstream_to_downstream[width_name], skeleton_coords_mid


def width_array_value_closest_to_coordinate(branch_obj, coordinate, verbose=False):
    """Return the width value(s) at the skeleton point(s) closest to coordinate."""
    coordinate = np.array(coordinate)
    single_flag = coordinate.ndim == 1
    coordinate = coordinate.reshape(-1, 3)

    widths, sk_coordinates = width_array_upstream_to_dowstream_with_skeletal_points(branch_obj)
    closest_widths = widths[
        nu.closest_idx_for_each_coordinate(coordinate, sk_coordinates, closest_idx_algorithm="linalg")
    ]

    return closest_widths[0] if single_flag else closest_widths


# --------------- endpoint offsets --------------

def endpoint_type_with_offset(branch_obj, endpoint_type="upstream", offset=1000, plot=False, verbose=False):
    """Return the skeleton coordinate offset distance from an endpoint."""
    endpoint_coordinate = getattr(branch_obj, f"endpoint_{endpoint_type}")
    return sk.skeleton_coordinate_offset_from_endpoint(
        branch_obj.skeleton,
        offset_distance=offset,
        endpoint_coordinate=endpoint_coordinate,
        plot_coordinate=plot,
    )


def endpoint_upstream_with_offset(branch_obj, offset=1000, plot=False, verbose=False):
    return endpoint_type_with_offset(branch_obj, endpoint_type="upstream", offset=offset, plot=plot, verbose=verbose)


def endpoint_downstream_with_offset(branch_obj, offset=1000, plot=False, verbose=False):
    return endpoint_type_with_offset(branch_obj, endpoint_type="downstream", offset=offset, plot=plot, verbose=verbose)


# --------------- parameter system --------------

global_parameters_dict_default = dict(
    offset_skeleton_vector=500,
    comparison_distance_skeleton_vector=3000,
    extra_offset_skeleton_vector=6000,
    offset_width_endpoint=0,
)
attributes_dict_default = dict()

global_parameters_dict_microns = {}
attributes_dict_microns = {}

global_parameters_dict_h01 = dict()
attributes_dict_h01 = dict()
