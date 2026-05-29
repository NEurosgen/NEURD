
import copy
import networkx as nx
from datasci_tools import numpy_dep as np
from datasci_tools import general_utils as gu
import pandas as pd
top_of_layer_vector = np.array([0,-1,0])

def soma_starting_vector(limb_obj=None,
                        neuron_obj=None,
                        limb_idx=None,
                        soma_idx = 0,
                        soma_group_idx = None,
                        soma_center = None):
    """
    Will find the angle between the vector pointing to the
    top of the volume and the angle from the soma center to the starting skeleton
    coordinate
    
    
    """
    if limb_obj is None:
        curr_limb_obj = neuron_obj[limb_idx]
        
    else:
        curr_limb_obj = limb_obj
        
    if soma_group_idx is not None:
        curr_limb_obj.set_concept_network_directional(starting_soma=soma_idx,
                                                     soma_group_idx=soma_group_idx)
        
    st_coordinates = curr_limb_obj.current_starting_coordinate
    
    if soma_center is None:
        curr_soma = neuron_obj[f"S{soma_idx}"].mesh
        soma_center = tu.mesh_center_vertex_average(curr_soma)
    
    
    #1) Get starting angle of branch
    st_vector = st_coordinates - soma_center
    st_vector_norm = st_vector/np.linalg.norm(st_vector)
    return st_vector_norm

def soma_starting_angle(limb_obj=None,
                        neuron_obj=None,
                        limb_idx=None,
                        soma_idx = 0,
                        soma_group_idx = None,
                        soma_center = None,
                        y_vector = np.array([0,-1,0])):
    st_vector_norm = nst.soma_starting_vector(limb_obj=limb_obj,
                        neuron_obj=neuron_obj,
                        limb_idx=limb_idx,
                        soma_idx = soma_idx,
                        soma_group_idx = soma_group_idx,
                        soma_center = soma_center,)
    angle_from_top = np.round(nu.angle_between_vectors(y_vector,st_vector_norm),2)
    
    return angle_from_top

def fork_divergence_from_skeletons(upstream_skeleton,
                    downstream_skeletons,
                    downstream_starting_endpoint = None,
                    comparison_distance = 500,
                    skeletal_segment_size = 50,
                    plot_restrictions = False,
                    combining_function = np.sum,
                                  verbose=False):
    """
    Purpose: To compute the number for the fork splitting

    Pseudocode: 
    1) Find intersection point of all 3 branches
    2) for 2 downstream branch: 
       - restrict the skeleton to a certain distance from the start
    3) discretize the skeeletons so have x pieces
    4) Measure distance between each indexed point
    5) Have one way of aggregating the distances (median, mean)

    Application: If below a certain value then can indicate incorrect branching


    Ex:
    from neurd import neuron_statistics as nst
    nst.fork_divergence(upstream_skeleton = upstream_sk,
                        downstream_skeletons = downstream_sk,
                        comparison_distance = 500,
                        skeletal_segment_size = 50,
                        plot_restrictions = True,
                        combining_function = np.mean)
    
    """
    downstream_sk = downstream_skeletons
    upstream_sk = upstream_skeleton
    
    if downstream_starting_endpoint is None:
        joining_endpoint_1 = sk.shared_coordiantes(list(downstream_sk) + [upstream_skeleton],
                                 return_one=True)
    else:
        joining_endpoint_1 = downstream_starting_endpoint
        
    if verbose:
        print(f"joining_endpoint_1 = {joining_endpoint_1}")

    # resize the skeletons
    d_skeletons_resized = [sk.resize_skeleton_branch(k,segment_width=skeletal_segment_size)
                          for k in downstream_skeletons]
    
    if verbose:
#         print(f"After resizing:\n{[sk.calculate_skeleton_segment_distances(k) for k in d_skeletons_resized]}")
        pass
        

    d_skeletons_restricted = [sk.restrict_skeleton_to_distance_from_coordinate(k,
                                             coordinate = joining_endpoint_1,
                                             distance_threshold = comparison_distance+1,)
                              for k in d_skeletons_resized]
    d_skeletons_resized = [sk.resize_skeleton_branch(k,segment_width=skeletal_segment_size)
                           for k in d_skeletons_restricted]
                           
    if verbose:
        #print(f"After restricting:\n{[sk.calculate_skeleton_distance(k) for k in d_skeletons_restricted]}")
        pass
        
    if verbose:
        print(f"Segment sizes after restriction")
        for d in d_skeletons_resized:
            print(sk.calculate_skeleton_segment_distances(d))

    #align the skeletons


    d_ordered_paths = [sk.skeleton_coordinate_path_from_start(k,start_endpoint_coordinate=joining_endpoint_1) for k in d_skeletons_restricted]
    
    
    if verbose:
        #print(f"Before skeleton path from start: {d_skeletons_restricted}")
        #print(f"After skeleton path from start: {d_ordered_paths}")
        for d in d_ordered_paths:
            print(sk.calculate_skeleton_segment_distances(sk.skelton_coordinate_path_to_skeleton(d)))
        
    min_length = np.min([len(k) for k in d_ordered_paths])
    if verbose:
        print(f"min_length = {min_length}")
        
    path_distances = np.linalg.norm(d_ordered_paths[0][:min_length]-d_ordered_paths[1][:min_length],axis=1)

    return_value = combining_function(path_distances)
    
    return return_value


def fork_divergence_from_branch(limb_obj,
                                branch_idx,
                                verbose = False,
                                error_not_2_downstream=True,
                                total_downstream_skeleton_length_threshold = 4000,
                                individual_branch_length_threshold = 3000,
                                skip_value = np.inf,
                                plot_fork_skeleton = False,
                                upstream_sk_color="red",
                                downstream_sk_colors = None,
                                
                                #arguments for the fork divergence measurement
                                comparison_distance = 400,
                                skeletal_segment_size = 40,
                                plot_restrictions = False,
                                combining_function = np.mean,
                                **kwargs
                               ):
    """
    Purpose

    Pseudocode: 
    1) Get the branch where the error is
    2) Get all the upstream branch and all of the downstream branches of that
    3) Measure the sibling angles
    4) Collect together the skeletons in a list
    5) Run the fork splitting function

    Note: should only be done on 2 forks
    

    """

    return_value = None
    #2) Get all the upstream branch and all of the downstream branches of that
    upstream_node = xu.upstream_node(limb_obj.concept_network_directional,
                                     branch_idx)
    if upstream_node is None:
        return skip_value
    
    downstream_nodes = xu.downstream_nodes(limb_obj.concept_network_directional,upstream_node)

    if error_not_2_downstream and len(downstream_nodes) != 2:
        raise Exception(f"Not exactly 2 downstream nodes: {downstream_nodes}")


    upstream_sk = limb_obj[upstream_node].skeleton
    downstream_sk = [limb_obj[d].skeleton for d in downstream_nodes]

    if verbose:
        print(f"Upstream Node = {upstream_node}, downstream nodes = {downstream_nodes}")


    if (total_downstream_skeleton_length_threshold is not None and
        individual_branch_length_threshold is not None):
        d_len = np.array([sk.calculate_skeleton_distance(limb_obj[k].skeleton) for 
                         k in downstream_nodes])
        d_skeletal_len = np.array([nru.skeletal_length_over_downstream_branches(limb_obj,
                                                d,
                                                verbose=False) for d in downstream_nodes])
        below_threshold = np.where((d_skeletal_len<total_downstream_skeleton_length_threshold) | 
                                   (d_len < individual_branch_length_threshold))[0]
        if len(below_threshold) > 0:
            if verbose:
                print(f"Skipping this intersection because some of downstream skeletal lengths too short (min {total_downstream_skeleton_length_threshold}):")
                print(f" or the individual branch length was too short (min {individual_branch_length_threshold})")
                for j,(d,d_len) in enumerate(zip(downstream_nodes,d_skeletal_len)):
                    if j in below_threshold:
                        print(f"Brnach {d}: length {d_l} {d_len}")

            return_value = skip_value


    if return_value is None:
        #3) Measure the sibling angles
        sibling_angle = list(nru.find_sibling_child_skeleton_angle(curr_limb_obj=limb_obj,
                                             child_node=branch_idx).values())
        if verbose:
            print(f"sibling_angle = {sibling_angle}")

        #4) Collect together the skeletons in a list


        if downstream_sk_colors is None:
            from datasci_tools import matplotlib_utils as mu
            downstream_sk_colors = mu.generate_color_list(n_colors = len(downstream_nodes),
                                  colors_to_omit = [upstream_sk_color])


        
        return_value = nst.fork_divergence_from_skeletons(upstream_skeleton = upstream_sk,
                            downstream_skeletons = downstream_sk,
                            comparison_distance = comparison_distance,
                            skeletal_segment_size = skeletal_segment_size,
                            plot_restrictions = plot_restrictions,
                            combining_function = combining_function,
                                                         verbose=verbose)


    if verbose:
        print(f"Final Fork Divergence = {return_value}")
        
    return return_value


def child_angles(limb_obj,
    branch_idx,
    verbose = False,
    comparison_distance=1500,
    ):

    """
    Purpose: To measure all of the angles betweent he children nodes

    Psuedocode: 
    1) Get the downstream nodes --> if none or one then return empty dictionary

    For all downstream nodes:
    2) choose one of the downstream nodes and send to nru.find_sibling_child_skeleton_angle
    3) create a dictionary with the nodes in a tuple as the key and the angle between 
    them as the values

    will error if more than 2 children current
    
    Ex: 
    nst.child_angles(limb_obj = neuron_obj[6],
    branch_idx = 22,
    verbose = False,
    )

    """




    #1) Get the downstream nodes --> if none or one then return empty dictionary
    downstream_n = xu.downstream_nodes(limb_obj.concept_network_directional,
                            branch_idx)

    if len(downstream_n) in [0,1]:
        return {}
    elif len(downstream_n) > 2:
        raise Exception(f"Not implemented for number of children more than 2 and currently there are {len(downstream_n) }")
    else:
        pass

    #2) choose one of the downstream nodes and send to nru.find_sibling_child_skeleton_angle
    sibling_angles = nru.find_sibling_child_skeleton_angle(limb_obj,
                                         downstream_n[0],
                                            comparison_distance=comparison_distance)
    return_dict = {(downstream_n[0],k):v for k,v in sibling_angles.items()}

    if verbose:
        print(f"return_dict= {return_dict}")
    return return_dict


def angle_from_top(vector,
    vector_pointing_to_top = np.array([0,-1,0]),
                   verbose=True
    ):
    """
    Purpose: Will find the angle between
    a vector and the vector pointing towards the top
    of the volume
    
    """
    if vector.ndim == 2:
        vector = vector[-1] - vector[0] 
    st_vector_norm = vector/np.linalg.norm(vector)
    angle_from_top = np.round(nu.angle_between_vectors(vector_pointing_to_top,st_vector_norm),2)
    return angle_from_top


def children_skeletal_lengths(limb_obj,
                                        branch_idx,
                                        verbose = False,
                                        return_dict = True):
    """
    Purpose: To generate the downstream skeletal lengths of all children
    
    Pseudocode:
    1) Find the downstream nodes
    2) Compute the downstream skeletal length for each
    3) return as dictionary
    """
    
    vals = dict([(k,nru.skeletal_length_over_downstream_branches(limb_obj,
                                             branch_idx = k,
                                             include_branch_skeletal_length=True))
                  for k in xu.downstream_nodes(limb_obj.concept_network_directional,branch_idx)])
    if verbose:
        print(f"downstream values = {vals}")
    if return_dict:
        return vals
    else:
        return list(vals.values())

def children_skeletal_lengths_min(limb_obj,
                                        branch_idx,
                                        verbose = False):
    return_sk_len = nst.children_skeletal_lengths(limb_obj,
                                        branch_idx,
                                        verbose = verbose,
                                        return_dict = False)
    if len(return_sk_len) == 0:
        return 0
    else:
        return np.min(return_sk_len)

    
    
def upstream_skeletal_length(limb_obj,
                       branch_idx,
                        default = np.inf,
                        **kwargs):
    """
    Purpose: To return the skeletal length of the upstream branch
    
    Psuedocode: 
    1) Get the upstream branch
    2) return the width
    """
    up_node = xu.upstream_node(limb_obj.concept_network_directional,branch_idx)
    if up_node is None:
        return default
    else:
        return limb_obj[up_node].skeletal_length
    
    
def skeletal_length_along_path(limb_obj,
                              branch_path):
    return np.sum([limb_obj[k].skeletal_length for k in branch_path])
def total_upstream_skeletal_length(limb_obj,
                                  branch_idx,
                                   include_branch=False,
                                  **kwargs):
    """
    Purpose: To get all of the skeleton length from current branch to 
    starting branch
    
    """
    #print(f"include_branch = {include_branch}")
    
    branch_path = nru.branch_path_to_start_node(limb_obj,
                             branch_idx,
                             include_branch_idx = include_branch,
                            skeletal_length_min = None,
                            verbose = False)
    return skeletal_length_along_path(limb_obj,branch_path)

# --------- 6/17: Functions for helping pair branches with each other ----------#
def width_new(branch,width_new_name="no_spine_mean_mesh_center",
              width_new_name_backup = "no_spine_median_mesh_center",
              **kwargs):
    try:
        return branch.width_new[width_new_name]
    except:
        return branch.width_new[width_new_name_backup]




def parent_child_sk_angle(limb_obj,
                                branch_1_idx,
                                branch_2_idx,
                         **kwargs):
    return nst.find_parent_child_skeleton_angle_upstream_downstream(limb_obj,
                                                        branch_1_idx,
                                                        branch_2_idx,
                                                        branch_1_type="upstream",
                                                        branch_2_type = "downstream",
                                                        use_upstream_skeleton_restriction=True,
                                                        **kwargs)

def sibling_sk_angle(limb_obj,
                                branch_1_idx,
                                branch_2_idx,
                         **kwargs):
    return nst.find_parent_child_skeleton_angle_upstream_downstream(limb_obj,
                                                        branch_1_idx,
                                                        branch_2_idx,
                                                        branch_1_type="downstream",
                                                        branch_2_type = "downstream",
                                                        use_upstream_skeleton_restriction=True,
                                                        **kwargs)
# def none_to_some_synapses(limb_obj,
#                           branch_1_idx,
#                          branch_2_idx,
#                          synapse_type=None,):
#     """
#     Purpose: To indicate if there were no synapses and then some synapses
#     """
#     attr_name = "synapses"
#     if synapse_type is not None:
#         attr_name += f"_{synapse_type}"
        
#     n_syn_branch_1 = len(getattr(limb_obj[branch_1_idx],attr_name))
#     n_syn_branch_2 = len(getattr(limb_obj[branch_2_idx],attr_name))
    
#     if n_syn_branch_1 > 0 and n_syn_branch_2 > 0:
#         return False
    
#     if n_syn_branch_1 > 0 or n_syn_branch_2 > 0: 
#         return True
    
#     return False
    
# def n_synapses_diff(limb_obj,
#                       branch_1_idx,
#                       branch_2_idx,
#                   synapse_type=None,
#                   verbose = False):
#     """
#     Purpose: Will return the different in number of synapses
    
#     """
#     attr_name = "synapses"
#     if synapse_type is not None:
#         attr_name += f"_{synapse_type}"
        
#     n_syn_branch_1 = len(getattr(limb_obj[branch_1_idx],attr_name))
#     n_syn_branch_2 = len(getattr(limb_obj[branch_2_idx],attr_name))
    
#     if verbose:
#         print(f"Using {attr_name}")
#         print(f"n_syn_branch_1 = {n_syn_branch_1}, n_syn_branch_2 = {n_syn_branch_2}")
    
#     return np.abs(n_syn_branch_1 - n_syn_branch_2)

def n_synapses_diff(limb_obj,
                      branch_1_idx,
                      branch_2_idx,
                    synapse_type="synapses",
                    branch_1_direction = "upstream",
                    branch_2_direction = "downstream",
                    comparison_distance = 10000,
                    nodes_to_exclude=None,
                  verbose = False,
                   **kwargs):
    """
    Purpose: Will return the different in number of synapses
    
    """
    #print(f"synapse_type inside nst = {synapse_type}")
    n_syn_branch_1 = cnu.synapses_upstream_downstream(limb_obj,
                                                            branch_1_idx,
                                                             synapse_type=synapse_type,
                                                            direction=branch_1_direction,
                                                            distance=comparison_distance,
                                                            nodes_to_exclude=nodes_to_exclude)
    n_syn_branch_2 = cnu.synapses_upstream_downstream(limb_obj,
                                                            branch_2_idx,
                                                             synapse_type=synapse_type,
                                                            direction=branch_2_direction,
                                                            distance=comparison_distance,
                                                            nodes_to_exclude=nodes_to_exclude)
        
    n_syn_branch_1 = len(n_syn_branch_1)
    n_syn_branch_2 = len(n_syn_branch_2)
    
    if verbose:
        print(f"Using {synapse_type}")
        print(f"n_syn_branch_1 = {n_syn_branch_1}, n_syn_branch_2 = {n_syn_branch_2}")
    
    return np.abs(n_syn_branch_1 - n_syn_branch_2)

# def synapse_density_diff(limb_obj,
#                       branch_1_idx,
#                       branch_2_idx,
#                   synapse_type=None,
#                   verbose = False):
#     """
#     Purpose: Will return the different in number of synapses
    
#     """
#     attr_name = "synapse_density"
#     if synapse_type is not None:
#         attr_name += f"_{synapse_type}"
        
#     n_syn_branch_1 = getattr(limb_obj[branch_1_idx],attr_name)
#     n_syn_branch_2 = getattr(limb_obj[branch_2_idx],attr_name)
    
#     if verbose:
#         print(f"Using {attr_name}")
#         print(f"n_syn_branch_1 = {n_syn_branch_1}, n_syn_branch_2 = {n_syn_branch_2}")
    
#     return np.abs(n_syn_branch_1 - n_syn_branch_2)

def synapse_density_diff(limb_obj,
                      branch_1_idx,
                      branch_2_idx,
                  synapse_type="synapse_density",
                    branch_1_direction = "upstream",
                    branch_2_direction = "downstream",
                    comparison_distance = 10000,
                    nodes_to_exclude=None,
                  verbose = False,
                        **kwargs):
    """
    Purpose: Will return the different in number of synapses
    
    """
        
    n_syn_branch_1 = cnu.synapse_density_upstream_downstream(limb_obj,
                                                            branch_1_idx,
                                                             synapse_density_type=synapse_type,
                                                            direction=branch_1_direction,
                                                            distance=comparison_distance,
                                                            nodes_to_exclude=nodes_to_exclude)
    n_syn_branch_2 = cnu.synapse_density_upstream_downstream(limb_obj,
                                                            branch_2_idx,
                                                             synapse_density_type=synapse_type,
                                                            direction=branch_2_direction,
                                                            distance=comparison_distance,
                                                            nodes_to_exclude=nodes_to_exclude)
    
    if verbose:
        print(f"Using {synapse_type}")
        print(f"syn_density_branch_1 = {n_syn_branch_1}, syn_density_branch_2 = {n_syn_branch_2}")
    
    return np.abs(n_syn_branch_1 - n_syn_branch_2)


def compute_edge_attributes_locally(G,
                                          limb_obj,
                                         nodes_to_compute,
                                         edge_functions,
                                         arguments_for_all_edge_functions = None,
                                         verbose=False,
                                         directional = False,
                                         set_default_at_end = True,
                                         default_value_at_end = None,
                                         **kwargs):
    """
    Purpose: To iterate over graph edges and compute
    edge properties and store

    Pseudocode: 
    For each nodes to compute:
        get all of the edges for that node
        For each downstream partner:
            For each function:
                compute the value and store it in the edge
    Ex: 
    G = complete_graph_from_node_ids(all_branch_idx)

    nodes_to_compute = [upstream_branch]
    edge_functions = dict(sk_angle=nst.parent_child_sk_angle,
                         width_diff = nst.width_diff,
                          width_diff_percentage = nst.width_diff_percentage)

    compute_edge_attributes_between_nodes(G,
                                             nodes_to_compute,
                                             edge_functions,
                                             verbose=True,
                                             directional = False)

    """
    G = copy.deepcopy(G)
    
    if directional:
        neighbors_func = xu.downstream_nodes
    else:
        neighbors_func = xu.get_neighbors

    func_value_dict = dict()
    for n in nodes_to_compute:
        func_value_dict[n] = dict()
        down_nodes = neighbors_func(G,n)
        if verbose:
            print(f"Working on node {n}")

        for d in down_nodes:
            func_value_dict[n][d] = dict()
            if verbose:
                print(f"   Neighbor {d}")
            for func_name,func_info in edge_functions.items():
                if not callable(func_info):
                    func = func_info["function"]
                    if "arguments" in func_info.keys():
                        args = func_info["arguments"]
                    else:
                        args = dict()
                else:
                    func = func_info
                    args = dict()
                    
                if arguments_for_all_edge_functions is not None:
                    func_args = dict(arguments_for_all_edge_functions)
                else:
                    func_args = dict()
                
                func_args.update(args)
                
                func_value = func(limb_obj,n,d,**func_args)
            
                if verbose:
                    print(f"      {func_name}: {func_value}")
                func_value_dict[n][d][func_name] = func_value
    
    xu.apply_edge_attribute_dict_to_graph(G,func_value_dict)
    
    if set_default_at_end:
        for func_name,func_info in edge_functions.items():
            d_value = default_value_at_end
            try:
                if "default_value" in func_info:
                    d_value = func_info["defualt_value"]
            except:
                pass
            xu.set_edge_attribute_defualt(G,func_name,d_value)
                
    return G

def compute_edge_attributes_locally_upstream_downstream(
            limb_obj,
            upstream_branch,
            downstream_branches,
            offset=1500,
            comparison_distance = 2000,
            plot_extracted_skeletons = False,
            concept_network_comparison_distance = 10000,
            synapse_density_diff_type = "synapse_density_pre",
            n_synapses_diff_type = "synapses_pre",
            
    
    ):
    """
    To compute a graph storing the values for the
    edges between the nodes
    """
    all_branch_idx = np.hstack([downstream_branches,[upstream_branch]])
    G = xu.complete_graph_from_node_ids(all_branch_idx)
    
    nodes_to_compute = [upstream_branch]
    
    arguments_for_all_edge_functions = dict(
                                        #nodes_to_exclude=nodes_to_exclude,
                                           branch_1_direction="upstream",
                                            branch_2_direction="downstream",
                                           comparison_distance = concept_network_comparison_distance)
    
    edge_functions = dict(sk_angle=dict(function=nst.parent_child_sk_angle,
                                        arguments=dict(offset=offset,
                                                      comparison_distance=comparison_distance,
                                                      plot_extracted_skeletons=plot_extracted_skeletons)),
                         width_diff = nst.width_diff,
                          width_diff_percentage = nst.width_diff_percentage,
                         synapse_density_diff=dict(function=nst.synapse_density_diff,
                                               arguments=dict(synapse_type = synapse_density_diff_type)),
                          n_synapses_diff = dict(function=nst.n_synapses_diff,
                                                 arguments=dict(synapse_type=n_synapses_diff_type)),
                          #none_to_some_synapses = nst.none_to_some_synapses
                         )

    G_e_1 = nst.compute_edge_attributes_locally(G,
                                              limb_obj,
                                             nodes_to_compute,
                                             edge_functions,
                                                arguments_for_all_edge_functions=arguments_for_all_edge_functions,
                                             verbose=False,
                                             directional = False)
    
    nodes_to_compute = downstream_branches
    
    arguments_for_all_edge_functions = dict(
                                        #nodes_to_exclude=nodes_to_exclude,
                                           branch_1_direction="downstream",
                                            branch_2_direction="downstream",
                                           comparison_distance = concept_network_comparison_distance)
    
    edge_functions = dict(
                          sk_angle=dict(function=nst.sibling_sk_angle,
                                        arguments=dict(offset=offset,
                                                      comparison_distance=comparison_distance,
                                                plot_extracted_skeletons=plot_extracted_skeletons)),
                         width_diff = nst.width_diff,
                          width_diff_percentage = nst.width_diff_percentage,
                         synapse_density_diff=dict(function=nst.synapse_density_diff,
                                               arguments=dict(synapse_type = synapse_density_diff_type)),
                          n_synapses_diff = dict(function=nst.n_synapses_diff,
                                                 arguments=dict(synapse_type=n_synapses_diff_type)),
                         #none_to_some_synapses = nst.none_to_some_synapses
    )

    G_e_2 = nst.compute_edge_attributes_locally(G_e_1,
                                              limb_obj,
                                             nodes_to_compute,
                                             edge_functions,
                                                arguments_for_all_edge_functions=arguments_for_all_edge_functions,
                                             verbose=False,
                                             directional = False)
    return G_e_2
                
                
# --------- For the global deleteion functions ------------

def compute_edge_attributes_globally(G,
                                     edge_functions,
                                     edges_to_compute=None,
                                     arguments_for_all_edge_functions = None,
                                     verbose=False,
                                         set_default_at_end = True,
                                         default_value_at_end = None,
                                     **kwargs):
    """
    Purpose: to compute edge attributes
    that need the whole graph to be computed
    
    """
    G = copy.deepcopy(G)
    
    if edges_to_compute is None:
        edges_to_compute = xu.edges(G)
    
    other_edges_to_remove = []
    for e in edges_to_compute:
        if verbose:
            print(f"   Working on Edge {e}")
            
        for func_name,func_info in edge_functions.items():
            if not callable(func_info):
                func = func_info["function"]
                if "arguments" in func_info.keys():
                    args = func_info["arguments"]
                else:
                    args = dict()
            else:
                func = func_info
                args = dict()
                
                
            if arguments_for_all_edge_functions is not None:
                func_args = dict(arguments_for_all_edge_functions)
            else:
                func_args = dict()

            func_args.update(args)
                
            global_edge_dict = func(G,e[0],e[1],**func_args)
            
            if verbose:
                print(f"      {func_name}: {global_edge_dict}")
                
            xu.apply_edge_attribute_dict_to_graph(G,global_edge_dict,label=func_name)
            
            
    if set_default_at_end:
        for func_name,func_info in edge_functions.items():
            d_value = default_value_at_end
            try:
                if "default_value" in func_info:
                    d_value = func_info["defualt_value"]
            except:
                pass
            xu.set_edge_attribute_defualt(G,func_name,d_value)
    return G
            



# -------------- For node level edge attributes ------------- 3
def compute_edge_attributes_around_node(G,
                                        edge_functions,
                                        edge_functions_args = dict(),
                                        nodes_to_compute=None,
                                        arguments_for_all_edge_functions = None,
                                        verbose=False,
                                        #directional = False,
                                         set_default_at_end = True,
                                         default_value_at_end = None,
                                        **kwargs):
    """
    Purpose: To use all the edges around a node 
    to compute edge features

    """
    G = copy.deepcopy(G)
    
    if nodes_to_compute is None:
        nodes = list(G.nodes())
    
    if not nu.is_array_like(nodes_to_compute):
        nodes_to_compute = [nodes_to_compute]
        
    if verbose:
        print(f"nodes_to_compute = {nodes_to_compute}")

    for n in nodes_to_compute:
        node_edges = xu.node_to_edges(G,n)
        for func_name,func_info in edge_functions.items():
            if not callable(func_info):
                func = func_info["function"]
                if "arguments" in func_info.keys():
                    args = func_info["arguments"]
                else:
                    args = dict()
            else:
                func = func_info
                args = dict()
                
            if arguments_for_all_edge_functions is not None:
                func_args = dict(arguments_for_all_edge_functions)
            else:
                func_args = dict()

            func_args.update(args)
                
            node_edge_dict = func(G,node_edges,**func_args)
            
            if verbose:
                print(f"      {func_name}: {node_edge_dict}")
                
            xu.apply_edge_attribute_dict_to_graph(G,node_edge_dict,label=func_name)
            
            
    if set_default_at_end:
        for func_name,func_info in edge_functions.items():
            d_value = default_value_at_end
            try:
                if "default_value" in func_info:
                    d_value = func_info["defualt_value"]
            except:
                pass
            xu.set_edge_attribute_defualt(G,func_name,d_value)
    return G
            



def find_parent_child_skeleton_angle_upstream_downstream(limb_obj,
                                                         branch_1_idx,
                                                         branch_2_idx,
                                                         branch_1_type = "upstream",
                                                         branch_2_type = "downstream",
                                                        verbose = False,
                                                        offset=1500,
                                                        min_comparison_distance = 1000,
                                                        comparison_distance = 2000,
                                                         skeleton_resolution = 100,
                                                        plot_extracted_skeletons = False,
                                                         use_upstream_skeleton_restriction=True,
                                                         use_downstream_skeleton_restriction = True,
                                                         nodes_to_exclude = None,
                                                         **kwargs
                                                        ):
    """
    Purpose: to find the skeleton angle between a designated
    upstream and downstream branch
    
    Ex: 
    nru.find_parent_child_skeleton_angle_upstream_downstream(
        limb_obj = neuron_obj[0],
    branch_1_idx = 223,
    branch_2_idx = 224,
        plot_extracted_skeletons = True
    )
    
    
    Ex: 
    branch_idx = 140
    nru.find_parent_child_skeleton_angle_upstream_downstream(limb_obj,
                                                            nru.upstream_node(limb_obj,branch_idx),branch_idx,
                                                            verbose = True,
                                                            plot_extracted_skeletons=True,
                                                            comparison_distance=40000,
                                                            use_upstream_skeleton_restriction=True)
    """
    output_sk = []
    endpoints = []
    for b_idx,b_type in zip([branch_1_idx,branch_2_idx],
                            [branch_1_type,branch_2_type]):
        
        upstream_func = False
        if b_type == "downstream":
            b_endpt = nru.upstream_endpoint(limb_obj,
                                 b_idx)
            if use_downstream_skeleton_restriction:
                downstream_func = True
        else: 
            b_endpt = nru.downstream_endpoint(limb_obj,
                                 b_idx)
            if use_upstream_skeleton_restriction:
                upstream_func = True
        
        if upstream_func:
            curr_sk = nru.restrict_skeleton_from_start_plus_offset_upstream(
                                                                limb_obj,
                                                                b_idx,
                                                                offset=offset,
                                                                comparison_distance=comparison_distance,
                                                                min_comparison_distance=min_comparison_distance,
                                                                verbose=verbose,
                                                                start_coordinate=b_endpt,
                                                                skeleton_resolution = skeleton_resolution,
                                                                nodes_to_exclude = nodes_to_exclude)
        elif downstream_func:
            curr_sk = nru.restrict_skeleton_from_start_plus_offset_downstream(
                                                                limb_obj,
                                                                b_idx,
                                                                offset=offset,
                                                                comparison_distance=comparison_distance,
                                                                min_comparison_distance=min_comparison_distance,
                                                                verbose=verbose,
                                                                start_coordinate=b_endpt,
                                                                skeleton_resolution = skeleton_resolution,
                                                                nodes_to_exclude=nodes_to_exclude)
        else:
            curr_sk = sk.restrict_skeleton_from_start_plus_offset(limb_obj[b_idx].skeleton,
                                                   offset=offset,
                                                comparison_distance=comparison_distance,
                                                    min_comparison_distance=min_comparison_distance,
                                                verbose=verbose,
                                                 start_coordinate=b_endpt,
                                                                  skeleton_resolution = skeleton_resolution
                                                   )
        output_sk.append(curr_sk)
        endpoints.append(b_endpt)
        
        if verbose:
            print(f"{b_type} {b_idx} endpoint: {b_endpt}")

    up_sk = output_sk[0]
    d_sk = output_sk[1]
    
#     sk.restrict_skeleton_from_start_plus_offset(limb_obj[branch_1_idx].skeleton,
#                                                    offset=offset,
#                                                 comparison_distance=comparison_distance,
#                                                     min_comparison_distance=min_comparison_distance,
#                                                 verbose=verbose,
#                                                  start_coordinate=b_1_endpt,
#                                                    )

#     d_sk = sk.restrict_skeleton_from_start_plus_offset(limb_obj[branch_2_idx].skeleton,
#                                                    offset=offset,
#                                                 comparison_distance=comparison_distance,
#                                                     min_comparison_distance=min_comparison_distance,
#                                                 verbose=verbose,
#                                                  start_coordinate=b_2_endpt,
#                                                    )

    curr_angle = sk.parent_child_skeletal_angle(up_sk,d_sk)
    if verbose:
        print(f"curr_angle = {curr_angle}")
        

    return curr_angle

    
def ray_trace_perc(branch_obj,percentile=85):
    return tu.mesh_size(branch_obj.mesh,'ray_trace_percentile',percentile)




# ---------------- Getting Attributes Using the concept network walking ----------- #

def skeleton_upstream(limb_obj,
                            branch_idx,
                            nodes_to_exclude=None,
                            **kwargs):
    return cnu.skeleton_upstream(limb_obj,
                            branch_idx,
                            nodes_to_exclude=nodes_to_exclude,
                                       **kwargs)

def skeleton_downstream(limb_obj,
                            branch_idx,
                            nodes_to_exclude=None,
                            **kwargs):
    return cnu.skeleton_downstream(limb_obj,
                            branch_idx,
                            nodes_to_exclude=nodes_to_exclude,
                                         **kwargs)


def skeletal_length_upstream(limb_obj,
                            branch_idx,
                            nodes_to_exclude=None,
                            **kwargs):
    return cnu.skeletal_length_upstream(limb_obj,
                            branch_idx,
                            nodes_to_exclude=nodes_to_exclude,
                                       **kwargs)

def skeletal_length_downstream(limb_obj,
                            branch_idx,
                            nodes_to_exclude=None,
                            **kwargs):
    return cnu.skeletal_length_downstream(limb_obj,
                            branch_idx,
                            nodes_to_exclude=nodes_to_exclude,
                                         **kwargs)

def skeletal_length_downstream_total(limb_obj,
                            branch_idx,
                            nodes_to_exclude=None,
                            include_branch_in_dist = True,
                            **kwargs):
    return cnu.skeletal_length_downstream(limb_obj,
                            branch_idx,
                            nodes_to_exclude=nodes_to_exclude,
                            include_branch_in_dist=include_branch_in_dist,
                            only_non_branching=False,
                                         **kwargs)

def skeletal_length_upstream_total(limb_obj,
                            branch_idx,
                            nodes_to_exclude=None,
                            include_branch_in_dist = True,
                            **kwargs):
    return cnu.skeletal_length_upstream(limb_obj,
                            branch_idx,
                            nodes_to_exclude=nodes_to_exclude,
                            include_branch_in_dist=include_branch_in_dist,
                            only_non_branching=False,
                                         **kwargs)

def width_upstream(limb_obj,
                            branch_idx,
                            nodes_to_exclude=None,
                            **kwargs):
    return cnu.width_upstream(limb_obj,
                            branch_idx,
                            nodes_to_exclude=nodes_to_exclude,
                                       **kwargs)

def width_downstream(limb_obj,
                            branch_idx,
                            nodes_to_exclude=None,
                            **kwargs):
    return cnu.width_downstream(limb_obj,
                            branch_idx,
                            nodes_to_exclude=nodes_to_exclude,
                                         **kwargs)

def synapses_upstream(limb_obj,
                            branch_idx,
                            nodes_to_exclude=None,
                           synapse_type="synapses",
                            **kwargs):
    return cnu.synapses_upstream(limb_obj,
                            branch_idx,
                            nodes_to_exclude=nodes_to_exclude,
                                synapse_type=synapse_type,
                                       **kwargs)

def synapses_downstream(limb_obj,
                            branch_idx,
                            nodes_to_exclude=None,
                        synapse_type="synapses",
                            **kwargs):
    return cnu.synapses_downstream(limb_obj,
                            branch_idx,
                            synapse_type=synapse_type,
                            nodes_to_exclude=nodes_to_exclude,
                                         **kwargs)
def n_synapses_upstream(limb_obj,
                       branch_idx,**kwargs):
    return len(nst.synapses_upstream(limb_obj,
                            branch_idx,
                                       **kwargs))

def n_synapses_downstream(limb_obj,
                       branch_idx,**kwargs):
    return len(nst.synapses_upstream(limb_obj,
                            branch_idx,
                                       **kwargs))
def synapses_downstream_total(limb_obj,
                            branch_idx,
                             distance=np.inf,
                            **kwargs):
    return nst.synapses_downstream(limb_obj,
                            branch_idx,
                                   distance=distance,
                                   only_non_branching=False,
                                       **kwargs)
def synapses_upstream_total(limb_obj,
                            branch_idx,
                             distance=np.inf,
                            **kwargs):
    return nst.synapses_upstream(limb_obj,
                            branch_idx,
                                   distance=distance,
                                   only_non_branching=False,
                                       **kwargs)

def n_synapses_downstream_total(limb_obj,
                       branch_idx,**kwargs):
    return len(nst.synapses_downstream_total(limb_obj,
                            branch_idx,
                                       **kwargs))


def synapses_downstream_within_dist(limb_obj,
                            branch_idx,
                            synapse_type="synapses",
                            distance = 5000,
                                    
                            plot_synapses = False,
                            verbose = False,
                            **kwargs,):
    """
    purpose: to find thenumber of downstream
    postsyns within a certain downstream distance

    """

    post_syns = cnu.synapses_downstream(limb_obj,
                           branch_idx = branch_idx,
                           distance = distance,
                            only_non_branching=False,
                            include_branch_in_dist=False,
                            include_branch_idx=False,
                            synapse_type=synapse_type,
                                        **kwargs
                           )
    if verbose:
        print(f"# of postsyns = {len(post_syns)}")

    return post_syns

def synapses_post_downstream_within_dist(limb_obj,
                            branch_idx,
                            distance = 5000,
                            plot_synapses = False,
                            verbose = False,
                            **kwargs,):
    return synapses_downstream_within_dist(limb_obj,
                            branch_idx,
                            synapse_type="synapses_post",
                            distance = distance,
                            plot_synapses = plot_synapses,
                            verbose = verbose,
                            **kwargs,)

def synapses_pre_downstream_within_dist(limb_obj,
                            branch_idx,
                            distance = 5000,
                            plot_synapses = False,
                            verbose = False,
                            **kwargs,):
    return synapses_downstream_within_dist(limb_obj,
                            branch_idx,
                            synapse_type="synapses_pre",
                            distance = distance,
                            plot_synapses = plot_synapses,
                            verbose = verbose,
                            **kwargs,)






def fork_divergence(limb_obj,
    branch_idx,
    downstream_idxs = None,

    #arguments for skeletons
    skeleton_distance = 10000,

    error_not_2_downstream = True,

    #arguments for enforcing skipping rule for fork check
    total_downstream_skeleton_length_threshold = 0,#3000#4000
    individual_branch_length_threshold = 2000,#3000
    skip_value = np.inf,

    # for visualizing the current plot
    plot_fork_skeleton = False,

    #arguments for the fork divergence measurement
    comparison_distance = 400,
    skeletal_segment_size = 40,
    plot_restrictions = False,
    combining_function = np.mean,
                    
                nodes_to_exclude = None,
                   verbose = False):

    """
    Purpose: To run the fork divergence the children
    of an upstream node

    Pseudocode: 
    1) Get downstream nodes
    2) Apply skeletal length restrictions if any
    3) compute the fork divergence from the skeletons

    """


    upstream_node = branch_idx

    if downstream_idxs is None:
        downstream_nodes = cnu.downstream_nodes(limb_obj,branch_idx)
    else:
        downstream_nodes = downstream_idxs

    if verbose:
        print(f"downstream_nodes = {downstream_nodes}")

    upstream_sk = limb_obj[upstream_node].skeleton
    downstream_sk = [cnu.skeleton_downstream(limb_obj,d,distance=skeleton_distance)
                     for d in downstream_nodes]


    return_value = None


    if error_not_2_downstream and len(downstream_nodes) != 2:
        raise Exception(f"Not exactly 2 downstream nodes: {downstream_nodes}")

    if (total_downstream_skeleton_length_threshold is not None and
        individual_branch_length_threshold is not None):
        d_len = np.array([sk.calculate_skeleton_distance(limb_obj[k].skeleton) for 
                         k in downstream_nodes])
        d_skeletal_len = np.array([nru.skeletal_length_over_downstream_branches(limb_obj,
                                                d,
                                                verbose=False) for d in downstream_nodes])
        if verbose:
            print(f"skeletal length = {d_len}")
            print(f"downstream skeletal length = {d_skeletal_len}")

        below_threshold = np.where((d_skeletal_len<total_downstream_skeleton_length_threshold) | 
                                   (d_len < individual_branch_length_threshold))[0]
        if len(below_threshold) > 0:
            if verbose:
                print(f"Skipping this intersection because some of downstream skeletal lengths too short (min {total_downstream_skeleton_length_threshold}):")
                print(f" or the individual branch length was too short (min {individual_branch_length_threshold})")
                for j,(d,d_len) in enumerate(zip(downstream_nodes,d_skeletal_len)):
                    if j in below_threshold:
                        print(f"Brnach {d}: length {d_len}")

            return_value = skip_value



    if return_value is None:
        return_value = nst.fork_divergence_from_skeletons(upstream_skeleton = upstream_sk,
                                downstream_skeletons = downstream_sk,
                                comparison_distance = comparison_distance,
                                skeletal_segment_size = skeletal_segment_size,
                                plot_restrictions = plot_restrictions,
                                combining_function = combining_function,
                                                             verbose=verbose)

    if verbose:
        print(f"return_value = {return_value}")
        
    return return_value

def compute_node_attributes(G,
                            limb_obj,
                            node_functions,
                           verbose = False):
    """
    Purpose: To Compute node attributes given:
    - function
    - arguments for function
    - nodes to compute for (so can explicitely do for upstream and downstream)

    Each of this will be stored in a list of dictionaries
    """


    G_nodes = list(G.nodes())
    att_dict = dict([(k,dict()) for k in G_nodes])
    
    for f_info  in node_functions:
        f_name = f_info["name"]
        func = f_info["function"]
        nodes_to_compute = f_info.get("nodes_to_compute",G_nodes)
        if not nu.is_array_like(nodes_to_compute):
            nodes_to_compute = [nodes_to_compute]
        
        args = f_info.get("arguments",dict())
        default = f_info.get("default",None)
        
        if verbose:
            print(f"Working on {f_name} with args = {args}")

        nodes_not_compute = np.setdiff1d(G_nodes,nodes_to_compute)

        for n in nodes_to_compute:
            att_dict[n][f_name] = func(limb_obj=limb_obj,
                                       branch_idx=n,
                                       **args)

        if default is not None:
            for n in nodes_not_compute:
                att_dict[n][f_name] = default

    xu.set_node_attributes_dict(G,att_dict)
    return G


def node_functions_default(upstream_branch,
                   downstream_branches,
                  ):
    """
    To create the defautl node attributes
    wanting to compute
    """
    return [
    dict(name="skeletal_length_downstream",
     function = nst.skeletal_length_downstream,
     nodes_to_compute = downstream_branches),

    dict(name="skeletal_length_downstream_total",
     function = nst.skeletal_length_downstream_total,
     nodes_to_compute = downstream_branches,),
        
    dict(name="skeletal_length_upstream_total",
     function = nst.skeletal_length_upstream_total,
     nodes_to_compute = upstream_branch,),
        
    dict(name="skeletal_length_upstream",
     function = nst.skeletal_length_upstream,
     nodes_to_compute = upstream_branch,),
    
    dict(name="width_downstream",
     function = nst.width_downstream,
     nodes_to_compute = downstream_branches,),
    
    dict(name="width_upstream",
     function = nst.width_upstream,
     nodes_to_compute = upstream_branch,), 
    
    dict(name="n_synapses_upstream",
        function = nst.n_synapses_upstream,
        nodes_to_compute = upstream_branch),
    
    dict(name="n_synapses_downstream",
        function = nst.n_synapses_downstream,
        nodes_to_compute = downstream_branches),
    
    dict(name="n_synapses_downstream_total",
        function = nst.n_synapses_downstream_total,
        nodes_to_compute = downstream_branches),
    
    ]

def compute_node_attributes_upstream_downstream(G,
                                                limb_obj,
                                               upstream_branch,
                                               downstream_branches,
                                                node_functions=None,
                                               verbose = False):
    """
    Purpose: To attach node properties to a graph
    that references branches on a limb
    
    """
    if node_functions is None:
        node_functions = node_functions_default(upstream_branch=upstream_branch,
                            downstream_branches=downstream_branches)
        
    node_type_dict = gu.merge_dicts([{upstream_branch:dict(node_type="upstream")},
                        {k:dict(node_type="downstream") for k in downstream_branches}])
    
    xu.set_node_attributes_dict(G,node_type_dict)

    nst.compute_node_attributes(G,
                           limb_obj,
                            node_functions=node_functions,
                           verbose = verbose)
    
    if verbose:
        print(xu.node_df(G))
        
    return G


def synapse_closer_to_downstream_endpoint_than_upstream(branch_obj):
    """
    Purpose: Will indicate if there is a synapse that is closer to the downstream endpoint than upstream endpoint
    
    """
    return_value = False
    for syn in branch_obj.synapses:
        if syn.downstream_dist < syn.upstream_dist:
            return_value = True
    
    return return_value


    

def downstream_upstream_diff_of_most_downstream_syn(branch_obj,
                                                   default_value = 0):
    """
    Purpose: Determine the difference between 
    the closest downstream dist and 
    the farthest upstream dist

    Pseudocode: 
    1) Get the synapse with min of downstream dist
    2) Get the difference between downstream dist and upstream dist
    3) Return the difference
    """
    return_value = default_value
    syns = branch_obj.synapses
    if len(syns) > 0:
        d_dists = np.array([k.downstream_dist for k in syns])
        syn_idx = np.argmin(d_dists)
        d_min = d_dists[syn_idx]
        u_max = syns[syn_idx].upstream_dist
        return_value = d_min - u_max
    return return_value


# ------------ New Filter: 7/8 ----------
def fork_min_skeletal_distance_from_skeletons(downstream_skeletons,
                                             comparison_distance = 3000,
                                             offset = 700,
                                              skeletal_segment_size = 40,
                                             verbose = False,
                                             plot_skeleton_restriction=False,
                                              plot_min_pair = False
                                             ):
    """
    Purpose: To determine the min distance from two diverging
    skeletons with an offset
    """
    joining_endpoint_1 = sk.shared_coordiantes(downstream_skeletons,
                                 return_one=True)

    d_skeletons_resized = [sk.resize_skeleton_branch(k,segment_width=skeletal_segment_size)
                              for k in downstream_skeletons]

    new_sks = [sk.restrict_skeleton_from_start_plus_offset(k,
                                               offset = offset,
                                               comparison_distance = comparison_distance,
                                                start_coordinate = joining_endpoint_1,
                                               ) for k in d_skeletons_resized]


    distances_between_skeletons = sk.closest_distances_from_skeleton_vertices_to_base_skeleton(new_sks[0],
                                                              new_sks[1],
                                                              verbose= verbose,
                                                              plot_min_pair=plot_min_pair)
    min_dist = np.min(distances_between_skeletons)
    if verbose:
        print(f"min_dist= {min_dist}")
        print(f"distances_between_skeletons = {distances_between_skeletons}")
        
    return min_dist




def distance_from_soma(limb_obj,
                      branch_idx,
                      include_node_skeleton_dist = False,
                      verbose = False,
                      **kwargs):
    """
    Purpose: To find the distance away from the soma
    for a given set of branches
    
    Ex: 
    nst.distance_from_soma(limb_obj,190)
    """
    
    singular_flag = False
    if not nu.is_array_like(branch_idx):
        branches = [branch_idx]
        singular_flag = True
    else:
        branches = branch_idx
        
    branch_paths = [nru.branch_path_to_start_node(
                    limb_obj,k,include_branch_idx=include_node_skeleton_dist) for k in branches]
    
    return_value = [nru.sum_feature_over_branches(limb_obj,k,
                                                 feature_name = "skeletal_length") for k in branch_paths]
    
    if verbose:
        print(f"branch_paths = {branch_paths}")
        print(f"Path lengths = {return_value}")
        
        
    if singular_flag:
        return_value = return_value[0]
        
    return return_value

def distance_from_soma_euclidean(limb_obj,
                                branch_idx,):
    """
    Will return the euclidean distance of the upstream endpoint
    to the starting coordinate of the limb
    
    Ex: 
    branch_idx = 0
    limb_obj = neuron_obj_proof[0]
    nst.distance_from_soma_euclidean(limb_obj,branch_idx)
    """
    upstream_endpoint = nru.upstream_endpoint(limb_obj,branch_idx,return_endpoint_index=False)
    return np.linalg.norm(limb_obj.current_starting_coordinate - upstream_endpoint)



def width_basic(branch_obj):
    return branch_obj.width
                            
    

    

# ------- 7/26: To help identify axons  -----------
distance_away_from_endpoint= 6_000



# ------------- 7/28: for apical -----------------



def skeleton_perc_dist_match_ref_vector(limb_obj,
    branch_idx,
    max_angle = 30,
    min_angle = None,
    reference_vector = np.array([0,-1,0]),
    skeleton_resize_distance = 8000,
    plot_branch = False,
    verbose = False,
    **kwargs):
    """
    Purpose: 
    To find the percentage of skeleton that is
    within a certain angle of a comparison vector

    Pseudocode: 
    1) Resize the branch skeleton and order the skeleton
    2) For each segments of the skeleton:
    - extract the vector from the skeleton
    - find the angle betweeen the reference bector and the segment vector
    - if angle is below the threshold then count it as a match
    3) return the percentage of the matches

    """


    branch_obj = limb_obj[branch_idx]

    #branch_obj
    #1) Resize the branch skeleton and order the skeleton

    perc_match,dist_match = sk.percentage_skeleton_match_to_ref_vector(branch_obj.skeleton,
                                              reference_vector=reference_vector,
                                               max_angle=max_angle,
                                                min_angle=min_angle,
                                              start_endpoint_coordinate=nru.upstream_endpoint(limb_obj,branch_idx),
                                            segment_width=skeleton_resize_distance,
                                             plot_skeleton=False,
                                              verbose=verbose,
                                              return_match_length = True)
    if verbose:
        print(f"perc_match = {perc_match}")
        print(f"dist_match = {dist_match}")
        
    return perc_match,dist_match

def skeleton_dist_match_ref_vector(limb_obj,
    branch_idx,
    max_angle = 30,
    min_angle = None,
    reference_vector = np.array([0,-1,0]),
    skeleton_resize_distance = 8000,
    plot_branch = False,
    verbose = False,
    **kwargs):
    """
    Purpose: To return the amount of skeletal distance that matches
    a comparison vector
    
    Ex: 
    nst.skeleton_dist_match_ref_vector(neuron_obj[0],
                                       11,
                                       verbose = True)
    
    """
    perc_match,dist_match = nst.skeleton_perc_dist_match_ref_vector(limb_obj,
    branch_idx,
    max_angle = max_angle,
    min_angle = min_angle,
    reference_vector = reference_vector,
    skeleton_resize_distance = skeleton_resize_distance,
    plot_branch = plot_branch,
    verbose = verbose,)
    
    return dist_match

def skeleton_perc_match_ref_vector(limb_obj,
    branch_idx,
    max_angle = 30,
    min_angle = None,
    reference_vector = np.array([0,-1,0]),
    skeleton_resize_distance = 8000,
    plot_branch = False,
    verbose = False,
    **kwargs):
    """
    Purpose: To return the percentage of skeletal distance that matches
    a comparison vector
    
    """
    perc_match,dist_match = nst.skeleton_perc_dist_match_ref_vector(limb_obj,
    branch_idx,
    max_angle = max_angle,
    min_angle = min_angle,
    reference_vector = reference_vector,
    skeleton_resize_distance = skeleton_resize_distance,
    plot_branch = plot_branch,
    verbose = verbose,)
    
    return perc_match

def get_stat(obj,stat,**kwargs):
    """
    Purpose: Will either run the function 0n the  object
    of get the property of the function if it is a string
    
    Ex:
    nst.get_stat(limb_obj[0],"skeletal_length")
    
    """
    if type(stat) == str:
        return getattr(obj,stat)
    else:
        return stat(obj,**kwargs)
    
    
# --------- Statisitcs for canddidates --------------
def skeletal_length_over_candidate(neuron_obj,
                                  candidate,
                                  **kwargs):
    sk_len = nru.skeletal_length_over_limb_branch(neuron_obj,nru.nru.limb_branch_from_candidate(candidate))
    return sk_len




# ------- help with searching for labels -------------- #
def upstream_node_is_apical_shaft(limb_obj,branch_idx,verbose,**kwargs):
    return nru.upstream_node_has_label(limb_obj,branch_idx,label="apical_shaft",verbose = verbose)

def is_label_in_downstream_branches(limb_obj,
                                    branch_idx,
                                    label,
                                    all_downstream_nodes = False,
                                     verbose = False,
                                   ):
    """
    Purpose: To test if a label is in the downstream 
    nodes 

    1) Get all the downstream labels
    2) return the test if a certain label is in downstream labels
    
    Ex: 
    nst.is_label_in_downstream_branches(neuron_obj[1],5,"apical_shaft",verbose = True)
    """
    if not nu.is_array_like(label):
        label = [label]
        
    downstream_labels = nru.downstream_labels(limb_obj,branch_idx,
                          all_downstream_nodes = all_downstream_nodes,
                         verbose = verbose)
    
    common_labels = np.intersect1d(downstream_labels,label)
    if verbose:
        print(f"common_labels= {common_labels}")
        
    if len(common_labels) > 0:
        return True
    else:
        return False
    
def is_apical_shaft_in_downstream_branches(limb_obj,
                                    branch_idx,
                                    all_downstream_nodes = False,
                                     verbose = False,
                                   **kwargs):
    """
    Ex: 
    nst.is_apical_shaft_in_downstream_branches(neuron_obj[1],4,verbose = True)
    """
    return is_label_in_downstream_branches(limb_obj,
                                    branch_idx,
                                    label="apical_shaft",
                                    all_downstream_nodes = all_downstream_nodes,
                                     verbose = verbose,
                                   )

def is_axon_in_downstream_branches(limb_obj,
                                    branch_idx,
                                    all_downstream_nodes = False,
                                     verbose = False,
                                   **kwargs):
    """
    Ex: 
    nst.is_apical_shaft_in_downstream_branches(neuron_obj[1],4,verbose = True)
    """
    return is_label_in_downstream_branches(limb_obj,
                                    branch_idx,
                                    label="axon",
                                    all_downstream_nodes = all_downstream_nodes,
                                     verbose = verbose,
                                   )


# ---------- Functions over upstream and downstream branches ----------- #
def width_weighted_over_branches(limb_obj,
                                branches,
                                 width_func = None,
                                verbose = False):
    """
    Purpose: Find weighted width over branches

    Ex: 
    nst.width_weighted_over_branches(n_obj_2[6],
                                branches = [24,2])
    """
    if width_func is None:
        width_func = nst.width_new
        
    weight_width = cnu.weighted_feature_over_branches(limb_obj = limb_obj,
                                branches =branches,
                               direction=None,
                               verbose = verbose,
                               feature_function=width_func
    )
    
    return weight_width


def skeleton_dist_match_ref_vector_sum_over_branches(limb_obj,
                                                    branches,
                                                    max_angle,
                                                    min_angle=None,
                                                     direction = None,
                                                     verbose = False,
                                                    **kwargs):
    """
    Purpose: Find the amount of upstream skeletal distance
    that matches a certain angle

    """
    sk_dist = cnu.sum_feature_over_branches(limb_obj = limb_obj,
                                branches =branches,
                               direction=direction,
                               verbose = verbose,
                               feature_function=nst.skeleton_dist_match_ref_vector,
                                  use_limb_obj_and_branch_idx = True,
                                  max_angle=max_angle,
                                  min_angle = min_angle,)
    return sk_dist


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
    
    Ex: 
    from neurd import neuron_statistics as nst
    stats_dict_over_limb_branch(
        neuron_obj = neuron_obj_proof,
        limb_branch_dict = limb_branch_dict)
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
    verbose = False,
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

        if verbose:
            print(f"Total Number of Branches = {(n_branches)}")
            print(f"n_short_branches = {n_short_branches}, n_medium_branches = {n_medium_branches}, n_long_branches = {n_long_branches}")

        # calculating the skeletal lengths
        if in_um:
            divisor = 1000
        else:
            divisor = 1

        axon_length = np.sum(sk_branches_dist)/divisor
        axon_branch_length_median = np.median(sk_branches_dist)/divisor
        axon_branch_length_mean = np.mean(sk_branches_dist)/divisor

        if verbose:
            print(f"axon_length = {axon_length}, axon_branch_length_median = {axon_branch_length_median}, axon_branch_length_mean = {axon_branch_length_mean}")

        bbox_volume = sk.bbox_volume(skeleton)/volume_divisor
        bbox_corners = sk.bounding_box_corners(skeleton)
        bbox_corners_soma_relative = bbox_corners - soma_center
        if skeleton_aligned is not None:
            if verbose:
                print(f"Using skeleton aligned")
                print(f"Previous bbox_corners_soma_relative = {bbox_corners_soma_relative}")
            bbox_corners_aligned = sk.bounding_box_corners(skeleton_aligned)
            bbox_corners_soma_relative = bbox_corners_aligned - soma_center
            if verbose:
                print(f"NEW ALIGNED bbox_corners_soma_relative = {bbox_corners_soma_relative}")
            

        if verbose:
            print(f"bbox_volume = {bbox_volume}")
            print(f"bbox_corners = {bbox_corners}")
            print(f"bbox_corners_soma_relative = {bbox_corners_soma_relative}")


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
def soma_distance_branch_set(neuron_obj,
                                     attr_name,
                                     attr_func,
                                    ):
    """
    Purpose: Will set the skeletal distance to soma
    on each branch
    
    Pseudocode: 
    1) iterate through all of the limbs and branches
    2) Find the distnace from soma and store
    """
    for limb_idx in neuron_obj.get_limb_node_names():
        limb_obj = neuron_obj[limb_idx]
        for branch_idx in limb_obj.get_branch_names():
            s_dist = attr_func(limb_obj,branch_idx)
            setattr(limb_obj[branch_idx],attr_name,s_dist)
            
            
    
            
def centroid_stats_from_neuron_obj(neuron_obj,
                                  voxel_adjustment_vector=None,
                                  include_volume=True):
    if voxel_adjustment_vector is None:
        voxel_adjustment_vector=voxel_to_nm_scaling
        
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
                                  verbose= False,
                                   limb_branch_dict = None,
                                   neuron_obj_aligned=None,
                                  ):
    """
    Compute all the statistics for a neurons skeleton (should have only one soma)
    """
    if voxel_adjustment_vector is None:
        voxel_adjustment_vector = voxel_to_nm_scaling
    
    sk_dict = stats_dict_over_limb_branch(
        neuron_obj,
        limb_branch_dict=limb_branch_dict,
        stats_to_compute=["skeletal_length","n_branches"])

    sk_dict_2 = features_from_neuron_skeleton_and_soma_center(
        neuron_obj,
        verbose = verbose,
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

def skeleton_stats_compartment(
    neuron_obj,
    compartment,
    include_compartmnet_prefix=True,
    include_centroids = False,
    **kwargs):
    
    limb_branch_dict = getattr(neuron_obj,f"{compartment}_limb_branch_dict")
    return_dict = nst.skeleton_stats_from_neuron_obj(
        neuron_obj,
        limb_branch_dict = limb_branch_dict,
        include_centroids=include_centroids,
        **kwargs
        )
    if include_compartmnet_prefix:
        return_dict = {f"{compartment}_{k}":v for k,v in return_dict.items()}
        
    return return_dict
        



# --------------s / 9 ---------------
def farthest_distance_from_skeleton_to_mesh(
    obj,
    verbose = False,
    plot = False,
    **kwargs
    ):
    """
    Purposee: find the coordinate
    of the skeleton that has the 
    longest closest distance to the mesh
    
    Ex: 
    farthest_distance_from_skeleton_to_mesh(
    branch_obj,
    verbose = True,
    plot = True
    )
    """
    return tu.farthest_coordinate_to_faces(
        obj.mesh,
        obj.skeleton,
        return_distance = True,
        verbose = verbose,
        plot = plot,
    )

def limb_branch_from_stats_df(
    df
    ):
    """
    Purpose: To convert a dataframe to a limb branch dict
    """
    limb_branch_pairings = df[["limb","node"]].to_numpy()

    #gets a dictionary where key is the limb and value is a list of all the branches that were in the filtered dataframe
    limb_to_branch = dict([(k,np.sort(limb_branch_pairings[:,1][np.where(limb_branch_pairings[:,0]==k)[0]]).astype("int")) 
                           for k in np.unique(limb_branch_pairings[:,0])])
    return limb_to_branch

def coordinates_function_list(
    coordinates = None
    ):
    if coordinates is None:
        coordinates = ("mesh_center",
               "endpoint_downstream",
              "endpoint_upstream")
    return  np.concatenate([
        [f"{k}_{x}" for x in ["x","y","z"]] for k in coordinates
    ])


def stats_df(
    neuron_obj,
    functions_list=None,
    query = None,
    limb_branch_dict_restriction = None,
    function_kwargs=None,
    include_coordinates = False,
    coordinates = None,
    check_nans=False,
    ):
    """
    Purpose: To return the stats on neuron branches 
    that is used by the neuron searching to filter down
    
    Ex: 
    from neurd import neuron_statistics as nst

    limb_obj = neuron_obj[6]

    s_df = nst.stats_df(
        neuron_obj,
        functions_list = [ns.width_new,
        ns.skeletal_length,
        ns.n_synapses_post_downstream],
        limb_branch_dict_restriction=dict(L6=limb_obj.get_branch_names())
            )
    s_df
    """
    
    if functions_list is None:
        functions_list= []
        
    if query is not None:
        functions_list += ns.functions_list_from_query(query)
    
    if include_coordinates:
        neuron_obj = bu.set_branches_endpoints_upstream_downstream_idx(neuron_obj)
        functions_list += list(nst.coordinates_function_list(coordinates))
        
    return ns.query_neuron(neuron_obj,
                functions_list=functions_list,
                function_kwargs = function_kwargs,       
                return_dataframe_before_filtering=True,
                           limb_branch_dict_restriction=limb_branch_dict_restriction,
               query="",
                check_nans=check_nans)
    
    
        


def neuron_stats(
    neuron_obj,
    stats_to_ignore=None,
    include_skeletal_stats = False,
    include_centroids= False,
    voxel_adjustment_vector = None,
    cell_type_mode = False,
    **kwargs):
    
    """
    Purpose: Will compute a wide range of statistics 
    on a neurons object
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


    stats_dict = dict(
                    n_vertices = neuron_obj.n_vertices,
                    n_faces = neuron_obj.n_faces,

                    # axon_length/axon_area dropped: axon labeling removed in Phase 5
                    # (nru.axon_length/axon_area no longer exist).

                    max_soma_volume = neuron_obj.max_soma_volume,
                    max_soma_n_faces = neuron_obj.max_soma_n_faces,
                    max_soma_area = neuron_obj.max_soma_area,


                    n_not_processed_soma_containing_meshes = len(neuron_obj.not_processed_soma_containing_meshes),
                    n_error_limbs=neuron_obj.n_error_limbs,
                    n_same_soma_multi_touching_limbs=len(neuron_obj.same_soma_multi_touching_limbs),
                    n_multi_soma_touching_limbs = len(neuron_obj.multi_soma_touching_limbs),
                    n_somas=neuron_obj.n_somas,
                    n_limbs=neuron_obj.n_limbs,
                    n_branches=neuron_obj.n_branches,
                    max_limb_n_branches=neuron_obj.max_limb_n_branches,

                    skeletal_length=neuron_obj.skeletal_length,
                    max_limb_skeletal_length=neuron_obj.max_limb_skeletal_length,
                    median_branch_length=neuron_obj.median_branch_length,

                    width_median=neuron_obj.width_median, #median width from mesh center without spines removed
                    width_no_spine_median=neuron_obj.width_no_spine_median, #median width from mesh center with spines removed
                    width_90_perc=neuron_obj.width_90_perc, # 90th percentile for width without spines removed
                    width_no_spine_90_perc=neuron_obj.width_no_spine_90_perc,  # 90th percentile for width with spines removed

                    n_spines=neuron_obj.n_spines,
                    # n_boutons dropped: bouton detection (axon-based) removed in Phase 5.

                    spine_density=neuron_obj.spine_density, # n_spines/ skeletal_length
                    spines_per_branch=neuron_obj.spines_per_branch,

                    skeletal_length_eligible=neuron_obj.skeletal_length_eligible, # the skeletal length for all branches searched for spines
                    n_spine_eligible_branches=neuron_obj.n_spine_eligible_branches,
                    spine_density_eligible = neuron_obj.spine_density_eligible,
                    spines_per_branch_eligible = neuron_obj.spines_per_branch_eligible,

                    total_spine_volume=neuron_obj.total_spine_volume, # the sum of all spine volume
                    spine_volume_median = neuron_obj.spine_volume_median,
                    spine_volume_density=neuron_obj.spine_volume_density, #total_spine_volume/skeletal_length
                    spine_volume_density_eligible=neuron_obj.spine_volume_density_eligible, #total_spine_volume/skeletal_length_eligible
                    spine_volume_per_branch_eligible=neuron_obj.spine_volume_per_branch_eligible, #total_spine_volume/n_spine_eligible_branche



    )

    if stats_to_ignore is not None:
        for s in stats_to_ignore:
            del stats_dict[s]

    if include_skeletal_stats:
        sk_dict = features_from_neuron_skeleton_and_soma_center(neuron_obj,
                                              verbose = False,
                                              features_to_exclude=("length","n_branches"),
                                             )
        stats_dict.update(sk_dict)


    if include_centroids:
        cent_stats = centroid_stats_from_neuron_obj(neuron_obj,
                                                       voxel_adjustment_vector=voxel_adjustment_vector)
        stats_dict.update(cent_stats)


    return stats_dict


def euclidean_distance_from_soma_limb_branch(
    neuron_obj,
    less_than = False,
    distance_threshold = 10_000,
    endpoint_type = "downstream",
    verbose = False,
    plot = False,
    ):
    """
    Purpose: Find limb branch dict within or
    farther than a certain euclidean distance
    from all the soma pieces

    Pseudocode: 
    1) get the upstream endpoints of all
    """
   
    

    bu.set_branches_endpoints_upstream_downstream_idx(neuron_obj)
    soma_kd = tu.mesh_to_kdtree(neuron_obj["S0"].mesh)



    lb_dict = dict()
    for limb_idx in neuron_obj.get_limb_names():
        limb_obj = neuron_obj[limb_idx]
        for branch_idx in limb_obj.get_branch_names():
            branch_obj = limb_obj[branch_idx]
            dist,__ = soma_kd.query(getattr(branch_obj,f"endpoint_{endpoint_type}").reshape(-1,3))
            if less_than:
                add_flag = dist[0] < distance_threshold
            else:
                add_flag = dist[0] >= distance_threshold

            if add_flag:
                if limb_idx not in lb_dict:
                    lb_dict[limb_idx] = []
                lb_dict[limb_idx].append(branch_idx)

                if verbose:
                    print(f"Adding {limb_idx}, {branch_idx} because dist {dist[0]}")


    lb_dict = {k:np.array(v) for k,v in lb_dict.items()}
    
    
    return lb_dict


    


# -- 5/9 Addition for computing more statistics


def limb_node_query_dict(neuron_obj):
    search_features = [
        ns.n_synapses_pre,
        ns.synapse_pre_perc,
        ns.axon_width,
        ns.n_spines,
        ns.n_synapses_post_spine,
        ns.skeletal_length,
        ns.closest_mesh_skeleton_dist,
        ns.area,
        
        ns.synapse_pre_perc_downstream,
        ns.n_synapses_downstream,
        ns.n_synapses_post_downstream,
        
        ns.width_new,ns.skeletal_length,
        ns.n_synapses_post_downstream,
        ns.closest_mesh_skeleton_dist,
        ns.skeletal_length_downstream,
        ns.area,
        
        ns.is_axon_in_downstream_branches,
        
        # -- myelination --
        ns.synapse_density_post,
        ns.axon_width,
        #ns.is_axon_like,
        ns.is_axon,
        ns.distance_from_soma,
        ns.skeletal_length_downstream,
    ]
    
    feature_strs = [
        "ray_trace_perc","skeletal_length","n_downstream_nodes",
        "n_synapses_post","n_synapses_pre","n_faces_branch",
        "synapse_closer_to_downstream_endpoint_than_upstream",
        "downstream_upstream_diff_of_most_downstream_syn",
        "axon_width","skeletal_length",
        "n_downstream_nodes",
        "ray_trace_perc",
        "parent_width",
        "total_upstream_skeletal_length",
    ]
    
    ns_features_from_str = [getattr(ns,k) for k in feature_strs]
    
    search_features = list(set(search_features + ns_features_from_str))
    
    search_df = ns.generate_neuron_dataframe(
        neuron_obj,
        functions_list=search_features
    )
    
    def df_to_dict_iter(df):
        result = {}
        for _, row in df.iterrows():
            key = f"{row['limb']}_{row['node']}"
            # build a sub-dict of all other columns
            value = {col: row[col] for col in df.columns if col not in ('limb', 'node')}
            result[key] = value
        return result
    
    # usage
    limb_node_dict = df_to_dict_iter(search_df)
    return limb_node_dict

def branch_computed_dict(
    branch,
    default_value = None,
    catch_errors = False,):

    branch_dict = {}
        
    branch_attributes = [
        "width_downstream_extra_offset",
        "width_upstream_extra_offset",
        "max_skeleton_endpoint_dist",
        "skeleton_smooth_vector_downstream_extra_offset",
        "skeleton_smooth_vector_upstream_extra_offset",
        "skeleton_smooth_vector_downstream",
        "skeleton_smooth_vector_upstream",
    ]
    
    from . import branch_utils as bu

    for b_att in branch_attributes:
        try:
            branch_dict[b_att] = getattr(branch,b_att)
        except:
            if not catch_errors:
                raise e
            branch_dict[b_att] = default_value
    
    branch_functions = [
        bu.width_min,
        bu.width_max,
    ]

    for b_func in branch_functions:
        try:
            branch_dict[b_func.__name__] = b_func(branch)
        except Exception as e:
            if not catch_errors:
                raise e
            branch_dict[b_func.__name__] = default_value
        

    branch_dict_functions = [
        bu.internal_bend_dict_func,
    ]

    for b_dict_func in branch_dict_functions:
        try:
            branch_dict.update(b_dict_func(branch))
        except:
            if not catch_errors:
                raise e
            pass

    return branch_dict

def branch_limb_computed_dict(
    limb,
    branch_idx,
    default_value = None,
    catch_errors = False,):

    branch_dict = {}
    
    from . import limb_utils as lu
        
    branch_functions = [
        lu.downstream_endnode_skeletal_distance_from_soma,
        lu.sibling_angle_smooth_max,
        lu.sibling_angle_smooth_min,
        lu.sibling_angle_smooth_extra_offset_max,
        lu.sibling_angle_smooth_extra_offset_min,
        lu.n_children_with_skip_distance,
        lu.n_children,
        lu.parent_skeletal_angle_smooth,
        lu.parent_skeletal_angle_smooth_extra_offset,
    ]


    for b_func in branch_functions:
        try:
            branch_dict[b_func.__name__] = b_func(limb,branch_idx)
        except Exception as e:
            if not catch_errors:
                raise e
            branch_dict[b_func.__name__] = default_value
    return branch_dict




def limb_branches_aggr_features(
    limb,
    branches,
    features,
    aggr_type = "sum",
    default_value = None,
    add_aggr_suffix = True,
):
        
        
    def value_and_name(branch,feat):
        value = None
        if type(feat) == str:
            value = getattr(branch,feat,default_value)
            if value == default_value:
                func = getattr(ns,feat,None)
                if func is None:
                    func = getattr(nst,feat,None)
                    
                if func is not None:
                    value = func(branch)
            name = feat
        else:
            value = feat(branch)
            name = feat.__name__
            
        return name,value
    
        
    records = []
    for branch_idx in branches:
        local_dict= dict()
        branch = limb[branch_idx]
        for feat in features:
            name,value = value_and_name(branch,feat)
            local_dict[name] = value
        
        records.append(local_dict)
                
    df = pd.DataFrame.from_records(records)
    
    func = getattr(df,aggr_type)
    branch_attr_dict = func(skipna=True,numeric_only=True).to_dict()
    
    if add_aggr_suffix:
        branch_attr_dict = {f"{k}_{aggr_type}":v 
                                for k,v in branch_attr_dict.items()}
    return branch_attr_dict 

def limb_branch_sum_features(
    limb,
    branches,
    features=None,
    default_value = None,
    **kwargs
):
    
    if features is None:
        features = [
            "area",
            "mesh_volume",
            'n_spines',
            'n_synapses',
            'n_synapses_head',
            'n_synapses_neck',
            'n_synapses_no_head',
            'n_synapses_post',
            'n_synapses_pre',
            'n_synapses_shaft', 
            'n_synapses_spine',
            "skeletal_length",
            'total_spine_volume',
            #bu.width_min,
            #bu.width_max,
            
        ]
    
    return limb_branches_aggr_features(
        limb,
        branches,
        features=features,
        aggr_type = "sum",
        default_value = default_value,
        **kwargs
    ) 
    
def limb_branch_avg_and_sum_features(
    limb,
    branches,
    add_aggr_suffix = True,
    **kwargs
):
    avg_dict = limb_branch_avg_features(
        limb,
        branches,
        add_aggr_suffix=add_aggr_suffix,
        **kwargs
    )
    
    sum_dict = limb_branch_sum_features(
        limb,
        branches,
        add_aggr_suffix=add_aggr_suffix,
        **kwargs
    )
    
    return dict(**avg_dict,**sum_dict)

def parent_node_features(
    limb,
    branch_idx,
    default_value = None,
    add_parent_prefix = True,
    ):
    """
    Purpose
    -------
    
    Generate a an attribute dictionary of attributes for a node
    that serves as the parent of a list of downstream nodes
    
    Pseudocode
    ----------
    1. Compute the 
    """
    from . import limb_utils as lu
    
    branch = limb[branch_idx]
    def get_branch_attr(branch,feat):
        try:
            return getattr(branch,feat)
        except:
            return default_value
    
    def get_limb_branch_attr(limb_obj,branch_idx,func):
        try:
            return func(limb_obj,branch_idx)
        except:
            return None
            
    branch_attr = [
        "width_downstream",
        "width_upstream",
        'min_dist_synapses_pre_upstream',
         'min_dist_synapses_post_upstream',
         'min_dist_synapses_pre_downstream',
         'min_dist_synapses_post_downstream',
        "max_skeleton_endpoint_dist",
    ]
    branch_dict = {k:get_branch_attr(branch,k) 
                   for k in branch_attr}
    
    standard_branch_dict = nst.limb_branch_avg_and_sum_features(
        limb,[branch_idx],add_aggr_suffix = False)
    
    branch_dict.update(standard_branch_dict)
    
    limb_branch_funcs = [
        nst.distance_from_soma,
        
        lu.siblings_skeletal_angle_max,
        lu.siblings_skeletal_angle_min,
        lu.children_skeletal_angle_max,
        lu.children_skeletal_angle_min,
        lu.sibling_angle_smooth_max,
        lu.sibling_angle_smooth_min,
        lu.sibling_angle_smooth_extra_offset_max,
        lu.sibling_angle_smooth_extra_offset_min,
        lu.downstream_endnode_skeletal_distance_from_soma,
    ]
    
    limb_dict = {k.__name__:get_limb_branch_attr(limb,branch_idx,k) 
                    for k in limb_branch_funcs}
    
    branch_dict.update(limb_dict)
    
    if add_parent_prefix:
        branch_dict  = {f"parent_{k}":v 
                            for k,v in branch_dict.items()}
    return branch_dict



# ----------------- Parameters ------------------------


global_parameters_dict_default = dict(
)

attributes_dict_default = dict()

global_parameters_dict_microns = {}
attributes_dict_microns = {}

global_parameters_dict_h01 = {}

attributes_dict_h01 = dict()



#--- from neurd_packages ---
from . import branch_utils as bu
from . import concept_network_utils as cnu
from . import neuron_searching as ns
from . import neuron_utils as nru


#--- from mesh_tools ---
from mesh_tools import skeleton_utils as sk
from mesh_tools import trimesh_utils as tu

#--- from datasci_tools ---
from datasci_tools import general_utils as gu
from datasci_tools import networkx_utils as xu
from datasci_tools import numpy_dep as np
from datasci_tools import numpy_utils as nu
from datasci_tools import pandas_utils as pu
