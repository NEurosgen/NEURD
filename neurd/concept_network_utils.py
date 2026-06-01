import sys as _sys

import networkx as nx
import numpy as np

from datasci_tools import networkx_utils as xu
from datasci_tools import numpy_utils as nu
from . import neuron_utils as nru

cnu = _sys.modules[__name__]

non_branching_upstream = False

def distance_between_nodes_di(limb_obj,
                      start_idx,
                      destination_idx,
                      reverse_di_graph,):
    """
    Purpose: To determine the distance
    between two nodes along a path
    
    """
    branch_path = nru.branch_path_to_node(limb_obj,
                       start_idx=start_idx,
                       destination_idx=destination_idx,
                        include_branch_idx=False,
                        include_last_branch_idx=False,
                        reverse_di_graph=reverse_di_graph)
    
    if branch_path is None:
        return np.inf

    from neurd import neuron_statistics as nst  # P3: local import breaks concept_network_utils<->neuron_statistics cycle
    return nst.skeletal_length_along_path(limb_obj,branch_path)

def distance_between_nodes_di_upstream(limb_obj,
                      start_idx,
                      destination_idx):
    """
    Purpose: To determine the upstream distance
    from branch_idx to start_idx
    
    """
    return cnu.distance_between_nodes_di(limb_obj,
                      start_idx,
                      destination_idx,
                      reverse_di_graph=True)

def distance_between_nodes_di_downstream(limb_obj,
                      start_idx,
                      destination_idx):
    """
    Purpose: To determine the upstream distance
    from branch_idx to start_idx
    
    """
    return cnu.distance_between_nodes_di(limb_obj,
                      start_idx,
                      destination_idx,
                      reverse_di_graph=False)

def branches_within_distance(limb_obj,
                             branch_idx,
                            dist_func,
                            distance_threshold,
                            include_branch_idx = False):
    """
    Purpose: To find all branches with a certain downstream distance
    """
    branch_names = np.array(limb_obj.get_branch_names())
    branch_dist = np.array([dist_func(limb_obj,branch_idx,k) for k in branch_names])
    
    branches_within_dist = branch_names[(branch_dist<=distance_threshold) & 
                                       (branch_dist != np.inf)]
    
    if include_branch_idx:
        return branches_within_dist
    else:
        return branches_within_dist[branches_within_dist != branch_idx]

def branches_within_distance_upstream(limb_obj,
                             branch_idx,
                                     distance_threshold,
                                     include_branch_idx = False):
    return branches_within_distance(limb_obj,
                             branch_idx,
                            dist_func=cnu.distance_between_nodes_di_upstream,
                                   distance_threshold=distance_threshold,
                                   include_branch_idx = include_branch_idx)

def branches_within_distance_downstream(limb_obj,
                             branch_idx,
                                     distance_threshold,
                                       include_branch_idx = False):
    """
    Ex: 
    nst.branches_within_distance_downstream(limb_obj,223,
                                       2000)
    """
    return branches_within_distance(limb_obj,
                             branch_idx,
                            dist_func=cnu.distance_between_nodes_di_downstream,
                                   distance_threshold=distance_threshold,
                                   include_branch_idx = include_branch_idx)

#find all end nodes within a downstream threshold


def subgraph_around_branch(limb_obj,
                           branch_idx,
                           upstream_distance=0,
                           downstream_distance=0,
                           distance = None,
                           distance_attribute = "skeletal_length",
                          include_branch_in_upstream_dist=True,
                          include_branch_in_downstream_dist=True,
                          only_non_branching_downstream=True,
                           only_non_branching_upstream = non_branching_upstream, 
                           include_branch_idx = True,
                          return_branch_idxs=True,
                           plot_subgraph = False,
                           nodes_to_exclude=None,
                           nodes_to_include = None,
                          ):
    """
    Purpose: To return a subgraph around a certain 
    branch to find all the nodes upstream and/or downstream
    
    Pseudocode: 
    1) Find all the branches upstream of branch 
    (subtract the skeletal length of branch if included in upstream dist )
    2) Find all the branches downstrem of branch 
    (subtract the skeletal length of branch if included in upstream dist )
    3) Find the upstream and downstream nodes a certain distance away
    
    Ex: 
    cnu.subgraph_around_branch(limb_obj,
                           branch_idx=97,
                           upstream_distance=1000000,
                           downstream_distance=1000000,
                           distance = None,
                           distance_attribute = "skeletal_length",
                          include_branch_in_upstream_dist=True,
                          include_branch_in_downstream_dist=True,
                          only_non_branching_downstream=True,
                           include_branch_idx = False,
                          return_branch_idxs=True,
                           plot_subgraph = True,
                           verbose = False
                          )
    """
    
    if distance is not None:
        upstream_distance = downstream_distance = distance
    
    branch_idx_dist = getattr(limb_obj[branch_idx],distance_attribute)
    
    if include_branch_in_upstream_dist:
        upstream_distance = upstream_distance - branch_idx_dist
            
    if include_branch_in_downstream_dist:
        downstream_distance = downstream_distance - branch_idx_dist
    
    upstream_branches = cnu.branches_within_distance_upstream(limb_obj,
                                                          branch_idx,
                                                          upstream_distance,
                                                          include_branch_idx=include_branch_idx)
    downstream_branches = cnu.branches_within_distance_downstream(limb_obj,
                                                          branch_idx,
                                                          downstream_distance,
                                                          include_branch_idx=include_branch_idx)
    
    if nodes_to_exclude is None:
        nodes_to_exclude = limb_obj.nodes_to_exclude
        
    if nodes_to_exclude is not None:
        downstream_branches = np.setdiff1d(downstream_branches,nodes_to_exclude)
        upstream_branches = np.setdiff1d(upstream_branches,nodes_to_exclude)
    
    if nodes_to_include is not None:
        downstream_branches = np.intersect1d(downstream_branches,nodes_to_include)
        upstream_branches = np.intersect1d(upstream_branches,nodes_to_include)
    
    if only_non_branching_upstream:
        upstream_branches = np.intersect1d(upstream_branches,cnu.upstream_nodes_without_branching(limb_obj,branch_idx,
                                                                                                  nodes_to_exclude=nodes_to_exclude))
    
    if only_non_branching_downstream:
#         downstream_branches = np.setdiff1d(downstream_branches,cnu.branches_with_parent_branching(limb_obj))
        downstream_branches = np.intersect1d(downstream_branches,cnu.downstream_nodes_without_branching(limb_obj,branch_idx,
                                                                                                       nodes_to_exclude=nodes_to_exclude))
        
    total_branches = np.hstack([upstream_branches,downstream_branches])
    if include_branch_idx:
        if branch_idx not in total_branches:
            total_branches = np.hstack([total_branches,[branch_idx]])
        
    if return_branch_idxs:
        return total_branches
    
    sub_G = limb_obj.concept_network_directional.subgraph(total_branches)
    
    if plot_subgraph:
        nx.draw(sub_G,with_labels=True)
        
    return sub_G

def downstream_nodes_without_branching(limb_obj,
                                      branch_idx,
                                      nodes_to_exclude = None):
    """
    Purpose: To return all nodes that are 
    downstream of a branch but not after a branching point
    """
    if nodes_to_exclude is None:
        nodes_to_exclude = limb_obj.nodes_to_exclude
    
    downstream_nodes = []
    curr_node = branch_idx
    for i,b in enumerate(limb_obj.get_branch_names()):
        d_nodes = nru.downstream_nodes(limb_obj,curr_node)
        
        if nodes_to_exclude is not None:
            d_nodes = np.setdiff1d(d_nodes,nodes_to_exclude)
        
        if len(d_nodes) == 0 or len(d_nodes) > 1:
            return downstream_nodes
        elif len(d_nodes) == 1:
            curr_node = d_nodes[0]
            downstream_nodes.append(curr_node)
            
    return downstream_nodes

def upstream_nodes_without_branching(limb_obj,
                                      branch_idx,
                                    nodes_to_exclude = None):
    """
    Purpose: To return all nodes that are 
    downstream of a branch but not after a branching point
    """
    if nodes_to_exclude is None:
        nodes_to_exclude = limb_obj.nodes_to_exclude
    
    upstream_nodes = []
    curr_node = branch_idx
    for i,b in enumerate(limb_obj.get_branch_names()):
        u_node = nru.upstream_node(limb_obj,curr_node)
        if u_node is None:
            return upstream_nodes
        
        d_nodes = nru.downstream_nodes(limb_obj,u_node)
        
        if nodes_to_exclude is not None:
            d_nodes = np.setdiff1d(d_nodes,nodes_to_exclude)
        
        if len(d_nodes) > 1:
            return upstream_nodes
        elif len(d_nodes) == 1:
            curr_node = u_node
            upstream_nodes.append(curr_node)
            
    return upstream_nodes

# ------ 6/25: Helps find attributes that are downstream or upstream -------

def other_direction(direction):
    if direction=="upstream":
        return "downstream"
    elif direction == "downstream":
        return "upstream"
    else:
        raise Exception(f"unknown direction: {direction}")

def nodes_upstream_downstream(limb_obj,
                             branch_idx,
                             direction,
                             distance = np.inf,include_branch_in_dist = True,
                         only_non_branching = True,
                         include_branch_idx = True,
                         nodes_to_exclude = None,
                             nodes_to_include=None):
    """
    Will return nodes that are upstream or downstream by a certain dist
    """
    arg_dict = {f"include_branch_in_{direction}_dist":include_branch_in_dist,
                f"only_non_branching_{direction}":only_non_branching,
                f"{direction}_distance":distance,
                f"{other_direction(direction)}_distance":-1}
                
    # 1) Get all of the branches that are downstream
    # (either up to branch point or within certain distance)
    all_downstream_nodes = cnu.subgraph_around_branch(limb_obj,
                                                      branch_idx = branch_idx,
                                                      include_branch_idx=include_branch_idx,
                                                      nodes_to_exclude=nodes_to_exclude,
                                                      nodes_to_include=nodes_to_include,
                                                      **arg_dict
                                    )

    return all_downstream_nodes

def attribute_upstream_downstream(limb_obj,
                                 branch_idx,
                                  direction,
                                  attribute_name=None,
                                  attribute_func = None,
                                 concat_func = np.concatenate,
                                 distance = np.inf,
                         include_branch_in_dist = True,
                         only_non_branching = True,
                         include_branch_idx = True,
                         nodes_to_exclude = None,
                         return_nodes = False,
                                 ):

    """
    Purpose: To retrieve and concatenate
    the attributes of a branch and
    all of the branches downsream
    of the branch until there is a branching point
    or within a certain distance

    Pseudocode: 
    1) Get all of the branches that are downstream
    (either up to branch point or within certain distance)
    2) Get the attributes of the branch and all those downstream
    3) concatenate the attributes using the prescribed function

    """

    all_downstream_nodes = cnu.nodes_upstream_downstream(limb_obj,
                             branch_idx,
                             direction,
                             distance = distance,
                            include_branch_in_dist = include_branch_in_dist,
                         only_non_branching = only_non_branching,
                         include_branch_idx = include_branch_idx,
                         nodes_to_exclude = nodes_to_exclude,)
        
    #2) Get the attributes of the branch and all those downstream
    if attribute_func is None:
        down_attr = [getattr(limb_obj[k],attribute_name) for k in all_downstream_nodes]
    else:
        down_attr = [attribute_func(limb_obj[k]) for k in all_downstream_nodes]

    #3) concatenate the attributes using the prescribed function
    if len(down_attr) > 0 and concat_func is not None:
        down_attr_concat = concat_func(down_attr)
    else:
        down_attr_concat = down_attr 
        
    if return_nodes:
        return down_attr_concat,all_downstream_nodes
    else:
        return down_attr_concat





def synapses_upstream_downstream(limb_obj,
                       branch_idx,
                        direction,
                       distance = np.inf,
                       only_non_branching=True,
                        include_branch_in_dist = True,
                        include_branch_idx = True,
                        plot_synapses = False,
                        synapse_type="synapses",
                        return_nodes = False,
                        nodes_to_exclude = None,
                        **kwargs
                       ):
    """
    Purpose: To get the downstream synapses at a branch
    
    Ex: 
    syns = downstream_synapses(limb_obj,16,downstream_distance = 0, include_branch_in_downstream_dist = False,
                    only_non_branching_downstream=False,
                   plot_synapses=True)
                   
    E

    """
    syns,nodes = cnu.attribute_upstream_downstream(limb_obj = limb_obj,
    branch_idx = branch_idx,
    direction=direction,
    attribute_name = synapse_type,
    concat_func = np.concatenate,
    include_branch_idx=include_branch_idx,
    distance = distance,
    only_non_branching = only_non_branching,
    include_branch_in_dist = include_branch_in_dist,
    nodes_to_exclude=nodes_to_exclude,
    return_nodes = True,
     **kwargs)
    
    if return_nodes:
        return syns,nodes
    else:
        return syns
    
def synapses_downstream(limb_obj,
                       branch_idx,
                       distance = np.inf,
                       only_non_branching=True,
                        include_branch_in_dist = True,
                        include_branch_idx = True,
                        plot_synapses = False,
                        synapse_type="synapses",
                        return_nodes = False,
                        nodes_to_exclude = None,
                        **kwargs
                       ):
    return synapses_upstream_downstream(limb_obj,
                       branch_idx,
                         direction="downstream",
                       distance = distance,
                       only_non_branching=only_non_branching,
                        include_branch_in_dist = include_branch_in_dist,
                        include_branch_idx = include_branch_idx,
                        plot_synapses = plot_synapses,
                        synapse_type=synapse_type,
                        return_nodes = return_nodes,
                        nodes_to_exclude = nodes_to_exclude,
                        **kwargs
                       )

def synapses_upstream(limb_obj,
                       branch_idx,
                       distance = np.inf,
                       only_non_branching=True,
                        include_branch_in_dist = True,
                        include_branch_idx = True,
                        plot_synapses = False,
                        synapse_type="synapses",
                        return_nodes = False,
                        nodes_to_exclude = None,
                        **kwargs
                       ):
    return synapses_upstream_downstream(limb_obj,
                       branch_idx,
                         direction="upstream",
                       distance = distance,
                       only_non_branching=only_non_branching,
                        include_branch_in_dist = include_branch_in_dist,
                        include_branch_idx = include_branch_idx,
                        plot_synapses = plot_synapses,
                        synapse_type=synapse_type,
                        return_nodes = return_nodes,
                        nodes_to_exclude = nodes_to_exclude,
                        **kwargs
                       )
    
def weighted_attribute_upstream_downstream(limb_obj,
                                          branch_idx,
                                           direction,
                                          attribute_name,
                                           attribute_func = None,
                                           filter_away_zero_sk_lengths=True,
                                          **kwargs):
    sk_lengths,nodes = cnu.attribute_upstream_downstream(limb_obj = limb_obj,
    branch_idx = branch_idx,
    direction=direction,
    attribute_name = "skeletal_length",
    concat_func = None,
     return_nodes=True,
     **kwargs)

    attr_values,nodes = cnu.attribute_upstream_downstream(limb_obj = limb_obj,
    branch_idx = branch_idx,
    direction=direction,
    attribute_name = attribute_name,
    attribute_func=attribute_func,
    concat_func = None,
    return_nodes=True,           
     **kwargs)

    sk_lengths = np.array(sk_lengths)
    attr_values = np.array(attr_values)

    # if filter_away_zero_widths:
    if filter_away_zero_sk_lengths:
        keep_mask = sk_lengths > 0

        attr_values = attr_values[keep_mask]
        sk_lengths = sk_lengths[keep_mask]

    if len(attr_values) != len(sk_lengths):
        raise Exception("")

    if len(attr_values) > 0:
        return nu.weighted_average(attr_values,sk_lengths)
    else:
        return 0
    
def width_upstream_downstream(limb_obj,
    branch_idx,
    direction,
    distance = np.inf,
    only_non_branching=True,
    include_branch_in_dist = True,
    include_branch_idx = True,
    width_func = None,
    width_attribute = None,
    nodes_to_exclude = None,
    **kwargs):
    
    return cnu.weighted_attribute_upstream_downstream(limb_obj,
                                          branch_idx,
                                          direction=direction,
                                          attribute_name=width_attribute,
                                        attribute_func = width_func,
                                        include_branch_idx=include_branch_idx,
                                        distance = distance,
                                        only_non_branching = only_non_branching,
                                        include_branch_in_dist = include_branch_in_dist,
                                        nodes_to_exclude=nodes_to_exclude,
                                          **kwargs)

def width_upstream(limb_obj,
    branch_idx,
    distance = np.inf,
    only_non_branching=True,
    include_branch_in_dist = True,
    include_branch_idx = True,
    width_func = None,
    width_attribute = None,
    nodes_to_exclude = None,
                              **kwargs):
    """
    cnu.width_downstream(limb_obj,
    branch_idx = 65,
    distance = np.inf,
    only_non_branching=False,
    include_branch_in_dist = True,
    include_branch_idx = True,
    verbose = False,
    width_func = au.axon_width,
    width_attribute = None,
    return_nodes = False,
    nodes_to_exclude = None,)
    """
    
    return width_upstream_downstream(limb_obj,
    branch_idx,
    direction="upstream",
    distance = distance,
    only_non_branching=only_non_branching,
    include_branch_in_dist = include_branch_in_dist,
    include_branch_idx = include_branch_idx,
    width_func = width_func,
    width_attribute = width_attribute,
    nodes_to_exclude = nodes_to_exclude,
    **kwargs)

def width_downstream(limb_obj,
    branch_idx,
    distance = np.inf,
    only_non_branching=True,
    include_branch_in_dist = True,
    include_branch_idx = True,
    width_func = None,
    width_attribute = None,
    nodes_to_exclude = None,
    **kwargs):
    
    return width_upstream_downstream(limb_obj,
    branch_idx,
    direction="downstream",
    distance = distance,
    only_non_branching=only_non_branching,
    include_branch_in_dist = include_branch_in_dist,
    include_branch_idx = include_branch_idx,
    width_func = width_func,
    width_attribute = width_attribute,
    nodes_to_exclude = nodes_to_exclude,
    **kwargs)

def skeletal_length_upstream_downstream(limb_obj,
    branch_idx,
    direction,
    distance = np.inf,
    only_non_branching=True,
    include_branch_in_dist = True,
    include_branch_idx = True,
    return_nodes = False,
    nodes_to_exclude = None,
    **kwargs):
    """
    Purpose: To find the up and downstream width

    Pseudocode: 
    1) Get all up/down sk lengths
    2) Get all up/down widths
    3) Filter away non-zeros widths if argument set
    4) If arrays are non-empty, computed the weighted average

    """

    sk_len,nodes = cnu.attribute_upstream_downstream(limb_obj = limb_obj,
    branch_idx = branch_idx,
    direction=direction,
    attribute_name = "skeletal_length",
    concat_func = np.sum,
    include_branch_idx=include_branch_idx,
    distance = distance,
    only_non_branching = only_non_branching,
    include_branch_in_dist = include_branch_in_dist,
    nodes_to_exclude=nodes_to_exclude,
    return_nodes = True,
     **kwargs)
        
    if return_nodes:
        return sk_len,nodes
    else:
        return sk_len
    
def skeletal_length_upstream(limb_obj,
    branch_idx,
    distance = np.inf,
    only_non_branching=True,
    include_branch_in_dist = True,
    include_branch_idx = True,
    return_nodes = False,
    nodes_to_exclude = None,
    **kwargs):
    
    return skeletal_length_upstream_downstream(limb_obj,
    branch_idx,
    direction = "upstream",
    distance = distance,
    only_non_branching=only_non_branching,
    include_branch_in_dist = include_branch_in_dist,
    include_branch_idx = include_branch_idx,
    return_nodes = return_nodes,
    nodes_to_exclude = nodes_to_exclude,
    **kwargs)

def skeletal_length_downstream(limb_obj,
    branch_idx,
    distance = np.inf,
    only_non_branching=True,
    include_branch_in_dist = True,
    include_branch_idx = True,
    return_nodes = False,
    nodes_to_exclude = None,
    **kwargs):
    
    return skeletal_length_upstream_downstream(limb_obj,
    branch_idx,
    direction = "downstream",
    distance = distance,
    only_non_branching=only_non_branching,
    include_branch_in_dist = include_branch_in_dist,
    include_branch_idx = include_branch_idx,
    return_nodes = return_nodes,
    nodes_to_exclude = nodes_to_exclude,
    **kwargs)

# ------------ synapse density ---------- #

def synapse_density_upstream_downstream(limb_obj,
    branch_idx,
    direction,
    distance = np.inf,
    only_non_branching=True,
    include_branch_in_dist = True,
    include_branch_idx = True,
    synapse_density_type = "synapse_density",
    nodes_to_exclude = None,
                              **kwargs):
    """
    Purpose: To find the up and downstream width

    Pseudocode: 
    1) Get all up/down sk lengths
    2) Get all up/down widths
    3) Filter away non-zeros widths if argument set
    4) If arrays are non-empty, computed the weighted average

    """

    return cnu.weighted_attribute_upstream_downstream(limb_obj = limb_obj,
    branch_idx = branch_idx,
    direction=direction,
    attribute_name = synapse_density_type,
    include_branch_idx=include_branch_idx,
    distance = distance,
    only_non_branching = only_non_branching,
    include_branch_in_dist = include_branch_in_dist,
    nodes_to_exclude=nodes_to_exclude,
     **kwargs)

def downstream_nodes(limb_obj,
                    branch_idx):
    """
    Will give the downstream nodes excluding the 
    nodes to be excluded
    
    """
    down_nodes = nru.downstream_nodes(limb_obj,branch_idx)
    return np.setdiff1d(down_nodes,limb_obj.nodes_to_exclude)

def all_downtream_branches(limb_obj,
                          branch_idx):
    return xu.all_downstream_nodes(limb_obj.concept_network_directional,branch_idx)

def all_upstream_branches(limb_obj,
                          branch_idx):
    return xu.all_upstream_nodes(limb_obj.concept_network_directional,branch_idx)

def all_downstream_branches_from_branches(limb_obj,
                                         branches,
                                         include_original_branches=False):
    if not nu.is_array_like(branches):
        branches = [branches]
        
    all_downs = np.concatenate([cnu.all_downtream_branches(limb_obj,
                                           k) for k in branches])
    if include_original_branches:
        all_downs = np.concatenate([all_downs,branches])
    else:
        all_downs = np.setdiff1d(all_downs,branches)
        
    downstream_nodes = np.unique(all_downs)
    
    return downstream_nodes

def all_upstream_branches_from_branches(limb_obj,
                                         branches,
                                         include_original_branches=False):
    if not nu.is_array_like(branches):
        branches = [branches]
        
    all_downs = np.concatenate([cnu.all_upstream_branches(limb_obj,
                                           k) for k in branches])
    if include_original_branches:
        all_downs = np.concatenate([all_downs,branches])
    else:
        all_downs = np.setdiff1d(all_downs,branches)
        
    upstream_nodes = np.unique(all_downs)
    
    return upstream_nodes

# ---- helps with developing statistics over current/above/below branches
def feature_over_branches(
    limb_obj,
    branches,
    direction = None,#downstream or upstream
    include_original_branches_in_direction = False,
    # argument for computing the feature
    feature_name=None,
    feature_function=None,
    combining_function=None,
    return_skeletal_length = False,
    **kwargs):
    """
    Purpose: To find the average value over a list of branches

    Pseudocode: 
    1) convert the branches list into the branches
    that will be used to compute the statistic
    2) Compute the skeletal length for all the branches
    3) Compute the statistic for all the nodes
    
    Ex: 
    feature_over_branches(limb_obj = n_obj_2[6],
                                branches = [24,2],
                               direction="upstream",
                               verbose = True,
                               feature_function=ns.width
    )
    """

    #1) convert the branches list into the branches
    #that will be used to compute the statistic
    if direction is not None:
        branches = getattr(cnu,f"all_{direction}_branches_from_branches")(limb_obj,
                                                                         branches,
                                                                         include_original_branches=include_original_branches_in_direction)

    #2) Compute the skeletal length for all the branches
    sk_len = [limb_obj[k].skeletal_length for k in branches]

    #3) compute statistics over branches
    branches_val = nru.feature_over_branches(limb_obj,
                                            branch_list = branches,
                                            feature_name=feature_name,
                                            feature_function=feature_function,
                                            combining_function=combining_function,
                                            **kwargs)
    
    if return_skeletal_length:
        return branches_val,sk_len
    else:
        return branches_val
    
def sum_feature_over_branches(
    limb_obj,
    branches,
    direction = None,#downstream or upstream
    include_original_branches_in_direction = False,
    # argument for computing the feature
    feature_name=None,
    feature_function=None,
    combining_function=None,
    default_value = 0,
    **kwargs):
    """
    Purpose: To find the average value over a list of branches

    Pseudocode: 
    1) Find features over branches with skeletal length
    4) Do a weighted average based on skeletal length
    
    Ex: 
    cnu.sum_feature_over_branches(limb_obj = n_obj_2[6],
                                branches = [24,2],
                               direction="upstream",
                               verbose = True,
                               feature_function=ns.width
    )
    """
    branches_val,sk_len = cnu.feature_over_branches(
    limb_obj,
    branches,
    direction = direction,#downstream or upstream
    include_original_branches_in_direction = include_original_branches_in_direction,
    # argument for computing the feature
    feature_name=feature_name,
    feature_function=feature_function,
    combining_function=combining_function,
    return_skeletal_length = True,
    **kwargs)
    
    if len(sk_len) == 0:
        return_value = default_value
    else:
        return_value = np.sum(branches_val)

    return return_value
    
def all_downstream_nodes(limb_obj,branch_idx):
    return xu.all_downstream_nodes(limb_obj.concept_network_directional,
                                   branch_idx)