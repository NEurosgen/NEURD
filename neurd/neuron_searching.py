'''



Purpose: Module provides tools for helping to find the interesting branches
and limbs according to the query and functions that you define

** To create a limb function **
Have it return either one singular value or a dictionary mapping the 
branch idx to a 




'''
import copy
from copy import deepcopy
import pandas as pd
import re
import sys
from datasci_tools import function_utils as fcu
from datasci_tools import numpy_dep as np

comparison_distance_global = 1000
limb_function_append_name = "limb_ns"

def convert_neuron_to_branches_dataframe(current_neuron,
                                        limbs_to_process=None,
                                        limb_branch_dict_restriction=None):
    """axon_segment
    Purpose: 
    How to turn a concept map into a pandas table with only the limb_idx and node_idx
    
    Example: 
    neuron_df = convert_neuron_to_branches_dataframe(current_neuron = recovered_neuron)
    
    """
    curr_concept_network = nru.return_concept_network(current_neuron)
    
    if limbs_to_process is None:
        limb_idxs = nru.get_limb_names_from_concept_network(curr_concept_network)
    else:
        limb_idxs = [nru.get_limb_string_name(k) for k in limbs_to_process]
    
    
    limb_node_idx_dicts = []

    for l in limb_idxs:
        if limb_branch_dict_restriction is None:
            limb_node_idx_dicts += [dict(limb=l,node=int(k)) for k in 
                                curr_concept_network.nodes[l]["data"].concept_network.nodes()]
        else:
            if l in limb_branch_dict_restriction.keys():
                limb_node_idx_dicts += [dict(limb=l,node=int(k)) for k in 
                                    limb_branch_dict_restriction[l]]

    df = pd.DataFrame(limb_node_idx_dicts)
    return df

#wrapper to help with classifying funciton uses
class run_options:
    def __init__(self,run_type="Limb"):
        self.run_type = run_type

    def __call__(self, f):
        f.run_type = self.run_type
        return f

#------------------------------- Branch Functions ------------------------------------#

"""
Want to have functions that just operate off of branch characteristics or limb characteristics
"""

#Branch Functions

@run_options(run_type="Branch")
def n_faces_branch(curr_branch,name=None,branch_name=None,**kwargs):
    return len(curr_branch.mesh.faces)

@run_options(run_type="Branch")
def width(curr_branch,name=None,branch_name=None,**kwargs):
    return curr_branch.width


@run_options(run_type="Branch")
def skeleton_distance_branch(curr_branch,name=None,branch_name=None,**kwargs):
    try:
        #print(f"curr_branch.skeleton = {curr_branch.skeleton.shape}")
        return sk.calculate_skeleton_distance(curr_branch.skeleton)
    except:
        print(f"curr_branch.skeleton = {curr_branch.skeleton}")
        raise Exception("")
        
@run_options(run_type="Branch")
def skeletal_length(curr_branch,name=None,branch_name=None,**kwargs):
    try:
        #print(f"curr_branch.skeleton = {curr_branch.skeleton.shape}")
        return sk.calculate_skeleton_distance(curr_branch.skeleton)
    except:
        print(f"curr_branch.skeleton = {curr_branch.skeleton}")
        raise Exception("")
        

@run_options(run_type="Branch")
def n_spines(branch,limb_name=None,branch_name=None,**kwargs):
    if not branch.spines is None:
        return branch.n_spines
    else:
        return 0

@run_options(run_type="Branch")
def width_new(branch,limb_name=None,branch_name=None,width_new_name="no_spine_mean_mesh_center",
              width_new_name_backup = "no_spine_median_mesh_center",
              **kwargs):
    try:
        return branch.width_new[width_new_name]
    except:
        return branch.width_new[width_new_name_backup]


@run_options(run_type="Branch")
def n_boutons(branch,limb_name=None,branch_name=None,**kwargs):
    return nru.n_boutons(branch)


    

# ---------- new possible widths ----------------- #

@run_options(run_type="Branch")
def median_mesh_center(branch,limb_name=None,branch_name=None,**kwargs):
    return branch.width_new["median_mesh_center"]

@run_options(run_type="Branch")
def no_spine_mean_mesh_center(branch,limb_name=None,branch_name=None,**kwargs):
    return branch.width_new["no_spine_mean_mesh_center"]

@run_options(run_type="Branch")
def no_spine_median_mesh_center(branch,limb_name=None,branch_name=None,**kwargs):
    return branch.width_new["no_spine_median_mesh_center"]

# ---------- new possible widths ----------------- #





@run_options(run_type="Branch")
def no_spine_average_mesh_center(branch,limb_name=None,branch_name=None,**kwargs):
    return branch.width_new["no_spine_average_mesh_center"]

@run_options(run_type="Branch")
def spines_per_skeletal_length(branch,limb_name=None,branch_name=None,**kwargs):
    curr_n_spines = n_spines(branch)
    curr_skeleton_distance = sk.calculate_skeleton_distance(branch.skeleton)
    return curr_n_spines/curr_skeleton_distance

# --------- spine densities ---------------- #
@run_options(run_type="Branch")
def spine_density(branch,limb_name=None,branch_name=None,**kwargs):
    return branch.spine_density


# ----------- working with different labels -----------------#

            

@run_options(run_type="Branch")
def labels(branch,limb_name=None,branch_name=None,**kwargs):
    return branch.labels









        


#------------------------------- Limb Functions ------------------------------------#
"""
Rule: For the limb functions will either return
1) 1 number
2) 1 True/False Value
3) Array of 1 or 2 that matches the number of branches
4) a list of nodes that it applies to

"""
def convert_limb_function_return_to_dict(function_return,
                                        curr_limb_concept_network):
    """
    purpose: to take the returned
    value of a limb function and convert it to 
    a dictionary that maps all of the nodes to a certain value
    - capable of handling both a dictionary and a scalar value
    
    """
    
    #curr_limb_concept_network = curr_neuron_concept_network.nodes[limb_name]["data"].concept_network
    function_mapping_dict = dict()
    if np.isscalar(function_return):
        for branch_idx in curr_limb_concept_network.nodes():
            function_mapping_dict[branch_idx] = function_return
    elif set(list(function_return.keys())) == set(list(curr_limb_concept_network.nodes())):
        function_mapping_dict = function_return
    else:
        raise Exception("The value returned from limb function was not a scalar nor did it match the keys of the limb branches")  
    
    return function_mapping_dict

def convert_limb_function_return_to_limb_branch_dict(function_return,
                                        curr_limb_concept_network,
                                                    limb_name):
    """
    Purpose: returns a dictionary that maps limb to valid branches
    according to a function return that is True or False
    (only includes the branches that are true from the function_return)
    
    Result: retursn a dictionary like dict(L1=[3,5,8,9,10])
    """
    new_dict = convert_limb_function_return_to_dict(function_return,
                                        curr_limb_concept_network)
    return {limb_name:[k for k,v in new_dict.items() if v == True]}
    


@run_options(run_type="Limb")
def skeleton_distance_limb(curr_limb,limb_name=None,**kwargs):
    curr_skeleton = curr_limb.get_skeleton()
    return sk.calculate_skeleton_distance(curr_skeleton)

@run_options(run_type="Limb")
def n_faces_limb(curr_limb,limb_name=None,**kwargs):
    return len(curr_limb.mesh.faces)

@run_options(run_type="Limb")
def merge_limbs(curr_limb,limb_name=None,**kwargs):
    return "MergeError" in curr_limb.labels

@run_options(run_type="Limb")
def limb_error_branches(curr_limb,limb_name=None,**kwargs):
    error_nodes = nru.classify_endpoint_error_branches_from_limb_concept_network(curr_limb.concept_network)
    node_names = np.array(list(curr_limb.concept_network.nodes()))
    return dict([(k,k in error_nodes) for k in node_names])





@run_options(run_type="Limb")
def skeletal_distance_from_soma(curr_limb,
                    limb_name = None,
                    somas = None,
                    error_if_all_nodes_not_return=True,
                    include_node_skeleton_dist=True,
                    print_flag = False,
                    **kwargs
                            
    ):
    
    return nru.skeletal_distance_from_soma(curr_limb,
                    limb_name = None,
                    somas = None,
                    error_if_all_nodes_not_return=True,
                    include_node_skeleton_dist=True,
                    print_flag = False,
                    **kwargs)






    

axon_width_like_functions_list = [
    "width",
    "median_mesh_center",
    "n_spines",
    "n_faces_branch",
    "skeleton_distance_branch",
    "spines_per_skeletal_length",
    "no_spine_median_mesh_center",
]



# ------------- 1//22: Addition that accounts for newer higher fidelity spining --------- #








@run_options(run_type="Limb")
def axon_segment(curr_limb,limb_branch_dict=None,limb_name=None,
                 
                 #the parameters for the axon_segment_downstream_dendrites function
                 downstream_face_threshold=5000,
                 downstream_non_axon_percentage_threshold=0.3,
                 max_skeletal_length_can_flip=20000,
                 distance_for_downstream_check=40000,
                 print_flag=False,
                 
                 
                 #the parameters for the axon_segment_clean_false_positives function
                 width_match_threshold=50,
               width_type = "no_spine_median_mesh_center",
               must_have_spine=True,
             **kwargs):
    """
    Function that will go through and hopefully label all of the axon pieces on a limb
    """
    
    curr_limb_concept_network = curr_limb.concept_network
    #print(f"limb_branch_dict BEFORE = {limb_branch_dict}")
    downstream_filtered_limb_branch_dict = axon_segment_downstream_dendrites(curr_limb,limb_branch_dict,limb_name=limb_name,
             downstream_face_threshold=downstream_face_threshold,
             downstream_non_axon_percentage_threshold=downstream_non_axon_percentage_threshold,                
            max_skeletal_length_can_flip=max_skeletal_length_can_flip,                      
            distance_for_downstream_check=distance_for_downstream_check,
             print_flag=print_flag,
             **kwargs)
    """ Old way of doing before condensed
    #print(f"limb_branch_dict AFTER = {limb_branch_dict}")
    #unravel the output back into a dictionary mapping every node to a value
    #print(f"downstream_filtered_limb_branch_dict BEFORE= {downstream_filtered_limb_branch_dict}")
    downstream_filtered_limb_branch_dict =  convert_limb_function_return_to_dict(downstream_filtered_limb_branch_dict,
                                                curr_limb_concept_network)
    #print(f"downstream_filtered_limb_branch_dict AFTER= {downstream_filtered_limb_branch_dict}")
    #convert the dictionary mapping to a new limb_branch_dict just for that limb
    limb_branch_dict_downstream_filtered = {limb_name:[k for k,v in downstream_filtered_limb_branch_dict.items() if v == True]}
    #print(f"limb_branch_dict_downstream_filtered = {limb_branch_dict_downstream_filtered}")
    
    """
    
    limb_branch_dict_downstream_filtered = convert_limb_function_return_to_limb_branch_dict(downstream_filtered_limb_branch_dict,
                                                                                           curr_limb_concept_network,
                                                                                           limb_name)
    
    
    
    clean_false_positives_dict = axon_segment_clean_false_positives(curr_limb=curr_limb,
                                       limb_branch_dict=limb_branch_dict_downstream_filtered,
                                       limb_name=limb_name,
                                    width_match_threshold=width_match_threshold,
                                   width_type = width_type,
                                   must_have_spine=must_have_spine,
                                 print_flag=print_flag,
                                 **kwargs)
    
    limb_branch_dict_downstream_filtered = convert_limb_function_return_to_limb_branch_dict(clean_false_positives_dict,
                                                                                           curr_limb_concept_network,
                                                                                           limb_name)
    
    #print(f"limb_branch_dict_downstream_filtered before dendrite to axon flipping = {limb_branch_dict_downstream_filtered}")
    
    dendrit_to_axon_flip_branch_dict = flip_dendrite_to_axon(curr_limb,limb_branch_dict_downstream_filtered,limb_name=limb_name,
                                                            **kwargs)
    
    
    
    limb_branch_dict_downstream_filtered = convert_limb_function_return_to_limb_branch_dict(dendrit_to_axon_flip_branch_dict,
                                                                                           curr_limb_concept_network,
                                                                                           limb_name)
    
    #print(f"limb_branch_dict_downstream_filtered AFTER dendrite to axon flipping = {limb_branch_dict_downstream_filtered}")
    
    return dendrit_to_axon_flip_branch_dict

@run_options(run_type="Limb")
def axon_segment_downstream_dendrites(curr_limb,limb_branch_dict,limb_name=None,downstream_face_threshold=5000,
                                      downstream_non_axon_percentage_threshold = 0.5,
                                      max_skeletal_length_can_flip=20000,
                                      distance_for_downstream_check = 50000,
                 print_flag=False,
                 limb_starting_angle_dict=None,
                limb_starting_angle_threshold = 155,
                 #return_limb_branch_dict=False,
                 **kwargs):
    """
    Purpose: To filter the aoxn-like segments (so that does not mistake dendritic branches)
    based on the criteria that an axon segment should not have many non-axon upstream branches
    
    Example on how to run: 
    
    curr_limb_name = "L1"
    curr_limb = uncompressed_neuron.concept_network.nodes[curr_limb_name]["data"]
    ns = reload(ns)

    return_value = axon_segment(curr_limb,limb_branch_dict=limb_branch_dict,
                 limb_name=curr_limb_name,downstream_face_threshold=5000,
                     print_flag=False)
    return_value
    """
    
   
    
    
    if print_flag:
        print(f"downstream_face_threshold= {downstream_face_threshold}")
        print(f"downstream_non_axon_percentage_threshold = {downstream_non_axon_percentage_threshold}")
        print(f"max_skeletal_length_can_flip = {max_skeletal_length_can_flip}")
        print(f"distance_for_downstream_check = {distance_for_downstream_check}")
        #print(f"limb_branch_dict= {limb_branch_dict}")
        
    #curr_limb_branch_dict = kwargs["function_kwargs"]["limb_branch_dict"]
    
    curr_limb_branch_dict = limb_branch_dict
    
    if limb_name not in list(curr_limb_branch_dict.keys()):
        return False
    
    curr_axon_nodes = curr_limb_branch_dict[limb_name]
    
#     if print_flag:
#         print(f"curr_axon_nodes = {curr_axon_nodes}")
    
    curr_limb_copy = deepcopy(curr_limb) #deepcopying so don't change anything
    
    non_axon_nodes = []
    #1) Get all of the concept maps (by first getting all of the somas)
    touching_somas = [k["starting_soma"] for k in curr_limb_copy.all_concept_network_data]
    #2) For each of the concept maps: 
    for sm_start in touching_somas:
        curr_limb_copy.set_concept_network_directional(sm_start)
        curr_directional_network = curr_limb_copy.concept_network_directional
        
        #- For each node: 
        for n in curr_axon_nodes:
            
            # ----------- 1/28: Only want to go downstream for a certain extent, and only want to flip back to dendrite if small segments
            if curr_limb[n].skeletal_length > max_skeletal_length_can_flip:
                if print_flag:
                    print(f"Skipping a possible flip because the length is too long for threshold ({max_skeletal_length_can_flip}): {curr_limb[n].skeletal_length}")
                continue
            #a. Get all of the downstream nodes
            
            curr_downstream_nodes = nru.branches_within_skeletal_distance(limb_obj = curr_limb_copy,
                            start_branch = n,
                            max_distance_from_start = distance_for_downstream_check,
                            verbose = False,
                            include_start_branch_length = False,
                            include_node_branch_length = False,
                            only_consider_downstream = True)
            
            #curr_downstream_nodes = xu.downstream_edges(curr_directional_network,n)
            
            # if there are any downstream nodes
            if len(curr_downstream_nodes) > 0:
                #curr_downstream_nodes = np.concatenate(curr_downstream_nodes)
                #b. Get the total number of faces for all upstream non-axon nodes
                curr_non_axon_nodes = set([k for k in curr_downstream_nodes if k not in curr_axon_nodes])
                downstream_axon_nodes = set([k for k in curr_downstream_nodes if k in curr_axon_nodes])
                
                if len(curr_non_axon_nodes) > 0: 
                    non_axon_face_count = np.sum([len(curr_limb_copy.concept_network.nodes[k]["data"].mesh.faces) for k in curr_non_axon_nodes])
                    axon_face_count = np.sum([len(curr_limb_copy.concept_network.nodes[k]["data"].mesh.faces) for k  in downstream_axon_nodes])
                    perc_non_axon = non_axon_face_count / (non_axon_face_count + axon_face_count)
                    if print_flag:
                        print(f"Soma {sm_start}, limb {limb_name}, node {n} had {non_axon_face_count} non-axon downstream faces, {axon_face_count} axon downstream for a percentage of {perc_non_axon}")  
                    #if non_axon_face_count > downstream_face_threshold:
                    
                    # ----------- 1/20 addition: That factors in percentages and not just raw face count ------- #
                    if downstream_non_axon_percentage_threshold is not None:
                        if print_flag:
                            print(f"perc_non_axon for limb_{limb_name}_node_{n},  = {perc_non_axon}")
                        reverse_label = perc_non_axon > downstream_non_axon_percentage_threshold and non_axon_face_count > downstream_face_threshold
                    else:
                        reverse_label = non_axon_face_count > downstream_face_threshold
                    
                    if reverse_label:
                        non_axon_nodes.append(n)
                        if print_flag:
                            print(f"     Added {n} to non-axon list")
                else:
                    if print_flag:
                        print(f"Soma {sm_start}, limb {limb_name}, node {n} did not hae any NON-AXON downstream targets")
            else:
                if print_flag:
                    print(f"Soma {sm_start}, limb {limb_name}, node {n} did not hae any downstream targets")
    
    #compile all of the non-axon nodes
    total_non_axon_nodes = set(non_axon_nodes)
    
    if print_flag:
        print(f"total_non_axon_nodes = {total_non_axon_nodes}")
    #make a return dictionary that shows the filtered down axons
    return_dict = dict()
    for n in curr_limb_copy.concept_network.nodes():
        if n in curr_axon_nodes and n not in total_non_axon_nodes:
            return_dict[n] = True
        else:
            return_dict[n] = False
    
    return return_dict


# ------- 2/3: Will help flip dendrites back to axons (to help with axon identification) --------------- #

@run_options(run_type="Limb")
def flip_dendrite_to_axon(curr_limb,limb_branch_dict,limb_name=None,
    max_skeletal_length_can_flip_dendrite = 70000,
    downstream_axon_percentage_threshold = 0.85,
    distance_for_downstream_check_dendrite = 50000,
    downstream_face_threshold_dendrite=5000,

    axon_spine_density_max = 0.00015,
    axon_width_max = 600,

    significant_dendrite_downstream_density = 0.0002,
    significant_dendrite_downstream_length = 10000,

    print_flag=False,
                         **kwargs):

    """
    Pseudoode: 
    1) Flip the axon banch list into a dendrite list
    2) Iterate through all the dendrite nodes: 

    a. Run the following checks to exclude dendrite from being flipped:
    - max size
    - spine density
    - width

    b. Get all of the downstream nodes and if there are an downstream nodes:
        i) Get the # of axons and non axons downsream
        ii) If no axons then skip
        iii) Iterate through all the downstream nodes:
            Check for a significant spiny cell and if detect then skip
        iv) get the downstream axon percentage and total numbers
        v) if pass the percentage and total number threshold --> add to the list

    3) Generate a new limb branch dict
    
    Ex: 
    from neurd import neuron_searching as ns  
    curr_limb_idx = 3
    curr_limb = test_neuron[curr_limb_idx]
    limb_name = f"L{curr_limb_idx}"
    try:
        limb_branch_dict = {limb_name:current_axon_limb_branch_dict[limb_name]}
    except:
        limb_branch_dict = {limb_name:[]}

    flip_dendrite_to_axon(curr_limb,limb_branch_dict,limb_name)

    """

    if print_flag:
        print(f"downstream_face_threshold_dendrite= {downstream_face_threshold_dendrite}")
        print(f"downstream_axon_percentage_threshold = {downstream_axon_percentage_threshold}")
        print(f"max_skeletal_length_can_flip_dendrite = {max_skeletal_length_can_flip_dendrite}")
        print(f"distance_for_downstream_check_dendrite = {distance_for_downstream_check_dendrite}")

    curr_limb_branch_dict = limb_branch_dict

    if limb_name not in list(curr_limb_branch_dict.keys()):
        return False

    curr_axon_nodes = curr_limb_branch_dict[limb_name]
    curr_dendrite_nodes = np.setdiff1d(curr_limb.get_branch_names(),curr_axon_nodes)
    curr_dendrite_nodes

    curr_limb_copy = copy.deepcopy(curr_limb)
    non_dendrite_nodes = []
    #1) Get all of the concept maps (by first getting all of the somas)
    touching_somas = [k["starting_soma"] for k in curr_limb_copy.all_concept_network_data]
    #2) For each of the concept maps: 
    for sm_start in touching_somas:
        curr_limb_copy.set_concept_network_directional(sm_start)
        curr_directional_network = curr_limb_copy.concept_network_directional

        for n in curr_dendrite_nodes:
            """
            a. Run the following checks to exclude dendrite from being flipped:
            - max size
            - spine density
            - width
            """
            branch_obj =curr_limb_copy[n]
            curr_density = branch_obj.spine_density
            curr_width = branch_obj.width_new["median_mesh_center"]
            curr_length = branch_obj.skeletal_length  

            if not ((curr_density<axon_spine_density_max) and 
                   (curr_length< max_skeletal_length_can_flip_dendrite) and
                   (curr_width<axon_width_max)):
                if print_flag:
                    print(f"Skipping dendrite candidate {n} because failed the attribute filter with: \n"
                         f"curr_density = {curr_density}\n"
                         f"curr_width = {curr_width}\n"
                         f"curr_length = {curr_length}\n")
                continue

            """
            b. Get all of the downstream nodes and if there are an downstream nodes:
            i) Get the # of axons and non axons downsream
            ii) If no axons then skip
            iii) Iterate through all the downstream nodes:
                Check for a significant spiny cell and if detect then skip
            iv) get the downstream axon percentage and total numbers
            v) if pass the percentage and total number threshold --> add to the list
            """

            #i) Get the # of axons and non axons downsream
            curr_downstream_nodes = nru.branches_within_skeletal_distance(limb_obj = curr_limb_copy,
                                    start_branch = n,
                                    max_distance_from_start = distance_for_downstream_check_dendrite,
                                    verbose = False,
                                    include_start_branch_length = False,
                                    include_node_branch_length = False,
                                    only_consider_downstream = True)

            if print_flag:
                print(f"curr_downstream_nodes = {curr_downstream_nodes}")

            if len(curr_downstream_nodes)>0:
                #i) Get the # of axons and non axons downsream
                curr_non_axon_nodes = set([k for k in curr_downstream_nodes if k not in curr_axon_nodes])
                downstream_axon_nodes = set([k for k in curr_downstream_nodes if k in curr_axon_nodes])

                #ii) If no axons then skip
                if len(downstream_axon_nodes)==0:
                    if print_flag:
                        print(f"Skipping dendrite candidate {n} because no downstream axons")
                    continue

                continue_from_donwstream_dendrite_flag = False


                #iii) Iterate through all the downstream nodes:
                #Check for a significant spiny cell and if detect then skip

                for dn in curr_non_axon_nodes:
                    branch_d = curr_limb_copy[dn]

                    curr_density = branch_d.spine_density
                    curr_length = branch_d.skeletal_length

                    if ((curr_density > significant_dendrite_downstream_density) and
                        (curr_length > significant_dendrite_downstream_length)):

                        if print_flag:
                            print(f"Skipping dendrite candidate {n} because found downstream dendrite with \n"
                                     f"curr_density = {curr_density}\n"
                                     f"curr_length = {curr_length}")

                        continue_from_donwstream_dendrite_flag = True
                        break

                if continue_from_donwstream_dendrite_flag:
                    continue


                #iv) get the downstream axon percentage and total numbers
                non_axon_face_count = np.sum([len(curr_limb_copy.concept_network.nodes[k]["data"].mesh.faces) for k in curr_non_axon_nodes])
                axon_face_count = np.sum([len(curr_limb_copy.concept_network.nodes[k]["data"].mesh.faces) for k  in downstream_axon_nodes])
                perc_axon = axon_face_count / (non_axon_face_count + axon_face_count)

                if print_flag:
                    print(f"Soma {sm_start}, limb {limb_name}, node {n} had {non_axon_face_count} non-axon downstream faces, {axon_face_count}"
                          f" axon downstream for a percentage of {perc_axon}")  


                # ----------- 1/20 addition: That factors in percentages and not just raw face count ------- #
                if downstream_axon_percentage_threshold is not None:
                    if print_flag:
                        print(f"perc_axon for limb_{limb_name}_node_{n},  = {perc_axon}")
                    reverse_label = perc_axon > downstream_axon_percentage_threshold and axon_face_count > downstream_face_threshold_dendrite
                else:
                    reverse_label = axon_face_count > downstream_face_threshold_dendrite

                if reverse_label:
                    non_dendrite_nodes.append(n)
                    if print_flag:
                        print(f"     Added {n} to non-dendrite list")


    #compile all of the non-axon nodes
    total_non_dendrite_nodes = set(non_dendrite_nodes)

    if print_flag:
        print(f"total_non_dendrite_nodes = {total_non_dendrite_nodes}")
    #make a return dictionary that shows the filtered down axons

    return_dict = dict()
    for n in curr_limb_copy.concept_network.nodes():
        if n in curr_axon_nodes or n in total_non_dendrite_nodes:
            return_dict[n] = True
        else:
            return_dict[n] = False

    return return_dict
    

@run_options(run_type="Limb")
def axon_segment_clean_false_positives(curr_limb,
                                       limb_branch_dict,
                                       limb_name=None,
                                    width_match_threshold=50,
                                   width_type = "no_spine_average_mesh_center",
                                   must_have_spine=True,
                                       interest_nodes=[],
                                    #return_limb_branch_dict=False,
                                       false_positive_max_skeletal_length = 35000,
                                 print_flag=False,
                                 **kwargs):
    """
    Purpose: To help prevent the false positives
    where small end dendritic segments are mistaken for axon pieces
    by checking if the mesh transition in width is very constant between an upstream 
    node (that is a non-axonal piece) and the downstream node that is an axonal piece
    then this will change the axonal piece to a non-axonal piece label: 
    
    
    Idea: Can look for where width transitions are pretty constant with preceeding dendrite and axon
    and if very similar then keep as non-dendrite

    *** only apply to those with 1 or more spines

    Pseudocode: 
    1) given all of the axons

    For each axon node:
    For each of the directional concept networks
    1) If has an upstream node that is not an axon --> if not then continue
    1b) (optional) Has to have at least one spine or continues
    2) get the upstream nodes no_spine_average_mesh_center width array
    2b) find the endpoints of the current node
    3) Find which endpoints match from the node and the upstream node
    4) get the tangent part of the no_spine_average_mesh_center width array from the endpoints matching
    (this is either the 2nd and 3rd from front or last depending on touching AND that it is long enough)

    5) get the tangent part of the node based on touching

    6) if the average of these is greater than upstream - 50

    return an updated dictionary
    
    """
    if limb_name not in list(limb_branch_dict.keys()):
        if print_flag:
            print(f"{limb_name} not in curr_limb_branch_dict.keys so returning False")
        return False
    
    curr_axon_nodes = limb_branch_dict[limb_name]
    
    curr_limb_copy = deepcopy(curr_limb)
    
    non_axon_nodes = []
    
    #a) Get all of the concept maps (by first getting all of the somas)
    touching_somas = [k["starting_soma"] for k in curr_limb_copy.all_concept_network_data]
    
    #b) For each of the concept maps: 
    for sm_start in touching_somas:
        curr_limb_copy.set_concept_network_directional(sm_start)
        curr_directional_network = curr_limb_copy.concept_network_directional
        
        #- For each node: 
        for n in curr_axon_nodes:
            
            #if already added to the non-axons nodes then don't need to check anymore
            if n in non_axon_nodes:
                continue
            
            #1) If has an upstream node that is not an axon --> if not then continue
            curr_upstream_nodes = xu.upstream_edges_neighbors(curr_directional_network,n)
            
            try:
                upstream_node_temp = curr_upstream_nodes[0][0]
            except:
                upstream_node_list = curr_upstream_nodes
            else:
                upstream_node_list = curr_upstream_nodes[0]
            
            if len(upstream_node_list) == 0:
                continue
            if len(upstream_node_list) > 1:
                raise Exception(f"More than one upstream node for node {n}: {upstream_node_list}")
            
            upstream_node = upstream_node_list[0]
                
            if print_flag:
                print(f"n = {n}, upstream_node= {upstream_node}")
            
            if upstream_node in curr_axon_nodes:
                if print_flag:
                    print("Skipping because the upstream node is not a non-axon piece")
                continue
                
            if (false_positive_max_skeletal_length < 
                curr_limb_copy.concept_network.nodes[n]["data"].skeletal_length > 
                        false_positive_max_skeletal_length):
                if print_flag:
                    print(f"Skipping because the pice was larger than false_positive_max_skeletal_length = {false_positive_max_skeletal_length} ")
                continue
                    
        
            #1b) (optional) Has to have at least one spine or continues
            if must_have_spine:
                if not (curr_limb_copy.concept_network.nodes[n]["data"].spines) is None:
                    if (curr_limb_copy.concept_network.nodes[n]["data"].n_spines == 0):
                        if print_flag:
                            print(f"Not processing node {n} because there were no spines and  must_have_spine set to {must_have_spine}")
                        continue
                else:
                    if print_flag:
                        print(f"Not processing node {n} because spines were NONE and must_have_spine set to {must_have_spine}")
                    continue #if spines were None
                    
            
            #2- 5) get the tangent touching parts of the mesh
            width_array_1,width_array_2 = wu.find_mesh_width_array_border(curr_limb=curr_limb_copy,
                                 node_1 = n,
                                 node_2 = upstream_node,
                                width_name=width_type,
                                segment_start = 1,
                                segment_end = 6,
                                skeleton_segment_size = 500,
                                print_flag=False,
                                **kwargs
                                )
            
            #6) if the average of these is greater than upstream - 50 then add to the list of non axons
            #interest_nodes = [56,71]`x
            if n in interest_nodes or upstream_node in interest_nodes:
                print(f"width_array_1 = {width_array_1}")
                print(f"width_array_2 = {width_array_2}")
                print(f"np.mean(width_array_1) = {np.mean(width_array_1)}")
                print(f"np.mean(width_array_2) = {np.mean(width_array_2)}")
            
            if np.mean(width_array_1) >= (np.mean(width_array_2)-width_match_threshold):
                
                non_axon_nodes.append(n)
                if print_flag:
                    print(f"Adding node {n} to non_axon list with threshold {width_match_threshold} because \n"
                          f"   np.mean(width_array_1)  = {np.mean(width_array_1) }"
                          f"   np.mean(width_array_2)  = {np.mean(width_array_2) }")
                 
    #after checking all nodes and concept networks
    #compile all of the non-axon nodes
    total_non_axon_nodes = set(non_axon_nodes)
    
    if print_flag:
        print(f"total_non_axon_nodes = {total_non_axon_nodes}")
    #make a return dictionary that shows the filtered down axons
    return_dict = dict()
    for n in curr_limb_copy.concept_network.nodes():
        if n in curr_axon_nodes and n not in total_non_axon_nodes:
            return_dict[n] = True
        else:
            return_dict[n] = False
    
    return return_dict
            
    
# --------- 1/15: Additions to help find axon -----------------#

    


#------------------------------- Creating the Data tables from the neuron and functions------------------------------
def get_run_type(f):
    """
    Purpose: To decide whether a function is a limb or branch
    function

    Pseudocode: 
    1) Try and get the runtype
    2) Extract the name of the first parameter of the function
    3a) if "branch" in then branch, elif "limb" in then limb
    """
    try:
        curr_run_type = getattr(f,"run_type",)
    except:
        try:
            first_arg_name = fcu.arg_names(f)[0].lower()
            if "branch" in first_arg_name:
                curr_run_type = "Branch"
            elif "limb" in first_arg_name:
                curr_run_type = "Limb"
            else:
                raise Exception("No branch argument")
        except:
            curr_run_type = "Branch"
    else:
        pass
    
    return curr_run_type
    
    
def apply_function_to_neuron(current_neuron,current_function,function_kwargs=None,verbose=False
                            ):
    """
    Purpose: To retrieve a dictionary mapping every branch on every node
    to a certain value as defined by the function passed
    
    Example: 
    curr_function = width
    curr_function_mapping = apply_function_to_neuron(recovered_neuron,curr_function)
    
    """
    if function_kwargs is None:
        function_kwargs = dict()
    
    curr_neuron_concept_network = nru.return_concept_network(current_neuron)
    
    if "limbs_to_process" in function_kwargs.keys():
        curr_limb_names = [nru.get_limb_string_name(k) for k in function_kwargs["limbs_to_process"]]
    else:
        curr_limb_names = nru.get_limb_names_from_concept_network(curr_neuron_concept_network)
        
    limb_branch_dict_restriction =  function_kwargs.get("limb_branch_dict_restriction",None)
    
    if limb_branch_dict_restriction is None:
        limb_branch_dict_restriction = nru.neuron_limb_branch_dict(current_neuron)
    else:
        if verbose:
            print(f"Using the limb_branch_dict_restriction = {limb_branch_dict_restriction}")
        
    
    
    function_mapping = dict([(limb_name,dict()) for limb_name in curr_limb_names if limb_name in limb_branch_dict_restriction.keys()])

    #curr_run_type = getattr(current_function,"run_type","Branch")
    curr_run_type = get_run_type(current_function)
    #if it was a branch function that was passed
    if curr_run_type=="Branch":
        for limb_name in function_mapping.keys():
            curr_limb_concept_network = curr_neuron_concept_network.nodes[limb_name]["data"].concept_network
            for branch_idx in curr_limb_concept_network.nodes():
                
                # -- 1/29 Addition: Will only look at nodes in the limb branch dict restriction -- #
                if branch_idx not in limb_branch_dict_restriction[limb_name]:
                    continue
                    
#                 if limb_name == "L0" and branch_idx == 73:
#                     print(f"Computing {current_function} for !!!!!")
                    
                if "str" in str(type(current_function)):
                    function_mapping[limb_name][branch_idx] = getattr(curr_limb_concept_network.nodes[branch_idx]["data"],
                                                                     current_function)
                else:
                    function_mapping[limb_name][branch_idx] = current_function(curr_limb_concept_network.nodes[branch_idx]["data"],limb_name=limb_name,branch_name=branch_idx,**function_kwargs)
                
    elif curr_run_type=="Limb":
        #if it was a limb function that was passed
        """
        - for each limb:
          i) run the function and recieve the returned result
          2) If only a single value is returned --> make dict[limb_idx][node_idx] = value all with the same value
          3) if dictionary of values: 
             a. check that keys match the node_names
             b. make dict[limb_idx][node_idx] = value for all nodes using the dictionary
        
        """
        for limb_name in function_mapping.keys():
            function_return = current_function(curr_neuron_concept_network.nodes[limb_name]["data"],limb_name=limb_name,
                                               **function_kwargs)
            curr_limb_concept_network = curr_neuron_concept_network.nodes[limb_name]["data"].concept_network
            
            function_mapping[limb_name] =  convert_limb_function_return_to_dict(function_return,
                                                        curr_limb_concept_network)
            
            """ Older way of doing this before functionality was moved out to function
            if np.isscalar(function_return):
                for branch_idx in curr_limb_concept_network.nodes():
                    function_mapping[limb_name][branch_idx] = function_return
            elif set(list(function_return.keys())) == set(list(curr_limb_concept_network.nodes())):
                function_mapping[limb_name] = function_return
            else:
                raise Exception("The value returned from limb function was not a scalar nor did it match the keys of the limb branches")
            """
        
    else:
        raise Exception("Function recieved was neither a Branch nor a Limb")
        
    return function_mapping


def map_new_limb_node_value(current_df,mapping_dict,value_name):
    """
    To apply a dictionary to a neuron dataframe table
    
    mapping_dict = dict()
    for x,y in zip(neuron_df["limb"].to_numpy(),neuron_df["node"].to_numpy()):
        if x not in mapping_dict.keys():
            mapping_dict[x]=dict()
        mapping_dict[x][y] = np.random.randint(10)
        
    map_new_limb_node_value(neuron_df,mapping_dict,value_name="random_number")
    neuron_df
    """
    if len(current_df) > 0:
        current_df[value_name] = current_df.apply(lambda x: mapping_dict[x["limb"]][x["node"]], axis=1)
    return current_df

def generate_neuron_dataframe(current_neuron,
                              functions_list,
                              check_nans=True,
                              function_kwargs=dict()):
    """
    Purpose: With a neuron and a specified set of functions generate a dataframe
    with the values computed
    
    Arguments:
    current_neuron: Either a neuron object or the concept network of a neuron
    functions_list: List of functions to process the limbs and branches of the concept network
    check_nans : whether to check and raise an Exception if any nans in run
    
    Application: We will then later restrict using df.eval()
    
    Pseudocode: 
    1) convert the functions_list to a list
    2) Create a dataframe for the neuron
    3) For each function:
    a. get the dictionary mapping of limbs/branches to values
    b. apply the values to the dataframe
    4) return the dataframe
    
    Example: 
    returned_df = generate_neuron_dataframe(recovered_neuron,functions_list=[
    n_faces_branch,
    width,
    skeleton_distance_branch,
    skeleton_distance_limb,
    n_faces_limb,
    merge_limbs,
    limb_error_branches
    ])

    returned_df[returned_df["merge_limbs"] == True]
    """
    
    if not nu.is_array_like(functions_list):
        functions_list = [functions_list]
        
    #print(f"functions_list = {functions_list}")
    
    
    limbs_to_process = function_kwargs.get("limbs_to_process",None)
    limb_branch_dict_restriction=function_kwargs.get("limb_branch_dict_restriction",None)
    
    #2) Create a dataframe for the neuron
    curr_df = convert_neuron_to_branches_dataframe(current_neuron,
                                                  limbs_to_process=limbs_to_process,
                                                  limb_branch_dict_restriction=limb_branch_dict_restriction)
    
    """
    3) For each function:
    a. get the dictionary mapping of limbs/branches to values
    b. apply the values to the dataframe
    
    """
    
    for curr_function in functions_list:
        curr_function_mapping = apply_function_to_neuron(current_neuron,curr_function,function_kwargs)
        curr_function_name = getattr(curr_function,"__name__",curr_function)
        if curr_function_name[-len(limb_function_append_name):] == limb_function_append_name:
            curr_function_name = curr_function_name[:-(len(limb_function_append_name)+1)]
        map_new_limb_node_value(curr_df,curr_function_mapping,value_name=curr_function_name)
        
    if check_nans:
        if pu.n_nans_total(curr_df) > 0:
            print(f"Number of nans = {pu.n_nans_per_column(curr_df)}")
            su.compressed_pickle(curr_df,"curr_df")
            raise Exception("Some fo the data in the dataframe were incomplete")
            
    
    #4) return the dataframe
    return curr_df



# -------------------- Function that does full querying of neuron -------------------------- #


current_module = sys.modules[__name__]


def functions_list_from_query(
    query,
    verbose = False):
    """
    Purpose: To turn a query into a list of functions
    
    Ex: 
    functions_list_from_query(query = 
    "(n_synapses_pre >= 1) and (synapse_pre_perc >= 0.6) and (axon_width <= 270) and (n_spines <= 10) and (n_synapses_post_spine <= 3) and (skeletal_length > 2500) and (area > 1) and (closest_mesh_skeleton_dist < 500)",
      verbose = True                  )

    """
    r_obj = re.compile(r'\((.*?)[ =<> ]+(.*?)\)')
    curr_list = list(r_obj.finditer(query))
    found_functions = list(set([k.group(1).strip() for k in curr_list]))
    if verbose:
        print(f"# of found functions = {len(found_functions)}: {found_functions}")
    return found_functions


def query_neuron(
    concept_network,           
    query,
    functions_list=None,
    function_kwargs=None,
    query_variables_dict=None,
    return_dataframe=False,
    return_dataframe_before_filtering = False,
    return_limbs=False,
    return_limb_grouped_branches=True,
    limb_branch_dict_restriction = None,
    print_flag=False,
    limbs_to_process = None,
    plot_limb_branch_dict = False,
    check_nans = True,
    ):
    """
    *** to specify "limbs_to_process" to process just put in the function kwargs
    
    Purpose: Recieve a neuron object or concept map 
    representing a neuron and apply the query
    to find the releveant limbs, branches
    
    
    Possible Ouptuts: 
    1) filtered dataframe
    2) A list of the [(limb_idx,branches)] ** default
    3) A dictionary that makes limb_idx to the branches that apply (so just grouping them)
    4) Just a list of the limbs
    
    Arguments
    concept_network,
    feature_functios, #the list of str/functions that specify what metrics want computed (so can use in query)
    query, #df.query string that specifies how to filter for the desired branches/limbs
    local_dict=dict(), #if any variables in the query string whose values can be loaded into query (variables need to start with @)
    return_dataframe=False, #if just want the filtered dataframe
    return_limbs=False, #just want limbs in query returned
    return_limb_grouped_branches=True, #if want dictionary with keys as limbs and values as list of branches in the query
    print_flag=True,
    
    
    Example: 
    from os import sys
    sys.path.append("../../neurd_packages/meshAfterParty/meshAfterParty/")
    from importlib import reload
    
    from datasci_tools import pandas_utils as pu
    import pandas as pd
    from pathlib import Path
    
    
    compressed_neuron_path = Path("../test_neurons/test_objects/12345_2_soma_practice_decompress")

    from neurd import neuron_utils as nru
    nru = reload(nru)
    from neurd import neuron
    neuron=reload(neuron)

    from datasci_tools import system_utils as su

    with su.suppress_stdout_stderr():
        recovered_neuron = nru.decompress_neuron(filepath=compressed_neuron_path,
                          original_mesh=compressed_neuron_path)

    recovered_neuron
    
    ns = reload(ns)
    nru = reload(nru)

    list_of_faces = [1038,5763,7063,11405]
    branch_threshold = 31000
    current_query = "n_faces_branch in @list_of_faces or skeleton_distance_branch > @branch_threshold"
    local_dict=dict(list_of_faces=list_of_faces,branch_threshold=branch_threshold)


    functions_list=[
    n_faces_branch,
    "width",
    skeleton_distance_branch,
    skeleton_distance_limb,
    "n_faces_limb",
    merge_limbs,
    limb_error_branches
    ]

    returned_output = query_neuron(recovered_neuron,
                             functions_list,
                              current_query,
                              local_dict=local_dict,
                              return_dataframe=False,
                              return_limbs=False,
                              return_limb_grouped_branches=True,
                             print_flag=False)
    
    
    
    Example 2:  How to use the local dictionary with a list
    
    ns = reload(ns)

    current_functions_list = [
        "skeletal_distance_from_soma",
        "no_spine_average_mesh_center",
        "n_spines",
        "n_faces_branch",

    ]

    function_kwargs=dict(somas=[0],print_flag=False)
    query="skeletal_distance_from_soma > -1 and (limb in @limb_list)"
    query_variables_dict = dict(limb_list=['L1','L2',"L3"])

    limb_branch_dict_df = query_neuron(uncompressed_neuron,
                                       query=query,
                                          function_kwargs=function_kwargs,
                                          query_variables_dict=query_variables_dict,
                   functions_list=current_functions_list,
                                      return_dataframe=True)

    limb_branch_dict = query_neuron(uncompressed_neuron,
                                       query=query,
                   functions_list=current_functions_list,
                                       query_variables_dict=query_variables_dict,
                                       function_kwargs=function_kwargs,
                                      return_dataframe=False)
    
    
    """
    if functions_list is None:
        functions_list = functions_list_from_query(query)
        
    if function_kwargs is None:
        function_kwargs=dict()
        
    if query_variables_dict is None:
        query_variables_dict=dict()
    
    if limbs_to_process is not None:
        limbs_to_process = [nru.get_limb_string_name(k) for k in limbs_to_process]
        function_kwargs["limbs_to_process"] = limbs_to_process
    
    local_dict = query_variables_dict
    concept_network_old = concept_network
    #any preprocessing work
    if concept_network.__class__.__name__ == "Neuron":
        if print_flag:
            print("Extracting concept network from neuron object")
        concept_network = concept_network.concept_network
        
    if print_flag:
        print(f"functions_list = {functions_list}")
        
    final_feature_functions = []
    for f in functions_list:
        if "str" in str(type(f)):
            try:
                curr_feature = getattr(current_module, f,f)
                #curr_feature = 0
            except:
                raise Exception(f"The funciton {f} specified by string was not a pre-made funciton in neuron_searching module")
        elif callable(f):
            curr_feature = f
        else:
            raise Exception(f"Function item {f} was not a string or callable function")
        
        final_feature_functions.append(curr_feature)
    
    if print_flag:
        print(f"final_feature_functions = {final_feature_functions}")
        
        
    
    if limb_branch_dict_restriction is not None:
        function_kwargs["limb_branch_dict_restriction"] = limb_branch_dict_restriction
        
    #0) Generate a pandas table that originally has the limb index and node index
    returned_df = generate_neuron_dataframe(concept_network,functions_list=final_feature_functions,
                                           function_kwargs=function_kwargs,
                                           check_nans=check_nans)
    
    if return_dataframe_before_filtering:
        return returned_df
    
    if len(returned_df)>0:
        filtered_returned_df = returned_df.query(query,
                      local_dict=local_dict)
    else:
        filtered_returned_df = returned_df
    
    """
    Preparing output for returning
    """

    if return_dataframe:
        return filtered_returned_df
    
    
    if len(filtered_returned_df)>0:
        """ -- old method --
        limb_branch_pairings = filtered_returned_df[["limb","node"]].to_numpy()

        #gets a dictionary where key is the limb and value is a list of all the branches that were in the filtered dataframe
        limb_to_branch = dict([(k,np.sort(limb_branch_pairings[:,1][np.where(limb_branch_pairings[:,0]==k)[0]]).astype("int")) 
                               for k in np.unique(limb_branch_pairings[:,0])])
        """
        
        limb_to_branch = nst.limb_branch_from_stats_df(filtered_returned_df)
        
        
        if limb_branch_dict_restriction is not None:
            limb_to_branch = nru.limb_branch_intersection([limb_branch_dict_restriction,limb_to_branch])
        
        if return_limbs:
            return list(limb_to_branch.keys())

        if return_limb_grouped_branches:
            return limb_to_branch

        return limb_branch_pairings
    else:
        if plot_limb_branch_dict:
            print(f"No limb branch to plot")
        if return_limbs:
            return []
        if return_limb_grouped_branches:
            return dict()
        if return_limb_grouped_branches:
            return np.array([]).reshape(-1,2)
        
        







# ----- 4/19 ---------------


    





#raise Exception("Need to fix all of the relational skeletal angles")





# @run_options(run_type="Limb")
# def n_downstream_nodes(curr_limb,limb_name=None,
#                **kwargs):
#     output_dict = dict()
#     for b in curr_limb.get_branch_names():
#         downstream_nodes = xu.downstream_nodes(curr_limb.concept_network_directional,b)
#         output_dict[b] = len(downstream_nodes)
        
#     return output_dict






@run_options(run_type="Limb")
def skeletal_length_downstream(curr_limb,limb_name=None,
               **kwargs):
    output_dict = dict()
    for b in curr_limb.get_branch_names():
        curr_value = nru.skeletal_length_over_downstream_branches(curr_limb,
                                                b,
                                                verbose=False)
        output_dict[b] = curr_value
    return output_dict






def run_limb_function(limb_func,curr_limb,limb_name=None,
                    limb_branch_dict_restriction=None,
               **kwargs):
    """
    Will run a generic limb function
    """
    #print(f"limb_name = {limb_name},kwargs = {kwargs}")
    output_dict = dict()
    for b in curr_limb.get_branch_names():
        if limb_branch_dict_restriction is not None:
            if b not in limb_branch_dict_restriction[limb_name]:
                output_dict[b] = None
                continue
        #print(f"b = {b}")
        div = limb_func(branch_idx = b,
                limb_obj = curr_limb,
                verbose = False,
                **kwargs)
        output_dict[b] = div
    return output_dict

@run_options(run_type="Limb")
def upstream_skeletal_length(curr_limb,limb_name=None,
                    limb_branch_dict_restriction=None,
               **kwargs):
    return run_limb_function(nst.upstream_skeletal_length,
                            curr_limb=curr_limb,
                             limb_name=limb_name,
                             limb_branch_dict_restriction=limb_branch_dict_restriction,
                             **kwargs
                            )

@run_options(run_type="Limb")
def total_upstream_skeletal_length(curr_limb,limb_name=None,
                    limb_branch_dict_restriction=None,
               **kwargs):
    return run_limb_function(nst.total_upstream_skeletal_length,
                            curr_limb=curr_limb,
                             limb_name=limb_name,
                             limb_branch_dict_restriction=limb_branch_dict_restriction,
                             **kwargs
                            )



# ---------- 6/9: Synapse Features --------------#


# ----------- 6/21: v6 statistics -----------

# --------- 7/19: Helping with the new E/I classification --------



# ------- functions for apical classification ---------#









# ------------------ 2/9 autoproofredaing of human ---------

@run_options(run_type="Branch")
def closest_mesh_skeleton_dist(curr_branch,name=None,branch_name=None,**kwargs):
    return bu.closest_mesh_skeleton_dist(curr_branch)

@run_options(run_type="Branch")
def area(curr_branch,name=None,branch_name=None,**kwargs):
    return curr_branch.area

# ------------- skeletal angles -------------


def set_limb_functions_for_search(
    module,
    functions = None,
    append_name = limb_function_append_name,
    verbose = False):
    """
    Purpose: To add wrappers for all the functions
    so can operate in generating a neurons dataframe
    
    Pseudocode: 
    1) Get all of the functions in the module
    2) Filter the functions for only those that have limb in the first arg
    
    For all functions
    3) Send each of the functions through the wrapper
    4) Set the function in module with new name
    """
    if functions is None:
        all_func_names = fcu.all_functions_from_module(module,return_only_names=True)
    else:
        all_func_names = functions
        
    limb_funcs = [k for k in all_func_names if "limb" in fcu.arg_names(getattr(module,k))[0]]
    
    if verbose:
        print(f"all_func_name = {all_func_names}")
        print(f"limb_funcs = {limb_funcs}")
        
    def rename(newname):
        def decorator(f):
            f.__name__ = newname
            return f
        return decorator
        
    for f in limb_funcs:
        new_name = f"{f}_{append_name}"
        if verbose:
            print(f"Creating new function: {new_name}")
        f_func = getattr(module,f)
        def make_func(func):
            @rename(new_name)
            def dummy_func(curr_limb,limb_name=None,limb_branch_dict_restriction=None,**kwargs):
                return run_limb_function(func,curr_limb=curr_limb,
                                 limb_name=limb_name,
                                 limb_branch_dict_restriction=limb_branch_dict_restriction,
                                 **kwargs)
            return dummy_func
        setattr(module,new_name,make_func(f_func))

# --- for better controlling stitching errors --



#--- from neurd_packages ---
from . import branch_utils as bu
from . import neuron_statistics as nst
from . import neuron_utils as nru

from . import width_utils as wu

#--- from mesh_tools ---
from mesh_tools import skeleton_utils as sk

#--- from datasci_tools ---
from datasci_tools import networkx_utils as xu
from datasci_tools import numpy_dep as np
from datasci_tools import numpy_utils as nu
from datasci_tools import pandas_utils as pu
from datasci_tools import system_utils as su

