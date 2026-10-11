"""Evaluate branch and limb features and query neurons through dataframes."""
import copy
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
    """Return one dataframe row per selected limb and branch."""
    curr_concept_network = nru.return_concept_network(current_neuron)
    
    if limbs_to_process is None:
        limb_idxs = nru.get_limb_names_from_concept_network(curr_concept_network)
    else:
        limb_idxs = [nru.get_limb_string_name(k) for k in limbs_to_process]
    limb_node_idx_dicts = []
    for l in limb_idxs:
        if limb_branch_dict_restriction is None:
            branches = curr_concept_network.nodes[l]["data"].concept_network.nodes()
        elif l in limb_branch_dict_restriction:
            branches = limb_branch_dict_restriction[l]
        else:
            continue
        limb_node_idx_dicts.extend(dict(limb=l, node=int(k)) for k in branches)
    return pd.DataFrame(limb_node_idx_dicts, columns=["limb", "node"])

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
    except Exception:
        print(f"curr_branch.skeleton = {curr_branch.skeleton}")
        raise Exception("")
        
@run_options(run_type="Branch")
def skeletal_length(curr_branch,name=None,branch_name=None,**kwargs):
    try:
        #print(f"curr_branch.skeleton = {curr_branch.skeleton.shape}")
        return sk.calculate_skeleton_distance(curr_branch.skeleton)
    except Exception:
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
    except Exception:
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
    

#------------------------------- Creating the Data tables from the neuron and functions------------------------------
def get_run_type(f):
    """Use a feature's declared run type, then its first argument's name."""
    try:
        return getattr(f, "run_type")
    except Exception:
        try:
            first_arg_name = fcu.arg_names(f)[0].lower()
            if "branch" in first_arg_name:
                return "Branch"
            elif "limb" in first_arg_name:
                return "Limb"
        except Exception:
            pass
    return "Branch"
    
    
def apply_function_to_neuron(current_neuron,current_function,function_kwargs=None,verbose=False
                            ):
    """Evaluate a branch feature or broadcast a limb feature to its branches."""
    if function_kwargs is None:
        function_kwargs = dict()
    curr_neuron_concept_network = nru.return_concept_network(current_neuron)
    if "limbs_to_process" in function_kwargs.keys():
        curr_limb_names = [nru.get_limb_string_name(k) for k in function_kwargs["limbs_to_process"]]
    else:
        curr_limb_names = nru.get_limb_names_from_concept_network(curr_neuron_concept_network)
    limb_branch_dict_restriction = function_kwargs.get("limb_branch_dict_restriction", None)
    if limb_branch_dict_restriction is None:
        limb_branch_dict_restriction = nru.neuron_limb_branch_dict(current_neuron)
    elif verbose:
        print(f"Using the limb_branch_dict_restriction = {limb_branch_dict_restriction}")
    function_mapping = {name: {} for name in curr_limb_names
                        if name in limb_branch_dict_restriction.keys()}
    curr_run_type = get_run_type(current_function)
    if curr_run_type == "Branch":
        for limb_name in function_mapping.keys():
            curr_limb_concept_network = curr_neuron_concept_network.nodes[limb_name]["data"].concept_network
            for branch_idx in curr_limb_concept_network.nodes():
                if branch_idx not in limb_branch_dict_restriction[limb_name]:
                    continue
                curr_branch = curr_limb_concept_network.nodes[branch_idx]["data"]
                if "str" in str(type(current_function)):
                    value = getattr(curr_branch, current_function)
                else:
                    value = current_function(curr_branch, limb_name=limb_name,
                                             branch_name=branch_idx, **function_kwargs)
                function_mapping[limb_name][branch_idx] = value
    elif curr_run_type == "Limb":
        for limb_name in function_mapping.keys():
            curr_limb = curr_neuron_concept_network.nodes[limb_name]["data"]
            function_return = current_function(curr_limb, limb_name=limb_name, **function_kwargs)
            curr_limb_concept_network = curr_neuron_concept_network.nodes[limb_name]["data"].concept_network
            function_mapping[limb_name] = convert_limb_function_return_to_dict(
                function_return, curr_limb_concept_network)
    else:
        raise Exception("Function recieved was neither a Branch nor a Limb")
    return function_mapping


def map_new_limb_node_value(current_df,mapping_dict,value_name):
    """Add a dataframe column from a ``{limb: {branch: value}}`` mapping."""
    if len(current_df) > 0:
        current_df[value_name] = current_df.apply(lambda x: mapping_dict[x["limb"]][x["node"]], axis=1)
    return current_df

def generate_neuron_dataframe(current_neuron,
                              functions_list,
                              check_nans=True,
                              function_kwargs=dict()):
    """Evaluate branch or limb features into a dataframe for querying.

    Feature names become columns; ``function_kwargs`` can restrict limbs and
    branches. With ``check_nans=True``, incomplete feature values raise.
    """
    if not nu.is_array_like(functions_list):
        functions_list = [functions_list]
    curr_df = convert_neuron_to_branches_dataframe(current_neuron,
        limbs_to_process=function_kwargs.get("limbs_to_process", None),
        limb_branch_dict_restriction=function_kwargs.get("limb_branch_dict_restriction", None))
    for curr_function in functions_list:
        curr_function_mapping = apply_function_to_neuron(current_neuron,curr_function,function_kwargs)
        curr_function_name = getattr(curr_function,"__name__",curr_function)
        if curr_function_name[-len(limb_function_append_name):] == limb_function_append_name:
            curr_function_name = curr_function_name[:-(len(limb_function_append_name)+1)]
        map_new_limb_node_value(curr_df,curr_function_mapping,value_name=curr_function_name)
    if check_nans:
        nans_per_column = curr_df.isnull().sum()
        if nans_per_column.sum() > 0:
            raise Exception("Some of the data in the dataframe were incomplete; "
                            f"nans per column = {nans_per_column}")
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
    """Select branches using a pandas query over branch and limb features.

    ``functions_list`` accepts callables and names from this module or branch
    attributes. If omitted, feature names are extracted from ``query``.
    ``query_variables_dict`` provides values referenced with ``@name``.
    ``function_kwargs`` supplies feature arguments; the limb restrictions limit
    which rows are evaluated.

    Returns a ``{limb: branch_indices}`` mapping by default. The dataframe flags
    return feature rows before or after filtering; ``return_limbs`` returns
    the matching limb names. With ``return_limb_grouped_branches=False``, return
    an ordered ``(limb, node)`` array, including an array of shape ``(0, 2)``
    when no rows match. Pair and dataframe results keep selected row order and
    duplicates; grouped results sort branches, and an explicit restriction
    intersects them with its branch indices.

    Example::

        query_neuron(neuron, "(width > @minimum) and (n_faces_branch > 100)",
                     query_variables_dict={"minimum": 115})
    """
    if functions_list is None:
        functions_list = functions_list_from_query(query)
    if function_kwargs is None:
        function_kwargs=dict()
    if query_variables_dict is None:
        query_variables_dict=dict()
    if limbs_to_process is not None:
        function_kwargs["limbs_to_process"] = [nru.get_limb_string_name(k) for k in limbs_to_process]
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
            except Exception:
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
    returned_df = generate_neuron_dataframe(concept_network,functions_list=final_feature_functions,
                                           function_kwargs=function_kwargs,
                                           check_nans=check_nans)
    if return_dataframe_before_filtering:
        return returned_df
    filtered_returned_df = (returned_df.query(query, local_dict=query_variables_dict)
                            if len(returned_df) > 0 else returned_df)
    if return_dataframe:
        return filtered_returned_df
    if len(filtered_returned_df) == 0 and plot_limb_branch_dict:
        print("No limb branch to plot")
    if return_limbs or return_limb_grouped_branches:
        limb_to_branch = (limb_branch_from_stats_df(filtered_returned_df)
                          if len(filtered_returned_df) > 0 else {})
        if len(filtered_returned_df) > 0 and limb_branch_dict_restriction is not None:
            limb_to_branch = nru.limb_branch_intersection([limb_branch_dict_restriction,limb_to_branch])
        return list(limb_to_branch.keys()) if return_limbs else limb_to_branch
    return filtered_returned_df[["limb", "node"]].to_numpy()
        
        
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


@run_options(run_type="Branch")
def closest_mesh_skeleton_dist(curr_branch,name=None,branch_name=None,**kwargs):
    return bu.closest_mesh_skeleton_dist(curr_branch)

@run_options(run_type="Branch")
def area(curr_branch,name=None,branch_name=None,**kwargs):
    return curr_branch.area

#--- from neurd_packages ---
from . import branch_utils as bu
from . import neuron_utils as nru

#--- from mesh_tools ---
from mesh_tools import skeleton_utils as sk

#--- from datasci_tools ---
from datasci_tools import numpy_utils as nu

