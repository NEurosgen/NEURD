'''



Purpose of this file: To help the development of the neuron object
1) Concept graph methods
2) Preprocessing pipeline for creating the neuron object from a meshs




'''
import copy
from copy import deepcopy
import networkx as nx

import time

from trimesh.ray import ray_pyembree
from datasci_tools import numpy_dep as np


soma_face_offset = 2


#importing at the bottom so don't get any conflicts

#for meshparty preprocessing

# tools for restricting 

# -------------- 7/22 Help Filter Bad Branches ------------------ #
def classify_error_branch(curr_branch,width_to_face_ratio=5):
    curr_width = curr_branch.width
    curr_face_count = len(curr_branch.mesh.faces)
    if curr_width/curr_face_count > width_to_face_ratio:
        return True
    else:
        return False
    
def classify_endpoint_error_branches_from_limb_concept_network(curr_concept_network,**kwargs):
    """
    Purpose: To identify all endpoints of concept graph where the branch meshes/skeleton
    are likely a result of bad skeletonization or meshing:
    
    Applications: Can get rid of these branches later
    
    Pseudocode: 
    1) Get all of the endpoints of the concept network
    2) Get all of the branch objects for the endpoints
    3) Return the idx's of the branch objects that test positive for being an error branch
    """
    
    #1) Get all of the endpoints of the concept network
    end_nodes = xu.get_nodes_of_degree_k(curr_concept_network,1)
    
    #2) Get all of the branch objects for the endpoints
    end_node_branches = [curr_concept_network.nodes[k]["data"] for k in end_nodes]
    
    #3) Return the idx's of the branch objects that test positive for being an error branch
    total_error_end_node_branches = []
    
    for en_idx,e_branch in zip(end_nodes,end_node_branches):
        if classify_error_branch(e_branch):
            total_error_end_node_branches.append(en_idx)
    
    return total_error_end_node_branches
    

# -------------- tools for the concept networks ------------------ #







def get_limb_names_from_concept_network(concept_network):
    """
    Purpose: Function that takes in either a neuron object
    or the concept network and returns just the concept network
    depending on the input
    
    """
    return [k for k in concept_network.nodes() if "L" in k]

    

def return_concept_network(current_neuron):
    """
    Purpose: Function that takes in either a neuron object
    or the concept network and returns just the concept network
    depending on the input
    
    """
    if current_neuron.__class__.__name__ == "Neuron":
        curr_concept_network = current_neuron.concept_network
    #elif type(current_neuron) == type(xu.GraphOrderedEdges()):
    elif current_neuron.__class__.__name__ == "GraphOrderedEdges":
        curr_concept_network = current_neuron
    else:
        exception_string = (f"current_neuron not a Neuron object or Graph Ordered Edges instance: {type(current_neuron)}"
                       f"\n {current_neuron.__class__.__name__}"
                       f"\n {xu.GraphOrderedEdges().__class__.__name__}"
                           f"\n {current_neuron.__class__.__name__ == xu.GraphOrderedEdges().__class__.__name__}")
        print(exception_string)
        raise Exception("")
    return curr_concept_network
    

def convert_limb_concept_network_to_neuron_skeleton(curr_concept_network,check_connected_component=True):
    """
    Purpose: To take a concept network that has the branch 
    data within it to the skeleton for that limb
    
    Pseudocode: 
    1) Get the nodes names of the branches 
    2) Order the node names
    3) For each node get the skeletons into an array
    4) Stack the array
    5) Want to check that skeleton is connected component
    
    Example of how to run: 
    full_skeleton = convert_limb_concept_network_to_neuron_skeleton(recovered_neuron.concept_network.nodes["L1"]["data"].concept_network)
    
    """
    sorted_nodes = np.sort(list(curr_concept_network.nodes()))
    #print(f"sorted_nodes = {sorted_nodes}")
    full_skeleton = sk.stack_skeletons([curr_concept_network.nodes[k]["data"].skeleton for k in sorted_nodes])
    if check_connected_component:
        sk.check_skeleton_one_component(full_skeleton)
    return full_skeleton
    
def get_starting_info_from_concept_network(concept_networks):
    """
    Purpose: To turn a dictionary that maps the soma indexes to a concept map
    into just a list of dictionaries with all the staring information
    
    Ex input:
    concept_networks = {0:concept_network, 1:concept_network,}
    
    Ex output:
    [dict(starting_soma=..,starting_node=..
            starting_endpoints=...,starting_coordinate=...,touching_soma_vertices=...)]
    
    Pseudocode: 
    1) get the soma it's connect to
    2) get the node that has the starting coordinate 
    3) get the endpoints and starting coordinate for that nodes
    """
    
    
    output_dicts = []
    for current_soma,curr_concept_network_list in concept_networks.items():
        for curr_concept_network in curr_concept_network_list:
            curr_output_dict = dict()
            # 1) get the soma it's connect to
            curr_output_dict["starting_soma"] = current_soma

            # 2) get the node that has the starting coordinate 
            starting_node = xu.get_starting_node(curr_concept_network)
            curr_output_dict["starting_node"] = starting_node

            endpoints_dict = xu.get_node_attributes(curr_concept_network,attribute_name="endpoints",node_list=[starting_node],
                           return_array=False)

            curr_output_dict["starting_endpoints"] = endpoints_dict[starting_node]

            starting_node_dict = xu.get_node_attributes(curr_concept_network,attribute_name="starting_coordinate",node_list=[starting_node],
                           return_array=False)
            #get the starting coordinate of the starting dict
            curr_output_dict["starting_coordinate"] = starting_node_dict[starting_node]

            if "touching_soma_vertices" in curr_concept_network.nodes[starting_node].keys():
                curr_output_dict["touching_soma_vertices"] = curr_concept_network.nodes[starting_node]["touching_soma_vertices"]
            else:
                curr_output_dict["touching_soma_vertices"] = None
                
            #soma starting group
            if "soma_group_idx" in curr_concept_network.nodes[starting_node].keys():
                curr_output_dict["soma_group_idx"] = curr_concept_network.nodes[starting_node]["soma_group_idx"]
            else:
                curr_output_dict["soma_group_idx"] = None
                
            

            curr_output_dict["concept_network"] = curr_concept_network
            output_dicts.append(curr_output_dict)
    
    return output_dicts





def convert_concept_network_to_directional(concept_network,
                                        node_widths=None,
                                          no_cycles=True,
                                          suppress_disconnected_errors=False,
                                          verbose=False):
    """
    Pseudocode: 
    0) Create a dictionary with the keys as all the nodes and empty list as values
    1) Get the starting node
    2) Find all neighbors of starting node
    2b) Add the starting node to the list of all the nodes it is neighbors to
    3) Add starter node to the "procesed_nodes" so it is not processed again
    4) Add each neighboring node to the "to_be_processed" list

    5) Start loop that will continue until "to_be_processed" is done
    a. Get the next node to be processed
    b. Get all neighbors
    c. For all nodes who are not currently in the curr_nodes's list from the lookup dictionary
    --> add the curr_node to those neighbor nodes lists
    d. For all nodes not already in the to_be_processed or procesed_nodes, add them to the to_be_processed list
    ...
    z. when no more nodes in to_be_processed list then reak

    6) if the no_cycles option is selected:
    - for every neruong with multiple neurons in list, choose the one that has the branch width that closest matches

    7) convert the incoming edges dictionary to edge for a directional graph
    
    Example of how to use: 
    
    example_concept_network = nx.from_edgelist([[1,2],[2,3],[3,4],[4,5],[2,5],[2,6]])
    nx.draw(example_concept_network,with_labels=True)
    plt.show()
    xu.set_node_attributes_dict(example_concept_network,{1:dict(starting_coordinate=np.array([1,2,3]))})

    directional_ex_concept_network = convert_concept_network_to_directional(example_concept_network,no_cycles=True)
    nx.draw(directional_ex_concept_network,with_labels=True)
    plt.show()

    node_widths = {1:0.5,2:0.61,3:0.73,4:0.88,5:.9,6:0.4}
    directional_ex_concept_network = convert_concept_network_to_directional(example_concept_network,no_cycles=True,node_widths=node_widths)
    nx.draw(directional_ex_concept_network,with_labels=True)
    plt.show()
    """

    curr_limb_concept_network = concept_network
    mesh_widths = node_widths

    #if only one node in concept_network then return
    if len(curr_limb_concept_network.nodes()) <= 1:
        if verbose:
            print("Concept graph size was 1 or less so returning original")
        return nx.DiGraph(curr_limb_concept_network)

    #0) Create a dictionary with the keys as all the nodes and empty list as values
    incoming_edges_to_node = dict([(k,[]) for k in curr_limb_concept_network.nodes()])
    to_be_processed_nodes = []
    processed_nodes = []
    max_iterations = len(curr_limb_concept_network.nodes()) + 100

    #1) Get the starting node 
    starting_node = xu.get_starting_node(curr_limb_concept_network)

    #2) Find all neighbors of starting node
    curr_neighbors = xu.get_neighbors(curr_limb_concept_network,starting_node)

    #2b) Add the starting node to the list of all the nodes it is neighbors to
    for cn in curr_neighbors:
        incoming_edges_to_node[cn].append(starting_node)

    #3) Add starter node to the "procesed_nodes" so it is not processed again
    processed_nodes.append(starting_node)

    #4) Add each neighboring node to the "to_be_processed" list
    to_be_processed_nodes.extend([k for k in curr_neighbors if k not in processed_nodes ])
    # print(f"incoming_edges_to_node AT START= {incoming_edges_to_node}")
    # print(f"processed_nodes_AT_START = {processed_nodes}")
    # print(f"to_be_processed_nodes_AT_START = {to_be_processed_nodes}")

    #5) Start loop that will continue until "to_be_processed" is done
    for i in range(max_iterations):
    #     print("\n")
    #     print(f"processed_nodes = {processed_nodes}")
    #     print(f"to_be_processed_nodes = {to_be_processed_nodes}")

        if len(to_be_processed_nodes) == 0:
            break
        #a. Get the next node to be processed
        
        curr_node = to_be_processed_nodes.pop(0)
        #print(f"curr_node = {curr_node}")
        #b. Get all neighbors
        curr_node_neighbors = xu.get_neighbors(curr_limb_concept_network,curr_node)
        #print(f"curr_node_neighbors = {curr_node_neighbors}")
        #c. For all nodes who are not currently in the curr_nodes's list from the lookup dictionary
        #--> add the curr_node to those neighbor nodes lists
        for cn in curr_node_neighbors:
            if cn == curr_node:
                raise Exception("found a self connection in network graph")
            if cn not in incoming_edges_to_node[curr_node]:
                incoming_edges_to_node[cn].append(curr_node)

            #d. For all nodes not already in the to_be_processed or procesed_nodes, add them to the to_be_processed list
            if cn not in to_be_processed_nodes and cn not in processed_nodes:
                to_be_processed_nodes.append(cn)


        # add the nodes to those been processed
        processed_nodes.append(curr_node)


        #z. when no more nodes in to_be_processed list then reak
        

    #print(f"incoming_edges_to_node = {incoming_edges_to_node}")
    #6) if the no_cycles option is selected:
    #- for every neruong with multiple neurons in list, choose the one that has the branch width that closest matches

    incoming_lengths = [k for k,v in incoming_edges_to_node.items() if len(v) >= 1]
    
    if not suppress_disconnected_errors:
        if len(incoming_lengths) != len(curr_limb_concept_network.nodes())-1:
            raise Exception("after loop in directed concept graph, not all nodes have incoming edges (except starter node)")

    if no_cycles == True:
        if verbose:
            print("checking and resolving cycles")
        #get the nodes with multiple incoming edges
        multi_incoming = dict([(k,v) for k,v in incoming_edges_to_node.items() if len(v) >= 2])


        if len(multi_incoming) > 0:
            if verbose:
                print("There are loops to resolve and 'no_cycles' parameters set requires us to fix eliminate them")
            #find the mesh widths of all the incoming edges and the current edge

            #if mesh widths are available then go that route
            if not mesh_widths is None:
                if verbose:
                    print("Using mesh_widths for resolving loops")
                for curr_node,incoming_nodes in multi_incoming.items():
                    curr_node_width = mesh_widths[curr_node]
                    incoming_nodes_width_difference = [np.linalg.norm(mesh_widths[k]- curr_node_width) for k in incoming_nodes]
                    winning_incoming_node = incoming_nodes[np.argmin(incoming_nodes_width_difference).astype("int")]
                    incoming_edges_to_node[curr_node] = [winning_incoming_node]
            else: #if not mesh widths available then just pick the longest edge
                """
                Get the coordinates of all of the nodes
                """
                node_coordinates_dict = xu.get_node_attributes(curr_limb_concept_network,attribute_name="coordinates",return_array=False)
                if set(list(node_coordinates_dict.keys())) != set(list(incoming_edges_to_node.keys())):
                    if verbose:
                        print("The keys of the concept graph with 'coordinates' do not match the keys of the edge dictionary")
                        print("Just going to use the first incoming edge by default")
                    for curr_node,incoming_nodes in multi_incoming.items():
                        winning_incoming_node = incoming_nodes[0]
                        incoming_edges_to_node[curr_node] = [winning_incoming_node]
                else: #then have coordinate information
                    if verbose:
                        print("Using coordinate distance to pick the winning node")
                    curr_node_coordinate = node_coordinates_dict[curr_node]
                    incoming_nodes_distance = [np.linalg.norm(node_coordinates_dict[k]- curr_node_coordinate) for k in incoming_nodes]
                    winning_incoming_node = incoming_nodes[np.argmax(incoming_nodes_distance).astype("int")]
                    incoming_edges_to_node[curr_node] = [winning_incoming_node]
        else:
            if verbose:
                print("No cycles to fix")


        #check that all have length of 1
        multi_incoming = dict([(k,v) for k,v in incoming_edges_to_node.items() if len(v) == 1])
        
        if not suppress_disconnected_errors:
            if len(multi_incoming) != len(curr_limb_concept_network.nodes()) - 1:
                raise Exception("Inside the no_cycles but at the end all of the nodes only don't have one incoming cycle"
                               f"multi_incoming = {multi_incoming}")

    #7) convert the incoming edges dictionary to edge for a directional graph
    total_edges = []

    if no_cycles:
        for curr_node,incoming_nodes in multi_incoming.items():
            curr_incoming_edges = [(j,curr_node) for j in incoming_nodes]
            total_edges += curr_incoming_edges
    else:
        for curr_node,incoming_nodes in incoming_edges_to_node.items():
            curr_incoming_edges = [(j,curr_node) for j in incoming_nodes]
            total_edges += curr_incoming_edges


    #creating the directional network
    curr_limb_concept_network_directional = nx.DiGraph(nx.create_empty_copy(curr_limb_concept_network,with_data=True))
    curr_limb_concept_network_directional.add_edges_from(total_edges)

    return curr_limb_concept_network_directional


def branches_to_concept_network(curr_branch_skeletons,
                             starting_coordinate,
                              starting_edge,
                                touching_soma_vertices=None,
                                soma_group_idx=None,
                                starting_soma=None,
                             max_iterations= 1000000,
                               verbose=False):
    """
    Will change a list of branches into
    """
    # local import to break the neuron_utils <-> neuron module-load cycle (P1):
    # Branch is the only thing neuron_utils needs from neuron, and only here.
    from neurd.neuron import Branch

    if verbose:
        print(f"Starting_edge inside branches_to_conept = {starting_edge}")
    
    start_time = time.time()
    processed_nodes = []
    edge_endpoints_to_process = []
    concept_network_edges = []

    """
    If there is only one branch then just pass back a one-node graph 
    with no edges
    """
    if len(curr_branch_skeletons) == 0:
        raise Exception("Passed no branches to be turned into concept network")
    
    if len(curr_branch_skeletons) == 1:
        concept_network = xu.GraphOrderedEdges()
        concept_network.add_node(0)
        
        starting_node = 0
        #print("setting touching_soma_vertices 1")
        attrs = {starting_node:{"starting_coordinate":starting_coordinate,
                                "endpoints":Branch(starting_edge).endpoints,
                               "touching_soma_vertices":touching_soma_vertices,
                                "soma_group_idx":soma_group_idx,
                               "starting_soma":starting_soma}
                                }
        
        xu.set_node_attributes_dict(concept_network,attrs)
        #print(f"Recovered touching vertices after 1 = {xu.get_all_nodes_with_certain_attribute_key(concept_network,'touching_soma_vertices')}")
        
        #add the endpoints 
        return concept_network

    # 0) convert each branch to one segment and build a graph from it
    
    
    # 8-29 debug
    #curr_branch_meshes_downsampled = [sk.resize_skeleton_branch(b,n_segments=1) for b in curr_branch_skeletons]
    curr_branch_meshes_downsampled = []
    for i,b in enumerate(curr_branch_skeletons):
        try:
            curr_branch_meshes_downsampled.append(sk.resize_skeleton_branch(b,n_segments=1))
        except:
            if verbose:
                print(f"The following branch {i} could not be downsampled: {b}")
            raise Exception("not downsampled branch")
        
    
    """
    In order to solve the problem that once resized there could be repeat edges
    
    Pseudocode: 
    1) predict the branches that are repeats and then create a map 
    of the non-dom (to be replaced) and dominant (the ones to replace)
    2) Get an arange list of the branch idxs and then delete the non-dominant ones
    3) Run the whole concept map process
    4) At the end for each non-dominant one, at it in (with it's idx) and copy
    the edges of the dominant one that it was mapped to
    
    
    """
    
    downsampled_skeleton = sk.stack_skeletons(curr_branch_meshes_downsampled)
    # curr_sk_graph_debug = sk.convert_skeleton_to_graph_old(downsampled_skeleton)
    # nx.draw(curr_sk_graph_debug,with_labels = True)

    #See if touching row matches the original: 
    

    all_skeleton_vertices = downsampled_skeleton.reshape(-1,3)
    unique_rows,indices = np.unique(all_skeleton_vertices,return_inverse=True,axis=0)
    
    reshaped_indices = np.sort(indices.reshape(-1,2),axis=1)
    unique_edges,unique_edges_indices = np.unique(reshaped_indices,axis = 0,return_inverse=True)
    from collections import Counter
    multiplicity_edge_counter = dict(Counter(unique_edges_indices))
    #this will give the unique edge that appears multiple times
    duplicate_edge_identifiers = [k for k,v in multiplicity_edge_counter.items() if v > 1] 
    
    #for keeping track of original indexes
    original_idxs = np.arange(0,len(curr_branch_meshes_downsampled))
    
    """
    This will delete any branches that have the same two common endpoints
    """
    if len(duplicate_edge_identifiers) > 0:
        if verbose:
            print(f"There were {len(duplicate_edge_identifiers)} duplication nodes found")
        all_conn_comp = []
        for d in duplicate_edge_identifiers:
            all_conn_comp.append(list(np.where(unique_edges_indices == [d] )[0]))

        domination_map = dict()
        for curr_comp in all_conn_comp:
            dom_node = curr_comp[0]
            non_dom_nodes = curr_comp[1:]
            for n_dom in non_dom_nodes:
                domination_map[n_dom] = dom_node
        if verbose:
            print(f"domination_map = {domination_map}")
        

        to_delete_rows = list(domination_map.keys())

        #delete all of the non dominant rows from the indexes and the skeletons
        original_idxs = np.delete(original_idxs,to_delete_rows,axis=0)
        curr_branch_meshes_downsampled = [k for i,k in enumerate(curr_branch_meshes_downsampled) if i not in to_delete_rows]
    
    #print(f"curr_branch_meshes_downsampled[24] = {curr_branch_meshes_downsampled[24]}")
    curr_stacked_skeleton = sk.stack_skeletons(curr_branch_meshes_downsampled)
    #print(f"curr_stacked_skeleton[24] = {curr_stacked_skeleton[24]}")

    branches_graph = sk.convert_skeleton_to_graph(curr_stacked_skeleton) #can recover the original skeleton
#     print(f"len(curr_stacked_skeleton) = {len(curr_stacked_skeleton)}")
#     print(f"len(branches_graph.edges_ordered()) = {len(branches_graph.edges_ordered())}")
#     print(f"(branches_graph.edges_ordered())[24] = {(branches_graph.edges_ordered())[24]}")
#     print(f"coordinates = (branches_graph.edges_ordered())[24] = {xu.get_node_attributes(branches_graph,node_list=(branches_graph.edges_ordered())[24])}")


    #************************ need to just make an edges lookup dictionary*********#


    #1) Identify the starting node on the starting branch
    starting_node = xu.get_nodes_with_attributes_dict(branches_graph,dict(coordinates=starting_coordinate))
    
    if verbose:
        print(f"At the start, starting_node (in terms of the skeleton, that shouldn't match the starting edge) = {starting_node}")
    if len(starting_node) != 1:
        raise Exception(f"The number of starting nodes found was not exactly one: {starting_node}")
    #1b) Add all edges incident and their other node label to a list to check (add the first node to processed nodes list)
    incident_edges = xu.node_to_edges(branches_graph,starting_node)
    #print(f"incident_edges = {incident_edges}")
    # #incident_edges_idx = edge_to_index(incident_edges)

    # #adding them to the list to be processed (gets the edge and the downstream edge)
    edge_endpoints_to_process = [(edges,edges[edges != starting_node ]) for edges in incident_edges]
    processed_nodes.append(starting_node)

    #need to add all of the newly to look edges and the current edge to the concept_network_edges
    """
    Pseudocode: 
    1) convert starting edge to the node identifiers
    2) iterate through all the edges to process and add the combos where the edge does not match
    """
    edge_coeff= []
    for k in starting_edge:
        edge_coeff.append(xu.get_nodes_with_attributes_dict(branches_graph,dict(coordinates=k))[0])
    
    
    for curr_edge,edge_enpt in edge_endpoints_to_process:
        if not np.array_equal(np.sort(curr_edge),np.sort(edge_coeff)):
            #add to the concept graph
            concept_network_edges += [(np.array(curr_edge),np.array(edge_coeff))]
        else:
            starting_node_edge = curr_edge
            if verbose:
                print("printing out current edge:")
                print(xu.get_node_attributes(branches_graph,node_list=starting_node_edge))
        
    
    for i in range(max_iterations):
        #print(f"==\n\n On iteration {i}==")
        if len(edge_endpoints_to_process) == 0:
            if verbose:
                print(f"edge_endpoints_to_process was empty so exiting loop after {i} iterations")
            break

        #2) Pop the edge edge number,endpoint of the stack
        edge,endpt = edge_endpoints_to_process.pop(0)
        #print(f"edge,endpt = {(edge,endpt)}")
        #- if edge already been processed then continue
        if endpt in processed_nodes:
            #print(f"Already processed endpt = {endpt} so skipping")
            continue
        #a. Find all edges incident on this node
        incident_edges = xu.node_to_edges(branches_graph,endpt)
        #print(f"incident_edges = {incident_edges}")

        considering_edges = [k for k in incident_edges if not np.array_equal(k,edge) and not np.array_equal(k,np.flip(edge))]
        #print(f"considering_edges = {considering_edges}")
        #b. Create edges from curent edge to those edges incident with it
        concept_network_edges += [(edge,k) for k in considering_edges]

        #c. Add the current node as processed
        processed_nodes.append(endpt)

        #d. For each edge incident add the edge and the other connecting node to the list
        new_edge_processing = [(e,e[e != endpt ]) for e in considering_edges]
        edge_endpoints_to_process = edge_endpoints_to_process + new_edge_processing
        #print(f"edge_endpoints_to_process = {edge_endpoints_to_process}")

    if len(edge_endpoints_to_process)>0:
        raise Exception(f"Reached max_interations of {max_iterations} and the edge_endpoints_to_process not empty")

    #flattening the connections so we can get the indexes of these edges
    flattened_connections = np.array(concept_network_edges).reshape(-1,2)
    
    orders = xu.get_edge_attributes(branches_graph,edge_list=flattened_connections)
    #******
    
    fixed_idx_orders = original_idxs[orders]
    concept_network_edges_fixed = np.array(fixed_idx_orders).reshape(-1,2)

    
    # # edge_endpoints_to_process
    #print(f"concept_network_edges_fixed = {concept_network_edges_fixed}")
    concept_network = xu.GraphOrderedEdges()
    #print("type(concept_network) = {type(concept_network)}")
    concept_network.add_edges_from([k for k in concept_network_edges_fixed])
    
    #add the endpoints as attributes to each of the nodes
    node_endpoints_dict = dict()
    old_ordered_edges = branches_graph.edges_ordered()
    for edge_idx,curr_branch_graph_edge in enumerate(old_ordered_edges):
        new_edge_idx = original_idxs[edge_idx]
        curr_enpoints = np.array(xu.get_node_attributes(branches_graph,node_list=curr_branch_graph_edge)).reshape(-1,3)
        node_endpoints_dict[new_edge_idx] = dict(endpoints=curr_enpoints)
        xu.set_node_attributes_dict(concept_network,node_endpoints_dict)
    
    
    
    
    #add the starting coordinate to the corresponding node
    #print(f"starting_node_edge right before = {starting_node_edge}")
    starting_order = xu.get_edge_attributes(branches_graph,edge_list=[starting_node_edge]) 
    #print(f"starting_order right before = {starting_order}")
    if len(starting_order) != 1:
        raise Exception(f"Only one starting edge index was not found,starting_order={starting_order} ")
    
    starting_edge_index = original_idxs[starting_order[0]]
    if verbose:
        print(f"starting_node in concept map (that should match the starting edge) = {starting_edge_index}")
    #attrs = {starting_node[0]:{"starting_coordinate":starting_coordinate}} #old way that think uses the wrong starting_node
    attrs = {starting_edge_index:{"starting_coordinate":starting_coordinate,"touching_soma_vertices":touching_soma_vertices,"soma_group_idx":soma_group_idx,"starting_soma":starting_soma}} 
    #print("setting touching_soma_vertices 2")
    xu.set_node_attributes_dict(concept_network,attrs)
    #print(f"Recovered touching vertices after 2 = {xu.get_all_nodes_with_certain_attribute_key(concept_network,'touching_soma_vertices')}")
    
    #want to set all of the edge endpoints on the nodes as well just for a check
    
    
    if verbose:
        print(f"Total time for branches to concept conversion = {time.time() - start_time}\n")
    
    
    # Add back the nodes that were deleted
    if len(duplicate_edge_identifiers) > 0:
        if verbose:
            print("Working on adding back the edges that were duplicates")
        for non_dom,dom in domination_map.items():
            #print(f"Re-adding: {non_dom}")
            #get the endpoints attribute
            # local_node_endpoints_dict
            
            curr_neighbors = xu.get_neighbors(concept_network,dom)  
            new_edges = np.vstack([np.ones(len(curr_neighbors))*non_dom,curr_neighbors]).T
            concept_network.add_edges_from(new_edges)
            
            curr_endpoint = xu.get_node_attributes(concept_network,attribute_name="endpoints",node_list=[dom])[0]
            #print(f"curr_endpoint in add back = {curr_endpoint}")
            add_back_attribute_dict = {non_dom:dict(endpoints=curr_endpoint)}
            #print(f"To add dict = {add_back_attribute_dict}")
            xu.set_node_attributes_dict(concept_network,add_back_attribute_dict)
            
    return concept_network


# --------------  END OF COMPRESSION OF NEURON ---------------- #

# --------------  7/23 To help with visualizations of neuron ---------------- #





def limb_label(name_input,force_int=True):
    if type(name_input) == str:
        return name_input
    if force_int:
        name_input = int(name_input)
    if type(name_input) == int or type(name_input) == float:
        return f"L{int(name_input)}"
    else:
        raise Exception(f"Recieved unexpected type ({type(name_input)}) for limb name")

def limb_idx(name_input):
    if "str" in str(type(name_input)):
        return int(name_input[1:])
    elif type(name_input) == int or type(name_input) == float:
        return int(name_input)
    else:
        raise Exception(f"Recieved unexpected type ({type(name_input)}) for limb name")
    
    
# --------------- 8/5 --------------------------#
def branch_mesh_no_spines(branch):
    """
    Purpose: To return the branch mesh without any spines
    """
    original_mesh_flag = False
    if not branch.spines is None:
        if len(branch.spines) > 0:
            ex_branch_no_spines_mesh = tu.original_mesh_faces_map(branch.mesh,
                                    tu.combine_meshes(branch.spines),
                                   matching=False,
                                   print_flag=False,
                                   match_threshold = 0.001,
                                                            return_mesh=True,
                                                                 )
        else:
            original_mesh_flag = True
    else: 
        original_mesh_flag = True
    
    if original_mesh_flag:
        ex_branch_no_spines_mesh = branch.mesh
        
    return ex_branch_no_spines_mesh

#xu.endpoint_connectivity(end_1,end_2)


# ---------------------- 8/31: For querying and axon searching --------------------------- #
    
# --- 9/2: Mesh correspondence that helps deal with the meshparty data  ----
def sdf_filter(curr_branch,curr_limb,size_threshold=20,
               return_sdf_mean=False,
               ray_inter=None,
              n_std_dev = 1):
    """
    Purpose: to eliminate edge parts of meshes that should
    not be on the branch mesh correspondence
    
    Pseudocode
    The filtering step (Have a size threshold for this maybe?):
    1) Calculate the sdf values for all parts of the mesh
    2) Restrict the faces to only thos under mean + 1.5*std_dev
    3) split the mesh and only keep the biggest one

    Example: 
    
    limb_idx = 0
    branch_idx = 20
    branch_idx = 3
    #branch_idx=36
    filtered_branch_mesh, filtered_branch_mesh_idx = sdf_filter(double_neuron_processed[limb_idx][branch_idx],double_neuron_processed[limb_idx],
                                                               n_std_dev=1)
    filtered_branch_mesh.show()

    """
    

    
    #1) Calculate the sdf values for all parts of the mesh
    ray_trace_width_array = tu.ray_trace_distance(curr_limb.mesh,face_inds=curr_branch.mesh_face_idx,ray_inter=ray_inter)
    ray_trace_width_array_mean = np.mean(ray_trace_width_array[ray_trace_width_array>0])
    #apply the size threshold
    if len(curr_branch.mesh.faces)<20:
        if return_sdf_mean:
            return curr_branch.mesh,np.arange(0,len(curr_branch.mesh_face_idx)),ray_trace_width_array_mean
        else:
            return curr_branch.mesh,np.arange(0,len(curr_branch.mesh_face_idx))
    
    #2) Restrict the faces to only thos under mean + 1.5*std_dev
    ray_trace_mask = ray_trace_width_array < (ray_trace_width_array_mean + n_std_dev*np.std(ray_trace_width_array))
    filtered_mesh = curr_limb.mesh.submesh([curr_branch.mesh_face_idx[ray_trace_mask]],append=True,repair=False)

    
    #3) split the mesh and only keep the biggest one
    filtered_split_meshes, filtered_split_meshes_idx = tu.split(filtered_mesh)
    
    if return_sdf_mean:
        return filtered_split_meshes[0],filtered_split_meshes_idx[0],ray_trace_width_array_mean
    else:
        return filtered_split_meshes[0],filtered_split_meshes_idx[0]
    

# ------------------------------ 9/1 To help with mesh correspondence -----------------------------------------------------#


def apply_adaptive_mesh_correspondence_to_neuron(current_neuron,
                                                apply_sdf_filter=False,
                                                n_std_dev=1):

    
    for limb_idx in np.sort(current_neuron.get_limb_node_names()):
        
        ex_limb = current_neuron.concept_network.nodes[limb_idx]["data"]
        if apply_sdf_filter:
            print("Using SDF filter")
            ray_inter = ray_pyembree.RayMeshIntersector(ex_limb.mesh)
            
        
        segment_mesh_faces = dict()
        for branch_idx in np.sort(ex_limb.concept_network.nodes()):
            print(f"---- Working on limb {limb_idx} branch {branch_idx} ------")
            ex_branch = ex_limb.concept_network.nodes[branch_idx]["data"]

            #1) get all the neighbors 1 hop away in connectivity
            #2) Assemble a mesh of all the surrounding neighbors
            one_hop_neighbors = xu.get_neighbors(ex_limb.concept_network,branch_idx)
            if len(one_hop_neighbors) > 0:
                two_hop_neighbors = np.concatenate([xu.get_neighbors(ex_limb.concept_network,k) for k in one_hop_neighbors])
                branches_for_surround = np.unique([branch_idx] + list(one_hop_neighbors) + list(two_hop_neighbors))


                surround_mesh_faces = np.concatenate([ex_limb.concept_network.nodes[k]["data"].mesh_face_idx for k in branches_for_surround])
                surrounding_mesh = ex_limb.mesh.submesh([surround_mesh_faces],append=True,repair=False)

                #3) Send the skeleton and the surrounding mesh to the mesh adaptive distance --> gets back indices
                return_value = cu.mesh_correspondence_adaptive_distance(curr_branch_skeleton=ex_branch.skeleton,
                                                     curr_branch_mesh=surrounding_mesh)

                if len(return_value) == 2:
                    remaining_indices, width = return_value
                    final_limb_indices = surround_mesh_faces[remaining_indices]
                else: #if mesh correspondence couldn't be found
                    print("Mesh correspondence couldn't be found so using defaults")
                    final_limb_indices = ex_branch.mesh_face_idx
                    width = ex_branch.width

            else: #if mesh correspondence couldn't be found
                print("Mesh correspondence couldn't be found so using defaults")
                final_limb_indices = ex_branch.mesh_face_idx
                width = ex_branch.width



            """  How we would get the final mesh  
            branch_mesh_filtered = ex_limb.mesh.submesh([final_limb_indices],append=True,repair=False) 

            """
            #5b) store the width measurement based back in the mesh object
            ex_branch.width_new["adaptive"] = width

            if apply_sdf_filter:
                #---------- New step:  Further filter the limb indices
                new_branch_mesh = ex_limb.mesh.submesh([final_limb_indices],append=True,repair=False)
                new_branch_obj = copy.deepcopy(ex_branch)
                new_branch_obj.mesh = new_branch_mesh
                new_branch_obj.mesh_face_idx = final_limb_indices

                filtered_branch_mesh,filtered_branch_mesh_idx,filtered_branch_sdf_mean= sdf_filter(curr_branch=new_branch_obj,
                                                                                                       curr_limb=ex_limb,
                                                                                                       return_sdf_mean=True,
                                                                                                       ray_inter=ray_inter,
                                                                                                      n_std_dev=n_std_dev)
                final_limb_indices = final_limb_indices[filtered_branch_mesh_idx]
            
            segment_mesh_faces[branch_idx] = final_limb_indices

        #This ends up fixing any conflicts in labeling
        face_lookup = invert_mapping(segment_mesh_faces,total_keys=np.arange(0,len(ex_limb.mesh.faces)))
        #original_labels = set(list(itertools.chain.from_iterable(list(face_lookup.values()))))
        #original_labels = gu.get_unique_values_dict_of_lists(face_lookup)
        original_labels = np.arange(0,len(ex_limb))

        face_coloring_copy = cu.resolve_empty_conflicting_face_labels(curr_limb_mesh = ex_limb.mesh,
                                                                                        face_lookup=face_lookup,
                                                                                        no_missing_labels = list(original_labels),
                                                                     max_submesh_threshold=50000)

        divided_submeshes,divided_submeshes_idx = tu.split_mesh_into_face_groups(ex_limb.mesh,face_coloring_copy)

        #now reassign the new divided supmeshes
        for branch_idx in ex_limb.concept_network.nodes():
            ex_branch = ex_limb.concept_network.nodes[branch_idx]["data"]

            ex_branch.mesh = divided_submeshes[branch_idx]
            ex_branch.mesh_face_idx = divided_submeshes_idx[branch_idx]
            ex_branch.mesh_center = tu.mesh_center_vertex_average(ex_branch.mesh)
            
            #need to change the preprocessed_data to reflect the change
            limb_idx_used = int(limb_idx[1:])
            current_neuron.preprocessed_data["limb_correspondence"][limb_idx_used][branch_idx]["branch_mesh"] = ex_branch.mesh 
            current_neuron.preprocessed_data["limb_correspondence"][limb_idx_used][branch_idx]["branch_face_idx"] = ex_branch.mesh_face_idx
            
            

    
# --------- 9/9 Helps with splitting the mesh limbs ------------ #




# ---- 11/20 functions that will help compute statistics of the neuron object -----------





def n_somas(neuron_obj):
    return len(neuron_obj.get_soma_node_names())

def n_limbs(neuron_obj):
    return len(neuron_obj.get_limb_node_names())

def n_branches_per_limb(neuron_obj):
    return [len(ex_limb.get_branch_names()) for ex_limb in neuron_obj]

def n_branches(neuron_obj):
    return np.sum(neuron_obj.n_branches_per_limb)

def skeleton_length_per_limb(neuron_obj):
    return [sk.calculate_skeleton_distance(limb.skeleton) for limb in neuron_obj]

def skeletal_length(neuron_obj):
    return np.sum(neuron_obj.skeleton_length_per_limb)


def max_limb_n_branches(neuron_obj):
    if len(neuron_obj.n_branches_per_limb)>0:
        return np.max(neuron_obj.n_branches_per_limb)
    else:
        return None

def max_limb_skeletal_length(neuron_obj):
    if len(neuron_obj.skeleton_length_per_limb) > 0:
        return np.max(neuron_obj.skeleton_length_per_limb)
    else:
        return None

def all_skeletal_lengths(neuron_obj):
    all_skeletal_lengths = []
    for curr_limb in neuron_obj:
        for curr_branch in curr_limb:
            curr_branch_sk_len = sk.calculate_skeleton_distance(curr_branch.skeleton)
            all_skeletal_lengths.append(curr_branch_sk_len)
    return np.array(all_skeletal_lengths)

def median_branch_length(neuron_obj):
    if len(all_skeletal_lengths(neuron_obj))>0:
        return np.round(np.median(all_skeletal_lengths(neuron_obj)),3)
    else:
        return None
    

# -- width data --
def all_medain_mesh_center_widths(neuron_obj):
    all_widths = []
    for curr_limb in neuron_obj:
        for curr_branch in curr_limb:
            curr_width = curr_branch.width_new["median_mesh_center"]
            if curr_width < np.inf:
                all_widths.append(curr_width)
    return np.array(all_widths)

def all_no_spine_median_mesh_center_widths(neuron_obj):
    all_widths = []
    for curr_limb in neuron_obj:
        for curr_branch in curr_limb:
            curr_width = curr_branch.width_new["no_spine_median_mesh_center"]
            if curr_width < np.inf:
                all_widths.append(curr_width)
    return np.array(all_widths)

def width_median(neuron_obj):
    if len(all_medain_mesh_center_widths(neuron_obj)) > 0:
        return np.round(np.median(all_medain_mesh_center_widths(neuron_obj)),3)
    else:
        return None

def width_no_spine_median(neuron_obj):
    if len(all_no_spine_median_mesh_center_widths(neuron_obj)) > 0:
        return np.round(np.median(all_no_spine_median_mesh_center_widths(neuron_obj)),3)
    else:
        return None

def width_perc(neuron_obj,perc=90):
    if len(all_medain_mesh_center_widths(neuron_obj)) > 0:
        return np.round(np.percentile(all_medain_mesh_center_widths(neuron_obj),perc),3)
    else:
        return None

def width_no_spine_perc(neuron_obj,perc=90):
    if len(all_no_spine_median_mesh_center_widths(neuron_obj)) > 0:
        return np.round(np.percentile(all_no_spine_median_mesh_center_widths(neuron_obj),perc),3)
    else:
        return None



# -- spine data --

def calculate_spines_skeletal_length(neuron_obj):
    if neuron_obj.spines is None:
        return None
    sk_len_array = []
    for k in neuron_obj.spines:
        try:
            curr_skeletal_length = sk.calculate_skeleton_distance(sk.surface_skeleton(k))
        except:
            curr_skeletal_length = 0 

        sk_len_array.append(curr_skeletal_length)
    neuron_obj.spines_skeletal_length = np.array(sk_len_array)
    return sk_len_array
    

def n_spines(neuron_obj,skeletal_length_max=None):
    if skeletal_length_max is None:
        from neurd import parameters
        skeletal_length_max = parameters.params.skeletal_length_max_n_spines
    if neuron_obj.spines is None:
        return 0
    else:
        if skeletal_length_max is not None:
            if not hasattr(neuron_obj,"spines_skeletal_length")  or len(neuron_obj.spines_skeletal_length) != len(neuron_obj.spines):
#                 sk_len_array = []
#                 for k in neuron_obj.spines:
#                     try:
#                         curr_skeletal_length = sk.calculate_skeleton_distance(sk.surface_skeleton(k))
#                     except:
#                         curr_skeletal_length = 0 
                        
#                     sk_len_array.append(curr_skeletal_length)
#                 neuron_obj.spines_skeletal_length = np.array(sk_len_array)
                
                neuron_obj.spines_skeletal_length = calculate_spines_skeletal_length(neuron_obj)
                                                                            
                #neuron_obj.spines_skeletal_length = np.array([sk.calculate_skeleton_distance(sk.surface_skeleton(k)) for k in neuron_obj.spines])
            valid_spine_idx = np.where(np.array(neuron_obj.spines_skeletal_length)<skeletal_length_max)[0]
            return len(valid_spine_idx)
        else:
            return len(neuron_obj.spines)
    
def n_boutons(neuron_obj):
    if neuron_obj.boutons is None:
        return 0
    else:
        return len(neuron_obj.boutons)
    
def n_web(neuron_obj):
    if neuron_obj.web is None:
        return 0
    else:
        return 1
    
def compute_mesh_attribute_volume(branch_obj,
                                  mesh_attribute,
                                 max_hole_size=2000,
                                 self_itersect_faces=False):
    if getattr(branch_obj,mesh_attribute) is None:
        setattr(branch_obj,f"{mesh_attribute}_volume",None)
    else:
        vol_list = [tu.mesh_volume(sp,verbose=False) for sp in
                   getattr(branch_obj,mesh_attribute)]
        setattr(branch_obj,f"{mesh_attribute}_volume",vol_list)

def feature_list_over_object(obj,
                            feature_name):
    """
    Purpose: Will compile a list of all of the 
    """
    obj._index = -1
    total_feature = []
    for b in obj:
        if not getattr(b,feature_name) is None:
            total_feature += list(getattr(b,feature_name))
    return total_feature

def compute_feature_over_object(obj,
                               feature_name):
    obj._index = -1
    for b in obj:
        getattr(b,f"compute_{feature_name}")()
    
def spine_density(neuron_obj):
    skeletal_length = neuron_obj.skeletal_length
    if skeletal_length > 0:
        spine_density = neuron_obj.n_spines/skeletal_length
    else:
        spine_density = 0
    return spine_density

def spines_per_branch(neuron_obj):
    if neuron_obj.n_branches > 0:
        spines_per_branch = neuron_obj.n_spines/neuron_obj.n_branches
    else:
        spines_per_branch = 0
    return spines_per_branch
    
def n_spine_eligible_branches(neuron_obj):
    n_spine_eligible_branches = 0
    for curr_limb in neuron_obj:
        for curr_branch in curr_limb:
            if not curr_branch.spines is None:
                n_spine_eligible_branches += 1
    return n_spine_eligible_branches

def spine_eligible_branch_lengths(neuron_obj):
    spine_eligible_branch_lengths = []
    for curr_limb in neuron_obj:
        for curr_branch in curr_limb:
            if not curr_branch.spines is None:
                curr_branch_sk_len = sk.calculate_skeleton_distance(curr_branch.skeleton)
                spine_eligible_branch_lengths.append(curr_branch_sk_len)
    return spine_eligible_branch_lengths

def skeletal_length_eligible(neuron_obj):
    return np.round(np.sum(neuron_obj.spine_eligible_branch_lengths),3)

def spine_density_eligible(neuron_obj):
    #spine eligible density and per branch
    if neuron_obj.skeletal_length_eligible > 0:
        spine_density_eligible = neuron_obj.n_spines/neuron_obj.skeletal_length_eligible
    else:
        spine_density_eligible = 0
    
    return spine_density_eligible

def spines_per_branch_eligible(neuron_obj):
    if neuron_obj.n_spine_eligible_branches > 0:
        spines_per_branch_eligible = np.round(neuron_obj.n_spines/neuron_obj.n_spine_eligible_branches,3)
    else:
        spines_per_branch_eligible = 0
    
    return spines_per_branch_eligible


# ------- all the spine volume stuff -----------
def total_spine_volume(neuron_obj):
    if neuron_obj.n_spines > 0:
        spines_vol = np.array(neuron_obj.spines_volume)
        return np.sum(spines_vol)
        
    else:
        return 0

def spine_volume_median(neuron_obj):
    spines_vol = np.array(neuron_obj.spines_volume)
    if neuron_obj.n_spines > 0:
        #spine_volume_median
        valid_spine_vol = spines_vol[spines_vol>0]

        if len(valid_spine_vol) > 0:
            spine_volume_median = np.median(valid_spine_vol)
        else:
            spine_volume_median = 0
        
        return spine_volume_median
        
    else:
        return 0
    
def spine_volume_density(neuron_obj):
    if neuron_obj.n_spines > 0:
        if neuron_obj.skeletal_length_eligible > 0:
            spine_volume_density_eligible = neuron_obj.total_spine_volume/neuron_obj.skeletal_length
        else:
            spine_volume_density_eligible = 0
        
        return spine_volume_density_eligible
        
    else:
        return 0


def spine_volume_density_eligible(neuron_obj):
    if neuron_obj.n_spines > 0:
        if neuron_obj.skeletal_length > 0:
            spine_volume_density = neuron_obj.total_spine_volume/neuron_obj.skeletal_length_eligible
        else:
            spine_volume_density = 0
        
        return spine_volume_density
        
    else:
        return 0
    
def spine_volume_per_branch_eligible(neuron_obj):
    if neuron_obj.n_spines > 0:
        if neuron_obj.n_spine_eligible_branches > 0:
            spine_volume_per_branch_eligible = neuron_obj.total_spine_volume/neuron_obj.n_spine_eligible_branches
        else:
            spine_volume_per_branch_eligible = 0
        
        return spine_volume_per_branch_eligible
        
    else:
        return 0
    
    
# -------------- 11 / 26 To help with erroring------------------------------#


def align_and_restrict_branch(base_branch,
                              common_endpoint=None,
                              width_name= "no_spine_median_mesh_center",
                              width_name_backup= "no_spine_median_mesh_center",
                             offset=500,
                             comparison_distance=2000,
                             skeleton_segment_size=1000,
                              verbose=False,
                             ):
    
    if width_name not in base_branch.width_array.keys():
        width_name = width_name_backup
        
    #Now just need to do the resizing (and so the widths calculated will match this)
    base_skeleton_ordered = sk.resize_skeleton_branch(base_branch.skeleton,skeleton_segment_size)

    if not common_endpoint is None:
        #figure out if need to flip or not:
        if np.array_equal(common_endpoint,base_skeleton_ordered[-1][-1]):

            base_width_ordered = np.flip(base_branch.width_array[width_name])
            base_skeleton_ordered = sk.flip_skeleton(base_skeleton_ordered)
            flip_flag = True
            if verbose:
                print("Base needs flipping")
                print(f"Skeleton after flip = {base_skeleton_ordered}")
        elif np.array_equal(common_endpoint,base_skeleton_ordered[0][0]):
            base_width_ordered = base_branch.width_array[width_name]
            flip_flag = False
        else:
            raise Exception("No matching endpoint")
    else:
        base_width_ordered = base_branch.width_array[width_name]
        
    # apply the cutoff distance
    if verbose:
        print(f"Base offset = {offset}")
        
    
    (skeleton_minus_buffer,
     offset_indexes,
     offset_success) = sk.restrict_skeleton_from_start(base_skeleton_ordered,
                                                                    offset,
                                                                     subtract_cutoff=True)
   
    
    base_final_skeleton = None
    base_final_indexes = None

    if offset_success:
        
        (skeleton_comparison,
         comparison_indexes,
         comparison_success) = sk.restrict_skeleton_from_start(skeleton_minus_buffer,
                                                                        comparison_distance,
                                                                         subtract_cutoff=False)
        
        if comparison_success:
            if verbose:
                print("Base: Long enough for offset and comparison length")
            base_final_skeleton = skeleton_comparison
            base_final_indexes = offset_indexes[comparison_indexes]

        else:
            if verbose:
                print("Base: Passed the offset phase but was not long enough for comparison")
    else:
        if verbose:
            print("Base: Was not long enough for offset")


    if base_final_skeleton is None:
        if verbose:
            print("Base: Not using offset ")
        (base_final_skeleton,
         base_final_indexes,
         _) = sk.restrict_skeleton_from_start(base_skeleton_ordered,
                                                                        comparison_distance,
                                                                         subtract_cutoff=False)
        

    
    base_final_widths = base_width_ordered[np.clip(base_final_indexes,0,len(base_width_ordered)-1)]
    base_final_seg_lengths = sk.calculate_skeleton_segment_distances(base_final_skeleton,cumsum=False)
    
    return base_final_skeleton,base_final_widths,base_final_seg_lengths

    
    
    

global_comparison_distance = 3000



    

            
    

    
    



def get_limb_string_name(limb_idx,start_letter="L"):
    if limb_idx is None:
        return None
    if type(limb_idx) == int or "int" in str(type(limb_idx)) or "float" in str(type(limb_idx)):
        return f"{start_letter}{limb_idx}" 
    elif type(limb_idx) == str or "str" in str(type(limb_idx)):
        return limb_idx
    else:
        raise Exception("Not int or string input")
        
def get_limb_int_name(limb_name):
    if limb_name is None:
        return None
    if type(limb_name) == int:
        return limb_name
    elif type(limb_name) == str:
        return int(limb_name[1:])
    else:
        raise Exception("Not int or string input")
        
        
def get_soma_int_name(soma_name):
    limb_name = soma_name
    if limb_name is None:
        return None
    if type(limb_name) == int:
        return limb_name
    elif type(limb_name) == str:
        return int(limb_name[1:])
    else:
        raise Exception("Not int or string input")
        



 
    
    
    
    
def all_soma_connnecting_endpionts_from_starting_info(starting_info):
    all_endpoints = []
    try:
        for limb_idx,limb_start_v in starting_info.items():
            for soma_idx,soma_v in limb_start_v.items():
                for soma_group_idx,group_v in soma_v.items():
                    all_endpoints.append(group_v["endpoint"])
    except:
        for soma_idx,soma_v in starting_info.items():
            for soma_group_idx,group_v in soma_v.items():
                all_endpoints.append(group_v["endpoint"])
        
    if len(all_endpoints) > 0:
        all_endpoints = np.unique(np.vstack(all_endpoints),axis=0)
    return all_endpoints
    
    

    
    
def get_matching_concept_network_data(limb_obj,soma_idx=None,soma_group_idx=None,
                                     starting_node=None,
                                     verbose=False):
    
    if type(soma_idx) == str:
        soma_idx = int(soma_idx[1:])
    
    if soma_idx is None and (soma_group_idx is None) and starting_node is None:
        raise Exception("All soma, soma_group and starting node descriptions are None")
        
    matching_concept_network_dicts_idx = np.arange(len(limb_obj.all_concept_network_data))
  
    if soma_idx is not None:
        soma_matches = np.array([i for i,k in enumerate(limb_obj.all_concept_network_data) if k["starting_soma"] == soma_idx])
        matching_concept_network_dicts_idx = np.intersect1d(matching_concept_network_dicts_idx,soma_matches)
        
    if soma_group_idx is not None:
        soma_matches = np.array([i for i,k in enumerate(limb_obj.all_concept_network_data) if k["soma_group_idx"] == soma_group_idx])
        matching_concept_network_dicts_idx = np.intersect1d(matching_concept_network_dicts_idx,soma_matches)
        
    if starting_node is not None:
        soma_matches = np.array([i for i,k in enumerate(limb_obj.all_concept_network_data) if k["starting_node"] == starting_node])
        matching_concept_network_dicts_idx = np.intersect1d(matching_concept_network_dicts_idx,soma_matches)
        
    if verbose:
        print(f"matching_concept_network_dicts_idx = {matching_concept_network_dicts_idx}")
        
    return [limb_obj.all_concept_network_data[k] for k in matching_concept_network_dicts_idx]
    
    
    
# ----------- 1/15: For Automatic Axon and Apical Classification ---------------#
                    

            
    
    
    
    
def skeletal_distance_from_soma(curr_limb,
                    limb_name = None,
                    somas = None,
                    error_if_all_nodes_not_return=True,
                    include_node_skeleton_dist=True,
                    print_flag = False,
                    branches = None,
                    **kwargs
                            
    ):

    """
    Purpose: To determine the skeletal distance away from 
    a soma a branch piece is
    
    Pseudocode: 
    0) Create dictionary that will store all of the results
    For each directional concept network
    1) Find the starting node
    For each node: 
    1)find the shortest path from starting node to that node
    2) convert the path into skeletal distance of each node 
    and then add up
    3) Map of each of distances to the node in a dictionary and return
    - replace a previous one if smaller
    
    Example: 
    skeletal_distance_from_soma(
                    limb_name = "L1"
                    curr_limb = uncompressed_neuron.concept_network.nodes[limb_name]["data"]
                    print_flag = True
                    #soma_list=None
                    somas = [0,1]
                    check_all_nodes_in_return=True
    )

    """
    if print_flag:
        print(f"\n\n------Working on Limb ({limb_name})-------")
        print(f"Starting nodes BEFORE copy = {xu.get_starting_node(curr_limb.concept_network,only_one=False)}")

    curr_limb_copy =  deepcopy(curr_limb)
    
    if print_flag:
        print(f"Starting nodes after copy = {xu.get_starting_node(curr_limb_copy.concept_network,only_one=False)}")

    #0) Create dictionary that will store all of the results
    return_dict = dict()

    #For each directional concept network
    if somas is None:
        touching_somas = [k["starting_soma"] for k in curr_limb_copy.all_concept_network_data]
    else:
        if not nu.is_array_like(somas):
            somas = [somas]
        touching_somas = somas

    if print_flag:
        print(f"Performing analysis for somas: {touching_somas}")

    
    nodes_to_process = curr_limb_copy.get_branch_names()
        
    for sm_start in touching_somas:
        #1) Find the starting node
        if print_flag:
            print(f"--> Working on soma {sm_start}")
        try:
            curr_limb_copy.set_concept_network_directional(sm_start)
        except:
            if print_flag:
                print(f"Limb ({limb_name}) was not connected to soma {sm_start} accordinag to all concept networks")
            continue
        curr_directional_network = curr_limb_copy.concept_network_directional
        starting_node = curr_limb_copy.current_starting_node
        
        if branches is not None:
            nodes_to_process = branches
        else:
            nodes_to_process = curr_directional_network.nodes()

        #For each node: 
        for n in nodes_to_process:
            #1)find the shortest path from starting node to that node
            #( could potentially not be there because it is directional)
            try:
                curr_shortest_path = nx.shortest_path(curr_directional_network,starting_node,n)
            except:
                #return_dict[n] = np.inf
                continue
            #2) convert the path into skeletal distance of each node and then add up
            if not include_node_skeleton_dist:
                path_length = np.sum([sk.calculate_skeleton_distance(curr_directional_network.nodes[k]["data"].skeleton)
                               for k in curr_shortest_path[:-1]])
            else:
                path_length = np.sum([sk.calculate_skeleton_distance(curr_directional_network.nodes[k]["data"].skeleton)
                               for k in curr_shortest_path])


            #3) Map of each of distances to the node in a dictionary and return
            #- replace a previous one if smaller

            if n in return_dict.keys():
                if path_length < return_dict[n]:
                    return_dict[n] = path_length
            else:
                return_dict[n] = path_length
    if print_flag:
        print(f"\nBefore Doing the dictionary correction, return_dict={return_dict}\n")
    #check that the return dict has all of the nodes
    for n in nodes_to_process:
        if n not in return_dict.keys():
            return_dict[n] = np.inf
   
    if error_if_all_nodes_not_return:
        #if set(list(return_dict.keys())) != set(list(curr_limb_copy.concept_network.nodes())):
        if set(list(return_dict.keys())) != set(list(nodes_to_process)):
            raise Exception("return_dict keys do not exactly match the curr limb nodes")
            
    return return_dict
    





    
def all_concept_network_data_to_limb_network_stating_info(all_concept_network_data):
    """
    Purpose: Will conver the concept network data list of dictionaries into a 
    the dictionary representation of only the limb touching vertices and
    endpoints of the limb_network_stating_info in the preprocessed data
    
    Pseudocode: 
    Iterate through all of the network dicts and store as
    soma--> soma_group_idx --> dict(touching_verts,
                                    endpoint)
                                    
    stored in the concept network as 
    touching_soma_vertices
    starting_coordinate
    
    """
    limb_network = dict()
    for k in all_concept_network_data:
        soma_idx = k["starting_soma"]
        soma_group_idx = k["soma_group_idx"]
        
        if soma_idx not in limb_network.keys():
            limb_network[soma_idx] = dict()
            
        limb_network[soma_idx][soma_group_idx] = dict(touching_verts=k["touching_soma_vertices"],
                                                     endpoint = k["starting_coordinate"])
        
    return limb_network
    

def clean_all_concept_network_data(all_concept_network_data,
                                  verbose=False):
    
    """
    Purpose: To make sure that there are
    no duplicate entries of that starting nodes
    and either to combine the soma touching points
    or just keep the largest one

    Pseudocode: 
    1) Start with an empty dictionary
    For all the dictionaries:
    2)  store the result
    indexed by starting soma and starting node
    3) If an entry already existent --> then either add the soma touching
    vertices (and unique) to the list or replace it if longer

    4) Turn the one dictionary into a list of dictionaries
    like the all_concept_network_data attribute

    5) Replace the all_concept_network_data


    """

    new_network_data = dict()

    for n_dict in all_concept_network_data:
        starting_soma = n_dict["starting_soma"]
        starting_node = n_dict["starting_node"]

        if starting_soma not in new_network_data.keys():
            new_network_data[starting_soma] = dict()

        if starting_node in new_network_data[starting_soma].keys():
            if (len(new_network_data[starting_soma][starting_node]["touching_soma_vertices"]) < 
                len(n_dict["touching_soma_vertices"])):
                if verbose:
                    print(f"Replacing the Soma_{starting_soma}_Node_{starting_node} dictionary")
                new_network_data[starting_soma][starting_node] = n_dict
            else:
                if verbose:
                    print(f"Skipping the Soma_{starting_soma}_Node_{starting_node} dictionary because smaller")
        else:
            new_network_data[starting_soma][starting_node] = n_dict

    #4) Turn the one dictionary into a list of dictionaries
    #like the all_concept_network_data attribute

    new_network_list = []
    for soma_idx,soma_info in new_network_data.items():
        for idx,(starting_node,node_info) in enumerate(soma_info.items()):
            node_info["soma_group_idx"] = idx
            new_network_list.append(node_info)

    return new_network_list



def clean_neuron_all_concept_network_data(neuron_obj,verbose=False):
    """
    Will go through and clean all of the concept network data
    in all the limbs of a Neuron
    """
    for j,curr_limb in enumerate(neuron_obj):
        if verbose:
            print(f"\n\n---- Working on Limb {j} ----")
            
            
        cleaned_network = clean_all_concept_network_data(curr_limb.all_concept_network_data,
                                                                          verbose=verbose)
        
        if verbose:
            print(f"cleaned_network = {cleaned_network}\n\n")
        
        curr_limb.all_concept_network_data = cleaned_network
        
        #setting the concept network
        st_soma = curr_limb.all_concept_network_data[0]["starting_soma"]
        st_node = curr_limb.all_concept_network_data[0]["starting_node"]
        curr_limb.set_concept_network_directional(starting_soma=st_soma,
                                                 starting_node=st_node)
        
        # --------- 1/24: Cleaning the preprocessed data as well -----------#
        if verbose:
            print(f"cleaned_network = {cleaned_network}")
            
        new_limb_network = all_concept_network_data_to_limb_network_stating_info(cleaned_network)
        
        if verbose:
            print(f"\n---------\nnew_limb_network = {new_limb_network}\n---------\n")
        neuron_obj.preprocessed_data["limb_network_stating_info"][j] = new_limb_network
        
        if verbose:
            print(f"curr_limb.all_concept_network_data = {curr_limb.all_concept_network_data}\n\n")
            
#         neuron_obj[j] = curr_limb
    
#     return neuron_obj




        


        
    
    

def sum_feature_over_limb_branch_dict(neuron_obj,
                                       limb_branch_dict,
                                       feature=None,
                                      branch_func_instead_of_feature = None,
                                     feature_function=None):
    """
    Purpose: To sum the value of some feature over the branches
    specified by the limb branch dict
    """
    
    feature_total = 0
    
    for limb_name, branch_list in limb_branch_dict.items():
        for b in branch_list:
            if feature == "n_branches":
                feature_value = 1
            elif branch_func_instead_of_feature is not None:
                feature_value = branch_func_instead_of_feature(neuron_obj[limb_name][b])
            else:
                feature_value = getattr(neuron_obj[limb_name][b],feature)
            if feature_function is not None:
                feature_value = feature_function(feature_value)
            feature_total += feature_value
            
    return feature_total

def feature_over_limb_branch_dict(neuron_obj,
                                       limb_branch_dict,
                                       feature=None,
                                     feature_function=None,
                                  feature_from_fuction = None,
                                  feature_from_fuction_kwargs = None,
                                  keep_seperate=False,
                                  branch_func_instead_of_feature = None,
                                 skip_None=True):
    """
    Purpose: To sum the value of some feature over the branches
    specified by the limb branch dict
    
    
    """
    
    feature_total = []
    
    for limb_name, branch_list in limb_branch_dict.items():
        for b in branch_list:
            if feature_from_fuction is not None:
                if feature_from_fuction_kwargs is None:
                    feature_from_fuction_kwargs = dict()
                feature_value = feature_from_fuction(neuron_obj[limb_name][b],**feature_from_fuction_kwargs)
            else:
                if branch_func_instead_of_feature is not None:
                    feature_value = branch_func_instead_of_feature(neuron_obj[limb_name][b])
                else:
                    feature_value = getattr(neuron_obj[limb_name][b],feature)
                if feature_function is not None:
                    feature_value = feature_function(feature_value)
            
            if skip_None and feature_value is None:
                continue
            
            if keep_seperate:
                feature_total.append(feature_value)
            else:
                if nu.is_array_like(feature_value):
                    feature_total+= list(feature_value)
                else:
                    feature_total.append(feature_value)
            
    return feature_total



    
# ------ 2/1: Utils for a lot of the edge functions ----------- #



    
    
    

def branches_within_skeletal_distance(limb_obj,
                                    start_branch,
                                    max_distance_from_start,
                                    verbose = False,
                                    include_start_branch_length = False,
                                    include_node_branch_length = False,
                                    only_consider_downstream = False):

    """
    Purpose: to find nodes within a cetain skeletal distance of a certain 
    node (can be restricted to only those downstream)

    Pseudocode: 
    1) Get the directed concept grpah
    2) Get all of the downstream nodes of the node
    3) convert directed concept graph into an undirected one
    4) Get a subgraph using all of the downstream nodes
    5) For each node: 
    - get the shortest path from the node to the starting node
    - add up the skeleton distance (have options for including each endpoint)
    - if below the max distance then add
    6) Return nodes


    Ex: 
    start_branch = 53
        
    viable_downstream_nodes = branches_within_skeletal_distance(limb_obj = current_neuron[6],
                                start_branch = start_branch,
                                max_distance_from_start = 50000,
                                verbose = False,
                                include_start_branch_length = False,
                                include_node_branch_length = False,
                                only_consider_downstream = True)

    limb_branch_dict=dict(L6=viable_downstream_nodes+[start_branch])


    """

    curr_limb = limb_obj



    viable_downstream_nodes = []

    dir_nx = curr_limb.concept_network_directional

    #2) Get all of the downstream nodes of the node

    if only_consider_downstream:
        all_downstream_nodes = list(xu.all_downstream_nodes(dir_nx,start_branch))
    else:
        all_downstream_nodes = list(dir_nx.nodes())
        all_downstream_nodes.remove(start_branch)

    if len(all_downstream_nodes) == 0:
        if verbose:
            print(f"No downstream nodes to test")

        return []

    if verbose:
        print(f"Number of downstream nodes = {all_downstream_nodes}")

    #3) convert directed concept graph into an undirected one
    G_whole = nx.Graph(dir_nx)

    #4) Get a subgraph using all of the downstream nodes
    G = G_whole.subgraph(all_downstream_nodes + [start_branch])

    for n in all_downstream_nodes:

        #- get the shortest path from the node to the starting node
        try:
            curr_shortest_path = nx.shortest_path(G,start_branch,n)
        except:
            if verbose:
                print(f"Continuing because No path between start node ({start_branch}) and node {n}")
            continue 


        if not include_node_branch_length:
            curr_shortest_path = curr_shortest_path[:-1]

        if not include_start_branch_length:
            curr_shortest_path = curr_shortest_path[1:]

        total_sk_length_of_path = np.sum([curr_limb[k].skeletal_length for k in curr_shortest_path])

        if total_sk_length_of_path <= max_distance_from_start:
            viable_downstream_nodes.append(n)
        else:
            if verbose:
                print(f"Branch {n} was too far from the start node : {total_sk_length_of_path} (threshold = {max_distance_from_start})")

    return viable_downstream_nodes

def neuron_limb_branch_dict(neuron_obj):
    """
    Purpose: To develop a limb branch dict represnetation
    of the limbs and branchs of a neuron
    
    """
    limb_branch_dict_new = dict()
    
    if neuron_obj.__class__.__name__ == "Neuron":
        for limb_name in neuron_obj.get_limb_node_names():
            limb_branch_dict_new[limb_name] = neuron_obj[limb_name].get_branch_names()
    else:
        net = neuron_obj
        curr_limb_names = [k for k in net.nodes() if "L" in k]
        for limb_name in curr_limb_names:
            limb_branch_dict_new[limb_name] = np.array(list(net.nodes[limb_name]["data"].concept_network.nodes()))

        
    return limb_branch_dict_new


def limb_branch_combining(
                           limb_branch_dict_list,
                           combining_function,
                           verbose=False):
    """
    Purpose: To get every node that is not in limb branch dict
    
    Ex: 
    invert_limb_branch_dict(curr_neuron_obj,limb_branch_return,
                       verbose=True)
    """
    if len(limb_branch_dict_list) == 0:
        return dict()
    all_keys = nu.union1d_multi_list([list(k.keys()) for k in limb_branch_dict_list])
    
    
    limb_branch_dict_new = dict()
    for limb_name in all_keys:
        
        if verbose:
            print(f"\n--- Working on limb {limb_name}")
        
        curr_branches = [k.get(limb_name,[]) for k in limb_branch_dict_list]
        
        leftover_branches = nu.function_over_multi_lists(curr_branches,combining_function)
        
        if verbose:
            print(f"combining_function = {combining_function}")
            print(f"curr_branches = {curr_branches}")
            print(f"leftover_branches = {leftover_branches}")
            
        if len(leftover_branches)>0:
            limb_branch_dict_new[limb_name] = leftover_branches
            
    return limb_branch_dict_new



def limb_branch_intersection(limb_branch_dict_list):
    
    return limb_branch_combining(
                           limb_branch_dict_list,
                           np.intersect1d,
                           verbose=False
    )



# ----------- For rules with doubling back, width jumps, high degree nodes, train track crossings -------- #


def max_soma_volume(neuron_obj,
                    divisor = 1_000_000_000):
    """
    Will find the largest number of faces out of all the somas
    
    """
    #soma_volumes = [neuron_obj[k].volume/divisor for k in neuron_obj.get_soma_node_names()] 
    
    try:
        soma_volumes = [tu.mesh_volume(neuron_obj[k].mesh)/divisor for k in neuron_obj.get_soma_node_names()] 
    except:
        return 0
    
    largest_volume = np.max(soma_volumes)
    return largest_volume

def max_soma_n_faces(neuron_obj):
    """
    Will find the largest number of faces out of all the somas
    
    """
    soma_areas = [len(neuron_obj[k].mesh.faces) for k in neuron_obj.get_soma_node_names()] 
    largest_soma_area = np.max(soma_areas)
    return largest_soma_area

def max_soma_area(neuron_obj):
    """
    Will find the largest number of faces out of all the somas
    
    """
    soma_n_faces = [neuron_obj[k].area for k in neuron_obj.get_soma_node_names()] 
    largest_n_faces = np.max(soma_n_faces)
    return largest_n_faces



def soma_centers(neuron_obj,
                 soma_name=None,
                voxel_adjustment=False,
                 voxel_adjustment_vector = None,
                 return_int_form=True,
                return_single=True):
    """
    Will come up with the centers predicted for each of the somas in the neuron
    """
    if voxel_adjustment_vector is None:
        voxel_adjustment_vector = voxel_to_nm_scaling
    
    if soma_name is None:
#         current_soma_means = np.array([tu.mesh_center_vertex_average(neuron_obj[s_name].mesh) 
#                                for s_name in neuron_obj.get_soma_node_names()])
        current_soma_means = np.array([neuron_obj[soma_name].mesh_center
                               for s_name in neuron_obj.get_soma_node_names()])
    else:
        current_soma_means = np.array([neuron_obj[soma_name].mesh_center]).reshape(-1,3)
    
    if voxel_adjustment and voxel_adjustment_vector is not None:
        
        current_soma_means = current_soma_means/voxel_adjustment_vector
        
    if return_int_form:
        current_soma_means = current_soma_means.astype("int")
        
    if return_single:
        if len(current_soma_means) != 1:
            raise Exception(f"Not just one soma center: {current_soma_means}")
        current_soma_means=current_soma_means[0]
        
    return current_soma_means
    

    

# ---- 2/15: For helping with backtracking synapses back to the somas -------- #

    
    





def shared_skeleton_endpoints_for_connected_branches(limb_obj,
                                                    branch_1,
                                                    branch_2,
                                                   verbose=False,
                                                    check_concept_network_connectivity=True):
    """
    Purpose: To find the shared skeleton endpoint of 
    branches that are connected in the concept network
    
    Ex:
    shared_skeleton_endpoints_for_connected_branches(neuron_obj[5],
                                                0,1,
                                                verbose=True)
    
    """
    curr_concept_network = limb_obj.concept_network
    if check_concept_network_connectivity:
        if branch_1 not in xu.get_neighbors(curr_concept_network,branch_2):
            raise Exception(f"Branches {branch_1} and {branch_2} are not connected in the concept network")

    shared_endpoints = sk.shared_endpoint(limb_obj[branch_1].skeleton,
                                          limb_obj[branch_2].skeleton,
                                          return_possibly_two=True)
    if verbose:
        print(f"shared_endpoints = {shared_endpoints}")
        
    return shared_endpoints



    

def limb_branch_dict_to_skeleton(neuron_obj,limb_branch_dict):
    """
    Purpose: turn a limb_branch_dict into
    the corresponding skeleton of branches
    stacked together
    
    Pseudocode:
    1) Get the skeletons over the limb branch dict
    2) Stack the skeletons
    
    """
    branch_skeletons = feature_over_limb_branch_dict(neuron_obj,
                                       limb_branch_dict,
                                       feature="skeleton",
                                  keep_seperate=True,
                                 skip_None=True)
    return np.array(sk.stack_skeletons(branch_skeletons))

def axon_skeleton(neuron_obj):
    axon_limb_branch_dict = neuron_obj.axon_limb_branch_dict
    return limb_branch_dict_to_skeleton(neuron_obj,axon_limb_branch_dict)

def dendrite_skeleton(neuron_obj):
    dendrite_limb_branch_dict = neuron_obj.dendrite_limb_branch_dict
    return limb_branch_dict_to_skeleton(neuron_obj,dendrite_limb_branch_dict)


def mesh_without_mesh_attribute(obj,
                               mesh_attribute):
    """
    Purpose: To return the branch mesh without any spines
    """
    original_mesh_flag = False
    if hasattr(obj,mesh_attribute) and getattr(obj,mesh_attribute) is not None:
        if len(getattr(obj,mesh_attribute)) > 0:
            ex_branch_no_spines_mesh = tu.original_mesh_faces_map(obj.mesh,
                                    tu.combine_meshes(getattr(obj,mesh_attribute)),
                                   matching=False,
                                   print_flag=False,
                                   match_threshold = 0.001,
                                                            return_mesh=True,
                                                                 )
        else:
            original_mesh_flag = True
    else: 
        original_mesh_flag = True
    
    if original_mesh_flag:
        ex_branch_no_spines_mesh = obj.mesh
        
    return ex_branch_no_spines_mesh

def mesh_without_boutons(obj):
    return mesh_without_mesh_attribute(obj,
                                      mesh_attribute="boutons")





    

    
def neuron_mesh_from_branches(neuron_obj,
                             plot_mesh=False):
    """
    Purpose: To reconstruct the mesh of neuron
    from all of the branch obejcts

    Pseudocode:
    Iterate through all the limbs:
        iterate through all the branches
            add to big list

    Add some to big list

    concatenate list into mesh
    """
    from mesh_tools import trimesh_utils as tu

    neuron_mesh_list = []
    for limb_obj in neuron_obj:
        for branch_obj in limb_obj:
            neuron_mesh_list.append(branch_obj.mesh)
    neuron_mesh_list += neuron_obj.get_soma_meshes()

    neuron_mesh_from_branches = tu.combine_meshes(neuron_mesh_list)
    return neuron_mesh_from_branches





def all_downstream_branches(limb_obj,
                           branch_idx):
    """
    Will return all of the branches that are downstream
    of the branch_idx
    
    """
    return xu.all_downstream_nodes(limb_obj.concept_network_directional,branch_idx)

def feature_over_branches(limb_obj,
                                 branch_list,
                                 feature_name=None,
                                  feature_function=None,
                          use_limb_obj_and_branch_idx = False,
                         combining_function=None,
                         verbose = False,
                         **kwargs):
    """
    To calculate a certain feature over 
    all the branches in a list
    """
    feature_list = []
    for b in branch_list:
        branch_obj = limb_obj[b]
        if feature_name is not None:
            curr_val = getattr(branch_obj,feature_name)
        elif feature_function is not None:
            if not use_limb_obj_and_branch_idx:
                curr_val = feature_function(branch_obj,**kwargs)
            else:
                curr_val = feature_function(limb_obj=limb_obj,branch_idx=b,**kwargs)
        else:
            raise Exception("Need to set either feature_name or feature function")
            
        feature_list.append(curr_val)
        
    if verbose:
        print(f"feature_list (before combining) = {feature_list}")
    if combining_function is not None:
        if len(feature_list)>0:
            feature_list = combining_function(feature_list)
        else:
            feature_list = 0
    
    if verbose:
        print(f"feature_list (after combining) = {feature_list}")
    
    return feature_list

def sum_feature_over_branches(limb_obj,
                                 branch_list,
                                 feature_name=None,
                                  feature_function=None,
                         verbose = False):
    return feature_over_branches(limb_obj,
                                 branch_list,
                                 feature_name=feature_name,
                                  feature_function=feature_function,
                                 combining_function = np.sum,
                         verbose = verbose)

def skeletal_length_over_downstream_branches(limb_obj,
                                            branch_idx,
                                            combining_function = np.sum,
                                             include_branch_skeletal_length=True,
                                             nodes_to_exclude = None,
                                            verbose=False):
    """
    Will compute how much skeleton there is downstream of a certain node
    
    """
    downstream_branches = list(all_downstream_branches(limb_obj,branch_idx))
    
    if nodes_to_exclude is not None:
        downstream_branches = list(np.setdiff1d(downstream_branches,nodes_to_exclude))
    
    if include_branch_skeletal_length:
        downstream_branches.append(branch_idx)
        
    sk_sum =feature_over_branches(limb_obj,
                              branch_list = downstream_branches,
                              feature_name="skeletal_length",
                              combining_function = combining_function,
                              verbose = verbose
                             )
    if verbose:
        print(f"Total skeleton downstream = {sk_sum}")
        
    return sk_sum







    
    
def upstream_node(limb_obj,branch):
    return xu.upstream_node(limb_obj.concept_network_directional,branch)



def neighbor_endpoint(limb_obj,
                     branch_idx,
                     verbose = False,
                     return_endpoint_index=False,
                     neighbor_type="upstream"):
    """
    Pseudocode: 
    1) Find the upsream node
    2a) if upstream node is None then use the current current starting node
    2b) if upstream node, find the common skeleton point between the 2

    """

    upstream_node = upstream_node(limb_obj,branch_idx)
    if verbose:
        print(f"upstream_node for {branch_idx}: {upstream_node}")

    if upstream_node is None:
        common_endpoint = limb_obj.current_starting_coordinate
    else:
        shared_endooints = shared_skeleton_endpoints_for_connected_branches(limb_obj,
                                                        branch_idx,
                                                        upstream_node,
                                                        check_concept_network_connectivity=False).reshape(-1,3)
        common_endpoint = shared_endooints[0]

        if verbose:
            print(f"shared_endooints = {shared_endooints}")

    
        
    if verbose:
        print(f"common_endpoint = {common_endpoint}")
        
    if neighbor_type == "upstream":
        pass
    elif neighbor_type == "downstream":
        endpoint_index = nu.matching_row_index(limb_obj[branch_idx].endpoints,common_endpoint)
        common_endpoint = limb_obj[branch_idx].endpoints[1-endpoint_index]
        if verbose:
            print(f"The donwsream endpoint is {common_endpoint}")
    else:
        raise Exception(f"Unknown neighbor_type: {neighbor_type}")
        
    if return_endpoint_index:
        endpoint_index = nu.matching_row_index(limb_obj[branch_idx].endpoints,common_endpoint)
        if verbose:
            print(f"endpoints = {limb_obj[branch_idx].endpoints}")
            print(f"return_endpoint_index = {endpoint_index}")
        return endpoint_index
        
        
    return common_endpoint


def upstream_endpoint(limb_obj,
                     branch_idx,
                     verbose = False,
                     return_endpoint_index=False):
    """
    Purpose: To get the coordinate of the part of the 
    skeleton connecting to the upstream branch
    
    branch_idx = 263
    
    """
    return neighbor_endpoint(limb_obj,
                     branch_idx,
                     verbose,
                     return_endpoint_index,
                     neighbor_type="upstream")



def upstream_downstream_endpoint_idx(
    limb_obj,
    branch_idx,
    verbose = False,
    ):
    """
    To get the upstream and downstream endpoint idx
    returned (upstream_idx,downstream_idx)
    """
    
    up_idx = upstream_endpoint(limb_obj,
                     branch_idx,
                     verbose = verbose,
                     return_endpoint_index=True)
    return (up_idx,1-up_idx)
    
    

    
def branch_path_to_node(limb_obj,
                             start_idx,
                            destination_idx,
                             include_branch_idx = False,
                              include_last_branch_idx = True,
                            skeletal_length_min = None,
                            verbose = False,
                            reverse_di_graph=True,
                            starting_soma_for_di_graph = None,
                       ):
    """
    Purpose: Will find the branch objects on the path
    from current branch to the starting coordinate

    Application: Will know what width objects to compare to
    for width jump

    Pseudocode: 
    1) Get the starting coordinate of brnach
    2) Find the shortest path from branch_idx to starting branch
    3) Have option to include starting branch or not
    3) If skeletal length threshold is set then:
    a. get skeletal length of all branches on path
    b. Filter out all branches that are not above the skeletal length threshold


        

    """
    if starting_soma_for_di_graph is not None:
        limb_obj.set_concept_network_directional(soma_group_idx=starting_soma_for_di_graph,
                                                suppress_disconnected_errors=True)
        G = limb_obj.concept_network_directional
    elif reverse_di_graph:
        G = xu.reverse_DiGraph(limb_obj.concept_network_directional)
    else:
        G = limb_obj.concept_network_directional

    #2) Find the shortest path from branch_idx to starting branch
    try:
        shortest_path = nx.shortest_path(G,
                                         start_idx,destination_idx)
    except:
        shortest_path = None
        
    if verbose:
        print(f"shortest_path = {shortest_path}")
        
    if shortest_path is None:
        if verbose:
            print(f"No path between Nodes so returning None")
        return shortest_path

    if not include_branch_idx:
        shortest_path = shortest_path[1:]

    shortest_path = np.array(shortest_path)
    
    if not include_last_branch_idx:
        shortest_path = shortest_path[:-1]

    if skeletal_length_min is not None:
        sk_len = np.array([limb_obj[k].skeletal_length for k in shortest_path])
        shortest_path = shortest_path[sk_len>skeletal_length_min]
        if verbose:
            print(f"shortest_path AFTER skeletal length filtering above"
                  f" skeletal_length_min ({skeletal_length_min}): \n   {shortest_path}")

    return shortest_path

  




# --------- 7/28: For the apical classification ------------
def skeleton_over_limb_branch_dict(neuron_obj,
                                   limb_branch_dict,
                                   stack_skeletons = True,
                                   plot_skeleton = False,
                                  ):
    """
    Purpose: To collect the meshes over a limb branch dict
    
    mesh_over_limb_branch_dict(neuron_obj,
                              limb_branch_from_candidate(apical_candidates[0]),
                              plot_mesh=True)
    """

    individual_sk = feature_over_limb_branch_dict(neuron_obj,
                                     limb_branch_dict=limb_branch_dict,
                                      keep_seperate = True,
                                     feature="skeleton")
    if stack_skeletons:
        individual_sk = sk.stack_skeletons(individual_sk)
        
        
    return individual_sk












    

    


            

            

    

    

def shortest_path(limb_obj,start_branch_idx,destiation_branch_idx,
                 plot_path=False):
    shortest_p = np.array(xu.shortest_path(limb_obj.concept_network,start_branch_idx,destiation_branch_idx))
    return shortest_p

    
    
# ---------- 10/22 -------------











# ------- 12/27: 




    
    




def parent_node(
    limb_obj,
    branch_idx,
    verbose=False):
    """
    Purpose: to get the parent 
    branch of a branch_idx
    
    """
    parent_node = xu.upstream_node(limb_obj.concept_network_directional,branch_idx)
    if verbose:
        print(f"parent_node of branch {branch_idx} = branch {parent_node}")
        
    return parent_node








def recalculate_endpoints_and_order_skeletons_for_branch(branch_obj):
    branch_obj.calculate_endpoints()
    branch_obj.order_skeleton_by_smallest_endpoint()
    branch_obj._skeleton_graph = None
    branch_obj._endpoints_nodes = None
    
def recalculate_endpoints_and_order_skeletons_over_neuron(neuron_obj):
    """
    Purpose: Recalculate endpoints
    and order the skeletons
    """

    for limb_idx in neuron_obj.get_limb_node_names():
        limb_obj = neuron_obj[limb_idx]
        for branch_idx in limb_obj.get_branch_names():
            #branch_obj = limb_obj[branch_idx]
            recalculate_endpoints_and_order_skeletons_for_branch(neuron_obj[limb_idx][branch_idx])


        

    #package where can use the Branches class to help do branch skeleton analysis
    
    
def calculate_decomposition_products(
    neuron_obj,
    store_in_obj = False,
    verbose = False,
    ):
    
    # ---- basic statistics of neuron
    stats_dict = neuron_obj.neuron_stats(stats_to_ignore = [
                     # n_boutons/axon_length/axon_area removed from neuron_stats (Phase 5).
                     "max_soma_volume",
                     "max_soma_n_faces",],
        include_skeletal_stats = True,
        include_centroids= True,
        #voxel_adjustment_vector=voxel_adjustment_vector,
    )
    
    # --- generating the skeleton 
    skeleton = neuron_obj.skeleton
    
    # --- skeleton stats
    from neurd.neuron_statistics import skeleton_stats_from_neuron_obj  # P3: local import breaks neuron_utils<->neuron_statistics cycle
    sk_stats = skeleton_stats_from_neuron_obj(
            neuron_obj,
            include_centroids=True,
            verbose = verbose,
    )
    
    stats_dict.update(sk_stats)
    decomp_products = pipeline.StageProducts(
        skeleton=skeleton,
        **stats_dict,
    )

    if store_in_obj:
        neuron_obj.pipeline_products.set_stage_attrs(
            decomp_products,
            stage = "decomposition"
        )
        
    return decomp_products



  
    


# ---- 4/4 adjustments ---


            
                    

#--- from neurd_packages ---

# P1/P3: neuron_utils is now a true intra-package SINK (imports nothing from neurd at
# module level), so all 9 modules that import it load without a cycle. The three names it
# used (neuron.Branch, concept_network_utils.subgraph_around_branch,
# neuron_statistics.skeleton_stats_from_neuron_obj) are now local imports at their call sites.

#--- from mesh_tools ---
from mesh_tools import compartment_utils as cu
from mesh_tools import skeleton_utils as sk
from mesh_tools import trimesh_utils as tu

#--- from datasci_tools ---
from datasci_tools.general_utils import  invert_mapping
from datasci_tools import networkx_utils as xu
from datasci_tools import numpy_dep as np
from datasci_tools import numpy_utils as nu
from datasci_tools import pipeline

