
import copy
from copy import deepcopy
from dataclasses import dataclass
import itertools

import networkx as nx
from scipy.spatial import KDTree
import time
import trimesh
from datasci_tools import numpy_dep as np
from datasci_tools import general_utils as gu
#--- from neurd_packages ---
from . import neuron
from . import neuron_utils as nru

from .soma_extraction_utils  import extract_soma_center
from . import parameters
from . import submesh_ops


#--- from mesh_tools ---
from mesh_tools import compartment_utils as cu
from mesh_tools import meshparty_skeletonize as m_sk
from mesh_tools import skeleton_utils as sk
from mesh_tools import trimesh_utils as tu

from datasci_tools import networkx_utils as xu
from datasci_tools import numpy_utils as nu
from datasci_tools import system_utils as su
from datasci_tools.tqdm_utils import tqdm
#importing at the bottom so don't get any conflicts

#for meshparty preprocessing


process_version = 10 #no skeleton jumping hopefully


min_distance_threshold = 0.00001

# Stitch config — hardcoded constants (former signature defaults; never tuned, not from
# parameters.params). Module-level so preprocess_limb + the stitch helpers read them directly
# instead of threading them as arguments.
meshparty_segment_size = 100
move_MAP_stitch_to_end_or_branch = True
distance_to_move_point_threshold = 500
prevent_MP_starter_branch_stitches = False
check_correspondence_branches = True

#--------------- default arguments to use ----------#

        
def mesh_correspondence_first_pass(
    mesh,
    skeleton=None,
    skeleton_branches=None,
    distance_by_mesh_center=True,
    remove_inside_pieces_threshold = 0,
    skeleton_segment_width = 1000,
    initial_distance_threshold = 3000,
    skeletal_buffer = 100,
    backup_distance_threshold = 6000,
    backup_skeletal_buffer = 300,
    connectivity="edges",
    ):
    """
    Will come up with the mesh correspondences for all of the skeleton
    branches: where there can be overlaps and empty faces
    
    """
    curr_limb_mesh = mesh
    curr_limb_sk = skeleton
    
    if remove_inside_pieces_threshold > 0:
        curr_limb_mesh_indices = tu.remove_mesh_interior(curr_limb_mesh,
                                                 size_threshold_to_remove=remove_inside_pieces_threshold,
                                                 try_hole_close=False,
                                                 return_face_indices=True,
                                                )
        curr_limb_mesh = curr_limb_mesh.submesh([curr_limb_mesh_indices],append=True,repair=False)
    else:
        curr_limb_mesh_indices = np.arange(len(curr_limb_mesh.faces))
    
    if skeleton_branches is None:
        if skeleton is None:
            raise Exception("Both skeleton and skeleton_branches is None")
        curr_limb_branches_sk_uneven = sk.decompose_skeleton_to_branches(curr_limb_sk) #the line that is decomposing to branches
    else:
        curr_limb_branches_sk_uneven = skeleton_branches 

    #Doing the limb correspondence for all of the branches of the skeleton
    local_correspondence = dict()
    for j,curr_branch_sk in tqdm(enumerate(curr_limb_branches_sk_uneven)):
        local_correspondence[j] = dict()

        
        returned_data = cu.mesh_correspondence_adaptive_distance(curr_branch_sk,
                                      curr_limb_mesh,
                                     skeleton_segment_width = skeleton_segment_width,
                                     distance_by_mesh_center=distance_by_mesh_center,
                                    distance_threshold = initial_distance_threshold,
                                    buffer = skeletal_buffer,
                                                                connectivity=connectivity)
        if len(returned_data) == 0:
            print("Got nothing from first pass so expanding the mesh correspondnece parameters ")
            returned_data = cu.mesh_correspondence_adaptive_distance(curr_branch_sk,
                                      curr_limb_mesh,
                                     skeleton_segment_width = skeleton_segment_width,
                                     distance_by_mesh_center=distance_by_mesh_center,
                                    buffer=backup_skeletal_buffer,
                                     distance_threshold=backup_distance_threshold,
                                    return_closest_face_on_empty=True,
                                        connectivity=connectivity)
            
        # Need to just pick the closest face is still didn't get anything
        
        # ------ 12/3 Addition: Account for correspondence that does not work so just picking the closest face
        curr_branch_face_correspondence, width_from_skeleton = returned_data
        
            
        if len(curr_branch_face_correspondence) > 0:
            curr_submesh = curr_limb_mesh.submesh([list(curr_branch_face_correspondence)],append=True,repair=False)
        else:
            curr_submesh = trimesh.Trimesh(vertices=np.array([]),faces=np.array([]))


        local_correspondence[j]["branch_skeleton"] = curr_branch_sk
        local_correspondence[j]["correspondence_mesh"] = curr_submesh
        local_correspondence[j]["correspondence_face_idx"] = curr_limb_mesh_indices[curr_branch_face_correspondence]
        local_correspondence[j]["width_from_skeleton"] = width_from_skeleton
        
        
    
    return local_correspondence


def check_skeletonization_and_decomp(
    skeleton,
    local_correspondence):
    """
    Purpose: To check that the decomposition and skeletonization went well
    
    
    """
    #couple of checks on how the decomposition went:  for each limb
    #1) if shapes of skeletons cleaned and divided match
    #2) if skeletons are only one component
    #3) if you downsample the skeletons then still only one component
    #4) if any empty meshes
    cleaned_branch = skeleton
    empty_submeshes = []

    print(f"Limb decomposed into {len(local_correspondence)} branches")

    #get all of the skeletons and make sure that they from a connected component
    divided_branches = [local_correspondence[k]["branch_skeleton"] for k in local_correspondence]
    divided_skeleton_graph = sk.convert_skeleton_to_graph(
                                    sk.stack_skeletons(divided_branches))

    divided_skeleton_graph_recovered = sk.convert_graph_to_skeleton(divided_skeleton_graph)

    cleaned_limb_skeleton = cleaned_branch
 


    #check that it is all one component
    divided_skeleton_graph_n_comp = nx.number_connected_components(divided_skeleton_graph)

    cleaned_limb_skeleton_graph = sk.convert_skeleton_to_graph(cleaned_limb_skeleton)
    cleaned_limb_skeleton_graph_n_comp = nx.number_connected_components(cleaned_limb_skeleton_graph)

    if divided_skeleton_graph_n_comp > 1 or cleaned_limb_skeleton_graph_n_comp > 1:
        raise Exception(f"One of the decompose_skeletons or cleaned skeletons was not just one component : {divided_skeleton_graph_n_comp,cleaned_limb_skeleton_graph_n_comp}")

    #check that when we downsample it is not one component:
    curr_branch_meshes_downsampled = [sk.resize_skeleton_branch(b,n_segments=1) for b in divided_branches]
    downsampled_skeleton = sk.stack_skeletons(curr_branch_meshes_downsampled)
    curr_sk_graph_debug = sk.convert_skeleton_to_graph(downsampled_skeleton)


    con_comp = list(nx.connected_components(curr_sk_graph_debug))
    if len(con_comp) > 1:
        raise Exception(f"There were more than 1 component when downsizing: {[len(k) for k in con_comp]}")
    else:
        print(f"The downsampled branches number of connected components = {len(con_comp)}")


    for j,v in local_correspondence.items():
        if len(v["correspondence_mesh"].faces) == 0:
            empty_submeshes.append(j)


    if len(empty_submeshes) > 0:
        raise Exception(f"Found empyt meshes after branch mesh correspondence: {empty_submeshes}")
        

def correspondence_1_to_1(
    mesh,
    local_correspondence,
    curr_limb_endpoints_must_keep=None,
    curr_soma_to_piece_touching_vertices=None,
    must_keep_labels=dict(),
    fill_to_soma_border=True,
    input_sub=None,
                    ):
    """
    Will Fix the 1-to-1 Correspondence of the mesh
    correspondence for the limbs and make sure that the
    endpoints that are designated as touching the soma then
    make sure the mesh correspondnece reaches the soma limb border

    has an optional argument must_keep_labels that will allow you to specify some labels that are a must keep

    Phase B: `input_sub` (a submesh_ops.SubMesh whose .mesh IS `mesh` and whose parent chain reaches
    the limb) makes each branch's `branch_face_idx` come out in the LIMB frame by construction --
    folding the split's input-frame indices through `input_sub.root_face_idx()`. This is the single
    seam that replaces the callers' manual `parent_idx[branch_face_idx]` remaps (and, for the stitch
    floating path, the frame-desync that `_rebuild_limb_frames` repairs post-hoc). When `input_sub`
    is None the output stays in the input-mesh frame -- identical to the pre-Phase-B behaviour.
    """
    
    if len(submesh_ops.split(mesh))>1:
        su.compressed_pickle(mesh,"mesh")
        raise Exception("Mesh passed to correspondence_1_to_1 is not just one mesh")
    
    mesh_start_time = time.time()
    print(f"\n\n--- Working on 1-to-1 correspondence-----")

    #geting the current limb mesh

    no_missing_labels = list(local_correspondence.keys()) #counts the number of divided branches which should be the total number of labels
    curr_limb_mesh = mesh

    #set up the face dictionary
    face_lookup = dict([(j,[]) for j in range(0,len(curr_limb_mesh.faces))])

    for j,branch_piece in local_correspondence.items():
        curr_faces_corresponded = branch_piece["correspondence_face_idx"]

        for c in curr_faces_corresponded:
            face_lookup[c].append(j)

    original_labels = set(list(itertools.chain.from_iterable(list(face_lookup.values()))))

    if len(original_labels) != len(no_missing_labels):
        raise Exception(f"len(original_labels) != len(no_missing_labels) for original_labels = {len(original_labels)},no_missing_labels = {len(no_missing_labels)}")

    if max(original_labels) + 1 > len(original_labels):
        raise Exception("There are some missing labels in the initial labeling")


    #here is where can call the function that resolves the face labels
    try:
        face_coloring_copy = cu.resolve_empty_conflicting_face_labels(
                         curr_limb_mesh = curr_limb_mesh,
                         face_lookup=face_lookup,
                         no_missing_labels = list(original_labels),
                        must_keep_labels=must_keep_labels,
                        branch_skeletons = [local_correspondence[k]["branch_skeleton"] for k in local_correspondence.keys()],
        )
    except Exception as e:
        # CLASS-B legibility: this resolve is the floating-piece / dense-limb correspondence step.
        # "missing labels was not resolved" here means some skeleton segment cannot claim a
        # connected face patch (geometric limit), not a corrupt input. Surface WHEN/WHY, re-raise
        # unchanged (no behavior change) so any existing handler still sees the original error.
        print(f"[correspondence_1_to_1] resolve_empty_conflicting_face_labels FAILED on a "
              f"{len(curr_limb_mesh.faces)}-face piece with {len(original_labels)} skeleton "
              f"segment(s) -> {type(e).__name__}: {e}")
        raise

    """  9/17 Addition: Will make sure that the desired starting node is touching the soma border """
    """
    Pseudocode:
    For each soma it is touching
    0) Get the soma border
    1) Find the label_to_expand based on the starting coordinate
    a. Get the starting coordinate

    soma_to_piece_touching_vertices=None
    endpoints_must_keep

    """

    #curr_limb_endpoints_must_keep --> stores the endpoints that should be connected to the soma
    #curr_soma_to_piece_touching_vertices --> maps soma to  a list of grouped touching vertices

    if fill_to_soma_border:
        if (not curr_limb_endpoints_must_keep is None) and (not curr_soma_to_piece_touching_vertices is None):
            for sm,soma_border_list in curr_soma_to_piece_touching_vertices.items():
                for curr_soma_border,st_coord in zip(soma_border_list,curr_limb_endpoints_must_keep[sm]):

                    #1) Find the label_to_expand based on the starting coordinate
                    divided_branches = [v["branch_skeleton"] for v in local_correspondence.values()]
                    label_to_expand = sk.find_branch_skeleton_with_specific_coordinate(divded_skeleton=divided_branches,
                                                                                       current_coordinate=st_coord)[0]


                    face_coloring_copy = cu.waterfill_starting_label_to_soma_border(curr_limb_mesh,
                                                       border_vertices=curr_soma_border,
                                                        label_to_expand=label_to_expand,
                                                       total_face_labels=face_coloring_copy,
                                                       print_flag=True)


    # -- splitting the mesh pieces into individual pieces
    divided_submeshes,divided_submeshes_idx = tu.split_mesh_into_face_groups(curr_limb_mesh,face_coloring_copy)

    # Phase B: with input_sub, fold the split's input-frame indices to the LIMB frame by composition
    # (branch_face_idx = input_sub.root_face_idx()[input_idx]); without it, keep the input-frame idx.
    if input_sub is not None:
        assert input_sub.mesh is curr_limb_mesh, \
            "correspondence_1_to_1: input_sub.mesh must be the correspondence-input mesh"
        _input_root = input_sub.root_face_idx()

    #-- check that all the split mesh pieces are one component --#
    local_correspondence_revised = deepcopy(local_correspondence)
    #save off the new data as branch mesh
    for k in local_correspondence_revised.keys():
        _input_idx = divided_submeshes_idx[k]
        local_correspondence_revised[k]["branch_mesh"] = divided_submeshes[k]
        local_correspondence_revised[k]["branch_face_idx"] = (
            _input_idx if input_sub is None else _input_root[_input_idx])

        #clean the limb correspondence that we do not need
        del local_correspondence_revised[k]["correspondence_mesh"]
        del local_correspondence_revised[k]["correspondence_face_idx"]
    

    return local_correspondence_revised


def _run_mesh_correspondence(mesh, skeleton_branches, **one_to_one_kwargs):
    """The recurring "first-pass mesh correspondence, then resolve 1-to-1" pair.

    Equivalent to:
        local = mesh_correspondence_first_pass(mesh=mesh, skeleton_branches=skeleton_branches)
        return correspondence_1_to_1(mesh=mesh, local_correspondence=local, **one_to_one_kwargs)

    Use only where the intermediate first-pass result is not needed on its own (i.e. it feeds
    straight into correspondence_1_to_1). Sites that reuse the first-pass result, pass
    skeleton= instead of skeleton_branches=, or split the two calls across a try boundary keep
    the explicit two-call form.
    """
    local_correspondence = mesh_correspondence_first_pass(mesh=mesh, skeleton_branches=skeleton_branches)
    return correspondence_1_to_1(mesh=mesh, local_correspondence=local_correspondence, **one_to_one_kwargs)


def filter_soma_touching_vertices_dict_by_mesh(mesh,
                                              curr_piece_to_soma_touching_vertices,
                                              verbose=True):
    """
    Purpose: Will take the soma to touching vertics
    and filter it for only those that touch the particular mesh piece

    Pseudocode:
    1) Build a KDTree of the mesh
    2) Create an output dictionary to store the filtered soma touching vertices
    For the original soma touching vertices, iterating through all the somas
        For each soma_touching list:
            Query the mesh KDTree and only keep the coordinates whose distance is equal to 0

    If empty dictionary then return None? (have option for this)
    
    Ex: 
    return_value = filter_soma_touching_vertices_dict_by_mesh(
    mesh = mesh_pieces_for_MAP[0],
    curr_piece_to_soma_touching_vertices = piece_to_soma_touching_vertices[1]
    )

    """
    
    if curr_piece_to_soma_touching_vertices is None:
        if verbose:
            print("In filter_soma_touching_vertices_dict_by_mesh: curr_piece_to_soma_touching_vertices was None so returning none")
        return None

    #1) Build a KDTree of the mesh
    curr_mesh_tree = KDTree(mesh.vertices)

    #2) Create an output dictionary to store the filtered soma touching vertices
    output_soma_touching_vertices = dict()

    for sm_idx,border_verts_list in curr_piece_to_soma_touching_vertices.items():
        for b_verts in border_verts_list:
            dist,closest_nodes = curr_mesh_tree.query(b_verts)
            match_verts = b_verts[dist==0]
            if len(match_verts)>0:
                if sm_idx not in output_soma_touching_vertices.keys():
                    output_soma_touching_vertices[sm_idx] = []
                output_soma_touching_vertices[sm_idx].append(match_verts)
    if len(output_soma_touching_vertices) == 0:
        return None
    else:
        return output_soma_touching_vertices
    
    
# ----------------- When refactoring the limb decomposition function ------ #

def find_if_stitch_point_on_end_or_branch(matched_branches_skeletons,
                                                              stitch_coordinate,
                                                              verbose=False):
                    
                    
                        # Step A: Find if stitch point is on endpt/branch point or in middle
                        stitch_point_on_end_or_branch = False
                        if len(matched_branches_skeletons) == 0:
                            raise Exception("No matching branches found for soma extending point")
                        elif len(matched_branches_skeletons)>1:
                            if verbose:
                                print(f"Multiple Branches for MP soma extending connection point {matched_branches_skeletons.shape}")
                            stitch_point_on_end_or_branch = True
                        else:# len(match_sk_branches)==1:
                            if verbose:
                                print(f"Only one Branch for MP soma Extending connection point {matched_branches_skeletons.shape}")
                            if len(nu.matching_rows(sk.find_branch_endpoints(matched_branches_skeletons[0]),
                                                    stitch_coordinate))>0:
                                stitch_point_on_end_or_branch =True

                        return stitch_point_on_end_or_branch
                    

def _split_branch_entry(branch_dict, old_key, first, second):
    """Replace branch_dict[old_key] with `first` and append `second` at max_key+1, then reorder
    keys; returns the reordered dict. The del+reassign-at-old_key preserves the original idiom
    exactly. Shared by attach_floating (main-branch cut) and _overwrite_stitched_entries (Part 16).
    """
    del branch_dict[old_key]
    branch_dict[old_key] = first
    branch_dict[np.max(list(branch_dict.keys())) + 1] = second
    return gu.order_dict_by_keys(branch_dict)


@dataclass
class _FloatingMatch:
    """The floating piece closest to any main limb, chosen for stitching this iteration."""
    winning_float: int
    match_main_limb: int
    dist: float
    main_stitch_point: object
    floating_stitch_point: object
    winning_main_skeleton: object


def _find_closest_floating_piece(limb_correspondence_cp, floating_limbs_skeleton_endpoints,
                                 floating_limbs_to_process):
    """Steps a-d: build each main limb's full skeleton, find (via KDTree) the minimum endpoint
    distance from every still-unprocessed floating piece to every main limb, and return the
    closest (float, main-limb) pair as a `_FloatingMatch`.
    """
    #a) Get full skeletons of limbs for all limbs in limb correspondence
    main_limb_skeletons = []
    for main_idx in np.sort(list(limb_correspondence_cp.keys())):
        main_limb_skeletons.append(sk.stack_skeletons([k["branch_skeleton"] for k in limb_correspondence_cp[main_idx].values()]))

    #b) Find the minimum distance (and the node it corresponds to) for each floating piece between their 
    #endpoints and all skeleton points of limbs 
    floating_piece_min_distance_all_main_limbs = dict([(float_idx,[]) for float_idx in floating_limbs_to_process])
    for main_idx,main_limb_sk in enumerate(main_limb_skeletons):

        main_skeleton_coordinates = sk.skeleton_unique_coordinates(main_limb_sk)
        main_kdtree = KDTree(main_skeleton_coordinates)

        for float_idx in floating_piece_min_distance_all_main_limbs.keys():

            dist,closest_node = main_kdtree.query(floating_limbs_skeleton_endpoints[float_idx])
            min_dist_idx = np.argmin(dist)
            min_dist = dist[min_dist_idx]
            min_dist_closest_node = main_skeleton_coordinates[closest_node[min_dist_idx]]
            floating_piece_min_distance_all_main_limbs[float_idx].append([min_dist,min_dist_closest_node,floating_limbs_skeleton_endpoints[float_idx][min_dist_idx]])


    winning_float = -1
    winning_float_match_main_limb = -1
    main_limb_stitch_point = None
    floating_limb_stitch_point = None
    winning_float_dist = np.inf


    #c) Find the floating piece that has the closest distance
    #--> winning piece

    #For the winning piece
    #d) Get the closest coordinate on the matching limb

    for f_idx,dist_data in floating_piece_min_distance_all_main_limbs.items():

        dist_data_array = np.array(dist_data)
        closest_main_limb = np.argmin(dist_data_array[:,0])
        closest_main_dist = dist_data_array[closest_main_limb][0]

        if closest_main_dist < winning_float_dist:

            winning_float = f_idx
            winning_float_match_main_limb = closest_main_limb
            winning_float_dist = closest_main_dist
            main_limb_stitch_point = dist_data_array[closest_main_limb][1]
            floating_limb_stitch_point = dist_data_array[closest_main_limb][2]

    winning_main_skeleton = main_limb_skeletons[winning_float_match_main_limb]

    return _FloatingMatch(
        winning_float=winning_float,
        match_main_limb=winning_float_match_main_limb,
        dist=winning_float_dist,
        main_stitch_point=main_limb_stitch_point,
        floating_stitch_point=floating_limb_stitch_point,
        winning_main_skeleton=winning_main_skeleton,
    )


def _preprocess_floating_pieces(floating_meshes, filter_end_node_length_meshparty, verbose):
    """Part 0/1: keep only floating meshes above the face threshold and run each through
    preprocess_limb (best-effort -- a piece that fails to decompose is skipped, CLASS-C guard).
    Returns the list of per-piece limb correspondences. Config read from parameters.params
    (the sole caller overrides none of it).
    """
    floating_piece_face_threshold = parameters.params.floating_piece_face_threshold
    size_threshold_MAP_stitch = parameters.params.size_threshold_MAP_stitch
    axon_width_preprocess_limb_max = parameters.params.axon_width_preprocess_limb_max
    use_adaptive_invalidation_d = parameters.params.use_adaptive_invalidation_d_floating
    mp_only_revised_invalidation_d = parameters.params.mp_only_revised_invalidation_d

    floating_limbs_above_threshold = [k for k in floating_meshes if len(k.faces)>floating_piece_face_threshold]

    #1) Run all significant floating pieces through preprocess_limb
    
    debug_corr = True
    if debug_corr:
        print(f"Starting the floating pieces preprocessing")
    
    #limb_cong = {}
    #with su.suppress_stdout_stderr():
    with su.suppress_stdout_stderr() if not debug_corr else su.dummy_context_mgr():
        
        floating_limbs_correspondence = []
        for j,k in enumerate(floating_limbs_above_threshold):
            if debug_corr:
                print(f"Floating {j}: {k}")
                st_time = time.time()
            
            neuron_params = dict(
                width_threshold_MAP=parameters.params.width_threshold_MAP,
                size_threshold_MAP=size_threshold_MAP_stitch,
                axon_width_preprocess_limb_max=axon_width_preprocess_limb_max,
                use_adaptive_invalidation_d=use_adaptive_invalidation_d,
                mp_only_revised_invalidation_d=mp_only_revised_invalidation_d,
                mp_only_invalidation_d_axon_buffer=parameters.params.mp_only_invalidation_d_axon_buffer,
                mp_only_revised_invalidation_d_reference=parameters.params.mp_only_revised_invalidation_d_reference,
                mp_only_revised_width_reference=parameters.params.mp_only_revised_width_reference,
                verbose=verbose,
            )
            limb_params = dict(
                invalidation_d=parameters.params.invalidation_d,
                smooth_neighborhood=1,
                combine_close_skeleton_nodes_threshold_meshparty=700,
                filter_end_node_length_meshparty=filter_end_node_length_meshparty,
                use_meshafterparty=True,
                use_adaptive_invalidation_d=use_adaptive_invalidation_d,
                axon_width_preprocess_limb_max=axon_width_preprocess_limb_max,
            )

            # CLASS-C guard: a floating piece's MAP skeletonization can leave a degenerate/empty
            # leftover submesh deep in mesh_subtraction_by_skeleton -> trimesh "too many indices for
            # array: array is 1-dimensional". Floating-piece stitching is best-effort, so skip a piece
            # that won't decompose instead of crashing the whole neuron. Everything downstream derives
            # from floating_limbs_correspondence, so not appending keeps all indices consistent.
            try:
                curr_corr = preprocess_limb(
                    mesh=k,
                    neuron_params=neuron_params,
                    limb_params=limb_params,
                    soma_touching_vertices_dict=None,
                    return_concept_network=False,
                    error_on_no_starting_coordinates=False,
                )
            except Exception as _float_err:
                print(f"[stitch] floating piece {j} (faces={len(k.faces)}) failed to decompose "
                      f"({type(_float_err).__name__}: {_float_err}); skipping it")
                continue
                
            floating_limbs_correspondence.append(curr_corr)
            
            
            if debug_corr:
                print(f"--> time = {time.time() - st_time}")
                st_time = time.time()
    return floating_limbs_correspondence


def _stitch_floating_piece_into_limb(limb_correspondence_cp, match, winning_floating_correspondence,
                                     excluded_node_coordinates, distance_to_move_point_threshold,
                                     verbose):
    """Steps e-k for the winning floating piece: snap the main-limb stitch point to a nearby
    end/branch node, find the main + floating branches, and (if the stitch would split a main
    branch) cut + re-correspond it, then graft the floating branches onto the main limb. Mutates
    limb_correspondence_cp in place. On a CLASS-B re-correspondence failure it returns early
    having mutated nothing; the caller marks the piece processed either way.
    """
    winning_float = match.winning_float
    winning_float_match_main_limb = match.match_main_limb
    winning_float_dist = match.dist
    main_limb_stitch_point = match.main_stitch_point
    floating_limb_stitch_point = match.floating_stitch_point
    winning_main_skeleton = match.winning_main_skeleton

    #e) Try and move closest coordinate to an endpoint or high degree node

    main_limb_stitch_point,change_status = sk.move_point_to_nearest_branch_end_point_within_threshold(
                                                        skeleton=winning_main_skeleton,
                                                        coordinate=main_limb_stitch_point,
                                                        distance_to_move_point_threshold = distance_to_move_point_threshold,
                                                        verbose=verbose,
                                                        consider_high_degree_nodes=True,
                                                        excluded_node_coordinates=excluded_node_coordinates

                                                        )
    if verbose:
        print(f"Status of Main limb stitch point moved = {change_status}")

#     #checking that match was right


    #f) Find the branch on the main limb that corresponds to the stitch point
    main_limb_branches = np.array([k["branch_skeleton"] for k in limb_correspondence_cp[winning_float_match_main_limb].values()])
    match_sk_branches = sk.find_branch_skeleton_with_specific_coordinate(main_limb_branches,
                        current_coordinate=main_limb_stitch_point)

    #g) Find whether the stitch point is on an endpoint/high degree node or will end up splitting the branch
    #AKA stitch_point_on_end_or_branch
    stitch_point_on_end_or_branch = find_if_stitch_point_on_end_or_branch(
                                                            matched_branches_skeletons= main_limb_branches[match_sk_branches],
                                                             stitch_coordinate=main_limb_stitch_point,
                                                              verbose=False)

    #h) Find the branch on the floating limb where the closest end point is
    winning_float_branches = np.array([k["branch_skeleton"] for k in winning_floating_correspondence.values()])
    match_float_branches = sk.find_branch_skeleton_with_specific_coordinate(winning_float_branches,
                        current_coordinate=floating_limb_stitch_point)

    if len(match_float_branches) > 1:
        raise Exception("len(match_float_branches) was greater than 1 in the floating pieces stitch")

    if verbose:
        print("\n\n")
        print(f"match_sk_branches = {match_sk_branches}")
        print(f"match_float_branches = {match_float_branches}")
        print(f"stitch_point_on_end_or_branch = {stitch_point_on_end_or_branch}")


    """
    Stitching process:
    i) if not stitch_point_on_end_or_branch
       1. cut the main limb branch where stitch is
       2. do mesh correspondence with the new stitches
       3. (just give the both the same old width)
       4. replace the old entry in the limb corresondence with one of the new skeleton cuts
          and add on the other skeletons cuts to the end

    j) Add a skeletal segment from floating limb stitch point to main limb stitch point
    k) Add the floating limb branches to the end of the limb correspondence
    l) Marks the floating piece as processed

    """

    # ---------- Begin stitching process ---------------
    if not stitch_point_on_end_or_branch:
        main_branch = match_sk_branches[0]
        #1. cut the main limb branch where stitch is
        matching_branch_sk = sk.cut_skeleton_at_coordinate(skeleton=main_limb_branches[main_branch],
                                                                   cut_coordinate = main_limb_stitch_point)
        #2. do mesh correspondence with the new stitchess
        stitch_mesh = limb_correspondence_cp[winning_float_match_main_limb][main_branch]["branch_mesh"]

        local_correspondnece = mesh_correspondence_first_pass(mesh=stitch_mesh,
                                                  skeleton_branches=matching_branch_sk)

        # CLASS-B guard: cutting the main branch + re-correspondence can fail when the cut
        # leaves a skeleton segment with no connected face patch (resolve_empty_conflicting_
        # face_labels -> "missing labels was not resolved"). Floating-piece stitching is
        # best-effort, so skip this piece instead of crashing the whole neuron. Safe to skip
        # here: nothing has been mutated yet (limb_correspondence_cp is only edited below).
        # We must mark the piece processed before `continue`, else the while loop reselects
        # the same winning_float forever.
        try:
            local_correspondence_revised = correspondence_1_to_1(mesh=stitch_mesh,
                                                        local_correspondence=local_correspondnece)
        except Exception as _stitch_err:
            print(f"[stitch] floating piece {winning_float} -> main limb "
                  f"{winning_float_match_main_limb} (dist {winning_float_dist:.1f}): cut-branch "
                  f"re-correspondence failed ({type(_stitch_err).__name__}: {_stitch_err}); "
                  f"skipping this floating piece")
            return  # nothing mutated yet; caller marks winning_float processed

        #3. (just give the both the same old width)
        old_width = limb_correspondence_cp[winning_float_match_main_limb][main_branch]["width_from_skeleton"]
        for branch_idx in local_correspondence_revised.keys():
            local_correspondence_revised[branch_idx]["width_from_skeleton"] = old_width

        #4. replace the old entry in the limb corresondence with one of the new skeleton cuts
        #and add on the other skeletons cuts to the end
        limb_correspondence_cp[winning_float_match_main_limb] = _split_branch_entry(
            limb_correspondence_cp[winning_float_match_main_limb], main_branch,
            local_correspondence_revised[0], local_correspondence_revised[1])


    #j) Add a skeletal segment from floating limb stitch point to main limb stitch point
    skeleton = winning_floating_correspondence[match_float_branches[0]]["branch_skeleton"]
    adjusted_floating_sk_branch = sk.stack_skeletons([skeleton,np.array([floating_limb_stitch_point,main_limb_stitch_point])])
            

    winning_floating_correspondence[match_float_branches[0]]["branch_skeleton"] = adjusted_floating_sk_branch

    #k) Add the floating limb branches to the end of the limb correspondence
    curr_limb_key_len = np.max(list(limb_correspondence_cp[winning_float_match_main_limb].keys()))
    for float_idx,flaot_data in winning_floating_correspondence.items():
        limb_correspondence_cp[winning_float_match_main_limb][curr_limb_key_len + 1 + float_idx] = flaot_data


def attach_floating_pieces_to_limb_correspondence(
        limb_correspondence,
        floating_meshes,
        distance_to_move_point_threshold = 4000,
        filter_end_node_length_meshparty = 1000,
        verbose = False,
        excluded_node_coordinates=np.array([]),
    **kwargs):
    """
    Purpose: To take a limb correspondence and add on the floating pieces
    that are significant and close enough to a limb

    Pseudocode:
    0) Filter the floating pieces for only those above certain face count
    1) Run all significant floating pieces through preprocess_limb
    2) Get all full skeleton endpoints (degree 1) for all floating pieces


    Start loop until all floating pieces have been added
    a) Get full skeletons of limbs for all limbs in limb correspondence
    b) Find the minimum distance (and the node it corresponds to) for each floating piece between their 
    endpoints and all skeleton points of limbs
    c) Find the floating piece that has the closest distance
    --> winning piece

    For the winning piece
    d) Get the closest coordinate on the matching limb
    e) Try and move closest coordinate to an endpoint or high degree node
    f) Find the branch on the main limb that corresponds to the stitch point
    g) Find whether the stitch point is on an endpoint/high degree node or will end up splitting the branch
    AKA stitch_point_on_end_or_branch
    h) Find the branch on the floating limb where the closest end point is

    At this point have
    - main limb stitch point and branches (and whether not splitting will be required)  [like MAP]
    - floating limb stitch point and branch [like MP]

    Stitching process:
    i) if not stitch_point_on_end_or_branch
    - cut the main limb branch where stitch is
    - do mesh correspondence with the new stitches
    - (just give the both the same old width)
    - replace the old entry in the limb corresondence with one of the new skeleton cuts
    and add on the other skeletons cuts to the end

    j) Add a skeletal segment from floating limb stitch point to main limb stitch point
    k) Add the floating limb branches to the end of the limb correspondence
    l) Marks the floating piece as processed


    """

    # max_stitch_distance is the only config still needed here; the rest moved into
    # _preprocess_floating_pieces. The sole caller (_stitch_floating_pieces) overrides none.
    max_stitch_distance = parameters.params.max_stitch_distance

    limb_correspondence_cp = limb_correspondence

    floating_limbs_correspondence = _preprocess_floating_pieces(
        floating_meshes, filter_end_node_length_meshparty, verbose)

    #2) Get all full skeleton endpoints (degree 1) for all floating pieces
    floating_limbs_skeleton = [sk.stack_skeletons([k["branch_skeleton"] for k in l_c.values()]) for l_c in floating_limbs_correspondence]
    floating_limbs_skeleton_endpoints = [sk.find_skeleton_endpoint_coordinates(k) for k in floating_limbs_skeleton]

    floating_limbs_to_process = np.arange(0,len(floating_limbs_skeleton))

    #Start loop until all floating pieces have been added
    while len(floating_limbs_to_process)>0:

        match = _find_closest_floating_piece(
            limb_correspondence_cp, floating_limbs_skeleton_endpoints, floating_limbs_to_process)
        winning_floating_correspondence = floating_limbs_correspondence[match.winning_float]

        if verbose:
            print(f"winning_float = {match.winning_float}")
            print(f"winning_float_match_main_limb = {match.match_main_limb}")
            print(f"winning_float_dist = {match.dist}")
            print(f"main_limb_stitch_point = {match.main_stitch_point}")
            print(f"floating_limb_stitch_point = {match.floating_stitch_point}")

        #c/d) if the closest piece is farther than the max, stop stitching entirely
        if match.dist > max_stitch_distance:
            print(f"The closest float distance was {match.dist} which was greater than the maximum stitch distance {max_stitch_distance}\n"
                 " --> so ending the floating mesh stitch processs")
            return limb_correspondence_cp

        _stitch_floating_piece_into_limb(
            limb_correspondence_cp, match, winning_floating_correspondence,
            excluded_node_coordinates, distance_to_move_point_threshold, verbose)

        #l) Marks the floating piece as processed
        floating_limbs_to_process = np.setdiff1d(floating_limbs_to_process,[match.winning_float])

    return limb_correspondence_cp


def calculate_limb_concept_networks(limb_correspondence,
                                    network_starting_info,
                                   run_concept_network_checks=True,
                                   verbose=False):
    """
    Can take a limb correspondence and the starting vertices and endpoints
    and create a list of concept networks organized by 
    [soma_idx] --> list of concept networks 
                    (because could possibly have mulitple starting points on the same soma)
    
    """
    
    limb_correspondence_individual = limb_correspondence
    divided_skeletons = np.array([limb_correspondence_individual[k]["branch_skeleton"] for k in np.sort(list(limb_correspondence_individual.keys()))])


    # -------------- Part 18: Getting Concept Networks  [soma_idx] --> list of concept networks -------#

    """
    Concept Network Pseudocode:

    Step 0: Compile the Limb correspondence into final form

    Make sure these have the same list
    limb_to_soma_touching_vertices_list,limb_to_endpoints_must_keep_list

    For every dictionary in zip(limb_to_soma_touching_vertices_list,
                            limb_to_endpoints_must_keep_list):

        #make sure the dicts have same keys
        For every key (represents the soma) in the dictionary:

            #make sure the lists have the same sizes
            For every item in the list (which would be a list of endpoints or list of groups of vertexes):
                #At this point have the soma, the endpoint and the touching vertices

                1) find the branch with the endoint that must keep
                    --> if multiple endpoints then error
                2) Call the branches_to_concept_network with the
                   a. divided skeletons
                   b. closest endpoint
                   c. endpoints of branch (from the branch found)
                   d. touching soma vertices

                3) Run the checks on the concept network

    """

    
    limb_to_soma_concept_networks = dict()

    for soma_idx in network_starting_info.keys():
        
        if soma_idx not in list(limb_to_soma_concept_networks.keys()):
            limb_to_soma_concept_networks[soma_idx] = []
        for soma_group_idx,st_dict in network_starting_info[soma_idx].items():
            t_verts = st_dict["touching_verts"]
            endpt = st_dict["endpoint"]
            if verbose:
                print(f"\n\n---------Working on soma_idx = {soma_idx}, soma_group_idx {soma_group_idx}, endpt = {endpt}---------")


            #1) find the branch with the endoint that must keep
            # ---------------- 11/17 Addition: If the endpoint does not match a skeleton point anymore then just get the closest endpoint of mesh that has touching vertices

            start_branch = sk.find_branch_skeleton_with_specific_coordinate(divded_skeleton=divided_skeletons,
                                                            current_coordinate=endpt)[0]


            start_branch_endpoints = sk.find_branch_endpoints(divided_skeletons[start_branch])

            #2) Call the branches_to_concept_network with the
            curr_limb_concept_network = nru.branches_to_concept_network(curr_branch_skeletons=divided_skeletons,
                                                                  starting_coordinate=endpt,
                                                                  starting_edge=start_branch_endpoints,
                                                                  touching_soma_vertices=t_verts,
                                                                       soma_group_idx=soma_group_idx,
                                                                       verbose=verbose)
            if verbose:
                print("Done generating concept network \n\n")


            if run_concept_network_checks:
                #3) Run the checks on the concept network
                #3.1: check to make sure the starting coordinate was recovered

                recovered_touching_piece = xu.get_nodes_with_attributes_dict(curr_limb_concept_network,dict(starting_coordinate=endpt))

                if verbose:
                    print(f"recovered_touching_piece = {recovered_touching_piece}")
                if recovered_touching_piece[0] != start_branch:
                    raise Exception(f"For limb and soma {soma_idx} the recovered_touching and original touching do not match\n"
                                   f"recovered_touching_piece = {recovered_touching_piece}, original_touching_pieces = {start_branch}")


                #3.2: Check number of nodes match the number of divided skeletons
                if len(curr_limb_concept_network.nodes()) != len(divided_skeletons):
                    raise Exception("The number of nodes in the concept graph and number of branches passed to it did not match\n"
                                  f"len(curr_limb_concept_network.nodes())={len(curr_limb_concept_network.nodes())}, len(curr_limb_divided_skeletons)= {len(divided_skeletons)}")

                #3.3: Check that concept network is a connected component
                if nx.number_connected_components(curr_limb_concept_network) > 1:
                    raise Exception("There was more than 1 connected components in the concept network")


                #3.4 Make sure the oriiginal divided skeleton endpoints match the concept map endpoints
                for j,un_resized_b in enumerate(divided_skeletons):
                    """
                    Pseudocode: 
                    1) get the endpoints of the current branch
                    2) get the endpoints in the concept map
                    3) compare
                    - if not equalt then break
                    """
                    #1) get the endpoints of the current branch
                    b_endpoints = neuron.Branch(un_resized_b).endpoints
                    #2) get the endpoints in the concept map
                    graph_endpoints = xu.get_node_attributes(curr_limb_concept_network,attribute_name="endpoints",node_list=[j])[0]
                    if not xu.compare_endpoints(b_endpoints,graph_endpoints):
                        raise Exception(f"The node {j} in concept graph endpoints do not match the endpoints of the original branch\n"
                                       f"original_branch_endpoints = {b_endpoints}, concept graph node endpoints = {graph_endpoints}")

            limb_to_soma_concept_networks[soma_idx].append(curr_limb_concept_network)

    return limb_to_soma_concept_networks                    


def filter_limb_correspondence_for_end_nodes(limb_correspondence,
                                             mesh,
                                             starting_info=None,
                                             filter_end_node_length=4000,
                                             error_on_no_starting_coordinates = True,
                                             plot_new_correspondence = False,
                                             error_on_starting_coordinates_not_endnodes = True,
                                            verbose = True,
                                             
                                             
                                            ):
    """
    Pseudocode:
    1) Get all of the starting coordinates
    2) Assemble the entire skeleton and run the skeleton cleaning process
    3) Decompose skeleton into branches and find out mappin gof old branches to new ones
    4) Assemble the new width and mesh face idx for all new branches
    - width: do weighted average by skeletal length
    - face_idx: concatenate
    5) Make face_lookup and Run waterfilling algorithm to fill in rest
    6) Get the divided meshes and face idx from waterfilling
    7) Store everything back inside a correspondence dictionary
    """

    limb_correspondence_individual=limb_correspondence
    limb_mesh_mparty = mesh
    network_starting_info_revised_cleaned = starting_info


    lc_skeletons = [v["branch_skeleton"] for v in limb_correspondence_individual.values()]
    lc_branch_meshes = [v["branch_mesh"] for v in limb_correspondence_individual.values()]
    lc_branch_face_idx = [v["branch_face_idx"] for v in limb_correspondence_individual.values()]
    lc_width_from_skeletons = [v["width_from_skeleton"] for v in limb_correspondence_individual.values()]

    #1) Get all of the starting coordinates
                
    if starting_info is not None:
        all_starting_coords = nru.all_soma_connnecting_endpionts_from_starting_info(network_starting_info_revised_cleaned)
    else:
        all_starting_coords = []
                
    
    if error_on_no_starting_coordinates:
        if len(all_starting_coords) == 0:
            raise Exception(f"No starting coordinates found: network_starting_info_revised_cleaned = {network_starting_info_revised_cleaned} ")

    # ---------- 1/5/2021: Will check all starting points as end nodes and if not all degree 1 then error --------#
    

    #2) Assemble the entire skeleton and run the skeleton cleaning process
    curr_limb_sk_cleaned,rem_branches = sk.clean_skeleton(sk.stack_skeletons(lc_skeletons),
                         distance_func=sk.skeletal_distance,
                         min_distance_to_junction=filter_end_node_length,
                        endpoints_must_keep=all_starting_coords,
                         return_skeleton=True,
                         print_flag=False,
                        return_removed_skeletons=True,
                        error_if_endpoints_must_keep_not_endnode=error_on_starting_coordinates_not_endnodes)

    if verbose:
        print(f"Removed {len(rem_branches)} skeletal branches")

    #3) Decompose skeleton into branches and find out mappin gof old branches to new ones
    cleaned_branches = sk.decompose_skeleton_to_branches(curr_limb_sk_cleaned)
    
    if len(cleaned_branches) == 0:
        print("There were no branches after cleaning limb correspondence")
        return limb_correspondence
    
    original_br_mapping = sk.map_between_branches_lists(lc_skeletons,cleaned_branches)

    # 4) Assemble the new width and mesh face idx for all new branches
    # - width: do weighted average by skeletal length
    # - face_idx: concatenate

    new_width_from_skeletons = []
    new_branch_face_idx = []

    for j,cl_b in enumerate(cleaned_branches):
        or_idx = np.where(original_br_mapping==j)[0]

        #doing the width
        total_skeletal_length = 0
        weighted_width = 0
        if len(or_idx) > 1:
            for oi in or_idx:
                curr_sk_len = sk.calculate_skeleton_distance(lc_skeletons[oi])
                weighted_width += curr_sk_len*lc_width_from_skeletons[oi]
                total_skeletal_length+=curr_sk_len
            final_width = weighted_width/total_skeletal_length
            new_width_from_skeletons.append(final_width)
        else:
            new_width_from_skeletons.append(lc_width_from_skeletons[or_idx[0]])


        #doing the face_idx
        new_branch_face_idx.append(np.concatenate([lc_branch_face_idx[k] for k in or_idx]))


    #5) Make face_lookup and Run waterfilling algorithm to fill in rest

    face_lookup = {j:[] for j in range(0,len(limb_mesh_mparty.faces))}
    face_lookup_marked = gu.invert_mapping(new_branch_face_idx)
    ky = list(face_lookup.keys())
    if verbose:
        print(f"For marked faces: {print(np.max(ky),len(ky))}")
    face_lookup.update(face_lookup_marked)


    original_labels = np.arange(0,len(cleaned_branches))

    face_coloring_copy = cu.resolve_empty_conflicting_face_labels(curr_limb_mesh = limb_mesh_mparty,
                                                                    face_lookup=face_lookup,
                                                                    no_missing_labels = list(original_labels))


    #6) Get the divided meshes and face idx from waterfilling
    # -- splitting the mesh pieces into individual pieces
    divided_submeshes,divided_submeshes_idx = tu.split_mesh_into_face_groups(limb_mesh_mparty,face_coloring_copy,
                                                                            return_dict=False)


    #7) Store everything back inside a correspondence dictionary
    limb_correspondence_individual_filtered = dict()
    for j,curr_sk in enumerate(cleaned_branches):
        local_dict = dict(branch_skeleton=curr_sk,
                          width_from_skeleton=new_width_from_skeletons[j],
                         branch_mesh=divided_submeshes[j],
                         branch_face_idx=divided_submeshes_idx[j])
        limb_correspondence_individual_filtered[j] = local_dict


    if plot_new_correspondence:
        plot_limb_correspondence(limb_correspondence_individual_filtered)

    return limb_correspondence_individual_filtered
def _new_invalidation_d(
    width_median,
    lowest_value,
    width_ref,
    inv_d_ref,
    ax_width,
    max_invalidation_d,
    verbose=False,
):
    """
    Новый invalidation_d по ширине (линейная интерполяция между параметрами).
        slope     = (inv_d_ref - lowest_value) / (width_ref - ax_width)
        new_inv_d = slope * (width - width_ref) + inv_d_ref
    """
    change_x = width_ref - ax_width
    if change_x == 0:
        return max_invalidation_d

    slope = (inv_d_ref - lowest_value) / change_x
    new_d = slope * (width_median - width_ref) + inv_d_ref
    final_d = max(min(new_d, max_invalidation_d), lowest_value)

    if verbose:
        print(f"new_invalidation_d = {new_d} (max = {max_invalidation_d}), final = {final_d}")
    return final_d


def _run_skeletonization_pass(
    limb_mesh_mparty,
    root_curr,
    meshparty_segment_size,
    limb_cfg,
    verbose=False,
):
    """Один проход: скелетонизация меша + разбиение на ветви.

    Возвращает (segment_branches, divided_submeshes, divided_submeshes_idx,
                segment_widths_median).
    """
    sk_obj = m_sk.skeletonize_mesh_largest_component(
        limb_mesh_mparty,
        root=root_curr,
        invalidation_d=limb_cfg["invalidation_d"],
        smooth_neighborhood=limb_cfg["smooth_neighborhood"],
        filter_mesh=False,
    )

    if verbose:
        print(f"meshparty_segment_size = {meshparty_segment_size}")

    return m_sk.skeleton_obj_to_branches(
        sk_obj,
        mesh=limb_mesh_mparty,
        meshparty_segment_size=meshparty_segment_size,
        combine_close_skeleton_nodes_threshold=limb_cfg["combine_close_skeleton_nodes_threshold_meshparty"],
        filter_end_node_length=limb_cfg["filter_end_node_length_meshparty"],
    )
def _decide_next_limb_cfg(
    segment_branches,
    segment_widths_median,
    limb_cfg,
    neuron_cfg,
    verbose=False,
):
    """Решает, нужен ли ещё один проход, и с какими параметрами.

    Возвращает новый limb_cfg для повторного прогона, либо None — значит
    текущий результат финальный.
    """
    width_median = m_sk.width_median_weighted(segment_widths_median, segment_branches)
    pieces_above_threshold = np.where(
        segment_widths_median > neuron_cfg["width_threshold_MAP"]
    )[0]

    if verbose:
        print(f"width_median = {width_median}")
        print(f"segment_widths_median = {segment_widths_median}")
        print(f"# pieces_above_threshold = {len(pieces_above_threshold)}")

    p = parameters.params

    # 1. Тонкая ветвь → параметры аксона
    if width_median <= neuron_cfg["axon_width_preprocess_limb_max"]:
        if verbose:
            print("Using the axon parameters")
        return {
            **limb_cfg,
            "combine_close_skeleton_nodes_threshold_meshparty": p.combine_close_skeleton_nodes_threshold_meshparty_axon,
            "filter_end_node_length_meshparty": p.filter_end_node_length_meshparty_axon,
            "filter_end_node_length": p.filter_end_node_length_axon,
            "invalidation_d": p.invalidation_d_axon,
            "smooth_neighborhood": p.smooth_neighborhood_axon,
        }

    # 2. Нет MAP-кусков → пересчёт invalidation_d (интерполяция по ширине)
    if (
        len(pieces_above_threshold) == 0
        and neuron_cfg["mp_only_revised_invalidation_d"]
        and limb_cfg["invalidation_d"] != p.invalidation_d_axon
    ):
        if verbose:
            print("Using MP only revised invalidation")
        return {
            **limb_cfg,
            "invalidation_d": _new_invalidation_d(
                width_median,
                lowest_value=p.invalidation_d_axon + neuron_cfg["mp_only_invalidation_d_axon_buffer"],
                width_ref=neuron_cfg["mp_only_revised_width_reference"],
                inv_d_ref=neuron_cfg["mp_only_revised_invalidation_d_reference"],
                ax_width=neuron_cfg["axon_width_preprocess_limb_max"],
                max_invalidation_d=limb_cfg["invalidation_d"],
                verbose=True,
            ),
        }

    # 3. Есть MAP-куски → финал
    return None
def _cycle_for_something(
    limb_mesh_mparty,
    root_curr,
    meshparty_segment_size,
    neuron_cfg,
    limb_cfg,
):
    verbose = neuron_cfg.get("verbose", False)

    result = _run_skeletonization_pass(
        limb_mesh_mparty, root_curr, meshparty_segment_size, limb_cfg, verbose,
    )

    if not neuron_cfg["use_adaptive_invalidation_d"]:
        return result

    new_limb_cfg = _decide_next_limb_cfg(
        result[0], result[3], limb_cfg, neuron_cfg, verbose,
    )
    if new_limb_cfg is None:
        return result  # первый проход финальный

    # ровно один повторный прогон с пересчитанными параметрами
    return _run_skeletonization_pass(
        limb_mesh_mparty, root_curr, meshparty_segment_size, new_limb_cfg, verbose,
    )

def _build_filtered_connectivity(
    sublimb_meshes_MP,
    sublimb_meshes_MAP,
    limb_mesh_mparty,
    connectivity_type,
):
    """Один проход: связность мешей + фильтрация рёбер MP→MAP.

    Возвращает (mesh_conn, mesh_conn_vertex_groups, mesh_conn_old).
    """
    mesh_conn, mesh_conn_vertex_groups = tu.mesh_list_connectivity(
        meshes=sublimb_meshes_MP + sublimb_meshes_MAP,
        main_mesh=limb_mesh_mparty,
        connectivity=connectivity_type,
        min_common_vertices=1,
        return_vertex_connection_groups=True,
        return_largest_vertex_connection_group=True,
        print_flag=False,
    )
    mesh_conn_old = copy.deepcopy(mesh_conn)

    n_mp = len(sublimb_meshes_MP)
    mesh_conn_filt = []
    mesh_conn_vertex_groups_filt = []
    for j, (m1, m2) in enumerate(mesh_conn):
        # ребро должно идти от MP-куска к MAP-куску
        if m1 < n_mp and m2 >= n_mp:
            mesh_conn_filt.append([m1, m2])
            mesh_conn_vertex_groups_filt.append(mesh_conn_vertex_groups[j])
        else:
            print(f"Edge {(m1, m2)} was not kept")

    return np.array(mesh_conn_filt), mesh_conn_vertex_groups_filt, mesh_conn_old
def _connectivity_is_valid(mesh_conn, n_mp, n_map):
    """Проверяет, что граф связности — один компонент со всеми узлами.

    Возвращает True/False; не бросает исключений.
    """
    G = nx.from_edgelist(mesh_conn)
    if len(G) != n_mp + n_map:
        return False
    return len(list(nx.connected_components(G))) == 1

def _resolve_mesh_connectivity(
    sublimb_meshes_MP,
    sublimb_meshes_MAP,
    limb_mesh_mparty,
    sublimb_skeletons_MP,
    sublimb_skeletons_MAP,
    connectivity_type,
):
    n_mp = len(sublimb_meshes_MP)
    n_map = len(sublimb_meshes_MAP)

    def attempt(conn_type):
        return _build_filtered_connectivity(
            sublimb_meshes_MP, sublimb_meshes_MAP, limb_mesh_mparty, conn_type,
        )

    # ── Попытка 1: текущий connectivity_type ──────────────────────────
    mesh_conn, mesh_conn_vertex_groups, mesh_conn_old = attempt(connectivity_type)

    if _connectivity_is_valid(mesh_conn, n_mp, n_map):
        print(f"Successful mesh connectivity with type {connectivity_type}")
        return mesh_conn, mesh_conn_vertex_groups, connectivity_type

    # уже на "vertices" — дальше отступать некуда, это ошибка
    if connectivity_type == "vertices":
        raise Exception("Something went wrong in the connectivity")

    # ── Попытка 2: переключаемся на "vertices" ───────────────────────
    print(f"Failed on connection type {connectivity_type}")
    connectivity_type = "vertices"
    print(f"so changing type to {connectivity_type}")

    mesh_conn, mesh_conn_vertex_groups, mesh_conn_old = attempt(connectivity_type)

    if _connectivity_is_valid(mesh_conn, n_mp, n_map):
        print(f"Successful mesh connectivity with type {connectivity_type}")
        return mesh_conn, mesh_conn_vertex_groups, connectivity_type

    # "vertices" тоже не сошёлся — финальная ошибка
    raise Exception("Something went wrong in the connectivity")


def _decompose_map_piece(
    map_sub,
    sublimb_idx,
    soma_touching_vertices_dict,
    filter_end_node_length,
    perform_cleaning_checks,
    combine_close_skeleton_nodes,
    combine_close_skeleton_nodes_threshold,
    use_surface_after_CGAL,
    surface_reconstruction_size,
    remove_mesh_interior_face_threshold,
    error_on_bad_cgal_return,
    max_stitch_distance_CGAL,
    distance_by_mesh_center,
):
    """Part 9 (один MAP-кусок): CGAL-скелетонизация + mesh correspondence.

    Возвращает (local_correspondence_revised, curr_limb_endpoints_must_keep,
                curr_soma_to_piece_touching_vertices). Списки must_keep/touching
        обновляет вызывающий цикл (чтобы helper был без побочных эффектов).
    """
    print(f"--- Working on MAP piece {sublimb_idx}---")
    mesh_start_time = time.time()
    # Phase B: map_sub is a submesh_ops.SubMesh (mesh + limb-frame face_idx + parent=limb_mesh_mparty).
    # The rest of the body uses `mesh` unchanged; the :1399 remap now composes off map_sub's provenance.
    mesh = map_sub.mesh
    curr_soma_to_piece_touching_vertices = filter_soma_touching_vertices_dict_by_mesh(
        mesh=mesh,
        curr_piece_to_soma_touching_vertices=soma_touching_vertices_dict,
    )

    # ---- 0) Generating the Clean skeletons  -------------------------------------------#
    if curr_soma_to_piece_touching_vertices is not None:
        curr_total_border_vertices = dict([(k, np.vstack(v)) for k, v in curr_soma_to_piece_touching_vertices.items()])
    else:
        curr_total_border_vertices = None

    # The C++ CGAL skeletonizer (calcification_param) refuses a non-closed mesh: it returns 4
    # ("Not closed mesh"), writes no .cgal file, and skeleton_utils then SILENTLY falls back to
    # meshparty ("..._skeleton.cgal not found so skipping" -> meshparty skeletonization). That swap
    # is harmless for decomposition (meshparty carries ~all skeletonization on H01 — typically a
    # single MAP piece per neuron), but it is invisible in the logs, so a real CGAL regression would
    # hide behind it. Surface the fallback loudly: CGAL contributes here ONLY if the piece is closed.
    if not tu.is_watertight(mesh):
        print(f"[MAP {sublimb_idx}] mesh not watertight -> CGAL skeletonizer will return 4 "
              f"('Not closed mesh') and SILENTLY fall back to meshparty skeletonization "
              f"(faces={len(mesh.faces)}). This is expected on H01; flagged so a genuine CGAL "
              f"failure is not mistaken for it.")

    try:
        cleaned_branch, curr_limb_endpoints_must_keep = sk.skeletonize_and_clean_connected_branch_CGAL(
            mesh=mesh,
            curr_soma_to_piece_touching_vertices=curr_soma_to_piece_touching_vertices,
            total_border_vertices=curr_total_border_vertices,
            filter_end_node_length=filter_end_node_length,
            perform_cleaning_checks=perform_cleaning_checks,
            combine_close_skeleton_nodes=combine_close_skeleton_nodes,
            combine_close_skeleton_nodes_threshold=combine_close_skeleton_nodes_threshold,
            use_surface_after_CGAL=use_surface_after_CGAL,
            surface_reconstruction_size=surface_reconstruction_size,
            remove_mesh_interior_face_threshold=remove_mesh_interior_face_threshold,
            error_on_bad_cgal_return=error_on_bad_cgal_return,
            max_stitch_distance=max_stitch_distance_CGAL,
        )
    except Exception as e:
        # CLASS-C legibility: skeletonization of this MAP piece blew up (e.g. mesh_subtraction_by_
        # skeleton leaves a degenerate/empty submesh -> trimesh "too many indices for array").
        # Report the piece geometry that triggered it; re-raise unchanged (no behavior change).
        try:
            _wt = tu.is_watertight(mesh)
        except Exception:
            _wt = "?"
        print(f"[MAP {sublimb_idx}] skeletonize_and_clean_connected_branch_CGAL FAILED "
              f"(faces={len(mesh.faces)}, vertices={len(mesh.vertices)}, watertight={_wt}) "
              f"-> {type(e).__name__}: {e}")
        raise

    if curr_limb_endpoints_must_keep is None:
        print("Inside MAP decomposition and curr_limb_endpoints_must_keep was None")

    if len(cleaned_branch) == 0:
        raise Exception(f"Found a zero length skeleton for MAP piece {sublimb_idx}")

    # ---- 1) Generating Initial Mesh Correspondence -------------------------------------------#
    print(f"Working on limb correspondence for #{sublimb_idx} MAP piece")
    local_correspondence = mesh_correspondence_first_pass(
        mesh=mesh,
        skeleton=cleaned_branch,
        distance_by_mesh_center=distance_by_mesh_center,
        connectivity="edges",
        remove_inside_pieces_threshold=100,
    )

    #------------- 2) Doing Some checks on the initial corespondence -------- #
    if perform_cleaning_checks:
        check_skeletonization_and_decomp(skeleton=cleaned_branch, local_correspondence=local_correspondence)

    # -------3) Finishing off the face correspondence so get 1-to-1 correspondence of mesh face to skeletal piece
    # Phase B: pass map_sub as input_sub so branch_face_idx comes back in the LIMB frame by
    # construction (the old :1399 mesh_idx[branch_face_idx] remap now lives inside correspondence_1_to_1).
    local_correspondence_revised = correspondence_1_to_1(
        mesh=mesh,
        local_correspondence=local_correspondence,
        curr_limb_endpoints_must_keep=curr_limb_endpoints_must_keep,
        curr_soma_to_piece_touching_vertices=curr_soma_to_piece_touching_vertices,
        input_sub=map_sub,
    )

    print(f"Total time for MAP sublimb #{sublimb_idx} mesh processing = {time.time() - mesh_start_time}")
    return local_correspondence_revised, curr_limb_endpoints_must_keep, curr_soma_to_piece_touching_vertices


def _fix_mp_soma_extension(
    segment_branches,
    divided_submeshes,
    divided_submeshes_idx,
    segment_widths_median,
    curr_soma_to_piece_touching_vertices,
    limb_mesh_mparty,
):
    """Part 10 (11/9 addition): для MP-сублимба, касающегося границы сомы,
    достраивает soma-extending ветви и пересчитывает correspondence.

    Мутирует/возвращает (segment_branches, divided_submeshes, divided_submeshes_idx,
    segment_widths_median) плюс endpts_total, touching_total и флаг
    no_soma_extension_add (для диагностического принта в вызывающем цикле).
    """
    no_soma_extension_add = True

    endpts_total = dict()
    curr_soma_to_piece_touching_vertices_total = dict()
    for sm_idx, sm_bord_verts_list in curr_soma_to_piece_touching_vertices.items():
        #will be used for later
        endpts_total[sm_idx] = []
        curr_soma_to_piece_touching_vertices_total[sm_idx] = []

        for sm_bord_verts in sm_bord_verts_list:
            #1) Get the mesh pieces that are touching the border
            matching_mesh_idx = tu.filter_meshes_by_containing_coordinates(
                mesh_list=divided_submeshes,
                nullifying_points=sm_bord_verts,
                filter_away=False,
                distance_threshold=min_distance_threshold,
                return_indices=True,
            )
            #2) concatenate all meshes and skeletons that are touching
            if len(matching_mesh_idx) <= 0:
                raise Exception("None of branches were touching the border vertices when fixing MP pieces")

            touch_mesh = tu.combine_meshes(divided_submeshes[matching_mesh_idx])
            touch_sk = sk.stack_skeletons(segment_branches[matching_mesh_idx])

            local_curr_soma_to_piece_touching_vertices = {sm_idx: [sm_bord_verts]}
            new_sk, endpts, new_branch_info = sk.create_soma_extending_branches(
                current_skeleton=touch_sk,
                skeleton_mesh=touch_mesh,
                soma_to_piece_touching_vertices=local_curr_soma_to_piece_touching_vertices,
                return_endpoints_must_keep=True,
                return_created_branch_info=True,
                check_connected_skeleton=False,
            )

            # ---- 12/30 Addition Check if the endpoint found is an endnode or not and if not then manually add branch ---
            curr_endnode = endpts[sm_idx][0]
            match_sk_branches = sk.find_branch_skeleton_with_specific_coordinate(
                segment_branches, current_coordinate=curr_endnode
            )

            if len(match_sk_branches) > 1:
                border_average_coordinate = np.mean(sm_bord_verts, axis=0)
                new_branch_sk = np.vstack([curr_endnode, border_average_coordinate]).reshape(-1, 2, 3)
                br_info = dict(new_branch=new_branch_sk, border_verts=sm_bord_verts)
                endpts_total[sm_idx].append(border_average_coordinate)
            else:
                br_info = new_branch_info[sm_idx][0]
                endpts_total[sm_idx].append(endpts[sm_idx][0])
            # -------------------- End of 12/30 Addition ------------------

            #3) Add the info to the new running lists
            curr_soma_to_piece_touching_vertices_total[sm_idx].append(sm_bord_verts)

            #4) Skip if no new branch was added
            if br_info is None:
                print("The new branch info was none so skipping \n")
                continue

            #4 If new branch was made then
            no_soma_extension_add = False

            #1) Get the newly added branch (and the original vertex which is the first row)
            br_new, sm_bord_verts = br_info["new_branch"], br_info["border_verts"]

            curr_soma_to_piece_touching_vertices_MP = {sm_idx: [sm_bord_verts]}
            endpoints_must_keep_MP = {sm_idx: [br_new[0][1]]}

            orig_vertex = br_new[0][0]

            #2) Find the branches that have that coordinate (could be multiple)
            match_sk_branches = sk.find_branch_skeleton_with_specific_coordinate(
                segment_branches, current_coordinate=orig_vertex
            )

            stitch_point_on_end_or_branch = find_if_stitch_point_on_end_or_branch(
                matched_branches_skeletons=segment_branches[match_sk_branches],
                stitch_coordinate=orig_vertex,
                verbose=False,
            )

            if not stitch_point_on_end_or_branch:
                matching_branch_sk = sk.cut_skeleton_at_coordinate(
                    skeleton=segment_branches[match_sk_branches][0], cut_coordinate=orig_vertex
                )
            else:
                matching_branch_sk = segment_branches[match_sk_branches]

            #3) Find the mesh and skeleton of the winning branch
            matching_branch_mesh_idx = np.array(divided_submeshes_idx)[match_sk_branches]
            extend_soma_mesh_idx = np.concatenate(matching_branch_mesh_idx)
            extend_soma_mesh = limb_mesh_mparty.submesh([extend_soma_mesh_idx], append=True, repair=False)
            # Phase B: the correspondence-input mesh as an explicit SubMesh of the limb; passed as
            # input_sub to _run_mesh_correspondence so branch_face_idx comes back in the LIMB frame
            # (the old extend_soma_mesh_idx[...] remap now lives inside correspondence_1_to_1).
            extend_soma_sub = submesh_ops.SubMesh(extend_soma_mesh, extend_soma_mesh_idx, limb_mesh_mparty)

            #4) Add newly created branch to skeleton and divide the skeleton into branches (could make 2 or 3)
            sk.check_skeleton_connected_component(sk.stack_skeletons(list(matching_branch_sk) + [br_new]))

            #5) Run Adaptive mesh correspondnece using branches and mesh
            # GETTING MESHES THAT ARE NOT FULLY CONNECTED!!
            local_correspondence_revised = _run_mesh_correspondence(
                extend_soma_mesh,
                list(matching_branch_sk) + [br_new],
                curr_limb_endpoints_must_keep=endpoints_must_keep_MP,
                curr_soma_to_piece_touching_vertices=curr_soma_to_piece_touching_vertices_MP,
                input_sub=extend_soma_sub,
            )

            new_submeshes = [k["branch_mesh"] for k in local_correspondence_revised.values()]
            # branch_face_idx is already in the limb frame (input_sub above), so no remap here.
            new_submeshes_idx = [k["branch_face_idx"] for k in local_correspondence_revised.values()]
            new_skeletal_branches = [k["branch_skeleton"] for k in local_correspondence_revised.values()]

            #calculate the new width
            ray_inter = tu.ray_pyembree.RayMeshIntersector(limb_mesh_mparty)
            new_widths = []
            for new_s_idx in new_submeshes_idx:
                curr_ray_distance = tu.ray_trace_distance(mesh=limb_mesh_mparty, face_inds=new_s_idx, ray_inter=ray_inter)
                curr_width_median = np.median(curr_ray_distance[curr_ray_distance != 0])
                if (not np.isnan(curr_width_median)) and (curr_width_median > 0):
                    new_widths.append(curr_width_median)
                else:
                    print(f"USING A DEFAULT WIDTH BECAUSE THE NEWLY COMPUTED ONE WAS {curr_width_median}: {segment_widths_median[match_sk_branches[0]]}")
                    new_widths.append(segment_widths_median[match_sk_branches[0]])

            segment_branches = np.array([k for i, k in enumerate(segment_branches) if i not in match_sk_branches] + new_skeletal_branches)

            divided_submeshes = np.delete(divided_submeshes, match_sk_branches, axis=0)
            divided_submeshes = np.append(divided_submeshes, new_submeshes, axis=0)

            divided_submeshes_idx = np.array([k for i, k in enumerate(divided_submeshes_idx) if i not in match_sk_branches] + new_submeshes_idx)

            segment_widths_median = np.delete(segment_widths_median, match_sk_branches, axis=0)
            segment_widths_median = np.append(segment_widths_median, new_widths, axis=0)

            try:
                sk.check_skeleton_connected_component(sk.stack_skeletons(segment_branches))
            except:
                su.compressed_pickle(local_correspondence_revised, "local_correspondence_revised")
            print("checked segment branches after soma add on")

    return (
        segment_branches,
        divided_submeshes,
        divided_submeshes_idx,
        segment_widths_median,
        endpts_total,
        curr_soma_to_piece_touching_vertices_total,
        no_soma_extension_add,
    )


def _merge_map_mp_correspondence(limb_correspondence_MAP, limb_correspondence_MP):
    """Part 17: схлопывает MAP- и MP-correspondence в один плоский dict
    {branch_idx -> branch_dict}, последовательно перенумеровывая ветви."""
    limb_correspondence_individual = dict()
    counter = 0
    for sublimb_branches in limb_correspondence_MAP.values():
        for branch_dict in sublimb_branches.values():
            limb_correspondence_individual[counter] = branch_dict
            counter += 1
    for sublimb_branches in limb_correspondence_MP.values():
        for branch_dict in sublimb_branches.values():
            limb_correspondence_individual[counter] = branch_dict
            counter += 1
    return limb_correspondence_individual


def _rearrange_network_starting_info(
    limb_to_soma_touching_vertices_list,
    limb_to_endpoints_must_keep_list,
    soma_touching_vertices_dict,
):
    """Part 18.1: перегруппировывает сырые touching-vertices/endpoints в
    soma_idx -> border_group_idx -> [dict(touching_verts, endpoint)]."""
    network_starting_info_revised = dict()
    for v_list_dict, enpts_list_dict in zip(
        limb_to_soma_touching_vertices_list, limb_to_endpoints_must_keep_list
    ):
        if set(list(v_list_dict.keys())) != set(list(enpts_list_dict)):
            raise Exception("Soma keys not match for touching vertices and endpoints")
        for sm_idx in v_list_dict.keys():
            v_list_soma = v_list_dict[sm_idx]
            endpt_soma = enpts_list_dict[sm_idx]
            if len(v_list_soma) != len(endpt_soma):
                raise Exception(f"touching vertices list and endpoint list not match size for soma {sm_idx}")

            all_border_vertex_groups = soma_touching_vertices_dict[sm_idx]

            for v_l, endpt in zip(v_list_soma, endpt_soma):
                matching_border_group = []
                for i, curr_border_group in enumerate(all_border_vertex_groups):
                    if nu.test_matching_vertices_in_lists(curr_border_group, v_l, verbose=True):
                        matching_border_group.append(i)

                if len(matching_border_group) == 0 or len(matching_border_group) > 1:
                    raise Exception(f"Matching border groups was not exactly 1: {matching_border_group}")

                winning_border_group = matching_border_group[0]

                if sm_idx not in network_starting_info_revised.keys():
                    network_starting_info_revised[sm_idx] = dict()
                if winning_border_group not in network_starting_info_revised[sm_idx].keys():
                    network_starting_info_revised[sm_idx][winning_border_group] = []
                network_starting_info_revised[sm_idx][winning_border_group].append(
                    dict(touching_verts=v_l, endpoint=endpt)
                )
    return network_starting_info_revised


def _clean_network_starting_info(
    network_starting_info_revised,
    limb_correspondence_individual,
    soma_touching_vertices_dict,
):
    """Part 18.2: для каждого (soma, border group) выбирает один стартовый
    endpoint — на скелете, ближайший к центру границы; если таких нет, ищет
    новую degree-1 точку на касающихся границу ветвях."""
    sorted_keys = np.sort(list(limb_correspondence_individual.keys()))
    curr_branches = [limb_correspondence_individual[k]["branch_skeleton"] for k in sorted_keys]
    curr_meshes = [limb_correspondence_individual[k]["branch_mesh"] for k in sorted_keys]

    network_starting_info_revised_cleaned = dict()
    for soma_idx in network_starting_info_revised.keys():
        network_starting_info_revised_cleaned[soma_idx] = dict()
        for bound_g_idx, endpoint_list in network_starting_info_revised[soma_idx].items():
            endpoint_list = np.array(endpoint_list)

            filter_on_skeleton_list = []
            for zz, endpt_dict in enumerate(endpoint_list):
                #a. filter to only those with an endpoint that is on a branch of the skeleton
                sk_indices = sk.find_branch_skeleton_with_specific_coordinate(
                    divded_skeleton=curr_branches, current_coordinate=endpt_dict["endpoint"]
                )
                if len(sk_indices) > 0:
                    filter_on_skeleton_list.append(zz)

            endpoint_list_filt = endpoint_list[filter_on_skeleton_list]

            curr_border_group_coordinates = soma_touching_vertices_dict[soma_idx][bound_g_idx]
            boundary_mean = np.mean(curr_border_group_coordinates, axis=0)

            if len(endpoint_list_filt) == 1:
                print("Only one endpoint after filtering away the endpoints that are not on the skeleton")
                winning_dict = endpoint_list_filt[0]
            #b2: If more --> pick the one with the endpoint closest to the average fo the vertex group
            elif len(endpoint_list_filt) > 1:
                print(f"MORE THAN one endpoint after filtering away the endpoints that are not on the skeleton: {len(endpoint_list_filt)}")
                viable_endpoints = [endpt_dict["endpoint"] for endpt_dict in endpoint_list_filt]
                distanes_from_mean = np.linalg.norm(viable_endpoints - boundary_mean, axis=1)
                winning_endpoint_idx = np.argmin(distanes_from_mean)
                winning_dict = endpoint_list_filt[winning_endpoint_idx]
            #if there was no clear winner
            else:
                print("Having to find a new branch point")
                #i) get all meshes that touch the vertex group (and keep the vertices that overlap)
                mesh_indices_on_border = tu.filter_meshes_by_containing_coordinates(
                    curr_meshes,
                    nullifying_points=curr_border_group_coordinates,
                    filter_away=False,
                    distance_threshold=min_distance_threshold,
                    return_indices=True,
                )
                if len(mesh_indices_on_border) == 0:
                    raise Exception("There were no meshes that were touching the boundary group")

                total_skeleton_graph = sk.convert_skeleton_to_graph(sk.stack_skeletons(curr_branches))
                skeleton_branches_on_border = [k for n, k in enumerate(curr_branches) if n in mesh_indices_on_border]
                skeleton_branches_on_border_endpoints = np.array([sk.find_branch_endpoints(k) for k in skeleton_branches_on_border])

                viable_endpoints = []
                for enpt in skeleton_branches_on_border_endpoints.reshape(-1, 3):
                    curr_enpt_node = xu.get_graph_node_by_coordinate(total_skeleton_graph, enpt, return_single_value=True)
                    curr_enpt_degree = xu.get_node_degree(total_skeleton_graph, curr_enpt_node)
                    if curr_enpt_degree == 1:
                        viable_endpoints.append(enpt)

                if len(viable_endpoints) == 0:
                    print("No branch endpoints were degree 1 so just using all endpoints")
                    viable_endpoints = skeleton_branches_on_border_endpoints.reshape(-1, 3)

                distanes_from_mean = np.linalg.norm(viable_endpoints - boundary_mean, axis=1)
                winning_endpoint = viable_endpoints[np.argmin(distanes_from_mean)]

                sk_indices = sk.find_branch_skeleton_with_specific_coordinate(
                    divded_skeleton=curr_branches, current_coordinate=winning_endpoint
                )
                winning_branch = np.intersect1d(mesh_indices_on_border, sk_indices)
                if len(winning_branch) == 0:
                    raise Exception("There was no winning branch for the creation of a new soma extending branch")
                winning_branch_single = winning_branch[0]

                winning_touching_vertices = tu.filter_vertices_by_mesh(curr_meshes[winning_branch_single], curr_border_group_coordinates)
                winning_dict = dict(touching_verts=winning_touching_vertices, endpoint=winning_endpoint)

            network_starting_info_revised_cleaned[soma_idx][bound_g_idx] = winning_dict
    return network_starting_info_revised_cleaned


def _safe_frame_faces(meshes, li):
    """Best-effort len(meshes[li].faces); None if unavailable."""
    try:
        return len(meshes[li].faces)
    except Exception:
        return None


@dataclass
class _SublimbGrouping:
    """Parts 4-8 output: one limb's meshparty decomposition split into MAP and MP sublimbs.

    MAP sublimbs (`map_subs`) are wide branches that need CGAL re-skeletonization,
    grouped by mesh+skeleton connectivity. MP sublimbs (`sublimb_*`) are everything else. When
    no MAP candidates survive, there is a single whole-limb MP sublimb and `MAP_flag` is False.
    """
    map_subs: list   # list[submesh_ops.SubMesh]: MAP sublimb mesh + limb-frame face_idx + parent
    sublimb_meshes_MP: list
    sublimb_mesh_branches_MP: list
    sublimb_mesh_idx_branches_MP: list
    sublimb_skeleton_branches: list
    widths_MP: list
    MAP_flag: bool


def _group_into_map_mp_sublimbs(
    segment_branches,
    divided_submeshes,
    divided_submeshes_idx,
    segment_widths_median,
    limb_mesh_mparty,
    use_meshafterparty,
    width_threshold_MAP,
    size_threshold_MAP,
):
    """Parts 4-8: from the meshparty decomposition, find the wide branches that warrant MAP
    (CGAL) re-skeletonization, group them into connected MAP sublimbs, and put the rest into
    MP sublimbs; fall back to a single whole-limb MP sublimb when no MAP candidates survive.

    Returns a `_SublimbGrouping`. Behavior identical to the inline Part 4-8 block.
    """
    if use_meshafterparty:
        print("Attempting to use MeshAfterParty Skeletonization and Mesh Correspondence")
        # --------------- Part 4: Find Individual Branches that could be MAP processed because of width ------------- #
        #gettin the branches that should be passed through MAP skeletonization
        pieces_above_threshold = np.where(segment_widths_median>width_threshold_MAP)[0]

        mesh_large_idx = [divided_submeshes_idx[k] for k in pieces_above_threshold]
    else:
        print("Only Using MeshParty Skeletonization and Mesh Correspondence")
        mesh_large_idx = []


    print("Another print")
    # Phase B: map_subs bundles each MAP sublimb mesh + its limb-frame face_idx + parent
    # (limb_mesh_mparty) into one submesh_ops.SubMesh, replacing the old parallel
    # mesh_pieces_for_MAP / mesh_pieces_for_MAP_face_idx arrays.
    map_subs = []


    if len(mesh_large_idx) > 0: #will only continue processing if found MAP candidates

        # --------------- Part 5: Find mesh connectivity and group MAP branch candidates into MAP sublimbs ------------- #
        print(f"Found len(mesh_large_idx) MAP candidates: {[len(k) for k in mesh_large_idx]}")

        #finds the connectivity edges of all the MAP candidates
        mesh_large_connectivity = tu.mesh_list_connectivity(meshes = mesh_large_idx,
                                                            connectivity="edges",
                                main_mesh = limb_mesh_mparty,
                                print_flag = False)

        """ 1/3/21s
        Big Conclusion from debugging: the large mesh pieces themselves (before combining into map pieces)
        themselves aren't totally connected by edges (can be split)

        - so even if large pieces do have a shared edge and you combine them together,
        they can still be split by the edges into multiple pieces because the original pieces
        could be split into multiple pieces


        """


        G = nx.Graph()
        G.add_nodes_from(np.arange(len(mesh_large_idx)))
        G.add_edges_from(mesh_large_connectivity)
        conn_comp = list(nx.connected_components(G))

        filtered_pieces = []

        for cc in conn_comp:
            total_cc_size = np.sum([len(mesh_large_idx[k]) for k in cc])
            if total_cc_size>size_threshold_MAP:
                filtered_pieces.append(pieces_above_threshold[list(cc)])


        if len(filtered_pieces) > 0:
            # --------------- Part 6: If Found MAP sublimbs, Get the meshes and mesh_idxs of the sublimbs ------------- #
            #all the pieces that will require MAP mesh correspondence and skeletonization
            #(already organized into their components)
            _map_face_idx = [np.concatenate(divided_submeshes_idx[k]) for k in filtered_pieces]
            map_subs = [submesh_ops.SubMesh(limb_mesh_mparty.submesh([idx], append=True, repair=False), idx, limb_mesh_mparty)
                        for idx in _map_face_idx]


            # --------------- Part 7: If Found MAP sublimbs, Get the meshes and mesh_idxs of the sublimbs ------------- #
            # ********* if there are no pieces leftover then will automatically make all the lists below just empty (don't need to if.. else.. the case)****
            pieces_idx_MP = np.delete(np.arange(len(divided_submeshes_idx)),np.concatenate(filtered_pieces))

            skeleton_MP = [segment_branches[k] for k in pieces_idx_MP]
            skeleton_connectivity_MP = sk.skeleton_list_connectivity(
                                            skeletons=skeleton_MP
                                            )

            G = nx.Graph()
            G.add_nodes_from(np.arange(len(skeleton_MP)))
            G.add_edges_from(skeleton_connectivity_MP)
            sublimbs_MP = list(nx.connected_components(G))
            sublimbs_MP_orig_idx = [pieces_idx_MP[list(k)] for k in sublimbs_MP]


            #concatenate into sublimbs the skeletons and meshes
            sublimb_mesh_idx_branches_MP = [divided_submeshes_idx[k] for k in sublimbs_MP_orig_idx]
            sublimb_mesh_branches_MP = [[limb_mesh_mparty.submesh([ki],append=True,repair=False)
                                        for ki in k] for k in sublimb_mesh_idx_branches_MP]
            sublimb_meshes_MP = [limb_mesh_mparty.submesh([np.concatenate(k)],append=True,repair=False)
                                                         for k in sublimb_mesh_idx_branches_MP]

            sublimb_skeleton_branches = [segment_branches[k] for k in sublimbs_MP_orig_idx]
            widths_MP = [segment_widths_median[k] for k in sublimbs_MP_orig_idx]


    # --------------- Part 8: If No MAP sublimbs found, set the MP sublimb lists to just the whole MP branch decomposition ------------- #

    #if no sublimbs need to be decomposed with MAP then just reassign all of the previous MP processing to the sublimb_MPs


    if len(map_subs) == 0:
        print('no MAP pieces')
        sublimb_meshes_MP = [limb_mesh_mparty] #trimesh pieces that have already been passed through MP skeletonization (may not need)
        # -- the decomposition information ---
        sublimb_mesh_branches_MP = [divided_submeshes] #the mesh branches for all the disconnected sublimbs
        sublimb_mesh_idx_branches_MP = [divided_submeshes_idx] #The mesh branches idx that have already passed through MP skeletonization
        sublimb_skeleton_branches = [segment_branches]#the skeleton bnraches for all the sublimbs
        widths_MP = [segment_widths_median] #the mesh branches widths for all the disconnected groups

        MAP_flag = False
    else:
        MAP_flag = True

    return _SublimbGrouping(
        map_subs=map_subs,
        sublimb_meshes_MP=sublimb_meshes_MP,
        sublimb_mesh_branches_MP=sublimb_mesh_branches_MP,
        sublimb_mesh_idx_branches_MP=sublimb_mesh_idx_branches_MP,
        sublimb_skeleton_branches=sublimb_skeleton_branches,
        widths_MP=widths_MP,
        MAP_flag=MAP_flag,
    )


def _decompose_mp_sublimbs(
    sublimb_meshes_MP,
    sublimb_skeleton_branches,
    sublimb_mesh_branches_MP,
    sublimb_mesh_idx_branches_MP,
    widths_MP,
    MAP_flag,
    soma_touching_vertices_dict,
    limb_mesh_mparty,
    *,
    print_fusion_steps,
    fusion_time,
):
    """Part 10: run each MP sublimb through `_fix_mp_soma_extension` and build its correspondence.

    Returns `(limb_correspondence_MP, endpoints_additions, soma_touching_additions)`:
    the per-sublimb correspondence dict, plus the soma-extension must-keep endpoints /
    soma-touching vertices for the caller to append (in order) to the limb-level
    accumulators. `fusion_time` is consumed only for the diagnostic timing prints (not
    returned — the caller does not read it after Part 10). Behavior identical to the
    inline Part-10 loop.
    """
    limb_correspondence_MP = dict()
    endpoints_additions = []
    soma_touching_additions = []

    for sublimb_idx, mesh in enumerate(sublimb_meshes_MP):
        print(f"---- Working on MP Decomposition #{sublimb_idx} ----")
        mesh_start_time = time.time()

        if len(sublimb_meshes_MP) == 1 and not MAP_flag:
            print("Using Quicker soma_to_piece_touching_vertices because no MAP and only one sublimb_mesh piece ")
            curr_soma_to_piece_touching_vertices = soma_touching_vertices_dict
        else:
            if soma_touching_vertices_dict is not None:
                print("Computing the current soma touching verts dict manually")
                curr_soma_to_piece_touching_vertices = filter_soma_touching_vertices_dict_by_mesh(
                                                    mesh = mesh,
                                                    curr_piece_to_soma_touching_vertices = soma_touching_vertices_dict
                                                    )
            else:
                curr_soma_to_piece_touching_vertices = None

        if print_fusion_steps:
            print(f"MP filtering soma verts: {time.time() - fusion_time }")
            fusion_time = time.time()

        #creating all of the sublimb groups
        segment_branches = np.array(sublimb_skeleton_branches[sublimb_idx])

        branch = mesh
        divided_submeshes = np.array(sublimb_mesh_branches_MP[sublimb_idx])
        divided_submeshes_idx = sublimb_mesh_idx_branches_MP[sublimb_idx]
        segment_widths_median = widths_MP[sublimb_idx]


        if curr_soma_to_piece_touching_vertices is None:
            print(f"Do Not Need to Fix MP Decomposition {sublimb_idx} so just continuing")

        else:
            # 11/9 addition: build soma-extending branches where the MP mesh touches a soma border
            print(f"Fixing Possible Soma Extension Branch for Sublimb {sublimb_idx}")
            (
                segment_branches,
                divided_submeshes,
                divided_submeshes_idx,
                segment_widths_median,
                endpts_total,
                curr_soma_to_piece_touching_vertices_total,
                no_soma_extension_add,
            ) = _fix_mp_soma_extension(
                segment_branches,
                divided_submeshes,
                divided_submeshes_idx,
                segment_widths_median,
                curr_soma_to_piece_touching_vertices,
                limb_mesh_mparty,
            )

            endpoints_additions.append(endpts_total)
            soma_touching_additions.append(curr_soma_to_piece_touching_vertices_total)

            if no_soma_extension_add:
                print("No soma extending branch was added for this sublimb even though it had a soma border (means they already existed)")


        #building the limb correspondence
        limb_correspondence_MP[sublimb_idx] = dict()

        for zz,b_sk in enumerate(segment_branches):
            limb_correspondence_MP[sublimb_idx][zz] = dict(
                branch_skeleton = b_sk,
                width_from_skeleton = segment_widths_median[zz],
                branch_mesh = divided_submeshes[zz],
                branch_face_idx = divided_submeshes_idx[zz]
                )

    return limb_correspondence_MP, endpoints_additions, soma_touching_additions


def _decompose_map_sublimbs(
    map_subs,
    soma_touching_vertices_dict,
    *,
    filter_end_node_length,
    perform_cleaning_checks,
    combine_close_skeleton_nodes,
    combine_close_skeleton_nodes_threshold,
    use_surface_after_CGAL,
    surface_reconstruction_size,
    remove_mesh_interior_face_threshold,
    error_on_bad_cgal_return,
    max_stitch_distance_CGAL,
    distance_by_mesh_center,
):
    """Part 9: run each MAP sublimb mesh through `_decompose_map_piece` (CGAL skeleton).

    Returns `(limb_correspondence_MAP, endpoints_additions, soma_touching_additions)`:
    the per-sublimb correspondence dict keyed by sublimb index, plus the must-keep
    endpoints / soma-touching-vertices that the caller appends (in order) to the
    limb-level accumulator lists. Behavior identical to the inline Part-9 loop.
    """
    limb_correspondence_MAP = dict()
    endpoints_additions = []
    soma_touching_additions = []
    global_start_time = time.time()

    for sublimb_idx, map_sub in enumerate(map_subs):
        (
            local_correspondence_revised,
            curr_limb_endpoints_must_keep,
            curr_soma_to_piece_touching_vertices,
        ) = _decompose_map_piece(
            map_sub=map_sub,
            sublimb_idx=sublimb_idx,
            soma_touching_vertices_dict=soma_touching_vertices_dict,
            filter_end_node_length=filter_end_node_length,
            perform_cleaning_checks=perform_cleaning_checks,
            combine_close_skeleton_nodes=combine_close_skeleton_nodes,
            combine_close_skeleton_nodes_threshold=combine_close_skeleton_nodes_threshold,
            use_surface_after_CGAL=use_surface_after_CGAL,
            surface_reconstruction_size=surface_reconstruction_size,
            remove_mesh_interior_face_threshold=remove_mesh_interior_face_threshold,
            error_on_bad_cgal_return=error_on_bad_cgal_return,
            max_stitch_distance_CGAL=max_stitch_distance_CGAL,
            distance_by_mesh_center=distance_by_mesh_center,
        )

        if curr_limb_endpoints_must_keep is not None:
            endpoints_additions.append(curr_limb_endpoints_must_keep)
            soma_touching_additions.append(curr_soma_to_piece_touching_vertices)

        limb_correspondence_MAP[sublimb_idx] = local_correspondence_revised

    print(f"Total time for MAP sublimb processing {time.time() - global_start_time}")
    return limb_correspondence_MAP, endpoints_additions, soma_touching_additions


def _reroute_branch_endpoint(branch_skeleton, remove_point, target_point, else_point):
    """Reroute a branch skeleton's stitch endpoint to `target_point`, smoothed.

    Convert the branch to a graph, drop the node at `remove_point`, add + smooth a segment
    from the kept neighbour to `target_point` (or, when the graph collapses to <=1 node, a
    straight segment to `else_point`), then resize to `meshparty_segment_size`. `else_point`
    is a separate arg only to preserve a pre-existing asymmetry at the Part-14 call site;
    normally it equals `target_point`.
    """
    MP_stitch_branch_graph = sk.convert_skeleton_to_graph(branch_skeleton)
    stitch_node = xu.get_nodes_with_attributes_dict(MP_stitch_branch_graph, dict(coordinates=remove_point))[0]
    stitch_neighbors = xu.get_neighbors(MP_stitch_branch_graph, stitch_node)
    if len(stitch_neighbors) != 1:
        raise Exception("Not just one neighbor for stitch point of MP branch")
    keep_neighbor = stitch_neighbors[0]
    keep_neighbor_coordinates = xu.get_node_attributes(MP_stitch_branch_graph, node_list=[keep_neighbor])[0]
    MP_stitch_branch_graph.remove_node(stitch_node)
    try:
        if len(MP_stitch_branch_graph) > 1:
            new_MP_skeleton = sk.add_and_smooth_segment_to_branch(skeleton=sk.convert_graph_to_skeleton(MP_stitch_branch_graph),
                                            skeleton_stitch_point=keep_neighbor_coordinates,
                                             new_stitch_point=target_point)
        else:
            print("Not even attempting smoothing segment because once keep_neighbor_coordinates")
            new_MP_skeleton = np.vstack([keep_neighbor_coordinates, else_point]).reshape(-1, 2, 3)
    except:
        su.compressed_pickle(MP_stitch_branch_graph, "MP_stitch_branch_graph")
        su.compressed_pickle(keep_neighbor_coordinates, "keep_neighbor_coordinates")
        su.compressed_pickle(target_point, "target_point")
        raise Exception("Something went wrong with add_and_smooth_segment_to_branch")
    return sk.resize_skeleton_branch(new_MP_skeleton, segment_width=meshparty_segment_size)


def _find_map_stitch_point(map_correspondence, v_g, av_vert, curr_skeleton_MAP, total_keep_endpoints):
    """Find the MAP-side stitch point for one MP<->MAP connection.

    Picks the MAP skeleton point closest to the connection centroid whose branch mesh actually
    touches the border vertices, optionally snapped to a nearby end/branch node. Returns
    (MAP_stitch_point, MAP_stitch_point_on_end_or_branch, MAP_branches_with_stitch_point,
    MAP_pieces_idx_touching_border).
    """
    # -------------- 11/9 NEW METHOD FOR FINDING MAP STITCH POINT ------------ #
    # (finds a MAP stitch point whose branch mesh actually touches the border vertices)
    o_keys = np.sort(list(map_correspondence.keys()))
    curr_MAP_branch_meshes = np.array([map_correspondence[k]["branch_mesh"]
                                     for k in o_keys])
    curr_MAP_branch_skeletons = np.array([map_correspondence[k]["branch_skeleton"]
                                     for k in o_keys])

    MAP_pieces_idx_touching_border = tu.filter_meshes_by_containing_coordinates(mesh_list=curr_MAP_branch_meshes,
                                   nullifying_points=v_g,
                                    filter_away=False,
                                   distance_threshold=min_distance_threshold,
                                   return_indices=True)

    MAP_branches_considered = curr_MAP_branch_skeletons[MAP_pieces_idx_touching_border]
    curr_skeleton_MAP_for_stitch = sk.stack_skeletons(MAP_branches_considered)

    #3) Find the closest skeletal point on MAP pairing (MAP stitch)
    MAP_skeleton_coords = np.unique(curr_skeleton_MAP_for_stitch.reshape(-1,3),axis=0)


    MAP_stitch_point = MAP_skeleton_coords[np.argmin(np.linalg.norm(MAP_skeleton_coords-av_vert,axis=1))]

    # --------- 11/13: Making so could possibly stitch to another point that was already stitched to
    curr_br_endpts = np.array([sk.find_branch_endpoints(k) for k in MAP_branches_considered]).reshape(-1,3)
    curr_br_endpts_unique = np.unique(curr_br_endpts,axis=0)


    #3b) Consider if the stitch point is close enough to end or branch node in skeleton:
    # and if so then reassign
    if move_MAP_stitch_to_end_or_branch:
        MAP_stitch_point_new,change_status = sk.move_point_to_nearest_branch_end_point_within_threshold(
                                                skeleton=curr_skeleton_MAP,
                                                coordinate=MAP_stitch_point,
                                                distance_to_move_point_threshold = distance_to_move_point_threshold,
                                                verbose=False,
                                                possible_node_coordinates=curr_br_endpts_unique,
                                                excluded_node_coordinates=total_keep_endpoints,
                                                )
        MAP_stitch_point=MAP_stitch_point_new


    #4) Find the branches that have that MAP stitch point:

    MAP_branches_with_stitch_point = sk.find_branch_skeleton_with_specific_coordinate(
        divded_skeleton=curr_MAP_branch_skeletons,
        current_coordinate = MAP_stitch_point
    )


    MAP_stitch_point_on_end_or_branch = False
    if len(MAP_branches_with_stitch_point)>1:
        MAP_stitch_point_on_end_or_branch = True
    elif len(MAP_branches_with_stitch_point)==1:
        if len(nu.matching_rows(sk.find_branch_endpoints(curr_MAP_branch_skeletons[MAP_branches_with_stitch_point[0]]),
                                MAP_stitch_point))>0:
            MAP_stitch_point_on_end_or_branch=True
    else:
        raise Exception("No matching MAP values")
    return (MAP_stitch_point, MAP_stitch_point_on_end_or_branch,
            MAP_branches_with_stitch_point, MAP_pieces_idx_touching_border)


def _find_mp_stitch_point(mp_correspondence, v_g, av_vert, total_keep_endpoints,
                          all_map_stitch_points):
    """Find the MP-side stitch point (winning_vertex) for one MP<->MAP connection.

    Restricts to MP branch meshes touching the border, picks the endpoint closest to the
    connection centroid, finds the branch(es) carrying it, and flags keep_MP_stitch_static when
    that vertex was already used as a MAP stitch point. Returns (winning_vertex,
    MP_branches_with_stitch_point, keep_MP_stitch_static, conn, curr_MP_branch_skeletons).
    """
    ord_keys = np.sort(list(mp_correspondence.keys()))
    curr_MP_branch_meshes = [mp_correspondence[k]["branch_mesh"] for k in ord_keys]


    # 11/9 Addition: New way that filters meshes by their touching of the vertex connection group (this could possibly be an empty group)
    conn = tu.filter_meshes_by_containing_coordinates(mesh_list=curr_MP_branch_meshes,
                                   nullifying_points=v_g,
                                    filter_away=False,
                                   distance_threshold=min_distance_threshold,
                                   return_indices=True)

    if len(conn) == 0:
        print("Connectivity was 0 for the MP mesh groups touching the vertex group so not restricting by that anymore")
        sk_conn = np.arange(0,len(curr_MP_branch_meshes))
    else:
        sk_conn = conn



    #1) Get the endpoint vertices of the MP skeleton branches (so every endpoint or high degree node)
    #(needs to be inside loop because limb correspondence will change)
    curr_MP_branch_skeletons = [mp_correspondence[k]["branch_skeleton"] for k in sk_conn]
    endpoint_nodes_coordinates = np.array([sk.find_branch_endpoints(k) for k in curr_MP_branch_skeletons])
    endpoint_nodes_coordinates = np.unique(endpoint_nodes_coordinates.reshape(-1,3),axis=0)

    """ ---------- 1 /5: Take out the possible endpoints --------------------"""
    if prevent_MP_starter_branch_stitches:
        endpoint_nodes_coordinates = nu.setdiff2d(endpoint_nodes_coordinates,total_keep_endpoints)


    #2) Find the closest endpoint vertex to the vertex connection group (this is MP stitch point)

    winning_vertex = endpoint_nodes_coordinates[np.argmin(np.linalg.norm(endpoint_nodes_coordinates-av_vert,axis=1))]


    #2b) Find the branch points where the winning vertex is located
    curr_MP_branch_skeletons = [mp_correspondence[k]["branch_skeleton"] for k in np.sort(list(mp_correspondence.keys()))]
    MP_branches_with_stitch_point = sk.find_branch_skeleton_with_specific_coordinate(
        divded_skeleton=curr_MP_branch_skeletons,
        current_coordinate = winning_vertex
    )


    # -------- 11/13 addition: Will see if the MP stitch point was already a MAP stitch point ---- #
    if len(nu.matching_rows(np.array(all_map_stitch_points),winning_vertex)) > 0:
        keep_MP_stitch_static = True
    else:
        keep_MP_stitch_static = False
    return (winning_vertex, MP_branches_with_stitch_point, keep_MP_stitch_static,
            conn, curr_MP_branch_skeletons)


@dataclass
class _StitchCtx:
    """Per-connection state for one MP<->MAP stitch iteration (Parts 15-16).

    Bundles the shared limb references plus the per-connection values that flow from the
    stitch-point finders / Parts 13-14 into the re-correspondence (`_recorrespond_stitch`) and
    entry-overwrite (`_overwrite_stitched_entries`) steps, so each takes a single `ctx` arg
    instead of a 12-15 parameter signature. The three trailing fields are produced by
    `_recorrespond_stitch` and consumed by `_overwrite_stitched_entries`.
    """
    # shared references (identical for every connection in the limb; mutated in place)
    limb_correspondence_MAP: dict
    limb_correspondence_MP: dict
    limb_mesh_mparty: object
    # per-connection identity + stitch-point results (from the finders / Parts 13-14)
    MP_idx: int
    MAP_idx: int
    MAP_stitch_point_on_end_or_branch: bool
    MAP_branches_with_stitch_point: object
    MAP_pieces_idx_touching_border: object
    keep_MP_stitch_static: bool
    conn: object
    MP_branches_with_stitch_point: object
    curr_MAP_sk: object
    curr_MP_sk: object
    curr_MAP_sk_final: object = None
    cut_flag: bool = False
    # produced by _recorrespond_stitch (Part 15), consumed by _overwrite_stitched_entries (Part 16)
    MAP_pieces_for_correspondence: object = None
    local_correspondence_stitch_revised: object = None
    MP_branches_for_correspondence: object = None


def _recorrespond_stitch(ctx):
    """Part 15: gather the stitched MAP+MP branches, re-run mesh correspondence across the
    join (cutting the MAP mesh first when the stitch split a branch), and stash the results
    (`MAP_pieces_for_correspondence`, `local_correspondence_stitch_revised`,
    `MP_branches_for_correspondence`, possibly-revised `curr_MAP_sk`) back onto `ctx` for
    `_overwrite_stitched_entries`. Body identical to the inline Part 15 block it replaces.
    """
    MAP_branches_with_stitch_point = ctx.MAP_branches_with_stitch_point
    MAP_pieces_idx_touching_border = ctx.MAP_pieces_idx_touching_border
    limb_correspondence_MAP = ctx.limb_correspondence_MAP
    MAP_idx = ctx.MAP_idx
    curr_MAP_sk = ctx.curr_MAP_sk
    cut_flag = ctx.cut_flag
    limb_mesh_mparty = ctx.limb_mesh_mparty
    conn = ctx.conn
    MP_branches_with_stitch_point = ctx.MP_branches_with_stitch_point
    limb_correspondence_MP = ctx.limb_correspondence_MP
    MP_idx = ctx.MP_idx
    curr_MP_sk = ctx.curr_MP_sk

    # -------------- Part 15: Gets all of the skeletons and Mesh to divide u and does mesh correspondence -------#
    # ------------- revise IDX so still references the whole limb mesh -----------#

    # -------------- 11/10 Addition accounting for not all MAP pieces always touching each other --------------------#
    if len(MAP_branches_with_stitch_point) > 1:
        print("\nRevising the MAP pieces index:")
        MAP_pieces_for_correspondence = nu.intersect1d(MAP_pieces_idx_touching_border,MAP_branches_with_stitch_point)
        curr_MAP_sk = [limb_correspondence_MAP[MAP_idx][k]["branch_skeleton"] for k in MAP_pieces_for_correspondence]
    else:
        MAP_pieces_for_correspondence = MAP_branches_with_stitch_point

    curr_MAP_meshes_idx = [limb_correspondence_MAP[MAP_idx][k]["branch_face_idx"] for k in MAP_pieces_for_correspondence]

    # Have to adjust based on if the skeleton were split

    if cut_flag:
        #Then it was cut and have to do mesh correspondence to find what label to cut
        if len(curr_MAP_meshes_idx) > 1:
            raise Exception("MAP_pieces_for_correspondence was longer than 1 and cut flag was set")
        pre_stitch_mesh_idx = curr_MAP_meshes_idx[0]
        pre_stitch_mesh = limb_mesh_mparty.submesh([pre_stitch_mesh_idx],append=True,repair=False)
        local_correspondence_stitch_revised_MAP = _run_mesh_correspondence(
                                                    pre_stitch_mesh, curr_MAP_sk,
                                                    curr_limb_endpoints_must_keep=None,
                                                    curr_soma_to_piece_touching_vertices=None)

        #Need to readjust the mesh correspondence idx
        for k,v in local_correspondence_stitch_revised_MAP.items():
            local_correspondence_stitch_revised_MAP[k]["branch_face_idx"] = pre_stitch_mesh_idx[local_correspondence_stitch_revised_MAP[k]["branch_face_idx"]]

        curr_MAP_meshes_idx = [v["branch_face_idx"] for v in local_correspondence_stitch_revised_MAP.values()]
    else:
        local_correspondence_stitch_revised_MAP = dict([(gg,limb_correspondence_MAP[MAP_idx][kk]) for gg,kk in enumerate(MAP_pieces_for_correspondence)])

        for gg,kk in enumerate(MAP_pieces_for_correspondence):
            local_correspondence_stitch_revised_MAP[gg]["branch_skeleton"] = curr_MAP_sk[gg]


    #To make sure that the MAP never gives up ground on the labels
    must_keep_labels_MAP = dict()
    must_keep_counter = 0
    for kk,b_idx in enumerate(curr_MAP_meshes_idx):
        #must_keep_labels_MAP.update(dict([(ii,kk) for ii in range(must_keep_counter,must_keep_counter+len(b_idx))]))
        must_keep_labels_MAP[kk] = np.arange(must_keep_counter,must_keep_counter+len(b_idx))
        must_keep_counter += len(b_idx)


    #this is where should send only the MP that apply
    MP_branches_for_correspondence,conn_idx,MP_branches_with_stitch_point_idx = nu.intersect1d(conn,MP_branches_with_stitch_point,return_indices=True)

    curr_MP_meshes_idx = [limb_correspondence_MP[MP_idx][k]["branch_face_idx"] for k in MP_branches_for_correspondence]
    curr_MP_sk_for_correspondence = [curr_MP_sk[zz] for zz in MP_branches_with_stitch_point_idx]

    stitching_mesh_idx = np.concatenate(curr_MAP_meshes_idx + curr_MP_meshes_idx)
    stitching_mesh = limb_mesh_mparty.submesh([stitching_mesh_idx],append=True,repair=False)
    stitching_skeleton_branches = curr_MAP_sk + curr_MP_sk_for_correspondence
    """

    ****** NEED TO GET THE RIGHT MESH TO RUN HE IDX ON SO GETS A GOOD MESH (CAN'T BE LIMB_MESH_MPARTY)
    BUT MUST BE THE ORIGINAL MAP MESH

    mesh_pieces_for_MAP
    sublimb_meshes_MP

    mesh_pieces_for_MAP_face_idx
    sublimb_meshes_MP_face_idx

    stitching_mesh = tu.combine_meshes(curr_MAP_meshes + curr_MP_meshes)
    stitching_skeleton_branches = curr_MAP_sk + curr_MP_sk

    """

    # ******************************** this is where should do thing about no mesh correspondence ***************** #

    # -------- 12/22: Trying to do the re-correspondence but if doesn't work then just resort to old one --------- #

    try:

        #3) Run mesh correspondence to get new meshes and mesh_idx and widths
        local_correspondence_stitch_revised = _run_mesh_correspondence(
                                                    stitching_mesh, stitching_skeleton_branches,
                                                    curr_limb_endpoints_must_keep=None,
                                                    curr_soma_to_piece_touching_vertices=None,
                                                    must_keep_labels=must_keep_labels_MAP)

        #Need to readjust the mesh correspondence idx
        for k,v in local_correspondence_stitch_revised.items():
            local_correspondence_stitch_revised[k]["branch_face_idx"] = stitching_mesh_idx[local_correspondence_stitch_revised[k]["branch_face_idx"]]
    except:
        print("Errored in 1 to 1 correspondence in stitching so just reverting to the original mesh assignments")
        # Setting the correspondence manually because the adaptive way did not work
        local_counter = 0
        local_correspondence_stitch_revised = dict()

        # setting the MAP parts (the new skeletons have already been adjusted)
        for k in local_correspondence_stitch_revised_MAP:
            local_correspondence_stitch_revised[local_counter] = local_correspondence_stitch_revised_MAP[k]
            local_counter += 1

        # setting the MP parts (the new skeletons have not been adjusted yet so adjusting them here)
        for mp_idx, k in enumerate(MP_branches_for_correspondence):
            local_correspondence_stitch_revised[local_counter] = limb_correspondence_MP[MP_idx][k] 
            local_correspondence_stitch_revised[local_counter]["branch_skeleton"] = curr_MP_sk[mp_idx]
            local_counter += 1

    ctx.MAP_pieces_for_correspondence = MAP_pieces_for_correspondence
    ctx.local_correspondence_stitch_revised = local_correspondence_stitch_revised
    ctx.MP_branches_for_correspondence = MP_branches_for_correspondence
    ctx.curr_MAP_sk = curr_MAP_sk


def _overwrite_stitched_entries(ctx):
    """Part 16: overwrite the old MAP/MP branch entries with the re-corresponded stitched
    results on `ctx` (adding one new MAP branch when the stitch split a branch, and fixing up
    skeletons left out of the correspondence). Mutates the correspondence dicts in place.
    Body identical to the inline Part 16 block it replaces.
    """
    MAP_stitch_point_on_end_or_branch = ctx.MAP_stitch_point_on_end_or_branch
    limb_correspondence_MAP = ctx.limb_correspondence_MAP
    MAP_idx = ctx.MAP_idx
    MAP_branches_with_stitch_point = ctx.MAP_branches_with_stitch_point
    local_correspondence_stitch_revised = ctx.local_correspondence_stitch_revised
    MAP_pieces_for_correspondence = ctx.MAP_pieces_for_correspondence
    keep_MP_stitch_static = ctx.keep_MP_stitch_static
    curr_MAP_sk_final = ctx.curr_MAP_sk_final
    curr_MAP_sk = ctx.curr_MAP_sk
    limb_correspondence_MP = ctx.limb_correspondence_MP
    MP_idx = ctx.MP_idx
    MP_branches_for_correspondence = ctx.MP_branches_for_correspondence
    MP_branches_with_stitch_point = ctx.MP_branches_with_stitch_point
    curr_MP_sk = ctx.curr_MP_sk

    # -------------- Part 16: Overwrite old branch entries (and add on one new to MAP if required a split) -------#


    #4a) If MAP_stitch_point_on_end_or_branch is False
    #- Delete the old MAP branch parts and replace with new MAP ones
    if not MAP_stitch_point_on_end_or_branch:
        print("Deleting branches from dictionary")
        #replace the split branch with the two new cut halves, reordering keys
        limb_correspondence_MAP[MAP_idx] = _split_branch_entry(
            limb_correspondence_MAP[MAP_idx], MAP_branches_with_stitch_point[0],
            local_correspondence_stitch_revised[0], local_correspondence_stitch_revised[1])

    else: #4b) Revise the meshes,  mesh_idx, and widths of the MAP pieces if weren't broken up
        for j,curr_MAP_idx_fixed in enumerate(MAP_pieces_for_correspondence): 
            limb_correspondence_MAP[MAP_idx][curr_MAP_idx_fixed] = local_correspondence_stitch_revised[j]
        #want to update all of the skeletons just in case was altered by keep_MP_stitch_static and not included in correspondence
        if keep_MP_stitch_static:
            if len(MAP_branches_with_stitch_point) != len(curr_MAP_sk_final):
                raise Exception("MAP_branches_with_stitch_point not same size as curr_MAP_sk_final")
            for gg,map_idx_curr in enumerate(MAP_branches_with_stitch_point):
                limb_correspondence_MAP[MAP_idx][map_idx_curr]["branch_skeleton"] = curr_MAP_sk_final[gg]


    for j,curr_MP_idx_fixed in enumerate(MP_branches_for_correspondence): #************** right here just need to make only the ones that applied
        limb_correspondence_MP[MP_idx][curr_MP_idx_fixed] = local_correspondence_stitch_revised[j+len(curr_MAP_sk)]


    #5b) Fixing the branch skeletons that were not included in the correspondence
    MP_leftover,MP_leftover_idx = nu.setdiff1d(MP_branches_with_stitch_point,MP_branches_for_correspondence)

    for curr_MP_leftover,curr_MP_leftover_idx in zip(MP_leftover,MP_leftover_idx):
        limb_correspondence_MP[MP_idx][curr_MP_leftover]["branch_skeleton"] = curr_MP_sk[curr_MP_leftover_idx]


def _assemble_sublimbs(limb_correspondence):
    """Combine each sublimb's branch meshes/skeletons into per-sublimb (meshes, skeletons) lists.

    Order-preserving: mirrors the two identical Part-11 loops it replaces (byte-identical).
    """
    meshes, skeletons = [], []
    for _sublimb_key, sublimb_v in limb_correspondence.items():
        meshes.append(tu.combine_meshes([bv["branch_mesh"] for bv in sublimb_v.values()]))
        skeletons.append(sk.stack_skeletons([bv["branch_skeleton"] for bv in sublimb_v.values()]))
    return meshes, skeletons


def _stitch_map_and_mp(
    limb_correspondence_MAP,
    limb_correspondence_MP,
    limb_mesh_mparty,
    total_keep_endpoints,
):
    """Parts 11-16: stitch each MP<->MAP connection, mutating limb_correspondence_MAP /
    limb_correspondence_MP in place (branches re-cut / re-corresponded at the join), and
    return them. Callers guard with `len(MAP)>0 and len(MP)>0` -- this assumes both are
    non-empty. Behavior identical to the inline Part 11-16 block it replaces.
    """
    # -------------- Part 11: Getting Sublimb Mesh and Skeletons and Gets connectivitiy by Mesh -------#
    # -------------(filtering connections to only MP to MAP edges)--------------- #

    # ---- Doing the mesh connectivity ---------#
    sublimb_meshes_MP,  sublimb_skeletons_MP  = _assemble_sublimbs(limb_correspondence_MP)
    sublimb_meshes_MAP, sublimb_skeletons_MAP = _assemble_sublimbs(limb_correspondence_MAP)

    mesh_conn, mesh_conn_vertex_groups, connectivity_type = _resolve_mesh_connectivity(
        sublimb_meshes_MP,
        sublimb_meshes_MAP,
        limb_mesh_mparty,
        sublimb_skeletons_MP,
        sublimb_skeletons_MAP,
        connectivity_type="edges",
    )

    #adjust the connection indices for MP and MAP indices
    mesh_conn_adjusted = np.vstack([mesh_conn[:,0],mesh_conn[:,1]-len(sublimb_meshes_MP)]).T


    """
    Pseudocode:
    For each connection edge:
        For each vertex connection group:
            1) Get the endpoint vertices of the MP skeleton
            2) Find the closest endpoint vertex to the vertex connection group (this is MP stitch point)
            3) Find the closest skeletal point on MAP pairing (MAP stitch) 
            4) Find the branches that have that MAP stitch point:
            5A) If the number of branches corresponding to stitch point is multipled
                --> then we are stitching at a branching oint
                i) Just add the skeletal segment from MP_stitch to MAP stitch to the MP skeletal segment
                ii) 

    """


    # -------------- STITCHING PHASE -------#
    stitch_counter = 0
    all_map_stitch_points = []
    for (MP_idx,MAP_idx),v_g in zip(mesh_conn_adjusted,mesh_conn_vertex_groups):
        print(f"\n---- Working on {(MP_idx,MAP_idx)} connection-----")

        curr_skeleton_MAP = sk.stack_skeletons([branch_v["branch_skeleton"] for branch_v in limb_correspondence_MAP[MAP_idx].values()])


        av_vert = np.mean(v_g,axis=0)

        # ---------------- Doing the MAP part first -------------- #
        (MAP_stitch_point, MAP_stitch_point_on_end_or_branch,
         MAP_branches_with_stitch_point, MAP_pieces_idx_touching_border) = _find_map_stitch_point(
            limb_correspondence_MAP[MAP_idx], v_g, av_vert, curr_skeleton_MAP, total_keep_endpoints)
        all_map_stitch_points.append(MAP_stitch_point)  # add the map stitch point to the history

        # ---------------- Doing the MP Part --------------------- #
        (winning_vertex, MP_branches_with_stitch_point, keep_MP_stitch_static,
         conn, curr_MP_branch_skeletons) = _find_mp_stitch_point(
            limb_correspondence_MP[MP_idx], v_g, av_vert, total_keep_endpoints, all_map_stitch_points)



        # -------------- Part 13: Will Adjust the MP branches that have the stitch point so extends to the MAP stitch point -------#
        curr_MP_sk = []
        for b_idx in MP_branches_with_stitch_point:
            if not keep_MP_stitch_static:
                # reroute the MP branch so its winning_vertex endpoint extends to MAP_stitch_point
                curr_MP_sk.append(_reroute_branch_endpoint(
                    curr_MP_branch_skeletons[b_idx],
                    remove_point=winning_vertex, target_point=MAP_stitch_point,
                    else_point=MAP_stitch_point))
            else:
                curr_MP_sk.append(curr_MP_branch_skeletons[b_idx])


        #2) Get skeletons and meshes from MP and MAP pieces
        curr_MAP_sk = [limb_correspondence_MAP[MAP_idx][k]["branch_skeleton"] for k in MAP_branches_with_stitch_point]

        #2.1) Going to break up the MAP skeleton if need be
        """
        Pseudocode:
        a) check to see if it needs to be broken up
        If it does:
        b) Convert the skeleton into a graph
        c) Find the node of the MAP stitch point (where need to do the breaking)
        d) Find the degree one nodes
        e) For each degree one node:
        - Find shortest path from stitch node to end node
        - get a subgraph from that path
        - convert graph to a skeleton and save as new skeletons

        """
        # -------------- Part 14: Breaks Up MAP skeleton into 2 pieces if Needs (because MAP stitch point not on endpoint or branch point)  -------#

        #a) check to see if it needs to be broken up
        cut_flag = False
        if not MAP_stitch_point_on_end_or_branch:
            if len(curr_MAP_sk) > 1:
                raise Exception(f"There was more than one skeleton for MAP skeletons even though MAP_stitch_point_on_end_or_branch = {MAP_stitch_point_on_end_or_branch}")


            skeleton_to_cut = curr_MAP_sk[0]
            curr_MAP_sk = sk.cut_skeleton_at_coordinate(skeleton=skeleton_to_cut,
                                                        cut_coordinate=MAP_stitch_point)
            cut_flag=True


        curr_MAP_sk_final = None
        # ------ 11/13 Addition: need to adjust the MAP points if have to keep MP static
        if keep_MP_stitch_static:
            curr_MAP_sk_final = []
            for map_skel in curr_MAP_sk:
                # FIX: the <=1-node fallback now targets winning_vertex to match target_point. The
                # original used MAP_stitch_point here -- a copy-paste asymmetry from the Part-13
                # block (whose target IS MAP_stitch_point). Only affects the rare path where the
                # branch graph collapses to a single node after the stitch node is removed.
                curr_MAP_sk_final.append(_reroute_branch_endpoint(
                    map_skel,
                    remove_point=MAP_stitch_point, target_point=winning_vertex,
                    else_point=winning_vertex))
            curr_MAP_sk = copy.deepcopy(curr_MAP_sk_final)


        ctx = _StitchCtx(
            limb_correspondence_MAP=limb_correspondence_MAP,
            limb_correspondence_MP=limb_correspondence_MP,
            limb_mesh_mparty=limb_mesh_mparty,
            MP_idx=MP_idx,
            MAP_idx=MAP_idx,
            MAP_stitch_point_on_end_or_branch=MAP_stitch_point_on_end_or_branch,
            MAP_branches_with_stitch_point=MAP_branches_with_stitch_point,
            MAP_pieces_idx_touching_border=MAP_pieces_idx_touching_border,
            keep_MP_stitch_static=keep_MP_stitch_static,
            conn=conn,
            MP_branches_with_stitch_point=MP_branches_with_stitch_point,
            curr_MAP_sk=curr_MAP_sk,
            curr_MP_sk=curr_MP_sk,
            curr_MAP_sk_final=curr_MAP_sk_final,
            cut_flag=cut_flag,
        )
        _recorrespond_stitch(ctx)        # Part 15: re-run mesh correspondence at the join
        _overwrite_stitched_entries(ctx)  # Part 16: overwrite old branch entries w/ stitched


        print(f" Finished with {(MP_idx,MAP_idx)} \n\n\n")
        stitch_counter += 1

        if check_correspondence_branches:
            sk.check_correspondence_branches_have_2_endpoints(limb_correspondence_MAP[MAP_idx])
            sk.check_correspondence_branches_have_2_endpoints(limb_correspondence_MP[MP_idx])
    return limb_correspondence_MAP, limb_correspondence_MP


def preprocess_limb(
    mesh,
    neuron_params,
    limb_params,
    soma_touching_vertices_dict=None,
    return_concept_network=True,
    return_concept_network_starting_info=False,
    error_on_no_starting_coordinates=True,
):
    """Декомпозирует одну ветвь (limb) на ветви скелета и mesh-correspondence.

    neuron_params — параметры уровня нейрона (общие для всех веток):
        width_threshold_MAP, size_threshold_MAP, axon_width_preprocess_limb_max,
        use_adaptive_invalidation_d, mp_only_* , verbose.
    limb_params — параметры конкретного прохода скелетонизации:
        invalidation_d, smooth_neighborhood, combine_close_skeleton_nodes_threshold_meshparty,
        filter_end_node_length_meshparty, use_meshafterparty, ...
    Остальные скалярные параметры берутся из parameters.params.
    """
    p = parameters.params

    # ── параметры уровня нейрона ───────────────────────────────────────
    width_threshold_MAP = neuron_params["width_threshold_MAP"]
    size_threshold_MAP = neuron_params["size_threshold_MAP"]

    # ── config-driven параметры из глобального конфига ─────────────────
    filter_end_node_length = p.filter_end_node_length
    surface_reconstruction_size = p.surface_reconstruction_size
    remove_mesh_interior_face_threshold = p.remove_mesh_interior_face_threshold
    max_stitch_distance_CGAL = p.max_stitch_distance_CGAL
    # min_distance_threshold — module-level constant (см. строку 39)

    # ── фиксированные константы (бывшие дефолты сигнатуры) ─────────────
    # meshparty_segment_size / move_MAP_stitch_to_end_or_branch / distance_to_move_point_threshold /
    # prevent_MP_starter_branch_stitches / check_correspondence_branches — module-level constants now.
    distance_by_mesh_center = True
    perform_cleaning_checks = True
    combine_close_skeleton_nodes = True
    combine_close_skeleton_nodes_threshold = 700
    use_surface_after_CGAL = True
    error_on_bad_cgal_return = False
    filter_end_nodes_from_correspondence = True
    run_concept_network_checks = True
    print_fusion_steps = True


    curr_limb_time = time.time()

    limb_mesh_mparty = mesh

    #will store a list of all the endpoints tha tmust be kept:
    limb_to_endpoints_must_keep_list = []
    limb_to_soma_touching_vertices_list = []

    # --------------- Part 1 and 2: Getting Border Vertices and Setting the Root------------- #
    fusion_time = time.time()
    #will eventually get the current root from soma_to_piece_touching_vertices[i]
    if soma_touching_vertices_dict is not None:
        root_curr = soma_touching_vertices_dict[list(soma_touching_vertices_dict.keys())[0]][0][0]
    else:
        root_curr = None


    if print_fusion_steps:
        print(f"Time for preparing soma vertices and root: {time.time() - fusion_time }")
        fusion_time = time.time()

    # --------------- Part 3: Meshparty skeletonization and Decomposition ------------- #
    (
        segment_branches,
        divided_submeshes,
        divided_submeshes_idx,
        segment_widths_median,
    ) = _cycle_for_something(
        limb_mesh_mparty,
        root_curr,
        meshparty_segment_size=meshparty_segment_size,
        neuron_cfg=neuron_params,
        limb_cfg=limb_params,
    )


    # --------------- Parts 4-8: split the meshparty decomposition into MAP / MP sublimbs ------------- #
    _grp = _group_into_map_mp_sublimbs(
        segment_branches,
        divided_submeshes,
        divided_submeshes_idx,
        segment_widths_median,
        limb_mesh_mparty,
        limb_params['use_meshafterparty'],
        width_threshold_MAP,
        size_threshold_MAP,
    )
    map_subs = _grp.map_subs
    sublimb_meshes_MP = _grp.sublimb_meshes_MP
    sublimb_mesh_branches_MP = _grp.sublimb_mesh_branches_MP
    sublimb_mesh_idx_branches_MP = _grp.sublimb_mesh_idx_branches_MP
    sublimb_skeleton_branches = _grp.sublimb_skeleton_branches
    widths_MP = _grp.widths_MP
    MAP_flag = _grp.MAP_flag

    # ------------------- At this point have the correct division between MAP and MP ------------------------

    # -------------- Part 9: Doing the MAP decomposition ------------------ #
    limb_correspondence_MAP, _map_endpoints_add, _map_soma_touching_add = _decompose_map_sublimbs(
        map_subs,
        soma_touching_vertices_dict,
        filter_end_node_length=filter_end_node_length,
        perform_cleaning_checks=perform_cleaning_checks,
        combine_close_skeleton_nodes=combine_close_skeleton_nodes,
        combine_close_skeleton_nodes_threshold=combine_close_skeleton_nodes_threshold,
        use_surface_after_CGAL=use_surface_after_CGAL,
        surface_reconstruction_size=surface_reconstruction_size,
        remove_mesh_interior_face_threshold=remove_mesh_interior_face_threshold,
        error_on_bad_cgal_return=error_on_bad_cgal_return,
        max_stitch_distance_CGAL=max_stitch_distance_CGAL,
        distance_by_mesh_center=distance_by_mesh_center,
    )
    limb_to_endpoints_must_keep_list.extend(_map_endpoints_add)
    limb_to_soma_touching_vertices_list.extend(_map_soma_touching_add)


    # ----------------- Part 10: Doing the MP Decomposition ---------------------- #
    limb_correspondence_MP, _mp_endpoints_add, _mp_soma_touching_add = _decompose_mp_sublimbs(
        sublimb_meshes_MP,
        sublimb_skeleton_branches,
        sublimb_mesh_branches_MP,
        sublimb_mesh_idx_branches_MP,
        widths_MP,
        MAP_flag,
        soma_touching_vertices_dict,
        limb_mesh_mparty,
        print_fusion_steps=print_fusion_steps,
        fusion_time=fusion_time,
    )
    limb_to_endpoints_must_keep_list.extend(_mp_endpoints_add)
    limb_to_soma_touching_vertices_list.extend(_mp_soma_touching_add)


    if check_correspondence_branches:
        sk.check_correspondence_branches_have_2_endpoints(limb_correspondence_MAP)
        sk.check_correspondence_branches_have_2_endpoints(limb_correspondence_MP)
        
    total_keep_endpoints = []
    for entry in limb_to_endpoints_must_keep_list:
        for k,v in entry.items():
            total_keep_endpoints.append(v)
            
    if len(total_keep_endpoints)>0:
        total_keep_endpoints = np.vstack(total_keep_endpoints)
        
    total_keep_endpoints = np.array(total_keep_endpoints)
    
    # Only want to perform this step if both MP and MAP pieces
    if len(limb_correspondence_MAP)>0 and len(limb_correspondence_MP)>0:
        limb_correspondence_MAP, limb_correspondence_MP = _stitch_map_and_mp(
            limb_correspondence_MAP, limb_correspondence_MP, limb_mesh_mparty, total_keep_endpoints)
    else:
        print("There were not both MAP and MP pieces so skipping the stitch resolving phase")

    print(f"Time for decomp of Limb = {time.time() - curr_limb_time}")


    # -------------- Part 17: Grouping the MP and MAP Correspondence into one correspondence dictionary -------#
    limb_correspondence_individual = _merge_map_mp_correspondence(
        limb_correspondence_MAP, limb_correspondence_MP
    )

    # -------------- Part 18: filter the network starting info into a clean presentation ------------ #
    # 1) перегруппировать в soma_idx -> border_group -> [dict(touching_verts, endpoint)]
    network_starting_info_revised = _rearrange_network_starting_info(
        limb_to_soma_touching_vertices_list,
        limb_to_endpoints_must_keep_list,
        soma_touching_vertices_dict,
    )
    # 2) выбрать единственный стартовый endpoint на каждую (soma, border group)
    network_starting_info_revised_cleaned = _clean_network_starting_info(
        network_starting_info_revised,
        limb_correspondence_individual,
        soma_touching_vertices_dict,
    )

    # -------------- Part 18b: Filter the limb correspondence for any short stubs ------------ #
    if filter_end_nodes_from_correspondence:
        limb_correspondence_individual = filter_limb_correspondence_for_end_nodes(limb_correspondence=limb_correspondence_individual,
                                                     mesh=limb_mesh_mparty,
                                                     starting_info=network_starting_info_revised_cleaned,
                                                    filter_end_node_length=filter_end_node_length,
                                                    error_on_no_starting_coordinates=error_on_no_starting_coordinates,
                                                    error_on_starting_coordinates_not_endnodes= prevent_MP_starter_branch_stitches

                                                    )

    if not return_concept_network:
        if return_concept_network_starting_info: #because may want to calculate the concept networks later
            return limb_correspondence_individual,network_starting_info_revised_cleaned
        else:
            return limb_correspondence_individual
    else:
        limb_to_soma_concept_networks = calculate_limb_concept_networks(limb_correspondence_individual,
                                                                        network_starting_info_revised_cleaned,
                                                                        run_concept_network_checks=run_concept_network_checks,
                                                                       )


    return limb_correspondence_individual,limb_to_soma_concept_networks


def _extract_single_soma(mesh, segment_id):
    """Заменяет оригинальную Фазу 1 и 2. Ищет строго одну сому."""
    (soma_mesh_list, _, total_soma_list_sdf, _, _) = extract_soma_center(
        segment_id, mesh.vertices, mesh.faces
    )

    if not soma_mesh_list:
        raise ValueError(f"No soma found for segment {segment_id}")

    main_soma = soma_mesh_list[0]
    soma_sdf = total_soma_list_sdf[0]

    return main_soma, soma_sdf

def _segment_limbs_from_soma(main_mesh, soma_mesh, params):
    """
    Заменяет Фазу 3 (поиск кусков, касающихся сомы).
    Вычитает сому из общего меша и находит точки соединения веток.

    Returns:
        branch_meshes: list of limb meshes touching the soma
        floating_meshes: list of disconnected pieces
        soma_touching_vertices: list, one entry per branch_mesh;
            each entry is {soma_idx: [touching_vertex_array]} for preprocess_limb
        soma_to_piece_connectivity: {0: [0..N-1]} (limbs index branch_meshes positionally)
        branch_meshes_orig_idx: list parallel to branch_meshes -- each limb's faces in the
            ORIGINAL neuron mesh (level-2 provenance carried from the split_significant SubMeshes)
        soma_faces_idx: the soma's faces in the original neuron mesh (geometric faces_by_match)
    """
    soma_faces_idx = submesh_ops.faces_by_match(main_mesh, soma_mesh)
    non_soma_faces = np.delete(np.arange(len(main_mesh.faces)), soma_faces_idx)
    non_soma_mesh = main_mesh.submesh([non_soma_faces], append=True, repair=False)

    sig_pieces_sm, insignificant_sm = submesh_ops.split_significant(
        non_soma_mesh, params.size_threshold_MAP, return_insignificant=True
    )
    sig_pieces = [s.mesh for s in sig_pieces_sm]
    insignificant_limbs = [s.mesh for s in insignificant_sm]

    
    connected_pieces, connected_vertices = tu.mesh_pieces_connectivity(
        main_mesh=main_mesh, central_piece=soma_mesh, periphery_pieces=sig_pieces,
        return_vertices=True
    )

    branch_meshes = [sig_pieces[i] for i in connected_pieces]
    # Phase B level-2 provenance: each limb's faces in the ORIGINAL neuron mesh, carried from the
    # split_significant SubMeshes (root_face_idx into non_soma_mesh, composed through non_soma_faces).
    # Lets preprocess supply limb_mehses_face_idx so neuron.py skips the geometric KDTree fallback for
    # CLEAN (non-grown) limbs; grown (stitch-rebuilt) limbs fall back to the owned geometric match.
    branch_meshes_orig_idx = [non_soma_faces[sig_pieces_sm[i].root_face_idx()] for i in connected_pieces]
    floating_meshes = [p for i, p in enumerate(sig_pieces) if i not in connected_pieces]
    floating_meshes.extend(insignificant_limbs)

    # Build per-limb soma_touching_vertices_dict: {0: [border_verts_array]}
    soma_touching_vertices = [
        {0: [connected_vertices[j]]} for j in range(len(connected_pieces))
    ]
    # connectivity must index branch_meshes positionally (0..N-1), NOT the original
    # sig_pieces indices in `connected_pieces` — branch_meshes is the densified list
    # [sig_pieces[i] for i in connected_pieces], and downstream limb nodes are named
    # L{j} by enumerate(limb_meshes). Using the sparse sig_pieces indices here makes
    # the concept-network limb nodes (L2,L5,…) disagree with the data-bearing L0,L1,…
    # and blows up later as KeyError 'data'.
    soma_to_piece_connectivity = {0: list(range(len(branch_meshes)))}
    return (branch_meshes, floating_meshes, soma_touching_vertices, soma_to_piece_connectivity,
            branch_meshes_orig_idx, soma_faces_idx)

def _drop_trimesh_caches(mesh):
    """Clear trimesh's lazy derived-array cache (triangles, edges, face_adjacency, ...).

    These caches total ~6.6x a mesh's raw vertices+faces and are the dominant RAM holder
    mid-decomposition (see _decompose_limbs). They recompute on demand, so dropping them on
    meshes we retain but won't immediately touch again costs nothing but peak memory.
    """
    c = getattr(mesh, "_cache", None)
    if c is not None:
        c.clear()


def _decompose_limbs(branch_meshes, soma_touching_vertices, params):
    """Заменяет Фазу 4A (Скелетизация).

    Args:
        soma_touching_vertices: list parallel to branch_meshes;
            each element is soma_touching_vertices_dict for preprocess_limb
    """
    limb_correspondence = {}
    limb_network_starts = {}

    # ── параметры уровня нейрона (общие для всех веток) ────────────────
    neuron_params = dict(
        width_threshold_MAP=params.width_threshold_MAP,
        size_threshold_MAP=params.size_threshold_MAP,
        axon_width_preprocess_limb_max=params.axon_width_preprocess_limb_max,
        use_adaptive_invalidation_d=params.use_adaptive_invalidation_d,
        mp_only_revised_invalidation_d=params.mp_only_revised_invalidation_d,
        mp_only_invalidation_d_axon_buffer=params.mp_only_invalidation_d_axon_buffer,
        mp_only_revised_invalidation_d_reference=params.mp_only_revised_invalidation_d_reference,
        mp_only_revised_width_reference=params.mp_only_revised_width_reference,
        verbose=True,
    )

    # ── параметры прохода скелетонизации (одинаковые для всех веток) ───
    limb_params = dict(
        invalidation_d=params.invalidation_d,
        smooth_neighborhood=1,
        combine_close_skeleton_nodes_threshold_meshparty=700,
        filter_end_node_length_meshparty=params.filter_end_node_length,
        use_meshafterparty=True,
        use_adaptive_invalidation_d=params.use_adaptive_invalidation_d,
        axon_width_preprocess_limb_max=params.axon_width_preprocess_limb_max,
    )

    for idx, (limb_mesh, touching_verts_dict) in enumerate(
        zip(branch_meshes, soma_touching_vertices)
    ):
        corr, net_start = preprocess_limb(
            mesh=limb_mesh,
            neuron_params=neuron_params,
            limb_params=limb_params,
            soma_touching_vertices_dict=touching_verts_dict,
            return_concept_network=False,
            return_concept_network_starting_info=True,
        )
        limb_correspondence[idx] = corr
        limb_network_starts[idx] = net_start

        # Drop the trimesh cache on the just-consumed limb input mesh (its derived arrays are
        # not needed again here). The produced branch meshes are deliberately NOT cleared in
        # this loop — they are used heavily downstream (adaptive correspondence, widths), so
        # clearing now would just force immediate recomputes; their caches are cleared once at
        # the end of construction instead (Neuron._clear_mesh_caches). The single largest RAM
        # holder, the full neuron mesh, is cleared right after segmentation (preprocess_neuron).
        _drop_trimesh_caches(limb_mesh)

    return limb_correspondence, limb_network_starts

def _stitch_floating_pieces(
    limb_correspondence,
    floating_meshes,
    limb_network_starts,
    soma_mesh,
    params
):
    """
    Пришивает отсоединенные (плавающие) участки меша обратно к основному скелету.
    """
    if not limb_correspondence or not floating_meshes:
        return limb_correspondence

    # limb_network_starts is {limb_idx: per_limb_starting_info};
    # all_soma_connnecting_endpionts_from_starting_info handles this nested form
    excluded_node_coordinates = nru.all_soma_connnecting_endpionts_from_starting_info(
        limb_network_starts
    )

    outside_perc_threshold = 80
    meshes_to_stitch = [
        m for m in floating_meshes
        if tu.n_vertices_outside_mesh_bbox(m, [soma_mesh], return_percentage=True) > outside_perc_threshold
    ]

    stitched_correspondence = attach_floating_pieces_to_limb_correspondence(
        limb_correspondence,
        floating_meshes=meshes_to_stitch,
        excluded_node_coordinates=excluded_node_coordinates,
        verbose=False,
    )

    return stitched_correspondence


def _rebuild_limb_frames(limb_correspondence, limb_meshes):
    """Make every limb's branch_face_idx a clean partition of a self-consistent limb mesh.

    `_stitch_floating_pieces` appends floating-piece and re-cut branches whose branch_face_idx live
    in a FOREIGN mesh frame (the floating piece / stitch mesh), never remapped into the limb mesh.
    Downstream that surfaces as `index N is out of bounds for axis 0 with size N` in
    apply_adaptive_mesh_correspondence_to_neuron (class A), or as branches silently aliasing the
    wrong faces (overlap). Decomposition itself is correct — only the stitch desyncs the frame.

    For any limb whose branches no longer form a clean partition of its stored mesh, rebuild a
    consistent frame:
        limb mesh       = tu.combine_meshes(branch meshes, in branch-key order)
        branch_face_idx = contiguous [offset, offset + n) ranges into that combined mesh
    combine_meshes preserves face order and count (verified, even for fully duplicate geometry), so
    each range exactly recovers its branch's faces. Only inconsistent (stitched) limbs are rebuilt;
    clean limbs are left untouched, so neurons that never stitch are byte-for-byte unaffected. The
    floating-piece faces now genuinely live in the limb mesh, which is the correct post-stitch state.
    Returns limb_meshes (mutated in place and returned for convenience).
    """
    import numpy as _np
    rebuilt = 0
    for li, corr in limb_correspondence.items():
        n_faces = _safe_frame_faces(limb_meshes, li)
        arrs = [_np.asarray(d["branch_face_idx"]).ravel() for d in corr.values()
                if d.get("branch_face_idx") is not None]
        if not arrs:
            continue
        allidx = _np.concatenate(arrs)
        clean = (n_faces is not None
                 and int(allidx.max()) < n_faces
                 and int(_np.unique(allidx).size) == int(allidx.size))
        if clean:
            continue

        ordered_keys = sorted(corr.keys())
        branch_mesh_list = [corr[k]["branch_mesh"] for k in ordered_keys]
        combined = tu.combine_meshes(branch_mesh_list)
        offset = 0
        for k in ordered_keys:
            n = len(corr[k]["branch_mesh"].faces)
            corr[k]["branch_face_idx"] = _np.arange(offset, offset + n)
            offset += n
        limb_meshes[li] = combined
        rebuilt += 1
        print(f"[rebuild-limb-frame] limb {li}: stitch-induced desync (stored {n_faces} faces, "
              f"max branch idx {int(allidx.max())}) -> rebuilt from {len(ordered_keys)} branch "
              f"meshes into {len(combined.faces)} faces")
    if rebuilt:
        print(f"[rebuild-limb-frame] rebuilt {rebuilt} limb frame(s) after stitching")
    return limb_meshes


def _build_concept_networks(limb_correspondence_stitched, limb_network_starts):
    """
    Формирует графы концептов (Concept Networks) для каждой ветви.
    """
    limb_concept_networks = {}

    for limb_idx, correspondence in limb_correspondence_stitched.items():
        net_start = limb_network_starts[limb_idx]
        limb_to_soma_concept_network = calculate_limb_concept_networks(
            correspondence,
            net_start,
            run_concept_network_checks=True
        )
        limb_concept_networks[limb_idx] = limb_to_soma_concept_network

    return limb_concept_networks
from .parameters import params
def preprocess_neuron(
    mesh,
    segment_id,
    params = params,
    verbose=True
):
    """
    Главный пайплайн предобработки одиночного нейрона.
    Ожидает строго одну клетку без глии.

    Конфиг датасета (microns/h01) должен быть выбран вызывателем заранее через
    parameters.params.use(...); здесь он НЕ перебивается.
    """
    start_time = time.time()
    

    soma_mesh, soma_sdf = _extract_single_soma(mesh, segment_id)

    (branch_meshes, floating_meshes, soma_touching_vertices, soma_to_piece_connectivity,
     branch_meshes_orig_idx, soma_faces_idx) = _segment_limbs_from_soma(
        mesh, soma_mesh, params
    )

    # Segmenting the full neuron mesh built its heaviest trimesh caches — on the big H01
    # neuron `vertex_adjacency_graph` alone is ~1.6 GB and `vertex_faces` ~0.9 GB (split_by_
    # vertices populates them). The full mesh is NOT used past this point (the rest of
    # preprocessing works on the per-limb `branch_meshes`, and the returned dict carries
    # branch_meshes / soma_mesh, never `mesh`), so dropping its cache here removes the single
    # largest RAM holder for the long _decompose_limbs phase. Recomputes on demand if touched.
    _drop_trimesh_caches(mesh)

    limb_correspondence, limb_network_starts = _decompose_limbs(
        branch_meshes, soma_touching_vertices, params
    )

    limb_correspondence_stitched = _stitch_floating_pieces(
        limb_correspondence,
        floating_meshes,
        limb_network_starts,
        soma_mesh,
        params
    )

    # FIX (class A): _stitch_floating_pieces leaves floating/cut branches in a foreign mesh frame.
    # Rebuild a self-consistent frame (limb mesh = combined branch meshes, contiguous face idx) for
    # any limb whose partition is no longer clean. Only broken (stitched) limbs are touched.
    branch_meshes = _rebuild_limb_frames(limb_correspondence_stitched, branch_meshes)

    # Phase B (#3): supply the level-2 (limb/soma <-> original mesh) face maps so neuron.py skips its
    # geometric KDTree fallbacks. CLEAN limbs (mesh unchanged by the rebuild -> same face count as the
    # carried provenance) use the exact provenance; GROWN limbs (stitch-rebuilt, mesh gained foreign
    # floating geometry) fall back to the owned geometric match. Soma reuses the faces_by_match already
    # computed during segmentation (avoids neuron.py:2268 recomputing it).
    limb_mehses_face_idx = [
        np.sort(orig_idx) if len(orig_idx) == len(limb_mesh.faces)
        else submesh_ops.faces_by_match(mesh, limb_mesh)
        for limb_mesh, orig_idx in zip(branch_meshes, branch_meshes_orig_idx)
    ]

    limb_concept_networks = _build_concept_networks(
        limb_correspondence_stitched,
        limb_network_starts
    )

    if verbose:
        print(f"Total preprocessing time: {time.time() - start_time:.2f}s")

    return {
        "soma_meshes": [soma_mesh],
        "soma_volumes": [tu.mesh_volume(soma_mesh, watertight_method="convex_hull")],
        "soma_sdfs": [soma_sdf],
        "soma_touching_vertices": soma_touching_vertices,
        "soma_to_piece_connectivity" : soma_to_piece_connectivity,
        "limb_meshes": branch_meshes,
        "limb_mehses_face_idx": limb_mehses_face_idx,
        "soma_meshes_face_idx": [soma_faces_idx],
        "floating_meshes": floating_meshes,
        "limb_correspondence": limb_correspondence_stitched,
        "limb_concept_networks": limb_concept_networks,
        "limb_network_stating_info": limb_network_starts,
    }
    
 


