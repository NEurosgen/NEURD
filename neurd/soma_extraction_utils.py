from copy import deepcopy
import os
from pathlib import Path
from scipy.spatial import KDTree
import random
import time
import trimesh
from datasci_tools import numpy_dep as np



#from mesh_tools.trimesh_utils import split_significant_pieces,split
#from datasci_tools import numpy_utils as np


soma_connectivity="edges"
"""
Checking the new validation checks
"""
def side_length_ratios(current_mesh):
    """
    Will compute the ratios of the bounding box sides
    To be later used to see if there is skewness
    """

    # bbox = current_mesh.bounding_box_oriented.vertices
    bbox = current_mesh.bounding_box_oriented.vertices
    x_axis_unique = np.unique(bbox[:,0])
    y_axis_unique = np.unique(bbox[:,1])
    z_axis_unique = np.unique(bbox[:,2])
    x_length = (np.max(x_axis_unique) - np.min(x_axis_unique)).astype("float")
    y_length = (np.max(y_axis_unique) - np.min(y_axis_unique)).astype("float")
    z_length = (np.max(z_axis_unique) - np.min(z_axis_unique)).astype("float")
    #print(x_length,y_length,z_length)
    #compute the ratios:
    xy_ratio = float(x_length/y_length)
    xz_ratio = float(x_length/z_length)
    yz_ratio = float(y_length/z_length)
    side_ratios = [xy_ratio,xz_ratio,yz_ratio]
    flipped_side_ratios = []
    for z in side_ratios:
        if z < 1:
            flipped_side_ratios.append(1/z)
        else:
            flipped_side_ratios.append(z)
    return flipped_side_ratios

def side_length_check(current_mesh,side_length_ratio_threshold=3):
    side_length_ratio_names = ["xy","xz","yz"]
    side_ratios = side_length_ratios(current_mesh)
    pass_threshold = [(k <= side_length_ratio_threshold) and
                      (k >= 1/side_length_ratio_threshold) for k in side_ratios]
    for i,(rt,truth) in enumerate(zip(side_ratios,pass_threshold)):
        if not truth:
            print(f"{side_length_ratio_names[i]} = {rt} ratio was beyong {side_length_ratio_threshold} multiplier")

    if False in pass_threshold:
        return False
    else:
        return True


def largest_mesh_piece(msh):
    mesh_splits_inner = list(msh.split(only_watertight=False))
    total_mesh_split_lengths_inner = [len(k.faces) for k in mesh_splits_inner]
    sort_idx = np.flip(np.argsort(total_mesh_split_lengths_inner)).tolist()
    ordered_mesh_splits_inner = [mesh_splits_inner[i] for i in sort_idx]
    return ordered_mesh_splits_inner[0]

def soma_volume_ratio(current_mesh,
                     #watertight_method="fill_holes,
                     watertight_method="poisson",
                      max_value = 1000,
                     ):
    """
    bounding_box_oriented: rotates the box to be less volume
    bounding_box : does not rotate the box and makes it axis aligned
    
    ** checks to see if closed mesh and if not then make closed **
    """
    poisson_temp_folder = Path.cwd() / "Poisson_temp"
    poisson_temp_folder.mkdir(parents=True,exist_ok=True)
    with meshlab.Poisson(poisson_temp_folder,overwrite=True) as Poisson_obj_temp:

        #get the largest piece
        lrg_mesh = largest_mesh_piece(current_mesh)
        if not lrg_mesh.is_watertight:
            #lrg_mesh.export("lrg_mesh_in_soma_volume.off")
            if watertight_method == "poisson":
                print("Using Poisson Surface Reconstruction for watertightness in soma_volume_ratio")
                #run the Poisson Surface reconstruction and get the largest piece
                new_mesh_inner,poisson_file_obj = Poisson_obj_temp(vertices=lrg_mesh.vertices,
                       faces=lrg_mesh.faces,
                       return_mesh=True,
                       delete_temp_files=True,
                       segment_id=random.randint(0,999999))
                lrg_mesh = largest_mesh_piece(new_mesh_inner)
            elif watertight_method == "fill_holes":
                print("Using the close holes feature for watertightness in soma_volume_ratio")
                fill_hole_obj = meshlab.FillHoles(max_hole_size=2000,
                                                 self_itersect_faces=False)

                mesh_filled_holes,fillholes_file_obj = fill_hole_obj(   
                                                    vertices=lrg_mesh.vertices,
                                                     faces=lrg_mesh.faces,
                                                     return_mesh=True,
                                                     delete_temp_files=True,
                                                    )
                lrg_mesh = largest_mesh_piece(mesh_filled_holes)
                
                    
            else:
                raise Exception(f"Unimplemented watertight_method requested: {watertight_method}")

        #turn the mesh into a closed mesh based on 
        print(f"mesh.is_watertight = {lrg_mesh.is_watertight}")
        
        if not lrg_mesh.is_watertight:
            lrg_mesh = lrg_mesh.convex_hull
        
        ratio_val = lrg_mesh.bounding_box.volume/lrg_mesh.volume
    #     if ratio_val < 1:
    #         raise Exception("Less than 1 value in volume ratio computation")
    if ratio_val < max_value:
        return ratio_val
    else:
        return max_value

def soma_volume_check(current_mesh,multiplier=8,verbose = True):
    ratio_val= soma_volume_ratio(current_mesh)
    if verbose:
        print("Inside sphere validater: ratio_val = " + str(ratio_val))
    if np.abs(ratio_val) > multiplier:
        return False
    return True



# -------------- Function that will extract the soma ------- #


def filter_away_inside_soma_pieces(
                            main_mesh_total,
                            pieces_to_test,
                            significance_threshold=2000,
                            n_sample_points=3,
                            required_outside_percentage = 0.9,
                            print_flag = False,
                            return_inside_pieces=False,
                            ):
    if not nu.is_array_like(main_mesh_total):
        main_mesh_total = [main_mesh_total]
    
    if not nu.is_array_like(pieces_to_test):
        pieces_to_test = [pieces_to_test]
        
    if len(pieces_to_test) == 0:
        print("pieces_to_test was empty so returning empty list or pieces")
        return pieces_to_test
    
    significant_pieces = [m for m in pieces_to_test if len(m.faces) >= significance_threshold]
    
    print(f"There were {len(significant_pieces)} pieces found after size threshold")
    if len(significant_pieces) <=0:
        print("THERE WERE NO MESH PIECES GREATER THAN THE significance_threshold")
        return []
    
    final_mesh_pieces = []
    inside_pieces = []
    

    
    for i,mesh in enumerate(significant_pieces):
        outside_flag = True
        for j,main_mesh in enumerate(main_mesh_total):
            #gets the number of samples on the mesh to test (only the indexes)
            idx = np.random.choice(len(mesh.vertices),n_sample_points , replace=False)
            
            
            #gets the sample's vertices
            points = mesh.vertices[idx,:]

            start_time = time.time()

            #find the signed distance from the sampled vertices to the main mesh
            # Points outside the mesh will be negative
            # Points inside the mesh will be positive
            signed_distance = trimesh.proximity.signed_distance(main_mesh,points)

            #gets the 
            outside_percentage = sum(signed_distance <= 0)/n_sample_points
            
            if print_flag:
                print(f"Mesh piece {i} has outside_percentage {outside_percentage}, idx = {idx}")
            
            if outside_percentage < required_outside_percentage:
                if print_flag: 
                    print(f"Mesh piece {i} ({mesh}) inside mesh {j} :( ")
                outside_flag = False
                inside_pieces.append(mesh)
                break
        if outside_flag:
            if print_flag: 
                print(f"Mesh piece {i} OUTSIDE all meshes (corrected)")
            final_mesh_pieces.append(mesh)
        
    if return_inside_pieces:
        return final_mesh_pieces,inside_pieces
    else:
        return final_mesh_pieces


# subtacting the soma

''' old version 
def subtract_soma(current_soma,main_mesh,
                 significance_threshold=200,
                 distance_threshold = 550):
    print("\ninside Soma subtraction")
    start_time = time.time()
    face_midpoints_soma = np.mean(current_soma.vertices[current_soma.faces],axis=1)


    curr_mesh_bbox_restriction,faces_bbox_inclusion = (
                    tu.bbox_mesh_restriction(main_mesh,
                                             current_soma.bounds,
                                            mult_ratio=1.3)
    )

    face_midpoints_neuron = np.mean(curr_mesh_bbox_restriction.vertices[curr_mesh_bbox_restriction.faces],axis=1)

    soma_kdtree = KDTree(face_midpoints_soma)

    distances,closest_node = soma_kdtree.query(face_midpoints_neuron)

    distance_passed_faces  = distances<distance_threshold
    
    """ Older way of doing difference
    
    
    faces_to_keep = np.array(list(set(np.arange(0,len(main_mesh.faces))).difference(set(faces_bbox_inclusion[distance_passed_faces]))))
    """
    
    #newer way: using numpy functions
    faces_to_keep = np.delete(np.arange(len(main_mesh.faces)),
                                    faces_bbox_inclusion[distance_passed_faces])

    """
    #didn't work
    distance_passed_faces  = distances>=distance_threshold
    faces_to_keep = faces_bbox_inclusion[distance_passed_faces]
    
    """
    
    
    without_soma_mesh = main_mesh.submesh([faces_to_keep],append=True)
    
    
    

    #get the significant mesh pieces
    mesh_pieces = split_significant_pieces(without_soma_mesh,significance_threshold=significance_threshold)
    print(f"mesh pieces in subtact soma BEFORE the filtering inside pieces = {mesh_pieces}")
    
    mesh_pieces = filter_away_inside_soma_pieces(current_soma,mesh_pieces,
                                                           significance_threshold=significance_threshold)
    print(f"mesh pieces in subtact soma AFTER the filtering inside pieces = {mesh_pieces}")
    print(f"Total Time for soma mesh cancellation = {np.round(time.time() - start_time,3)}")
    return mesh_pieces
'''

def subtract_soma(current_soma_list,main_mesh,
                 significance_threshold=200,
                 distance_threshold = 1500,
                  connectivity="edges",
                 ):
    if type(current_soma_list) == type(trimesh.Trimesh()):
        current_soma_list = [current_soma_list]
    
    if type(current_soma_list) != list:
        raise Exception("Subtract soma was not passed a trimesh object or list for it's soma parameter")

        
    print("\ninside Soma subtraction")
    start_time = time.time()
    current_soma = tu.combine_meshes(current_soma_list)
    face_midpoints_soma = current_soma.triangles_center

    all_bounds = [k.bounds for k in  current_soma_list]
    

    curr_mesh_bbox_restriction,faces_bbox_inclusion = (
                    tu.bbox_mesh_restriction(main_mesh,
                                            all_bounds ,
                                            mult_ratio=1.3)
    )

    face_midpoints_neuron = curr_mesh_bbox_restriction.triangles_center

    soma_kdtree = KDTree(face_midpoints_soma)

    distances,closest_node = soma_kdtree.query(face_midpoints_neuron)

    distance_passed_faces  = distances<distance_threshold

    """ Older way of doing difference


    faces_to_keep = np.array(list(set(np.arange(0,len(main_mesh.faces))).difference(set(faces_bbox_inclusion[distance_passed_faces]))))
    """

    #newer way: using numpy functions
    faces_to_keep = np.delete(np.arange(len(main_mesh.faces)),
                                    faces_bbox_inclusion[distance_passed_faces])

    """
    #didn't work
    distance_passed_faces  = distances>=distance_threshold
    faces_to_keep = faces_bbox_inclusion[distance_passed_faces]

    """


    without_soma_mesh = main_mesh.submesh([faces_to_keep],append=True)




    #get the significant mesh pieces
    mesh_pieces = tu.split_significant_pieces(without_soma_mesh,significance_threshold=significance_threshold,
                                             connectivity=connectivity)
    
    # ----- 11/22 turns out weren't even using this part --------- #
#     print(f"mesh pieces in subtact soma BEFORE the filtering inside pieces = {mesh_pieces}")

#     current_mesh_pieces = filter_away_inside_soma_pieces(current_soma,mesh_pieces,
#                                          significance_threshold=significance_threshold,
#                                                         n_sample_points=5,
#                                                         required_outside_percentage=0.9)
#     print(f"mesh pieces in subtact soma AFTER the filtering inside pieces = {mesh_pieces}")
    print(f"Total Time for soma mesh cancellation = {np.round(time.time() - start_time,3)}")
    
    
    return mesh_pieces

def find_soma_centroids(soma_mesh_list):
    """
    Will return a list of soma centers if given one mesh or list of meshes
    the center is just found by averaging the vertices
    """
    if not nu.is_array_like(soma_mesh_list):
        soma_mesh_list = [soma_mesh_list]
    soma_mesh_list_centers = [np.array(np.mean(k.vertices,axis=0)).astype("float")
                           for k in soma_mesh_list]
    return soma_mesh_list_centers


def find_soma_centroid_containing_meshes(soma_mesh_list,
                                            split_meshes,
                                        verbose=False):
    """
    Purpose: Will find the mesh piece that most likely has the 
    soma that was found by the poisson soma finding process
    
    """
    containing_mesh_indices=dict([(i,[]) for i,sm_c in enumerate(soma_mesh_list)])
    for k,sm_mesh in enumerate(soma_mesh_list):
        sm_center = tu.mesh_center_vertex_average(sm_mesh)
        viable_meshes = np.array([j for j,m in enumerate(split_meshes) 
                 if trimesh.bounds.contains(m.bounds,sm_center.reshape(-1,3))
                        ])
        if verbose:
            print(f"viable_meshes = {viable_meshes}")
        if len(viable_meshes) == 0:
            raise Exception(f"The Soma {k} with mesh {sm_center} was not contained in any of the boundying boxes")
        elif len(viable_meshes) == 1:
            containing_mesh_indices[k] = viable_meshes[0]
        else:
            #find which mesh is closer to the soma midpoint (NOT ACTUALLY WHAT WE WANT)
            min_distances_to_soma = []
            dist_min_to_soma = []
            for v_i in viable_meshes:
                # build the KD Tree
                viable_neuron_kdtree = KDTree(split_meshes[v_i].vertices)
                distances,closest_node = viable_neuron_kdtree.query(sm_mesh.vertices.reshape(-1,3))
                min_distances_to_soma.append(np.sum(distances))
                dist_min_to_soma.append(np.min(distances))
            if verbose:
                print(f"min_distances_to_soma = {min_distances_to_soma}")
                print(f"dist_min_to_soma = {dist_min_to_soma}")
            containing_mesh_indices[k] = viable_meshes[np.argmin(min_distances_to_soma)]

    return containing_mesh_indices
    
def grouping_containing_mesh_indices(containing_mesh_indices):
    """
    Purpose: To take a dictionary that maps the soma indiece to the 
             mesh piece containing the indices: {0: 0, 1: 0}
             
             and to rearrange that to a dictionary that maps the mesh piece
             to a list of all the somas contained inside of it 
             
    Pseudocode: 
    1) get all the unique mesh pieces and create a dictionary with an empty list
    2) iterate through the containing_mesh_indices dictionary and add each
       soma index to the list of the containing mesh index
    3) check that none of the lists are empty or else something has failed
             
    """
    
    unique_meshes = np.unique(list(containing_mesh_indices.values()))
    mesh_groupings = dict([(i,[]) for i in unique_meshes])
    
    #2) iterate through the containing_mesh_indices dictionary and add each
    #   soma index to the list of the containing mesh index
    
    for soma_idx, mesh_idx in containing_mesh_indices.items():
        mesh_groupings[mesh_idx].append(soma_idx)
    
    #3) check that none of the lists are empty or else something has failed
    len_lists = [len(k) for k in mesh_groupings.values()]
    
    if 0 in len_lists:
        raise Exception("One of the lists is empty when grouping somas lists")
        
    return mesh_groupings



""" ---------- 9/23: Addition to help filter away false somas"""


def original_mesh_soma(
    mesh,
    original_mesh,
    bbox_restriction_multiplying_ratio = 1.7,
    match_distance_threshold = 1500,
    mesh_significance_threshold = 1000,
    return_inside_pieces = True,
    return_multiple_pieces_above_threshold = True,
    soma_size_threshold = 8000,
    verbose = False,
    ):
    """
    Purpose: To take an approximation of the soma mesh (usually from a poisson surface reconstruction)
    and map it to faces on the original mesh

    Pseudocode: 
    1) restrict the larger mesh with a bounding box or current
    2) Remove all interior pieces
    3) Save the interior pieces if asked for a pass-back
    4) Split the Main Mesh
    5) Find the Meshes that contain the soma
    6) Map to the original with a high distance threshold
    7) Split the new mesh and take the largest
    """

    soma_mesh_list = [mesh]

    #1) restrict the larger mesh with a bounding box or current
    restricted_big_mesh,_ = tu.bbox_mesh_restriction(original_mesh,mesh,mult_ratio=bbox_restriction_multiplying_ratio)

    # -- Split into largest piece -- #
#     restr_split = tu.split_significant_pieces(restricted_big_mesh,
#                                               significance_threshold=mesh_significance_threshold,connectivity="edges")
#     restr_mesh_to_test = restr_split[0]

    restr_mesh_to_test = restricted_big_mesh



    #2) Remove all interior pieces
    orig_mesh_to_map,inside_pieces = tu.remove_mesh_interior(restr_mesh_to_test,return_removed_pieces=True,size_threshold_to_remove=300)

    """ Old way that did a lot of splits: but not doing splits anymore
#     split_meshes = tu.split_significant_pieces(restr_without_interior,
#                                                significance_threshold=mesh_significance_threshold,connectivity="edges")

#     split_meshes = [restr_without_interior]


#     #5) Find the Meshes that contain the soma
#     containing_mesh_indices = find_soma_centroid_containing_meshes(soma_mesh_list,
#                                                 split_meshes,
#                                                 verbose=True)

#     soma_containing_meshes = grouping_containing_mesh_indices(containing_mesh_indices)

#     soma_touching_meshes = [split_meshes[k] for k in soma_containing_meshes.keys()]

#     if len(soma_touching_meshes) != 1:
#         raise Exception(f"soma_touching_meshes not of size 1 {soma_touching_meshes}")

#     orig_mesh_to_map = soma_touching_meshes[0]
    """


    #6) Map to the original with a high distance threshold
    prelim_soma_mesh = submesh_ops.faces_by_match(orig_mesh_to_map, mesh,
                              match_threshold=match_distance_threshold,
                              return_mesh=True)


    #7) Split the new mesh and take the largest
    split_meshes_after_backtrack = [s.mesh for s in submesh_ops.split_significant(
                                prelim_soma_mesh, mesh_significance_threshold,
                                connectivity=soma_connectivity)]
    if verbose:
        print(f"split_meshes_after_backtrack = {split_meshes_after_backtrack}")
        print(f"soma_size_threshold = {soma_size_threshold}")
        
    if len(split_meshes_after_backtrack) == 0:
        return_mesh = None
    else:
        if return_multiple_pieces_above_threshold:
            return_mesh = [k for k in split_meshes_after_backtrack if len(k.faces)>soma_size_threshold]

    if return_inside_pieces:
        return return_mesh,inside_pieces
    else:
        return return_mesh


    
def extract_soma_center(
    segment_id=12345,
    current_mesh_verts=None,
    current_mesh_faces=None,
    mesh = None,

    max_somas=None,  # if set (e.g. 1 for single-neuron files), stop the multi-piece
                     # search once this many somas are found, skipping the per-piece
                     # interior-removal/segmentation of the remaining mesh pieces.
    outer_decimation_ratio= None,
    large_mesh_threshold = None,#60000,
    large_mesh_threshold_inner = None, #was changed so dont filter away som somas
    soma_width_threshold = None,
    soma_size_threshold = None, #changed this to smaller so didn't filter some somas away
    inner_decimation_ratio = None,

    segmentation_clusters =3,
    segmentation_smoothness = 0.2,

    volume_mulitplier=None,
    #side_length_ratio_threshold=3
    side_length_ratio_threshold=None,
    soma_size_threshold_max=None,#240000,#192000, #this puts at 12000 once decimated, another possible is 256000
    delete_files=True,

    #------- 11/12 Additions --------------- #

    #these arguments are for removing inside pieces
    size_threshold_to_remove=None, #size accounting for the decimation


    last_size_threshold = None,#1300,

    largest_hole_threshold = None,
    max_fail_loops = None,#np.inf,
    verbose=False,

    backtrack_soma_size_threshold=None,
    backtrack_match_distance_threshold=1500,

    filter_inside_somas=True,

    second_pass_size_threshold = None,

    **kwargs
    ):
    
    # -------- Soma parameters -----------
    if outer_decimation_ratio is None:
        outer_decimation_ratio = parameters.params.outer_decimation_ratio

    if large_mesh_threshold is None:
        large_mesh_threshold = parameters.params.large_mesh_threshold

    if large_mesh_threshold_inner is None:
        large_mesh_threshold_inner = parameters.params.large_mesh_threshold_inner

    if inner_decimation_ratio is None:
        inner_decimation_ratio = parameters.params.inner_decimation_ratio

    # Experimental speed knobs (default = unchanged): the soma is a big smooth blob and
    # its accuracy is not a target here, so decimating the soma-detection mesh harder
    # cheapens the WHOLE soma stage at once (Poisson + SDF segmentation + meshlab
    # Decimator/Interior). NEURD_SOMA_OUTER_DECIM / NEURD_SOMA_INNER_DECIM override the
    # 0.25 ratios (smaller = fewer faces kept = faster, but risks falling below the
    # soma_size_threshold -> "No Somas").
    import os as _os
    _eo = _os.environ.get("NEURD_SOMA_OUTER_DECIM")
    if _eo:
        outer_decimation_ratio = float(_eo)
    _ei = _os.environ.get("NEURD_SOMA_INNER_DECIM")
    if _ei:
        inner_decimation_ratio = float(_ei)
        
        
    if max_fail_loops is None:
        max_fail_loops = parameters.params.max_fail_loops
    if size_threshold_to_remove is None:
        size_threshold_to_remove = parameters.params.size_threshold_to_remove
    if soma_width_threshold is None:
        soma_width_threshold = parameters.params.soma_width_threshold
    if soma_size_threshold is None:
        soma_size_threshold = parameters.params.soma_size_threshold
    if soma_size_threshold_max is None:
        soma_size_threshold_max = parameters.params.soma_size_threshold_max
    if volume_mulitplier is None:
        volume_mulitplier = parameters.params.volume_mulitplier
    if side_length_ratio_threshold is None:
        side_length_ratio_threshold = parameters.params.side_length_ratio_threshold
    if backtrack_soma_size_threshold is None:
        backtrack_soma_size_threshold = parameters.params.backtrack_soma_size_threshold
    if last_size_threshold is None:
        last_size_threshold = parameters.params.last_size_threshold
    if largest_hole_threshold is None:
        largest_hole_threshold = parameters.params.largest_hole_threshold
    if second_pass_size_threshold is None:
        second_pass_size_threshold = parameters.params.second_pass_size_threshold
        
    
    
    """
    Purpose: To extract the soma features

    Arguments for soma extraction:
    # **Note: All of the thresholds get scaled by the decimation ratios applied before

    outer_decimation_ratio= 0.25 #decimation ratio for 1st round of decimation
    large_mesh_threshold = 20000 #minimum face count threshold after 1st round of decimation
    large_mesh_threshold_inner = 13000 #minimum face count threshold after poisson reconstruction and split
    inner_decimation_ratio = 0.25 #decimation ratio for 2nd round of decimation after poisson reconstruction
    max_fail_loops = 10 #number of times soma finding can fail in finding an invalid soma in the outer/inner decimation loop

    # other cleaning methods to run on segments after first decimation
    size_threshold_to_remove=1000, #minium number of faces of mesh piece to remove

    #after 2nd round of decimation and mesh segmentation is applied
    soma_width_threshold = 0.32 # minimuum sdf values of mesh segments
    soma_size_threshold = 9000 # minimum face count threshold of mesh segments
    soma_size_threshold_max=1_200_000# maximum face count threshold of mesh segments

    #parameters for checking possible viable somas after mesh segmentation
    volume_mulitplier=8 #for soma_volume_check funcion that makes sure not highly skewed bounding box volume to mesh volume 
    side_length_ratio_threshold=6 #for side_length_check function making sure not highly swkewed x,y,z side length ratios
    delete_files=True #deletes all files in temp folder

    backtrack_soma_size_threshold=8000 #minimum number of faces for a backtracked mesh

    #filters at the very end for meshes that made it thorugh
    last_size_threshold = 2000 #min faces count
    largest_hole_threshold = 17000 #maximum hole length for a soma to allow the segmentaiton split at end
    """

    global_start_time = time.time()

    #Adjusting the thresholds based on the decimations
    large_mesh_threshold = large_mesh_threshold*outer_decimation_ratio
    large_mesh_threshold_inner = large_mesh_threshold_inner*outer_decimation_ratio
    soma_size_threshold = soma_size_threshold*outer_decimation_ratio
    soma_size_threshold_max = soma_size_threshold_max*outer_decimation_ratio

    #adjusting for inner decimation
    soma_size_threshold = soma_size_threshold*inner_decimation_ratio
    soma_size_threshold_max = soma_size_threshold_max*inner_decimation_ratio
    
    print(f"Current Arguments Using (adjusted for decimation):\n large_mesh_threshold= {large_mesh_threshold}"
                 f" \nlarge_mesh_threshold_inner = {large_mesh_threshold_inner}"
                  f" \nsoma_size_threshold = {soma_size_threshold}"
                 f" \nsoma_size_threshold_max = {soma_size_threshold_max}"
                 f"\nouter_decimation_ratio = {outer_decimation_ratio}"
                 f"\ninner_decimation_ratio = {inner_decimation_ratio}")


    # ------------------------------


    temp_folder = f"./{segment_id}"
    temp_object = Path(temp_folder)
    #make the temp folder if it doesn't exist
    temp_object.mkdir(parents=True,exist_ok=True)

    #making the decimation and poisson objections
    Dec_outer = meshlab.Decimator(outer_decimation_ratio,temp_folder,overwrite=True)
    Dec_inner = meshlab.Decimator(inner_decimation_ratio,temp_folder,overwrite=True)
    Poisson_obj = meshlab.Poisson(temp_folder,overwrite=True)

    if mesh is None:
        recov_orig_mesh = trimesh.Trimesh(vertices=current_mesh_verts,faces=current_mesh_faces)
    else:
        recov_orig_mesh = mesh

    #Step 1: Decimate the Mesh and then split into the seperate pieces
    new_mesh,output_obj = Dec_outer(vertices=recov_orig_mesh.vertices,
             faces=recov_orig_mesh.faces,
             segment_id=segment_id,
             return_mesh=True,
             delete_temp_files=False)

    # if remove_inside_pieces:
    #     print("removing mesh interior after decimation")
    #     new_mesh = tu.remove_mesh_interior(new_mesh,size_threshold_to_remove=size_threshold_to_remove)

    #preforming the splits of the decimated mesh

    mesh_splits = list(new_mesh.split(only_watertight=False))

    #get the largest mesh
    mesh_lengths = np.array([len(split.faces) for split in mesh_splits])


    total_mesh_split_lengths = [len(k.faces) for k in mesh_splits]
    sort_idx = np.flip(np.argsort(total_mesh_split_lengths)).tolist()
    ordered_mesh_splits = [mesh_splits[i] for i in sort_idx]
    list_of_largest_mesh = [k for k in ordered_mesh_splits if len(k.faces) > large_mesh_threshold]

    print(f"Total found significant pieces before Poisson = {list_of_largest_mesh}")

    #if no significant pieces were found then will use smaller threshold
    if len(list_of_largest_mesh)<=0:
        print(f"Using smaller large_mesh_threshold because no significant pieces found with {large_mesh_threshold}")
        list_of_largest_mesh = [k for k in ordered_mesh_splits if len(k.faces) > large_mesh_threshold/2]

    total_soma_list = []
    total_poisson_list = []
    total_soma_list_sdf = []

    filtered_soma_list_components = []

    for i in range(2):
        if len(filtered_soma_list_components)>0 or (i==1 and second_pass_size_threshold is None):
            if verbose:
                print(f"Not need to do a second pass because already found a soma")
                
            if delete_files:
                #now erase all of the files used
                from shutil import rmtree

                #remove the directory with the meshes
                rmtree(str(temp_object.absolute()))

                #removing the temporary files
                temp_folder = Path("./temp")
                temp_files = [x for x in temp_folder.glob('**/*')]
                seg_temp_files = [x for x in temp_files if str(segment_id) in str(x)]

                for f in seg_temp_files:
                    f.unlink()
            break
            
        if i == 1:
            if verbose:
                print(f"Using backup size thresholds")

            soma_size_threshold = second_pass_size_threshold
            last_size_threshold = second_pass_size_threshold
            backtrack_soma_size_threshold = second_pass_size_threshold


        
        #start iterating through where go through all pieces before the poisson reconstruction
        no_somas_found_in_big_loop = 0
        for i,largest_mesh in enumerate(list_of_largest_mesh):
            # single-neuron short-circuit: once enough somas are found, skip the
            # remaining pieces (each costs an interior-removal meshlab spawn + segmentation).
            if max_somas is not None and len(total_soma_list) >= max_somas:
                print(f"Reached max_somas={max_somas}; skipping {len(list_of_largest_mesh)-i} remaining piece(s)")
                break
            print(f"----- working on large mesh #{i}: {largest_mesh}")

            try:
                largest_mesh = tu.remove_mesh_interior(largest_mesh,
                                                       size_threshold_to_remove=size_threshold_to_remove,
                                                      try_hole_close=False)
            except:
                print("Unable to remove inside pieces in list_of_largest_mesh")

            somas_found_in_big_loop = False

            largest_file_name = str(output_obj.stem) + "_largest_piece.off"
            pre_largest_mesh_path = temp_object / Path(str(output_obj.stem) + "_largest_piece.off")
            pre_largest_mesh_path = pre_largest_mesh_path.absolute()
            print(f"pre_largest_mesh_path = {pre_largest_mesh_path}")
            # ******* This ERRORED AND CALLED OUR NERUON NONE: 77697401493989254 *********
            new_mesh_inner,poisson_file_obj = Poisson_obj(vertices=largest_mesh.vertices,
                       faces=largest_mesh.faces,
                       return_mesh=True,
                       mesh_filename=largest_file_name,
                       delete_temp_files=False)


            #splitting the Poisson into the largest pieces and ordering them
            mesh_splits_inner = list(new_mesh_inner.split(only_watertight=False))
            if len(mesh_splits_inner) > 0:
                total_mesh_split_lengths_inner = [len(k.faces) for k in mesh_splits_inner]
                sort_idx_inner = np.flip(np.argsort(total_mesh_split_lengths_inner)).tolist()
                ordered_mesh_splits_inner = [mesh_splits_inner[i] for i in sort_idx_inner]

                list_of_largest_mesh_inner = [k for k in ordered_mesh_splits_inner if len(k.faces) > large_mesh_threshold_inner]
            else:
                list_of_largest_mesh_inner = []
            print(f"Total found significant pieces AFTER Poisson = {list_of_largest_mesh_inner}")

            n_failed_inner_soma_loops = 0
            for j, largest_mesh_inner in enumerate(list_of_largest_mesh_inner):
                to_add_list = []
                to_add_list_sdf = []

                print(f"----- working on mesh after poisson #{j}: {largest_mesh_inner}")

                largest_mesh_path_inner = str(poisson_file_obj.stem) + "_largest_inner.off"

                #Decimate the inner poisson piece
                largest_mesh_path_inner_decimated,output_obj_inner = Dec_inner(
                                    vertices=largest_mesh_inner.vertices,
                                     faces=largest_mesh_inner.faces,
                                    mesh_filename=largest_mesh_path_inner,
                                     return_mesh=True,
                                     delete_temp_files=False)

                dec_splits = [s.mesh for s in submesh_ops.split_significant(
                    largest_mesh_path_inner_decimated, 15, connectivity=soma_connectivity)]
                print(f"\n-------Splits after inner decimation len = {len(dec_splits)}--------\n")

                if len(dec_splits) == 0:
                    print("There were no signifcant splits after inner decimation")
                    n_failed_inner_soma_loops += 1
                else:
                    print(f"done exporting decimated mesh: {largest_mesh_path_inner}")

                    largest_mesh_path_inner_decimated_clean = dec_splits[0]
                    """ # ----------- 1/12: Addition that does the segmentation again -------------------- #"""

                    for ii in range(3):
                        print(f"\n    --- On segmentation loop {ii} --")
                        print(f"largest_mesh_path_inner_decimated_clean = {largest_mesh_path_inner_decimated_clean}\n")

                        '''
                        faces = np.array(largest_mesh_path_inner_decimated_clean.faces)
                        verts = np.array(largest_mesh_path_inner_decimated_clean.vertices)

                        # may need to do some processing


                        segment_id_new = int(str(segment_id) + f"{i}{j}")
                        #print(f"Before the classifier the pymeshfix_clean = {pymeshfix_clean}")
                        verts_labels, faces_labels, soma_value,classifier = wcda.extract_branches_whole_neuron(
                                                import_Off_Flag=False,
                                                segment_id=segment_id_new,
                                                vertices=verts,
                                                 triangles=faces,
                                                pymeshfix_Flag=False,
                                                 import_CGAL_Flag=False,
                                                 return_Only_Labels=True,
                                                 clusters=3,
                                                 smoothness=0.2,
                                                soma_only=True,
                                                return_classifier = True
                                                )
                        print(f"soma_sdf_value = {soma_value}")


                        #total_poisson_list.append(largest_mesh_path_inner_decimated)

                        # Save all of the portions that resemble a soma
                        median_values = np.array([v["median"] for k,v in classifier.sdf_final_dict.items()])
                        segmentation = np.array([k for k,v in classifier.sdf_final_dict.items()])

                        #order the compartments by greatest to smallest
                        sorted_medians = np.flip(np.argsort(median_values))
                        print(f"segmentation[sorted_medians],median_values[sorted_medians] = {(segmentation[sorted_medians],median_values[sorted_medians])}")
                        print(f"Sizes = {[classifier.sdf_final_dict[g]['n_faces'] for g in segmentation[sorted_medians]]}")
                        print(f"soma_size_threshold = {soma_size_threshold}")
                        print(f"soma_size_threshold_max={soma_size_threshold_max}")

                        valid_soma_segments_width = [g for g,h in zip(segmentation[sorted_medians],median_values[sorted_medians]) if ((h > soma_width_threshold)
                                                                            and (classifier.sdf_final_dict[g]["n_faces"] > soma_size_threshold)
                                                                            and (classifier.sdf_final_dict[g]["n_faces"] < soma_size_threshold_max))]
                        valid_soma_segments_sdf = [h for g,h in zip(segmentation[sorted_medians],median_values[sorted_medians]) if ((h > soma_width_threshold)
                                                                            and (classifier.sdf_final_dict[g]["n_faces"] > soma_size_threshold)
                                                                            and (classifier.sdf_final_dict[g]["n_faces"] < soma_size_threshold_max))]

                        print("valid_soma_segments_width")
                        '''

                        """# ----------- 1/14 Addition that will just use the trimesh segmentation function ------------"""
                        
                        print(f"largest_mesh_path_inner_decimated_clean = {largest_mesh_path_inner_decimated_clean}")
                        print(f"soma_size_threshold = {soma_size_threshold}")
                        print(f"soma_size_threshold_max = {soma_size_threshold_max}")
                        print(f"soma_width_threshold = {soma_width_threshold}")
                        
                        divided_meshes,divided_meshes_sdf = tu.mesh_segmentation(mesh=largest_mesh_path_inner_decimated_clean,clusters=segmentation_clusters,
                                                                                 smoothness=segmentation_smoothness)

                        divided_meshes_len = np.array([len(k.faces) for k in divided_meshes])
                        valid_indexes = np.where((divided_meshes_len > soma_size_threshold) & 
                                                 (divided_meshes_len < soma_size_threshold_max) & 
                                                (np.array(divided_meshes_sdf) > soma_width_threshold))[0]

                        valid_soma_meshes = [divided_meshes[k] for k in valid_indexes]
                        valid_soma_segments_width = [divided_meshes_sdf[k] for k in valid_indexes]
                        



                        """# =------------- 1/12: Addition that will repeat this loop --------------"""
                        if len(valid_soma_meshes) > 0:
                            break
                        else:
                            """
                            Pseudocode: 
                            Get the largest mesh segment

                            Old:
                            new_mesh_try_faces = np.where(classifier.labels_list == nu.mode_1d(classifier.labels_list))[0]
                            largest_mesh_path_inner_decimated_clean = largest_mesh_path_inner_decimated_clean.submesh([new_mesh_try_faces],append=True)
                            """
                            largest_mesh_path_inner_decimated_clean = divided_meshes[0]

                    if len(valid_soma_segments_width) > 0:
                        print(f"      ------ Found {len(valid_soma_segments_width)} viable somas: {valid_soma_segments_width}")
                        somas_found_in_big_loop = True
                        #get the meshes only if signfiicant length

                        for soma_mesh,sdf in zip(valid_soma_meshes,valid_soma_segments_width):


                            # ---------- No longer doing the extra checks in here --------- #


                            curr_side_len_check = side_length_check(soma_mesh,side_length_ratio_threshold)
                            curr_volume_check = soma_volume_check(soma_mesh,volume_mulitplier)


                            if curr_side_len_check and curr_volume_check:
                                #check if we can split this into two
                                to_add_list.append(soma_mesh)
                                to_add_list_sdf.append(sdf)

                                """  -----------Removed 1/12: When trying to force a split between them 
                                possible_smoothness = [0.2,0.05,0.01]
                                for smooth_value in possible_smoothness:
                                    #1) Run th esegmentation algorithm again to segment the mesh (had to run the higher smoothing to seperate some)
                                    mesh_extra, mesh_extra_sdf = tu.mesh_segmentation(soma_mesh,clusters=3,smoothness=smooth_value,verbose=True)
                                    mesh_extra = np.array(mesh_extra)

                                    #2) Filter out meshes by sizs and sdf threshold
                                    mesh_extra_lens = np.array([len(kk.faces) for kk in mesh_extra])
                                    filtered_meshes_idx = np.where((mesh_extra_lens >= soma_size_threshold) & (mesh_extra_lens <= soma_size_threshold_max) & (mesh_extra_sdf>soma_width_threshold))[0]

                                    if len(filtered_meshes_idx) >= 2:
                                        if verbose:
                                            print(f"Breakin on smoothness: {smooth_value}")
                                        break

                                if len(filtered_meshes_idx) >= 2:
                                    filtered_meshes = mesh_extra[filtered_meshes_idx]
                                    filtered_meshes_sdf = mesh_extra_sdf[filtered_meshes_idx]

                                    to_add_list_retry = []
                                    to_add_list_sdf_retry = []

                                    for f_m,f_m_sdf in zip(filtered_meshes,filtered_meshes_sdf):
                                        curr_side_len_check_retry = side_length_check(f_m,side_length_ratio_threshold)
                                        curr_volume_check_retry = soma_volume_check(f_m,volume_mulitplier)

                                        if curr_side_len_check_retry and curr_volume_check_retry:
                                            to_add_list_retry.append(f_m)
                                            to_add_list_sdf_retry.append(f_m_sdf)

                                    if len(to_add_list_retry)>1:
                                        if verbose:
                                            print("Using the new feature that split the soma further into more groups")
                                        to_add_list += to_add_list_retry
                                        to_add_list_sdf += to_add_list_sdf_retry

                                    else:
                                        to_add_list.append(soma_mesh)
                                        to_add_list_sdf.append(sdf)


                                else:
                                    to_add_list.append(soma_mesh)
                                    to_add_list_sdf.append(sdf)
                                """

                            else:
                                # ---------- 1/7 Addition: Trying one more additional cgal segmentation to see if there is actually a soma ---
                                """
                                Pseudocode: 
                                1) Run th esegmentation algorithm again to segment the mesh
                                2) Filter out meshes by sizs and sdf threshold
                                3) If there are any remaining meshes, pick the largest sdf mesh and test for volume and side length check
                                --> if matches then adds
                                """

                                print(f"->Attempting retry of soma because failed first checks: "
                                         f"soma_mesh = {soma_mesh}, curr_side_len_check = {curr_side_len_check}, curr_volume_check = {curr_volume_check}")
                                #1) Run th esegmentation algorithm again to segment the mesh
                                mesh_extra, mesh_extra_sdf = tu.mesh_segmentation(soma_mesh,clusters=3,smoothness=0.2,verbose=True)
                                mesh_extra = np.array(mesh_extra)

                                #2) Filter out meshes by sizs and sdf threshold
                                mesh_extra_lens = np.array([len(kk.faces) for kk in mesh_extra])
                                filtered_meshes_idx = np.where((mesh_extra_lens >= soma_size_threshold) & (mesh_extra_lens <= soma_size_threshold_max) & (mesh_extra_sdf>soma_width_threshold))[0]


                                if len(filtered_meshes_idx) > 0:
                                    filtered_meshes = mesh_extra[filtered_meshes_idx]
                                    filtered_meshes_sdf = mesh_extra_sdf[filtered_meshes_idx]

                                    sdf_winning_index = np.argmax(filtered_meshes_sdf)
                                    soma_mesh_retry = filtered_meshes[sdf_winning_index]
                                    sdf_retry = filtered_meshes_sdf[sdf_winning_index]

                                    curr_side_len_check_retry = side_length_check(soma_mesh_retry,side_length_ratio_threshold)
                                    curr_volume_check_retry = soma_volume_check(soma_mesh_retry,volume_mulitplier)

                                    if curr_side_len_check_retry and curr_volume_check_retry:
                                        to_add_list.append(soma_mesh_retry)
                                        to_add_list_sdf.append(sdf_retry)
                                    else:
                                        print(f"--->This soma mesh was not added because failed retry of sphere validation:\n "
                                             f"soma_mesh = {soma_mesh_retry}, curr_side_len_check = {curr_side_len_check_retry}, curr_volume_check = {curr_volume_check_retry}")
                                        continue
                                else:
                                    print(f"Could not find valid soma mesh in retry")
                                    continue


                        n_failed_inner_soma_loops = 0

                    else:
                        n_failed_inner_soma_loops += 1

                total_soma_list_sdf += to_add_list_sdf
                total_soma_list += to_add_list

                # --------------- KEEP TRACK IF FAILED TO FIND SOMA (IF TOO MANY FAILS THEN BREAK)
                if n_failed_inner_soma_loops >= max_fail_loops:
                    print(f"breaking inner loop because {max_fail_loops} soma fails in a row")
                    break


            # --------------- KEEP TRACK IF FAILED TO FIND SOMA (IF TOO MANY FAILS THEN BREAK)
            if somas_found_in_big_loop == False:
                no_somas_found_in_big_loop += 1
                if no_somas_found_in_big_loop >= max_fail_loops:
                    print(f"breaking because {max_fail_loops} fails in a row in big loop")
                    break

            else:
                no_somas_found_in_big_loop = 0





        total_soma_list_revised = total_soma_list
        total_soma_list_revised_sdf = total_soma_list_sdf

        run_time = time.time() - global_start_time

        print(f"\n\n\n Total time for run = {time.time() - global_start_time}")
        print(f"Before Filtering the number of somas found = {len(total_soma_list_revised)}")


        #need to erase all of the temporary files ******
        #import shutil
        #shutil.rmtree(directory)

        """
        Running the extra tests that depend on
        - border vertices
        - how well the poisson matches the backtracked soma to the real mesh
        - other size checks

        """
        filtered_soma_list = []
        filtered_soma_list_sdf = []

        for yy,(soma_mesh,curr_soma_sdf) in enumerate(zip(total_soma_list_revised,total_soma_list_revised_sdf)):
            if verbose:
                print(f"\n---Performing Soma Mesh Backtracking to original mesh for poisson soma {yy}")
            soma_mesh_poisson = deepcopy(soma_mesh)
                #print("About to find original mesh")

            try:
                if verbose:
                    print(f"backtrack_soma_size_threshold = {backtrack_soma_size_threshold}")
                soma_mesh_list,_ = original_mesh_soma(
                                                original_mesh = recov_orig_mesh,
                                                mesh=soma_mesh_poisson,
                                                soma_size_threshold=backtrack_soma_size_threshold,
                                                match_distance_threshold=backtrack_match_distance_threshold,
                                                verbose = verbose)

            except:
                import traceback
                traceback.print_exc()
                print("--->This soma mesh was not added because Was not able to backtrack soma to mesh")
                continue

            if soma_mesh_list is None:
                print("--->This soma mesh was not added because Was not able to backtrack soma to mesh")
                continue



            if verbose: 
                print(f"After backtrack the found {len(soma_mesh_list)} possible somas: {soma_mesh_list} ")

            for rr,soma_mesh in enumerate(soma_mesh_list):
                if verbose:
                    print(f"\n--- working on backtrack soma {rr}: {soma_mesh}")

                curr_side_len_check = side_length_check(soma_mesh,side_length_ratio_threshold)
                curr_volume_check = soma_volume_check(soma_mesh,volume_mulitplier)


                # -------- 1/12 Addition: Does a second round of segmentation after to see if can split somas at all ---- #
                if (not curr_side_len_check) or (not curr_volume_check):
                    print("Trying backtrack segmentation")
                    mesh_tests,mesh_tests_sdf = tu.mesh_segmentation(soma_mesh,clusters=3,smoothness=0.2)

                    soma_mesh_filtered = []
                    soma_mesh_sdf_filtered = []

                    for m_test,m_test_sdf in zip(mesh_tests,mesh_tests_sdf):

                        if len(m_test.faces) >= backtrack_soma_size_threshold and m_test_sdf >=soma_width_threshold:

                            if side_length_check(m_test,side_length_ratio_threshold) and soma_volume_check(m_test,volume_mulitplier):

                                soma_mesh_filtered.append(m_test)
                                soma_mesh_sdf_filtered.append(m_test_sdf)

                    if len(soma_mesh_filtered) == 0:
                        print(f"--->This soma mesh was not added because it did not pass the sphere validation EVEN AFTER SEGMENTATION:\n "
                         f"soma_mesh = {soma_mesh}, curr_side_len_check = {curr_side_len_check}, curr_volume_check = {curr_volume_check}")
                        continue
                else:


                    soma_mesh_filtered = [soma_mesh]
                    soma_mesh_sdf_filtered = [curr_soma_sdf]



                #tu.write_neuron_off(soma_mesh_poisson,"original_poisson.off")
                #If made it through all the checks then add to final list
                filtered_soma_list += soma_mesh_filtered
                filtered_soma_list_sdf += soma_mesh_sdf_filtered

        """
        Need to delete all files in the temp folder *****
        """

        # ----------- 11 /11 Addition that does a last step segmentation of the soma --------- #
        #return total_soma_list, run_time
        #return total_soma_list_revised,run_time,total_soma_list_revised_sdf

        """
        Things we should ask about the segmentation:

        Advantages: 
        1) could help filter away negatives

        Disadvantages:
        1) Can actually cut up the soma and then filter away the soma (not what we want)
        2) Could introduce a big hole (don't think can guard against this)
        """


        #filtered_soma_list_saved = copy.deepcopy(filtered_soma_list)

        if len(filtered_soma_list) > 0:
            filtered_soma_list_revised = []
            filtered_soma_list_sdf_revised = []
            for f_soma,f_soma_sdf in zip(filtered_soma_list,filtered_soma_list_sdf):

                print("Skipping the segmentatio filter at end")
                if not (len(f_soma.faces) >= last_size_threshold and f_soma_sdf >= soma_width_threshold):
                    print(f"Soma (size = {len(f_soma.faces)}, width={soma_width_threshold}) did not pass thresholds (size threshold={last_size_threshold}, width threshold = {soma_width_threshold}) ")
                    continue


                print("removing mesh interior before segmentation")
                f_soma = tu.remove_mesh_interior(f_soma,size_threshold_to_remove=size_threshold_to_remove)

                print("Doing the soma segmentation filter at end")

                meshes_split,meshes_split_sdf = tu.mesh_segmentation(
                    mesh = f_soma,
                    smoothness=0.5
                )

                #applying the soma width and the soma size threshold
                above_width_threshold_mask = meshes_split_sdf>=soma_width_threshold
                meshes_split_sizes = np.array([len(k.faces) for k in meshes_split])
                above_size_threshold_mask = meshes_split_sizes >= last_size_threshold

                above_width_threshold_idx = np.where(above_width_threshold_mask & above_size_threshold_mask)[0]
                if len(above_width_threshold_idx) == 0:
                    print(f"No split meshes were above the width threshold ({soma_width_threshold}) and size threshold ({last_size_threshold}) so continuing")
                    print(f"So just going with old somas")

                    f_soma_final = f_soma
                    f_soma_sdf_final = f_soma_sdf

                else:
                    meshes_split = np.array(meshes_split)
                    meshes_split_sdf = np.array(meshes_split_sdf)

                    meshes_split_filtered = meshes_split[above_width_threshold_idx]
                    meshes_split_sdf_filtered = meshes_split_sdf[above_width_threshold_idx]

                    #way to choose the index of the top candidate
                    top_candidate = 0

                    largest_hole_before_seg = tu.largest_hole_length(f_soma)
                    largest_hole_after_seg = tu.largest_hole_length(meshes_split_filtered[top_candidate])

                    print(f"Largest hole before segmentation = {largest_hole_before_seg}, after = {largest_hole_after_seg},")
                    if largest_hole_before_seg > 0:
                          print(f"\nratio = {largest_hole_after_seg/largest_hole_before_seg}, difference = {largest_hole_after_seg - largest_hole_before_seg}")

                    if largest_hole_after_seg < largest_hole_threshold:
                        f_soma_final = meshes_split_filtered[top_candidate]
                        f_soma_sdf_final = meshes_split_sdf_filtered[top_candidate]
                    else:
                        f_soma_final = f_soma
                        f_soma_sdf_final = f_soma_sdf


                filtered_soma_list_revised.append(f_soma_final)
                filtered_soma_list_sdf_revised.append(f_soma_sdf_final)




            filtered_soma_list = np.array(filtered_soma_list_revised)
            filtered_soma_list_sdf = np.array(filtered_soma_list_sdf_revised)

            """
            # ----------- 1/7/21 ---------------#
            Now was to stitch the somas together if they are touching

            """
            if len(filtered_soma_list)>1:
                connected_meshes_components = tu.mesh_list_connectivity(meshes=filtered_soma_list,
                                         main_mesh=recov_orig_mesh,
                                                            return_connected_components=True)

                filtered_soma_list_components = np.array([submesh_ops.combine(filtered_soma_list[k]) for k in connected_meshes_components])
                filtered_soma_list_sdf_components = np.array([np.mean(filtered_soma_list_sdf[k]) for k in connected_meshes_components])
            elif len(filtered_soma_list)==1:
                filtered_soma_list_components = filtered_soma_list
                filtered_soma_list_sdf_components = filtered_soma_list_sdf
            else:
                filtered_soma_list_components = []
                filtered_soma_list_sdf_components = np.array([])
        else:
            filtered_soma_list_components = []
            filtered_soma_list_sdf_components = np.array([])



        #----------- 1/9 Addition: Final Size Threshold ------------- #
        if verbose:
            print(f"filtered_soma_list_components = {filtered_soma_list_components}")
        filtered_soma_list_components_new = []
        filtered_soma_list_sdf_components_new = []

        for soma_mesh, soma_mesh_sdf in zip(filtered_soma_list_components,filtered_soma_list_sdf_components):
            # --------- 1/9: Extra Size Threshold For Somas ------------- #
            if len(soma_mesh.faces) < backtrack_soma_size_threshold:
                print(f"--->This soma mesh with size {len(soma_mesh.faces)} was not bigger than the threshold {backtrack_soma_size_threshold}")
                continue
            else:
                filtered_soma_list_components_new.append(soma_mesh)
                filtered_soma_list_sdf_components_new.append(soma_mesh_sdf)

        filtered_soma_list_components = filtered_soma_list_components_new
        filtered_soma_list_sdf_components = np.array(filtered_soma_list_sdf_components_new)

        if filter_inside_somas:
            if len(filtered_soma_list_components)>1:
                keep_indices = tu.filter_away_inside_meshes(mesh_list = filtered_soma_list_components,
                                            distance_type="shortest_vertex_distance",
                                            distance_threshold = 2000,
                                            inside_percentage_threshold = 0.20,
                                            verbose = False,
                                            return_meshes = False,
                                            )
            else:
                keep_indices = np.arange(len(filtered_soma_list_components))

            filtered_soma_list_components = [k for i,k in enumerate(filtered_soma_list_components) if i in keep_indices]
            filtered_soma_list_sdf_components = filtered_soma_list_sdf_components[keep_indices]

    

    
    return list(filtered_soma_list_components),run_time,filtered_soma_list_sdf_components


def soma_indentification(
    mesh_decimated,
    verbose=False,
    **soma_extraction_parameters
    ):

    (total_soma_list,
     run_time,
     total_soma_list_sdf) = extract_soma_center(
        mesh = mesh_decimated,
        verbose = verbose,
        **soma_extraction_parameters
    )

    soma_products = pipeline.StageProducts(
        soma_extraction_parameters = soma_extraction_parameters,
        soma_meshes=total_soma_list,
        soma_run_time=run_time,
        soma_sdfs=total_soma_list_sdf,
    )

    return soma_products
    
    
#--- from mesh_tools ---
from mesh_tools import meshlab
from mesh_tools import trimesh_utils as tu

#--- from datasci_tools ---
from datasci_tools import numpy_dep as np
from datasci_tools import numpy_utils as nu
from datasci_tools import pipeline

from . import parameters
from . import submesh_ops

