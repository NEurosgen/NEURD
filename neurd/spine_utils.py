'''



To Do: Want to add how close a spine is to upstream and downstream endpoint



'''
import copy
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from pathlib import Path
from scipy.spatial import KDTree
try:
    import seaborn as sns
except ImportError:  # pragma: no cover
    sns = None
import time
import time 
from datasci_tools import numpy_dep as np
from datasci_tools import general_utils as gu
from datasci_tools import matplotlib_utils as mu

try:
    import cgal_Segmentation_Module as csm
except:
    pass


volume_divisor = 1000_000_000

spine_attributes = ["mesh_face_idx",
                      "mesh",
                      "neck_face_idx",
                      "head_face_idx",
                      "neck_sdf",
                      "head_sdf",
                      "head_width",
                      "neck_width",
                      "volume",
                    "spine_id",
                    "sdf",
                    #features to help with electrical properties
                     "endpoints_dist",
                     "upstream_dist",
                     "downstream_dist",
                     "coordinate",
                    "coordinate_border_verts",
                     "closest_sk_coordinate",
                     "closest_branch_face_idx",
                     "closest_face_dist",
                      "closest_face_coordinate",
                      "soma_distance",
                      "soma_distance_euclidean",
                      "compartment",
                      "limb_idx",
                      "branch_idx",
                        "skeleton",
                        "skeletal_length",
                    
                    #attributes for more spine features
                    "bbox_oriented_side_lengths",
                    "head_bbox_oriented_side_lengths",
                    "neck_bbox_oriented_side_lengths",
                    "head_mesh_splits",
                    "head_mesh_splits_face_idx",
                    
                    # attributes from the branch_obj
                    "compartment",
                    "branch_width_overall",
                    "branch_skeletal_length",
                    "branch_width_at_base",
                     ]
head_neck_shaft_dict = dict(no_label=-1,
                            head=0,
                           neck=1,
                           shaft=2,
                           no_head=3,
                           bouton=4,
                           non_bouton=5)

    

    

head_neck_shaft_dict_inverted = gu.invert_mapping(head_neck_shaft_dict,one_to_one=True)


neck_color_default = "gold"#"yellow"#,"pink"#,"yellow"# "aqua"
head_color_default = "red"
no_head_color_default = "black"
shaft_color_default = "lime"
bouton_color_default = "orange"
base_color_default = "blue"
center_color_default = "orange"

spine_bouton_labels_colors_dict = dict(
head = "red",
neck = "aqua",
shaft = "plum",
no_head = "limegreen",
bouton = "greenyellow",
non_bouton = "aqua",
no_label = "royalblue"
)

# ---------- queries for differnt types of spines -----------
spine_volume_to_spine_area_min = 0.01
# slab_query_1_dj = "((shaft_border_area > 0.18) AND (spine_volume < 0.05)) OR (shaft_border_area is NULL)"
# slab_query_2_dj = f"((NOT (spine_volume is NULL)) AND (spine_volume_to_spine_area < {spine_volume_to_spine_area_min}))"

slab_query_1 = "((shaft_border_area > 0.18) and (spine_volume < 0.05)) or (shaft_border_area != shaft_border_area)"
slab_query_2 = f"((spine_volume == spine_volume) and (spine_volume_to_spine_area < {spine_volume_to_spine_area_min}))"
slab_query_3 = f"(shaft_border_area > 0.5)"
slab_query_4 = f"(spine_to_shaft_border_area <= 2)"
slab_query_5 = f"(spine_area_per_faces) < 0.005"
long_merge = f"(head_skeletal_length > 2000)"
too_fat_of_neck = f"(neck_width_ray_80_perc > 400) and (n_heads > 0)"


measurable_features = ("spine_area",
            "spine_n_faces",
            "spine_volume",
            "spine_skeletal_length",
            "spine_width_ray",
            "spine_width_ray_80_perc",
                      )

n_heads_max = 1
min_distance_from_endpt = 2000
default_spine_restrictions = [
    f"downstream_dist > {min_distance_from_endpt}",
    f"upstream_dist > {min_distance_from_endpt}",
    f"n_heads <= {n_heads_max}",
]

spine_synapse_rename_dict  =dict(
            spine_compartment = "spine_compartment",
            syn_spine_volume = "spine_volume",
            syn_spine_area = "spine_area",
            syn_spine_width_ray_80_perc = "spine_width_ray_80_perc"
        )

def false_positive_queries(
    table_type = "pandas",
    invert = False,
    include_axons = True):
    return_queries = [
        slab_query_1,
        slab_query_2,
        slab_query_3,
        slab_query_4,
        slab_query_5,
        long_merge,
        #too_fat_of_neck
    ]
    if include_axons:
        return_queries.append("compartment == 'axon'")
    if invert:
        return_queries = [f"not ({k})" for k in return_queries]
    if table_type == "dj":
        from datasci_tools import dj_utils as dju
        return_queries  = [dju.dj_query_from_pandas_query(k) for k in return_queries]

    return return_queries

def spine_table_restriction_high_confidence(
    table_type = "pandas",
    include_default_restrictions=True,
    return_query_str = False,):
    return_value = false_positive_queries(table_type,invert = True)
    if include_default_restrictions:
        additional_queries = default_spine_restrictions.copy()
        if table_type == "dj":
            from datasci_tools import dj_utils as dju
            additional_queries = [dju.dj_query_from_pandas_query(k) for k in additional_queries]
        return_value += additional_queries
        
    if return_query_str:
        return_value = pu.query_str_from_list(return_value,table_type = table_type)
        
    return return_value
    

def filter_away_fp_from_df(
    df,
    fp_queries = None,
    verbose = False,
    eta = 0.000001
    ):
    if verbose:
        print(f"Attempting to filter away fp")
    
    if fp_queries is None:
        fp_queries = false_positive_queries()
        
    df["spine_volume_to_spine_area"] = df["spine_volume"] / (df["spine_area"] + eta)
    df["spine_to_shaft_border_area"] = df["spine_area"] / (df["shaft_border_area"] + eta)
    df["spine_area_per_faces"] = df["spine_area"]/df["spine_n_faces"]
    
    return_df = pu.query_table_from_list(
        df,
        restrictions = [f"not ({k})" for k in fp_queries],
        verbose_filtering = verbose, 
    )
    
    return return_df



class Spine:
    """
    Classs that will hold information about a spine extracted from a neuron

    Attributes
    ----------
    mesh_face_idx: a list of face indices of the branch that belong to the spine mesh
    mesh: the submesh of the branch that represent the spine (mesh_face_idx indexed into the branch mesh
    neck_face_idx: a list of face indices of the spine’s mesh that were classified as the neck (can be empty if not detected)
    head_face_idx:  list of face indices of the spine’s mesh that were classified as the head (can be empty if not detected)
    neck_sdf: the sdf value of the neck submesh from the clustering algorithm used to segment the head from the neck 
    head_sdf: the sdf value of the head submesh from the clustering algorithm used to segment the head from the neck 
    head_width: a width approximation using ray tracing of the head submesh
    neck_width:  a width approximation using ray tracing of the head submesh
    volume: volume of entire mesh
    spine_id: unique identifier for spine
    sdf:  the sdf value of the spine submesh from the clustering algorithm used to segment the spine from the branch mesh
    endpoints_dist: skeletal walk distance of the skeletal point closest to the start of the spine protrusion to the branch skeletal endpoints
    upstream_dist: skeletal walk distance of the skeletal point closest to the start of the spine protrusion to the upstream branch skeletal endpoint
    downstream_dist: skeletal walk distance of the skeletal point closest to the start of the spine protrusion to the downstream branch skeletal endpoint
    coordinate_border_verts: 
    coordinate: one coordinate of the border vertices to be used for spine locations
    bbox_oriented_side_lengths
    head_bbox_oriented_side_lengths
    neck_bbox_oriented_side_lengths
    head_mesh_splits
    head_mesh_splits_face_idx
    branch_width_overall
    branch_skeletal_length
    branch_width_at_base
    skeleton: surface skeleton over the spine mesh
    skeletal_length: length of spine skeleton

    # -- attributes similar to those of spine attribute
    closest_branch_face_idx
    closest_sk_coordinate: 3D location in space of closest skeletal point on branch for which spine is located
    closest_face_coordinate: center coordinate of closest mesh face  on branch for which spine is located
    closest_face_dist: distance from synapse coordinate to closest_face_coordinate
    soma_distance: skeletal walk distance from synapse to soma
    soma_distance_euclidean: straight path distance from synapse to soma center
    compartment: the compartment of the branch that the spine is located on
    limb_idx: the limb identifier that the spine is located on
    branch_idx: the branch identifier that the spine is located on

    """
    def __init__(
        self,
        mesh,
        calculate_spine_attributes=False,
        branch_obj = None,
        **kwargs):
        if type(mesh) == spu.Spine:
            for k,v in mesh.export().items():
                #setattr(self,k,v)
                try:
                    setattr(self,k,v)
                except:
                    setattr(self,f"_{k}",v)
            return 
            
        for a in spine_attributes:
            
            try:
                setattr(self,a,None)
            except:
                setattr(self,f"_{a}",None)
                
        self.mesh = mesh
        #if synapse_dict is not None:
        for k,v in kwargs.items():
            if k in spine_attributes:
                try:
                    setattr(self,k,v)
                except:
                    setattr(self,f"_{k}",v)
                
        if calculate_spine_attributes:
            self.calculate_spine_attributes(branch_obj=branch_obj)
            
                
    def calculate_spine_attributes(self,branch_obj=None):
        spu.calculate_spine_attributes(self,branch_obj=branch_obj)
        
        
    def export(
        self,  
        attributes_to_skip = None,
        attributes_to_add = None,
        **kwargs):
        return spu.export(
            self,
            attributes_to_skip=attributes_to_skip,
            attributes_to_add=attributes_to_add,
            **kwargs
            )
        
        
    @property
    def base_coordinate(self):
        return self.coordinate
    @property
    def base_coordinate_x_nm(self):
        return self.base_coordinate[0]
    @property
    def base_coordinate_y_nm(self):
        return self.base_coordinate[1]
    @property
    def base_coordinate_z_nm(self):
        return self.base_coordinate[2]
    
    @property
    def bbox_oriented_side_lengths(self):
        if self._bbox_oriented_side_lengths is None:
            self._bbox_oriented_side_lengths = meshu.bounding_box_side_lengths_sorted(self.mesh)
        return self._bbox_oriented_side_lengths
    
    @property
    def head_bbox_oriented_side_lengths(self):
        if self.head_mesh is None:
            return [None,None,None]
        if len(self.head_mesh.faces) == 0:
            return [0,0,0]
        if self._head_bbox_oriented_side_lengths is None:
            self._head_bbox_oriented_side_lengths = meshu.bounding_box_side_lengths_sorted(self.head_mesh)
        return self._head_bbox_oriented_side_lengths
    
    @property
    def neck_bbox_oriented_side_lengths(self):
        if self.neck_mesh is None:
            return [None,None,None]
        if len(self.neck_mesh.faces) == 0:
            return [0,0,0]
        if self._neck_bbox_oriented_side_lengths is None:
            self._neck_bbox_oriented_side_lengths = meshu.bounding_box_side_lengths_sorted(self.neck_mesh)
        return self._neck_bbox_oriented_side_lengths
    
    @property
    def spine_bbox_oriented_side_max(self):
        return self.bbox_oriented_side_lengths[0]
    bbox_oriented_side_max = spine_bbox_oriented_side_max
    @property
    def spine_bbox_oriented_side_middle(self):
        return self.bbox_oriented_side_lengths[1]
    bbox_oriented_side_middle = spine_bbox_oriented_side_middle
    @property
    def spine_bbox_oriented_side_min(self):
        return self.bbox_oriented_side_lengths[2]
    bbox_oriented_side_min = spine_bbox_oriented_side_min
    
    @property
    def head_bbox_oriented_side_max(self):
        return self.head_bbox_oriented_side_lengths[0]
    @property
    def head_bbox_oriented_side_middle(self):
        return self.head_bbox_oriented_side_lengths[1]
    @property
    def head_bbox_oriented_side_min(self):
        return self.head_bbox_oriented_side_lengths[2]
    
    @property
    def neck_bbox_oriented_side_max(self):
        return self.neck_bbox_oriented_side_lengths[0]
    @property
    def neck_bbox_oriented_side_middle(self):
        return self.neck_bbox_oriented_side_lengths[1]
    @property
    def neck_bbox_oriented_side_min(self):
        return self.neck_bbox_oriented_side_lengths[2]
    
    @property
    def head_mesh(self):
        return spu.head_mesh(self)
    
#     def head_mesh_splits(
#         self,
#         n_faces_threshold = 10):
#         if self.n_faces_head == 0:
#             return []
#         else:
#             return tu.split_significant_pieces(
#                 self.head_mesh,
#                 significance_threshold=n_faces_threshold,
#                 connectivity="vertices"
#             )

    @property
    def head_mesh_splits(self):
        if self.head_mesh is None:
            return None
        if self._head_mesh_splits is None:
            self._head_mesh_splits,self._head_mesh_splits_face_idx = spu.split_head_mesh(
                self,
            )
        return self._head_mesh_splits 
    
    @property
    def head_mesh_splits_face_idx(self):
        if self.head_mesh is None:
            return None
        if self._head_mesh_splits_face_idx is None:
            self._head_mesh_splits,self._head_mesh_splits_face_idx = spu.split_head_mesh(
                self,
            )
            
        return self._head_mesh_splits_face_idx
    
    @property
    def n_heads(self):
        if self.head_mesh_splits is None:
            return None
        return len(self.head_mesh_splits)
    
    def head_mesh_splits_from_index(self,index):
        return spu.head_mesh_splits_from_index(self,index=index)
    
    @property
    def head_mesh_splits_max(self):
        return spu.head_mesh_splits_from_index(self,index=0)
    
    @property
    def head_mesh_splits_min(self):
        return spu.head_mesh_splits_from_index(self,index=self.n_heads-1)
    
    @property
    def n_faces(self):
        return len(self.mesh.faces)
    
    @property
    def n_vertices(self):
        return len(self.mesh.vertices)
    
    @property
    def n_faces_head(self):
        if self.head_face_idx is None:
            return None
        return len(self.head_face_idx)
    
    @property
    def n_faces_neck(self):
        if self.neck_face_idx is None:
            return None
        return len(self.neck_face_idx)
    
    @property
    def neck_mesh(self):
        return spu.neck_mesh(self)
    
    @property
    def area(self):
        return self.mesh.area
    
    @property
    def no_head_mesh(self):
        return spu.no_head_mesh(self)
    
    @property
    def head_exist(self):
        return spu.head_exist(self)
    
    @property
    def no_head_face_idx(self):
        if self.head_exist:
            return np.array([])
        return self.neck_face_idx
    
    @property
    def skeletal_length(self):
        if self.skeleton is None:
            self.calculate_skeleton()
        return self._skeletal_length
    
    @property
    def skeleton(self):
        if self._skeleton is None:
            self.calculate_skeleton()
        return self._skeleton
    
    def calculate_head_neck(self,**kwargs):
        (self.head_face_idx,
         self.neck_face_idx,
         self.head_sdf,
         self.neck_sdf,
         self.head_width,
         self.neck_width) = spine_head_neck(
            mesh=self.mesh,
            return_meshes = False,
            no_head_coordinates = self.coordinate_border_verts,
            return_sdf = True,
            **kwargs)
        self._head_mesh_splits = None
    def calculate_face_idx(self,
                      original_mesh=None,
                    original_mesh_kdtree=None,
                          **kwargs):
        self.mesh_face_idx = spu.calculate_face_idx(self,
                                                   original_mesh=original_mesh,
                                                   original_mesh_kdtree=original_mesh_kdtree,
                                                   **kwargs)
        
        
    def plot_head_neck(self,**kwargs):
        spu.plot_head_neck(self)
        
    def calculate_closest_mesh_sk_coordinates(self,branch_obj,**kwargs):
        spu.calculate_spine_obj_mesh_skeleton_coordinates(branch_obj,self,**kwargs)
    
    def calculate_volume(self):
        if self.volume is None:
            self.volume = spu.volume_from_spine(self)
    
    def calculate_skeleton(self):
        self._skeleton = spu.skeleton_from_spine(self)
        self._skeletal_length = sk.calculate_skeleton_distance(self.skeleton)
        
    @property
    def mesh_center(self):
        return self.mesh.centroid
    
    @property
    def mesh_center_x_nm(self):
        return self.mesh_center[0]
    @property
    def mesh_center_y_nm(self):
        return self.mesh_center[1]
    @property
    def mesh_center_z_nm(self):
        return self.mesh_center[2]
    
    @property
    def sdf_mean(self):
        return np.mean(self.sdf)
    
    @property
    def sdf_median(self):
        return np.median_sdf(self.sdf)
    
    @property
    def sdf_90_perc(self):
        return np.percentile(self.sdf,90)
    
    @property
    def sdf_70_perc(self):
        return np.percentile(self.sdf,70)
    
    @property
    def endpoint_dist_0(self):
        if self.endpoints_dist is None:
            return None
        else:
            return self.endpoints_dist[0]
    @property   
    def endpoint_dist_1(self):
        if self.endpoints_dist is None:
            return None
        else:
            return self.endpoints_dist[1]
        
    @property
    def area_of_border_verts(self):
        return area_of_border_verts(self)
    coordinate_border_verts_area = area_of_border_verts
    border_area = area_of_border_verts
    shaft_border_area = area_of_border_verts
    
    #--- Mesh Attribute Functions for spine -----
    @property
    def spine_width_ray(self,**kwargs):
        return width_ray_from_compartment(self,compartment = 'spine',**kwargs)

    @property
    def spine_width_ray_80_perc(self,**kwargs):
        return width_ray_80_perc_from_compartment(self,compartment = 'spine',**kwargs)

    @property
    def spine_area(self,**kwargs):
        return area_from_compartment(self,compartment = 'spine',**kwargs)

    @property
    def spine_volume(self,**kwargs):
        return volume_from_compartment(self,compartment = 'spine',**kwargs)

    @property
    def spine_skeletal_length(self,**kwargs):
        return skeletal_length_from_compartment(self,compartment = 'spine',**kwargs)

    @property
    def spine_n_faces(self,**kwargs):
        return n_faces(self,compartment = 'spine',**kwargs)

    @property
    def spine_bbox_min_x_nm(self,**kwargs):
        return bbox_min_x_nm_from_compartment(self,compartment = 'spine',**kwargs)

    @property
    def spine_bbox_min_y_nm(self,**kwargs):
        return bbox_min_y_nm_from_compartment(self,compartment = 'spine',**kwargs)

    @property
    def spine_bbox_min_z_nm(self,**kwargs):
        return bbox_min_z_nm_from_compartment(self,compartment = 'spine',**kwargs)

    @property
    def spine_bbox_max_x_nm(self,**kwargs):
        return bbox_max_x_nm_from_compartment(self,compartment = 'spine',**kwargs)

    @property
    def spine_bbox_max_y_nm(self,**kwargs):
        return bbox_max_y_nm_from_compartment(self,compartment = 'spine',**kwargs)

    @property
    def spine_bbox_max_z_nm(self,**kwargs):
        return bbox_max_z_nm_from_compartment(self,compartment = 'spine',**kwargs)

    #--- Mesh Attribute Functions for head -----
    @property
    def head_width_ray(self,**kwargs):
        return width_ray_from_compartment(self,compartment = 'head',**kwargs)

    @property
    def head_width_ray_80_perc(self,**kwargs):
        return width_ray_80_perc_from_compartment(self,compartment = 'head',**kwargs)

    @property
    def head_area(self,**kwargs):
        return area_from_compartment(self,compartment = 'head',**kwargs)

    @property
    def head_volume(self,**kwargs):
        return volume_from_compartment(self,compartment = 'head',**kwargs)

    @property
    def head_skeletal_length(self,**kwargs):
        return skeletal_length_from_compartment(self,compartment = 'head',**kwargs)

    @property
    def head_n_faces(self,**kwargs):
        return n_faces(self,compartment = 'head',**kwargs)

    @property
    def head_bbox_min_x_nm(self,**kwargs):
        return bbox_min_x_nm_from_compartment(self,compartment = 'head',**kwargs)

    @property
    def head_bbox_min_y_nm(self,**kwargs):
        return bbox_min_y_nm_from_compartment(self,compartment = 'head',**kwargs)

    @property
    def head_bbox_min_z_nm(self,**kwargs):
        return bbox_min_z_nm_from_compartment(self,compartment = 'head',**kwargs)

    @property
    def head_bbox_max_x_nm(self,**kwargs):
        return bbox_max_x_nm_from_compartment(self,compartment = 'head',**kwargs)

    @property
    def head_bbox_max_y_nm(self,**kwargs):
        return bbox_max_y_nm_from_compartment(self,compartment = 'head',**kwargs)

    @property
    def head_bbox_max_z_nm(self,**kwargs):
        return bbox_max_z_nm_from_compartment(self,compartment = 'head',**kwargs)

    #--- Mesh Attribute Functions for neck -----
    @property
    def neck_width_ray(self,**kwargs):
        return width_ray_from_compartment(self,compartment = 'neck',**kwargs)

    @property
    def neck_width_ray_80_perc(self,**kwargs):
        return width_ray_80_perc_from_compartment(self,compartment = 'neck',**kwargs)

    @property
    def neck_area(self,**kwargs):
        return area_from_compartment(self,compartment = 'neck',**kwargs)

    @property
    def neck_volume(self,**kwargs):
        return volume_from_compartment(self,compartment = 'neck',**kwargs)

    @property
    def neck_skeletal_length(self,**kwargs):
        return skeletal_length_from_compartment(self,compartment = 'neck',**kwargs)

    @property
    def neck_n_faces(self,**kwargs):
        return n_faces(self,compartment = 'neck',**kwargs)

    @property
    def neck_bbox_min_x_nm(self,**kwargs):
        return bbox_min_x_nm_from_compartment(self,compartment = 'neck',**kwargs)

    @property
    def neck_bbox_min_y_nm(self,**kwargs):
        return bbox_min_y_nm_from_compartment(self,compartment = 'neck',**kwargs)

    @property
    def neck_bbox_min_z_nm(self,**kwargs):
        return bbox_min_z_nm_from_compartment(self,compartment = 'neck',**kwargs)

    @property
    def neck_bbox_max_x_nm(self,**kwargs):
        return bbox_max_x_nm_from_compartment(self,compartment = 'neck',**kwargs)

    @property
    def neck_bbox_max_y_nm(self,**kwargs):
        return bbox_max_y_nm_from_compartment(self,compartment = 'neck',**kwargs)

    @property
    def neck_bbox_max_z_nm(self,**kwargs):
        return bbox_max_z_nm_from_compartment(self,compartment = 'neck',**kwargs)
    
    # -- general mesh properties of spine
    @property
    def face_area_mean(self):
        return tu.face_area_mean(self.mesh)
    
    @property
    def face_area_sum(self):
        return tu.face_area_sum(self.mesh)
    
    @property
    def boundary_edges_lengths_sum(self):
        return tu.boundary_edges_lengths_sum(self.mesh)
    
    @property
    def face_area_max(self):
        return tu.face_area_max(self.mesh)
    
    @property 
    def area_to_boundary_length_ratio(self):
        return tu.area_to_boundary_length_ratio(self.mesh)
    
computed_attributes_to_export = (
        "area",
        "sdf_mean",
        "n_faces",
        "n_faces_head",
        "n_faces_neck",
        "mesh_center",
        "sdf_mean",
        "sdf_90_perc",
        "sdf_70_perc",
        "bbox_oriented_side_max",
        "bbox_oriented_side_middle",
        "bbox_oriented_side_min",
        "n_heads",
        "endpoint_dist_0",
        "endpoint_dist_1"
)
    
    
def calculate_soma_distance_euclidean(spine_obj,soma_center=None):
    if soma_center is None:
        return
    else:
        spine_obj.soma_distance_euclidean = np.linalg.norm(soma_center - spine_obj.mesh_center)
        
def calculate_soma_distance_skeletal(spine_obj,upstream_skeletal_length=None):
    if upstream_skeletal_length is None or spine_obj.upstream_dist is None:
        return
    spine_obj.soma_distance = spine_obj.upstream_dist + upstream_skeletal_length 
    
    

def calculate_branch_width_at_base(spine_obj,branch_obj):
    spine_obj.branch_width_at_base = bu.width_array_value_closest_to_coordinate(
                branch_obj,
                spine_obj.base_coordinate
    )
    
branch_overall_features_for_spine = ["width_overall","skeletal_length"]#branch_width_at_base
def calculate_branch_overall_features(
    spine_obj,
    branch_obj,
    branch_features = None):
    if branch_features is None:
        branch_features = branch_overall_features_for_spine

    branch_features = nu.to_list(branch_features)

    for feat in branch_features:
        if "str" in str(type(feat)):
            curr_value = getattr(branch_obj,feat)
        else:
            curr_value = feat(branch_obj)

        setattr(spine_obj,f"branch_{feat}",curr_value)
        
    
        
    
def calculate_spine_attributes(
    spine_obj,
    branch_obj=None,
    calculate_coordinates = True,
    calculate_head_neck = False,
    branch_shaft_mesh_face_idx=None,
    soma_center = None,
    upstream_skeletal_length = None,
    branch_features = None,
    verbose_time=False,
    mesh = None,
    **kwargs):
    """
    Purpose
    -----------
    Given a spine mesh (and potentially the branch object it resides on) calculates descriptive statistics

    Pseudocode
    ----------------
    1) calculates the volume
    2) calculates the bounding box side lengths
    3) calculates branch relative statistics:
        a) closest mesh/skeleton coordinate
        b) distance of spine base to each skeleton endpoint
        c) width of branch obj at the base of the spine
    4) Optionally calculates the head and neck clusters and statistics. spu.calculate_head_neck

    Global Parameters to Set
    ------------------------


    Parameters
    ----------
    spine_obj : _type_
        _description_
    branch_obj : _type_, optional
        _description_, by default None
    calculate_coordinates : bool, optional
        _description_, by default True
    calculate_head_neck : bool, optional
        _description_, by default False
    branch_shaft_mesh_face_idx : _type_, optional
        _description_, by default None
    soma_center : _type_, optional
        _description_, by default None
    upstream_skeletal_length : _type_, optional
        _description_, by default None
    branch_features : _type_, optional
        _description_, by default None
    verbose_time : bool, optional
        _description_, by default False
    mesh : _type_, optional
        _description_, by default None

    Returns
    -------
    _type_
        _description_
    """
    
    if verbose_time:
        st = time.time()
    spine_obj.skeleton
    
    if verbose_time:
        print(f"Time for skeleton = {time.time() - st}")
        st = time.time()
    spine_obj.calculate_volume()
    if verbose_time:
        print(f"Time for volume = {time.time() - st}")
        st = time.time()
    spine_obj.bbox_oriented_side_lengths
    if verbose_time:
        print(f"Time for bbox = {time.time() - st}")
        st = time.time()
    
    if calculate_coordinates:
        if branch_obj is not None:
            spine_obj.calculate_closest_mesh_sk_coordinates(
                branch_obj,
                verbose_time=verbose_time,
                branch_shaft_mesh_face_idx=branch_shaft_mesh_face_idx,
            )
            if verbose_time:
                print(f"Time for closest_coordinate = {time.time() - st}")
                st = time.time()
            spu.calculate_endpoints_dist(branch_obj,spine_obj)
            if verbose_time:
                print(f"Time for calculate_endpoints_dist = {time.time() - st}")
                st = time.time()
            if branch_obj.endpoints_upstream_downstream_idx is not None:
                spu.calculate_upstream_downstream_dist_from_up_idx(
                    spine_obj,
                    branch_obj.endpoints_upstream_downstream_idx[0]
                )
                if verbose_time:
                    print(f"Time for calculate_upstream_downstream_dist_from_up_idx = {time.time() - st}")
                    st = time.time()
                    
                calculate_branch_width_at_base(spine_obj,branch_obj)
            
            
    
    if calculate_head_neck:
        if mesh is None:
            mesh = branch_obj.mesh
        spine_obj = spu.calculate_spine_obj_mesh_skeleton_coordinates(
            spine_obj=spine_obj,
            mesh = mesh,
        )
        spine_obj.calculate_head_neck(**kwargs)
        if verbose_time:
            print(f"Time for calculate_head_neck = {time.time() - st}")
            st = time.time()
        spine_obj.head_mesh_splits
        if verbose_time:
            print(f"Time for head_mesh_splits = {time.time() - st}")
            st = time.time()
            
    
    calculate_soma_distance_euclidean(spine_obj,soma_center)
    calculate_soma_distance_skeletal(spine_obj,upstream_skeletal_length)
    if branch_obj is not None:
        calculate_branch_overall_features(spine_obj,branch_obj,branch_features=branch_features,)
    
    return spine_obj

def mesh_minus_spine_objs(
    spine_objs,
    mesh = None,
    branch_obj=None,
    return_idx = False,):
    """
    Purpose: To get the shaft mesh of a branch
    given a list of spines
    """
    if mesh is None:
        mesh = branch_obj.mesh
    idx = np.delete(
        np.arange(len(mesh.faces)),np.hstack([k.mesh_face_idx for k in spine_objs]
                                      ).astype('int')
    )
    
    if return_idx:
        return idx
    else:
        return mesh.submesh([idx],append=True)

def calculate_spine_attributes_for_list(
    spine_objs,
    branch_obj = None,
    calculate_coordinates = True,
    calculate_head_neck=False,
    verbose_time=False,
    mesh = None,
    **kwargs
    ):
    if calculate_coordinates and len(spine_objs) > 0:
        branch_shaft_mesh_face_idx = mesh_minus_spine_objs(
            spine_objs,
            branch_obj=branch_obj,
            mesh = mesh,
            return_idx = True,
        )
    else:
        branch_shaft_mesh_face_idx = None
    return [spu.calculate_spine_attributes(
        k,
        branch_obj = branch_obj,
        calculate_coordinates=calculate_coordinates,
        calculate_head_neck=calculate_head_neck,
        verbose_time=verbose_time,
        branch_shaft_mesh_face_idx=branch_shaft_mesh_face_idx,
        mesh=mesh,
        **kwargs
    ) for k in spine_objs]
    
        
def is_spine_obj(obj):
    #return str(type(obj)) == str(spu.Spine)
    return obj.__class__ == spu.Spine
        
def calculate_face_idx(spine_obj,
                      original_mesh=None,
                    original_mesh_kdtree=None,
                      **kwargs):
    """
    Purpose: To calculate the original faces 
    of the spine to a reference mesh
    """
    if original_mesh is None and original_mesh_kdtree is None:
        raise Exception("Either original_mesh or original_mesh_kdtree needs to be non None")
    
    return tu.original_mesh_faces_map(original_mesh = original_mesh,
                                     original_mesh_kdtree=original_mesh_kdtree,
                                     submesh = spine_obj.mesh,
                                      exact_match=True,
                                     **kwargs)

def plot_head_neck(spine_obj,
                   neck_color = neck_color_default,
                    head_color = head_color_default,
                   no_head_color = no_head_color_default,
                  verbose = True):
    neck_mesh =spine_obj.neck_mesh
    head_mesh = spine_obj.head_mesh
    no_head_mesh = spine_obj.no_head_mesh
    
    if verbose:
        print(f"head_mesh ({head_color}): {head_mesh}")
        print(f"neck_mesh ({neck_color}): {neck_mesh}")
        print(f"no_head_mesh ({no_head_color}): {no_head_mesh}")

    
    
def mesh_from_name_or_idx(
    spine_obj,
    name=None,
    idx=None,
    largest_component = False):
    if idx is None:
        if name is not None:
            if "face_idx" not in name:
                name = f"{name}_face_idx"
        idx = getattr(spine_obj,name,None)
    
    if idx is None:
        return None
    elif len(idx) == 0:
        return tu.empty_mesh()
    else:
        mesh =  spine_obj.mesh.submesh([idx],append=True)
        if largest_component:
            mesh = tu.largest_conn_comp(mesh)
            
        return mesh
    
def head_mesh(spine_obj):
    return mesh_from_name_or_idx(spine_obj,name="head")
def neck_mesh(spine_obj):
    return mesh_from_name_or_idx(spine_obj,name="neck",largest_component=False)
def no_head_mesh(spine_obj):
    return mesh_from_name_or_idx(spine_obj,name="no_head")
def head_mesh_splits_face_idx_by_index(spine_obj,index):
    if spine_obj.head_mesh is None:
        return None
    return np.array(spine_obj.head_face_idx)[np.where(spine_obj.head_mesh_splits_face_idx == index)[0]]
def head_mesh_splits_from_index(spine_obj,index):
    return mesh_from_name_or_idx(spine_obj,idx = head_mesh_splits_face_idx_by_index(spine_obj,index))
"""
def head_mesh(spine_obj):
    if spine_obj.head_face_idx is None or len(spine_obj.head_face_idx) == 0:
        return tu.empty_mesh()
    else:
        return spine_obj.mesh.submesh([spine_obj.head_face_idx],append=True)
    
def neck_mesh(spine_obj):
    if spine_obj.neck_face_idx is None:
        return tu.empty_mesh()
    else:
        return spine_obj.mesh.submesh([spine_obj.neck_face_idx],append=True)
    
def no_head_mesh(spine_obj):
    if spine_obj.no_head_face_idx is None or len(spine_obj.no_head_face_idx) == 0:
        return tu.empty_mesh()
    else:
        return spine_obj.mesh.submesh([spine_obj.no_head_face_idx],append=True)
"""
    
def head_exist(spine_obj):
    if spine_obj.head_face_idx is None or len(spine_obj.head_face_idx) == 0:
        return False
    return True



def export(
    spine_obj,
    attributes_to_skip = None,
    attributes_to_add = None,
    suppress_errors = True,
    attributes = None,
    default_value = None):
    
    if attributes is None:
        curr_attributes = spine_attributes
        if attributes_to_skip is not None:
            attributes_to_skip= nu.to_list(attributes_to_skip)
            curr_attributes = np.setdiff1d(curr_attributes,attributes_to_skip)

        if attributes_to_add is not None:
            attributes_to_add=nu.to_list(attributes_to_add)
            curr_attributes = np.union1d(curr_attributes,attributes_to_add)
    else:
        curr_attributes = attributes

    return_dict = {}
    for k in curr_attributes:
        try:
            curr_value = getattr(spine_obj,k,default_value)
        except:
            if suppress_errors:
                curr_value = default_value
            else:
                raise Exception("")
                
        return_dict[k] = curr_value
    return return_dict
        
    #return {k:getattr(spine_obj,k,None) for k in curr_attributes}
    




def set_branch_spines_obj(branch_obj,
                               calculate_mesh_face_idx = True,
                                verbose = False):
    """
    Purpose: To set the spine 0bj
    attribute for a branch

    Pseudocode: 
    1) store the spine mesh and the volume
    2) calculate the neck face idx and sdf
    2) optional: calculate the mesh face_idx
    """
    curr_spines = branch_obj.spines
    curr_spines_vol = branch_obj.spines_volume
    
    if curr_spines is None:
        branch_obj.spines_obj = None
        return 

    spines_obj = []
    branch_kd = tu.mesh_kdtree_face(branch_obj.mesh)

    if not verbose:
        tqu.turn_off_tqdm()
    else:
        tqu.turn_on_tqdm()

    for s,vol in tqu.tqdm(zip(curr_spines,curr_spines_vol)):
        sp_obj = spu.Spine(mesh=s,volume=vol)
        sp_obj.calculate_head_neck()

        if calculate_mesh_face_idx:
            sp_obj.calculate_face_idx(original_mesh=branch_obj.mesh,
                                      original_mesh_kdtree=branch_kd)
        spines_obj.append(sp_obj)

    branch_obj.spines_obj = spines_obj
    
def set_neuron_spine_attribute(neuron_obj,func,
                              verbose = False):
    """
    Purpose: To set the spines obj
    for all branches in the neuron obj
    """
    for limb_name in neuron_obj.get_limb_names():
        if verbose:
            print(f"Working on Limb {limb_name}")
        limb_obj = neuron_obj[limb_name]
        for b_idx in limb_obj.get_branch_names():
            if verbose:
                print(f"--- branch {b_idx}")
            func(branch_obj = limb_obj[b_idx])
    
def set_neuron_spines_obj(neuron_obj,
                         verbose = False):
    spu.set_neuron_spine_attribute(neuron_obj,
                               func = spu.set_branch_spines_obj,
                              verbose=verbose)

            
def set_branch_head_neck_shaft_idx(branch_obj,
                                  plot_face_idx=False,
                                    add_no_head_label = True,
                                  verbose = False):
    branch_obj.head_neck_shaft_idx = spu.head_neck_shaft_idx_from_branch(
                    branch_obj,
                    plot_face_idx  = plot_face_idx,
                    add_no_head_label = add_no_head_label,
                    verbose = verbose)
    
def set_neuron_head_neck_shaft_idx(neuron_obj,
                                   add_no_head_label= True,
                         verbose = False):
    set_neuron_spine_attribute(neuron_obj,
                               func = spu.set_branch_head_neck_shaft_idx,
                              verbose=verbose)
    
def set_branch_synapses_head_neck_shaft(branch_obj,
                                        verbose = False
                                       ):
    """
    Purpose: To use the head_neck_shaft_idx
    of the branch objects to give the 
    synapses of a branch the head_neck_shaft label

    Pseudocode: 
    If the branch has any 
    1) Build a KDTree of the branch mesh
    2) find which faces are the closest for the coordinates of all the synapses
    3) Assign the closest face and the head_neck_shaft to the synapse objects
    """
    spines_obj = branch_obj.spines_obj
    if branch_obj.synapses is not None and len(branch_obj.synapses)>0:
        mesh_kd = tu.mesh_kdtree_face(branch_obj.mesh)

        synapse_coords = np.array([s.coordinate for s in branch_obj.synapses]).reshape(-1,3)

        dist,closest_face = mesh_kd.query(synapse_coords)
 
        for j,s in enumerate(branch_obj.synapses):
            s.closest_branch_face_idx = closest_face[j]
            s.head_neck_shaft = branch_obj.head_neck_shaft_idx[s.closest_branch_face_idx]
            
def set_neuron_synapses_head_neck_shaft(neuron_obj,
                                       verbose=False):
    set_neuron_spine_attribute(neuron_obj,
                               func = spu.set_branch_synapses_head_neck_shaft,
                              verbose=verbose)

def add_head_neck_shaft_spine_objs(neuron_obj,
                                   add_synapse_labels=True,
                                   filter_spines_for_size = True,
                                   add_distance_attributes = True,
                                  verbose = False):
    """
    Will do the additionaly processing that
    adds the spine objects to a neuron and then
    creates the head_neck_shaft_idx for the branches 
    
    Application: Can be used later to map synapses
    to the accurate label
    """
    st = time.time()
    
    if filter_spines_for_size:
        
        neuron_obj= spu.filter_spines_by_size(neuron_obj,
                                verbose = verbose)
        if verbose:
            print(f"Total time for fitlering spines by size {time.time() - st}")
            st = time.time()
    
    spu.set_neuron_spines_obj(neuron_obj)
    if verbose:
        print(f"Total time for set_neuron_spines_obj {time.time() - st}")
        st = time.time()
    
    spu.set_neuron_head_neck_shaft_idx(neuron_obj)
    if verbose:
        print(f"Total time for set_neuron_head_neck_shaft_idx {time.time() - st}")
        st = time.time()
    if add_synapse_labels:
        spu.set_neuron_synapses_head_neck_shaft(neuron_obj)
        if verbose:
            print(f"Total time for set_neuron_synapses_head_neck_shaft {time.time() - st}")
            st = time.time()
            
    if add_distance_attributes:
        neuron_obj= spu.calculate_spine_obj_attr_for_neuron(
                neuron_obj,
                verbose = verbose)
        if verbose:
            print(f"Total time for calculate_spine_obj_attr_for_neuron {time.time() - st}")
            st = time.time()
    
    return neuron_obj
            
    
def spines_head_meshes(obj):
    return [k.head_mesh for k in obj.spines_obj]
def spines_neck_meshes(obj):
    return [k.neck_mesh for k in obj.spines_obj]
def spines_no_head_meshes(obj):
    return [k.no_head_mesh for k in obj.spines_obj]

def spines(neuron_obj):
    if type(neuron_obj) != list:
        curr_spines = neuron_obj.spines_obj
    else:
        curr_spines = neuron_obj
    return curr_spines

def spines_head(neuron_obj):
    curr_spines = spu.spines(neuron_obj)
    return [k for k in curr_spines if k.head_exist]
def spines_no_head(neuron_obj):
    curr_spines = spu.spines(neuron_obj)
    return [k for k in curr_spines if not k.head_exist]
def spines_neck(neuron_obj):
    return spu.spines_head(neuron_obj)

def n_spines(neuron_obj):
    return len(spu.spines(neuron_obj))


def plot_spines_head_neck(neuron_obj,
                         head_color = head_color_default,
                         neck_color = neck_color_default,
                          no_head_color = no_head_color_default,
                          bouton_color = bouton_color_default,
                          mesh_alpha = 0.5,
                          verbose=False,
                          show_at_end=True,
                          combine_meshes = True,
                         ):
    
    meshes_colors = []
    
    spine_heads = spu.spines_head_meshes(neuron_obj)
    meshes_colors += [head_color]*len(spine_heads)
    
    spine_necks = spu.spines_neck_meshes(neuron_obj)
    meshes_colors += [neck_color]*len(spine_necks)
    
    spine_no_heads = spu.spines_no_head_meshes(neuron_obj)
    meshes_colors += [no_head_color]*len(spine_no_heads)
    
    boutons = neuron_obj.boutons
    meshes_colors += [bouton_color]*len(boutons)
    
    if verbose:
        print(f"# of spine_heads = {len(spine_heads)}, # of spine_necks= {len(spine_necks)-len(spine_no_heads)}, # of spine_no_heads = {len(spine_no_heads)}")
    
    if combine_meshes:
        meshes_colors = [head_color,neck_color,no_head_color,bouton_color]
        spine_heads = [tu.combine_meshes(spine_heads)]
        spine_necks = [tu.combine_meshes(spine_necks)]
        spine_no_heads = [tu.combine_meshes(spine_no_heads)]
        boutons = [tu.combine_meshes(boutons)]
        meshes = spine_heads+spine_necks+spine_no_heads+boutons
        
    
    
# -------------- End of Spines Obj -------------- #
    
    
connectivity = "edges"

"""   DON'T NEED THIS FUNCTION ANYMORE BECAUSE REPLACED BY TRIMESH_UTILS MESH_SEGMENTATION
def cgal_segmentation(written_file_location,
                      clusters=2,
                      smoothness=0.03,
                      return_sdf=True,
                     print_flag=False,
                     delete_temp_file=True):
    
    if written_file_location[-4:] == ".off":
        cgal_mesh_file = written_file_location[:-4]
    else:
        cgal_mesh_file = written_file_location
    if print_flag:
        print(f"Going to run cgal segmentation with:"
             f"\nFile: {cgal_mesh_file} \nclusters:{clusters} \nsmoothness:{smoothness}")

    csm.cgal_segmentation(cgal_mesh_file,clusters,smoothness)

    #read in the csv file
    cgal_output_file = Path(cgal_mesh_file + "-cgal_" + str(np.round(clusters,2)) + "_" + "{:.2f}".format(smoothness) + ".csv" )
    cgal_output_file_sdf = Path(cgal_mesh_file + "-cgal_" + str(np.round(clusters,2)) + "_" + "{:.2f}".format(smoothness) + "_sdf.csv" )

    cgal_data = np.genfromtxt(str(cgal_output_file.absolute()), delimiter='\n')
    cgal_sdf_data = np.genfromtxt(str(cgal_output_file_sdf.absolute()), delimiter='\n')
    
    if delete_temp_file:
        cgal_output_file.unlink()
        cgal_output_file_sdf.unlink()
        
    
    if return_sdf:
        return cgal_data,cgal_sdf_data
    else:
        return cgal_data"""

    
    
def get_spine_meshes_unfiltered_from_mesh(
    current_mesh,
    segment_name=None,
    clusters=None,
    smoothness=None,
    connectivity_type_for_shaft = None,#"vertices",
    shaft_expansion_method = "path_to_all_shaft_mesh",
    cgal_folder = Path("./cgal_temp"),
    delete_temp_file=True,
    return_sdf=False,
    return_mesh_idx = False,
    print_flag=False,
    shaft_threshold=None,
    ensure_mesh_conn_comp = False,
    plot_segmentation = False,
    plot_shaft = False,
    plot = False,
    ):
    """
    Purpose
    -----------
    First initial spine mesh detection on a branch mesh

    Pseudocode
    ----------------

    Global Parameters to Set
    ----------------

    """
    if connectivity_type_for_shaft is None:
        connectivity_type_for_shaft = connectivity_type_for_shaft_global
    
    if clusters is None:
        clusters = clusters_threshold_global
    
    if smoothness is None:
        smoothness = smoothness_threshold_global
        
    if shaft_threshold is None:
        shaft_threshold = shaft_threshold_global
    
    
    if segment_name is None:
        segment_name = f"{np.random.randint(10,1000)}_{np.random.randint(10,1000)}"
    
    #print(f"segment_name before cgal = {segment_name}")
    (spine_meshes,
     spine_meshes_idx,
     shaft_meshes,
     shaft_meshes_idx,
    cgal_sdf_data) = spine_data_returned= split_mesh_into_spines_shaft(current_mesh,
                               segment_name=segment_name,
                               clusters=clusters,
                              smoothness=smoothness,
                              cgal_folder = cgal_folder,
                              delete_temp_file=delete_temp_file,
                              return_sdf = True,
                              print_flag=print_flag,
                              shaft_threshold = shaft_threshold,
                              plot_segmentation = plot_segmentation,
                            plot_shaft=plot_shaft,
                                                                      )
    if len(spine_meshes) == 0:
        
        if return_sdf and return_mesh_idx:
            return  [],[],[]
        elif return_sdf or return_mesh_idx:
            return [],[]
        else:
            return []
    else:
        spine_mesh_names = [f"s{i}" for i,mesh in enumerate(spine_meshes)]
        shaft_mesh_names = [f"b{i}" for i,mesh in enumerate(shaft_meshes)]

        total_meshes = spine_meshes + shaft_meshes
        total_meshes_idx = spine_meshes_idx + shaft_meshes_idx
        total_names = spine_mesh_names + shaft_mesh_names

        total_edges = []
        for j,(curr_mesh,curr_mesh_idx) in enumerate(zip(total_meshes,total_meshes_idx)):
            touching_meshes = tu.mesh_pieces_connectivity(
                            main_mesh=current_mesh,
                            central_piece=curr_mesh_idx,
                            periphery_pieces=total_meshes_idx,
                            connectivity = connectivity_type_for_shaft)
            try:
                touching_meshes.remove(j)
            except:
                print(f"j = {j}")
                su.compressed_pickle(current_mesh,"current_mesh")
                su.compressed_pickle(curr_mesh_idx,"curr_mesh_idx")
                su.compressed_pickle(total_meshes_idx,"total_meshes_idx")
                su.compressed_pickle(total_names,"total_names")
                raise Exception("didn't do remove")
                
            #construct the edges
            curr_edges = [[total_names[j],total_names[h]] for h in touching_meshes]
            total_edges += curr_edges

        spine_graph = xu.remove_selfloops(nx.from_edgelist(total_edges))
        #nx.draw(spine_graph,with_labels=True)


        """
        How to determine whih parts that are the shaft
        1) start with biggest shaft
        2) Find the shoftest paths to all shaft parts
        3) add all the nodes that aren't already in the shaft category to the shaft category
        """

        #find the biggest shaft
        
        biggest_shaft = f"b{np.argmax([len(k.faces) for k in shaft_meshes])}"
        non_biggest_shaft = [k for k in shaft_mesh_names if k != biggest_shaft]

        final_shaft_mesh_names = shaft_mesh_names.copy()
        if len(non_biggest_shaft) > 0:
            #find all shortest paths from biggest shaft to non_biggest_shaft
            
            #shaft_shortest_paths = [nx.shortest_path(spine_graph,
            #source=biggest_shaft,target=curr_shaft) for curr_shaft in non_biggest_shaft]
            
            shaft_shortest_paths = []
            if shaft_expansion_method == "path_to_largest_shaft_mesh":
                for curr_shaft in non_biggest_shaft:
                    try:
                        c_path = nx.shortest_path(spine_graph,
                                                         source=biggest_shaft,target=curr_shaft) 
                    except:
                        #print(f"Mesh {curr_shaft} Seems to be not connected to mesh")
                        shaft_shortest_paths.append([curr_shaft])
                    else:
                        shaft_shortest_paths.append(c_path)
            elif shaft_expansion_method == "path_to_all_shaft_mesh":
                for shaft_1 in shaft_mesh_names:
                    for shaft_2 in shaft_mesh_names:
                        if shaft_1 != shaft_2:
                            try:
                                c_path = nx.shortest_path(spine_graph,
                                        source=shaft_1,target=shaft_2) 
                            except:
                                #print(f"{shaft_1} and {shaft_2} not connected on the same mesh")
                                shaft_shortest_paths.append([shaft_1,shaft_2])
                            else:
                                shaft_shortest_paths.append(c_path)
            else:
                raise Exception("")
                            
            

            new_shaft_meshes = [int(k[1:]) for k in np.unique(np.concatenate(shaft_shortest_paths)) if "s" in k]
            #print(f"new_shaft_meshes = {new_shaft_meshes}")
            final_shaft_mesh_names += [k for k in np.unique(np.concatenate(shaft_shortest_paths)) if "s" in k]
            final_shaft_meshes = shaft_meshes + [spine_meshes[k] for k in new_shaft_meshes]
            final_shaft_meshes_idx = np.unique(np.concatenate(shaft_meshes_idx + [spine_meshes_idx[k] for k in new_shaft_meshes]))
        else:
            final_shaft_meshes = shaft_meshes
            final_shaft_meshes_idx = np.unique(np.concatenate(shaft_meshes_idx))

        final_shaft_mesh_names = np.unique(final_shaft_mesh_names)

        final_spine_faces_idx = np.delete(np.arange(0,len(current_mesh.faces)), np.array(final_shaft_meshes_idx).astype('int'))

        """
        #Old way of getting all of the spines: by just dividing the mesh using disconnected components
        #after subtracting the shaft mesh

        spine_submesh = current_mesh.submesh([final_spine_faces_idx],append=True)
        spine_submesh_split = spine_submesh.split(only_watertight=False)

        """

        """
        #New way of extracting the spines using graphical methods

        Pseudocode:
        1) remove the shaft meshes from the graph
        2) get the connected components
        3) assemble the connected components total face_idx:
        a. get the sdf values that correspond to those
        b. get the submesh that corresponds to those

        """
        spine_graph.remove_nodes_from(final_shaft_mesh_names)

        spine_submesh_split=[]
        spine_submesh_split_idx = []
        spine_submesh_split_sdf = []
        for sp_list in list(nx.connected_components(spine_graph)):
            curr_spine_face_idx_split = np.concatenate([spine_meshes_idx[int(sp[1:])] for sp in sp_list ])
            spine_submesh_split_sdf.append(cgal_sdf_data[curr_spine_face_idx_split])
            spine_submesh_split_idx.append(curr_spine_face_idx_split)
            spine_submesh_split.append(current_mesh.submesh([curr_spine_face_idx_split],append=True))


        if print_flag or plot:
            print(f"\n\nTotal Number of Spines Found = {len(spine_submesh_split)}")


        #sort the list by size
        spine_length_orders = [len(k.faces) for k in spine_submesh_split]
        greatest_to_least = np.flip(np.argsort(spine_length_orders))
        spines_greatest_to_least =  np.array(spine_submesh_split)[greatest_to_least]
        spines_sdf_greatest_to_least = np.array(spine_submesh_split_sdf)[greatest_to_least]
        spine_submesh_split_idx = np.array(spine_submesh_split_idx)[greatest_to_least]
        
        
        if ensure_mesh_conn_comp:
            spines_greatest_to_least_new = []
            spine_submesh_split_idx_new = []
            sdf_new = []
            for k,k_idx,k_sdf in zip(
                spines_greatest_to_least,
                spine_submesh_split_idx,
                spines_sdf_greatest_to_least):
                sp_mesh,sp_idx = tu.largest_conn_comp(k,return_face_indices=True)
                spines_greatest_to_least_new.append(sp_mesh)
                spine_submesh_split_idx_new.append(k_idx[sp_idx])
                sdf_new.append(k_sdf[sp_idx])
                
            spines_greatest_to_least = np.array(spines_greatest_to_least_new)
            spine_submesh_split_idx = np.array(spine_submesh_split_idx_new)
            spines_sdf_greatest_to_least = np.array(sdf_new)
            
                
            
        if not return_sdf and not return_mesh_idx:
            return spines_greatest_to_least
        return_value = [spines_greatest_to_least]
        if return_sdf:
            return_value.append(spines_sdf_greatest_to_least)
        if return_mesh_idx:
            return_value.append(spine_submesh_split_idx)
            
        return return_value


    
        
        
"""
These filters didn't seem to work very well...

"""
def sdf_median_mean_difference(sdf_values):
    return np.abs(np.median(sdf_values) - np.mean(sdf_values)) 

def apply_sdf_filter(sdf_values,sdf_median_mean_difference_threshold = 0.025,
                    return_not_passed=False):
    pass_filter = []
    not_pass_filter = []
    for j,curr_sdf in enumerate(sdf_values):
        if sdf_median_mean_difference(curr_sdf)< sdf_median_mean_difference_threshold:
            pass_filter.append(j)
        else:
            not_pass_filter.append(j)
    if return_not_passed:
        return not_pass_filter
    else:
        return pass_filter



def filter_spine_meshes(spine_meshes,
                        spine_n_face_threshold=None,
                       spine_sk_length_threshold=None,
                       verbose = False):
    if spine_n_face_threshold is None:
        spine_n_face_threshold = spine_n_face_threshold_global
    if spine_sk_length_threshold is None:
        spine_sk_length_threshold = spine_sk_length_threshold_global
        
    keep_idx = np.arange(len(spine_meshes))
    if spine_n_face_threshold is not None:
        spine_n_faces = np.array([len(k.faces) for k in spine_meshes])
        keep_idx = np.intersect1d(keep_idx,np.where(spine_n_faces >= spine_n_face_threshold)[0])
        if verbose:
            print(f"After face threshold {spine_n_face_threshold}, keep_idx = {keep_idx}")
    if spine_sk_length_threshold is not None:
        spine_lens = np.array([spu.spine_length(k) for k in spine_meshes])
        keep_idx = np.intersect1d(keep_idx,np.where(spine_lens >= spine_sk_length_threshold)[0])
        if verbose:
            print(f"After sk length threshold {spine_sk_length_threshold}, keep_idx = {keep_idx}")
    return [k for i,k in enumerate(spine_meshes) if i in keep_idx]


#------------ 9/23 Addition -------------- #
def filter_out_border_spines(mesh,spine_submeshes,
                            border_percentage_threshold=None,                                                    
                             check_spine_border_perc=None,
                            verbose=False,
                            return_idx = False):
    """
    Purpose: Filter away spines by their percentage overlap with parent border vertices 
    """
    if border_percentage_threshold is None:
        border_percentage_threshold = border_percentage_threshold_global
        
    if check_spine_border_perc is None:
        check_spine_border_perc = check_spine_border_perc_global
    return tu.filter_away_border_touching_submeshes_by_group(mesh,spine_submeshes,
                                                             border_percentage_threshold=border_percentage_threshold,
                                                             inverse_border_percentage_threshold=check_spine_border_perc,
                                                             verbose = verbose,
                                                             return_meshes = not return_idx,
                                                            )

def filter_out_soma_touching_spines(
    spine_submeshes,
    soma_vertices=None,
    soma_kdtree=None,
    verbose=False,
    return_idx = False,
    ):
    """
    Purpose: To filter the spines that are touching the somae
    Because those are generally false positives picked up 
    by cgal segmentation
    
    Pseudocode
    1) Create a KDTree from the soma vertices
    2) For each spine:
    a) Do a query against the KDTree with vertices
    b) If any of the vertices have - distance then nullify


    """
    if soma_kdtree is None and not soma_vertices is None:
        soma_kdtree = KDTree(soma_vertices)
    if soma_kdtree is None and soma_verties is None:
        raise Exception("Neither a soma kdtree or soma vertices were given")

    if verbose:
        print(f"Number of spines before soma border filtering = {len(spine_submeshes)}")
    final_spines = []
    final_spines_idx = []
    for j,sp_mesh in enumerate(spine_submeshes):
        sp_dist,sp_closest = soma_kdtree.query(sp_mesh.vertices)
        n_match_vertices = np.sum(sp_dist==0)
        
        if n_match_vertices == 0:
            final_spines.append(sp_mesh)
            final_spines_idx.append(j)
        else:
            if verbose:
                print(f"Spine {j} was removed because had {n_match_vertices} border vertices")
    if verbose:
        print(f"Number of spines before soma border filtering = {len(final_spines)}")
    
    if return_idx:
        return final_spines_idx
    else:
        return final_spines


def spine_head_neck(
    mesh,
    cluster_options = (2,3,4),
    smoothness = None,#0.15,
    plot_segmentation = False,
    head_ray_trace_min = None,#240,
    head_face_min = None,#10,
    default_head_face_idx = np.array([]).astype("int"),
    default_head_sdf = -1,
    
    #for filtering away head meshes that have the following coordinate
    stop_segmentation_after_first_success = False,
    no_head_coordinates = None,
    only_allow_one_connected_component_neck = None,
    
    plot_head_neck = False,
    return_meshes = False,
    return_sdf = True,
    return_width=True,
    verbose = False,
    ):
    """
    Purpose
    -----------
    To determine the head and neck face indices clustering of a mesh representing a spine


    Pseudocode
    ----------------
    for clusters [2,3]:
    1) Run the segmentation algorithm (using cluster and smoothness thresholds)
    2) Filter meshes for all those with a value above the face count above threshold and above the ray trace percentage width value
    3) Store the meshes not face as neck and store sdf value as weighted average
    4a) If none then continue
    4b) If at least one, concatenate the faces of all of the spine heads into one array (and do weighted average of the sdf)
    5) Break if found a head
    6) Optionally plot the spine neck and head

    Global Parameters to Set
    ----------------
    head_smoothness:
        the cgal segmentation smoothness parameter for clustering a spine mesh into a head and neck

    head_ray_trace_min: float
        minimum width approximation (units) as a percentile of the ray trace values for a submesh to be in consideration as a spine head submesh 

    head_face_min: int
        minimum number of faces for a submesh to be in consideration as a spine head submesh 

    only_allow_one_connected_component_neck: bool
        whether to allow for the neck submesh to be multiple connected components (aka disconnected) or not



    Can optionally return:
    1) Meshes instead of face idx
    
    Ex: 
    curr_idx = 38
    sp_mesh = curr_branch.spines[curr_idx]
    spu.spine_head_neck(sp_mesh,
                        cluster_options=(2,3,4),
                        smoothness=0.15,
                        verbose = True,
                        plot_segmentation=True,
                       plot_head_neck=True)

    """
    
    if head_ray_trace_min is None:
        head_ray_trace_min = head_ray_trace_min_global
        
    if smoothness is None:
        smoothness = head_smoothness_global
        
    if head_face_min is None:
        head_face_min = head_face_min_global
        
    if only_allow_one_connected_component_neck is None:
        only_allow_one_connected_component_neck = only_allow_one_connected_component_neck_global
    

    neck_face_idx = None
    head_face_idx = default_head_face_idx

    neck_sdf = None
    head_sdf = default_head_sdf
    head_width = default_head_sdf

    winning_n_heads = 0
    neck_face_idx_win = neck_sdf_win = neck_width_win = None
    head_face_idx_win = np.array([])
    head_sdf_win = default_head_sdf
    head_width_win = default_head_sdf
    for c in cluster_options:
        if verbose:
            print(f"Using clusters {c}")

        meshes,sdfs,mesh_idx = tu.mesh_segmentation(mesh,
                                                    clusters = c,
                                                    smoothness = smoothness,
                                           return_meshes = True,
                                           return_sdf=True,
                                            plot_segmentation=plot_segmentation,
                                          return_ordered_by_size=False,
                                          return_mesh_idx = True)
        ray_trace_perc = np.array([tu.mesh_size(k,"ray_trace_percentile") for k in meshes])
        mesh_sizes = np.array([len(k.faces) for k in meshes])

        if verbose:
            print(f"sdfs={sdfs}, ray_trace_perc = {ray_trace_perc}, mesh_sizes = {mesh_sizes}")
            print(f"Thresholds: head_ray_trace_min = {head_ray_trace_min}, head_face_min = {head_face_min}")

        head_obj_idx = np.where((ray_trace_perc > head_ray_trace_min) & (mesh_sizes > head_face_min))[0]
        
        if verbose:
            print(f"head_obj_idx = {head_obj_idx}")
        
        """
        Purpose:
        --------
        Want to delete any meshes that have the border coordinates
        """
        
        if no_head_coordinates is not None:
            head_obj_idx_final = []
            for hm in head_obj_idx:
                curr_mesh = meshes[hm]
                closest_dist = tu.closest_mesh_distance_to_coordinates_fast(
                    curr_mesh,
                    coordinates =no_head_coordinates,
                    attribute = "vertices",
                    stop_after_0_dist = True,
                )
                
                if closest_dist > 0:
                    head_obj_idx_final.append(hm)
                    
            head_obj_idx = head_obj_idx_final
            if verbose:
                print(f"After filtering for no_head_coordinates: head_obj_idx = {head_obj_idx}")
        
        neck_obj_idx = np.delete(np.arange(len(meshes)),np.array(head_obj_idx).astype('int'))

        """
        Purpose:
        --------
        Ensure there is a neck submesh and optionally only one neck submesh (if more than one then negate the head segmentation)
        """
        if len(neck_obj_idx) == 0:
            neck_obj_idx = np.arange(len(meshes))
            head_obj_idx = np.array([]).astype('int')
        elif only_allow_one_connected_component_neck:
            n_neck_components = len(tu.connected_components_from_face_idx(
                mesh,face_idx = np.concatenate(mesh_idx[neck_obj_idx]),return_meshes = False))
            if verbose:
                print(f"n_neck_components = {n_neck_components}")
            if  n_neck_components> 1:
                if verbose:
                    print(f"More than one neck connected component so not valid")
                neck_obj_idx = np.arange(len(meshes))
                head_obj_idx = np.array([]).astype('int')
        else:
            pass

        if verbose:
            print(f"head_obj_idx = {head_obj_idx}")
            print(f"neck_obj_idx= {neck_obj_idx}")

        """
        Purpose:
        --------
        Calculate the weighted average head and neck sdf and width values
        """
        # save off the neck face idx
        neck_face_idx = np.concatenate(mesh_idx[neck_obj_idx])
        neck_sdf = nu.weighted_average(sdfs[neck_obj_idx],mesh_sizes[neck_obj_idx])
        neck_width = nu.weighted_average(ray_trace_perc[neck_obj_idx],mesh_sizes[neck_obj_idx])

        if verbose:
            print(f"neck_face_idx= {neck_face_idx}")
            print(f"neck_sdf = {neck_sdf}")
            print(f"neck_width = {neck_width}")

        if len(head_obj_idx) == 0:
            if neck_face_idx_win is None:
                neck_face_idx_win = neck_face_idx
                neck_sdf_win = neck_sdf
                neck_width_win = neck_width
            continue

        head_face_idx = np.concatenate(mesh_idx[head_obj_idx])
        head_sdf = nu.weighted_average(sdfs[head_obj_idx],mesh_sizes[head_obj_idx])
        head_width = nu.weighted_average(ray_trace_perc[head_obj_idx],mesh_sizes[head_obj_idx])

        if verbose:
            print(f"head_face_idx= {head_face_idx}")
            print(f"head_sdf = {head_sdf}")
            print(f"head_width = {head_width}")

        n_heads = len(tu.connected_components_from_face_idx(
            mesh,face_idx = head_face_idx,return_meshes = False)
        )
        if n_heads > winning_n_heads:
            winning_n_heads = n_heads
            
            neck_face_idx_win = neck_face_idx
            neck_sdf_win = neck_sdf
            neck_width_win = neck_width
            
            head_face_idx_win = head_face_idx
            head_sdf_win = head_sdf
            head_width_win = head_width
            
            if verbose:
                print(f"New winning number of heads = {winning_n_heads}")
            
            if stop_segmentation_after_first_success:
                break

    # Filtering away the heads that are 
#     if no_head_coordinates is not None:
#         if len(head_face_idx) > 0:
#             if verbose:
#                 print(f"Attempting to filter head synapses close to coordinates")
#             head_face_idx = tu.filter_away_connected_comp_in_face_idx_with_minimum_vertex_distance_to_coordinates(
#                 mesh = mesh,
#                 face_idx = head_face_idx,
#                 coordinates = no_head_coordinates,
#                 verbose = verbose,
#                 plot = False,
#             )
            
#             if len(head_face_idx) > 0:
#                 head_face_idx = np.hstack(head_face_idx)
                
#             neck_face_idx = np.delete(np.arange(len(mesh.faces)),head_face_idx)
            
    neck_face_idx = neck_face_idx_win
    neck_sdf = neck_sdf_win
    neck_width = neck_width_win

    head_face_idx = head_face_idx_win
    head_sdf = head_sdf_win
    head_width = head_width_win       
            
    neck_mesh = mesh.submesh([neck_face_idx],append=True)
    if len(head_face_idx) > 0:
        head_mesh = mesh.submesh([head_face_idx],append=True)
    else:
        head_mesh = tu.empty_mesh()
        
    
    
        
    if return_meshes:
        return_value= [head_mesh,neck_mesh]
    else:
        return_value = [head_face_idx.astype('int'),neck_face_idx.astype('int')]
        
    if return_sdf:
        return_value += [head_sdf,neck_sdf]
        
    if return_width:
        return_value += [head_width,neck_width]
        
    return return_value



def bouton_non_bouton_idx_from_branch(branch_obj,
                                     plot_branch_boutons=False,
                                     plot_face_idx=False,
                                     verbose = False):
    """
    Purpose: To add axon labels to the branch

    Ex: 
    return_face_idx = spu.bouton_non_bouton_idx_from_branch(branch_obj = neuron_obj[0][0],
    plot_branch_boutons = False,
    verbose = True,
    plot_face_idx = True,
    )

    """

    head_neck_shaft_idx = np.ones(branch_obj.mesh_face_idx.shape)*spu.head_neck_shaft_dict["non_bouton"]

    boutons = branch_obj.boutons
    if boutons is not None and len(boutons) > 0:
        boutons_idx = np.concatenate(tu.convert_meshes_to_face_idxes(boutons,branch_obj.mesh))

        if verbose:
            print(f"{len(boutons_idx)} bouton faces")

        head_neck_shaft_idx[boutons_idx] = spu.head_neck_shaft_dict["bouton"]


    return head_neck_shaft_idx

def head_neck_shaft_idx_from_branch(branch_obj,
    plot_face_idx  = False,
    add_no_head_label = True,
    verbose = False,
    process_axon_branches=True):
    """
    Purpose: To create an array
    mapping the mesh face idx of the branch to a
    label of head/neck/shaft

    Pseudocode: 
    1) Create an array the size of branch mesh 
    initialized to shaft
    2) iterate through all of the spine objects of the branch
        a) set all head index to head
        b) set all neck index to neck
        
    Ex: 
    spu.head_neck_shaft_idx_from_branch(branch_obj = neuron_obj_exc_syn[0][6],
    plot_face_idx  = True,
    verbose = True,)
    """

    if process_axon_branches and "axon" in branch_obj.labels:
        """
        Purpose: To add axon labels to the branch
        
        Psuedocode: 
        1) 
        """
        head_neck_shaft_idx = spu.bouton_non_bouton_idx_from_branch(branch_obj)
        if verbose:
            print(f'# of bouton faces = {np.sum(head_neck_shaft_idx == spu.head_neck_shaft_dict["bouton"])}')
            print(f'# of non_bouton faces = {np.sum(head_neck_shaft_idx == spu.head_neck_shaft_dict["non_bouton"])}')
    else:
        head_neck_shaft_idx = np.ones(branch_obj.mesh_face_idx.shape)*spu.head_neck_shaft_dict["shaft"]

        if branch_obj.spines_obj is not None:
            for sp_obj in branch_obj.spines_obj:
                if add_no_head_label and (not sp_obj.head_exist):
                    head_neck_shaft_idx[sp_obj.mesh_face_idx[sp_obj.neck_face_idx]] = spu.head_neck_shaft_dict["no_head"]
                    head_neck_shaft_idx[sp_obj.mesh_face_idx[sp_obj.head_face_idx]] = spu.head_neck_shaft_dict["head"]
                else:
                    head_neck_shaft_idx[sp_obj.mesh_face_idx[sp_obj.head_face_idx]] = spu.head_neck_shaft_dict["head"]
                    head_neck_shaft_idx[sp_obj.mesh_face_idx[sp_obj.neck_face_idx]] = spu.head_neck_shaft_dict["neck"]

        if verbose:
            print(f'# of shaft faces = {np.sum(head_neck_shaft_idx == spu.head_neck_shaft_dict["shaft"])}')
            print(f'# of head faces = {np.sum(head_neck_shaft_idx == spu.head_neck_shaft_dict["head"])}')
            print(f'# of neck faces = {np.sum(head_neck_shaft_idx == spu.head_neck_shaft_dict["neck"])}')
            print(f'# of no_head faces = {np.sum(head_neck_shaft_idx == spu.head_neck_shaft_dict["no_head"])}')

    return head_neck_shaft_idx

def spine_density(obj,um = True):
    """
    n_spine / skeletal length (um)
    """
    skeletal_length = obj.skeletal_length
    if um:
        skeletal_length = skeletal_length/1000
        
    if skeletal_length == 0:
        return 0
    return len(obj.spines_obj)/skeletal_length
    
def spine_volume_density(obj,um = True):
    """
    sum spine volume (um**3) / skeletal length (um)
    """
    skeletal_length = obj.skeletal_length
    if um:
        skeletal_length = skeletal_length/1000
    
    if skeletal_length == 0:
        return 0
    return np.sum([k.volume for k in obj.spines_obj])/skeletal_length


    
                
def spine_str_label(spine_label):
    """
    spu.spine_str_label(-2)
    """
    if spine_label is None:
        return "no_label"
    if type(spine_label) != str:
        spine_label =  head_neck_shaft_dict_inverted[spine_label]
    return spine_label

def spine_int_label(spine_label):
    if type(spine_label) == str:
        spine_label =  head_neck_shaft_dict[spine_label]
    return spine_label



# -------------- 12/6: Doing the spine calculation ----------------
def calculate_spines_on_branch(
    branch,
    
    # arguments for the intial segmentation
    clusters_threshold=None,#2,
    smoothness_threshold=None,
    shaft_threshold = None,
    plot_spines_before_filter = False,
    
    spine_n_face_threshold=None,
    spine_sk_length_threshold = None,#1350,
    plot_spines_after_face_threshold = False,
    
    
    filter_by_bounding_box_longest_side_length=None,
    side_length_threshold = None,
    plot_spines_after_bbox_threshold = False,
    
    
    filter_out_border_spines=None, #this seemed to cause a lot of misses
    border_percentage_threshold=None,
    check_spine_border_perc=None,
    plot_spines_after_border_filter = False,
    
    
    skeleton_endpoint_nullification=None,
    skeleton_endpoint_nullification_distance = None,
    plot_spines_after_skeleton_endpt_nullification = False,
    
    
    soma_vertex_nullification = None,
    soma_verts = None,
    soma_kdtree = None,
    plot_spines_after_soma_nullification = False,
    

    #-------1/20 Addition --------
    filter_by_volume = None,
    calculate_spine_volume=None,
    filter_by_volume_threshold = None, #calculated from experiments
    plot_spines_after_volume_filter = False,
    
    # ----- 4/14/25 Additions ----
    filter_by_face_area_mean = None,
    filter_by_face_area_mean_min = None,
    plot_spines_after_filter_by_face_area_mean = False,

    filter_by_boundary_to_area_ratio = None,
    filter_by_boundary_to_area_ratio_min = None,
    plot_spines_after_filter_by_boundary_to_area_ratio_min = False,
    
    
    print_flag = False,
    plot_segmentation = False,
    **kwargs,
    ):
    """
    Purpose
    -------
    Will calculate the spines on a branch object
    
    Pseudocode
    ----------------
    1) spu.get_spine_meshes_unfiltered_from_mesh
    2) filters spine meshes by minimum number of faces (spine_n_face_threshold) and minimum skeletal length (spine_sk_length_threshold)
    3) if requested (filter_by_bounding_box_longest_side_length), filters the meshes to have less than a certain length (side_length_threshold) for the longest side of their oriented bounding box. 
    To prevent false positive spines from long axon fragment merges
    4) if requested (filter_out_border_spines), filter out meshes that have:
        a) higher than a certain percentage (border_percentage_threshold) of the submesh vertices overlapping with border vertices (certices adjacent to open spaces in the mesh) on the parent mesh  
        b) higher than certain percentage (check_spine_border_perc_global) of the parent mesh’s border vertices overlapping with the submesh vertices

    5) if requested (skeleton_endpoint_nullification), filter away spines that are within a certain distance (skeleton_endpoint_nullification_distance) from the branch skeleton endpoints in order to avoid a high false positive class.
    6) if requested (soma_vertex_nullification), filter out spines that have vertices overlapping with vertices of the soma
    7) Creates a spine object for each of the spine meshes remaining after filtering:
        a) spu.calculate_spine_attributes
        
    Global Parameters to Set
    ----------------
    spine_n_face_threshold: int
        minimum number of mesh faces for a submesh to be in consideration for the spine classification

    # -- size filtering
    spine_sk_length_threshold: int
        minimum length (unit) of the surface skeleton of the submesh for it to be in consideration for the spine classification

    # -- bounding box filtering
    filter_by_bounding_box_longest_side_length: 

    side_length_threshold: 

    # -- border filtering
    filter_out_border_spines: bool
        whether to be perform spine filtering by considering how much spine submesh vertices overlap with border vertices of barents

    border_percentage_threshold: float
        maximum percentage a submesh vertices can overlap with border vertices (certices adjacent to open spaces in the mesh) on the parent mesh and still be in consideration for spine label  

    check_spine_border_perc_global: float
        maximum percentage that a the parent mesh’s border vertices can overlapping with the submesh vertices and that submesh still be in consideration for spine label  
    # -- skeleton filtering
    skeleton_endpoint_nullification: bool

    skeleton_endpoint_nullification_distance_global: float
        minimum distance

    # -- soma filtering
    soma_vertex_nullification: bool


    
    Ex: 
    curr_limb = neuron_obj[2]
    soma_verts = np.concatenate([neuron_obj[f"S{k}"].mesh.vertices for k in curr_limb.touching_somas()])

    branch = neuron_obj[2][7]
    sp_filt,sp_vol, spine_submesh_split_filtered_not = calculate_spines_on_branch(
        branch,

        shaft_threshold = 500,
        smoothness_threshold = 0.08,


        plot_spines_before_filter = False,
        plot_spines_after_face_threshold=False,

        plot_spines_after_bbox_threshold = True,
        plot_spines_after_border_filter = True,

        soma_verts = soma_verts,
        plot_spines_after_skeleton_endpt_nullification = True,
        plot_spines_after_soma_nullification = True,
        plot_spines_after_volume_filter = True,

        print_flag=True,

    )
    """
    
    if clusters_threshold is None:
        clusters_threshold = clusters_threshold_global
        
    if smoothness_threshold is None:
        smoothness_threshold = smoothness_threshold_global
        
    if shaft_threshold is None:
        shaft_threshold = shaft_threshold_global
        
    if calculate_spine_volume is None:
        calculate_spine_volume = calculate_spine_volume_global
    
    if spine_n_face_threshold is None:
        spine_n_face_threshold = spine_n_face_threshold_global
        
    if spine_sk_length_threshold is None:
        spine_sk_length_threshold = spine_sk_length_threshold_global
        
    if filter_by_bounding_box_longest_side_length is None:
        filter_by_bounding_box_longest_side_length = filter_by_bounding_box_longest_side_length_global
        
    if side_length_threshold is None:
        side_length_threshold = side_length_threshold_global
        
    if filter_out_border_spines is None:
        filter_out_border_spines = filter_out_border_spines_global
        
    if skeleton_endpoint_nullification is None:
        skeleton_endpoint_nullification = skeleton_endpoint_nullification_global
        
    if skeleton_endpoint_nullification_distance is None:
        skeleton_endpoint_nullification_distance = skeleton_endpoint_nullification_distance_global
        
    if soma_vertex_nullification is None:
        soma_vertex_nullification = soma_vertex_nullification_global
        
    if border_percentage_threshold is None:
        border_percentage_threshold = border_percentage_threshold_global
        
    if check_spine_border_perc is None:
        check_spine_border_perc = check_spine_border_perc_global
        
    if filter_by_volume is None:
        filter_by_volume = filter_by_volume_global
        
    if filter_by_volume_threshold is None:
        filter_by_volume_threshold = filter_by_volume_threshold_global
        
    if filter_by_face_area_mean is None:
        filter_by_face_area_mean = filter_by_face_area_mean_global
    if filter_by_face_area_mean_min is None:
        filter_by_face_area_mean_min = filter_by_face_area_mean_min_global
        
    if filter_by_boundary_to_area_ratio is None:
        filter_by_boundary_to_area_ratio = filter_by_boundary_to_area_ratio_global
    
    if filter_by_boundary_to_area_ratio_min is None:
        filter_by_boundary_to_area_ratio_min = filter_by_boundary_to_area_ratio_min_global
    
    cgal_path=Path("./cgal_temp")
    
    # Step 1: Initial segmentation to get unfiltered spines
    spine_submesh_split= spu.get_spine_meshes_unfiltered_from_mesh(
        branch.mesh,
        segment_name="no_name",
        clusters=clusters_threshold,
        smoothness=smoothness_threshold,
        cgal_folder = cgal_path,
        delete_temp_file=True,
        return_sdf=False,
        print_flag=False,
        shaft_threshold=shaft_threshold,
        plot_segmentation=plot_segmentation
    )
    
    
    if print_flag:
        print(f"--> n_spines found before filtering = {len(spine_submesh_split)}")
        
        
        
    # Step 2: Filter Spines by Face length
    if spine_n_face_threshold > 0 or spine_sk_length_threshold > 0:
        print(f"Filtering away by face and skeletal length")
        spine_submesh_split_filtered = spu.filter_spine_meshes(spine_submesh_split,
                                                            spine_n_face_threshold=spine_n_face_threshold,
                                                          spine_sk_length_threshold=spine_sk_length_threshold)
    else:
        spine_submesh_split_filtered = spine_submesh_split
    
    
        
        
    # Step 3: Filter Spines by Bounding Box
    if filter_by_bounding_box_longest_side_length:
        old_length = len(spine_submesh_split_filtered)
        spine_submesh_split_filtered = tu.filter_meshes_by_bounding_box_longest_side(spine_submesh_split_filtered,
                                                                                 side_length_threshold=side_length_threshold)
    
        
        
    if filter_out_border_spines:
        if print_flag:
            print("Using the filter_out_border_spines option")
        spine_submesh_split_filtered = spu.filter_out_border_spines(branch.mesh,
                                                                    spine_submesh_split_filtered,
                                                                    border_percentage_threshold=border_percentage_threshold,
                                                                    check_spine_border_perc=check_spine_border_perc,
                                                                    verbose=print_flag
                                                                   )
        
            
    if skeleton_endpoint_nullification:
        if print_flag:
            print("Using the skeleton_endpoint_nullification option")


        curr_branch_end_coords = sk.find_skeleton_endpoint_coordinates(branch.skeleton)
        spine_submesh_split_filtered = tu.filter_meshes_by_containing_coordinates(spine_submesh_split_filtered,
                                                    curr_branch_end_coords,
                                                    distance_threshold=skeleton_endpoint_nullification_distance)
        
            
    
    
    if soma_vertex_nullification:
        if soma_verts is not None:
            soma_kdtree = KDTree(soma_verts)
        if soma_kdtree is None:
            raise Exception("Requested endpoint nullification but no soma information given")
            
        if print_flag:
            print("Using the soma_vertex_nullification option")

        spine_submesh_split_filtered = spu.filter_out_soma_touching_spines(spine_submesh_split_filtered,
                                                    soma_kdtree=soma_kdtree)
        
            
            
    if calculate_spine_volume or filter_by_volume: 
        spine_volumes = np.array([tu.mesh_volume(k) for k in spine_submesh_split_filtered])
        
    if filter_by_volume:
        """
        Pseudocode: 
        1) Calculate the volumes of all the spines
        2) Filter those spines for only those above the volume

        """
        if len(spine_submesh_split_filtered) > 0:
            volume_kept_idx = np.where(spine_volumes > filter_by_volume_threshold)[0]
            if print_flag:
                print(f"Number of spines filtered away by volume = {len(spine_volumes) - len(volume_kept_idx)}")
            #spine_submesh_split_filtered_not = [spine_submesh_split_filtered[k] for k in range(len(spine_volumes)) if k not in volume_kept_idx]
            spine_submesh_split_filtered = [spine_submesh_split_filtered[k] for k in volume_kept_idx]
            
            spine_volumes = spine_volumes[volume_kept_idx]
            
        
    # --- 4/14/25 addition --
    if filter_by_face_area_mean:
        kept_idx = [i for i,k in enumerate(spine_submesh_split_filtered) if tu.face_area_mean(k)>=filter_by_face_area_mean_min]
        
        if print_flag:
            print(f"Number of spines filtered away by face_area_mean = {len(spine_submesh_split_filtered) - len(kept_idx)}")
        
        
        spine_submesh_split_filtered =[spine_submesh_split_filtered[k] for k in kept_idx]
        spine_volumes = [spine_volumes[k] for k in kept_idx]
        
            
    # --- 4/14/25 addition --
    if filter_by_boundary_to_area_ratio:
        kept_idx = [i for i,k in enumerate(spine_submesh_split_filtered) if tu.area_to_boundary_length_ratio(k)>=filter_by_boundary_to_area_ratio_min]
        
        if print_flag:
            print(f"Number of spines filtered away by boundary_to_area_ratio = {len(spine_submesh_split_filtered) - len(kept_idx)}")
        
        
        spine_submesh_split_filtered =[spine_submesh_split_filtered[k] for k in kept_idx]
        spine_volumes = [spine_volumes[k] for k in kept_idx]
        
                         
    
        
    if calculate_spine_volume:
        return spine_submesh_split_filtered,spine_volumes
    else:
        return spine_submesh_split_filtered
    
def calculate_spines_on_neuron(
    neuron_obj,
    limb_branch_dict=None,
    #---arguments for the query restriction -----
    #query="width > 400 and n_faces_branch>100",
    #query="median_mesh_center > 140 and n_faces_branch>100",#previous used median_mesh_center > 140
    query=None,#previous used median_mesh_center > 140
    plot_query = False,
    
    # limb specific arguments for spine calculation:
    soma_vertex_nullification = None,
    
    #---- arguments for the spine calculation on a branch ----
    
    #-- arguments for volume -- 
    calculate_spine_volume = None,
    
    print_flag=False,
    limb_branch_dict_exclude = None,
    **kwargs):
    """
    Purpose: Will calculate spines over a neuron object
    
    Pseudocode
    ----------------
    1) Calculates a limb_branch_dict over which to perform spine detection if not already given. Which branches are included are determined by:
        a) Generates width calculations if not already performed
        b) performs the search with the query to get the limb branch dict
    2) Iterates over all limbs in limb branch:
        a) Generates the soma touching vertices and creates a kdtree for them
        b) Iterates over all branches:
            i) spu.calculate_spines_on_branch → returns spines and spine volumes
            
    Global Parameters to Set
    ----------------
    query: str
        the query to restrict branches searched


    
    Ex: 
    spu.calculate_spines_on_neuron(
        recovered_neuron,
        plot_query = False,
        print_flag = True,
    )
    
    
    """
    
    if query is None:
        query = query_global
    
    if calculate_spine_volume is None:
        calculate_spine_volume = calculate_spine_volume_global
    
    if soma_vertex_nullification is None:
        soma_vertex_nullification = soma_vertex_nullification_global
    
    
    if query is None:
        query = query_global
    
    # --- Step 1: Applies query to see which branches to calculate spines over ---
    if limb_branch_dict is None:
        if type(query) == dict:
            functions_list = query["functions_list"]
            current_query = query["query"]
        else:
            functions_list = ["median_mesh_center","n_faces_branch"]
            current_query = query


        #check that have calculated the median mesh center if required
        if "median_mesh_center" in functions_list:
            if len(neuron_obj.get_limb_node_names())>0 and "median_mesh_center" not in neuron_obj[0][0].width_new.keys():
                print("The median_mesh_center was requested but has not already been calculated so calculating now.... ")

                wu.calculate_new_width_for_neuron_obj(neuron_obj,
                                                      no_spines=False,
                                       distance_by_mesh_center=True,
                                       summary_measure="median")
            else:
                print("The median_mesh_center was requested and HAS already been calculated")

        limb_branch_dict = ns.query_neuron(neuron_obj,
                           functions_list=functions_list,
                           query=current_query,
                                         plot_limb_branch_dict=plot_query)

    if print_flag:
        print(f"limb_branch_dict = {limb_branch_dict}")
        
        
    # --- Step 2: Calculating the Spines over ---
        
    for limb_idx in limb_branch_dict.keys():
        curr_limb = neuron_obj[limb_idx]
        
        if soma_vertex_nullification:
            soma_verts = np.concatenate([neuron_obj[f"S{k}"].mesh.vertices for k in curr_limb.touching_somas()])
            soma_kdtree = KDTree(soma_verts)
        else:
            soma_kdtree = None
            
        for branch_idx in limb_branch_dict[limb_idx]:
            if limb_branch_dict_exclude is not None:
                if limb_idx in limb_branch_dict_exclude:
                    if branch_idx in limb_branch_dict_exclude[limb_idx]:
                        if print_flag:
                            print(f"Skipping because in limb_branch exclude")
                        continue
            
            curr_branch = curr_limb[branch_idx]
            
            if print_flag:
                print(f"Working on limb {limb_idx} branch {branch_idx}")
                
                
            if calculate_spine_volume:
                branch_spines,branch_spines_vol  = calculate_spines_on_branch(
                    curr_branch,
                    soma_kdtree = soma_kdtree,
                    print_flag=print_flag,
                    calculate_spine_volume =calculate_spine_volume,
                    **kwargs
                )
                curr_branch.spines_volume = list(branch_spines_vol)
                already_calculated_volumes = True
            else:
                branch_spines  = calculate_spines_on_branch(
                    curr_branch,
                    soma_kdtree = soma_kdtree,
                    print_flag=print_flag,
                    calculate_spine_volume =calculate_spine_volume,
                    **kwargs
                )
                already_calculated_volumes = False
            
            curr_branch.spines = list(branch_spines)
            
            
            if calculate_spine_volume and not already_calculated_volumes:
                curr_branch.compute_spines_volume()
            
            # reset the spines obj so old spines won't override
            curr_branch.spines_obj = None
                
# --------------- for filtering spines: 1/25 ------------
    
def filter_spines_by_size_branch(
    branch_obj,
    spine_n_face_threshold=None,
    filter_by_volume_threshold = None,
    spine_sk_length_threshold= None,
    verbose = False,
    assign_back_to_obj = True,
    calculate_spines_length_on_whole_neuron = True,
    ):
    """
    Purpose: To filter away any of the 
    spines according to the size thresholds

    """
    if spine_n_face_threshold is None:
        spine_n_face_threshold= spine_n_face_threshold_global
    if filter_by_volume_threshold is None:
        filter_by_volume_threshold = filter_by_volume_threshold_global
    if spine_sk_length_threshold is None:
        spine_sk_length_threshold = spine_sk_length_threshold_global
        
    if verbose:
        print(f"Spines before filtering away: {len(branch_obj.spines_obj)}")
        
    if branch_obj.spines_obj is not None and len(branch_obj.spines_obj) > 0:
        s_objs_keep = [s_obj for s_obj in branch_obj.spines_obj
                              if ((len(s_obj.mesh.faces) > spine_n_face_threshold)
                               and (s_obj.volume > filter_by_volume_threshold)
                                 and (s_obj.skeletal_length > spine_sk_length_threshold))
                              ]
        if verbose:
            print(f"Spines before filtering away: {len(s_objs_keep)}")

        if assign_back_to_obj:
            branch_obj.spines_obj = s_objs_keep
            branch_obj.spines = [s.mesh for s in s_objs_keep]
            branch_obj.spines_volume = [s.volume for s in s_objs_keep]

        return_value = s_objs_keep
    elif branch_obj.spines is not None:
        new_spines = []
        new_spines_volume = []
        for spine_mesh,spine_vol in zip(branch_obj.spines,branch_obj.spines_volume):
            if ((len(spine_mesh.faces) >= spine_n_face_threshold) 
                and (spine_vol >= filter_by_volume_threshold)
               and (spine_length(spine_mesh) > spine_sk_length_threshold)):
                new_spines.append(spine_mesh)
                new_spines_volume.append(spine_vol)
                
        if assign_back_to_obj:
            branch_obj.spines = new_spines
            branch_obj.spines_volume = new_spines_volume
        
        return_value = new_spines
    else:
        return_value = None
        
    if calculate_spines_length_on_whole_neuron:
        nru.calculate_spines_skeletal_length(branch_obj)
        
    return return_value
    
def filter_spines_by_size(
    neuron_obj,
    spine_n_face_threshold=None,
    filter_by_volume_threshold = None,
    verbose = False,
    **kwargs
    ):
    st = time.time()
    for limb_obj in neuron_obj:
        for branch_obj in limb_obj:
            spu.filter_spines_by_size_branch(branch_obj,
                                             **kwargs
                                )
    if verbose:
        print(f"Total time for spine filtering: {time.time() - st}")
        
    return neuron_obj


# ------------- 2/7 adjustments -----------------
def adjust_obj_with_face_offset(
    spine_obj,
    face_offset,
    verbose = False,
    ):
    """
    Purpose: To adjust the spine properties that
    would be affected by a different face idx
    
    Ex: 
    b_test = neuron_obj[0][18]
    sp_obj = b_test.spines_obj[0]
    sp_obj.export()

    spu.adjust_spine_obj_with_face_offset(
        sp_obj,
        face_offset = face_offset,
        verbose = True
    ).export()
    """
    new_obj = copy.deepcopy(spine_obj)
    for k,v in spine_obj.export().items():
        if "face_idx" not in k:
            continue
        
        if v is None:
            continue
            
        if k in ["head_face_idx","neck_face_idx"]:
            continue
            
        if verbose:
            print(f"Adjusting {k} because face_idx and not None")
            
        try:
            setattr(new_obj,k,v + face_offset)
        except:
            setattr(new_obj,f"_{k}",v + face_offset)
        
    return new_obj

def spine_length(
    spine_mesh,
    verbose = False,
    surface_skeleton_method = "slower",
    plot = False):
    if is_spine_obj(spine_mesh):
        spine_mesh = spine_mesh.mesh
        
    #spine_mesh = tu.largest_conn_comp(spine_mesh)
    
    if surface_skeleton_method == "meshparty":
        curr_sk = sk.surface_skeleton(spine_mesh)
    else:
        curr_sk = sk.generate_surface_skeleton_slower(spine_mesh)
        
    curr_sk_length = sk.calculate_skeleton_distance(curr_sk)
    if verbose:
        print(f"skeletal length = {curr_sk_length}")
    return curr_sk_length


# -------------- for other properties per spine obj -------------

def calculate_spine_obj_mesh_skeleton_coordinates_for_branch(branch_obj):
    if branch_obj.spines_obj is None:
        return branch_obj
    branch_obj.spines_obj = [spu.calculate_spine_obj_mesh_skeleton_coordinates(branch_obj,k)
                            for k in branch_obj.spines_obj] 
    return branch_obj

def calculate_spine_obj_mesh_skeleton_coordinates(
    branch_obj=None,
    spine_obj=None,
    coordinate_method = 'first_coordinate',#"mean",
    plot_intersecting_vertices = False,
    plot_closest_skeleton_coordinate = False,
    spine_objs = None,
    branch_shaft_mesh_face_idx=None,
    verbose = False,
    verbose_time = False,
    mesh = None,
    skeleton = None,
    **kwargs
    ):


    """
    Will compute a lot of the properties of 
    spine objects that are equivalent to those
    computed in add_valid_synapses_to_neuron_obj

    The attributes include

    "endpoints_dist",
    "upstream_dist",
    "downstream_dist",
    "coordinate",
    "closest_sk_coordinate",
    "closest_face_idx",
    "closest_branch_face_idx",
    "closest_face_dist",
    "closest_face_coordinate",

    Pseudocode: 
    1) Make sure the branches have upstream and downstream set
    2) Find intersection of vertices between branch and shaft
    3) Find average vertices that make up the coordinate
    4) Find the closest mesh coordinate
    5) Find the closest skeeleton point

    """
    
    ''' Old method
    try:
        
        overlap_verts= tu.find_border_vertices(
            spine_obj.mesh,
            return_coordinates=True)
    except:
        branch_obj_minus_spine = tu.subtract_mesh(
            branch_obj.mesh,
            spine_obj.mesh,
        )
        overlap_verts = nu.intersect2d(
            spine_obj.mesh.vertices,
            branch_obj_minus_spine.vertices
        )
        


    if len(overlap_verts) == 0:
#         overlap_verts = spine_obj.mesh.vertices[0].reshape(-1,3)
#         if verbose:
#             print(f"Using first spine vertex as coordinate because no overlapping")
        overlap_verts = tu.closest_mesh_vertex_to_other_mesh(
            spine_obj.mesh,
            branch_obj.mesh_shaft,
            plot = False,
            verbose = False,
        ).reshape(-1,3)



    #3) Find average vertices that make up the coordinate    
    if coordinate_method == "mean":
        coordinate = np.mean(overlap_verts,axis = 0)
    else:
        coordinate = overlap_verts[0]
    if verbose:
        print(f"coordinate = {coordinate}")
    '''
    attr_to_set = [
        "coordinate",
        "closest_branch_face_idx",
        "closest_face_dist",
        "closest_face_coordinate",
        "coordinate_border_verts"
    ]
    
    if branch_obj is not None and mesh is None:
        mesh = branch_obj.mesh
    if branch_obj is not None and skeleton is None:
        skeleton = branch_obj.skeleton
    
    
    if spine_objs is not None:
        meshes_to_minus = [k.mesh for k in spine_objs]
    else:
        meshes_to_minus = None
        
    # find the overlapping vertices
    if branch_shaft_mesh_face_idx is not None:
        overlapping_vertices = tu.overlapping_vertices_from_face_lists(
            mesh,
            face_lists = [branch_shaft_mesh_face_idx,spine_obj.mesh_face_idx],
            return_idx = False
        )
    else:
        overlapping_vertices = None
        
    #print(f"overlapping_vertices = {overlapping_vertices}")
    
    try:
        coordinate,coordinate_border_verts = tu.coordinate_on_mesh_mesh_border(
            mesh=spine_obj.mesh,
            mesh_border=mesh,
            meshes_to_minus = meshes_to_minus,
            coordinate_method = coordinate_method,#"mean",
            overlapping_vertices=overlapping_vertices,
            verbose = verbose,
            verbose_time=verbose_time,
            return_winning_coordinate_group = True,
            plot=False,
        )
    except:
        coordinate = tu.closest_mesh_coordinate_to_other_mesh(
            spine_obj.mesh,
            mesh
        )
        
        coordinate_border_verts = coordinate
    


    #4) Find the closest mesh coordinate
    closest_branch_face_idx = tu.closest_face_to_coordinate(mesh,coordinate)
    closest_face_coordinate= mesh.triangles_center[closest_branch_face_idx]
    closest_face_dist = np.linalg.norm(coordinate - closest_face_coordinate)

    if verbose:
        print(f"closest_branch_face_idx = {closest_branch_face_idx}")
        print(f"closest_face_coordinate = {closest_face_coordinate}")
        print(f"closest_face_dist = {closest_face_dist}")

    #5) Find the closest skeeleton point
    closest_sk_coordinate = None
    if skeleton is not None:
        closest_sk_coordinate = sk.closest_skeleton_coordinate(skeleton,
                                    closest_face_coordinate)
        attr_to_set.append("closest_sk_coordinate")
    if verbose:
        print(f"closest_sk_coordinate= {closest_sk_coordinate}")


    
    for k in attr_to_set:
        setattr(spine_obj,k,eval(k))
        
    return spine_obj

def id_from_idx(
    limb_idx,
    branch_idx,
    spine_idx,
    ):
    
    limb_idx = nru.get_limb_int_name(limb_idx)
    name_as_string = f"{str(limb_idx).zfill(2)}{str(branch_idx).zfill(3)}{str(spine_idx).zfill(3)}"
    return int(name_as_string)
    
def calculate_spine_obj_attr_for_neuron(
    neuron_obj,
    verbose = False,
    create_id = True,
    **kwargs
    ):
    """
    Purpose: To set all of the
    neuron_obj spine attributes

    Pseudocode: 
    for limbs
        for branches
            1) calculate the mesh and skeleton info
            2) calculate_branch_attr_soma_distances_on_limb

    calculate the soma distance
    
    Ex: neuron_obj= spu.calculate_spine_obj_attr_for_neuron(neuron_obj,verbose = True)

    """
    
    for limb_idx in neuron_obj.get_limb_names():
        if verbose:
            print(f"--- Working on Limb {limb_idx}")
        limb_obj = neuron_obj[limb_idx]
        for branch_idx in limb_obj.get_branch_names():
            if verbose:
                print(f"     Branch {branch_idx}")
            branch_obj = limb_obj[branch_idx]
            #1) calculate the mesh and skeleton info
            limb_obj[branch_idx] = spu.calculate_spine_obj_mesh_skeleton_coordinates_for_branch(branch_obj)
            
            if create_id and branch_obj.spines_obj is not None:
                for s_idx,sp_ogj in enumerate(branch_obj.spines_obj):
                    branch_obj.spine_id = spu.id_from_idx(limb_idx,branch_idx,s_idx)

        #2) calculate_branch_attr_soma_distances_on_limb
        limb_obj = _branch_attr_soma_distances_on_limb(
            limb_obj,
            branch_attr="spines_obj",
            calculate_endpoints_dist_if_empty=True
        )
        neuron_obj[limb_idx] = limb_obj

    if verbose:
        print(f"Working on calculate_neuron_soma_distance_euclidean")
    _neuron_soma_distance_euclidean(neuron_obj, branch_attr="spines_obj")
    _set_limb_branch_idx_to_attr(neuron_obj, branch_attr="spines_obj")

    return neuron_obj

def calculate_endpoints_dist(branch_obj, spine_obj):
    _calculate_endpoints_dist(branch_obj, spine_obj)

def calculate_upstream_downstream_dist_from_up_idx(spine_obj, up_idx):
    _calculate_upstream_downstream_dist_from_up_idx(spine_obj, up_idx=up_idx)
    
    
# ------- for calculating properties of spines ----
def skeleton_from_spine(spine,plot=False):
    if "trimesh" in str(type(spine)):
        mesh = spine
    else:
        mesh = spine.mesh
        
    spine_sk = sk.surface_skeleton(mesh,plot=plot)
    return spine_sk



def volume_from_spine(spine,default_value = 0):
    try:
        return tu.mesh_volume(spine.mesh)
    except:
        return default_value


# ---------------------- 11/3: Calculating the spines with all of the information  -----
def spine_objs_with_border_sk_endpoint_and_soma_filter_from_scratch_on_branch_obj(
    branch_obj = None,
    plot_segmentation = False,
    ensure_mesh_conn_comp = True,
    plot_spines_before_filter = False,
    
    
    filter_out_border_spines = None,
    border_percentage_threshold = None,
    plot_spines_after_border_filter = False,

    skeleton_endpoint_nullification = False,
    skeleton_endpoint_nullification_distance = None,
    plot_spines_after_skeleton_endpt_nullification = False,

    soma_vertex_nullification = None,
    soma_verts = None,
    soma_kdtree = None,
    plot_spines_after_soma_nullification = None,
    
    plot = False,
    verbose = False,
    mesh = None,
    skeleton = None,
    **kwargs
    ):
    """
    Purpose
    -----------
    Performs spine detection on a branch object or a branch mesh (optionally with a skeleton) and then apply filtering before creating official Spine objects from each individually detected spine.

    Purpose Detailed
    -----------------------

    Pseudocode
    ----------------
    1) Generate initial spine mesh detection
    2) Filter out border spines:
        if requested (filter_out_border_spines), filter out meshes that have:
            a) higher than a certain percentage (border_percentage_threshold) of the submesh vertices overlapping with border vertices (certices adjacent to open spaces in the mesh) on the parent mesh  
            b) higher than certain percentage (check_spine_border_perc_global) of the parent mesh’s border vertices overlapping with the submesh vertices
    3) skeleton endpoint filtering:
        if requested (skeleton_endpoint_nullification), filter away spines that are within a certain distance (skeleton_endpoint_nullification_distance) from the branch skeleton endpoints in order to avoid a high false positive class.
    4) some vertex nullification:
        if requested (soma_vertex_nullification), filter out spines that have vertices overlapping with vertices of the soma
    5) Creates spine objects from all the remaining spine objects (will do head/neck segmentaiton)

    Global Parameters to Set
    ----------------
    # -- border filtering
    filter_out_border_spines: bool
        whether to be perform spine filtering by considering how much spine submesh vertices overlap with border vertices of shaft submesh

    border_percentage_threshold: float
        maximum percentage a submesh vertices can overlap with border vertices (vertices adjacent to open spaces in the mesh) on the parent mesh and still be in consideration for spine label  

    # -- skeleton filtering
    skeleton_endpoint_nullification: bool
        whether to filter away spines that are within a certain distance (skeleton_endpoint_nullification_distance) from the branch skeleton endpoints in order to avoid a high false positive class.

    skeleton_endpoint_nullification_distance_global: float
        minimum distance a spine mesh can be from the branch skeleton endpoints and not be filtered away when the skeleton_endpoint_nullification flag is set

    # -- soma filtering
    soma_vertex_nullification: bool
        when true will filter out spines that have vertices overlapping with vertices of the soma


    """
    
    if filter_out_border_spines is None:
        filter_out_border_spines = filter_out_border_spines_global
        
    if border_percentage_threshold is None:
        border_percentage_threshold = border_percentage_threshold_global
        
    if skeleton_endpoint_nullification is None:
        skeleton_endpoint_nullification = skeleton_endpoint_nullification_global
        
    if skeleton_endpoint_nullification_distance is None:
        skeleton_endpoint_nullification_distance = skeleton_endpoint_nullification_distance_global
        
    if soma_vertex_nullification is None:
        soma_vertex_nullification = soma_vertex_nullification_global

    
    if mesh is None:
        mesh = branch_obj.mesh
        
        

    (spines,
     spines_sdf,
     spines_mesh_idx) = spu.get_spine_meshes_unfiltered_from_mesh(
        mesh,
        delete_temp_file=True,
        return_sdf=True,
        return_mesh_idx = True,
        print_flag=False,
        plot_segmentation=plot_segmentation,
        ensure_mesh_conn_comp=ensure_mesh_conn_comp,
        plot = False,
        **kwargs
    )
    """
    Purpose:
    -------
    if requested (filter_by_bounding_box_longest_side_length), filters the meshes to have less than a certain length (side_length_threshold) for the longest side of their oriented bounding box. To prevent false positive spines from long axon fragment merges

    """

        



    spine_submesh_split_filtered = spines
    
    if len(spine_submesh_split_filtered) == 0:
        return spine_submesh_split_filtered
    """
    Purpose
    -------
    if requested (filter_out_border_spines), filter out meshes that have:
        a) higher than a certain percentage (border_percentage_threshold) of the submesh vertices overlapping with border vertices (certices adjacent to open spaces in the mesh) on the parent mesh  
        b) higher than certain percentage (check_spine_border_perc_global) of the parent mesh’s border vertices overlapping with the submesh vertices

    """
    if filter_out_border_spines:
        if verbose:
            print("Using the filter_out_border_spines option")
        spine_idx = spu.filter_out_border_spines(
            mesh,
            spine_submesh_split_filtered,
            border_percentage_threshold=border_percentage_threshold,
            check_spine_border_perc=check_spine_border_perc,
            verbose=verbose,
            return_idx = True,
        )
        
        spine_submesh_split_filtered = spine_submesh_split_filtered[spine_idx]
        spines_sdf=spines_sdf[spine_idx]
        spines_mesh_idx = spines_mesh_idx[spine_idx]

        if verbose:
            print(f"After filter_out_border_spines: # of spines = {len(filter_out_border_spines)}")

    if len(spine_submesh_split_filtered) == 0:
        return spine_submesh_split_filtered

    """
    Purpose
    -------
    if requested (skeleton_endpoint_nullification), filter away spines that are within a certain distance (skeleton_endpoint_nullification_distance) from the branch skeleton endpoints in order to avoid a high false positive class.
    """
    if skeleton_endpoint_nullification and (branch_obj is not None or skeleton is not None):
        if skeleton is None:
            skeleton = branch_obj.skeleton
        if verbose:
            print("Using the skeleton_endpoint_nullification option")


        curr_branch_end_coords = sk.find_skeleton_endpoint_coordinates(skeleton)
        spine_idx = tu.filter_meshes_by_containing_coordinates(
            spine_submesh_split_filtered,
            curr_branch_end_coords,
            distance_threshold=skeleton_endpoint_nullification_distance,
            return_indices=True
        )
        
        spine_submesh_split_filtered = spine_submesh_split_filtered[spine_idx]
        spines_sdf=spines_sdf[spine_idx]
        spines_mesh_idx = spines_mesh_idx[spine_idx]

            
        if verbose:
            print(f"After skeleton_endpoint_nullification: # of spines = {len(spine_submesh_split_filtered)}")


    if len(spine_submesh_split_filtered) == 0:
        return spine_submesh_split_filtered
    
    """
    Purpose:
    --------
    
    if requested (soma_vertex_nullification), filter out spines that have vertices overlapping with vertices of the soma
    """
    if soma_vertex_nullification and (soma_kdtree is not None or soma_kdtree is not None):
        if soma_verts is not None:
            soma_kdtree = KDTree(soma_verts)
        if soma_kdtree is None:
            raise Exception("Requested soma vertex but no soma information given")

        if verbose:
            print("Using the soma_vertex_nullification option")

        spine_idx = spu.filter_out_soma_touching_spines(spine_submesh_split_filtered,
                                                    soma_kdtree=soma_kdtree,
                                                       return_idx = True)
        
        spine_submesh_split_filtered = spine_submesh_split_filtered[spine_idx]
        spines_sdf=spines_sdf[spine_idx]
        spines_mesh_idx = spines_mesh_idx[spine_idx]

            
        if verbose:
            print(f"After soma_vertex_nullification: # of spines = {len(spine_submesh_split_filtered)}")
            
    
    """
    Purpose:
    --------
    Creates spine objects from all the remaining spine objects (will do head/neck segmentaiton)
    """
    # Creating the spine objects
    spine_objs = [spu.Spine(
        mesh = k,
        mesh_face_idx = k_idx,
        sdf = s
    ) for k,k_idx,s in zip(spine_submesh_split_filtered,spines_mesh_idx,spines_sdf,)]
    
    
    if verbose:
        print(f"Final Number of Spine Objects = {len(spine_objs)}")
        
    return spine_objs




def df_from_spine_objs(
    spine_objs,
    attributes_to_skip = (
        "mesh_face_idx",
        "mesh",
        "neck_face_idx",
        "head_face_idx",
        "sdf",
        "skeleton",
    ),
    attributes_to_add = computed_attributes_to_export,
    columns_at_front = computed_attributes_to_export,
    columns_at_back = None,
    attributes = None,
    add_volume_to_area_ratio = False,
    verbose = False,
    verbose_loop = False,
    ):
    """
    Purpose: make a spine attribute
    dataframe from a list of spines

    Pseudocode: 
    1) 
    """
    st = time.time()
    spine_objs = nu.to_list(spine_objs)
    
    dicts = []
    for j,k in enumerate(spine_objs):
        if verbose_loop:
            print(f"spine {j}")
        curr_dict = k.export(
            attributes_to_skip=attributes_to_skip,
            attributes_to_add=attributes_to_add,
            attributes=attributes,
        )
        if add_volume_to_area_ratio:
            curr_dict["spine_volume_to_spine_area"] = spu.spine_volume_to_spine_area(k)
        dicts.append(
            curr_dict
        )
        
        
    
    df = pd.DataFrame.from_records(dicts)
    
    if attributes is None:
        df = pu.order_columns(
            df,
            columns_at_front=columns_at_front,
            columns_at_back=columns_at_back,
        )
    if verbose:
        print(f"Time for generating df {time.time() - st}")
    
    return df


def plot_spine_objs(
    spine_objs,
    branch_obj = None,
    mesh = None,
    plot_mesh_centers = True,
    spine_color = "random",
    mesh_alpha = 1,
    ):
    

    if branch_obj is not None:
        mesh = branch_obj.mesh
        
    scatters = None
    if plot_mesh_centers:
        scatters = [np.vstack([k.mesh_center for k in spine_objs]).reshape(-1,3)]
def filter_spine_objs_from_restrictions(
    spine_objs,
    restrictions,
    spine_df = None,
    verbose=False,
    return_idx = False,
    joiner = "AND",
    plot = False,
    **kwargs
    ):
    """
    Purpose: Want to filter the spines with
    a list of queries
    """
    if spine_df is None:
        spine_df = spu.df_from_spine_objs(
            spine_objs,
            add_volume_to_area_ratio=True,
            **kwargs
        )
        
    
        
    idx = pu.query_table_from_list(
        spine_df,
        restrictions=restrictions,
        verbose_filtering=verbose,
        return_idx = True,
        joiner=joiner,
        **kwargs
    )
    
    spine_objs_new = [spine_objs[k] for k in idx]
    
    if return_idx:
        return idx
    
    if plot:
        plot_spine_objs(spine_objs_new)
    return spine_objs_new

query_spine_objs = filter_spine_objs_from_restrictions


    
    
def spine_volume_to_spine_area(spine_obj):
    if spine_obj.volume is None:
        return 0
    return (spine_obj.volume/spu.volume_divisor)/(spine_obj.area/spu.area_divisor)

def filter_spine_objs_by_size_bare_minimum(
    spine_objs,
    spine_n_face_threshold: int = None,#6,
    spine_sk_length_threshold: float = None,#306.6,
    filter_by_volume_threshold: float = None,#900496.186,
    bbox_oriented_side_max_min: int = None,#300,
    sdf_mean_min: float = None,#0,
    spine_volume_to_spine_area_min: float = None,
    verbose = False,
    ):
    """
    Purpose
    -----------
    Filters the spine objects for minimum feature requirements.

    Pseudocode
    ----------------
    1) Apply a list of simple attribute “greater than” queries to filter down the spine objects.


    Global Parameters to Set
    ----------------
    spine_n_face_threshold_bare_min: 
        minimum number of faces for a valid spine mesh

    spine_sk_length_threshold_bare_min: 
        minimum surface skeletal length (units) of a valid spine mesh

    filter_by_volume_threshold_bare_min:
        minimum volume (units) of a valid spine mesh

    bbox_oriented_side_max_min_bare_min:
        minimum side length (uints) of the oriented bounding box surrounding the spine mesh for valid spines

    sdf_mean_min_bare_min: 
        minimum mean sdf value (computed in the shaft/spine clustering step performed by the cgal clustering algorithm) for valid spine meshes

    spine_volume_to_spine_area_min_bare_min:
        minimum ratio of mesh volume (units^3) to mesh area (units^2) for valid spine meshes

    Notes
    --------
    1) Visualize the spines after these filtering


    Parameters
    ----------
    spine_objs : _type_
        _description_
    spine_n_face_threshold : int, optional
        _description_, by default None
    spine_sk_length_threshold : float, optional
        _description_, by default None
    filter_by_volume_threshold : float, optional
        _description_, by default None
    bbox_oriented_side_max_min : int, optional
        _description_, by default None
    sdf_mean_min : float, optional
        _description_, by default None
    spine_volume_to_spine_area_min : float, optional
        _description_, by default None
    verbose : bool, optional
        _description_, by default False

    Returns
    -------
    _type_
        _description_
    """
    if len(spine_objs) == 0:
        return spine_objs
    
    if spine_n_face_threshold is None:
        spine_n_face_threshold = spine_n_face_threshold_bare_min_global
        
    if spine_sk_length_threshold is None:
        spine_sk_length_threshold = spine_sk_length_threshold_bare_min_global
        
    if filter_by_volume_threshold is None:
        filter_by_volume_threshold = filter_by_volume_threshold_bare_min_global
        
    if bbox_oriented_side_max_min is None:
        bbox_oriented_side_max_min = bbox_oriented_side_max_min_bare_min_global
        
    if sdf_mean_min is None:
        sdf_mean_min = sdf_mean_min_bare_min_global
        
    if spine_volume_to_spine_area_min is None:
        spine_volume_to_spine_area_min = spine_volume_to_spine_area_min_bare_min_global
        
        
    #print(f"filter_by_volume_threshold = {filter_by_volume_threshold}")
    sp_objs_filt = spu.filter_spine_objs_from_restrictions(
        spine_objs,
        restrictions = [
            f"n_faces >= {spine_n_face_threshold}",
            f"(skeletal_length >= {spine_sk_length_threshold}) or (skeletal_length != skeletal_length)",
            f"(volume >= {filter_by_volume_threshold}) or (volume != volume)",
            f"sdf_mean > {sdf_mean_min}",
            f"bbox_oriented_side_max > {bbox_oriented_side_max_min}",
            f"spine_volume_to_spine_area >= {spine_volume_to_spine_area_min}",
        ],
        verbose = verbose,
    )

    return sp_objs_filt
    
    

        
def split_head_mesh(
    spine_obj,
    return_face_idx_map = True,
    plot = False,):
    """
    Purpose: want to divide mesh into connected
    components and optionally return the mask
    of mapping fae to component
    """
    head_meshes,head_face_idx = tu.connected_components_from_mesh(
        spine_obj.head_mesh,
        return_face_idx_map = True,
        plot=plot
    )
    
    if return_face_idx_map:
        return head_meshes,head_face_idx
    else:
        return head_meshes
    
def spine_mesh(spine_obj):
    return spine_obj.mesh
    
def mesh_from_compartment(
    spine_obj,
    compartment = "head",
    index = 0
    ):
    
    args = dict()
    if compartment == "head":
        mesh_func = spu.head_mesh_splits_from_index
        args = dict(index=index)
    elif compartment == "neck":
        mesh_func = spu.neck_mesh
    elif compartment == 'spine':
        mesh_func = spu.spine_mesh
    elif compartment == 'no_head':
        mesh_func = spu.neck_mesh
    else:
        raise Exception("")
    
        
    return mesh_func(spine_obj,**args)

def mesh_attribute_from_compartment(
    spine_obj,
    attribute_func,
    compartment = "head",
    index = 0,
    shaft_default_value = None,
    **kwargs
    ):
    
    if type(attribute_func) == str:
        attribute_func = eval(f"tu.{attribute_func}")
        
    if type(compartment) != str:
        compartment,index = compartment_index_from_id(compartment)
        
    if compartment == "shaft":
        return shaft_default_value
    
    curr_mesh = spu.mesh_from_compartment(spine_obj,compartment=compartment,index=index)
    if curr_mesh is None or len(curr_mesh.faces) == 0:
        return None
    return attribute_func(
        curr_mesh,
        **kwargs
    )
    
def width_ray_from_compartment(
    spine_obj,
    compartment = "head",
    index = 0,
    percentile = 50,
    default_value_if_empty = 0,
    ):
    
    return mesh_attribute_from_compartment(
        spine_obj,
        attribute_func=tu.width_ray_trace_perc,
        compartment = compartment,
        index = index,
        percentile=percentile,
        default_value_if_empty=default_value_if_empty,
        )

def width_ray_80_perc_from_compartment(
    spine_obj,
    compartment = "head",
    index = 0,
    default_value_if_empty = 0,
    ):
    
    return mesh_attribute_from_compartment(
        spine_obj,
        attribute_func=tu.width_ray_trace_perc,
        compartment = compartment,
        index = index,
        percentile=80,
        default_value_if_empty=default_value_if_empty,
        )

def area_from_compartment(
    spine_obj,
    compartment = "head",
    index = 0,
    ):
    

    return mesh_attribute_from_compartment(
        spine_obj,
        attribute_func=tu.area,
        compartment = compartment,
        index = index,
        )

def n_faces(
    spine_obj,
    compartment = "head",
    index = 0,
    ):
    
    def my_func(mesh):
        return len(mesh.faces)

    return mesh_attribute_from_compartment(
        spine_obj,
        attribute_func=my_func,
        compartment = compartment,
        index = index,
        )

def volume_from_compartment(
    spine_obj,
    compartment = "head",
    index = 0,
    ):

    return mesh_attribute_from_compartment(
        spine_obj,
        attribute_func=tu.mesh_volume,
        compartment = compartment,
        index = index,
        )

def skeletal_length_from_compartment(
    spine_obj,
    compartment = "head",
    index = 0,
    ):

    return mesh_attribute_from_compartment(
        spine_obj,
        attribute_func=tu.skeletal_length_from_mesh,
        compartment = compartment,
        index = index,
    )

bbox_oriented = False
def bbox_min_x_nm_from_compartment(
    spine_obj,
    compartment = "head",
    index = 0,
    ):

    return mesh_attribute_from_compartment(
        spine_obj,
        attribute_func=tu.bbox_min_x,
        compartment = compartment,
        index = index,
        oriented=bbox_oriented,
    )

def bbox_min_y_nm_from_compartment(
    spine_obj,
    compartment = "head",
    index = 0,
    ):

    return mesh_attribute_from_compartment(
        spine_obj,
        attribute_func=tu.bbox_min_y,
        compartment = compartment,
        index = index,
        oriented=bbox_oriented,
    )

def bbox_min_z_nm_from_compartment(
    spine_obj,
    compartment = "head",
    index = 0,
    ):

    return mesh_attribute_from_compartment(
        spine_obj,
        attribute_func=tu.bbox_min_z,
        compartment = compartment,
        index = index,
        oriented=bbox_oriented,
    )

def bbox_max_x_nm_from_compartment(
    spine_obj,
    compartment = "head",
    index = 0,
    ):

    return mesh_attribute_from_compartment(
        spine_obj,
        attribute_func=tu.bbox_max_x,
        compartment = compartment,
        index = index,
        oriented=bbox_oriented,
    )

def bbox_max_y_nm_from_compartment(
    spine_obj,
    compartment = "head",
    index = 0,
    ):

    return mesh_attribute_from_compartment(
        spine_obj,
        attribute_func=tu.bbox_max_y,
        compartment = compartment,
        index = index,
        oriented=bbox_oriented,
    )

def bbox_max_z_nm_from_compartment(
    spine_obj,
    compartment = "head",
    index = 0,
    ):

    return mesh_attribute_from_compartment(
        spine_obj,
        attribute_func=tu.bbox_max_z,
        compartment = compartment,
        index = index,
        oriented=bbox_oriented,
    )

def spine_compartment_mesh_functions(
    compartments = ("spine","head",'neck',),
    stats_functions = (
        "width_ray_from_compartment",
        "width_ray_80_perc_from_compartment",
        "area_from_compartment",
        "volume_from_compartment",
        "skeletal_length_from_compartment",
        "n_faces",
        "bbox_min_x_nm_from_compartment",
        "bbox_min_y_nm_from_compartment",
        "bbox_min_z_nm_from_compartment",
        
        "bbox_max_x_nm_from_compartment",
        "bbox_max_y_nm_from_compartment",
        "bbox_max_z_nm_from_compartment",
    ),
    verbose = True
    ):

    """
    Purpose: To generate the size 
    functions for all compartments: spine,head,neck,

    Pseudocode: 
    1) Iterate through all compartments
    2) print out the formatted function
    """

    function_names = []
    function_declaration_str = ""
    for comp in compartments:
        function_declaration_str += (f"#--- Mesh Attribute Functions for {comp} -----\n")
        for st in stats_functions: 
            func_name = f"{comp}_{st.replace('_from_compartment','')}"
            function_names.append(func_name)
            function_declaration_str += (
            f"@property\ndef {func_name}(self,**kwargs):\n"
            f"    return {st}(self,compartment = '{comp}',**kwargs)\n\n"
            )

    if verbose:
        print(f"{function_declaration_str}")
    return function_names

    
def plot_spines_objs_with_head_neck_and_coordinates(
    spine_objs,
    branch_obj=None,
    mesh = None,
    head_color = "red",
    neck_color = "aqua",
    no_head_color = "black",
    base_coordinate_color = "pink",
    center_coordinate_color = "orange",
    mesh_alpha = 0.8,
    verbose = False
    ):
    """
    Purpose: Want to plot from a list of spines
    all of the head,necks, centroids and 
    coordinates of spines

    Pseudocode: 
    For each spine:
    a) get the head/neck and put into mesh lists
    b) Get the coordinates and mesh_center and put into scatter
    3) Plot all  with the branch mesh
    """


    if branch_obj is None:
        mesh = mesh
    else:
        mesh = branch_obj.mesh
        
    head_meshes = []
    neck_meshes = []
    no_head_meshes = []
    base_coords = []
    center_coords = []
    for s in spine_objs:
        if s.n_heads == 0:
            no_head_meshes.append(s.no_head_mesh)
        else:
            head_meshes.append(s.head_mesh)
            neck_meshes.append(s.neck_mesh)
        center_coords.append(s.mesh_center)
        if s.coordinate is None:
            continue
        base_coords.append(s.coordinate)
        
    #print(f"head_meshes = {head_meshes}")

    base_coords = np.array(base_coords).reshape(-1,3)
    center_coords = np.array(center_coords).reshape(-1,3)
    
    # concatenating the meshes
    head_meshes = [tu.combine_meshes(head_meshes)]
    neck_meshes = [tu.combine_meshes(neck_meshes)]
    no_head_meshes = [tu.combine_meshes(no_head_meshes)]
    
    if verbose:
        print(f"head_meshes = {head_meshes}")
        print(f"neck_meshes = {neck_meshes}")
        print(f"no_head_meshes = {no_head_meshes}")

    colors = (
        [head_color]*len(head_meshes) +
        [neck_color]*len(neck_meshes) + 
        [no_head_color]*len(no_head_meshes)
    )

def spine_objs_bare_minimum_filt_with_attr_from_branch_obj(
    branch_obj=None,
    soma_verts_on_limb=None,
    soma_kdtree_on_limb=None,
    plot_unfiltered_spines = False,
    plot_filtered_spines = False,
    verbose = False,
    
    #for calculatin distance of spine
    soma_center = None,
    upstream_skeletal_length = None,
    
    #for branch features
    branch_features = None,
    mesh = None,
    skeleton = None,
    **kwargs
    ):
    """
    Purpose
    -----------
    Performs spine detection on a branch object or a branch mesh (optionally with a skeleton)

    Purpose Detailed
    -----------------------

    Pseudocode
    ----------------
    1) Generates spine objects
    2) Calculate spine attributes for each spine object
    3) Apply bare minimum spine attribute filtering (filter_spine_objs_by_size_bare_minimum)

    
        
    Analysis Roadmap
    ----------------
    spu.spine_objs_bare_minimum_filt_with_attr_from_branch_obj
        spu.spine_objs_with_border_sk_endpoint_and_soma_filter_from_scratch_on_branch_obj
        
            spu.get_spine_meshes_unfiltered_from_mesh
                spu.get_spine_meshes_unfiltered_from_mesh
                    spu.split_mesh_into_spines_shaft:
                        tu.mesh_segmentation
                            VISUALIZATION: visualization that enough chopped up
                            PARAMETER CHANGE:
                                smoothness_threshold
                                clusters_threshold

                        spu.restrict_meshes_to_shaft_meshes_without_coordinates
                            VISUALIZATION: look at initial shaft seaparation (c)
                            PARAMETER CHANGE:
                                shaft_close_hole_area_top_2_mean_max
                                shaft_mesh_volume_max
                                shaft_mesh_n_faces_min

                VISUALIZATION: individual spines prior to individual spine filtering

            VISUALIZATION: spines after filtering (or each step after filtering)
            PARAMETER CHANGE:
                # -- border filtering
                filter_out_border_spines
                border_percentage_threshold
                # -- skeleton filtering
                skeleton_endpoint_nullification
                skeleton_endpoint_nullification_distance
                # -- soma filtering
                soma_vertex_nullification: bool


        filter_spine_objs_by_size_bare_minimum
            VISUALIZATION: spines before and after substitution
            PARAMETER CHANGE:                
                spine_n_face_threshold_bare_min
                spine_sk_length_threshold_bare_min
                filter_by_volume_threshold_bare_min
                bbox_oriented_side_max_min_bare_min
                sdf_mean_min_bare_min
                spine_volume_to_spine_area_min_bare_min
                
        spu.calculate_spine_attributes_for_list
            spu.calculate_spine_attributes:
                spu.calculate_head_neck:
                    VISUALIZATION: spine head/neck subdivision
                    PARAMETER CHANGE:
                        head_smoothness
                        head_ray_trace_min
                        head_face_min
                        only_allow_one_connected_component_neck

    """


    if mesh is None:
        mesh = branch_obj.mesh
    if skeleton is None and branch_obj is not None:
        skeleton = branch_obj.skeleton
    
    if soma_kdtree_on_limb is None and soma_verts_on_limb is not None:
        soma_kdtree_on_limb = KDTree(soma_verts_on_limb)
    sp_objs = spu.spine_objs_with_border_sk_endpoint_and_soma_filter_from_scratch_on_branch_obj(
        branch_obj=branch_obj,
        soma_kdtree = soma_kdtree_on_limb,
        verbose = verbose,
        plot = plot_unfiltered_spines,
        mesh=mesh,
        skeleton = skeleton,
    )
    
    #raise Exception("")

    sp_objs = spu.calculate_spine_attributes_for_list(
        sp_objs,
        calculate_coordinates=False,
    )

    if verbose:
        print(f"Before filtering len(sp_objs) = {len(sp_objs)}")

    #filters the spine objects
    sp_objs_filt = spu.filter_spine_objs_by_size_bare_minimum(sp_objs)
    if verbose:
        print(f"AFTER filtering len(sp_objs_filt) = {len(sp_objs_filt)}")


    #calculates the center and closest face idx
    sp_objs_filt = spu.calculate_spine_attributes_for_list(
        sp_objs_filt,
        branch_obj = branch_obj,
        calculate_coordinates=True,
        calculate_head_neck = True,
        verbose_time=False,
        soma_center=soma_center,
        upstream_skeletal_length=upstream_skeletal_length,
        mesh = mesh,
        **kwargs
    )

    if plot_filtered_spines:
        spu.plot_spines_objs_with_head_neck_and_coordinates(
            sp_objs_filt,
            branch_obj = branch_obj,
            mesh = mesh
        )

        
    return sp_objs_filt


def id_from_compartment_index(
    compartment,
    index = 0):
    if compartment == "shaft":
        return -3
    elif compartment == "no_head":
        return -2
    elif compartment == "neck":
        return -1
    elif compartment == "head":
        return 0 + index
    else:
        raise Exception("")
        
def compartment_index_from_id(id):
    if id == -3:
        return "shaft",0
    elif id == -2:
        return "no_head",0
    elif id == -1:
        return "neck",0
    elif id >= 0:
        return "head",id
    else:
        raise Exception("")
        
        
def compartment_idx_for_mesh_face_idx_of_spine(spine_obj):
    """
    Purpose: Create a face map for that spines mesh_face_idx
    from the head,neck, and no_label 
    
    Ex: 
    spine_obj = output_spine_objs[5]
    spu.plot_head_neck(spine_obj)
    tu.split_mesh_into_face_groups(
        spine_obj.mesh,
        spu.compartment_idx_for_mesh_face_idx_of_spine(spine_obj),
        plot=True,
    )
    """
    empty_array = np.ones(len(spine_obj.mesh.faces))
    if spine_obj.n_heads == 0:
        return empty_array * id_from_compartment_index("no_head")
    else:
        curr_idx = empty_array * id_from_compartment_index("neck")
        curr_idx[spine_obj.head_face_idx] = spine_obj.head_mesh_splits_face_idx
        return curr_idx
    
def face_idx_map_from_spine_objs(
    spine_objs,
    branch_obj = None,
    mesh = None, 
    no_spine_index = -1,
    plot = False,
    ):
    """
    Purpose: from a branch
    mesh and spine objs on that branch mesh
    create an array (N,2) that maps every face to the shaft
    or a spine index and the compartment
    """
    if mesh is None:
        mesh = branch_obj.mesh


    face_map = np.ones((len(branch_obj.mesh.faces),2))*no_spine_index
    face_map[:,1] = spu.id_from_compartment_index("shaft")
    for j,s in enumerate(spine_objs):
        face_map[s.mesh_face_idx,0] = j
        face_map[s.mesh_face_idx,1] = spu.compartment_idx_for_mesh_face_idx_of_spine(s)

    if plot:
        tu.split_mesh_into_face_groups(
            branch_obj.mesh,
            face_map[:,1],
            plot = True,
        )

    return face_map

    

def synapse_df_with_spine_match(
    branch_obj,
    spine_objs,
    plot_face_idx_map = False,
    attributes_to_append = (
    "volume",
    "width_ray_80_perc",
    "area"
    ),

    attribute_rename_dict = dict(
        volume = "spine_volume"
    ),
    spine_id_column = "spine_id",
    spine_compartment_column ="spine_compartment",
    verbose = False,
    #filter_away_non_spine_synapses = True,
    ):
    """
    Purpose: Create a dataframe that maps all
    syanpses on the branch to the spine id and
    size of the spine id

    Pseudocode: 
    1) Creae an array the same size as # of faces
    of branch where the values are (spine_id, comp #)
    2) Use the synapse objects of branch
    """


    face_idx_map = spu.face_idx_map_from_spine_objs(
        spine_objs = spine_objs,
        branch_obj = branch_obj,
        plot = plot_face_idx_map,
    )


    syn_df = pd.DataFrame()

    if verbose:
        print(f"# of synases = {len(syn_df)}")
    
    return syn_df

    


                        
def spine_objs_and_synapse_df_computed_from_branch_idx(
    branch_obj=None,
    limb_obj = None,
    branch_idx = None,
    soma_verts_on_limb = None,
    soma_kdtree_on_limb = None,
    upstream_skeletal_length = None,
    plot_branch_mesh_before_spine_detection = False,
    
    #spine detection phase:
    plot_unfiltered_spines  = False,
    plot_filtered_spines = False,
    
    #branch features
    branch_features = None,
    
    verbose = False,
    verbose_computation = False,
    **kwargs
    ):
    """
    Purpose: from a branch object
    will generate spine objects and the
    synapse df of synaspes onto spines
    """
    
    if not verbose_computation:
        tqu.turn_off_tqdm()
    if limb_obj is not None and branch_idx is not None:
        upstream_skeletal_length = nst.total_upstream_skeletal_length(limb_obj,branch_idx)
        branch_obj = limb_obj[branch_idx]


    if soma_kdtree_on_limb is None:
        if soma_verts_on_limb is None:
            soma_verts_on_limb =limb_obj.current_touching_soma_vertices
        soma_kdtree_on_limb = KDTree(soma_verts_on_limb)

    #raise Exception("")
    output_spine_objs = spu.spine_objs_bare_minimum_filt_with_attr_from_branch_obj(
        branch_obj,
        soma_kdtree_on_limb =soma_kdtree_on_limb,
        plot_unfiltered_spines = plot_unfiltered_spines,
        plot_filtered_spines = plot_filtered_spines,
        soma_center = np.mean(soma_verts_on_limb,axis=0),
        upstream_skeletal_length=upstream_skeletal_length,
        verbose = verbose_computation,
        branch_features=branch_features,
    )


    syn_df = spu.synapse_df_with_spine_match(
        branch_obj,
        spine_objs = output_spine_objs,
        verbose = verbose_computation,
    )

    if verbose:
        print(f"# of output spines = {len(output_spine_objs)}")
        print(f"# of spine_synapses = {len(syn_df)}")
        
    if not verbose_computation:
        tqu.turn_on_tqdm()

    return output_spine_objs,syn_df

def spine_id_from_limb_branch_spine_idx(
    limb_idx,
    branch_idx,
    spine_idx = 0):
    """
    Purpose: Defines the method used for creating the spine id
    [limb,2][branches,4][spine,4]
    """
    limb_idx = nru.limb_idx(limb_idx)
    return int(f"{limb_idx}{str(branch_idx).zfill(4)}{str(spine_idx).zfill(4)}")


def spine_id_add_from_limb_branch_idx(limb_idx,branch_idx):
    return spine_id_from_limb_branch_spine_idx(limb_idx,branch_idx)



spine_compartments = ("spine_head","spine_neck","spine_no_head","shaft")
spine_compartments_no_prefix = [k.replace('spine_','') for k in spine_compartments]
                
    #return meshes
    


def features_to_export_for_db():
    total_features = [
        "spine_id",

        #coordinates
        "base_coordinate_x_nm",
        "base_coordinate_y_nm",
        "base_coordinate_z_nm",
        "mesh_center_x_nm",
        "mesh_center_y_nm",
        "mesh_center_z_nm",
        
        #head features
        "n_heads",
        "shaft_border_area",
    
        
        #distances from things
        "downstream_dist",
        "upstream_dist",
        "soma_distance_euclidean",
        "soma_distance",
        
        #branch features
        "compartment",
        "branch_width_overall",
        "branch_skeletal_length",
        "branch_width_at_base",
        
    ]

    #spine size features (and repeat for head/neck)
    size_features = [
        "area",
        "n_faces",
        "volume",
        "skeletal_length",
        "width_ray",
        "width_ray_80_perc",
        "bbox_oriented_side_max",
        "bbox_oriented_side_middle",
        "bbox_oriented_side_min",
#         "bbox_min_x_nm",
#         "bbox_min_y_nm",
#         "bbox_min_z_nm",
#         "bbox_max_x_nm",
#         "bbox_max_y_nm",
#         "bbox_max_z_nm",
    ]
    
    for prefix in ["spine","head","neck"]:
        total_features+= [f"{prefix}_{k}" for k in size_features]

    return total_features

    
    
# ---- function ofr extracting new spine_objs from neuron_obj ---


def plot_spine_coordinates_from_spine_df(
    mesh,
    spine_df,
    coordinate_types = (
        "base_coordinate",
        "mesh_center",
        "head_bbox_max",
        "head_bbox_min",
        "neck_bbox_max",
        "neck_bbox_min",
    )
    ):
    
    """
    Ex: 
    scats = spu.plot_spine_coordinates_from_spine_df(
        mesh = neuron_obj.mesh,
        spine_df=spine_df,
    )
    """

    scatters = [pu.coordinates_from_df(
        spine_df,k,filter_away_nans = True ) for k in 
        coordinate_types
    ]
    
    
    #return scatters
    
    
    
def synapse_spine_match_df_filtering(
    syn_df):
    """
    Purpose: To map columns and 
    filter away columns of synapse df
    for database write
    """

    if len(syn_df) == 0:
        return syn_df
    
    syn_df_renamed = pu.map_column_with_dict_slow(
        syn_df,
        column = "spine_compartment",
        dict_map = {-3:"shaft",-2:"no_head",-1:'neck'},
        default_value = "head",
        verbose = True,
        in_place=False
    )
    
    syn_df_renamed = syn_df_renamed[["syn_id","spine_id","spine_compartment","spine_volume","area","width_ray_80_perc"]]
    syn_df_renamed = pu.rename_columns(
        syn_df_renamed,
        dict(syn_id = "synapse_id",
             area = "spine_area",
            width_ray_80_perc = "spine_width_ray_80_perc")
    )

    return syn_df_renamed

area_divisor = 1_000_000
scale_dict_default = dict(
    volume = 1/1_000_000_000,
    area = 1/1_000_000
)
def scale_stats_df(
    df,
    scale_dict = scale_dict_default,
    in_place = False,
    ):
    """
    Purpose: Want to scale certain columns
    of dataframe by divisors if have keyword
    """
    if not in_place:
        df = df.copy()
    
    
    for k,scale in scale_dict.items():
        for col in df.columns:
            if k in col:
                df.loc[~df[col].isna(),col] = df.loc[~df[col].isna(),col].astype('int')*scale
                
                
    return df




def seg_split_spine(
    df= None,
    segment_id= None,
    split_index= None,
    spine_id = None,
    return_dicts = True,
    ):
    """
    Purpose: Get segment,split_index and spine_ids
    
    Ex: 
    spu.seg_split_spine(spine_df_trans_umap)

    spu.seg_split_spine(
        segment_id = 864691135730167737,
        split_index = 0,
        spine_id = [0,11],
        df = None,
    )
    """
    if df is not None:
        if "segment_id" in df.columns:
            segment_id = df.segment_id.to_list()
        if "split_index" in df.columns:
            split_index = df.split_index.to_list()
        if "spine_id" in df.columns:
            spine_id = df.spine_id.to_list()

    segment_id = nu.to_list(segment_id)
    split_index = nu.to_list(split_index)
    spine_id=nu.to_list(spine_id)

    if len(segment_id) != len(spine_id):
        if len(segment_id) == 1:
            segment_id = segment_id*len(spine_id)
        else:
            raise Exception("")

    if len(split_index) != len(spine_id):
        if len(split_index) == 1:
            split_index = split_index*len(spine_id)
        else:
            raise Exception("")

    if return_dicts:
        return [dict(segment_id=s,split_index = sp,spine_id = spi)
               for s,sp,spi in zip(segment_id,split_index,spine_id)]
    return segment_id,split_index,spine_id

seg_split_spine_from_df = seg_split_spine

def area_of_border_verts(spine_obj,default_value = 0):
    if spine_obj.coordinate_border_verts is None:
        return default_value
    return tu.area_of_vertex_boundary(
        spine_obj.mesh,
        spine_obj.coordinate_border_verts
    )

# ----- new shaft filtering function --------------
def restrict_meshes_to_shaft_meshes_without_coordinates(
    meshes,
    close_hole_area_top_2_mean_max = None, #110_000,
    mesh_volume_max = None,#0.3e9,
    #n_faces_max = 500,
    n_faces_min = None,
    verbose = False,
    plot = False,
    return_idx = True,
    return_all_shaft_if_none = True,
    ):
    """
    Purpose
    -----------
    To restrict a list of meshes to those with a high probability of being a shaft mesh

    Pseudocode
    ------------
    1) restricts meshes to those greater than a certain volume (shaft_mesh_volume_max) or greater than a certain mean top 2 hole area (shaft_close_hole_area_top_2_mean_max) because both could be indicative of a shaft mesh
        functions used: trimesh_utils.close_hole_area_top_2_mean, trimesh_utils.mesh_volume
    2) restricts meshes to face threshold (shaft_mesh_n_faces_min)
    3) Returns the indices of the meshes remaining after filtering

    Global Parameters to Set
    ----------------
    shaft_close_hole_area_top_2_mean_max: float
        a minimum mesh area (nm^2)  threshold for the mean of the top 2 holes (ideally the connecting ends of the tube like shaft sections from the mesh cluster (from the CGAL algorithm) to be potentially considered a part of the neuron shaft. (The max refers to a max suffix is in reference to spines)

    shaft_mesh_volume_max: int
        the minimum mesh volume (nm^3) for a mesh cluster (from the CGAL algorithm) to be potentially considered a part of the neuron shaft. (The max refers to a max suffix is in reference to spines)

    shaft_mesh_n_faces_min: int
        A minimum number of faces on a submesh cluster early in the pipeline to be potentially considered a part of the neuron shaft.


    """
    if close_hole_area_top_2_mean_max is None:
        close_hole_area_top_2_mean_max = shaft_close_hole_area_top_2_mean_max_global
    if mesh_volume_max is None:
        mesh_volume_max = shaft_mesh_volume_max_global
    if n_faces_min is None:
        n_faces_min = shaft_mesh_n_faces_min_global
    
    query = [
        f"(close_hole_area_top_2_mean > {close_hole_area_top_2_mean_max}) or (mesh_volume > {mesh_volume_max})",
        #f"(close_hole_area_top_2_mean > {close_hole_area_top_2_mean_max}) or (n_faces > {n_faces_max})",
        f"(n_faces > {n_faces_min})"
    ]

    shaft_meshes_idx= tu.query_meshes_from_stats(
        meshes,
        functions = [
            "close_hole_area_top_2_mean",
            "n_faces",
            "mesh_volume"
        ],
        query = query,
        verbose = verbose,
        plot = plot,
        return_idx = return_idx
    )
    
    if len(shaft_meshes_idx) == 0 and return_all_shaft_if_none:
        shaft_meshes_idx = np.arange(len(meshes))

    return shaft_meshes_idx

def split_mesh_into_spines_shaft(
    current_mesh,
    segment_name="",
    clusters=None,
    smoothness=None,
    cgal_folder = Path("./cgal_temp"),
    delete_temp_file=True,
    shaft_threshold = None,
     return_sdf = True,
    print_flag = False,
    plot_segmentation = False,
    plot_shaft = False,
    plot_shaft_buffer = 0,
    **kwargs
    ):
    """
    Purpose
    -----------
    Determine the final classification of shaft submeshes and then return all connected components (floating islands) of the submesh after the shaft has been removed as individual spine meshes. In other words, generates the meshes for each individual spine (prior to individual spine filtering steps).

    Pseudocode
    ----------------
    1) runs a segmentation on the mesh with given clusters and smoothness.	
    a) generates a mapping of the clusters the sdf values for each face
    2) splits the segmentation into separate meshes
    3) spu.restrict_meshes_to_shaft_meshes_without_coordinates
    4) Divide the meshes into spine meshes and shaft meshes (first pass)
    5) Given first pass of the shaft and spine classification, reclassifies some spine meshes as shaft meshes to ensure complete graph connectivity between all shaft submeshes using the following algorithm
            1) start with biggest shaft
            2) Find the shortest paths to all shaft parts
            3) add all the submeshes that aren't already in the shaft category to the shaft category
    6) Creates a total spine submesh by removing all of the shaft submeshes from the original mesh, divides up this total spine submesh into connected components (separate islands) as separate meshes so each mesh represents a single spine
    7)  Sorts the spine meshes from largest (number of faces) to smallest
    8) Returns spine meshes and the sdf values (generated in the segmentation step) associated with them



    Global Parameters to Set
    ----------------
    smoothness_threshold: 
        The smoothness parameter for the cgal mesh segmentation algorithm used as an initial intermediate over-segmentation step in the spine detection. The smaller the smoothness value, generally the more number of underlying clusters on the mesh identified and the more spines that can be potentially identified at the higher risk of false positives (although the spine algorithm attempts to filter away false positives from the underlying mesh segmentation). Source: https://doc.cgal.org/4.6/Surface_mesh_segmentation/index.html

    clusters_threshold:
        The clusters parameter for the cgal mesh segmentation algorithm used as an initial intermediate over-segmentation step in the spine detection. The larger the number of clusters the more spines can be potentially identified at the higher risk of false positives (although the spine algorithm attempts to filter away false positives from the underlying mesh segmentation). Source: https://doc.cgal.org/4.6/Surface_mesh_segmentation/index.html



    Parameters
    ----------
    current_mesh : _type_
        _description_
    segment_name : str, optional
        _description_, by default ""
    clusters : _type_, optional
        _description_, by default None
    smoothness : _type_, optional
        _description_, by default None
    cgal_folder : _type_, optional
        _description_, by default Path("./cgal_temp")
    delete_temp_file : bool, optional
        _description_, by default True
    shaft_threshold : _type_, optional
        _description_, by default None
    return_sdf : bool, optional
        _description_, by default True
    print_flag : bool, optional
        _description_, by default False
    plot_segmentation : bool, optional
        _description_, by default False
    plot_shaft : bool, optional
        _description_, by default False
    plot_shaft_buffer : int, optional
        _description_, by default 0

    Returns
    -------
    _type_
        _description_
    """
    if clusters is None:
        clusters = clusters_threshold_global
    
    if smoothness is None:
        smoothness = smoothness_threshold_global
        
    if shaft_threshold is None:
        shaft_threshold = shaft_threshold_global
    
    

    #print(f"plot_segmentation= {plot_segmentation}")
    cgal_data,cgal_sdf_data = tu.mesh_segmentation(current_mesh,
                                                  cgal_folder=cgal_folder,
                                                   clusters=clusters,
                                                   smoothness=smoothness,
                                                   return_sdf=True,
                                                   delete_temp_files=delete_temp_file,
                                                   return_meshes=False,
                                                   return_ordered_by_size=False,
                                                   plot_segmentation = plot_segmentation,
                                                  )
    
    #get a look at how many groups and what distribution:
    from collections import Counter
    if print_flag:
        print(f"Counter of data = {Counter(cgal_data)}")

    #gets the meshes that are split using the cgal labels
    split_meshes,split_meshes_idx = tu.split_mesh_into_face_groups(current_mesh,cgal_data,return_idx=True,
                                   check_connect_comp = False)
    
    
    
    split_meshes,split_meshes_idx
    
    
    if len(split_meshes.keys()) <= 1:
        print("There was only one mesh found from the spine process and mesh split, returning empty array")
        if return_sdf:
            return [],[],[],[],[]
        else:
            return [],[],[],[]
        
    
    
#     #Applying a length threshold to get all other possible shaft meshes
#     for spine_id,spine_mesh in split_meshes.items():
#         if len(spine_mesh.faces) < shaft_threshold:
#             spine_meshes.append(spine_mesh)
#             spine_meshes_idx.append(split_meshes_idx[spine_id])
#         else:
#             shaft_meshes.append(spine_mesh)
#             shaft_meshes_idx.append(split_meshes_idx[spine_id])

    meshes = np.array(list(split_meshes.values()))
    meshes_idx = np.array(list(split_meshes.keys()))
    
    #return meshes
    
    sh_idx = spu.restrict_meshes_to_shaft_meshes_without_coordinates(
        meshes,
        **kwargs
    )
    
    shaft_meshes = [meshes[k] for k in sh_idx]
    shaft_meshes_idx = [split_meshes_idx[meshes_idx[k]] for k in sh_idx]
    
    sp_idx = np.delete(np.arange(len(meshes)),sh_idx).astype('int')
    spine_meshes = [meshes[k] for k in sp_idx]
    spine_meshes_idx = [split_meshes_idx[meshes_idx[k]] for k in sp_idx]
    
 
    if len(shaft_meshes) == 0:
        if print_flag:
            print("No shaft meshes detected")
        if return_sdf:
            return [],[],[],[],[]
        else:
            return [],[],[],[]
 
    if len(spine_meshes) == 0:
        if print_flag:
            print("No spine meshes detected")
            
    if return_sdf:
        return spine_meshes,spine_meshes_idx,shaft_meshes,shaft_meshes_idx,cgal_sdf_data
    else:
        return spine_meshes,spine_meshes_idx,shaft_meshes,shaft_meshes_idx

    
def synapse_attribute_dict_from_synapse_df(
    df,
    attribute = "synapse_coords",
    suffix = None,
    verbose = False,):
    """
    Purpose: To extract the coordinates of 
    all the spine categories from a synapse_spine_df

    Psuedocode: 
    Iterate through all of the spine categories
    1) Restrict the dataframe to just categories
    2) Extract coordinates
    3) Put in dictionary
    """
    if suffix is None:
        suffix = f"_{attribute}"
    coordinate_dict = dict()
    for cat in spu.spine_compartments:
        curr_cat = cat.replace(f"spine_","")
        if len(df) > 0:
            curr_df = df.query(f"spine_compartment=='{curr_cat}'")
        else:
            curr_df = []
            
        if attribute == "synapse_coords":
            if len(curr_df) == 0:
                coords = np.array([]).reshape(-1,3)
            else:
                coords = pu.coordinates_from_df(curr_df,name = "synapse")
        elif attribute == "synapse_id":
            if len(curr_df) == 0:
                coords = np.array([]).reshape(-1)
            else:
                coords = curr_df["synapse_id"].to_numpy()
        else:
            raise Exception("")
        
        coordinate_dict[f"{cat}{suffix}"] = coords
    
    if verbose:
        for k,v in coordinate_dict.items():
            print(f"{k}:{len(v)}")
    
    return coordinate_dict


    
    



def spine_compartment_synapses(df,compartment):
    compartment= nu.to_list(compartment)
    return df.query(f"spine_compartment in {list(compartment)}")

def shaft_synapses(df):
    return spine_compartment_synapses(df,compartment="shaft")
def n_shaft_synapses(df):
    return len(shaft_synapses(df))



# --------- exporting the stats --------------


spine_features_no_head = (
        'spine_area',
        'spine_n_faces',
        'spine_skeletal_length',
        'spine_volume',
        'spine_width_ray',
        )

spine_features_head_neck = (
        'head_area',
        'head_n_faces',
        'head_skeletal_length',
        'head_volume',
        'head_width_ray',
        'head_width_ray_80_perc',
        
        'neck_area',
        'neck_n_faces',
        'neck_skeletal_length',
        'neck_volume',
        'neck_width_ray',
        'neck_width_ray_80_perc',
        )

spine_features_n_syn_no_head = (
        'spine_n_no_head_syn',
        'spine_max_no_head_syn_size',
        'spine_max_no_head_sp_vol',
)

spine_features_n_syn_head_neck = (
        'spine_n_head_syn',
        'spine_n_neck_syn',
        'spine_max_head_syn_size',
        'spine_max_neck_syn_size',
       'spine_max_head_sp_vol',
       'spine_max_neck_sp_vol',
)






def number_of_columns(df):
    return [k for k in df.columns if "_n_" in k or k[:2] == 'n_']


# ----------------- Parameters ------------------------

global_parameters_dict_default_spine_identification = dict(
    query="median_mesh_center > 115 and n_faces_branch>100",#previous used median_mesh_center > 140
    calculate_spine_volume=True,
    connectivity_type_for_shaft="vertices",
    clusters_threshold=5,#3,#2,
    smoothness_threshold=0.08,#0.12,#0.08,
    shaft_close_hole_area_top_2_mean_max = 110_000,
    shaft_mesh_volume_max = 0.3e9,
    shaft_mesh_n_faces_min = 10,
    shaft_threshold=300,
    
    # --- the new bare minimum thresholds ----
    spine_n_face_threshold_bare_min = 6,
    spine_sk_length_threshold_bare_min = 306.6,
    filter_by_volume_threshold_bare_min = 900496.186,
    bbox_oriented_side_max_min_bare_min = 300,
    spine_volume_to_spine_area_min_bare_min = 0.008,
    sdf_mean_min_bare_min = 0,

    spine_n_face_threshold=25,
    spine_sk_length_threshold = 1_000,
    filter_by_bounding_box_longest_side_length=True,
    side_length_threshold = 5000,
    filter_out_border_spines=False, #this seemed to cause a lot of misses
    skeleton_endpoint_nullification=True,
    skeleton_endpoint_nullification_distance = 2000,
    soma_vertex_nullification = True,
    border_percentage_threshold=0.3,
    check_spine_border_perc=0.4,

    #-------1/20 Addition --------
    filter_by_volume = True,
    filter_by_volume_threshold = 19_835_293, #calculated from experiments   
    
    # ----- 4/14/25 additions
    filter_by_face_area_mean = False,
    filter_by_face_area_mean_min = 0,

    filter_by_boundary_to_area_ratio = False,
    filter_by_boundary_to_area_ratio_min = 0,
    
)

global_parameters_dict_default_head_neck_shaft = dict(
    head_smoothness = 0.09,#0.15,
    head_ray_trace_min = 240,
    head_face_min = 10,
    only_allow_one_connected_component_neck = False,
    
)

global_parameters_dict_default = gu.merge_dicts([
    global_parameters_dict_default_spine_identification,
    global_parameters_dict_default_head_neck_shaft
    
])

attributes_dict_default = dict(
)    

global_parameters_dict_microns = {}
attributes_dict_microns = {}

global_parameters_dict_h01_spine_identification = dict(
    spine_n_face_threshold=15,
    spine_sk_length_threshold = 1_000,
    filter_by_volume_threshold = 50_000_000,#19835293, #calculated from experiments   
    
    clusters_threshold=5,
    smoothness_threshold=0.08,#0.12,#0.08,
    shaft_close_hole_area_top_2_mean_max = 260_000,
    shaft_mesh_volume_max = 0.6e9,
    shaft_mesh_n_faces_min = 10,
    
    # bare minimum filters 
    spine_volume_to_spine_area_min_bare_min = 0.01,
    spine_n_face_threshold_bare_min = 10,
    
)
global_parameters_dict_h01_head_neck_shaft = {}

global_parameters_dict_h01 = gu.merge_dicts([
    global_parameters_dict_h01_spine_identification,
    global_parameters_dict_h01_head_neck_shaft
    
])


attributes_dict_h01 = {}



# --- inlined from branch_attr_utils (wave 5.5) ---

def _calculate_endpoints_dist(branch_obj, attr_obj):
    attr_obj.endpoints_dist = [sk.skeleton_path_between_skeleton_coordinates(
        starting_coordinate=attr_obj.closest_sk_coordinate,
        destination_node=j,
        skeleton_graph=branch_obj.skeleton_graph,
        only_skeleton_distance=True,
    ) for j in branch_obj.endpoints_nodes]

def _calculate_upstream_downstream_dist_from_up_idx(attr_obj, up_idx):
    down_idx = 1 - up_idx
    attr_obj.downstream_dist = attr_obj.endpoints_dist[down_idx]
    attr_obj.upstream_dist = attr_obj.endpoints_dist[1 - down_idx]

def _branch_attr_soma_distances_on_limb(limb_obj, branch_attr, calculate_endpoints_dist_if_empty=True, verbose=False):
    bu.set_branches_endpoints_upstream_downstream_idx_on_limb(limb_obj)
    for branch_idx in limb_obj.get_branch_names():
        branch_obj = limb_obj[branch_idx]
        upstream_dist = nst.total_upstream_skeletal_length(limb_obj, branch_idx)
        upstream_endpoint_idx = branch_obj.endpoints_upstream_downstream_idx[0]
        curr_attr_list = getattr(branch_obj, branch_attr)
        if curr_attr_list is not None:
            for attr_obj in curr_attr_list:
                if attr_obj.endpoints_dist is None or attr_obj.endpoints_dist[upstream_endpoint_idx] == -1:
                    if calculate_endpoints_dist_if_empty:
                        _calculate_endpoints_dist(branch_obj, attr_obj)
                        bu.set_endpoints_upstream_downstream_idx_on_branch(limb_obj, branch_idx)
                        down_idx = branch_obj.endpoints_upstream_downstream_idx[1]
                        attr_obj.downstream_dist = attr_obj.endpoints_dist[down_idx]
                        attr_obj.upstream_dist = attr_obj.endpoints_dist[1 - down_idx]
                    else:
                        raise Exception("Endpoint distance was not calculated yet")
                _calculate_upstream_downstream_dist_from_up_idx(attr_obj, upstream_endpoint_idx)
                attr_obj.soma_distance = attr_obj.upstream_dist + upstream_dist
    return limb_obj

def _neuron_soma_distance_euclidean(neuron_obj, branch_attr, verbose=False):
    soma_center = neuron_obj["S0"].mesh_center
    for attr_obj in getattr(neuron_obj, branch_attr):
        attr_obj.soma_distance_euclidean = np.linalg.norm(soma_center - attr_obj.coordinate)

def _set_limb_branch_idx_to_attr(neuron_obj, branch_attr):
    for limb_idx in neuron_obj.get_limb_names(return_int=True):
        limb_obj = neuron_obj[limb_idx]
        for branch_idx in limb_obj.get_branch_names():
            branch_obj = limb_obj[branch_idx]
            attr_list = getattr(branch_obj, branch_attr)
            if attr_list is not None:
                for s in attr_list:
                    s.limb_idx = limb_idx
                    s.branch_idx = branch_idx

# ---

#--- from neurd_packages ---
from . import branch_utils as bu
from . import neuron_searching as ns
from . import neuron_statistics as nst
from . import neuron_utils as nru

from . import width_utils as wu

#--- from mesh_tools ---
from mesh_tools import skeleton_utils as sk
from mesh_tools import trimesh_utils as tu

#--- from datasci_tools ---
from datasci_tools import general_utils as gu
from datasci_tools import matplotlib_utils as mu
from datasci_tools import mesh_utils as meshu
from datasci_tools import networkx_utils as xu
from datasci_tools import numpy_dep as np
from datasci_tools import numpy_utils as nu
from datasci_tools import pandas_utils as pu
from datasci_tools import statistics_utils as stu
from datasci_tools import system_utils as su
from datasci_tools import tqdm_utils as tqu

from . import spine_utils as spu