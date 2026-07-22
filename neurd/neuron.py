
import networkx as nx
from typing import NamedTuple
from scipy.spatial import KDTree
import sys
import time
from datasci_tools import numpy_dep as np

from . import neuron_utils as nru
from . import neuron_statistics as nst

current_module = sys.modules[__name__]

branch_mesh_attributes = ["spines","boutons"]
object_attributes = ["synapses","spines_obj"]
computed_attribute_list = [
    "width_array",
    "width_array_skeletal_lengths",
    "width_new",
    "spines_volume",
    "boutons_volume",
    "labels","boutons_cdfs","web_cdf","web","head_neck_shaft_idx"] + branch_mesh_attributes + object_attributes

def convert_soma_to_piece_connectivity_to_graph(soma_to_piece_connectivity):
    """
    Pseudocode:
    1) Create the edges with the new names from the soma_to_piece_connectivity
    2) Create a GraphOrderedEdges from the new edges

    Ex:

    concept_network = convert_soma_to_piece_connectivity_to_graph(current_mesh_data[0]["soma_to_piece_connectivity"])
    nx.draw(concept_network,with_labels=True)
    """

    total_edges = []
    for soma_key,list_of_limbs in soma_to_piece_connectivity.items():
        total_edges += [[f"S{soma_key}",f"L{curr_limb}"] for curr_limb in list_of_limbs]

    print(f"total_edges = {total_edges}")
    concept_network = xu.GraphOrderedEdges()
    concept_network.add_edges_from(total_edges)
    return concept_network

class Branch:
    """
    Class that will hold one continus skeleton
    piece that has no branching
    """

    def __init__(self,
                skeleton,
                width=None,
                mesh=None,
                mesh_face_idx=None,
                 labels=[] #for any labels of that branch
        ):

        self.skeleton=skeleton.reshape(-1,2,3)
        self._skeleton_smooth = None
        self._skeleton_graph = None
        self._endpoints_nodes = None
        self.mesh=mesh
        self.width=width
        self.mesh_face_idx = mesh_face_idx
        self._mesh_volume = None
        self.endpoints_upstream_downstream_idx = None

        #calculate the end coordinates of skeleton branch
        self.calculate_endpoints()
        self.order_skeleton_by_smallest_endpoint()
        self.mesh_center = None
        if not self.mesh is None:
            self.mesh_center = tu.mesh_center_vertex_average(self.mesh)
        self.labels=labels
        if not nu.is_array_like(self.labels):
            self.labels=[self.labels]

        self.spines = None
        self.boutons = None
        self.spines_volume = None
        self.boutons_volume = None
        self.boutons_cdfs = None

        self.web = None
        self.web_cdf = None
        self.width_new = dict()
        self.width_array = dict()
        self.width_array_skeletal_lengths = None
        self.synapses = []
        self.spines_obj = []

        self.head_neck_shaft_idx = None

        self._skeleton_vector_upstream = None
        self._skeleton_vector_downstream = None
        self._width_downstream = None
        self._width_upstream = None

        self._skeleton_vector_upstream_extra_offset = None
        self._skeleton_vector_downstream_extra_offset = None
        self._width_downstream_extra_offset = None
        self._width_upstream_extra_offset = None

        self._skeleton_smooth_vector_downstream_extra_offset = None
        self._skeleton_smooth_vector_upstream_extra_offset = None
        self._skeleton_smooth_vector_downstream = None
        self._skeleton_smooth_vector_upstream = None

    def calculate_endpoints(self):
        self.endpoints = sk.find_branch_endpoints(self.skeleton)

    def order_skeleton_by_smallest_endpoint(self):
        self.skeleton = sk.order_skeleton(
                        self.skeleton,
        )

    @property
    def endpoint_upstream(self):
        return self.endpoints[self.endpoints_upstream_downstream_idx[0]]
    @property
    def endpoint_downstream(self):
        return self.endpoints[self.endpoints_upstream_downstream_idx[1]]

    @property
    def endpoint_upstream_with_offset(self):
        return bu.endpoint_upstream_with_offset(self)

    @property
    def endpoint_downstream_with_offset(self):
        return bu.endpoint_downstream_with_offset(self)

    @property
    def width_array_upstream_to_downstream(self):
        return bu.width_array_upstream_to_downstream(self,verbose = False)

    @property
    def width_array_skeletal_lengths_upstream_to_downstream(self):
        return bu.width_array_skeletal_lengths_upstream_to_downstream(self,verbose = False)

    @property
    def skeletal_coordinates_upstream_to_downstream(self):
        return bu.skeletal_coordinates_upstream_to_downstream(self,verbose = False)

    @property
    def skeletal_coordinates_dist_upstream_to_downstream(self):
        return bu.skeletal_coordinates_dist_upstream_to_downstream(self,verbose = False)

    @property
    def mesh_shaft(self):
        return bu.mesh_shaft(self)

    @property
    def mesh_shaft_idx(self):
        return bu.mesh_shaft_idx(self)

    @property
    def mesh_center_x(self):
        return self.mesh_center[0]
    @property
    def mesh_center_y(self):
        return self.mesh_center[1]
    @property
    def mesh_center_z(self):
        return self.mesh_center[2]

    @property
    def endpoint_upstream_x(self):
        return self.endpoint_upstream[0]
    @property
    def endpoint_upstream_y(self):
        return self.endpoint_upstream[1]
    @property
    def endpoint_upstream_z(self):
        return self.endpoint_upstream[2]

    @property
    def endpoint_downstream_x(self):
        return self.endpoint_downstream[0]
    @property
    def endpoint_downstream_y(self):
        return self.endpoint_downstream[1]
    @property
    def endpoint_downstream_z(self):
        return self.endpoint_downstream[2]

    @property
    def skeleton_vector_upstream(self):
        """
        the skelelton vector near upstream coordinate where the vector is oriented in the skeletal walk direction away from the soma
        """
        if self._skeleton_vector_upstream is None:
            self._skeleton_vector_upstream = bu.skeleton_vector_upstream(self)
        return self._skeleton_vector_upstream

    @property
    def skeleton_vector_downstream(self):
        """
        the skelelton vector near downstream coordinate where the vector is oriented in the skeletal walk direction away from the soma
        """

        if self._skeleton_vector_downstream is None:
            self._skeleton_vector_downstream = bu.skeleton_vector_downstream(self)
        return self._skeleton_vector_downstream

    @property
    def skeleton_vector_upstream_extra_offset(self):
        """
        the skelelton vector near upstream coordinate where the vector is oriented in the skeletal walk direction away from the soma
        """
        if self._skeleton_vector_upstream_extra_offset is None:
            self._skeleton_vector_upstream_extra_offset = bu.skeleton_vector_upstream_extra_offset(self)
        return self._skeleton_vector_upstream_extra_offset

    @property
    def skeleton_vector_downstream_extra_offset(self):
        """
        the skelelton vector near downstream coordinate where the vector is oriented in the skeletal walk direction away from the soma
        """

        if self._skeleton_vector_downstream_extra_offset is None:
            self._skeleton_vector_downstream_extra_offset = bu.skeleton_vector_downstream_extra_offset(self)
        return self._skeleton_vector_downstream_extra_offset

    @property
    def skeleton_smooth_vector_upstream(self):
        """
        the skelelton vector near upstream coordinate where the vector is oriented in the skeletal walk direction away from the soma
        """
        if self._skeleton_smooth_vector_upstream is None:
            self._skeleton_smooth_vector_upstream = bu.skeleton_vector_upstream(self,skeleton_attribute="skeleton_smooth")
        return self._skeleton_smooth_vector_upstream

    @property
    def skeleton_smooth_vector_downstream(self):
        """
        the skelelton vector near downstream coordinate where the vector is oriented in the skeletal walk direction away from the soma
        """

        if self._skeleton_smooth_vector_downstream is None:
            self._skeleton_smooth_vector_downstream = bu.skeleton_vector_downstream(self,skeleton_attribute="skeleton_smooth")
        return self._skeleton_smooth_vector_downstream

    @property
    def skeleton_smooth_vector_upstream_extra_offset(self):
        """
        the skelelton vector near upstream coordinate where the vector is oriented in the skeletal walk direction away from the soma
        """
        if self._skeleton_smooth_vector_upstream_extra_offset is None:
            self._skeleton_smooth_vector_upstream_extra_offset = bu.skeleton_vector_upstream_extra_offset(self,skeleton_attribute="skeleton_smooth")
        return self._skeleton_smooth_vector_upstream_extra_offset

    @property
    def skeleton_smooth_vector_downstream_extra_offset(self):
        """
        the skelelton vector near downstream coordinate where the vector is oriented in the skeletal walk direction away from the soma
        """

        if self._skeleton_smooth_vector_downstream_extra_offset is None:
            self._skeleton_smooth_vector_downstream_extra_offset = bu.skeleton_vector_downstream_extra_offset(self,skeleton_attribute="skeleton_smooth")
        return self._skeleton_smooth_vector_downstream_extra_offset

    @property
    def width_upstream(self):
        if self._width_upstream is None:
            self._width_upstream = bu.width_upstream(self)
        return self._width_upstream

    @property
    def width_downstream(self):
        if self._width_downstream is None:
            self._width_downstream = bu.width_downstream(self)
        return self._width_downstream

    @property
    def width_upstream_extra_offset(self):
        if self._width_upstream_extra_offset is None:
            self._width_upstream_extra_offset = bu.width_upstream_extra_offset(self)
        return self._width_upstream_extra_offset

    @property
    def width_downstream_extra_offset(self):
        if self._width_downstream_extra_offset is None:
            self._width_downstream_extra_offset = bu.width_downstream_extra_offset(self)
        return self._width_downstream_extra_offset

    @property
    def skeleton_smooth(self):
        if self._skeleton_smooth is None:
            self._skeleton_smooth = sk.smooth_skeleton_transform(
                self.skeleton,
                transform = "gaussian",
            )
        return self._skeleton_smooth

    @property
    def min_dist_synapses_pre_upstream(self):
        return bu.min_dist_synapses_pre_upstream(self)
    @property
    def min_dist_synapses_post_upstream(self):
        return bu.min_dist_synapses_post_upstream(self)

    @property
    def min_dist_synapses_pre_downstream(self):
        return bu.min_dist_synapses_pre_downstream(self)
    @property
    def min_dist_synapses_post_downstream(self):
        return bu.min_dist_synapses_post_downstream(self)

    @property
    def mesh_volume(self):
        if self._mesh_volume is None:
            try:
                self._mesh_volume = tu.mesh_volume(self.mesh)
            except:
                self._mesh_volume = 0
        divisor=1_000_000_000
        return self._mesh_volume/divisor

    @property
    def area(self):
        """
        dictionary mapping the index to the
        """
        divisor=1_000_000
        return self.mesh.area/divisor

    @property
    def skeletal_length(self):
        return sk.calculate_skeleton_distance(self.skeleton)

    @property
    def n_spines(self):
        return nru.n_spines(self)

    @property
    def n_boutons(self):
        return nru.n_boutons(self)

    @property
    def n_web(self):
        return nru.n_web(self)

    @property
    def spine_density(self):
        return nru.spine_density(self)

    @property
    def total_spine_volume(self):
        return nru.total_spine_volume(self)

    @property
    def spine_volume_median(self):
        return nru.spine_volume_median(self)

    @property
    def spine_volume_density(self):
        return nru.spine_volume_density(self)

    @property
    def skeletal_length_eligible(self):
        return sk.calculate_skeleton_distance(self.skeleton)

    def compute_spines_volume(self,
              max_hole_size=2000,
              self_itersect_faces=False):
        nru.compute_mesh_attribute_volume(self,
                                         mesh_attribute="spines",
                                         max_hole_size=max_hole_size,
                                         self_itersect_faces=self_itersect_faces)
    def compute_boutons_volume(self,
              max_hole_size=2000,
              self_itersect_faces=False):
        nru.compute_mesh_attribute_volume(self,
                                         mesh_attribute="boutons",
                                         max_hole_size=max_hole_size,
                                         self_itersect_faces=self_itersect_faces)

    # -------- 6/8 Addition For quick skeleton ------- #
    @property
    def skeleton_graph(self):
        if self._skeleton_graph is None:
            self._skeleton_graph = sk.convert_skeleton_to_graph(self.skeleton)
        return self._skeleton_graph

    @property
    def endpoints_nodes(self):
        if self._endpoints_nodes is None:
            self._endpoints_nodes = [xu.get_graph_node_by_coordinate(self.skeleton_graph,k)
                                    for k in self.endpoints]

        return self._endpoints_nodes

    @property
    def axon_compartment(self):
        if "axon" in self.labels:
            return "axon"
        else:
            return "dendrite"

def limb_mesh_from_branches(limb_obj,
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

    neuron_mesh_list = []

    for branch_obj in limb_obj:
        neuron_mesh_list.append(branch_obj.mesh)

    neuron_mesh_from_branches = tu.combine_meshes(neuron_mesh_list)
    return neuron_mesh_from_branches

class _StartingState(NamedTuple):
    """A limb's soma-attachment state, unpacked from its concept_network_dict.

    All fields are None (and concept_network None, all_concept_network_data []) when the
    dict is empty -- a limb that touches no soma.
    """
    concept_network: object
    all_concept_network_data: list
    current_starting_coordinate: object
    current_starting_node: object
    current_starting_endpoints: object
    current_starting_soma: object
    current_touching_soma_vertices: object
    current_soma_group_idx: object


def _load_starting_state(concept_network_dict):
    """Pick the limb's active concept network + starting attributes from its dict."""
    if len(concept_network_dict) == 0:
        return _StartingState(None, [], None, None, None, None, None, None)

    data = nru.get_starting_info_from_concept_network(concept_network_dict)
    current = data[0]
    return _StartingState(
        concept_network=concept_network_dict[current["starting_soma"]][current["soma_group_idx"]],
        all_concept_network_data=data,
        current_starting_coordinate=current["starting_coordinate"],
        current_starting_node=current["starting_node"],
        current_starting_endpoints=current["starting_endpoints"],
        current_starting_soma=current["starting_soma"],
        current_touching_soma_vertices=current["touching_soma_vertices"],
        current_soma_group_idx=current["soma_group_idx"],
    )


def _attach_branches(concept_network, curr_limb_correspondence):
    """Build a Branch per correspondence entry and store it as node "data".

    Returns suppress_disconnected_errors=True if any branch was not already a node of
    the concept network (it had to be added, so the graph is not fully connected).
    """
    suppress_disconnected_errors = False
    for j,branch_data in curr_limb_correspondence.items():
        branch_obj = Branch(
                            skeleton=branch_data["branch_skeleton"],
                            width=branch_data["width_from_skeleton"],
                            mesh=branch_data["branch_mesh"],
                           mesh_face_idx=branch_data["branch_face_idx"],
                            labels=[],
        )
        if j not in concept_network:
            concept_network.add_node(j)
            suppress_disconnected_errors = True
        xu.set_node_data(concept_network,
                        node_name=j,
                        curr_data=branch_obj,
                        curr_data_label="data")
    return suppress_disconnected_errors


#: the attributes that mark a node as the limb's soma-attachment point
_STARTING_ATTR_KEYS = ("starting_coordinate", "touching_soma_vertices",
                       "soma_group_idx", "starting_soma")


def _clear_starting_attrs(concept_network):
    """Remove the starting-point attributes from every node that currently carries them."""
    previous_starting_node = xu.get_starting_node(concept_network, only_one=False)
    if len(previous_starting_node) != 1:
        print(f"**** Warning there were {len(previous_starting_node)} starting nodes in concept "
              f"networks\nprevious_starting_node = {previous_starting_node}")
    for prev_st_node in previous_starting_node:
        for key in _STARTING_ATTR_KEYS:
            del concept_network.nodes[prev_st_node][key]
    return previous_starting_node


def _apply_starting_attrs(concept_network, match, starting_soma):
    """Write the starting-point attributes onto the node named by `match`.

    Note starting_soma is the caller's argument, not match["starting_soma"] -- the two
    can differ and the original wrote the argument here.
    """
    attrs = {match["starting_node"]: {
        "starting_coordinate": match["starting_coordinate"],
        "touching_soma_vertices": match["touching_soma_vertices"],
        "soma_group_idx": match["soma_group_idx"],
        "starting_soma": starting_soma}}
    xu.set_node_attributes_dict(concept_network, attrs)


class Limb:
    """
    Class that will hold one continus skeleton
    piece that has no branching (called a limb)

    3) Limb Process: For each limb made
    a. Build all the branches from the
        - mesh
        - skeleton
        - width
        - branch_face_idx
    b. Pick the top concept graph (will use to store the nodes)
    c. Put the branches as "data" in the network
    d. Get all of the starting coordinates and starting edges and put as member attributes in the limb
    """
    def __init__(self,
                             mesh,
                             curr_limb_correspondence=None,
                             concept_network_dict=None,
                             mesh_face_idx=None,
                            labels=None,
                             deleted_edges = None,
                             created_edges = None,
                            verbose=False):
        """Build a limb: its branches (as concept-network node data) and its
        soma-attachment starting state."""
        if labels is None:
            labels = []

        if deleted_edges is None:
            deleted_edges = []

        if created_edges is None:
            created_edges = []

        self.axon_spines =np.array([])
        self.short_thick_endnodes = np.array([])

        self._index = -1
        self.mesh=mesh

        #checking that some of arguments aren't None
        if curr_limb_correspondence is None:
            raise Exception("curr_limb_correspondence is None before start of Limb processing in init")
        if concept_network_dict is None:
            raise Exception("concept_network_dict is None before start of Limb processing in init")

        #make sure it is a list
        if not nu.is_array_like(labels):
            labels=[labels]

        self.labels=labels

        #All the stuff dealing with the concept graph
        for field, value in _load_starting_state(concept_network_dict)._asdict().items():
            setattr(self, field, value)

        self.mesh_face_idx = mesh_face_idx
        self.mesh_center = (tu.mesh_center_vertex_average(self.mesh)
                            if self.mesh is not None else None)

        # a. Build the branches and store each as its node's "data" in the concept network
        suppress_disconnected_errors = _attach_branches(self.concept_network, curr_limb_correspondence)

        self.deleted_edges =deleted_edges
        self.created_edges = created_edges

        if self.current_starting_coordinate is not None:

            self.set_concept_network_edges_from_current_starting_data()
            self.concept_network_directional = self.convert_concept_network_to_directional(no_cycles = True,
                                                                                suppress_disconnected_errors=suppress_disconnected_errors)
        else:
            self.concept_network_directional = None

    @property
    def mesh_from_branches(self):
        return limb_mesh_from_branches(self)

    @property
    def current_starting_soma_vertices(self):
        return self.current_touching_soma_vertices

    def set_branches_endpoints_upstream_downstream_idx(self):
        bu.set_branches_endpoints_upstream_downstream_idx_on_limb(self)

    @property
    def mesh_volume(self):
        return nru.sum_feature_over_branches(self,
                             self.get_branch_names(),
                             "mesh_volume")

    @property
    def branches(self):
        return [k for k in self]

    @property
    def area(self):
        """
        dictionary mapping the index to the
        """
        if len(self.branch_objects) > 0:
            return np.sum([k.area for k in self])
        else:
            return 0

    @property
    def skeletal_length(self):
        """
        dictionary mapping the index to the
        """
        if len(self.branch_objects) > 0:
            return np.sum([k.skeletal_length for k in self])
        else:
            return 0

    @property
    def branch_objects(self):
        """
        dictionary mapping the index to the
        """
        return dict([(i,k) for i,k in enumerate(self)])

    @property
    def n_branches(self):
        """
        number of branches in the limb
        """
        return len(self)

    @property
    def network_starting_info(self):
        """
        Purpose: will generate the dictionary that is organized
        soma_idx --> soma_group_idx --> dict(touching_verts,endpoint)

        that can be used to generate a concept network from

        """
        st_dict = dict()
        for st in self.all_concept_network_data:
            soma_idx = st["starting_soma"]
            soma_group_idx = st["soma_group_idx"]
            if st["starting_soma"] not in st_dict.keys():
                st_dict[soma_idx] = dict()

            st_dict[soma_idx][soma_group_idx] = dict(touching_verts=st["touching_soma_vertices"],
                                                    endpoint=st["starting_coordinate"])
        return st_dict

    @property
    def all_starting_coordinates(self):
        """
        Purpose: will generate the dictionary that is organized
        soma_idx --> soma_group_idx --> dict(touching_verts,endpoint)

        that can be used to generate a concept network from

        """
        start_coordinates = []
        for st in self.all_concept_network_data:
            start_coordinates.append(st["starting_coordinate"])

        start_coordinates = np.array(start_coordinates).reshape(-1,3)
        return start_coordinates

    @property
    def all_starting_nodes(self):
        """
        Purpose: will generate the dictionary that is organized
        soma_idx --> soma_group_idx --> dict(touching_verts,endpoint)

        that can be used to generate a concept network from

        """
        starting_nodes = []
        for st in self.all_concept_network_data:
            starting_nodes.append(st["starting_node"])

        return starting_nodes

    @property
    def limb_correspondence(self):
        self._index = -1
        limb_corr = dict()
        for idx in self.get_branch_names():
            b = self[idx]
            curr_width = b.width
            try:
                curr_width = b.width_new["median_mesh_center"]
            except:
                pass

            limb_corr[idx] = dict(branch_skeleton=b.skeleton,
                                  width_from_skeleton = curr_width,
                                 branch_mesh = b.mesh,
                                 branch_face_idx = b.mesh_face_idx,
                                 )

        return limb_corr

    @property
    def divided_skeletons(self):
        curr_corr = self.limb_correspondence
        return np.array([curr_corr[k]["branch_skeleton"] for k in np.sort(list(curr_corr.keys()))])

    @property
    def n_spines(self):
        return nru.n_spines(self)

    @property
    def n_boutons(self):
        return nru.n_boutons(self)

    @property
    def n_web(self):
        return nru.n_web(self)

    @property
    def spines(self):
        return nru.feature_list_over_object(self,"spines")

    @property
    def synapses(self):
        return nru.feature_list_over_object(self,"synapses")

    @property
    def spines_obj(self):
        return nru.feature_list_over_object(self,"spines_obj")

    @property
    def boutons(self):
        return nru.feature_list_over_object(self,"boutons")

    @property
    def web(self):
        return nru.feature_list_over_object(self,"web")

    @property
    def spines_volume(self):
        return list(nru.feature_list_over_object(self,"spines_volume"))
    @property
    def boutons_volume(self):
        return nru.feature_list_over_object(self,"boutons_volume")

    def compute_spines_volume(self):
        nru.compute_feature_over_object(self,"spines_volume")

    def compute_boutons_volume(self):
        nru.compute_feature_over_object(self,"boutons_volume")

    def get_branch_names(self,ordered=True,return_int=True):
        node_names = np.sort(list(self.concept_network.nodes()))
        if return_int:
            return node_names.astype("int")
        else:
            return node_names

    def get_skeleton(self,check_connected_component=True):
        """
        Purpose: Will return the entire skeleton of all the branches
        stitched together

        """
        return nru.convert_limb_concept_network_to_neuron_skeleton(self.concept_network,
                             check_connected_component=check_connected_component)

    @property
    def skeleton(self,check_connected_component=False):
        """
        Purpose: Will return the entire skeleton of all the branches
        stitched together

        """
        return nru.convert_limb_concept_network_to_neuron_skeleton(self.concept_network,
                             check_connected_component=check_connected_component)

    @property
    def skeleton_smooth(self):
        return sk.stack_skeletons([self[k].skeleton_smooth for k in self.get_branch_names()])

    def get_concept_network_data_by_soma(self,soma_idx=None):
        #compile a dictionary of all of the starting material
        return_dict = dict()
        for curr_data in self.all_concept_network_data:
            return_dict[curr_data["starting_soma"]] = dict([(k,v) for k,v in curr_data.items() if k != "starting_soma"])
        if not soma_idx is None:
            return return_dict[soma_idx]
        else:
            return return_dict

    def get_concept_network_data_by_soma_and_idx(self,soma_idx,soma_group_idx):
        return_dict = []
        for curr_data in self.all_concept_network_data:
            if curr_data["soma_group_idx"] == soma_group_idx and curr_data["starting_soma"] == soma_idx:
                return_dict.append(curr_data)

        if len(return_dict) != 1:
            raise Exception(f"Did not find exactly one starting dictionary for soma_idx {soma_idx}, soma_group_idx {soma_group_idx}: {len(return_dict)} ")
        else:
            return return_dict[0]

    @property
    def concept_network_data_by_soma(self):
        #compile a dictionary of all of the starting material
        return_dict = dict()
        for curr_data in self.all_concept_network_data:
            return_dict[curr_data["starting_soma"]] = dict([(k,v) for k,v in curr_data.items() if k != "starting_soma"])
        return return_dict

    @property
    def concept_network_data_by_starting_node(self):
        #compile a dictionary of all of the starting material
        return_dict = dict()
        for curr_data in self.all_concept_network_data:
            return_dict[curr_data["starting_node"]] = dict([(k,v) for k,v in curr_data.items() if k != "starting_node"])
        return return_dict

    def touching_somas(self):
        """
        The soma identifiers that a current limb is adjacent two (useful for finding paths to cut for multi-soma or multi-touch limbs
        """
        return [k["starting_soma"] for k in self.all_concept_network_data if k["starting_soma"] >= 0]

    def get_skeleton_soma_starting_node(self,soma,print_flag=False):
        """
        Purpose: from the all

        """
        if type(soma) == str:
            soma = int(soma[1:])

        limb_starting_coordinate = self.concept_network_data_by_soma[soma]["starting_coordinate"]

        if print_flag:
            print(f"limb_starting_coordinate = {limb_starting_coordinate}")
        limb_skeleton_graph = sk.convert_skeleton_to_graph(self.skeleton)

        sk_starting_node = xu.get_nodes_with_attributes_dict(limb_skeleton_graph,
                                      attribute_dict=dict(coordinates=limb_starting_coordinate))
        if len(sk_starting_node) != 1:
            raise Exception(f"Not exactly one skeleton starting node: sk_starting_node = {sk_starting_node}")
        return sk_starting_node[0]

    def get_starting_branch_by_soma(self,soma,print_flag=False):
        """
        Purpose: from the all

        """
        if type(soma) == str:
            soma = int(soma[1:])

        return self.concept_network_data_by_soma[soma]["starting_node"]

    def get_soma_by_starting_node(self,starting_node,print_flag=False):
        """
        Purpose: from the all

        """

        return self.concept_network_data_by_starting_node[starting_node]["starting_soma"]

    def get_soma_group_by_starting_node(self,starting_node,print_flag=False):
        """
        Purpose: from the all

        """

        return self.concept_network_data_by_starting_node[starting_node]["soma_group_idx"]

    def find_branch_by_skeleton_coordinate(self,target_coordinate):

        """
        Purpose: To be able to find the branch where the skeleton point resides

        Pseudocode:
        For each branch:
        1) get the skeleton
        2) ravel the skeleton into a numpy array
        3) searh for that coordinate:
        - if returns a non empty list then add to list

        """
        matching_node = []
        for n in self.concept_network.nodes():
            curr_skeleton_points = self.concept_network.nodes[n]["data"].skeleton.reshape(-1,3)
            row_matches = nu.matching_rows(curr_skeleton_points,target_coordinate)
            if len(row_matches) > 0:
                matching_node.append(n)

        if len(matching_node) > 1:
            print(f"***Warning More than one branch skeleton matches the desired corrdinate: {matching_node}")
        elif len(matching_node) == 1:
            matching_node = matching_node[0]
        else:
            raise Exception("No matching branches found")

        return matching_node

    def convert_concept_network_to_directional(self,no_cycles = True,width_source=None,print_flag=False,
                                              suppress_disconnected_errors=False,
                                              convert_concept_network_to_directional_verbose = False):
        """Directional version of this limb's concept network, rooted at its starting node.

        Resolves a per-branch width (see width_source) and delegates to
        nru.convert_concept_network_to_directional.
        """
        if self.concept_network is None:
            raise Exception("Cannot use convert_concept_nextwork_to_directional on limb if concept_network is None")

        curr_limb_concept_network = self.concept_network

        #make sure that there is one and only one starting node embedded in the graph
        try:
            xu.get_starting_node(curr_limb_concept_network)
        except:
            print("There was not exactly one starting nodes in the current self.concept_network"
                  " when trying to convert to concept network ")
            xu.get_starting_node(curr_limb_concept_network)

        if width_source is None:
            #check to see if the mesh center width is available
            try:
                if "no_spine_average_mesh_center" in curr_limb_concept_network.nodes[0]["data"].width_new.keys():
                    width_source = "no_spine_average_mesh_center"
                else:
                    width_source = "width"
            except:
                width_source = "width"

        if print_flag:
            print(f"width_source = {width_source}, type = {type(width_source)}")

        if width_source == "width":
            if print_flag:
                print("Using the default width")
            node_widths = dict([(k,curr_limb_concept_network.nodes[k]["data"].width) for k in curr_limb_concept_network.nodes() ])
        else:
            if print_flag:
                print(f"Using the {width_source} in width_new that was calculated")
            node_widths = dict([(k,curr_limb_concept_network.nodes[k]["data"].width_new[width_source]) for k in curr_limb_concept_network.nodes() ])

        if print_flag:
            print(f"node_widths= {node_widths}")

        directional_concept_network = nru.convert_concept_network_to_directional(
            curr_limb_concept_network,
            node_widths=node_widths,
            no_cycles=no_cycles,
            suppress_disconnected_errors =suppress_disconnected_errors,
            verbose = convert_concept_network_to_directional_verbose)

        return directional_concept_network

    def set_concept_network_directional(self,starting_soma=None,soma_group_idx=0,starting_node=None,print_flag=False,
                                       suppress_disconnected_errors=False,no_cycles = True,
                                        convert_concept_network_to_directional_verbose = False,**kwargs):
        """
        Re-root this limb's concept network on a different starting soma/node.

        1) resolve the target soma + group from the args
        2) clear the old starting attributes off every node
        3) write the new starting attributes onto the matching node
        4) sync the self.current_* attributes and rebuild the directional network
        """
        # 1) resolve which soma / group we are re-rooting on
        if not starting_node is None:
            soma_group_idx = self.get_soma_group_by_starting_node(starting_node)
            starting_soma = self.get_soma_by_starting_node(starting_node)

        if soma_group_idx is None:
            soma_group_idx = 0
        else:
            soma_group_idx = nru.get_soma_int_name(soma_group_idx)

        if soma_group_idx == -1:
            soma_group_idx = self.current_soma_group_idx

        match = nru.get_matching_concept_network_data(self, soma_idx=starting_soma,
                                                      soma_group_idx=soma_group_idx,
                                                      starting_node=starting_node,
                                                      verbose=False)[0]

        # 2) move the starting attributes from the old node(s) onto the matching one
        _clear_starting_attrs(self.concept_network)
        _apply_starting_attrs(self.concept_network, match, starting_soma)
        xu.get_starting_node(self.concept_network)  # asserts exactly one starting node

        # 3) sync self.current_* from the match (starting_soma here is the match's, not the arg)
        self.current_starting_coordinate = match["starting_coordinate"]
        self.current_starting_node = match["starting_node"]
        self.current_starting_endpoints = match["starting_endpoints"]
        self.current_starting_soma = match["starting_soma"]
        self.current_touching_soma_vertices = match["touching_soma_vertices"]
        self.current_soma_group_idx = match["soma_group_idx"]

        # 4) rebuild the edges + directional network from the new starting info
        self.set_concept_network_edges_from_current_starting_data()

        if print_flag or convert_concept_network_to_directional_verbose:
            self.concept_network_directional = self.convert_concept_network_to_directional(no_cycles = no_cycles,print_flag=print_flag,
                                                                                           suppress_disconnected_errors=suppress_disconnected_errors,
                                                                                           convert_concept_network_to_directional_verbose=convert_concept_network_to_directional_verbose,
                                                                                           **kwargs)
        else:
            with su.suppress_stdout_stderr():
                self.concept_network_directional = self.convert_concept_network_to_directional(no_cycles = no_cycles,print_flag=print_flag,
                                                                                               suppress_disconnected_errors=suppress_disconnected_errors,
                                                                                               **kwargs)

    def set_concept_network_edges_from_current_starting_data(self,verbose=False):
        new_concept_network = nru.branches_to_concept_network(curr_branch_skeletons= self.divided_skeletons,
                                                                  starting_coordinate=self.current_starting_coordinate,
                                                                  starting_edge=self.current_starting_endpoints,
                                                                  touching_soma_vertices=self.current_touching_soma_vertices,
                                                                       soma_group_idx=self.current_soma_group_idx,
                                                                       verbose=False)

        self.concept_network.remove_edges_from(list(self.concept_network.edges()))
        self.concept_network.add_edges_from(list(new_concept_network.edges()))
        self.concept_network.remove_edges_from(self.deleted_edges)
        self.concept_network.add_edges_from(self.created_edges)

    # ----------------- 9/2 To help with compression ------------------------- #
    def get_attribute_dict(self,attribute_name):
        attribute_dict = dict()
        for branch_idx in self.get_branch_names():
            curr_branch = self[branch_idx]

            if hasattr(curr_branch,attribute_name):
                if attribute_name in branch_mesh_attributes:
                    if not getattr(curr_branch,attribute_name) is None:
                        curr_attr = getattr(curr_branch,attribute_name)
                        if nu.is_array_like(curr_attr):
                            attribute_dict[branch_idx] = [tu.original_mesh_faces_map(curr_branch.mesh,k) for k in curr_attr]
                        else:
                            attribute_dict[branch_idx] =tu.original_mesh_faces_map(curr_branch.mesh,curr_attr)
                    else:
                        attribute_dict[branch_idx] = None
                elif attribute_name in object_attributes:
                    curr_attr = getattr(curr_branch,attribute_name)
                    if curr_attr is not None:
                        if nu.is_array_like(curr_attr):
                            attribute_dict[branch_idx] = [k.export() for k in curr_attr]
                        else:
                            attribute_dict[branch_idx] = curr_attr.export()
                    else:
                        attribute_dict[branch_idx] = None
                else:
                    att_val = getattr(curr_branch,attribute_name)
                    if attribute_name == "spines_volume" and att_val is not None:
                        att_val  = list(att_val)
                    try:
                        attribute_dict[branch_idx] = getattr(curr_branch,attribute_name)
                    except:
                        attribute_dict[branch_idx] = None
            else:
                attribute_dict[branch_idx] = None

        return attribute_dict

    def set_attribute_dict(self,attribute_name,attribute_dict,verbose=False):
        for branch_idx,curr_branch in enumerate(self):
            if branch_idx in list(attribute_dict.keys()):
                if attribute_name in branch_mesh_attributes:
                    if not attribute_dict[branch_idx] is None:
                        try:
                            setattr(curr_branch,attribute_name,
                                    [curr_branch.mesh.submesh([k],append=True,repair=False) for k in attribute_dict[branch_idx]])
                        except:
                            setattr(curr_branch,attribute_name,
                                    curr_branch.mesh.submesh([attribute_dict[branch_idx]],append=True,repair=False))
                    else:
                        setattr(curr_branch,attribute_name,None)
                elif attribute_name in object_attributes:
                    setattr(curr_branch,attribute_name,None)
                else:
                    att_val = attribute_dict[branch_idx]
                    if attribute_name == "spines_volume" and att_val is not None:
                        att_val  = list(att_val)
                    setattr(curr_branch,attribute_name,att_val)
            else:
                if verbose:
                    print(f"Skipping attributes for Branch {branch_idx} because not in dictionary")

    def set_computed_attribute_data(self,computed_attribute_data,print_flag=False):
        start_time = time.time()

        if computed_attribute_data is None:
            return

        for k,v in computed_attribute_data.items():
            self.set_attribute_dict(k,v)

    def get_computed_attribute_data(self,
                                    attributes = computed_attribute_list,
                                    one_dict=True,
                                    print_flag=False):
        start_time = time.time()

        lookup_values = []
        lookup_dict = dict()
        for a in attributes:
            current_lookup_value = self.get_attribute_dict(a)
            lookup_values.append(current_lookup_value)
            if one_dict:
                lookup_dict[a] = current_lookup_value

        if print_flag:
            print(f"Total time for spine/bouton/width compression = {time.time() - start_time}")

        if one_dict:
            return lookup_dict
        else:
            return lookup_values

    # Defining some useful built in functions
    def __getitem__(self,key):
        return self.concept_network.nodes[key]["data"]
    def __setitem__(self,key,newvalue):
        self.concept_network.nodes[key]["data"] = newvalue
    def __len__(self):
        return len(list(self.concept_network.nodes()))
    def __iter__(self):
        return self
    def __next__(self):
        self._index += 1
        sorted_node_indexes = np.sort(list(self.concept_network.nodes()))
        if self._index >= len(self):
            self._index = -1
            raise StopIteration
        else:
            return self[sorted_node_indexes[self._index]]

    # ---------- 6/29 To help with navigating the concept network segments
    @property
    def nodes_to_exclude(self):
        return np.concatenate([self.axon_spines,self.short_thick_endnodes])

class Soma:
    """
    Class that will hold one continus skeleton
    piece that has no branching

    Properties that are housed:
     'mesh',
     'mesh_center',
     'mesh_face_idx',
     'sdf',
     'side_length_ratios',
     'volume_ratio'

    """

    def __init__(self,mesh,mesh_face_idx=None,sdf=None,volume_ratio=None,
                volume=None,synapses=None):
        #Accounting for the fact that could recieve soma object

        self.mesh=mesh
        self.sdf=sdf
        self.mesh_face_idx = mesh_face_idx
        if volume_ratio is None:
            self.volume_ratio = sm.soma_volume_ratio(self.mesh,
                                                    )
        else:
            print("Using precomputed volume ratio")
            self.volume_ratio = volume_ratio
        self.side_length_ratios = sm.side_length_ratios(self.mesh)
        self.mesh_center = tu.mesh_center_vertex_average(self.mesh)
        self._volume = volume

        if synapses is None:
            synapses = []

        self.synapses = synapses

        if volume is None:
            self.volume

    @property
    def compartment(self):
        return "soma"

    @property
    def area(self):
        return self.mesh.area/1_000_000

    @property
    def volume(self,watertight_method="poisson",divisor = 1_000_000_000):
        """
        Will compute the volume of the soma
        """
        if self._volume is None:
            self._volume = tu.mesh_volume(self.mesh,
                                         watertight_method=watertight_method)

        return self._volume/divisor

    @property
    def mesh_volume(self,**kwargs):
        return self.volume

    #     Old defaul paramters   def __init__(
    #     self,
    #     mesh,

    #     ):

def _soma_piece_graph(soma_to_piece_connectivity):
    """The networkx graph that holds most of a Neuron's data, from soma connectivity."""
    if isinstance(soma_to_piece_connectivity, nx.Graph):
        return soma_to_piece_connectivity
    if isinstance(soma_to_piece_connectivity, dict):
        return convert_soma_to_piece_connectivity_to_graph(soma_to_piece_connectivity)
    raise Exception(f"Recieved an incompatible type of {type(soma_to_piece_connectivity)} for the concept_network")


def _attach_somas(concept_network, preprocessed_data, soma_volume_ratios):
    """Create the Soma objects and hang them off the "S<j>" nodes as ["data"]."""
    soma_meshes = preprocessed_data["soma_meshes"]
    # soma_meshes_face_idx is supplied by preprocess_neuron since 63d2584 -- the
    # geometric KDTree fallback that used to regenerate it is gone
    for j,(curr_soma,curr_soma_face_idx,current_sdf,curr_volume_ratio) in enumerate(zip(
            soma_meshes,
            preprocessed_data["soma_meshes_face_idx"],
            preprocessed_data["soma_sdfs"],
            soma_volume_ratios)):
        soma_obj = Soma(curr_soma,
                        mesh_face_idx=curr_soma_face_idx,
                        sdf=current_sdf,
                        volume_ratio=curr_volume_ratio,
                        volume=None)

        soma_name = f"S{j}"
        # a soma is not necessarily already a node of the connectivity graph
        if soma_name in concept_network.nodes():
            xu.set_node_data(curr_network=concept_network,
                             node_name=soma_name,
                             curr_data=soma_obj,
                             curr_data_label="data")
        else:
            print(f"Did not have {soma_name} in concept network so adding it")
            concept_network.add_node(soma_name,data=soma_obj)


def _attach_limbs(concept_network, preprocessed_data):
    """Create the Limb objects and hang them off the "L<j>" nodes as ["data"].

    Each limb takes its correspondence and concept networks by index, so the four
    preprocessed lists stay aligned.
    """
    limb_correspondence = preprocessed_data["limb_correspondence"]
    limb_concept_networks = preprocessed_data["limb_concept_networks"]

    for j,(curr_limb_mesh,curr_limb_mesh_face_idx) in enumerate(zip(
            preprocessed_data["limb_meshes"],
            preprocessed_data["limb_mehses_face_idx"])):
        limb_obj = Limb(
                            mesh=curr_limb_mesh,
                            curr_limb_correspondence=limb_correspondence[j],
                            concept_network_dict=limb_concept_networks[j],
                            mesh_face_idx=curr_limb_mesh_face_idx,
                        )

        limb_name = f"L{j}"
        if limb_name not in concept_network.nodes():
            concept_network.add_node(limb_name)

        xu.set_node_data(curr_network=concept_network,
                         node_name=limb_name,
                         curr_data=limb_obj,
                         curr_data_label="data")


def _refine_neuron(neuron_obj, calculate_spines=False):
    """Everything after the objects exist: correspondence, spines, widths, cleanup."""
    print(f"--- 5) Doing the adaptive mesh correspondence on the meshparty preprocessing ---")
    nru.apply_adaptive_mesh_correspondence_to_neuron(neuron_obj)

    print(f"self.n_limbs = {neuron_obj.n_limbs}")
    if neuron_obj.n_limbs > 0:
        if calculate_spines and not neuron_obj.spines_already_computed():
            print("7a) calculating spines because didn't exist")
            spu.calculate_spines_on_neuron(neuron_obj)

        for w in ["no_spine_median_mesh_center"]:
            print(w)
            wu.calculate_new_width_for_neuron_obj(neuron_obj,width_name=w)

        nru.clean_neuron_all_concept_network_data(neuron_obj)
    else:
        print("Skipping the width and spine calculation because no limbs")

    nru.recalculate_endpoints_and_order_skeletons_over_neuron(neuron_obj)

    try:
        bu.set_branches_endpoints_upstream_downstream_idx(neuron_obj)
    except:
        pass


class Neuron:
    """
    Neuron class docstring:
    Will

    Purpose:
    An object oriented approach to housing the data
    about a single neuron mesh and the secondary
    data that can be gleamed from this. For instance
    - skeleton
    - compartment labels
    - soma centers
    - subdivided mesh into cable pieces

    Pseudocode:

    1) Create Neuron Object (through __init__)
    a. Add the small non_soma_list_meshes
    b. Add whole mesh
    c. Add soma_to_piece_connectivity as concept graph and it will be turned into a concept map

    2) Creat the soma meshes
    a. Create soma mesh objects
    b. Add the soma objects as ["data"] attribute of all of the soma nodes

    3) Limb Process: For each limb (use an index to iterate through limb_correspondence,current_mesh_data and limb_concept_network/lables)
    a. Build all the branches from the
        - mesh
        - skeleton
        - width
        - branch_face_idx
    b. Pick the top concept graph (will use to store the nodes)
    c. Put the branches as "data" in the network
    d. Get all of the starting coordinates and starting edges and put as member attributes in the limb

    """
    def __init__(self,mesh,segment_id = None, calculate_spines = False):
        """Decompose `mesh` into somas and limbs of branches.

        The heavy lifting is `preprocess_neuron`; this turns its output into the
        object graph (`concept_network`, with Soma/Limb objects as node "data") and
        then refines it.
        """
        neuron_start_time = time.time()

        self._index = -1          # in order to become an iterable
        self.mesh = mesh
        self._mesh_kdtree = None

        if segment_id is None:
            segment_id = np.random.randint(100000000)
            print(f"picking a random 7 digit segment id: {segment_id}")
        self.segment_id = segment_id

        print("--- 0) Having to preprocess the Neuron becuase no preprocessed data\nPlease wait this could take a while.....")
        from neurd import preprocess_neuron as pre  # P3: local import breaks neuron<->preprocess_neuron cycle
        preprocessed_data = pre.preprocess_neuron(mesh,segment_id=segment_id)

        self.nucleus_id = preprocessed_data.get("nucleus_id",None)
        self.split_index = preprocessed_data.get("split_index",None)
        self.preprocessed_data = preprocessed_data

        # preprocess_neuron does not return soma_volume_ratios, so they are always computed here
        soma_volume_ratios = [sm.soma_volume_ratio(j) for j in preprocessed_data["soma_meshes"]]
        preprocessed_data["soma_volume_ratios"] = soma_volume_ratios
        print(f"--- 1) Finished unpacking preprocessed materials: {time.time() - neuron_start_time}")

        self.concept_network = _soma_piece_graph(preprocessed_data["soma_to_piece_connectivity"])
        _attach_somas(self.concept_network, preprocessed_data, soma_volume_ratios)
        _attach_limbs(self.concept_network, preprocessed_data)
        print(f"--- 4) Finshed generating Limb objects and adding them to concept graph: {time.time() - neuron_start_time}")

        # Drop large preprocessed_data keys that are now fully materialised into
        # Limb/Soma objects inside concept_network. Kept: limb_correspondence and
        # limb_network_stating_info (mutated by neuron_utils after construction).
        for _stale_key in ("limb_meshes", "limb_concept_networks", "soma_meshes"):
            preprocessed_data.pop(_stale_key, None)

        _refine_neuron(self, calculate_spines=calculate_spines)
        self._clear_mesh_caches()

    def _clear_mesh_caches(self):
        """Clear trimesh lazy-property caches on the full mesh, every Limb mesh, and every
        Branch mesh, at the end of construction.

        trimesh populates expensive derived arrays on first access — vertex_adjacency_graph,
        vertex_faces, triangles, edges, face_adjacency. On the big H01 neuron these dominate
        RAM: the full mesh's caches are ~1.6 GB and the 153 branch meshes together hold ~1.4 GB
        of cache (vs only ~0.1 GB of raw vertices+faces — a ~15x blow-up). The cached arrays are
        not needed once construction is done and recompute on demand if any later code path asks.

        Branches WERE previously left alone ("small, accessed frequently"); that holds per-branch
        but not in aggregate on a many-branch neuron, and the segmentation deliverable
        (process_all_neurons) saves branch mesh+skeleton to disk then drops the Neuron, so nothing
        re-reads the caches. Clearing them frees ~0.5-1 GB off the retained object. Any downstream
        analysis that does touch a branch mesh transparently repopulates that one branch's cache.
        """
        def _clear(m):
            c = getattr(m, "_cache", None)
            if c is not None:
                c.clear()

        if hasattr(self, "mesh"):
            _clear(self.mesh)
        for limb_name in self.get_limb_node_names() if hasattr(self, "concept_network") else []:
            limb = self.concept_network.nodes[limb_name].get("data")
            if limb is None:
                continue
            _clear(getattr(limb, "mesh", None))
            for branch_name in (limb.concept_network.nodes() if hasattr(limb, "concept_network") else []):
                branch = limb.concept_network.nodes[branch_name].get("data")
                if branch is not None:
                    _clear(getattr(branch, "mesh", None))

    @property
    def limbs(self):
        return [k for k in self]

    def __getattr__(self,k):
        if k[:2] == "__":
            raise AttributeError(k)
        if hasattr(self,"pipeline_products"):
            return getattr(self.pipeline_products,k)
        else:
            return self.__getattribute__(k)

    def calculate_decomposition_products(
        self,
        store_in_obj = False,):
        return nru.calculate_decomposition_products(
            self,
            store_in_obj = store_in_obj,
        )

    def get_total_n_branches(self):
        return np.sum([len(self.concept_network.nodes[li]["data"].concept_network.nodes()) for li in self.get_limb_node_names()])

    def get_skeleton(self,check_connected_component=True):
        return nru.get_whole_neuron_skeleton(self,
                                 check_connected_component=check_connected_component)

    def get_attribute_dict(self,attribute_name):
        attribute_dict = dict()
        for limb_idx in self.get_limb_node_names(return_int=True):
            curr_limb = self[limb_idx]
            attribute_dict[limb_idx] = curr_limb.get_attribute_dict(attribute_name)

        return attribute_dict

    def set_attribute_dict(self,attribute_name,attribute_dict):
        for limb_idx,curr_limb in enumerate(self):
            if limb_idx in list(attribute_dict.keys()):
                curr_limb.set_attribute_dict(attribute_name,attribute_dict[limb_idx])
            else:
                pass

    def set_computed_attribute_data(self,computed_attribute_data,print_flag=False):
        start_time = time.time()

        if computed_attribute_data is None:
            return

        for k,v in computed_attribute_data.items():
            self.set_attribute_dict(k,v)

        if print_flag:
            print(f"Total time for spine/width compression = {time.time() - start_time}")

    def get_computed_attribute_data(self,
                                    attributes = computed_attribute_list,
                                    one_dict=True,
                                    print_flag=False):
        start_time = time.time()

        lookup_values = []
        lookup_dict = dict()
        for a in attributes:
            current_lookup_value = self.get_attribute_dict(a)
            lookup_values.append(current_lookup_value)
            if one_dict:
                lookup_dict[a] = current_lookup_value

        if print_flag:
            print(f"Total time for spine/width compression = {time.time() - start_time}")

        if one_dict:
            return lookup_dict
        else:
            return lookup_values

    def calculate_new_width(self,**kwargs):
        wu.calculate_new_width_for_neuron_obj(self,**kwargs)

    # ------------ 9/24: Function that will see if spines are already comuted ------------ #
    def spines_already_computed(self):
        """
        Pseudocode:
        1) Iterate through all of limbs and branches
        2) If find one instance where spines not None, return True
        3) If none found, return False

        """
        found_spines=False

        self._index = -1

        for limb in self:
            limb._index = -1
            if found_spines == True:
                break
            for branch in limb:
                if not branch.spines is None:
                    found_spines=True
                    break
            limb._index = -1
        self._index = -1
        return found_spines
    #------------------ some useful built in functions ------------------ #

    def __getitem__(self,key):
        if type(key) == int or type(key) == np.int64 or type(key) == np.int32:
            key = f"L{key}"
        return self.concept_network.nodes[key]["data"]
    def __setitem__(self,key,newvalue):
        if type(key) == int or type(key) == np.int64 or type(key) == np.int32:
            key = f"L{key}"
        self.concept_network.nodes[key]["data"] = newvalue
    def __len__(self):
        return len(list(self.get_limb_node_names()))
    def __iter__(self):
        return self
    def __next__(self):
        self._index += 1
        if self._index >= len(self):
            self._index = -1
            raise StopIteration
        else:
            return self[self._index]

    def get_soma_meshes(self):
        """
        Gives the same output that running the soma identifier would

        Retunrs: a list containing the following elements
        1) list of soma meshes (N)
        2) scalar value of time it took to process (dummy 0)
        3) list of soma sdf values (N)

        """
        soma_meshes = [self.concept_network.nodes[k]["data"].mesh for k in sorted(self.get_soma_node_names())]
        return soma_meshes

    def get_somas(self):
        """
        Gives the same output that running the soma identifier would

        Retunrs: a list containing the following elements
        1) list of soma meshes (N)
        2) scalar value of time it took to process (dummy 0)
        3) list of soma sdf values (N)

        """
        soma_meshes = [self.concept_network.nodes[k]["data"].mesh for k in sorted(self.get_soma_node_names())]
        somas_sdfs = [self.concept_network.nodes[k]["data"].sdf for k in sorted(self.get_soma_node_names())]
        somas =[soma_meshes,0,somas_sdfs]
        return somas

    def get_limb_node_names(self,return_int=False):
        with_l_names = [k for k in self.concept_network.nodes() if "L" in k]
        if return_int:
            sorted_numbers = np.sort([int(k[1:]) for k in with_l_names])
            return [int(k) for k in sorted_numbers]
        else:
            return with_l_names

    def get_limb_names(self,return_int=False):
        with_l_names = [k for k in self.concept_network.nodes() if "L" in k]
        if return_int:
            sorted_numbers = np.sort([int(k[1:]) for k in with_l_names])
            return [int(k) for k in sorted_numbers]
        else:
            return with_l_names

    def get_branch_node_names(self,limb_idx):
        limb_idx = nru.limb_label(limb_idx)
        curr_limb_obj = self.concept_network.nodes[limb_idx]["data"]
        return list(curr_limb_obj.concept_network.nodes())
    def get_soma_node_names(self,int_label=False):
        soma_names = [k for k in self.concept_network.nodes() if "S" in k]
        if int_label:
            return [int(k[1:]) for k in soma_names]
        else:
            return soma_names

    def get_soma_indexes(self):
        return self.get_soma_node_names(int_label=True)

    from datasci_tools import system_utils as su

     # Here was old not actie code  def calculate_spines_old
    @property
    def limb_area(self):
        self._index = -1
        return np.sum([k.area for k in self])

    @property
    def soma_area(self):
        soma_node_names = self.get_soma_node_names()
        return np.sum([self[k].area for k in soma_node_names])

    @property
    def area(self):
        return self.limb_area

    @property
    def area_with_somas(self):
        return self.limb_area + self.soma_area

    @property
    def limb_mesh_volume(self):
        self._index = -1
        return np.sum([k.mesh_volume for k in self])

    @property
    def soma_mesh_volume(self):
        soma_node_names = self.get_soma_node_names()
        return np.sum([self[k].volume for k in soma_node_names])

    @property
    def mesh_volume(self):
        return self.limb_mesh_volume

    @property
    def mesh_volume_with_somas(self):
        return self.limb_mesh_volume + self.soma_mesh_volume

    @property
    def spines(self):
        return nru.feature_list_over_object(self,"spines")
    @property
    def boutons(self):
        return nru.feature_list_over_object(self,"boutons")

    @property
    def synapses(self):
        return nru.feature_list_over_object(self,"synapses")

    @property
    def spines_obj(self):
        return nru.feature_list_over_object(self,"spines_obj")

    @property
    def web(self):
        return nru.feature_list_over_object(self,"web")

    @property
    def spines_volume(self):
        return list(nru.feature_list_over_object(self,"spines_volume"))
    @property
    def boutons_volume(self):
        return nru.feature_list_over_object(self,"boutons_volume")

    def compute_spines_volume(self):
        nru.compute_feature_over_object(self,"spines_volume")

    def compute_boutons_volume(self):
        nru.compute_feature_over_object(self,"boutons_volume")

    @property
    def axon_mesh(self):
        return nru.axon_mesh(self)

    @property
    def dendrite_mesh(self):
        return nru.dendrite_mesh(self)

    @property
    def axon_skeleton(self):
        return nru.axon_skeleton(self)

    @property
    def axon_skeleton(self):
        return nru.axon_skeleton(self)

    @property
    def dendrite_skeleton(self):
        return nru.dendrite_skeleton(self)

    @property
    def axon_limb_name(self):
        if len(self.axon_limb_branch_dict) == 0:
            return None
        else:
            return list(self.axon_limb_branch_dict.keys())[0]

    @property
    def axon_limb(self):
        ax_limb_name = self.axon_limb_name

        if ax_limb_name is not None:
            return self[ax_limb_name]
        else:
            return None

    @property
    def axon_limb_idx(self):
        ax_name = self.axon_limb_name
        if ax_name is None:
            return ax_name
        else:
            return nru.get_limb_int_name(ax_name)

    # ---- 11/20 functions that will help compute statistics of the neuron object ----------

    # -- skeleton and branch data ---

    @property
    def n_limbs(self):
        return nru.n_limbs(self)

    @property
    def n_branches_per_limb(self):
        return nru.n_branches_per_limb(self)

    @property
    def n_branches(self):
        return nru.n_branches(self)

    @property
    def skeleton_length_per_limb(self):
        return nru.skeleton_length_per_limb(self)

    @property
    def skeletal_length(self):
        return nru.skeletal_length(self)

    @property
    def max_limb_skeletal_length(self):
        return nru.max_limb_skeletal_length(self)

    @property
    def max_limb_n_branches(self):
        return nru.max_limb_n_branches(self)

    @property
    def median_branch_length(self):
        return nru.median_branch_length(self)

    # -- width data --
    @property
    def width_median(self):
        return nru.width_median(self)

    @property
    def width_no_spine_median(self):
        return nru.width_no_spine_median(self)

    @property
    def width_90_perc(self):
        return nru.width_perc(self,perc=90)
    @property
    def width_no_spine_90_perc(self):
        return nru.width_no_spine_perc(self,perc=90)

    # -- spine entries--
    @property
    def n_spines(self):
        return nru.n_spines(self)

    @property
    def n_boutons(self):
        return nru.n_boutons(self)

    @property
    def spine_density(self):
        return nru.spine_density(self)

    @property
    def spines_per_branch(self):
        return nru.spines_per_branch(self)

    @property
    def n_spine_eligible_branches(self):
        return nru.n_spine_eligible_branches(self)

    @property
    def spine_eligible_branch_lengths(self):
        return nru.spine_eligible_branch_lengths(self)
    @property
    def skeletal_length_eligible(self):
        return nru.skeletal_length_eligible(self)

    @property
    def spine_density_eligible(self):
        return nru.spine_density_eligible(self)

    @property
    def spines_per_branch_eligible(self):
        return nru.spines_per_branch_eligible(self)

    # ------ spine volume issues ----
    @property
    def total_spine_volume(self):
        return nru.total_spine_volume(self)

    @property
    def spine_volume_median(self):
        return nru.spine_volume_median(self)

    @property
    def spine_volume_density(self):
        return nru.spine_volume_density(self)

    @property
    def spine_volume_density_eligible(self):
        return nru.spine_volume_density_eligible(self)

    @property
    def spine_volume_per_branch_eligible(self):
        return nru.spine_volume_per_branch_eligible(self)
    @property
    def max_soma_n_faces(self):
        return nru.max_soma_n_faces(self)

    @property
    def max_soma_volume(self):
        return nru.max_soma_volume(self)

    @property
    def max_soma_area(self):
        return nru.max_soma_area(self)

    @property
    def n_vertices(self):
        return len(self.mesh.vertices)

    @property
    def n_faces(self):
        return len(self.mesh.faces)

    @property
    def axon_length(self):
        return nru.axon_length(self)

    @property
    def axon_area(self):
        return nru.axon_area(self)

    @property
    def limb_branch_dict(self):
        return nru.neuron_limb_branch_dict(self)

    def neuron_stats(self,stats_to_ignore=None,
                     include_skeletal_stats = False,
                    include_centroids= False,
                     voxel_adjustment_vector = None,
                     cell_type_mode = False,
                    **kwargs):
        return nst.neuron_stats(self,
                    stats_to_ignore=stats_to_ignore,
                     include_skeletal_stats = include_skeletal_stats,
                    include_centroids= include_centroids,
                     voxel_adjustment_vector = voxel_adjustment_vector,
                     cell_type_mode = cell_type_mode,
                    **kwargs)

    @property
    def mesh_kdtree(self):
        """A kdtree of the original mesh"""
        if self._mesh_kdtree is None:
            self._mesh_kdtree = KDTree(self.mesh.triangles_center)
        return self._mesh_kdtree

    @property
    def mesh_from_branches(self):
        return nru.neuron_mesh_from_branches(self)

    # ------- 6/9: The pre and post of the errored meshes
    @property
    def n_mesh_errored_synapses(self):
        return len(self.mesh_errored_synapses)

    @property
    def n_mesh_errored_synapses_pre(self):
        return len(self.mesh_errored_synapses_pre)

    @property
    def n_mesh_errored_synapses_post(self):
        return len(self.mesh_errored_synapses_post)

    @property
    def n_distance_errored_synapses(self):
        return len(self.distance_errored_synapses)

    @property
    def n_distance_errored_synapses_pre(self):
        return len(self.distance_errored_synapses_pre)

    @property
    def n_distance_errored_synapses_post(self):
        return len(self.distance_errored_synapses_post)

    @property
    def merge_filter_locations(self):
        """
        a nested dictionary datastructure storing the information on where the merge error filters were trigger.
        The order that the merge error filters was applied matters because the branches that triggered the filter are only those that had not triggered an early applied filter, and thus it was not already filtered away.
        Note: this product includes all branches that triggered the filter at this stage, regardless if they were downstream of another
        The datastructure is organized in the following way:

        merge error filter name: --> dict
            limb name:
                list of 2x3 arrays that stores coordinates of the endpoints the skeleton of the branch that triggered the merge error filter (1st coordinate is upstream branch)
        """
        try:
            return self.pipeline_products["auto_proof"]["split_locations_before_filter"]
        except:
            return dict()

#--- from neurd_packages ---

from . import branch_utils as bu
# P3: `preprocess_neuron as pre` is now a local import in __init__ (only use site) to
# break the neuron<->preprocess_neuron cycle.
from . import soma_extraction_utils as sm
from . import spine_utils as spu
from . import width_utils as wu

#--- from mesh_tools ---
from mesh_tools import skeleton_utils as sk
from mesh_tools import trimesh_utils as tu

#--- from datasci_tools ---

from datasci_tools import networkx_utils as xu
from datasci_tools import numpy_dep as np
from datasci_tools import numpy_utils as nu
from datasci_tools import system_utils as su
from datasci_tools import pipeline as pl
