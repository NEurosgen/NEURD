

from datasci_tools import numpy_dep as np

# ---------- statistics ----------------

default_relation_value = -1

def parent_skeletal_angle(
    limb_obj,
    branch_idx,
    verbose = False,
    default_value = None,
    skeletal_angle_attr = "skeleton_vector_[dir]",
    skeleton_attribute = "skeleton",
    **kwargs):
    """
    Purpose: to get the branching angle with parent
    from the skeleton vectors

    Pseudocode: 
    1) Get parent branch
    2) Get parent and child vector
    3) Get the angle between the two
    
    Ex: 
    from neurd import limb_utils as lu
    lu.parent_skeletal_angle(
    branch_idx = 2,
    limb_obj = neuron_obj[1],
    verbose = True,
    )
    """
    
    skeletal_angle_attr=skeletal_angle_attr.replace("skeleton",skeleton_attribute)
    
    upstream_attr = skeletal_angle_attr.replace('[dir]','upstream')
    downstream_attr = skeletal_angle_attr.replace('[dir]','downstream')
    
    if limb_obj[branch_idx].endpoints_upstream_downstream_idx is None:
        bu.set_branches_endpoints_upstream_downstream_idx_on_limb(limb_obj)
    
    parent_idx = nru.parent_node(limb_obj,branch_idx)
    if verbose:
        print(f"parent_idx = {parent_idx}")
    
    if  parent_idx is None:
        return default_value
    
    branch_obj = limb_obj[parent_idx]
    branch_obj_2 = limb_obj[branch_idx]
    return np.round(nu.angle_between_vectors(
        getattr(branch_obj,downstream_attr),
        getattr(branch_obj_2,upstream_attr)),2
    )
    
def parent_skeletal_angle_extra_offset(
    limb_obj,
    branch_idx,
    verbose = False,
    default_value = None,
    skeletal_angle_attr = "skeleton_vector_[dir]",
    **kwargs):
    return parent_skeletal_angle(
        limb_obj,
        branch_idx,
        verbose = verbose,
        default_value = default_value,
        skeletal_angle_attr = "skeleton_vector_[dir]_extra_offset",
        **kwargs)
    
    












    
def width_upstream(
    limb_obj,
    branch_idx,
    verbose = False,
    min_skeletal_length = 2000,
    skip_low_skeletal_length_upstream = True,
    default_value = 10000000):
    """
    Purpoose: To get the width of the upstream segement
    
    Pseudocode:
    1) Get the parent node
    2) Get the parent width
    
    Ex: 
    from neurd import limb_utils as lu
    lu.width_upstream(neuron_obj[1],5,verbose = True)
    """
    parent_node = nru.parent_node(limb_obj,branch_idx)
    
    if min_skeletal_length is None:
        min_skeletal_length = 0
        
    parent_sk_length = None
    width = default_value
    
    while parent_node is not None:
        parent_sk_length = limb_obj[parent_node].skeletal_length
        if parent_sk_length < min_skeletal_length:
            parent_node = nru.parent_node(limb_obj,parent_node)
        else:
            break

    if parent_node is not None:
        width = nru.width(limb_obj[parent_node])
        parent_sk_length = limb_obj[parent_node].skeletal_length
    
    if verbose:
        print(f"parent_node = {parent_node} (width = {width}, parent_sk_length = {parent_sk_length})")
        
    return width


    

# ------------ automatically create limb functions out of existing functions ------









#--- from neurd_packages ---
from . import branch_utils as bu
from . import neuron_searching as ns
from . import neuron_statistics as nst
from . import neuron_utils as nru
from . import concept_network_utils as cnu

#--- from datasci_tools ---
from datasci_tools import networkx_utils as xu
from datasci_tools import numpy_dep as np
from datasci_tools import numpy_utils as nu


from mesh_tools import trimesh_utils as tu
from datasci_tools import (
    numpy_utils as nu,
    ipyvolume_utils as ipvu,
)

from . import limb_utils as lu

ns.set_limb_functions_for_search(lu,verbose = False)




