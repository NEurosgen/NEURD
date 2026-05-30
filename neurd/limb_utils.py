

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
    parent_skeletal_angle(
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


# P2: self-import `from . import limb_utils as lu` removed. set_limb_functions_for_search
# needs the module object itself, so reference it explicitly via sys.modules.
import sys as _sys
ns.set_limb_functions_for_search(_sys.modules[__name__], verbose = False)




