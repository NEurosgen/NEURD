import sys as _sys

import numpy as np
from datasci_tools import numpy_utils as nu

from . import branch_utils as bu
from . import neuron_searching as ns
from . import neuron_utils as nru


default_relation_value = -1


def parent_skeletal_angle(
    limb_obj,
    branch_idx,
    verbose=False,
    default_value=None,
    skeletal_angle_attr="skeleton_vector_[dir]",
    skeleton_attribute="skeleton",
    **kwargs,
):
    """Return the branching angle between branch_idx and its parent, from skeleton vectors."""
    skeletal_angle_attr = skeletal_angle_attr.replace("skeleton", skeleton_attribute)
    upstream_attr = skeletal_angle_attr.replace("[dir]", "upstream")
    downstream_attr = skeletal_angle_attr.replace("[dir]", "downstream")

    if limb_obj[branch_idx].endpoints_upstream_downstream_idx is None:
        bu.set_branches_endpoints_upstream_downstream_idx_on_limb(limb_obj)

    parent_idx = nru.parent_node(limb_obj, branch_idx)
    if verbose:
        print(f"parent_idx = {parent_idx}")

    if parent_idx is None:
        return default_value

    branch_obj = limb_obj[parent_idx]
    branch_obj_2 = limb_obj[branch_idx]
    return np.round(
        nu.angle_between_vectors(
            getattr(branch_obj, downstream_attr),
            getattr(branch_obj_2, upstream_attr),
        ),
        2,
    )


def parent_skeletal_angle_extra_offset(
    limb_obj,
    branch_idx,
    verbose=False,
    default_value=None,
    skeletal_angle_attr="skeleton_vector_[dir]",
    **kwargs,
):
    return parent_skeletal_angle(
        limb_obj,
        branch_idx,
        verbose=verbose,
        default_value=default_value,
        skeletal_angle_attr="skeleton_vector_[dir]_extra_offset",
        **kwargs,
    )


# set_limb_functions_for_search needs the module object — reference via sys.modules.
ns.set_limb_functions_for_search(_sys.modules[__name__], verbose=False)
