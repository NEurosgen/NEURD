"""Raw-neuron mesh decomposition into somas, limbs, branches and skeletons.

Spine detection is explicit and disabled by default. Dataset parameters and CGAL
rays are process globals; concurrent runs with different settings need isolated
processes. Existing callers that configure ``parameters.params`` themselves can
omit ``dataset`` to retain the selected dataset.
"""
from neurd import neuron
from neurd.run_config import RunConfig


def segmentation_pipeline(
    mesh,
    segment_id=12345,
    verbose=False,
    max_somas=None,
    *,
    calculate_spines=None,
    dataset=None,
    refinement=None,
    sdf_profile=None,
    sdf_rays=None,
    config=None,
):
    """Decompose a ``trimesh.Trimesh`` using the shared raw-neuron configuration.

    ``calculate_spines`` defaults to False. With no dataset argument, retain the
    dataset already selected by ``neurd.set_volume_params``. New callers should
    pass ``dataset='microns'`` or ``'h01'`` explicitly. The reference profile uses
    25 SDF rays; ``sdf_profile='fast'`` uses 12. Named H01 refinement policies
    require spines and remain experimental.

    Supply either a ``RunConfig`` or individual configuration arguments. The
    returned Neuron exposes ``somas`` and ``limbs[i].branches[j].mesh/skeleton``;
    enabled spine meshes and volumes live on the branches. ``max_somas=1`` bounds
    the soma search for a known single-neuron mesh. Progress may still be printed
    when ``verbose=False``.
    """
    explicit = (calculate_spines, dataset, refinement, sdf_profile, sdf_rays)
    if config is not None:
        if not isinstance(config, RunConfig):
            raise TypeError("config must be a RunConfig")
        if any(value is not None for value in explicit):
            raise ValueError("Pass either config or individual configuration arguments")
    else:
        from neurd import parameters
        config = RunConfig(
            dataset=parameters.params.data_type if dataset is None else dataset,
            calculate_spines=False if calculate_spines is None else calculate_spines,
            refinement="off" if refinement is None else refinement,
            sdf_profile="reference" if sdf_profile is None else sdf_profile,
            sdf_rays=sdf_rays,
        )
    config.apply()
    neuron_obj = neuron.Neuron(
        mesh=mesh, segment_id=segment_id, verbose=verbose, max_somas=max_somas,
        calculate_spines=config.calculate_spines, spine_kwargs=config.spine_kwargs(),
    )
    neuron_obj.run_config = config.as_dict()
    return neuron_obj
