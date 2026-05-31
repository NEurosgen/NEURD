"""Single explicit source of truth for tunable parameters.

Replaces the previous mechanism that monkey-patched ``*_global`` module
variables (one per parameter, per module) at runtime via
``set_volume_params`` / ``parameter_utils``. Here every tunable lives in one
importable object with two modes:

    from neurd import parameters
    parameters.params.offset_skeleton_vector      # current value
    parameters.params.use("h01")                  # switch mode

``microns`` uses the base values; ``h01`` applies the deltas in ``_H01`` on
top of them. The object is created in ``microns`` mode at import, so reads
always succeed without an explicit setup call.
"""


# Base values (microns). Grown module-by-module as the global dicts are retired.
_BASE = dict(
    # branch_utils
    offset_skeleton_vector=500,
    comparison_distance_skeleton_vector=3000,
    extra_offset_skeleton_vector=6000,
    offset_width_endpoint=0,
    # neuron_utils
    skeletal_length_max_n_spines=3000,
    # soma_extraction_utils
    nucleus_min=700,
    nucleus_max=None,
    glia_volume_threshold_in_um=2500,
    glia_n_faces_threshold=400_000,
    glia_n_faces_min=100_000,
    outer_decimation_ratio=0.25,
    large_mesh_threshold=20_000,
    large_mesh_threshold_inner=13_000,
    inner_decimation_ratio=0.25,
    max_fail_loops=10,
    remove_inside_pieces=True,
    size_threshold_to_remove=1_000,
    pymeshfix_clean=False,
    check_holes_before_pymeshfix=False,
    second_poisson=False,
    soma_width_threshold=0.32,
    soma_size_threshold=9_000,
    soma_size_threshold_max=1_200_000,
    volume_mulitplier=8,
    side_length_ratio_threshold=6,
    perform_pairing=False,
    backtrack_soma_mesh_to_original=True,
    backtrack_soma_size_threshold=8000,
    poisson_backtrack_distance_threshold=None,
    close_holes=False,
    boundary_vertices_threshold=None,
    last_size_threshold=2000,
    segmentation_at_end=True,
    largest_hole_threshold=17000,
    second_pass_size_threshold=None,
    # preprocess_neuron
    width_threshold_MAP=500,
    size_threshold_MAP=2000,
    size_threshold_MAP_stitch=2000,
    apply_expansion=False,
    max_stitch_distance=8000,
    max_stitch_distance_CGAL=5000,
    filter_end_node_length=4000,
    axon_width_preprocess_limb_max=200,
    limb_remove_mesh_interior_face_threshold=0,
    surface_reconstruction_size=1000,
    floating_piece_face_threshold=50,
    invalidation_d=12000,
    remove_mesh_interior_face_threshold=0,
    mp_only_revised_invalidation_d=False,
    mp_only_invalidation_d_axon_buffer=None,
    mp_only_revised_invalidation_d_reference=None,
    mp_only_revised_width_reference=None,
    use_adaptive_invalidation_d=False,
    use_adaptive_invalidation_d_floating=True,
    combine_close_skeleton_nodes_threshold_meshparty_axon=1300,
    filter_end_node_length_meshparty_axon=1150,
    filter_end_node_length_axon=1150,
    invalidation_d_axon=1500,
    smooth_neighborhood_axon=0,
)

# h01-only deltas, applied on top of _BASE when mode == "h01".
_H01 = dict(
    # neuron_utils
    skeletal_length_max_n_spines=6_000,
    # soma_extraction_utils
    glia_n_faces_threshold=3_000_000,
    glia_n_faces_min=3_000_000,
    large_mesh_threshold=40_000,
    large_mesh_threshold_inner=20_000,
    soma_size_threshold=15_000,
    soma_size_threshold_max=2_000_000,
    backtrack_soma_size_threshold=15_000,
    last_size_threshold=15_000,
    second_pass_size_threshold=5_000,
    # preprocess_neuron
    width_threshold_MAP=1000,
    size_threshold_MAP=10_000,
    size_threshold_MAP_stitch=14_000,
    apply_expansion=True,
    max_stitch_distance=13_000,
    max_stitch_distance_CGAL=13_000,
    axon_width_preprocess_limb_max=350,
    limb_remove_mesh_interior_face_threshold=150,
    floating_piece_face_threshold=500,
    use_adaptive_invalidation_d=True,
    combine_close_skeleton_nodes_threshold_meshparty_axon=1700,
    invalidation_d_axon=2500,
)


class _Params:
    def __init__(self):
        self.use("microns")

    def use(self, data_type):
        """Switch parameter mode in place ("microns" or "h01")."""
        data = dict(_BASE)
        if data_type == "h01":
            data.update(_H01)
        elif data_type != "microns":
            raise ValueError(f"Unknown data_type {data_type!r} (expected 'microns' or 'h01')")
        self._data = data
        self.data_type = data_type

    def __getattr__(self, name):
        try:
            return self.__dict__["_data"][name]
        except KeyError:
            raise AttributeError(name)


params = _Params()
