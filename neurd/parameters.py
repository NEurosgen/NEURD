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
)

# h01-only deltas, applied on top of _BASE when mode == "h01".
_H01 = dict(
    # branch_utils: no h01 overrides
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
