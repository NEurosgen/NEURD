"""Configuration shared by raw-neuron API calls and isolated batch workers.

Dataset parameters and CGAL's ray count are process globals. Apply a configuration
once before constructing a neuron; use separate processes for concurrent datasets.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from importlib.machinery import EXTENSION_SUFFIXES
import importlib
import os


RAY_PROFILES = {"reference": 25, "fast": 12}
REFINEMENT_POLICIES = ("off", "h01_current", "h01_smooth_heads")


def runtime_environment(config, *, environment=None):
    """Record behavior flags, effective import patches and numeric thread limits."""
    import neurd
    names = (
        "NEURD_LEGACY_MESH_LOADER", "NEURD_OWNED_SKELETAL_DISTANCE", "NEURD_REAL_POISSON", "NEURD_POISSON_DEPTH",
        "NEURD_POISSON_BACKEND", "NEURD_POISSON_ITERS", "NEURD_SKIP_SOMA_END_SPLIT",
        "NEURD_SOMA_OUTER_DECIM", "NEURD_SOMA_INNER_DECIM", "NEURD_DECIMATE_SINGLE_STAGE",
        "NEURD_MESHLAB_DECIMATE", "NEURD_LEGACY_VERTEX_COMPONENTS",
        "NEURD_LEGACY_MATCHING_VERTICES", "NEURD_ENABLE_COMPRESSED_PICKLE", "NEURD_CGAL_TIMINGS",
        "NEURD_LEGACY_CGAL_ASSEMBLY", "NEURD_LEGACY_SHAFT_PATHS",
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS", "PYTHONHASHSEED",
    )
    environment = os.environ if environment is None else environment
    values = {name: environment.get(name) for name in names}
    # apply() changes this variable in the worker, so fingerprint its configured
    # effective value rather than an unrelated ambient value in the supervisor.
    values["NEURD_SDF_RAYS"] = str(config.sdf_rays)
    return dict(environment=values, import_patches=dict(neurd.PATCH_STATUS))


def require_compiled_cgal():
    """Reject the Python stand-in before an expensive spine-enabled run."""
    backend = importlib.import_module("cgal_Segmentation_Module")
    path = str(getattr(backend, "__file__", ""))
    if not any(path.endswith(suffix) for suffix in EXTENSION_SUFFIXES):
        raise RuntimeError(
            "Spine detection requires the compiled CGAL segmentation backend; "
            "install cgal/cgal_segmentation with CGAL headers available."
        )
    return path


@dataclass(frozen=True)
class RunConfig:
    """Validated raw-neuron settings; refinement remains opt-in and H01-only.

    ``sdf_rays=None`` resolves from ``sdf_profile``. Explicit ray counts are
    recorded as ``custom`` unless they match the selected profile. The API's
    compatibility default excludes spines; CLI callers select that mode explicitly.
    """

    dataset: str = "microns"
    calculate_spines: bool = False
    refinement: str = "off"
    sdf_profile: str = "reference"
    sdf_rays: int | None = None

    def __post_init__(self):
        if self.dataset not in ("microns", "h01"):
            raise ValueError("dataset must be 'microns' or 'h01'")
        if not isinstance(self.calculate_spines, bool):
            raise ValueError("calculate_spines must be a boolean")
        if self.refinement not in REFINEMENT_POLICIES:
            raise ValueError(f"refinement must be one of {REFINEMENT_POLICIES}")
        if self.refinement != "off" and not self.calculate_spines:
            raise ValueError("A refinement policy requires calculate_spines=True")
        if self.refinement != "off" and self.dataset != "h01":
            raise ValueError("The named refinement policies are available only for H01")
        if self.sdf_profile not in RAY_PROFILES:
            raise ValueError(f"sdf_profile must be one of {tuple(RAY_PROFILES)}")
        rays = RAY_PROFILES[self.sdf_profile] if self.sdf_rays is None else self.sdf_rays
        # The native extension parses this budget with C++ atoi (signed int).
        if isinstance(rays, bool) or not isinstance(rays, int) or not 0 < rays <= 2_147_483_647:
            raise ValueError("sdf_rays must be a positive 32-bit integer")
        object.__setattr__(self, "sdf_rays", rays)

    def refinement_config(self):
        if self.refinement == "off":
            return None
        from neurd.spine_refinement import RefinementConfig

        overrides = dict(
            grow_ragged_heads=True,
            trim_shaft_connections=True,
            trim_single_protrusions=True,
            filter_low_relief_sheets=True,
            retry_with_mesh_axis=True,
            refine_remaining_heads=True,
            filter_incomplete_heads=True,
        )
        if self.refinement == "h01_smooth_heads":
            overrides.update(
                screen_smooth_incomplete_heads=True,
                smooth_head_minimum_added_clearance_nm=100.0,
            )
        return RefinementConfig(**overrides)

    def spine_kwargs(self):
        return dict(
            geometric_refinement=self.refinement != "off",
            geometric_refinement_config=self.refinement_config(),
        )

    def as_dict(self):
        data = asdict(self)
        if self.sdf_rays != RAY_PROFILES[self.sdf_profile]:
            data["sdf_profile"] = "custom"
        config = self.refinement_config()
        data["refinement_config"] = asdict(config) if config is not None else None
        return data

    def apply(self):
        """Apply process globals once, before soma/limb/spine calculations."""
        if self.calculate_spines:
            require_compiled_cgal()
        from neurd import parameters

        parameters.params.use(self.dataset)
        os.environ["NEURD_SDF_RAYS"] = str(self.sdf_rays)


def add_run_arguments(parser, *, default_spines=True):
    """Expose the same configuration options in both raw mesh runners."""
    parser.add_argument("--dataset", choices=("microns", "h01"), default="microns")
    parser.add_argument(
        "--spines", action=argparse.BooleanOptionalAction,
        default=default_spines, help="calculate raw branch spines (requires compiled CGAL)",
    )
    parser.add_argument("--refinement", choices=REFINEMENT_POLICIES, default="off")
    parser.add_argument("--sdf-profile", choices=tuple(RAY_PROFILES), default="reference")
    parser.add_argument("--sdf-rays", type=int, help="explicit positive ray count; recorded in results")


def config_from_args(args):
    return RunConfig(
        dataset=args.dataset, calculate_spines=args.spines, refinement=args.refinement,
        sdf_profile=args.sdf_profile, sdf_rays=args.sdf_rays,
    )
