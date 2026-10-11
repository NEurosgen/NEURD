"""NEURD package entry points.

Compatibility shims and provider patches are installed by the cached bootstrap
module before the public entry points below are defined.
"""

from . import _bootstrap

PATCH_STATUS = _bootstrap.PATCH_STATUS
_patch_ok = _bootstrap._patch_ok
_patch_skipped = _bootstrap._patch_skipped
_patch_failed = _bootstrap._patch_failed
Path = _bootstrap.Path
_STUBBED_ROOTS = _bootstrap._STUBBED_ROOTS
_StubModule = _bootstrap._StubModule
_StubLoader = _bootstrap._StubLoader
_StubFinder = _bootstrap._StubFinder

# Preserve historical bindings that exist only when their patch stage reached them.
if hasattr(_bootstrap, "_decimator_init_orig"):
    _decimator_init_orig = _bootstrap._decimator_init_orig
if hasattr(_bootstrap, "_decimator_init_capture"):
    _decimator_init_capture = _bootstrap._decimator_init_capture
if hasattr(_bootstrap, "_decimate_inprocess"):
    _decimate_inprocess = _bootstrap._decimate_inprocess
if hasattr(_bootstrap, "_BaseKDTree_sk"):
    _BaseKDTree_sk = _bootstrap._BaseKDTree_sk
if hasattr(_bootstrap, "_KDTree2DTolerant"):
    _KDTree2DTolerant = _bootstrap._KDTree2DTolerant

from .version import __version__

default_data_type = "microns"


def set_volume_params(
    volume=default_data_type,
    verbose=False,
    verbose_loop=False,
):
    """Switch the active parameter set ("microns" or "h01").

    Canonical entry point now that parameters live in `neurd.parameters`;
    delegates to the single explicit params object instead of monkey-patching
    `*_global` variables across every module.
    """
    from . import parameters
    parameters.params.use(volume)
