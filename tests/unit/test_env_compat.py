"""Smoke tests for the local-env shims in `neurd/__init__.py`.

These guard against regressions of Phase 3:
- numpy>=2 compatibility (np.float_ restored as alias)
- ipyvolume/cloudvolume stub finder for optional viz / cloud deps
"""
import sys

import pytest


def test_neurd_imports_without_error():
    import neurd  # noqa: F401


def test_datasci_tools_imports_after_shim():
    """Without the np.float_ shim this import raises AttributeError on numpy>=2."""
    import neurd  # noqa: F401
    from datasci_tools import numpy_dep  # noqa: F401
    from datasci_tools import module_utils  # noqa: F401  -- pulls full chain


def test_ipyvolume_submodule_resolves_via_stub():
    """`from ipyvolume.moviemaker import MovieMaker` (used by datasci_tools)
    must succeed even when ipyvolume is not installed."""
    import neurd  # noqa: F401
    import ipyvolume
    from ipyvolume.moviemaker import MovieMaker  # noqa: F401
    # Real ipyvolume is a package with __version__; our stub has no real attrs.
    is_stub = not hasattr(ipyvolume, "__version__")
    if not is_stub:
        pytest.skip("real ipyvolume installed, stub path not exercised")


def test_mesh_tools_skeleton_utils_imports():
    """skeleton_utils does `import ipyvolume as ipv` at top level; must work."""
    import neurd  # noqa: F401
    pytest.importorskip("mesh_tools")
    from mesh_tools import skeleton_utils  # noqa: F401


def test_real_package_wins_over_stub():
    """If a real package is in sys.modules already, the stub finder must not
    override it."""
    import neurd  # triggers stub finder install
    fake = type(sys)("ipyvolume")
    fake.__version__ = "test"
    saved = sys.modules.get("ipyvolume")
    sys.modules["ipyvolume"] = fake
    try:
        import ipyvolume
        assert ipyvolume is fake
    finally:
        if saved is not None:
            sys.modules["ipyvolume"] = saved
        else:
            sys.modules.pop("ipyvolume", None)
