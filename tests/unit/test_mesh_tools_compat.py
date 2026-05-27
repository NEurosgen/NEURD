"""Verify mesh_tools submodules import with modern deps (open3d>=0.19, numpy 2)."""
import pytest


def test_trimesh_utils_imports():
    import neurd  # noqa: F401
    pytest.importorskip("mesh_tools")
    pytest.importorskip("open3d")
    from mesh_tools import trimesh_utils  # noqa: F401


def test_skeleton_utils_imports():
    import neurd  # noqa: F401
    pytest.importorskip("mesh_tools")
    from mesh_tools import skeleton_utils  # noqa: F401


def test_meshparty_version_modern():
    meshparty = pytest.importorskip("meshparty")
    major = int(meshparty.__version__.split(".")[0])
    assert major >= 2, f"Expected meshparty>=2, got {meshparty.__version__}"


def test_open3d_version_modern():
    o3d = pytest.importorskip("open3d")
    parts = o3d.__version__.split(".")
    major, minor = int(parts[0]), int(parts[1])
    assert (major, minor) >= (0, 19), f"Expected open3d>=0.19, got {o3d.__version__}"
