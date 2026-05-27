"""numpy>=2 compatibility checks for the shim in `neurd/__init__.py`."""
import numpy as np


def test_numpy_version_is_2_or_higher():
    major = int(np.__version__.split(".")[0])
    assert major >= 2, f"Expected numpy>=2 for this env, got {np.__version__}"


def test_numpy_float64_basics():
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    assert arr.mean() == 2.0
    assert np.float64(1.5) == 1.5


def test_shim_restores_float_alias():
    import neurd  # noqa: F401  -- activates shim
    assert np.float_ is np.float64


def test_shim_restores_int_complex_aliases():
    import neurd  # noqa: F401
    assert np.int_ is np.int64
    assert np.complex_ is np.complex128
