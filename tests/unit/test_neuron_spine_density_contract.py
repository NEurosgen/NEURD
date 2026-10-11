"""Spine volume densities divide only by their own positive length denominator.

The total=0/eligible>0 fixture deliberately does not describe a physical neuron:
it isolates the two scalar contracts when callers supply enriched objects.
"""
from types import SimpleNamespace
import warnings

import numpy as np
import pytest

from tests.unit import skip_if_datasci_tools_unusable

skip_if_datasci_tools_unusable()

from neurd import neuron_utils as nru


@pytest.mark.parametrize("scalar", [int, float, np.float64])
@pytest.mark.parametrize("n_spines", [0, 2])
@pytest.mark.parametrize("total,eligible,expected", [
    (12, 4, (2, 6)),
    (12, 0, (2, 0)),
    (0, 4, (0, 6)),
    (0, 0, (0, 0)),
])
def test_each_density_uses_its_own_length_oracle(scalar, n_spines, total, eligible, expected):
    obj = SimpleNamespace(n_spines=n_spines, total_spine_volume=scalar(24),
                          skeletal_length=scalar(total),
                          skeletal_length_eligible=scalar(eligible))
    if n_spines == 0:
        expected = (0, 0)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with np.errstate(divide="raise", invalid="raise"):
            actual = (nru.spine_volume_density(obj), nru.spine_volume_density_eligible(obj))
    assert actual == expected
    assert all(np.isfinite(value) for value in actual)


@pytest.mark.parametrize("function,length_name", [
    (nru.spine_volume_density, "skeletal_length"),
    (nru.spine_volume_density_eligible, "skeletal_length_eligible"),
])
def test_positive_density_preserves_exact_scalar_division(function, length_name):
    obj = SimpleNamespace(n_spines=2, total_spine_volume=np.float64(1.234567890123456e9),
                          skeletal_length=np.float64(125.33333),
                          skeletal_length_eligible=np.float64(70.121))
    expected = obj.total_spine_volume / getattr(obj, length_name)
    actual = function(obj)
    assert type(actual) is type(expected)
    assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize("function,length_name", [
    (nru.spine_volume_density, "skeletal_length"),
    (nru.spine_volume_density_eligible, "skeletal_length_eligible"),
])
@pytest.mark.parametrize("length", [-4.0, np.nan])
def test_nonpositive_and_nan_denominators_retain_greater_than_zero_policy(function, length_name, length):
    obj = SimpleNamespace(n_spines=2, total_spine_volume=24.0,
                          skeletal_length=12.0, skeletal_length_eligible=4.0)
    setattr(obj, length_name, length)
    assert function(obj) == 0


@pytest.mark.parametrize("function", [nru.spine_volume_density, nru.spine_volume_density_eligible])
def test_zero_spine_count_short_circuits_volume_and_lengths(function):
    assert function(SimpleNamespace(n_spines=0)) == 0
