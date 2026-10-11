"""Soma centers remain in nm unless an explicit voxel conversion is requested."""
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest
import trimesh

from tests.unit import skip_if_datasci_tools_unusable

skip_if_datasci_tools_unusable()

from neurd import neuron_statistics as stats, neuron_utils as nru
from neurd.neuron import Limb, Neuron, Soma


def neuron_with_somas(centers):
    obj = object.__new__(Neuron)
    obj._index = -1
    obj.pipeline_products = SimpleNamespace()
    obj.concept_network = nx.Graph()
    for name, center in centers:
        mesh = trimesh.creation.box(extents=[2, 4, 6])
        mesh.vertices += np.asarray(center)
        soma = Soma(mesh, sdf=1, volume_ratio=1, volume=48)
        obj.concept_network.add_node(name, data=soma)
    return obj


def add_limb(obj):
    limb = object.__new__(Limb)
    limb._index = -1
    limb.concept_network = nx.Graph()
    axis = np.asarray([[[1., 0., 0.], [5., 0., 0.]]])
    limb.concept_network.add_node(0, data=SimpleNamespace(skeleton=axis, skeletal_length=4.))
    limb.all_concept_network_data = [dict(starting_soma=0, starting_coordinate=np.asarray([1., 0., 0.]))]
    obj.concept_network.add_node("L0", data=limb)
    obj.concept_network.add_edge("S0", "L0")
    return obj


def test_nm_centers_need_no_scale_and_preserve_graph_order():
    obj = neuron_with_somas([("S2", [-5.5, 7.5, 13.5]), ("S0", [10.5, -20.5, 30.5])])
    np.testing.assert_array_equal(nru.soma_centers(obj, return_int_form=False, return_single=False),
                                  [[-5.5, 7.5, 13.5], [10.5, -20.5, 30.5]])
    np.testing.assert_array_equal(nru.soma_centers(obj, return_single=False), [[-5, 7, 13], [10, -20, 30]])
    with pytest.raises(Exception, match="Not just one soma center"):
        nru.soma_centers(obj)


@pytest.mark.parametrize("integer", [False, True])
def test_selected_soma_and_default_single_shape(integer):
    obj = neuron_with_somas([("S2", [1.5, -2.5, 3.5]), ("S0", [4.5, 5.5, -6.5])])
    expected = np.asarray([4.5, 5.5, -6.5])
    if integer:
        expected = expected.astype(int)
    actual = nru.soma_centers(obj, soma_name="S0", return_int_form=integer)
    np.testing.assert_array_equal(actual, expected)
    assert actual.shape == (3,)
    np.testing.assert_array_equal(nru.soma_centers(obj, soma_name="S0", return_int_form=integer,
                                                return_single=False), expected.reshape(1, 3))


def test_empty_centers_keep_n_by_three_shape_and_default_single_error():
    obj = neuron_with_somas([])
    for integer in (True, False):
        actual = nru.soma_centers(obj, return_int_form=integer, return_single=False)
        assert actual.shape == (0, 3)
        assert actual.dtype == (np.dtype(int) if integer else np.dtype(float))
    with pytest.raises(Exception, match="Not just one soma center"):
        nru.soma_centers(obj)


def test_anisotropic_conversion_truncates_negative_coordinates_toward_zero():
    obj = neuron_with_somas([("S0", [-5.5, 7.5, 13.5])])
    np.testing.assert_array_equal(nru.soma_centers(obj, voxel_adjustment=True,
                                                voxel_adjustment_vector=[2, 3, 4], return_int_form=False),
                                  [-2.75, 2.5, 3.375])
    np.testing.assert_array_equal(nru.soma_centers(obj, voxel_adjustment=True,
                                                voxel_adjustment_vector=[2, 3, 4]), [-2, 2, 3])


def test_scale_is_not_consulted_for_nm_only_output():
    obj = neuron_with_somas([("S0", [2, 4, 6])])
    np.testing.assert_array_equal(nru.soma_centers(obj, voxel_adjustment_vector=[0, np.nan]), [2, 4, 6])


@pytest.mark.parametrize("scale", [None, [0, 1, 1], [-1, 1, 1], [1, np.nan, 1], [1, np.inf, 1],
                                  4, [1, 2], [[1, 2, 3]], ["a", "b", "c"], [1j, 1, 1]])
def test_invalid_voxel_scales_raise_explicit_value_error(scale):
    obj = neuron_with_somas([("S0", [2, 4, 6])])
    with pytest.raises(ValueError, match="nm-per-voxel"):
        nru.soma_centers(obj, voxel_adjustment=True, voxel_adjustment_vector=scale)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.longdouble])
def test_valid_scale_retains_numpy_division_dtype_and_values(dtype):
    obj = neuron_with_somas([("S0", [-5.5, 7.5, 13.5])])
    scale = np.asarray([2, 3, 4], dtype=dtype)
    expected = obj["S0"].mesh_center / scale
    actual = nru.soma_centers(obj, soma_name="S0", voxel_adjustment=True,
                             voxel_adjustment_vector=scale, return_int_form=False)
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("volume", [True, False])
def test_centroid_stats_retain_explicit_vector_keys_and_units(volume):
    obj = neuron_with_somas([("S0", [-5.5, 7.5, 13.5])])
    actual = stats.centroid_stats_from_neuron_obj(obj, voxel_adjustment_vector=[2, 3, 4], include_volume=volume)
    expected = dict(centroid_x_nm=-5.5, centroid_y_nm=7.5, centroid_z_nm=13.5,
                    centroid_x=-2, centroid_y=2, centroid_z=3)
    if volume:
        expected["centroid_volume"] = 48 / 1_000_000_000
    assert actual == expected
    with pytest.raises(ValueError, match="explicit voxel_adjustment_vector"):
        stats.centroid_stats_from_neuron_obj(obj, include_volume=volume)


def test_skeleton_stats_without_centroids_require_no_voxel_scale():
    obj = add_limb(neuron_with_somas([("S0", [0, 0, 0])]))
    actual = stats.skeleton_stats_from_neuron_obj(obj, include_centroids=False)
    assert actual["skeletal_length"] == 4.
    assert actual["n_branches"] == 1
    assert actual["n_limbs"] == 1
    assert not any(key.startswith("centroid_") for key in actual)


def test_skeleton_stats_with_centroids_require_explicit_vector():
    obj = add_limb(neuron_with_somas([("S0", [0, 0, 0])]))
    with pytest.raises(ValueError, match="explicit voxel_adjustment_vector"):
        stats.skeleton_stats_from_neuron_obj(obj)
    actual = stats.skeleton_stats_from_neuron_obj(obj, voxel_adjustment_vector=[2, 3, 4])
    assert actual["centroid_x_nm"] == actual["centroid_x"] == 0
    assert actual["centroid_y_nm"] == actual["centroid_y"] == 0
    assert actual["centroid_z_nm"] == actual["centroid_z"] == 0
