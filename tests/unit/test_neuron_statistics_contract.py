"""The legacy summary can read slim objects without inventing preprocessing data."""
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest
import trimesh

from tests.unit import skip_if_datasci_tools_unusable

skip_if_datasci_tools_unusable()

from neurd import neuron_statistics as stats
from neurd.neuron import Branch, Limb, Neuron, Soma


LEGACY_KEYS = (
    "n_vertices", "n_faces", "max_soma_volume", "max_soma_n_faces", "max_soma_area",
    "n_not_processed_soma_containing_meshes", "n_error_limbs",
    "n_same_soma_multi_touching_limbs", "n_multi_soma_touching_limbs",
    "n_somas", "n_limbs", "n_branches", "max_limb_n_branches", "skeletal_length",
    "max_limb_skeletal_length", "median_branch_length", "width_median", "width_no_spine_median",
    "width_90_perc", "width_no_spine_90_perc", "n_spines", "spine_density", "spines_per_branch",
    "skeletal_length_eligible", "n_spine_eligible_branches", "spine_density_eligible",
    "spines_per_branch_eligible", "total_spine_volume", "spine_volume_median",
    "spine_volume_density", "spine_volume_density_eligible", "spine_volume_per_branch_eligible",
)
UPSTREAM_ATTRIBUTES = {
    "n_not_processed_soma_containing_meshes": "not_processed_soma_containing_meshes",
    "n_error_limbs": "n_error_limbs",
    "n_same_soma_multi_touching_limbs": "same_soma_multi_touching_limbs",
    "n_multi_soma_touching_limbs": "multi_soma_touching_limbs",
}


def make_neuron(soma_count=2, short_axis=False):
    neuron = Neuron.__new__(Neuron)
    neuron._index = -1
    neuron.concept_network = nx.Graph()
    meshes = []
    for index in range(soma_count):
        extents = [1000, 1000, 1000] if index == 0 else [2000, 3000, 4000]
        mesh = trimesh.creation.box(extents=extents)
        soma = Soma(mesh, sdf=1, volume_ratio=1, volume=mesh.volume)
        neuron.concept_network.add_node(f"S{index}", data=soma)
        meshes.append(mesh)
    axes = ([[[1., 0., 0.], [5., 0., 0.]]],) if short_axis else (
        [[0., 0., 0.], [4000., 0., 0.]],
        [[4000., 0., 0.], [12000., 0., 0.]],
        [[0., 0., 0.], [0., 2000., 0.]],
    )
    limb_axes = [axes] if short_axis else [axes[:2], axes[2:]]
    branch_index = 0
    for limb_index, coordinates in enumerate(limb_axes):
        limb = Limb.__new__(Limb)
        limb._index = -1
        limb.concept_network = nx.Graph()
        limb.all_concept_network_data = [dict(starting_soma=0, starting_coordinate=np.asarray(coordinates[0]).reshape(-1, 3)[0])]
        for node, axis in enumerate(coordinates):
            mesh = trimesh.creation.box(extents=[100, 100, 100])
            branch = Branch(np.asarray(axis), mesh=mesh, width=100)
            branch.width_new = dict(median_mesh_center=[100, 300, 500][branch_index],
                                    no_spine_median_mesh_center=[80, 250, 450][branch_index])
            if short_axis:
                branch.spines = []
                branch.spines_volume = []
            elif branch_index == 0:
                spine = trimesh.creation.box(extents=[2, 3, 4])
                branch.spines = [spine, spine.copy()]
                branch.spines_volume = [spine.volume, spine.volume]
            elif branch_index == 1:
                branch.spines = []
                branch.spines_volume = []
            limb.concept_network.add_node(node, data=branch)
            meshes.append(mesh)
            branch_index += 1
        neuron.concept_network.add_node(f"L{limb_index}", data=limb)
        if soma_count:
            neuron.concept_network.add_edge("S0", f"L{limb_index}")
    neuron.mesh = trimesh.util.concatenate(meshes)
    neuron.spines_skeletal_length = np.asarray([] if short_axis else [100., 200.])
    return neuron


def enrich(neuron):
    neuron.not_processed_soma_containing_meshes = [object(), object()]
    neuron.n_error_limbs = np.int32(3)
    neuron.same_soma_multi_touching_limbs = [object()]
    neuron.multi_soma_touching_limbs = [object()] * 4
    return neuron


def expected_core():
    return dict(
        n_vertices=40, n_faces=60, max_soma_volume=24., max_soma_n_faces=12, max_soma_area=52.,
        n_somas=2, n_limbs=2, n_branches=3, max_limb_n_branches=2, skeletal_length=14000.,
        max_limb_skeletal_length=12000., median_branch_length=4000.,
        width_median=300., width_no_spine_median=250., width_90_perc=460., width_no_spine_90_perc=410.,
        n_spines=2, spine_density=2 / 14000, spines_per_branch=2 / 3,
        skeletal_length_eligible=12000., n_spine_eligible_branches=2,
        spine_density_eligible=2 / 12000, spines_per_branch_eligible=1.,
        total_spine_volume=48., spine_volume_median=24., spine_volume_density=48 / 14000,
        spine_volume_density_eligible=48 / 12000, spine_volume_per_branch_eligible=24.,
    )


def test_slim_neuron_summary_has_actual_geometry_width_and_spine_statistics():
    neuron = make_neuron()
    actual = neuron.neuron_stats(include_upstream_diagnostics=False)
    assert actual == expected_core()
    assert tuple(actual) == tuple(key for key in LEGACY_KEYS if key not in UPSTREAM_ATTRIBUTES)
    assert not any(hasattr(neuron, name) for name in UPSTREAM_ATTRIBUTES.values())


def test_enriched_default_retains_legacy_order_diagnostic_counts_and_scalar_types():
    actual = stats.neuron_stats(enrich(make_neuron()))
    expected = expected_core()
    expected.update(n_not_processed_soma_containing_meshes=2, n_error_limbs=np.int32(3),
                    n_same_soma_multi_touching_limbs=1, n_multi_soma_touching_limbs=4)
    assert actual == expected
    assert tuple(actual) == LEGACY_KEYS
    assert type(actual["n_error_limbs"]) is np.int32
    assert type(actual["n_branches"]) is np.int64
    assert type(actual["width_median"]) is np.float64


@pytest.mark.parametrize("key,attribute", UPSTREAM_ATTRIBUTES.items())
def test_requested_missing_diagnostics_raise_actionable_attribute_error(key, attribute):
    neuron = enrich(make_neuron())
    delattr(neuron, attribute)
    with pytest.raises(AttributeError, match=f"{key}.*{attribute}.*include_upstream_diagnostics=False"):
        stats.neuron_stats(neuron)


def test_disabled_diagnostic_getters_are_never_called(monkeypatch):
    def blocked(_):
        raise AssertionError("Upstream metadata must not be read")
    for attribute in UPSTREAM_ATTRIBUTES.values():
        monkeypatch.setattr(Neuron, attribute, property(blocked), raising=False)
    assert stats.neuron_stats(make_neuron(), include_upstream_diagnostics=False) == expected_core()


def test_ignore_list_bypasses_unavailable_diagnostic_getters_with_default_policy():
    actual = stats.neuron_stats(make_neuron(), stats_to_ignore=list(UPSTREAM_ATTRIBUTES))
    assert actual == expected_core()


def test_ignored_core_getters_are_not_evaluated(monkeypatch):
    def blocked(_):
        raise AssertionError("Ignored core statistic must not be read")
    for key in ("width_median", "spine_density"):
        monkeypatch.setattr(Neuron, key, property(blocked))
    actual = stats.neuron_stats(make_neuron(), stats_to_ignore=["width_median", "spine_density"],
                               include_upstream_diagnostics=False)
    expected = expected_core()
    del expected["width_median"], expected["spine_density"]
    assert actual == expected


@pytest.mark.parametrize("ignored,key", [(["typo"], "typo"),
                                         (["n_faces", "n_faces"], "n_faces"),
                                         (["centroid_x"], "centroid_x")])
def test_unknown_and_duplicate_ignore_names_retain_key_error(ignored, key):
    with pytest.raises(KeyError) as error:
        stats.neuron_stats(make_neuron(), stats_to_ignore=ignored, include_upstream_diagnostics=False)
    assert error.value.args == (key,)


@pytest.mark.parametrize("soma_count", [0, 1, 3])
def test_selected_soma_count_comes_from_actual_graph(soma_count):
    neuron = make_neuron(soma_count=soma_count)
    ignored = [key for key in LEGACY_KEYS if key != "n_somas"]
    assert neuron.neuron_stats(stats_to_ignore=ignored, include_upstream_diagnostics=False) == {"n_somas": soma_count}


def test_cell_type_mode_preserves_forced_switches_and_overrides_ignore_list(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("Cell type mode must disable skeletal features")
    monkeypatch.setattr(stats, "features_from_neuron_skeleton_and_soma_center", blocked)
    neuron = make_neuron()
    actual = stats.neuron_stats(neuron, stats_to_ignore=["n_faces"], cell_type_mode=True,
                               include_skeletal_stats=True, include_centroids=False,
                               voxel_adjustment_vector=[2, 3, 4])
    expected = expected_core()
    del expected["n_somas"], expected["spine_density"]
    expected.update(centroid_x_nm=0., centroid_y_nm=0., centroid_z_nm=0.,
                    centroid_x=0, centroid_y=0, centroid_z=0, centroid_volume=1.)
    assert actual == expected
    assert "n_faces" in actual


def test_centroid_requests_require_explicit_scale_but_plain_summary_does_not():
    neuron = make_neuron()
    assert stats.neuron_stats(neuron, include_upstream_diagnostics=False) == expected_core()
    with pytest.raises(ValueError, match="explicit voxel_adjustment_vector"):
        stats.neuron_stats(neuron, include_upstream_diagnostics=False, include_centroids=True)


def test_skeletal_statistics_work_without_unsupported_verbose_keyword():
    neuron = make_neuron(soma_count=1, short_axis=True)
    actual = stats.neuron_stats(neuron, include_upstream_diagnostics=False, include_skeletal_stats=True)
    assert actual["skeletal_length"] == 4.
    assert actual["branch_length_mean"] == actual["branch_length_median"] == 0.005
    assert actual["n_short_branches"] == 1
    assert actual["n_long_branches"] == actual["n_medium_branches"] == 0
    assert actual["bbox_x_min"] == 0.
    assert actual["bbox_x_max"] == 5.
    assert actual["bbox_volume"] == 0.


def test_ordinary_statistic_exceptions_are_propagated():
    expected = AttributeError("Specific property computation failed")
    fields = {key: value for key, value in expected_core().items() if key != "width_median"}
    class Enriched(SimpleNamespace):
        @property
        def width_median(self):
            raise expected
    with pytest.raises(AttributeError) as error:
        stats.neuron_stats(Enriched(**fields), include_upstream_diagnostics=False)
    assert error.value is expected


def test_extra_kwargs_remain_ignored():
    assert stats.neuron_stats(make_neuron(), include_upstream_diagnostics=False, legacy_unused="value") == expected_core()
