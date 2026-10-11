"""Decomposition products retain their structured result and optional storage."""
import copy
import pickle
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest
import trimesh

from tests.unit import skip_if_datasci_tools_unusable

skip_if_datasci_tools_unusable()

from datasci_tools.pipeline import PipelineProducts, StageProducts
from neurd import neuron_statistics as stats, neuron_utils as nru
from neurd.neuron import Branch, Limb, Neuron, Soma


DIAGNOSTICS = (
    "n_not_processed_soma_containing_meshes",
    "n_error_limbs",
    "n_same_soma_multi_touching_limbs",
    "n_multi_soma_touching_limbs",
)


class RecordingNeuron:
    def __init__(self):
        self.events = []
        self.axis = np.asarray([[[1., 2., 3.], [4., 5., 6.]]])

    def neuron_stats(self, **kwargs):
        self.events.append(("neuron_stats", kwargs))
        return dict(core=7, overlap=1)

    @property
    def skeleton(self):
        self.events.append(("skeleton",))
        return self.axis


def record_skeleton_stats(monkeypatch, neuron):
    def skeleton_stats(obj, include_centroids=True, voxel_adjustment_vector=None):
        assert obj is neuron
        neuron.events.append(("skeleton_stats", include_centroids, voxel_adjustment_vector))
        return dict(overlap=2, tail=9)
    monkeypatch.setattr(stats, "skeleton_stats_from_neuron_obj", skeleton_stats)


@pytest.mark.parametrize("centroids", [False, True])
@pytest.mark.parametrize("diagnostics", [False, True])
def test_flags_preserve_callback_order_result_keys_and_verbose_compatibility(monkeypatch, centroids, diagnostics):
    neuron = RecordingNeuron()
    record_skeleton_stats(monkeypatch, neuron)
    scale = [2, 3, 4]
    products = nru.calculate_decomposition_products(
        neuron, verbose=True, include_centroids=centroids,
        voxel_adjustment_vector=scale, include_upstream_diagnostics=diagnostics,
    )
    assert isinstance(products, StageProducts)
    assert products.skeleton is neuron.axis
    assert list(products.__dict__) == ["skeleton", "core", "overlap", "tail"]
    assert (products.core, products.overlap, products.tail) == (7, 2, 9)
    expected_kwargs = dict(stats_to_ignore=["max_soma_volume", "max_soma_n_faces"],
                           include_skeletal_stats=True, include_centroids=centroids,
                           voxel_adjustment_vector=scale)
    if not diagnostics:
        expected_kwargs["include_upstream_diagnostics"] = diagnostics
    assert list(neuron.events[0][1]) == list(expected_kwargs)
    assert neuron.events == [
        ("neuron_stats", expected_kwargs),
        ("skeleton",), ("skeleton_stats", centroids, scale),
    ]
    assert not hasattr(neuron, "pipeline_products")


def test_defaults_keep_enriched_diagnostics_and_centroid_requests(monkeypatch):
    neuron = RecordingNeuron()
    record_skeleton_stats(monkeypatch, neuron)
    nru.calculate_decomposition_products(neuron)
    initial = neuron.events[0][1]
    assert initial["include_centroids"] is True
    assert list(initial) == ["stats_to_ignore", "include_skeletal_stats", "include_centroids"]
    assert "include_upstream_diagnostics" not in initial
    assert "voxel_adjustment_vector" not in initial
    assert neuron.events[-1] == ("skeleton_stats", True, None)


def test_strict_legacy_neuron_stats_callback_retains_its_default_keyword_set(monkeypatch):
    class LegacyNeuron(RecordingNeuron):
        def neuron_stats(self, stats_to_ignore, include_skeletal_stats, include_centroids):
            self.events.append(("strict_neuron_stats", stats_to_ignore,
                                include_skeletal_stats, include_centroids))
            return dict(core=7, overlap=1)
    neuron = LegacyNeuron()
    record_skeleton_stats(monkeypatch, neuron)
    products = nru.calculate_decomposition_products(neuron, False, True)
    assert products.overlap == 2
    assert neuron.events[0] == ("strict_neuron_stats", ["max_soma_volume", "max_soma_n_faces"], True, True)


def test_neuron_wrapper_forwards_every_control(monkeypatch):
    neuron = Neuron.__new__(Neuron)
    received = []
    marker = object()
    def calculate(obj, **kwargs):
        received.append((obj, kwargs))
        return marker
    monkeypatch.setattr(nru, "calculate_decomposition_products", calculate)
    assert neuron.calculate_decomposition_products(
        True, include_centroids=False, voxel_adjustment_vector=[2, 3, 4],
        include_upstream_diagnostics=False, verbose=True,
    ) is marker
    assert received == [(neuron, dict(store_in_obj=True, include_centroids=False,
                                     voxel_adjustment_vector=[2, 3, 4],
                                     include_upstream_diagnostics=False, verbose=True))]


def test_requested_storage_initializes_missing_container_after_calculation(monkeypatch):
    neuron = RecordingNeuron()
    record_skeleton_stats(monkeypatch, neuron)
    products = nru.calculate_decomposition_products(neuron, store_in_obj=True)
    assert isinstance(neuron.pipeline_products, PipelineProducts)
    assert neuron.pipeline_products.stages == ["decomposition"]
    saved = neuron.pipeline_products.decomposition
    assert isinstance(saved, StageProducts)
    assert saved.skeleton is products.skeleton
    assert saved.core == products.core and saved.tail == products.tail


def test_existing_stage_and_other_stages_merge_in_the_same_container(monkeypatch):
    neuron = RecordingNeuron()
    record_skeleton_stats(monkeypatch, neuron)
    container = PipelineProducts(decomposition=dict(keep=11, overlap=-1), previous=dict(value=4))
    neuron.pipeline_products = container
    old_stage = container.decomposition
    previous_stage = container.previous
    products = nru.calculate_decomposition_products(neuron, store_in_obj=True)
    assert neuron.pipeline_products is container
    assert container.decomposition is old_stage
    assert container.previous is previous_stage and previous_stage.value == 4
    assert old_stage.keep == 11 and old_stage.overlap == products.overlap == 2


def test_compatible_custom_writer_receives_the_stageproducts_object(monkeypatch):
    neuron = RecordingNeuron()
    record_skeleton_stats(monkeypatch, neuron)
    received = []
    container = SimpleNamespace(set_stage_attrs=lambda value, *, stage: received.append((value, stage)))
    neuron.pipeline_products = container
    products = nru.calculate_decomposition_products(neuron, store_in_obj=True)
    assert neuron.pipeline_products is container
    assert received == [(products, "decomposition")]


@pytest.mark.parametrize("writer", [None, 3])
def test_present_incompatible_storage_raises_without_replacing_container(monkeypatch, writer):
    neuron = RecordingNeuron()
    record_skeleton_stats(monkeypatch, neuron)
    container = SimpleNamespace(set_stage_attrs=writer, existing=11)
    neuron.pipeline_products = container
    with pytest.raises(TypeError, match="callable set_stage_attrs"):
        nru.calculate_decomposition_products(neuron, store_in_obj=True)
    assert neuron.pipeline_products is container and container.existing == 11


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("failure", ["neuron_stats", "skeleton_stats"])
def test_calculation_failures_do_not_create_container_or_write_stages(monkeypatch, existing, failure):
    neuron = RecordingNeuron()
    record_skeleton_stats(monkeypatch, neuron)
    if existing:
        neuron.pipeline_products = PipelineProducts(decomposition=dict(keep=11), previous=dict(value=4))
        container = neuron.pipeline_products
        before = container.export()
    def fail(*args, **kwargs):
        raise RuntimeError("calculation failed")
    if failure == "neuron_stats":
        monkeypatch.setattr(neuron, "neuron_stats", fail)
    else:
        monkeypatch.setattr(stats, "skeleton_stats_from_neuron_obj", fail)
    with pytest.raises(RuntimeError, match="calculation failed"):
        nru.calculate_decomposition_products(neuron, store_in_obj=True)
    if existing:
        assert neuron.pipeline_products is container and container.export() == before
    else:
        assert not hasattr(neuron, "pipeline_products")


def real_neuron(somas=1, with_limb=True):
    neuron = Neuron.__new__(Neuron)
    neuron._index = -1
    neuron.mesh = trimesh.creation.box(extents=[2, 4, 6])
    neuron.concept_network = nx.Graph()
    for index in range(somas):
        mesh = trimesh.creation.box(extents=[2, 4, 6])
        mesh.vertices += [4.5 + index * 10, -8.5, 12.5]
        neuron.concept_network.add_node(f"S{index}", data=Soma(mesh, sdf=1, volume_ratio=1, volume=48))
    if with_limb:
        limb = Limb.__new__(Limb)
        limb._index = -1
        limb.concept_network = nx.Graph()
        axis = np.asarray([[[1., 0., 0.], [1001., 0., 0.]]])
        branch = Branch(axis, mesh=trimesh.creation.box(), width=6)
        branch.width_new = dict(median_mesh_center=6., no_spine_median_mesh_center=5.)
        limb.concept_network.add_node(0, data=branch)
        limb.all_concept_network_data = [dict(starting_soma=0, starting_coordinate=axis[0, 0])]
        neuron.concept_network.add_node("L0", data=limb)
        neuron.concept_network.add_edge("S0", "L0")
    return neuron


@pytest.mark.parametrize("somas,with_limb", [(1, False), (1, True), (2, True)])
def test_real_slim_neuron_products_without_diagnostics_or_centroids(somas, with_limb):
    neuron = real_neuron(somas, with_limb)
    products = neuron.calculate_decomposition_products(
        include_centroids=False, include_upstream_diagnostics=False,
    )
    assert products.n_somas == somas
    assert products.n_limbs == products.n_branches == int(with_limb)
    assert products.skeletal_length == 1000 * int(with_limb)
    assert products.n_spines == 0
    assert not any(key in products for key in DIAGNOSTICS)
    assert not any(key.startswith("centroid_") for key in products)
    assert not hasattr(neuron, "pipeline_products")
    if with_limb:
        np.testing.assert_array_equal(products.skeleton, neuron.skeleton)
    else:
        assert products.skeleton == []
        assert products.bbox_volume == 0


@pytest.mark.parametrize("somas", [1, 2])
def test_real_centroids_require_scale_and_retain_s0_reference(somas):
    neuron = real_neuron(somas)
    with pytest.raises(ValueError, match="explicit voxel_adjustment_vector"):
        neuron.calculate_decomposition_products(store_in_obj=True, include_upstream_diagnostics=False)
    assert not hasattr(neuron, "pipeline_products")
    products = neuron.calculate_decomposition_products(
        include_upstream_diagnostics=False, voxel_adjustment_vector=[2, 3, 4],
    )
    assert (products.centroid_x_nm, products.centroid_y_nm, products.centroid_z_nm) == (4.5, -8.5, 12.5)
    assert (products.centroid_x, products.centroid_y, products.centroid_z) == (2, -2, 3)
    assert products.centroid_volume == 48 / 1_000_000_000


def test_enriched_diagnostics_remain_real_counts_when_requested():
    neuron = real_neuron()
    neuron.not_processed_soma_containing_meshes = [object(), object()]
    neuron.n_error_limbs = 3
    neuron.same_soma_multi_touching_limbs = [4]
    neuron.multi_soma_touching_limbs = [5, 6]
    products = neuron.calculate_decomposition_products(include_centroids=False)
    assert tuple(getattr(products, key) for key in DIAGNOSTICS) == (2, 3, 1, 2)


def test_missing_requested_diagnostics_do_not_initialize_storage():
    neuron = real_neuron()
    with pytest.raises(AttributeError, match="include_upstream_diagnostics"):
        neuron.calculate_decomposition_products(store_in_obj=True, include_centroids=False)
    assert not hasattr(neuron, "pipeline_products")


def test_zero_soma_failure_does_not_fabricate_reference_or_initialize_storage():
    neuron = real_neuron(somas=0, with_limb=False)
    with pytest.raises(ValueError, match="zero-size array"):
        neuron.calculate_decomposition_products(
            store_in_obj=True, include_centroids=False, include_upstream_diagnostics=False,
        )
    assert not hasattr(neuron, "pipeline_products")


def test_stored_real_products_survive_copy_and_pickle():
    neuron = real_neuron()
    products = neuron.calculate_decomposition_products(
        store_in_obj=True, include_centroids=False, include_upstream_diagnostics=False,
    )
    for restored in (copy.deepcopy(neuron), pickle.loads(pickle.dumps(neuron))):
        assert isinstance(restored.pipeline_products, PipelineProducts)
        assert restored.pipeline_products is not neuron.pipeline_products
        assert restored.pipeline_products.decomposition.n_somas == 1
        assert restored.n_branches == 1
        np.testing.assert_array_equal(restored.pipeline_products.decomposition.skeleton, products.skeleton)
