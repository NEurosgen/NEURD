"""Unit tests for neurd.volume_utils — the abstract DataInterface base."""
import pytest

from tests.unit.leaves import skip_if_datasci_tools_unusable

skip_if_datasci_tools_unusable()

from neurd import volume_utils as vu


class _Dummy(vu.DataInterface):
    def align_array(self): ...
    def align_mesh(self): ...
    def align_skeleton(self): ...
    def align_neuron_obj(self): ...
    def unalign_neuron_obj(self): ...
    def segment_id_to_synapse_dict(self, **kw): ...


def test_data_interface_is_abstract():
    with pytest.raises(TypeError):
        vu.DataInterface(source="x")


def test_data_interface_init_and_set_synapse_filepath():
    d = _Dummy(source="src", voxel_to_nm_scaling=(4, 4, 40))
    assert d.source == "src"
    assert d.voxel_to_nm_scaling == (4, 4, 40)
    assert d.synapse_filepath is None
    d.set_synapse_filepath("/tmp/s.csv")
    assert d.synapse_filepath == "/tmp/s.csv"


def test_nuclei_from_segment_id_default_returns_pair_of_none():
    d = _Dummy(source="src")
    ids, centers = d.nuclei_from_segment_id(segment_id=123)
    assert ids is None and centers is None


def test_nuclei_classification_info_default_dict_shape():
    d = _Dummy(source="src")
    info = d.nuclei_classification_info_from_nucleus_id(nuclei=1)
    assert set(info) == {
        "external_cell_type",
        "external_cell_type_n_nuc",
        "external_cell_type_fine",
        "external_cell_type_fine_n_nuc",
        "external_cell_type_fine_e_i",
    }
    assert all(v is None for v in info.values())


def test_nuclei_from_segment_id_return_centers_false():
    """Regression: previously raised NameError because of `nucleus_ids` typo."""
    d = _Dummy(source="src")
    assert d.nuclei_from_segment_id(segment_id=123, return_centers=False) is None
