"""Missing attributes must behave normally, including before construction."""
import copy
import pickle
from types import SimpleNamespace

import pytest

from tests.unit import skip_if_datasci_tools_unusable

skip_if_datasci_tools_unusable()

from neurd.neuron import Neuron


@pytest.mark.parametrize('name', ['pipeline_products', 'missing_feature'])
def test_missing_attributes_without_pipeline_products(name):
    neuron = Neuron.__new__(Neuron)
    assert not hasattr(neuron, name)
    assert getattr(neuron, name, 'default') == 'default'
    with pytest.raises(AttributeError, match=name):
        getattr(neuron, name)


def test_optional_pipeline_product_delegation():
    neuron = Neuron.__new__(Neuron)
    neuron.pipeline_products = SimpleNamespace(stage_metric=12, segment_id=99)
    neuron.segment_id = 7
    assert neuron.stage_metric == 12
    assert neuron.segment_id == 7
    assert getattr(neuron, 'missing_feature', None) is None


def test_real_upstream_pipeline_products_remain_compatible():
    from datasci_tools.pipeline import PipelineProducts
    neuron = Neuron.__new__(Neuron)
    neuron.pipeline_products = PipelineProducts(decomposition={'stage_metric': 12})
    assert neuron.stage_metric == 12


@pytest.mark.parametrize('name', ['__missing__', '__setstate__'])
def test_dunder_attributes_are_not_delegated(name):
    neuron = Neuron.__new__(Neuron)
    neuron.pipeline_products = SimpleNamespace(**{name: 'stage value'})
    with pytest.raises(AttributeError, match=name):
        getattr(neuron, name)


def test_copy_and_pickle_without_pipeline_products():
    neuron = Neuron.__new__(Neuron)
    neuron.segment_id = 7
    neuron.preprocessed_data = {'metadata': [1, 2]}
    for restored in (copy.deepcopy(neuron), pickle.loads(pickle.dumps(neuron))):
        assert restored.segment_id == 7
        assert restored.preprocessed_data == neuron.preprocessed_data
        assert restored.preprocessed_data is not neuron.preprocessed_data
        assert not hasattr(restored, 'pipeline_products')
