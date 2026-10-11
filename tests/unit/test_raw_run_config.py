"""Shared API policies and spine/width ordering, without heavy decomposition."""
import importlib
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.unit import skip_if_datasci_tools_unusable

skip_if_datasci_tools_unusable()

from neurd import neuron, parameters
from neurd.run_config import RunConfig

pipeline = importlib.import_module('neurd.segmentation_pipeline')
config_module = importlib.import_module('neurd.run_config')


@pytest.mark.parametrize('options', [
    dict(dataset='unknown'), dict(calculate_spines=1), dict(refinement='unknown'),
    dict(refinement='h01_current'),
    dict(calculate_spines=True, refinement='h01_current'),
    dict(sdf_profile='unknown'), dict(sdf_rays=0), dict(sdf_rays=-1),
    dict(sdf_rays=True), dict(sdf_rays=12.5), dict(sdf_rays=2**31),
])
def test_invalid_settings_fail_before_processing(options):
    with pytest.raises(ValueError):
        RunConfig(**options)


def test_ray_profiles_are_explicit_and_custom_counts_are_recorded():
    assert RunConfig().sdf_rays == 25
    assert RunConfig(sdf_profile='fast').sdf_rays == 12
    assert RunConfig(sdf_rays=16).as_dict()['sdf_profile'] == 'custom'


@pytest.mark.parametrize('policy,filename', [
    ('h01_current', 'h01_spines_current.json'),
    ('h01_smooth_heads', 'h01_spines_smooth_heads.json'),
])
def test_named_policies_match_existing_experiment_configs(policy, filename):
    from neurd.spine_refinement import RefinementConfig
    path = Path(__file__).resolve().parents[2] / 'configs' / filename
    expected = json.loads(json.dumps(asdict(RefinementConfig(**json.loads(path.read_text())))))
    actual = RunConfig(dataset='h01', calculate_spines=True, refinement=policy).as_dict()
    assert json.loads(json.dumps(actual['refinement_config'])) == expected


def test_api_preserves_selected_dataset_and_default_spines_are_off(monkeypatch):
    previous = parameters.params.data_type
    parameters.params.use('h01')
    monkeypatch.setenv('NEURD_SDF_RAYS', '999')
    calls = []
    def build(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace()
    monkeypatch.setattr(pipeline.neuron, 'Neuron', build)
    try:
        result = pipeline.segmentation_pipeline('mesh', segment_id=7, max_somas=1)
        assert calls[0]['calculate_spines'] is False
        assert calls[0]['spine_kwargs']['geometric_refinement'] is False
        assert result.run_config['dataset'] == 'h01'
        assert result.run_config['sdf_rays'] == 25
        assert parameters.params.data_type == 'h01'
    finally:
        parameters.params.use(previous)


def test_configured_api_passes_refinement_before_neuron_widths(monkeypatch):
    monkeypatch.setattr(config_module, 'require_compiled_cgal', lambda: '/native.so')
    captured = []
    monkeypatch.setattr(pipeline.neuron, 'Neuron', lambda **kwargs: captured.append(kwargs) or SimpleNamespace())
    config = RunConfig(dataset='h01', calculate_spines=True, refinement='h01_smooth_heads', sdf_profile='fast')
    result = pipeline.segmentation_pipeline('mesh', config=config)
    assert captured[0]['calculate_spines'] is True
    assert captured[0]['spine_kwargs']['geometric_refinement'] is True
    assert captured[0]['spine_kwargs']['geometric_refinement_config'].screen_smooth_incomplete_heads
    assert result.run_config['sdf_rays'] == 12
    with pytest.raises(ValueError, match='either'):
        pipeline.segmentation_pipeline('mesh', config=config, calculate_spines=False)


def test_python_backend_fails_before_any_neuron_construction(monkeypatch):
    monkeypatch.setattr(config_module.importlib, 'import_module', lambda name: SimpleNamespace(__file__='/fallback.py'))
    monkeypatch.setattr(pipeline.neuron, 'Neuron', lambda **kwargs: pytest.fail('Neuron must not start'))
    with pytest.raises(RuntimeError, match='compiled CGAL'):
        pipeline.segmentation_pipeline('mesh', calculate_spines=True)


def test_refinement_kwargs_preserve_spines_then_widths_then_cleanup(monkeypatch):
    events = []
    obj = SimpleNamespace(n_limbs=1, spines_already_computed=lambda: False)
    monkeypatch.setattr(neuron.nru, 'apply_adaptive_mesh_correspondence_to_neuron', lambda obj: events.append('correspondence'))
    monkeypatch.setattr(neuron.spu, 'calculate_spines_on_neuron', lambda obj, **kw: events.append(('spines', kw)))
    monkeypatch.setattr(neuron.wu, 'calculate_new_width_for_neuron_obj', lambda obj, **kw: events.append('widths'))
    monkeypatch.setattr(neuron.nru, 'clean_neuron_all_concept_network_data', lambda obj: events.append('cleanup'))
    monkeypatch.setattr(neuron.nru, 'recalculate_endpoints_and_order_skeletons_over_neuron', lambda obj: events.append('endpoints'))
    monkeypatch.setattr(neuron.bu, 'set_branches_endpoints_upstream_downstream_idx', lambda obj: None)
    neuron._refine_neuron(obj, calculate_spines=True, spine_kwargs={'geometric_refinement': True})
    assert events == ['correspondence', ('spines', {'geometric_refinement': True}), 'widths', 'cleanup', 'endpoints']
