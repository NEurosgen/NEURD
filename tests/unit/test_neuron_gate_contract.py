"""The whole-neuron gate compares matching modes and records raw-run provenance."""
import importlib.util
from pathlib import Path

import pytest

from tests.unit import skip_if_datasci_tools_unusable

skip_if_datasci_tools_unusable()

spec = importlib.util.spec_from_file_location(
    'neuron_gate_under_test',Path(__file__).resolve().parents[1]/'tools/neuron_gate.py')
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


@pytest.mark.parametrize('baseline_has_spines',[False,True])
def test_tolerance_rejects_mismatched_spine_modes(baseline_has_spines):
    common = {key:1 for key, (kind,_) in gate._TOL.items() if kind != 'skip'}
    with_spines = dict(common,n_spines_total=0)
    baseline,current = (with_spines,common) if baseline_has_spines else (common,with_spines)
    assert any('spine modes differ' in message for message in gate._compare_tolerant(baseline,current))
    assert gate._compare_tolerant(with_spines,with_spines) == []


def test_gate_reuses_native_and_environment_provenance_even_without_spines(monkeypatch,tmp_path):
    import copy
    import process_all_neurons as batch
    from neurd.h01_batch import digest_json
    from neurd.run_config import RunConfig

    mesh = tmp_path/'neuron.off'
    mesh.write_text('fixture')
    signature = dict(code_sha256={},dependencies=dict(skeletonizer=dict(sha256='native-binary')),
                     runtime_environment=dict(environment={'NEURD_REAL_POISSON':'1'},import_patches={}))
    monkeypatch.setattr(batch,'_run_signature',lambda *args:copy.deepcopy(signature))
    monkeypatch.setattr(gate.subprocess,'check_output',lambda args,**kw:'head' if args[1]=='rev-parse' else '')
    receipt = gate._provenance(mesh,RunConfig(),1.0)
    assert receipt['dependencies']['skeletonizer']['sha256'] == 'native-binary'
    assert receipt['runtime_environment']['environment']['NEURD_REAL_POISSON'] == '1'
    assert 'tests/tools/neuron_gate.py' in receipt['source_sha256']
    assert 'tests/tools/neuron_metrics.py' in receipt['source_sha256']
    assert receipt['run_hash'] == digest_json(receipt['provenance'])
    seeded = gate._provenance(mesh, RunConfig(), 1.0, seed=123)
    assert seeded['random_seed'] == seeded['provenance']['gate_random_seed'] == 123
    assert seeded['run_hash'] != receipt['run_hash']


def test_gate_applies_seed_after_reader_and_before_pipeline(monkeypatch, tmp_path):
    import numpy as np
    from types import SimpleNamespace

    mesh = tmp_path/'neuron.off'
    mesh.write_text('fixture')
    private = tmp_path/'work'
    private.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(gate.tempfile, 'mkdtemp', lambda **kwargs:str(private))
    monkeypatch.setattr(gate.sys, 'argv', ['neuron_gate', '--neuron', str(mesh), '--seed', '123'])
    previous = np.random.get_state()
    observed = []
    def reader(path):
        np.random.seed(987)
        np.random.random(999)  # Loading cannot determine the pipeline RNG state.
        return SimpleNamespace(vertices=[0], faces=[0])
    def pipeline(*args, **kwargs):
        observed.append(np.random.randint(1_000_000))
        return 'neuron'
    monkeypatch.setattr(gate, 'load_mesh', reader)
    monkeypatch.setattr(gate, 'segmentation_pipeline', pipeline)
    monkeypatch.setattr(gate, 'extract_metrics', lambda *args, **kwargs:{})
    try:
        gate.main()
        assert observed == [np.random.RandomState(123).randint(1_000_000)]
    finally:
        np.random.set_state(previous)
