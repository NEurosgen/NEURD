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
    import process_all_neurons as batch
    from neurd.h01_batch import digest_json
    from neurd.run_config import RunConfig

    mesh = tmp_path/'neuron.off'
    mesh.write_text('fixture')
    signature = dict(code_sha256={},dependencies=dict(skeletonizer=dict(sha256='native-binary')),
                     runtime_environment=dict(environment={'NEURD_REAL_POISSON':'1'},import_patches={}))
    monkeypatch.setattr(batch,'_run_signature',lambda *args:signature)
    monkeypatch.setattr(gate.subprocess,'check_output',lambda args,**kw:'head' if args[1]=='rev-parse' else '')
    receipt = gate._provenance(mesh,RunConfig(),1.0)
    assert receipt['dependencies']['skeletonizer']['sha256'] == 'native-binary'
    assert receipt['runtime_environment']['environment']['NEURD_REAL_POISSON'] == '1'
    assert 'tests/tools/neuron_gate.py' in receipt['source_sha256']
    assert 'tests/tools/neuron_metrics.py' in receipt['source_sha256']
    assert receipt['run_hash'] == digest_json(receipt['provenance'])
