"""The library exporter does not activate the batch runner's process policy."""
import os
from pathlib import Path
import subprocess
import sys


def test_library_export_has_no_cli_thread_or_process_context_side_effects(tmp_path):
    script = '''
import importlib.abc
import json
import multiprocessing
import os
from pathlib import Path
import sys
from types import SimpleNamespace

threads = ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'OMP_NUM_THREADS')
before_environment = {name: os.environ.get(name) for name in threads}
before_start_method = multiprocessing.get_start_method(allow_none=True)

class BlockRunnerImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {'process_all_neurons', 'process_all_neurons_parallel'}:
            raise AssertionError('Library export imported a CLI runner: ' + fullname)

sys.meta_path.insert(0, BlockRunnerImports())
def unexpected_context(*args, **kwargs):
    raise AssertionError('Library export created a multiprocessing context')
multiprocessing.get_context = unexpected_context

from neurd.neuron_export import save_segmentation
output = Path(sys.argv[1])
neuron = SimpleNamespace(limbs=[], concept_network=None,
                         run_config={'dataset': 'h01', 'calculate_spines': False})
save_segmentation(neuron, output)
summary = json.loads((output / 'summary.json').read_text())
assert summary['data_type'] == 'h01'
assert summary['run_config'] == neuron.run_config
assert summary['n_limbs'] == summary['total_spines'] == 0
assert summary['soma_present'] is False
assert sorted(path.name for path in output.iterdir()) == ['summary.json']
assert {name: os.environ.get(name) for name in threads} == before_environment
assert multiprocessing.get_start_method(allow_none=True) == before_start_method
assert 'process_all_neurons' not in sys.modules
assert 'process_all_neurons_parallel' not in sys.modules
print('PUBLIC_EXPORT_PASS')
'''
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    for name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'OMP_NUM_THREADS'):
        env.pop(name, None)
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, '-c', script, str(tmp_path / 'export')],
        cwd=root, env=env, capture_output=True, text=True, timeout=45,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'PUBLIC_EXPORT_PASS' in result.stdout
