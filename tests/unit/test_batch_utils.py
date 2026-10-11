"""Shared provenance stays lightweight and invalidates changed source snapshots."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest


HELPERS_PATH = Path(__file__).resolve().parents[2]/'neurd/batch_utils.py'
spec = importlib.util.spec_from_file_location('batch_utils_under_test', HELPERS_PATH)
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)


def test_standalone_completion_helpers_do_not_import_neurd_or_numerical_backends(tmp_path):
    script = '''
import importlib.abc
import importlib.util
from pathlib import Path
import sys

class BlockNumericalImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'neurd', 'numpy', 'scipy', 'trimesh', 'mesh_tools'}:
            raise AssertionError('Unexpected heavy import: ' + fullname)

sys.meta_path.insert(0, BlockNumericalImports())
spec = importlib.util.spec_from_file_location('standalone_helpers', sys.argv[1])
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)
output = Path(sys.argv[2])
helpers.save(output/'value.json', {'finite': 1.0})
assert len(helpers.sha(output/'value.json')) == 64
assert len(helpers.digest_json({'finite': 1.0})) == 64
assert helpers.separate_roots(output/'source', output/'destination')
assert not helpers.complete_valid(output, 'run', {})
print('LIGHTWEIGHT_HELPERS_PASS')
'''
    result = subprocess.run(
        [sys.executable, '-c', script, str(HELPERS_PATH), str(tmp_path)],
        cwd=tmp_path, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'),
        capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'LIGHTWEIGHT_HELPERS_PASS' in result.stdout


@pytest.mark.parametrize('changed', [
    'neurd/batch_utils.py', 'neurd/science.py', 'process_all_neurons_parallel.py',
])
def test_raw_fingerprint_includes_shared_module_and_runner_source(tmp_path, monkeypatch, changed):
    from neurd.run_config import RunConfig

    for relative in ('process_all_neurons.py', 'process_all_neurons_parallel.py',
                     'neurd/batch_utils.py', 'neurd/science.py'):
        path = tmp_path/relative
        path.parent.mkdir(exist_ok=True)
        path.write_text('# source snapshot\n')
    monkeypatch.setattr(helpers, 'dependency_signature', lambda: {})
    monkeypatch.setattr(helpers.importlib.metadata, 'version', lambda name: 'fixture-version')
    monkeypatch.setattr(helpers.importlib.util, 'find_spec', lambda name: None)
    config = RunConfig()
    before = helpers.raw_run_signature(config, source_root=tmp_path)
    assert set(before['code_sha256']) == {
        'process_all_neurons.py', 'process_all_neurons_parallel.py',
        'neurd/batch_utils.py', 'neurd/science.py',
    }
    assert before['schema_version'] == 1
    assert before['decimation_ratio'] == 0.0 and before['max_somas'] is None
    assert before['config'] == config.as_dict()
    (tmp_path/changed).write_text('# changed source snapshot\n')
    after = helpers.raw_run_signature(config, source_root=tmp_path)
    assert helpers.digest_json(before) != helpers.digest_json(after)
    assert after['code_sha256'][changed] == helpers.sha(tmp_path/changed)
    assert before['dependencies'] == after['dependencies']
    assert before['runtime_environment'] == after['runtime_environment']


def test_installed_package_fingerprint_does_not_require_repository_runners(tmp_path, monkeypatch):
    from neurd.run_config import RunConfig

    package = tmp_path/'neurd'
    package.mkdir()
    (package/'batch_utils.py').write_text('# installed helper\n')
    (package/'segmentation_pipeline.py').write_text('# installed pipeline\n')
    monkeypatch.setattr(helpers, 'dependency_signature', lambda: {})
    monkeypatch.setattr(helpers.importlib.metadata, 'version', lambda name: 'fixture-version')
    monkeypatch.setattr(helpers.importlib.util, 'find_spec', lambda name: None)
    result = helpers.raw_run_signature(RunConfig(), source_root=tmp_path)
    assert set(result['code_sha256']) == {
        'neurd/batch_utils.py', 'neurd/segmentation_pipeline.py',
    }
    assert all(result['code_sha256'][str(path.relative_to(tmp_path))] == helpers.sha(path)
               for path in package.glob('*.py'))
