"""Native builds must find environment/system headers and fail explicitly."""
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys

import pytest
import setuptools


ROOT = Path(__file__).resolve().parents[2]
SETUPS = [ROOT/'cgal'/package/'setup.py' for package in ('cgal_segmentation','cgal_skeleton_param')]


@pytest.mark.parametrize('setup_path',SETUPS,ids=lambda path:path.parent.name)
@pytest.mark.parametrize('cgal,eigen',[
    ('/fake/env/include','/fake/env/include/eigen3'),
    ('/fake/base/include','/fake/base/include/eigen3'),
    ('/usr/include','/usr/include/eigen3'),
    ('/fake/env/include','/usr/include/eigen3'),
])
def test_native_setup_finds_env_base_and_system_headers(monkeypatch,setup_path,cgal,eigen):
    files = {cgal+'/CGAL/version.h',eigen+'/Eigen/Core'}
    captured = []
    monkeypatch.setattr(sys,'prefix','/fake/env')
    monkeypatch.setattr(sys,'base_prefix','/fake/base')
    monkeypatch.setattr(os.path,'isfile',lambda path:path in files)
    monkeypatch.setattr(setuptools,'setup',lambda **kwargs:captured.append(kwargs))
    runpy.run_path(str(setup_path))
    extension = captured[0]['ext_modules'][0]
    assert cgal in extension.include_dirs and eigen in extension.include_dirs
    assert extension.include_dirs[:2] == [cgal,eigen]
    assert '/fake/env/lib' in extension.library_dirs and '/fake/base/lib' in extension.library_dirs


@pytest.mark.parametrize('setup_path',SETUPS,ids=lambda path:path.parent.name)
@pytest.mark.parametrize('missing',['CGAL/version.h','Eigen/Core'])
def test_native_setup_explains_missing_headers(monkeypatch,setup_path,missing):
    monkeypatch.setattr(os.path,'isfile',lambda path:not path.endswith(missing))
    monkeypatch.setattr(setuptools,'setup',lambda **kwargs:pytest.fail('Missing headers must stop the build'))
    with pytest.raises(RuntimeError,match=f'Missing {missing}.*searched:.*libcgal-dev'):
        runpy.run_path(str(setup_path))


@pytest.mark.parametrize('setup_path',SETUPS,ids=lambda path:path.parent.name)
def test_native_setup_prefers_active_prefix_headers(monkeypatch,setup_path):
    captured = []
    monkeypatch.setattr(sys,'prefix','/fake/env')
    monkeypatch.setattr(sys,'base_prefix','/fake/base')
    monkeypatch.setattr(os.path,'isfile',lambda path:True)
    monkeypatch.setattr(setuptools,'setup',lambda **kwargs:captured.append(kwargs))
    runpy.run_path(str(setup_path))
    assert captured[0]['ext_modules'][0].include_dirs[0] == '/fake/env/include'


def installer_environment(tmp_path,headers=True,fail_backend=False,allow_missing=False,pure_python_backend=False):
    """Stub Python commands so installer policy runs without installing packages."""
    fake_bin = tmp_path/'venv/bin'
    fake_bin.mkdir(parents=True)
    python = fake_bin/'python'
    python.write_text(f'''#!{sys.executable}
import json, os, sys, types
with open(os.environ['INSTALL_LOG'],'a') as log:
    log.write(json.dumps(sys.argv[1:])+'\\n')
if sys.argv[1] == '-c':
    print('3.12')
elif sys.argv[1] == '-':
    payload = sys.stdin.read()
    if len(sys.argv) == 2:
        present = {{'/usr/include/CGAL/version.h','/usr/include/eigen3/Eigen/Core'}}
        os.path.isfile = lambda path:os.environ['FAKE_MISSING_HEADERS'] == '0' and path in present
    else:
        sys.argv = [sys.argv[0], *sys.argv[2:]]
        module = types.ModuleType(sys.argv[1])
        module.__file__ = '/fake/backend.py' if os.environ['FAKE_PURE_PYTHON_BACKEND'] == '1' else '/fake/backend.so'
        setattr(module,sys.argv[2],lambda:None)
        sys.modules[sys.argv[1]] = module
    exec(compile(payload,'installer-probe','exec'))
elif sys.argv[1:4] == ['-m','pip','install']:
    if os.environ['FAKE_FAILED_BACKEND'] == '1' and sys.argv[-1].endswith('cgal_skeleton_param'):
        sys.exit(1)
else:
    sys.exit('Unexpected installer command')
''')
    python.chmod(0o755)
    (fake_bin/'activate').write_text(f'export PATH="{fake_bin}:$PATH"\n')
    log = tmp_path/'commands.jsonl'
    env = dict(os.environ,PYTHON=str(python),VENV_DIR=str(fake_bin.parent),INSTALL_LOG=str(log),
               NEURD_ALLOW_MISSING_CGAL='1' if allow_missing else '0',
               FAKE_MISSING_HEADERS='0' if headers else '1',FAKE_FAILED_BACKEND='1' if fail_backend else '0',
               FAKE_PURE_PYTHON_BACKEND='1' if pure_python_backend else '0')
    result = subprocess.run(['bash',str(ROOT/'scripts/install_local.sh')],cwd=ROOT,env=env,
                            text=True,capture_output=True,timeout=20)
    return result,[json.loads(line) for line in log.read_text().splitlines()]


def test_installer_missing_headers_stops_before_package_installation(tmp_path):
    result,commands = installer_environment(tmp_path,headers=False)
    assert result.returncode != 0
    assert not any(command[:2] == ['-m','pip'] for command in commands)
    assert 'Both native CGAL backends are required' in result.stderr


def test_installer_failed_backend_stops_without_claiming_complete(tmp_path):
    result,commands = installer_environment(tmp_path,fail_backend=True)
    assert result.returncode != 0
    assert not any(command[-1] == './cgal/cgal_segmentation' for command in commands)
    assert 'failed to build or import' in result.stderr
    assert 'Install complete' not in result.stdout


@pytest.mark.parametrize('headers,fail_backend',[(False,False),(True,True)])
def test_installer_explicit_development_opt_out(tmp_path,headers,fail_backend):
    result,_ = installer_environment(tmp_path,headers=headers,fail_backend=fail_backend,allow_missing=True)
    assert result.returncode == 0,result.stderr
    assert 'Development install complete with missing native CGAL backends' in result.stderr
    assert 'both native CGAL backends verified' not in result.stdout


def test_installer_success_requires_verification_of_both_native_modules(tmp_path):
    result,commands = installer_environment(tmp_path)
    assert result.returncode == 0,result.stderr
    assert ['-','calcification_param_Module','calcification_param'] in commands
    assert ['-','cgal_Segmentation_Module','cgal_segmentation'] in commands
    assert 'both native CGAL backends verified' in result.stdout


def test_installer_rejects_python_fallback_even_when_build_reports_success(tmp_path):
    result,_ = installer_environment(tmp_path,pure_python_backend=True)
    assert result.returncode != 0
    assert 'is not a compiled extension' in result.stderr
    assert 'Install complete' not in result.stdout
