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


def installer_environment(tmp_path,headers=True,fail_backend=False,allow_missing=False,pure_python_backend=False,
                          repo=ROOT,cwd=None,venv_setting=None,create_venv=False,
                          requested_version='3.12',active_version='3.12'):
    """Stub Python commands so installer policy runs without installing packages."""
    if repo != ROOT:
        (repo/'scripts').mkdir(parents=True)
        (repo/'scripts/install_local.sh').write_bytes((ROOT/'scripts/install_local.sh').read_bytes())
    if venv_setting is None:
        venv_setting = str(tmp_path/'venv')
    venv_path = Path(venv_setting) if venv_setting else Path('.venv')
    if not venv_path.is_absolute():
        venv_path = repo/venv_path
    fake_bin = venv_path/'bin'
    requested_python = tmp_path/'requested_python'
    fake_python = f'''#!{sys.executable}
import json, os, pathlib, shlex, sys, types
with open(os.environ['INSTALL_LOG'],'a') as log:
    log.write(json.dumps({{'argv':sys.argv[1:],'cwd':os.getcwd(),'executable':sys.argv[0]}})+'\\n')
if sys.argv[1] == '-c':
    version_key = 'FAKE_REQUESTED_VERSION' if sys.argv[0] == os.environ['FAKE_REQUESTED_PYTHON'] else 'FAKE_ACTIVE_VERSION'
    print(os.environ[version_key])
elif sys.argv[1:3] == ['-m','venv']:
    bin_path = pathlib.Path(sys.argv[3])/'bin'
    bin_path.mkdir(parents=True)
    python = bin_path/'python'
    python.write_bytes(pathlib.Path(__file__).read_bytes())
    python.chmod(0o755)
    (bin_path/'activate').write_text('export PATH='+shlex.quote(str(bin_path))+':"$PATH"\\n')
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
'''
    requested_python.write_text(fake_python)
    requested_python.chmod(0o755)
    if not create_venv:
        fake_bin.mkdir(parents=True)
        python = fake_bin/'python'
        python.write_text(fake_python)
        python.chmod(0o755)
        (fake_bin/'activate').write_text(f'export PATH="{fake_bin}:$PATH"\n')
    log = tmp_path/'commands.jsonl'
    env = dict(os.environ,PYTHON=str(requested_python),INSTALL_LOG=str(log),
               NEURD_ALLOW_MISSING_CGAL='1' if allow_missing else '0',
               FAKE_MISSING_HEADERS='0' if headers else '1',FAKE_FAILED_BACKEND='1' if fail_backend else '0',
               FAKE_PURE_PYTHON_BACKEND='1' if pure_python_backend else '0',
               FAKE_REQUESTED_PYTHON=str(requested_python),FAKE_REQUESTED_VERSION=requested_version,
               FAKE_ACTIVE_VERSION=active_version)
    if venv_setting:
        env['VENV_DIR'] = venv_setting
    else:
        env.pop('VENV_DIR',None)
    result = subprocess.run(['bash',str(repo/'scripts/install_local.sh')],cwd=cwd or repo,env=env,
                            text=True,capture_output=True,timeout=20)
    return result,[json.loads(line)['argv'] for line in log.read_text().splitlines()]


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


@pytest.mark.parametrize('venv_kind',['default','relative','absolute'])
def test_installer_resolves_repository_and_venv_from_unrelated_directory(tmp_path,venv_kind):
    repo = tmp_path/'source tree'
    cwd = tmp_path/'unrelated directory'
    cwd.mkdir()
    settings = {'default':'','relative':'local environments/neurd','absolute':str(tmp_path/'external environment')}
    setting = settings[venv_kind]
    expected_venv = Path(setting) if venv_kind == 'absolute' else repo/(setting or '.venv')
    result,commands = installer_environment(tmp_path,repo=repo,cwd=cwd,venv_setting=setting,create_venv=True)
    assert result.returncode == 0,result.stderr
    assert ['-m','venv',str(expected_venv)] in commands
    records = [json.loads(line) for line in (tmp_path/'commands.jsonl').read_text().splitlines()]
    installs = [record for record in records if record['argv'][:3] == ['-m','pip','install']]
    assert len(installs) == 6
    assert all(record['cwd'] == str(repo) for record in installs)
    assert all(record['executable'] == str(expected_venv/'bin/python') for record in installs)
    assert ['-m','pip','install','-r','requirements.txt'] in commands
    assert ['-m','pip','install','-e','.[dev]'] in commands
    assert ['-m','pip','install','./cgal/cgal_skeleton_param'] in commands
    assert ['-m','pip','install','./cgal/cgal_segmentation'] in commands
    assert f'source {expected_venv}/bin/activate' in result.stdout
    assert not (cwd/'.venv').exists()


@pytest.mark.parametrize('active_version,allow_missing',[('3.9',False),('3.13',True)])
def test_installer_rejects_unsupported_reused_environment_before_installation(tmp_path,active_version,allow_missing):
    result,commands = installer_environment(tmp_path,requested_version='3.12',active_version=active_version,
                                           allow_missing=allow_missing)
    assert result.returncode != 0
    assert len(commands) == 2
    assert all(command[0] == '-c' for command in commands)
    assert f'Activated environment requires Python 3.10-3.12 (got {active_version})' in result.stderr
    assert 'Choose a new VENV_DIR' in result.stderr
    assert 'Install complete' not in result.stdout


@pytest.mark.parametrize('active_version',['3.10','3.11','3.12'])
def test_installer_accepts_supported_reused_environment(tmp_path,active_version):
    result,commands = installer_environment(tmp_path,requested_version='3.12',active_version=active_version)
    assert result.returncode == 0,result.stderr
    assert ['-','calcification_param_Module','calcification_param'] in commands
    assert ['-','cgal_Segmentation_Module','cgal_segmentation'] in commands


@pytest.mark.parametrize('requested_version',['3.9','3.13'])
def test_installer_preserves_requested_interpreter_version_guard(tmp_path,requested_version):
    result,commands = installer_environment(tmp_path,requested_version=requested_version,active_version='3.12')
    assert result.returncode != 0
    assert len(commands) == 1
    assert f'requires Python 3.10-3.12 (got {requested_version})' in result.stdout
