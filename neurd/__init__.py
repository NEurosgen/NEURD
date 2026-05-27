# Compatibility shim: datasci_tools uses np.float_ / np.int_ / np.complex_ which
# were removed in numpy 2. Restore them as aliases for their canonical dtypes
# before importing datasci_tools. Safe no-op on numpy<2 (attributes already exist).
import numpy as _np
if not hasattr(_np, 'float_'):
    _np.float_ = _np.float64
if not hasattr(_np, 'int_'):
    _np.int_ = _np.int64
if not hasattr(_np, 'complex_'):
    _np.complex_ = _np.complex128
del _np

# Stub optional visualization / cloud deps so that modules which do top-level
# `import ipyvolume` / submodule imports (e.g. `from ipyvolume.moviemaker import
# MovieMaker` in datasci_tools, `import ipyvolume as ipv` in mesh_tools) can be
# loaded without the real packages installed. A meta-path finder intercepts any
# `<pkg>` or `<pkg>.*` import and returns a stub module whose attributes are
# all None. If the real package is installed, it wins (importlib checks
# sys.modules and earlier finders first), so this is a no-op in that case.
import sys as _sys
import importlib.util as _ilutil
from importlib.abc import Loader as _Loader, MetaPathFinder as _MPF
from types import ModuleType as _ModuleType

_STUBBED_ROOTS = ('ipyvolume', 'cloudvolume')

class _StubModule(_ModuleType):
    def __getattr__(self, name):
        return None

class _StubLoader(_Loader):
    def create_module(self, spec):
        mod = _StubModule(spec.name)
        mod.__path__ = []  # mark as package so submodule imports work
        return mod
    def exec_module(self, module):
        pass

class _StubFinder(_MPF):
    def find_spec(self, fullname, path, target=None):
        root = fullname.split('.', 1)[0]
        if root in _STUBBED_ROOTS:
            # Only stub if the real package is genuinely missing.
            real = _ilutil.find_spec(root) if fullname == root else None
            if fullname == root and real is not None and not isinstance(
                _sys.modules.get(root), _StubModule
            ):
                return None
            return _ilutil.spec_from_loader(fullname, _StubLoader(), is_package=True)
        return None

_sys.meta_path.append(_StubFinder())
del _sys, _ilutil, _Loader, _MPF, _ModuleType

from pathlib import Path

from datasci_tools import module_utils as modu

from .version import __version__

default_data_type = "microns"


def set_volume_params(
    volume=default_data_type,
    verbose=False,
    verbose_loop=False,
):
    directory = Path(f"{__file__}").parents[0]
    modu.all_modules_set_global_parameters_and_attributes(
        data_type=volume,
        directory=directory,
        verbose=verbose,
        verbose_loop=verbose_loop,
        from_package="neurd",
    )
