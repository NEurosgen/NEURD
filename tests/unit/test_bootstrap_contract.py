"""Initial-import contracts with isolated providers and no installed-provider mutation."""

import builtins
import importlib.abc
import importlib.machinery
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace
import warnings

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "neurd/_bootstrap.py"
PATCHES = (
    "poisson_noop", "fillholes_noop", "compressed_pickle_noop", "decimator_open3d",
    "off_renumber_repair", "skeleton_kdtree_1d", "vertex_components_scipy",
    "matching_vertices_ckdtree",
)
OPT_OUTS = {
    "NEURD_ENABLE_COMPRESSED_PICKLE": "compressed_pickle_noop",
    "NEURD_MESHLAB_DECIMATE": "decimator_open3d",
    "NEURD_LEGACY_VERTEX_COMPONENTS": "vertex_components_scipy",
    "NEURD_LEGACY_MATCHING_VERTICES": "matching_vertices_ckdtree",
}


class _NumpyProvider(ModuleType):
    def __getattr__(self, name):
        if name in {"float_", "int_", "complex_", "in1d"}:
            raise AttributeError(name)
        return getattr(np, name)


class _Providers:
    """Run actual bootstrap source against fresh modules in a fresh namespace."""

    def __init__(self, *, environment=None, cgal="absent", fail_poisson=False):
        self.events = []
        self.calls = []
        self.real_roots = set()
        self.numpy = _NumpyProvider("numpy")
        self.numpy_dep = ModuleType("datasci_tools.numpy_dep")
        self.system_utils = ModuleType("datasci_tools.system_utils")
        self.system_utils.compressed_pickle = self._original
        self.numpy_utils = ModuleType("datasci_tools.numpy_utils")
        self.numpy_utils.get_matching_vertices = self._original
        self.trimesh_utils = ModuleType("mesh_tools.trimesh_utils")
        self.trimesh_utils.vertex_components = self._original
        self.trimesh_utils.load_mesh_no_processing = self._original
        calls = self.calls

        class Meshlab:
            @staticmethod
            def fetch_mesh_from_off(path):
                return path

        class Poisson(Meshlab):
            __call__ = staticmethod(_Providers._original)

        class FillHoles(Meshlab):
            __call__ = staticmethod(_Providers._original)

        class Decimator(Meshlab):
            __call__ = staticmethod(_Providers._original)

            def __init__(self, decimation_ratio, temp_folder, overwrite=False, **kwargs):
                calls.append(("decimator_init", decimation_ratio, temp_folder, overwrite, kwargs))
                self.temp_folder_obj = Path(temp_folder)

        class KDTree:
            def __init__(self, data, *args, **kwargs):
                self.data = data
                self.tag = "original-tree"
                calls.append(("tree_init", data, args, kwargs))

            def query(self, points, *args, **kwargs):
                calls.append(("tree_query", points, args, kwargs))
                return "query-result"

        self.meshlab = ModuleType("mesh_tools.meshlab")
        for cls in (Meshlab, Poisson, FillHoles, Decimator):
            setattr(self.meshlab, cls.__name__, cls)
        self.original_decimator_init = Decimator.__init__
        self.original_tree = KDTree
        self.skeleton_utils = ModuleType("mesh_tools.skeleton_utils")
        self.skeleton_utils.KDTree = KDTree
        self.modules = {
            "numpy": self.numpy,
            "datasci_tools.numpy_dep": self.numpy_dep,
            "datasci_tools.system_utils": self.system_utils,
            "datasci_tools.numpy_utils": self.numpy_utils,
            "mesh_tools.meshlab": self.meshlab,
            "mesh_tools.trimesh_utils": self.trimesh_utils,
            "mesh_tools.skeleton_utils": self.skeleton_utils,
            "os": SimpleNamespace(environ=dict(environment or {})),
        }
        for root in ("datasci_tools", "mesh_tools"):
            package = ModuleType(root)
            for name, module in self.modules.items():
                if name.startswith(root + "."):
                    setattr(package, name.split(".")[1], module)
            self.modules[root] = package
        self.package = ModuleType("neurd")
        self.package._cgal_segmentation = ModuleType("neurd._cgal_segmentation")
        self.package.submesh_ops = SimpleNamespace(vertex_components=self._vertex_components)
        self.modules["neurd"] = self.package
        self.system = SimpleNamespace(modules={}, meta_path=[])
        self.modules["sys"] = self.system
        native_spec = importlib.machinery.ModuleSpec("cgal_Segmentation_Module", loader=None)
        if cgal == "preloaded":
            self.cgal_provider = ModuleType("cgal_Segmentation_Module")
            self.cgal_provider.__spec__ = native_spec
            self.system.modules["cgal_Segmentation_Module"] = self.cgal_provider
        else:
            self.cgal_provider = None

        def find_spec(name):
            self.events.append(("find_spec", name))
            assert name == "cgal_Segmentation_Module"
            return native_spec if cgal in {"installed", "preloaded"} else None

        def find_real_root(name):
            self.events.append(("find_real_root", name))
            return object() if name in self.real_roots else None

        util = SimpleNamespace(find_spec=find_spec, spec_from_loader=importlib.util.spec_from_loader)
        machinery = SimpleNamespace(PathFinder=SimpleNamespace(find_spec=find_real_root))
        self.modules.update({
            "importlib": SimpleNamespace(util=util, machinery=machinery),
            "importlib.util": util,
            "importlib.machinery": machinery,
        })
        self.fail_poisson = fail_poisson

    @staticmethod
    def _original(*args, **kwargs):
        return "upstream-result"

    def _vertex_components(self, mesh):
        self.calls.append(("vertex_components", mesh))
        return "component-result"

    def _import(self, name, globals=None, locals=None, fromlist=(), level=0):
        self.events.append(("import", name, tuple(fromlist or ()), level))
        if self.fail_poisson and name == "mesh_tools.meshlab" and "Poisson" in fromlist:
            raise ImportError("controlled Poisson failure")
        if name == "" and level == 1:
            assert tuple(fromlist) == ("_cgal_segmentation",)
            return self.package
        if name in self.modules:
            return self.modules[name] if fromlist else self.modules[name.split(".")[0]]
        return builtins.__import__(name, globals, locals, fromlist, level)

    def execute(self, source=None, *, module_name="neurd._bootstrap"):
        namespace = {
            "__name__": module_name, "__package__": "neurd",
            "__builtins__": dict(vars(builtins), __import__=self._import),
        }
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            exec(compile(source or SOURCE.read_text(), str(SOURCE), "exec"), namespace)
        self.namespace = namespace
        self.warnings = caught
        return namespace


def test_default_installation_order_and_numpy_aliases():
    providers = _Providers()
    state = providers.execute()
    assert list(state["PATCH_STATUS"].items()) == [(name, True) for name in PATCHES]
    assert not providers.warnings
    assert providers.numpy.float_ is np.float64
    assert providers.numpy.int_ is np.int64
    assert providers.numpy.complex_ is np.complex128
    assert providers.numpy.in1d is providers.numpy_dep.in1d is np.isin
    assert providers.meshlab.Poisson.__call__.__name__ == "_poisson_inprocess_noop"
    assert providers.meshlab.FillHoles.__call__.__name__ == "_fillholes_inprocess_noop"
    assert providers.meshlab.Decimator.__call__.__name__ == "_decimate_inprocess"
    assert providers.meshlab.Meshlab.fetch_mesh_from_off.__name__ == "_fetch_mesh_from_off_repaired"
    assert providers.system_utils.compressed_pickle(object(), "unused", return_size=True) == 0


@pytest.mark.parametrize("flag,patch", OPT_OUTS.items())
@pytest.mark.parametrize("value", ["0", "1"])
def test_only_exact_one_opts_out(flag, patch, value):
    providers = _Providers(environment={flag: value})
    state = providers.execute()
    expected = [(name, None if name == patch and value == "1" else True) for name in PATCHES]
    assert list(state["PATCH_STATUS"].items()) == expected
    assert not providers.warnings
    binding = {
        "compressed_pickle_noop": providers.system_utils.compressed_pickle,
        "decimator_open3d": providers.meshlab.Decimator.__call__,
        "vertex_components_scipy": providers.trimesh_utils.vertex_components,
        "matching_vertices_ckdtree": providers.numpy_utils.get_matching_vertices,
    }[patch]
    assert (binding is _Providers._original) == (value == "1")


def test_failed_patch_warns_and_later_patches_install():
    providers = _Providers(fail_poisson=True)
    state = providers.execute()
    assert list(state["PATCH_STATUS"].items()) == [
        (name, False if name == "poisson_noop" else True) for name in PATCHES
    ]
    assert providers.meshlab.Poisson.__call__ is _Providers._original
    assert len(providers.warnings) == 1
    warning = providers.warnings[0]
    assert warning.category is RuntimeWarning
    assert str(warning.message) == (
        "neurd perf/compat patch 'poisson_noop' did NOT install "
        "(ImportError: controlled Poisson failure). The pipeline still runs, but slower and without "
        "this fix -- most likely the upstream mesh_tools/datasci_tools API moved."
    )
    # stacklevel=2 points to the failed installer rather than the warning helper.
    lines = SOURCE.read_text().splitlines()
    assert warning.filename == str(SOURCE)
    assert lines[warning.lineno - 1].strip() == '_patch_failed("poisson_noop", _e)'


@pytest.mark.parametrize("cgal", ["absent", "installed", "preloaded"])
def test_cgal_provider_selection_precedes_mesh_tools(cgal):
    providers = _Providers(cgal=cgal)
    providers.execute()
    find_index = providers.events.index(("find_spec", "cgal_Segmentation_Module"))
    mesh_index = next(i for i, event in enumerate(providers.events)
                      if event[:2] == ("import", "mesh_tools.meshlab"))
    assert find_index < mesh_index
    fallback = ("import", "", ("_cgal_segmentation",), 1)
    assert (fallback in providers.events) == (cgal == "absent")
    if cgal == "absent":
        assert providers.events.index(fallback) < mesh_index
        assert providers.system.modules["cgal_Segmentation_Module"] is providers.package._cgal_segmentation
    elif cgal == "preloaded":
        assert providers.system.modules["cgal_Segmentation_Module"] is providers.cgal_provider
    else:
        assert "cgal_Segmentation_Module" not in providers.system.modules
    assert "dotmotif.executors.Neo4jExecutor" in providers.system.modules


@pytest.mark.parametrize("root", ["ipyvolume", "cloudvolume"])
def test_stub_finder_missing_root_and_real_path_precedence(root):
    providers = _Providers()
    state = providers.execute()
    assert len(providers.system.meta_path) == 1
    finder = providers.system.meta_path[0]
    assert isinstance(finder, state["_StubFinder"])
    assert finder.find_spec("unrelated", None) is None
    for fullname in (root, root + ".submodule"):
        spec = finder.find_spec(fullname, None)
        module = spec.loader.create_module(spec)
        spec.loader.exec_module(module)
        assert isinstance(module, state["_StubModule"])
        assert module.__name__ == fullname
        assert module.__path__ == []
        assert module.unknown_attribute is None
    providers.real_roots.add(root)
    assert finder.find_spec(root, None) is None


@pytest.mark.parametrize("root", ["ipyvolume", "cloudvolume"])
def test_preloaded_optional_package_wins_without_finder_call(root, monkeypatch):
    providers = _Providers()
    state = providers.execute()
    finder = providers.system.meta_path[0]
    real = ModuleType(root)
    real.__version__ = "controlled-real-package"
    monkeypatch.setitem(sys.modules, root, real)
    monkeypatch.setattr(sys, "meta_path", [finder, *sys.meta_path])
    before = list(providers.events)
    assert importlib.import_module(root) is real
    assert providers.events == before
    assert not isinstance(real, state["_StubModule"])


@pytest.mark.parametrize("root", ["ipyvolume", "cloudvolume"])
def test_real_optional_package_and_submodule_on_disk_win(root, tmp_path, monkeypatch):
    package = tmp_path / root
    package.mkdir()
    (package / "__init__.py").write_text('marker = "real-package"\n')
    (package / "controlled_contract_submodule.py").write_text('marker = "real-submodule"\n')
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, root, raising=False)
    monkeypatch.delitem(sys.modules, root + ".controlled_contract_submodule", raising=False)
    providers = _Providers()
    providers.modules["importlib"].machinery = importlib.machinery
    providers.modules["importlib.machinery"] = importlib.machinery
    state = providers.execute()
    finder = providers.system.meta_path[0]
    monkeypatch.setattr(sys, "meta_path", [*sys.meta_path, finder])
    real = importlib.import_module(root)
    submodule = importlib.import_module(root + ".controlled_contract_submodule")
    assert real.marker == "real-package"
    assert submodule.marker == "real-submodule"
    assert not isinstance(real, state["_StubModule"])
    assert not isinstance(submodule, state["_StubModule"])


def test_delayed_wrappers_keep_original_providers_and_forward_arguments():
    providers = _Providers()
    state = providers.execute()
    assert state["_decimator_init_orig"] is providers.original_decimator_init
    decimator = providers.meshlab.Decimator(0.25, "controlled-temp", overwrite=True, extra=7)
    assert decimator._decimation_ratio == 0.25
    assert providers.calls == [("decimator_init", 0.25, "controlled-temp", True, {"extra": 7})]
    assert state["_BaseKDTree_sk"] is providers.original_tree
    tree = providers.skeleton_utils.KDTree([0, 2, 4], "constructor-arg", leafsize=9)
    assert tree.tag == "original-tree"
    assert tree.query([2.1], "query-arg", k=2) == "query-result"
    np.testing.assert_array_equal(providers.calls[1][1], [[0], [2], [4]])
    np.testing.assert_array_equal(providers.calls[2][1], [[2.1]])
    assert providers.calls[1][1].dtype == providers.calls[2][1].dtype == np.float64
    assert providers.calls[1][2:] == (("constructor-arg",), {"leafsize": 9})
    assert providers.calls[2][2:] == (("query-arg",), {"k": 2})
    points = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    tree_2d = providers.skeleton_utils.KDTree(points)
    assert tree_2d._tree.data is points
    assert tree_2d.query(points) == "query-result"
    assert providers.calls[-1][1] is points
    mesh = object()
    assert providers.trimesh_utils.vertex_components(mesh) == "component-result"
    assert providers.calls[-1] == ("vertex_components", mesh)


def test_package_exposes_same_registry_and_helpers():
    import neurd
    from neurd import _bootstrap

    for name in ("PATCH_STATUS", "_patch_ok", "_patch_skipped", "_patch_failed", "Path",
                 "_STUBBED_ROOTS", "_StubModule", "_StubLoader", "_StubFinder"):
        assert getattr(neurd, name) is getattr(_bootstrap, name)
    for name in ("_decimator_init_orig", "_decimator_init_capture", "_decimate_inprocess",
                 "_BaseKDTree_sk", "_KDTree2DTolerant"):
        assert hasattr(neurd, name) == hasattr(_bootstrap, name)
        if hasattr(_bootstrap, name):
            assert getattr(neurd, name) is getattr(_bootstrap, name)
    assert importlib.import_module("neurd") is neurd
    assert importlib.import_module("neurd._bootstrap") is _bootstrap


def test_fresh_import_respects_all_opt_outs_and_normal_module_cache():
    script = """
import importlib
import neurd
from datasci_tools import numpy_utils, system_utils
from mesh_tools import meshlab, trimesh_utils
from neurd import _bootstrap
assert neurd.PATCH_STATUS is _bootstrap.PATCH_STATUS
assert all(neurd.PATCH_STATUS[name] is None for name in (
    'compressed_pickle_noop', 'decimator_open3d',
    'vertex_components_scipy', 'matching_vertices_ckdtree'))
assert all(neurd.PATCH_STATUS[name] is True for name in (
    'poisson_noop', 'fillholes_noop', 'off_renumber_repair', 'skeleton_kdtree_1d'))
for function in (system_utils.compressed_pickle, meshlab.Decimator.__call__,
                 trimesh_utils.vertex_components, numpy_utils.get_matching_vertices):
    assert not function.__module__.startswith('neurd')
assert meshlab.Poisson.__call__.__module__ == 'neurd._bootstrap'
assert importlib.import_module('neurd') is neurd
assert importlib.import_module('neurd._bootstrap') is _bootstrap
print('BOOTSTRAP_INITIAL_IMPORT_PASS')
"""
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    environment.update({flag: "1" for flag in OPT_OUTS})
    result = subprocess.run([sys.executable, "-c", script], cwd=ROOT, env=environment,
                            capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "BOOTSTRAP_INITIAL_IMPORT_PASS" in result.stdout
