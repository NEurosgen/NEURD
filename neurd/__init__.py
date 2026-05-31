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
# np.in1d was removed in numpy 2 (use np.isin). Restore on the numpy module so
# direct `np.in1d` calls resolve to the isin replacement.
if not hasattr(_np, 'in1d'):
    _np.in1d = _np.isin
del _np

# datasci_tools.numpy_dep does `from numpy import *`, which in numpy 2 does NOT
# bind `in1d` (removed from numpy.__all__), so setting it on the numpy module alone
# is not enough. Set it directly on numpy_dep so callers like
# numpy_utils.intersect_indices (uses ndep.in1d) work.
try:
    import numpy as _np_in1d
    from datasci_tools import numpy_dep as _ndep_in1d
    if not hasattr(_ndep_in1d, 'in1d'):
        _ndep_in1d.in1d = _np_in1d.isin
    del _np_in1d, _ndep_in1d
except Exception:
    pass

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
        if root not in _STUBBED_ROOTS:
            return None
        import importlib.util as _ilu_local
        import importlib.machinery as _ilm_local
        # Prefer the real package if it is actually installed. Use PathFinder (the
        # sys.path-based finder) rather than importlib.util.find_spec, which would
        # re-invoke meta_path — including this finder — and recurse.
        if fullname == root and _ilm_local.PathFinder.find_spec(root) is not None:
            return None
        return _ilu_local.spec_from_loader(fullname, _StubLoader(), is_package=True)

_sys.meta_path.append(_StubFinder())
del _sys, _ilutil, _Loader, _MPF, _ModuleType

# Register a pure-Python stand-in for the compiled CGAL extension
# `cgal_Segmentation_Module` BEFORE mesh_tools.trimesh_utils is imported (it does
# `import cgal_Segmentation_Module as csm` at module load). The original was built
# inside the removed Docker image; the stub keeps the segmentation pipeline runnable
# on a plain env. A genuinely installed extension wins (we only register if absent).
import sys as _sys2
import importlib.util as _ilu2
if _ilu2.find_spec('cgal_Segmentation_Module') is None and \
        'cgal_Segmentation_Module' not in _sys2.modules:
    from . import _cgal_segmentation as _cgal_seg_stub
    _sys2.modules['cgal_Segmentation_Module'] = _cgal_seg_stub
    del _cgal_seg_stub
del _ilu2, _sys2

# The reimerlab dotmotif fork eagerly imports its Neo4jExecutor at `import dotmotif`,
# which drags in heavy Neo4j-only deps (py2neo, tamarind, dask) that NEURD never uses —
# its graph filters run on the NetworkX executor. Pre-register a stub Neo4jExecutor
# module so dotmotif's `from .Neo4jExecutor import Neo4jExecutor` resolves to the stub and
# skips that whole import chain. (Also sidesteps the upstream `raise e(...)` bug in
# datasci_tools.dotmotif_utils that fires on the resulting ImportError.)
import sys as _sys3
from types import ModuleType as _MT3
if 'dotmotif.executors.Neo4jExecutor' not in _sys3.modules:
    _neo4j_stub = _MT3('dotmotif.executors.Neo4jExecutor')
    _neo4j_stub.Neo4jExecutor = type('Neo4jExecutor', (object,), {})
    _sys3.modules['dotmotif.executors.Neo4jExecutor'] = _neo4j_stub
    del _neo4j_stub
del _sys3, _MT3

from pathlib import Path

from datasci_tools import module_utils as modu

# MeshLab Poisson is a silent NO-OP on MeshLabServer 2020.09 here, but costs ~60 s of
# subprocess+xvfb spawn per call (~9 calls ≈ 536 s — the single biggest cost in the
# pipeline, see PIPELINE.md / profile). The Screened Poisson filter does not actually
# reconstruct on this build: captured (input, output) pairs are BYTE-IDENTICAL (vertices
# and faces unchanged), and the wrapper returns the input mesh. Soma detection works
# anyway — the sphere validator / volume ratio are robust to non-watertight meshes — so
# Poisson never contributed to the result. (A prior fix patching its script filters to
# emit <filter>/<Param> XML did not make the filter run; verified still a no-op.)
#
# Replicate that EXACT behavior in-process: skip the subprocess, return the input mesh.
# This is byte-for-byte behavior-preserving (validated by test_segmentation_pipeline)
# and removes ~536 s. A real watertight reconstruction (open3d) would CHANGE the soma
# detector's input and is intentionally NOT done here — it is a separate quality
# experiment to evaluate against a soma baseline.
try:
    from mesh_tools.meshlab import Poisson as _Poisson

    def _poisson_inprocess_noop(
        self, vertices=[], faces=[], segment_id=None, return_mesh=True,
        input_mesh_path="", mesh_filename="", printout=True,
        delete_temp_files=True, **kwargs,
    ):
        import os as _os
        import random as _r
        import trimesh as _tm
        if segment_id is None:
            segment_id = _r.randint(100, 100000)
        if len(mesh_filename) <= 0:
            mesh_filename = f"neuron_{segment_id}.off"
        # output path mirrors the real wrapper ({input_stem}_poisson.off); only its
        # `.stem` is consumed downstream (for naming the next temp file), not its bytes.
        output_obj = self.temp_folder_obj / f"{Path(mesh_filename).stem}_poisson.off"
        if len(vertices) == 0 and input_mesh_path:
            mesh = self.fetch_mesh_from_off(str(input_mesh_path))
        else:
            mesh = _tm.Trimesh(vertices=vertices, faces=faces, process=False)
        # EXPERIMENT (NEURD_REAL_POISSON=1): restore a real watertight reconstruction
        # (open3d) in place of the no-op, to test whether the original Docker pipeline's
        # working MeshLab Poisson is what kept fragmented H01 limb meshes connected
        # (some H01 neurons crash in correspondence_1_to_1 on a disconnected limb mesh).
        # Off by default -> identical to the no-op; on -> reconstructs + writes to disk
        # so return_mesh=False callers read the reconstructed bytes.
        if _os.environ.get("NEURD_REAL_POISSON") == "1":
            try:
                from neurd import _mesh_ops as _mo
                mesh = _mo.poisson_surface_reconstruction_meshlab(mesh)
                self.temp_folder_obj.mkdir(parents=True, exist_ok=True)
                mesh.export(str(output_obj))
            except Exception as _e:
                print(f"[real-poisson] failed, passthrough: {type(_e).__name__}: {_e}")
        return (mesh, output_obj) if return_mesh else output_obj

    _Poisson.__call__ = _poisson_inprocess_noop
    del _Poisson, _poisson_inprocess_noop
except Exception:
    pass

# MeshLab FillHoles is likewise BROKEN on this build: the Close-Holes script fails
# (returncode 255, "filter Remove Faces from Non Manifold Edges not found"), and every
# caller already catches that and continues with the unmodified mesh (e.g.
# trimesh_utils.remove_mesh_interior: `try: mesh = fill_holes(mesh) except: continue`,
# and soma_volume_ratio's fill-holes branch). So FillHoles is effectively a no-op that
# still spawns xvfb+meshlabserver. Replace its __call__ with an in-process pass-through
# returning the input mesh — behavior-preserving (caller keeps the original mesh either
# way), no subprocess. (A real hole-fill, e.g. trimesh/pymeshfix, would CHANGE the mesh
# and is intentionally not done here.)
try:
    from mesh_tools.meshlab import FillHoles as _FillHoles

    def _fillholes_inprocess_noop(
        self, vertices=[], faces=[], segment_id=None, return_mesh=True,
        input_mesh_path="", mesh_filename="", printout=True,
        delete_temp_files=True, **kwargs,
    ):
        import random as _r
        import trimesh as _tm
        if segment_id is None:
            segment_id = _r.randint(100, 100000)
        if len(mesh_filename) <= 0:
            mesh_filename = f"neuron_{segment_id}.off"
        fname = getattr(self, "filter_name", "fill_holes")
        output_obj = self.temp_folder_obj / f"{Path(mesh_filename).stem}_{fname}.off"
        if len(vertices) == 0 and input_mesh_path:
            mesh = self.fetch_mesh_from_off(str(input_mesh_path))
        else:
            mesh = _tm.Trimesh(vertices=vertices, faces=faces, process=False)
        return (mesh, output_obj) if return_mesh else output_obj

    _FillHoles.__call__ = _fillholes_inprocess_noop
    del _FillHoles, _fillholes_inprocess_noop
except Exception:
    pass

# MeshLabServer 2020.09 OFF-exporter bug: after a vertex-deleting filter (e.g. the
# interior-removal chain in remove_mesh_interior), it writes the compacted vertex
# block (only surviving vertices) but leaves the face indices in the ORIGINAL
# numbering. trimesh then reads a self-inconsistent mesh (faces index past the
# vertex array) which later crashes in vertex_faces / split_by_vertices with
# "axis 0 index N exceeds matrix dimension M". The surviving vertices are written
# in ascending original-index order and are exactly the referenced set, so we can
# renumber the faces self-contained: map each original index to its position in
# sorted(unique(faces)). Self-guarding: only triggers when faces reference beyond
# the vertex count, so well-formed output (incl. vertex-ADDING filters like Poisson)
# is returned untouched.
try:
    import numpy as _np_ml
    import trimesh as _trimesh_ml
    from mesh_tools import trimesh_utils as _tu_ml
    from mesh_tools.meshlab import Meshlab as _Meshlab

    def _fetch_mesh_from_off_repaired(mesh_path):
        _mesh_obj = Path(mesh_path).absolute()
        if not _mesh_obj.exists():
            raise FileNotFoundError('Mesh file missing')
        if _mesh_obj.suffix != '.off':
            raise TypeError('mesh path must be to an .off file')
        _mesh = _tu_ml.load_mesh_no_processing(_mesh_obj)
        _faces = _np_ml.asarray(_mesh.faces)
        if len(_faces) and _faces.max() >= len(_mesh.vertices):
            _uniq = _np_ml.unique(_faces)
            if len(_uniq) == len(_mesh.vertices):
                _new_faces = _np_ml.searchsorted(_uniq, _faces)
                _mesh = _trimesh_ml.Trimesh(
                    vertices=_np_ml.asarray(_mesh.vertices),
                    faces=_new_faces,
                    process=False,
                )
        return _mesh

    _Meshlab.fetch_mesh_from_off = staticmethod(_fetch_mesh_from_off_repaired)
    del _Meshlab, _fetch_mesh_from_off_repaired
except Exception:
    pass

# Newer pykdtree strictly requires 2D data_pts, but upstream mesh_tools.skeleton_utils
# builds KDTrees from 1D cumulative skeletal-distance arrays (e.g. in
# coordinates_from_downstream_dist) -> "data_pts array should have exactly 2 dimensions".
# Wrap skeleton_utils.KDTree so 1D inputs are reshaped to (N,1); 2D inputs pass through
# unchanged, so all other (vertex-array) KDTree uses behave identically.
try:
    import numpy as _np_kd
    from mesh_tools import skeleton_utils as _sku
    _BaseKDTree_sk = _sku.KDTree

    class _KDTree2DTolerant:
        def __init__(self, data, *a, **k):
            data = _np_kd.asarray(data)
            self._ndim1 = data.ndim == 1
            if self._ndim1:
                data = _np_kd.ascontiguousarray(data.reshape(-1, 1), dtype=_np_kd.float64)
            self._tree = _BaseKDTree_sk(data, *a, **k)

        def query(self, pts, *a, **k):
            pts = _np_kd.asarray(pts)
            if self._ndim1 and pts.ndim == 1:
                pts = _np_kd.ascontiguousarray(pts.reshape(-1, 1), dtype=_np_kd.float64)
            return self._tree.query(pts, *a, **k)

        def __getattr__(self, name):
            return getattr(self._tree, name)

    _sku.KDTree = _KDTree2DTolerant
    del _sku
except Exception:
    pass

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
