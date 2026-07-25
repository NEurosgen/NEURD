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

# --- perf/compat patch registry ------------------------------------------------------
# Every patch below rewrites a function that lives in mesh_tools / datasci_tools, so a
# rename upstream makes it silently stop applying: the build still succeeds, just ~15 min
# slower and without the OFF-renumber repair. That failure mode is invisible, so each
# patch records its outcome here and a failure warns loudly instead of `pass`-ing.
#
#   PATCH_STATUS[name] is True  -> installed
#                         False -> tried and FAILED (a warning was emitted)
#                         None  -> deliberately skipped via its NEURD_* env flag
#
# tests/unit/test_perf_patches.py asserts nothing is False and that the fast paths are
# actually the ones bound. Read that test before renaming anything here.
import warnings as _warnings

PATCH_STATUS = {}


def _patch_ok(name):
    PATCH_STATUS[name] = True


def _patch_skipped(name):
    """Opted out on purpose through the patch's env flag -- not a failure."""
    PATCH_STATUS[name] = None


def _patch_failed(name, exc):
    PATCH_STATUS[name] = False
    _warnings.warn(
        f"neurd perf/compat patch {name!r} did NOT install "
        f"({type(exc).__name__}: {exc}). The pipeline still runs, but slower and without "
        f"this fix -- most likely the upstream mesh_tools/datasci_tools API moved.",
        RuntimeWarning,
        stacklevel=2,
    )


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
                # NEURD_POISSON_DEPTH: octree depth (default 11 = Docker; 9 = fast+good).
                # NEURD_POISSON_BACKEND: "meshlab" (default, Docker-accurate) or "open3d"
                #   (faster; subsamples point cloud, may over-segment thin neurites at depth=11
                #   but worth testing at depth 8-9 where coarsening already smooths gaps).
                # NEURD_POISSON_ITERS: Gauss-Seidel iters (meshlab only, default 8 = Docker).
                _depth = int(_os.environ.get("NEURD_POISSON_DEPTH", "11"))
                _backend = _os.environ.get("NEURD_POISSON_BACKEND", "meshlab")
                if _backend == "open3d":
                    mesh = _mo.poisson_surface_reconstruction(mesh, depth=_depth)
                else:
                    _iters = int(_os.environ.get("NEURD_POISSON_ITERS", "8"))
                    mesh = _mo.poisson_surface_reconstruction_meshlab(mesh, depth=_depth, iters=_iters)
                self.temp_folder_obj.mkdir(parents=True, exist_ok=True)
                mesh.export(str(output_obj))
            except Exception as _e:
                print(f"[real-poisson] failed, passthrough: {type(_e).__name__}: {_e}")
        return (mesh, output_obj) if return_mesh else output_obj

    _Poisson.__call__ = _poisson_inprocess_noop
    del _Poisson, _poisson_inprocess_noop
    _patch_ok("poisson_noop")
except Exception as _e:
    _patch_failed("poisson_noop", _e)

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
    _patch_ok("fillholes_noop")
except Exception as _e:
    _patch_failed("fillholes_noop", _e)

# datasci_tools.compressed_pickle: EVERY call reachable in a build is a debug dump-before-raise --
# it pickles the big curr_limb_mesh to disk right before an Exception that the caller CATCHES and
# swallows (neuron_utils.py:867, "keep the pre-refinement partition"). On a dense H01 neuron that
# fires on many limbs -> ~98 s (4.6%) of pure wasted bz2 writes (py-spy wall attribution). No
# production output uses compressed_pickle.
#
# NEURD's own dump sites have since been deleted outright; what remains are the calls inside
# mesh_tools.compartment_utils.resolve_empty_conflicting_face_labels (:1274/:1424), which is a
# LOCKED upstream path -- hence this patch is still needed. No-op it by default; set
# NEURD_ENABLE_COMPRESSED_PICKLE=1 to restore (e.g. fixture generation). The raise still fires, so
# behaviour is identical -- only the wasted dump is skipped.
try:
    import os as _os_cp
    if _os_cp.environ.get("NEURD_ENABLE_COMPRESSED_PICKLE") != "1":
        from datasci_tools import system_utils as _su_cp

        def _compressed_pickle_noop(obj, filename, return_size=False, *args, **kwargs):
            return 0 if return_size else None

        _su_cp.compressed_pickle = _compressed_pickle_noop
        del _su_cp, _compressed_pickle_noop
        _patch_ok("compressed_pickle_noop")
    else:
        _patch_skipped("compressed_pickle_noop")
except Exception as _e:
    _patch_failed("compressed_pickle_noop", _e)

# MeshLab Decimator (Quadric Edge Collapse) is real work, but each call forks
# xvfb+meshlabserver and round-trips the mesh through OFF. On the big H01 neuron the
# soma-extraction decimations (outer on the full ~1.6M-face mesh + per-Poisson inner)
# are ~13 calls ≈ 104 s of subprocess time (profile_cumulative: meshlab.py:375 __call__).
# open3d's quadric decimation is the same algorithm in-process; _mesh_ops.decimate is
# already fidelity-validated against meshlab (tests/integration/test_mesh_ops_fidelity.py)
# and the user does not need pristine soma quality. Replace __call__ in-process by default;
# set NEURD_MESHLAB_DECIMATE=1 to keep the original meshlab subprocess (A/B / fidelity check).
#
# decimation_ratio is baked into the .mls script (not stored on self), so __init__ is
# wrapped to also stash it on the instance for the in-process path. A 0 / >=1 ratio means
# "no decimation" in the meshlab script (TargetPerc), so we pass the mesh through unchanged.
try:
    import os as _os_dec
    if _os_dec.environ.get("NEURD_MESHLAB_DECIMATE") != "1":
        from mesh_tools.meshlab import Decimator as _Decimator

        _decimator_init_orig = _Decimator.__init__

        def _decimator_init_capture(self, decimation_ratio, temp_folder, overwrite=False, **kwargs):
            _decimator_init_orig(self, decimation_ratio, temp_folder, overwrite=overwrite, **kwargs)
            self._decimation_ratio = decimation_ratio

        def _decimate_inprocess(
            self, vertices=[], faces=[], segment_id=None, return_mesh=True,
            input_mesh_path="", mesh_filename="", printout=True,
            delete_temp_files=True, **kwargs,
        ):
            import random as _r
            import trimesh as _tm
            from neurd import _mesh_ops as _mo
            if segment_id is None:
                segment_id = _r.randint(100, 100000)
            if len(mesh_filename) <= 0:
                mesh_filename = f"neuron_{segment_id}.off"
            # output path mirrors the real wrapper ({input_stem}_decimated.off); only its
            # stem is consumed downstream for naming, not its bytes.
            output_obj = self.temp_folder_obj / f"{Path(mesh_filename).stem}_decimated.off"
            if len(vertices) == 0 and input_mesh_path:
                mesh = self.fetch_mesh_from_off(str(input_mesh_path))
            else:
                mesh = _tm.Trimesh(vertices=vertices, faces=faces, process=False)
            ratio = getattr(self, "_decimation_ratio", 0)
            # ratio 0 / >=1 == meshlab "no decimation"; otherwise quadric-decimate in-process.
            if 0 < ratio < 1:
                mesh = _mo.decimate(mesh, decimation_ratio=ratio)
            if not return_mesh:
                self.temp_folder_obj.mkdir(parents=True, exist_ok=True)
                mesh.export(str(output_obj))
                return output_obj
            return mesh, output_obj

        _Decimator.__init__ = _decimator_init_capture
        _Decimator.__call__ = _decimate_inprocess
        # NB: do NOT `del _decimator_init_orig` — the wrapped __init__ closes over it.
        _patch_ok("decimator_open3d")
    else:
        _patch_skipped("decimator_open3d")
except Exception as _e:
    _patch_failed("decimator_open3d", _e)

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
    _patch_ok("off_renumber_repair")
except Exception as _e:
    _patch_failed("off_renumber_repair", _e)

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
    _patch_ok("skeleton_kdtree_1d")
except Exception as _e:
    _patch_failed("skeleton_kdtree_1d", _e)

# trimesh_utils.vertex_components == [list(k) for k in nx.connected_components(
# mesh.vertex_adjacency_graph)]. Building that networkx graph (`add_edges_from` over
# edges_unique, one Python-level dict entry per edge) is ~205 s / 10% of a 187 MB H01 build's
# WALL (py-spy leaf attribution) -- the single biggest non-CGAL cost. It is reached only through
# `tu.split_by_vertices` (i.e. every `tu.split(connectivity="vertices")`), and 100% of that wall
# sits inside LOCKED mesh_tools call paths (resolve_face_labels 65 s, adaptive correspondence
# 65 s, skeleton_to_branches 51 s, spine shaft split 29 s), so no NEURD call-site migration can
# reach it -- patching the one function does.
#
# submesh_ops.vertex_components computes the identical partition with one scipy
# connected-components pass, and deliberately reproduces networkx's COMPONENT ORDER (first
# appearance in edges_unique), because split_by_vertices re-sorts pieces with the unstable
# np.flip(np.argsort(sizes)) whose tie order depends on input order. Gated byte-exact against the
# real networkx implementation, end-to-end through split_by_vertices, in
# tests/unit/test_vertex_components.py. Set NEURD_LEGACY_VERTEX_COMPONENTS=1 to restore networkx.
try:
    import os as _os_vc
    if _os_vc.environ.get("NEURD_LEGACY_VERTEX_COMPONENTS") != "1":
        from mesh_tools import trimesh_utils as _tu_vc

        def _vertex_components_scipy(mesh):
            from neurd import submesh_ops as _so_vc
            return _so_vc.vertex_components(mesh)

        _tu_vc.vertex_components = _vertex_components_scipy
        del _tu_vc, _vertex_components_scipy
        _patch_ok("vertex_components_scipy")
    else:
        _patch_skipped("vertex_components_scipy")
except Exception as _e:
    _patch_failed("vertex_components_scipy", _e)

# datasci_tools.numpy_utils.get_matching_vertices finds unordered vertex pairs within
# `equiv_distance` by building the FULL NxN coordinate distance matrix (pdist->squareform),
# then a copy of it, then an NxN np.eye -- 3 x O(N^2) allocations. On the skeleton node-combine
# path (skeleton_utils.py:1737, LOCKED mesh_tools) N is large, so this is ~25 GB of allocation
# churn on a big-H01 build (memray) and O(N^2) memory/time. The result is exactly a fixed-radius
# self-pairs query: cKDTree(points).query_pairs(equiv_distance) -- O(N log N) time, O(output)
# memory. Byte-exact vs the original over 1800 randomized cases (both ignore_diagonal modes,
# equiv_distance 0 / 0.02 / 0.5, with duplicate + jittered coords); patch the one datasci_tools
# function (mesh_tools calls it as nu.get_matching_vertices -> resolved at call time).
# Set NEURD_LEGACY_MATCHING_VERTICES=1 to restore the O(N^2) original.
try:
    import os as _os_mv
    if _os_mv.environ.get("NEURD_LEGACY_MATCHING_VERTICES") != "1":
        from datasci_tools import numpy_utils as _nu_mv

        def _get_matching_vertices_kdtree(possible_vertices, ignore_diagonal=True,
                                          equiv_distance=0, print_flag=False):
            import numpy as _np
            from scipy.spatial import cKDTree as _cKDTree
            pv = _np.asarray(possible_vertices).reshape(-1, 3)
            n = len(pv)
            if n < 2:
                return _np.empty((0, 2), dtype=_np.int64)
            pairs = _cKDTree(pv).query_pairs(r=equiv_distance,
                                             output_type="ndarray").astype(_np.int64)
            if not ignore_diagonal:
                # the original keeps self-pairs (i,i) when ignore_diagonal=False (dist 0 <= r).
                # No current caller uses this branch, but keep it exact.
                diag = _np.arange(n, dtype=_np.int64).reshape(-1, 1)
                diag = _np.hstack([diag, diag])
                pairs = _np.vstack([pairs, diag]) if len(pairs) else diag
            if len(pairs) == 0:
                return _np.empty((0, 2), dtype=_np.int64)
            # reproduce np.unique(np.sort(...,axis=1),axis=0): pairs are already i<j, sort lexicographically.
            return pairs[_np.lexsort((pairs[:, 1], pairs[:, 0]))]

        _nu_mv.get_matching_vertices = _get_matching_vertices_kdtree
        del _nu_mv, _get_matching_vertices_kdtree
        _patch_ok("matching_vertices_ckdtree")
    else:
        _patch_skipped("matching_vertices_ckdtree")
except Exception as _e:
    _patch_failed("matching_vertices_ckdtree", _e)

from .version import __version__

default_data_type = "microns"


def set_volume_params(
    volume=default_data_type,
    verbose=False,
    verbose_loop=False,
):
    """Switch the active parameter set ("microns" or "h01").

    Canonical entry point now that parameters live in `neurd.parameters`;
    delegates to the single explicit params object instead of monkey-patching
    `*_global` variables across every module.
    """
    from . import parameters
    parameters.params.use(volume)
