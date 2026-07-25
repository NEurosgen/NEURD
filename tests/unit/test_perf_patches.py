"""The import-time perf/compat patches must actually install.

`neurd/__init__.py` rewrites functions that live in `mesh_tools` / `datasci_tools`.
If one of those moves or is renamed upstream, the patch stops applying and *nothing
breaks* -- the build still succeeds, just ~15 min slower on a big H01 neuron and
without the OFF-renumber repair. That is the failure mode this file exists to catch.

It is not covered by the algorithm tests: `test_vertex_components.py` and
`test_matching_vertices.py` inline the ORIGINAL implementation and compare it against
the module-level function, so if the patch never installed they would compare the
original against itself and pass green.

Here we assert the registry has no failures AND that the fast implementations are the
ones actually bound at the call sites.
"""
import os
import subprocess
import sys

import pytest

from tests.unit import skip_if_datasci_tools_unusable

skip_if_datasci_tools_unusable()

import neurd  # noqa: E402

# Every patch neurd/__init__.py is expected to register. Adding a patch without adding
# it here (or vice versa) fails test_registry_covers_exactly_the_known_patches.
EXPECTED_PATCHES = {
    "poisson_noop",
    "fillholes_noop",
    "compressed_pickle_noop",
    "decimator_open3d",
    "off_renumber_repair",
    "skeleton_kdtree_1d",
    "vertex_components_scipy",
    "matching_vertices_ckdtree",
}

# The env flag that opts each patch out, for the ones that have one.
OPT_OUT_FLAG = {
    "compressed_pickle_noop": "NEURD_ENABLE_COMPRESSED_PICKLE",
    "decimator_open3d": "NEURD_MESHLAB_DECIMATE",
    "vertex_components_scipy": "NEURD_LEGACY_VERTEX_COMPONENTS",
    "matching_vertices_ckdtree": "NEURD_LEGACY_MATCHING_VERTICES",
}


def test_no_patch_failed():
    failed = sorted(k for k, v in neurd.PATCH_STATUS.items() if v is False)
    assert not failed, (
        f"perf/compat patches failed to install: {failed}. The pipeline still runs but "
        f"slower; check whether the upstream mesh_tools/datasci_tools API moved."
    )


def test_registry_covers_exactly_the_known_patches():
    assert set(neurd.PATCH_STATUS) == EXPECTED_PATCHES


@pytest.mark.parametrize("name", sorted(EXPECTED_PATCHES))
def test_patch_installed_unless_opted_out(name):
    """Every patch is installed, unless its env flag was set for this run."""
    status = neurd.PATCH_STATUS[name]
    flag = OPT_OUT_FLAG.get(name)
    if status is None:
        assert flag is not None and os.environ.get(flag) is not None, (
            f"{name} reports 'skipped' but no opt-out flag is set for it"
        )
        pytest.skip(f"{name} opted out via {flag}")
    assert status is True


def test_vertex_components_is_the_scipy_one():
    from mesh_tools import trimesh_utils as tu
    if neurd.PATCH_STATUS["vertex_components_scipy"] is None:
        pytest.skip("opted out via NEURD_LEGACY_VERTEX_COMPONENTS")
    # the patch binds a closure defined in neurd/__init__.py that delegates to submesh_ops
    assert tu.vertex_components.__module__.startswith("neurd"), (
        f"tu.vertex_components is {tu.vertex_components!r}, not the neurd scipy version -- "
        f"the ~205s networkx vertex-graph is back"
    )


def test_matching_vertices_is_the_kdtree_one():
    from datasci_tools import numpy_utils as nu
    if neurd.PATCH_STATUS["matching_vertices_ckdtree"] is None:
        pytest.skip("opted out via NEURD_LEGACY_MATCHING_VERTICES")
    assert nu.get_matching_vertices.__module__.startswith("neurd"), (
        f"nu.get_matching_vertices is {nu.get_matching_vertices!r}, not the cKDTree "
        f"version -- the O(N^2) distance matrix is back"
    )


def test_meshlab_wrappers_are_in_process():
    """Poisson / FillHoles / Decimator must not be the subprocess versions."""
    from mesh_tools import meshlab
    for cls, expected in (
        (meshlab.Poisson, "_poisson_inprocess_noop"),
        (meshlab.FillHoles, "_fillholes_inprocess_noop"),
    ):
        assert cls.__call__.__name__ == expected, (
            f"{cls.__name__}.__call__ is {cls.__call__.__name__}, expected {expected} -- "
            f"the xvfb+meshlabserver subprocess is back"
        )
    if neurd.PATCH_STATUS["decimator_open3d"] is not None:
        assert meshlab.Decimator.__call__.__name__ == "_decimate_inprocess"


def test_off_renumber_repair_installed():
    from mesh_tools.meshlab import Meshlab
    assert Meshlab.fetch_mesh_from_off.__name__ == "_fetch_mesh_from_off_repaired", (
        "the MeshLab OFF-exporter renumber repair is not installed -- vertex-deleting "
        "filters will crash later in vertex_faces/split_by_vertices"
    )


def test_skeleton_kdtree_accepts_1d():
    """The wrapper exists so 1-D cumulative skeletal-distance arrays work."""
    from mesh_tools import skeleton_utils as sk
    tree = sk.KDTree([0.0, 1.0, 2.0, 10.0])
    dist, idx = tree.query([1.1])
    assert int(idx[0] if hasattr(idx, "__len__") else idx) == 1


def test_failure_path_records_and_warns():
    """The detection mechanism itself: a failed patch must be loud and must fail the gate.

    Guards against the previous behaviour, where every patch ended in
    `except Exception: pass` and a broken patch was completely invisible.
    """
    saved = dict(neurd.PATCH_STATUS)
    try:
        with pytest.warns(RuntimeWarning, match="did NOT install"):
            neurd._patch_failed("poisson_noop", AttributeError("no attribute 'Poisson'"))
        assert neurd.PATCH_STATUS["poisson_noop"] is False
        # ... and that is exactly what test_no_patch_failed asserts against
        assert [k for k, v in neurd.PATCH_STATUS.items() if v is False] == ["poisson_noop"]
    finally:
        neurd.PATCH_STATUS.clear()
        neurd.PATCH_STATUS.update(saved)


def test_registry_reports_optout_when_flag_set():
    """Proves the registry tracks reality rather than always reporting success.

    Runs a fresh interpreter with NEURD_LEGACY_VERTEX_COMPONENTS=1: the patch must
    report 'skipped' and the networkx implementation must be the one bound.
    """
    env = dict(os.environ, NEURD_LEGACY_VERTEX_COMPONENTS="1")
    code = (
        "import neurd;"
        "from mesh_tools import trimesh_utils as tu;"
        "print(neurd.PATCH_STATUS['vertex_components_scipy'],"
        "      tu.vertex_components.__module__.startswith('neurd'))"
    )
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True,
                         text=True, timeout=600)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip().splitlines()[-1] == "None False", (
        f"expected the opt-out to be visible in PATCH_STATUS, got {out.stdout.strip()!r}"
    )
