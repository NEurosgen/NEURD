"""Equivalence gate for the get_matching_vertices KDTree replacement.

`datasci_tools.numpy_utils.get_matching_vertices` finds unordered vertex pairs within
`equiv_distance` by building the full NxN coordinate distance matrix (pdist -> squareform),
a copy of it, and an NxN np.eye -- 3x O(N^2). On the skeleton node-combine path
(skeleton_utils.py:1737, LOCKED) N is large, so it is ~25 GB of allocation churn and a big
wall cost on a big-H01 build (memray/py-spy). NEURD monkeypatches it with a
`cKDTree.query_pairs` version (O(N log N); see `neurd/__init__.py`), so the patched result must
be indistinguishable from the original -- including row order, since callers consume the pairs
directly.

We compare against the ORIGINAL implementation (inlined here, so the test still checks the real
algorithm even with the monkeypatch live) over randomized point clouds with exact-duplicate and
jittered coordinates, both `ignore_diagonal` modes, and several `equiv_distance` values.
"""
import neurd  # noqa: F401  -- activates numpy shims + the get_matching_vertices monkeypatch
import numpy as np
import pytest
from scipy.spatial.distance import pdist, squareform

from tests.unit import skip_if_datasci_tools_unusable

skip_if_datasci_tools_unusable()

from datasci_tools import numpy_utils as nu  # noqa: E402  (after the skip guard)


def _original_get_matching_vertices(possible_vertices, ignore_diagonal=True, equiv_distance=0):
    """The ORIGINAL O(N^2) implementation, inlined (nu.get_matching_vertices is monkeypatched)."""
    pv = np.asarray(possible_vertices).reshape(-1, 3)
    dist_matrix = squareform(pdist(pv, "euclidean"))
    matching = np.array(np.where(dist_matrix <= equiv_distance)).T
    if ignore_diagonal:
        result = matching[matching[:, 0] != matching[:, 1]]
    else:
        result = matching
    if len(result) > 0:
        return np.unique(np.sort(result, axis=1), axis=0)
    return result


def _as_pairs(a):
    a = np.asarray(a)
    return a.reshape(-1, 2) if a.size else np.empty((0, 2), dtype=np.int64)


def test_patch_is_active():
    """The KDTree replacement is the live implementation (guards against a silent revert)."""
    assert nu.get_matching_vertices.__name__ == "_get_matching_vertices_kdtree"


@pytest.mark.parametrize("equiv_distance", [0, 0.02, 0.5])
@pytest.mark.parametrize("ignore_diagonal", [True, False])
def test_matches_original_randomized(equiv_distance, ignore_diagonal):
    rng = np.random.default_rng(0)
    for trial in range(150):
        n = int(rng.integers(2, 60))
        pts = rng.integers(0, 8, size=(n, 3)).astype(float)  # forces many exact-coord duplicates
        if trial % 3 == 0:
            pts += rng.normal(0, 0.01, pts.shape)             # sub-equiv_distance jitter
        got = _as_pairs(nu.get_matching_vertices(
            pts.copy(), ignore_diagonal=ignore_diagonal, equiv_distance=equiv_distance))
        exp = _as_pairs(_original_get_matching_vertices(
            pts.copy(), ignore_diagonal=ignore_diagonal, equiv_distance=equiv_distance))
        assert got.shape == exp.shape, f"trial {trial}: {got.shape} vs {exp.shape}"
        assert np.array_equal(got, exp), f"trial {trial}: pairs differ"


def test_degenerate_inputs():
    """Fewer than 2 points and no-match clouds return a well-formed empty (0, 2) array."""
    for pts in (np.empty((0, 3)), np.array([[1.0, 2.0, 3.0]])):
        out = _as_pairs(nu.get_matching_vertices(pts))
        assert out.shape == (0, 2)
    spread = _as_pairs(nu.get_matching_vertices(np.arange(30).reshape(-1, 3).astype(float),
                                                equiv_distance=0))
    assert spread.shape == (0, 2)
