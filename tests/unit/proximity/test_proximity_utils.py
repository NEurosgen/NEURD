"""Unit tests for the pure helpers in neurd.proximity_utils.

Most of the module is wired to a datajoint database (via the global ``vdi``
namespace). Only the small DB-free helpers are covered here.
"""
import pytest

from tests.unit import skip_if_datasci_tools_unusable

skip_if_datasci_tools_unusable()
pytest.importorskip("pandas")
pytest.importorskip("networkx")
# Transitively required: neurd → h01_volume_utils → microns_volume_utils → datajoint.
pytest.importorskip("datajoint")

import networkx as nx
import numpy as np
import pandas as pd

from neurd import proximity_utils as pxu


# ---------------- synapse_coordinates_from_df ----------------


def test_synapse_coordinates_from_df_returns_float_xyz_array():
    df = pd.DataFrame(
        {
            "synapse_x_nm": [1, 2, 3],
            "synapse_y_nm": [10, 20, 30],
            "synapse_z_nm": [100, 200, 300],
            "extra_column": ["a", "b", "c"],
        }
    )
    coords = pxu.synapse_coordinates_from_df(df)

    assert coords.shape == (3, 3)
    assert coords.dtype == np.float64
    np.testing.assert_array_equal(coords[0], [1.0, 10.0, 100.0])


def test_synapse_coordinates_from_df_handles_empty_df():
    df = pd.DataFrame(
        {"synapse_x_nm": [], "synapse_y_nm": [], "synapse_z_nm": []}
    )
    coords = pxu.synapse_coordinates_from_df(df)
    assert coords.shape == (0, 3)


# ---------------- A_prox_from_G_prox / A_syn_from_G_prox ----------------


def _toy_proximity_graph():
    g = nx.DiGraph()
    g.add_edge("a", "b", n_synapses=3)
    g.add_edge("a", "c", n_synapses=0)
    g.add_edge("b", "c", n_synapses=1)
    return g


def test_A_prox_from_G_prox_binarises_adjacency():
    g = _toy_proximity_graph()
    A = pxu.A_prox_from_G_prox(g, nodelist=["a", "b", "c"])

    # Default adjacency_matrix in networkx_utils uses unweighted edges → 1 per
    # edge — the binarisation step is a no-op for those, but ensures the matrix
    # is 0/1 only.
    assert set(np.unique(A)).issubset({0, 1})
    # Edges a→b, a→c, b→c → 3 ones in the upper triangle.
    assert A.sum() == 3


def test_A_syn_from_G_prox_binarises_weighted_adjacency():
    g = _toy_proximity_graph()
    A = pxu.A_syn_from_G_prox(g, nodelist=["a", "b", "c"])

    # Even though the underlying weights are 0/1/3, the function binarises.
    assert set(np.unique(A)).issubset({0, 1})
    # a→b (3 synapses) and b→c (1 synapse) survive; a→c (0) is dropped.
    assert A.sum() == 2
