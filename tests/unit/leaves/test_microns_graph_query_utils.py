"""Unit tests for neurd.microns_graph_query_utils.

Only the pandas-query functions are exercised — the load_node_df / load_edge_df
functions hit disk and are covered by patching the loaders.
"""
from unittest.mock import patch

import pytest

from tests.unit.leaves import skip_if_datasci_tools_unusable

skip_if_datasci_tools_unusable()
pytest.importorskip("pandas")

import pandas as pd

from neurd import microns_graph_query_utils as mqu


@pytest.fixture
def node_df():
    return pd.DataFrame(
        {
            "cell_type": ["excitatory", "inhibitory", "excitatory"],
            "soma_x_nm": [1.0, 2.0, 3.0],
            "soma_y_nm": [1.0, 2.0, 3.0],
            "soma_z_nm": [1.0, 2.0, 3.0],
        }
    )


def test_soma_centers_from_node_df_shape(node_df):
    centers = mqu.soma_centers_from_node_df(node_df)
    assert centers.shape == (3, 3)


def test_node_df_from_query_filters(node_df):
    out = mqu.node_df_from_query("cell_type == 'excitatory'", node_df=node_df)
    assert len(out) == 2


def test_node_df_from_attribute_value_string(node_df):
    out = mqu.node_df_from_attribute_value(
        attribute_type="cell_type", attribute_value="inhibitory", node_df=node_df
    )
    assert len(out) == 1


def test_excitatory_inhibitory_counts(node_df):
    n_exc, n_inh = mqu.n_excitatory_n_inhibitory_nodes(node_df=node_df)
    assert (n_exc, n_inh) == (2, 1)


def test_excitatory_cells_node_df(node_df):
    out = mqu.excitatory_cells_node_df(node_df=node_df)
    assert len(out) == 2
    assert set(out["cell_type"]) == {"excitatory"}


def test_inhibitory_cells_node_df(node_df):
    out = mqu.inhibitory_cells_node_df(node_df=node_df)
    assert len(out) == 1
    assert set(out["cell_type"]) == {"inhibitory"}


def test_soma_centers_from_node_query_returns_df(node_df):
    centers, sub = mqu.soma_centers_from_node_query(
        query="cell_type == 'inhibitory'",
        node_df=node_df,
        return_query_df=True,
    )
    assert centers.shape == (1, 3)
    assert len(sub) == 1


def test_load_node_df_delegates_to_decompress_pickle():
    with patch("datasci_tools.system_utils.decompress_pickle", return_value="OK") as p:
        assert mqu.load_node_df() == "OK"
        p.assert_called_once_with(mqu.default_node_df_path)


def test_load_edge_df_delegates_to_gzip_to_df():
    with patch("datasci_tools.pandas_utils.gzip_to_df", return_value="OK") as p:
        assert mqu.load_edge_df() == "OK"
        p.assert_called_once_with(mqu.default_edge_df_path)
