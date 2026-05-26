"""Helpers for querying a MICrONS-style connectome graph stored as a
pandas DataFrame (node_df) or networkx graph."""

from datasci_tools import networkx_utils as xu
from datasci_tools import pandas_utils as pu
from datasci_tools import system_utils as su


default_node_df_path = "/platinum_graph/Data/G_query_v6_filtered_node_df.csv"
default_edge_df_path = "/platinum_graph/Data/G_query_v6_filtered_edge_and_node_df.gzip"
default_graph_path = "/platinum_graph/Data/G_query_v6_filtered.pbz2"


def load_node_df(filepath=None):
    if filepath is None:
        filepath = default_node_df_path
    return su.decompress_pickle(filepath)


def load_edge_df(filepath=None):
    if filepath is None:
        filepath = default_edge_df_path
    return pu.gzip_to_df(filepath)


# ----------------- Node querying ------------------------- #
def soma_centers_from_node_df(node_df):
    return node_df[["soma_x_nm", "soma_y_nm", "soma_z_nm"]].to_numpy()


def node_df_from_query(
    query,
    G=None,
    node_df=None,
    verbose=False,
    **kwargs,
):
    if node_df is None:
        node_df = xu.node_df(G)

    curr_df = node_df.query(query)

    if verbose:
        print(f"# of nodes: {len(curr_df)}")
    return curr_df


def node_df_from_attribute_value(
    attribute_type=None,
    attribute_value=None,
    query=None,
    G=None,
    node_df=None,
    **kwargs,
):
    if query is None:
        if isinstance(attribute_value, str):
            query = f"{attribute_type} == '{attribute_value}'"
        else:
            query = f"{attribute_type} == {attribute_value}"

    return node_df_from_query(query=query, G=G, node_df=node_df, **kwargs)


def soma_centers_from_node_query(
    query,
    G=None,
    node_df=None,
    verbose=False,
    return_query_df=False,
):
    """Apply `query` to node_df, return soma centers (and optionally the sub-df)."""
    if node_df is None:
        node_df = xu.node_df(G)

    sub_df = node_df.query(query)
    sub_df_centers = soma_centers_from_node_df(sub_df)

    if verbose:
        print(f"# of cells in query = {len(sub_df_centers)}")

    if return_query_df:
        return sub_df_centers, sub_df
    return sub_df_centers


def excitatory_cells_node_df(G=None, node_df=None, **kwargs):
    return node_df_from_attribute_value(
        attribute_type="cell_type",
        attribute_value="excitatory",
        G=G,
        node_df=node_df,
        **kwargs,
    )


def inhibitory_cells_node_df(G=None, node_df=None, **kwargs):
    return node_df_from_attribute_value(
        attribute_type="cell_type",
        attribute_value="inhibitory",
        G=G,
        node_df=node_df,
        **kwargs,
    )


def n_excitatory_n_inhibitory_nodes(G=None, node_df=None, verbose=False):
    n_excitatory = len(excitatory_cells_node_df(G=G, node_df=node_df))
    n_inhibitory = len(inhibitory_cells_node_df(G=G, node_df=node_df))

    if verbose:
        print(f"n_excitatory = {n_excitatory},n_inhibitory = {n_inhibitory} ")
    return n_excitatory, n_inhibitory
