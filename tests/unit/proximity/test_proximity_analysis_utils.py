"""Unit tests for the pure helpers in neurd.proximity_analysis_utils.

The bulk of the module pulls data from datajoint via the global ``vdi``
namespace; only the DB-free helpers (``conversion_rate``, ``conversion_df``,
``print_n_dict``) are exercised here.
"""
import pytest

from tests.unit import skip_if_datasci_tools_unusable

skip_if_datasci_tools_unusable()
pytest.importorskip("pandas")
pytest.importorskip("seaborn")
pytest.importorskip("datajoint")  # the module imports `datajoint as dj`

import numpy as np
import pandas as pd

from neurd import proximity_analysis_utils as pxa


# ---------------- conversion_rate ----------------


def test_conversion_rate_on_empty_df_is_nan():
    out = pxa.conversion_rate(pd.DataFrame({"n_synapses": []}))
    assert np.isnan(out)


def test_conversion_rate_is_mean_of_n_synapses_per_proximity():
    df = pd.DataFrame({"n_synapses": [0, 0, 1, 2]})
    # sum(0+0+1+2)=3, len=4 → 0.75
    assert pxa.conversion_rate(df) == pytest.approx(0.75)


# ---------------- conversion_df ----------------


def test_conversion_df_groups_by_compartment_and_computes_conversion():
    df = pd.DataFrame(
        {
            "presyn": [1, 1, 2, 2],
            "postsyn": [10, 10, 20, 20],
            "n_synapses": [0, 2, 1, 1],
            "postsyn_compartment": ["basal", "basal", "apical", "apical"],
            "proximity_dist": [100.0, 200.0, 150.0, 250.0],
            "presyn_width": [400.0, 500.0, 300.0, 350.0],
        }
    )
    out = pxa.conversion_df(
        df,
        separate_compartments=True,
        separate_by_neuron_pairs=False,
    )

    # Two compartment groups → two output rows.
    assert len(out) == 2
    assert set(out["postsyn_compartment"]) == {"basal", "apical"}

    basal = out[out["postsyn_compartment"] == "basal"].iloc[0]
    apical = out[out["postsyn_compartment"] == "apical"].iloc[0]

    # basal: 0+2 synapses across 2 proximities → conversion = 1.0
    assert basal["n_synapses"] == 2
    assert basal["n_proximity"] == 2
    assert basal["conversion"] == pytest.approx(1.0)

    # apical: 1+1 synapses across 2 proximities → conversion = 1.0
    assert apical["n_synapses"] == 2
    assert apical["n_proximity"] == 2
    assert apical["conversion"] == pytest.approx(1.0)


def test_conversion_df_separate_by_neuron_pairs_keeps_each_pair():
    df = pd.DataFrame(
        {
            "presyn": [1, 1, 2],
            "postsyn": [10, 10, 20],
            "n_synapses": [0, 3, 5],
            "postsyn_compartment": ["basal"] * 3,
            "proximity_dist": [100.0, 200.0, 150.0],
            "presyn_width": [400.0, 500.0, 300.0],
        }
    )
    out = pxa.conversion_df(
        df,
        separate_compartments=False,
        separate_by_neuron_pairs=True,
    )
    # Two unique (presyn, postsyn) pairs.
    assert len(out) == 2

    pair_1_10 = out[(out["presyn"] == 1) & (out["postsyn"] == 10)].iloc[0]
    assert pair_1_10["n_synapses"] == 3
    assert pair_1_10["n_proximity"] == 2
    assert pair_1_10["conversion"] == pytest.approx(1.5)


# ---------------- print_n_dict ----------------


def test_print_n_dict_formats_categories(capsys):
    n_dict = {
        "exc→exc": {"datapoint": [0.123, 1.456], "n_prox": [10, 20]},
        "exc→inh": {"datapoint": [2.0], "n_prox": [5]},
    }
    out = pxa.print_n_dict(n_dict, verbose=False)

    assert "exc→exc:" in out
    assert "0.12 (n_prox = 10)" in out
    assert "1.46 (n_prox = 20)" in out
    assert "exc→inh:" in out
    assert "2.0 (n_prox = 5)" in out

    # verbose=False → no stdout
    captured = capsys.readouterr()
    assert captured.out == ""


def test_print_n_dict_verbose_prints_to_stdout(capsys):
    n_dict = {"foo": {"datapoint": [0.5], "n_prox": [42]}}
    pxa.print_n_dict(n_dict, verbose=True)
    captured = capsys.readouterr()
    assert "foo:" in captured.out
    assert "n_prox = 42" in captured.out
