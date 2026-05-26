"""Smoke tests for neurd.nature_paper_plotting (figure-generation helpers)."""
import pytest

from tests.unit.leaves import skip_if_datasci_tools_unusable

skip_if_datasci_tools_unusable()
pytest.importorskip("seaborn")

import pandas as pd

from neurd import nature_paper_plotting as npp


def test_palette_dicts_have_expected_keys():
    assert set(npp.exc_inh_palette) == {"excitatory", "inhibitory"}
    assert set(npp.exc_inh_combination_palette) == {
        "Exc Onto Exc",
        "Exc Onto Inh",
        "Inh Onto Exc",
        "Inh Onto Inh",
    }


def test_plot_edit_labels_subset_smoke():
    edits_df = pd.DataFrame(
        {"edit_type": ["foo"] * 5, "x": [1, 2, 3, 4, 5], "y": [1, 2, 3, 4, 5]}
    )
    # Should not raise.
    npp.plot_edit_labels_subset(edits_df, edit_labels=["foo"], x="x", y="y", bins=5)
