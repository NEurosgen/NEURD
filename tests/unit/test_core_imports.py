"""Smoke tests: the core-clump modules must import on a bare env.

Guards the import-order / lazy-import fixes that made `from neurd import
spine_utils` work without `datajoint`/`dotmotif`/`caveclient` installed:
- PRE-1: `function_utils as fcu` is imported at the top of `neuron_searching`
  (was used before its import on line 2370).
- PRE-2: `dj_utils` is lazily imported inside `spine_utils` (upstream
  `raise e(...)` bug fires only when `datajoint` is missing).
- PRE-3: `dotmotif_utils` is lazily imported inside `graph_filters`.

If any core module regresses to a module-load dependency on a broken/optional
chain, the corresponding import below fails.
"""
import importlib

import pytest

from tests.unit import skip_if_datasci_tools_unusable

skip_if_datasci_tools_unusable()

# Core clump (DEPS_PLAN §0) + graph_filters (PRE-3 fix lives there).
CORE_MODULES = [
    "apical_utils",
    "axon_utils",
    "branch_utils",
    "cell_type_utils",
    "classification_utils",
    "concept_network_utils",
    "error_detection",
    "graph_filters",
    "limb_utils",
    "neuron_searching",
    "neuron_statistics",
    "neuron_utils",
    "neuron_visualizations",
    "preprocess_neuron",
    "proofreading_utils",
    "soma_extraction_utils",
    "spine_utils",
    "synapse_utils",
]


@pytest.mark.parametrize("mod_name", CORE_MODULES)
def test_core_module_imports(mod_name):
    importlib.import_module(f"neurd.{mod_name}")


def test_neuron_searching_has_fcu_at_module_level():
    """PRE-1 regression: `fcu` must resolve at module load, not after line 2370."""
    ns = importlib.import_module("neurd.neuron_searching")
    assert hasattr(ns, "fcu")
