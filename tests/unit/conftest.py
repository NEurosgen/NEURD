"""Run export contracts through both the library API and retained CLI aliases."""
import importlib

import pytest


@pytest.fixture(params=('process_all_neurons', 'neurd.neuron_export'), ids=('cli', 'library'))
def raw_exporter(request):
    return importlib.import_module(request.param)
