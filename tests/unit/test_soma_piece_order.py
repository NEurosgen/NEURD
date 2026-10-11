"""Soma ordering preserves NumPy tie choices and empty-input exceptions."""
from types import SimpleNamespace

import numpy as np
import pytest

from tests.unit import skip_if_datasci_tools_unusable

skip_if_datasci_tools_unusable()

from neurd import soma_extraction_utils as soma


@pytest.mark.parametrize("sizes", [[], [1], [4, 1, 2], [2, 2, 2],
                                  [3, 1, 3, 2, 3], [8] * 30, list(range(50))])
def test_largest_piece_uses_the_original_order_even_for_ties(sizes):
    pieces = [SimpleNamespace(faces=range(size), position=index)
              for index, size in enumerate(sizes)]
    calls = []

    def split(**kwargs):
        calls.append(kwargs)
        return iter(pieces)

    mesh = SimpleNamespace(split=split)
    expected = [pieces[index] for index in np.flip(np.argsort(sizes)).tolist()]
    actual = soma._ordered_splits(mesh)
    assert len(actual) == len(expected)
    assert all(left is right for left, right in zip(actual, expected))
    if not expected:
        with pytest.raises(IndexError):
            soma.largest_mesh_piece(mesh)
    else:
        assert soma.largest_mesh_piece(mesh) is expected[0]
    assert calls == [{"only_watertight": False}, {"only_watertight": False}]
