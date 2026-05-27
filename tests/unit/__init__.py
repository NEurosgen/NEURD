"""Shared probes for the unit-test suite.

``pytest.importorskip`` only catches ``ImportError``. The ``datasci_tools``
chain can also blow up at import time with ``AttributeError`` (e.g. on
numpy>=2 where ``numpy.float_`` was removed). Test modules call
``skip_if_datasci_tools_unusable()`` at module top level to fall back to a
clean skip in that case.
"""
from typing import Optional

import pytest


def _reason() -> Optional[str]:
    try:
        import datasci_tools.module_utils  # noqa: F401
    except Exception as exc:  # pragma: no cover
        return f"datasci_tools unusable: {exc.__class__.__name__}: {exc}"
    return None


_SKIP_REASON = _reason()


def skip_if_datasci_tools_unusable() -> None:
    if _SKIP_REASON is not None:
        pytest.skip(_SKIP_REASON, allow_module_level=True)
