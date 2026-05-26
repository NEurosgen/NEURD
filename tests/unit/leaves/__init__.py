"""Probe whether the ``datasci_tools`` chain is importable.

Test modules call ``skip_if_datasci_tools_unusable()`` at module top level
because ``pytest.importorskip`` only handles ``ImportError`` — it can't catch
the ``AttributeError`` raised by ``datasci_tools.numpy_dep`` on numpy>=2.
"""
import pytest


def _reason() -> str | None:
    try:
        import datasci_tools.module_utils  # noqa: F401
    except Exception as exc:  # pragma: no cover
        return f"datasci_tools unusable: {exc.__class__.__name__}: {exc}"
    return None


_SKIP_REASON = _reason()


def skip_if_datasci_tools_unusable() -> None:
    if _SKIP_REASON is not None:
        pytest.skip(_SKIP_REASON, allow_module_level=True)
