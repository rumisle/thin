"""thin: a GPU kernel compiler and its runtime.

The root's public names are loaded on first use: importing
``thin`` loads none of the modules behind them.
"""

from __future__ import annotations

import importlib

_LAZY = {
    "compile": "thin.runtime",
    "declare": "thin.runtime",
    "overriding": "thin.runtime",
    "use_store": "thin.runtime",
    "cdiv": "thin.runtime",
    "KernelDef": "thin.runtime",
    "Kernel": "thin.runtime",
    "Bound": "thin.runtime",
    "buffer": "thin.runtime",
    "Refused": "thin.runtime",
    "targeting": "thin.target",
    "Spec": "thin.opspec",
    "Int": "thin.opspec",
    "Choice": "thin.opspec",
    "Bool": "thin.opspec",
    "Field": "thin.opspec",
    "Tensor": "thin.opspec",
    "In": "thin.opspec",
    "Out": "thin.opspec",
    "InOut": "thin.opspec",
    "implements": "thin.opspec",
    "implementations": "thin.opspec",
    "candidates": "thin.opspec",
    "lower": "thin.opspec",
    "lowered": "thin.opspec",
    "check": "thin.opspec",
    "group": "thin.opspec.group",
}

__all__ = sorted(_LAZY)


def __getattr__(name: str) -> object:
    """Loads a public name's module on first use."""
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module 'thin' has no attribute '{name}'")
    if name == "group":
        return importlib.import_module(module)
    return getattr(importlib.import_module(module), name)
