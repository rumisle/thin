"""Ops, their implementations, checking them, and groups.

An op's spec (a ``Spec`` subclass with fields and a signature of tensors)
says what is computed; implementations registered against it say how
kernels compute it; ``lower`` turns an op into the Bounds its binding
makes, and ``check`` runs and judges them. ``thin.opspec.group`` holds
groups of ranks and their memory.

Importing this package loads no compiler and neither numpy nor torch:
those load when a tensor is made or a check runs.
"""

from __future__ import annotations

from thin.opspec import group
from thin.opspec._check import SCHEDS, Failure, check
from thin.opspec._impl import (
    NEEDS,
    Implementation,
    candidates,
    implementations,
    implements,
    lower,
    lowered,
)
from thin.opspec._spec import Bool, Choice, Field, Int, Spec
from thin.opspec._tensor import In, InOut, Out, Tensor

__all__ = [
    "NEEDS",
    "SCHEDS",
    "Bool",
    "Choice",
    "Failure",
    "Field",
    "Implementation",
    "In",
    "InOut",
    "Int",
    "Out",
    "Spec",
    "Tensor",
    "candidates",
    "check",
    "group",
    "implementations",
    "implements",
    "lower",
    "lowered",
]
