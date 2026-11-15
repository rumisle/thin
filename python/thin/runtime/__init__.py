"""Compile, bind, run and store.

The names the package root exposes come from the modules here, each
loaded on first use; the RankGroup protocol is what a Bound asks of a
group's rank handle.

Importing this module, or any of the definition modules (``params``,
``kernel``, ``bound``), loads no compiler, no native extension and neither
numpy nor torch: those load inside the calls that need them.
"""

from __future__ import annotations

import importlib
from typing import Protocol

from thin.runtime.errors import Collision, DeviceFault, LaunchError, NotEmulated, Refused

__all__ = [
    "GUARD_FILL",
    "Bound",
    "Collision",
    "Device",
    "DeviceFault",
    "EmulatedDevice",
    "Kernel",
    "KernelDef",
    "LaunchError",
    "NotEmulated",
    "Program",
    "RankGroup",
    "Refused",
    "buffer",
    "cdiv",
    "compile",
    "declare",
    "dispatch",
    "emulated_group",
    "generator_identity",
    "made_by_buffer",
    "overriding",
    "program",
    "use_store",
]

_LAZY = {
    "compile": "thin.runtime.kernel",
    "use_store": "thin.runtime.kernel",
    "cdiv": "thin.runtime.kernel",
    "KernelDef": "thin.runtime.kernel",
    "Kernel": "thin.runtime.kernel",
    "declare": "thin.runtime.params",
    "generator_identity": "thin.runtime.identity",
    "overriding": "thin.runtime.params",
    "Bound": "thin.runtime.bound",
    "buffer": "thin.runtime.bound",
    "made_by_buffer": "thin.runtime.bound",
    "Device": "thin.runtime.device",
    "Program": "thin.runtime.device",
    "program": "thin.runtime.device",
    "dispatch": "thin.runtime.device",
    "GUARD_FILL": "thin.runtime.device",
    "EmulatedDevice": "thin.runtime.emulated",
    "emulated_group": "thin.runtime.emulated",
}

_NAMED = {"emulated_group": "group"}
"""A root name defined under another name in its module."""


def __getattr__(name: str) -> object:
    """Loads a public name's module on first use."""
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module 'thin.runtime' has no attribute '{name}'")
    return getattr(importlib.import_module(module), _NAMED.get(name, name))


class Team(Protocol):
    """A group's shared tables, as the runtime uses them."""

    held: object
    """The group's held table."""

    def peers_of(self, v) -> list | None:
        """Every rank's copy of the group tensor v is a copy of, or None."""
        ...

    def mapped(self, kernel, name: str, nbytes: int) -> list:
        """Every rank's buffer of a mapped argument, at least nbytes."""
        ...

    def holds(self, kernel, name: str, nbytes: int) -> bool:
        """Whether such buffers exist already."""
        ...


class RankGroup(Protocol):
    """One rank's handle on a group, as a Bound uses it."""

    rank: int
    """This handle's rank."""

    device: object
    """The rank's GPU index, or the emulated tag."""

    team: Team
    """The group's tables."""
