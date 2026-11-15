"""What the library asks the current target for: its kernel-library hooks.

Every spelling, policy and size the library would otherwise hard-code is
one of the target's ``lib_hooks``. A hook the target does not supply yet
is refused where the library first needs it, naming the hook.
"""

from __future__ import annotations

from thin.target import target


def hook(name: str) -> str:
    """The current target's hook ``name``; ValueError if it supplies none."""
    t = target()
    value = t.lib_hooks[name]
    if value is None:
        raise ValueError(f"{t.name} supplies no {name} hook for the kernel library yet")
    return value


def granule() -> int:
    """The lanes a wave is counted by: the target's wave width."""
    return int(hook("lane_granule"))


def coherent() -> dict:
    """The cache fields of an access coherent across workgroups."""
    return {"scope": hook("coherent")}


def system() -> dict:
    """The cache fields of an access to memory other devices map."""
    return {"scope": hook("system")}


def returning() -> dict:
    """The cache fields of an atomic that returns its old value."""
    return {"th": hook("atomic_return")}
