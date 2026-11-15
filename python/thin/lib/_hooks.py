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


def _cache_fields(default_key: str, raw: str) -> dict:
    """Named choices (`SCOPE_*`, `TH_*`) use `default_key`; bare flags map to `True`."""
    if raw.isupper():
        return {default_key: raw}
    return {flag: True for flag in raw.split()}


def coherent() -> dict:
    """The cache fields of an access coherent across workgroups."""
    return _cache_fields("scope", hook("coherent"))


def system() -> dict:
    """The cache fields of an access to memory other devices map."""
    return _cache_fields("scope", hook("system"))


def returning() -> dict:
    """The cache fields of an atomic that returns its old value."""
    return _cache_fields("th", hook("atomic_return"))
