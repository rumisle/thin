"""The build options record and its context, without the compiler.

Kept apart from thin.codegen so the runtime reads the options in force (they
are part of a render's address) without loading the native
extension: a store hit renders nothing.
"""

from __future__ import annotations

import contextlib
import contextvars
import dataclasses
from collections.abc import Iterator
from dataclasses import dataclass, field

__all__ = ["Options", "current", "output_pairs", "using"]


def _opt(default, output: bool, doc: str):
    """An option's field: its default and "output" column."""
    return field(default=default, metadata={"output": output, "doc": doc})


@dataclass(frozen=True)
class Options:
    """The build options record.

    Each field's metadata ``output`` is column: whether the option can
    change what a render stores or whether it succeeds. A new option must say.
    """

    audit: bool = _opt(False, True, "Emit loop bodies twice, and audit the placement.")
    keep_dead: bool = _opt(False, True, "Skip the dead-write check.")
    safe_barrier: bool = _opt(False, True, "Wait for every counter at 0 before each barrier.")
    waits_unique: bool = _opt(False, True, "Refuse counts that are not the only fixpoint.")
    safe_waits: bool = _opt(False, True, "Every placed count 0.")
    regions: bool = _opt(True, True, "Use lane regions' EXEC tokens in placement.")
    peak_list: int = _opt(24, False, "Values a pressure error lists.")
    peak_report: bool = _opt(False, False, "Report the values at the peak.")
    pressure: bool = _opt(False, False, "Report live dwords along the program.")
    lds_wave_order: bool = _opt(True, True, "One wave's program order may separate LDS regions.")


_DEFAULTS = Options()
_CURRENT: contextvars.ContextVar[Options] = contextvars.ContextVar("thin.options")


def current() -> Options:
    """The options in force: a kernel takes this record when it is made."""
    return _CURRENT.get(_DEFAULTS)


@contextlib.contextmanager
def using(**changes) -> Iterator[Options]:
    """Kernels made inside the block take the current options with these changes.

    Blocks nest; an unknown option name is a TypeError.
    """
    token = _CURRENT.set(dataclasses.replace(current(), **changes))
    try:
        yield _CURRENT.get()
    finally:
        _CURRENT.reset(token)


_OUTPUT = tuple(sorted(f.name for f in dataclasses.fields(Options) if f.metadata["output"]))
"""The output options' names, sorted."""
_PAIRS: dict = {}
"""Options record -> its pairs (records are frozen)."""


def output_pairs(options: Options | None = None) -> tuple:
    """(name, repr of its value) of every output option, sorted by name."""
    o = current() if options is None else options
    found = _PAIRS.get(o)
    if found is None:
        found = _PAIRS[o] = tuple((n, repr(getattr(o, n))) for n in _OUTPUT)
    return found
