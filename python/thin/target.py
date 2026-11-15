"""The Python view of targets.

Targets are native (thin._native). This module gives each target, form,
slot, field and named register one Python object, built from the native
record on first use, and adds selection by context (``targeting``) and the
flag helper. Every rule stays native: lookups, spellings and the returning
hook are calls into the target.
"""

from __future__ import annotations

import contextlib
import contextvars
import threading
from collections.abc import Iterator
from types import MappingProxyType

from thin import _native

__all__ = [
    "Field",
    "Flags",
    "Form",
    "Phys",
    "Slot",
    "Target",
    "flag_bits",
    "target",
    "target_names",
    "targeting",
]


class Phys:
    """A named register: one object per register, for every target."""

    __slots__ = ("dwords", "file", "name", "whole")

    def __init__(self, name: str, dwords: int, file: str, whole: str):
        self.name = name
        self.dwords = dwords
        self.file = file
        self.whole = whole

    def __repr__(self) -> str:
        return self.name

    def __copy__(self) -> Phys:
        return self

    def __deepcopy__(self, memo) -> Phys:
        return self


PHYS: MappingProxyType = MappingProxyType({row[0]: Phys(*row) for row in _native.phys()})
"""The named registers by name."""


def flag_bits() -> list[tuple[str, int]]:
    """(name, bit) of every form flag, in table order."""
    return _native.flag_bits()


class Flags:
    """One int attribute per form flag."""

    @staticmethod
    def names(f: int) -> str:
        """The set flag names joined by ``|``, or ``NONE``."""
        names = [name for name, bit in flag_bits() if f >> bit & 1]
        return "|".join(names) or "NONE"


for _name, _bit in flag_bits():
    setattr(Flags, _name, 1 << _bit)


class Slot:
    """An explicit operand slot of a form."""

    __slots__ = ("dwords", "kind", "literal", "modifiers", "name", "role", "sgpr_widths", "widths")

    def __init__(self, kind, dwords, role, name, sgpr_widths, modifiers, literal, widths):
        self.kind = kind
        self.dwords = dwords
        self.role = role
        self.name = name
        self.sgpr_widths = frozenset(sgpr_widths)
        self.modifiers = modifiers
        self.literal = literal
        self.widths = list(widths)


class Field:
    """A field a form takes."""

    __slots__ = ("bits", "choices", "kind", "name", "required", "signed")

    def __init__(self, name, kind, bits, signed, required, choices):
        self.name = name
        self.kind = kind
        self.bits = bits
        self.signed = signed
        self.required = required
        self.choices = list(choices)


class Form:
    """A form: what every instruction of one spelling and operand count shares."""

    __slots__ = (
        "arity",
        "base",
        "canon",
        "counter",
        "dpp",
        "dst_half",
        "dst_sel",
        "encoding",
        "fields",
        "flags",
        "id",
        "implicit_defs",
        "implicit_uses",
        "kind",
        "mem",
        "mem_bytes",
        "mnemonic",
        "name",
        "pseudo",
        "returning",
        "slots",
        "target",
        "unordered",
        "variadic",
        "wait_text",
    )

    def __init__(self, target_name: str, head, tail):
        (self.id, self.name, self.mnemonic, self.base, self.canon, self.kind) = head[:6]
        (self.counter, self.unordered, self.flags, self.encoding) = head[6:]
        self.target = target_name
        self.arity, self.returning = tail[0], tail[1]
        self.slots = tuple(Slot(*s) for s in tail[2])
        self.variadic = tail[3]
        self.fields = tuple(Field(*f) for f in tail[4])
        self.implicit_defs = frozenset(tail[5])
        self.implicit_uses = frozenset(tail[6])
        self.mem_bytes, self.pseudo = tail[7], tail[8]
        self.dpp, self.dst_sel, self.mem, self.wait_text, self.dst_half = tail[9]

    def __repr__(self) -> str:
        return f"Desc({self.name})"


class Target:
    """One target's view: its record, its forms and its queries."""

    def __init__(self, name: str):
        wave, lds_max, chip, counters, waits, barrier, window = _native.target_info(name)
        self.name = name
        self.wave = wave
        self.lds_max = lds_max
        (self.xcds, self.cus, self.simds, self.cu_lds, self.vgprs, self.vgprs_per_lane) = chip[:6]
        (self.sgpr_max, self.user_sgprs, self.wave_slots, self.preload_max) = chip[6:10]
        self.wgids = list(chip[10])
        self.counters = dict(counters)
        self.waits = [(m, list(fields)) for m, fields in waits]
        self.barrier = [(m, list(ops)) for m, ops in barrier]
        self.hazard_window = window
        #: The kernel library's hooks: each a value, or None ("not yet").
        self.lib_hooks = dict(_native.target_lib_hooks(name))
        #: The instruction-cache invalidate: (instruction, nop, nops, per_cu) or None.
        self.icache_inv = _native.target_icache_inv(name)
        self.phys = PHYS
        all_forms = [Form(name, head, tail) for head, tail in _native.target_forms(name)]
        self._forms = [f for f in all_forms if f.id >= 0]
        self._own = {f.name: f for f in all_forms if f.id < 0}

    def allocated_vgprs(self, vgpr: int, agpr: int) -> int:
        """The vector registers allocated for (vgpr, agpr)."""
        return _native.allocated_vgprs(self.name, vgpr, agpr)

    def spell(self, op: str) -> str:
        """The mnemonic of a dotted op; ValueError when the target has none."""
        mnemonic = _native.spell(self.name, op)
        if mnemonic is None:
            raise ValueError(f"{op}: {self.name} has no such op")
        return mnemonic

    def returning(self, name: str, th: str | None = None) -> bool:
        """Whether an atomic with this temporal hint (``th``) is its returning form."""
        return _native.returning(self.name, name, th)

    def form(self, mnemonic: str, arity: int, returning: bool = False) -> Form:
        """The form of a key; KeyError when there is none."""
        found = _native.form_id(self.name, mnemonic, arity, returning)
        if found is None:
            raise KeyError(f"{self.name}: no form {mnemonic} with {arity} operands")
        return self._forms[found]

    def forms(self) -> list[Form]:
        """Every target form in id order, own ops excluded."""
        return list(self._forms)

    def pseudo(self, name: str) -> Form:
        """An own op by name; KeyError when there is none."""
        try:
            return self._own[name]
        except KeyError:
            raise KeyError(f"{self.name}: no own op {name}") from None

    def __repr__(self) -> str:
        return f"Target({self.name})"


_TARGETS: dict[str, Target] = {}
_BUILDING = threading.Lock()
_CURRENT: contextvars.ContextVar[str | None] = contextvars.ContextVar("thin.target", default=None)


def target_names() -> list[str]:
    """The known target names in registry order."""
    return _native.target_names()


def target(name: str | None = None) -> Target:
    """The named target, else the current one, else the default."""
    if name is None:
        name = _CURRENT.get() or target_names()[0]
    view = _TARGETS.get(name)
    if view is None:
        _native.check_target(name)
        with _BUILDING:
            view = _TARGETS.get(name) or Target(name)
            _TARGETS[name] = view
    return view


@contextlib.contextmanager
def targeting(name: str) -> Iterator[Target]:
    """Makes a target current for the block and yields it."""
    view = target(name)
    token = _CURRENT.set(name)
    try:
        yield view
    finally:
        _CURRENT.reset(token)
