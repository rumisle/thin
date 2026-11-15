"""The IR's Python API.

The attribute readers are in:mod:`thin.ir.facts` and the line helpers in
:mod:`thin.ir.stream`; neither is re-exported here.

The names writers and the Python layers use. Instructions, the snapshot,
the analyses, the verifier and the printer are native (thin._native); this
module holds what is about the writer's world: values as objects, the
stream of rows, source locations, Diag rendering and the pass runner.
"""

from __future__ import annotations

import os
import sys
import sysconfig
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, ClassVar

from thin import _native
from thin.target import PHYS, Flags, Phys, target, target_names

__all__ = [
    "ABS",
    "ASSUME",
    "B32",
    "CLAIMS",
    "COMMENT",
    "DISCARD",
    "LABEL",
    "LOCS",
    "NEG",
    "PSEUDO_OPS",
    "UNDEFINED",
    "UNIFORM",
    "WAITS",
    "WAIT_ALL",
    "WAIT_LEFT",
    "WAIT_REGS",
    "WAIT_SMEM",
    "WAIT_TAGS",
    "Ablation",
    "AliasOk",
    "AsmError",
    "AssignError",
    "BlockRef",
    "Comment",
    "Diag",
    "ExecRegion",
    "ExprError",
    "Fence",
    "FlowError",
    "Hazard",
    "Imm",
    "InRegion",
    "Instr",
    "LdsAccess",
    "LocTable",
    "Loop",
    "M0Set",
    "MemError",
    "NoFixpoint",
    "Pass",
    "Phys",
    "Places",
    "Program",
    "Published",
    "RangeError",
    "ScheduleError",
    "Snapshot",
    "Staged",
    "Stream",
    "Tag",
    "Type",
    "VReg",
    "Value",
    "WaitUses",
    "WavesTaken",
    "always",
    "annotation",
    "from_native",
    "group_value",
    "internal",
    "is_wait",
    "name_text",
    "own",
    "placed_text",
    "positions",
    "program",
    "request",
    "run",
    "short",
    "text",
    "verify",
    "wait",
]

FILES = ("v", "s", "a")
"""The register file letters, in the order v < s < a."""

NEG = 1
"""The modifier bit for negation."""

ABS = 2
"""The modifier bit for absolute value."""


# --- Values and operands -----------------------------------------------


@dataclass(frozen=True)
class Type:
    """A frozen (name, dwords) record."""

    name: str
    """Free text: ``u32``, ``f32x2``, ``mask``."""

    dwords: int
    """The size of one element."""


B32 = Type("b32", 1)
"""The type of register-file group values."""


class Value:
    """A virtual register: n dwords of one file, not SSA.

    Identity is the object: copies are the object itself.
    """

    __slots__ = (
        "__weakref__",
        "_align",
        "_file",
        "_id",
        "_n",
        "_name",
        "_part_numbers",
        "_parts",
        "_type",
        "covered",
        "index",
        "loc",
        "pin",
        "same_as",
    )

    def __init__(self, id, file, n, align, type, name, loc=0, index=0):
        if file not in FILES:
            raise ValueError(f"register file {file!r}")
        self._id, self._file, self._n, self._align = id, file, n, align
        self._type, self._name = type, name
        self.loc, self.index = loc, index
        self.pin = None
        """A physical register the writer or the ABI fixed, or None."""
        self.same_as = None
        """The value whose registers this one shares, or None."""
        self.covered = False
        """Whether its partial-EXEC writes cover the lanes its reads use."""
        self._parts = None
        self._part_numbers: dict[str, int] = {}

    @property
    def id(self) -> int:
        """Its number in its program."""
        return self._id

    @property
    def file(self) -> str:
        """Its register file letter."""
        return self._file

    @property
    def n(self) -> int:
        """Its dwords."""
        return self._n

    @property
    def align(self) -> int:
        """Its alignment in dwords."""
        return self._align

    @property
    def type(self) -> Type:
        """Its type."""
        return self._type

    @property
    def name(self) -> str:
        """Its name."""
        return self._name

    @property
    def parts(self) -> dict:
        """Hand-laid names to dword ranges, made at first access."""
        if self._parts is None:
            self._parts = {}
        return self._parts

    @property
    def whole(self) -> VReg:
        """``VReg(v, 0, n)``."""
        return VReg(self, 0, self._n)

    def __getitem__(self, k) -> VReg:
        """Element k, or a contiguous slice of elements (a step is refused)."""
        d = self._type.dwords
        if isinstance(k, slice):
            start, stop, step = k.indices(self._n // d)
            if step != 1:
                raise ValueError(f"{self!r}[{k.start}:{k.stop}:{k.step}]: registers are contiguous")
            return VReg(self, start * d, max(stop - start, 0) * d)
        return VReg(self, k * d, d)

    def __copy__(self) -> Value:
        return self

    def __deepcopy__(self, memo) -> Value:
        return self

    def __repr__(self) -> str:
        return f"%{self._file}{self._id}:{self._name}"


def group_value(id, file, n, align, name, index=0) -> Value:
    """A value of type B32 with an unknown loc."""
    return Value(id, file, n, align, B32, name, 0, index)


class VReg:
    """Dwords lo..lo+n-1 of a value; ``part`` is not identity."""

    __slots__ = ("lo", "n", "part", "value")

    def __init__(self, value, lo, n, part=""):
        self.value, self.lo, self.n, self.part = value, lo, n, part

    @property
    def file(self) -> str:
        """The value's file letter."""
        return self.value.file

    def dwords(self) -> range:
        """``range(lo, lo+n)``."""
        return range(self.lo, self.lo + self.n)

    def dword(self, k) -> VReg:
        """``VReg(value, lo+k, 1)``, keeping the part."""
        return VReg(self.value, self.lo + k, 1, self.part)

    @property
    def through(self) -> str:
        """The name the register goes through: ``name`` or ``name.part``."""
        name = self.value.name
        return f"{name}.{self.part}" if self.part else name

    def __eq__(self, other) -> bool:
        if not isinstance(other, VReg):
            return NotImplemented
        return other.value is self.value and (other.lo, other.n) == (self.lo, self.n)

    def __hash__(self) -> int:
        return hash((id(self.value), self.lo, self.n))

    def __copy__(self) -> VReg:
        return self

    def __deepcopy__(self, memo) -> VReg:
        return self

    def __repr__(self) -> str:
        v = self.value
        at = f"+{self.lo}" if self.lo else ""
        return f"{v.file}:{v.name}[{v.index}]x{self.n}{at}"


@dataclass(frozen=True)
class Imm:
    """An immediate and the writer's spelling."""

    value: int | float
    """The bits as the slot reads them, or a float."""

    text: str = ""
    """The writer's text, kept only where str(value) differs."""


@dataclass(frozen=True)
class BlockRef:
    """A branch target, named by the label that starts the block."""

    name: str
    """The label."""


def _part_number(value, part: str) -> int:
    """A hand-laid part's number on its value: 1, 2, … as first met; 0 for none."""
    numbers = getattr(value, "_part_numbers", None) if part else None
    if numbers is None:
        return 0
    return numbers.setdefault(part, len(numbers) + 1)


def _native_reg(r: VReg) -> tuple:
    """A register as the native side takes it."""
    v = r.value
    return (v.id, v.file, r.lo, r.n, _part_number(v, r.part))


def _native_op(op) -> tuple:
    """An operand as the native side takes it; a non-operand is a TypeError."""
    if isinstance(op, VReg):
        return ("r", _native_reg(op))
    if isinstance(op, Phys):
        return ("p", op.name)
    if isinstance(op, Imm):
        return ("f" if isinstance(op.value, float) else "i", op.value, op.text)
    if isinstance(op, BlockRef):
        return ("b", op.name)
    raise TypeError(f"an operand: {op!r}")


def _op_of(row: _Row, item: tuple):
    """An operand read back: the register object given, else rebuilt."""
    kind, arg, text = item
    if kind == "r":
        return row.regs[arg]
    if kind == "p":
        return PHYS[arg]
    if kind == "b":
        return BlockRef(arg)
    return Imm(arg, text)


# --- Attributes --------------------------------------------------------

NONE = 0
"""An attribute's ``on`` for any form."""


@dataclass(frozen=True)
class Comment:
    """A note printed after the line."""

    on: ClassVar[int] = NONE
    text: str
    """The note; an empty one removes the attribute."""


@dataclass(frozen=True)
class Ablation:
    """The instruction is dropped when the build ablates one of these classes."""

    on: ClassVar[int] = NONE
    classes: frozenset[str]
    """The ablation classes."""


@dataclass(frozen=True)
class AliasOk:
    """Reads the alias check must not report."""

    on: ClassVar[int] = NONE
    pairs: frozenset[tuple[str, str | None]]
    """(read, written or None) pairs."""


@dataclass(frozen=True)
class Tag:
    """The writer's marker for wait bookkeeping."""

    on: ClassVar[int] = NONE
    value: tuple
    """The tag; ``("noorder", …)`` is noted apart."""


@dataclass(frozen=True)
class Staged:
    """An operand staged into a scratch SGPR before the instruction."""

    on: ClassVar[int] = NONE
    stage: tuple
    """What is staged."""


@dataclass(frozen=True)
class WaitUses:
    """Registers a wait just before this instruction named: uses of it."""

    on: ClassVar[int] = NONE
    regs: tuple
    """The registers."""


@dataclass(frozen=True)
class InRegion:
    """The lane region the instruction is in."""

    on: ClassVar[int] = NONE
    id: int
    """The region's id."""


@dataclass(frozen=True)
class Hazard:
    """Placed by the hazard pass, which drops and re-places these."""

    on: ClassVar[int] = NONE


@dataclass(frozen=True)
class Places:
    """Names an op that writes no register answers to."""

    on: ClassVar[int] = Flags.VMEM | Flags.DS
    names: tuple[str, ...]
    """The names."""


@dataclass(frozen=True)
class Published:
    """A device-scope atomic the coherence check must not judge."""

    on: ClassVar[int] = Flags.ATOMIC
    why: str
    """The reason."""


@dataclass(frozen=True)
class LdsAccess:
    """An LDS access the layout placed; its region is kept by identity."""

    on: ClassVar[int] = Flags.DS | Flags.VMEM
    region: Any
    """The region."""
    what: str
    """What it accesses."""
    offset: Any = None
    """Its offset."""
    field: int = 0
    """Its field."""
    ordered: str = ""
    """Its ordering."""


@dataclass(frozen=True)
class M0Set:
    """An M0 write of this constant that a DMA made ahead of itself."""

    on: ClassVar[int] = NONE
    value: int
    """The constant."""


@dataclass(frozen=True)
class WavesTaken:
    """The waves that take this skip branch; each below 64."""

    on: ClassVar[int] = Flags.BRANCH
    waves: frozenset[int]
    """The waves."""


@dataclass(frozen=True)
class Fence:
    """An LDS fence: nothing in the LDS lives past it."""

    on: ClassVar[int] = Flags.BARRIER


_ATTRS = {
    c.__name__: c
    for c in (
        Comment,
        Ablation,
        AliasOk,
        Tag,
        Staged,
        WaitUses,
        InRegion,
        Hazard,
        Places,
        Published,
        LdsAccess,
        M0Set,
        WavesTaken,
        Fence,
    )
}

_PAYLOAD: dict[type, Callable[[Any], Any]] = {
    Comment: lambda a: a.text,
    Ablation: lambda a: list(a.classes),
    AliasOk: lambda a: list(a.pairs),
    Tag: lambda a: a.value,
    Staged: lambda a: a,
    WaitUses: lambda a: (a, [_native_reg(r) for r in a.regs]),
    InRegion: lambda a: a.id,
    Hazard: lambda a: None,
    Places: lambda a: list(a.names),
    Published: lambda a: a.why,
    LdsAccess: lambda a: (a, (id(a.region), a.field, a.what, a.ordered)),
    M0Set: lambda a: a.value,
    WavesTaken: lambda a: list(a.waves),
    Fence: lambda a: None,
}

_REBUILD: dict[type, Callable[[Any], Any]] = {
    Comment: Comment,
    Ablation: lambda p: Ablation(frozenset(p)),
    AliasOk: lambda p: AliasOk(frozenset(p)),
    Tag: Tag,
    Staged: lambda p: p,
    WaitUses: lambda p: p,
    InRegion: InRegion,
    Hazard: lambda p: Hazard(),
    Places: Places,
    Published: Published,
    LdsAccess: lambda p: p,
    M0Set: M0Set,
    WavesTaken: lambda p: WavesTaken(frozenset(p)),
    Fence: lambda p: Fence(),
}


def _attr_name(cls) -> str:
    """The class's name, if it is an attribute class of the IR."""
    if _ATTRS.get(getattr(cls, "__name__", None)) is not cls:
        raise TypeError(f"{cls!r}: not an attribute of the IR")
    return cls.__name__


# --- Instructions ------------------------------------------------------


class _Row:
    """A stream row.

    It holds the native instruction, the register operand objects (by
    operand index, None elsewhere), its loc and the stream that holds it.
    """

    __slots__ = ("desc", "ins", "loc", "regs", "stream")

    def __init__(self, desc, ins, regs, loc, stream):
        self.desc, self.ins, self.regs, self.loc, self.stream = desc, ins, regs, loc, stream

    def copy(self, stream) -> _Row:
        """A new row of `stream` sharing this one's immutable data."""
        return _Row(self.desc, self.ins, self.regs, self.loc, stream)


class Instr:
    """Names a row of a stream, or a row it owns while no stream has it.

    Two handles are equal iff they name the same row: compare with ``==``.
    """

    __slots__ = ("_row",)

    def __init__(self, desc, ops, fields=None, *, mods=None, loc=0, attrs=None):
        ops = tuple(ops)
        native = [_native_op(op) for op in ops]
        items = list((fields or {}).items())
        ins = _native.Ins(
            desc.target, desc.id, native, items, None if mods is None else bytes(mods)
        )
        regs = tuple(op if isinstance(op, VReg) else None for op in ops)
        self._row = _Row(desc, ins, regs, loc, None)
        for a in (attrs or {}).values():
            self.put(a)

    @classmethod
    def _of(cls, row: _Row) -> Instr:
        """The handle of a row."""
        i = cls.__new__(cls)
        i._row = row
        return i

    def __eq__(self, other) -> bool:
        return isinstance(other, Instr) and other._row is self._row

    def __hash__(self) -> int:
        return id(self._row)

    def __deepcopy__(self, memo) -> Instr:
        return self

    @property
    def desc(self):
        """Its form."""
        return self._row.desc

    @property
    def ops(self) -> tuple:
        """The operands: register objects as given, the others rebuilt."""
        row = self._row
        return tuple(_op_of(row, item) for item in row.ins.ops())

    @property
    def fields(self) -> dict:
        """The fields, in stored order."""
        return dict(self._row.ins.fields())

    def field(self, name: str):
        """The field's value, or None (also for an unknown name)."""
        return self._row.ins.field(name)

    @property
    def mods(self) -> bytes | None:
        """The modifier bytes, one per slot, or None."""
        return self._row.ins.mods()

    def _facts(self) -> tuple:
        return self._row.ins.facts()

    @property
    def defs(self) -> tuple:
        """The register defs: the first operand object equal to each."""
        return tuple(self._row.regs[k] for k in self._facts()[1])

    @property
    def uses(self) -> tuple:
        """The register uses: the first operand object equal to each."""
        return tuple(self._row.regs[k] for k in self._facts()[2])

    @property
    def regs(self) -> tuple:
        """The register operand objects in order; empty for an own op."""
        if self.desc.flags & Flags.PSEUDO:
            return ()
        return tuple(r for r in self._row.regs if r is not None)

    @property
    def kind(self) -> str:
        """The form's kind name."""
        return self.desc.kind

    @property
    def counter(self) -> str | None:
        """The form's counter."""
        return self.desc.counter

    @property
    def op(self) -> str:
        """The mnemonic."""
        return self.desc.mnemonic

    @property
    def flags(self) -> int:
        """The form's flags."""
        return self.desc.flags

    def has(self, flag_bits: int) -> bool:
        """Whether the form has every one of these flags."""
        return flag_bits != 0 and self.desc.flags & flag_bits == flag_bits

    @property
    def writes(self) -> frozenset[str]:
        """The named registers it writes."""
        return frozenset(self._facts()[3])

    def phys_defs(self) -> frozenset[str]:
        """The named registers it writes."""
        return self.writes

    def phys_uses(self) -> frozenset[str]:
        """The named registers it reads."""
        return frozenset(self._facts()[4])

    @property
    def sgpr_valu_def(self) -> bool:
        """Whether it is a VALU that writes an SGPR."""
        return self._facts()[5][0]

    @property
    def writes_exec(self) -> bool:
        """Whether it writes EXEC."""
        return self._facts()[5][1]

    @property
    def m0_write(self) -> bool:
        """Whether it writes M0."""
        return self._facts()[5][2]

    @property
    def is_hazard(self) -> bool:
        """Whether the hazard pass placed it."""
        return "Hazard" in self._row.ins.attr_names()

    @property
    def target(self) -> str | None:
        """The first block operand's name."""
        return next((op.name for op in self.ops if isinstance(op, BlockRef)), None)

    @property
    def loc(self) -> int:
        """Its source location."""
        return self._row.loc

    @loc.setter
    def loc(self, loc: int) -> None:
        self._row.loc = loc

    @property
    def where(self) -> str:
        """``short(loc)``, or "" for loc 0."""
        return LOCS.short(self._row.loc) if self._row.loc else ""

    @property
    def text(self) -> str:
        """The printer's line with placeholder registers."""
        return text(self)

    def attr(self, cls):
        """A new instance of the attribute of that class, or None."""
        payload = self._row.ins.attr(_attr_name(cls))
        return None if payload is None else _REBUILD[cls](payload)

    @property
    def attrs(self) -> dict | None:
        """Class to instance for each present attribute, or None when there is none."""
        names = self._row.ins.attr_names()
        return {_ATTRS[n]: self.attr(_ATTRS[n]) for n in names} or None

    def put(self, attr) -> None:
        """Replaces the attribute of attr's class; empty data removes it."""
        cls = type(attr)
        name = _attr_name(cls)
        self._row.ins = self._row.ins.put(name, _PAYLOAD[cls](attr))

    def drop(self, cls) -> None:
        """Removes the attribute of that class."""
        self._row.ins = self._row.ins.drop(_attr_name(cls))

    def set_ops(self, ops) -> None:
        """Rebuilds the instruction with new operands, keeping fields, mods and attributes.

        A snapshot already taken keeps the instruction as it was: it holds
        the row's data at the time, not the row.
        """
        ops = tuple(ops)
        row = self._row
        row.ins = row.ins.with_ops([_native_op(op) for op in ops])
        row.regs = tuple(op if isinstance(op, VReg) else None for op in ops)

    def width_errors(self) -> list[str]:
        """One message per scalar register its slot's SGPR widths exclude."""
        return self._row.ins.width_errors([repr(op) for op in self.ops])

    def __repr__(self) -> str:
        ops = ", ".join(repr(op) for op in self.ops)
        return f"<{self.desc.name} {ops}>" if ops else f"<{self.desc.name}>"


# --- The IR's own ops --------------------------------------------------

_WAIT_KINDS = {"WaitRegs", "WaitTags", "WaitAll", "WaitLeft", "WaitSmem"}
_ANNOTATED = {"Discard", "Uniform", "Undefined", "Assume"}


def own(name: str, on=None):
    """The own op ``name`` (``ir.wait_regs``, ``ir.label``…) of a target.

    ``on`` is a target or its name; by default the current one
    (``thin.target.targeting``). Own ops are per target: a marker made for
    one target is that target's form.
    """
    if on is None or isinstance(on, str):
        on = target(on)
    return on.pseudo(name)


_default_own = target(target_names()[0]).pseudo
WAIT_REGS = _default_own("ir.wait_regs")
"""The default target's ``ir.wait_regs``; ``own`` gives another target's."""
WAIT_TAGS = _default_own("ir.wait_tags")
"""The default target's ``ir.wait_tags``."""
WAIT_ALL = _default_own("ir.wait_all")
"""The default target's ``ir.wait_all``."""
WAIT_LEFT = _default_own("ir.wait_left")
"""The default target's ``ir.wait_left``."""
WAIT_SMEM = _default_own("ir.wait_smem")
"""The default target's ``ir.wait_smem``."""
DISCARD = _default_own("ir.discard")
"""The default target's ``ir.discard``."""
ASSUME = _default_own("ir.assume")
"""The default target's ``ir.assume``."""
UNIFORM = _default_own("ir.uniform")
"""The default target's ``ir.uniform``."""
UNDEFINED = _default_own("ir.undefined")
"""The default target's ``ir.undefined``."""
LABEL = _default_own("ir.label")
"""The default target's ``ir.label``."""
COMMENT = _default_own("ir.comment")
"""The default target's ``ir.comment``."""
PSEUDO_OPS = {
    op.name: op
    for op in (
        WAIT_REGS,
        WAIT_TAGS,
        WAIT_ALL,
        WAIT_LEFT,
        WAIT_SMEM,
        DISCARD,
        ASSUME,
        UNIFORM,
        UNDEFINED,
        LABEL,
        COMMENT,
    )
}
"""The default target's own ops by name."""
WAITS = (WAIT_REGS, WAIT_TAGS, WAIT_ALL, WAIT_LEFT, WAIT_SMEM)
"""The default target's wait markers' ops (``is_wait`` tests any target's)."""
CLAIMS = (DISCARD, ASSUME, UNIFORM, UNDEFINED)
"""The default target's claims' ops."""


def wait(kind, counter, args, said="", ablated=False, *, on=None) -> Instr:
    """A wait marker of that kind on a target, by default the current one.

    ``counter`` must be one of that target's counters.
    """
    if kind not in ("regs", "tags", "all", "left", "smem"):
        raise ValueError(f"no wait request {kind!r}")
    op = own(f"ir.wait_{kind}", on)
    choices = next((f.choices for f in op.fields if f.name == "counter"), ())
    if choices and str(counter) not in choices:
        raise ValueError(f"wait: {counter!r} is not a counter of {op.target} ({choices})")
    fields: dict[str, Any] = {"counter": str(counter), "said": said}
    if ablated:
        fields["ablated"] = True
    ops: tuple = ()
    if kind == "regs":
        ops, fields["regions"] = tuple(args[0]), tuple(args[1])
    elif kind == "tags":
        fields["tags"] = tuple(args[0])
    elif kind == "left":
        fields["left"] = args[0]
    return Instr(op, ops, fields)


def is_wait(s: Instr) -> bool:
    """Whether s is a wait marker: by own-op kind, so on any target."""
    return s.desc.pseudo in _WAIT_KINDS


def request(s: Instr) -> tuple:
    """A wait marker's request: (counter, kind, …) as lists them."""
    counter = s.field("counter")
    kind = s.desc.pseudo
    if kind == "WaitRegs":
        return (counter, "regs", s.ops, s.field("regions") or ())
    if kind == "WaitTags":
        return (counter, "tags", s.field("tags"))
    if kind == "WaitLeft":
        return (counter, "left", s.field("left"))
    return (counter, "all" if kind == "WaitAll" else "smem")


# --- The printer ------------------------------------------


def _own(i: Instr) -> bool:
    return bool(i.desc.flags & Flags.PSEUDO)


def _line(i: Instr, spell: Callable[[VReg], str]) -> str:
    return i._row.ins.line([spell(r) for r in i._row.regs if r is not None])


def text(i: Instr) -> str:
    """The line with placeholder registers ``{0}``, ``{1}``…; an own op's are reprs."""
    if _own(i):
        return _line(i, repr)
    numbers = iter(range(len(i._row.regs)))
    return _line(i, lambda r: f"{{{next(numbers)}}}")


def annotation(i: Instr, reg_text: Callable[[VReg], str]) -> str | None:
    """A claim (discard, uniform, undefined or assume)'s line; None for anything else.

    ``reg_text`` spells each register, and a repr stands in where it cannot.
    """
    if i.desc.pseudo not in _ANNOTATED:
        return None

    def spell(r: VReg) -> str:
        try:
            return reg_text(r)
        except ValueError:
            return repr(r)

    return _line(i, spell)


def placed_text(r: VReg, base: int) -> str:
    """A register spelled physically, its value at ``base``."""
    return _native.placed_text(repr(r), r.file, base, r.lo, r.n, r.value.n)


def name_text(r: VReg, group: int) -> str:
    """A register spelled by name, its value in a group of ``group`` values."""
    v = r.value
    where = LOCS.short(v.loc) if getattr(v, "loc", 0) else None
    return _native.name_text((r.file, v.name, v.index, group, r.lo, r.n, v.n, where))


# --- The stream --------------------------------------------------------


class Stream:
    """Behaves like a list of Instr; edits rewrite the layout, never the rows."""

    __slots__ = ("_layout",)

    def __init__(self, xs=None):
        self._layout: list[_Row] = []
        self.extend(xs or ())

    def _take(self, x: Instr) -> _Row:
        """The row of this stream x names: moved in, its own, or a copy."""
        row = x._row
        if row.stream is None:
            row.stream = self
            return row
        return row if row.stream is self else row.copy(self)

    def _at(self, k: int) -> int:
        n = len(self._layout)
        if not -n <= k < n:
            raise IndexError("stream index out of range")
        return k % n

    def __len__(self) -> int:
        return len(self._layout)

    def __bool__(self) -> bool:
        return bool(self._layout)

    def __getitem__(self, k):
        if isinstance(k, slice):
            return [Instr._of(row) for row in self._layout[k]]
        return Instr._of(self._layout[self._at(k)])

    def __setitem__(self, k, x) -> None:
        if not isinstance(k, slice):
            self._layout[self._at(k)] = self._take(x)
            return
        xs = list(x)
        if k.step not in (None, 1):
            size = len(range(*k.indices(len(self._layout))))
            if size != len(xs):
                raise ValueError(
                    f"attempt to assign a sequence of size {len(xs)} "
                    f"to an extended slice of size {size}"
                )
        self._layout[k] = [self._take(i) for i in xs]

    def __delitem__(self, k) -> None:
        if isinstance(k, slice):
            del self._layout[k]
        else:
            del self._layout[self._at(k)]

    def __iter__(self):
        k = 0
        while k < len(self._layout):
            yield Instr._of(self._layout[k])
            k += 1

    def __reversed__(self):
        k = len(self._layout) - 1
        while 0 <= k < len(self._layout):
            yield Instr._of(self._layout[k])
            k -= 1

    def __contains__(self, x) -> bool:
        return isinstance(x, Instr) and any(row is x._row for row in self._layout)

    def append(self, x: Instr) -> None:
        """Takes x at the end."""
        self._layout.append(self._take(x))

    def extend(self, xs: Iterable[Instr]) -> None:
        """Takes each of xs at the end, xs read in full first."""
        self._layout.extend([self._take(x) for x in list(xs)])

    def insert(self, k: int, x: Instr) -> None:
        """Takes x at position k, clamped like list.insert."""
        self._layout.insert(k, self._take(x))

    def pop(self, k: int = -1) -> Instr:
        """Removes and returns the handle at k."""
        return Instr._of(self._layout.pop(self._at(k)))

    def remove(self, x: Instr) -> None:
        """Removes x's first position."""
        del self._layout[self.index(x)]

    def clear(self) -> None:
        """Empties the layout; the rows stay."""
        self._layout.clear()

    def index(self, x: Instr) -> int:
        """The first position of x."""
        for k, row in enumerate(self._layout):
            if isinstance(x, Instr) and row is x._row:
                return k
        raise ValueError("not in the stream")

    def rfind(self, x: Instr, before: int | None = None) -> int | None:
        """The last position of x below ``before``, or None."""
        stop = len(self._layout) if before is None else min(before, len(self._layout))
        for k in range(stop - 1, -1, -1):
            if self._layout[k] is x._row:
                return k
        return None

    def count(self, x: Instr) -> int:
        """How many positions name x."""
        return sum(1 for row in self._layout if isinstance(x, Instr) and row is x._row)

    def mix(self) -> dict[str, int]:
        """Kind name to count over the layout, without labels and comments."""
        out: dict[str, int] = {}
        for row in self._layout:
            kind = row.desc.kind
            if kind not in ("label", "comment"):
                out[kind] = out.get(kind, 0) + 1
        return out


def positions(ins, xs) -> list[int | None]:
    """The first position of each x in a stream or a list, or None; ins read once."""
    first: dict[int, int] = {}
    for k, i in enumerate(list(ins)):
        first.setdefault(id(i._row), k)
    return [first.get(id(x._row)) for x in xs]


# --- Programs and snapshots --------------------------------------------


@dataclass(frozen=True)
class Loop:
    """A loop the front end built, by its first and last instruction."""

    stem: str
    """The loop's stem."""
    first: Any
    """Its first instruction."""
    last: Any
    """Its last instruction."""
    conservative: bool = False
    """Whether its analysis is conservative."""


@dataclass(frozen=True)
class ExecRegion:
    """A lane region; ``save`` is None when every lane is on at entry."""

    id: int
    """Its id."""
    entry: Any
    """Its first instruction."""
    save: Any
    """The saved EXEC, or None."""
    checked: bool = False
    """Whether the full-EXEC assertion at entry is verified."""


class Program:
    """A kernel's IR: target, body stream, values, loops and lane regions."""

    def __init__(self, target, symbol=""):
        self.target = target
        self.symbol = symbol
        self.body = Stream()
        self.values: list[Value] = []
        self.loops: list[Loop] = []
        self.regions: list[ExecRegion] = []

    def value(self, file, n, type, *, align=1, name="", loc=0) -> Value:
        """Appends a value numbered by its position."""
        v = Value(len(self.values), file, n, align, type, name, loc)
        self.values.append(v)
        return v


class Snapshot:
    """An immutable native view of a stream; passes read it natively."""

    __slots__ = ("native", "rows")

    def __init__(self, native, rows):
        self.native = native
        """The native snapshot."""
        self.rows = rows
        """The rows it was made of, in layout order (their locs spell its messages)."""

    def wave_classes(self) -> list[dict[int, bool] | None]:
        """Per wave class its resolved skip branches, or None for every wave."""
        return [None if c is None else dict(c) for c in self.native.wave_classes()]


def program(unit, ins=None) -> Snapshot:
    """The snapshot of a unit's kernel, or of a free list of instructions.

    Gathering from a kernel reads the Builder's kernel and comes
    with it; a free list's registers are physical.
    """
    if unit is not None and hasattr(unit, "settle"):
        native = unit._n.snapshot()
        return Snapshot(native, list(unit.ins._layout))
    if ins is None:
        raise ValueError("a Program: a kernel or its instructions")
    rows = [i._row for i in ins]
    name = rows[0].desc.target if rows else target_names()[0]
    return Snapshot(_native.Snap.of_list(name, [row.ins for row in rows]), rows)


def verify(program, printable=False) -> None:
    """Raises a Diag at the body's first instruction that breaks an invariant.

    The checks are; with ``printable``, a wait marker fails.
    """
    rows = program.body._layout
    values = {}
    for row in rows:
        for r in row.regs:
            if r is not None:
                v = r.value
                pinned = getattr(v, "pin", None) is not None
                values[v.id] = (repr(v), v.n, getattr(v, "align", 1), pinned)
    found = _native.verify(program.target.name, [row.ins for row in rows], values, printable)
    if found is not None:
        k, msg = found
        raise Diag(msg, rows[k].loc)


_verify_program = verify


# --- Source locations -------------------------------------------------


_COMPREHENSIONS = frozenset({"<listcomp>", "<dictcomp>", "<setcomp>", "<genexpr>"})
"""The code names of comprehension frames, which locations skip."""


class LocTable:
    """Interns frames, names and fusions to small ints; Loc 0 is unknown.

    ``rule(code)`` says whether a code object is internal (not a writer's);
    its verdict is cached per code object.
    """

    def __init__(self, rule: Callable[[Any], bool]):
        self._rule = rule
        self._verdicts: dict[int, tuple[Any, bool]] = {}
        self._entries: list[tuple] = [("unknown",)]
        self._index: dict[tuple, int] = {}

    def __len__(self) -> int:
        return len(self._entries)

    def _internal(self, code) -> bool:
        cached = self._verdicts.get(id(code))
        if cached is None:
            cached = self._verdicts[id(code)] = (code, bool(self._rule(code)))
        return cached[1]

    def _intern(self, entry: tuple, key: tuple) -> int:
        loc = self._index.get(key)
        if loc is None:
            loc = self._index[key] = len(self._entries)
            self._entries.append(entry)
        return loc

    def capture(self, skip: int = 1) -> int:
        """The calling writer's Loc: up to four of its frames, outermost interned first."""
        f = sys._getframe(1)
        for _ in range(skip - 1):
            f = f.f_back if f is not None else None
        while f is not None and self._internal(f.f_code):
            f = f.f_back
        frames = []
        while f is not None and not self._internal(f.f_code) and len(frames) < 4:
            # Comprehensions are frames of their own only before Python
            # 3.12; leaving them out makes a chain the same on every version.
            if f.f_code.co_name not in _COMPREHENSIONS:
                frames.append((f.f_code, f.f_lineno))
            f = f.f_back
        loc = 0
        for code, line in reversed(frames):
            loc = self._intern(("frame", code, line, loc), ("frame", id(code), line, loc))
        return loc

    def name(self, text: str, inner: int = 0) -> int:
        """A name around another Loc."""
        return self._intern(("name", text, inner), ("name", text, inner))

    def fuse(self, *locs: int, what: str = "") -> int:
        """One Loc made of several, 0s and repeats dropped.

        Without ``what``, one Loc is itself and none is 0.
        """
        kept = tuple(dict.fromkeys(loc for loc in locs if loc))
        if not what and len(kept) <= 1:
            return kept[0] if kept else 0
        return self._intern(("fused", kept, what), ("fused", kept, what))

    def lines(self, loc: int) -> list[str]:
        """The Loc's lines: a frame chain innermost first, names and fusions indented."""
        entry = self._entries[loc]
        if entry[0] == "frame":
            out = []
            while entry[0] == "frame":
                _, code, line, caller = entry
                out.append(f"{os.path.relpath(code.co_filename)}:{line} ({code.co_name})")
                entry = self._entries[caller]
            return out
        if entry[0] == "name":
            inner = ["  " + x for x in self.lines(entry[2])] if entry[2] else []
            return [entry[1], *inner]
        if entry[0] == "fused":
            head = [entry[2]] if entry[2] else []
            return head + ["  " + x for member in entry[1] for x in self.lines(member)]
        return ["(unknown)"]

    def short(self, loc: int) -> str:
        """``<basename>:<line>`` of the Loc's frame, or ``(unknown)``."""
        entry = self._entries[loc]
        while entry[0] in ("name", "fused"):
            nxt = entry[2] if entry[0] == "name" else (entry[1][0] if entry[1] else 0)
            entry = self._entries[nxt]
        if entry[0] != "frame":
            return "(unknown)"
        return f"{os.path.basename(entry[1].co_filename)}:{entry[2]}"

    def where(self, loc: int) -> str:
        """`` (file:line)`` of a Loc, or "" for one the table does not know (0 included)."""
        return f" ({self.short(loc)})" if 0 < loc < len(self._entries) else ""


_PACKAGE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LIBRARY = os.path.join(_PACKAGE, "lib")
_SYSTEM = tuple(
    os.path.abspath(p)
    for key in ("stdlib", "platstdlib", "purelib", "platlib")
    if (p := sysconfig.get_paths().get(key))
)


def _inside(path: str, root: str) -> bool:
    return path == root or path.startswith(root + os.sep)


def internal(code) -> bool:
    """Whether a code object is the compiler's or Python's, not a writer's."""
    path = os.path.abspath(code.co_filename)
    if os.path.basename(path).startswith(("test_", "fuzz_")):
        return False
    if _inside(path, _LIBRARY):  # the kernel library is a writer, wherever it is installed
        return False
    return _inside(path, _PACKAGE) or any(_inside(path, root) for root in _SYSTEM)


LOCS = LocTable(internal)
"""The process's one location table."""


def short(loc: int) -> str:
    """``LOCS.short(loc)``."""
    return LOCS.short(loc)


# --- Diagnostics ------------------------------------------------------


class Diag(ValueError):
    """A user-facing refusal at a Loc; ``str(e)`` is ``render()``."""

    def __init__(self, msg: str = "", loc: int = 0, notes: tuple = ()) -> None:
        super().__init__(msg)
        self.loc = loc
        self.msg = msg
        self.notes = tuple(notes)

    def note(self, loc: int, msg: str) -> Diag:
        """Appends a note and returns self."""
        self.notes = (*self.notes, (loc, msg))
        return self

    def render(self) -> str:
        """The message, its location chain and its notes, from LOCS.

        Each caller is one ``at`` line; the table's own indentation of a
        named location's frames is dropped, so ``at`` aligns them.
        """
        head = f"{LOCS.short(self.loc)}: {self.msg}" if self.loc else self.msg
        chain = ["    at " + x.lstrip() for x in LOCS.lines(self.loc)[1:]] if self.loc else []
        notes = [f"  note: {LOCS.short(loc)}: {msg}" for loc, msg in self.notes]
        return "\n".join([head, *chain, *notes])

    def __str__(self) -> str:
        return self.render()


class ScheduleError(Diag):
    """A kernel the compiler refuses to build."""


class ExprError(ScheduleError):
    """Refused by the expressions."""


class FlowError(ScheduleError):
    """Refused by control flow."""


class MemError(ScheduleError):
    """Refused by the LDS layout or an access."""


class AssignError(ScheduleError):
    """Refused by register assignment."""


class RangeError(ScheduleError):
    """An address or operand not provably in bounds."""


class NoFixpoint(ValueError):
    """The wait-count search found no fixpoint."""


class AsmError(ValueError):
    """The encoder refused a line."""


_SCHEDULE_CLASSES = {
    "Schedule": ScheduleError,
    "Expr": ExprError,
    "Flow": FlowError,
    "Mem": MemError,
    "Assign": AssignError,
    "Range": RangeError,
}


def _spell(parts, line, where) -> str:
    """A native message's text.

    Lines are spelled by ``line(k)``, sources by ``where(k)`` (dropped
    without it).
    """
    pieces = []
    for kind, k, n, text_ in parts:
        if kind == "text":
            pieces.append(text_)
        elif kind == "where":
            pieces.append(where(k) if where else "")
        elif line is None:
            raise TypeError(f"a native message with an {kind} part (instruction {k}) and no line")
        else:
            pieces.append(line(k) if kind == "instr" else line(k)[:n])
    return "".join(pieces)


def from_native(cls_name: str, loc: int, parts, notes, line=None, where=None) -> Exception:
    """The exception a native Diag stands for.

    A part is ``(kind, k, n, text)``: ``text``, ``instr`` (instruction k's
    line), ``where`` (its source) or ``clip`` (the first n characters of
    its line).
    """
    msg = _spell(parts, line, where)
    spelled = tuple((nloc, _spell(nparts, line, where)) for nloc, nparts in notes)
    if cls_name == "NoFixpoint":
        return NoFixpoint(msg)
    if cls_name == "Asm":
        return AsmError(msg)
    if cls_name == "Diag":
        return Diag(msg, loc, spelled)
    return _SCHEDULE_CLASSES[cls_name](msg, loc, spelled)


# --- The pass runner --------------------------------------------------


def always(unit) -> bool:
    """A pass's default ``when``: always run."""
    return True


@dataclass(frozen=True)
class Pass:
    """A transform ``run(unit)`` or a check ``run(unit, snapshot)``."""

    name: str
    """The timing key."""
    run: Callable = field(compare=False)
    """The pass."""
    check: bool = False
    """Whether it reads a snapshot and changes nothing."""
    when: Callable = always
    """Whether it runs on a unit."""


def run(unit, passes, *, verify=True, printable=False):
    """Runs the passes in order, timing each into ``unit.pass_times``.

    Checks between two transforms share one snapshot. Returns the snapshot
    the last checks read, or None if no check ran after the last transform.
    """
    snap = None
    for p in passes:
        if not p.when(unit):
            continue
        start = time.perf_counter()
        try:
            if p.check:
                if snap is None:
                    snap = program(unit)
                p.run(unit, snap)
            else:
                p.run(unit)
                snap = None
        finally:
            unit.pass_times[p.name] = unit.pass_times.get(p.name, 0.0) + time.perf_counter() - start
    if verify:
        start = time.perf_counter()
        try:
            _verify_program(unit.program, printable)
        finally:
            unit.pass_times["verify"] = (
                unit.pass_times.get("verify", 0.0) + time.perf_counter() - start
            )
    return snap
