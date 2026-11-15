"""Register files and typed values: handles on the native Builder's values."""

from __future__ import annotations

import weakref

from thin import _native
from thin.dsl._expr import Expr, leaf
from thin.dsl._types import TYPES
from thin.ir import LOCS, VReg

BUILDERS: weakref.WeakValueDictionary = weakref.WeakValueDictionary()
"""The live Builders by identity: a value names its Builder by it."""


def typed(native) -> TypedValue:
    """The Python handle of a native typed value."""
    x = TypedValue.__new__(TypedValue)
    x._typed = native
    x._reg = None
    return x


def _bound(i) -> str:
    return "" if i is None else str(i)


def _update(name):
    def op(self, other):
        b = self.rf.b
        b._at()
        t = _native.tree_update(name, leaf(self), leaf(other))
        _native.assign(b._n, self._typed, ("t", t))
        return self

    return op


def _update_divmod(name):
    def op(self, d):
        b = self.rf.b
        b._at()
        t = getattr(Expr, {"div": "__floordiv__", "mod": "__mod__"}[name])(self, d)
        _native.assign(b._n, self._typed, ("t", t._n))
        return self

    return op


class TypedValue(Expr):
    """The author's handle on some dwords of a group.

    Temporary display names may repeat `t`; numeric value ids distinguish them.
    """

    __slots__ = ("__weakref__", "_reg", "_typed")

    @property
    def rf(self) -> RegFile:
        """Its register file."""
        return BUILDERS[self._typed.builder].files[self._typed.file]

    @property
    def file(self) -> str:
        """Its file letter."""
        return self._typed.file

    @property
    def name(self) -> str:
        """Its group's name."""
        return self._typed.name

    @property
    def type(self):
        """Its type."""
        return TYPES[self._typed.ty]

    @property
    def n(self) -> int:
        """Its elements."""
        return self._typed.n

    @property
    def sub(self) -> int:
        """Its dword offset into its group's first value."""
        return self._typed.sub

    @property
    def label(self) -> str | None:
        """Its label."""
        return self._typed.label

    @property
    def part(self) -> str:
        """Its part (``[1:3][0]``, ``.hi``)."""
        return self._typed.part

    @property
    def width(self) -> int:
        """Its dwords."""
        return self._typed.width

    @property
    def where(self) -> str:
        """The author's line."""
        return self._typed.where_

    def __asm_reg__(self) -> VReg:
        """Its register (made once)."""
        if self._reg is None:
            value = self.rf.b.values[self._typed.value]
            self._reg = VReg(value, self._typed.sub, self._typed.width)
        return self._reg

    def __getitem__(self, k):
        if isinstance(k, bool) or not isinstance(k, (int, slice)):
            raise TypeError(f"{self!r}[{k!r}]: an element index or a slice")
        if isinstance(k, int):
            return typed(self._typed.element(k))
        step = "" if k.step is None else f":{k.step}"
        written = f"{_bound(k.start)}:{_bound(k.stop)}{step}"
        return typed(self._typed.slice(k.start, k.stop, k.step, written))

    def __setitem__(self, k, x) -> None:
        """A statement: writes a tree into the view."""
        view = self if k is Ellipsis else self[k]
        b = self.rf.b
        b._at()
        _native.assign(b._n, view._typed, leaf(x))

    @property
    def lo(self) -> TypedValue:
        """The low half of a u64, ptr, mask or f32x2."""
        return typed(self._typed.half(0))

    @property
    def hi(self) -> TypedValue:
        """The high half of a u64, ptr, mask or f32x2."""
        return typed(self._typed.half(1))

    @property
    def mul(self) -> TypedValue:
        """A divisor argument's multiplier."""
        return typed(self._typed.divisor_part(1))

    @property
    def shr(self) -> TypedValue:
        """A divisor argument's shift."""
        return typed(self._typed.divisor_part(2))

    def __len__(self) -> int:
        return self._typed.n

    def __iter__(self):
        return (self[j] for j in range(self._typed.n))

    def __repr__(self) -> str:
        return repr(self._typed)

    __iadd__ = _update("add")
    __isub__ = _update("sub")
    __imul__ = _update("mul")
    __iand__ = _update("and")
    __ior__ = _update("or")
    __ixor__ = _update("xor")
    __ilshift__ = _update("shl")
    __irshift__ = _update("shr")
    __ifloordiv__ = _update_divmod("div")
    __imod__ = _update_divmod("mod")


def as_reg(x):
    """The register a typed value stands for; anything else as it is."""
    if isinstance(x, (list, tuple)) and any(isinstance(e, TypedValue) for e in x):
        return type(x)(as_reg(e) for e in x)
    protocol = getattr(x, "__asm_reg__", None)
    return x if protocol is None else protocol()


class Groups:
    """A file's groups: name to its list of values, in declaration order."""

    __slots__ = ("_f",)

    def __init__(self, f):
        self._f = f

    def _all(self) -> dict:
        values = self._f.b.values
        return {name: [values[k] for k in ids] for name, ids in self._f._facts()[1]}

    def __getitem__(self, name):
        return self._all()[name]

    def get(self, name, default=None):
        """The group's values, or default."""
        return self._all().get(name, default)

    def __contains__(self, name) -> bool:
        return name in self._all()

    def __len__(self) -> int:
        return len(self._all())

    def __iter__(self):
        return iter(self._all())

    def keys(self):
        """The group names."""
        return self._all().keys()

    def items(self):
        """(name, values) pairs."""
        return self._all().items()

    def values(self):
        """The groups' value lists."""
        return self._all().values()


def _maker(type_name):
    def make(self, init=None, *, n=1, name=None, at=None, same_as=None):
        return self._make(type_name, init, n, name, at, same_as)

    make.__name__ = type_name
    make.__doc__ = f"A new {type_name} value of this file ."
    return make


class RegFile:
    """A register file of a Builder."""

    __slots__ = ("__weakref__", "_b", "letter")

    def __init__(self, b, letter: str):
        self._b = weakref.ref(b)
        self.letter = letter

    @property
    def b(self):
        """Its Builder."""
        b = self._b()
        if b is None:
            raise ValueError(
                f"{self.letter}: a value before the Builder "
                "(make the Builder first: it names the values)"
            )
        return b

    def _facts(self) -> tuple:
        return _native.file_facts(self.b._n, self.letter)

    def _make(self, type_name, init, n, name, at, same_as):
        b = self.b
        b._at()
        exact = isinstance(n, int) and not isinstance(n, bool)
        y = None if same_as is None else _same_as(self, same_as)
        x = None if init is None else leaf(init)
        native = _native.value_make(
            b._n, self.letter, type_name, n if exact else None, repr(n), name, at, y, x
        )
        return typed(native)

    u32 = _maker("u32")
    i32 = _maker("i32")
    f32 = _maker("f32")
    u64 = _maker("u64")
    ptr = _maker("ptr")
    f32x2 = _maker("f32x2")

    def mask(self, init=None, *, name=None, at=None, same_as=None):
        """A new lane mask of this file's wave."""
        return self._make("mask", init, 1, name, at, same_as)

    @property
    def limit(self) -> int:
        """Its size."""
        return self._facts()[0]

    @property
    def vals(self) -> Groups:
        """Group name to its list of values."""
        return Groups(self)

    @property
    def values(self) -> list:
        """The file's values in order."""
        return [v for v in self.b.values if v.file == self.letter]

    @property
    def entered(self) -> list[str]:
        """The groups the hardware fills at entry."""
        return self._facts()[2]

    @property
    def pinned(self) -> list[str]:
        """The pinned groups in pin order."""
        return self._facts()[3]

    @property
    def lanes(self) -> list[str]:
        """The lanes-covered names."""
        return self._facts()[4]

    def alloc(self, name, count, width=1, align=1) -> list:
        """Declares a group; returns its values."""
        b = self.b
        b._at()
        ids = _native.file_alloc(b._n, self.letter, name, count, width, align)
        return [b.values[k] for k in ids]

    def pin(self, name, at) -> None:
        """Pins the group's values from ``at``."""
        _native.file_pin(self.b._n, self.letter, name, at)
        self.b._sync_values(links=True)

    def same_as(self, name, other) -> None:
        """The group shares the other's registers."""
        _native.file_same_as(self.b._n, self.letter, name, other)
        self.b._sync_values(links=True)

    def reg(self, name, idx=0, width=None, sub=0, label="") -> VReg:
        """``VReg(group[idx], sub, width or the value's n)``."""
        b = self.b
        value, _, lo, n = _native.file_reg(b._n, self.letter, name, idx, width, sub)
        return VReg(b.values[value], lo, n)

    def value_of(self, name) -> str:
        """The group at the end of the first value's same_as chain."""
        return _native.file_value_of(self.b._n, self.letter, name)

    def number(self, r) -> int:
        """The register's number (its value's first register plus lo)."""
        return _native.file_number(self.b._n, self.letter, _native_reg(r), repr(r), False)

    def phys(self, r) -> int:
        """The register's number once final."""
        return _native.file_number(self.b._n, self.letter, _native_reg(r), repr(r), True)

    def used(self) -> int:
        """The cursor before a placement, the highest placed register + 1 after."""
        return _native.file_used(self.b._n, self.letter)

    def overlap(self, r1, r2) -> bool:
        """Same file and overlapping numbers."""
        return self.b.overlap(r1, r2)

    def apply(self, placement) -> None:
        """Places each value at its register."""
        _native.file_apply(self.b._n, self.letter, [tuple(p) for p in placement])
        self.b._sync_values(links=True)

    def entry(self, *groups) -> list:
        """Declares the registers the hardware fills at entry."""
        b = self.b
        b._at()
        out = _native.file_entry(b._n, self.letter, [tuple(g) for g in groups])
        b._sync_values(links=True)
        return [typed(x) for x in out]

    def lanes_covered(self, *names) -> RegFile:
        """Adds lanes-covered names."""
        _native.file_lanes_covered(self.b._n, self.letter, list(names))
        return self

    def __repr__(self) -> str:
        return f"<file {self.letter}>"


def _same_as(f, y):
    native = getattr(y, "_typed", None)
    if native is None:
        raise ValueError(f"same_as={y!r}: a whole value of the {f.letter} file")
    return native


def _native_reg(r) -> tuple:
    v = r.value
    return (v.id, v.file, r.lo, r.n)


def capture() -> tuple[int, str]:
    """The author's location and its short text."""
    loc = LOCS.capture()
    return loc, LOCS.short(loc) if loc else ""
