"""The LDS layout, regions, addresses and pointers: handles on the native layout."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass

from thin import _native
from thin.dsl._expr import leaf, tree
from thin.dsl._types import DslType
from thin.dsl._values import typed


@dataclass(frozen=True)
class Xor:
    """A swizzle.

    Element c along ``into`` is stored at ``c ^ ((idx[by] >> shift) & mask)``.
    """

    by: int
    """The dimension whose index keys it."""
    into: int
    """The dimension it permutes."""
    mask: int
    """The key's mask."""
    shift: int = 0
    """The key's shift."""


def _name_of(x) -> str:
    return getattr(x, "__name__", None) or type(x).__name__


def _off(x):
    return tree(x) if isinstance(x, _native.DslTree) else x


class Address:
    """An LDS address: a region, a byte offset and a label."""

    __slots__ = ("_a", "_l")

    def __init__(self, layout, a):
        self._l, self._a = layout, a

    def _address(self):
        return self._a

    @property
    def region(self) -> Region:
        """Its region."""
        return Region(self._l, self._a.region)

    @property
    def offset(self):
        """Its byte offset: an int or a tree."""
        return _off(self._a.off)

    @property
    def text(self) -> str:
        """Its text."""
        return self._a.text()

    def __add__(self, t):
        _refuse_two(self, t)
        return Address(self._l, self._a.plus(leaf(t)))

    __radd__ = __add__

    def __sub__(self, t):
        return Address(self._l, self._a.minus(leaf(t)))

    def __repr__(self) -> str:
        return self._a.text()


def _refuse_two(a, t):
    if getattr(t, "_address", None) is not None:
        from thin.ir import MemError

        raise MemError(f"{a!r} + {t!r}: an address plus a byte offset, not two addresses")


class Pointer:
    """A pointer into a region: a value holding an element's address."""

    __slots__ = ("_l", "_p")

    def __init__(self, layout, p):
        self._l, self._p = layout, p

    @property
    def value(self):
        """The value holding the address."""
        return typed(self._p.value)

    @property
    def region(self) -> Region:
        """The region it points into."""
        return Region(self._l, self._p.region)

    def at(self, *d) -> Address:
        """The element d further on (constant indices)."""
        b = self._l.b
        b._at()
        return Address(self._l, _native.lds_pointer_at(b._n, self._p, [leaf(x) for x in d]))

    def __asm_reg__(self):
        return self.value.__asm_reg__()

    @property
    def _typed(self):
        return self._p.value

    def __repr__(self) -> str:
        return repr(self.value)


class Region:
    """A region of the layout."""

    __slots__ = ("_k", "_l")

    def __init__(self, layout, k: int):
        self._l, self._k = layout, k

    def _info(self) -> tuple:
        return _native.lds_info(self._l.b._n, self._k)

    def _address(self):
        return _native.lds_address(self._l.b._n, self._k, ("i", 0))

    def __eq__(self, other) -> bool:
        return isinstance(other, Region) and other._l is self._l and other._k == self._k

    def __hash__(self) -> int:
        return hash((id(self._l), self._k))

    @property
    def name(self) -> str:
        """Its name."""
        return self._info()[0]

    @property
    def shape(self) -> tuple:
        """Its shape, outermost first."""
        return tuple(self._info()[1])

    @property
    def elem(self) -> str:
        """Its element as declared."""
        return self._info()[2]

    @property
    def strides(self) -> tuple:
        """Its strides in bytes."""
        return tuple(self._info()[4])

    @property
    def size(self) -> int:
        """Its bytes."""
        return self._info()[5]

    @property
    def align(self) -> int:
        """Its alignment."""
        return self._info()[7]

    @property
    def swizzle(self) -> Xor | None:
        """Its swizzle, or None."""
        s = self._info()[9]
        return None if s is None else Xor(*s)

    @property
    def parent(self) -> Region | None:
        """The region it is a part of, or None."""
        p = self._info()[8]
        return None if p is None else Region(self._l, p[0])

    @property
    def base(self) -> int:
        """Its byte offset in the layout (freezes the layout)."""
        return _native.lds_base(self._l.b._n, self._k)

    def parts(self, *names) -> list[Region]:
        """The outermost dimension's entries as regions of their own."""
        return [Region(self._l, k) for k in _native.lds_parts(self._l.b._n, self._k, list(names))]

    def tags(self) -> list[str]:
        """The names an op on it answers to for a wait."""
        return _native.lds_tags(self._l.b._n, self._k)

    def offset(self, *idx, swizzled=True):
        """The byte offset of element idx: an int or a tree."""
        return _off(_native.lds_offset(self._l.b._n, self._k, [leaf(x) for x in idx], swizzled))

    def stored(self, *idx) -> tuple:
        """The stored place of a logical element (the swizzle applied)."""
        return tuple(
            _off(x) for x in _native.lds_stored(self._l.b._n, self._k, [leaf(x) for x in idx])
        )

    def at(self, *idx) -> Address:
        """``A + A.offset(*idx)``."""
        return self + self.offset(*idx)

    def pointer(self, *idx, name=None) -> Pointer:
        """A value holding the address of element idx."""
        b = self._l.b
        b._at()
        return Pointer(self._l, _native.lds_pointer(b._n, self._k, [leaf(x) for x in idx], name))

    def __add__(self, t) -> Address:
        _refuse_two(self, t)
        return Address(self._l, _native.lds_address(self._l.b._n, self._k, leaf(t)))

    __radd__ = __add__

    def __asm_wait__(self, b):
        """The region as a wait's thing."""
        return ("g", self.name)

    def __repr__(self) -> str:
        return self.name


class Layout:
    """The kernel's LDS layout."""

    __slots__ = ("b",)

    def __init__(self, b):
        self.b = b

    @contextlib.contextmanager
    def _group(self, union, name, align, at):
        _native.lds_open(self.b._n, union, name, align, at)
        try:
            yield self
        finally:
            _native.lds_close(self.b._n)

    def struct(self, name=None, *, align=None, at=None):
        """A group whose members follow one another (a context manager)."""
        return self._group(False, name, align, at)

    def union(self, name=None, *, align=None, at=None):
        """A group whose members share their start (a context manager)."""
        return self._group(True, name, align, at)

    def region(
        self,
        name,
        shape=(),
        elem="u8",
        *,
        pitch=None,
        nbytes=None,
        align=None,
        at=None,
        swizzle=None,
    ):
        """Declares a region in the current group."""
        typed_elem = isinstance(elem, DslType)
        # Anything else is refused by its name (a function's, a class's) as
        # an unknown element.
        e = elem.name if typed_elem else elem if isinstance(elem, str) else _name_of(elem)
        dims = list(pitch.items()) if isinstance(pitch, dict) else None
        row = None if isinstance(pitch, dict) else pitch
        sw = None if swizzle is None else (swizzle.by, swizzle.into, swizzle.mask, swizzle.shift)
        args = (list(shape), e, row, dims, nbytes, align, at, sw)
        return Region(self, _native.lds_region(self.b._n, name, args, typed_elem))

    @property
    def size(self) -> int:
        """The layout's end (freezes it)."""
        return _native.lds_size(self.b._n)

    def leaves(self) -> list[Region]:
        """The regions in layout order."""
        return [Region(self, k) for k in _native.lds_leaves(self.b._n)]

    @property
    def regions(self) -> dict:
        """Name to region."""
        return {name: Region(self, k) for k, name in enumerate(_native.lds_names(self.b._n))}

    def table(self) -> list[str]:
        """The layout's table lines."""
        return _native.lds_table(self.b._n)
