"""The language's types and the argument kinds."""

from __future__ import annotations

from thin import _native

_DWORDS = {
    "u32": 1,
    "i32": 1,
    "f32": 1,
    "u64": 2,
    "ptr": 2,
    "mask": 2,
    "mask32": 1,
    "f32x2": 2,
    "cond": 0,
}


class Kind:
    """An argument kind: a type, ranged or not, own memory, peers or a library's."""

    __slots__ = ("_k",)

    def __init__(self, k):
        self._k = k

    def multiple_of(self, m) -> Kind:
        """``.multiple_of(m)``."""
        exact = isinstance(m, int) and not isinstance(m, bool)
        return Kind(self._k.multiple_of(m if exact else None, repr(m)))

    def divisor(self) -> Kind:
        """``.divisor()``: a u32 the kernel divides by."""
        return Kind(self._k.divisor())

    @property
    def uncached(self) -> Kind:
        """``.uncached``: every access bypasses the L2."""
        return Kind(self._k.uncached())

    def __repr__(self) -> str:
        return self._k.text()


class DslType:
    """A DSL type: one object per type, compared by identity."""

    __slots__ = ("dwords", "name")

    def __init__(self, name: str):
        self.name = name
        self.dwords = _DWORDS[name]

    def __call__(self, x):
        """``T(x)``: x's bits read as T."""
        from thin.dsl._expr import leaf, tree
        from thin.dsl._values import typed

        native = getattr(x, "_typed", None)
        if self.name == "f32x2" and native is not None:
            return typed(_native.cast_pair(x.rf.b._n, native))
        return tree(_native.tree_cast(self.name, leaf(x)))

    def within(self, lo, hi) -> Kind:
        """``T.within(lo, hi)``: a ranged argument type."""
        return Kind(_native.kind_within(self.name, lo, hi))

    @property
    def kind(self) -> Kind:
        """The type as an argument kind."""
        return Kind(_native.kind_type(self.name))

    def __reduce__(self):
        return (_type, (self.name,))

    def __copy__(self) -> DslType:
        return self

    def __deepcopy__(self, memo) -> DslType:
        return self

    def __repr__(self) -> str:
        return self.name


TYPES = {name: DslType(name) for name in _DWORDS}
"""The types by name."""


def _type(name: str) -> DslType:
    return TYPES[name]


u32, i32, f32, u64, ptr, mask, mask32, f32x2, cond = (TYPES[n] for n in _DWORDS)

scratch = Kind(_native.kind_own("scratch"))
"""Own memory, any bytes on entry."""
kept = Kind(_native.kind_own("kept"))
"""Own memory, zero at first, left as found."""
carried = Kind(_native.kind_own("carried"))
"""Own memory, zero at first, what a launch leaves the next reads."""
gens = Kind(_native.kind_own("gens"))
"""Carried generation words starting at 2^32 - 3."""


def mapped(world) -> Kind:
    """A group's carried memory, one buffer per rank."""
    exact = isinstance(world, int) and not isinstance(world, bool)
    return Kind(_native.kind_ranked(False, world if exact else None, repr(world)))


def peers(world) -> Kind:
    """Every rank's copy of a group-shared tensor."""
    exact = isinstance(world, int) and not isinstance(world, bool)
    return Kind(_native.kind_ranked(True, world if exact else None, repr(world)))


def lib_arg(kind, library: str) -> Kind:
    """An argument a library adds and loads itself."""
    k = kind if isinstance(kind, Kind) else kind.kind
    return Kind(_native.kind_lib(k._k, library))
