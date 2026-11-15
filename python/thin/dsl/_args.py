"""The argument block and the kernel's entry."""

from __future__ import annotations

from types import MappingProxyType

from thin import _native
from thin.dsl._expr import leaf
from thin.dsl._types import DslType, Kind
from thin.dsl._values import typed


class Args:
    """A kernel's arguments, laid out natively."""

    __slots__ = ("_a",)

    def __init__(self, at=None, **kinds):
        """Declares the block in keyword order; ``at`` places arguments at byte offsets."""
        given = [(name, _kind(k)) for name, k in kinds.items()]
        self._a = _native.DslArgs(given, dict(at or {}))

    @property
    def size(self) -> int:
        """The end of the last argument."""
        return self._a.size

    def names(self) -> list[str]:
        """Every argument's name in order."""
        return self._a.names()

    def own(self) -> list[str]:
        """The arguments without a library."""
        return self._a.own()

    def __getitem__(self, name) -> MappingProxyType:
        record = self._a.record(name)
        if record is None:
            raise KeyError(name)
        return MappingProxyType(record)

    def __contains__(self, name) -> bool:
        return self._a.record(name) is not None

    def __iter__(self):
        return iter(self._a.names())

    def __len__(self) -> int:
        return len(self._a.names())


def _kind(k):
    if isinstance(k, Kind):
        return k._k
    if isinstance(k, DslType):
        return k.kind._k
    return k


class Ranked(tuple):
    """A mapped or peers argument: each rank's pointer, then ``.own``."""

    own: object

    def __new__(cls, ranks, own):
        t = super().__new__(cls, ranks)
        t.own = own
        return t


def _loaded(x):
    if isinstance(x, tuple):
        return Ranked([typed(r) for r in x[0]], typed(x[1]))
    return typed(x)


class Entry:
    """The kernel's entry."""

    __slots__ = ("_b", "args")

    def __init__(self, b, args):
        self._b, self.args = b, args

    def _info(self) -> tuple:
        return _native.entry_info(self._b._n)

    @property
    def karg(self):
        """The argument pointer."""
        return typed(self._info()[0])

    @property
    def tid(self):
        """The lane's id in the workgroup."""
        return typed(self._info()[1])

    @property
    def wgid(self):
        """The workgroup id x."""
        return typed(self._info()[2])

    @property
    def wgid_y(self):
        """The workgroup id y, or None."""
        y = self._info()[3]
        return None if y is None else typed(y)

    @property
    def entry_moves(self) -> list:
        """The values the entry's workgroup-id moves copy, in order.

        The pipeline's drop_entry deletes the moves nothing reads.
        """
        return [typed(t) for t in _native.entry_moves(self._b._n)]

    @property
    def preload(self) -> int:
        """The preloaded dwords."""
        return self._info()[4]

    @property
    def wg(self) -> int:
        """The workgroup id dimensions."""
        return self._info()[5]

    @property
    def wgsize(self):
        """The workgroup's lanes, or None."""
        return self._info()[6]

    @property
    def lane(self):
        """The lane's index in its wave."""
        self._b._at()
        return typed(_native.entry_lane(self._b._n))

    @property
    def wave(self):
        """The wave's index in the workgroup, or 0."""
        self._b._at()
        w = _native.entry_wave(self._b._n)
        return w if isinstance(w, int) else typed(w)

    def load(self, *names, comment="") -> list:
        """The named arguments' values, loaded."""
        self._b._at()
        return [_loaded(x) for x in _native.entry_load(self._b._n, list(names), comment)]

    def load_rank(self, name, d, comment=""):
        """Rank d's pointer of a mapped or peers argument."""
        self._b._at()
        return typed(_native.entry_load_rank(self._b._n, name, leaf(d), comment))
