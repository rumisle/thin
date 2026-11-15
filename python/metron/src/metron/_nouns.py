"""The nouns of a race: Point, Arm, Call, State, the arm makers."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from types import MappingProxyType

__all__ = ["KINDS", "RESERVED", "UNCACHED", "Arm", "Call", "Point", "State", "co_arm", "torch_arm"]

#: The owned kinds.
KINDS = (
    "scratch",
    "kept",
    "carried",
    "gens",
    "kept.uncached",
    "carried.uncached",
    "gens.uncached",
    "mapped",
)
#: The kinds whose pools are uncached memory (`mapped` always is).
UNCACHED = ("kept.uncached", "carried.uncached", "gens.uncached", "mapped")
#: Names a point's tensors may not take: the Call's own attributes.
RESERVED = ("owned", "stream", "rank", "peers")


def _frozen(d) -> MappingProxyType:
    return MappingProxyType(dict(d))


class Point:
    """The inputs, the outputs' shapes and the check of a race."""

    __slots__ = ("check", "inputs", "key", "meta", "outputs")

    def __init__(self, key, inputs, outputs, check=None, meta=None):
        """A point; a list of dicts per side makes a group, one dict a rank."""
        group = isinstance(inputs, list)
        ins = inputs if group else [inputs]
        outs = outputs if isinstance(outputs, list) else [outputs]
        if len(ins) != len(outs) or not ins:
            raise ValueError(f"point {key}: {len(ins)} ranks' inputs, {len(outs)} ranks' outputs")
        for i, o in zip(ins, outs, strict=True):
            both = sorted(set(i) & set(o))
            reserved = sorted(n for n in (*i, *o) if n in RESERVED)
            not_tensors = sorted(n for n, v in (*i.items(), *o.items()) if not hasattr(v, "dtype"))
            if both or reserved or not_tensors or not o:
                raise ValueError(
                    f"point {key}: names in and out {both}, reserved {reserved}, not tensors "
                    f"{not_tensors}, outputs {sorted(o)}: none of the first three, "
                    "one output at least"
                )
        object.__setattr__(self, "key", key)
        object.__setattr__(self, "inputs", [_frozen(i) for i in ins] if group else _frozen(inputs))
        object.__setattr__(
            self, "outputs", [_frozen(o) for o in outs] if group else _frozen(outputs)
        )
        object.__setattr__(self, "check", check)
        object.__setattr__(self, "meta", _frozen(meta or {}))

    def __setattr__(self, name, value):
        raise AttributeError(f"Point is immutable ({name})")

    def ranks(self) -> tuple[list, list]:
        """(list of input dicts, list of output dicts), one per rank."""
        if isinstance(self.inputs, list):
            return list(self.inputs), list(self.outputs)
        return [self.inputs], [self.outputs]

    @property
    def world(self) -> int:
        """Its ranks."""
        return len(self.ranks()[0])

    def __repr__(self) -> str:
        return f"Point({self.key!r})"


def _owned_ok(entry) -> bool:
    if not (isinstance(entry, tuple) and len(entry) == 2):
        return False
    size, kind = entry
    if kind not in KINDS:
        return False
    if isinstance(size, bytes):
        return kind.startswith("kept") and len(size) > 0
    return type(size) is int and size > 0


@dataclass(frozen=True)
class Arm:
    """One way to compute a point's outputs."""

    name: str
    """Its name (in the record)."""
    launch: object
    """``launch(a)``: enqueues one call on ``a.stream``, given a Call."""
    fingerprint: str = ""
    """What identifies the arm's code."""
    owned: dict = field(default_factory=dict)
    """name -> (size, kind): its own memory (KINDS)."""
    checked: bool = True
    """False: wrong on purpose, judged but not by the check, and timed."""
    read: tuple = ()
    """Owned names read back after its judged replay (rank 0's)."""
    meta: dict = field(default_factory=dict)
    """What an adapter keeps beside the arm."""

    def __post_init__(self):
        owned = {n: tuple(e) if isinstance(e, list) else e for n, e in dict(self.owned).items()}
        bad = {n: e for n, e in owned.items() if not _owned_ok(e)}
        unknown = sorted(n for n in self.read if n not in owned)
        if not self.name or bad or unknown:
            raise ValueError(
                f"arm {self.name!r}: owned {bad} - (bytes > 0, one of {KINDS}) each, kept's bytes "
                f"or its first contents; read {unknown}: of owned"
            )
        object.__setattr__(self, "owned", _frozen(owned))
        object.__setattr__(self, "read", tuple(self.read))
        object.__setattr__(self, "meta", _frozen(self.meta))

    def unchecked(self) -> Arm:
        """The same arm with ``checked=False``."""
        return replace(self, checked=False)


class Call:
    """What an arm's launch is given for a call at copy i on rank r."""

    __slots__ = ("_peers", "_views", "owned", "rank", "stream")

    def __init__(self, views, owned, stream, rank=0, peers=None):
        """The views of copy i, the arm's buffers, the stream and the rank."""
        self._views, self.owned, self.stream, self.rank = views, owned, stream, rank
        self._peers = peers or (lambda name: [views[name]])

    def peers(self, name) -> list:
        """Every rank's view of copy i of that tensor (a group-shared tensor)."""
        return self._peers(name)

    def __getattr__(self, name):
        views = object.__getattribute__(self, "_views")
        if name in views:
            return views[name]
        raise AttributeError(f"the point has no tensor {name!r}")


@dataclass(frozen=True)
class State:
    """Which inputs stay warm and whether the code is cold."""

    warm: frozenset = frozenset()
    """Input names that are not rotated."""
    code: str = "warm"
    """"warm", or "cold": the instruction cache invalidated before every call."""

    def __post_init__(self):
        if self.code not in ("warm", "cold"):
            raise ValueError(f"state code {self.code!r}: 'warm' or 'cold'")
        object.__setattr__(self, "warm", frozenset(self.warm))

    def said(self) -> dict:
        """{"warm": sorted names, "code": code}."""
        return {"warm": sorted(self.warm), "code": self.code}


def torch_arm(name, fn, fingerprint=None) -> Arm:
    """An arm of a tensor-library call writing the point's outputs."""
    if fingerprint is None:
        import torch

        fingerprint = f"torch {torch.__version__}"
    return Arm(name, fn, fingerprint)


def co_arm(name, co, symbol, grid, kernarg, block=(256, 1), **kw) -> Arm:
    """An arm launching kernel `symbol` of code object `co` at `grid`.

    Its argument block is `kernarg(a)`.
    """
    from metron import _device

    def launch(a):
        p = _device.program(co, symbol, block)
        _device.launch(p, grid, kernarg(a), a.stream)

    kw.setdefault("fingerprint", hashlib.sha256(co).hexdigest()[:16])
    return Arm(name, launch, **kw)
