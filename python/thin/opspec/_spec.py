"""Specs, their fields and registry, and ops."""

from __future__ import annotations

import operator
import os
import re
import sys
from typing import ClassVar

from thin.opspec._tensor import check_given, place_of

__all__ = ["Bool", "Choice", "Field", "Int", "Spec"]

_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)+")
_PKG = "thin"


class Field(int):
    """A run-time field: an int that also carries the range the kernel serves."""

    def __new__(cls, value, lo, hi):
        """The field of value v serving lo.. hi."""
        f = super().__new__(cls, value)
        f.lo, f.hi = lo, hi
        return f

    @property
    def range(self) -> tuple[int, int]:
        """The served range ``(lo, hi)``."""
        return (self.lo, self.hi)

    def __str__(self) -> str:
        return int.__repr__(self)

    def __repr__(self) -> str:
        return f"{int.__repr__(self)} (served {self.lo} .. {self.hi})"


class Int:
    """An integer field in lo.. hi, both inclusive, none leaving a side open."""

    def __init__(self, default=None, lo=None, hi=None):
        """Refuses bounds that are no ints or cross, and a default outside them."""
        for side, b in (("lo", lo), ("hi", hi)):
            if b is not None and (isinstance(b, bool) or not isinstance(b, int)):
                raise TypeError(f"Int({side}={b!r}): an int or None")
        if lo is not None and hi is not None and lo > hi:
            raise ValueError(f"Int(lo={lo}, hi={hi}): lo above hi")
        self.default, self.lo, self.hi = default, lo, hi
        if default is not None:
            self.accept("Int", "default", default)

    def accept(self, spec: str, name: str, v):
        """The value an op holds for v: a plain int, or the Field itself."""
        ok = not isinstance(v, bool)
        if ok:
            try:
                operator.index(v)
            except TypeError:
                ok = False
        if not ok:
            raise TypeError(f"{spec}: {name} = {v!r}: an int")
        if not isinstance(v, Field):
            v = operator.index(v)
        if (self.lo is not None and v < self.lo) or (self.hi is not None and v > self.hi):
            raise ValueError(f"{spec}: {name} = {v} outside {self.lo} .. {self.hi}")
        return v

    def __repr__(self) -> str:
        return f"Int(default={self.default!r}, lo={self.lo!r}, hi={self.hi!r})"


def _member(options, v) -> bool:
    return any(type(o) is type(v) and o == v for o in options)


class Choice:
    """One of a few str, bool or int values, compared with their type."""

    def __init__(self, *options, default=None):
        """Refuses no options, other types, repeats and a default not among them."""
        if not options:
            raise ValueError("a Choice of one value at least")
        bad = [o for o in options if type(o) not in (str, bool, int)]
        if bad:
            raise TypeError(f"a Choice's values are str, bool or int, not {bad}")
        if any(_member(options[:i], o) for i, o in enumerate(options)):
            raise ValueError(f"a Choice's values once each: {options}")
        if default is not None and not _member(options, default):
            raise ValueError(f"a Choice's default {default!r}: one of {options}")
        self.options, self.default = options, default

    def accept(self, spec: str, name: str, v):
        """V itself when it is an option by type and value."""
        if not _member(self.options, v):
            raise ValueError(f"{spec}: {name} = {v!r}: one of {self.options}")
        return v

    def __repr__(self) -> str:
        return f"Choice{self.options!r}"


def Bool(default=None) -> Choice:  # noqa: N802 - the name
    """``Choice(False, True, default=default)``."""
    return Choice(False, True, default=default)


_REGISTRY: dict[str, tuple[type, str, str]] = {}
"""name → (class, real path of its file, qualified name)."""


def _place(cls) -> str:
    """The real path of the file holding the class statement (its code's file name)."""
    frame = sys._getframe(2)  # the class statement's frame (type.__new__ is native)
    while frame is not None and frame.f_code.co_name == "__init_subclass__":
        frame = frame.f_back
    path = frame.f_code.co_filename if frame is not None else ""
    if not path or path.startswith("<"):
        module = sys.modules.get(cls.__module__)
        path = getattr(module, "__file__", None) or path
    return os.path.realpath(path) if path and not path.startswith("<") else path


class Spec:
    """An op's contract; an instance is an op.

    Declared as ``class X(Spec, name="ns.op", group=None)``. A subclass
    defines ``signature``, ``inputs`` and ``check`` and may set ``dists``.
    """

    dists: ClassVar[dict] = {}
    """The data parameters a check runs at: name → values."""

    _name: str | None = None
    _group_field: str | None = None

    def __init_subclass__(cls, *, name=None, group=None, **kwargs):
        """Checks the group field, then the name, then registers the class."""
        super().__init_subclass__(**kwargs)
        q = cls.__qualname__
        # A field may not shadow an op's own attribute (key(), name(),...):
        # the names Spec defines, and group, which is the handle's keyword.
        own = {n for n in vars(Spec) if not n.startswith("__")} | {"group", "rank", "tensors"}
        clash = sorted(n for n in cls.fields() if n in own)
        if clash:
            raise TypeError(f"{q}: fields {clash} shadow an op's own attributes")
        # Each spec has its own dists: a subclass that sets none gets a copy.
        if "dists" not in vars(cls):
            cls.dists = dict(cls.dists)
        if group is not None:
            if not isinstance(cls.fields().get(group), Int):
                raise TypeError(f"{q}: group='{group}' names the Int field of the group's size")
            cls._group_field = group
        if name is None:
            raise TypeError(
                f'{q}: a spec names itself - class {cls.__name__}({_PKG}.Spec, name="namespace.op")'
            )
        if not isinstance(name, str) or not _NAME.fullmatch(name):
            raise ValueError(
                f'{q}: spec name {name!r}: dotted identifiers, a namespace first ("example.vadd")'
            )
        place = _place(cls)
        held = _REGISTRY.get(name)
        if held is not None and (held[1], held[2]) != (place, q):
            raise TypeError(
                f"spec name '{name}': {q} ({place}) and {held[2]} ({held[1]}) - one name one spec"
            )
        _REGISTRY[name] = (cls, place, q)
        cls._name = name

    @classmethod
    def name(cls) -> str:
        """The name."""
        return cls._name

    @classmethod
    def fields(cls) -> dict:
        """Name → field, from the most basic class to this one."""
        out: dict = {}
        for klass in reversed(cls.__mro__):
            for n, v in vars(klass).items():
                if isinstance(v, (Int, Choice)):
                    out[n] = v
        return out

    @classmethod
    def group_field(cls) -> str | None:
        """The Int field of a group spec's world, or None."""
        return cls._group_field

    def signature(self) -> dict:
        """Name → Tensor, a function of the field values."""
        raise NotImplementedError(f"{self.name()}: signature(op)")

    def inputs(self, seed, **dist):
        """The inputs of one run."""
        raise NotImplementedError(f"{self.name()}: inputs(op, seed, **dist)")

    def check(self, inputs, outputs):
        """None when the outputs are right, else why not."""
        raise NotImplementedError(f"{self.name()}: check(op, inputs, outputs)")

    def __init__(self, group=None, **kw):
        """An op: field values and tensors by name, checked in order."""
        spec = self.name()
        fields = type(self).fields()
        for n, f in fields.items():
            v = kw.get(n, f.default)
            if v is None:
                raise TypeError(f"{spec}: field {n} not given")
            self.__dict__[n] = f.accept(spec, n, v)
        sig = self.signature()
        clash = [n for n in sig if n in fields]
        if clash:
            raise TypeError(f"{spec}: tensors {clash} named as fields")
        unknown = sorted(n for n in kw if n not in fields and n not in sig)
        if unknown:
            raise TypeError(f"{spec}: unknown {unknown}")
        tensors = {n: v for n, v in kw.items() if n in sig}
        for n, v in tensors.items():
            check_given(spec, n, sig[n], v)
        places = {n: place_of(v) for n, v in tensors.items()}
        if len(set(places.values())) > 1:
            raise ValueError(f"{spec}: tensors on one device, not {places}")
        self.__dict__["_tensors"] = tensors
        self._take_group(spec, group, fields)
        for n, t in sig.items():
            if not t.shared:
                continue
            if self.group_field() is None:
                raise TypeError(f"{spec}: tensor {n} shared - a group spec's")
            given = tensors.get(n)
            if given is not None and (group is None or group.team.peers_of(given) is None):
                raise ValueError(
                    f"{spec}: tensor {n} is shared: its group's copy (g.tensor), given with the "
                    "group"
                )

    def _take_group(self, spec, group, fields) -> None:
        if group is not None:
            world = self.group_field()
            if world is None:
                raise TypeError(f"{spec}: not a group spec - an op of it takes no group=")
            if self.__dict__[world] != group.world:
                raise ValueError(
                    f"{spec}: {world} = {self.__dict__[world]}, its group's world {group.world}"
                )
        self.__dict__["_group"] = group

    def __getattr__(self, name):
        tensors = self.__dict__.get("_tensors", {})
        if name in tensors:
            return tensors[name]
        raise AttributeError(name)

    @property
    def tensors(self) -> dict:
        """The given tensors, name → value, as given."""
        return self.__dict__["_tensors"]

    @property
    def group(self):
        """The rank's handle, or None."""
        return self.__dict__["_group"]

    @property
    def rank(self):
        """The handle's rank, or None."""
        group = self.__dict__["_group"]
        return None if group is None else group.rank

    def field_values(self) -> dict:
        """Name → value of every field, in field order."""
        return {n: self.__dict__[n] for n in type(self).fields()}

    def key(self) -> str:
        """``"<name>{<f>=<v>,…}"`` over the fields sorted by name."""
        values = self.field_values()
        return self.name() + "{" + ",".join(f"{n}={values[n]}" for n in sorted(values)) + "}"

    def with_tensors(self, tensors, group=None):
        """A new op of the same fields with these tensors and handle."""
        return type(self)(group=group or self.group, **self.field_values(), **dict(tensors))

    def __repr__(self) -> str:
        return f"<op {self.key()}>"
