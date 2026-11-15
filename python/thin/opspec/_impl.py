"""Implementations, their pairs and served ranges, and lowering."""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

from thin.opspec._spec import Field, Int

__all__ = [
    "NEEDS",
    "Implementation",
    "candidates",
    "implementations",
    "implements",
    "lower",
    "lowered",
]

NEEDS = ("co_resident", "overread")
"""What a deployment must honour."""


@dataclass(frozen=True, eq=False)
class Implementation:
    """A binding function registered against a spec."""

    spec: type
    """The spec class it implements."""
    fn: object
    """The binding: ``fn(op, **tunables)`` → a Bound or a list of Bounds."""
    tunables: dict = field(default_factory=dict)
    """name → the tuple of its values."""
    covers: object = None
    """``covers(op, **tunables)`` → whether the pair admits the op; None: always."""
    serves: object = None
    """``serves(op, **tunables)`` → {Int field: (lo, hi)}; None: nothing served."""
    needs: dict = field(default_factory=dict)
    """co_resident and overread, as declared."""

    @property
    def name(self) -> str:
        """``"<spec name>/<function name>"``."""
        return f"{self.spec.name()}/{self.fn.__name__}"

    def __repr__(self) -> str:
        return f"<implementation {self.name}>"


_REGISTRY: dict[type, list[Implementation]] = {}


def _check_needs(spec, needs: dict) -> None:
    bad = [k for k in needs if k not in NEEDS]
    if bad:
        raise ValueError(f"{spec.name()}: needs {bad}: one of {list(NEEDS)}")
    if "overread" in needs:
        over = needs["overread"]
        ok = isinstance(over, dict) and all(
            isinstance(v, int) and not isinstance(v, bool) and v >= 1 for v in over.values()
        )
        if not ok:
            raise ValueError(
                f"{spec.name()}: needs overread {over!r}: {{tensor: bytes}}, a positive int each"
            )


def implements(spec, *, tunables=None, covers=None, serves=None, needs=None):
    """A decorator registering a binding function against spec."""
    needs = dict(needs or {})
    _check_needs(spec, needs)
    table = {n: tuple(v) for n, v in (tunables or {}).items()}
    empty = sorted(n for n, v in table.items() if not v)
    if empty:
        raise ValueError(f"{spec.name()}: tunables {empty} list no values")
    taken = sorted(n for n in table if n in ("seeds", "device", "scheds", "dists", "impl"))
    if taken:
        raise ValueError(f"{spec.name()}: tunables {taken} are check()'s own keywords")

    def register(fn):
        record = Implementation(spec, fn, table, covers, serves, needs)
        _REGISTRY.setdefault(spec, []).append(record)
        fn.implementation = record
        return fn

    return register


def implementations(spec) -> list:
    """The spec class's implementations in declaration order."""
    return list(_REGISTRY.get(spec, ()))


def served(impl: Implementation, op, tunables: dict) -> dict:
    """The pair's served ranges at op, validated."""
    if impl.serves is None:
        return {}
    ranges = dict(impl.serves(op, **tunables))
    fields = type(op).fields()
    for n, (lo, hi) in ranges.items():
        f = fields.get(n)
        if not isinstance(f, Int):
            raise ValueError(f"{impl.name} serves {n}: not an Int field of {op.name()}")
        inside = lo <= hi and (f.lo is None or lo >= f.lo) and (f.hi is None or hi <= f.hi)
        if not inside:
            raise ValueError(
                f"{impl.name} {tunables} serves {n} {lo} .. {hi}: outside the field's "
                f"{f.lo} .. {f.hi}"
            )
    return {n: (lo, hi) for n, (lo, hi) in ranges.items()}


def _excluding(op, ranges: dict) -> list[str]:
    values = op.field_values()
    return [
        f"serves {n} {lo} .. {hi}, not {values[n]}"
        for n, (lo, hi) in ranges.items()
        if not lo <= values[n] <= hi
    ]


def _covers(impl, op, tunables) -> tuple[bool, dict]:
    ranges = served(impl, op, tunables)
    if _excluding(op, ranges):
        return False, ranges
    return impl.covers is None or bool(impl.covers(op, **tunables)), ranges


def _pairs(impl: Implementation):
    names = list(impl.tunables)
    for values in itertools.product(*(impl.tunables[n] for n in names)):
        yield dict(zip(names, values, strict=True))


def candidates(op):
    """Yields (implementation, tunables, served ranges) for every covering pair."""
    for impl in implementations(type(op)):
        for tunables in _pairs(impl):
            ok, ranges = _covers(impl, op, tunables)
            if ok:
                yield impl, tunables, ranges


def _named(op, impl) -> Implementation:
    spec = type(op)
    if isinstance(impl, str):
        found = [i for i in implementations(spec) if i.name.endswith("/" + impl)]
        if not found:
            names = [i.name for i in implementations(spec)]
            raise ValueError(f"{spec.name()}: no implementation {impl!r} (of {names})")
        record = found[0]
    else:
        record = getattr(impl, "implementation", None)
        if record is None:
            name = getattr(impl, "__qualname__", repr(impl))
            raise ValueError(f"{spec.name()}: {name} is no registered implementation (@implements)")
    if record.spec is not spec:
        raise ValueError(f"{record.name} implements {record.spec.name()}, not {spec.name()}")
    return record


def _view(op, ranges: dict):
    """The served view: served fields as Fields, with the op's group."""
    if not ranges:
        return op
    values = {n: Field(v, *ranges[n]) if n in ranges else v for n, v in op.field_values().items()}
    return type(op)(group=op.group, **values, **op.tensors)


def arg_fact(rec, name: str):
    """A fact of a kernel's argument record: a mapping's key or an object's attribute."""
    if isinstance(rec, dict):
        return rec.get(name)
    return getattr(rec, name, None)


def _honoured(impl, tunables, ranges, bounds) -> None:
    """Every served field is an argument of a kernel declared over the whole range."""
    for n, (lo, hi) in ranges.items():
        taking = [b for b in bounds if n in b.kernel.args]
        if not taking:
            raise ValueError(
                f"{impl.name} {tunables} serves {n}, but no kernel it binds takes {n}: a served "
                "field is a kernel argument"
            )
        for b in taking:
            rec = b.kernel.args[n]
            lo_, hi_ = arg_fact(rec, "lo"), arg_fact(rec, "hi")
            bits = 8 * arg_fact(rec, "bytes")
            signed = str(arg_fact(rec, "kind") or "").startswith("i")
            klo = (-(1 << (bits - 1)) if signed else 0) if lo_ is None else lo_
            khi = ((1 << (bits - 1 if signed else bits)) - 1) if hi_ is None else hi_
            if not klo <= lo <= hi <= khi:
                raise ValueError(
                    f"{impl.name} {tunables} serves {n} {lo} .. {hi}, but {b.kernel.name} takes "
                    f"{n} in {klo} .. {khi}"
                )


def _needs_met(impl, tunables, op, bounds) -> None:
    """The Bounds say what the implementation declares."""
    declared = bool(impl.needs.get("co_resident", False))
    co = [b.kernel.name for b in bounds if b.co_resident]
    if bool(co) != declared:
        raise ValueError(
            f"{impl.name} {tunables}: needs co_resident {declared}, its kernels "
            f"{co or 'none'} co-resident"
        )
    over = impl.needs.get("overread", {})
    signature = op.signature()
    strangers = [n for n in over if n not in signature]
    if strangers:
        raise ValueError(f"{impl.name}: needs overread {strangers}: not tensors of {op.name()}")
    for b in bounds:
        for arg, nbytes in b.overread.items():
            bound = b.args.get(arg)
            tensor = next((n for n, v in op.tensors.items() if v is bound), None)
            if tensor is None or nbytes > over.get(tensor, 0):
                said = "no tensor of the op there" if tensor is None else over.get(tensor, 0)
                raise ValueError(
                    f"{impl.name} {tunables}: {b.kernel.name} overreads {arg} by {nbytes} bytes, "
                    f"the implementation declares {said}"
                )


def lowered(op, impl=None, **tunables):
    """(implementation, tunables, Bounds) of op."""
    missing = sorted(n for n in op.signature() if n not in op.tensors)
    if missing:
        raise TypeError(f"{op.key()}: lowered without its tensors {missing}")
    if op.group_field() is not None and op.group is None:
        raise TypeError(
            f"{op.key()}: a group op is one rank's call - made with group= its rank's handle"
        )
    if impl is not None:
        record = _named(op, impl)
        unknown = sorted(set(tunables) - set(record.tunables))
        if unknown:
            raise ValueError(
                f"{record.name}: no tunables {unknown} (its tunables: {sorted(record.tunables)})"
            )
        # A value outside the declared list is taken: an experiment's own.
        chosen = {n: v[0] for n, v in record.tunables.items()}
        chosen.update(tunables)
        ok, ranges = _covers(record, op, chosen)
        if not ok:
            why = "; ".join(_excluding(op, ranges))
            raise ValueError(
                f"{record.name} {chosen} does not cover {op.key()}" + (f" ({why})" if why else "")
            )
    else:
        if tunables:
            raise ValueError(
                f"{op.key()}: tunables {sorted(tunables)} without an implementation: name it "
                "(impl=) - the first covering one would take its own"
            )
        first = next(candidates(op), None)
        if first is None:
            raise ValueError(f"no implementation covers {op.key()}")
        record, chosen, ranges = first
    out = record.fn(_view(op, ranges), **chosen)
    bounds = list(out) if isinstance(out, (list, tuple)) else [out]
    for b in bounds:
        b.group = op.group
    _honoured(record, chosen, ranges, bounds)
    _needs_met(record, chosen, op, bounds)
    return record, chosen, bounds


def lower(op, impl=None, **tunables) -> list:
    """The Bounds alone of ``lowered``."""
    return lowered(op, impl, **tunables)[2]
