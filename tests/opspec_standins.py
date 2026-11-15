"""Stand-in specs, kernels and Bounds for the opspec unit tests.

A kernel is a name, an address and an argument table; a Bound is the
runtime's record over such a kernel, with a ``run`` and an ``emulate`` that
record the call and write the outputs. Specs are made from field and
signature expressions by a class statement compiled under this file's name,
so their place in the registry is this file. (The golden tests run the
vectors' own texts, tests/test_opspec.py.)
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

import numpy as np

FILE = os.path.realpath(__file__)


def dsl():
    """The names a field or signature expression is evaluated with."""
    import thin.opspec as o

    return {n: getattr(o, n) for n in ("Int", "Choice", "Bool", "In", "Out", "InOut")}


def define(src: str, scope=None) -> dict:
    """Runs class statements compiled under this file's name; returns the namespace."""
    import thin.opspec as o

    ns = {"Spec": o.Spec, "thin": __import__("thin"), **dsl(), **(scope or {})}
    exec(compile(src, FILE, "exec"), ns)
    return ns


def spec_of(desc: dict, base=None, qualname="S", extra: dict | None = None) -> type:
    """A spec class from {name, fields: {n: expr}, signature: expr, group?}."""
    import thin.opspec as o

    body = {n: eval(e, dsl()) for n, e in desc.get("fields", {}).items()}
    sig = desc.get("signature", "{}")

    def signature(op):
        return eval(sig, {**dsl(), **op.field_values()})

    body["signature"] = signature
    body.update(extra or {})
    kw = {"name": desc["name"]}
    if desc.get("group"):
        kw["group"] = desc["group"]
    ns = define(
        f"class {qualname}(_Base, **_kw):\n    locals().update(_body)\n",
        {"_Base": base or o.Spec, "_kw": kw, "_body": body},
    )
    return ns[qualname]


def masked(text: str) -> str:
    """A message with each source path in parentheses written ``(<file>)``."""
    return re.sub(r"\((/[^)]*)\)", "(<file>)", text)


def field_fact(v):
    """A field value as the golden data encodes it."""
    from thin.opspec import Field

    if isinstance(v, Field):
        return {"Field": int(v), "range": list(v.range)}
    return {"type": type(v).__name__, "value": v}


@dataclass(eq=False)
class Kernel:
    """A stand-in kernel: a name, an address and argument records."""

    name: str
    args: dict = field(default_factory=dict)
    address: str = ""

    def __post_init__(self):
        self.address = self.address or f"standin/{self.name}"


def kernel(name: str, kargs: dict | None = None) -> Kernel:
    """A kernel from {arg: [lo, hi, bytes]} (lo/hi null: undeclared)."""
    args = {
        n: {"name": n, "lo": lo, "hi": hi, "bytes": nbytes, "kind": "u32", "owned": None}
        for n, (lo, hi, nbytes) in (kargs or {}).items()
    }
    return Kernel(name, args)


CALLS: list = []
"""The recorded runs: {call, kernel, sched}."""


def bound(k, args, grid=(1, 1, 1), co_resident=False, overread=None, effect=None):
    """A runtime Bound over a stand-in kernel whose emulate and run record and run effect."""
    from thin.runtime.bound import Bound

    class StandIn(Bound):
        def emulate(self, sched=None):
            CALLS.append({"call": "emulate", "kernel": self.kernel.name, "sched": sched})
            if callable(effect):
                effect(self)
            return {}

        def run(self):
            CALLS.append({"call": "run", "kernel": self.kernel.name})
            if callable(effect):
                effect(self)

    b = StandIn(k, tuple(grid), dict(args), co_resident, dict(overread or {}))
    b.effect = effect
    return b


def bound_fact(b) -> dict:
    """A Bound as the golden data encodes it."""
    return {
        "kernel": b.kernel.name,
        "grid": list(b.grid),
        "co_resident": b.co_resident,
        "overread": dict(b.overread),
        "group": None if b.group is None else repr(b.group),
    }


def binding(name: str, make, seen: list):
    """A binding function named name that records what it saw and returns make(op, t)."""

    def fn(op, **tunables):
        seen.append(
            {
                "fields": {n: field_fact(v) for n, v in op.field_values().items()},
                "group": None if op.group is None else repr(op.group),
                "tunables": dict(tunables),
            }
        )
        return make(op, tunables)

    fn.__name__ = fn.__qualname__ = name
    return fn


# --- runner.json's specs -----------------------------------------------------------------


def add_spec():
    """unit.add: z = x + y of f32, n elements, dists {scale: [1.0, 1e30]}."""
    import thin.opspec as o

    def signature(op):
        return {"x": o.In("f32", op.n), "y": o.In("f32", op.n), "z": o.Out("f32", op.n)}

    def inputs(op, seed, *, scale=1.0):
        i = np.arange(op.n, dtype=np.float64)
        x = ((i + seed) * scale).astype(np.float32)
        y = np.full(op.n, seed * scale, dtype=np.float64).astype(np.float32)
        return {"x": x, "y": y}

    def check(op, inputs, outputs):
        want = inputs["x"] + inputs["y"]
        got = outputs["z"]
        wrong = np.flatnonzero(got.view(np.uint32) != want.view(np.uint32))
        if not len(wrong):
            return None
        i = int(wrong[0])
        return f"z[{i}] = {float(got[i])}, want {float(want[i])} ({len(wrong)} wrong)"

    extra = {"signature": signature, "inputs": inputs, "check": check}
    extra["dists"] = {"scale": [1.0, 1e30]}
    return spec_of({"name": "unit.add", "fields": {"n": "Int(lo=1)"}}, qualname="Add", extra=extra)


def gsum_spec():
    """unit.gsum_run: every rank's y is the sum of the ranks' x."""
    import thin.opspec as o

    def signature(op):
        return {"x": o.In("u32", op.N), "y": o.Out("u32", op.N)}

    def inputs(op, seed):
        n = op.world if not GSUM_SHORT else 1
        return [{"x": (np.arange(op.N) + 10 * r + seed).astype(np.uint32)} for r in range(n)]

    def check(op, inputs, outputs):
        want = sum(i["x"].astype(np.uint64) for i in inputs).astype(np.uint32)
        for r, out in enumerate(outputs):
            wrong = np.flatnonzero(out["y"] != want)
            if len(wrong):
                i = int(wrong[0])
                return f"rank {r}: y[{i}] = {hex(int(out['y'][i]))}, want {hex(int(want[i]))}"
        return None

    desc = {"name": "unit.gsum_run", "group": "world"}
    desc["fields"] = {"world": "Int(lo=2, hi=8)", "N": "Int(lo=1)"}
    extra = {"signature": signature, "inputs": inputs, "check": check}
    return spec_of(desc, qualname="GsumRun", extra=extra)


GSUM_SHORT = False
"""Whether test.gsum_run's inputs give one rank's only."""


def gpull_spec():
    """unit.gpull_run: rank r's out is the sum over the ranks' y[r]; y shared."""
    import thin.opspec as o

    def signature(op):
        return {"y": o.In("u32", op.world, op.N, shared=True), "out": o.Out("u32", op.N)}

    def inputs(op, seed):
        return [
            {"y": np.full((op.world, op.N), 100 * r + seed, dtype=np.uint32)}
            for r in range(op.world)
        ]

    def check(op, inputs, outputs):
        for r, out in enumerate(outputs):
            want = sum(i["y"][r].astype(np.uint64) for i in inputs).astype(np.uint32)
            if not np.array_equal(out["out"], want):
                return f"rank {r}: out = {out['out'].tolist()}, want {want.tolist()}"
        return None

    desc = {"name": "unit.gpull_run", "group": "world"}
    desc["fields"] = {"world": "Int(lo=2, hi=8)", "N": "Int(lo=1)"}
    extra = {"signature": signature, "inputs": inputs, "check": check}
    return spec_of(desc, qualname="GpullRun", extra=extra)
