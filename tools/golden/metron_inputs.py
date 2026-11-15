"""Construct synthetic host tensors, arms and fault actions for metron tests.

These functions are shared input constructors, not stored expected results.
They never read a golden document. Outputs are evaluated by metron itself.
"""

from __future__ import annotations

import pytest


def _torch():
    return pytest.importorskip("torch")


def _xy():
    torch = _torch()
    return torch.arange(4, dtype=torch.float32), torch.empty(4, device="meta")


def _point_case(text):
    import metron

    x, y = _xy()
    make = {
        "a name both in and out": lambda: metron.Point("p", {"x": x}, {"x": y}),
        "a reserved name": lambda: metron.Point("p", {"stream": x}, {"y": y}),
        "an output reserved": lambda: metron.Point("p", {"x": x}, {"owned": y}),
        "not a tensor": lambda: metron.Point("p", {"x": x}, {"y": [1, 2]}),
        "no output": lambda: metron.Point("p", {"x": x}, {}),
        "no input is fine": lambda: metron.Point("p", {}, {"y": y}),
        "names rank and peers are reserved": lambda: metron.Point(
            "p", {"rank": x, "peers": x}, {"y": y}
        ),
        "several faults at once": lambda: metron.Point(
            "p", {"x": x, "stream": x}, {"x": y, "q": 3}
        ),
        "group: ranks differ": lambda: metron.Point("p", [{"x": x}, {"x": x}], [{"y": y}]),
        "group: no ranks": lambda: metron.Point("p", [], []),
        "group: one rank's fault": lambda: metron.Point("p", [{"x": x}, {"x": x}], [{"y": y}, {}]),
        "group of two ok": lambda: metron.Point("p", [{"x": x}, {"x": x}], [{"y": y}, {"y": y}]),
    }[text]
    return make()


def _owned_of(case):
    out = {}
    for name, entry in case["owned"].items():
        if isinstance(entry, str):  # the case "not a pair": the entry is the int itself
            out[name] = int(entry)
            continue
        size, kind = entry
        out[name] = (bytes.fromhex(size) if isinstance(size, str) else size, kind)
    return out


def _result(fn):
    try:
        return fn()
    except ValueError as e:
        return {"error": "ValueError", "message": str(e)}


def _hygiene_tensor(case):
    torch = _torch()
    t = case["tensor"]
    shape, dtype = tuple(t["shape"]), getattr(torch, t["dtype"])
    n = 1
    for s in shape:
        n *= s
    name = case["name"]
    if name in ("zeros_f32", "small_zeros", "just_under", "just_at"):
        x = torch.zeros(n, dtype=dtype)
    elif name == "ones_f32":
        x = torch.ones(n, dtype=dtype)
    elif name == "arange_f32":
        x = torch.arange(n, dtype=dtype)
    elif name.startswith("one_off"):
        x = torch.zeros(n, dtype=dtype)
        x[{"one_off_outside_slices": 32868, "one_off_in_last_slice": n - 1}.get(name, 0)] = 1
    elif name == "bf16_sevens":
        x = torch.full((n,), 7.0, dtype=dtype)
    elif name == "u8_pattern":
        x = (torch.arange(n) % 256).to(dtype)
    elif name == "u8_const":
        x = torch.full((n,), 0x3C, dtype=dtype)
    else:
        x = torch.full((n,), -2.0, dtype=dtype)
    return x.reshape(shape)


def _host():
    from metron import _device

    return _device.HostBackend()


def _rle(t, words=False) -> list:
    data = t.view(t.new_empty(0).dtype).numpy().tobytes() if not words else None
    if words:
        import numpy as np

        vals = [f"{v:08x}" for v in np.frombuffer(t.numpy().tobytes(), "<u4")]
    else:
        vals = list(data)
    out = []
    for v in vals:
        if out and out[-1][0] == v:
            out[-1][1] += 1
        else:
            out.append([v, 1])
    return out


def _arms_of(case):
    import metron

    return [metron.Arm(a["name"], lambda c: None, owned=_owned_of(a)) for a in case["arms"]]


def _judging_point(group: bool):
    torch = _torch()
    import metron

    def wrong_at(x, z):
        bad = torch.nonzero(z != x + 1)
        return None if not len(bad) else f"z[{int(bad[0])}] wrong"

    def check(inputs, outputs):
        if not group:
            return wrong_at(inputs["x"], outputs["z"])
        for r, (i, o) in enumerate(zip(inputs, outputs, strict=True)):
            why = wrong_at(i["x"], o["z"])
            if why:
                return f"rank {r}: {why}"
        return None

    x = torch.arange(8, dtype=torch.float32)
    z = torch.empty(8, device="meta")
    if group:
        return metron.Point("j", [{"x": x}, {"x": x.clone()}], [{"z": z}, {"z": z}], check)
    return metron.Point("j", {"x": x}, {"z": z}, check)


def _judging_layout(case):
    import metron
    from metron import _race

    arm = metron.Arm(
        "a", lambda c: None, owned={"k": (8, "kept"), "s": (16, "scratch")}, checked=case["checked"]
    )
    warm = {"x"} if case["case"] == "warm input changed" else set()
    point = _judging_point(case["group"])
    lay = _race._Layout(_host(), [arm], point, metron.State(warm), 4, 2048)
    lay.restore()
    return lay, arm


def _write_right(lay, copies=None):
    for rk in lay.ranks:
        for i in copies if copies is not None else range(4):
            rk.copies["z"].view(i).copy_(rk.copies["x"].view(i) + 1)


def _act(lay, text):
    """Does what a judging case says, after restore and fresh."""
    r0, r1 = lay.ranks[0], lay.ranks[-1]
    z, x = r0.copies["z"], r0.copies["x"]
    gap = lambda rk, i: rk.copies["z"].row(i)[rk.copies["z"].nbytes :]  # noqa: E731
    if text.startswith("every copy right") or text in ("kept buffer left changed",
            "kept pool changed outside the arm's buffers", "owned pool gap written",
            "scratch left dirty is fine"):  # fmt: skip
        _write_right(lay)
    if text == "copy 2 wrong, others right":
        _write_right(lay)
        z.view(2)[3] = 99
    elif text == "two distinct wrong copies (1 and 3) and copy 0 right":
        _write_right(lay)
        z.view(1)[5] = 99
        z.view(3)[6] = 99
    elif text == "input copy 1 changed (and outputs wrong)":
        x.view(1)[2] = 50
    elif text == "an input's gap written in copy 0":
        x.row(0)[x.nbytes + 7] = 0
    elif text == "warm input changed":
        x.view(0)[0] = 50
    elif text.startswith("output gap written"):
        _write_right(lay)
        which = {"every copy alike": range(4), "copy 0 and copy 2 only": (0, 2),
                 "copy 3 only": (3,)}[text.removeprefix("output gap written in ")]  # fmt: skip
        for i in which:
            gap(r0, i)[0] = 1
    elif text == "unchecked arm whose output gap is written":
        gap(r0, 3)[0] = 1
    elif text == "kept buffer left changed":
        r0.pools.view("a", "k")[0] = 1
    elif text == "kept pool changed outside the arm's buffers":
        r0.pools.pools["kept"][0][100] = 1
    elif text == "owned pool gap written":
        buf, size = r0.pools.pools["scratch"]
        buf[size + 5] = 0
    elif text == "scratch left dirty is fine":
        r0.pools.view("a", "s").fill_(3)
    elif text == "input changed and kept changed: input first":
        x.view(1)[0] = 9
        r0.pools.view("a", "k")[0] = 1
    elif text == "group: rank 1 copy 2 wrong":
        _write_right(lay)
        r1.copies["z"].view(2)[0] = 99
    elif text == "group: rank 1 input changed":
        _write_right(lay)
        r1.copies["x"].view(1)[0] = 9
    elif text == "group: rank 1 output gap":
        _write_right(lay)
        gap(r1, 1)[0] = 1
    elif text == "group: rank 1 kept changed":
        _write_right(lay)
        r1.pools.view("a", "k")[0] = 1


def _ops():
    """Synthetic operations for compiler-adapter boundary checks.

    Cover passed buffers, overreads, unrelated arguments and in-place tensors.
    """
    import numpy as np

    import thin
    from examples.vadd import build_vadd
    from examples.vadd_op import VAdd
    from thin.opspec import In, InOut, Int, Out, Spec, implements
    from thin.runtime.bound import buffer

    if hasattr(_ops, "made"):
        return _ops.made

    def spec(name, inplace=False):
        class S(Spec, name=name):
            n = Int(lo=1)

            def signature(op):  # noqa: N805 - the spelling
                if inplace:
                    return {"x": InOut("f32", op.n)}
                return {"x": In("f32", op.n), "y": In("f32", op.n), "z": Out("f32", op.n)}

            def inputs(op, seed):  # noqa: N805
                return VAdd(n=op.n).inputs(seed)

            def check(op, inputs, outputs):  # noqa: N805
                return VAdd(n=op.n).check(inputs, outputs)

        return S

    serves = {"serves": lambda op, **_: {"n": (1, (1 << 30) - 1)}, "tunables": {"BLOCK": (256,)}}
    two, near, far, own = (spec(f"test.m_{s}") for s in ("two", "near", "far", "own"))
    inplace = spec("test.m_inplace", inplace=True)

    def k(op, BLOCK):  # noqa: N803
        return thin.compile(build_vadd, BLOCK=BLOCK, n=op.n.range)

    @implements(two, **serves)
    def two_impl(op, BLOCK):  # noqa: N803
        mid = buffer(op.n * 4, like=op.z)
        return [
            k(op, BLOCK)(x=op.x, y=op.y, z=mid, n=op.n),
            k(op, BLOCK)(x=mid, y=op.y, z=op.z, n=op.n),
        ]

    two_impl.__name__ = "two"

    def overreading(by):
        def bind(op, BLOCK):  # noqa: N803
            kk = k(op, BLOCK)
            args = {"x": op.x, "y": op.y, "z": op.z, "n": op.n}
            return kk.bind(-(-op.n // BLOCK), args, overread={"x": by})

        return bind

    near_fn = overreading(4096)
    near_fn.__name__ = "near"
    implements(near, needs={"overread": {"x": 4096}}, **serves)(near_fn)
    far_fn = overreading(8192)
    far_fn.__name__ = "far"
    implements(far, needs={"overread": {"x": 8192}}, **serves)(far_fn)

    def own_fn(op, BLOCK):  # noqa: N803
        return k(op, BLOCK)(x=np.zeros(op.n, np.float32), y=op.y, z=op.z, n=op.n)

    own_fn.__name__ = "own"
    implements(own, **serves)(own_fn)
    _ops.made = {"two": two, "near": near, "far": far, "own": own, "inplace": inplace}
    return _ops.made


def _tensor_said(t):
    return {
        "dtype": str(t.dtype).removeprefix("torch."),
        "shape": list(t.shape),
        "device": t.device.type,
    }


def _arms(case):
    import metron.thin as mn
    from examples.vadd_op import VAdd, add

    made = case["made_by"]
    p = mn.point(VAdd(n=1000))
    if made == "impls(point)":
        return mn.impls(p)
    if made.startswith("impl(point, add, BLOCK=512"):
        return [mn.impl(p, add, BLOCK=512, name="mine", checked=False)]
    if made.startswith("impl(point, add)"):
        return [mn.impl(p, add)]
    ops = _ops()
    spec = ops["two"] if "two kernels" in made else ops["near"]
    q = mn.point(spec(n=1000))
    return [mn.impl(q, "two" if "two kernels" in made else "near")]


def _refusal(text):
    import metron.thin as mn
    from examples.vadd import build_vadd
    from examples.vadd_op import VAdd, add

    p = mn.point(VAdd(n=1000))
    ops = _ops()
    if text.startswith("impl(point, add, build="):
        return mn.impl(p, add, build={build_vadd: {"n": (1, 4096)}})
    if text.startswith("a kernel argument"):
        return mn.impl(mn.point(ops["own"](n=1000)), "own")
    if text == "an overread past the gap":
        return mn.impl(mn.point(ops["far"](n=1000)), "far")
    if text == "a tensor written in place":
        return mn.point(ops["inplace"](n=4))
    arm = mn.impl(p, add)
    import metron

    r = metron.Race("p", {}, {arm.name: metron.ArmResult()}, "untimed", 0, {}, [], {}, {}, {}, 0.0)
    return mn.laps(arm, r)
