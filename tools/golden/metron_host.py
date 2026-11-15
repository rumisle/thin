"""Construct host tensor/arm inputs and record metron's actual decisions."""

from __future__ import annotations

import metron
from metron import _memory, _nouns, _race
from tools.golden import metron_inputs as inputs


def result(function):
    """Record a scalar result or the precise public refusal."""
    try:
        return function()
    except Exception as error:
        return dict(error=type(error).__name__, message=str(error))


def nouns_data():
    """Point/arm/state validity, ownership kinds and race contract boundaries."""
    point_names = (
        "a name both in and out",
        "a reserved name",
        "an output reserved",
        "not a tensor",
        "no output",
        "no input is fine",
        "names rank and peers are reserved",
        "several faults at once",
        "group: ranks differ",
        "group: no ranks",
        "group: one rank's fault",
        "group of two ok",
    )
    cases = [
        dict(what="Point", case=name, result=result(lambda n=name: inputs._point_case(n) and "ok"))
        for name in point_names
    ]
    arm_inputs = [
        ("size 0", [0, "scratch"]),
        ("bytes for scratch", ["01", "scratch"]),
        ("bytes for kept", ["01", "kept"]),
        ("bytes for kept.uncached", ["0102", "kept.uncached"]),
        ("empty bytes for kept", ["", "kept"]),
        ("unknown kind", [8, "pinned"]),
        ("a bool size", [True, "scratch"]),
        ("a negative size", [-4, "carried"]),
        ("not a pair", "8"),
    ]
    arms = [dict(what="Arm", case=name, owned=dict(w=entry), read=[]) for name, entry in arm_inputs]
    arms.append(
        dict(
            what="Arm", case="every kind", owned={kind: [8, kind] for kind in _nouns.KINDS}, read=[]
        )
    )
    for name, read in (("read of a name not owned", ["t"]), ("read of an owned name", ["w"])):
        arms.append(dict(what="Arm", case=name, owned=dict(w=[8, "kept"]), read=read))
    arms.append(dict(what="Arm", case="empty name", owned={}, read=[]))
    for c in arms:
        c["result"] = result(
            lambda c=c: (
                metron.Arm(
                    "" if c["case"] == "empty name" else "a",
                    lambda a: None,
                    owned=inputs._owned_of(c),
                    read=tuple(c["read"]),
                )
                and "ok"
            )
        )
    cases += arms
    for code in ("warm", "cold", "hot"):
        cases.append(
            dict(
                what="State",
                case=f"code {code}",
                result=result(lambda c=code: metron.State(code=c) and "ok"),
            )
        )
    cases.append(
        dict(
            what="State.said",
            case="warm {y, x}, cold",
            result=metron.State(warm={"y", "x"}, code="cold").said(),
        )
    )
    for name, names, warm in (
        ("an arm twice", ["a", "a", "b", "b"], []),
        ("no arm", [], []),
        ("warm not an input", ["a"], ["w", "y"]),
        ("warm an input", ["a", "b"], ["x"]),
        ("all at once", ["a", "a"], ["w"]),
    ):
        point = inputs._point_case("no input is fine")
        x, y = inputs._xy()
        point = metron.Point("p", dict(x=x), dict(y=y))
        value = result(
            lambda names=names, point=point, warm=warm: (
                _race._refuse(
                    [metron.Arm(n, lambda a: None) for n in names],
                    point,
                    metron.State(warm=frozenset(warm)),
                )
                or "ok"
            )
        )
        cases.append(dict(what="race refusal", case=name, arms=names, warm=warm, result=value))
    cases.append(
        dict(
            what="Arm.unchecked",
            case="checked becomes false",
            result=metron.Arm("a", lambda a: None).unchecked().checked,
        )
    )
    return dict(
        notation=(
            "Synthetic constructor products and contract perturbations, evaluated through metron."
        ),
        kinds=list(_nouns.KINDS),
        uncached_kinds=[k for k in _nouns.KINDS if k in _nouns.UNCACHED],
        reserved=list(_nouns.RESERVED),
        cases=cases,
    )


def hygiene_data():
    """Constant patterns, sample gaps and exact threshold edges."""
    n = _memory.HYGIENE
    plans = [
        ("zeros_f32", "float32", [n]),
        ("ones_f32", "float32", [n]),
        ("arange_f32", "float32", [n]),
        ("small_zeros", "float32", [16]),
        ("just_under", "float32", [n // 4 - 1]),
        ("just_at", "float32", [n // 4]),
        *(
            (name, "float32", [n])
            for name in ("one_off_outside_slices", "one_off_in_last_slice", "one_off_first")
        ),
        ("bf16_sevens", "bfloat16", [n // 2]),
        ("u8_pattern", "uint8", [n * 2]),
        ("u8_const", "uint8", [n * 2]),
        ("matrix_const", "float32", [256, 256]),
    ]
    cases = [
        dict(
            name=name,
            tensor=dict(dtype=dtype, shape=shape, device="cpu"),
            contents=f"Generated {dtype} pattern {name}, shape {shape}.",
        )
        for name, dtype, shape in plans
    ]
    tensors = {c["name"]: inputs._hygiene_tensor(c) for c in cases}
    for c in cases:
        c["flagged"] = bool(metron.hygiene({c["name"]: tensors[c["name"]]}))
    return dict(
        notation=(
            "Native hygiene decisions on constants, "
            "sampled-slice perturbations and "
            "threshold sizes."
        ),
        cases=cases,
        flags=metron.hygiene(tensors),
    )


def layout_data():
    """Input/output stride geometry and per-kind pool placement/initialization."""
    torch = inputs._torch()
    copies = []
    for dtype, shape, io in (
        ("float32", [1], "in"),
        ("float32", [4], "out"),
        ("uint8", [0], "in"),
        ("uint8", [17], "out"),
        ("bfloat16", [1024, 3], "in"),
        ("float32", [1 << 20], "in"),
    ):
        tensor = (
            torch.zeros(shape, dtype=getattr(torch, dtype))
            if io == "in"
            else torch.empty(shape, dtype=getattr(torch, dtype), device="meta")
        )
        c = _memory.Copies(inputs._host(), 0, "t", tensor, io, 3)
        c.restore()
        row = c.row(0)
        copies.append(
            dict(
                tensor=dict(dtype=dtype, shape=shape, device="cpu"),
                io=io,
                nbytes=c.nbytes,
                stride=c.stride,
                gap_byte=int(row[c.nbytes]),
                first_byte=int(row[0]) if c.nbytes else None,
            )
        )
    plans = [
        [
            dict(name="a", owned=dict(w=[300, "scratch"], k=[16, "kept"])),
            dict(name="b", owned=dict(v=[8, "scratch"], u=[600, "scratch"], c=[4, "carried"])),
        ],
        [
            dict(name="a", owned=dict(k=[bytes((7, 8)).hex(), "kept"], z=[10, "kept"])),
            dict(name="b", owned=dict(k=[bytes([9] * 300).hex(), "kept"])),
        ],
        [
            dict(
                name="a",
                owned=dict(
                    g=[8, "gens"],
                    c=[300, "carried.uncached"],
                    h=[8, "gens.uncached"],
                    m=[12, "mapped"],
                    k=[bytes([1]).hex(), "kept.uncached"],
                ),
            )
        ],
        [dict(name="a", owned={})],
    ]
    owned = []
    for arms in plans:
        c = dict(arms=arms)
        pools = _memory.Pools(inputs._host(), 0, inputs._arms_of(c))
        c.update(
            at=[dict(arm=a, name=n, kind=k, offset=o, nbytes=b) for a, n, k, o, b in pools.places],
            pools={k: size for k, (_, size) in pools.pools.items()},
            contents={
                k: inputs._rle(buf[:size], words=k.startswith("gens"))
                for k, (buf, size) in pools.pools.items()
            },
            gap={
                k: inputs._rle(buf[size : size + _memory.GAP])
                for k, (buf, size) in pools.pools.items()
            },
        )
        owned.append(c)
    refusal = dict(
        case="an input of stride_max - GAP + 1 bytes",
        result=result(lambda: _memory.stride_of("huge", _memory.STRIDE_MAX - _memory.GAP + 1)),
    )
    return dict(
        notation=(
            "Native copy geometry and owned-pool "
            "allocation/initialization for explicit "
            "host inputs."
        ),
        copies=copies,
        owned=owned,
        copies_refusal=refusal,
    )


def judging_data():
    """Fault injection over every copy, memory class and group rank."""
    scenarios = (
        "every copy right",
        "every copy right, judged twice (memo)",
        "copy 2 wrong, others right",
        "two distinct wrong copies (1 and 3) and copy 0 right",
        "nothing written: copy 0 judged",
        "input copy 1 changed (and outputs wrong)",
        "an input's gap written in copy 0",
        "warm input changed",
        "output gap written in every copy alike",
        "output gap written in copy 0 and copy 2 only",
        "output gap written in copy 3 only",
        "unchecked arm with wrong outputs",
        "unchecked arm whose output gap is written",
        "kept buffer left changed",
        "kept pool changed outside the arm's buffers",
        "owned pool gap written",
        "scratch left dirty is fine",
        "input changed and kept changed: input first",
        "group: rank 1 copy 2 wrong",
        "group: rank 1 input changed",
        "group: rank 1 output gap",
        "group: rank 1 kept changed",
    )
    cases = []
    for name in scenarios:
        c = dict(
            case=name, group=name.startswith("group:"), checked=not name.startswith("unchecked")
        )
        c["copies"] = 4
        lay, arm = inputs._judging_layout(c)
        inputs._act(lay, name)
        c.update(wrong=lay.judge(arm), check_calls=lay.check_calls)
        if "judged twice" in name:
            lay.check_calls = 0
            c["again_same_outputs"] = dict(wrong=lay.judge(arm), check_calls=lay.check_calls)
        cases.append(c)
    return dict(
        notation="Native correctness/memory judging after explicit synthetic fault actions.",
        cases=cases,
    )


def adapter_data():
    """Evaluate the compiler adapter using thin's own vector-add example and test ops."""
    import metron.thin as mn
    import thin
    from examples.vadd_op import VAdd

    torch = inputs._torch()
    before = thin.use_store(None)
    try:
        points = []
        for n, seed, dist in ((1000, 0, None), (4096, 3, dict(scale=1e30))):
            op = VAdd(n=n)
            p = mn.point(op, seed, dist)
            right = dict(z=p.inputs["x"] + p.inputs["y"])
            points.append(
                dict(
                    op=f"example.vadd{{n={n}}}",
                    seed=seed,
                    dist=dist,
                    key=p.key,
                    inputs={n: inputs._tensor_said(t) for n, t in p.inputs.items()},
                    outputs={n: inputs._tensor_said(t) for n, t in p.outputs.items()},
                    check_right=p.check(p.inputs, right),
                    check_wrong=p.check(p.inputs, dict(z=torch.zeros_like(right["z"]))),
                )
            )
        plans = [("impls(point)", f"example.vadd/add {{'BLOCK': {block}}}") for block in (256, 512)]
        plans += [
            ("impl(point, add, BLOCK=512, name='mine', checked=False)", "mine"),
            (
                "impl(point, add) (a tunable not given takes its first value)",
                "example.vadd/add {'BLOCK': 256}",
            ),
            (
                "impl of an implementation of two kernels passing a buffer",
                "test.m_two/two {'BLOCK': 256}",
            ),
            (
                "impl of an implementation overreading x by 4096 bytes (accepted)",
                "test.m_near/near {'BLOCK': 256}",
            ),
        ]
        arms = []
        for made, name in plans:
            a = next(a for a in inputs._arms(dict(made_by=made)) if a.name == name)
            arms.append(
                dict(
                    made_by=made,
                    name=name,
                    owned={n: list(v) for n, v in a.owned.items()},
                    checked=a.checked,
                    read=list(a.read),
                    fingerprint_words=len(a.fingerprint.split(" ")),
                )
            )
        refusal_names = (
            (
                "impl(point, add, build={build_vadd: "
                "{'n': (1, 4096)}}): the lowering's own "
                "refusal passes through"
            ),
            "a kernel argument neither a tensor of the op nor a passed buffer",
            "an overread past the gap",
            "a tensor written in place",
            "laps of an arm none of whose kernels takes laps",
        )
        refusals = [
            dict(case=name, result=result(lambda name=name: inputs._refusal(name)))
            for name in refusal_names
        ]
    finally:
        thin.use_store(before)
    return dict(
        notation="Thin-compiled operation points, arms and adapter refusal boundaries.",
        points=points,
        arms=arms,
        refusals=refusals,
    )
