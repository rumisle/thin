"""Execute generated DSL kernels and compare their results on the CPU."""

from __future__ import annotations

import numpy as np
import pytest

LANES = 64
TREES = 12
U32_EDGES = [0, 1, 2, 16, 17, 64, 65, 1 << 31, (1 << 32) - 1, (1 << 32) - 16, (1 << 32) - 17]
F32_EDGES = [0.0, 1.0, -1.0, 0.5, 2.0, -16.0, 64.0, 65.0, 1e30, 3.0e-39]


@pytest.fixture
def no_store():
    import thin

    before = thin.use_store(None)
    yield
    thin.use_store(before)


def _u32_tree(rng, depth):
    """(dsl builder, numpy function) of a random u32 tree."""
    if depth == 0 or rng.random() < 0.25:
        pick = rng.integers(3)
        if pick == 0:
            return (lambda a, c: a), (lambda a, c: a)
        if pick == 1:
            return (lambda a, c: c), (lambda a, c: c)
        k = int(rng.choice(U32_EDGES))
        return (lambda a, c: a * 0 + k), (lambda a, c: np.full_like(a, k))
    op = rng.choice(["+", "-", "*", "&", "|", "^", "<<", ">>"])
    if op in ("<<", ">>"):
        s = int(rng.integers(32))
        dl, nl = _u32_tree(rng, depth - 1)
        if op == "<<":
            return (lambda a, c: dl(a, c) << s), (lambda a, c: nl(a, c) << np.uint32(s))
        return (lambda a, c: dl(a, c) >> s), (lambda a, c: nl(a, c) >> np.uint32(s))
    (dl, nl), (dr, nr) = _u32_tree(rng, depth - 1), _u32_tree(rng, depth - 1)
    fns = {
        "+": lambda x, y: x + y,
        "-": lambda x, y: x - y,
        "*": lambda x, y: x * y,
        "&": lambda x, y: x & y,
        "|": lambda x, y: x | y,
        "^": lambda x, y: x ^ y,
    }
    f = fns[op]
    return (lambda a, c: f(dl(a, c), dr(a, c))), (lambda a, c: f(nl(a, c), nr(a, c)))


I32_EDGES = [0, 1, -1, 2, -16, -17, 64, 65, -(1 << 31), (1 << 31) - 1]


def _i32_tree(rng, depth):
    """(dsl builder, numpy function) of a random i32 tree (>> arithmetic)."""
    if depth == 0 or rng.random() < 0.25:
        pick = rng.integers(3)
        if pick == 0:
            return (lambda a, c: a), (lambda a, c: a)
        if pick == 1:
            return (lambda a, c: c), (lambda a, c: c)
        k = int(rng.choice(I32_EDGES))
        return (lambda a, c: a * 0 + k), (lambda a, c: np.full_like(a, k))
    op = rng.choice(["+", "-", "*", "&", "|", "^", ">>"])
    if op == ">>":
        s = int(rng.integers(32))
        dl, nl = _i32_tree(rng, depth - 1)
        return (lambda a, c: dl(a, c) >> s), (lambda a, c: nl(a, c) >> np.int32(s))
    (dl, nl), (dr, nr) = _i32_tree(rng, depth - 1), _i32_tree(rng, depth - 1)
    fns = {
        "+": lambda x, y: x + y,
        "-": lambda x, y: x - y,
        "*": lambda x, y: x * y,
        "&": lambda x, y: x & y,
        "|": lambda x, y: x | y,
        "^": lambda x, y: x ^ y,
    }
    f = fns[op]
    return (lambda a, c: f(dl(a, c), dr(a, c))), (lambda a, c: f(nl(a, c), nr(a, c)))


def _f32_tree(rng, depth):
    """(dsl builder, numpy function) of a random f32 tree."""
    if depth == 0 or rng.random() < 0.25:
        pick = rng.integers(3)
        if pick == 0:
            return (lambda a, c: a), (lambda a, c: a)
        if pick == 1:
            return (lambda a, c: c), (lambda a, c: c)
        k = float(rng.choice(F32_EDGES))
        return (lambda a, c: a * 0.0 + k), (lambda a, c: a * np.float32(0.0) + np.float32(k))
    op = rng.choice(["+", "-", "*", "neg", "abs"])
    if op in ("neg", "abs"):
        # Source modifiers: a negated or absolute operand of the parent op.
        dl, nl = _f32_tree(rng, depth - 1)
        if op == "neg":
            return (lambda a, c: -dl(a, c)), (lambda a, c: -nl(a, c))
        return (lambda a, c: abs(dl(a, c))), (lambda a, c: np.abs(nl(a, c)))
    (dl, nl), (dr, nr) = _f32_tree(rng, depth - 1), _f32_tree(rng, depth - 1)
    fns = {"+": lambda x, y: x + y, "-": lambda x, y: x - y, "*": lambda x, y: x * y}
    f = fns[op]
    return (lambda a, c: f(dl(a, c), dr(a, c))), (lambda a, c: f(nl(a, c), nr(a, c)))


def _trees(kind: str, seed: int) -> list:
    rng = np.random.default_rng(seed)
    tree = {"u32": _u32_tree, "i32": _i32_tree, "f32": _f32_tree}[kind]
    return [tree(rng, 4) for _ in range(TREES)]


def trees_kernel(kind: str = "u32", seed: int = 0):
    """A kernel storing TREES random trees of `kind` drawn from `seed` (a generator)."""
    from thin import KernelDef
    from thin.dsl import Args, O, ptr, u32

    trees = _trees(kind, seed)

    def body(b, k):
        V = b.v  # noqa: N806 - the DSL's spelling
        make = {"u32": V.u32, "i32": V.i32, "f32": V.f32}[kind]
        x, y, z, _ = k.load()
        i = V.u32(k.tid, name="i")
        off = V.u32(i * 4, name="off")
        a, c = make(name="a"), make(name="c")
        b.emit(O.global_load(a, off, x))
        b.emit(O.global_load(c, off, y))
        b.wait(a, c, "a, c landed")
        for t, (dsl, _) in enumerate(trees):
            r = make(dsl(a, c), name=f"r{t}")
            o = V.u32(off + 4 * LANES * t, name=f"o{t}")
            b.emit(O.global_store(o, r, z))

    def host(bind, x, y, z, n):
        return bind(grid=1, args={"x": x, "y": y, "z": z, "n": n})

    args = Args(x=ptr, y=ptr, z=ptr, n=u32.within(1, LANES))
    return KernelDef(f"trees_{kind}", args, body, wgsize=LANES, max_grid=(1,), host=host)


@pytest.mark.parametrize("kind", ["u32", "i32", "f32"])
def test_random_expression_trees_match_numpy(no_store, kind):
    import thin

    seed = {"u32": 16, "i32": 18, "f32": 17}[kind]
    trees = _trees(kind, seed)
    rng = np.random.default_rng(seed + 100)
    if kind == "u32":
        a = rng.integers(0, 1 << 32, LANES, dtype=np.uint64).astype(np.uint32)
        c = np.array(U32_EDGES * 6, dtype=np.uint64)[:LANES].astype(np.uint32)
    elif kind == "i32":
        a = rng.integers(-(1 << 31), 1 << 31, LANES, dtype=np.int64).astype(np.int32)
        c = np.array(I32_EDGES * 7, dtype=np.int64)[:LANES].astype(np.int32)
    else:
        a = rng.standard_normal(LANES).astype(np.float32) * np.float32(100)
        c = np.array(F32_EDGES * 7, dtype=np.float32)[:LANES]
    z = np.zeros(LANES * len(trees), a.dtype)
    out = thin.compile(trees_kernel, kind=kind, seed=seed)(x=a, y=c, z=z, n=LANES).emulate()
    got = out["z"].reshape(len(trees), LANES)
    with np.errstate(all="ignore"):
        for t, (_, want) in enumerate(trees):
            w = want(a, c).astype(a.dtype)
            assert got[t].view(np.uint32).tolist() == w.view(np.uint32).tolist() or (
                kind == "f32" and _same_nan_class(got[t], w)
            ), f"tree {t}"


def _same_nan_class(got, want) -> bool:
    """NaNs are canonical on the emulator; numpy keeps payloads."""
    g, w = got.view(np.uint32), want.view(np.uint32)
    nan = np.isnan(want)
    return bool(np.all((g == w) | (nan & (g == 0x7FC00000))))


# --- Encoding moves ------------------------------------------------------

PLACES = ("vgpr", "sgpr", "inline", "literal")
#: Two-operand VALU operators per type, and numpy's meaning of each.
U32_OPS = {
    "+": lambda x, y: x + y,
    "-": lambda x, y: x - y,
    "*": lambda x, y: x * y,
    "&": lambda x, y: x & y,
    "|": lambda x, y: x | y,
    "^": lambda x, y: x ^ y,
    "<<": lambda x, y: x << (y & np.uint32(31)),
    ">>": lambda x, y: x >> (y & np.uint32(31)),
}
F32_OPS = {"+": lambda x, y: x + y, "-": lambda x, y: x - y, "*": lambda x, y: x * y}
_CONST = {"u32": {"inline": 7, "literal": 0x12345}, "f32": {"inline": 2.0, "literal": 3.75}}
_SCALAR = {"u32": 5, "f32": 1.5}


def _apply(op, a, b):
    return {
        "+": lambda: a + b,
        "-": lambda: a - b,
        "*": lambda: a * b,
        "&": lambda: a & b,
        "|": lambda: a | b,
        "^": lambda: a ^ b,
        "<<": lambda: a << b,
        ">>": lambda: a >> b,
    }[op]()


def _placed(kind, place, vec, scal):
    return {"vgpr": vec, "sgpr": scal}.get(place, _CONST[kind].get(place))


def _pairs():
    """Every placement pair with at least one register (two constants fold in Python)."""
    return [(p, q) for p in PLACES for q in PLACES if "vgpr" in (p, q) or "sgpr" in (p, q)]


def moves_kernel(kind="u32", op="+"):
    """out[pair][tid] = a op b for every placement pair of _pairs()."""
    from thin import KernelDef
    from thin.dsl import Args, O, f32, ptr, u32

    def body(b, k):
        V = b.v  # noqa: N806 - the DSL's spelling
        x, out, s = k.load()
        a = (V.u32 if kind == "u32" else V.f32)(name="a")
        b.emit(O.global_load(a, V.u32(k.tid * 4, name="off"), x))
        b.wait(a, "a landed")
        for j, (p, q) in enumerate(_pairs()):
            lhs, rhs = _placed(kind, p, a, s), _placed(kind, q, a, s)
            r = (V.u32 if kind == "u32" else V.f32)(_apply(op, lhs, rhs), name=f"r{j}")
            b.emit(O.global_store(V.u32((j * LANES + k.tid) * 4, name="o"), r, out))

    args = Args(x=ptr, out=ptr, s=u32 if kind == "u32" else f32)
    return KernelDef(f"moves_{kind}", args, body, wgsize=LANES, max_grid=(1,))


def _move_checks(text: str) -> list[str]:
    """properties of each VALU line of a rendered body on gfx1201: at
    most two constant-bus sources (distinct SGPRs, named registers and
    literals), one literal value at most, a VGPR src1 in the short form."""
    bad = []
    for line in text.splitlines():
        ins = line.strip().split("//")[0].strip()
        if not ins.startswith("v_") or ins.startswith(("v_nop", "v_readfirstlane")):
            continue
        name, _, rest = ins.partition(" ")
        srcs = [o.strip() for o in rest.split(",")][1:]
        scalar = {o for o in srcs if o[0] == "s" or o.startswith(("vcc", "m0"))}
        literal = {
            o for o in srcs if o[0] not in "sv" and not o.startswith("m0") and not _inline(o)
        }
        if len(scalar) + len(literal) > 2:
            bad.append(f"three constant-bus sources: {ins}")
        if len(literal) > 1:
            bad.append(f"two literals: {ins}")
        e32 = not name.endswith("_e64") and len(srcs) == 2
        if e32 and _is_vop2(name) and not srcs[1].startswith("v"):
            bad.append(f"a short form's src1 not a VGPR: {ins}")
    return bad


def _inline(o: str) -> bool:
    try:
        v = float(o) if "." in o else int(o, 0)
    except ValueError:
        return False
    return (isinstance(v, int) and -16 <= v <= 64) or v in (
        0.5,
        -0.5,
        1.0,
        -1.0,
        2.0,
        -2.0,
        4.0,
        -4.0,
    )


def _is_vop2(name: str) -> bool:
    """Whether the target has a short (VOP2) form of the mnemonic."""
    from thin.target import targeting

    with targeting("gfx1201") as t:
        return any(f.name == name and f.encoding == "ENC_VOP2" for f in t.forms())


@pytest.mark.parametrize(
    ("kind", "op"), [("u32", o) for o in U32_OPS] + [("f32", o) for o in F32_OPS]
)
def test_encoding_moves_keep_the_bus_and_the_value(no_store, kind, op):
    """every placement of {VGPR, SGPR, inline, literal} in the two
    sources of a VALU operator: the rendered instructions keep rules and
    the results equal numpy's."""
    import thin

    k = thin.compile(moves_kernel, kind=kind, op=op)
    assert _move_checks(k.asm) == []
    rng = np.random.default_rng(len(op) * 31 + (kind == "f32"))
    if kind == "u32":
        a = rng.integers(0, 1 << 32, LANES, dtype=np.uint64).astype(np.uint32)
        a[:4] = [0, 1, 31, (1 << 32) - 1]
    else:
        a = (rng.standard_normal(LANES) * 10).astype(np.float32)
    pairs = _pairs()
    out = np.zeros(LANES * len(pairs), a.dtype)
    k.bind(grid=1, args={"x": a, "out": out, "s": _SCALAR[kind]}).emulate()
    got = out.reshape(len(pairs), LANES)
    dt = np.uint32 if kind == "u32" else np.float32
    fn = (U32_OPS if kind == "u32" else F32_OPS)[op]
    with np.errstate(all="ignore"):
        for j, (p, q) in enumerate(pairs):
            lhs = np.asarray(_placed(kind, p, a, _SCALAR[kind]), dt)
            rhs = np.asarray(_placed(kind, q, a, _SCALAR[kind]), dt)
            want = np.broadcast_to(fn(lhs, rhs), (LANES,)).astype(dt)
            assert got[j].view(np.uint32).tolist() == want.view(np.uint32).tolist(), (p, q)


# --- Lane regions --------------------------------------------------

WG = 64
#: Each region's lanes, as the snippet below opens them (tid is the workgroup's lane).
REGIONS = {
    1: lambda t: t < 40,
    2: lambda t: (t < 40) & (t % 2 == 0),
    3: lambda t: (t >= 8) & (t < 16),
    4: lambda t: ((t % 32) >> 2) % 2 == 1,  # the int mask 0xF0F0F0F0 in each 32-lane wave
    5: lambda t: t < 20,
    6: lambda t: (t < 32) & (t >= 16),
    7: lambda t: t >= 0,
}


def lanes_kernel():
    """out[r][tid] = r inside region r, for supported instruction forms and a narrow."""
    from thin import KernelDef
    from thin.dsl import Args, O, ptr

    def body(b, k):
        V, S = b.v, b.s  # noqa: N806 - the DSL's spelling
        (out,) = k.load()
        t = k.tid

        def mark(r):
            o = V.u32((r * WG + t) * 4, name=f"o{r}")
            b.emit(O.global_store(o, V.u32(r, name=f"m{r}"), out))

        with b.lanes(t < 40):  # a condition, save-less
            mark(1)
            with b.lanes((t & 1) == 0, save=True):  # nested, saved
                mark(2)
            mark(1)  # restored to the outer region's lanes
        with b.lanes(t >= 8, t < 16):  # two conditions
            mark(3)
        with b.lanes(0xF0F0F0F0):  # an int mask
            mark(4)
        m = S.mask(t < 20, name="m")
        with b.lanes(m):  # a mask value
            mark(5)
        with b.lanes(t > 1000, skip=True):  # no lane: skipped
            mark(9)
        with b.lanes(t < 32):
            b.narrow(t >= 16)
            mark(6)
        mark(7)  # every lane again

    return KernelDef("lanes", Args(out=ptr), body, wgsize=WG, max_grid=(1,))


def test_the_exec_inside_a_region_is_the_entrys_and_its_masks(no_store):
    """the emulator's EXEC in each region is the entry's AND the
    masks; nesting restores; a narrow narrows; a skipped empty region runs nothing."""
    import thin

    k = thin.compile(lanes_kernel)
    out = np.zeros(10 * WG, np.uint32)
    k.bind(grid=1, args={"out": out}).emulate()
    got = out.reshape(10, WG)
    t = np.arange(WG)
    for r, lanes in REGIONS.items():
        want = np.where(lanes(t), r, 0)
        assert got[r].tolist() == want.tolist(), r
    assert not got[9].any() and not got[8].any() and not got[0].any()


# --- Magic division -----------------------------------------------

X_EDGES = [0, 1, 2, 1 << 31, (1 << 32) - 2, (1 << 32) - 1]


def _dividends(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 1 << 32, n, dtype=np.uint64).astype(np.uint32)
    x[: len(X_EDGES)] = X_EDGES
    return x


def div_kernel(ds=(3,)):
    """out[j] = x // ds[j] and out[len + j] = x % ds[j], each a constant divisor."""
    from thin import KernelDef
    from thin.dsl import Args, O, ptr, u32

    def body(b, k):
        V = b.v  # noqa: N806 - the DSL's spelling
        x, out, n = k.load()
        i = V.u32(k.wgid * LANES + k.tid, name="i")
        with b.lanes(i < n):
            v = V.u32(name="v")
            b.emit(O.global_load(v, V.u32(i * 4, name="off"), x))
            b.wait(v, "x landed")
            for j, d in enumerate(ds):
                for half, r in enumerate((V.u32(v // d, name="q"), V.u32(v % d, name="r"))):
                    o = V.u32(((half * len(ds) + j) * n + i) * 4, name="o")
                    b.emit(O.global_store(o, r, out))

    args = Args(x=ptr, out=ptr, n=u32.within(1, 1 << 16))
    return KernelDef("div_const", args, body, wgsize=LANES, max_grid=(1 << 10,))


def div_arg_kernel():
    """out = x // d, then x % d, d a divisor argument (its bind packs the pair)."""
    from thin import KernelDef
    from thin.dsl import Args, O, ptr, u32

    def body(b, k):
        V = b.v  # noqa: N806 - the DSL's spelling
        x, out, n, d = k.load()
        i = V.u32(k.wgid * LANES + k.tid, name="i")
        with b.lanes(i < n):
            v = V.u32(name="v")
            b.emit(O.global_load(v, V.u32(i * 4, name="off"), x))
            b.wait(v, "x landed")
            b.emit(O.global_store(V.u32(i * 4, name="o"), V.u32(v // d, name="q"), out))
            b.emit(O.global_store(V.u32((n + i) * 4, name="o"), V.u32(v % d, name="r"), out))

    args = Args(x=ptr, out=ptr, n=u32.within(1, 1 << 16), d=u32.within(1, 1 << 31).divisor())
    return KernelDef("div_arg", args, body, wgsize=LANES, max_grid=(1 << 10,))


def _check_const(ds, x):
    import thin

    k = thin.compile(div_kernel, ds=tuple(ds))
    out = np.zeros(2 * len(ds) * len(x), np.uint32)
    k.bind(grid=-(-len(x) // LANES), args={"x": x, "out": out, "n": len(x)}).emulate()
    got = out.reshape(2, len(ds), len(x))
    for j, d in enumerate(ds):
        assert (got[0, j] == x // np.uint32(d)).all(), d
        assert (got[1, j] == x % np.uint32(d)).all(), d


def _check_arg(ds, x):
    import thin

    k = thin.compile(div_arg_kernel)
    for d in ds:
        out = np.zeros(2 * len(x), np.uint32)
        args = {"x": x, "out": out, "n": len(x), "d": int(d)}
        k.bind(grid=-(-len(x) // LANES), args=args).emulate()
        assert (out[: len(x)] == x // np.uint32(d)).all(), d
        assert (out[len(x) :] == x % np.uint32(d)).all(), d


def _magic_ds() -> list[int]:
    import json

    from tests.golden import data_path

    return [c["d"] for c in json.loads((data_path("dsl", "magic.json")).read_text())["cases"]]


def test_magic_division_by_the_vectors_divisors(no_store):
    """x // d and x % d exact for every d of magic.json, constant
    and as a divisor argument; the full sweep is the opt-in test below."""
    x = _dividends(2048, 0)
    _check_const(_magic_ds(), x)
    rng = np.random.default_rng(1)
    _check_arg(_magic_ds()[:4] + [int(d) for d in rng.integers(1, 1 << 31, 8)], x[:512])


@pytest.mark.skipif(
    __import__("os").environ.get("THIN_EXHAUSTIVE") != "1", reason="set THIN_EXHAUSTIVE=1"
)
@pytest.mark.slow("exhaustive")
def test_magic_division_by_1000_random_divisors(no_store):
    """in full: 1000 random d on 20000 random x plus the edges."""
    rng = np.random.default_rng(2)
    ds = [int(d) for d in rng.integers(1, 1 << 31, 1000)]
    x = _dividends(20000, 3)
    for at in range(0, len(ds), 50):
        _check_const(_magic_ds() + ds[at : at + 50] if at == 0 else ds[at : at + 50], x)
    _check_arg(ds, x)


# --- Rewrites ------------------------------------------------------------

#: Each rewrite as (name, the expression over x, y and z = y & 7, the instruction it makes).
REWRITES = [
    ("fold +-", lambda x, y, z: (x + 0x7FFF_FFF0) - 0x7FFF_FFFF, None),
    ("fold -+", lambda x, y, z: (x - 300) + 44, None),
    ("fold ++ wraps", lambda x, y, z: (x + 0xFFFF_FFF0) + 0x20, None),
    ("low mask", lambda x, y, z: (y + (x << 8)) & 0xFF, None),
    ("low mask mirrored", lambda x, y, z: ((x << 5) + y) & 31, None),
    ("low mask of a product", lambda x, y, z: (y + x * 16) & 15, None),
    ("x * 1", lambda x, y, z: x * 1, None),
    ("x & all ones", lambda x, y, z: x & 0xFFFF_FFFF, None),
    ("x | 0", lambda x, y, z: x | 0, None),
    ("0 + x", lambda x, y, z: 0 + x, None),
    ("x << 0", lambda x, y, z: x << 0, None),
    ("x * 2^k", lambda x, y, z: x * 64, "lshlrev"),
    ("2^k * x", lambda x, y, z: 8 * x, "lshlrev"),
    ("shift add", lambda x, y, z: x + (y << 3), "lshl_add"),
    ("shift add mirrored", lambda x, y, z: (y << 9) + x, "lshl_add"),
    ("product add", lambda x, y, z: x + y * 4, "lshl_add"),
    ("shift or", lambda x, y, z: x | (y << 2), "lshl_or"),
    ("shift by a tree", lambda x, y, z: x + (y << z), "lshl_add"),
    ("bit field", lambda x, y, z: (x >> 5) & 0x3FF, "bfe"),
    ("bit field to the top", lambda x, y, z: (x >> 20) & 0xFFF, "bfe"),
]
#: The scalar file's shift adds (k 1.. 4) and the rest that apply to it.
S_REWRITES = [
    ("s shift add k=1", lambda x, y, z: x + (y << 1), "lshl1_add"),
    ("s shift add k=4", lambda x, y, z: (y << 4) + x, "lshl4_add"),
    ("s product add", lambda x, y, z: x + y * 2, "lshl1_add"),
    ("s fold", lambda x, y, z: (x + 5) - 9, None),
    ("s low mask", lambda x, y, z: (y + (x << 4)) & 7, None),
]


def _numpy(fn, x, y):
    with np.errstate(over="ignore"):
        return fn(x, y, y & np.uint32(7)).astype(np.uint32)


def rewrites_kernel():
    """out[j][tid]: rewrite j on vector x, y; then the scalar rewrites on the
    scalar arguments sx, sy, stored by every lane."""
    from thin import KernelDef
    from thin.dsl import Args, O, ptr, u32

    def body(b, k):
        V, S = b.v, b.s  # noqa: N806 - the DSL's spelling
        x, y, out, sx, sy = k.load()
        off = V.u32(k.tid * 4, name="off")
        a, c = V.u32(name="a"), V.u32(name="c")
        b.emit(O.global_load(a, off, x))
        b.emit(O.global_load(c, off, y))
        b.wait(a, c, "x, y landed")
        for j, (_, fn, _) in enumerate(REWRITES):
            r = V.u32(fn(a, c, c & 7), name=f"r{j}")
            b.emit(O.global_store(V.u32((j * LANES + k.tid) * 4, name="o"), r, out))
        base = len(REWRITES)
        for j, (_, fn, _) in enumerate(S_REWRITES):
            r = V.u32(S.u32(fn(sx, sy, sy & 7), name=f"s{j}"), name=f"sv{j}")
            b.emit(O.global_store(V.u32(((base + j) * LANES + k.tid) * 4, name="o"), r, out))

    args = Args(x=ptr, y=ptr, out=ptr, sx=u32, sy=u32)
    return KernelDef("rewrites", args, body, wgsize=LANES, max_grid=(1,))


def test_rewrites_are_exact(no_store):
    """each lowered expression equals its unrewritten meaning (numpy's
    wrapping arithmetic) on random operands, and makes its instruction."""
    import thin

    k = thin.compile(rewrites_kernel)
    for name, _, made in REWRITES + S_REWRITES:
        if made is not None:
            assert f"_{made}_" in k.asm or f"_{made}_b32" in k.asm, (name, made)
    rng = np.random.default_rng(7)
    for trial in range(3):
        a = rng.integers(0, 1 << 32, LANES, dtype=np.uint64).astype(np.uint32)
        c = rng.integers(0, 1 << 32, LANES, dtype=np.uint64).astype(np.uint32)
        a[:3], c[:3] = [0, (1 << 32) - 1, 1 << 31], [(1 << 32) - 1, 0, 7]
        sx, sy = (int(v) for v in rng.integers(0, 1 << 32, 2, dtype=np.uint64))
        n = len(REWRITES) + len(S_REWRITES)
        out = np.zeros(n * LANES, np.uint32)
        args = {"x": a, "y": c, "out": out, "sx": sx, "sy": sy}
        k.bind(grid=1, args=args).emulate()
        got = out.reshape(n, LANES)
        for j, (name, fn, _) in enumerate(REWRITES):
            assert got[j].tolist() == _numpy(fn, a, c).tolist(), (trial, name)
        for j, (name, fn, _) in enumerate(S_REWRITES):
            want = int(_numpy(fn, np.array([sx], np.uint32), np.array([sy], np.uint32))[0])
            assert (got[len(REWRITES) + j] == want).all(), (trial, name)


# --- Control-flow nests ----------------------------------------------------

M32 = (1 << 32) - 1


def _cond(rng, loop_vars):
    """A random wave-uniform condition: (operand, comparison, constant)."""
    operand = rng.choice(["a", "b", "c", *loop_vars])
    return str(operand), str(rng.choice(["==", "<", ">="])), int(rng.integers(0, 4))


def _nest(rng, depth, loop_vars, ids):
    """A random statement list: marks, if-chains (1-4 arms, with and without
    otherwise), counted loops with breaks, and exits."""
    out = []
    for _ in range(int(rng.integers(1, 4))):
        kind = rng.choice(["mark", "chain", "loop", "exit"], p=[0.4, 0.3, 0.2, 0.1])
        if depth == 0 or kind == "mark":
            out.append(("mark", next(ids)))
        elif kind == "chain":
            arms = [(_cond(rng, loop_vars), _nest(rng, depth - 1, loop_vars, ids))
                    for _ in range(int(rng.integers(1, 5)))]  # fmt: skip
            other = _nest(rng, depth - 1, loop_vars, ids) if rng.random() < 0.5 else None
            out.append(("chain", arms, other))
        elif kind == "loop":
            var = f"i{len(loop_vars)}"
            body = _nest(rng, depth - 1, [*loop_vars, var], ids)
            at = int(rng.integers(0, len(body) + 1))
            body.insert(at, ("break", _cond(rng, [var])))
            out.append(("loop", var, int(rng.integers(1, 4)), body))
        else:
            out.append(("exit", _cond(rng, loop_vars)))
    return out


class _ExitError(Exception):
    pass


class _BreakError(Exception):
    pass


def _holds(cond, env) -> bool:
    operand, cmp, k = cond
    v = env[operand]
    return {"==": v == k, "<": v < k, ">=": v >= k}[cmp]


def _run(nest, env, h):
    """The Python semantics: the path's hash, h = h * 33 + mark."""
    for st in nest:
        if st[0] == "mark":
            h = (h * 33 + st[1]) & M32
        elif st[0] == "chain":
            taken = next((body for cond, body in st[1] if _holds(cond, env)), st[2])
            if taken is not None:
                h = _run(taken, env, h)
        elif st[0] == "loop":
            _, var, n, body = st
            for i in range(n):
                env[var] = i
                try:
                    h = _run(body, env, h)
                except _BreakError as b:
                    h = b.args[0]
                    break
        elif st[0] == "break":
            if _holds(st[1], env):
                raise _BreakError(h)
        elif _holds(st[1], env):
            raise _ExitError(h)
    return h


def _emit(b, nest, env, h, loops):
    for st in nest:
        if st[0] == "mark":
            h[...] = h * 33 + st[1]
        elif st[0] == "chain":
            arms, other = st[1], st[2]
            for j, (cond, body) in enumerate(arms):
                with b.when(_dsl_cond(cond, env)) if j == 0 else b.elif_(_dsl_cond(cond, env)):
                    _emit(b, body, env, h, loops)
            if other is not None:
                with b.otherwise():
                    _emit(b, other, env, h, loops)
        elif st[0] == "loop":
            _, var, n, body = st
            for lp in b.loop(n, name=f"loop{var}"):
                _emit(b, body, {**env, var: lp.i}, h, [*loops, lp])
        elif st[0] == "break":
            loops[-1].break_when(_dsl_cond(st[1], env))
        else:
            b.exit_when(_dsl_cond(st[1], env), to="L_out")


def _dsl_cond(cond, env):
    operand, cmp, k = cond
    v = env[operand]
    return {"==": v == k, "<": v < k, ">=": v >= k}[cmp]


def _has_exit(nest) -> bool:
    for st in nest:
        if st[0] == "exit":
            return True
        if st[0] == "chain" and any(_has_exit(body) for _, body in st[1]):
            return True
        if st[0] == "chain" and st[2] is not None and _has_exit(st[2]):
            return True
        if st[0] == "loop" and _has_exit(st[3]):
            return True
    return False


def _random_nest(seed):
    import itertools

    return _nest(np.random.default_rng(seed), 3, [], itertools.count(1))


def flow_kernel(seed=0):
    """out[tid] = the hash of the marks a random nest of `seed` passes."""
    from thin import KernelDef
    from thin.dsl import Args, O, ptr, u32

    nest = _random_nest(seed)

    def body(b, k):
        out, a, bb, c = k.load()
        h = b.s.u32(0, name="h")
        _emit(b, nest, {"a": a, "b": bb, "c": c}, h, [])
        if _has_exit(nest):
            b.put_label("L_out", from_branches=True)
        b.emit(O.global_store(b.v.u32(k.tid * 4, name="o"), b.v.u32(h, name="hv"), out))

    args = Args(out=ptr, a=u32, b=u32, c=u32)
    return KernelDef(f"flow{seed}", args, body, wgsize=LANES, max_grid=(1,))


#: Seeds whose nests hold a chain, a loop and an exit each.
FLOW_SEEDS = [3, 4, 5, 7, 8, 9, 13, 15]


@pytest.mark.parametrize("seed", FLOW_SEEDS)
def test_random_control_flow_nests_take_the_python_path(no_store, seed):
    """random nests of chains (1-4 arms, otherwise, arms that
    branch out) and counted loops with breaks and exits: the path the emulator
    takes is the one Python's semantics give, for random uniform values."""
    import thin

    k = thin.compile(flow_kernel, seed=seed)
    nest = _random_nest(seed)
    rng = np.random.default_rng(100 + seed)
    for _ in range(8):
        env = {n: int(v) for n, v in zip("abc", rng.integers(0, 5, 3), strict=True)}
        try:
            want = _run(nest, dict(env), 0)
        except _ExitError as e:
            want = e.args[0]
        out = np.zeros(LANES, np.uint32)
        k.bind(grid=1, args={"out": out, **env}).emulate()
        assert (out == want).all(), (env, int(out[0]), want)


# --- Regression checks: wide constants, the sign of zero ---


@pytest.mark.parametrize("p", __import__("tests.dsl_edges").dsl_edges.PS)
def test_u64_plus_minus_every_edge_constant_scalar(no_store, p):
    """(p ± c) mod 2^64 bit for bit for c, the SALU chain."""
    import thin
    from tests import dsl_edges as E  # noqa: N812 - the cases, as E

    k = thin.compile(E.gen_wide_scalar)
    out = np.zeros(E.SLOTS, np.uint64)
    got = np.asarray(k.bind((1,), {"p": p, "out": out}).emulate().get("out", out)).view(np.uint64)
    assert got.tolist() == E.wide_want(p)


def test_u64_plus_minus_every_edge_constant_vector(no_store):
    """Wide VALU constants keep high halves when lowered into VGPRs."""
    import thin
    from tests import dsl_edges as E  # noqa: N812 - the cases, as E

    k = thin.compile(E.gen_wide_vector)
    ps = np.array(E.PS + [0] * (32 - len(E.PS)), np.uint64)
    out = np.zeros(32 * E.SLOTS, np.uint64)
    got = np.asarray(k.bind((1,), {"ps": ps, "out": out}).emulate().get("out", out)).view(np.uint64)
    for lane, p in enumerate(E.PS):
        assert got[lane * E.SLOTS : (lane + 1) * E.SLOTS].tolist() == E.wide_want(p), hex(p)


def test_the_borrow_chain_follows_the_isa_guide(no_store):
    """v.sub.co, v.subrev.co, v.subb.co under a partial EXEC: results and
    borrow masks (0 in inactive lanes) as the guide's pseudocode."""
    import thin
    from tests import dsl_edges as E  # noqa: N812 - the cases, as E

    k = thin.compile(E.gen_borrow)
    xs, ys = E.borrow_inputs()
    d, masks = np.zeros(96, np.uint32), np.zeros(3, np.uint32)
    got = k.bind((1,), {"xs": xs, "ys": ys, "d": d, "masks": masks}).emulate()
    want_d, want_m = E.borrow_want()
    rows = np.asarray(got.get("d", d)).reshape(3, 32)[:, : E.ACTIVE]
    assert rows.tolist() == want_d.reshape(3, 32)[:, : E.ACTIVE].tolist()
    assert [hex(m) for m in np.asarray(got.get("masks", masks))] == [hex(m) for m in want_m]


@pytest.mark.parametrize("c", [1 << 64, -(1 << 63) - 1, 1.5, 0.0])
def test_a_u64_constant_out_of_range_or_a_float_is_refused(c):
    from thin import dsl

    b = dsl.Builder()
    p = b.s.u64(name="p", at=0)
    with pytest.raises(
        dsl.ExprError, match=r"outside the u64 range|a float constant where a u64 is"
    ):
        b.s.u64(p + c)


def test_the_sign_of_zero(no_store):
    """V.f32(-0.0), S.f32(-0.0), f * -0.0, f + -0.0 (and reversed) for f in
    ±0.0 and others: IEEE's sign bits; -0.0 emitted as 0x80000000."""
    import thin
    from tests import dsl_edges as E  # noqa: N812 - the cases, as E

    k = thin.compile(E.gen_zero)
    assert "0x80000000" in k.asm or "2147483648" in k.asm
    src = np.zeros(32, np.float32)
    src[: len(E.ZERO_INPUTS)] = E.ZERO_INPUTS
    out = np.zeros(32 * 6, np.uint32)
    got = np.asarray(k.bind((1,), {"src": src, "out": out}).emulate().get("out", out)).view(
        np.uint32
    )
    want = E.zero_want()
    assert [hex(x) for x in got[: len(want)]] == [hex(x) for x in want]


def scan_kernel(op: str = "add"):
    """Each lane stores scan(x, op, 32) of the wave's 32 inputs."""
    from thin import KernelDef
    from thin.dsl import Args, O, ptr, scan

    def body(b, k):
        V = b.v  # noqa: N806 - the DSL's spelling
        x, y = k.load("x", "y")
        off = V.u32(k.tid * 4, name="off")
        a = V.f32(name="a")
        b.emit(O.global_load(a, off, x))
        b.wait(a, "a landed")
        r = V.f32(scan(a, op, 32), name="r")
        b.emit(O.global_store(off, r, y))

    return KernelDef(f"scan_{op}", Args(x=ptr, y=ptr), body, wgsize=32, max_grid=(1,))


@pytest.mark.parametrize(
    ("op", "fold"), [("add", np.add), ("max", np.maximum), ("min", np.minimum)]
)
def test_a_32_lane_scan_is_the_wave_s_prefix(no_store, op, fold):
    """scan over a 32-lane wave: each lane holds the fold of lanes 0.. itself
    (small integers, so every order of adds is exact)."""
    import thin

    rng = np.random.default_rng(7)
    x = rng.integers(-8, 9, 32).astype(np.float32)
    y = np.zeros(32, np.float32)
    k = thin.compile(scan_kernel, op=op)
    got = np.asarray(k.bind((1,), {"x": x, "y": y}).emulate().get("y", y))
    assert got.tolist() == fold.accumulate(x).tolist()


def across_kernel(op: str = "add", lanes: int = 32):
    """Each lane stores across(x, op, lanes) of its group of the wave's inputs."""
    from thin import KernelDef
    from thin.dsl import Args, O, across, ptr

    def body(b, k):
        V = b.v  # noqa: N806 - the DSL's spelling
        x, y = k.load("x", "y")
        off = V.u32(k.tid * 4, name="off")
        a = V.f32(name="a")
        b.emit(O.global_load(a, off, x))
        b.wait(a, "a landed")
        r = V.f32(across(a, op, lanes), name="r")
        b.emit(O.global_store(off, r, y))

    return KernelDef(f"across_{op}_{lanes}", Args(x=ptr, y=ptr), body, wgsize=32, max_grid=(1,))


@pytest.mark.parametrize("lanes", [2, 4, 8, 16, 32])
@pytest.mark.parametrize(("op", "fold"), [("add", np.sum), ("max", np.max), ("min", np.min)])
def test_across_folds_each_group_into_every_lane_of_it(no_store, op, fold, lanes):
    """across over groups of 2 .. 32 lanes (32 crosses the wave's two rows
    with v_permlanex16_b32); small integers, so every order of adds is exact."""
    import thin

    rng = np.random.default_rng(11)
    x = rng.integers(-8, 9, 32).astype(np.float32)
    y = np.zeros(32, np.float32)
    k = thin.compile(across_kernel, op=op, lanes=lanes)
    got = np.asarray(k.bind((1,), {"x": x, "y": y}).emulate().get("y", y))
    want = np.repeat(fold(x.reshape(-1, lanes), axis=1), lanes)
    assert got.tolist() == want.tolist()


def minmax_kernel():
    """Each lane stores min(a, b, c), max(a, b, c, d) and med3(a, b, c) of its inputs."""
    from thin import KernelDef
    from thin.dsl import Args, O, max, med3, min, ptr

    def body(b, k):
        V = b.v  # noqa: N806 - the DSL's spelling
        x, y = k.load("x", "y")
        xs = [V.f32(name=f"x{j}") for j in range(4)]
        for j, v in enumerate(xs):
            b.emit(O.global_load(v, V.u32(k.tid * 16 + 4 * j, name=f"o{j}"), x))
        b.wait(*xs, "inputs landed")
        a, c, d, e = xs
        for j, r in enumerate([min(a, c, d), max(a, c, d, e), med3(a, c, d)]):
            o = V.u32(k.tid * 12 + 4 * j, name=f"q{j}")
            b.emit(O.global_store(o, V.f32(r, name=f"r{j}"), y))

    return KernelDef("minmax", Args(x=ptr, y=ptr), body, wgsize=32, max_grid=(1,))


def test_min_and_max_of_more_than_two_operands(no_store):
    import thin

    rng = np.random.default_rng(3)
    x = rng.standard_normal((32, 4)).astype(np.float32)
    y = np.zeros((32, 3), np.float32)
    k = thin.compile(minmax_kernel)
    got = np.asarray(k.bind((1,), {"x": x, "y": y}).emulate().get("y", y)).reshape(32, 3)
    assert got[:, 0].tolist() == x[:, :3].min(axis=1).tolist()
    assert got[:, 1].tolist() == x.max(axis=1).tolist()
    assert got[:, 2].tolist() == np.median(x[:, :3], axis=1).astype(np.float32).tolist()
