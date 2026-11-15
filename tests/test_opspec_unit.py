"""Unit tests of opspec beyond its vectors."""

from __future__ import annotations

import itertools
import random
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from tests import opspec_standins as st

# --- The registry ---------------------------------------------------------------


def test_one_place_twice_is_accepted_two_places_refused():
    src = 'class R(Spec, name="unit.reg"):\n    pass\n'
    a, b = st.define(src)["R"], st.define(src)["R"]
    assert a is not b and b.name() == "unit.reg"
    with pytest.raises(TypeError, match="one name one spec"):
        st.define('class R2(Spec, name="unit.reg"):\n    pass\n')


def test_a_subclass_needs_its_own_name_and_implementations_are_per_class():
    from thin.opspec import implementations, implements

    base = st.define('class B(Spec, name="unit.base"):\n    pass\n')["B"]
    with pytest.raises(TypeError, match="a spec names itself"):
        st.define("class C(_B):\n    pass\n", {"_B": base})
    sub = st.define('class C(_B, name="unit.sub"):\n    pass\n', {"_B": base})["C"]

    def fn(op):
        return []

    implements(base)(fn)
    implements(base)(fn)
    assert [i.name for i in implementations(base)] == ["unit.base/fn"] * 2
    assert implementations(sub) == []


# --- Fields ---------------------------------------------------------------------------


def _int_spec(lo, hi):
    return st.spec_of({"name": "unit.int", "fields": {"n": f"Int(lo={lo!r}, hi={hi!r})"}})


def test_int_property():
    from thin.opspec import Field

    rng = random.Random(3)
    for _ in range(200):
        lo = rng.choice([None, rng.randrange(-50, 50)])
        hi = rng.choice([None, rng.randrange(-50, 50)])
        if lo is not None and hi is not None and lo > hi:
            with pytest.raises(ValueError, match="lo above hi"):
                _int_spec(lo, hi)
            continue
        spec = _int_spec(lo, hi)
        v = rng.randrange(-80, 80)
        given = rng.choice([v, np.int32(v), np.int64(v), Field(v, -1, 1), True, float(v)])
        inside = (lo is None or v >= lo) and (hi is None or v <= hi)
        if isinstance(given, (bool, float)):
            with pytest.raises(TypeError, match="an int"):
                spec(n=given)
        elif not inside:
            with pytest.raises(ValueError, match="outside"):
                spec(n=given)
        else:
            got = spec(n=given).n
            assert got == v
            assert type(got) is (Field if isinstance(given, Field) else int)


def test_choice_is_type_exact():
    from thin.opspec import Choice

    c = Choice(True, 1, "a")
    for v, ok in [(True, True), (1, True), ("a", True), (np.int64(1), False), (1.0, False)]:
        if ok:
            assert c.accept("s", "f", v) is v
        else:
            with pytest.raises(ValueError, match="one of"):
                c.accept("s", "f", v)
    with pytest.raises(ValueError, match="one of"):
        Choice(True).accept("s", "f", 1)


# --- Storage ----------------------------------------------------------------------------


def test_storage_and_poison_agree_with_formats():
    from thin import formats
    from thin.opspec import In

    rng = random.Random(5)
    for name, d in formats.DTYPES.items():
        per = 8 // d.width if d.width < 8 else 1
        for _ in range(5):
            shape = [rng.randrange(1, 4) for _ in range(rng.randrange(1, 3))]
            shape[-1] *= per
            t = In(name, *shape)
            numpy, storage = t.storage()
            assert numpy == d.numpy
            assert storage == (*shape[:-1], shape[-1] // per)
            p = t.poisoned()
            assert p.dtype.name == numpy and p.shape == storage
            assert set(p.tobytes()) <= {0xFF}
            try:
                import torch
            except ImportError:
                continue
            assert tuple(torch.zeros(storage, dtype=getattr(torch, d.torch)).shape) == storage


# --- candidates and lowering ------------------------------------------------------


def _plain_spec(name):
    return st.spec_of({"name": name, "fields": {"n": "Int(lo=1)"}, "signature": "{}"})


def test_candidates_product_order_on_random_tables():
    from thin.opspec import candidates, implements

    rng = random.Random(9)
    for trial in range(20):
        spec = _plain_spec(f"unit.cand{trial}")
        table = {
            f"t{i}": rng.sample(range(10), rng.randrange(1, 3)) for i in range(rng.randrange(0, 3))
        }
        implements(spec, tunables=table)(st.binding("f", lambda op, t: [], []))
        with pytest.raises(ValueError, match=r"tunables \['e'\] list no values"):
            implements(spec, tunables={**table, "e": []})
        got = [t for _, t, _ in candidates(spec(n=1))]
        want = [dict(zip(table, vs, strict=True)) for vs in itertools.product(*table.values())]
        assert got == want


def test_served_ranges_are_coverage_and_the_first_pair_lowers():
    from thin.opspec import candidates, implements, lowered

    spec = _plain_spec("unit.served")
    k = st.kernel("k", {"n": [1, 1000, 4]})
    for lo, hi in [(1, 10), (11, 100), (1, 1000)]:
        name = f"b{lo}_{hi}"
        fn = st.binding(name, lambda op, t: st.bound(k, {"n": op.n}), [])
        implements(spec, serves=lambda op, lo=lo, hi=hi, **_: {"n": (lo, hi)})(fn)
    rng = random.Random(1)
    for _ in range(50):
        n = rng.randrange(1, 1001)
        names = [i.name.split("/")[1] for i, _, _ in candidates(spec(n=n))]
        want = [f"b{lo}_{hi}" for lo, hi in [(1, 10), (11, 100), (1, 1000)] if lo <= n <= hi]
        assert names == want
        assert lowered(spec(n=n))[0].name.endswith(want[0])
    impl = lowered(spec(n=5), "b1_1000")[0]
    assert lowered(spec(n=5), impl.fn)[0] is impl


def _vadd_spec(name):
    def signature(op):
        from thin.opspec import In, Out

        return {"x": In("f32", op.n), "y": In("f32", op.n), "z": Out("f32", op.n)}

    return st.spec_of(
        {"name": name, "fields": {"n": "Int(lo=1, hi=(1 << 30) - 1)"}},
        extra={"signature": signature},
    )


def _vadd_op(spec, n):
    return spec(n=n, **{t: np.zeros(n, np.float32) for t in "xyz"})


@pytest.fixture
def no_store():
    import thin

    before = thin.use_store(None)
    yield
    thin.use_store(before)


def test_the_served_view_lowers_a_range_to_one_kernel(no_store):
    import thin
    from examples.vadd import build_vadd
    from thin.opspec import implements, lower

    spec = _vadd_spec("unit.vadd")
    top = (1 << 30) - 1

    @implements(spec, serves=lambda op, **_: {"n": (1, top)})
    def add(op):
        k = thin.compile(build_vadd, n=op.n.range)
        return k(x=op.x, y=op.y, z=op.z, n=op.n)

    kernels = {id(lower(_vadd_op(spec, n))[0].kernel) for n in (1, 777, 5000)}
    assert len(kernels) == 1
    (b,) = lower(_vadd_op(spec, 777))
    assert (b.kernel.args["n"]["lo"], b.kernel.args["n"]["hi"]) == (1, top)
    assert b.args["n"] == 777


def test_honoured_ranges_and_needs_with_real_kernels(no_store):
    import thin
    from examples.vadd import build_vadd
    from thin.opspec import implements, lower

    spec = _vadd_spec("unit.vadd_needs")

    @implements(spec, serves=lambda op, **_: {"n": (1, 1 << 20)})
    def narrow(op):
        return thin.compile(build_vadd, n=(1, 4096))(x=op.x, y=op.y, z=op.z, n=op.n)

    @implements(spec, needs={"co_resident": True})
    def claims(op):
        return thin.compile(build_vadd)(x=op.x, y=op.y, z=op.z, n=op.n)

    @implements(spec, needs={"overread": {"x": 16}})
    def reads(op):
        k = thin.compile(build_vadd)
        args = {"x": op.x, "y": op.y, "z": op.z, "n": op.n}
        return k.bind(1, args, overread={"x": 64})

    op = _vadd_op(spec, 8)
    with pytest.raises(
        ValueError, match=r"serves n 1 \.\. 1048576, but vadd_b256 takes n in 1 \.\. 4096"
    ):
        lower(op, narrow)
    with pytest.raises(ValueError, match="needs co_resident True, its kernels none co-resident"):
        lower(op, claims)
    with pytest.raises(ValueError, match="overreads x by 64 bytes, the implementation declares 16"):
        lower(op, reads)


# --- check -----------------------------------------------------------------------------


def _add_with(effects):
    from thin.opspec import implements

    add = st.add_spec()
    for name, effect in effects.items():
        fn = st.binding(
            name,
            lambda op, t, name=name, effect=effect: st.bound(
                st.kernel(name), {n: getattr(op, n) for n in "xyz"}, effect=effect
            ),
            [],
        )
        implements(add)(fn)
    return add


def _short(b):
    b.args["z"][:-1] = (b.args["x"] + b.args["y"])[:-1]


def test_check_run_count_and_failures_and_rerun():
    import thin

    add = _add_with({"short": _short})
    st.CALLS.clear()
    op = add(n=8)
    fails = thin.check(op, impl="short", seeds=(0, 1))
    assert len(st.CALLS) == 2 * 2 * len(thin.opspec.SCHEDS)
    assert len(fails) == len(st.CALLS)
    f = fails[5]
    st.CALLS.clear()
    again = eval(f.rerun(), {"thin": thin, "op": op})
    assert len(st.CALLS) == 1
    assert [(g.seed, g.dist, g.sched, g.why) for g in again] == [(f.seed, f.dist, f.sched, f.why)]
    with pytest.raises(ValueError, match="dists"):
        thin.check(op, dists={"nope": [1]})


def test_a_fault_is_a_failure():
    import thin
    from thin.runtime.errors import DeviceFault

    def fault(b):
        raise DeviceFault("boom")

    add = _add_with({"bad": fault})
    fails = thin.check(add(n=4), impl="bad", seeds=(0,), scheds=(None,), dists={"scale": [1.0]})
    assert [f.why for f in fails] == ["fault: boom"]


def test_a_group_check_makes_one_group_and_closes_it(monkeypatch):
    import thin
    from thin.opspec import group, implements

    made, closed = [], []
    real = group.emulated

    def counting(world):
        handles = real(world)
        made.append(world)
        original = handles[0].team.close
        monkeypatch.setattr(handles[0].team, "close", lambda: (closed.append(world), original())[1])
        return handles

    monkeypatch.setattr(group, "emulated", counting)
    monkeypatch.setattr(group, "emulate", lambda bounds, sched=None, staged=(): None)
    gsum = st.gsum_spec()
    for world in range(2, 9):

        @implements(gsum)
        def anything(op):
            return st.bound(st.kernel("k"), {})

        made.clear(), closed.clear()
        thin.check(
            gsum(world=world, N=2), impl="anything", seeds=(0, 1), scheds=(None, {"jitter": 1})
        )
        assert made == [world] and closed == [world]


# --- Groups' GPU kinds ------------------------------------------------------------------


def test_the_ipc_exchange_record_and_its_refusal():
    from thin.opspec import _gpu
    from thin.runtime.errors import LaunchError

    handle = bytes(range(64))
    rec = _gpu.record(handle, 300, "k:w")
    assert rec == handle + (300).to_bytes(8, "little") + b"k:w"
    sent = []

    def same(mine):
        sent.append(mine)
        return [mine, bytes(64) + mine[64:]]

    assert len(_gpu.exchange(2, 0, same, handle, 300, "k:w")) == 2 and sent == [rec]

    def differs(mine):
        return [mine, bytes(64) + (8).to_bytes(8, "little") + b"k:w"]

    with pytest.raises(
        LaunchError,
        match=r"rank 0: 300 bytes of k:w; every rank's: \[\(300, 'k:w'\), \(8, 'k:w'\)\]",
    ):
        _gpu.exchange(2, 0, differs, handle, 300, "k:w")
    with pytest.raises(LaunchError, match="allocate differently"):
        _gpu.exchange(2, 0, lambda mine: [mine], handle, 300, "k:w")


def test_a_local_group_refuses_more_ranks_than_gpus():
    torch = pytest.importorskip("torch")
    from thin.opspec import group
    from thin.runtime.errors import LaunchError

    n = torch.cuda.device_count()
    with pytest.raises(LaunchError, match=f"a local group of {n + 1}: {n} GPUs here"):
        group.local(n + 1)


def test_ipc_refuses_a_rank_outside_the_world():
    from thin.opspec import group

    with pytest.raises(ValueError, match="rank 2 of a group of 2"):
        group.ipc(2, 2, lambda b: [b])


# --- The runtime's joint run of a group ----------------------


class _Memory:
    """One memory two host devices share, as a group's emulated devices do."""

    def __init__(self):
        self.allocs: dict[int, bytearray] = {}
        self.next = 1 << 20
        self.launches: list = []


def _shared_device(memory, rank):
    from thin.runtime.device import Device

    class Shared(Device):
        guard = 0

        def alloc_bytes(self, n, fill, meta):
            base, memory.next = memory.next, memory.next + max(n, 1) + 256
            memory.allocs[base] = bytearray([fill]) * n
            return memory.allocs[base], base

        def write_bytes(self, a, o, d):
            a[o : o + len(d)] = d

        def read_bytes(self, a, o, n):
            return bytes(a[o : o + n])

        def copy_bytes(self, dst, do, src, so, n):
            dst[do : do + n] = src[so : so + n]

        def fill_bytes(self, a, o, n, byte):
            a[o : o + n] = bytes([byte]) * n

        def sync(self, what):
            pass

        def free(self):
            pass

        def load(self, image, code_object, asm, block, key):
            return "code"

        def run(self, code, key, grid, block, kernarg, what, cooperative):
            memory.launches.append((self.rank, kernarg))
            own_mapped = int.from_bytes(kernarg[16:24], "little")  # this rank's, appended
            memory.allocs[own_mapped][0] = 10 + self.rank

    return Shared(rank)


def test_the_joint_run_of_a_group_maps_every_rank():
    from thin import formats
    from thin.opspec import group
    from thin.runtime.bound import Bound, joint_run

    desc = bytearray(64)
    desc[8:12] = (48).to_bytes(4, "little")
    meta = {"amdhsa.kernels": [{".name": "k", ".symbol": "k.kd", ".max_flat_workgroup_size": 32}]}
    co = formats.write("k", b"\0" * 4, bytes(desc), meta, 78)
    args = {
        "m": {"name": "m", "offset": 0, "bytes": 24, "kind": "ptr", "owned": "mapped", "ranks": 2},
        "p": {"name": "p", "offset": 24, "bytes": 24, "kind": "ptr", "owned": None, "ranks": 2},
    }
    k = SimpleNamespace(
        name="k",
        address="unit/k",
        args=args,
        co=co,
        asm="",
        manifest=SimpleNamespace(kernarg_bytes=48, needs={}),
    )
    ranks = group.emulated(2)
    copies = [ranks[q].tensor("t", "u32", (2,)) for q in range(2)]
    copies[1][:] = 7
    memory = _Memory()
    devices = [_shared_device(memory, 0), _shared_device(memory, 1)]
    bounds = [Bound(k, (1, 1, 1), {"m": 8, "p": copies[r]}, False, {}, ranks[r]) for r in range(2)]
    joint_run(bounds, devices, ranks[0].team.held, lambda b: b.group.rank, ())
    for rank, kernarg in memory.launches:
        m = [int.from_bytes(kernarg[i : i + 8], "little") for i in range(0, 24, 8)]
        p = [int.from_bytes(kernarg[i : i + 8], "little") for i in range(24, 48, 8)]
        assert m[2] == m[rank] and p[2] == p[rank] and m[0] != m[1] and p[0] != p[1]
        assert bytes(memory.allocs[p[1]][:4]) == (7).to_bytes(4, "little")
    held = ranks[0].team.mapped(k, "m", 8)
    assert [b[0] for b in held] == [10, 11]


# --- Import cost -----------------------------------------------------------------


def test_opspec_loads_no_compiler_nor_numpy():
    code = "import sys, thin.opspec; print('\\n'.join(sys.modules))"
    out = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
    loaded = set(out.stdout.split())
    heavy = (
        "thin.dsl",
        "thin.codegen",
        "thin.ir",
        "thin.target",
        "thin.emu",
        "thin._native",
        "thin.formats",
    )
    assert not {m for m in loaded if m.startswith(heavy) or m in {"numpy", "torch"}}


# --- Groups on the running emulator -----------------------------------------------------


def _system_scope(k):
    """The Kernel with every global access at SCOPE_SYS.

    gfx1201 IPC requires scope >= SCOPE_DEV on group memory. This helper
    explicitly sets system scope
    to the rendered.s and encodes that, so a group kernel runs on gfx1201.
    """
    import dataclasses
    import re

    from thin import codegen

    asm = re.sub(
        r"^(\tglobal_\S+ [^/\n]*?)(\s*(//.*)?)$", r"\1 scope:SCOPE_SYS\2", k.asm, flags=re.M
    )
    return dataclasses.replace(k, asm=asm, co=codegen.code_object(asm), address=k.address + "+sys")


def gen_peek():
    """out[0] = word 0 of rank 1's copy of the shared input p."""
    from thin import KernelDef
    from thin.dsl import Args, O, peers, ptr

    def body(b, k):
        V = b.v  # noqa: N806 - the DSL's spelling
        p, out = k.load()
        v = V.u32(name="v")
        zero = V.u32(0, name="zero")
        b.emit(O.global_load(v, zero, p[1]))
        b.wait(v, "the peer's word")
        b.emit(O.global_store(zero, v, out))

    return KernelDef("peek", Args(p=peers(2), out=ptr), body, wgsize=32, max_grid=(1,))


def test_shared_inputs_are_staged_until_their_rank_starts(no_store):
    """rank 0 reads rank 1's copy before rank 1 started (slow) and sees
    0xFF; rank 1 reads it once started and sees its bytes."""
    import thin
    from thin.opspec import group
    from thin.runtime.bound import Bound

    k = _system_scope(thin.compile(gen_peek))
    ranks = group.emulated(2)
    copies = [ranks[q].tensor("p", "u32", (8,)) for q in range(2)]
    for q in range(2):
        copies[q][:] = 100 + q
    outs = [np.zeros(1, np.uint32) for _ in range(2)]
    bounds = [
        Bound(k, (1, 1, 1), {"p": copies[r], "out": outs[r]}, group=ranks[r]) for r in range(2)
    ]
    group.emulate(bounds, sched=group.slow(2), staged=copies)
    assert [int(o[0]) for o in outs] == [0xFFFF_FFFF, 101]
    ranks[0].close()


def test_a_group_kernel_from_the_dsl_alone_is_refused_its_ipc_access(no_store):
    """on gfx1201: without a scope the maker can set, a peer's copy
    is read without the IPC policy, a DeviceFault."""
    import thin
    from thin.opspec import group
    from thin.runtime.bound import Bound
    from thin.runtime.errors import DeviceFault

    k = thin.compile(gen_peek)
    ranks = group.emulated(2)
    copies = [ranks[q].tensor("p", "u32", (8,)) for q in range(2)]
    out = [np.zeros(1, np.uint32) for _ in range(2)]
    bounds = [
        Bound(k, (1, 1, 1), {"p": copies[r], "out": out[r]}, group=ranks[r]) for r in range(2)
    ]
    with pytest.raises(DeviceFault, match="without scope ≥ SCOPE_DEV or a non-temporal th"):
        group.emulate(bounds)
    ranks[0].close()


def gen_swap(mutant=False):
    """Each rank stores (its x, flag 1) into every rank's mapped slot of its
    own, polls its peer's slot until the flag is 1, and writes y = x + the
    peer's x. The mutant reads the peer's slot once, without the poll."""
    from thin import KernelDef
    from thin.dsl import Args, O, mapped, popcount, ptr, u32

    def body(b, k):
        V, S = b.v, b.s  # noqa: N806 - the DSL's spelling
        m, x, y, rank = k.load()
        zero = V.u32(0, name="zero")
        w = V.u32(n=2, name="w")
        b.emit(O.global_load(w[0], zero, x))
        b.wait(w[0], "x landed")
        w[1] = 1
        mine = V.u32(rank * 8, name="mine")
        for q in range(2):
            b.emit(O.global_store(mine, w, m[q], comment=f"-> rank {q}"))
        theirs = V.u32((1 - rank) * 8, name="theirs")
        got = V.u32(n=2, name="got")
        if mutant:
            b.emit(O.global_load(got, theirs, m.own))
            b.wait(got, "the peer's word")
        else:
            for lp in b.loop(name="poll"):
                b.emit(O.global_load(got, theirs, m.own))
                b.wait(got, "the round")
                stale = S.mask(got[1] != 1, name="stale")
                lp.break_when(popcount(stale) == 0, comment="the peer's word in")
        b.emit(O.global_store(zero, V.u32(w[0] + got[0], name="sum"), y))

    args = Args(m=mapped(2), x=ptr, y=ptr, rank=u32.within(0, 1))
    return KernelDef("swap", args, body, wgsize=32, max_grid=(1,))


def _swap(mutant, sched):
    import thin
    from thin.opspec import group
    from thin.runtime.bound import Bound

    k = _system_scope(thin.compile(gen_swap, mutant=mutant))
    ranks = group.emulated(2)
    ys = [np.zeros(1, np.uint32) for _ in range(2)]
    bounds = [
        Bound(
            k,
            (1, 1, 1),
            {"m": 16, "x": np.array([5 + 2 * r], np.uint32), "y": ys[r], "rank": r},
            group=ranks[r],
        )
        for r in range(2)
    ]
    group.emulate(bounds, sched=sched)
    ranks[0].close()
    return [int(v[0]) for v in ys]


def test_a_mutant_is_caught_by_group_emulate_under_slow(no_store):
    """without its poll, rank 0 reads the slow rank's slot before rank 1
    stored it; with the poll, both ranks sum both inputs under every schedule."""
    from thin.opspec import group

    assert _swap(False, None) == [12, 12]
    assert _swap(False, group.slow(2)) == [12, 12]
    assert _swap(True, group.slow(2)) != [12, 12]


def test_check_of_a_real_op_on_the_emulated_device(no_store):
    """check(op) of vadd on the emulator passes; a binding one element
    short fails every run with the element it left."""
    import thin
    from examples.vadd import build_vadd
    from thin.opspec import implements

    add = st.add_spec()

    @implements(add)
    def vadd(op):
        return thin.compile(build_vadd)(x=op.x, y=op.y, z=op.z, n=op.n)

    @implements(add)
    def vadd_short(op):
        return thin.compile(build_vadd)(x=op.x, y=op.y, z=op.z, n=op.n - 1)

    op = add(n=300)
    assert thin.check(op, impl="vadd", seeds=(0,)) == []
    fails = thin.check(op, impl="vadd_short", seeds=(0,))
    runs = len(add.dists["scale"]) * len(thin.opspec.SCHEDS)
    assert len(fails) == runs and all(f.why.startswith("z[299] = ") for f in fails), fails[:1]


def test_check_takes_emulated_or_gpu_only():
    from thin.opspec import check

    spec = _plain_spec("unit.devices")
    with pytest.raises(ValueError, match=r"check\(device='cuda'\): 'emulated' or 'gpu'"):
        check(spec(n=1), device="cuda")


def test_a_signed_arguments_default_range_is_signed():
    from types import SimpleNamespace

    from thin.opspec._impl import _honoured

    rec = SimpleNamespace(lo=None, hi=None, bytes=4, kind="i32")
    bound = SimpleNamespace(kernel=SimpleNamespace(name="k", args={"n": rec}))
    impl = SimpleNamespace(name="unit/f")
    _honoured(impl, {}, {"n": (-5, 5)}, [bound])  # within i32
    with pytest.raises(ValueError, match=r"takes n in -2147483648 \.\. 2147483647"):
        _honoured(impl, {}, {"n": (0, 1 << 31)}, [bound])


def test_a_tunable_may_not_take_checks_keywords():
    from thin.opspec import implements

    spec = _plain_spec("unit.keywords")
    with pytest.raises(ValueError, match=r"tunables \['seeds'\] are check\(\)'s own keywords"):
        implements(spec, tunables={"seeds": (1,)})


def test_a_groups_ranks_lower_to_one_implementation():
    from thin.opspec import check, implements

    spec = st.spec_of(
        {
            "name": "unit.split",
            "fields": {"world": "Int(lo=2, hi=8)", "n": "Int(lo=1)"},
            "signature": "{}",
            "group": "world",
        },
        extra={"inputs": lambda op, seed: [{}, {}], "check": lambda op, i, o: None},
    )
    for which in (0, 1):
        implements(spec, covers=lambda op, which=which, **_: op.rank == which)(
            st.binding(f"r{which}", lambda op, t: [], [])
        )
    with pytest.raises(ValueError, match=r"the ranks lower to .*: one implementation"):
        check(spec(world=2, n=1), seeds=(0,), scheds=(None,))
