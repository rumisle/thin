"""A value cannot reuse a load destination while its write is still in flight.

Random kernels run clean with the load hold enabled; the negative control
turns it off and must expose premature reuse."""

from __future__ import annotations

import random

import numpy as np
import pytest

SEEDS = range(40)


@pytest.fixture(autouse=True)
def no_store():
    import thin

    before = thin.use_store(None)
    yield
    thin.use_store(before)


def gen_fuzz(seed: int = 0, control: bool = False):
    """A random kernel of loads, waits and uses (`control` only names it apart)."""
    from thin import KernelDef
    from thin.dsl import Args, ptr, u32

    def body(b, k):
        from thin.dsl import O

        rng = random.Random(seed)
        V, S = b.v, b.s  # noqa: N806
        x, y = k.load()  # the arguments: scalar loads on kmcnt
        lds = b.lds()
        t = lds.region("T", (32,), u32)
        off = V.u32(k.tid * 16, name="off")
        lane = V.u32(k.tid, name="lane")
        b.emit(O.ds_write(t.at(lane), lane))
        acc = V.u32(0, name="acc")
        live = []  # loaded values not yet waited for

        def load():
            kind = rng.choice(["g1", "g2", "g4", "ds"])
            if kind == "ds":
                v = V.u32(name=f"d{len(live)}")
                b.emit(O.ds_read(v, t.at(lane)))
            else:
                v = V.u32(n=int(kind[1]), name=f"g{len(live)}")
                b.emit(O.global_load(v, off, x))
            live.append(v)

        def use():
            if not live:
                return
            v = live.pop(rng.randrange(len(live)))
            if rng.random() < 0.5:
                b.wait(v)
            else:  # everything in flight
                b.wait(v, *live)
                live.clear()
            n = v.n if hasattr(v, "n") else 1
            for j in rng.sample(range(n), rng.randint(1, n)):
                acc[...] = acc + (v[j] if n > 1 else v)

        for _ in range(rng.randint(4, 12)):
            r = rng.random()
            if r < 0.45:
                load()
            elif r < 0.75:
                use()
            elif r < 0.85 and live:  # a load never read: dead at its issue
                live.pop(rng.randrange(len(live)))
                tmp = V.u32(k.tid + 7, name=f"t{seed}")
                acc[...] = acc + tmp
            else:
                trips = S.u32(2, name="trips")
                for _lp in b.loop(trips, conservative=True):
                    load()
                    use()
        if live:
            b.wait(*live)
        o4 = V.u32(k.tid * 4, name="o4")
        b.emit(O.global_store(o4, acc, y))

    return KernelDef(f"fuzz{seed}{'c' if control else ''}", Args(x=ptr, y=ptr), body,
                     wgsize=32, max_grid=(1,))  # fmt: skip


def _run(k):
    x = np.arange(32 * 4, dtype=np.uint32)
    y = np.zeros(32, np.uint32)
    return np.asarray(k.bind((1,), {"x": x, "y": y}).emulate().get("y", y))


def _compile(seed, control=False):
    import thin
    from thin import codegen

    with codegen.using(keep_dead=True):
        return thin.compile(gen_fuzz, seed=seed, control=control)


def test_random_kernels_never_reuse_a_loading_register():
    ran = 0
    for seed in SEEDS:
        try:
            k = _compile(seed)
        except Exception as e:  # a generated shape the compiler refuses
            assert "still being loaded" not in str(e)
            continue
        _run(k)  # raises on a pending register touched (RegisterInFlight)
        ran += 1
    assert ran >= len(SEEDS) // 2, ran


def test_without_the_load_hold_the_emulator_catches_the_reuse():
    """The negative control: the allocator without load holds lets a value take a
    register a load is still writing, and the run raises RegisterInFlight."""
    from thin import _native

    caught = 0
    old = _native.hold_loads(False)
    try:
        for seed in SEEDS:
            try:
                k = _compile(seed, control=True)
                _run(k)
            except Exception as e:
                caught += "still being loaded" in str(e)
    finally:
        _native.hold_loads(old)
    assert caught > 0
