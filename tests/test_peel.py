"""Peeled loops through the normal Python path: compiled,
rendered and emulated. The peel keeps the copies up to the first repeat, the
steady copy under the loop's label, and the back branch and end label."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def no_store():
    import thin

    before = thin.use_store(None)
    yield
    thin.use_store(before)


def gen_peeled(peel: int = 1, trips: int = 4, wait_first: bool = False):
    """acc = sum of i over `trips` iterations; with wait_first, the first
    iteration also waits for a load issued before the loop (a body that
    differs once, so the peel keeps one iteration out)."""
    from thin import KernelDef
    from thin.dsl import Args, ptr

    def body(b, k):
        from thin.dsl import O

        V, S = b.v, b.s  # noqa: N806
        x, y = k.load()
        off = V.u32(k.tid * 4, name="off")
        g = V.u32(name="g")
        b.emit(O.global_load(g, off, x))
        n = S.u32(trips, name="n")
        acc = V.u32(0, name="acc")
        for lp in b.loop(n, peel=peel, name="p"):
            if wait_first:
                b.wait(g)
                acc[...] = acc + g
            acc[...] = acc + lp.i
        if not wait_first:
            b.wait(g)
            acc[...] = acc + g
        b.emit(O.global_store(off, acc, y))

    return KernelDef("peeled", Args(x=ptr, y=ptr), body, wgsize=32, max_grid=(1,))


def _ops(asm: str) -> list:
    out = []
    for line in asm.split("\t.section")[0].splitlines():
        code = line.split("//")[0].strip()
        if code and not code.startswith((";", ".amdgcn", ".text", ".protected", ".globl",
                                         ".p2align", ".type")):  # fmt: skip
            out.append(code)
    return out


@pytest.mark.parametrize(
    ("peel", "trips", "wait_first"), [(1, 4, False), (2, 5, False), (2, 3, True)]
)
def test_a_peeled_loop_compiles_renders_and_runs(peel, trips, wait_first):
    import thin

    k = thin.compile(gen_peeled, peel=peel, trips=trips, wait_first=wait_first)
    ops = _ops(k.asm)
    assert ".Lp_1:" in ops and ".Lp_end_2:" in ops, "\n".join(ops)
    top, end = ops.index(".Lp_1:"), ops.index(".Lp_end_2:")
    assert ops[end - 1] == "s_branch .Lp_1"  # the steady copy loops back
    assert top < end
    x = np.arange(32, dtype=np.uint32) * 10
    y = np.zeros(32, np.uint32)
    got = np.asarray(k.bind((1,), {"x": x, "y": y}).emulate().get("y", y))
    per = trips * x if wait_first else x
    assert got.tolist() == (per + sum(range(trips))).tolist()


def test_the_steady_copy_follows_the_peeled_iterations():
    """wait_first: the first iteration waits for g; later ones find it retired,
    so the steady copy has no wait and the peeled one does."""
    import thin

    ops = _ops(thin.compile(gen_peeled, peel=2, trips=3, wait_first=True).asm)
    top = ops.index(".Lp_1:")
    assert any(o.startswith("s_wait_loadcnt") for o in ops[:top])
    assert not any(o.startswith("s_wait_loadcnt") for o in ops[top:])
