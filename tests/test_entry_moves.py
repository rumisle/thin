"""Keep workgroup-id entry moves in built streams and final assembly."""

from __future__ import annotations

import numpy as np
import pytest

from tests import notation


@pytest.fixture(autouse=True)
def no_store():
    import thin

    before = thin.use_store(None)
    yield
    thin.use_store(before)


def _moves(src: str):
    """(the rendered ttmp moves, the built ones, k.entry_moves' names)."""
    from thin import codegen

    b = notation.run_snippet(src)
    built = [line for line in notation.built(b) if "ttmp" in line]  # before the pipeline
    rendered = [line.split("//")[0].strip() for line in codegen.render(b) if "ttmp" in line]
    return rendered, built


ENTRY = 'k = b.entry(Args(x=ptr), wg=2)\n(x,) = k.load()\noff = V.u32(k.tid * 4, name="off")\n'


def test_a_body_reading_neither_id_renders_no_move():
    rendered, built = _moves(ENTRY + "b.emit(O.global_store(off, off, x))\n")
    assert rendered == [] and len(built) == 2


def test_reading_only_wgid_y_keeps_only_its_move():
    rendered, _ = _moves(
        ENTRY + 'y = V.u32(k.wgid_y, name="y")\nb.emit(O.global_store(off, y, x))\n'
    )
    assert len(rendered) == 1 and rendered[0].endswith("ttmp7")


def test_a_read_inside_an_arm_or_a_loop_keeps_the_move():
    arm = ENTRY + 'w = V.u32(0, name="w")\nwith b.lanes(k.tid < 3):\n    w += k.wgid\n'
    arm += "b.emit(O.global_store(off, w, x))\n"
    rendered, _ = _moves(arm)
    assert len(rendered) == 1 and rendered[0].endswith("ttmp9")
    loop = ENTRY + 'n = S.u32(name="n", at=8)\nacc = V.u32(0, name="acc")\n'
    loop += "for lp in b.loop(n):\n    acc += k.wgid\nb.emit(O.global_store(off, acc, x))\n"
    rendered, _ = _moves(loop)
    assert len(rendered) == 1 and rendered[0].endswith("ttmp9")


def gen_peeled_id(read: bool = True):
    """wg = 2, k.wgid read only inside a peeled loop's body (copied by the peel)."""
    from thin import KernelDef
    from thin.dsl import Args, ptr

    def body(b, k):
        from thin.dsl import O

        V, S = b.v, b.s  # noqa: N806
        (x,) = k.load()
        n = S.u32(4, name="n")
        acc = V.u32(0, name="acc")
        for lp in b.loop(n, peel=1, name="p"):
            acc[...] = acc + (k.wgid if read else lp.i)
        off = V.u32(k.tid * 4, name="off")
        b.emit(O.global_store(off, acc, x))

    return KernelDef("peeled_id", Args(x=ptr), body, wgsize=32, wg=2, max_grid=(4, 4))


def test_an_id_read_only_in_a_peeled_body_keeps_its_move():
    """drop_entry runs before the deferred transforms, so an id read only in a
    peeled loop's body keeps its move; compiled, rendered and emulated."""
    import thin

    def ttmp(asm):
        return [line.split("//")[0].strip() for line in asm.splitlines() if "ttmp" in line]

    k = thin.compile(gen_peeled_id, read=True)
    assert ttmp(k.asm) == ["s_mov_b32 s2, ttmp9"] or [t.endswith("ttmp9") for t in ttmp(k.asm)] == [
        True
    ]
    out = np.zeros(32, np.uint32)
    got = np.asarray(k.bind((3, 1), {"x": out}).emulate().get("x", out))
    assert got.tolist() == [4 * 2] * 32  # the last workgroup (x = 2) stores last
    assert ttmp(thin.compile(gen_peeled_id, read=False).asm) == []


def test_entry_moves_lists_the_copied_ids_even_when_dropped():
    from thin import codegen

    src = ENTRY + "b.emit(O.global_store(off, off, x))\nout = (k.entry_moves, k.wgid, k.wgid_y)\n"
    scope_b = notation.run_snippet(src.replace("out = ", "b._moves_seen = "))
    moves, x, y = scope_b._moves_seen
    assert [repr(v) for v in moves] == [repr(x), repr(y)]
    codegen.render(scope_b)


def gen_ids(both: bool = True):
    """wg = 2: both ids stored, or only y (the x move dropped)."""
    from thin import KernelDef
    from thin.dsl import Args, ptr

    def body(b, k):
        from thin.dsl import O

        V = b.v  # noqa: N806
        (out,) = k.load()
        wg = V.u32(k.wgid_y * 4 + k.wgid if both else k.wgid_y, name="wg")
        off = V.u32(wg * 4, name="off")
        with b.lanes(k.tid == 0):
            b.emit(O.global_store(off, wg, out))

    return KernelDef("ids", Args(out=ptr), body, wgsize=32, wg=2, max_grid=(4, 4))


def test_the_ids_read_are_unchanged_on_the_emulator():
    """Per workgroup, the emulated ids are the grid's, with a move dropped or not."""
    import thin

    for both in (True, False):
        k = thin.compile(gen_ids, both=both)
        out = np.zeros(16, np.uint32)
        got = np.asarray(k.bind((4, 3), {"out": out}).emulate().get("out", out))
        if both:
            assert got[:12].tolist() == list(range(12))
        else:
            assert got[:3].tolist() == [0, 1, 2]
