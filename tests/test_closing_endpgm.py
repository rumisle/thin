"""Closing s_endpgm is omitted after a body that already ends control flow."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def no_store():
    import thin

    before = thin.use_store(None)
    yield
    thin.use_store(before)


def gen_ending(ending: str = "endpgm"):
    """Stores 7 at x[tid], then ends as `ending` says."""
    from thin import KernelDef
    from thin.dsl import Args, ptr

    def body(b, k):
        from thin.dsl import O

        V = b.v  # noqa: N806
        (x,) = k.load()
        off = V.u32(k.tid * 4, name="off")
        seven = V.u32(7, name="seven")
        b.emit(O.global_store(off, seven, x))
        if ending == "endpgm":
            b.emit(O.op("s_endpgm"))
        elif ending == "branch":
            b.emit(O.op("s_branch", ".Ldone"))
            b.emit(O.op("s_nop", 0))  # unreachable
            b.put_label(".Ldone")
            b.emit(O.op("s_endpgm"))
            b.emit(O.op("s_branch", ".Ldone"))
        elif ending == "cbranch":  # ends in a conditional branch, never taken
            s = b.s.u32(k.wgid, name="s")
            b.put_label(".Lagain")
            b.emit(O.op("s_cmp_eq_u32", s, 99))
            b.emit(O.op("s_cbranch_scc1", ".Lagain"))
        elif ending == "label":
            s = b.s.u32(k.wgid, name="s")
            b.emit(O.op("s_cmp_eq_u32", s, 0))
            b.emit(O.op("s_cbranch_scc0", ".Lend"))
            b.emit(O.op("s_nop", 0))
            b.put_label(".Lend", from_branches=True)
        else:  # endpgm, a comment, then a label
            s = b.s.u32(k.wgid, name="s")
            b.emit(O.op("s_cmp_eq_u32", s, 0))
            b.emit(O.op("s_cbranch_scc0", ".Lend"))
            b.emit(O.op("s_endpgm"))
            b.comment("the end")
            b.put_label(".Lend", from_branches=True)

    return KernelDef(f"ends_{ending}", Args(x=ptr), body, wgsize=32, max_grid=(4,))


OWN = {"endpgm": 1, "branch": 1, "cbranch": 0, "label": 0, "comment": 1}
"""The s_endpgm lines each body emits itself."""


@pytest.mark.parametrize(
    ("ending", "closing"),
    [("endpgm", 0), ("branch", 0), ("cbranch", 1), ("label", 1), ("comment", 1)],
)
def test_the_closing_endpgm_only_where_reachable(ending, closing):
    import thin
    from thin import codegen

    k = thin.compile(gen_ending, ending=ending)
    ops = [line.split("//")[0].split() for line in k.asm.splitlines()]
    assert sum(1 for o in ops if o[:1] == ["s_endpgm"]) == OWN[ending] + closing
    assert codegen.code_object(k.asm)  # it assembles
    # Every path ends at an s_endpgm: the emulation of each workgroup (each
    # takes another branch) finishes and stores.
    out = np.zeros(32, np.uint32)
    for g in range(4):
        got = np.asarray(k.bind((g + 1,), {"x": out.copy()}).emulate().get("x", out))
        assert got.tolist() == [7] * 32, (ending, g)
