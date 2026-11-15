"""Undefined claims on a CFG edge."""

import numpy as np
import pytest

from tests import notation
from thin import codegen

HEAD = """\
k = b.entry(Args(x=ptr), wg=1)
(x,) = k.load()
off = V.u32(k.tid * 4, name="off")
s = S.u32(k.wgid, name="s")
y = V.u32(k.tid + 1, name="y")
b.emit(O.op("s_cmp_eq_u32", s, 0))
"""


def _lines(src):
    return [ln.split("//")[0].rstrip() for ln in codegen.render(notation.run_snippet(src))]


def test_a_claim_before_a_label_is_on_the_fall_edge_with_no_nop():
    got = _lines(
        HEAD
        + 'b.undefined(y, why="w")\n'
        + 'b.put_label(".La")\n'
        + "b.emit(O.global_store(off, off, x))\n"
    )
    at = got.index(".La:")
    assert got[at - 1] == "; undefined on fall: v2 (w)"
    assert not any(ln.strip().startswith("s_nop") for ln in got)


def test_a_taken_claim_stands_just_before_its_branch():
    got = _lines(
        HEAD
        + 'b.undefined(y, why="w", edge="taken")\n'
        + 'b.emit(O.op("s_cbranch_scc0", ".Lt"))\n'
        + 'b.put_label(".Lt")\n'
    )
    at = got.index("\ts_cbranch_scc0 .Lt")
    assert got[at - 1] == "; undefined on taken: v2 (w)"


def test_a_claim_right_after_a_label_is_a_plain_claim():
    got = _lines(
        HEAD + 'b.put_label(".La")\nb.undefined(y, why="w")\nb.emit(O.global_store(off, off, x))\n'
    )
    at = got.index(".La:")
    assert got[at + 1] == "; undefined: v2 (w)"


def test_a_taken_claim_not_before_a_branch_is_refused():
    src = HEAD + 'b.undefined(y, why="w", edge="taken")\nb.emit(O.global_store(off, y, x))\n'
    with pytest.raises(Exception, match=r"edge='taken'\): the next instruction is not a branch"):
        codegen.render(notation.run_snippet(src))
    with pytest.raises(Exception, match=r"edge is None or 'taken'"):
        notation.run_snippet(HEAD + 'b.undefined(y, why="w", edge="fall")\n')


def test_the_value_ends_only_on_the_claimed_edge():
    """The taken edge's target reads y, the fall path writes it first: with the
    claim y is dead past its def, and t (after a label) takes its register."""

    def src(claim):
        return (
            "k = b.entry(Args(x=ptr), wg=1)\n(x,) = k.load()\n"
            'off = V.u32(k.tid * 4, name="off")\ns = S.u32(k.wgid, name="s")\n'
            'y = V.u32(k.tid + 1, name="y")\nb.put_label(".La")\n'
            't = V.u32(k.tid * 3, name="t")\nb.emit(O.global_store(off, t, x))\n'
            'b.emit(O.op("s_cmp_eq_u32", s, 0))\n'
            + ('b.undefined(y, why="w", edge="taken")\n' if claim else "")
            + 'b.emit(O.op("s_cbranch_scc0", ".Lt"))\ny[...] = 5\n'
            'b.put_label(".Lt")\nb.emit(O.global_store(off, y, x))\n'
        )

    assert "\tv_add_nc_u32 v2, 1, v0" in _lines(src(True))
    assert "\tv_mul_lo_u32 v2, v0, 3" in _lines(src(True))
    assert "\tv_mul_lo_u32 v3, v0, 3" in _lines(src(False))


def _run(body):
    """Runs one wave of `body` (s[0:1] the kernargs: one output pointer); the
    stored dwords."""
    from thin.emu import Cluster, Memory

    text = (
        "\ts_load_b64 s[4:5], s[0:1], 0x0\n"
        "\ts_wait_kmcnt 0\n"
        "\tv_lshlrev_b32 v1, 2, v0\n"
        "\tv_mov_b32 v2, 7\n" + body + "\tglobal_store_b32 v1, v2, s[4:5]\n\ts_endpgm\n"
    )
    m = Memory()
    y = m.alloc("y", 128)
    k = m.alloc("k", 8, tagged=False)
    k.write(0, np.array([y.base], dtype=np.uint64).tobytes())
    Cluster(text, {"wgsize": 32, "vgpr": 4}, m, [k]).run()
    return set(np.frombuffer(y.data, np.uint32).tolist())


POISON = 0x7FC17FC1


@pytest.mark.parametrize(("cmp", "want"), [("0", {POISON}), ("1", {7})])
def test_the_emulator_poisons_on_the_taken_edge_only(cmp, want):
    # s_cmp_eq_u32 0, cmp: scc 1 when cmp is 0, then cbranch_scc1 is taken.
    body = (
        f"\ts_cmp_eq_u32 0, {cmp}\n"
        "; undefined on taken: v2 (t)\n"
        "\ts_cbranch_scc1 .Lt\n"
        "\ts_nop 0\n"
        ".Lt:\n"
    )
    assert _run(body) == want


@pytest.mark.parametrize(("cmp", "want"), [("0", {7}), ("1", {POISON})])
def test_the_emulator_poisons_on_the_fall_edge_only(cmp, want):
    body = (
        f"\ts_cmp_eq_u32 0, {cmp}\n"
        "\ts_cbranch_scc1 .Lt\n"
        "\ts_nop 0\n"
        "; undefined on fall: v2 (f)\n"
        ".Lt:\n"
    )
    assert _run(body) == want


def test_the_emulator_keys_edge_claims_by_their_edge():
    from thin.emu import undefined_claims

    text = (
        "\ts_nop 0\n; undefined on taken: v2 (t)\n\ts_cbranch_scc1 .Lt\n"
        "\ts_nop 0\n; undefined on fall: v3 (f)\n.Lt:\n\ts_endpgm\n"
    )
    assert undefined_claims(text) == {"edge:1->3": [("v", 2, 1)], "edge:2->3": [("v", 3, 1)]}
