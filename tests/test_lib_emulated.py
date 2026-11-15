"""Execute supported library protocols with the gfx1201 CPU model."""

from __future__ import annotations

import numpy as np
import pytest

WORDS = 4


@pytest.fixture
def no_store():
    import thin

    before = thin.use_store(None)
    yield
    thin.use_store(before)


def _gens(per):
    from thin.lib import xwg

    return xwg.Gens(per=per, words=WORDS, wgsize=64)


def gen_gens(per="member", run=0):
    """Each wave stores the g, g1 and tag it read: out[wg][wave] = (g, g1, tag);
    `run` only separates the tests' held memory (a kernel address each)."""
    from thin import KernelDef
    from thin.dsl import Args, O, ptr

    decl = _gens(per)

    def body(b, k):
        (out,) = k.load("out")
        g = decl.begin(b, k, at=k.wgid if per == "member" else 0)
        t = g.tag()
        if per == "party":
            # The completer: wave 0 of workgroup 0, after every wave of it read.
            b.barrier("every wave read the word")
            with b.when(k.wave == 0):
                g.advance()
        o = b.v.u32((k.wgid * 2 + k.wave) * 12, name="o")
        for j, x in enumerate([g.g, g.g1, t]):
            b.emit(O.global_store(o, b.v.u32(x, name=f"x{j}"), out, offset=4 * j))

    args = Args(out=ptr, **decl.args())
    return KernelDef(f"gens_{per}", args, body, wgsize=64, max_grid=(WORDS,))


def _launch(k, per, grid, sched=None):
    out = np.zeros(WORDS * 2 * 3, np.uint32)
    bound = k.bind(grid=grid, args={"out": out, **_gens(per).sizes()})
    bound.emulate(sched)
    word = np.frombuffer(bound.owned["xwg_gen"].tobytes(), np.uint32)
    return out.reshape(WORDS, 2, 3)[:grid], word


def test_member_generations_cross_the_wrap_with_their_tags(no_store):
    """Three launches from 2^32 - 3; each wave's g1 and tag follow its g.

    A member is one wave: a wgsize-64 workgroup is two 32-lane waves, each
    with a word of its own, so each word advances by one a launch.
    """
    import thin
    from thin.lib.xwg import GEN_START, full_tag

    k = thin.compile(gen_gens, per="member")
    word = GEN_START
    for _ in range(3):
        got, words = _launch(k, "member", WORDS)
        for wg in range(WORDS):
            for g, g1, t in got[wg].tolist():
                assert g == word, (wg, g, word)
                assert g1 == (g + 1) % 2**32 and t == full_tag(g1)
        word = (word + 1) % 2**32
        assert words.tolist() == [word] * (2 * WORDS)
    assert word == 0  # the third launch crossed 2^32


def test_the_waves_of_a_workgroup_keep_their_own_generations(no_store):
    """With a quantum of 1 both waves read before either stores; each still
    reads and advances its own word."""
    import thin
    from thin.lib.xwg import GEN_START

    k = thin.compile(gen_gens, per="member", run=1)
    got, words = _launch(k, "member", WORDS, sched={"quantum": 1})
    assert (got[:, :, 0] == GEN_START).all()
    assert words.tolist() == [GEN_START + 1] * (2 * WORDS)


def test_a_member_generation_needs_the_same_grid(no_store):
    import thin
    from thin.runtime.errors import LaunchError

    k = thin.compile(gen_gens, per="member", run=2)
    why = "per-member generations: every workgroup of every launch advances its own word"
    assert k.manifest.needs["same_grid"] == {"xwg_gen": why}
    _launch(k, "member", WORDS)
    with pytest.raises(LaunchError, match=r"grid \(2, 1, 1\) on its held memory xwg_gen"):
        _launch(k, "member", 2)


def test_a_partys_completer_advances_its_word_once(no_store):
    """per="party": every wave reads the party's word; after a barrier wave 0
    advances it, so the word moves by one a launch, and the grid may change."""
    import thin
    from thin.lib.xwg import GEN_START, full_tag

    k = thin.compile(gen_gens, per="party")
    assert "same_grid" not in k.manifest.needs
    for launch, grid in enumerate([1, 1, 1]):
        got, words = _launch(k, "party", grid)
        g = (GEN_START + launch) % 2**32
        assert got[0].tolist() == [[g, (g + 1) % 2**32, full_tag((g + 1) % 2**32)]] * 2
        assert words[0] == (g + 1) % 2**32


def gen_scaled(laps=False):
    """out[tid] = 3 * tid, with laps declared off when `laps`."""
    from thin import KernelDef
    from thin.dsl import Args, O, ptr
    from thin.lib import Laps

    decl = Laps(["a"], 64, keep="sum", on=False)

    def body(b, k):
        (out,) = k.load("out")
        lp = decl.begin(b, k) if laps else None
        if lp is not None:
            lp.lap("a")
        v = b.v.u32(k.tid * 3, name="v")
        if lp is not None:
            lp.record()
        b.emit(O.global_store(b.v.u32(k.tid * 4, name="o"), v, out))
        if lp is not None:
            lp.end()

    args = Args(out=ptr, **(decl.args() if laps else {}))
    return KernelDef("scaled", args, body, wgsize=64, max_grid=(1,))


def test_laps_off_add_nothing(no_store):
    """No argument, no clock read, the same code and the same result."""
    import thin

    plain, off = thin.compile(gen_scaled), thin.compile(gen_scaled, laps=True)
    assert list(off.args) == ["out"]

    def code(k):
        return [
            line for line in k.asm.splitlines() if line.startswith("\t") and "." not in line[:3]
        ]

    assert code(off) == code(plain)
    assert not any("getreg" in line or "sendmsg" in line for line in code(off))
    for k in (plain, off):
        out = np.zeros(64, np.uint32)
        k.bind(grid=1, args={"out": out}).emulate()
        assert out.tolist() == [3 * t for t in range(64)]


@pytest.mark.parametrize("keep", ["last", "first", "sum"])
def test_laps_record_and_decode_on_the_emulator(no_store, keep):
    """Every wave keeps two rounds; the emulated clocks run at 20x the constant one."""
    import thin
    from tests.laps_kernels import gen_laps, laps_decl

    decl = laps_decl(keep)
    k = thin.compile(gen_laps, keep=keep)
    out = np.zeros(64 * 2, np.uint32)
    bound = k.bind(grid=2, args={"out": out, **decl.sizes(2)})
    bound.emulate()
    assert out.tolist() == [4 * t for t in range(128)]
    t = decl.decode(bound.owned["laps"], 2)
    assert t.rounds == [[2, 2], [2, 2]]
    assert {m for row in t.clock_mhz for m in row} == {2000.0}
    if keep == "sum":
        assert all(v > 0 for n in decl.names for row in t.laps[n] for v in row)
    else:
        phases = t.phases()
        assert len(phases["store"]) == 2 * 2 * 2 and all(p > 0 for p in phases["store"])
    assert "never taken" not in t.summary()


def test_the_example_vadd_times_its_waves_with_laps(no_store):
    """examples/vadd.py with laps: z is x + y and every wave's two laps are kept."""
    import thin
    from examples.vadd import build_vadd

    k = thin.compile(build_vadd, laps=True)
    n = 1000
    x, y = np.arange(n, dtype=np.float32), np.full(n, 0.5, np.float32)
    z = np.zeros(n, np.float32)
    bound = k(x=x, y=y, z=z, n=n)
    bound.emulate()
    assert z.tolist() == (x + y).tolist()
    t = k.definition.meta["laps"].decode(bound.owned["laps"], 4)
    assert t.rounds == [[1] * 8] * 4
    phases = t.phases()
    assert len(phases["load"]) == len(phases["store"]) == 32
    assert all(p > 0 for p in phases["load"] + phases["store"])


def test_a_round_s_first_lap_is_taken_with_nothing_to_measure_from(no_store):
    """One round a wave: its first lap has no lap before it, which the
    summary says rather than calling it never taken."""
    import thin
    from examples.vadd import build_vadd

    k = thin.compile(build_vadd, laps=True)
    n = 1024
    x, y, z = np.ones(n, np.float32), np.ones(n, np.float32), np.zeros(n, np.float32)
    call = k(x=x, y=y, z=z, n=n)
    call.emulate()
    summary = k.definition.meta["laps"].decode(call.owned["laps"], call.grid[0]).summary()
    assert "  start        taken, with no lap before it to measure from" in summary
    assert "never taken" not in summary
