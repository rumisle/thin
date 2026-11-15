"""IR properties beyond the fixed golden cases."""

from __future__ import annotations

import contextlib
import enum
import os

import pytest

from thin import codegen, ir
from thin.ir import facts, stream
from thin.target import PHYS, target


@pytest.fixture(scope="module")
def t():
    return target("gfx1201")


def _values(t):
    p = ir.Program(t)
    a = p.value("v", 1, ir.B32, name="a")
    q = p.value("v", 4, ir.B32, name="q")
    s = p.value("s", 1, ir.B32, name="s")
    return p, a, q, s


def _refusal(p) -> str | None:
    try:
        ir.verify(p)
    except ir.Diag as e:
        return e.msg
    return None


# --- the verifier -----------------------------------------------------------------


def test_a_width_outside_the_slots_choices_is_named_as_a_tuple(t):
    p, a, q, _ = _values(t)
    p.body.append(ir.Instr(t.form("global_load_b32", 3), (a[0], q.whole, PHYS["off"])))
    assert _refusal(p) == "`global_load_b32`: operand 1 is 4 dwords; slot OPR_VGPR takes (1, 2)"


class _Small(enum.IntEnum):
    THREE = 3


class _Text(str):
    pass


@pytest.mark.parametrize(
    ("fields", "error"),
    [
        ({"offset": _Small.THREE}, None),
        ({"offset": 2**130}, f"`ds_load_b32`: offset:{2**130} outside 0..65535"),
        ({"offset": None}, "`ds_load_b32`: offset:None outside 0..65535"),
    ],
)
def test_int_fields_take_any_python_int_in_range(t, fields, error):
    p, a, _, _ = _values(t)
    p.body.append(ir.Instr(t.form("ds_load_b32", 2), (a[0], a[0]), fields))
    assert _refusal(p) == error


def test_str_subclasses_are_texts_and_choices(t):
    p, *_ = _values(t)
    p.body.append(
        ir.Instr(t.pseudo("ir.wait_all"), (), {"counter": _Text("ds"), "said": _Text("x")})
    )
    assert _refusal(p) is None


def test_the_first_failing_instruction_and_check_win(t):
    p, a, _, s = _values(t)
    add = t.form("v_add_nc_u32", 3)
    p.body.extend(
        [
            ir.Instr(add, (a[0], a[0], a[0])),
            ir.Instr(add, (a[0], s[0], ir.BlockRef("x")), {"offset": 1}, attrs={0: ir.Fence()}),
            ir.Instr(add, (s[0], a[0], a[0])),
        ]
    )
    assert _refusal(p) == "`v_add_nc_u32`: operand 2 is a block; slot OPR_VGPR takes vsrc"


def test_the_diag_carries_the_instructions_loc(t):
    p, a, _, s = _values(t)
    p.body.append(ir.Instr(t.form("v_add_nc_u32", 3), (s[0], a[0], a[0]), loc=5))
    with pytest.raises(ir.Diag) as raised:
        ir.verify(p)
    assert raised.value.loc == 5


# --- attributes -------------------------------------------------------------------


def test_attributes_round_trip_and_drop(t):
    i = ir.Instr(t.form("s_cbranch_scc1", 1), (ir.BlockRef("L"),))
    region = object()
    every = [
        ir.Comment("c"),
        ir.Ablation(frozenset({"x"})),
        ir.AliasOk(frozenset({("r", None)})),
        ir.Tag(("dma", 3)),
        ir.Staged(("mov", 1)),
        ir.InRegion(4),
        ir.Hazard(),
        ir.Places(("n",)),
        ir.Published("why"),
        ir.LdsAccess(region, "w", field=2, ordered="o"),
        ir.M0Set(9),
        ir.WavesTaken(frozenset({1, 63})),
        ir.Fence(),
    ]
    for a in every:
        i.put(a)
        assert i.attr(type(a)) == a
    assert list(i.attrs) == [type(a) for a in every]
    assert i.attr(ir.LdsAccess).region is region
    for a in every:
        i.drop(type(a))
        assert i.attr(type(a)) is None
    assert i.attrs is None


def test_a_bad_wave_leaves_the_old_attribute(t):
    i = ir.Instr(t.form("s_cbranch_scc1", 1), (ir.BlockRef("L"),))
    i.put(ir.WavesTaken(frozenset({2})))
    with pytest.raises(ValueError, match="wave 70"):
        i.put(ir.WavesTaken(frozenset({70})))
    assert i.attr(ir.WavesTaken) == ir.WavesTaken(frozenset({2}))


def test_helpers_read_attributes(t):
    i = stream.made(
        ir.Instr(t.form("s_nop", 1), (ir.Imm(0),)), comment="c", tag=("t",), abl=("x",), loc=3
    )
    assert (facts.said(i), facts.tag(i), facts.staged(i), facts.region(i), i.loc) == (
        "c",
        ("t",),
        (),
        None,
        3,
    )
    facts.say(i, "")
    assert facts.said(i) == "" and facts.waves_taken(None) is None
    assert facts.label(stream.label("L")) == "L"


# --- the stream and snapshots -----------------------------------------------------


def test_a_snapshot_never_sees_later_edits(t):
    p, *_ = _values(t)
    branch = ir.Instr(t.form("s_cbranch_scc1", 1), (ir.BlockRef("L"),))
    p.body.extend([branch, stream.label("L")])
    snap = ir.program(None, list(p.body))
    assert snap.wave_classes() == [None]
    branch.put(ir.WavesTaken(frozenset({0})))
    branch.set_ops([ir.BlockRef("M")])
    p.body.clear()
    assert snap.native.wave_classes() == [None]
    later = ir.program(None, [branch]).native
    assert later.wave_classes() == [None]


def test_a_free_list_refuses_values_below_zero(t):
    v = ir.Value(-1, "v", 1, 1, ir.B32, "x")
    i = ir.Instr(t.form("v_mov_b32", 2), (v[0], ir.Imm(0)))
    with pytest.raises(ValueError, match="a value numbered below 0"):
        ir.program(None, [i])
    with pytest.raises(ValueError, match="a kernel or its instructions"):
        ir.program(None)


def test_streams_behave_like_lists(t):
    xs = [ir.Instr(t.form("s_nop", 1), (ir.Imm(k),)) for k in range(5)]
    s = ir.Stream(xs)
    assert s[-1] == xs[4] and s[1:3] == xs[1:3] and len(s) == 5
    s[1:3] = list(s[3:5])
    assert list(s) == [xs[0], xs[3], xs[4], xs[3], xs[4]]
    with pytest.raises(ValueError, match="a sequence of size 1 to an extended slice of size 3"):
        s[::2] = [xs[0]]
    with pytest.raises(IndexError, match="stream index out of range"):
        s[9]
    s.insert(-100, xs[2])
    assert s.index(xs[2]) == 0 and s.pop() == xs[4]
    s.remove(xs[3])
    assert s.count(xs[3]) == 1 and ir.positions(s, [xs[3], xs[1]]) == [3, None]
    with pytest.raises(ValueError, match="not in the stream"):
        s.index(xs[1])
    assert s.mix() == {"nop": 4}


def test_iteration_is_live(t):
    xs = [ir.Instr(t.form("s_nop", 1), (ir.Imm(k),)) for k in range(3)]
    s = ir.Stream(xs)
    seen = []
    for i in s:
        seen.append(i)
        if len(seen) == 1:
            del s[-1]
    assert seen == xs[:2]


def test_put_on_a_streams_instr_changes_its_row(t):
    i = ir.Instr(t.form("s_nop", 1), (ir.Imm(0),))
    s = ir.Stream([i])
    s[0].put(ir.Comment("x"))
    assert facts.said(i) == "x" and s[0] == i


# --- locations --------------------------------------------------------------------


def _chain(table, depth):
    return table.capture() if depth == 0 else _chain(table, depth - 1)


def test_capture_interns_and_caps_at_four_frames():
    table = ir.LocTable(lambda code: code.co_name not in ("_chain", "_twice"))

    def _twice():
        return [_chain(table, 6) for _ in range(2)]

    a, b = _twice()
    assert a == b and len(table.lines(a)) == 4


def test_with_blocks_are_internal_frames():
    table = ir.LocTable(ir.internal)
    seen = []

    @contextlib.contextmanager
    def block():
        seen.append(ir.LOCS.capture())
        yield

    with block():
        pass
    assert ir.LOCS.lines(seen[0])[0].split(":")[0] == os.path.relpath(__file__)
    assert len(table) == 1


def test_the_default_rule():
    here = compile("", __file__, "exec")
    package = os.path.dirname(os.path.dirname(ir.__file__))
    library = compile("", os.path.join(package, "lib", "k.py"), "exec")
    ours = compile("", ir.__file__, "exec")
    unnormalized = compile(  # a writer's file by a path with "..": normalized first
        "",
        os.path.join(os.path.dirname(__file__), "..", "examples", "x.py"),
        "exec",
    )
    assert not ir.internal(here) and not ir.internal(library) and not ir.internal(unnormalized)
    assert ir.internal(ours) and ir.internal(contextlib.contextmanager.__code__)


def test_fuse_collapses():
    table = ir.LocTable(lambda code: False)
    a = table.name("a")
    assert table.fuse(0, a, a) == a and table.fuse() == 0
    assert table.fuse(a, a, what="w") == table.fuse(a, what="w") != a


# --- the pass runner --------------------------------------------------------------


def test_pass_times_accumulate_and_are_kept_when_a_pass_raises(t):
    unit = type("U", (), {})()
    unit.program, unit.pass_times = ir.Program(t), {"t": 1.0}

    def boom(u):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        ir.run(unit, [ir.Pass("t", lambda u: None), ir.Pass("t", boom)])
    assert unit.pass_times["t"] > 1.0 and "verify" not in unit.pass_times
    assert ir.run(unit, [], verify=False) is None and "verify" not in unit.pass_times


# --- options ----------------------------------------------------------------------


def test_options_nest_and_refuse_unknown_names():
    assert codegen.current() == codegen.Options()
    with codegen.using(audit=True):
        with codegen.using(peak_list=3) as inner:
            assert (inner.audit, inner.peak_list) == (True, 3)
        assert codegen.current().peak_list == 24
    assert not codegen.current().audit
    with pytest.raises(TypeError), codegen.using(nosuch=1):
        pass


def test_the_kernel_library_is_a_writer_when_installed_with_the_system(monkeypatch):
    """Found installing the wheel: thin/lib under site-packages (one of Python's
    paths) was taken as internal, so library frames lost their attribution."""
    package = os.path.dirname(os.path.dirname(ir.__file__))
    monkeypatch.setattr(ir, "_SYSTEM", (os.path.dirname(package),))
    library = compile("", os.path.join(package, "lib", "xwg.py"), "exec")
    ours = compile("", ir.__file__, "exec")
    assert not ir.internal(library) and ir.internal(ours)
