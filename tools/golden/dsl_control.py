"""Control-flow DSL snippets: Builder claims, branches, loops, lane regions and waits.

Each plan specifies a program and its intended outcome. Thin supplies the
actual outcome, refusal text and rendered instructions.
"""

from __future__ import annotations

import builtins
import contextlib
import io
import pathlib
import textwrap
from dataclasses import dataclass

from tools.golden import ROOT, encode, snippets

BASE = "crates/thin-dsl/tests/data/dsl/"
FILES = ("builder.json", "flow.json", "lanes.json", "waits.json")
OUTPUTS = tuple(BASE + name for name in FILES)

NOTATION = (
    "Each case is a DSL snippet (`src`) run on a fresh Builder for gfx1201, with `b` the "
    "Builder, `V`, `S` its files and every name of the language imported. `built` is the "
    "stream the Builder made, before any pass: one line per instruction, `<form name> "
    "<operands> {fields}; <attributes>`. A register operand is written as VReg repr "
    "(`<file>:<value name>[<index in group>]x<dwords>` and `+<lo>` when lo != 0); a named "
    "register by its name; an immediate as its value, with the writer's spelling in quotes "
    "when it has one (`16'0x10'`); a block as `@label`. Source modifiers are written `-x`, "
    "`|x|`. Fields are in stored order. Attributes in a fixed order: comment, tag, abl, "
    "alias_ok, staged, wait_uses, in_region, places, published, lds=(region, what, field, "
    "offset: none or var, ordered), m0_set, waves_taken, fence. `values` lists the "
    "program's values in id order: `id file name n= align= [index=] [pin=] [same_as=]`. "
    "`records` holds the Program's lane regions and loops, the deferred records' kinds, "
    "each file's entered / pinned groups and lanes-covered names, and the wave index. "
    "`error` is the refusal (exception class and message) raised while the snippet ran. "
    "`render` (the whole pipeline's output) is the final.s; `render_error` a refusal of the "
    "pipeline's passes."
)

INTENTS = ("built", "render", "render_error", "error")


@dataclass(frozen=True)
class Plan:
    """A snippet, its intended outcome and the behavior it exercises."""

    name: str
    intent: str
    behavior: str
    src: str


def plan(name, intent, behavior, src) -> Plan:
    """Declare a program and the stage/behavior it must cover."""
    assert intent in INTENTS, intent
    return Plan(name, intent, behavior, textwrap.dedent(src).strip("\n"))


# Shared declaration lines, so each program's own operations stand out.
LANE_PAIR = 'col, val = V.u32(name="col", at=2), V.u32(name="val", at=3)\n'
SCALAR_PAIR = 'base, flag = S.u32(name="base", at=4), S.u32(name="flag", at=5)\n'
SEL = 'sel = S.u32(name="sel", at=3)\n'
PQ = 'p, q = S.u32(name="p", at=1), S.u32(name="q", at=2)\n'
KEEP = 'keep = S.mask(name="keep", at=6)\n'
WIDX = 'widx = S.u32(name="widx", at=1)\n'
MEM = 'src = S.ptr(name="src", at=2)\nidx = V.u32(name="idx", at=1)\n'


# --- builder.json -------------------------------------------------------------

BUILDER = [
    plan(
        "stream_markers",
        "render",
        "nop, comment, plain and fenced barrier, label lower into the stream and render",
        textwrap.dedent("""
        b.nop(2)
        b.comment("stage the tile")
        b.barrier(fence=True)
        b.put_label(".Lstage")
        b.barrier("rows published")
        b.nop()
        """),
    ),
    plan(
        "claims_recorded",
        "render",
        "uniform (named SCC and SGPRs), undefined, assume and discard become ir statements",
        LANE_PAIR
        + SCALAR_PAIR
        + textwrap.dedent("""
        b.uniform("scc", flag, base, why="the flag is broadcast from one wave")
        b.undefined(val, why="the first trip writes val")
        b.assume(col, lo=16, hi=63, why="upper lanes only")
        b.discard(col, "the column index is not reused")
        """),
    ),
    plan(
        "ablation_drops_marked",
        "render",
        "an ablated class drops both the ablatable() block and an op marked abl=, keeps others",
        LANE_PAIR
        + textwrap.dedent("""
        b.ablate = {"tail"}
        b.emit(O.op("v_mov_b32", val, 7))
        with b.ablatable(None, "tail"):
            b.emit(O.op("v_add_nc_u32", val, 5, val))
        b.emit(O.op("v_lshlrev_b32", val, 3, val, abl=("tail",)))
        """),
    ),
    plan(
        "ablation_class_kept",
        "render",
        "ablatable() marks its ops with the classes while nothing is ablated",
        textwrap.dedent("""
        dst = V.u32(name="dst", at=6)
        with b.ablatable("warm", "pro"):
            b.emit(O.op("v_mov_b32", dst, 21))
        """),
    ),
    plan(
        "alias_waiver_dead_write",
        "render_error",
        "alias_ok by register, by name and over= marks reads; a dead write is still refused",
        LANE_PAIR
        + textwrap.dedent("""
        k = S.u32(name="k", at=6)
        with b.alias_ok("val#1", col, over=(k,), why="the aliasing branch is never taken"):
            b.emit(O.op("v_add_nc_u32", val, col, col))
        with b.alias_ok("tmp", why="a name not made yet"):
            b.emit(O.op("v_mov_b32", val, 17))
        """),
    ),
    plan(
        "alias_waiver_read_after",
        "render",
        "alias_ok marks survive a full render when the waived value is read",
        LANE_PAIR
        + textwrap.dedent("""
        with b.alias_ok(col, why="col and val share a lane"):
            b.emit(O.op("v_sub_nc_u32", val, col, val))
        b.emit(O.op("v_xor_b32", col, val, col))
        """),
    ),
    plan(
        "hold_then_release",
        "render",
        "hold marks dwords, other writes pass, release lets the held register be written",
        LANE_PAIR
        + textwrap.dedent("""
        b.hold("callee", col)
        b.emit(O.op("v_mov_b32", val, 13))
        b.release("callee")
        b.emit(O.op("v_mov_b32", col, val))
        """),
    ),
    plan(
        "needs_recorded",
        "render",
        "need(same_grid, arg=) and need(co_resident) are recorded and readable as b.needs",
        textwrap.dedent("""
        b.need("same_grid", "a generation counter is shared", arg="gen")
        b.need("co_resident", "spins on a peer's flag")
        assert sorted(b.needs) == ["co_resident", "same_grid"]
        assert b.needs["same_grid"] == {"gen": "a generation counter is shared"}
        """),
    ),
    plan(
        "position_after_emit",
        "render_error",
        "b.pos counts emissions, b.last and b.emitted(start) see the latest; dead write render",
        LANE_PAIR
        + textwrap.dedent("""
        b.emit(O.op("v_mov_b32", val, 40))
        mark = b.pos
        b.emit(O.op("v_not_b32", val, 41))
        assert b.pos - mark == 1
        assert b.last.op == "v_not_b32"
        assert [i.op for i in b.emitted(mark)] == ["v_not_b32"]
        assert [i.op for i in b.emitted(0, mark)] == ["v_mov_b32"]
        """),
    ),
    plan(
        "issued_per_counter",
        "render_error",
        "b.issued counts loads/km per counter on the path; a load over an in-flight one fails",
        MEM
        + textwrap.dedent("""
        q = V.u32(name="q", at=4)
        cnt = S.u32(name="cnt", at=8)
        b.emit(O.s_load(cnt, src))
        b.emit(O.op("global_load_b32", q, idx, src))
        b.emit(O.op("global_load_b32", q, idx, src))
        assert [b.issued(c) for c in ("load", "km", "store", "ds")] == [2, 1, 0, 0]
        """),
    ),
    plan(
        "held_register_written",
        "error",
        "writing a register held by a label is refused",
        LANE_PAIR
        + textwrap.dedent("""
        b.hold("epilogue", col, val)
        b.emit(O.op("v_add_nc_u32", val, col, 1))
        """),
    ),
    plan(
        "held_twice",
        "error",
        "holding a dword already held by another label is refused",
        LANE_PAIR
        + textwrap.dedent("""
        b.hold("outer", val)
        b.hold("inner", col, val)
        """),
    ),
    plan(
        "emit_after_seal",
        "error",
        "after seal() an instruction emission is refused",
        LANE_PAIR
        + textwrap.dedent("""
        b.emit(O.op("v_mov_b32", val, 3))
        b.seal()
        b.emit(O.op("v_mov_b32", val, 4))
        """),
    ),
    plan(
        "nop_after_seal",
        "error",
        "after seal() a builder statement (nop) is refused the same way",
        textwrap.dedent("""
        b.seal()
        b.nop(1)
        """),
    ),
    plan(
        "ablated_branch",
        "error",
        "control flow marked with an ablated class is refused",
        textwrap.dedent("""
        b.ablate = {"skip"}
        b.emit(O.op("s_cbranch_scc1", ".Lout", abl=("skip",)))
        """),
    ),
    plan(
        "uniform_without_reason",
        "error",
        "uniform() with an empty why is refused",
        SCALAR_PAIR + 'b.uniform(base, flag, why="")\n',
    ),
    plan(
        "undefined_without_reason",
        "error",
        "undefined() with an empty why is refused",
        LANE_PAIR + 'b.undefined(col, val, why="")\n',
    ),
    plan(
        "undefined_special_register",
        "error",
        "undefined() of a named (non-file) register is refused",
        'b.undefined(O.M0, why="m0 is set later")\n',
    ),
    plan(
        "undefined_exec_register",
        "error",
        "undefined() of exec is refused the same way",
        'b.undefined(O.EXEC, why="exec is restored later")\n',
    ),
    plan(
        "assume_without_reason",
        "error",
        "assume() with an empty why is refused",
        LANE_PAIR + 'b.assume(val, lo=1, hi=8, why="")\n',
    ),
    plan(
        "assume_inverted_range",
        "error",
        "assume() with lo above hi is refused with the u32 range rule",
        LANE_PAIR + 'b.assume(val, lo=200, hi=100, why="backwards")\n',
    ),
    plan(
        "assume_above_u32",
        "error",
        "assume() with hi at 2^32 is refused with the u32 range rule",
        LANE_PAIR + 'b.assume(col, hi=1 << 32, why="too wide")\n',
    ),
    plan(
        "assume_special_register",
        "error",
        "assume() of a named register is refused",
        'b.assume(O.VCC, hi=1, why="a flag")\n',
    ),
    plan(
        "discard_without_reason",
        "error",
        "discard() with an empty why is refused",
        LANE_PAIR + 'b.discard(val, "")\n',
    ),
    plan(
        "alias_ok_without_reason",
        "error",
        "alias_ok() with an empty why is refused",
        LANE_PAIR
        + textwrap.dedent("""
        with b.alias_ok("val#1", over=(col,), why=""):
            b.nop(0)
        """),
    ),
    plan(
        "need_unknown",
        "error",
        "need() of an unknown kind is refused",
        'b.need("exclusive", "one kernel at a time")\n',
    ),
    plan(
        "need_same_grid_without_arg",
        "error",
        "need(same_grid) without its argument is refused",
        'b.need("same_grid", "generations")\n',
    ),
    plan(
        "issued_unknown_counter",
        "error",
        "issued() of a counter outside load/store/ds/km is refused",
        'b.issued("lgkm")\n',
    ),
    plan(
        "second_entry",
        "error",
        "a second entry() is refused",
        textwrap.dedent("""
        b.entry(Args(z=u32))
        b.entry(Args(z=u32))
        """),
    ),
    plan(
        "fork_away_from_branch",
        "error",
        "restoring a fork not taken at a branch is refused, naming the instructions around it",
        LANE_PAIR
        + textwrap.dedent("""
        b.emit(O.op("v_mov_b32", col, 5))
        snap = b.fork()
        b.emit(O.op("v_add_nc_u32", val, col, 2))
        b.cnt = snap
        """),
    ),
    plan(
        "fork_first_instruction",
        "error",
        "a fork before any instruction is refused, showing (none) before it",
        LANE_PAIR
        + textwrap.dedent("""
        snap = b.fork()
        b.emit(O.op("v_mov_b32", val, 6))
        b.cnt = snap
        """),
    ),
    plan(
        "claims_inside_lanes",
        "render",
        "uniform and assume inside a lane region carry its in_region mark",
        LANE_PAIR
        + SCALAR_PAIR
        + textwrap.dedent("""
        with b.lanes(col >= 32):
            b.uniform(flag, why="written before the region")
            b.assume(col, lo=32, hi=63, why="the region's lanes")
        """),
    ),
]


# --- flow.json ----------------------------------------------------------------

FLOW = [
    plan(
        "if_elif_else_chain",
        "render",
        "when / elif_ / otherwise lowers to a branch chain joined after the last arm",
        SEL
        + textwrap.dedent("""
        out = S.u32(name="out")
        with b.when(sel < 2):
            out[...] = 100
        with b.elif_(sel == 5):
            out[...] = 200
        with b.otherwise():
            out[...] = sel + 7
        out += sel
        """),
    ),
    plan(
        "unless_then_lone_when",
        "render",
        "unless takes its arm on a false condition; a lone when has no else",
        SEL
        + textwrap.dedent("""
        out = S.u32(5, name="out")
        with b.unless(sel != 9):
            out[...] = 6
        with b.when(sel >= 12):
            out[...] = 8
        out += 2
        """),
    ),
    plan(
        "elif_ladder_without_else",
        "render",
        "a chain of elif_ arms ends without otherwise",
        SEL
        + textwrap.dedent("""
        out = S.u32(1, name="out")
        with b.when(sel > 10):
            out[...] = 4
        with b.elif_(sel > 3):
            out[...] = 9
        with b.elif_(sel == 0):
            out[...] = 16
        out ^= 5
        """),
    ),
    plan(
        "junctions_both_senses",
        "render",
        "& and | junctions in when and unless short-circuit through junction labels",
        PQ
        + textwrap.dedent("""
        r = S.u32(1, name="r")
        with b.unless((p != 3) | (q > 8)):
            r[...] = 10
        with b.when((p < 4) & (q == 0)):
            r[...] = 20
        with b.unless((p == 9) & (q != 2)):
            r[...] = 30
        with b.when((p >= 1) | (q <= 6)):
            r[...] = 40
        r += q
        """),
    ),
    plan(
        "junction_three_and",
        "render",
        "a three-part & junction",
        PQ
        + textwrap.dedent("""
        r = S.u32(2, name="r")
        with b.when((q > 1) & (p != q) & (p == 6)):
            r[...] = 3
        r += 1
        """),
    ),
    plan(
        "junction_three_or_nested",
        "render",
        "a three-part | junction, and an | nested inside &",
        PQ
        + textwrap.dedent("""
        r = S.u32(2, name="r")
        with b.when((p == 2) | (q == 4) | (p > q)):
            r[...] = 5
        with b.when(((p == 7) | (q < 2)) & (q != p)):
            r[...] = 6
        r += 1
        """),
    ),
    plan(
        "wave_reductions_and_bit_test",
        "render",
        "all/any of a lane compare, a mask and vcc, and a single-bit test, as branch conditions",
        LANE_PAIR
        + KEEP
        + textwrap.dedent("""
        word = S.u32(name="word", at=1)
        r = S.u32(0, name="r")
        with b.when(all(col != 0)):
            r[...] = 1
        with b.when(any(keep)):
            r[...] = 2
        with b.when((word & 1 << 9) == 0):
            r[...] = 3
        with b.when(all(O.VCC)):
            r[...] = 4
        with b.when(any(val >= 12)):
            r[...] = 5
        r += 3
        """),
    ),
    plan(
        "scalar_compare_instructions",
        "render",
        "an s_cmp / s_bitcmp instruction given as the condition",
        SEL
        + textwrap.dedent("""
        r = S.u32(0, name="r")
        with b.when(O.op("s_cmp_lg_u32", sel, 12)):
            r[...] = 1
        with b.unless(O.op("s_bitcmp0_b32", sel, 6)):
            r[...] = 2
        with b.when(O.op("s_cmp_lt_i32", sel, -3)):
            r[...] = 3
        r += 1
        """),
    ),
    plan(
        "middle_arm_terminates",
        "render",
        "an arm ending in s_endpgm needs no jump to the chain's end",
        SEL
        + textwrap.dedent("""
        r = S.u32(0, name="r")
        with b.when(sel == 1):
            r[...] = 7
        with b.elif_(sel == 2):
            b.emit(O.op("s_endpgm"))
        with b.otherwise():
            r[...] = 9
        r += 4
        """),
    ),
    plan(
        "first_arm_terminates",
        "render",
        "the first (unless) arm ending in s_endpgm, its otherwise reached by the branch",
        SEL
        + textwrap.dedent("""
        r = S.u32(0, name="r")
        with b.unless(sel > 6):
            b.emit(O.op("s_endpgm"))
        with b.otherwise():
            r[...] = 11
        r += 1
        """),
    ),
    plan(
        "named_join_label",
        "render",
        "end= names the chain's join label",
        SEL
        + textwrap.dedent("""
        r = S.u32(0, name="r")
        with b.unless(sel == 4, end=".Ljoin"):
            r[...] = 8
        with b.otherwise():
            r[...] = 12
        r += sel
        """),
    ),
    plan(
        "comment_closes_chain",
        "render",
        "a comment between two when arms closes the first chain; the second starts anew",
        SEL
        + textwrap.dedent("""
        r = S.u32(0, name="r")
        with b.when(sel != 0):
            r[...] = 3
        b.comment("two separate chains")
        with b.when(sel < 8):
            r[...] = 4
        r += 6
        """),
    ),
    plan(
        "instruction_closes_chain",
        "render",
        "an instruction between two when arms closes the first chain",
        SEL
        + textwrap.dedent("""
        r = S.u32(0, name="r")
        with b.when(sel == 3):
            r[...] = 5
        r += 2
        with b.when(sel == 4):
            r[...] = 6
        r += 1
        """),
    ),
    plan(
        "nested_chain_in_first_arm",
        "render",
        (
            "a when/elif_ chain inside the first arm; the "
            "outer chain continues with elif_ and otherwise"
        ),
        PQ
        + textwrap.dedent("""
        r = S.u32(0, name="r")
        with b.when(p == 0):
            with b.when(q < 5):
                r[...] = 1
            with b.elif_(q < 9):
                r[...] = 2
        with b.elif_(p == 1):
            r[...] = 3
        with b.otherwise():
            with b.unless(q == 0):
                r[...] = 4
        r += 1
        """),
    ),
    plan(
        "counted_loop_named",
        "render",
        "a counted loop on an SGPR trip count, named, its index read in the body",
        textwrap.dedent("""
        trips = S.u32(name="trips", at=5)
        total = V.u32(3, name="total")
        for it in b.loop(trips, name="row"):
            total += it.i * 3
        """),
    ),
    plan(
        "constant_count_with_start",
        "render",
        "a constant trip count with a nonzero start",
        textwrap.dedent("""
        total = V.u32(1, name="total")
        for it in b.loop(6, start=2):
            total += it.i + 5
        """),
    ),
    plan(
        "loop_own_index",
        "render",
        "i= gives the loop the index register to count in, from start=",
        textwrap.dedent("""
        trips = S.u32(name="trips", at=5)
        col = S.u32(name="col")
        total = V.u32(0, name="total")
        for it in b.loop(trips, i=col, start=1):
            total += col
        """),
    ),
    plan(
        "until_with_break",
        "render",
        "an until= loop with a break_when inside",
        textwrap.dedent("""
        lim = S.u32(name="lim", at=1)
        v = S.u32(1, name="v")
        for it in b.loop(until=v > 90):
            v += 11
            it.break_when(v >= lim)
        """),
    ),
    plan(
        "repeat_break_junction",
        "render",
        "a repeat loop left only by a junction break_when, code after it",
        textwrap.dedent("""
        lim = S.u32(name="lim", at=1)
        v = S.u32(2, name="v")
        for it in b.loop():
            v += 5
            it.break_when((v > 30) & (v != lim))
        v += 9
        """),
    ),
    plan(
        "repeat_break_or",
        "render",
        "a repeat loop left by an | junction break_when with a comment",
        textwrap.dedent("""
        lim = S.u32(name="lim", at=1)
        v = S.u32(0, name="v")
        for it in b.loop():
            v += 4
            it.break_when((v == lim) | (v >= 64), comment="enough")
        v += 1
        """),
    ),
    plan(
        "peeled_counted",
        "render",
        "peel= on a named counted loop",
        textwrap.dedent("""
        trips = S.u32(name="trips", at=5)
        total = V.u32(0, name="total")
        for it in b.loop(trips, peel=2, name="pk"):
            total += it.i
        """),
    ),
    plan(
        "peeled_once",
        "render",
        "peel=1 with a value carried across trips",
        textwrap.dedent("""
        trips = S.u32(name="trips", at=5)
        total = V.u32(4, name="total")
        for it in b.loop(trips, peel=1):
            total ^= it.i
        """),
    ),
    plan(
        "value_made_in_body",
        "render",
        "a value made inside the body on every pass of the expansion",
        textwrap.dedent("""
        trips = S.u32(name="trips", at=5)
        total = V.u32(0, name="total")
        for it in b.loop(trips):
            sq = V.u32(it.i + 4, name="sq")
            total += sq
        """),
    ),
    plan(
        "exit_when_to_labels",
        "render",
        "exit_when to the default L_end and to a named label from a junction",
        textwrap.dedent("""
        code = S.u32(name="code", at=2)
        r = S.u32(0, name="r")
        b.exit_when(code > 40)
        r[...] = 5
        b.exit_when((code == 3) | (code == 8), to=".Lfinish")
        r += code
        b.put_label("L_end", from_branches=True)
        b.put_label(".Lfinish", from_branches=True)
        b.emit(O.op("s_endpgm"))
        """),
    ),
    plan(
        "loop_reads_arguments",
        "render",
        "kernel arguments loaded at entry are landed before a loop that reads them",
        textwrap.dedent("""
        k = b.entry(Args(cnt=u32, step=u32))
        cnt, step = k.load()
        total = V.u32(0, name="total")
        for it in b.loop(cnt):
            total += step
        """),
    ),
    plan(
        "audit_steady_loop",
        "render",
        "with audit on, a body that is the same each pass passes the steady-state check",
        textwrap.dedent("""
        import dataclasses
        b.options = dataclasses.replace(b.options, audit=True)
        trips = S.u32(name="trips", at=5)
        total = V.u32(9, name="total")
        for it in b.loop(trips):
            total += it.i * 4
        """),
    ),
    plan(
        "lane_cond_in_when",
        "error",
        "a lane (vector) compare as a branch condition is refused",
        LANE_PAIR
        + textwrap.dedent("""
        with b.when(col >= 6):
            pass
        """),
    ),
    plan(
        "lane_cond_in_unless",
        "error",
        "unless() with a lane compare is refused the same way",
        LANE_PAIR
        + textwrap.dedent("""
        with b.unless(val != 0):
            pass
        """),
    ),
    plan(
        "lane_compare_instruction",
        "error",
        "a v_cmp instruction as a branch condition is refused",
        LANE_PAIR
        + textwrap.dedent("""
        with b.when(O.op("v_cmp_lt_u32", O.VCC, 20, col)):
            pass
        """),
    ),
    plan(
        "non_compare_instruction",
        "error",
        "a scalar ALU instruction that is no compare is refused as a condition",
        SEL
        + textwrap.dedent("""
        with b.when(O.op("s_and_b32", sel, sel, 15)):
            pass
        """),
    ),
    plan(
        "bare_register_condition",
        "error",
        "a bare SGPR value is not a condition",
        textwrap.dedent("""
        word = S.u32(name="word", at=1)
        with b.when(word):
            pass
        """),
    ),
    plan(
        "elif_first",
        "error",
        "elif_ with no chain open is refused",
        SEL
        + textwrap.dedent("""
        with b.elif_(sel != 2):
            pass
        """),
    ),
    plan(
        "otherwise_after_comment",
        "error",
        "otherwise after a comment closed the chain is refused",
        SEL
        + textwrap.dedent("""
        with b.when(sel > 1):
            pass
        b.comment("closes it")
        with b.otherwise():
            pass
        """),
    ),
    plan(
        "elif_after_label",
        "error",
        "elif_ after a label closed the chain is refused",
        SEL
        + textwrap.dedent("""
        with b.when(sel > 1):
            pass
        b.put_label(".Lmid")
        with b.elif_(sel == 0):
            pass
        """),
    ),
    plan(
        "elif_after_else",
        "error",
        "elif_ after otherwise is refused (otherwise is the last arm)",
        SEL
        + textwrap.dedent("""
        with b.unless(sel == 3):
            pass
        with b.otherwise():
            pass
        with b.elif_(sel == 4):
            pass
        """),
    ),
    plan(
        "otherwise_twice",
        "error",
        "a second otherwise after otherwise is refused",
        SEL
        + textwrap.dedent("""
        with b.when(sel == 3):
            pass
        with b.otherwise():
            pass
        with b.otherwise():
            pass
        """),
    ),
    plan(
        "count_and_until",
        "error",
        "a loop given both a count and until= is refused",
        textwrap.dedent("""
        trips = S.u32(name="trips", at=5)
        v = S.u32(name="v", at=1)
        for it in b.loop(trips, until=v == 3):
            pass
        """),
    ),
    plan(
        "repeat_with_start",
        "error",
        "start= on a repeat loop (no count) is refused",
        textwrap.dedent("""
        for it in b.loop(start=3):
            pass
        """),
    ),
    plan(
        "until_with_index",
        "error",
        "i= on an until loop is refused",
        textwrap.dedent("""
        v = S.u32(name="v", at=1)
        j = S.u32(name="j")
        for it in b.loop(until=v > 2, i=j):
            pass
        """),
    ),
    plan(
        "peel_with_until",
        "error",
        "peel= on an until loop is refused",
        textwrap.dedent("""
        v = S.u32(name="v", at=1)
        for it in b.loop(until=v != 0, peel=2):
            pass
        """),
    ),
    plan(
        "zero_trips",
        "error",
        "a constant loop whose start equals its count is refused (0 trips)",
        textwrap.dedent("""
        for it in b.loop(5, start=5):
            pass
        """),
    ),
    plan(
        "negative_trips",
        "error",
        "a constant loop starting past its count is refused",
        textwrap.dedent("""
        for it in b.loop(3, start=7):
            pass
        """),
    ),
    plan(
        "python_break_then_instruction",
        "error",
        "a Python break out of a loop body is refused at the next emission",
        textwrap.dedent("""
        trips = S.u32(name="trips", at=5)
        v = S.u32(0, name="v")
        for it in b.loop(trips):
            v += 1
            break
        v += 2
        """),
    ),
    plan(
        "python_break_then_comment",
        "error",
        "a Python break out of a loop body is refused at the next comment",
        textwrap.dedent("""
        trips = S.u32(name="trips", at=5)
        for it in b.loop(trips, name="scan"):
            break
        b.comment("after")
        """),
    ),
    plan(
        "body_value_width_changes",
        "error",
        "a body pass making a value at another width is refused",
        textwrap.dedent("""
        import dataclasses
        b.options = dataclasses.replace(b.options, audit=True)
        trips = S.u32(name="trips", at=5)
        widths = [2]
        for it in b.loop(trips):
            S.u32(0, n=widths[0], name="pair")
            widths[0] = 1
        """),
    ),
    plan(
        "audit_body_drifts",
        "error",
        "with audit on, a body that differs on its second pass is refused",
        textwrap.dedent("""
        import dataclasses
        b.options = dataclasses.replace(b.options, audit=True)
        trips = S.u32(name="trips", at=5)
        total = V.u32(1, name="total")
        step = [4]
        for it in b.loop(trips):
            total += step[0]
            step[0] *= 2
        """),
    ),
    plan(
        "lane_junction_in_when",
        "error",
        "an | junction of lane compares in a branch is refused",
        LANE_PAIR
        + textwrap.dedent("""
        with b.when((col > 2) | (col < 1)):
            pass
        """),
    ),
    plan(
        "mixed_junction_in_when",
        "error",
        "a junction mixing a scalar and a lane compare in a branch is refused",
        LANE_PAIR
        + SEL
        + textwrap.dedent("""
        with b.when((sel == 0) & (val < 4)):
            pass
        """),
    ),
    plan(
        "junction_as_mask",
        "render",
        "an | junction of scalar compares made into a mask value",
        PQ + "S.mask((p != 0) | (q < 3))\n",
    ),
    plan(
        "and_junction_as_named_mask",
        "render",
        "an & junction made into a named mask value",
        PQ + 'S.mask((p > 4) & (q == p), name="both")\n',
    ),
]


# --- lanes.json ---------------------------------------------------------------

LANES = [
    plan(
        "nested_skip_region",
        "render",
        "a save-less region with a nested skip= region inside",
        LANE_PAIR
        + textwrap.dedent("""
        with b.lanes(col >= 8):
            val -= 1
            with b.lanes(val == 0, skip=True):
                val[...] = col + 1
        """),
    ),
    plan(
        "mask_and_condition_lists",
        "render",
        "a mask value alone, a mask then a compare, and two compares then a mask",
        LANE_PAIR
        + KEEP
        + textwrap.dedent("""
        with b.lanes(col < 30, keep):
            val += 6
        with b.lanes(keep):
            val += 5
        with b.lanes(val != 1, col > 2, keep):
            val += 4
        """),
    ),
    plan(
        "scalar_conditions",
        "render",
        "an SCC condition alone and after a mask",
        LANE_PAIR
        + SEL
        + KEEP
        + textwrap.dedent("""
        with b.lanes(keep, sel < 9):
            val += 8
        with b.lanes(sel != 3):
            val += 7
        """),
    ),
    plan(
        "explicit_and_fresh_save",
        "render",
        "save= a given mask register, and save=True with a comment",
        LANE_PAIR
        + textwrap.dedent("""
        held = S.mask(name="held")
        with b.lanes(col > 11, val < 7, save=True, comment="pair"):
            val += 3
        with b.lanes(col != 2, save=held):
            val += 1
        """),
    ),
    plan(
        "narrow_in_region",
        "render",
        "narrow() inside a region by a lane compare, then by a mask",
        LANE_PAIR
        + KEEP
        + textwrap.dedent("""
        with b.lanes(col != 0):
            val += 4
            b.narrow(val > 10)
            val += 1
            b.narrow(keep, comment="kept rows")
            val += 2
        """),
    ),
    plan(
        "all_lanes_constant",
        "render",
        "an all-lanes constant (with None) opens no region",
        LANE_PAIR
        + textwrap.dedent("""
        with b.lanes(None, -1):
            val += 2
        """),
    ),
    plan(
        "constant_mask_and_exec_instruction",
        "render",
        "a constant mask, and an instruction writing exec, as region entries",
        LANE_PAIR
        + KEEP
        + textwrap.dedent("""
        with b.lanes(0x0F0F):
            val += 1
        with b.lanes(O.op("s_and_b32", O.EXEC, keep, 0xFFFF)):
            val += 2
        """),
    ),
    plan(
        "compare_instruction_last",
        "render",
        "a v_cmp instruction as the last condition after a mask",
        LANE_PAIR
        + KEEP
        + textwrap.dedent("""
        with b.lanes(keep, O.op("v_cmp_ne_u32", O.VCC, 4, col)):
            val += 3
        """),
    ),
    plan(
        "raw_exec_block",
        "render",
        "raw_exec(why) inside a region lets an exec write through",
        LANE_PAIR
        + textwrap.dedent("""
        with b.lanes(col < 12):
            val += 2
            with b.raw_exec("lane 1 alone for the reduction"):
                b.emit(O.op("s_mov_b32", O.EXEC, 2))
            val += 1
        """),
    ),
    plan(
        "consecutive_regions",
        "render",
        "two save-less regions back to back",
        LANE_PAIR
        + textwrap.dedent("""
        with b.lanes(col < 20):
            val += 5
        with b.lanes(val >= 2):
            val += 6
        """),
    ),
    plan(
        "store_region_before_endpgm",
        "render",
        "a region with a store closed right before s_endpgm",
        LANE_PAIR
        + textwrap.dedent("""
        dst = S.ptr(name="dst", at=4)
        with b.lanes(val != 0):
            b.emit(O.global_store(col, val, dst, offset=8))
        b.emit(O.op("s_endpgm"))
        """),
    ),
    plan(
        "wave_blocks",
        "render",
        "wave blocks over single, multiple and overlapping wave ranges",
        WIDX
        + textwrap.dedent("""
        b.wave_index(widx, 8)
        acc = V.u32(name="acc", at=0)
        with b.waves([4, 5, 6, 7]):
            acc[...] = acc + 9
        with b.waves([2]):
            acc[...] = acc + 8
        with b.waves([1, 2, 3]):
            acc[...] = acc + 7
        with b.waves([6, 7]):
            acc[...] = acc + 6
        """),
    ),
    plan(
        "wave_block_by_wave",
        "render",
        "a wave block with by_wave=True",
        WIDX
        + textwrap.dedent("""
        b.wave_index(widx, 2, by_wave=True)
        acc = V.u32(name="acc", at=0)
        with b.waves([0]):
            acc[...] = acc + 3
        """),
    ),
    plan(
        "entry_wave_and_lane",
        "render",
        "the entry's wave and lane indices for a 256-lane workgroup",
        textwrap.dedent("""
        k = b.entry(wgsize=256)
        g = V.u32(k.lane + k.wave, name="g")
        g += k.wave
        """),
    ),
    plan(
        "exec_write_in_region",
        "error",
        "an unnamed exec write inside a region is refused",
        LANE_PAIR
        + textwrap.dedent("""
        with b.lanes(col >= 4):
            b.emit(O.op("s_or_b32", O.EXEC, O.EXEC, 1))
        """),
    ),
    plan(
        "saved_exec_overwritten",
        "error",
        "writing the region's saved exec register is refused",
        LANE_PAIR
        + textwrap.dedent("""
        held = S.mask(name="held")
        with b.lanes(val < 3, save=held):
            b.emit(O.op("s_and_b32", held, held, 3))
        """),
    ),
    plan(
        "branch_leaves_region",
        "error",
        "a branch out of a region is refused",
        LANE_PAIR
        + textwrap.dedent("""
        with b.lanes(col != 7):
            b.emit(O.op("s_cbranch_scc1", ".Lescape"))
        b.put_label(".Lescape")
        """),
    ),
    plan(
        "wide_save_register",
        "error",
        "a 64-bit save register on wave32 is refused",
        LANE_PAIR
        + textwrap.dedent("""
        wide = S.u64(name="wide")
        with b.lanes(col < 2, save=wide):
            pass
        """),
    ),
    plan(
        "narrow_without_region",
        "error",
        "narrow() outside a region is refused",
        LANE_PAIR + "b.narrow(col != 5)\n",
    ),
    plan(
        "narrow_by_scc",
        "error",
        "narrow() by a scalar compare is refused",
        LANE_PAIR
        + SEL
        + textwrap.dedent("""
        with b.lanes(col > 0):
            b.narrow(sel > 1)
        """),
    ),
    plan(
        "scc_after_two_compares",
        "error",
        "a scalar condition after two lane conditions is refused",
        LANE_PAIR
        + SEL
        + textwrap.dedent("""
        with b.lanes(col < 3, val > 1, sel == 0):
            pass
        """),
    ),
    plan(
        "scc_with_given_save",
        "error",
        "a scalar condition with an explicit save register is refused",
        SEL
        + textwrap.dedent("""
        held = S.mask(name="held")
        with b.lanes(sel == 2, save=held):
            pass
        """),
    ),
    plan(
        "register_name_as_condition",
        "error",
        "a register name string as a region condition is refused",
        textwrap.dedent("""
        with b.lanes("vcc_lo"):
            pass
        """),
    ),
    plan(
        "compare_instruction_first",
        "error",
        "a compare instruction before another condition is refused",
        LANE_PAIR
        + textwrap.dedent("""
        with b.lanes(O.op("v_cmp_lt_u32", O.VCC, 6, val), col < 9):
            pass
        """),
    ),
    plan(
        "exec_instruction_saved",
        "error",
        "an exec-writing entry instruction with a save register is refused",
        KEEP
        + textwrap.dedent("""
        held = S.mask(name="held")
        with b.lanes(O.op("s_and_b32", O.EXEC, keep, 0xF), save=held):
            pass
        """),
    ),
    plan(
        "exec_instruction_nested",
        "error",
        "an exec-writing entry instruction nested without save is refused",
        LANE_PAIR
        + KEEP
        + textwrap.dedent("""
        with b.lanes(val >= 1):
            with b.lanes(O.op("s_and_b32", O.EXEC, keep, 0xFF)):
                pass
        """),
    ),
    plan(
        "waves_before_index",
        "error",
        "a wave block before wave_index is refused",
        textwrap.dedent("""
        with b.waves([1, 2]):
            pass
        """),
    ),
    plan(
        "waves_with_gap",
        "error",
        "a non-contiguous wave set is refused",
        WIDX
        + textwrap.dedent("""
        b.wave_index(widx, 8)
        with b.waves([1, 3]):
            pass
        """),
    ),
    plan(
        "waves_beyond_count",
        "error",
        "a wave set reaching past the workgroup's waves is refused",
        WIDX
        + textwrap.dedent("""
        b.wave_index(widx, 2)
        with b.waves([1, 2]):
            pass
        """),
    ),
    plan(
        "raw_exec_without_reason",
        "error",
        "raw_exec() with an empty why is refused",
        textwrap.dedent("""
        with b.raw_exec(""):
            b.nop(0)
        """),
    ),
]


# --- waits.json ---------------------------------------------------------------

WAITS = [
    plan(
        "wait_values_each_counter",
        "render",
        "waits named by value (vector loads, a scalar load), alone and as a list, with comments",
        MEM
        + textwrap.dedent("""
        lo, hi = V.f32(name="lo"), V.f32(name="hi")
        cnt = S.u32(name="cnt")
        b.emit(O.s_load(cnt, src, offset=12))
        b.emit(O.global_load(hi, idx, src, offset=32))
        b.emit(O.global_load(lo, idx, src))
        b.wait([hi, cnt], comment="hi and cnt")
        hi[...] = hi + 0.5
        b.wait(lo, "lo landed")
        lo[...] = lo * hi
        """),
    ),
    plan(
        "wait_counters_and_left",
        "render",
        "waits by counter (LGKM, VM with a comment) and left= on one counter",
        MEM
        + textwrap.dedent("""
        lo = V.f32(name="lo")
        b.emit(O.global_load(lo, idx, src, offset=4))
        b.wait(O.LGKM)
        b.wait(O.VM, left=2)
        b.wait(O.VM, comment="all vm")
        """),
    ),
    plan(
        "wait_tags",
        "render",
        "waits by DMA tags, one and several with written=True",
        MEM
        + textwrap.dedent("""
        lo, hi = V.f32(name="lo"), V.f32(name="hi")
        b.emit(O.global_load(lo, idx, src, tag=("copy", 3)))
        b.emit(O.global_load(hi, idx, src, offset=64, tag=("copy", 4)))
        b.wait(tags=[("copy", 4)], comment="second copy")
        b.wait(tags=[("copy", 3), ("copy", 4)], written=True)
        """),
    ),
    plan(
        "wait_region",
        "render",
        "a wait on an LDS region after a ds write to it",
        textwrap.dedent("""
        tid = V.u32(name="tid", at=2)
        word = V.u32(name="word", at=3)
        L = b.lds()
        T = L.region("T", (128,), "u32")
        b.emit(O.ds_write(T.at(tid), word))
        b.wait(T)
        """),
    ),
    plan(
        "wait_in_loop_for_later_load",
        "render_error",
        "a wait in a loop body for a value loaded before and again later in the body",
        MEM
        + textwrap.dedent("""
        trips = S.u32(name="trips", at=5)
        lo = V.f32(name="lo")
        b.emit(O.global_load(lo, idx, src, offset=16))
        for it in b.loop(trips):
            b.wait(lo)
            lo[...] = lo - 3.0
            b.emit(O.global_load(lo, idx, src, offset=16))
        """),
    ),
    plan(
        "wait_deferred_to_later_load",
        "render_error",
        "a wait on a value first loaded later in the loop body is deferred to that load",
        MEM
        + textwrap.dedent("""
        trips = S.u32(name="trips", at=5)
        hi = V.f32(1.0, name="hi")
        for it in b.loop(trips):
            b.wait(hi, "the previous trip's hi")
            hi[...] = hi * 4.0
            b.emit(O.global_load(hi, idx, src, offset=24))
        b.wait(hi)
        """),
    ),
    plan(
        "wait_ablated",
        "render",
        "a wait and its load both dropped by an ablated class",
        MEM
        + textwrap.dedent("""
        lo = V.f32(name="lo")
        b.ablate = {"fetch"}
        b.emit(O.global_load(lo, idx, src, offset=40), abl=("fetch",))
        b.wait(lo, abl=("fetch",), comment="dropped")
        """),
    ),
    plan(
        "arguments_waited_at_first_read",
        "render",
        "argument loads are waited for at each argument's first read, across arms",
        textwrap.dedent("""
        k = b.entry(Args(buf=ptr, lim=u32, add=u32))
        buf, lim, add = k.load()
        r = S.u32(name="r")
        with b.when(lim > 7):
            r[...] = 4
        with b.otherwise():
            r[...] = add + 3
        r += lim
        """),
    ),
    plan(
        "land_arguments",
        "render",
        "land_arguments(why) waits for every pending argument load",
        textwrap.dedent("""
        k = b.entry(Args(lim=u32, add=u32, buf=ptr))
        lim, add, buf = k.load()
        r = S.u32(add + 2, name="r")
        b.land_arguments("before the epilogue")
        r += lim
        """),
    ),
    plan(
        "wait_for_nothing",
        "error",
        "a wait with nothing to wait for is refused",
        'b.wait(comment="empty")\n',
    ),
    plan(
        "left_on_two_counters",
        "error",
        "left= with two counters is refused",
        "b.wait(O.LGKM, O.VM, left=2)\n",
    ),
    plan(
        "wait_alu_value",
        "error",
        "a wait on a value an ALU op wrote is refused",
        textwrap.dedent("""
        t = V.u32(5, name="t")
        b.wait(t, "t")
        """),
    ),
    plan(
        "wait_unwritten_scalar",
        "error",
        "a wait on a pinned SGPR nothing wrote is refused",
        textwrap.dedent("""
        word = S.u32(name="word", at=7)
        b.wait(word)
        """),
    ),
    plan(
        "wait_on_number",
        "error",
        "a wait on a number is refused",
        "b.wait(7)\n",
    ),
    plan(
        "deferred_never_loaded",
        "error",
        "a wait in a loop on a value no load (before or later in the body) writes is refused",
        textwrap.dedent("""
        trips = S.u32(name="trips", at=5)
        hi = V.f32(2.0, name="hi")
        for it in b.loop(trips, name="spin"):
            b.wait(hi, "hi")
            hi[...] = hi - 1.0
        """),
    ),
    plan(
        "counter_by_name",
        "render",
        "waits by counter name strings",
        textwrap.dedent("""
        b.wait("km")
        b.wait("store")
        """),
    ),
    plan(
        "tag_never_issued",
        "error",
        "a wait for a DMA tag no op carried is refused, naming it",
        MEM
        + textwrap.dedent("""
        lo = V.f32(name="lo")
        b.emit(O.global_load(lo, idx, src, tag=("copy", 1)))
        b.wait(tags=[("copy", 1), ("copy", 2)])
        """),
    ),
    plan(
        "unknown_counter",
        "error",
        "a wait on a counter the target lacks is refused, listing its counters",
        'b.wait(Counter("gds"))\n',
    ),
    plan(
        "region_without_ops",
        "error",
        "a wait on an LDS region nothing touched is refused",
        textwrap.dedent("""
        tid = V.u32(name="tid", at=2)
        L = b.lds()
        T = L.region("T", (32,), "u32")
        U = L.region("U", (32,), "u32")
        b.emit(O.ds_write(T.at(tid), tid))
        b.wait(U)
        """),
    ),
    plan(
        "written_by_bare_store",
        "error",
        "wait(tags=, written=True) on a bare store that writes no register is refused",
        MEM
        + textwrap.dedent("""
        b.emit(O.op("global_store_b32", idx, idx, src, tag=("out",)))
        b.wait(tags=[("out",)], written=True)
        """),
    ),
]

PLANS = {"builder.json": BUILDER, "flow.json": FLOW, "lanes.json": LANES, "waits.json": WAITS}


# --- Evaluation ----------------------------------------------------------------


@contextlib.contextmanager
def no_fixture_reads():
    """Refuse any open of checked-in test data while the programs are evaluated."""
    data = (ROOT / "crates").as_posix(), (ROOT / "tests" / "data").as_posix()

    def guard(file):
        """Construct guard inputs."""
        if isinstance(file, (str, pathlib.PurePath)):
            path = pathlib.Path(file).resolve().as_posix()
            if path.startswith(data) and "/tests/data/" in path + "/":
                raise PermissionError(f"golden generator opened test data: {path}")

    saved = builtins.open, io.open, pathlib.Path.open

    def checked_open(file, *args, **kwargs):
        """Construct checked open inputs."""
        guard(file)
        return saved[0](file, *args, **kwargs)

    def checked_path_open(self, *args, **kwargs):
        """Construct checked path open inputs."""
        guard(self)
        return saved[2](self, *args, **kwargs)

    builtins.open = io.open = checked_open
    pathlib.Path.open = checked_path_open
    try:
        yield
    finally:
        builtins.open, io.open, pathlib.Path.open = saved


def outcome(case) -> str:
    """Identify the stage reached by an evaluated program."""
    for key in ("error", "render_error", "render"):
        if key in case:
            return key
    return "built"


def evaluate(p: Plan) -> dict:
    """Run one plan through thin; refusals are recorded exactly as thin raises them."""
    compile(p.src, "kernel.py", "exec")
    case = snippets.evaluate(p.name, p.src)
    refusal = case.get("render_error")
    if refusal is not None:
        # The vector runner executes each snippet as kernel.py; exec names it <string>.
        refusal["message"] = refusal["message"].replace("<string>", "kernel.py")
    got = outcome(case)
    if got != p.intent:
        detail = case.get("error") or case.get("render_error") or ""
        raise AssertionError(f"{p.name}: written for {p.intent}, thin gave {got} {detail}")
    return case


def documents() -> dict:
    """Evaluate the source plans under the fixture-I/O guard."""
    with no_fixture_reads():
        return {
            name: {"notation": NOTATION, "cases": [evaluate(p) for p in plans]}
            for name, plans in PLANS.items()
        }


def generate() -> dict:
    """Encode every authored control-flow program document."""
    names = [p.name for plans in PLANS.values() for p in plans]
    for plans in PLANS.values():
        seen = [p.name for p in plans]
        if len(seen) != len(set(seen)):
            raise ValueError("duplicate case names")
    del names
    return {BASE + name: encode(doc) for name, doc in documents().items()}
