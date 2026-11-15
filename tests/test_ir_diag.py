"""Native messages as Python spells them."""

import pytest

from thin import ir


def _line(k):
    return f"s_mov_b32 s{k}, 1"


def test_parts_spell_through_the_callbacks():
    parts = [("instr", 2, 0, ""), ("where", 2, 0, ""), ("text", 0, 0, ": "), ("clip", 3, 6, "")]
    error = ir.from_native("Schedule", 0, parts, [], _line, lambda k: f" (k.py:{k})")
    assert isinstance(error, ir.ScheduleError)
    assert error.msg == "s_mov_b32 s2, 1 (k.py:2): s_mov_"


def test_where_parts_drop_without_a_where_callback():
    error = ir.from_native("Flow", 0, [("instr", 1, 0, ""), ("where", 1, 0, "")], [], _line)
    assert isinstance(error, ir.FlowError) and error.msg == "s_mov_b32 s1, 1"


def test_a_where_of_an_unknown_loc_spells_nothing():
    assert ir.LOCS.where(0) == ""
    assert ir.LOCS.where(len(ir.LOCS) + 5) == ""


def test_notes_and_classes_follow_the_native_class():
    notes = [(0, [("text", 0, 0, "the load")])]
    error = ir.from_native("Diag", 7, [("text", 0, 0, "refused")], notes)
    assert type(error) is ir.Diag and error.loc == 7 and error.notes == ((0, "the load"),)
    assert type(ir.from_native("NoFixpoint", 0, [], [])) is ir.NoFixpoint
    assert type(ir.from_native("Asm", 0, [], [])) is ir.AsmError


def test_an_instruction_part_without_a_line_is_an_internal_error():
    with pytest.raises(TypeError, match="instr part"):
        ir.from_native("Schedule", 0, [("instr", 3, 0, "")], [])
