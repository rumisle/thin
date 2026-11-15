"""Golden checks of assembly printing and register spelling."""

from __future__ import annotations

import pytest

from tests.golden import cases
from thin import ir
from thin.ir import stream
from thin.target import PHYS, target

PARAMS = ("vector", "case_id", "case", "doc")

BASES = {"x#1": 0, "y#1": 1, "z#1": 4, "q": 8, "p#1": 4, "d#1": 8, "s1#1": 1, "w#1": 20, "w2#1": 21}
"""Where printer.json places each value (its notation; w and w2 as its lines show)."""


class _File:
    """A register file with pinned values: by name until placed, then physical."""

    def __init__(self, groups: dict[str, int]):
        self.groups = groups
        self.placed = False

    def name_text(self, r: ir.VReg) -> str:
        return ir.name_text(r, self.groups[r.value.name])

    def text(self, r: ir.VReg) -> str:
        if not self.placed:
            return self.name_text(r)
        return ir.placed_text(r, BASES[r.value.name] + r.value.index)


def _instr(case):
    """The case's instruction and register files."""
    values: dict[tuple, ir.Value] = {}
    groups: dict[str, dict[str, int]] = {"v": {}, "s": {}}

    def op(o):
        if "reg" in o:
            key = (o["file"], o["reg"], o["index"])
            if key not in values:
                values[key] = ir.Value(
                    len(values), o["file"], o["value_n"], 1, ir.B32, o["reg"], 0, o["index"]
                )
            groups[o["file"]][o["reg"]] = 3 if o["reg"] == "q" else 1
            return ir.VReg(values[key], o["lo"], o["n"])
        if "phys" in o:
            return PHYS[o["phys"]]
        if "block" in o:
            return ir.BlockRef(o["block"])
        value = o["imm"]
        return ir.Imm(float(value["float"]) if isinstance(value, dict) else value, o["text"])

    def field(v):
        return tuple(v["tuple"]) if isinstance(v, dict) else v

    t = target("gfx1201")
    ops = [op(o) for o in case["ops"]]
    form = (
        t.pseudo(case["form"]) if case["form"].startswith("ir.") else t.form(case["form"], len(ops))
    )
    mods = None if case["mods"] is None else bytes(case["mods"])
    fields = {k: field(v) for k, v in case["fields"].items()}
    i = ir.Instr(form, ops, fields, mods=mods)
    i.put(ir.Comment(case["comment"]))
    for attr in case["attrs"]:
        i.put(eval(attr, {"Places": ir.Places}))
    return i, {f: _File(g) for f, g in groups.items()}


@pytest.mark.parametrize(PARAMS, cases("codegen", "printer.json", "cases"))
def test_printer(vector, case_id, case, doc):
    i, files = _instr(case)
    before = {"text": i.text, "bare": stream.bare(i, files), "named": stream.named(i, files)}
    assert before == case["before_placement"]
    for f in files.values():
        f.placed = True
    placed = {
        "bare": stream.bare(i, files),
        "named": stream.named(i, files),
        "render": stream.render(i, files),
        "passed": stream.passed(i, files),
        "annotation": ir.annotation(i, lambda r: files[r.file].text(r)),
    }
    assert placed == case["placed"]
