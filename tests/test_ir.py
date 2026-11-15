"""Golden checks of the Python instruction and analysis API."""

from __future__ import annotations

import os

import pytest

from tests.golden import cases, whole
from thin import ir
from thin.ir import facts, stream
from thin.target import PHYS, target

PARAMS = ("vector", "case_id", "case", "doc")


@pytest.fixture(scope="module")
def t():
    return target("gfx1201")


def _program(t):
    """verify.json's and model.json's values: a, b (v), p (v, 2), s (s), s2 (s, 2)."""
    p = ir.Program(t)
    for name, file, n in (
        ("a", "v", 1),
        ("b", "v", 1),
        ("p", "v", 2),
        ("s", "s", 1),
        ("s2", "s", 2),
    ):
        p.value(file, n, ir.B32, name=name)
    return p


# --- model.json ----------------------------------------------------------------


def _model_values(t, key, case):
    p = _program(t)
    v = next(v for v in p.values if v.name == key)
    file, n = case.split(", ")
    assert (v.file, v.n) == (file, int(n.split()[0]))


def _round_trip(t, case):
    given = eval(case["given"])
    i = ir.Instr(t.form("v_add_nc_u32", 3), (), {"offset": given})
    back = i.field("offset")
    assert repr(back) == case["back"]
    assert (type(back) is type(given)) == case["same_type"]
    assert (back is given) == case["identical"]


def _attribute(t, case):
    p = _program(t)
    names = {c: getattr(ir, c) for c in ir.__all__ if c[0].isupper()}
    put = case["put"]
    a = ir.WaitUses((p.values[1][0],)) if put.startswith("WaitUses") else eval(put, names)
    i = ir.Instr(t.form("v_add_nc_u32", 3), ())
    i.put(a)
    got = i.attr(type(a))
    if case["attr_after"] == "None":
        assert got is None
    elif isinstance(a, (ir.Staged, ir.WaitUses, ir.LdsAccess)):
        assert got is a and repr(got) == case["attr_after"]
    else:
        assert got == eval(case["attr_after"], names)
        assert got is not a


def _error(t, case):
    form = t.form("v_add_nc_u32", 3)
    calls = {
        "WavesTaken wave 64": lambda: ir.Instr(form, ()).put(ir.WavesTaken(frozenset({64}))),
        "unknown field name": lambda: ir.Instr(form, (), {"nosuch": 1}),
        "operand of another type": lambda: ir.Instr(form, (5,)),
        "attr of a non-attribute class": lambda: ir.Instr(form, ()).attr(int),
    }
    with pytest.raises(BaseException) as raised:
        calls[case["case"]]()
    assert type(raised.value).__name__ == case["error"]
    assert raised.value.args[0] == case["message"]


def _reprs(t, key, want):
    p = _program(t)
    a, b, pv, s2 = p.values[0], p.values[1], p.values[2], p.values[4]
    got = {
        "value": lambda: repr(a),
        "value_s2": lambda: repr(s2),
        "vreg_whole": lambda: repr(pv.whole),
        "vreg_dword1": lambda: repr(pv.whole.dword(1)),
        "vreg_slice": lambda: repr(pv[1:2]),
        "vreg_index": lambda: repr(pv[1]),
        "through": lambda: pv.whole.through,
        "through_part": lambda: ir.VReg(pv, 0, 1, "lo").through,
        "vreg_eq_ignores_part": lambda: ir.VReg(pv, 0, 1, "lo") == pv[0],
        "instr": lambda: repr(ir.Instr(t.form("v_add_nc_u32", 3), (a[0], a[0], ir.Imm(4)))),
        "dwords": lambda: list(pv.whole.dwords()),
    }[key]()
    assert got == want
    del b


def _instr_facts(t, key, want):
    p = _program(t)
    a, b, s = p.values[0], p.values[1], p.values[3]
    i = ir.Instr(t.form("v_add_nc_u32", 3), (a[0], a[0], b[0]))
    claim = ir.Instr(t.pseudo("ir.uniform"), (s[0],), {"why": "w"})
    got = {
        "defs": lambda: repr(i.defs),
        "uses": lambda: repr(i.uses),
        "regs": lambda: repr(i.regs),
        "writes": lambda: sorted(i.writes),
        "phys_uses": lambda: sorted(i.phys_uses()),
        "kind": lambda: i.kind,
        "op": lambda: i.op,
        "where_unknown": lambda: i.where,
        "target_none": lambda: i.target,
        "pseudo_regs": lambda: repr(claim.regs),
    }[key]()
    assert got == want


@pytest.mark.parametrize(PARAMS, cases("ir", "model.json"))
def test_model(vector, case_id, case, doc, t):
    section, key = case_id.rstrip("]").split("[", 1)
    if section == "values":
        _model_values(t, key, case)
    elif section == "field_round_trip":
        _round_trip(t, case)
    elif section == "attributes":
        _attribute(t, case)
    elif section == "errors":
        _error(t, case)
    elif section == "reprs":
        _reprs(t, key, case)
    else:
        _instr_facts(t, key, case)


# --- verify.json -----------------------------------------------------------------


def _verify_program(t, name):
    """Build the mapped authored input, independently of its expected outcome."""
    from tools.golden.ir_inputs import verify_program

    return verify_program(t, name)


@pytest.mark.parametrize(PARAMS, cases("ir", "verify.json", "cases"))
def test_verify(vector, case_id, case, doc, t):
    p = _verify_program(t, case["name"])
    if case["error"] is None:
        ir.verify(p, case["printable"])
        return
    with pytest.raises(ir.Diag) as raised:
        ir.verify(p, case["printable"])
    assert raised.value.msg == case["error"]


# --- loc.json ---------------------------------------------------------------------

_SOURCES = {
    "/w/lib/helpers.py": "def helper(locs):\n    return locs.capture()\n",
    "/w/user/kern.py": (
        "def inner(locs):\n    return helper(locs)\n"
        "def outer(locs):\n    return inner(locs)\n"
        "def deep(locs, n):\n    return deep(locs, n - 1) if n else inner(locs)\n"
    ),
    "/w/user/main.py": (
        "def main(locs):\n"
        "    ab = [outer(locs) for _ in range(2)]\n"
        "    c = inner(locs)\n"
        "    d = deep(locs, 6)\n"
        "    e = outer(locs)\n"
        "    return ab, c, d, e\n"
    ),
}


def _scripted_locs(table):
    """loc.json's call tree run against `table`: every named Loc."""
    space: dict = {}
    for path, source in _SOURCES.items():
        exec(compile(source, path, "exec"), space)
    (a, b), c, d, e = space["main"](table)
    return {
        "a": a,
        "b": b,
        "c": c,
        "deep": d,
        "e": e,
        "name": table.name("peel copy 1", a),
        "name_no_inner": table.name("hazard"),
        "fuse": table.fuse(a, c, what="merged wait"),
        "fuse_dups": table.fuse(0, a, a),
        "fuse_unknowns": table.fuse(0, 0),
        "fuse_single": table.fuse(a, what=""),
        "fuse_two_no_what": table.fuse(c, a),
        "unknown": 0,
    }


@pytest.mark.parametrize(PARAMS, whole("ir", "loc.json"))
def test_loc(vector, case_id, case, doc, monkeypatch):
    table = ir.LocTable(lambda code: not code.co_filename.startswith("/w/user/"))
    monkeypatch.chdir("/")
    locs = _scripted_locs(table)
    for name, want in case["locs"].items():
        loc = locs[name]
        assert loc == want["loc"], name
        assert table.lines(loc) == want["lines"], name
        assert table.short(loc) == want["short"], name
    assert len(table) == case["len_after"]
    monkeypatch.setattr(ir, "LOCS", table)
    a = locs["a"]
    rendered = {
        "error_with_notes": ir.Diag("reads v5 while its load may be in flight", a)
        .note(a, "the load")
        .note(0, "unknown note"),
        "named": ir.Diag("named", locs["name"]),
        "unknown": ir.Diag("no location"),
    }
    for name, want in case["renders"].items():
        assert str(rendered[name]) == want, name


# --- passes.json ------------------------------------------------------------------


class _Unit:
    def __init__(self, program):
        self.program = program
        self.pass_times: dict[str, float] = {}


@pytest.mark.parametrize(PARAMS, whole("ir", "passes.json"))
def test_pass_runner(vector, case_id, case, doc, t, monkeypatch):
    made, calls = [], []
    make = ir.program

    def snapshot(unit, ins=None):
        made.append(make(None, list(unit.program.body)))
        return made[-1]

    monkeypatch.setattr(ir, "program", snapshot)

    def transform(name):
        def go(unit):
            calls.append([name, None])
            unit.program.body.append(stream.comment(f"after {name} {len(calls)}", on=t))

        return ir.Pass(name, go)

    def check(name, when=ir.always):
        def go(unit, snap):
            calls.append([name, next(k for k, s in enumerate(made) if s is snap)])

        return ir.Pass(name, go, check=True, when=when)

    passes = []
    for step in case["pipeline"]:
        name = step.split()[0]
        if "(transform)" in step:
            passes.append(transform(name))
        else:
            when = (lambda unit: False) if "when false" in step else ir.always
            passes.append(check(name, when=when))
    assert [p.name for p in passes] == [x.split()[0] for x in case["pipeline"]]
    unit = _Unit(ir.Program(t))
    returned = ir.run(unit, passes)
    assert calls == case["calls"]
    assert len(made) == case["programs_made"]
    assert returned is case["returned"]
    assert sorted(unit.pass_times) == case["pass_times_keys"]


# --- stream.json ------------------------------------------------------------------


class _Unplaced:
    """A register file before placement: every spelling is by name."""

    def text(self, r):
        return ir.name_text(r, 1)

    name_text = text


def _notes(i):
    """An instruction as stream.json writes what made() put on it."""
    return {
        "op": i.desc.name,
        "fields": i.fields,
        "comment": facts.said(i),
        "tag": list(facts.tag(i)),
        "abl": sorted(a.classes) if (a := i.attr(ir.Ablation)) else [],
        "stage": list(facts.staged(i)),
        "loc": i.loc,
    }


@pytest.mark.parametrize(PARAMS, whole("ir", "stream.json"))
def test_stream(vector, case_id, case, doc, t):
    s1, s2 = ir.Stream(), ir.Stream()
    x = ir.Instr(t.form("s_endpgm", 0), ())
    s1.append(x)
    s1.append(x)
    s2.append(x)
    files = {f: _Unplaced() for f in "vs"}
    branch = ir.Instr(t.form("s_cbranch_scc1", 1), (ir.BlockRef(".Lend_3"),))
    with pytest.raises(TypeError) as unknown:
        stream.comment("x", nosuch=1)
    got = {
        "same_stream_twice_equal_positions": [s1[0] == s1[1], s1[0] == x],
        "in_s2": x in s2,
        "s2_copy_equal_to_x": s2[0] == x,
        "count_s1": s1.count(x),
        "rfind": s1.rfind(x),
        "rfind_before_1": s1.rfind(x, 1),
        "index": s1.index(x),
        "passed_label": stream.passed(stream.label(".Lloop_12"), files),
        "passed_branch": stream.passed(branch, files),
        "comment_notes": {
            "call": case["comment_notes"]["call"],
            "made": _notes(eval(case["comment_notes"]["call"], {"comment": stream.comment})),
        },
        "comment_unknown_keyword": type(unknown.value).__name__,
        "waits_notes": {
            "call": case["waits_notes"]["call"],
            "made": [
                _notes(i)
                for i in eval(case["waits_notes"]["call"], {"waits": stream.waits, "on": t})
            ],
        },
    }
    assert got == case


def test_loc_relpath_is_from_the_working_directory(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    table = ir.LocTable(lambda code: False)
    loc = table.capture()
    assert table.lines(loc)[0].startswith(os.path.relpath(__file__))


# --- Named registers and own ops per target --------------------------------------------------


def test_named_registers_are_whole_in_writes_and_reads():
    """A v_cmpx's implicit EXEC and an explicit exec_lo both read as `exec`."""
    from thin.target import PHYS, target

    t = target("gfx1201")
    cmpx = ir.Instr(t.form("v_cmpx_gt_u32", 2), ())
    mov = ir.Instr(t.form("s_mov_b32", 2), (PHYS["exec_lo"], ir.Imm(0)))
    assert cmpx.writes == mov.writes == frozenset({"exec"})
    assert "exec" in cmpx.phys_uses()


def test_wait_markers_are_the_named_target_s():
    from thin.target import target, targeting

    with targeting("gfx1201"):
        here = ir.wait("all", "load", ())
    assert here.desc is target("gfx1201").pseudo("ir.wait_all")
    assert ir.wait("all", "load", (), on=target("gfx1201")).desc is here.desc
    assert ir.is_wait(here)
    with pytest.raises(ValueError, match="'vmcnt' is not a counter of gfx1201"):
        ir.wait("all", "vmcnt", ())
    assert stream.label("L", on="gfx1201").desc is target("gfx1201").pseudo("ir.label")


def test_a_label_defined_twice_is_refused():
    t = target("gfx1201")
    p = _verify_program(t, "valid_branch_loop")
    p.body.extend((stream.label("L"), stream.label("L")))
    with pytest.raises(ir.Diag, match="label L is defined twice"):
        ir.verify(p)


def test_the_verifier_checks_dpp_controls_and_special_slots(t):
    p = _verify_program(t, "valid_branch_loop")
    a = p.values[0][0]
    dpp = t.form("v_mov_b32_dpp", 2)
    masks = {"row_mask": 15, "bank_mask": 15}
    p.body.append(ir.Instr(dpp, (a, a), {"dpp": "row_shr:16", **masks}))
    with pytest.raises(ir.Diag, match=r"dpp `row_shr:16`: not a DPP16 control"):
        ir.verify(p)
    p.body[-1] = ir.Instr(dpp, (a, a), {"dpp": "quad_perm:[3,2,1,0]", **masks})
    ir.verify(p)
    cndmask = next(f for f in t.forms() if f.name == "v_cndmask_b32" and len(f.slots) == 4)
    p.body[-1] = ir.Instr(cndmask, (a, a, a, PHYS["vcc_hi"]))
    with pytest.raises(ir.Diag, match="operand 3 is vcc_hi; slot OPR_VCC takes vcc_lo"):
        ir.verify(p)
    p.body[-1] = ir.Instr(cndmask, (a, a, a, PHYS["vcc_lo"]))
    ir.verify(p)
