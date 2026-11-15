"""Executable Python IR inputs for model, location, pass-runner and stream data.

All recorded answers are produced by the public/native API.  Literal strings in
this module describe calls or construct inputs; none are expected diagnostics
copied from a golden file.
"""

from __future__ import annotations

import os
from contextlib import contextmanager

from thin import ir
from thin.ir import facts, stream
from thin.target import target

# ------------------------------------------------------------- model.json


def _program(t):
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


def _round_trips(t):
    # Expression text is the input consumed by the existing test as well.
    expressions = (
        "False",
        "7",
        "-170141183460469231731687303715884105728",
        "'cache'",
        "()",
        "(1, 1, 0)",
        "('dev', 'lu')",
        "(('noorder',), ('dma', 7, 2))",
        "(3, 'region')",
        "None",
        "1.25",
        "[9]",
    )
    out = []
    for text in expressions:
        given = eval(text)
        i = ir.Instr(t.form("v_add_nc_u32", 3), (), {"offset": given})
        back = i.field("offset")
        out.append(
            {
                "given": text,
                "back": repr(back),
                "same_type": type(back) is type(given),
                "identical": back is given,
            }
        )
    return out


def _attributes(t):
    puts = (
        "Comment(text='authored note')",
        "Comment(text='')",
        "Ablation(classes=frozenset({'schedule', 'memory'}))",
        "Ablation(classes=frozenset())",
        "AliasOk(pairs=frozenset({('left', 'right'), ('single', None)}))",
        "Tag(value=('load', 4, 2))",
        "Tag(value=('unordered',))",
        "Tag(value=())",
        "InRegion(id=11)",
        "Hazard()",
        "M0Set(value=19)",
        "WavesTaken(waves=frozenset({1, 4}))",
        "WavesTaken(waves=frozenset())",
        "Staged(stage=('load', 6))",
        "WaitUses((p.values[1][0],))",
    )
    names = {name: getattr(ir, name) for name in ir.__all__ if name[0].isupper()}
    out = []
    for put in puts:
        p = _program(t)
        attr = eval(put, {**names, "p": p})
        i = ir.Instr(t.form("v_add_nc_u32", 3), ())
        i.put(attr)
        after = i.attr(type(attr))
        # repr(frozenset) follows the process hash seed. Serialize the actual
        # returned members in a canonical constructor expression, without
        # changing their type or omitting any value.
        if isinstance(after, ir.AliasOk):
            shown = f"AliasOk(pairs=frozenset({sorted(after.pairs, key=repr)!r}))"
        elif isinstance(after, ir.Ablation):
            shown = f"Ablation(classes=frozenset({sorted(after.classes)!r}))"
        elif isinstance(after, ir.WavesTaken):
            shown = f"WavesTaken(waves=frozenset({sorted(after.waves)!r}))"
        else:
            shown = repr(after)
        out.append({"put": put, "attr_after": shown})
    return out


def _attempt(call):
    try:
        call()
    except BaseException as error:
        return {"error": type(error).__name__, "message": error.args[0]}
    raise AssertionError("authored invalid API call unexpectedly succeeded")


def _model_errors(t):
    form = t.form("v_add_nc_u32", 3)
    calls = {
        "WavesTaken wave 64": lambda: ir.Instr(form, ()).put(ir.WavesTaken(frozenset({64}))),
        "unknown field name": lambda: ir.Instr(form, (), {"nosuch": 1}),
        "operand of another type": lambda: ir.Instr(form, (5,)),
        "attr of a non-attribute class": lambda: ir.Instr(form, ()).attr(int),
    }
    return [dict(case=name, **_attempt(call)) for name, call in calls.items()]


def _reprs(t):
    p = _program(t)
    a, pair, s2 = p.values[0], p.values[2], p.values[4]
    return {
        "value": repr(a),
        "value_s2": repr(s2),
        "vreg_whole": repr(pair.whole),
        "vreg_dword1": repr(pair.whole.dword(1)),
        "vreg_slice": repr(pair[1:2]),
        "vreg_index": repr(pair[1]),
        "through": pair.whole.through,
        "through_part": ir.VReg(pair, 0, 1, "lo").through,
        "vreg_eq_ignores_part": ir.VReg(pair, 0, 1, "lo") == pair[0],
        "instr": repr(ir.Instr(t.form("v_add_nc_u32", 3), (a[0], a[0], ir.Imm(4)))),
        "dwords": list(pair.whole.dwords()),
    }


def _instruction_facts(t):
    p = _program(t)
    a, b, s = p.values[0], p.values[1], p.values[3]
    i = ir.Instr(t.form("v_add_nc_u32", 3), (a[0], a[0], b[0]))
    claim = ir.Instr(t.pseudo("ir.uniform"), (s[0],), {"why": "uniform input"})
    return {
        "defs": repr(i.defs),
        "uses": repr(i.uses),
        "regs": repr(i.regs),
        "writes": sorted(i.writes),
        "phys_uses": sorted(i.phys_uses()),
        "kind": i.kind,
        "op": i.op,
        "where_unknown": i.where,
        "target_none": i.target,
        "pseudo_regs": repr(claim.regs),
    }


def model_data():
    """Evaluate model inputs against value, field, attribute and instruction APIs."""
    t = target("gfx1201")
    return {
        "target": t.name,
        "values": {
            v.name: f"{v.file}, {v.n} dword" + ("s" if v.n != 1 else "") for v in _program(t).values
        },
        "field_round_trip": _round_trips(t),
        "attributes": _attributes(t),
        "errors": _model_errors(t),
        "reprs": _reprs(t),
        "instr_facts": _instruction_facts(t),
    }


# --------------------------------------------------------------- loc.json


def _location_sources():
    """A source-level call tree exercising capture, interning, naming and fusion.

    Stable file/function names identify stack frames in the location records.
    """
    return {
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


@contextmanager
def _working_directory(path):
    before = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(before)


def loc_data():
    """Execute the authored call tree and render its native-independent locations."""
    table = ir.LocTable(lambda code: not code.co_filename.startswith("/w/user/"))
    namespace = {}
    for path, source in _location_sources().items():
        exec(compile(source, path, "exec"), namespace)
    with _working_directory("/"):
        (a, b), c, deep, e = namespace["main"](table)
        named = table.name("peel copy 1", a)
    locs = {
        "a": a,
        "b": b,
        "c": c,
        "deep": deep,
        "e": e,
        "name": named,
        "name_no_inner": table.name("hazard"),
        "fuse": table.fuse(a, c, what="merged wait"),
        "fuse_dups": table.fuse(0, a, a),
        "fuse_unknowns": table.fuse(0, 0),
        "fuse_single": table.fuse(a, what=""),
        "fuse_two_no_what": table.fuse(c, a),
        "unknown": 0,
    }
    with _working_directory("/"):
        loc_records = {
            name: {"loc": loc, "lines": table.lines(loc), "short": table.short(loc)}
            for name, loc in locs.items()
        }
        original = ir.LOCS
        ir.LOCS = table
        try:
            diagnostics = {
                "error_with_notes": ir.Diag("reads v5 while its load may be in flight", a)
                .note(a, "the load")
                .note(0, "unknown note"),
                "named": ir.Diag("named", named),
                "unknown": ir.Diag("no location"),
            }
            renders = {name: str(diag) for name, diag in diagnostics.items()}
        finally:
            ir.LOCS = original
    return {
        "rule": (
            "Executable helper, inner, outer and recursive deep functions are compiled from "
            "memory. Internal frames are those outside /w/user. main captures two shared "
            "locations, a direct location, six recursive frames (truncated to four), and one "
            "final location. The API then interns names and fused locations; all paths are "
            "relative to the working directory /."
        ),
        "calls": (
            "two outer captures; one inner; deep recursion; another outer; name with and "
            "without an inner location; fusions with text, duplicate ids, zero ids, one id "
            "and no text; diagnostics with notes, a name and no location"
        ),
        "locs": loc_records,
        "len_after": len(table),
        "renders": renders,
    }


# ------------------------------------------------------------ passes.json


class _Unit:
    def __init__(self, program):
        self.program = program
        self.pass_times = {}


def passes_data():
    """Run an authored transform/check pipeline and observe its snapshots."""
    t = target("gfx1201")
    unit = _Unit(ir.Program(t))
    calls = []
    snapshots = []

    def transform(name):
        """Construct transform inputs."""

        def apply(unit):
            """Construct apply inputs."""
            calls.append([name, None])
            # An actual source-level transform: each invocation adds a comment.
            unit.program.body.append(stream.comment(f"after {name} {len(calls)}", on=t))

        return ir.Pass(name, apply)

    def check(name, when=ir.always):
        """Construct check inputs."""

        def inspect(unit, snap):
            """Construct inspect inputs."""
            if not any(snap is known for known in snapshots):
                snapshots.append(snap)
            k = next(k for k, known in enumerate(snapshots) if snap is known)
            calls.append([name, k])

        return ir.Pass(name, inspect, check=True, when=when)

    passes = [
        transform("stage"),
        check("entry"),
        check("entry_again"),
        check("suppressed", when=lambda unit: False),
        transform("fold"),
        check("before_spill"),
        check("after_spill"),
        transform("stage"),
    ]
    # The public runner expects a kernel unit. Adapt this free Program only at
    # snapshot construction; Snapshot itself remains the native API result.
    real_program = ir.program

    def free_snapshot(unit, ins=None):
        """Construct free snapshot inputs."""
        return real_program(None, list(unit.program.body) if ins is None else ins)

    ir.program = free_snapshot
    try:
        returned = ir.run(unit, passes)
    finally:
        ir.program = real_program
    return {
        "notation": (
            "An executable Pass pipeline. calls records transforms as [name, null] and checks "
            "as [name, snapshot index]. programs_made counts distinct snapshots observed by "
            "checks; returned is 'a program' or null; pass_times_keys are measured timing keys."
        ),
        "pipeline": [
            f"{p.name} ({'check' if p.check else 'transform'}"
            f"{', when false' if p.name == 'suppressed' else ''})"
            for p in passes
        ],
        "calls": calls,
        "programs_made": len(snapshots),
        "returned": "a program" if returned is not None else None,
        "pass_times_keys": sorted(unit.pass_times),
    }


# ------------------------------------------------------------ stream.json


class _Unplaced:
    def text(self, register):
        """Construct text inputs."""
        return ir.name_text(register, 1)

    name_text = text


def _notes(i):
    return {
        "op": i.desc.name,
        "fields": i.fields,
        "comment": facts.said(i),
        "tag": list(facts.tag(i)),
        "abl": sorted(a.classes) if (a := i.attr(ir.Ablation)) else [],
        "stage": list(facts.staged(i)),
        "loc": i.loc,
    }


def stream_data():
    """Exercise row identity, rendering, comments, waits and keyword refusal."""
    t = target("gfx1201")
    first, second = ir.Stream(), ir.Stream()
    ending = ir.Instr(t.form("s_endpgm", 0), ())
    first.append(ending)
    first.append(ending)
    second.append(ending)
    files = {file: _Unplaced() for file in "vs"}
    branch = ir.Instr(t.form("s_cbranch_scc1", 1), (ir.BlockRef(".Lend_3"),))
    try:
        stream.comment("invalid", unsupported=True)
    except TypeError as error:
        unknown_keyword = type(error).__name__
    else:
        raise AssertionError("unknown stream.comment keyword unexpectedly accepted")
    comment_call = (
        'comment("tile ready", comment="local copy completed", tag=("tile", 3), '
        'abl=("warmup", "probe"), stage=("mov", 17), loc=11)'
    )
    wait_call = (
        'waits(on, {"load": 0, "store": 1, "ds": 3, "km": 0}, comment="all engines", loc=13)'
    )
    return {
        "same_stream_twice_equal_positions": [first[0] == first[1], first[0] == ending],
        "in_s2": ending in second,
        "s2_copy_equal_to_x": second[0] == ending,
        "count_s1": first.count(ending),
        "rfind": first.rfind(ending),
        "rfind_before_1": first.rfind(ending, 1),
        "index": first.index(ending),
        "passed_label": stream.passed(stream.label(".Lloop_73"), files),
        "passed_branch": stream.passed(branch, files),
        "comment_notes": {
            "call": comment_call,
            "made": _notes(eval(comment_call, {"comment": stream.comment})),
        },
        "comment_unknown_keyword": unknown_keyword,
        "waits_notes": {
            "call": wait_call,
            "made": [_notes(i) for i in eval(wait_call, {"waits": stream.waits, "on": t})],
        },
    }
