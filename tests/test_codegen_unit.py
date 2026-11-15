"""Unit tests of the pipeline beyond the golden data."""

import json

import pytest

from tests import notation
from tests.golden import load
from tools.golden.codegen import STORE_SRC


def _kernel(name):
    return next(c for c in load("codegen", "kernels.json")["kernels"] if c["name"] == name)


def test_a_kernel_built_twice_gives_the_same_code_object():
    from thin import codegen

    case = _kernel("loop_filter_b64")
    first = codegen.code_object(notation.compile_case(case)[1].text)
    second = codegen.code_object(notation.compile_case(case)[1].text)
    assert first == second


def test_split_manifest_gives_back_the_manifest_written():
    from thin import codegen

    _, art = notation.compile_case(_kernel("scaled_merge_b128"))
    body, m = codegen.split_manifest(art.text)
    assert m == art.manifest
    assert body + m.line == art.text
    assert codegen.Manifest.of(json.dumps(m.record())) == m


def test_manifest_refuses_an_unknown_key():
    from thin import codegen

    with pytest.raises(ValueError, match="unknown key"):
        codegen.Manifest.of('{"symbol": "k", "colour": 1}')


def test_finish_peels_up_to_the_first_repeat():
    from thin import codegen

    b = notation.run_snippet(
        'n = S.u32(name="n", at=0)\nacc = V.u32(0, name="acc")\n'
        'for lp in b.loop(n, peel=1, name="p"):\n    acc += lp.i\n'
    )
    codegen.finish(b, nops=False)
    labels = [i.field("name") for i in b.ins if i.kind == "label"]
    assert labels.count(".Lp_1") == 1
    assert b.deferred == []
    assert list(b.pass_times) == ["drop_entry", "deferred", "waits"]


def test_a_peeled_body_with_no_steady_state_is_refused():
    from thin import codegen
    from thin.ir import ScheduleError

    b = notation.run_snippet(
        'n = S.u32(name="n", at=0)\nacc = V.u32(0, name="acc")\nk = [0]\n'
        'for lp in b.loop(n, peel=1, name="p"):\n    k[0] += 1\n    acc += k[0]\n'
    )
    with pytest.raises(
        ScheduleError, match="loop body p has no steady state after 1 peeled iterations"
    ):
        codegen.finish(b, nops=False)


def test_a_dead_exit_is_kept_or_commented():
    from thin import codegen

    b = notation.run_snippet(
        'x, y = V.u32(name="x", at=0), V.u32(name="y", at=1)\ns = S.u32(name="s", at=0)\n'
        'm = S.mask(name="m", at=4)\nwith b.lanes(s == 0):\n    y += 1\n'
        "with b.lanes(m, s == 1):\n    y += 2\n"
    )
    exits = len(b.deferred)
    codegen.finish(b)
    dead = [
        i
        for i in b.ins
        if i.kind == "comment" and "dead (exec set again" in (i.field("text") or "")
    ]
    kept = [r for r in b.program.regions if r.id == 2 and r.checked]
    assert exits == 1 and len(dead) + len(kept) == 1


@pytest.mark.parametrize("src,hazards", [(STORE_SRC, 0), (_kernel("scaled_merge_b128")["src"], 1)])
def test_placement_runs_pre_place_then_post_place(src, hazards):
    from thin import codegen

    b = codegen.derive(lambda: notation.run_snippet(src))
    assert list(b.pass_times) == ["drop_entry", "waits", "regalloc", "hazards"]
    assert b.hazards == hazards


def test_placing_twice_is_refused():
    from thin import codegen
    from thin.ir import AssignError

    b = codegen.derive(lambda: notation.run_snippet(_kernel("scaled_merge_b128")["src"]))
    with pytest.raises(AssignError, match="placed already"):
        codegen.place(b)


def test_using_refuses_an_unknown_option():
    from thin import codegen

    with pytest.raises(TypeError), codegen.using(colour=True):
        pass


def test_the_independent_checker_finds_nothing_in_a_compiled_kernel():
    """A compiled kernel's waits pass the independent split-counter check."""
    from thin import codegen

    _, art = notation.compile_case(_kernel("scaled_merge_b128"))
    assert codegen.counter_waits(art.text) == []


def test_the_audit_passes_a_kernel_whose_hazards_placed_an_s_delay_alu():
    """The audit leaves out what the hazard pass placed: the first run has none."""
    from thin import codegen

    src = _kernel("scalar_reuse_b64")["src"]
    with codegen.using(audit=True):
        b = codegen.derive(lambda: notation.run_snippet(src))
    assert any(i.desc.name == "s_delay_alu" for i in b.ins)
