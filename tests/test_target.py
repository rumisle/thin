"""Golden checks of Python target views against the core target data."""

import copy

import pytest

from tests.golden import cases, load

PARAMS = ("vector", "case_id", "case", "doc")
VIEW_QUERIES = {"spell", "returning", "form"}
VIEW_TARGET = {"name", "wave", "lds_max", "chip", "barrier", "hazards", "waits"}


def _view_cases():
    keep = []
    for param in cases("target", "target_facts.json"):
        section, key = param.values[1][:-1].split("[")
        if section in ("phys", "flags") or (section, key) in {
            *(("queries", q) for q in VIEW_QUERIES),
            *(("target", k) for k in VIEW_TARGET),
        }:
            keep.append(param)
    return keep


def _form_record(form) -> dict:
    from thin.target import Flags

    flags = Flags.names(form.flags)
    return {
        "id": form.id,
        "name": form.name,
        "op": form.mnemonic,
        "base": form.base,
        "canon": form.canon,
        "kind": form.kind,
        "counter": form.counter,
        "unordered": form.unordered,
        "flags": [] if flags == "NONE" else flags.split("|"),
        "encoding": form.encoding,
        "arity": form.arity,
        "returning": form.returning,
        "slots": [
            {
                "kind": s.kind,
                "dwords": s.dwords,
                "role": s.role,
                "type": s.name,
                "sgpr_widths": sorted(s.sgpr_widths),
                "modifiers": s.modifiers,
                "literal": s.literal,
                "widths": s.widths,
            }
            for s in form.slots
        ],
        "variadic": form.variadic,
        "fields": [
            {
                "name": f.name,
                "kind": f.kind,
                "bits": f.bits,
                "signed": f.signed,
                "required": f.required,
                "choices": f.choices,
            }
            for f in form.fields
        ],
        "implicit_defs": sorted(form.implicit_defs, key=_phys_number),
        "implicit_uses": sorted(form.implicit_uses, key=_phys_number),
        "mem_bytes": form.mem_bytes,
        "pseudo": form.pseudo or "None",
        "dpp": form.dpp,
        "dst_sel": form.dst_sel,
        "mem": form.mem,
        "wait_text": form.wait_text,
    }


def _phys_number(name: str) -> int:
    from thin.target import PHYS

    return list(PHYS).index(name)


@pytest.mark.parametrize(PARAMS, cases("target", "forms.json.gz"))
def test_form(vector, case_id, case, doc):
    from thin.target import target

    t = target("gfx1201")
    form = t.forms()[case["id"]] if case["id"] >= 0 else t.pseudo(case["name"])
    assert _form_record(form) == case


@pytest.mark.parametrize(PARAMS, _view_cases())
def test_target_fact(vector, case_id, case, doc):
    from thin.target import PHYS, flag_bits, target

    t = target("gfx1201")
    section, key = case_id[:-1].split("[")
    if section == "flags":
        assert list(case) == list(flag_bits()[int(key)])
    elif section == "phys":
        p = PHYS[case["name"]]
        want = case
        assert {"name": p.name, "dwords": p.dwords, "file": p.file, "whole": p.whole} == want
        assert copy.deepcopy(p) is p and repr(p) == p.name
    elif section == "target":
        _target_fact(t, key, case)
    elif key == "spell":
        for op, mnemonic in case:
            if mnemonic is None:
                with pytest.raises(ValueError, match="has no such op"):
                    t.spell(op)
            else:
                assert t.spell(op) == mnemonic
    elif key == "returning":
        assert [t.returning(name, th) for name, th, _ in case] == [r for *_, r in case]
    elif key == "form":
        for mnemonic, arity, returning, id_ in case:
            if id_ is None:
                with pytest.raises(KeyError):
                    t.form(mnemonic, arity, returning)
            else:
                assert t.form(mnemonic, arity, returning).id == id_


def _target_fact(t, key: str, case) -> None:
    if key == "chip":
        got = {
            "xcds": t.xcds,
            "cus": t.cus,
            "simds": t.simds,
            "lds": t.cu_lds,
            "vgprs": t.vgprs,
            "vgprs_per_lane": t.vgprs_per_lane,
            "sgprs": t.sgpr_max,
            "user_sgprs": t.user_sgprs,
            "preload": t.preload_max,
            "wave_slots": t.wave_slots,
            "wgids": t.wgids,
        }
    elif key == "barrier":
        got = [[m, ops] for m, ops in t.barrier]
    elif key == "hazards":
        got = {**case, "window": t.hazard_window}
    elif key == "waits":
        got = [
            {"mnemonic": m, "fields": f, "text": w["text"]}
            for (m, f), w in zip(t.waits, case, strict=True)
        ]
    else:
        got = getattr(t, key)
    assert got == case


def test_selection_and_errors():
    from thin.target import Flags, target, target_names, targeting

    assert target_names() == ["gfx1201"]
    assert target() is target("gfx1201")
    with targeting("gfx1201") as t:
        assert target() is t
    with pytest.raises(ValueError, match="not a target"):
        target("gfx9999")
    with pytest.raises(ValueError, match="not a target"), targeting("gfx9999"):
        pass
    t = target()
    with pytest.raises(KeyError, match="no form"):
        t.form("v_add_f32", 9)
    with pytest.raises(KeyError):
        t.pseudo("ir.nothing")
    assert t.form("v_add_f32", 3) is t.form("v_add_f32", 3)
    assert repr(t.form("v_add_f32", 3)) == "Desc(v_add_f32)"
    assert t.pseudo("ir.wait_regs").fields[1].choices == list(t.counters)
    assert t.counters == {"load": "loadcnt", "store": "storecnt", "ds": "dscnt", "km": "kmcnt"}
    assert Flags.names(Flags.VALU | Flags.DPP) == "VALU|DPP" and Flags.names(0) == "NONE"
    assert t.allocated_vgprs(25, 0) == 48


def test_the_vectors_name_one_target():
    assert load("target", "target_facts.json")["target"]["name"] == "gfx1201"
