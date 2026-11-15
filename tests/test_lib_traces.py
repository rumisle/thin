"""Golden checks of library hand-off protocol events on gfx1201."""

import re

import pytest

from tests import notation
from tests.golden import cases

PARAMS = ("vector", "case_id", "case", "doc")

#: gfx1201's cache fields for the abstract policies, as its hooks give them.
_POLICY = {
    ("scope", "SCOPE_DEV"): "coherent",
    ("scope", "SCOPE_SYS"): "system",
    ("th", "TH_ATOMIC_RETURN"): "return",
}
#: gfx1201's counters by the wait groups the traces name.
_GROUP = {"load": "vm", "store": "vm", "ds": "lgkm", "km": "lgkm"}
#: The core clock's low half, as gfx1201's clock hook reads it.
_CYCLES_LO = 29 | 31 << 11


def _vector_skeleton(events) -> list:
    out = []
    for e in events:
        m = re.search(r" @lanes(\d+)$", e)
        region = int(m.group(1)) if m else None
        e = e[: m.start()] if m else e
        if e.startswith("stmt "):
            continue
        if e.startswith("wait regs "):
            tok = ("wait", re.search(r'said="(.*?)"', e).group(1))
        elif e.startswith("wait all "):
            group, said = re.match(r'wait all (\w+) said="(.*?)"', e).groups()
            tok = ("wait all", group, said)
        elif e.startswith("lanes enter"):
            tok = ("lanes enter",)
        elif e == "lanes restore":
            tok = ("lanes restore",)
        elif e.startswith("label "):
            tok = ("label",)
        elif e.startswith("branch-if "):
            tok = ("branch-if",)
        elif e.startswith("branch "):
            tok = ("branch",)
        elif e.startswith("sleep "):
            tok = ("sleep", int(e.split()[1]))
        elif e.startswith(("spin ", "mask.shr ")):
            tok = (e.split()[0],)
        else:
            policy = re.search(r"\{policy=(\w+)", e)
            tok = ("mem", e.split()[0], policy.group(1) if policy else None)
        out.append((tok, region))
    return out


def _built_skeleton(b) -> list:
    from thin import ir

    out, last = [], None
    for i in b.ins:
        name = i.desc.name
        a = i.attrs or {}
        region = a[ir.InRegion].id if ir.InRegion in a else None
        if name == "ir.wait_regs":
            key = (notation.built_line(b, i).split(" {")[0], i.fields["said"])
            if key == last:
                continue
            last, tok = key, ("wait", i.fields["said"])
        elif name == "ir.wait_all":
            key = ("all", i.fields["said"])
            if key == last:
                continue
            last, tok = key, ("wait all", _GROUP[i.fields["counter"]], i.fields["said"])
        else:
            last = None
            ops = [repr(op) for op in i.ops]
            if name == "s_mov_b32" and ops and ops[0] == "exec_lo":
                tok = ("lanes restore",) if ops[1] in ("-1", "Imm(value=-1, text='')") else None
                tok = tok or ("lanes enter",)
                region = None
            elif name == "ir.label":
                tok = ("label",)
            elif name.startswith("s_cbranch_"):
                tok = ("branch-if",)
            elif name == "s_branch":
                tok = ("branch",)
            elif name == "s_sleep":
                tok = ("sleep", i.ops[0].value)
            elif name == "s_sendmsg_rtn_b64":
                tok = ("mem", "clock.real", None)
            elif name == "s_getreg_b32":
                if i.ops[1].value != _CYCLES_LO:
                    continue
                tok = ("mem", "clock.core", None)
            elif name == "s_add_co_u32" and _comment(i).startswith("[spin]"):
                tok = ("spin",)
            elif name == "s_lshr_b32" and _comment(i).startswith("xwg: the stale quads"):
                tok = ("mask.shr",)
            elif name.startswith("global_atomic"):
                returns = len(i.ops) == 4
                tok = ("mem", i.desc.canon.split("+")[0], "return" if returns else None)
            elif name.startswith(("s_load", "global_")):
                cache = [(k, i.fields[k]) for k in ("th", "scope") if k in i.fields]
                policy = _POLICY[cache[0]] if cache else None
                tok = ("mem", i.desc.canon, policy)
            else:
                continue
        out.append((tok, region))
    return out


def _comment(i) -> str:
    from thin import ir

    a = i.attrs or {}
    return a[ir.Comment].text if ir.Comment in a else ""


@pytest.mark.parametrize(PARAMS, cases("lib", "traces.json", "cases"))
def test_trace(vector, case_id, case, doc):
    if "error" in case:
        with pytest.raises(ValueError) as caught:
            notation.run_snippet(case["src"], library=True)
        assert str(caught.value) == case["error"]["message"]
        return
    b = notation.run_snippet(case["src"], library=True)
    b.settle()
    assert _built_skeleton(b) == _vector_skeleton(case["events"])
    assert b.needs == case.get("needs", {})
