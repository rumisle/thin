"""Serialize settled native Builder events into target-independent protocol tokens."""

from __future__ import annotations

import json

from tests import notation
from thin import ir

POLICIES = {
    ("scope", "SCOPE_DEV"): "coherent",
    ("scope", "SCOPE_SYS"): "system",
    ("th", "TH_ATOMIC_RETURN"): "return",
}
COUNTERS = {"load": "vm", "store": "vm", "ds": "lgkm", "km": "lgkm"}


def events(builder):
    """Read actual instruction fields and claims; no precomputed events are inputs."""
    result, previous_wait = [], None
    builder.settle()
    for instruction in builder.ins:
        name = instruction.desc.name
        attrs = instruction.attrs or {}
        region = attrs[ir.InRegion].id if ir.InRegion in attrs else None
        comment = attrs[ir.Comment].text if ir.Comment in attrs else ""
        operands = [repr(op) for op in instruction.ops]
        said = instruction.fields.get("said", "")
        if name in ("ir.wait_regs", "ir.wait_all"):
            key = (
                (name, notation.built_line(builder, instruction).split(" {")[0], said)
                if name == "ir.wait_regs"
                else ("all", said)
            )
            if key == previous_wait:
                continue
            previous_wait = key
            text = (
                f"wait regs [] said={json.dumps(said)}"
                if name == "ir.wait_regs"
                else f"wait all {COUNTERS[instruction.fields['counter']]} said={json.dumps(said)}"
            )
        else:
            previous_wait = None
            if name == "s_mov_b32" and operands[0] == "exec_lo":
                text = (
                    "lanes restore"
                    if operands[1] in ("-1", "Imm(value=-1, text='')")
                    else "lanes enter"
                )
                region = None
            elif name == "ir.label":
                text = "label .Lnative"
            elif name.startswith("s_cbranch_"):
                text = "branch-if .Lnative"
            elif name == "s_branch":
                text = "branch .Lnative"
            elif name == "s_sleep":
                text = f"sleep {instruction.ops[0].value}"
            elif name == "s_sendmsg_rtn_b64":
                text = "clock.real"
            elif name == "s_getreg_b32":
                if instruction.ops[1].value != (29 | (31 << 11)):
                    continue
                text = "clock.core"
            elif name == "s_add_co_u32" and comment.startswith("[spin]"):
                text = "spin native"
            elif name == "s_lshr_b32" and comment.startswith("xwg: the stale quads"):
                text = "mask.shr native"
            elif name.startswith("global_atomic"):
                text = instruction.desc.canon.split("+")[0]
                if len(instruction.ops) == 4:
                    text += " {policy=return}"
            elif name.startswith(("s_load", "global_")):
                text = instruction.desc.canon
                fields = [
                    (field, instruction.fields[field])
                    for field in ("th", "scope")
                    if field in instruction.fields
                ]
                if fields:
                    text += " {policy=" + POLICIES[fields[0]] + "}"
            else:
                # Complete own statements are retained for human inspection; protocol
                # checks independently select only observable synchronization events.
                text = "stmt " + notation.built_line(builder, instruction)
        if region is not None:
            text += f" @lanes{region}"
        result.append(text)
    return result
