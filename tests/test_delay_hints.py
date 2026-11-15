"""Recompute delay dependencies independently from rendered assembly."""

from __future__ import annotations

import glob
import json
import re

import pytest

from tests import notation
from tests.golden import data_path, load

TRANS = ("v_exp_", "v_log_", "v_rcp_", "v_rsq_", "v_sqrt_", "v_sin_", "v_cos_")
NO_DEST = ("s_cmp", "s_bitcmp", "s_cbranch", "s_branch", "s_nop", "s_wait", "s_endpgm",
           "s_barrier", "s_setpc", "s_sendmsg", "s_sleep", "s_setreg", "s_setprio", "s_trap",
           "s_icache", "s_denorm_mode", "s_round_mode", "s_delay_alu", "s_code_end",
           "s_ttrace", "s_dcache")  # fmt: skip
REG = re.compile(r"\b([vs])(?:(\d+)|\[(\d+):(\d+)\])|\b(vcc(?:_lo|_hi)?)\b")


def _regs(text: str) -> set:
    out = set()
    for m in REG.finditer(text):
        if m.group(5):
            name = m.group(5)
            out |= {("s", 106)} if name == "vcc_lo" else {("s", 107)} if name == "vcc_hi" else {
                ("s", 106), ("s", 107)}  # fmt: skip
        elif m.group(2) is not None:
            out.add((m.group(1), int(m.group(2))))
        else:
            out |= {(m.group(1), r) for r in range(int(m.group(3)), int(m.group(4)) + 1)}
    return out


def _operands(rest: str) -> list:
    return [o.strip() for o in rest.split(",")] if rest else []


def _code(line: str):
    """(op, operand text) of an instruction line, else None (labels, directives)."""
    text = line.split("//")[0].split(";")[0].strip()
    if not text or text.endswith(":") or text.startswith("."):
        return None
    op, _, rest = text.partition(" ")
    return op, rest.strip()


MERGES = ("v_writelane", "v_cvt_pk_fp8", "v_cvt_pk_bf8", "v_cvt_sr_fp8", "v_cvt_sr_bf8")
"""VALU ops that merge into their destination, so they read it too."""


def _valu(op: str) -> bool:
    return op.startswith("v_")


def _writes_reads(op: str, rest: str) -> tuple:
    ops = _operands(rest)
    if not ops:
        return set(), set()
    if _valu(op):
        if op.startswith("v_cmpx"):
            return set(), _regs(rest)
        n = 2 if ("_co_" in op or op.startswith("v_div_scale")) else 1
        # A DPP without bound_ctrl keeps the old value in lanes with no
        # source lane, so it reads its destination too.
        merges = op.startswith(MERGES) or (op.endswith("_dpp") and "bound_ctrl" not in rest)
        reads = _regs(",".join(ops[n:])) | (_regs(ops[0]) if merges else set())
        return _regs(",".join(ops[:n])), reads
    if op.startswith("s_") and not op.startswith(NO_DEST) and not op.startswith("s_store"):
        return _regs(ops[0]), _regs(",".join(ops[1:]))
    loads = ("_load", "_read", "atomic")
    if any(k in op for k in loads) and not op.startswith(("buffer_store", "global_store")):
        return _regs(ops[0]), _regs(",".join(ops[1:]))
    return set(), _regs(rest)


def stale_hints(text: str) -> list:
    """Every hint of the text naming a producer whose shared registers were overwritten."""
    raw = list(map(_code, text.splitlines()))  # None: a label, directive or blank
    labels = {
        k for k, line in enumerate(text.splitlines()) if line.split("//")[0].strip().endswith(":")
    }
    bad = []
    for at, c in enumerate(raw):
        if c is None or c[0] != "s_delay_alu":
            continue
        value = int(c[1].split()[0], 0)
        j = next((q for q in range(at + 1, len(raw))
                  if raw[q] is not None and raw[q][0] != "s_delay_alu"), None)  # fmt: skip
        if j is None:
            continue
        rop, rrest = raw[j]
        _, reads = _writes_reads(rop, rrest)
        for ident in (value & 0xF, (value >> 7) & 0xF):
            if not 1 <= ident <= 7:
                continue
            trans, n = ident >= 5, ident - 4 if ident >= 5 else ident
            seen, p = 0, None
            for q in range(j - 1, -1, -1):
                if q in labels:
                    break
                if raw[q] is None:
                    continue
                qop = raw[q][0]
                if qop.startswith(("s_branch", "s_cbranch")):
                    break
                if _valu(qop) and (not trans or qop.startswith(TRANS)):
                    seen += 1
                    if seen == n:
                        p = q
                        break
            if p is None:
                bad.append(f"{rop} {rrest}: no producer for id {ident}")
                continue
            shared = _writes_reads(*raw[p])[0] & reads
            for q in range(p + 1, j):
                if raw[q] is not None:
                    shared -= _writes_reads(*raw[q])[0]
            if not shared:
                bad.append(f"{rop} {rrest}: id {ident} names `{' '.join(raw[p])}`, overwritten")
    return bad


def _renders():
    out = [(f"kernels.json {c['name']}", c) for c in load("codegen", "kernels.json")["kernels"]]
    paths = [
        *sorted(glob.glob(str(data_path("dsl", "*.json")))),
        str(data_path("lib", "gfx1201.json")),
    ]
    for path in paths:
        with open(path) as f:
            doc = json.load(f)
        for k, c in enumerate(doc.get("cases", [])):
            r = c.get("render") if isinstance(c, dict) else None
            if r:
                out.append(
                    (f"{path.split('/')[-1]}[{k}]", "\n".join(r) if isinstance(r, list) else r)
                )
    return out


@pytest.mark.parametrize(("name", "what"), _renders(), ids=[n for n, _ in _renders()])
def test_no_hint_names_an_overwritten_producer(name, what):
    text = notation.compile_case(what)[1].text if isinstance(what, dict) else what
    assert stale_hints(text) == []


def test_the_checker_finds_the_stale_trans_hint():
    """The dependency-chain example requires: instid1(TRANS32_DEP_1) (641)."""
    text = "v_rcp_iflag_f32 v6, v6\nv_mul_f32 v6, v6, v7\ns_delay_alu 641\nv_cvt_u32_f32 v6, v6\n"
    assert len(stale_hints(text)) == 1
    assert stale_hints(text.replace("641", "1")) == []
