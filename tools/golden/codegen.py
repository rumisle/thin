"""Code-generation programs evaluated by thin's public and native APIs.

DSL, assembly, metadata, manifest and IR plans are constructed here. The
compiler, printer and native helper compute encodings and diagnostics.
"""

from __future__ import annotations

import builtins
import contextlib
import hashlib
import io
import json
import re
import subprocess

from tools.golden import ROOT, encode

BASE = "crates/thin-codegen/tests/data/codegen/"
OUTPUTS = tuple(
    BASE + name
    for name in (
        "kernels.json",
        "refusals.json",
        "verifier.json",
        "printer.json",
        "abi.json",
        "manifest.json",
        "metadata.json",
    )
)

KERNEL_NOTATION = (
    "Independently authored gfx1201 kernels. src runs with a fresh Builder, V, S "
    "and the public DSL names; derive places it and compile_kernel(symbol, "
    "wgsize=, max_grid=) produces s, the manifest, summary and pass timing keys. "
    "Placement seconds are masked as #.## s. code_object fields, including all "
    "digests, are computed from the native encoding and code-object reader."
)
REFUSAL_NOTATION = (
    "Authored public-API calls. build() runs src on a fresh gfx1201 Builder. "
    "Each call either evaluates ok_of to ok or records the actual exception class "
    "and message; placement seconds are masked as #.## s."
)
VERIFIER_NOTATION = (
    "Hand-authored split-counter assembly. counter_waits and its native records "
    "are computed by the independent checker and sorted as it returns them."
)
PRINTER_NOTATION = (
    "Authored IR instruction operands printed through thin.ir.stream. Values use "
    "the printer test's pinned bases. before_placement is computed before the "
    "register files are placed; all placed forms and annotations are computed "
    "afterwards."
)
ABI_NOTATION = (
    "gfx1201 argument declarations are evaluated with the public DSL. Residency "
    "and budget outputs are computed by the native ABI from the explicit numeric "
    "inputs. Argument refusals record the actual exception."
)
MANIFEST_NOTATION = (
    "manifest_line output and split_manifest results computed by thin-codegen "
    "from independently authored records and assembly text."
)
METADATA_NOTATION = (
    "YAML subsets and assembly inputs authored here; metadata documents and "
    "AsmError diagnostics are computed by thin-codegen's metadata reader."
)


# ---------------------------------------------------------------------------
# End-to-end kernels


KERNEL_PLANS = [
    dict(
        name="scaled_merge_b128",
        target="predicated f32 data flow with scalar-argument and two VMEM waits",
        symbol="authored_scaled_merge",
        wgsize=128,
        max_grid=[32768],
        src="""k = b.entry(Args(left=ptr, right=ptr, result=ptr,
                      count=u32.within(2, 1 << 24)), wg=1, wgsize=128)
left, right, result, count = k.load()
pos = V.u32(k.tid + k.wgid * 128, name="pos")
with b.lanes(pos < count):
    byte_at = V.u32(pos * 4, name="byte_at")
    x, y = V.f32(name="x"), V.f32(name="y")
    b.emit(O.global_load(x, byte_at, left))
    b.emit(O.global_load(y, byte_at, right))
    b.wait(x, y, "both input vectors ready")
    mixed = V.f32(x * 1.25 + y, name="mixed")
    b.emit(O.global_store(byte_at, mixed, result))
""",
    ),
    dict(
        name="scalar_reuse_b64",
        target="scalar destination rewrites after VALU reads, delay and ALU waits",
        symbol="authored_scalar_reuse",
        wgsize=64,
        max_grid=[8192],
        src="""k = b.entry(Args(out=ptr, count=u32.within(3, 1 << 19)), wg=1, wgsize=64)
out, count = k.load()
lane = V.u32(k.wgid * 64 + k.tid, name="lane")
a = S.u32(count * 13 + 2, name="a")
c = S.u32(count * 5 + 7, name="c")
d = S.u32(count * 17 + 1, name="d")
e = S.u32(count * 3 + 11, name="e")
first = V.u32(lane + a, name="first")
second = V.u32(first + c, name="second")
third = V.u32(second ^ d, name="third")
acc = V.u32(third + e, name="acc")
with b.lanes(lane < count):
    where = V.u32(lane * 4, name="where")
    b.emit(O.global_store(where, acc, out))
""",
    ),
    dict(
        name="guarded_permute_b64",
        target="v_cmpx to permlane16 dependency and its separating move",
        symbol="authored_guarded_permute",
        wgsize=64,
        max_grid=[4096],
        src="""k = b.entry(Args(out=ptr, count=u32.within(2, 1 << 18)), wg=1, wgsize=64)
out, count = k.load()
lane = V.u32(k.wgid * 64 + k.tid, name="lane")
signal = V.u32(lane ^ 5, name="signal")
with b.lanes(lane < count):
    where = V.u32(lane * 4, name="where")
    control = S.u32(0x60517243, name="control")
    shuffled = V.u32(name="shuffled")
    with b.raw_exec("stop a lane immediately before the shuffle"):
        b.emit(O.op("v_cmpx_ne_u32", 0, signal))
    b.emit(O.op("v.permlane16.b32", shuffled, signal, control, control))
    stored = V.u32(shuffled + lane, name="stored")
    b.emit(O.global_store(where, stored, out))
""",
    ),
    dict(
        name="matrix_recurrence_b32",
        target="back-to-back WMMA accumulator dependency, nop and neg_lo rendering",
        symbol="authored_matrix_recurrence",
        wgsize=32,
        max_grid=[1],
        src="""k = b.entry(Args(lhs=ptr, rhs=ptr, out=ptr), wg=1, wgsize=32)
lhs, rhs, out = k.load()
lane = V.u32(k.tid, name="lane")
address = V.u32(lane * 16, name="address")
x, y = V.u32(n=4, name="x"), V.u32(n=4, name="y")
b.emit(O.op("global_load_b128", x, address, lhs))
b.emit(O.op("global_load_b128", y, address, rhs, offset=16))
b.wait(x, y, "matrix fragments ready")
first, second = V.f32(n=8, name="first"), V.f32(n=8, name="second")
b.emit(O.op("v_wmma_f32_16x16x16_f16", first, x, y, 0))
b.emit(O.op("v_wmma_f32_16x16x16_f16", second, first[0:4], y, first, neg_lo=(0, 0, 1)))
out_address = V.u32(lane * 64, name="out_address")
for j, part in enumerate((second[4:8], first[4:8], second[0:4], first[0:4])):
    b.emit(O.op("global_store_b128", out_address, part, out, offset=16 * j or None))
""",
    ),
    dict(
        name="loop_filter_b64",
        target="counted loop, in-body load wait, backward branch and delay notes",
        symbol="authored_loop_filter",
        wgsize=64,
        max_grid=[8192],
        src="""k = b.entry(Args(src=ptr, dst=ptr, rounds=u32.within(2, 48)), wg=1, wgsize=64)
src, dst, rounds = k.load()
lane = V.u32(k.wgid * 64 + k.tid, name="lane")
address = V.u32(lane * 4, name="address")
state = V.f32(name="state")
b.emit(O.op("v_mov_b32", state, 0x3F800000))
for _ in b.loop(rounds, name="filter"):
    sample = V.f32(name="sample")
    b.emit(O.global_load(sample, address, src))
    b.wait(sample, "sample arrived")
    state[...] = (state - sample) * 0.5 + sample
b.emit(O.global_store(address, state, dst))
""",
    ),
    dict(
        name="lds_exchange_b96",
        target="LDS table, barrier, per-wave address proof and race checking",
        symbol="authored_lds_exchange",
        wgsize=96,
        max_grid=[4096],
        src="""k = b.entry(Args(out=ptr), wg=1, wgsize=96)
(out,) = k.load()
layout = b.lds()
slots = layout.region("slots", (96,), "u32")
value = V.u32((k.tid ^ 0x55) + 17, name="value")
b.emit(O.ds_write(slots.at(k.tid), value))
b.wait(slots)
b.barrier("all slots published")
received = V.u32(name="received")
b.emit(O.ds_read(received, slots.at((k.tid + 48) % 96)))
b.wait(received)
where = V.u32((k.wgid * 96 + k.tid) * 4, name="where")
result = V.u32(received + k.tid, name="result")
b.emit(O.global_store(where, result, out))
""",
    ),
    dict(
        name="plane_store_b32",
        target="two workgroup dimensions, uniform/assume claims and comment output",
        symbol="authored_plane_store",
        wgsize=32,
        max_grid=[32, 16],
        src="""k = b.entry(Args(out=ptr, width=u32.within(1, 32)), wg=2, wgsize=32)
out, width = k.load()
b.uniform(width, why="one launch width for every wave")
column = S.u32(k.wgid * width + 1, name="column")
row = S.u32(k.wgid_y * 32, name="row")
b.comment("form a plane index from both workgroup coordinates")
base = S.u32(row + column, name="base")
index = V.u32(base * 32 + k.tid, name="index")
b.assume(index, lo=1, hi=(1 << 16) - 1, why="bounded authored grid")
where = V.u32(index * 4, name="where")
payload = V.u32(index ^ width, name="payload")
b.emit(O.global_store(where, payload, out))
""",
    ),
]


STORE_SRC = """k = b.entry(Args(out=ptr), wg=1, wgsize=32)
(out,) = k.load()
slot = V.u32(k.tid * 4, name="slot")
payload = V.u32(k.tid + 9, name="payload")
b.emit(O.global_store(slot, payload, out))
"""

LDS_SRC = """k = b.entry(Args(out=ptr), wg=1, wgsize=96)
(out,) = k.load()
layout = b.lds()
values = layout.region("values", (96,), "u32")
value = V.u32(k.tid + 3, name="value")
b.emit(O.ds_write(values.at(k.tid), value))
b.wait(values)
"""

BUDGET_SRC = """k = b.entry(Args(out=ptr), wg=1, wgsize=1024)
(out,) = k.load()
terms = [V.u32(k.tid ^ (j * 37 + 5), name=f"term{j}") for j in range(196)]
total = V.u32(terms[0] * 3 + terms[1], name="total")
for term in terms[2:]:
    total[...] = total ^ term
where = V.u32(k.tid * 4, name="where")
b.emit(O.global_store(where, total, out))
"""


REFUSAL_PLANS = [
    (
        "bad_leading_digit",
        STORE_SRC,
        'art = compile_kernel(derive(build), "7authored", wgsize=32, max_grid=(1,))',
        None,
        "refusal",
    ),
    (
        "bad_leading_dollar",
        STORE_SRC,
        'art = compile_kernel(derive(build), "$authored", wgsize=32, max_grid=(1,))',
        None,
        "refusal",
    ),
    (
        "different_args",
        STORE_SRC,
        'art = compile_kernel(derive(build), "authored_k", wgsize=32,\n'
        "    args=Args(out=ptr), max_grid=(1,))",
        None,
        "refusal",
    ),
    (
        "entry_args_reused",
        STORE_SRC,
        "placed = derive(build)\n"
        'art = compile_kernel(placed, "authored_k", wgsize=32,\n'
        "    args=placed.entered.args, max_grid=(1,))",
        "art.manifest.kernarg_bytes",
        "success",
    ),
    (
        "wrong_lds_size",
        LDS_SRC,
        'art = compile_kernel(derive(build), "authored_lds", wgsize=96, lds=768, max_grid=(1,))',
        None,
        "refusal",
    ),
    (
        "layout_lds_size",
        LDS_SRC,
        'art = compile_kernel(derive(build), "authored_lds", wgsize=96, lds=384, max_grid=(1,))',
        "art.manifest.lds",
        "success",
    ),
    (
        "large_workgroup_budget",
        BUDGET_SRC,
        'art = compile_kernel(derive(build), "authored_heavy", wgsize=1024, max_grid=(1,))',
        None,
        "refusal",
    ),
    (
        "audit_regular_waits",
        KERNEL_PLANS[0]["src"],
        "with using(audit=True):\n"
        '    art = compile_kernel(derive(build), "authored_scaled_merge", wgsize=128,\n'
        "        max_grid=(32768,))",
        '"s_wait_loadcnt 0" in art.text',
        "success",
    ),
    (
        "audit_alu_hazards",
        KERNEL_PLANS[1]["src"],
        "with using(audit=True):\n"
        '    art = compile_kernel(derive(build), "authored_scalar_reuse", wgsize=64,\n'
        "        max_grid=(8192,))",
        '"s_wait_alu" in art.text',
        "success",
    ),
    (
        "occupied_pin",
        """k = b.entry(Args(out=ptr), wg=1, wgsize=32)
(out,) = k.load()
address = V.u32(k.tid * 12, name="address", at=3)
data = V.u32((k.tid + 1) * 12, name="data", at=3)
b.emit(O.global_store(address, data, out))
""",
        'art = compile_kernel(derive(build), "authored_pin", wgsize=32, max_grid=(1,))',
        None,
        "refusal",
    ),
    (
        "pin_beyond_file",
        """k = b.entry(Args(out=ptr), wg=1, wgsize=32)
(out,) = k.load()
wide = V.u32(n=4, name="wide", at=254)
for j in range(4):
    wide[j][...] = k.tid + j * 11
where = V.u32(k.tid * 4, name="where")
b.emit(O.op("global_store_b128", where, wide, out))
""",
        'art = compile_kernel(derive(build), "authored_wide_pin", wgsize=32, max_grid=(1,))',
        None,
        "refusal",
    ),
    (
        "same_as_moves_pin",
        """k = b.entry(Args(out=ptr), wg=1, wgsize=32)
(out,) = k.load()
source = V.u32((k.tid ^ 3) * 4, name="source")
alias = V.u32(name="alias", at=7, same_as=source)
b.emit(O.global_store(source, alias, out))
""",
        'art = compile_kernel(derive(build), "authored_alias", wgsize=32, max_grid=(1,))',
        None,
        "refusal",
    ),
]


# ---------------------------------------------------------------------------
# Independent assembly wait checker


def _asm(name, target, lines, refused=True):
    return dict(name=name, target=target, text="\n".join(lines) + "\n", refused=refused)


VERIFIER_PLANS = [
    _asm(
        "unwaited_vector_result",
        "loadcnt read before retirement",
        ["authored:", "\tglobal_load_b32 v7, v0, s[8:9]", "\tv_add_nc_u32 v8, v7, 1", "\ts_endpgm"],
    ),
    _asm(
        "retired_vector_result",
        "a zero load wait retires the result",
        [
            "authored:",
            "\tglobal_load_b32 v7, v0, s[8:9]",
            "\ts_wait_loadcnt 0",
            "\tv_xor_b32 v8, v7, v0",
            "\ts_endpgm",
        ],
        refused=False,
    ),
    _asm(
        "older_of_two_loads",
        "ordered loadcnt one retirement",
        [
            "authored:",
            "\tglobal_load_b32 v10, v0, s[8:9]",
            "\tglobal_load_b32 v11, v0, s[8:9] offset:12",
            "\ts_wait_loadcnt 1",
            "\tv_or_b32 v12, v10, v0",
            "\tv_sub_u32 v13, v11, v12",
            "\ts_endpgm",
        ],
    ),
    _asm(
        "overwrite_pending_load",
        "destination overwrite is also a read hazard",
        [
            "authored:",
            "\tglobal_load_b32 v12, v1, s[8:9]",
            "\tglobal_load_b32 v12, v1, s[8:9] offset:20",
            "\ts_wait_loadcnt 0",
            "\ts_endpgm",
        ],
    ),
    _asm(
        "lds_counter_independent",
        "loadcnt does not retire dscnt",
        [
            "authored:",
            "\tds_load_b32 v4, v0",
            "\tglobal_load_b32 v9, v0, s[8:9]",
            "\ts_wait_loadcnt 0",
            "\tv_xor_b32 v10, v4, v9",
            "\ts_endpgm",
        ],
    ),
    _asm(
        "lds_store_orders_wait",
        "a younger DS store makes dscnt one sufficient",
        [
            "authored:",
            "\tds_load_b32 v6, v0",
            "\tds_store_b32 v0, v11",
            "\ts_wait_dscnt 1",
            "\tv_or_b32 v7, v6, v0",
            "\ts_endpgm",
        ],
        refused=False,
    ),
    _asm(
        "kmcnt_is_not_ordered_count",
        "only zero retires scalar loads",
        [
            "authored:",
            "\ts_load_b32 s20, s[0:1], 0x8",
            "\ts_load_b32 s21, s[0:1], 0xc",
            "\ts_wait_kmcnt 1",
            "\ts_or_b32 s22, s20, 1",
            "\ts_wait_kmcnt 0",
            "\ts_add_co_u32 s23, s21, s22",
            "\ts_endpgm",
        ],
    ),
    _asm(
        "atomic_return_load_counter",
        "returning atomics use loadcnt, others use storecnt",
        [
            "authored:",
            "\tglobal_atomic_add_u32 v13, v0, v2, s[8:9] th:TH_ATOMIC_RETURN",
            "\tglobal_atomic_add_u32 v0, v4, s[8:9]",
            "\tv_xor_b32 v14, v13, v2",
            "\ts_endpgm",
        ],
    ),
    _asm(
        "join_with_new_pending_load",
        "control-flow merge preserves the post-join pending load",
        [
            "authored:",
            "\tglobal_load_b32 v8, v0, s[8:9]",
            "\ts_wait_loadcnt 0",
            "\ts_cbranch_scc1 .Lmerge",
            "\tglobal_load_b32 v9, v0, s[8:9] offset:8",
            "\ts_wait_loadcnt 0",
            ".Lmerge:",
            "\tglobal_load_b32 v9, v0, s[8:9] offset:16",
            "\tv_sub_u32 v10, v9, v8",
            "\ts_endpgm",
        ],
    ),
    _asm(
        "pending_store_data",
        "a store is a use of its pending data operand",
        [
            "authored:",
            "\tglobal_load_b64 v[16:17], v0, s[8:9]",
            "\tglobal_store_b64 v0, v[16:17], s[10:11]",
            "\ts_endpgm",
        ],
    ),
    _asm(
        "one_branch_has_wait",
        "conditional wait leaves one path pending",
        [
            "authored:",
            "\tglobal_load_b32 v18, v0, s[8:9]",
            "\ts_cbranch_scc1 .Lafter",
            "\ts_wait_loadcnt 0x0",
            ".Lafter:",
            "\tv_or_b32 v19, v18, v0",
            "\ts_endpgm",
        ],
    ),
    _asm(
        "backedge_pending_load",
        "loop state includes previous iteration's load",
        [
            ".Lrepeat:",
            "\tv_xor_b32 v20, v21, v0",
            "\tglobal_load_b32 v21, v0, s[8:9] offset:24",
            "\ts_cbranch_scc1 .Lrepeat",
            "\ts_endpgm",
        ],
    ),
    _asm(
        "dead_unwaited_read",
        "unconditional control flow skips the unsafe read",
        [
            "authored:",
            "\tglobal_load_b32 v22, v0, s[8:9]",
            "\ts_branch .Ldone",
            "\tv_add_nc_u32 v23, v22, 1",
            ".Ldone:",
            "\ts_endpgm",
        ],
        refused=False,
    ),
    _asm(
        "wide_destination_member",
        "range expansion tracks every loaded dword",
        [
            "authored:",
            "\tglobal_load_b128 v[12:15], v0, s[8:9]",
            "\tv_or_b32 v16, v14, v0",
            "\ts_endpgm",
        ],
    ),
    _asm(
        "per_wave_control_classes",
        "skip annotations define classes with safe reads",
        [
            "authored:",
            "\tglobal_load_b32 v3, v0, s[8:9]",
            "\tglobal_load_b32 v4, v0, s[8:9] offset:4",
            "\ts_cbranch_execz .Lfinished                 // by_wave skip 0,2,4 of 6",
            "\ts_wait_loadcnt 0x0",
            "\tv_add_nc_u32 v5, v3, v4",
            ".Lfinished:",
            "\ts_endpgm",
        ],
        refused=False,
    ),
    _asm(
        "scalar_load_overwrite",
        "overwriting scalar ranges is checked on kmcnt",
        [
            "authored:",
            "\ts_load_b64 s[10:11], s[0:1], 0x10",
            "\ts_load_b64 s[10:11], s[0:1], 0x18",
            "\ts_wait_kmcnt 0x0",
            "\ts_endpgm",
        ],
    ),
    _asm(
        "footer_text_ignored",
        "checking ends at the rodata section",
        [
            "authored:",
            "\ts_endpgm",
            "\t.section\t.rodata",
            "\ts_load_b32 s9, s[0:1], 0x0",
            "\ts_xor_b32 s10, s9, s9",
        ],
        refused=False,
    ),
    _asm(
        "comments_do_not_touch_registers",
        "register words in comments are ignored",
        ["authored:", "\tglobal_load_b32 v24, v0, s[8:9] // v24 and s99 are prose", "\ts_endpgm"],
        refused=False,
    ),
    _asm(
        "returned_message_pending",
        "sendmsg return registers use kmcnt",
        [
            "authored:",
            "\ts_sendmsg_rtn_b64 s[14:15], 137",
            "\ts_xor_b32 s16, s15, 1",
            "\ts_wait_kmcnt 0x0",
            "\ts_add_co_u32 s17, s14, s16",
            "\ts_endpgm",
        ],
    ),
]


# ---------------------------------------------------------------------------
# Printer instruction constructors


def _r(reg, file="v", index=0, value_n=1, lo=0, n=1):
    return dict(reg=reg, file=file, index=index, value_n=value_n, lo=lo, n=n)


def _i(imm, text=""):
    return dict(imm={"float": repr(imm)} if isinstance(imm, float) else imm, text=text)


def _p(form, ops=(), fields=None, mods=None, comment="", attrs=()):
    return dict(
        form=form, ops=list(ops), fields=fields or {}, mods=mods, comment=comment, attrs=list(attrs)
    )


PRINTER_SPECS = [
    (
        "integer_e32_comment",
        _p("v_add_nc_u32", [_r("y#1"), _r("x#1"), _i(5)], comment="y = x + five"),
    ),
    (
        "e64_sub_modifiers",
        _p("v_sub_f32_e64", [_r("x#1"), _r("y#1"), _i(0.375)], {"clamp": True}, [1, 2, 0]),
    ),
    (
        "false_flag_omitted",
        _p("v_mul_f32_e64", [_r("y#1"), _r("x#1"), _i(-2.5)], {"clamp": False}),
    ),
    (
        "dpp_default_bound",
        _p(
            "v_mov_b32_dpp",
            [_r("x#1"), _r("y#1")],
            {"dpp": "row_shr:3", "row_mask": 12, "bank_mask": 5, "bound_ctrl": False},
        ),
    ),
    (
        "dpp_explicit_bound",
        _p(
            "v_mov_b32_dpp",
            [_r("y#1"), _r("x#1")],
            {"dpp": "row_ror:7", "row_mask": 9, "bank_mask": 6, "bound_ctrl": True},
        ),
    ),
    (
        "store_off_b64",
        _p("global_store_b64", [_r("x#1"), _r("z#1", value_n=4, n=2), {"phys": "off"}]),
    ),
    (
        "scalar_quad_offset",
        _p(
            "s_load_b128",
            [_r("d#1", "s", value_n=4, n=4), _r("p#1", "s", value_n=2, n=2), _i(32, "0x20")],
        ),
    ),
    ("ds_wait_field", _p("s_wait_dscnt", fields={"dscnt": 2})),
    (
        "packed_selectors",
        _p(
            "v_pk_mul_f16",
            [_r("y#1"), _r("x#1"), _r("y#1")],
            {"op_sel": {"tuple": [0, 1]}, "op_sel_hi": {"tuple": [1, 1]}, "neg_lo": None},
        ),
    ),
    (
        "hex_immediate",
        _p("v_mov_b32", [_r("x#1"), _i(32, "0x20")]),
    ),
    (
        "negative_immediate",
        _p("v_mov_b32", [_r("y#1"), _i(-7)]),
    ),
    (
        "large_integer",
        _p("v_mov_b32", [_r("x#1"), _i(1 << 33)]),
    ),
    (
        "large_exponent_float",
        _p("v_mov_b32", [_r("y#1"), _i(3.25e12)]),
    ),
    (
        "small_exponent_float",
        _p("v_mov_b32", [_r("x#1"), _i(2e-7)]),
    ),
    (
        "signed_zero",
        _p("v_mov_b32", [_r("y#1"), _i(-0.0)]),
    ),
    (
        "rounded_float_repr",
        _p("v_mov_b32", [_r("x#1"), _i(98765432.0)]),
    ),
    (
        "negative_infinity",
        _p("v_mov_b32", [_r("y#1"), _i(float("-inf"))]),
    ),
    (
        "ordinary_float",
        _p("v_mov_b32", [_r("x#1"), _i(-4.5)]),
    ),
    (
        "wide_value_member",
        _p("v_mov_b32", [_r("z#1", value_n=4, lo=1), _r("y#1")]),
    ),
    (
        "wide_pair_slices",
        _p(
            "v_lshlrev_b64",
            [_r("z#1", value_n=4, lo=0, n=2), _i(3), _r("z#1", value_n=4, lo=2, n=2)],
        ),
    ),
    (
        "group_last_member",
        _p("v_mov_b32", [_r("q", index=2), _r("q", index=0)]),
    ),
    (
        "physical_vcc_exec",
        _p("s_mov_b32", [{"phys": "vcc_lo"}, {"phys": "exec_lo"}]),
    ),
    ("m0_from_placed_sgpr", _p("s_mov_b32", [{"phys": "m0"}, _r("p#1", "s", value_n=2)])),
    ("authored_branch_label", _p("s_cbranch_scc0", [{"block": ".Lloop_37"}])),
    ("authored_label", _p("ir.label", fields={"name": ".Lloop_37"})),
    (
        "authored_comment",
        _p("ir.comment", fields={"text": "; independent marker"}),
    ),
    (
        "named_alu_counts",
        _p("s_wait_alu", fields={"depctr_va_vdst": 1, "depctr_sa_sdst": 0}),
    ),
    (
        "load_wait_comment",
        _p("s_wait_loadcnt", fields={"loadcnt": 5}, comment="fragment ready"),
    ),
    (
        "operand_wait_count",
        _p("s_wait_loadcnt", [_i(7, "0x7")]),
    ),
    (
        "uniform_reason",
        _p(
            "ir.uniform",
            [_r("p#1", "s", value_n=2, n=2), {"phys": "scc"}],
            {"why": "launch-wide control"},
        ),
    ),
    (
        "assume_lane_range",
        _p("ir.assume", [_r("y#1")], {"lo": 2, "hi": 30, "why": "compact lane id"}),
    ),
    (
        "undefined_middle_pair",
        _p("ir.undefined", [_r("z#1", value_n=4, lo=1, n=2)], {"why": "consumed pair"}),
    ),
    (
        "discard_other_register",
        _p("ir.discard", [_r("w#1")], {"why": "unused tail"}),
    ),
    (
        "global_cache_fields",
        _p(
            "global_load_b32",
            [_r("w#1"), _r("x#1"), _r("p#1", "s", value_n=2, n=2)],
            {"offset": 48, "th": "TH_LOAD_BYPASS", "scope": "SCOPE_SYS"},
        ),
    ),
    (
        "ds_places_order",
        _p(
            "ds_load_b32",
            [_r("w2#1"), _r("x#1")],
            {"offset": 12},
            attrs=["Places(names=('tile', 'left'))"],
        ),
    ),
]


# ---------------------------------------------------------------------------
# ABI declarations and numeric requests


ARG_PLANS = [
    (
        "two_buffers_and_counts",
        "Args(first=ptr, second=ptr, count=u32, checksum=u32)",
    ),
    (
        "scalar_padding_before_pointer",
        "Args(tag=u32, buffer=ptr, scale=f32)",
    ),
    (
        "all_kinds_reordered",
        "Args(buffer=ptr, bits=u64, real=f32, signed=i32, count=u32)",
    ),
    ("signed_range", "Args(height=i32.within(-12, 240))"),
    (
        "divisor_layout",
        "Args(blocks=u32, quantum=u32.within(3, 1 << 12).divisor())",
    ),
    ("stride_multiple", "Args(stride=u32.within(6, 90).multiple_of(6))"),
    (
        "owned_memory_mix",
        "Args(workspace=scratch, persistent=kept.uncached, rollback=carried)",
    ),
    (
        "rank_shape_declaration",
        "Args(segments=u32, world=mapped(3), fallback=ptr)",
    ),
    ("five_peers", "Args(team=peers(5), cutoff=u32)"),
    (
        "library_state_first",
        "Args(state=lib_arg(kept, 'trace'), source=ptr)",
    ),
    (
        "explicit_sparse_offsets",
        "Args(at={'marker': 48, 'payload': 80}, first=ptr, marker=f32, payload=u64)",
    ),
    ("misaligned_explicit_pointer", "Args(at={'vector': 12}, vector=ptr)"),
    ("overlap_explicit_scalar", "Args(at={'stop': 6}, base=ptr, stop=u32)"),
    ("unknown_explicit_name", "Args(at={'missing': 24}, base=ptr)"),
    ("fractional_range_refused", "Args(scale=f32.within(2, 16))"),
    ("empty_declaration", "Args()"),
]

ARG_REFUSALS = {
    "misaligned_explicit_pointer",
    "overlap_explicit_scalar",
    "unknown_explicit_name",
    "fractional_range_refused",
}

RESIDENCY_INPUTS = [
    dict(lds=0, vgpr=1, agpr=0, wgsize=32),
    dict(lds=0, vgpr=32, agpr=8, wgsize=64),
    dict(lds=4096, vgpr=16, agpr=0, wgsize=128),
    dict(lds=16384, vgpr=80, agpr=16, wgsize=256),
    dict(lds=65536, vgpr=0, agpr=0, wgsize=64),
    dict(lds=32768, vgpr=192, agpr=0, wgsize=512),
    dict(lds=1024, vgpr=256, agpr=0, wgsize=32),
    dict(lds=0, vgpr=0, agpr=0, wgsize=1024),
    dict(lds=8192, vgpr=48, agpr=48, wgsize=256),
    dict(lds=131072, vgpr=24, agpr=0, wgsize=128),
    dict(lds=0, vgpr=256, agpr=0, wgsize=1024),
]

BUDGET_INPUTS = [
    dict(lds=0, vgpr=256, agpr=0, sgpr=106, wgsize=64),
    dict(lds=0, vgpr=0, agpr=257, sgpr=8, wgsize=32),
    dict(lds=0, vgpr=176, agpr=24, sgpr=12, wgsize=1024),
    dict(lds=0, vgpr=160, agpr=16, sgpr=12, wgsize=256),
    dict(lds=0, vgpr=12, agpr=0, sgpr=108, wgsize=128),
    dict(lds=66000, vgpr=12, agpr=0, sgpr=12, wgsize=64),
    dict(lds=69632, vgpr=264, agpr=16, sgpr=120, wgsize=1024),
    dict(lds=128, vgpr=8, agpr=8, sgpr=16, wgsize=32),
]


# ---------------------------------------------------------------------------
# Metadata documents and assembly


def _yaml(name, *lines):
    return dict(name=name, yaml=list(lines))


METADATA_PLANS = [
    _yaml(
        "complete_nested_document",
        "amdhsa.kernels:",
        "  - .args:",
        "      - .offset: 8",
        "        .size: 40",
        "        .value_kind: by_value",
        "    .language: HIP",
        "    .language_version:",
        "      - 3",
        "      - 1",
        "    .max_flat_workgroup_size: 128",
        "    .name: authored_kernel",
        "    .uses_dynamic_stack: true",
        "    .wavefront_size: 64",
        "amdhsa.target: custom-target",
        "amdhsa.version:",
        "  - 2",
        "  - 1",
    ),
    _yaml("negative_decimal", "delta: -73"),
    _yaml("zero_in_nested_map", "limits:", "  count: 0"),
    _yaml("padded_decimal_refused", "count: 00091"),
    _yaml("boolean_pair", "enabled: true", "  ", "visible: false"),
    _yaml("capital_false_refused", "flag: False"),
    _yaml("capital_yes_refused", "enabled: YES"),
    _yaml("flow_sequence_refused", "values: [4, 8]"),
    _yaml("single_quoted_refused", "mode: 'fast'"),
    _yaml(
        "plain_string_punctuation",
        "target: rdna-4 wave 1.2",
    ),
    _yaml("duplicate_nested_key", "outer:", "  value: 7", "  value: 9"),
    _yaml("nested_key_without_value", "root:", "  child:"),
    _yaml("unkeyed_line_refused", "not-a-key"),
    _yaml("blank_and_whitespace_lines", "alpha: 4", "", "   ", "beta: 8"),
    _yaml(
        "nested_scalar_map",
        "config:",
        "  width: 64",
        "  height: 16",
        "  label: tile-A",
    ),
    _yaml(
        "same_indent_scalar_list",
        "samples:",
        "- 7",
        "- -2",
        "- 0",
    ),
    _yaml(
        "dedent_left_over",
        "root:",
        "    nested: 1",
        "  orphan: 2",
    ),
    _yaml(
        "three_list_maps",
        "items:",
        "  - index: 1",
        "    label: first",
        "  - index: 2",
        "    label: second",
        "  - index: 3",
    ),
    dict(
        name="wavefront_type_refused",
        yaml=["amdhsa.kernels:", "  - .wavefront_size: wide64"],
    ),
    dict(
        name="language_value_refused",
        yaml=["amdhsa.kernels:", "  - .language: Rust"],
    ),
    dict(
        name="dynamic_stack_type_refused",
        yaml=["amdhsa.kernels:", "  - .uses_dynamic_stack: 1"],
    ),
    _yaml(
        "kernel_keys_ignored_elsewhere",
        ".uses_dynamic_stack: 0",
        ".language: Rust",
    ),
    dict(
        name="assembly_without_metadata",
        s="\t.text\nauthored_empty:\n\ts_mov_b32 s0, 0\n\ts_endpgm\n",
    ),
    dict(
        name="assembly_without_document_start",
        s=(
            "\t.text\nauthored:\n\ts_endpgm\n\t.amdgpu_metadata\n"
            "value: 9\n...\n\t.end_amdgpu_metadata\n"
        ),
    ),
    dict(
        name="assembly_without_metadata_end",
        s="\t.text\nauthored:\n\t.amdgpu_metadata\n---\nvalue: 9\n...\n",
    ),
]

METADATA_REFUSALS = {
    "padded_decimal_refused",
    "capital_false_refused",
    "capital_yes_refused",
    "flow_sequence_refused",
    "single_quoted_refused",
    "duplicate_nested_key",
    "nested_key_without_value",
    "unkeyed_line_refused",
    "dedent_left_over",
    "wavefront_type_refused",
    "language_value_refused",
    "dynamic_stack_type_refused",
    "assembly_without_metadata",
    "assembly_without_document_start",
    "assembly_without_metadata_end",
}

MANIFEST_FULL = {
    "symbol": "authored_manifest",
    "kernarg_bytes": 32,
    "lds": 1024,
    "wgsize": 128,
    "wgy": 2,
    "vgpr": 11,
    "agpr": 3,
    "sgpr": 18,
    "args": [
        {"name": "buffer", "offset": 0, "bytes": 8, "kind": "ptr"},
        {
            "name": "factor",
            "offset": 8,
            "bytes": 16,
            "kind": "u32",
            "lo": 2,
            "hi": 127,
            "divisor": True,
            "multiple": 3,
            "owned": "scratch",
            "library": "sampler",
            "mem": "uncached",
            "ranks": 3,
        },
    ],
    "build": {"recipe": {"rows": 8}},
    "resident": 2,
    "mix": {"valu": 7, "vmem": 2},
    "notes": ["authored diagnostic"],
    "grid": [8, 2, 1],
    "phases": {
        "v": [
            {
                "at": "v_xor_b32 v:a, v:b, v:c",
                "base": 4,
                "peak": 9,
                "phase": "(derived)",
                "size": None,
                "used": 12,
            }
        ]
    },
    "max_grid": [32, 4],
    "needs": {"co_resident": "paired consumer"},
}
MANIFEST_LINE_PLANS = [
    (
        "sparse_record_defaults",
        {"symbol": "authored_ping", "wgsize": 128, "vgpr": 5, "sgpr": 7},
    ),
    ("complete_record", MANIFEST_FULL),
    (
        "unicode_note",
        {
            "symbol": "authored_unicode",
            "wgsize": 32,
            "vgpr": 2,
            "sgpr": 4,
            "notes": ["latency µs → tile"],
        },
    ),
]


# ---------------------------------------------------------------------------
# Evaluation helpers


def native(requests):
    """Evaluate requests in thin-codegen's owned, fixture-independent example."""
    result = subprocess.run(
        ["cargo", "run", "-q", "-p", "thin-codegen", "--example", "golden_authored"],
        cwd=ROOT,
        input=json.dumps(requests),
        text=True,
        capture_output=True,
        check=True,
        timeout=120,
    )
    return json.loads(result.stdout)


def _masked(text):
    return re.sub(r"\d+\.\d\d s", "#.## s", text)


def kernels_document():
    """Compile and encode each authored kernel with the complete pipeline."""
    from tests import notation
    from thin import codegen, formats

    kernels = []
    for plan in KERNEL_PLANS:
        case = {"src": plan["src"]}
        b, artifact = notation.compile_case(
            {
                **case,
                "symbol": plan["symbol"],
                "wgsize": plan["wgsize"],
                "max_grid": plan["max_grid"],
            }
        )
        image = codegen.code_object(artifact.text)
        obj = formats.read(image)
        record = artifact.manifest.record()
        manifest = {
            k: v for k, v in record.items() if k not in ("phases", "max_grid", "needs") or v
        }
        kernels.append(
            {
                "name": plan["name"],
                "what": plan["target"],
                "symbol": plan["symbol"],
                "wgsize": plan["wgsize"],
                "max_grid": plan["max_grid"],
                "wg": b.entered.wg,
                "src": plan["src"],
                "args": artifact.manifest.args,
                "s": artifact.text,
                "manifest_line": artifact.manifest.line.removesuffix("\n"),
                "manifest": manifest,
                "summary": _masked(artifact.summary),
                "pass_times_keys": list(b.pass_times),
                "code_object": {
                    "sha256": hashlib.sha256(image).hexdigest(),
                    "bytes": len(image),
                    "text_hex": obj.text.hex(),
                    "descriptor_hex": obj.descriptor.hex(),
                    "e_flags": obj.e_flags,
                    "digests": dict(formats.code(image).digests),
                },
            }
        )
        if codegen.counter_waits(artifact.text):
            raise AssertionError(f"{plan['name']}: compiler emitted an unsafe wait schedule")
    return {"notation": KERNEL_NOTATION, "kernels": kernels}


def refusals_document():
    """Run authored calls and record their observed success or refusal."""
    from tests import notation
    from thin import codegen, dsl

    cases = []
    for name, src, call, ok_of, expected in REFUSAL_PLANS:
        case = {"name": name, "src": src, "call": call}
        scope = {attr: getattr(dsl, attr) for attr in dsl.__all__}
        scope.update(using=codegen.using, build=lambda src=src: notation.run_snippet(src))
        try:
            exec(call, scope)
        except Exception as error:
            if ok_of is not None:
                raise
            case.update(refused=type(error).__name__, message=_masked(str(error)))
            outcome = "refusal"
        else:
            if ok_of is None:
                raise AssertionError(f"{name}: unexpected success")
            case.update(ok_of=ok_of, ok=eval(ok_of, scope))
            outcome = "success"
        if outcome != expected:
            raise AssertionError(f"{name}: {outcome}, planned {expected}")
        cases.append(case)
    return {"notation": REFUSAL_NOTATION, "cases": cases}


def verifier_document():
    """Run the native independent checker over every assembly input."""
    results = native([{"op": "verifier", "text": plan["text"]} for plan in VERIFIER_PLANS])
    cases = []
    for plan, result in zip(VERIFIER_PLANS, results, strict=True):
        refused = bool(result["counter_waits"])
        if refused != plan["refused"]:
            raise AssertionError(f"{plan['name']}: wait findings do not match its planned class")
        cases.append({"name": plan["name"], "text": plan["text"], **result})
    return {"notation": VERIFIER_NOTATION, "cases": cases}


class _PrintFile:
    """One printer register file; placement switches names to physical text."""

    def __init__(self, groups):
        from thin import ir

        self._ir = ir
        self.groups = groups
        self.placed = False

    def name_text(self, reg):
        """Construct name text inputs."""
        return self._ir.name_text(reg, self.groups[reg.value.name])

    def text(self, reg):
        """Construct text inputs."""
        if not self.placed:
            return self.name_text(reg)
        return self._ir.placed_text(reg, PRINTER_BASES[reg.value.name] + reg.value.index)


PRINTER_BASES = {
    "x#1": 0,
    "y#1": 1,
    "z#1": 4,
    "q": 8,
    "p#1": 4,
    "d#1": 8,
    "s1#1": 1,
    "w#1": 20,
    "w2#1": 21,
}


def _print_case(spec):
    from thin import ir
    from thin.ir import stream
    from thin.target import PHYS, target

    values, groups = {}, {"v": {}, "s": {}}

    def operand(data):
        """Construct operand inputs."""
        if "reg" in data:
            key = data["file"], data["reg"], data["index"]
            if key not in values:
                values[key] = ir.Value(
                    len(values),
                    data["file"],
                    data["value_n"],
                    1,
                    ir.B32,
                    data["reg"],
                    0,
                    data["index"],
                )
            groups[data["file"]][data["reg"]] = 3 if data["reg"] == "q" else 1
            return ir.VReg(values[key], data["lo"], data["n"])
        if "phys" in data:
            return PHYS[data["phys"]]
        if "block" in data:
            return ir.BlockRef(data["block"])
        value = data["imm"]
        if isinstance(value, dict):
            value = float(value["float"])
        return ir.Imm(value, data["text"])

    ops = [operand(data) for data in spec["ops"]]
    t = target("gfx1201")
    form = (
        t.pseudo(spec["form"]) if spec["form"].startswith("ir.") else t.form(spec["form"], len(ops))
    )
    fields = {k: tuple(v["tuple"]) if isinstance(v, dict) else v for k, v in spec["fields"].items()}
    mods = None if spec["mods"] is None else bytes(spec["mods"])
    instruction = ir.Instr(form, ops, fields, mods=mods)
    instruction.put(ir.Comment(spec["comment"]))
    for attr in spec["attrs"]:
        instruction.put(eval(attr, {"Places": ir.Places}))
    files = {name: _PrintFile(value) for name, value in groups.items()}
    before = {
        "text": instruction.text,
        "bare": stream.bare(instruction, files),
        "named": stream.named(instruction, files),
    }
    for file in files.values():
        file.placed = True
    placed = {
        "bare": stream.bare(instruction, files),
        "named": stream.named(instruction, files),
        "render": stream.render(instruction, files),
        "passed": stream.passed(instruction, files),
        "annotation": ir.annotation(instruction, lambda r: files[r.file].text(r)),
    }
    return before, placed


def printer_document():
    """Compute every printer case before and after physical placement."""
    cases = []
    for name, spec in PRINTER_SPECS:
        before, placed = _print_case(spec)
        cases.append({"name": name, **spec, "before_placement": before, "placed": placed})
    return {"notation": PRINTER_NOTATION, "cases": cases}


def _abi_arg(plan):
    from thin import dsl
    from thin.target import targeting

    name, src = plan
    case = {"name": name, "src": src}
    scope = {attr: getattr(dsl, attr) for attr in dsl.__all__}
    try:
        with targeting("gfx1201"):
            block = eval(src, scope)
        case["ok"] = {"size": block.size, "args": [dict(block[n]) for n in block.names()]}
    except Exception as error:
        case.update(refused=type(error).__name__, message=str(error))
    return case


def abi_document():
    """Evaluate declarations and native residency/budget requests."""
    requests = [{"op": "residency", **item} for item in RESIDENCY_INPUTS]
    requests += [{"op": "budget", "name": "k", **item} for item in BUDGET_INPUTS]
    results = native(requests)
    split = len(RESIDENCY_INPUTS)
    residency = [
        {**item, **result} for item, result in zip(RESIDENCY_INPUTS, results[:split], strict=True)
    ]
    budget = []
    for item, result in zip(BUDGET_INPUTS, results[split:], strict=True):
        case = dict(item)
        if result["message"] is None:
            case["ok"] = None
        else:
            case.update(refused="ScheduleError", message=result["message"])
        budget.append(case)
    document = {
        "notation": ABI_NOTATION,
        "residency": residency,
        "budget": budget,
        "args": [_abi_arg(plan) for plan in ARG_PLANS],
    }
    for case in document["args"]:
        expected = case["name"] in ARG_REFUSALS
        if ("ok" not in case) != expected:
            raise AssertionError(f"{case['name']}: unexpected argument outcome")
    return document


def manifest_document():
    """Serialize records, then split texts containing the fresh lines."""
    line_results = native(
        [{"op": "manifest_line", "record": record} for _name, record in MANIFEST_LINE_PLANS]
    )
    lines = [
        {"name": name, "manifest": result["ok"]}
        for (name, _record), result in zip(MANIFEST_LINE_PLANS, line_results, strict=True)
    ]
    sparse, _full, unicode = (item["manifest"] for item in lines)
    split_plans = [
        (
            "final_line_after_body",
            "\tv_xor_b32 v0, v0, v0\n\t.end_amdgpu_metadata\n" + sparse,
        ),
        (
            "extra_trailing_newlines",
            "\ts_nop 0\n\ts_endpgm\n" + sparse + "\n\n",
        ),
        ("plain_assembly", "\ts_endpgm\n"),
        (
            "manifest_before_instruction",
            sparse + "\tv_add_nc_u32 v1, v0, 1\n",
        ),
        ("single_manifest_line", unicode),
    ]
    split_results = native([{"op": "manifest_split", "text": text} for _n, text in split_plans])
    splits = [
        {"name": name, "text": text, **result["ok"]}
        for (name, text), result in zip(split_plans, split_results, strict=True)
    ]
    return {"notation": MANIFEST_NOTATION, "lines": lines, "split": splits}


def metadata_document():
    """Parse every YAML or assembly input with the native metadata reader."""
    requests = []
    for plan in METADATA_PLANS:
        requests.append(
            {"op": "metadata_yaml", "yaml": plan["yaml"]}
            if "yaml" in plan
            else {"op": "metadata_s", "s": plan["s"]}
        )
    results = native(requests)
    cases = []
    for plan, result in zip(METADATA_PLANS, results, strict=True):
        case = {k: v for k, v in plan.items() if k in ("name", "yaml", "s")}
        if "ok" in result:
            case["ok"] = result["ok"]
        else:
            case.update(refused="AsmError", message=result["message"])
        expected = plan["name"] in METADATA_REFUSALS
        if ("ok" not in case) != expected:
            raise AssertionError(f"{plan['name']}: unexpected metadata outcome")
        cases.append(case)
    return {"notation": METADATA_NOTATION, "cases": cases}


# ---------------------------------------------------------------------------


@contextlib.contextmanager
def no_test_data():
    """Fail generation if Python code tries to open any checked-in test data."""
    real_open, real_io_open = builtins.open, io.open

    def guard(opener):
        """Construct guard inputs."""

        def checked(file, *args, **kwargs):
            """Construct checked inputs."""
            if "tests/data" in str(file).replace("\\", "/"):
                raise PermissionError(f"codegen generator opened test data: {file}")
            return opener(file, *args, **kwargs)

        return checked

    builtins.open, io.open = guard(real_open), guard(real_io_open)
    try:
        yield
    finally:
        builtins.open, io.open = real_open, real_io_open


def generate():
    """Construct every program and compute all outputs without reading fixtures."""
    docs = {
        "kernels.json": kernels_document,
        "refusals.json": refusals_document,
        "verifier.json": verifier_document,
        "printer.json": printer_document,
        "abi.json": abi_document,
        "manifest.json": manifest_document,
        "metadata.json": metadata_document,
    }
    previous = None
    import thin

    previous = thin.use_store(None)
    try:
        with no_test_data():
            return {BASE + name: encode(make()) for name, make in docs.items()}
    finally:
        thin.use_store(previous)
