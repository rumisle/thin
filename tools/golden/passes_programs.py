"""Independently authored programs for the compiler-pass golden generator.

Only input plans live here.  Results, diagnostics, placements, and count maps are
always obtained from ``thin-passes`` by :mod:`tools.golden.passes`.
"""

from __future__ import annotations

import random


def _case(name, input, **extra):
    return {"name": name, "input": list(input), **extra}


def _finish(lines):
    return [*lines, "s_endpgm"]


def _wait4(reg):
    """Requests in the source notation, including deliberately empty counters."""
    return [f"@wait {counter} regs {reg}" for counter in ("load", "store", "ds", "km")]


def _launch(args=(), wgsize=64, max_grid=(), **extra):
    return {
        "args": [list(a) for a in args],
        "wgsize": wgsize,
        "wgy": extra.pop("wgy", 0),
        "max_grid": list(max_grid),
        "preload": extra.pop("preload", 0),
        "tid": extra.pop("tid", None),
        **extra,
    }


def _arg(offset, bytes=4, lo=None, hi=None, divisor=False):
    return [offset, bytes, lo is not None, lo, hi, divisor]


def alias_plans():
    """Aliasing across linear paths, branches, loops, entry state and waivers."""
    pair = [("early", ["v", 12, 1, False]), ("late", ["v", 12, 1, False]), ("sink", ["v", 29, 1])]
    plans = [
        _case(
            "linear read observes the other logical value",
            ["v_xor_b32 %early, v21, 7", "v_or_b32 %sink, %late, 3", "s_endpgm"],
        ),
        _case(
            "rewrite makes the second logical value available",
            [
                "v_xor_b32 %early, v21, 7",
                "v_or_b32 %sink, %early, 3",
                "v_and_b32 %late, v22, 15",
                "v_add_nc_u32 %sink, %late, 9",
                "s_endpgm",
            ],
        ),
        _case(
            "conditional definition leaves a bypass path",
            [
                "v_mov_b32 %early, 17",
                "v_mov_b32 %sink, %early",
                "s_cmp_ge_u32 s31, 2",
                "s_cbranch_scc1 .Lauth_alias_merge",
                "v_sub_nc_u32 %late, 91, v20",
                ".Lauth_alias_merge:",
                "v_xor_b32 %sink, %late, 5",
                "s_endpgm",
            ],
        ),
        _case(
            "memory overwrite cannot retire an older logical reader",
            [
                "v_mov_b32 %early, 23",
                "global_load_b32 %late, v[44:45], off offset:12",
                "v_mul_lo_u32 %sink, %early, 3",
                "s_wait_loadcnt 0",
                "v_add_nc_u32 %sink, %late, 1",
                "s_endpgm",
            ],
        ),
        _case(
            "masked write read under its own alias is self consistent",
            [
                "v_mov_b32 %early, 29",
                "v_mov_b32 %sink, %early",
                "s_mov_b32 exec_lo, 15",
                "v_add_nc_u32 %late, v20, 4",
                "v_mov_b32 %sink, %late",
                "s_mov_b32 exec_lo, -1",
                "s_endpgm",
            ],
        ),
        _case(
            "whole read after a lane partial overwrite still detects clobber",
            [
                "v_mov_b32 %late, 31",
                "v_mov_b32 %sink, %late",
                "s_mov_b32 exec_lo, 7",
                "v_xor_b32 %early, v21, 11",
                "v_mov_b32 %sink, %early",
                "v_mov_b32 %late, 37",
                "s_mov_b32 exec_lo, -1",
                "v_or_b32 %sink, %late, 1",
                "s_endpgm",
            ],
        ),
        _case(
            "masked consumers intentionally focus on the newly written lanes",
            [
                "v_mov_b32 %late, 41",
                "v_mov_b32 %sink, %late",
                "s_mov_b32 exec_lo, 31",
                "v_sub_nc_u32 %early, v21, 2",
                "v_mov_b32 %sink, %early",
                "v_mov_b32 %late, 43",
                "v_xor_b32 %sink, %late, 1",
                "s_mov_b32 exec_lo, -1",
                "s_endpgm",
            ],
        ),
        _case(
            "fused accumulation has an implicit destination read",
            [
                "v_mov_b32 %early, 47",
                "v_mov_b32 %sink, %early",
                "v_fmac_f32 %late, v46, v47",
                "s_endpgm",
            ],
        ),
    ]
    for p in plans:
        p.update(groups=pair, entry=[])
    plans.extend(
        [
            _case(
                "hardware initialized vector entry and an unwritten peer alias",
                ["v_add_nc_u32 %sink, %lane_id, 5", "v_xor_b32 %sink, %derived, 9", "s_endpgm"],
                groups=[
                    ("lane_id", ["v", 0, 1]),
                    ("derived", ["v", 0, 1, False]),
                    ("sink", ["v", 33, 1]),
                ],
                entry=["lane_id"],
            ),
            _case(
                "scalar aliases follow the same provenance rules",
                ["s_or_b32 %left_s, s38, 6", "s_and_b32 %out_s, %right_s, 14", "s_endpgm"],
                groups=[
                    ("left_s", ["s", 36, 1, False]),
                    ("right_s", ["s", 36, 1, False]),
                    ("out_s", ["s", 46, 1]),
                ],
                entry=[],
            ),
            _case(
                "narrow overwrite of one member invalidates a wide group read",
                [
                    "v_lshlrev_b64 %wide, 1, v[52:53]",
                    "v_xor_b32 %member, v22, 13",
                    "global_store_b64 v54, %wide, s[8:9]",
                    "s_endpgm",
                ],
                groups=[
                    ("wide", ["v", 14, 2, False]),
                    ("member", ["v", 15, 1, False]),
                    ("sink", ["v", 35, 1]),
                ],
                entry=[],
            ),
            _case(
                "matching reader waiver suppresses the provenance diagnostic",
                ["v_mov_b32 %early, 53", "(alias_ok late) v_or_b32 %sink, %late, 3", "s_endpgm"],
                groups=pair,
                entry=[],
            ),
            _case(
                "writer constrained waiver does not waive the actual writer",
                [
                    "v_mov_b32 %early, 59",
                    "(alias_ok late over sink) v_or_b32 %sink, %late, 3",
                    "s_endpgm",
                ],
                groups=pair,
                entry=[],
            ),
            _case(
                "back edge carries a different name into the loop head",
                [
                    "v_mov_b32 %late, 61",
                    ".Lauth_alias_head:",
                    "v_add_nc_u32 %sink, %late, 6",
                    "v_xor_b32 %early, v20, 3",
                    "v_mov_b32 %sink, %early",
                    "s_cbranch_scc1 .Lauth_alias_head",
                    "s_endpgm",
                ],
                groups=pair,
                entry=[],
            ),
        ]
    )
    return plans


def checks_plans():
    """Author check successes and refusal counterexamples."""
    plans = [
        _case(
            "both sides of a branch overwrite the initial value",
            [
                "v_or_b32 v6, v1, 5",
                "s_cmp_lg_u32 s21, 0",
                "s_cbranch_scc0 .Lck_left",
                "v_xor_b32 v6, v2, 9",
                "s_branch .Lck_join",
                ".Lck_left:",
                "v_sub_nc_u32 v6, 99, v3",
                ".Lck_join:",
                "v_add_nc_u32 v7, v6, 1",
                "s_endpgm",
            ],
            check="dead writes",
            refuses=True,
        ),
        _case(
            "one side consumes before its peer overwrites",
            [
                "v_or_b32 v6, v1, 5",
                "s_cmp_lg_u32 s21, 0",
                "s_cbranch_scc0 .Lck_left",
                "v_add_nc_u32 v8, v6, 2",
                "s_branch .Lck_join",
                ".Lck_left:",
                "v_sub_nc_u32 v6, 99, v3",
                ".Lck_join:",
                "v_add_nc_u32 v7, v6, 1",
                "s_endpgm",
            ],
            check="dead writes",
        ),
        _case(
            "kernel exit is not considered a dead overwrite",
            ["v_xor_b32 v10, v8, 3", "s_endpgm"],
            check="dead writes",
        ),
        _case(
            "disabled lanes preserve the old vector value",
            [
                "v_mov_b32 v9, 77",
                "s_mov_b32 exec_lo, 31",
                "v_xor_b32 v9, v4, 1",
                "s_mov_b32 exec_lo, -1",
                "v_add_nc_u32 v11, v9, 1",
                "s_endpgm",
            ],
            check="dead writes",
        ),
        _case(
            "full EXEC write is dead only after restoration",
            [
                "v_mov_b32 v9, 79",
                "s_mov_b32 exec_lo, 15",
                "s_mov_b32 exec_lo, -1",
                "v_xor_b32 v9, v4, 1",
                "v_add_nc_u32 v11, v9, 1",
                "s_endpgm",
            ],
            check="dead writes",
            refuses=True,
        ),
        _case(
            "immediate full width replacement is dead",
            ["v_mov_b32 v12, 81", "v_add_nc_u32 v12, v1, 7", "v_xor_b32 v13, v12, 3", "s_endpgm"],
            check="dead writes",
            refuses=True,
        ),
        _case(
            "explicit discard is a consumer",
            [
                "v_mov_b32 v12, 83",
                "; discard v:x[0]x1: lane values no longer needed",
                "v_add_nc_u32 v12, v1, 7",
                "v_xor_b32 v13, v12, 3",
                "s_endpgm",
            ],
            check="dead writes",
        ),
        _case(
            "scalar replacement is outside the vector dead write check",
            ["s_mov_b32 s19, 85", "s_add_co_u32 s19, s19, 3", "s_and_b32 s20, s19, 7", "s_endpgm"],
            check="dead writes",
        ),
        _case(
            "multiply accumulate reads its original destination",
            ["v_mov_b32 v15, 87", "v_fmac_f32 v15, v5, v7", "v_xor_b32 v16, v15, 3", "s_endpgm"],
            check="dead writes",
        ),
        _case(
            "op sel half write leaves the other half observable",
            [
                "v_mov_b32 v17, 89",
                "v_add_f16_e64 v17, v8, v9 op_sel:[0,0,1]",
                "v_add_nc_u32 v18, v17, 1",
                "s_endpgm",
            ],
            check="dead writes",
        ),
        _case(
            "overwritten dword is diagnosed even beside a live dword",
            [
                "v_mov_b32 v20, 91",
                "v_mov_b32 v21, 93",
                "v_xor_b32 v20, v5, 7",
                "global_store_b64 v[50:51], v[20:21], s[12:13]",
                "s_endpgm",
            ],
            check="dead writes",
            refuses=True,
        ),
        _case(
            "loop head consumes the previous iteration write",
            [
                "v_mov_b32 v22, 0",
                ".Lck_loop:",
                "v_sub_nc_u32 v23, v22, v4",
                "v_add_nc_u32 v22, v5, 3",
                "s_cbranch_scc1 .Lck_loop",
                "s_endpgm",
            ],
            check="dead writes",
        ),
        _case(
            "select below join has a comparison on each predecessor",
            [
                "s_cmp_lg_u32 s30, 0",
                "s_cbranch_scc0 .Lscc_b",
                "s_cmp_ge_u32 s31, 7",
                "s_branch .Lscc_join",
                ".Lscc_b:",
                "s_cmp_le_u32 s32, 19",
                ".Lscc_join:",
                "s_cselect_b32 s33, 11, 22",
                "s_endpgm",
            ],
            check="SCC readers",
        ),
        _case(
            "arithmetic on one predecessor makes joined SCC unknown",
            [
                "s_cmp_lg_u32 s30, 0",
                "s_cbranch_scc0 .Lscc_b",
                "s_cmp_ge_u32 s31, 7",
                "s_branch .Lscc_join",
                ".Lscc_b:",
                "s_sub_co_u32 s32, s32, 5",
                ".Lscc_join:",
                "s_cselect_b32 s33, 11, 22",
                "s_endpgm",
            ],
            check="SCC readers",
            refuses=True,
        ),
        _case(
            "SCC producing shift invalidates a later select",
            [
                "s_cmp_le_u32 s34, 29",
                "s_lshl2_add_u32 s35, s35, 3",
                "s_cselect_b32 s36, 4, 8",
                "s_endpgm",
            ],
            check="SCC readers",
            refuses=True,
        ),
        _case(
            "non SCC mov can sit between compare and select",
            ["s_cmp_gt_u32 s34, 31", "s_mov_b32 s35, 73", "s_cselect_b32 s36, 6, 9", "s_endpgm"],
            check="SCC readers",
        ),
        _case(
            "unsigned add feeds its adjacent carry step",
            ["s_add_co_u32 s37, s38, 13", "s_add_co_ci_u32 s39, s40, 1", "s_endpgm"],
            check="SCC readers",
        ),
        _case(
            "comparison is not a carry producer",
            ["s_cmp_eq_u32 s37, 17", "s_add_co_ci_u32 s39, s40, 1", "s_endpgm"],
            check="SCC readers",
            refuses=True,
        ),
        _case(
            "signed overflow is not unsigned carry",
            ["s_add_co_i32 s37, s38, 15", "s_add_co_ci_u32 s39, s40, 1", "s_endpgm"],
            check="SCC readers",
            refuses=True,
        ),
        _case(
            "bitwise SCC is not carry chain state",
            ["s_or_b32 s37, s38, 16", "s_add_co_ci_u32 s39, s40, 1", "s_endpgm"],
            check="SCC readers",
            refuses=True,
        ),
        _case(
            "entry select has no comparison predecessor",
            ["s_cselect_b32 s41, 14, 28", "s_endpgm"],
            check="SCC readers",
            refuses=True,
        ),
        _case(
            "two adjacent carry steps form one chain",
            [
                "s_sub_co_u32 s42, s43, 21",
                "s_sub_co_ci_u32 s44, s45, 1",
                "s_add_co_ci_u32 s46, s47, 1",
                "s_endpgm",
            ],
            check="SCC readers",
        ),
        _case(
            "conditional branch consumption is outside the select checker",
            ["s_add_co_u32 s48, s48, 19", "s_cbranch_scc1 .Lscc_exit", ".Lscc_exit:", "s_endpgm"],
            check="SCC readers",
        ),
        _case(
            "save less entry is legal with full EXEC",
            ["v_mov_b32 v24, 3", "v_cmp_lt_u32 vcc_lo, v24, 100", "s_endpgm"],
            check="save-less regions",
            entries=[1],
        ),
        _case(
            "save less entry rejects a narrowed EXEC",
            ["s_mov_b32 exec_lo, 63", "v_cmp_lt_u32 vcc_lo, v24, 100", "s_endpgm"],
            check="save-less regions",
            entries=[1],
            refuses=True,
        ),
        _case(
            "full mask restoration makes the later region legal",
            [
                "s_mov_b32 exec_lo, 63",
                "s_mov_b32 exec_lo, -1",
                "v_cmp_lt_u32 vcc_lo, v24, 100",
                "s_endpgm",
            ],
            check="save-less regions",
            entries=[2],
        ),
        _case(
            "saved full scalar mask restores all lanes",
            [
                "s_mov_b32 s56, exec_lo",
                "s_mov_b32 exec_lo, 127",
                "s_mov_b32 exec_lo, s56",
                "v_cmp_lt_u32 vcc_lo, v24, 100",
                "s_endpgm",
            ],
            check="save-less regions",
            entries=[3],
        ),
        _case(
            "join carries inconsistent EXEC facts",
            [
                "s_cmp_lg_u32 s57, 0",
                "s_cbranch_scc0 .Lexec_merge",
                "s_mov_b32 exec_lo, 255",
                ".Lexec_merge:",
                "v_cmp_lt_u32 vcc_lo, v24, 100",
                "s_endpgm",
            ],
            check="save-less regions",
            entries=[4],
            refuses=True,
        ),
        _case(
            "comparison produced EXEC is not a constant full mask",
            [
                "v_cmp_ge_u32 vcc_lo, v24, 17",
                "s_mov_b32 exec_lo, vcc_lo",
                "v_xor_b32 v25, v4, 3",
                "s_endpgm",
            ],
            check="save-less regions",
            entries=[2],
            refuses=True,
        ),
        _case(
            "high bit constant remains a known partial EXEC",
            [
                "s_mov_b32 exec_lo, 1073741824",
                "s_mov_b32 exec_lo, -1",
                "v_mov_b32 v26, 6",
                "s_endpgm",
            ],
            check="save-less regions",
            entries=[1],
            refuses=True,
        ),
    ]
    return plans


def hazards_plans():
    """Exercise each native rule with separate data flow and fresh labels."""
    out = []

    def add(name, *lines):
        """Construct add inputs."""
        out.append(_case(name, _finish(lines)))

    def tracked(read, write, tail):
        """Construct tracked inputs."""
        return [f"v_cndmask_b32_e64 v17, v6, v7, {read}", f"s_cselect_b32 {write}, 5, 27", tail]

    add(
        "tracked scalar destination followed by scalar consumer",
        *tracked("s10", "s11", "s_add_co_u32 s18, s11, 9"),
    )
    for read, write in (("s8", "s8"), ("s9", "s8"), ("s8", "s9"), ("s13", "s13")):
        add(
            f"carry pair pairing {read} then {write}",
            f"v_add_co_ci_u32_e64 v18, vcc_lo, 3, v6, {read}",
            f"s_cselect_b32 {write}, 17, 29",
            f"s_or_b32 s19, {write}, 7",
        )
    add(
        "scalar selected value used as a vector carry selector",
        "v_add_co_ci_u32_e64 v18, vcc_lo, 3, v6, s8",
        "s_cselect_b32 s9, 17, 29",
        "v_add_co_ci_u32_e64 v20, vcc_lo, 5, v18, s9",
    )
    add(
        "selected VCC consumed by scalar arithmetic",
        "v_cndmask_b32 v16, v8, v9, vcc_lo",
        "s_cselect_b32 vcc_lo, 33, 47",
        "s_xor_b32 s20, vcc_lo, 5",
    )
    add(
        "selected VCC consumed by vector conditional move",
        "v_cndmask_b32 v16, v8, v9, vcc_lo",
        "s_cselect_b32 vcc_lo, 33, 47",
        "v_cndmask_b32 v21, v10, v11, vcc_lo",
    )
    add(
        "vector carry result then VCC readers",
        "v_add_co_ci_u32_e64 v19, s14, 1, v6, vcc_lo",
        "v_add_co_u32 v22, vcc_lo, v7, v19",
        "v_add_co_ci_u32 v23, vcc_lo, v9, v10, vcc_lo",
    )
    add(
        "scalar VCC copy commits vector state before next vector writer",
        "v_add_co_ci_u32_e64 v19, s14, 1, v6, vcc_lo",
        "v_add_co_u32 v22, vcc_lo, v7, v19",
        "s_mov_b32 s23, vcc_lo",
        "v_add_co_ci_u32 v24, vcc_lo, v9, v10, vcc_lo",
    )
    add(
        "merged SALU and VALU condition states both reach consumer",
        "v_cndmask_b32 v18, v7, v8, vcc_lo",
        "s_cmp_eq_u32 s25, 0",
        "s_cbranch_scc0 .Lhz_scalar",
        "v_cmp_gt_f32_e64 vcc_lo, v4, v5",
        "s_branch .Lhz_merge",
        ".Lhz_scalar:",
        "s_cselect_b32 vcc_lo, 51, 63",
        ".Lhz_merge:",
        "v_cndmask_b32 v26, v12, v13, vcc_lo",
    )
    add(
        "loop back edge establishes a scalar pair at its header",
        "s_nop 0",
        ".Lhz_head:",
        "s_getpc_b64 s[26:27]",
        "s_add_co_u32 s29, s26, 3",
        "v_cndmask_b32_e64 v20, v8, v9, s26",
        "s_cbranch_scc0 .Lhz_head",
    )
    add(
        "unpaired program counter write requires no wait",
        "s_getpc_b64 s[30:31]",
        "s_add_co_u32 s32, s30, 5",
        "v_cndmask_b32_e64 v22, v10, v11, s30",
    )
    add(
        "unpaired state crosses an ordinary control label",
        "s_getpc_b64 s[34:35]",
        "s_add_co_u32 s36, s34, 5",
        "s_branch .Lhz_plain",
        ".Lhz_plain:",
        "v_cndmask_b32_e64 v23, v10, v11, s34",
    )
    add(
        "vector carry output has immediate and later dependent readers",
        "v_add_co_ci_u32_e64 v25, s38, 2, v14, s38",
        "v_add_co_u32 v27, s38, v15, v25",
        "v_add_co_ci_u32_e64 v28, s39, v17, v18, s38",
    )
    add(
        "scalar copy between vector reads of one scalar selector",
        "v_add_co_ci_u32_e64 v25, s38, 2, v14, s38",
        "v_add_co_u32 v27, s38, v15, v25",
        "s_mov_b32 s40, s38",
        "v_add_co_ci_u32_e64 v28, s39, v17, v18, s38",
    )
    add(
        "buffer memory operation does not commit vector selector state",
        "v_add_co_ci_u32_e64 v25, s38, 2, v14, s38",
        "v_add_co_u32 v27, s38, v15, v25",
        "buffer_load_b32 v31, off, s[44:47], s48",
        "v_add_co_ci_u32_e64 v28, s39, v17, v31, s38",
    )
    add(
        "scalar memory operation does not commit vector selector state",
        "v_add_co_ci_u32_e64 v25, s38, 2, v14, s38",
        "v_add_co_u32 v27, s38, v15, v25",
        "s_load_b32 s49, s[50:51], 12",
        "v_add_co_ci_u32_e64 v28, s39, v17, v18, s38",
    )
    add(
        "ordinary scalar comparison destination is not tracked",
        "v_cmp_eq_f32_e64 s52, v5, v6",
        "s_cselect_b32 s52, 43, 47",
        "s_add_co_u32 s53, s52, 7",
    )
    add(
        "SALU nops before a selected write do not retire its pairing",
        "v_add_co_ci_u32_e64 v25, vcc_lo, 2, v14, s54",
        *(["s_nop 0"] * 6),
        "s_cselect_b32 s55, 11, 13",
        "s_add_co_u32 s56, s55, 7",
    )
    add(
        "SALU nops after a selected write do not retire its pairing",
        "v_add_co_ci_u32_e64 v25, vcc_lo, 2, v14, s54",
        "s_cselect_b32 s55, 11, 13",
        *(["s_nop 0"] * 6),
        "s_add_co_u32 s56, s55, 7",
    )
    for op, load, waits in (
        ("buffer", "buffer_load_b32 v33, off, s[60:63], s64", ["s_wait_loadcnt 0"]),
        ("global", "global_load_b32 v33, v[34:35], off offset:16", ["s_wait_loadcnt 0"]),
        ("flat", "flat_load_b32 v33, v[34:35]", ["s_wait_loadcnt 0", "s_wait_dscnt 0"]),
    ):
        add(
            f"{op} completion establishes the VMEM commit rule",
            "v_add_co_ci_u32_e64 v29, s65, 2, v20, s65",
            load,
            *waits,
            "s_getpc_b64 s[64:65]",
            "s_add_co_u32 s68, s64, 11",
        )
    add(
        "four DS nops age out tracked scalar pairs",
        "v_add_co_ci_u32_e64 v30, s69, 2, v21, s69",
        *(["ds_nop"] * 4),
        "s_getpc_b64 s[68:69]",
        "s_add_co_u32 s72, s68, 13",
    )
    add(
        "three DS nops retain tracked scalar pairs",
        "v_add_co_ci_u32_e64 v30, s69, 2, v21, s69",
        *(["ds_nop"] * 3),
        "s_getpc_b64 s[68:69]",
        "s_add_co_u32 s72, s68, 13",
    )
    add(
        "raw numeric ALU wait commits selected scalar state",
        "v_cndmask_b32_e64 v32, v15, v16, s75",
        "s_cselect_b32 s74, 5, 31",
        "s_wait_alu 0xfff8",
        "s_add_co_u32 s75, s74, 17",
    )
    add(
        "explicit symbolic ALU wait is reused",
        "v_cndmask_b32_e64 v32, v15, v16, s75",
        "s_cselect_b32 s74, 5, 31",
        "s_wait_alu depctr_sa_sdst(0)",
        "s_add_co_u32 s75, s74, 17",
    )

    def cmpx_case(name, middle, permute, e64=False, nops=0):
        """Construct cmpx case inputs."""
        cmpop = "v_cmpx_gt_f32_e64" if e64 else "v_cmpx_gt_f32"
        add(
            name,
            "v_mov_b32 v36, 9",
            f"{cmpop} 19, v36",
            "s_branch .Lhz_lane",
            ".Lhz_lane:",
            *middle,
            *(["v_nop"] * nops),
            permute,
        )

    cmpx_case(
        "SALU work between CMPX and permlane does not remove CMPX hazard",
        ["s_or_b32 s77, s76, 5"],
        "v_permlane16_b32 v37, v35, s77, s76",
    )
    cmpx_case(
        "permlanex16 directly follows CMPX across a basic block",
        [],
        "v_permlanex16_b32 v38, v34, s79, s78",
    )
    cmpx_case(
        "variable permlane directly follows CMPX across a basic block",
        [],
        "v_permlane16_var_b32 v39, v33, v32",
    )
    cmpx_case(
        "one vector nop does not retire CMPX state",
        [],
        "v_permlane16_b32 v40, v31, s81, s80",
        nops=1,
    )
    cmpx_case(
        "seven vector nops do not retire CMPX state",
        [],
        "v_permlane16_b32 v41, v30, s83, s82",
        nops=7,
    )
    add(
        "e64 CMPX also protects its subsequent permlane",
        "v_cmpx_gt_f32_e64 23, v28",
        "v_permlane16_b32 v42, v29, s85, s84",
    )
    cmpx_case(
        "real VALU work retires CMPX state",
        ["v_mul_f32 v43, v26, 2.0"],
        "v_permlane16_b32 v44, v27, s87, s86",
    )
    add(
        "CMPX hazard survives on only one incoming path",
        "v_cmpx_ge_f32 29, v25",
        "s_cbranch_scc0 .Lhz_permute",
        "v_add_f32 v45, v24, 1.0",
        ".Lhz_permute:",
        "v_permlane16_b32 v46, v23, s89, s88",
    )

    first = "v_wmma_f32_16x16x16_f16 v[16:23], v[48:51], v[52:55], v[16:23]"
    add(
        "second WMMA consumes previous result in its A operand",
        first,
        "v_wmma_f32_16x16x16_bf16 v[80:87], v[16:19], v[56:59], v[80:87]",
    )
    add(
        "second WMMA consumes previous result in its B operand",
        first,
        "v_wmma_f32_16x16x16_f16 v[88:95], v[60:63], v[16:19], v[88:95]",
    )
    add(
        "accumulator C overlap alone needs no WMMA spacing nop",
        first,
        "v_wmma_f16_16x16x16_f16 v[16:19], v[64:67], v[68:71], v[16:19]",
    )
    add(
        "integer WMMA B input overlaps an earlier floating result",
        first,
        "v_wmma_i32_16x16x16_iu8 v[96:103], v[72:73], v[16:17], v[96:103]",
    )
    add(
        "SWM MAC index register overlaps earlier matrix result",
        first,
        "v_swmmac_f32_16x16x32_bf16 v[104:111], v[74:77], v[78:85], v18",
    )
    add(
        "SWM MAC accumulator overlap alone is permitted",
        first,
        "v_swmmac_f32_16x16x32_f16 v[16:23], v[86:89], v[90:97], v112",
    )
    add(
        "ordinary VALU instruction spaces two matrix operations",
        first,
        "v_sub_f32 v124, v114, 2.0",
        "v_wmma_f32_16x16x16_f16 v[120:127], v[16:19], v[128:131], v[120:127]",
    )
    add(
        "SALU instruction does not space two matrix operations",
        first,
        "s_xor_b32 s90, s90, 3",
        "v_wmma_f32_16x16x16_f16 v[132:139], v[16:19], v[140:143], v[132:139]",
    )

    add(
        "one instruction VALU dependence gets dep1 delay",
        "v_mul_f32 v150, v151, 3.0",
        "v_add_f32 v152, v150, 1.0",
    )
    add(
        "one intervening VALU instruction yields dep2 delay",
        "v_mul_f32 v153, v154, 3.0",
        "v_xor_b32 v155, v156, 7",
        "v_sub_f32 v157, 9.0, v153",
    )
    add(
        "youngest of two producing VALU writes determines delay",
        "v_add_f32 v158, v159, 1.0",
        "v_mul_f32 v160, v161, 2.0",
        "v_sub_f32 v162, v158, v160",
    )
    add(
        "four VALU distance ignores an interleaved SALU operation",
        "v_mov_b32 v163, 13",
        "v_xor_b32 v164, v165, 3",
        "s_mov_b32 s91, 17",
        "v_mov_b32 v166, 15",
        "v_mov_b32 v167, 19",
        "v_add_f32 v168, v163, 2.0",
    )
    add(
        "five VALU distance is beyond the insertion window",
        "v_mov_b32 v169, 13",
        "v_xor_b32 v170, v171, 3",
        "v_mov_b32 v172, 15",
        "v_mov_b32 v173, 19",
        "v_mov_b32 v174, 21",
        "v_add_f32 v175, v169, 2.0",
    )
    add(
        "VMEM wide address read needs no arithmetic delay hint",
        "v_mov_b32 v176, 31",
        "global_load_b128 v[180:183], v176, s[92:93]",
        "s_wait_loadcnt 0",
    )
    add(
        "VCC comparison to conditional move gets a VALU delay",
        "v_cmp_gt_f32_e64 vcc_lo, v177, v178",
        "v_cndmask_b32_e64 v179, v184, v185, vcc_lo",
    )
    add(
        "memory wait retires arithmetic dependence bookkeeping",
        "v_lshl_add_u32 v186, s94, 6, v0",
        "s_wait_kmcnt 0",
        "v_cmp_lt_u32_e64 vcc_lo, s95, v186",
    )
    add(
        "memory destination overwrite replaces older VALU provenance",
        "v_mov_b32 v187, 33",
        "global_load_b32 v187, v188, s[96:97]",
        "v_add_f32 v189, v187, 4.0",
    )
    add(
        "ALU wait does not masquerade as a memory wait",
        "v_mov_b32 v190, 35",
        "s_wait_alu depctr_va_sdst(0)",
        "v_xor_b32 v191, v190, 7",
    )
    add(
        "transcendental result feeds a VALU consumer",
        "v_exp_f32 v192, v193",
        "v_mul_f32 v194, v192, 2.0",
    )
    add(
        "simultaneous transcendental and VALU dependencies are encoded together",
        "v_exp_f32 v195, v196",
        "v_mul_f32 v197, v198, 2.0",
        "v_add_f32 v199, v195, v197",
    )
    add(
        "second transcendental operation creates dep2 on the older result",
        "v_exp_f32 v200, v201",
        "v_sqrt_f32 v202, v203",
        "v_mul_f32 v204, v200, 3.0",
    )
    add(
        "existing delay instruction satisfies the VALU dependence",
        "v_mul_f32 v205, v206, 3.0",
        "s_delay_alu 1",
        "v_add_f32 v207, v205, 4.0",
    )
    add(
        "matrix result does not become a normal arithmetic dependency",
        first,
        "v_add_f32 v208, v16, 2.0",
    )
    add(
        "matrix input still depends on the immediately preceding VALU result",
        "v_add_f32 v209, v210, 2.0",
        "v_wmma_f32_16x16x16_f16 v[212:219], v[209:212], v[220:223], v[212:219]",
    )
    add(
        "untargeted label preserves the arithmetic producer distance",
        "v_mul_f32 v224, v225, 2.0",
        ".Lhz_annotation:",
        "v_add_f32 v226, v224, 3.0",
    )
    add(
        "targeted label begins a fresh hazard block",
        "v_mul_f32 v227, v228, 2.0",
        "s_cbranch_scc0 .Lhz_target",
        ".Lhz_target:",
        "v_add_f32 v229, v227, 3.0",
    )
    add(
        "inserted CMPX copy counts in subsequent arithmetic distance",
        "v_cmpx_ge_f32 37, v230",
        "v_add_f32 v231, v232, 5.0",
        "s_xor_b32 s98, s98, 1",
        "v_cmpx_ge_f32 41, v230",
        "v_permlane16_b32 v233, v231, s99, s98",
    )
    out[-1]["show_rerun"] = True
    add(
        "VALU overwrite ends transcendental tracking before a conversion",
        "v_rcp_iflag_f32 v234, v235",
        "v_mul_f32 v234, v234, 3.0",
        "v_cvt_u32_f32 v234, v234",
    )
    add(
        "transcendental overwrite replaces older VALU tracking",
        "v_mov_b32 v236, 45",
        "v_exp_f32 v236, v237",
        "v_add_f32 v238, v236, 6.0",
    )
    add(
        "partial VALU overwrite preserves adjacent transcendental dependency",
        "v_exp_f32 v239, v240",
        "v_sqrt_f32 v241, v242",
        "v_mov_b32 v239, 47",
        "v_add_f32 v243, v239, v241",
    )
    add(
        "newest of repeated writers remains the tracked producer",
        "v_exp_f32 v244, v245",
        "v_mul_f32 v244, v244, 2.0",
        "v_sqrt_f32 v244, v244",
        "v_add_f32 v246, v244, 1.0",
    )
    return out


def transfer_plans():
    """Fresh interval bounds for every transfer family and its edge cases."""
    specs = [
        ("s_mov_b32", [[11, 27]]),
        ("v_mov_b32", [[0, 4294967294]]),
        ("v_readfirstlane_b32", [[19, 19]]),
        ("s_cselect_b32", [[7, 7], [3, 3]]),
        ("v_cndmask_b32", [[6, 22], [-9, 2]]),
        ("s_add_co_u32", [[3, 73], [9, 9]]),
        ("v_add_nc_u32", [[0, 4294967294], [2, 2]]),
        ("v_add_nc_u32", [[-1073741824, -3], [1073741824, 1073741824]]),
        ("v_add_nc_u32", [[0, 4294967295], [1, 4294967295]]),
        ("s_sub_co_u32", [[17, 41], [2, 29]]),
        ("v_sub_nc_u32", [[0, 31], [32, 32]]),
        ("v_subrev_nc_u32", [[3, 5], [20, 20]]),
        ("s_lshl_b32", [[1, 63], [3, 3]]),
        ("v_lshlrev_b32", [[3, 3], [1, 63]]),
        ("v_lshlrev_b32", [[1, 3], [2, 127]]),
        ("v_lshlrev_b32", [[33, 33], [0, 127]]),
        ("v_lshlrev_b32", [[30, 30], [1, 5]]),
        ("s_lshl2_add_u32", [[2, 21], [73, 73]]),
        ("v_lshl_add_u32", [[1, 13], [3, 3], [11, 11]]),
        ("v_add_lshl_u32", [[2, 17], [2, 2], [4, 4]]),
        ("s_mul_i32", [[-7, 2], [3, 8]]),
        ("v_mul_lo_u32", [[0, 2097152], [0, 4096]]),
        ("v_mul_u32_u24", [[0, 71], [0, 513]]),
        ("v_mad_u32_u24", [[0, 71], [0, 513], [9, 9]]),
        ("v_mul_u32_u24", [[-2, 1], [7, 7]]),
        ("s_lshr_b32", [[0, 4095], [1, 4]]),
        ("v_lshrrev_b32", [[1, 3], [24, 4000]]),
        ("v_lshrrev_b32", [[0, 3], [-3, 4000]]),
        ("s_and_b32", [[0, 2047], None]),
        ("v_and_b32", [[15, 15], [0, 2047]]),
        ("v_and_b32", [None, None]),
        ("s_or_b32", [[0, 9], [0, 34]]),
        ("v_xor_b32", [[0, 15], [0, 6]]),
        ("v_lshl_or_b32", [[0, 5], [3, 3], [0, 31]]),
        ("v_lshl_or_b32", [[0, 0], [36, 36], [0, 9]]),
        ("v_lshl_or_b32", [[1, 1], [30, 30], [0, 1]]),
        ("s_min_u32", [[2, 201], [7, 93]]),
        ("v_max_u32", [[2, 201], [7, 93]]),
        ("v_bfe_u32", [[22136, 22136], [3, 3], [6, 6]]),
        ("v_bfe_u32", [[0, 2000], [5, 5], [4, 4]]),
        ("v_bfe_u32", [[0, 2000], None, [0, 6]]),
        ("s_bfe_u32", [[44043, 44043], [524294, 524294]]),
        ("s_bfe_u32", [[0, 2000], [327690, 327690]]),
        ("v_add_f32", [[0, 2], [0, 3]]),
        ("v_add_nc_u32", [[2, 7], None]),
    ]
    return [
        {"name": f"interval transfer {i:02d}: {op}", "op": op, "operands": ops, "k": 4 + i % 3}
        for i, (op, ops) in enumerate(specs)
    ]


def ranges_plans():
    """Author range-proof programs with launch assumptions."""
    plans = []
    args3 = [_arg(0, 8), _arg(8, 8), _arg(20, 4, 1, 300000)]
    grid_program = [
        "s_load_b128 s[12:15], s[0:1], 0",
        "s_load_b32 s9, s[0:1], 20",
        "v_lshl_add_u32 v6, ttmp9, 7, v0",
        "s_wait_kmcnt 0",
        "v_cmp_gt_u32 vcc_lo, s9, v6",
        "s_mov_b32 exec_lo, vcc_lo",
        "v_lshlrev_b32 v7, 2, v6",
        "ds_load_b32 v8, v7",
        "s_wait_dscnt 0",
        "ds_store_b32 v7, v8",
        "s_mov_b32 exec_lo, -1",
        "s_endpgm",
    ]
    plans.append(
        _case(
            "grid bounded vector address proves both DS operations",
            grid_program,
            launch=_launch(args3, 128, [2048]),
        )
    )
    plans.append(
        _case(
            "same vector address remains unproven without a grid bound",
            grid_program,
            launch=_launch(args3, 128),
        )
    )
    plans.append(
        _case(
            "oversized grid makes the scaled address reject wrapping",
            grid_program,
            launch=_launch(args3, 128, [33554432]),
            refuses=True,
        )
    )
    wide = _launch(wgsize=2147483648)
    plans.append(
        _case(
            "lane mask narrows addresses and restoration recovers entry facts",
            [
                "v_cmp_gt_u32 vcc_lo, 96, v0",
                "s_mov_b32 exec_lo, vcc_lo",
                "v_lshlrev_b32 v10, 1, v0",
                "ds_load_b32 v11, v10",
                "s_mov_b32 exec_lo, -1",
                "s_wait_dscnt 0",
                "ds_load_b32 v12, v10",
                "s_wait_dscnt 0",
                "s_endpgm",
            ],
            launch=wide,
        )
    )
    plans.append(
        _case(
            "SGPR comparison mask narrows just like VCC",
            [
                "v_cmp_gt_u32 s17, 96, v0",
                "s_and_saveexec_b32 s18, s17",
                "v_lshlrev_b32 v10, 1, v0",
                "ds_load_b32 v11, v10",
                "s_mov_b32 exec_lo, s18",
                "s_wait_dscnt 0",
                "s_endpgm",
            ],
            launch=wide,
        )
    )
    plans.append(
        _case(
            "restoring EXEC before address construction loses narrowed bound",
            [
                "v_cmp_gt_u32 vcc_lo, 96, v0",
                "s_mov_b32 exec_lo, vcc_lo",
                "s_mov_b32 exec_lo, -1",
                "v_lshlrev_b32 v10, 3, v0",
                "ds_load_b32 v11, v10",
                "s_wait_dscnt 0",
                "s_endpgm",
            ],
            launch=wide,
            refuses=True,
        )
    )
    plans.append(
        _case(
            "scalar versus lane comparison constrains the scalar inside region",
            [
                "s_load_b32 s19, s[0:1], 20",
                "s_wait_kmcnt 0",
                "v_cmp_gt_u32 vcc_lo, s19, v0",
                "s_mov_b32 exec_lo, vcc_lo",
                "v_lshlrev_b32 v10, 4, v0",
                "ds_load_b32 v11, v10",
                "s_mov_b32 exec_lo, -1",
                "s_wait_dscnt 0",
                "s_endpgm",
            ],
            launch=wide,
            refuses=True,
        )
    )
    plans.append(
        _case(
            "signed upper comparison alone retains a possibly negative lower endpoint",
            [
                "v_cmp_lt_i32 vcc_lo, v0, 96",
                "s_mov_b32 exec_lo, vcc_lo",
                "v_lshlrev_b32 v10, 1, v0",
                "ds_load_b32 v11, v10",
                "s_mov_b32 exec_lo, -1",
                "s_wait_dscnt 0",
                "s_endpgm",
            ],
            launch=wide,
        )
    )
    plans.append(
        _case(
            "declared scalar argument interval propagates through add and scale",
            [
                "s_load_b32 s20, s[0:1], 20",
                "s_wait_kmcnt 0",
                "v_add_nc_u32 v13, s20, v0",
                "v_lshlrev_b32 v14, 3, v13",
                "ds_load_b32 v15, v14",
                "s_wait_dscnt 0",
                "s_endpgm",
            ],
            launch=_launch([_arg(0, 8), _arg(8, 8), _arg(20, 4, 2, 500)], 128),
        )
    )
    plans.append(
        _case(
            "subtract then re-add recovers a positive constant address",
            [
                "v_sub_nc_u32 v16, 7, v0",
                "v_add_nc_u32 v17, 300, v16",
                "v_add_nc_u32 v17, v17, v0",
                "ds_load_b32 v18, v17",
                "s_wait_dscnt 0",
                "s_endpgm",
            ],
        )
    )
    plans.append(
        _case(
            "small constant minus lane id crosses below zero",
            ["v_sub_nc_u32 v19, 5, v0", "ds_load_b32 v20, v19", "s_wait_dscnt 0", "s_endpgm"],
            launch=_launch(wgsize=128),
            refuses=True,
        )
    )
    plans.append(
        _case(
            "right shift does not erase earlier wrap provenance",
            [
                "v_sub_nc_u32 v19, 5, v0",
                "v_lshrrev_b32 v21, 2, v19",
                "ds_load_b32 v20, v21",
                "s_wait_dscnt 0",
                "s_endpgm",
            ],
            launch=_launch(wgsize=128),
            refuses=True,
        )
    )
    plans.append(
        _case(
            "unknown raw vector offset is reported rather than assumed",
            ["ds_load_b32 v22, v23", "s_wait_dscnt 0", "s_endpgm"],
        )
    )
    plans.append(
        _case(
            "u24 multiplier rejects a scaled value beyond 24 bits",
            ["v_lshlrev_b32 v24, 22, v0", "v_mul_u32_u24 v25, v24, 5", "s_endpgm"],
            refuses=True,
        )
    )
    plans.append(
        _case(
            "u24 multiplier rejects an entirely unknown scalar operand",
            [
                "s_load_b32 s21, s[0:1], 0",
                "s_wait_kmcnt 0",
                "v_mul_u32_u24 v25, s21, 5",
                "s_endpgm",
            ],
            refuses=True,
        )
    )
    plans.append(
        _case(
            "u24 product inside its operand bound can feed DS",
            [
                "v_lshlrev_b32 v24, 9, v0",
                "v_mul_u32_u24 v25, v24, 5",
                "ds_load_b32 v26, v25",
                "s_wait_dscnt 0",
                "s_endpgm",
            ],
            launch=_launch(wgsize=128),
        )
    )
    plans.append(
        _case(
            "global memory address reports scalar and vector offset facts",
            ["global_load_b32 v27, v0, s[28:29]", "s_wait_loadcnt 0", "s_endpgm"],
        )
    )
    plans.append(
        _case(
            "explicit DS immediate contributes to the address byte range",
            [
                "v_lshlrev_b32 v28, 2, v0",
                "ds_load_b32 v29, v28 offset:65280",
                "s_wait_dscnt 0",
                "s_endpgm",
            ],
        )
    )
    plans.append(
        _case(
            "M0 arithmetic is followed alongside the explicit address",
            [
                "s_mov_b32 m0, 512",
                "s_add_co_u32 m0, m0, 12",
                "v_lshlrev_b32 v30, 2, v0",
                "ds_load_b32 v31, v30",
                "s_wait_dscnt 0",
                "s_endpgm",
            ],
        )
    )
    plans.append(
        _case(
            "comparison without a lane region does not narrow the later address",
            [
                "v_cmp_gt_u32 vcc_lo, 96, v0",
                "v_lshlrev_b32 v32, 3, v0",
                "ds_load_b32 v33, v32",
                "s_wait_dscnt 0",
                "s_endpgm",
            ],
            launch=wide,
            refuses=True,
        )
    )
    plans.append(
        _case(
            "loop-carried address widens conservatively",
            [
                "v_mov_b32 v34, 8",
                ".Lrange_head:",
                "v_add_nc_u32 v34, 12, v34",
                "ds_load_b32 v35, v34",
                "s_wait_dscnt 0",
                "s_cbranch_scc1 .Lrange_head",
                "s_endpgm",
            ],
        )
    )
    plans.append(
        _case(
            "explicit thread id subset supplies the address bounds",
            ["v_lshlrev_b32 v36, 2, v0", "ds_load_b32 v37, v36", "s_wait_dscnt 0", "s_endpgm"],
            launch=_launch(wgsize=96, tid=[16, 47]),
        )
    )
    plans.append(
        _case(
            "explicit loaded data claim controls the subsequent DS proof",
            [
                "global_load_b32 v38, v0, s[30:31]",
                "s_wait_loadcnt 0",
                "; assume: v:x[0]x1+1 in [4, 900] (independent data contract)",
                "v_lshlrev_b32 v39, 2, v38",
                "ds_load_b32 v40, v39",
                "s_wait_dscnt 0",
                "s_endpgm",
            ],
        )
    )
    plans.append(
        _case(
            "buffer VOFFEN offset is described by the range report",
            ["buffer_load_b32 v41, v0, s[36:39], 0 offen", "s_wait_loadcnt 0", "s_endpgm"],
        )
    )
    plans.append(
        _case(
            "plain lane-scaled DS address is fully bounded",
            ["v_lshlrev_b32 v42, 2, v0", "ds_load_b32 v43, v42", "s_wait_dscnt 0", "s_endpgm"],
        )
    )
    plans.append(
        _case(
            "workgroup Y id participates in a large power-of-two scale",
            [
                "v_mov_b32 v44, ttmp7",
                "v_lshlrev_b32 v45, 11, v44",
                "ds_load_b32 v46, v45",
                "s_wait_dscnt 0",
                "s_endpgm",
            ],
            launch=_launch(wgy=1, max_grid=[3, 2097152]),
        )
    )
    plans.append(
        _case(
            "preloaded bounded scalar supplies a DS address",
            ["v_lshlrev_b32 v47, 3, s2", "ds_load_b32 v48, v47", "s_wait_dscnt 0", "s_endpgm"],
            launch=_launch([_arg(0, 4, 0, 60)], preload=1),
        )
    )
    plans.append(
        _case(
            "divisor materialization with variable shift is deliberately not inferred",
            [
                "s_load_b128 s[40:43], s[0:1], 0",
                "s_wait_kmcnt 0",
                "v_lshlrev_b32 v49, s42, v0",
                "ds_load_b32 v50, v49",
                "s_wait_dscnt 0",
                "s_endpgm",
            ],
            launch=_launch([_arg(0, 16, 1, 65536, divisor=True)]),
        )
    )
    plans.append(
        _case(
            "overwriting kernarg base prevents later argument range recovery",
            [
                "s_nop 0",
                "s_mov_b32 s0, 0",
                "s_load_b32 s44, s[0:1], 20",
                "s_wait_kmcnt 0",
                "v_add_nc_u32 v51, s44, v0",
                "ds_load_b32 v52, v51",
                "s_wait_dscnt 0",
                "s_endpgm",
            ],
            launch=_launch(args3),
        )
    )
    for plan in plans:
        plan.setdefault("launch", _launch())
    return plans


def regalloc_plans():
    """Author register-pressure, lifetime and placement programs."""
    plans = [
        _case(
            "sequential vector values reuse disjoint storage",
            [
                "v_mov_b32 %first, 7",
                "v_xor_b32 %second, %first, %tid",
                "v_add_nc_u32 %answer, %second, 11",
                "v_sub_nc_u32 %answer, %answer, %tid",
            ],
            decl=[["first", 1], ["second", 1], ["answer", 1]],
            lanes_covered=[],
        ),
        _case(
            "unwritten symbolic input is refused at its first use",
            ["v_or_b32 %computed, %mystery, 13", "v_add_nc_u32 %result, %computed, %tid"],
            decl=[["computed", 1], ["result", 1], ["mystery", 1]],
            lanes_covered=[],
            refuses=True,
        ),
        _case(
            "unread wide load remains allocated through its landing wait",
            [
                "v_mov_b32 %seeds[0], 3",
                "v_mov_b32 %seeds[1], 5",
                "v_mov_b32 %seeds[2], 9",
                "buffer_load_b128 %packet[0]/4, %tid, %resource[0]/4, 0 offen",
                "v_mov_b32 %work, 17",
                "v_xor_b32 %work, %work, %tid",
                "@wait",
                "v_add_nc_u32 %work, %work, %seeds[0]",
                "v_add_nc_u32 %work, %work, %seeds[1]",
                "v_add_nc_u32 %work, %work, %seeds[2]",
            ],
            decl=[["seeds", 3], ["packet", 1, 4, 4], ["work", 1]],
            lanes_covered=[],
        ),
        _case(
            "read wide load releases storage after read and a label",
            [
                "v_mov_b32 %seeds[0], 3",
                "v_mov_b32 %seeds[1], 5",
                "v_mov_b32 %seeds[2], 9",
                "buffer_load_b128 %packet[0]/4, %tid, %resource[0]/4, 0 offen",
                "@wait",
                "v_add_nc_u32 %seeds[0], %packet[0]/4.2, %seeds[0]",
                ".Lrelease:",
                "v_mov_b32 %work, 17",
                "v_xor_b32 %work, %work, %tid",
                "@wait",
                "v_add_nc_u32 %work, %work, %seeds[0]",
                "v_add_nc_u32 %work, %work, %seeds[1]",
                "v_add_nc_u32 %work, %work, %seeds[2]",
            ],
            decl=[["seeds", 3], ["packet", 1, 4, 4], ["work", 1]],
            lanes_covered=[],
        ),
        _case(
            "unread portion of a load window persists until a block boundary",
            [
                "v_mov_b32 %seeds[0], 3",
                "v_mov_b32 %seeds[1], 5",
                "v_mov_b32 %seeds[2], 9",
                "buffer_load_b128 %packet[0]/4, %tid, %resource[0]/4, 0 offen",
                "@wait",
                "v_add_nc_u32 %seeds[0], %packet[0]/4.1, %seeds[0]",
                "v_mov_b32 %work, 17",
                "v_xor_b32 %work, %work, %tid",
                "@wait",
                "v_add_nc_u32 %work, %work, %seeds[0]",
                "v_add_nc_u32 %work, %work, %seeds[1]",
                "v_add_nc_u32 %work, %work, %seeds[2]",
            ],
            decl=[["seeds", 3], ["packet", 1, 4, 4], ["work", 1]],
            lanes_covered=[],
        ),
    ]
    partial_program = [
        "v_mov_b32 %old, 21",
        ".Lpartial:",
        "v_mov_b32 %local, 23",
        "v_xor_b32 %local, %local, %tid",
        "v_add_nc_u32 %combined, %local, 2",
        "s_and_saveexec_b32 %saved_mask, vcc_lo",
        "v_xor_b32 %old, %tid, 9",
        "s_mov_b32 exec_lo, -1",
        "v_add_nc_u32 %combined, %combined, %old",
    ]
    plans.append(
        _case(
            "lane partial overwrite extends the previous value live range",
            partial_program,
            decl=[["local", 1], ["old", 1], ["combined", 1]],
            lanes_covered=[],
        )
    )
    plans.append(
        _case(
            "declared lane coverage permits immediate old value reuse",
            partial_program,
            decl=[["local", 1], ["old", 1], ["combined", 1]],
            lanes_covered=["old"],
        )
    )
    plans.extend(
        [
            _case(
                "four dword tuple receives aligned placement",
                [
                    "v_mov_b32 %index, 29",
                    "buffer_load_b128 %quad[0]/4, %index, %resource[0]/4, 0 offen",
                    "@wait",
                    "v_xor_b32 %index, %quad[0]/4.0, %index",
                ],
                decl=[["index", 1], ["quad", 1, 4, 4]],
                lanes_covered=[],
            ),
            _case(
                "two dword load respects even pair alignment",
                [
                    "v_mov_b32 %index, 31",
                    "v_mov_b32 %spacer, 33",
                    "buffer_load_b64 %duo/2, %index, %resource[0]/4, 0 offen",
                    "@wait",
                    "v_xor_b32 %index, %spacer, %index",
                    "v_add_nc_u32 %index, %duo/2.0, %index",
                ],
                decl=[["index", 1], ["spacer", 1], ["duo", 1, 2]],
                lanes_covered=[],
            ),
        ]
    )
    pressure = [f"v_mov_b32 %many[{i}], {i * 3 + 2}" for i in range(24)]
    pressure += [f"v_xor_b32 %many[{i}], %many[{i}], %tid" for i in reversed(range(24))]
    plans.append(
        _case(
            "one value over the available bank reports its pressure peak",
            pressure,
            decl=[["many", 24]],
            lanes_covered=[],
            refuses=True,
        )
    )
    plans.extend(
        [
            _case(
                "first fit follows first occupancy with declaration tie break",
                [
                    "v_mov_b32 %middle, 35",
                    "v_mov_b32 %early, 37",
                    "v_sub_nc_u32 %last, %early, %middle",
                    "v_xor_b32 %last, %last, %tid",
                ],
                decl=[["early", 1], ["middle", 1], ["last", 1]],
                lanes_covered=[],
            ),
            _case(
                "single instruction destination cannot overlap its live operand",
                [
                    "v_mov_b32 %source, 39",
                    "v_or_b32 %dest, %source, 7",
                    "v_add_nc_u32 %dest, %dest, %tid",
                ],
                decl=[["source", 1], ["dest", 1]],
                lanes_covered=[],
            ),
            _case(
                "dead vector write survives to the next label for hazards",
                [
                    "v_mov_b32 %dead, 41",
                    "v_mov_b32 %running, 43",
                    "v_xor_b32 %running, %running, %tid",
                    ".Lwindow_end:",
                    "v_mov_b32 %later, 45",
                    "v_add_nc_u32 %later, %later, %running",
                ],
                decl=[["dead", 1], ["running", 1], ["later", 1]],
                lanes_covered=[],
            ),
            _case(
                "loop back edge keeps carried and body values simultaneously live",
                [
                    "v_mov_b32 %carried, 0",
                    ".Lalloc_head:",
                    "v_xor_b32 %carried, %carried, %tid",
                    "v_mov_b32 %body, 47",
                    "v_add_nc_u32 %body, %body, %carried",
                    "s_cbranch_scc1 .Lalloc_head",
                    "v_sub_nc_u32 %exit_value, %carried, %body",
                ],
                decl=[["carried", 1], ["body", 1], ["exit_value", 1]],
                lanes_covered=[],
            ),
            _case(
                "FMAC destination is both input and output for liveness",
                [
                    "v_mov_b32 %factor, 49",
                    "v_mov_b32 %acc, 51",
                    "v_fmac_f32 %acc, %factor, %factor",
                    "v_mov_b32 %consumer, 0",
                    "v_add_nc_u32 %consumer, %consumer, %acc",
                ],
                decl=[["factor", 1], ["acc", 1], ["consumer", 1]],
                lanes_covered=[],
            ),
            _case(
                "waiting for younger global load also retires older load",
                [
                    "buffer_load_b32 %older, %tid, %resource[0]/4, 0 offen",
                    "buffer_load_b32 %younger, %tid, %resource[0]/4, 0 offen",
                    "@wait %younger",
                    ".Lloads:",
                    "v_add_nc_u32 %value, %younger, %tid",
                    "v_mov_b32 %reuse, 53",
                    "v_xor_b32 %reuse, %reuse, %value",
                    "v_add_nc_u32 %reuse, %reuse, %tid",
                ],
                decl=[["older", 1], ["younger", 1], ["value", 1], ["reuse", 1]],
                lanes_covered=[],
            ),
            _case(
                "waiting for older load leaves younger destination in flight",
                [
                    "buffer_load_b32 %older, %tid, %resource[0]/4, 0 offen",
                    "buffer_load_b32 %younger, %tid, %resource[0]/4, 0 offen",
                    "@wait %older",
                    ".Lloads:",
                    "v_add_nc_u32 %value, %older, %tid",
                    "v_mov_b32 %reuse, 53",
                    "v_xor_b32 %reuse, %reuse, %value",
                    "v_add_nc_u32 %reuse, %reuse, %tid",
                ],
                decl=[["older", 1], ["younger", 1], ["value", 1], ["reuse", 1]],
                lanes_covered=[],
            ),
            _case(
                "discard consumes the value at its explicit block endpoint",
                [
                    "v_mov_b32 %retained, 55",
                    ".Ldiscard_scope:",
                    "v_mov_b32 %other, %tid",
                    "v_xor_b32 %other, %other, %tid",
                    "; discard %retained: intentional end of use",
                ],
                decl=[["retained", 1], ["other", 1], ["unused", 1]],
                lanes_covered=[],
            ),
        ]
    )
    return plans


def waits_plans():
    """Author explicit wait, CFG and pipeline cases."""
    plans = []
    plans.append(
        _case(
            "two independent VMEM loads have distinct register waits",
            [
                "global_load_b32 v7, v[12:13], off",
                "global_load_b32 v8, v[12:13], off offset:20",
                *_wait4("v7"),
                "v_mul_f32 v9, v7, 2.0",
                *_wait4("v8"),
                "v_add_f32 v9, v8, v9",
                "s_endpgm",
            ],
        )
    )
    plans.append(
        _case(
            "younger VMEM wait implicitly completes the older load",
            [
                "global_load_b32 v7, v[12:13], off",
                "global_load_b32 v8, v[12:13], off offset:20",
                *_wait4("v8"),
                "v_mul_f32 v9, v8, 3.0",
                "v_add_f32 v9, v7, v9",
                "s_endpgm",
            ],
        )
    )
    plans.append(
        _case(
            "join wait uses the fewest younger operations among incoming paths",
            [
                "global_load_b32 v10, v[14:15], off",
                "s_cmp_lg_u32 s17, 0",
                "s_cbranch_scc0 .Lwait_join",
                "global_load_b32 v11, v[14:15], off offset:8",
                *_wait4("v11"),
                "v_mul_f32 v11, 3.0, v11",
                "global_store_b32 v[14:15], v11, off offset:8",
                ".Lwait_join:",
                *_wait4("v10"),
                "v_add_f32 v10, 2.0, v10",
                "global_store_b32 v[14:15], v10, off",
                "s_endpgm",
            ],
        )
    )
    plans.append(
        _case(
            "load issued on only one arm is absent from the other path",
            [
                "global_load_b32 v10, v[14:15], off",
                "s_cmp_lg_u32 s17, 0",
                "s_cbranch_scc0 .Lwait_join",
                "global_load_b32 v11, v[14:15], off offset:8",
                ".Lwait_join:",
                *_wait4("v10"),
                "v_add_f32 v10, 2.0, v10",
                "global_store_b32 v[14:15], v10, off",
                "s_endpgm",
            ],
        )
    )
    plans.append(
        _case(
            "separate counters still imply a wait for an unmarked DS consumer",
            [
                "global_load_b32 v16, v[18:19], off",
                "s_load_b32 s20, s[22:23], 16",
                "ds_load_b32 v17, v0 offset:8",
                "ds_load_b32 v21, v0 offset:12",
                *_wait4("v16 s20"),
                "v_add_f32 v16, s20, v16",
                *_wait4("v17"),
                "v_add_f32 v16, v17, v16",
                "v_add_f32 v16, v21, v16",
                "global_store_b32 v[18:19], v16, off",
                "s_endpgm",
            ],
            refuses=True,
        )
    )
    plans.append(
        _case(
            "scalar KM counter always drains to zero rather than in-order distance",
            [
                "s_load_b32 s24, s[26:27], 4",
                "s_load_b32 s25, s[26:27], 20",
                *_wait4("s24"),
                "s_or_b32 s28, s24, 3",
                "s_add_co_u32 s28, s25, s28",
                "s_endpgm",
            ],
        )
    )
    plans.append(
        _case(
            "all requests drain both VM load and store counter groups",
            [
                "global_load_b32 v22, v[24:25], off",
                "global_store_b32 v[24:25], v23, off offset:16",
                "@wait load all",
                "@wait store all",
                "v_xor_b32 v22, v22, 7",
                "s_endpgm",
            ],
        )
    )
    plans.append(
        _case(
            "left count misses a younger register that its consumer implies",
            [
                "global_load_b32 v26, v[28:29], off",
                "global_load_b32 v27, v[28:29], off offset:4",
                "global_load_b32 v30, v[28:29], off offset:8",
                "@wait load left 2",
                "v_add_f32 v31, v26, v27",
                "v_add_f32 v31, v30, v31",
                "s_endpgm",
            ],
            refuses=True,
        )
    )
    plans.append(
        _case(
            "same counter overwrite of an in-flight destination is refused",
            [
                "global_load_b32 v32, v[34:35], off",
                "global_load_b32 v32, v[34:35], off offset:24",
                *_wait4("v32"),
                "v_add_f32 v33, v32, 1.0",
                "s_endpgm",
            ],
            refuses=True,
        )
    )
    plans.append(
        _case(
            "ALU overwrite of an in-flight destination is refused",
            ["global_load_b32 v36, v[38:39], off", "v_mov_b32 v36, 57", "s_endpgm"],
            refuses=True,
        )
    )
    plans.append(
        _case(
            "adjacent VMEM and DS requests remain target ordered waits",
            [
                "global_load_b32 v40, v[42:43], off",
                "ds_load_b32 v41, v0 offset:16",
                *_wait4("v40"),
                *_wait4("v41"),
                "v_add_f32 v44, v40, v41",
                "s_endpgm",
            ],
        )
    )
    plans.append(
        _case(
            "handwritten VM wait merges with a weaker generated request",
            [
                "global_load_b32 v45, v[46:47], off",
                "global_load_b32 v48, v[46:47], off offset:4",
                "s_wait_loadcnt 1",
                *_wait4("v48"),
                "v_add_f32 v49, v45, v48",
                "s_endpgm",
            ],
        )
    )
    plans.append(
        _case(
            "repeated exhausted request inserts no redundant wait",
            [
                "global_load_b32 v50, v[52:53], off",
                *_wait4("v50"),
                *_wait4("v50"),
                "v_mul_f32 v54, v50, 2.0",
                "s_endpgm",
            ],
        )
    )
    plans.append(
        _case(
            "double buffered loop agrees on entry and back edge distances",
            [
                "s_mov_b32 s30, 1",
                "global_load_b32 v55, v[56:57], off",
                ".Ldouble_buffer:",
                "global_load_b32 v58, v[56:57], off offset:4",
                *_wait4("v55"),
                "v_add_f32 v59, v55, v59",
                "global_load_b32 v55, v[56:57], off offset:8",
                *_wait4("v58"),
                "v_add_f32 v59, v58, v59",
                "s_add_co_u32 s30, s30, 2",
                "s_cmp_lt_u32 s30, s31",
                "s_cbranch_scc1 .Ldouble_buffer",
                *_wait4("v55"),
                "v_add_f32 v59, v55, v59",
                "s_endpgm",
            ],
        )
    )
    plans.append(
        _case(
            "post loop marker observes loop body issue history",
            [
                "s_mov_b32 s32, 0",
                "global_load_b32 v60, v[62:63], off",
                "global_load_b32 v64, v[62:63], off offset:4",
                ".Lpost_loop:",
                "global_load_b32 v61, v[62:63], off offset:8",
                *_wait4("v61"),
                "v_add_f32 v65, v61, v65",
                "s_add_co_u32 s32, s32, 3",
                "s_cmp_lt_u32 s32, s33",
                "s_cbranch_scc1 .Lpost_loop",
                *_wait4("v60"),
                "v_add_f32 v65, v60, v65",
                "v_add_f32 v65, v64, v65",
                "s_endpgm",
            ],
        )
    )
    plans.append(
        _case(
            "typed data atomic cannot overtake another outstanding data atomic",
            [
                "global_atomic_add_f32 v[66:67], v68, off",
                "global_atomic_add_u32 v[66:67], v69, off offset:4",
                "s_endpgm",
            ],
            refuses=True,
        )
    )
    plans.append(
        _case(
            "two returnless data atomics may remain outstanding together",
            [
                "global_atomic_add_f32 v[70:71], v72, off",
                "global_atomic_add_f32 v[70:71], v73, off offset:4",
                "s_endpgm",
            ],
        )
    )
    plans.append(
        _case(
            "pipeline nonconservative loop rejects stronger entry requirement",
            [],
            pipeline=True,
            refuses=True,
            source=_pipeline_source(False),
        )
    )
    plans.append(
        _case(
            "pipeline conservative loop materializes entry waits every iteration",
            [],
            pipeline=True,
            source=_pipeline_source(True),
        )
    )
    plans.append(
        _case(
            "path sensitive four step VM staircase after conditionally issued loads",
            [
                "s_cmp_lg_u32 s34, 0",
                "s_cbranch_scc0 .Lstair_join",
                "global_load_b32 v74, v[80:81], off",
                "global_load_b32 v75, v[80:81], off offset:4",
                "global_load_b32 v76, v[80:81], off offset:8",
                ".Lstair_join:",
                "global_load_b32 v77, v[80:81], off offset:12",
                "global_load_b32 v78, v[80:81], off offset:16",
                "global_load_b32 v79, v[80:81], off offset:20",
                "global_load_b32 v82, v[80:81], off offset:24",
                *_wait4("v77"),
                "v_add_f32 v83, v77, v83",
                *_wait4("v78"),
                "v_add_f32 v83, v78, v83",
                *_wait4("v79"),
                "v_add_f32 v83, v79, v83",
                *_wait4("v82"),
                "v_add_f32 v83, v82, v83",
                "s_endpgm",
            ],
        )
    )
    plans.append(
        _case(
            "staircase explicitly names conditionally older registers as well",
            [
                "s_cmp_lg_u32 s34, 0",
                "s_cbranch_scc0 .Lstair_join",
                "global_load_b32 v74, v[80:81], off",
                "global_load_b32 v75, v[80:81], off offset:4",
                "global_load_b32 v76, v[80:81], off offset:8",
                ".Lstair_join:",
                "global_load_b32 v77, v[80:81], off offset:12",
                "global_load_b32 v78, v[80:81], off offset:16",
                "global_load_b32 v79, v[80:81], off offset:20",
                "global_load_b32 v82, v[80:81], off offset:24",
                *_wait4("v77 v74 v75 v76"),
                "v_add_f32 v83, v77, v83",
                *_wait4("v78"),
                "v_add_f32 v83, v78, v83",
                *_wait4("v79"),
                "v_add_f32 v83, v79, v83",
                *_wait4("v82"),
                "v_add_f32 v83, v82, v83",
                "s_endpgm",
            ],
        )
    )
    plans.append(
        _case(
            "path overwrite in a staircase exposes older in-flight destination",
            [
                "s_cmp_lg_u32 s34, 0",
                "s_cbranch_scc0 .Loverwrite_join",
                "global_load_b32 v84, v[88:89], off",
                "global_load_b32 v85, v[88:89], off offset:4",
                ".Loverwrite_join:",
                "global_load_b32 v84, v[88:89], off offset:8",
                "global_load_b32 v85, v[88:89], off offset:12",
                "global_load_b32 v86, v[88:89], off offset:16",
                *_wait4("v84"),
                "v_add_f32 v87, v84, v87",
                *_wait4("v85"),
                "v_add_f32 v87, v85, v87",
                *_wait4("v86"),
                "v_add_f32 v87, v86, v87",
                "s_endpgm",
            ],
            refuses=True,
        )
    )
    plans.append(
        _case(
            "conditional loop before staircase leaves older path registers nameable",
            [
                "s_cmp_lg_u32 s34, 0",
                "s_cbranch_scc0 .Lloop_stair_join",
                "global_load_b32 v90, v[96:97], off",
                "global_load_b32 v91, v[96:97], off offset:4",
                "s_cmp_eq_u32 s35, 0",
                "s_cbranch_scc0 .Lloop_stair_join",
                ".Lone_path_loop:",
                "global_load_b32 v92, v[96:97], off offset:8",
                *_wait4("v92"),
                "v_add_f32 v93, v92, v93",
                "s_sub_co_u32 s35, s35, 2",
                "s_cmp_eq_u32 s35, 0",
                "s_cbranch_scc0 .Lone_path_loop",
                ".Lloop_stair_join:",
                "global_load_b32 v94, v[96:97], off offset:12",
                "global_load_b32 v95, v[96:97], off offset:16",
                "global_load_b32 v98, v[96:97], off offset:20",
                *_wait4("v94 v90 v91"),
                "v_add_f32 v99, v94, v99",
                *_wait4("v95"),
                "v_add_f32 v99, v95, v99",
                *_wait4("v98"),
                "v_add_f32 v99, v98, v99",
                "s_endpgm",
            ],
        )
    )
    return plans


def _pipeline_source(conservative):
    return [
        "global_load_b32 v104, v[100:101], off",
        "global_load_b32 v102, v[100:101], off offset:20",
        f"loop rounds=s40 (conservative={conservative}):",
        "  @wait v102",
        "  v_mul_f32 v103, v102, 1.5",
        "  global_load_b32 v102, v[100:101], off offset:24",
        "  global_load_b32 v105, v[100:101], off offset:28",
        "  @wait v105",
        "  v_add_f32 v103, v105, v103",
        "@wait v104",
        "v_add_f32 v103, v104, v103",
        "global_store_b32 v[100:101], v103, off",
        "s_endpgm",
    ]


def _pipeline_body():
    """A fully spelled CFG used to compute the pipeline's final body via waits."""
    return [
        "global_load_b32 v104, v[100:101], off",
        "global_load_b32 v102, v[100:101], off offset:20",
        "s_mov_b32 s36, 0",
        "s_cmp_lt_u32 s36, s40",
        "s_cbranch_scc0 .Lk_end_2",
        ".Lk_1:",
        "@wait load regs v102",
        "v_mul_f32 v103, v102, 1.5",
        "global_load_b32 v102, v[100:101], off offset:24",
        "global_load_b32 v105, v[100:101], off offset:28",
        "@wait load regs v105",
        "v_add_f32 v103, v105, v103",
        "s_add_co_u32 s36, s36, 1",
        "s_cmp_lt_u32 s36, s40",
        "s_cbranch_scc1 .Lk_1",
        ".Lk_end_2:",
        "@wait load regs v104",
        "v_add_f32 v103, v104, v103",
        "global_store_b32 v[100:101], v103, off",
        "s_endpgm",
    ]


def solve_plans():
    """Seeded abstract CFGs, independent of all prior occupancy input rows."""
    plans = []
    lane_choices = [1, 2, 4, 15, 255, 65535, 0xFFFFFFFF, "cur", ["region", 1], ["region", 2], None]
    for case_index in range(16):
        rng = random.Random(0x51A700 + case_index * 7919)
        n = 6 + case_index % 8
        key_count = 8
        succ = []
        for row in range(n):
            if row == n - 1:
                edges = []
            elif row == n - 2:
                edges = [n - 1]
            else:
                edges = [row + 1]
                if rng.random() < 0.43:
                    edges.append(rng.randrange(row + 2, n))
            if 2 < row < n - 2 and rng.random() < 0.28:
                edges.append(rng.randrange(0, row))
            if case_index in (2, 7, 12) and row == n // 2:
                edges.append(row)
            succ.append(sorted(set(edges)))

        def keys(rng=rng, key_count=key_count):
            """Construct keys inputs."""
            return sorted(k for k in range(key_count) if rng.random() < 0.14)

        rows = {
            field: [] for field in ("reads", "reads_every_lane", "writes", "kills", "lane_kills")
        }
        for _ in range(n):
            for field in rows:
                rows[field].append(keys())

        def lane(rng=rng):
            """Construct lane inputs."""
            return rng.choice(lane_choices[:-1])

        def exec_lane(rng=rng):
            """Construct exec lane inputs."""
            return rng.choice(lane_choices)

        held = [keys() if rng.random() < 0.24 else [] for _ in range(n)]
        pinned = sorted(rng.sample(range(key_count), case_index % 3)) if case_index % 4 == 3 else []
        plans.append(
            {
                "name": f"seeded occupancy graph {case_index:02d} ({n} rows)",
                "succ": succ,
                **rows,
                "read_lanes": [lane() for _ in range(n)],
                "exec_in": [exec_lane() for _ in range(n)],
                "writes_exec": [rng.random() < 0.17 for _ in range(n)],
                "exec_out": [
                    rng.choice(lane_choices) if rng.random() < 0.22 else None for _ in range(n)
                ],
                "pinned": pinned,
                "held": held,
                "window_ends": [rng.random() < 0.18 for _ in range(n)],
            }
        )
    return plans
