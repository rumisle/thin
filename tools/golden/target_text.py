"""Author parser/assembler and hazard programs; leave target/encoding facts recorded."""

from __future__ import annotations

import itertools
import json
import subprocess

from tools.golden import ROOT, encode, no_checked_data
from tools.golden.passes_programs import hazards_plans

BASE = "crates/thin-targets/tests/data/target/"
OUTPUTS = tuple(BASE + file + ".json" for file in ("read_s", "line_syntax", "hazards"))


def native(requests):
    """Evaluate source requests with the actual target reader, assembler and hazard model."""
    result = subprocess.run(
        ["cargo", "run", "-q", "-p", "thin-targets", "--example", "golden_text"],
        cwd=ROOT,
        input=json.dumps(requests),
        text=True,
        capture_output=True,
        timeout=120,
        check=True,
    )
    return json.loads(result.stdout)


def read_plans():
    """Independent assembly translation units and malformed-directive perturbations."""
    return [
        ("ordinary_text", ".text\nreader_tile:\n\ts_nop 3\n\ts_endpgm\n"),
        (
            "target_comments_loop",
            (
                '.amdgcn_target "amdgcn-amd-amdhsa--gfx1201"\n'
                ".text\n"
                ".Ltile:\n"
                "\ts_mov_b32 s22, 17 // seed\n"
                "\ts_xor_b32 s22, s22, 3 ; update\n"
                ".p2align 5\n"
                "\ts_cbranch_scc1 .Ltile\n"
            ),
        ),
        ("code_without_text", "\ts_nop 3\n"),
        ("foreign_directive", ".text\n\ts_nop 2\n.inject 9\n"),
        ("alignment_fill", ".text\n.p2align 5, 0xff\n"),
        ("alignment_too_large", ".text\n.p2align 14\n"),
        ("duplicate_target", '.amdgcn_target "first"\n// gap\n.amdgcn_target "second"\n'),
        (
            "descriptor_block",
            (
                ".text\n\ts_nop 2\n\ts_endpgm\n.section "
                ".rodata\n.amdhsa_kernel "
                "reader_tile\n.amdhsa_next_free_vgpr "
                "7\n.amdhsa_next_free_sgpr "
                "12\n.end_amdhsa_kernel\n"
            ),
        ),
        (
            "duplicate_descriptor_field",
            (
                ".amdhsa_kernel "
                "reader_tile\n.amdhsa_next_free_sgpr "
                "8\n.amdhsa_next_free_sgpr "
                "12\n.end_amdhsa_kernel\n"
            ),
        ),
        (
            "metadata_document",
            ".amdgpu_metadata\n---\nlayout:\n  - 7\n  - 11\n...\n.end_amdgpu_metadata\n",
        ),
        ("metadata_without_markers", ".amdgpu_metadata\nlayout: 7\n.end_amdgpu_metadata\n"),
        ("unterminated_metadata", ".amdgpu_metadata\n---\nlayout: 7\n...\n"),
        ("text_sections", ".text\nleft:\n\ts_nop 3\n.section .rodata\n.text\nright:\n\ts_endpgm\n"),
        ("code_in_rodata", ".section .rodata\n\ts_nop 7\n"),
    ]


def syntax_plans(vocab):
    """Grammar products and instruction/assembly plans."""
    ints = [
        "00",
        "37",
        "-37",
        "+29",
        "0x2B",
        "0X2b",
        "0o35",
        "0b1101",
        "2_048",
        "2__7",
        "_9",
        "3_",
        "023",
        "0000",
        "\t19 ",
        "0x_ab",
        "0b",
        "",
        "+",
        "2.75",
        "0b3",
        str(1 << 140),
    ]
    classify = [
        "37",
        "-37",
        "0x2b",
        "-0x2B",
        "0X2b",
        "2.0",
        "-3.",
        ".75",
        "2e4",
        "2.0e-4",
        "2.e+3",
        "-.75e2",
        "e7",
        "2.0e",
        "tile",
        ".Ltile",
        "Tile7",
        "_row",
        "tile.row",
        ".Ltile_row",
        "8tile",
        ".",
        ".row",
    ]
    nv, ns = vocab["reg_count"]["v"], vocab["reg_count"]["s"]
    operands = [
        "v37",
        f"v{nv - 1}",
        f"v{nv}",
        f"s{ns - 1}",
        f"s{ns}",
        "s[30:31]",
        "s[31:30]",
        "v[32:39]",
        f"v[{nv - 3}:{nv + 1}]",
        "a1",
        "vcc",
        "vcc_lo",
        "exec_lo",
        "m0",
        "null",
        "scc",
        "ttmp9",
        "ttmp7",
        "off",
        "-v37",
        "--v37",
        "|v37|",
        "-|v37|",
        "|",
        "-37",
        "-0x2b",
        "2.0",
        "-3.75",
        "0.75",
        ".Ltile",
        "tile",
        "hwreg(HW_REG_MISSING)",
        "gpr_idx(SRC1,SRC2)",
        "gpr_idx(SRC4)",
        "gpr_idx(DST",
        "s[30:31",
        "v[row:col]",
        "!",
        "2e4",
        "-Tile7",
        "vccz",
        "execz",
        "exec",
        "-.75",
        "|Tile7|",
    ]
    # Native vocabulary expands positive hardware-register and named-register coverage.
    operands += [f"hwreg({name})" for name, _ in vocab["hwreg"]]
    operands += [f"hwreg({name}, 1, 7)" for name, _ in vocab["hwreg"]]
    operands += [name for name, _, _ in vocab["specials"]]
    a = "v_mul_f32 v42, v43, v44"
    g = "global_load_b64 v[36:37], v38, s[40:41]"
    lines = [
        a,
        "s_xor_b32 s30, s31, 0x35",
        g + " offset:-32 th:TH_LOAD_RT scope:SCOPE_SYS",
        g + " offset:-32 offset:64",
        "ds_store_b64 v52, v[54:55] offset:65535",
        "v_mul_f32_dpp v60, v61, v62 quad_perm:[2,3,0,1] row_mask:0x7 bank_mask:0xb bound_ctrl:0",
        "v_mul_f32_dpp v60, v61, v62 row_mirror",
        "v_pk_mul_f16 v64, v65, v66 op_sel:[0,1] op_sel_hi:[1,0] clamp",
        "s_wait_alu depctr_hold_cnt(1) depctr_va_vdst(0)",
        "s_wait_alu depctr_hold_cnt(1)&depctr_va_vdst(0x0)",
        "s_wait_alu depctr_hold_cnt(1),depctr_va_vdst(0)",
        "s_wait_alu 0xfff7",
        "s_wait_alu depctr_hold_cnt(0) depctr_hold_cnt(1)",
        "s_wait_alu lgkmcnt(0)",
        "s_wait_alu depctr_va_vdst(-3)",
        "s_wait_loadcnt 0",
        "s_endpgm",
        "s_cbranch_scc1 .Ltile",
        a + " discarded",
        a + " Audit:3",
        a + " scope:SCOPE_INVALID",
        a + " Indices:[0,z]",
        "s_getreg_b32 s36, hwreg(HW_REG_MISSING, 3, 2)",
        "v_mov_b32 v39, s[44:45]",
        "s_load_b64 s[36:37], s[38:39], 0x8 offset:12 offen idxen",
        "v_sub_f32 v62, -v63, |v64|",
        "v_sub_f32 v70,v71,,,v72",
    ]
    assembly = [
        {
            "name": "loop_and_forward_exit",
            "items": [
                ["label", ".Lbegin"],
                ["ins", "s_xor_b32 s20, s21, 1"],
                ["ins", "s_cbranch_scc1 .Lbegin"],
                ["ins", "s_branch .Lfinish"],
                ["label", ".Lfinish"],
                ["ins", "v_mul_f32 v42, v43, v44"],
                ["ins", "s_endpgm"],
            ],
        },
        {
            "name": "aligned_vector_body",
            "items": [
                ["ins", "v_mul_f32 v42, v43, v44"],
                ["align", 5],
                ["label", ".Laligned"],
                ["ins", "s_endpgm"],
            ],
        },
        {
            "name": "aligned_literal",
            "items": [["ins", "s_and_b32 s34, s35, 0x5a00f"], ["align", 4], ["ins", "s_endpgm"]],
        },
        {
            "name": "unresolved_branch",
            "items": [["ins", "s_nop 3"], ["ins", "s_branch .Lmissing_tile"]],
        },
        {
            "name": "duplicate_label",
            "items": [["label", ".Ltile"], ["ins", "s_nop 3"], ["label", ".Ltile"]],
        },
        {
            "name": "unsupported_opcode",
            "items": [["ins", "s_nop 3"], ["ins", "v_no_such_instruction v54"]],
        },
        {
            "name": "alignment_then_missing_label",
            "items": [
                ["ins", "s_branch .Lgap"],
                ["align", 11],
                ["align", 11],
                ["align", 12],
                ["ins", "s_nop 3"],
            ],
        },
    ]
    for count, end in ((32768, True), (32767, False), (32767, True)):
        items = [
            ["ins", "s_branch .Lpadding_end"],
            ["repeat", count, "s_nop 3"],
            ["label", ".Lpadding_end"],
        ]
        if end:
            items.append(["ins", "s_endpgm"])
        assembly.append(
            {"name": f"forward_boundary_{count}_end_{end}", "items": items, "summary": True}
        )
    return {
        "py_int": ints,
        "classify": classify,
        "operands": operands,
        "lines": lines,
        "assemble": assembly,
    }


def hazard_programs():
    """Fresh hazard-family programs plus precise boundaries/special-register cases."""
    base = [(plan["name"], list(plan["input"])) for plan in hazards_plans()]
    extra = []

    def add(name, *lines):
        """Construct add inputs."""
        extra.append((name, list(lines)))

    for n in (4, 3):
        add(
            f"DS tracking boundary {n}",
            "v_add_co_ci_u32_e64 v34, s24, 1, v35, s24",
            *(["ds_nop"] * n),
            "s_getpc_b64 s[24:25]",
            "s_xor_b32 s27, s24, 0x33",
            "s_endpgm",
        )
    add(
        "wrong ALU wait field leaves SALU state",
        "v_mov_b32 v38, s20",
        "s_cselect_b32 s20, 7, 11",
        "s_wait_alu depctr_va_sdst(0)",
        "s_and_b32 s21, s20, 15",
        "s_endpgm",
    )
    add(
        "indirect branch drains vector scalar destination",
        "v_cndmask_b32_e64 v42, v43, v44, s28",
        "v_cmp_lt_f32_e64 s29, v45, v42",
        "s_setpc_b64 s[48:49]",
    )
    add(
        "swappc conservatively tracks scalar pairs",
        "v_mul_f32 v42, s28, v43",
        "s_swappc_b64 s[42:43], s[30:31]",
        "s_mov_b32 s47, 7",
        "s_xor_b32 s46, s47, 5",
        "s_endpgm",
    )
    add(
        "whole pair rewritten then consumed",
        "v_cndmask_b32_e64 v40, v41, v42, s[14:15]",
        "s_mov_b64 s[14:15], -1",
        "v_cndmask_b32_e64 v43, v40, v45, s[14:15]",
        "s_endpgm",
    )
    merged = list(base[10][1])
    merged[-2] = "s_mov_b32 s46, vcc_lo"
    add("joined condition consumed only by SALU", *merged)
    for special in ("ttmp7", "ttmp9", "m0"):
        add(
            f"special scalar tracking {special}",
            f"v_mul_f32 v46, {special}, v47",
            f"s_mov_b32 {special}, 11",
            f"s_xor_b32 s36, {special}, 0x35",
            "s_endpgm",
        )
    for same, read in ((True, None), (False, "s60"), (False, "s28")):
        dest = "s28" if same else "s29"
        between = [f"s_mov_b32 s40, {read}"] if read else []
        add(
            f"readfirstlane tracking {dest} commit {read}",
            "v_mov_b32 v54, s28",
            f"v_readfirstlane_b32 {dest}, v54",
            *between,
            f"v_mul_f32 v55, {dest}, v56",
            "s_endpgm",
        )
    add(
        "CMPX then permlane64",
        "v_cmpx_ne_u32 7, v50",
        "v_permlane64_b32 v51, v52",
        "s_endpgm",
    )
    add(
        "conditional wait does not retire every path",
        "v_xor_b32 v62, s20, v63",
        "s_mov_b32 s20, 5",
        "s_mov_b32 s20, 11",
        "s_cbranch_scc1 .Ljoin_wait",
        "s_wait_alu depctr_sa_sdst(0)",
        ".Ljoin_wait:",
        "s_xor_b32 s21, s20, 7",
        "s_endpgm",
    )
    add(
        "CMPX crosses a branch before permlane",
        "v_mov_b32 v50, 7",
        "v_cmpx_ne_u32 7, v50",
        "s_branch .Lafter_cmpx",
        ".Lafter_cmpx:",
        "s_xor_b32 s12, s13, 15",
        "v_permlane16_b32 v51, v52, s12, s14",
        "s_endpgm",
    )
    add(
        "three transcendental producers retain oldest dependency",
        "v_log_f32 v90, v91",
        "v_log_f32 v92, v93",
        "v_log_f32 v94, v95",
        "v_mul_f32 v96, v90, v97",
        "s_endpgm",
    )
    add(
        "SALU does not increase vector delay distance",
        "v_mul_f32 v80, v81, v82",
        "s_xor_b32 s45, s47, 9",
        "v_sub_f32 v83, v80, v80",
        "s_endpgm",
    )
    add(
        "VALU then TRANS dual delay",
        "v_mul_f32 v80, v81, v82",
        "v_log_f32 v84, v85",
        "v_fma_f32 v88, v80, v80, v84",
        "s_endpgm",
    )
    return base + extra


def generate():
    """Construct every input before evaluating it through native target APIs."""
    with no_checked_data():
        vocabulary = native([{"op": "vocab", "arg": None}])[0]
        reads = read_plans()
        syntax = syntax_plans(vocabulary)
        programs = hazard_programs()
        requests = [("read_s", text) for _, text in reads]
        for section, op in (
            ("py_int", "int"),
            ("classify", "classify"),
            ("operands", "operand"),
            ("lines", "line"),
            ("assemble", "assemble"),
        ):
            requests += [(op, arg) for arg in syntax[section]]
        requests += [("hazards", lines) for _, lines in programs]
        delays = sorted(
            {
                a | (skip << 4) | (b << 7)
                for a, skip, b in itertools.product(range(16), range(3), (0, 1, 5, 6))
            }
        )
        requests += [("delay_text", value) for value in delays]
        answers = iter(native([{"op": op, "arg": arg} for op, arg in requests]))
        read_data = [next(answers) for _ in reads]
        syntax_data = {section: [next(answers) for _ in plan] for section, plan in syntax.items()}
        hazards = []
        for name, _ in programs:
            result = next(answers)
            result["name"] = name
            hazards.append(result)
        hazard_data = {"cases": hazards, "delay_text": [next(answers) for _ in delays]}
        try:
            next(answers)
        except StopIteration:
            pass
        else:
            raise AssertionError("unconsumed native target answers")
        docs = read_data, syntax_data, hazard_data
        return {path: encode(doc) for path, doc in zip(OUTPUTS, docs, strict=True)}
