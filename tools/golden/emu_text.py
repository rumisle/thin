"""Assembly and claim programs for the emulator reader.

The native emulator computes parser results, claim diagnostics and messages.
Numeric result tables are not generated here.
"""

from __future__ import annotations

import json
import subprocess

from tools.golden import ROOT, encode, no_checked_data

BASE = "crates/thin-emu/tests/data/emu/"
OUTPUTS = (BASE + "reading.json", BASE + "claims.json")


READING_PLANS = (
    {
        "name": "register_forms_and_specials",
        "behavior": (
            "negated and absolute vector sources, "
            "register ranges, and named "
            "exec/m0/vcc/scc/off operands"
        ),
        "classification": "success",
        "text": (
            "\tv_fma_f32 v6, -v7, |v8|, v9\n"
            "\tv_mov_b64 v[10:11], s[12:13]\n"
            "\ts_mov_b32 m0, exec_lo\n"
            "\tv_cmp_ne_u32 vcc_lo, v14, v15\n"
            "\ts_cselect_b32 s26, s27, scc\n"
            "\tglobal_store_b64 v[20:21], v[22:23], off\n"
        ),
    },
    {
        "name": "numeric_operand_spellings",
        "behavior": (
            "negative integers, decimal and hexadecimal "
            "floats, movk sign extension, rounded "
            "decimal floats, and the unsigned 32-bit "
            "maximum"
        ),
        "classification": "success",
        "text": (
            "\tv_mov_b32 v12, -7\n"
            "\tv_mov_b32 v13, 1.25\n"
            "\tv_mov_b32 v14, 0x40490fdb\n"
            "\ts_movk_i32 s11, -23\n"
            "\tv_sub_f32 v15, -2.5, v13\n"
            "\tv_mul_f32 v16, 0.2, v13\n"
            "\tv_mov_b32 v17, 4294967295\n"
        ),
    },
    {
        "name": "scalar_and_vector_modifiers",
        "behavior": (
            "integer memory offset, bare clamp flag, and a list-valued packed-operation modifier"
        ),
        "classification": "success",
        "text": (
            "\tglobal_load_b64 v[12:13], v14, s[16:17] offset:24\n"
            "\tv_sub_f32 v18, v19, v20 clamp\n"
            "\tv_pk_add_f32 v[22:23], v[24:25], v[26:27] op_sel_hi:[1,0]\n"
        ),
    },
    {
        "name": "trivia_sections_and_control_labels",
        "behavior": (
            "column-zero sections with trimming, inline "
            "comment precedence and empty comments, "
            "dotted labels, branch operands, and barrier "
            "trivia"
        ),
        "classification": "success",
        "text": (
            "\ts_mov_b32 s21, 3\n"
            "; undefined: v40 (pending)\n"
            ";note_without_padding   \n"
            ".Lentry:\n"
            "exit_alias:\n"
            "\tv_mul_f32 v4, v5, v6 // vector note ; discarded\n"
            "\ts_nop 3 ; scalar note\n"
            "\ts_nop 4 //\n"
            "\ts_nop 5 ; ignored // trailing note\n"
            "\t; indented directive-looking line\n"
            ".Lsend:\n"
            "\ts_cbranch_scc1 .Lentry\n"
            "\ts_branch exit_alias\n"
            "\ts_barrier // LDS_FENCE T\n"
        ),
    },
    {
        "name": "later_label_definitions_win",
        "behavior": (
            "duplicate labels overwrite prior positions "
            "while dotted and undotted spellings remain "
            "distinct"
        ),
        "classification": "success",
        "text": ("root:\n\ts_nop 6\n.dup:\n\ts_nop 7\n.dup:\n\ts_nop 8\ndup:\n"),
    },
    {
        "name": "complete_assembly_translation_unit",
        "behavior": (
            "assembler directives and metadata are "
            "skipped while code, labels, comments, an "
            "LDS section, and the descriptor SGPR count "
            "are retained"
        ),
        "classification": "success",
        "text": (
            '\t.amdgcn_target "amdgcn-amd-amdhsa--T"\n'
            "\t.amdhsa_code_object_version 6\n"
            "\t.text\n"
            "\t.protected\treader_kernel\n"
            "\t.globl\treader_kernel\n"
            "\t.p2align\t7\n"
            "\t.type\treader_kernel,@function\n"
            "reader_kernel:\n"
            "; lds: 1024 B\n"
            "\tv_mov_b32 v9, 11                                      // seed\n"
            ".Lagain:\n"
            "\ts_add_u32 s7, s7, 2\n"
            "\ts_cbranch_scc0 .Lagain\n"
            "\ts_endpgm\n"
            ".Lreader_done:\n"
            "\t.size\treader_kernel, .Lreader_done-reader_kernel\n"
            '\t.section\t.rodata,"a",@progbits\n'
            "\t.p2align\t5, 0x0\n"
            "\t.amdhsa_kernel reader_kernel\n"
            "\t\t.amdhsa_user_sgpr_count 6\n"
            "\t\t.amdhsa_next_free_vgpr 10\n"
            "\t.end_amdhsa_kernel\n"
            "\t.amdgpu_metadata\n"
            "---\n"
            "amdhsa.version:\n"
            "  - 1\n"
            "  - 3\n"
            "...\n"
            "\t.end_amdgpu_metadata\n"
            '; thin.manifest {"reader":true}\n'
        ),
    },
    {
        "name": "unknown_special_register",
        "behavior": "a syntactically named but unmodelled special register is rejected",
        "classification": "refusal",
        "text": "\ts_nop 2\n\ts_mov_b32 s1, execz\n",
    },
    {
        "name": "malformed_operand_location",
        "behavior": (
            "the line reader reports an unrecognised operand with its source line and spelling"
        ),
        "classification": "refusal",
        "text": "\ts_nop 3\n\tv_mov_b32 v8, @address\n",
    },
    {
        "name": "vector_register_limit",
        "behavior": "v256 is refused as one past the final numbered vector register",
        "classification": "refusal",
        "text": "\tv_mov_b32 v8, v256\n",
    },
    {
        "name": "duplicate_clamp",
        "behavior": "repeating a modifier on one instruction is refused",
        "classification": "refusal",
        "text": "\tv_sub_f32 v30, v31, v32 clamp clamp\n",
    },
)


CLAIM_PLANS = (
    {
        "name": "uniform_register_inventory",
        "behavior": (
            "uniform scalar, range, and condition "
            "registers are collected while an "
            "unallocated register pattern is ignored"
        ),
        "classification": "success",
        "text": (
            "\ts_mov_b32 s17, 9\n"
            "; uniform: s17, s[20:22], scc (shared address)\n"
            "; uniform: s:x#9[3]x7 (unplaced scratch)\n"
            "\ts_nop 6\n"
        ),
    },
    {
        "name": "adjacent_uniform_claims_merge",
        "behavior": (
            "adjacent uniform claims merge registers and "
            "text, and a later claim anchors at its own "
            "instruction"
        ),
        "classification": "success",
        "text": (
            "\ts_mov_b32 s12, 4\n"
            "; uniform: s12 (base)\n"
            "; uniform: s13 (stride)\n"
            "\ts_add_u32 s18, s19, s20\n"
            "\ts_nop 2\n"
            "; uniform: vcc_lo, s[14:16] (mask and descriptor)\n"
            "\ts_endpgm\n"
        ),
    },
    {
        "name": "undefined_register_inventory",
        "behavior": (
            "undefined vector, vector-range, scalar, and "
            "accumulator-range registers are collected "
            "while an unplaced pattern is ignored"
        ),
        "classification": "success",
        "text": (
            "\tv_mov_b32 v11, 2\n"
            "; undefined: v12, v[14:16], s22, a[4:6] (dead after spill)\n"
            "; undefined: v:x#5[2]x3 (unplaced)\n"
            "\ts_nop 7\n"
        ),
    },
    {
        "name": "decimal_assume_intervals",
        "behavior": (
            "vector and scalar-range assume claims "
            "retain decimal bounds including the "
            "unsigned 32-bit maximum"
        ),
        "classification": "success",
        "text": (
            "\tv_mov_b32 v8, 1\n"
            "; assume: v9 in [2, 31] (small lane code)\n"
            "; assume: s[30:31] in [7, 4294967295] (unsigned payload)\n"
            "\ts_nop 4\n"
        ),
    },
    {
        "name": "hexadecimal_assume_bound_refused",
        "behavior": (
            "an assume interval with a hexadecimal lower bound is not a valid decimal claim"
        ),
        "classification": "refusal",
        "text": (
            "\tv_mov_b32 v4, 0\n"
            "; assume: v7 in [0x10, 63] (hex lower bound)\n"
            "; assume: s8 in [1, 9] (not reached)\n"
            "\ts_nop 3\n"
        ),
    },
    {
        "name": "assume_requires_explanation",
        "behavior": "an assume claim without a parenthesised explanation is refused",
        "classification": "refusal",
        "text": "\ts_mov_b32 s14, 2\n; assume: v15 in [1, 8]\n\ts_nop 5\n",
    },
    {
        "name": "uniform_cannot_precede_label",
        "behavior": (
            "a uniform claim immediately before a label "
            "is refused because it would apply to every "
            "incoming path"
        ),
        "classification": "refusal",
        "text": "\ts_nop 2\n; uniform: s19 (loop invariant)\n.Lheader_fresh:\n\ts_endpgm\n",
    },
    {
        "name": "uniform_rejects_vector_register",
        "behavior": "a vector register is not valid in a uniform claim",
        "classification": "refusal",
        "text": "\ts_nop 4\n; uniform: v21 (whole vector)\n\ts_endpgm\n",
    },
    {
        "name": "undefined_before_second_label_refused",
        "behavior": (
            "an undefined claim between labels is refused because it lacks an instruction anchor"
        ),
        "classification": "refusal",
        "text": (
            "\ts_cbranch_scc1 .Lwrite\n"
            "\tv_mov_b32 v7, 5\n"
            ".Lwrite:\n"
            "; undefined: v9 (not written on one arm)\n"
            ".Ljoin_fresh:\n"
            "\tv_mov_b32 v10, 1\n"
        ),
    },
    {
        "name": "undefined_has_explicit_anchor",
        "behavior": (
            "an undefined claim after a label succeeds when an instruction of its own follows"
        ),
        "classification": "success",
        "text": (
            "\ts_cbranch_scc1 .Lother\n"
            "\tv_mov_b32 v6, 3\n"
            ".Lother:\n"
            "; undefined: v7 (claim anchored to explicit nop)\n"
            "\ts_nop 9\n"
            ".Ldone:\n"
            "\tv_mov_b32 v8, 0\n"
        ),
    },
    {
        "name": "assume_cannot_precede_label",
        "behavior": (
            "an assume claim immediately before a label "
            "is refused because it would check every "
            "incoming path"
        ),
        "classification": "refusal",
        "text": (
            "\ts_cbranch_scc1 .Ltarget\n"
            "; assume: v12 in [4, 9] (would cover all incoming paths)\n"
            ".Ltarget:\n"
            "\tv_mov_b32 v13, 6\n"
        ),
    },
    {
        "name": "assume_after_label_is_anchored",
        "behavior": (
            "an assume claim after a label and before "
            "its instruction is accepted at that "
            "instruction"
        ),
        "classification": "success",
        "text": (
            "\ts_cbranch_scc1 .Lentry\n"
            ".Lentry:\n"
            "; assume: v16 in [2, 6] (check after joining label)\n"
            "\tv_mov_b32 v17, 8\n"
        ),
    },
    {
        "name": "indented_claim_is_trivia",
        "behavior": "an indented comment that resembles a uniform claim is not treated as a claim",
        "classification": "success",
        "text": "\ts_nop 7\n\t; uniform: s27 (indented trivia)\n\ts_nop 8\n",
    },
    {
        "name": "descriptor_only_user_sgprs",
        "behavior": "a descriptor-only fragment supplies its declared user SGPR count",
        "classification": "success",
        "text": (".amdhsa_kernel authored_empty\n.amdhsa_user_sgpr_count 7\n.end_amdhsa_kernel\n"),
    },
    {
        "name": "plain_program_has_no_claims",
        "behavior": (
            "a program without claims returns empty claim maps and the default user SGPR count"
        ),
        "classification": "success",
        "text": "\ts_nop 10\n",
    },
    {
        "name": "undefined_on_fallthrough_edge",
        "behavior": (
            "an undefined-on-fall claim before a label "
            "is keyed by the edge from the preceding "
            "instruction to the label target"
        ),
        "classification": "success",
        "text": (
            "\ts_cbranch_scc1 .Lalternate\n"
            "\tv_mov_b32 v18, 6\n"
            "; undefined on fall: v19 (fall-through arm did not produce it)\n"
            ".Lalternate:\n"
            "\tv_mov_b32 v20, 2\n"
        ),
    },
    {
        "name": "undefined_on_taken_edge_range",
        "behavior": (
            "an undefined-on-taken register range before "
            "a conditional branch is keyed by the "
            "branch-to-label edge"
        ),
        "classification": "success",
        "text": (
            "\tv_mov_b32 v22, 4\n"
            "; undefined on taken: v[24:26] (only fall-through initializes these)\n"
            "\ts_cbranch_scc1 .Lskip\n"
            "\tv_mov_b32 v27, 1\n"
            ".Lskip:\n"
            "\tv_mov_b32 v28, 9\n"
        ),
    },
    {
        "name": "taken_edge_claim_requires_branch",
        "behavior": "an undefined-on-taken claim not followed by a branch is refused",
        "classification": "refusal",
        "text": (
            "\tv_mov_b32 v29, 2\n"
            "; undefined on taken: v30 (no branch follows)\n"
            "\tv_mov_b32 v31, 3\n"
        ),
    },
    {
        "name": "fall_edge_claim_requires_label",
        "behavior": "an undefined-on-fall claim not followed by a label is refused",
        "classification": "refusal",
        "text": (
            "\tv_mov_b32 v32, 2\n; undefined on fall: v33 (no label follows)\n\tv_mov_b32 v34, 3\n"
        ),
    },
    {
        "name": "undefined_at_label_preinstruction",
        "behavior": (
            "an undefined claim at a label but before an "
            "instruction applies at that instruction, "
            "including on a backward branch"
        ),
        "classification": "success",
        "text": (
            ".Ltop_fresh:\n"
            "; undefined: v35 (poison at every incoming edge)\n"
            "\tv_mov_b32 v36, 4\n"
            "\ts_cbranch_scc1 .Ltop_fresh\n"
        ),
    },
)


MESSAGE_PLAN = {
    "sgpr": {
        "text": "assume: s18 in [2, 9] (bounded setup)",
        "pc": 17,
        "wave": 2,
        "value": 42,
    },
    "lane": {
        "text": "assume: v23 in [1, 7] (initialized lanes)",
        "pc": 18,
        "wave": 4,
        "lane": 5,
    },
    "held": [[7], [11, 12, 13], [256]],
}


def native(requests):
    """Evaluate text and message inputs using thin-emu's native example."""
    proc = subprocess.run(
        ["cargo", "run", "-q", "-p", "thin-emu", "--example", "golden_text_emu"],
        cwd=ROOT,
        input=json.dumps(requests),
        text=True,
        capture_output=True,
        timeout=120,
    )
    if proc.returncode:
        raise RuntimeError(f"native emulator text generation failed:\n{proc.stderr}")
    answers = json.loads(proc.stdout)
    if len(answers) != len(requests):
        raise AssertionError("native emulator returned a different request count")
    return answers


def generate() -> dict[str, bytes]:
    """Build fresh source plans, then serialize only native evaluator results."""
    with no_checked_data():
        requests = [{"reading": plan["text"]} for plan in READING_PLANS]
        requests += [{"claims": plan["text"]} for plan in CLAIM_PLANS]
        answers = native(requests)
        reading_answers = answers[: len(READING_PLANS)]
        claim_answers = answers[len(READING_PLANS) :]
        from thin.emu import POISON, assume_false, held_text

        message_inputs = {
            "sgpr": {**MESSAGE_PLAN["sgpr"], "lane": None},
            "lane": {**MESSAGE_PLAN["lane"], "value": POISON},
            "held": MESSAGE_PLAN["held"],
        }
        messages = {
            "assume_false_sgpr": assume_false(**message_inputs["sgpr"]),
            "assume_false_lane": assume_false(**message_inputs["lane"]),
            "held_text": held_text(message_inputs["held"]),
        }
        for plan, answer in zip(READING_PLANS, reading_answers, strict=True):
            refused = "error" in answer
            if refused != (plan["classification"] == "refusal"):
                raise AssertionError(f"{plan['name']}: wrong reader outcome: {answer}")
        for plan, answer in zip(CLAIM_PLANS, claim_answers, strict=True):
            if "reading_error" in answer:
                raise AssertionError(f"{plan['name']}: claim program did not parse: {answer}")
            refused = any("error" in answer[kind] for kind in ("uniform", "assume", "undefined"))
            if refused != (plan["classification"] == "refusal"):
                raise AssertionError(f"{plan['name']}: wrong claim outcome: {answer}")
        reading = {
            "notation": (
                "Independently authored gfx1201 assembly reader cases. text is parsed by the "
                "native emulator. Successful cases record ins as [mnemonic, operands, fields, "
                "line, normalized text, comment], labels, section notes, and user_sgprs. "
                "Operands use native register, literal, and label records; error cases record "
                "the native diagnostic."
            ),
            "cases": [
                {"name": plan["name"], "text": plan["text"], **answer}
                for plan, answer in zip(READING_PLANS, reading_answers, strict=True)
            ],
        }
        claims = {
            "notation": (
                "Independently authored claim fragments. Each case records native uniform, "
                "assume, and undefined results (or the kind's native error), plus user_sgprs. "
                "Instruction claims are keyed by instruction index and edge claims by "
                "edge:FROM->TO. messages are native assume-failure and held-value formatting "
                "results for the explicit inputs in this generator."
            ),
            "cases": [
                {"name": plan["name"], "text": plan["text"], **answer}
                for plan, answer in zip(CLAIM_PLANS, claim_answers, strict=True)
            ],
            "message_inputs": message_inputs,
            "messages": messages,
        }
        return dict(zip(OUTPUTS, (encode(reading), encode(claims)), strict=True))
