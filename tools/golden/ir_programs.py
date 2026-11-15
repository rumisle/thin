"""IR analysis programs and structured instruction plans.

The native helper evaluates these inputs without reading stored outputs.
"""

from __future__ import annotations


def _case(name, behavior, stream, *, waves=None):
    case = {"name": name, "stream": list(stream), "behavior": behavior}
    if waves is not None:
        case["waves"] = waves
    return case


def cfg_plans():
    """Graph shapes: fall-through, labels, unresolved edges, loops and waves."""
    wave_stream = [
        "op",
        "brw aside",
        "op",
        "op",
        "aside:",
        "op",
        "brw done",
        "op",
        "op",
        "done:",
        "end",
    ]
    return [
        _case("no_rows", "an empty instruction stream", []),
        _case("linear", "straight-line fall-through to an end", ["op", "end"]),
        _case(
            "self_cycle",
            "a label and conditional back-edge in one non-label block",
            ["cycle:", "op", "op", "br cycle", "end"],
        ),
        _case(
            "split_and_join",
            "two conditional arms, an unconditional edge and a join",
            ["op", "br right", "op", "op", "jmp joined", "right:", "op", "joined:", "end"],
        ),
        _case(
            "untargeted_labels",
            "labels not targeted by a branch, including the back-edge target",
            ["head:", "op", "note:", "op", "op", "br head", "end"],
        ),
        _case(
            "adjacent_target",
            "a conditional branch whose target is the next instruction",
            ["op", "op", "br nearby", "nearby:", "op", "end"],
        ),
        _case(
            "missing_targets",
            "conditional and unconditional references to absent labels",
            ["op", "br missing_cond", "op", "op", "jmp missing_jump", "end"],
        ),
        _case(
            "unreached_regions",
            "unreached rows before a label and after an end",
            ["op", "jmp live", "op", "orphan:", "op", "live:", "op", "end", "op"],
        ),
        _case(
            "repeated_label",
            "a repeated label name with branches resolving to its first row",
            ["twice:", "op", "br twice", "twice:", "op", "op", "br twice", "end"],
        ),
        _case(
            "inner_and_outer_cycles",
            "an inner loop nested in an outer loop",
            [
                "op",
                "outer:",
                "op",
                "inner:",
                "op",
                "op",
                "br inner",
                "op",
                "br outer",
                "end",
            ],
        ),
        _case(
            "implicit_fall_off",
            "a forward branch and no end-of-program instruction",
            ["op", "br later", "op", "op", "later:", "op"],
        ),
        _case(
            "wave_resolved_taken_first",
            "two wave-resolved conditional branches, first taken and second not",
            wave_stream,
            waves=[[1, True], [6, False]],
        ),
        _case(
            "wave_resolved_taken_second",
            "the same graph with the complementary branch decisions",
            wave_stream,
            waves=[[1, False], [6, True]],
        ),
    ]


def solver_plans():
    """Four separately laid-out graphs for all five native solver modes."""
    return [
        _case(
            "loop_fixpoint",
            "capping, widening, edge filtering, all-block and backward visits on a cycle",
            ["loop_head:", "op", "op", "br loop_head", "end"],
        ),
        _case(
            "branch_merge",
            "solver merges and visit order at a diamond join",
            ["br arm", "op", "op", "jmp merge", "arm:", "op", "merge:", "op", "end"],
        ),
        _case(
            "two_nesting_levels",
            "widening and backward propagation through nested cycles",
            [
                "op",
                "outside:",
                "op",
                "inside:",
                "op",
                "br inside",
                "op",
                "op",
                "br outside",
                "end",
            ],
        ),
        _case(
            "unreached_blocks",
            "forward and backward behavior in blocks unreachable from entry",
            ["jmp selected", "op", "op", "ignored:", "op", "selected:", "end", "op", "op"],
        ),
    ]


def exec_plans():
    """EXEC constants, saved copies, unknown values, branches and loops."""
    return [
        {
            "name": "immediate_masks",
            "lines": [
                "v_mov_b32 v21, 12",
                "s_mov_b32 exec_lo, 0x0f",
                "v_mov_b32 v22, 13",
                "s_mov_b32 exec_lo, -1",
                "v_mov_b32 v21, 14",
                "s_mov_b32 exec_lo, 0x40000000",
                "v_mov_b32 v22, 15",
                "s_mov_b32 exec_lo, 0x3fffffff",
                "s_endpgm",
            ],
        },
        {
            "name": "saved_then_restored",
            "lines": [
                "s_mov_b32 s12, exec_lo",
                "s_and_saveexec_b32 s14, s20",
                "v_mov_b32 v21, 20",
                "s_mov_b32 exec_lo, s14",
                "v_mov_b32 v22, 21",
                "s_mov_b32 exec_lo, 0x0b",
                "s_mov_b32 exec_lo, s12",
                "v_mov_b32 v23, 22",
                "s_endpgm",
            ],
        },
        {
            "name": "other_saves_leave_copy_known",
            "lines": [
                "s_and_saveexec_b32 s22, s24",
                "s_mov_b32 s23, 0",
                "s_mov_b32 exec_lo, s22",
                "s_and_saveexec_b32 s22, s24",
                "s_mov_b32 s25, 2",
                "s_mov_b32 exec_lo, s22",
                "s_and_saveexec_b32 s22, s24",
                "s_mov_b64 s[20:21], 0",
                "s_mov_b32 exec_lo, s22",
                "s_endpgm",
            ],
        },
        {
            "name": "saved_destination_overwritten",
            "lines": [
                "s_and_saveexec_b32 s26, s28",
                "s_mov_b32 s26, 4",
                "s_mov_b32 exec_lo, s26",
                "s_mov_b32 exec_lo, -4",
                "s_endpgm",
            ],
        },
        {
            "name": "two_unknown_saved_values",
            "lines": [
                "s_and_saveexec_b32 s30, s24",
                "s_and_saveexec_b32 s28, s26",
                "s_mov_b32 exec_lo, s30",
                "v_mov_b32 v25, 30",
                "s_mov_b32 exec_lo, s28",
                "v_mov_b32 v26, 31",
                "s_endpgm",
            ],
        },
        {
            "name": "direct_and_indirect_writers",
            "lines": [
                "v_cmpx_gt_u32 v27, v28",
                "v_mov_b32 v29, 40",
                "s_mov_b32 exec_lo, -1",
                "s_and_b32 exec_lo, exec_lo, s20",
                "v_mov_b32 v30, 41",
                "s_mov_b32 exec_lo, -1",
                "s_mov_b32 exec_hi, 9",
                "v_mov_b32 v31, 42",
                "s_mov_b32 exec_hi, s21",
                "v_mov_b32 v29, 43",
                "s_endpgm",
            ],
        },
        {
            "name": "path_sensitive_masks",
            "lines": [
                "s_cbranch_scc1 @other_path",
                "s_mov_b32 exec_lo, 0x11",
                "s_branch @meeting",
                "other_path:",
                "s_mov_b32 exec_lo, 0x11",
                "meeting:",
                "v_mov_b32 v32, 50",
                "s_cbranch_scc1 @second_path",
                "s_mov_b32 exec_lo, 0x22",
                "second_path:",
                "v_mov_b32 v33, 51",
                "s_endpgm",
            ],
        },
        {
            "name": "cyclic_and_unreachable",
            "lines": [
                "s_mov_b32 exec_lo, 0x2b",
                "again:",
                "v_mov_b32 v34, 60",
                "s_cbranch_scc1 @again",
                "s_endpgm",
                "v_mov_b32 v35, 61",
            ],
        },
    ]


def flight_plans():
    """Memory-counter retirement, path minima, cycles and insertion order."""
    cases = [
        (
            "three_loads_retire_in_order",
            [
                "global_load_b32 v11, v[20:21], off",
                "global_load_b32 v14, v[20:21], off",
                "global_load_b32 v15, v[22:23], off",
                "s_wait_loadcnt {loadcnt=2}",
                "v_mov_b32 v30, 70",
                "s_wait_loadcnt {loadcnt=1}",
                "v_mov_b32 v31, 71",
                "s_wait_loadcnt {loadcnt=0}",
                "s_endpgm",
            ],
            [0, 1, 2],
        ),
        (
            "separate_wait_domains",
            [
                "global_load_b32 v12, v[24:25], off",
                "ds_load_b32 v16, v26",
                "global_store_b32 v[24:25], v12, off",
                "ds_load_b32 v17, v27",
                "s_wait_dscnt {dscnt=1}",
                "v_mov_b32 v32, 72",
                "s_wait_loadcnt {loadcnt=0}",
                "v_mov_b32 v33, 73",
                "s_wait_storecnt {storecnt=0}",
                "s_endpgm",
            ],
            [0, 1, 2, 3],
        ),
        (
            "scalar_memory_count",
            [
                "s_load_b32 s14, s[10:11], 8",
                "s_load_b32 s15, s[10:11], 12",
                "s_wait_kmcnt {kmcnt=1}",
                "v_mov_b32 v34, 74",
                "s_wait_kmcnt {kmcnt=0}",
                "s_endpgm",
            ],
            [0, 1],
        ),
        (
            "operand_wait_retires_all",
            [
                "global_load_b32 v13, v[28:29], off",
                "s_wait_loadcnt 0",
                "v_mov_b32 v35, 75",
                "s_endpgm",
            ],
            [0],
        ),
        (
            "operand_wait_retires_prefix",
            [
                "global_load_b32 v13, v[28:29], off",
                "global_load_b32 v18, v[30:31], off",
                "s_wait_loadcnt 1",
                "v_mov_b32 v36, 76",
                "s_endpgm",
            ],
            [0, 1],
        ),
        (
            "track_only_second_load",
            [
                "global_load_b32 v13, v[28:29], off",
                "global_load_b32 v18, v[30:31], off",
                "s_wait_loadcnt {loadcnt=1}",
                "v_mov_b32 v37, 77",
                "s_endpgm",
            ],
            [1],
        ),
        (
            "join_keeps_earlier_load",
            [
                "global_load_b32 v13, v[28:29], off",
                "s_cbranch_scc1 @around",
                "global_load_b32 v18, v[30:31], off",
                "around:",
                "s_wait_loadcnt {loadcnt=1}",
                "v_mov_b32 v38, 78",
                "s_endpgm",
            ],
            [0, 2],
        ),
        (
            "load_reissued_by_loop",
            [
                "iterate:",
                "global_load_b32 v13, v[28:29], off",
                "v_mov_b32 v39, 79",
                "global_load_b32 v18, v[30:31], off",
                "s_wait_loadcnt {loadcnt=1}",
                "s_cbranch_scc1 @iterate",
                "s_wait_loadcnt {loadcnt=0}",
                "s_endpgm",
            ],
            [1, 3],
        ),
        (
            "explicit_non_program_load_order",
            [
                "s_cbranch_scc1 @right_loads",
                "global_load_b32 v18, v[30:31], off",
                "global_load_b32 v13, v[28:29], off",
                "s_branch @load_join",
                "right_loads:",
                "global_load_b32 v13, v[28:29], off",
                "global_load_b32 v18, v[30:31], off",
                "load_join:",
                "v_mov_b32 v40, 80",
                "s_endpgm",
            ],
            [5, 2, 1, 6],
        ),
    ]
    return [dict(name=n, lines=lines, loads=loads) for n, lines, loads in cases]


def m0_plans():
    """Known/unknown M0 states through straight-line code and CFG merges."""
    cases = [
        ("unannotated_writes", ["s_mov_b32 m0, 12", "s_mov_b32 m0, 12", "s_endpgm"]),
        (
            "marked_repetition_and_change",
            [
                "s_mov_b32 m0, 12 ; m0_set=12",
                "v_mov_b32 v41, 81",
                "s_mov_b32 m0, 12 ; m0_set=12",
                "s_mov_b32 m0, 13 ; m0_set=13",
                "s_mov_b32 m0, 13 ; m0_set=13",
                "s_endpgm",
            ],
        ),
        (
            "unknown_scalar_clears_fact",
            [
                "s_mov_b32 m0, 14 ; m0_set=14",
                "s_mov_b32 m0, s24",
                "s_mov_b32 m0, 14 ; m0_set=14",
                "s_endpgm",
            ],
        ),
        (
            "plain_write_does_not_establish_fact",
            ["s_mov_b32 m0, 15", "s_mov_b32 m0, 15 ; m0_set=15", "s_endpgm"],
        ),
        (
            "diamond_fact_intersection",
            [
                "s_mov_b32 m0, 16 ; m0_set=16",
                "s_cbranch_scc1 @right_m0",
                "s_mov_b32 m0, 17 ; m0_set=17",
                "right_m0:",
                "s_mov_b32 m0, 16 ; m0_set=16",
                "s_cbranch_scc1 @last_m0",
                "s_mov_b32 m0, 16 ; m0_set=16",
                "last_m0:",
                "s_mov_b32 m0, 16 ; m0_set=16",
                "s_endpgm",
            ],
        ),
        (
            "stable_loop_fact",
            [
                "s_mov_b32 m0, 18 ; m0_set=18",
                "m0_cycle:",
                "s_mov_b32 m0, 18 ; m0_set=18",
                "v_mov_b32 v42, 82",
                "s_cbranch_scc1 @m0_cycle",
                "s_endpgm",
            ],
        ),
        (
            "loop_with_changing_fact",
            [
                "s_mov_b32 m0, 19 ; m0_set=19",
                "m0_changing:",
                "s_mov_b32 m0, 19 ; m0_set=19",
                "s_mov_b32 m0, 20 ; m0_set=20",
                "s_cbranch_scc1 @m0_changing",
                "s_endpgm",
            ],
        ),
        (
            "unreached_repetition",
            [
                "s_branch @m0_target",
                "s_mov_b32 m0, 21 ; m0_set=21",
                "s_mov_b32 m0, 21 ; m0_set=21",
                "m0_middle:",
                "s_mov_b32 m0, 21 ; m0_set=21",
                "s_endpgm",
                "m0_target:",
                "s_mov_b32 m0, 21 ; m0_set=21",
                "s_endpgm",
            ],
        ),
        (
            "no_state_after_unconditional_end",
            [
                "s_mov_b32 m0, 22 ; m0_set=22",
                "s_cbranch_scc0 @m0_join",
                "s_endpgm",
                "s_mov_b32 m0, 23 ; m0_set=23",
                "m0_join:",
                "s_mov_b32 m0, 22 ; m0_set=22",
                "s_endpgm",
            ],
        ),
        ("entry_has_unknown_m0", ["s_mov_b32 m0, 17 ; m0_set=17", "s_endpgm"]),
    ]
    return [dict(name=n, lines=lines) for n, lines in cases]


def instruction_lines():
    """Instruction lines exercising register, operand and instruction facts."""
    return [
        "v_add_nc_u32 v51, v52, v53",
        "v_add_nc_u32 v51, v51, v51",
        "v_fmac_f32 v57, v58, v59",
        "v_writelane_b32 v57, s19, 1",
        "v_cmp_gt_u32 vcc_lo, v55, v56",
        "v_cmp_gt_u32 s[28:29], v55, v56",
        "v_cmpx_gt_u32 v57, v58",
        "v_cndmask_b32 v59, v60, v61, vcc_lo",
        "s_mov_b32 exec_lo, 0x5a",
        "s_mov_b32 exec_hi, 17",
        "s_mov_b32 s32, exec_lo",
        "s_and_saveexec_b32 s34, s36",
        "s_mov_b32 m0, 18",
        "s_add_co_u32 s37, s38, s39",
        "s_or_b32 vcc_lo, exec_lo, vcc_lo",
        "global_load_b32 v61, v[62:63], off",
        "global_store_b32 v[64:65], v61, off",
        "s_load_b32 s36, s[40:41], 8",
        "s_cbranch_scc1 @detour",
        "s_endpgm",
        "v_mov_b32_dpp v67, v68 {dpp=quad_perm:[2,3,0,1]|row_mask=15|bank_mask=15}",
        "v_mov_b32_dpp v67, v68 {dpp=row_shr:2|row_mask=15|bank_mask=15}",
        "v_mov_b32_dpp v67, v68 {dpp=row_shr:2|row_mask=15|bank_mask=15|bound_ctrl=true}",
        "v_mov_b32_dpp v67, v68 {dpp=quad_perm:[2,3,0,1]|row_mask=13|bank_mask=15}",
    ]


def dpp_plans():
    """Old-destination reads across bound control, controls and lane masks."""
    full = {"row_mask": 15, "bank_mask": 15}
    return [
        {},
        dict(row_mask=14, bank_mask=15),
        dict(full, bound_ctrl=True),
        dict(full, bound_ctrl=0),
        dict(full, dpp="row_mirror"),
        dict(full, dpp="row_half_mirror"),
        dict(full, dpp="row_ror:5"),
        dict(full, dpp="row_share:6"),
        dict(full, dpp="row_xmask:9"),
        dict(full, dpp="row_shl:2"),
        dict(full, dpp="row_newbcast:17"),
        {"row_mask": 13, "bank_mask": 11, "dpp": "row_mirror", "bound_ctrl": True},
        {"row_mask": True, "bank_mask": 15, "dpp": "row_mirror"},
    ]


def wave_program():
    """Three branch classes, with two attributed and one unresolved branch."""
    return [
        "s_cbranch_scc1 @left_wave ; waves=0b0101",
        "v_mov_b32 v70, 90",
        "left_wave:",
        "s_cbranch_scc1 @right_wave ; waves=0b0110",
        "v_mov_b32 v71, 91",
        "right_wave:",
        "s_cbranch_scc1 @wave_done",
        "v_mov_b32 v72, 92",
        "wave_done:",
        "s_endpgm",
    ]


WAVE_CASES = [
    ("five_by_wave", 5, True),
    ("seven_by_wave", 7, True),
    ("five_shared", 5, False),
]
