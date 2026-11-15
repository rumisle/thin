//! Exact branch-unrelaxation laws on thin-produced code objects.

mod common;

use thin_lift::branches::{unrelax_branches, RewriteError};
use thin_lift::schedule::unschedule;
use thin_targets::{decode, encode, Instruction};

/// Compile a new body with one of thin's own kernel headers and footers.
fn compiled(body: &str) -> (Vec<Instruction>, Vec<u8>) {
    let base = &common::kernels()[0];
    let header_end = base.find("\ts_mov_b32").unwrap();
    let footer_start = base.find("\t.section\t.rodata").unwrap();
    let text = format!("{}{body}\n{}", &base[..header_end], &base[footer_start..]);
    let object = thin_codegen::object::code_object(&text).unwrap();
    let parsed = thin_lift::object::read(&object).unwrap();
    let code = object[parsed.kernels[0].code.clone()].to_vec();
    (decode(thin_targets::default_name(), &code).unwrap(), code)
}

fn law(input: &[Instruction], code: &[u8], count: usize) {
    let (short, record) = unrelax_branches(input, None);
    assert_eq!(record.entries.len(), count);
    let restored = record.reverse(&short).unwrap();
    common::instructions_equal(&restored, input);
    assert_eq!(
        encode(thin_targets::default_name(), &restored).unwrap(),
        code
    );
}

fn jump(displacement: i64) -> String {
    let bits = displacement as u64;
    let high = (bits >> 32) as u32;
    let high = if high == u32::MAX {
        "-1".into()
    } else {
        high.to_string()
    };
    format!("s_getpc_b64 s[20:21]\ns_add_co_u32 s20, s20, {}\ns_add_co_ci_u32 s21, s21, {high}\ns_setpc_b64 s[20:21]\n", bits as u32)
}

#[test]
fn compiler_kernels_are_unchanged_when_they_have_no_long_jumps() {
    let target = thin_targets::default_name();
    for text in common::kernels() {
        let bytes = thin_codegen::object::code_object(&text).unwrap();
        let object = thin_lift::object::read(&bytes).unwrap();
        for k in object.kernels {
            let code = &bytes[k.code];
            let instructions = decode(target, code).unwrap();
            law(&instructions, code, 0);
        }
    }
}

#[test]
fn forward_backward_carry_and_inline_u64_sequences_round_trip() {
    let forward = format!("{}{}s_endpgm", jump(32), "s_nop 0\n".repeat(5));
    let (input, code) = compiled(&forward);
    law(&input, &code, 1);
    let (short, record) = unrelax_branches(&input, None);
    assert_eq!(short[0].branch_target(), Some(36));
    assert_eq!(record.entries[0].scratch_sgpr, 20);
    assert!(record.entries[0].writes_scc);
    assert_eq!(record.entries[0].displacement_sgpr, None);
    assert!(!record.entries[0].displacement_writes_scc);

    // Negative displacement needs low-word wrap and a high-word -1.
    let backward = format!("s_nop 0\n{}s_endpgm", jump(-8));
    let (input, code) = compiled(&backward);
    law(&input, &code, 1);
    assert_eq!(unrelax_branches(&input, None).0[1].branch_target(), Some(0));

    for constant in ["-4", "12"] {
        let body = format!("s_getpc_b64 s[4:5]\ns_add_nc_u64 s[4:5], {constant}, s[4:5]\ns_setpc_b64 s[4:5]\ns_nop 0\ns_endpgm");
        let (input, code) = compiled(&body);
        law(&input, &code, 1);
        assert!(!unrelax_branches(&input, None).1.entries[0].writes_scc);
    }
}

#[test]
fn the_end_boundary_and_commuted_sources_are_recognized() {
    let (input, code) = compiled(&jump(12));
    law(&input, &code, 1);
    assert_eq!(
        unrelax_branches(&input, None).0[0].branch_target(),
        Some(16)
    );
    let body = jump(32)
        .replace("s20, s20, 32", "s20, 32, s20")
        .replace("s21, s21, 0", "s21, 0, s21");
    let (input, code) = compiled(&format!("{body}{}s_endpgm", "s_nop 0\n".repeat(5)));
    law(&input, &code, 1);
}

#[test]
fn signed_sixteen_dword_range_edges_are_exact() {
    for (delta, expected) in [(131068i64, 1), (131072, 0), (-131072, 1), (-131076, 0)] {
        let body = if delta > 0 {
            // These positive constants require a literal, so the long
            // sequence is 20 bytes, not the inline sequence's 16.
            let nops = (delta + 4 - 20) as usize / 4;
            format!("{}{}s_endpgm", jump(delta), "s_nop 0\n".repeat(nops))
        } else {
            let nops = (-delta - 4) as usize / 4;
            format!("{}{}s_endpgm", "s_nop 0\n".repeat(nops), jump(delta))
        };
        let (input, code) = compiled(&body);
        law(&input, &code, expected);
    }
}

#[test]
fn multiple_rewrites_reverse_before_schedule_insertion() {
    let body = format!("s_nop 0\n{}s_wait_alu 0\n{}s_endpgm", jump(-8), jump(-32));
    let (input, code) = compiled(&body);
    let (body, schedule) = unschedule(&input);
    let (short, record) = unrelax_branches(&body, Some(&schedule));
    assert_eq!(record.entries.len(), 2);
    let restored_body = record.reverse(&short).unwrap();
    let restored = schedule.insert(&restored_body).unwrap();
    common::instructions_equal(&restored, &input);
    assert_eq!(
        encode(thin_targets::default_name(), &restored).unwrap(),
        code
    );
}

#[test]
fn noncanonical_and_unresolved_sequences_stay_untouched() {
    let canonical = jump(32);
    let variants = [
        canonical.replace("s_add_co_u32", "s_add_co_i32"),
        canonical.replace("s_add_co_ci_u32", "s_add_co_u32"),
        canonical.replace("s21, s21", "s21, s19"),
        canonical.replace("s_setpc_b64 s[20:21]", "s_setpc_b64 s[18:19]"),
        jump(13), // unaligned target
        jump(24), // aligned target past this program's end
        jump(6),  // target inside the low-add instruction
    ];
    for body in variants {
        let (input, code) = compiled(&format!("{body}s_endpgm"));
        law(&input, &code, 0);
    }
}

#[test]
fn known_incoming_edges_prevent_erasing_an_interior_entry() {
    let body = "s_branch .Linside\ns_getpc_b64 s[20:21]\n.Linside:\ns_add_co_u32 s20, 16, s20\ns_add_co_ci_u32 s21, 0, s21\ns_setpc_b64 s[20:21]\ns_nop 0\ns_endpgm";
    let (input, code) = compiled(body);
    law(&input, &code, 0);
}

#[test]
fn edited_results_and_malformed_records_are_refused() {
    let (input, _) = compiled(&format!("{}{}s_endpgm", jump(32), "s_nop 0\n".repeat(5)));
    let (mut short, mut record) = unrelax_branches(&input, None);
    assert_eq!(record.entries.len(), 1);
    short[0].offset += 4;
    assert!(matches!(
        record.reverse(&short),
        Err(RewriteError::Replacement)
    ));
    short[0] = record.entries[0].replacement;
    record.entries[0].position = short.len();
    assert!(matches!(
        record.reverse(&short),
        Err(RewriteError::Position)
    ));
    record.entries[0].position = 0;
    record.entries[0].original.clear();
    assert!(matches!(
        record.reverse(&short),
        Err(RewriteError::EmptyOriginal)
    ));
}

fn materialized_jump(a: u32, b: u32, high: i32) -> String {
    format!("s_getpc_b64 s[76:77]\ns_add_co_i32 s78, {a}, {b}\ns_add_co_u32 s76, s76, s78\ns_add_co_ci_u32 s77, s77, {high}\ns_setpc_b64 s[76:77]\n")
}

#[test]
fn materialized_displacements_fold_constants_and_record_every_clobber() {
    // The reviewed SGEMM displacement is 0x507c + 4, not 0x507c.
    let jump = materialized_jump(0x507c, 4, 0);
    let target = 4 + 0x5080;
    let padding = (target - 24) / 4;
    let (input, code) = compiled(&format!("{jump}{}s_endpgm", "s_nop 0\n".repeat(padding)));
    law(&input, &code, 1);
    let (short, record) = unrelax_branches(&input, None);
    assert_eq!(short[0].branch_target(), Some(target as u32));
    assert_eq!(record.entries[0].original.len(), 5);
    assert_eq!(record.entries[0].scratch_sgpr, 76);
    assert_eq!(record.entries[0].displacement_sgpr, Some(78));
    assert!(record.entries[0].displacement_writes_scc);
    assert!(record.entries[0].writes_scc);

    // Signed overflow in materialization is harmless: low-add overwrites
    // SCC and consumes only the wrapped displacement dword.
    let (input, code) = compiled(&materialized_jump(0x7fff_fffe, 0x7fff_fffe, -1));
    law(&input, &code, 1);
    assert_eq!(unrelax_branches(&input, None).0[0].branch_target(), Some(0));
}

#[test]
fn materialized_jumps_compose_with_removed_wait_targets() {
    let body = "s_getpc_b64 s[76:77]\ns_add_co_i32 s78, 20, 4\ns_delay_alu 1\ns_add_co_u32 s76, s76, s78\ns_add_co_ci_u32 s77, s77, 0\ns_setpc_b64 s[76:77]\ns_nop 0\ns_wait_loadcnt 0\ns_endpgm";
    let (input, code) = compiled(body);
    let (body, schedule) = unschedule(&input);
    // The target is the removed wait, at original offset 28.
    let analysis = thin_lift::branches::inspect_branches(&body, None);
    assert_eq!(
        analysis[0].outcome,
        thin_lift::branches::BranchOutcome::UnknownTarget
    );
    let (short, record) = unrelax_branches(&body, Some(&schedule));
    assert_eq!(record.entries.len(), 1);
    assert_eq!(short[0].branch_target(), Some(28));
    let restored = schedule.insert(&record.reverse(&short).unwrap()).unwrap();
    common::instructions_equal(&restored, &input);
    assert_eq!(
        encode(thin_targets::default_name(), &restored).unwrap(),
        code
    );
}

#[test]
fn materialization_does_not_admit_different_arithmetic_or_control_flow() {
    use thin_lift::branches::{inspect_branches, BranchOutcome};
    let canonical = materialized_jump(8, 4, -1);
    for body in [
        canonical.replace("s78, 8, 4", "s78, s79, 4"),
        canonical.replace("s78, 8, 4", "s76, 8, 4"),
        canonical.replace("s78, 8, 4", "s77, 8, 4"),
        canonical.replace("s76, s76, s78", "s76, s76, s79"),
        canonical.replace("s_add_co_i32", "s_add_co_u32"),
        canonical.replace("s_add_co_u32 s76", "s_add_co_i32 s76"),
    ] {
        let (input, code) = compiled(&format!("{body}s_endpgm"));
        law(&input, &code, 0);
        assert_eq!(
            inspect_branches(&input, None)[0].outcome,
            BranchOutcome::DifferentForm
        );
    }
    let setup = canonical.replace("s_setpc_b64 s[76:77]", "s_branch .Lcontinue");
    let (input, code) = compiled(&format!("{setup}.Lcontinue:\ns_endpgm"));
    law(&input, &code, 0);
    assert_eq!(
        inspect_branches(&input, None)[0].outcome,
        BranchOutcome::PcSetup
    );
}

#[test]
fn diagnostics_account_for_every_nonrewrite_reason() {
    use thin_lift::branches::{inspect_branches, BranchOutcome};
    for (delta, expected) in [
        (-8, BranchOutcome::TargetOutsideAddressSpace),
        (13, BranchOutcome::UnalignedTarget),
        (24, BranchOutcome::UnknownTarget),
    ] {
        let (input, _) = compiled(&format!("{}s_endpgm", jump(delta)));
        assert_eq!(inspect_branches(&input, None)[0].outcome, expected);
    }
    let delta = 131072;
    let padding = (delta + 4 - 20) / 4;
    let (input, _) = compiled(&format!(
        "{}{}s_endpgm",
        jump(delta),
        "s_nop 0\n".repeat(padding as usize)
    ));
    assert_eq!(
        inspect_branches(&input, None)[0].outcome,
        BranchOutcome::OutOfRange
    );
    let (input, _) = compiled("s_branch .Linside\ns_getpc_b64 s[20:21]\n.Linside:\ns_add_co_u32 s20, s20, 16\ns_add_co_ci_u32 s21, s21, 0\ns_setpc_b64 s[20:21]\ns_nop 0\ns_endpgm");
    assert_eq!(
        inspect_branches(&input, None)[0].outcome,
        BranchOutcome::InteriorEntry
    );
}

#[test]
fn removed_trailing_waits_preserve_the_original_end_boundary() {
    let (input, code) = compiled(&format!("{}s_wait_loadcnt 0", materialized_jump(16, 4, 0)));
    let (body, schedule) = unschedule(&input);
    let (short, record) = unrelax_branches(&body, Some(&schedule));
    assert_eq!(record.entries.len(), 1);
    assert_eq!(short[0].branch_target(), Some(24));
    let restored = schedule.insert(&record.reverse(&short).unwrap()).unwrap();
    common::instructions_equal(&restored, &input);
    assert_eq!(
        encode(thin_targets::default_name(), &restored).unwrap(),
        code
    );
}
