//! Call/return target laws on thin-produced code objects, never corpus bytes.

mod common;

use thin_ir::control::{Destination, EdgeKind, TransferKind};
use thin_lift::calls::{resolve_calls, Reason};
use thin_lift::interface::read_interface;
use thin_lift::program::{unrender, Program, Rendering};
use thin_lift::schedule::{unschedule, Schedule};

fn fixture(body: &str) -> (Program, Rendering, Schedule, Vec<thin_targets::Instruction>) {
    let base = &common::kernels()[0];
    let start = base.find("\ts_mov_b32").unwrap();
    let footer = base.find("\t.section\t.rodata").unwrap();
    let image = thin_codegen::object::code_object(&format!(
        "{}{body}\n{}",
        &base[..start],
        &base[footer..]
    ))
    .unwrap();
    let object = thin_lift::object::read(&image).unwrap();
    let kernel = &object.kernels[0];
    let interface = read_interface(
        &kernel.descriptor.unwrap(),
        kernel.metadata.as_ref().unwrap(),
    )
    .unwrap()
    .0;
    let input = thin_targets::decode(object.target, &image[kernel.code.clone()]).unwrap();
    let (body, schedule) = unschedule(&input);
    let (program, rendering) = unrender(
        thin_targets::get(object.target).unwrap(),
        &body,
        &schedule,
        &interface,
    )
    .unwrap();
    (program, rendering, schedule, input)
}
fn target(program: &Program, destination: Destination) -> Option<u32> {
    match destination {
        Destination::Instruction(row) => Some(program.addresses[row]),
        Destination::Exit => None,
    }
}
fn setup(pair: u32, temp: u32, material: u32, high: i32) -> String {
    format!("s_getpc_b64 s[{pair}:{}]\ns_add_co_i32 s{temp}, {material}, 4\ns_add_co_u32 s{pair}, s{pair}, s{temp}\ns_add_co_ci_u32 s{}, s{}, {high}\n",pair+1,pair+1,pair+1)
}

#[test]
fn joined_targets_call_both_subroutines_and_return_to_the_caller() {
    let body=format!("s_cmp_eq_u32 s2, 0\ns_cbranch_scc0 .Lleft\n{}s_branch .Ljoin\n.Lleft:\n{}s_branch .Ljoin\n.Ljoin:\ns_swappc_b64 s[30:31], s[12:13]\ns_endpgm\n{}v_mov_b32 v0, 1\ns_setpc_b64 s[30:31]\n{}v_mov_b32 v0, 2\ns_setpc_b64 s[30:31]",
        setup(12,14,240,0),setup(12,14,280,0),"s_nop 0\n".repeat(48),"s_nop 0\n".repeat(14));
    let (program, rendering, schedule, input) = fixture(&body);
    let old_cfg = program.snapshot.cfg(None).succ.clone();
    let old_indirect = program.indirect.clone();
    let (program, calls) = resolve_calls(program).unwrap();
    assert!(calls.complete());
    assert!(program.indirect.is_empty());
    let call = calls
        .sites
        .iter()
        .find(|s| matches!(s.kind, TransferKind::Call { .. }))
        .unwrap();
    assert_eq!(program.addresses[call.instruction], 56);
    assert_eq!(
        call.edges
            .iter()
            .filter_map(|e| target(&program, e.target))
            .collect::<Vec<_>>(),
        [256, 320]
    );
    assert_eq!(program.snapshot.cfg(None).succ[call.instruction].len(), 2);
    let returns: Vec<_> = calls
        .sites
        .iter()
        .filter(|s| matches!(s.kind, TransferKind::Return { .. }))
        .collect();
    assert_eq!(returns.len(), 2);
    for ret in returns {
        assert_eq!(ret.edges.len(), 1);
        assert_eq!(target(&program, ret.edges[0].target), Some(60));
        assert_eq!(
            ret.edges[0].kind,
            EdgeKind::Return {
                caller: call.instruction
            }
        );
    }
    let restored = rendering.render(&program).unwrap();
    common::instructions_equal(&schedule.insert(&restored).unwrap(), &input);
    let original = calls.forget(program).unwrap();
    assert_eq!(original.snapshot.cfg(None).succ, old_cfg);
    assert_eq!(original.indirect, old_indirect);
    assert!(original.snapshot.control_flow().is_none());
    common::instructions_equal(
        &schedule
            .insert(&rendering.render(&original).unwrap())
            .unwrap(),
        &input,
    );
}

#[test]
fn a_backward_relocated_target_preserves_unsigned_low_add_carry() {
    let body=format!("s_branch .Lmain\nv_mov_b32 v0, 1\ns_setpc_b64 s[30:31]\n.Lmain:\n{}s_swappc_b64 s[30:31], s[12:13]\ns_endpgm",setup(12,14,0xffff_fff0,-1));
    let (program, _, _, _) = fixture(&body);
    let (program, calls) = resolve_calls(program).unwrap();
    assert!(calls.complete());
    let call = calls
        .sites
        .iter()
        .find(|s| matches!(s.kind, TransferKind::Call { .. }))
        .unwrap();
    assert_eq!(target(&program, call.edges[0].target), Some(4));
    let ret = calls
        .sites
        .iter()
        .find(|s| matches!(s.kind, TransferKind::Return { .. }))
        .unwrap();
    assert_eq!(target(&program, ret.edges[0].target), Some(36));
}

#[test]
fn nested_calls_restore_a_saved_link_pair() {
    let body=format!("{}s_swappc_b64 s[30:31], s[12:13]\ns_endpgm\n{}s_mov_b64 s[32:33], s[30:31]\n{}s_swappc_b64 s[30:31], s[12:13]\ns_mov_b64 s[30:31], s[32:33]\ns_setpc_b64 s[30:31]\n{}v_mov_b32 v0, 7\ns_setpc_b64 s[30:31]",
        setup(12,14,248,0),"s_nop 0\n".repeat(57),setup(12,14,52,0),"s_nop 0\n".repeat(8));
    let (program, _, _, _) = fixture(&body);
    let (program, calls) = resolve_calls(program).unwrap();
    assert!(calls.complete());
    assert_eq!(
        calls
            .sites
            .iter()
            .filter(|s| matches!(s.kind, TransferKind::Call { .. }))
            .count(),
        2
    );
    let returns: Vec<_> = calls
        .sites
        .iter()
        .filter(|s| matches!(s.kind, TransferKind::Return { .. }))
        .map(|s| target(&program, s.edges[0].target).unwrap())
        .collect();
    assert_eq!(returns, [24, 280]);
}

#[test]
fn return_links_cover_more_than_two_hundred_fifty_six_callers() {
    let mut body = String::new();
    for caller in 0..300 {
        body += &setup(12, 14, 8184 - 24 * caller, 0);
        body += "s_swappc_b64 s[30:31], s[12:13]\n";
    }
    body += "s_endpgm\n";
    body += &"s_nop 0\n".repeat(247);
    body += "v_mov_b32 v0, 1\ns_setpc_b64 s[30:31]";
    let (program, _, _, _) = fixture(&body);
    let (program, calls) = resolve_calls(program).unwrap();
    assert!(calls.complete());
    let ret = calls
        .sites
        .iter()
        .find(|s| matches!(s.kind, TransferKind::Return { .. }))
        .unwrap();
    assert_eq!(ret.edges.len(), 300);
    assert!(ret
        .edges
        .iter()
        .all(|e| matches!(e.kind, EdgeKind::Return { .. })));
    assert_eq!(program.snapshot.cfg(None).succ[ret.instruction].len(), 300);
}

#[test]
fn unknown_and_clobbered_pairs_are_explicit_without_fallthrough() {
    let (program, _, _, _) = fixture("s_swappc_b64 s[30:31], s[12:13]\ns_endpgm");
    let (program, calls) = resolve_calls(program).unwrap();
    assert!(!calls.complete());
    assert_eq!(calls.sites[0].unresolved, Some(Reason::Input));
    assert!(program.snapshot.cfg(None).succ[0].is_empty());
    let (program, _, _, _) = fixture(
        "s_getpc_b64 s[12:13]\ns_mov_b32 s12, 0\ns_swappc_b64 s[30:31], s[12:13]\ns_endpgm",
    );
    let (_, calls) = resolve_calls(program).unwrap();
    assert_eq!(calls.sites[0].unresolved, Some(Reason::PartialAddress));
}

#[test]
fn swappc_captures_an_aliased_source_before_writing_its_link() {
    let body = format!(
        "{}s_swappc_b64 s[30:31], s[30:31]\ns_endpgm\n{}v_mov_b32 v0, 1\ns_setpc_b64 s[30:31]",
        setup(30, 14, 248, 0),
        "s_nop 0\n".repeat(57)
    );
    let (program, _, _, _) = fixture(&body);
    let (program, calls) = resolve_calls(program).unwrap();
    assert!(calls.complete());
    assert_eq!(target(&program, calls.sites[0].edges[0].target), Some(256));
    assert_eq!(target(&program, calls.sites[1].edges[0].target), Some(24));
}

#[test]
fn returns_resolve_through_removed_wait_boundaries_and_literal_widths() {
    let body=format!("{}s_swappc_b64 s[30:31], s[12:13]\ns_wait_kmcnt 0\nv_mov_b32 v0, 2\ns_endpgm\n{}v_mov_b32 v1, 1\ns_setpc_b64 s[30:31]",setup(12,14,248,0),"s_nop 0\n".repeat(55));
    let (program, rendering, schedule, input) = fixture(&body);
    assert_eq!(schedule.entries.len(), 1);
    let (program, calls) = resolve_calls(program).unwrap();
    assert!(calls.complete());
    assert_eq!(program.addresses[calls.sites[0].instruction], 20);
    assert_eq!(target(&program, calls.sites[1].edges[0].target), Some(28));
    common::instructions_equal(
        &schedule
            .insert(&rendering.render(&calls.forget(program).unwrap()).unwrap())
            .unwrap(),
        &input,
    );
}

#[test]
fn literal_word_targets_and_lost_carry_relations_are_refused() {
    let (program,_,_,_)=fixture("s_getpc_b64 s[12:13]\ns_mov_b32 s14, 0xdeadbeef\ns_add_co_u32 s12, s12, 4\ns_add_co_ci_u32 s13, s13, 0\ns_setpc_b64 s[12:13]\ns_endpgm");
    let (_, calls) = resolve_calls(program).unwrap();
    assert_eq!(calls.sites[0].unresolved, Some(Reason::Boundary));
    let (program,_,_,_)=fixture("s_getpc_b64 s[12:13]\ns_add_co_u32 s12, s12, 4\ns_cmp_eq_u32 s2, 0\ns_add_co_ci_u32 s13, s13, 0\ns_setpc_b64 s[12:13]\ns_endpgm");
    let (_, calls) = resolve_calls(program).unwrap();
    assert_eq!(calls.sites[0].unresolved, Some(Reason::Carry));
}

#[test]
fn complete_call_edges_remove_a_dead_argument_finding() {
    let body=format!("v_mov_b32 v0, 1\n{}s_swappc_b64 s[30:31], s[12:13]\nv_mov_b32 v0, 2\ns_endpgm\n{}v_add_f32 v1, v0, v0\ns_setpc_b64 s[30:31]",setup(12,14,244,0),"s_nop 0\n".repeat(55));
    let (program, _, _, _) = fixture(&body);
    let before = thin_passes::checks::dead_writes_all(&program.snapshot);
    assert_eq!(before.len(), 1);
    let first = thin_passes::checks::dead_writes(&program.snapshot).unwrap_err();
    assert_eq!(format!("{:?}", first.msg), format!("{:?}", before[0].msg));
    let (program, calls) = resolve_calls(program).unwrap();
    assert!(calls.complete());
    assert!(thin_passes::checks::dead_writes_all(&program.snapshot).is_empty());
}

#[test]
fn all_dead_findings_preserve_the_first_errors_existing_order() {
    let (program, _, _, _) =
        fixture("v_mov_b32 v1, 1\nv_mov_b32 v1, 2\nv_mov_b32 v2, 3\nv_mov_b32 v2, 4\ns_endpgm");
    let all = thin_passes::checks::dead_writes_all(&program.snapshot);
    assert_eq!(all.len(), 2);
    let rows: Vec<_> = all
        .iter()
        .map(|d| {
            d.msg
                .iter()
                .find_map(|p| match p {
                    thin_ir::Part::Instr(k) => Some(*k),
                    _ => None,
                })
                .unwrap()
        })
        .collect();
    assert_eq!(rows, [2, 0]);
    let first = thin_passes::checks::dead_writes(&program.snapshot).unwrap_err();
    assert_eq!(format!("{:?}", first.msg), format!("{:?}", all[0].msg));
}

#[test]
fn a_wide_constant_join_reports_its_scalar_domain_limit() {
    let mut body = String::from("s_mov_b32 s14, 0\n");
    for case in 1..65 {
        body += &format!("s_cmp_eq_u32 s2, {case}\ns_cbranch_scc1 .Lcase{case}\n");
    }
    body += "s_branch .Ljoin\n";
    for case in 1..65 {
        body += &format!(".Lcase{case}:\ns_mov_b32 s14, {case}\ns_branch .Ljoin\n");
    }
    body+=".Ljoin:\ns_getpc_b64 s[12:13]\ns_add_co_u32 s12, s12, s14\ns_add_co_ci_u32 s13, s13, 0\ns_setpc_b64 s[12:13]\ns_endpgm";
    let (program, _, _, _) = fixture(&body);
    let (_, calls) = resolve_calls(program).unwrap();
    assert_eq!(calls.sites[0].unresolved, Some(Reason::ScalarLimit));
}

#[test]
fn reanalysis_and_forget_preserve_existing_typed_control_flow() {
    let (program, _, _, _) = fixture("s_getpc_b64 s[12:13]\ns_setpc_b64 s[12:13]\ns_endpgm");
    let (program, first) = resolve_calls(program).unwrap();
    assert!(first.complete());
    let graph = program.snapshot.control_flow().cloned();
    let (program, second) = resolve_calls(program).unwrap();
    assert!(second.complete());
    assert_eq!(second.sites, first.sites);
    let program = second.forget(program).unwrap();
    assert_eq!(program.snapshot.control_flow(), graph.as_ref());
    assert!(first
        .forget(program)
        .unwrap()
        .snapshot
        .control_flow()
        .is_none());
}

fn emulate(
    body: &str,
    init: &mut dyn FnMut(&mut thin_emu::machine::wave::Wave),
) -> Result<thin_emu::machine::wave::Wave, thin_emu::EmuError> {
    // Execute only source accepted by thin's own code-object producer.
    let (physical, _, _, _) = fixture(body);
    let target = physical.snapshot.target();
    let program = thin_emu::read_program(body, target)?;
    let mut machine = thin_emu::machine::run::Machine::new(
        program,
        thin_emu::machine::facts::Facts::of(target),
        4,
        0,
        0,
    )?;
    let (mut waves, _) = machine.run_wg(
        &mut thin_emu::machine::memory::Memory::default(),
        &mut thin_emu::Tables::default(),
        0,
        (0, 0),
        1,
        0,
        Some(init),
    )?;
    Ok(waves.remove(0))
}

#[test]
fn getpc_emulation_uses_literal_widths_and_preserves_condition_registers() {
    let body = "s_getpc_b64 s[12:13]\ns_mov_b32 s4, 0x12345678\ns_getpc_b64 s[14:15]\ns_endpgm";
    let wave = emulate(body, &mut |w| {
        w.exec = 3;
        w.vcc = 0x25;
        w.scc = true;
    })
    .unwrap();
    assert_eq!(&wave.s[12..16], &[4, 0, 16, 0]);
    assert_eq!(wave.exec, 3);
    assert_eq!(wave.vcc, 0x25);
    assert!(wave.scc);
}

#[test]
fn aliased_swappc_emulation_reads_the_old_target_and_returns() {
    let body="s_getpc_b64 s[30:31]\ns_add_co_u32 s30, s30, 28\ns_add_co_ci_u32 s31, s31, 0\ns_swappc_b64 s[30:31], s[30:31]\ns_mov_b32 s4, 7\ns_endpgm\ns_nop 0\ns_nop 0\ns_mov_b32 s6, 9\ns_setpc_b64 s[30:31]";
    let wave = emulate(body, &mut |_| {}).unwrap();
    assert_eq!(wave.s[30], 16);
    assert_eq!(wave.s[4], 7);
    assert_eq!(wave.s[6], 9);
    assert!(wave.done);
}

#[test]
fn nested_call_emulation_preserves_a_saved_outer_link() {
    let mut body=String::from("s_getpc_b64 s[12:13]\ns_add_co_u32 s12, s12, 124\ns_add_co_ci_u32 s13, s13, 0\ns_swappc_b64 s[30:31], s[12:13]\ns_mov_b32 s4, 7\ns_endpgm\n");
    body += &"s_nop 0\n".repeat(25);
    body+="s_mov_b64 s[32:33], s[30:31]\ns_getpc_b64 s[12:13]\ns_add_co_u32 s12, s12, 120\ns_add_co_ci_u32 s13, s13, 0\ns_swappc_b64 s[30:31], s[12:13]\ns_mov_b64 s[30:31], s[32:33]\ns_setpc_b64 s[30:31]\n";
    body += &"s_nop 0\n".repeat(24);
    body += "s_mov_b32 s6, 9\ns_setpc_b64 s[30:31]";
    let wave = emulate(&body, &mut |_| {}).unwrap();
    assert_eq!(wave.s[4], 7);
    assert_eq!(wave.s[6], 9);
    assert_eq!(wave.s[30], 20);
    assert_eq!(wave.s[32], 20);
}

#[test]
fn setpc_emulation_maps_bytes_to_rows_and_faults_on_invalid_targets() {
    let body="s_getpc_b64 s[12:13]\ns_add_co_u32 s12, s12, 16\ns_add_co_ci_u32 s13, s13, 0\ns_setpc_b64 s[12:13]\ns_mov_b32 s4, 1\ns_mov_b32 s6, 2\ns_endpgm";
    let wave = emulate(body, &mut |_| {}).unwrap();
    assert_eq!(wave.s[4], thin_emu::machine::POISON);
    assert_eq!(wave.s[6], 2);
    let body = "s_setpc_b64 s[12:13]\ns_mov_b32 s4, 0xdeadbeef\ns_endpgm";
    for address in [6, 8, 4096] {
        let error = emulate(body, &mut |w| {
            w.s[12] = address;
            w.s[13] = 0;
        })
        .map(|_| ())
        .unwrap_err();
        assert_eq!(error.cause, Some(thin_emu::Cause::Fault));
        assert!(error.message.contains("PC target"));
    }
}

#[test]
fn call_returns_into_alignment_nops_reach_the_next_explicit_row() {
    let body="s_getpc_b64 s[12:13]\ns_add_co_u32 s12, s12, 60\ns_add_co_ci_u32 s13, s13, 0\ns_swappc_b64 s[30:31], s[12:13]\n.p2align 5\ns_mov_b32 s4, 7\ns_endpgm\n.p2align 6\ns_mov_b32 s6, 9\ns_setpc_b64 s[30:31]";
    let wave = emulate(body, &mut |_| {}).unwrap();
    assert_eq!(wave.s[30], 16);
    assert_eq!(wave.s[4], 7);
    assert_eq!(wave.s[6], 9);
}

#[test]
fn dependency_writes_literals_and_named_call_links_have_typed_refusals() {
    let (program, _, _, _) = fixture("s_getpc_b64 s[12:13]\ns_load_b64 s[12:13], s[0:1], 0\ns_swappc_b64 s[30:31], s[12:13]\ns_endpgm");
    let (_, calls) = resolve_calls(program).unwrap();
    assert_eq!(calls.sites[0].unresolved, Some(Reason::Written(1)));
    let (program, _, _, _) = fixture("s_mov_b64 s[12:13], 0\ns_setpc_b64 s[12:13]\ns_endpgm");
    let (_, calls) = resolve_calls(program).unwrap();
    assert_eq!(calls.sites[0].unresolved, Some(Reason::NotPcRelative));
    let (program, _, _, _) = fixture("s_getpc_b64 s[12:13]\ns_swappc_b64 vcc, s[12:13]\ns_endpgm");
    let (_, calls) = resolve_calls(program).unwrap();
    assert_eq!(calls.sites[0].unresolved, Some(Reason::Operand));
    assert!(matches!(
        calls.sites[0].kind,
        TransferKind::UnresolvedCall {
            continuation: Some(_)
        }
    ));
}

#[test]
fn independent_address_and_addend_alternatives_are_not_multiplied() {
    let body="s_cmp_eq_u32 s2, 0\ns_cbranch_scc0 .Lleft\ns_getpc_b64 s[12:13]\ns_mov_b32 s14, 128\ns_branch .Ljoin\n.Lleft:\ns_getpc_b64 s[12:13]\ns_mov_b32 s14, 256\n.Ljoin:\ns_add_co_u32 s12, s12, s14\ns_add_co_ci_u32 s13, s13, 0\ns_setpc_b64 s[12:13]\ns_endpgm";
    let (program, _, _, _) = fixture(body);
    let (_, calls) = resolve_calls(program).unwrap();
    assert_eq!(calls.sites[0].unresolved, Some(Reason::Correlation));
}

#[test]
fn negative_inline_64_bit_constants_keep_their_signed_displacements() {
    for (body, expected) in [
        ("s_getpc_b64 s[12:13]\ns_add_nc_u64 s[12:13], s[12:13], -4\ns_setpc_b64 s[12:13]\ns_endpgm", 0),
        ("s_mov_b64 s[14:15], -4\ns_getpc_b64 s[12:13]\ns_add_co_u32 s12, s12, s14\ns_add_co_ci_u32 s13, s13, s15\ns_setpc_b64 s[12:13]\ns_endpgm", 4),
    ] {
        let (program, _, _, _) = fixture(body);
        let (program, calls) = resolve_calls(program).unwrap();
        assert!(calls.complete());
        assert_eq!(target(&program, calls.sites[0].edges[0].target), Some(expected));
    }
}

#[test]
fn paired_shift_lints_identify_only_the_dead_dword() {
    for (reader, dead, live) in [("v_mov_b32 v4, v3", 2, 3), ("v_mov_b32 v4, v2", 3, 2)] {
        let body = format!("v_mov_b32 v0, 1\nv_mov_b32 v1, 2\nv_lshlrev_b64 v[2:3], 1, v[0:1]\n{reader}\nv_mov_b32 v2, 0\nv_mov_b32 v3, 0\ns_endpgm");
        let (program, _, _, _) = fixture(&body);
        let findings = thin_passes::checks::dead_write_findings(&program.snapshot);
        assert_eq!(findings.len(), 1);
        assert_eq!(
            findings[0]
                .dead_dwords
                .iter()
                .map(|r| (r.lo, r.n))
                .collect::<Vec<_>>(),
            [(dead, 1)]
        );
        assert_eq!(
            findings[0]
                .live_dwords
                .iter()
                .map(|r| (r.lo, r.n))
                .collect::<Vec<_>>(),
            [(live, 1)]
        );
        let text = format!("{:?}", findings[0].diagnostic.msg);
        assert!(text.contains("partial dead write"));
        assert!(text.contains("do not drop the whole instruction"));
    }
}

#[test]
fn both_pair_dwords_dead_is_distinct_from_both_live() {
    for (reader, count) in [("", 1), ("v_add_f32 v4, v2, v3", 0)] {
        let body = format!("v_mov_b32 v0, 1\nv_mov_b32 v1, 2\nv_lshrrev_b64 v[2:3], 1, v[0:1]\n{reader}\nv_mov_b32 v2, 0\nv_mov_b32 v3, 0\ns_endpgm");
        let (program, _, _, _) = fixture(&body);
        let findings = thin_passes::checks::dead_write_findings(&program.snapshot);
        assert_eq!(findings.len(), count);
        if count != 0 {
            assert_eq!(
                findings[0]
                    .dead_dwords
                    .iter()
                    .map(|r| r.lo)
                    .collect::<Vec<_>>(),
                [2, 3]
            );
            assert!(findings[0].live_dwords.is_empty());
        }
    }
}

#[test]
fn a_pair_dword_must_be_overwritten_on_both_sides_of_a_join() {
    for reader in ["", "v_mov_b32 v4, v2"] {
        let body = format!("v_mov_b32 v0, 1\nv_mov_b32 v1, 2\nv_lshlrev_b64 v[2:3], 1, v[0:1]\ns_cmp_eq_u32 s2, 0\ns_cbranch_scc0 .Lother\nv_mov_b32 v2, 0\ns_branch .Ljoin\n.Lother:\n{reader}\nv_mov_b32 v2, 1\n.Ljoin:\nv_mov_b32 v5, v3\nv_mov_b32 v3, 0\ns_endpgm");
        let (program, _, _, _) = fixture(&body);
        let findings = thin_passes::checks::dead_write_findings(&program.snapshot);
        let shift = findings
            .iter()
            .find(|f| program.snapshot.instrs()[f.instruction].form().canon == "v.lshlrev.b64");
        if reader.is_empty() {
            let shift = shift.unwrap();
            assert_eq!(
                shift.dead_dwords.iter().map(|r| r.lo).collect::<Vec<_>>(),
                [2]
            );
            assert_eq!(
                shift.live_dwords.iter().map(|r| r.lo).collect::<Vec<_>>(),
                [3]
            );
        } else {
            assert!(shift.is_none());
        }
    }
}

#[test]
fn an_in_place_right_shift_reads_the_old_high_word_but_can_kill_its_new_one() {
    let (program, _, _, _) = fixture("v_mov_b32 v2, 1\nv_mov_b32 v3, 2\nv_lshrrev_b64 v[2:3], 33, v[2:3]\nv_mov_b32 v4, v2\nv_mov_b32 v3, 0\ns_endpgm");
    let findings = thin_passes::checks::dead_write_findings(&program.snapshot);
    assert_eq!(findings.len(), 1);
    assert_eq!(findings[0].instruction, 2);
    assert_eq!(
        findings[0]
            .dead_dwords
            .iter()
            .map(|r| r.lo)
            .collect::<Vec<_>>(),
        [3]
    );
    assert_eq!(
        findings[0]
            .live_dwords
            .iter()
            .map(|r| r.lo)
            .collect::<Vec<_>>(),
        [2]
    );
}

/// Calls at two sites into one callee; materials put both at the callee.
fn two_calls(between: &str) -> String {
    let make = |m: [u32; 2]| {
        format!(
            "v_mov_b32 v5, 1\n{}s_swappc_b64 s[30:31], s[12:13]\nglobal_store_b32 v0, v5, s[0:1]\n{between}{}s_swappc_b64 s[30:31], s[12:13]\nv_mov_b32 v5, 3\nglobal_store_b32 v0, v5, s[0:1]\ns_endpgm\nv_mov_b32 v7, 0\ns_setpc_b64 s[30:31]",
            setup(12, 14, m[0], 0),
            setup(12, 14, m[1], 0)
        )
    };
    // Sizes do not depend on the (literal) materials: lay out once, then
    // aim each getpc's material at the callee.
    let (_, _, _, input) = fixture(&make([100, 100]));
    let getpc: Vec<u32> = (input.iter())
        .filter(|i| i.mnemonic == "s_getpc_b64")
        .map(|i| i.offset)
        .collect();
    let callee = input
        .iter()
        .find(|i| i.to_string() == "v_mov_b32_e32 v7, 0")
        .unwrap()
        .offset;
    make([callee - getpc[0] - 8, callee - getpc[1] - 8])
}

#[test]
fn a_write_is_dead_when_only_another_calls_continuation_reads_it() {
    let (program, _, _, _) = fixture(&two_calls("v_mov_b32 v5, 2\n"));
    let (program, calls) = resolve_calls(program).unwrap();
    assert!(calls.complete());
    let returns = calls
        .sites
        .iter()
        .find(|s| matches!(s.kind, TransferKind::Return { .. }));
    assert_eq!(
        returns.unwrap().edges.len(),
        2,
        "one callee, two continuations"
    );
    let dead: Vec<String> = thin_passes::checks::dead_write_findings(&program.snapshot)
        .iter()
        .map(|f| program.line(f.instruction).unwrap())
        .collect();
    // The second write of v5 reaches the first continuation's read only
    // by entering the callee from the second call and returning to the
    // first: no valid path.
    assert!(dead.contains(&"v_mov_b32 v5, 2".to_owned()), "{dead:?}");
    assert!(!dead.contains(&"v_mov_b32 v5, 1".to_owned()), "{dead:?}");
}
