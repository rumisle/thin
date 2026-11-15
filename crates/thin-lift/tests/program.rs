//! Physical IR laws on thin-produced objects and explicit entry bindings.

mod common;

use std::collections::HashMap;
use thin_ir::verify::{verify, ValueFacts};
use thin_ir::{Imm, Operand, Snapshot};
use thin_lift::interface::read_interface;
use thin_lift::program::{unrender, EntryValue};
use thin_lift::schedule::unschedule;
use thin_targets::{decode, encode};

fn fixture(body: &str) -> Vec<u8> {
    let base = &common::kernels()[0];
    let start = base.find("\ts_mov_b32").unwrap();
    let footer = base.find("\t.section\t.rodata").unwrap();
    thin_codegen::object::code_object(&format!("{}{body}\n{}", &base[..start], &base[footer..]))
        .unwrap()
}

fn laws(bytes: &[u8]) {
    let object = thin_lift::object::read(bytes).unwrap();
    let target = thin_targets::get(object.target).unwrap();
    for kernel in object.kernels {
        let input = decode(object.target, &bytes[kernel.code.clone()]).unwrap();
        let interface = read_interface(
            &kernel.descriptor.unwrap(),
            kernel.metadata.as_ref().unwrap(),
        )
        .unwrap()
        .0;
        let (body, schedule) = unschedule(&input);
        let (program, rendering) = unrender(target, &body, &schedule, &interface).unwrap();
        assert!(program
            .snapshot
            .values()
            .iter()
            .all(|v| v.pin == v.phys && v.pin == Some(0)));
        let facts: HashMap<_, _> = program
            .snapshot
            .values()
            .iter()
            .enumerate()
            .map(|(k, v)| {
                (
                    k as i32,
                    ValueFacts {
                        repr: format!("physical {}", v.file),
                        n: v.n,
                        align: v.align,
                        pinned: true,
                    },
                )
            })
            .collect();
        assert_eq!(
            verify(target, program.snapshot.instrs(), &facts, true),
            None
        );
        let restored_body = rendering.render(&program).unwrap();
        common::instructions_equal(&restored_body, &body);
        let restored = schedule.insert(&restored_body).unwrap();
        common::instructions_equal(&restored, &input);
        assert_eq!(
            encode(object.target, &restored).unwrap(),
            bytes[kernel.code]
        );
    }
}

#[test]
fn every_compiled_kernel_reproduces_its_decoded_instructions_and_bytes() {
    for text in common::kernels() {
        laws(&thin_codegen::object::code_object(&text).unwrap());
    }
}

#[test]
fn physical_fields_and_source_modifiers_are_verified_and_reversible() {
    laws(&fixture("v_cvt_pk_bf8_f32 v1, v2, v3 op_sel:[0,0,1]\nv_max_num_f32_e64 v4, v5, |v6|\ns_setreg_imm32_b32 hwreg(26, 0, 2), 0\ns_endpgm"));
}

#[test]
fn a_removed_wait_target_keeps_its_original_label() {
    laws(&fixture(
        "s_branch .Lwait\ns_nop 0\n.Lwait:\ns_wait_loadcnt 0\ns_mov_b32 s2, 5\ns_endpgm",
    ));
}

#[test]
fn entry_subranges_do_not_initialize_the_whole_pinned_file() {
    let bytes = fixture("s_endpgm");
    let object = thin_lift::object::read(&bytes).unwrap();
    let kernel = &object.kernels[0];
    let interface = read_interface(
        &kernel.descriptor.unwrap(),
        kernel.metadata.as_ref().unwrap(),
    )
    .unwrap()
    .0;
    let input = decode(object.target, &bytes[kernel.code.clone()]).unwrap();
    let (body, schedule) = unschedule(&input);
    let (program, _) = unrender(
        thin_targets::get(object.target).unwrap(),
        &body,
        &schedule,
        &interface,
    )
    .unwrap();
    assert_eq!(program.snapshot.entry().len(), 3);
    assert!(program.snapshot.entry().iter().all(|r| r.lo + r.n <= 3));
    assert!(program
        .entry
        .iter()
        .any(|b| b.value == EntryValue::WorkgroupId(0)));
    assert!(program.snapshot.values().iter().all(|v| v.n > 3));
}

#[test]
fn rendering_reads_current_ir_operands_not_a_saved_instruction_copy() {
    let bytes = fixture("s_mov_b32 s2, 5\ns_endpgm");
    let object = thin_lift::object::read(&bytes).unwrap();
    let kernel = &object.kernels[0];
    let interface = read_interface(
        &kernel.descriptor.unwrap(),
        kernel.metadata.as_ref().unwrap(),
    )
    .unwrap()
    .0;
    let input = decode(object.target, &bytes[kernel.code.clone()]).unwrap();
    let (body, schedule) = unschedule(&input);
    let target = thin_targets::get(object.target).unwrap();
    let (mut program, rendering) = unrender(target, &body, &schedule, &interface).unwrap();
    let mut rows = program.snapshot.instrs().to_vec();
    let mut ops = rows[0].ops().to_vec();
    ops[1] = Operand::Imm(Imm::int(6));
    rows[0] = rows[0].with_ops(ops).into();
    program.snapshot = Snapshot::new(target, program.snapshot.values().to_vec(), rows.into());
    let changed = rendering.render(&program).unwrap();
    assert_eq!(changed[0].operands()[1], thin_targets::decode::Opd::Int(6));
}

#[test]
fn wave_size_mismatch_is_a_typed_refusal_not_a_target_mutation() {
    let bytes = fixture("s_endpgm");
    let object = thin_lift::object::read(&bytes).unwrap();
    let kernel = &object.kernels[0];
    let mut interface = read_interface(
        &kernel.descriptor.unwrap(),
        kernel.metadata.as_ref().unwrap(),
    )
    .unwrap()
    .0;
    interface.wave_size = 64;
    let input = decode(object.target, &bytes[kernel.code.clone()]).unwrap();
    let (body, schedule) = unschedule(&input);
    match unrender(
        thin_targets::get(object.target).unwrap(),
        &body,
        &schedule,
        &interface,
    ) {
        Err(e) => assert_eq!(e.class, "wave size"),
        Ok(_) => panic!("wave64 is not the wave32 IR target"),
    }
}

fn checked(body: &str) -> thin_lift::program::Program {
    let bytes = fixture(body);
    let object = thin_lift::object::read(&bytes).unwrap();
    let kernel = &object.kernels[0];
    let interface = read_interface(
        &kernel.descriptor.unwrap(),
        kernel.metadata.as_ref().unwrap(),
    )
    .unwrap()
    .0;
    let input = decode(object.target, &bytes[kernel.code.clone()]).unwrap();
    let (body, schedule) = unschedule(&input);
    unrender(
        thin_targets::get(object.target).unwrap(),
        &body,
        &schedule,
        &interface,
    )
    .unwrap()
    .0
}

#[test]
fn physical_scc_accepts_defined_predicates_without_disabling_author_policies() {
    use thin_passes::checks::{physical_scc, scc_findings, SccReader};
    for (body, kind) in [
        (
            "s_and_b32 s0, 0, 0\ns_cselect_b32 s1, 1, 0\ns_endpgm",
            SccReader::Select,
        ),
        (
            "s_and_b32 s0, 0, 0\ns_add_co_ci_u32 s1, s1, 0\ns_endpgm",
            SccReader::Carry,
        ),
        (
            "s_cmp_lg_u32 s0, 0\ns_add_co_ci_u32 s1, s1, 0\ns_endpgm",
            SccReader::Carry,
        ),
    ] {
        let program = checked(body);
        assert!(physical_scc(&program.snapshot).is_empty());
        let findings = scc_findings(&program.snapshot);
        assert_eq!(findings.len(), 1);
        assert_eq!(findings[0].kind, kind);
        assert!(thin_passes::checks::scc_readers(&program.snapshot).is_err());
    }
}

#[test]
fn fused_shift_add_is_a_real_unsigned_carry_producer() {
    for shift in 1..=4 {
        let program = checked(&format!(
            "s_lshl{shift}_add_u32 s0, s2, s0\ns_add_co_ci_u32 s1, s1, 0\ns_endpgm"
        ));
        assert!(thin_passes::checks::scc_readers(&program.snapshot).is_ok());
        assert!(thin_passes::checks::physical_scc(&program.snapshot).is_empty());
    }
}

#[test]
fn physical_scc_rejects_entry_reads_and_propagates_undefined_carry() {
    for body in [
        "s_cselect_b32 s0, 1, 0\ns_endpgm",
        "s_cbranch_scc1 .Lend\n.Lend:\ns_endpgm",
    ] {
        let program = checked(body);
        assert_eq!(
            thin_passes::checks::physical_scc(&program.snapshot).len(),
            1
        );
    }
    let program = checked("s_add_co_ci_u32 s0, s0, 0\ns_cselect_b32 s1, 1, 0\ns_endpgm");
    assert_eq!(
        thin_passes::checks::physical_scc(&program.snapshot).len(),
        2
    );
}

#[test]
fn physical_scc_requires_a_defined_input_on_every_join_predecessor() {
    let valid = checked("s_cmp_eq_u32 s0, 0\ns_cbranch_scc0 .Lother\ns_and_b32 s1, s1, 7\ns_branch .Ljoin\n.Lother:\ns_cmp_lg_u32 s2, 0\n.Ljoin:\ns_cselect_b32 s3, 1, 0\ns_endpgm");
    assert!(thin_passes::checks::physical_scc(&valid.snapshot).is_empty());
    let invalid = checked("s_cbranch_execz .Lother\ns_branch .Ljoin\n.Lother:\ns_and_b32 s1, s1, 7\n.Ljoin:\ns_cselect_b32 s3, 1, 0\ns_endpgm");
    assert_eq!(
        thin_passes::checks::physical_scc(&invalid.snapshot).len(),
        1
    );
}

#[test]
fn dpp_controls_round_trip_and_unrepresentable_fields_are_refused() {
    laws(&fixture(
        "v_add_f32_dpp v0, v1, v2 row_mirror row_mask:15 bank_mask:15\ns_endpgm",
    ));
    for (field, class) in [("bound_ctrl:1", "rendering field"), ("fi:1", "field")] {
        let bytes = fixture(&format!(
            "v_add_f32_dpp v0, v1, v2 quad_perm:[0,1,2,3] {field}\ns_endpgm"
        ));
        let object = thin_lift::object::read(&bytes).unwrap();
        let kernel = &object.kernels[0];
        let interface = read_interface(
            &kernel.descriptor.unwrap(),
            kernel.metadata.as_ref().unwrap(),
        )
        .unwrap()
        .0;
        let input = decode(object.target, &bytes[kernel.code.clone()]).unwrap();
        let (body, schedule) = unschedule(&input);
        match unrender(
            thin_targets::get(object.target).unwrap(),
            &body,
            &schedule,
            &interface,
        ) {
            Err(e) => assert_eq!(e.class, class),
            Ok(_) => panic!("unrepresentable physical field must not be lost"),
        }
    }
}

#[test]
fn an_empty_physical_program_has_no_scc_readers() {
    let bytes = fixture("s_endpgm");
    let object = thin_lift::object::read(&bytes).unwrap();
    let kernel = &object.kernels[0];
    let interface = read_interface(
        &kernel.descriptor.unwrap(),
        kernel.metadata.as_ref().unwrap(),
    )
    .unwrap()
    .0;
    let (program, rendering) = unrender(
        thin_targets::get(object.target).unwrap(),
        &[],
        &Default::default(),
        &interface,
    )
    .unwrap();
    assert!(rendering.render(&program).unwrap().is_empty());
    assert!(thin_passes::checks::scc_findings(&program.snapshot).is_empty());
    assert!(thin_passes::checks::physical_scc(&program.snapshot).is_empty());
}

#[test]
fn typed_control_flow_overrides_indirect_fallthrough_and_invalidates_cache() {
    use thin_ir::control::{ControlFlow, Destination, Edge, EdgeKind, Transfer, TransferKind};
    let mut program = checked("s_setpc_b64 s[4:5]\ns_mov_b32 s0, 1\ns_endpgm");
    assert_eq!(program.snapshot.cfg(None).succ[0], [1]);
    program.snapshot = program
        .snapshot
        .with_control_flow(Some(ControlFlow {
            transfers: vec![Transfer {
                instruction: 0,
                kind: TransferKind::Jump,
                edges: vec![Edge {
                    target: Destination::Instruction(2),
                    kind: EdgeKind::Branch,
                }],
                complete: true,
            }],
        }))
        .unwrap();
    assert_eq!(program.snapshot.cfg(None).succ[0], [2]);
    program.snapshot = program.snapshot.with_control_flow(None).unwrap();
    assert_eq!(program.snapshot.cfg(None).succ[0], [1]);
}

#[test]
fn typed_returns_must_name_their_callers_fallthrough() {
    use thin_ir::control::{ControlFlow, Destination, Edge, EdgeKind, Transfer, TransferKind};
    let program = checked("s_swappc_b64 s[4:5], s[6:7]\ns_endpgm\ns_setpc_b64 s[4:5]");
    let link = *program.snapshot.instrs()[0].ops()[0].reg().unwrap();
    let flow = ControlFlow {
        transfers: vec![
            Transfer {
                instruction: 0,
                kind: TransferKind::Call {
                    link,
                    continuation: Destination::Instruction(1),
                },
                edges: vec![Edge {
                    target: Destination::Instruction(2),
                    kind: EdgeKind::Call,
                }],
                complete: true,
            },
            Transfer {
                instruction: 2,
                kind: TransferKind::Return { link },
                edges: vec![Edge {
                    target: Destination::Instruction(0),
                    kind: EdgeKind::Return { caller: 0 },
                }],
                complete: true,
            },
        ],
    };
    assert!(program.snapshot.with_control_flow(Some(flow)).is_err());
}

#[test]
fn the_schedule_goes_back_in_order_after_a_label_on_a_removed_wait() {
    let bytes = fixture(
        "s_branch .Lwait\ns_nop 0\n.Lwait:\ns_wait_loadcnt 0\ns_mov_b32 s2, 5\ns_wait_kmcnt 0\ns_endpgm",
    );
    let object = thin_lift::object::read(&bytes).unwrap();
    let kernel = &object.kernels[0];
    let interface = read_interface(
        &kernel.descriptor.unwrap(),
        kernel.metadata.as_ref().unwrap(),
    )
    .unwrap()
    .0;
    let input = decode(object.target, &bytes[kernel.code.clone()]).unwrap();
    let (body, schedule) = unschedule(&input);
    let target = thin_targets::get(object.target).unwrap();
    let (program, _) = unrender(target, &body, &schedule, &interface).unwrap();
    let s = program.scheduled(&schedule).unwrap();
    let forms: Vec<i32> = (s.instrs().iter())
        .filter(|i| i.form().kind != thin_target::Kind::Label)
        .map(|i| i.form().id)
        .collect();
    let original: Vec<i32> = (input.iter())
        .map(|i| thin_targets::physical_form(object.target, i).unwrap().id)
        .collect();
    assert_eq!(forms, original);
    let wait = (s.instrs().iter())
        .position(|i| i.form().flags.has(thin_target::Flags::WAITCNT))
        .unwrap();
    assert_eq!(s.instrs()[wait - 1].form().kind, thin_target::Kind::Label);
}
