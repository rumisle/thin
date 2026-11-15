//! The schedule insertion law on compiler outputs and boundary cases.

mod common;

use thin_lift::schedule::{is_scheduling, unschedule, Schedule, ScheduleError, Scheduled};
use thin_target::{assemble, read_s};
use thin_targets::{decode, encode, Instruction};

fn law(input: &[Instruction]) -> usize {
    let (body, schedule) = unschedule(input);
    assert!(body.iter().all(|i| !is_scheduling(i)));
    assert!(schedule
        .entries
        .iter()
        .all(|e| is_scheduling(&e.instruction)));
    let restored = schedule.insert(&body).unwrap();
    common::instructions_equal(&restored, input);
    let target = thin_targets::default_name();
    assert_eq!(
        encode(target, &restored).unwrap(),
        encode(target, input).unwrap()
    );
    schedule.entries.len()
}

#[test]
fn every_compiled_kernel_recovers_its_exact_schedule() {
    let mut removed = 0;
    let target = thin_targets::default_name();
    for text in common::kernels() {
        let object = thin_codegen::object::code_object(&text).unwrap();
        let parsed = thin_lift::object::read(&object).unwrap();
        for kernel in parsed.kernels {
            let code = &object[kernel.code];
            let instructions = decode(target, code).unwrap();
            removed += law(&instructions);
            let (body, schedule) = unschedule(&instructions);
            assert_eq!(
                encode(target, &schedule.insert(&body).unwrap()).unwrap(),
                code
            );
        }
    }
    assert!(removed > 20);
    eprintln!("thin's seven kernels: {removed} scheduling instructions recovered");
}

#[test]
fn leading_adjacent_trailing_waits_and_wait_targets_round_trip() {
    let target = thin_targets::default_name();
    let text = "s_wait_kmcnt 0\ns_delay_alu 1\ns_branch .Lwait\nv_add_f32 v0, -|v1|, v2\n.Lwait:\ns_wait_alu depctr_sa_sdst(0)\ns_wait_loadcnt 0\ns_endpgm\ns_wait_storecnt 7\n";
    let file = read_s(&format!(".text\n{text}\n"), false).unwrap();
    let code = assemble(&file.code, thin_targets::get(target).unwrap().hooks()).unwrap();
    let input = decode(target, &code).unwrap();
    assert_eq!(law(&input), 5);
    let (body, schedule) = unschedule(&input);
    assert_eq!(
        schedule
            .entries
            .iter()
            .map(|e| e.position)
            .collect::<Vec<_>>(),
        [0, 1, 4, 5, 7]
    );
    // Offsets and targets stay in the input address space until a later
    // relocation step; a branch can still name a removed wait's offset.
    assert_eq!(body[0].branch_target(), Some(input[4].offset));
}

#[test]
fn empty_all_schedule_and_no_schedule_round_trip() {
    assert_eq!(law(&[]), 0);
    let target = thin_targets::default_name();
    for (text, removed) in [
        ("s_endpgm", 0),
        ("s_wait_alu 0\ns_delay_alu 0\ns_wait_idle", 3),
    ] {
        let file = read_s(&format!(".text\n{text}\n"), false).unwrap();
        let code = assemble(&file.code, thin_targets::get(target).unwrap().hooks()).unwrap();
        assert_eq!(law(&decode(target, &code).unwrap()), removed);
    }
}

#[test]
fn malformed_positions_are_refused() {
    let i = Instruction::new(0, "s_wait_idle");
    let entry = |position| Scheduled {
        position,
        instruction: i,
    };
    let bad = Schedule {
        entries: vec![entry(0), entry(0)],
    };
    assert!(matches!(bad.insert(&[]), Err(ScheduleError::Order)));
    let bad = Schedule {
        entries: vec![entry(1), entry(0)],
    };
    assert!(matches!(bad.insert(&[]), Err(ScheduleError::Order)));
    let bad = Schedule {
        entries: vec![entry(1)],
    };
    assert!(matches!(
        bad.insert(&[]),
        Err(ScheduleError::Position { .. })
    ));
}
