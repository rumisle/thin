//! Generic wait-state engine tests with synthetic dependence rules.
//! gfx1201 uses its own hazard pass rather than these generic wait-state rules.

use std::sync::Arc;

use thin_ir::Instr;
use thin_passes::hazards::{self, Dep, HazardModel, Rule, Side, States};
use thin_target::Kind;
use thin_test_support::adapters::default_target;
use thin_test_support::asm;

/// A model of rules only.
struct Rules(Vec<Rule>);

impl HazardModel for Rules {
    fn rules(&self) -> &[Rule] {
        &self.0
    }
}

fn valu() -> Side {
    Side {
        kinds: vec![Kind::Valu],
        ..Side::default()
    }
}

/// RAW from a VALU to a VALU, `states` wait states.
fn raw(states: u32) -> Rules {
    Rules(vec![Rule {
        prev: valu(),
        cur: valu(),
        dep: Dep::Raw,
        states: States::Const(states),
    }])
}

/// The wait-state need per line.
fn needs(model: &Rules, lines: &[&str]) -> Vec<u32> {
    let t = default_target();
    let s = asm::snapshot(t, asm::read(t, lines).unwrap());
    hazards::needs(&s, model).iter().map(|n| n.states).collect()
}

#[test]
fn a_raw_rule_asks_its_states_and_an_s_nop_gives_them() {
    let m = raw(1);
    assert_eq!(
        needs(&m, &["v_mov_b32 v1, 0", "v_add_nc_u32 v2, v1, v1"]),
        [0, 1]
    );
    assert_eq!(
        needs(
            &m,
            &["v_mov_b32 v1, 0", "s_nop 0", "v_add_nc_u32 v2, v1, v1"]
        ),
        [0, 0, 0]
    );
    assert_eq!(
        needs(
            &raw(3),
            &["v_mov_b32 v1, 0", "s_nop 0", "v_add_nc_u32 v2, v1, v1"]
        ),
        [0, 0, 2]
    );
    assert_eq!(
        needs(&m, &["v_mov_b32 v1, 0", "v_add_nc_u32 v2, v3, v3"]),
        [0, 0]
    );
}

#[test]
fn a_forward_branch_into_a_label_run_is_a_predecessor() {
    let lines = [
        "v_mov_b32 v1, 0",
        "s_cbranch_scc1 .L",
        "s_mov_b32 s0, 0",
        "s_mov_b32 s1, 0",
        ".L:",
        "; a comment in the run",
        "v_add_nc_u32 v2, v1, v1",
    ];
    assert_eq!(needs(&raw(3), &lines), [0, 0, 0, 0, 0, 0, 2]);
}

#[test]
fn an_unconditional_branch_cuts_the_fall_through() {
    let lines = [
        "v_mov_b32 v1, 0",
        "s_branch .M",
        ".L:",
        "v_add_nc_u32 v2, v1, v1",
        ".M:",
        "s_endpgm",
    ];
    assert_eq!(needs(&raw(3), &lines), [0; 6]);
    let fall = [
        "v_mov_b32 v1, 0",
        "s_mov_b32 s0, 0",
        ".L:",
        "v_add_nc_u32 v2, v1, v1",
    ];
    assert_eq!(needs(&raw(3), &fall), [0, 0, 0, 2]);
}

#[test]
fn a_back_edge_need_goes_before_the_branch() {
    let lines = [
        ".L:",
        "v_add_nc_u32 v2, v1, v1",
        "v_mov_b32 v1, 0",
        "s_cbranch_scc1 .L",
        "s_endpgm",
    ];
    assert_eq!(needs(&raw(3), &lines), [0, 0, 0, 2, 0]);
}

#[test]
fn placing_twice_gives_the_same_stream_and_nothing_more_to_do() {
    let t = default_target();
    let m = raw(2);
    let lines = ["v_mov_b32 v1, 0", "v_add_nc_u32 v2, v1, v1", "s_endpgm"];
    let s = asm::snapshot(t, asm::read(t, &lines).unwrap());
    let once: Vec<Arc<Instr>> = hazards::apply(&s, &hazards::needs(&s, &m))
        .into_iter()
        .map(|p| p.0)
        .collect();
    let s1 = asm::snapshot(t, hazards::strip(&once));
    let twice: Vec<Arc<Instr>> = hazards::apply(&s1, &hazards::needs(&s1, &m))
        .into_iter()
        .map(|p| p.0)
        .collect();
    let spell = |is: &[Arc<Instr>]| is.iter().map(|i| asm::spell(&s, i)).collect::<Vec<_>>();
    assert_eq!(spell(&once), spell(&twice));
    assert_eq!(
        spell(&once),
        [
            "v_mov_b32 v1, 0",
            "s_nop 1",
            "v_add_nc_u32 v2, v1, v1",
            "s_endpgm"
        ]
    );
    let placed = asm::snapshot(t, once);
    assert!(hazards::clear(&placed, &m).is_ok());
    assert!(hazards::gave(&placed, &m, 1) && !hazards::gave(&placed, &m, 0));
}
