//! Partial writes: a 16-bit VALU result or a D16 load writes half of
//! its destination and keeps the other, so the IR reads the destination as
//! well as writing it. The dead-write check, register assignment and the
//! read-before-write refusal each follow that one fact.

use std::sync::Arc;

use thin_ir::{Gather, GatherFile, GatherValue, Instr, Reg, Snapshot, R, W};
use thin_passes::{checks, regalloc};
use thin_target::DstHalf;
use thin_test_support::adapters::default_target;
use thin_test_support::asm;

fn line(text: &str) -> Instr {
    asm::line(default_target(), text).expect("a line")
}

fn dead(lines: &[&str]) -> Option<String> {
    let t = default_target();
    let s = asm::snapshot(t, asm::read(t, lines).expect("lines"));
    checks::dead_writes(&s)
        .err()
        .map(|d| asm::message(&s, &d.msg))
}

#[test]
fn forms_name_the_half_they_write() {
    let half = |text: &str| line(text).form().dst_half;
    use DstHalf::*;
    let cases = [
        ("v_lshlrev_b16 v1, 1, v2", Some(OpSel)),
        ("v_max_num_f16 v1, v2, v3", Some(OpSel)),
        ("v_cvt_f16_f32 v1, v2", Some(OpSel)),
        ("v_fma_mixhi_f16 v1, v2, v3, v4", Some(Hi)),
        ("global_load_d16_b16 v1, v2, s[0:1]", Some(Lo)),
        ("global_load_d16_hi_b16 v1, v2, s[0:1]", Some(Hi)),
        ("ds_load_u16_d16_hi v1, v2", Some(Hi)),
        ("global_store_d16_hi_b16 v2, v1, s[0:1]", None),
        ("v_add_f32 v1, v2, v3", None),
        ("v_pack_b32_f16 v1, v2, v3", None),
        ("v_cmp_eq_f16_e64 s0, v2, v3", None),
        ("s_add_f16 s0, s1, s2", None),
    ];
    for (text, want) in cases {
        assert_eq!(half(text), want, "{text}");
    }
}

#[test]
fn a_half_write_reads_its_destination() {
    for text in [
        "v_lshlrev_b16 v1, 1, v2",
        "v_cvt_f16_f32 v1, v2",
        "global_load_d16_hi_b16 v1, v2, s[0:1]",
    ] {
        let i = line(text);
        assert_eq!(i.roles()[0], R | W, "{text}");
        let d = i.ops()[0].reg();
        assert!(i.uses().any(|r| Some(r) == d), "{text}");
        assert_eq!(i.kept_dst(), d, "{text}");
        assert!(i.keeps_old(false), "{text}");
        // The form keeps the operand table's role: the IR adds the read.
        assert_eq!(i.form().slots[0].role, W, "{text}");
    }
    let whole = line("v_add_f32 v1, v2, v3");
    assert_eq!(whole.roles()[0], W);
    let d = whole.ops()[0].reg();
    assert!(!whole.uses().any(|r| Some(r) == d) && whole.kept_dst().is_none());
}

#[test]
fn op_sel_picks_the_half() {
    let lo = line("v_max_num_f16_e64 v1, v2, v3");
    let hi = line("v_max_num_f16_e64 v1, v2, v3 op_sel:[0,0,1]");
    assert_eq!(lo.dst_half(), Some(DstHalf::Lo));
    assert_eq!(hi.dst_half(), Some(DstHalf::Hi));
    assert_eq!(hi.roles()[0], R | W);
}

#[test]
fn an_accumulator_or_a_source_is_a_data_read() {
    // v_fmac_f16 reads its destination as data: kept_dst is not it.
    let fmac = line("v_fmac_f16 v1, v2, v3");
    assert_eq!(fmac.roles()[0], R | W);
    assert!(fmac.kept_dst().is_none());
    let source = line("v_lshlrev_b16 v1, 1, v1");
    assert!(source.kept_dst().is_none());
}

#[test]
fn a_dpp_half_write_keeps_its_read() {
    let i = line("v_add_f16_dpp v1, v2, v3 quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf");
    assert_eq!(i.roles()[0], R | W);
    let whole = line("v_add_f32_dpp v1, v2, v3 quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf");
    assert_eq!(whole.roles()[0], W);
}

#[test]
fn the_write_a_half_write_keeps_is_not_dead() {
    let store = "global_store_b32 v0, v1, s[0:1]";
    for half in [
        "v_lshlrev_b16 v1, 1, v2",
        "v_cvt_f16_f32 v1, v2",
        "v_max_num_f16 v1, v2, v3",
        "global_load_d16_hi_b16 v1, v2, s[0:1]",
    ] {
        let got = dead(&[
            "v_mov_b32 v1, 0",
            half,
            "s_wait_loadcnt 0",
            store,
            "s_endpgm",
        ]);
        assert_eq!(got, None, "{half}");
    }
}

#[test]
fn a_dead_whole_write_is_still_reported() {
    let store = "global_store_b32 v0, v1, s[0:1]";
    let msg = dead(&["v_mov_b32 v1, 0", "v_mov_b32 v1, 2", store, "s_endpgm"]).expect("dead");
    assert!(
        msg.starts_with("`v_mov_b32 v1, 0`: its value is written over"),
        "{msg}"
    );
    // A half write a whole write follows is dead itself.
    let msg = dead(&["v_cvt_f16_f32 v1, v2", "v_mov_b32 v1, 2", store, "s_endpgm"]).expect("dead");
    assert!(
        msg.starts_with("`v_cvt_f16_f32 v1, v2`: its value is written over"),
        "{msg}"
    );
    let lines = [
        "global_load_d16_b16 v1, v2, s[0:1]",
        "s_wait_loadcnt 0",
        "v_mov_b32 v1, 2",
        store,
        "s_endpgm",
    ];
    let msg = dead(&lines).expect("dead");
    assert!(
        msg.starts_with("`global_load_d16_b16 v1, v2, s[0:1]`: its value"),
        "{msg}"
    );
}

/// A V-file kernel of one-dword values at distinct provisional registers
/// (tid pinned at v0), its lines over `%name`.
fn kernel(names: &[&str], lines: &[&str]) -> Snapshot {
    let t = default_target();
    let mut g = Gather::default();
    for (k, name) in names.iter().enumerate() {
        g.values.push(GatherValue {
            file: 'v',
            n: 1,
            align: 1,
            name: (*name).to_owned(),
            pin: (k == 0).then_some(0),
            register: Some(if k == 0 { 0 } else { 64 + k as u32 }),
            ..Default::default()
        });
    }
    g.files.push(GatherFile {
        letter: 'v',
        ..Default::default()
    });
    let resolve = |tok: &str| {
        let k = names
            .iter()
            .position(|n| Some(*n) == tok.strip_prefix('%'))?;
        Some(Reg::new(k as i32, 'v', 0, 1))
    };
    let instrs: Vec<Arc<Instr>> = (lines.iter())
        .map(|l| {
            asm::line_with(t, l, &resolve)
                .map(Arc::new)
                .expect("a line")
        })
        .collect();
    Snapshot::of_kernel(t, &g, instrs.into()).expect("a snapshot")
}

fn assign(s: &Snapshot) -> Result<regalloc::Placement, String> {
    let rules = thin_targets::hazards(thin_targets::default_name())
        .expect("a model")
        .rules();
    regalloc::assign(s, rules, &[('v', 24)], regalloc::Options::default())
        .map_err(|d| asm::message(s, &d.msg))
}

/// d is written whole, a label ends the dead-write window, t lives only
/// between, then a half write of d keeps d's high half: t must not share
/// d's register.
#[test]
fn the_kept_half_is_not_given_to_another_value() {
    for half in [
        "v_cvt_f16_f32 %d, %tid",
        "v_max_num_f16 %d, %tid, %tid",
        "v_lshlrev_b16 %d, 1, %tid",
        "global_load_d16_b16 %d, %tid, s[0:1]",
    ] {
        let s = kernel(
            &["tid", "d", "t"],
            &[
                "v_mov_b32 %d, 0x12340000",
                ".Lgap:",
                "v_add_nc_u32 %t, 5, %tid",
                "global_store_b32 %tid, %t, s[0:1]",
                half,
                "s_wait_loadcnt 0",
                "global_store_b32 %tid, %d, s[0:1]",
                "s_endpgm",
            ],
        );
        let p = assign(&s).expect("a placement");
        assert_ne!(p.bases[1], p.bases[2], "{half}");
    }
}

/// Without the half write, t may take d's register: the test above is not
/// vacuous.
#[test]
fn a_whole_write_frees_the_register() {
    let s = kernel(
        &["tid", "d", "t"],
        &[
            "v_mov_b32 %d, 0x12340000",
            ".Lgap:",
            "v_add_nc_u32 %t, 5, %tid",
            "global_store_b32 %tid, %t, s[0:1]",
            "v_add_f32 %d, %tid, %tid",
            "global_store_b32 %tid, %d, s[0:1]",
            "s_endpgm",
        ],
    );
    let p = assign(&s).expect("a placement");
    assert_eq!(p.bases[1], p.bases[2]);
}

/// Building a value from two halves needs no write before the first: the
/// half it keeps is carried through, not read as data. An accumulator
/// read before any write is still refused.
#[test]
fn a_fresh_register_takes_a_half_write() {
    let s = kernel(
        &["tid", "d"],
        &[
            "global_load_d16_b16 %d, %tid, s[0:1]",
            "s_wait_loadcnt 0",
            "global_load_d16_hi_b16 %d, %tid, s[2:3]",
            "s_wait_loadcnt 0",
            "global_store_b32 %tid, %d, s[0:1]",
            "s_endpgm",
        ],
    );
    assert!(assign(&s).is_ok());
    let s = kernel(
        &["tid", "d"],
        &[
            "v_fmac_f16 %d, %tid, %tid",
            "global_store_b32 %tid, %d, s[0:1]",
            "s_endpgm",
        ],
    );
    let msg = assign(&s).expect_err("a refusal");
    assert!(
        msg.contains("is read where no path from the entry wrote it"),
        "{msg}"
    );
}
