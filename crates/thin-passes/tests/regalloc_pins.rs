//! First fit never moves a pin: a pin another pin holds
//! and a pin past the file are refused, and only `same_as` gives a pinned
//! value another register (refused later, when the placement is applied).

use std::sync::Arc;

use thin_ir::{Gather, GatherFile, GatherValue, Reg, Snapshot};
use thin_passes::regalloc;
use thin_test_support::adapters::default_target;
use thin_test_support::asm;

/// A V-file kernel of one-value groups (name, dwords, pin, same_as) at
/// distinct provisional registers, and its lines over `%name`.
fn kernel(groups: &[(&str, u32, Option<u32>, Option<u32>)], lines: &[&str]) -> Snapshot {
    let t = default_target();
    let mut g = Gather::default();
    for (k, &(name, n, pin, same_as)) in groups.iter().enumerate() {
        g.values.push(GatherValue {
            file: 'v',
            n,
            align: n.min(2),
            name: name.to_owned(),
            pin,
            same_as,
            register: Some(pin.unwrap_or(64 + 4 * k as u32)),
            ..Default::default()
        });
    }
    g.files.push(GatherFile {
        letter: 'v',
        ..Default::default()
    });
    let resolve = |tok: &str| {
        let (name, dword) = tok
            .strip_prefix('%')?
            .split_once('.')
            .map_or((tok.strip_prefix('%')?, None), |(n, d)| (n, d.parse().ok()));
        let k = groups.iter().position(|x| x.0 == name)?;
        Some(dword.map_or(Reg::new(k as i32, 'v', 0, groups[k].1), |d| {
            Reg::new(k as i32, 'v', d, 1)
        }))
    };
    let instrs: Vec<Arc<thin_ir::Instr>> = lines
        .iter()
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

#[test]
fn a_pin_another_pin_holds_is_refused() {
    let s = kernel(
        &[
            ("tid", 1, Some(0), None),
            ("p", 1, Some(1), None),
            ("q", 1, Some(1), None),
        ],
        &[
            "v_mov_b32 %p, %tid",
            "v_mov_b32 %q, %tid",
            "v_add_nc_u32 %tid, %p, %q",
        ],
    );
    let msg = assign(&s).expect_err("a refusal");
    assert!(
        msg.starts_with("v: no room for q[0] (1 dwords, align 1;"),
        "{msg}"
    );
    assert!(
        msg.contains("): pinned at 1, which a pin before it holds. The peak:"),
        "{msg}"
    );
}

#[test]
fn a_pin_past_the_file_is_refused_as_past_the_file() {
    let s = kernel(
        &[("tid", 1, Some(0), None), ("p", 2, Some(23), None)],
        &[
            "v_mov_b32 %p.0, %tid",
            "v_mov_b32 %p.1, %tid",
            "v_add_nc_u32 %tid, %p.0, %p.1",
        ],
    );
    let msg = assign(&s).expect_err("a refusal");
    assert!(
        msg.contains("): pinned at 23, past the file's 24 registers. The peak:"),
        "{msg}"
    );
}

#[test]
fn same_as_alone_moves_a_pinned_value() {
    let s = kernel(
        &[
            ("tid", 1, Some(0), None),
            ("y", 1, None, None),
            ("z", 1, Some(5), Some(1)),
        ],
        &["v_mov_b32 %y, %tid", "v_add_nc_u32 %tid, %z, %y"],
    );
    let p = assign(&s).expect("a placement");
    assert_eq!(p.bases[2], p.bases[1]);
    assert_eq!(p.bases[1], Some(1));
}
