//! The occupancy solver on hand-made facts.

use thin_ir::control::Flow;
use thin_ir::live::{Exec, Lane, Rows};
use thin_ir::sets::Bits;
use thin_passes::regalloc::{solve, Facts};

/// A straight line of n instructions with nothing on them.
fn line(n: usize) -> Facts {
    let succ = (0..n)
        .map(|k| if k + 1 == n { vec![] } else { vec![k + 1] })
        .collect();
    Facts {
        live: Rows::new(succ, Vec::new()),
        writes: vec![Vec::new(); n],
        pinned: Vec::new(),
        held: vec![Bits::new(); n],
        window_ends: vec![false; n],
    }
}

#[test]
fn a_partial_write_keeps_the_other_lanes_live() {
    let mut f = line(5);
    f.writes[0] = vec![0];
    f.live.kills[0] = vec![0];
    f.window_ends[1] = true;
    f.writes[3] = vec![0];
    f.live.lane_kills[3] = vec![0];
    f.live.exec_in[3] = Exec::Mask(1);
    f.live.reads[4] = vec![0];
    f.live.read_lanes[4] = Lane::Mask(3);
    assert!(solve(&f).occupied[2].has(0), "lane 1 needs the first write");
    f.live.read_lanes[4] = Lane::Mask(1);
    let occ = solve(&f);
    assert!(!occ.occupied[2].has(0), "the second covers the lane read");
    assert_eq!(solve(&f), occ, "deterministic");
}

/// 0 writes k0; 1 calls 6; 2 (continuation 1) reads k0; 3 writes k1;
/// 4 calls 6; 5 (continuation 2) ends; 6 returns to 2 or 5.
fn two_calls() -> Facts {
    let mut f = line(7);
    f.live.succ = vec![
        vec![1],
        vec![6],
        vec![3],
        vec![4],
        vec![6],
        vec![],
        vec![2, 5],
    ];
    f.writes[0] = vec![0];
    f.live.kills[0] = vec![0];
    f.live.reads[2] = vec![0];
    f.writes[3] = vec![1];
    f.live.kills[3] = vec![1];
    f
}

#[test]
fn a_value_is_live_before_a_call_only_for_its_own_continuation() {
    let mut f = two_calls();
    let occupied = |f: &Facts, k: usize| solve(f).occupied[k].has(0);
    assert!(
        occupied(&f, 4),
        "without flows: live through the other call"
    );
    f.live.flows = vec![
        vec![Flow::Plain],
        vec![Flow::Call],
        vec![Flow::Plain],
        vec![Flow::Plain],
        vec![Flow::Call],
        vec![],
        vec![Flow::Return { caller: 1 }, Flow::Return { caller: 4 }],
    ];
    assert!(occupied(&f, 1) && occupied(&f, 6), "across its own call");
    assert!(
        !occupied(&f, 3) && !occupied(&f, 4),
        "not before the second call"
    );
}

#[test]
fn a_dead_write_holds_to_the_window_end() {
    let mut f = line(4);
    f.writes[0] = vec![5];
    f.window_ends[2] = true;
    let occ = solve(&f);
    let held: Vec<bool> = occ.occupied.iter().map(|o| o.has(5)).collect();
    assert_eq!(held, [true, true, true, false]);
}

#[test]
fn every_claim_on_an_edge_ends_its_keys() {
    let mut f = line(3);
    f.pinned = vec![0, 1];
    f.live.reads[2] = vec![0, 1];
    f.live.edge_kills[0] = vec![(1, vec![0]), (1, vec![1])];
    let occ = solve(&f);
    assert!(occ.occupied[1].has(0) && occ.occupied[1].has(1));
    assert!(
        !occ.occupied[0].has(0) && !occ.occupied[0].has(1),
        "both claims end their keys on the edge"
    );
}
