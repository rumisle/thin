//! Occupancy: the plain fixpoint that defines where each
//! key is occupied, on facts given as lists.

use thin_ir::cfg::Cfg;
use thin_ir::live::{self, Rows};
use thin_ir::sets::Bits;
use thin_ir::solve::solve_forward_all;

/// The facts the occupancy solver reads, per instruction, keys as
/// small ints: the liveness's rows and the writes and holds around them.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Facts {
    /// The liveness's rows; no key is live at an exit.
    pub live: Rows,
    /// The keys written.
    pub writes: Vec<Vec<usize>>,
    /// The keys written at entry.
    pub pinned: Vec<usize>,
    /// The keys held at it.
    pub held: Vec<Bits>,
    /// Whether it ends a dead write's window (a label or a branch).
    pub window_ends: Vec<bool>,
}

/// The occupancy solve's result.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Occupancy {
    /// Per instruction, the keys occupied there.
    pub occupied: Vec<Bits>,
    /// The keys read where no path wrote them.
    pub unwritten: Bits,
    /// Per instruction, the keys written on some path to it.
    pub written: Vec<Bits>,
}

/// The keys written on some path to each instruction (din).
fn written(f: &Facts) -> Vec<Bits> {
    let g = Cfg::from_succ_only(f.live.succ.clone());
    let mut pinned = Bits::new();
    pinned.add(f.pinned.iter().copied());
    let through = |d: &Bits, j: usize| {
        let mut d = d.clone();
        g.block(j).for_each(|k| d.add(f.writes[k].iter().copied()));
        d
    };
    let merge = |a: &Bits, b: &Bits| {
        let mut m = a.clone();
        m.or(b);
        m
    };
    let entry = solve_forward_all(&g, Bits::new(), vec![(0, pinned)], through, merge);
    let mut din = vec![Bits::new(); f.live.succ.len()];
    for (j, mut d) in entry.into_iter().enumerate() {
        for k in g.block(j) {
            din[k].clone_from(&d);
            d.add(f.writes[k].iter().copied());
        }
    }
    din
}

/// Where each key is occupied, and the keys read before any write.
pub fn solve(f: &Facts) -> Occupancy {
    let n = f.live.succ.len();
    let live = live::solve(&f.live);
    let written = written(f);
    let mut unwritten = Bits::new();
    // A dead write holds its keys up to its window's end: `dead` carries
    // the dead writes of the rows since the last end.
    let mut dead = Bits::new();
    let mut occupied = Vec::with_capacity(n);
    for (k, din) in written.iter().enumerate() {
        let mut read = Bits::new();
        read.add(f.live.reads[k].iter().copied());
        read.add(f.live.every[k].iter().copied());
        read.and_not(din);
        unwritten.or(&read);
        let mut occ = live.live_in[k].and(din);
        occ.add(f.writes[k].iter().copied());
        occ.or(&f.held[k]);
        occ.or(&dead);
        occupied.push(occ);
        if f.window_ends[k] {
            dead.clear();
        }
        let gone = f.writes[k]
            .iter()
            .copied()
            .filter(|&w| !live.live_out(k).has(w));
        dead.add(gone);
    }
    Occupancy {
        occupied,
        unwritten,
        written,
    }
}
