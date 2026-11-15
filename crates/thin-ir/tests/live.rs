//! The liveness solver against a reference: the row equations over
//! (context, lane, key) triples, iterated round robin until nothing
//! changes, on generated graphs with calls, partial EXEC and lane regions.

use std::collections::BTreeSet;

use thin_ir::control::{Context, Flow};
use thin_ir::live::{solve, Exec, Lane, Rows};
use thin_ir::sets::Bits;

/// A small xorshift generator.
struct Rng(u64);

impl Rng {
    fn below(&mut self, n: usize) -> usize {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 % n as u64) as usize
    }

    fn chance(&mut self, one_in: usize) -> bool {
        self.below(one_in) == 0
    }

    /// A few keys, some past the first word.
    fn keys(&mut self) -> Vec<usize> {
        const KEYS: [usize; 8] = [0, 1, 2, 3, 63, 64, 65, 129];
        (0..self.below(3))
            .map(|_| KEYS[self.below(KEYS.len())])
            .collect()
    }

    fn exec(&mut self) -> Exec {
        match self.below(4) {
            0 => Exec::Mask(1 + self.below(7) as u64),
            1 => Exec::Region(self.below(3) as i128),
            _ => Exec::Unknown,
        }
    }

    fn lane(&mut self) -> Lane {
        match self.below(4) {
            0 => Lane::Mask(1 + self.below(7) as u64),
            1 => Lane::Region(self.below(3) as i128),
            2 => Lane::Any,
            _ => Lane::Cur,
        }
    }
}

type Triple = (Context, Lane, usize);

/// Whether class c loses a lane kill made under EXEC e.
fn cut(c: Lane, e: Exec) -> bool {
    match (c, e) {
        (Lane::Cur, _) => true,
        (Lane::Mask(c), Exec::Mask(m)) => c & !m == 0,
        (Lane::Region(c), Exec::Region(r)) => c == r,
        _ => false,
    }
}

/// The triples before row k from those after it.
fn through(r: &Rows, k: usize, after: &BTreeSet<Triple>) -> BTreeSet<Triple> {
    let exec = if r.writes_exec[k] {
        r.exec_out[k]
    } else {
        r.exec_in[k]
    };
    let class = match exec {
        Exec::Mask(m) => Lane::Mask(m),
        Exec::Region(t) => Lane::Region(t),
        Exec::Unknown if r.writes_exec[k] => Lane::Any,
        Exec::Unknown => Lane::Cur,
    };
    let mut out = BTreeSet::new();
    for &(ctx, lane, key) in after {
        let lane = if lane == Lane::Cur { class } else { lane };
        let lane_cut = lane == Lane::Cur
            || lane != Lane::Any && r.exec_in[k] != Exec::Unknown && cut(lane, r.exec_in[k]);
        if r.kills[k].contains(&key) || lane_cut && r.lane_kills[k].contains(&key) {
            continue;
        }
        out.insert((ctx, lane, key));
    }
    out.extend(r.every[k].iter().map(|&key| (None, Lane::Any, key)));
    out.extend(r.reads[k].iter().map(|&key| (None, r.read_lanes[k], key)));
    out
}

/// The triples after row k from those before each successor.
fn after(r: &Rows, k: usize, before: &[BTreeSet<Triple>]) -> BTreeSet<Triple> {
    if r.succ[k].is_empty() {
        return r.exit.iter().map(|&key| (None, Lane::Any, key)).collect();
    }
    let mut out = BTreeSet::new();
    for (e, &t) in r.succ[k].iter().enumerate() {
        let flow = (r.flows.get(k)).map_or(Flow::Plain, |f| f[e]);
        let claimed =
            |key: usize| (r.edge_kills[k].iter()).any(|(to, keys)| *to == t && keys.contains(&key));
        for &(ctx, lane, key) in &before[t] {
            if let (Some(ctx), false) = (flow.back(k, ctx), claimed(key)) {
                out.insert((ctx, lane, key));
            }
        }
    }
    out
}

/// Per row, its live-in and live-out keys by the reference.
fn reference(r: &Rows) -> (Vec<Bits>, Vec<Bits>) {
    let n = r.succ.len();
    let mut before = vec![BTreeSet::new(); n];
    loop {
        let mut changed = false;
        for k in 0..n {
            let next = through(r, k, &after(r, k, &before));
            if next != before[k] {
                before[k] = next;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    let flat = |s: &BTreeSet<Triple>| {
        let mut b = Bits::new();
        b.add(s.iter().map(|t| t.2));
        b
    };
    let live_in = before.iter().map(flat).collect();
    let live_out = (0..n).map(|k| flat(&after(r, k, &before))).collect();
    (live_in, live_out)
}

/// Rows of a main body and two subroutines: calls into the first from
/// the body and into the second from the first (nested), returns to each
/// call's continuation, branches forward and back, partial EXEC, lane
/// regions, EXEC writes and edge claims.
fn generate(rng: &mut Rng) -> Rows {
    let n = 9 + rng.below(40);
    let (a, b) = (n / 2, n / 2 + (n - n / 2) / 2);
    let regions = [0..a, a..b, b..n];
    let mut succ: Vec<Vec<usize>> = (0..n).map(|k| vec![k + 1]).collect();
    let mut flows: Vec<Vec<Flow>> = vec![vec![Flow::Plain]; n];
    let mut calls: [Vec<usize>; 3] = Default::default();
    for (i, region) in regions.iter().enumerate() {
        let last = region.end - 1;
        for k in region.start..last {
            let mut call = false;
            match rng.below(8) {
                0 => succ[k] = vec![region.start + rng.below(region.len()), k + 1],
                1 => succ[k] = vec![region.start + rng.below(region.len())],
                2 if i < 2 && k + 1 < last => {
                    succ[k] = vec![regions[i + 1].start];
                    calls[i + 1].push(k);
                    call = true;
                }
                // A call whose callee starts at the next row.
                3 if rng.chance(4) => call = true,
                4 if rng.chance(3) => succ[k] = Vec::new(),
                _ => {}
            }
            succ[k].dedup();
            flows[k] = match call {
                true => vec![Flow::Call],
                false => vec![Flow::Plain; succ[k].len()],
            };
        }
        succ[last] = calls[i].iter().map(|&c| c + 1).collect();
        flows[last] = calls[i]
            .iter()
            .map(|&c| Flow::Return { caller: c })
            .collect();
    }
    if rng.chance(4) {
        // Arbitrary flows on every edge.
        for (k, f) in flows.iter_mut().enumerate() {
            *f = (0..succ[k].len())
                .map(|_| match rng.below(4) {
                    0 => Flow::Call,
                    1 => Flow::Return {
                        caller: rng.below(n),
                    },
                    _ => Flow::Plain,
                })
                .collect();
        }
    } else if rng.chance(5) {
        flows = Vec::new();
    }
    let mut r = Rows::new(succ, flows);
    for k in 0..n {
        r.exec_in[k] = rng.exec();
        if rng.chance(6) {
            r.writes_exec[k] = true;
            r.exec_out[k] = rng.exec();
        }
        match rng.below(6) {
            0 => r.reads[k] = rng.keys(),
            1 => {
                r.reads[k] = rng.keys();
                r.read_lanes[k] = rng.lane();
            }
            2 => r.every[k] = rng.keys(),
            3 => r.kills[k] = rng.keys(),
            4 => r.lane_kills[k] = rng.keys(),
            _ => {}
        }
        if rng.chance(3) {
            r.lane_kills[k].extend(rng.keys());
        }
        if let (Some(&t), true) = (r.succ[k].first(), rng.chance(6)) {
            r.edge_kills[k] = (0..1 + rng.below(2)).map(|_| (t, rng.keys())).collect();
        }
    }
    if rng.chance(2) {
        r.exit = rng.keys();
    }
    r
}

#[test]
fn the_solver_reaches_the_reference_fixpoint() {
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    for case in 0..3000 {
        let r = generate(&mut rng);
        let live = solve(&r);
        let (live_in, live_out) = reference(&r);
        assert_eq!(live.live_in, live_in, "case {case}: live-in of {r:?}");
        let got: Vec<&Bits> = (0..r.succ.len()).map(|k| live.live_out(k)).collect();
        let want: Vec<&Bits> = live_out.iter().collect();
        assert_eq!(got, want, "case {case}: live-out of {r:?}");
    }
}

#[test]
fn a_return_brings_back_only_its_own_calls_keys() {
    // 0 calls 4; 1 (continuation) reads k0; 2 calls 4; 3 ends;
    // 4 returns to 1 or 3. k0 is live into 4 from the first call only.
    let succ = vec![vec![4], vec![2], vec![4], vec![], vec![1, 3]];
    let flows = vec![
        vec![Flow::Call],
        vec![Flow::Plain],
        vec![Flow::Call],
        vec![],
        vec![Flow::Return { caller: 0 }, Flow::Return { caller: 2 }],
    ];
    let mut r = Rows::new(succ, flows);
    r.reads[1] = vec![0];
    let live = solve(&r);
    let has = |b: &[Bits]| b.iter().map(|b| b.has(0)).collect::<Vec<_>>();
    assert_eq!(has(&live.live_in), [true, true, false, false, true]);
    let out: Vec<bool> = (0..5).map(|k| live.live_out(k).has(0)).collect();
    assert_eq!(out, [true, false, false, false, true]);
}

#[test]
fn no_rows_have_no_liveness() {
    let live = solve(&Rows::new(Vec::new(), Vec::new()));
    assert!(live.live_in.is_empty());
}
