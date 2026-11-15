//! Instruction, snapshot and dataflow invariants.

use std::collections::{BTreeSet, VecDeque};
use std::sync::{Arc, OnceLock};

use thin_ir::cfg::{cfg_of, Cfg};
use thin_ir::sets::{Bits, Runs};
use thin_ir::solve::{solve_backward, solve_forward, solve_forward_all};
use thin_ir::{FieldId, FieldValue, Instr, Operand};
use thin_target::{Flags, Form, Kind};

/// A small deterministic generator (xorshift64*).
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// The forms of an abstract stream: an op, a label, a conditional branch,
/// an unconditional one and an end.
fn forms() -> &'static [Form; 5] {
    static FORMS: OnceLock<[Form; 5]> = OnceLock::new();
    FORMS.get_or_init(|| {
        [
            Form::plain(0, "op", Kind::Valu, Flags::VALU),
            Form::plain(-10, "ir.label", Kind::Label, Flags::PSEUDO),
            Form::plain(1, "br", Kind::Branch, Flags::BRANCH),
            Form::plain(2, "jmp", Kind::Branch, Flags::BRANCH | Flags::UNCOND),
            Form::plain(3, "end", Kind::Nop, Flags::ENDPGM | Flags::UNCOND),
        ]
    })
}

/// A random stream of `n` instructions over a few labels.
fn stream(rng: &mut Rng, n: usize) -> Vec<Arc<Instr>> {
    let f = forms();
    let name = |k: usize| -> Arc<str> { format!("L{k}").into() };
    (0..n)
        .map(|_| {
            let k = rng.below(5);
            let i = match k {
                1 => {
                    let field = (
                        FieldId::named("name").unwrap(),
                        FieldValue::Text(name(rng.below(4))),
                    );
                    Instr::new(&f[1], Vec::new(), vec![field], None)
                }
                2 | 3 => Instr::new(
                    &f[k],
                    vec![Operand::Block(name(rng.below(5)))],
                    Vec::new(),
                    None,
                ),
                _ => Instr::new(&f[k], Vec::new(), Vec::new(), None),
            };
            Arc::new(i)
        })
        .collect()
}

/// Checks the CFG's structural properties; blocks are maximal runs, a
/// branch always ending one, when no extra leaders split them.
fn check_cfg(g: &Cfg, instrs: &[Arc<Instr>], maximal: bool) {
    let n = g.succ.len();
    for t in 0..n {
        let mut want: Vec<usize> = Vec::new();
        for (k, s) in g.succ.iter().enumerate() {
            want.extend(s.iter().filter(|&&x| x == t).map(|_| k));
        }
        assert_eq!(g.preds[t], want, "preds are the transposed succ");
    }
    for j in 0..g.nblocks() {
        let r = g.block(j);
        for k in r.start..r.end - 1 {
            assert_eq!(
                g.succ[k],
                vec![k + 1],
                "a block's inner instruction falls through"
            );
            assert_eq!(g.preds[k + 1], vec![k], "a block has one entry");
        }
        let joined = r.start > 0
            && !instrs[r.start - 1].form().flags.has(Flags::BRANCH)
            && g.succ[r.start - 1] == vec![r.start]
            && g.preds[r.start] == vec![r.start - 1];
        assert!(
            !(maximal && joined),
            "block {j} could extend its predecessor"
        );
    }
    let mut seen = vec![false; n];
    let mut queue = VecDeque::from(if n > 0 { vec![0] } else { vec![] });
    if n > 0 {
        seen[0] = true;
    }
    while let Some(k) = queue.pop_front() {
        for &t in &g.succ[k] {
            if !std::mem::replace(&mut seen[t], true) {
                queue.push_back(t);
            }
        }
    }
    assert_eq!(g.reached(), seen, "reached is a BFS");
    let rpo = g.rpo();
    let reachable: BTreeSet<usize> = (0..n).filter(|&k| seen[k]).map(|k| g.block_of[k]).collect();
    assert_eq!(rpo.iter().copied().collect::<BTreeSet<_>>(), reachable);
    assert!(rpo.is_empty() || rpo[0] == 0, "rpo starts at block 0");
    let pos = |b: usize| rpo.iter().position(|&x| x == b).unwrap();
    for (q, t) in g.back_edges(&rpo) {
        assert!(pos(t) <= pos(q));
    }
    for &q in &rpo {
        for &t in &g.block_succ[q] {
            assert_eq!(g.back_edges(&rpo).contains(&(q, t)), pos(t) <= pos(q));
        }
    }
}

#[test]
fn cfgs_of_random_streams_hold_their_properties() {
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    for round in 0..400 {
        let instrs = stream(&mut rng, round % 24);
        for label_blocks in [false, true] {
            check_cfg(&cfg_of(&instrs, None, label_blocks), &instrs, !label_blocks);
        }
    }
}

/// The least fixpoint of a forward analysis by rounds over every block.
fn naive_forward(
    g: &Cfg,
    through: impl Fn(&BTreeSet<usize>, usize) -> BTreeSet<usize>,
) -> Vec<BTreeSet<usize>> {
    let mut state = vec![BTreeSet::new(); g.nblocks()];
    loop {
        let mut next = state.clone();
        for (j, s) in state.iter().enumerate() {
            let out = through(s, j);
            for &t in &g.block_succ[j] {
                next[t].extend(out.iter().copied());
            }
        }
        if next == state {
            return state;
        }
        state = next;
    }
}

#[test]
fn solvers_reach_the_naive_fixpoint() {
    let mut rng = Rng(7);
    let union = |a: &BTreeSet<usize>, b: &BTreeSet<usize>| a | b;
    for round in 0..300 {
        let instrs = stream(&mut rng, 1 + round % 20);
        let g = cfg_of(&instrs, None, false);
        let gen = |s: &BTreeSet<usize>, j: usize| {
            let mut s = s.clone();
            s.insert(j);
            s.remove(&((j + 1) % 4));
            s
        };
        let naive = naive_forward(&g, gen);
        let all = solve_forward_all(&g, BTreeSet::new(), vec![(0, BTreeSet::new())], gen, union);
        assert_eq!(all, naive);
        let reached = g.reached();
        let some = solve_forward(&g, vec![(0, BTreeSet::new())], gen, union, None);
        for j in 0..g.nblocks() {
            if reached[g.starts[j]] {
                assert!(some[j].as_ref().is_some_and(|s| s.is_subset(&naive[j])));
            } else {
                assert!(some[j].is_none());
            }
        }
    }
}

#[test]
fn backward_liveness_is_the_naive_fixpoint() {
    let mut rng = Rng(11);
    for round in 0..300 {
        let instrs = stream(&mut rng, 1 + round % 20);
        let g = cfg_of(&instrs, None, false);
        let through = |s: &BTreeSet<usize>, j: usize| {
            let mut s = s.clone();
            s.remove(&(j % 3));
            s.insert(j % 5);
            s
        };
        let merge = |l: &[&BTreeSet<usize>]| l.iter().fold(BTreeSet::new(), |a, b| &a | b);
        let got = solve_backward(&g, BTreeSet::new(), merge, through);
        let mut want = vec![BTreeSet::new(); g.nblocks()];
        loop {
            let next: Vec<_> = (0..g.nblocks())
                .map(|j| {
                    through(
                        &merge(
                            &g.block_succ[j]
                                .iter()
                                .map(|&s| &want[s])
                                .collect::<Vec<_>>(),
                        ),
                        j,
                    )
                })
                .collect();
            if next == want {
                break;
            }
            want = next;
        }
        assert_eq!(got, want);
    }
}

#[test]
fn bits_follow_a_reference_set() {
    let mut rng = Rng(3);
    let edges = [0, 1, 62, 63, 64, 65, 127, 128, 129, 191, 200];
    for _ in 0..200 {
        let (mut a, mut want) = (Bits::new(), BTreeSet::new());
        for _ in 0..40 {
            let x = edges[rng.below(edges.len())];
            let y = edges[rng.below(edges.len())];
            match rng.below(6) {
                0 => {
                    a.set(x);
                    want.insert(x);
                }
                1 => {
                    a.unset(x);
                    want.remove(&x);
                }
                2 => {
                    a.set_range(x, y);
                    want.extend(x..y);
                }
                3 => {
                    a.clear_range(x, y);
                    want.retain(|k| !(x..y).contains(k));
                }
                4 => {
                    let r = a.and_range(x, y);
                    assert_eq!(
                        r.ones(),
                        want.range(x..y.max(x)).copied().collect::<Vec<_>>()
                    );
                }
                _ => {
                    let mut o = Bits::new();
                    o.add([x, y]);
                    let ow: BTreeSet<usize> = [x, y].into();
                    let changes: Vec<(usize, bool)> = want
                        .symmetric_difference(&ow)
                        .map(|&k| (k, ow.contains(&k)))
                        .collect();
                    assert_eq!(a.changes(&o), changes);
                    assert_eq!(a.intersects(&o), !want.is_disjoint(&ow));
                    assert_eq!(a.meets(&o.sparse()), !want.is_disjoint(&ow));
                }
            }
            assert_eq!(a.ones(), want.iter().copied().collect::<Vec<_>>());
            assert_eq!(a.is_empty(), want.is_empty());
        }
    }
}

#[test]
fn runs_follow_a_reference_set() {
    let mut rng = Rng(5);
    for _ in 0..300 {
        let (mut a, mut want) = (Runs::new(), BTreeSet::new());
        let mut at = 0u32;
        for _ in 0..6 {
            at += rng.below(4) as u32;
            let end = at + rng.below(5) as u32;
            a.push_run(at, end);
            want.extend(at..end);
            at = end;
        }
        let mut b = Runs::new();
        let mut bw = BTreeSet::new();
        let mut at = rng.below(6) as u32;
        for _ in 0..4 {
            let end = at + 1 + rng.below(4) as u32;
            b.push_run(at, end);
            bw.extend(at..end);
            at = end + rng.below(5) as u32;
        }
        assert_eq!(a.first_common(&b), want.intersection(&bw).next().copied());
        assert_eq!(a.meets(&b), !want.is_disjoint(&bw));
        a.add(&b);
        want.extend(bw.iter().copied());
        let members: BTreeSet<u32> = a.runs().iter().flat_map(|&(x, y)| x..y).collect();
        assert_eq!(members, want);
        assert!(
            a.runs().windows(2).all(|w| w[0].1 < w[1].0),
            "canonical: sorted, apart"
        );
        assert_eq!(
            (a.first(), a.last()),
            (want.first().copied(), want.last().copied())
        );
        assert!((0..40).all(|r| a.has(r) == want.contains(&r)));
    }
}

/// A widening that keeps the old state ends the solve: the loop's state
/// grows under the merge, the widening refuses to move, and the block is
/// not visited again.
#[test]
fn a_widening_that_keeps_the_old_state_terminates() {
    use thin_ir::solve::Widen;
    let g = Cfg::from_succ_only(vec![vec![1], vec![0, 2], vec![]]);
    let mut visits = 0;
    let keep = |old: &u32, _: &u32| *old;
    let states = solve_forward(
        &g,
        vec![(0, 0u32)],
        |s, _| {
            visits += 1;
            assert!(visits < 1000, "the solver does not terminate");
            s + 1
        },
        |a, b| *a.max(b),
        Some(Widen {
            widen: &keep,
            patience: 1,
        }),
    );
    assert!(states.iter().all(Option::is_some));
}

/// One partial-write rule: lanes under a partial EXEC for per-lane units,
/// a writelane, a D16 access; never a scalar op.
#[test]
fn keeps_old_is_one_rule_for_every_unit() {
    let valu = Form::plain(0, "v_op", Kind::Valu, Flags::VALU);
    let mfma = Form::plain(1, "v_mfma", Kind::Matrix, Flags::VALU);
    let salu = Form::plain(2, "s_op", Kind::Salu, Flags::SALU);
    let lane = Form::plain(3, "v_writelane", Kind::Valu, Flags::VALU | Flags::WRITELANE);
    let d16 = Form::plain(4, "global_load_d16", Kind::Vmem, Flags::VMEM | Flags::D16);
    let forms: [&'static Form; 5] = [valu, mfma, salu, lane, d16].map(|f| &*Box::leak(Box::new(f)));
    let keeps =
        |f: &'static Form, partial| Instr::new(f, Vec::new(), Vec::new(), None).keeps_old(partial);
    assert_eq!(
        forms.map(|f| keeps(f, true)),
        [true, true, false, true, true]
    );
    assert_eq!(
        forms.map(|f| keeps(f, false)),
        [false, false, false, true, true]
    );
    // An SGPR destination is written whole, whatever EXEC is.
    let in_file = |f: &'static Form, file| {
        Instr::new(f, Vec::new(), Vec::new(), None).keeps_old_in(file, true)
    };
    assert_eq!(forms.map(|f| in_file(f, 's')), [false; 5]);
    assert_eq!(
        forms.map(|f| in_file(f, 'v')),
        forms.map(|f| keeps(f, true))
    );
}
