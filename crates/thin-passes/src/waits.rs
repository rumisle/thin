//! Wait counts: the count of every wait request, the waits
//! each instruction implies, the checks, and the target's wait
//! instructions in place of the markers.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use thin_ir::cfg::Cfg;
use thin_ir::solve::{solve_forward, solve_forward_on};
use thin_ir::{Attr, Diag, DiagClass, FieldValue, Instr, Part, Request, Snapshot, WaveClass};
use thin_target::{Flags, Kind, Target, WaitCounts};

use crate::hazards::wait_instrs;

/// A request's key: a marker's index, or an instruction's implied
/// (`use`) or publication (`pub`) request on a counter.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Key {
    /// The marker at this index.
    Marker(usize),
    /// Instruction k's implied request on a counter.
    Use(usize, usize),
    /// Instruction k's publication request on a counter.
    Pub(usize, usize),
}

impl Key {
    /// Its instruction, its rank (markers first) and its counter.
    fn order(self) -> (usize, u8, usize) {
        match self {
            Key::Marker(k) => (k, 0, 0),
            Key::Use(k, c) => (k, 1, c),
            Key::Pub(k, c) => (k, 2, c),
        }
    }

    /// The instruction it belongs to.
    pub fn index(self) -> usize {
        self.order().0
    }

    /// Whether it is an implied or a publication request.
    pub fn implied(self) -> bool {
        !matches!(self, Key::Marker(_))
    }
}

impl PartialOrd for Key {
    fn partial_cmp(&self, o: &Key) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}

impl Ord for Key {
    fn cmp(&self, o: &Key) -> Ordering {
        self.order().cmp(&o.order())
    }
}

/// A request's count; none is "no wait".
pub type Count = Option<u32>;

/// A count per request key; a missing key is "no wait".
pub type Decision = BTreeMap<Key, Count>;

/// The stronger of two counts: a count beats no wait, the smaller wins.
pub fn stronger(a: Count, b: Count) -> Count {
    match (a, b) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (a, b) => a.or(b),
    }
}

/// The cap on how many younger ops the state counts.
const CAP: u32 = 64;

/// A physical dword: (file, number).
type Dword = (char, u32);

/// The state at a point: per op in flight, the fewest younger ops
/// of its counter on any path; per dword, the loads that last wrote it.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct State {
    pend: BTreeMap<usize, u32>,
    rd: BTreeMap<Dword, BTreeSet<usize>>,
}

impl State {
    fn join(a: &State, b: &State) -> State {
        let mut out = a.clone();
        for (&op, &y) in &b.pend {
            let e = out.pend.entry(op).or_insert(y);
            *e = (*e).min(y);
        }
        for (d, ops) in &b.rd {
            out.rd.entry(*d).or_default().extend(ops);
        }
        out
    }
}

/// What an instruction does to the state.
#[derive(Clone, Debug)]
enum Act {
    Nothing,
    Mark(usize, Request),
    Wait(Vec<(usize, u32)>),
    Issue {
        counter: usize,
        unordered: bool,
        defs: Vec<Dword>,
        uses: Vec<Dword>,
    },
    Other {
        defs: Vec<Dword>,
        uses: Vec<Dword>,
    },
}

/// A loop the loop construct made, by its body's first and last
/// instruction.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Loop {
    /// Its stem.
    pub stem: String,
    /// Its body's first instruction.
    pub first: usize,
    /// Its body's last instruction.
    pub last: usize,
    /// Whether it is exempt from the steady-state check.
    pub conservative: bool,
}

/// The pass's explicit inputs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Options {
    /// Refuse counts that are not the only fixpoint (`waits_unique`).
    pub unique: bool,
    /// Every placed count 0 (`safe_waits`).
    pub safe: bool,
    /// The kernel is an ablation variant.
    pub ablation: bool,
}

/// The analysis of one snapshot.
pub struct Waits<'a> {
    s: &'a Snapshot,
    acts: Vec<Act>,
    classes: Vec<(WaveClass, Arc<Cfg>)>,
    coherent: Vec<bool>,
    pub_counter: Option<usize>,
}

impl<'a> Waits<'a> {
    /// The acts and classes of a snapshot.
    pub fn new(s: &'a Snapshot) -> Waits<'a> {
        let classes = s
            .wave_classes()
            .into_iter()
            .map(|c| (c.clone(), s.cfg(c.as_deref())))
            .collect();
        let acts = (0..s.instrs().len()).map(|k| act(s, k)).collect();
        let coherent: Vec<bool> = s.instrs().iter().map(|i| coherent(i)).collect();
        let firsts: BTreeSet<usize> = (s.instrs().iter().zip(&coherent))
            .filter(|(_, &c)| c)
            .filter_map(|(i, _)| i.form().counter)
            .collect();
        assert!(firsts.len() <= 1, "coherent ops on more than one counter ");
        Waits {
            s,
            acts,
            classes,
            coherent,
            pub_counter: firsts.into_iter().next(),
        }
    }

    fn largest(&self, c: usize) -> u32 {
        self.s.target().definition().counters[c].max
    }

    fn unordered(&self, op: usize) -> bool {
        let f = self.s.instrs()[op].form();
        f.counter.is_some() && f.unordered
    }

    fn counter(&self, op: usize) -> Option<usize> {
        self.s.instrs()[op].form().counter
    }

    /// The count a set of in-flight ops of counter c needs.
    fn count_of(&self, st: &State, ops: &BTreeSet<usize>, c: usize) -> Count {
        if ops.is_empty() {
            return None;
        }
        if ops.iter().any(|&op| self.unordered(op)) {
            return Some(0);
        }
        let fewest = ops.iter().map(|op| st.pend[op]).min().unwrap_or(0);
        Some(fewest.min(CAP).min(self.largest(c)))
    }

    /// Removes every op of counter c that a count of m retires.
    fn retire(&self, st: &mut State, c: usize, m: u32) {
        st.pend.retain(|&op, &mut y| {
            !(self.counter(op) == Some(c) && (m == 0 || y >= m && !self.unordered(op)))
        });
    }

    /// The in-flight loads that last wrote any of the dwords.
    fn writers(&self, st: &State, dwords: &[Dword]) -> BTreeSet<usize> {
        let found = dwords.iter().filter_map(|d| st.rd.get(d)).flatten();
        found
            .copied()
            .filter(|op| st.pend.contains_key(op))
            .collect()
    }

    /// A marker's count in a state.
    fn resolve(&self, st: &State, c: usize, r: &Request) -> Count {
        let of_c = |op: &usize| self.counter(*op) == Some(c);
        let ops: BTreeSet<usize> = match r {
            Request::Regs { regs, regions } => {
                let dwords: Vec<Dword> = regs.iter().flat_map(|r| dwords(self.s, r)).collect();
                let mut ops = self.writers(st, &dwords);
                if !regions.is_empty() {
                    let answers =
                        |op: usize| self.s.answers(op).iter().any(|a| regions.contains(a));
                    ops.extend(st.pend.keys().copied().filter(|&op| answers(op)));
                }
                ops.into_iter().filter(of_c).collect()
            }
            Request::Tags(tags) => (st.pend.keys().copied())
                .filter(|op| {
                    of_c(op)
                        && self.s.instrs()[*op]
                            .attrs()
                            .tag
                            .as_ref()
                            .is_some_and(|t| tags.contains(t))
                })
                .collect(),
            Request::All => st.pend.keys().copied().filter(of_c).collect(),
            Request::Left(n) => {
                let any = st.pend.keys().any(of_c);
                return any
                    .then(|| (*n).clamp(0, u32::MAX as i128) as u32)
                    .map(|n| n.min(self.largest(c)));
            }
            Request::Smem => return st.pend.keys().any(|&op| self.unordered(op)).then_some(0),
        };
        self.count_of(st, &ops, c)
    }

    /// The implied requests of an instruction and its publication request.
    fn implied(&self, st: &mut State, k: usize, decide: &mut dyn FnMut(Key, Count) -> Count) {
        let (defs, uses) = match &self.acts[k] {
            Act::Issue { defs, uses, .. } | Act::Other { defs, uses } => (defs, uses),
            _ => return,
        };
        // A load over a load still in flight needs a wait too, of its own
        // counter included: the data may land out of order, so both destinations must be protected.
        let mut t = self.writers(st, uses);
        t.extend(self.writers(st, defs));
        let counters: BTreeSet<usize> = t.iter().filter_map(|&op| self.counter(op)).collect();
        for cc in counters {
            let ops: BTreeSet<usize> = t
                .iter()
                .copied()
                .filter(|&op| self.counter(op) == Some(cc))
                .collect();
            if let Some(v) = decide(Key::Use(k, cc), self.count_of(st, &ops, cc)) {
                self.retire(st, cc, v);
            }
        }
        self.publication(st, k, decide);
    }

    /// A device-scope atomic's request for the coherent ops in flight.
    fn publication(&self, st: &mut State, k: usize, decide: &mut dyn FnMut(Key, Count) -> Count) {
        let i = &self.s.instrs()[k];
        let f = i.form();
        let device =
            f.flags.has(Flags::ATOMIC) && !i.attrs().noorder && i.attrs().published.is_none();
        let Some(c) = self.pub_counter.filter(|_| device) else {
            return;
        };
        let data = f.flags.has(Flags::FP_ATOMIC);
        let ops: BTreeSet<usize> = (st.pend.keys().copied())
            .filter(|&op| {
                self.coherent[op]
                    && !(data && self.s.instrs()[op].form().flags.has(Flags::FP_ATOMIC))
            })
            .collect();
        let count = (!ops.is_empty()).then(|| {
            ops.iter()
                .map(|op| st.pend[op])
                .min()
                .unwrap_or(0)
                .min(CAP)
                .min(self.largest(c))
        });
        if let Some(v) = decide(Key::Pub(k, c), count) {
            self.retire(st, c, v);
        }
    }

    /// One instruction's step.
    fn step(&self, st: &mut State, k: usize, decide: &mut dyn FnMut(Key, Count) -> Count) {
        match &self.acts[k] {
            Act::Nothing => {}
            Act::Mark(c, r) => {
                let here = self.resolve(st, *c, r);
                if let Some(v) = decide(Key::Marker(k), here) {
                    self.retire(st, *c, v);
                }
            }
            Act::Wait(counts) => counts.iter().for_each(|&(c, m)| self.retire(st, c, m)),
            Act::Issue {
                counter,
                unordered,
                defs,
                ..
            } => {
                self.implied(st, k, decide);
                if !unordered {
                    for (&op, y) in st.pend.iter_mut() {
                        if self.counter(op) == Some(*counter) {
                            *y = (*y + 1).min(CAP);
                        }
                    }
                }
                st.pend.insert(k, 0);
                for d in defs {
                    st.rd.insert(*d, BTreeSet::from([k]));
                }
            }
            Act::Other { defs, .. } => {
                self.implied(st, k, decide);
                defs.iter().for_each(|d| _ = st.rd.remove(d));
            }
        }
    }

    /// A block run from a state.
    fn run(
        &self,
        g: &Cfg,
        j: usize,
        st: &State,
        decide: &mut dyn FnMut(Key, Count) -> Count,
    ) -> State {
        let mut st = st.clone();
        g.block(j).for_each(|k| self.step(&mut st, k, decide));
        st
    }

    /// Per class, the block entry states with decisions D.
    fn flow(&self, d: &Decision, g: &Cfg) -> Vec<Option<State>> {
        if g.nblocks() == 0 {
            return Vec::new();
        }
        let through =
            |st: &State, j: usize| self.run(g, j, st, &mut |key, _| d.get(&key).copied().flatten());
        solve_forward(g, vec![(0, State::default())], through, State::join, None)
    }

    /// Every marker key, no wait.
    fn markers(&self) -> Decision {
        self.s
            .requests()
            .keys()
            .map(|&k| (Key::Marker(k), None))
            .collect()
    }

    /// The counts a decision leads to.
    pub fn f(&self, d: &Decision) -> Decision {
        let mut out = self.markers();
        let mut record = |key: Key, here: Count| {
            let e = out.entry(key).or_insert(here);
            *e = stronger(*e, here);
        };
        for (_, g) in &self.classes {
            for (j, entry) in self.flow(d, g).iter().enumerate() {
                let Some(entry) = entry else { continue };
                self.run(g, j, entry, &mut |key, here| {
                    record(key, here);
                    d.get(&key).copied().flatten()
                });
            }
        }
        out.retain(|k, v| !k.implied() || v.is_some());
        out
    }

    /// The fast search for a fixpoint.
    pub fn find(&self) -> Decision {
        let n = self.classes.len();
        let mut last: Vec<Decision> = vec![Decision::new(); n];
        let mut back: Vec<BTreeMap<usize, State>> = vec![BTreeMap::new(); n];
        for _ in 0..200 {
            let mut stable = true;
            for c in 0..n {
                let others = strongest((0..n).filter(|&o| o != c).map(|o| &last[o]));
                let (mine, exits) = self.find_pass(c, &others, &back[c]);
                if mine != last[c] || exits != back[c] {
                    stable = false;
                }
                (last[c], back[c]) = (mine, exits);
            }
            if stable {
                break;
            }
        }
        let mut out = self.markers();
        for (k, v) in strongest(last.iter()) {
            out.insert(k, v);
        }
        out.retain(|k, v| !k.implied() || v.is_some());
        out
    }

    /// One pass of `find` over class c: its decisions and the exit states
    /// of its back-edge sources.
    fn find_pass(
        &self,
        c: usize,
        others: &Decision,
        back: &BTreeMap<usize, State>,
    ) -> (Decision, BTreeMap<usize, State>) {
        let g = &self.classes[c].1;
        let rpo = g.rpo();
        let edges = g.back_edges(&rpo);
        let reached: BTreeSet<usize> = rpo.iter().copied().collect();
        let mut exits: BTreeMap<usize, State> = BTreeMap::new();
        let mut mine = Decision::new();
        for &b in &rpo {
            let mut entry = (b == 0).then(State::default);
            for &p in g.block_preds[b].iter().filter(|p| reached.contains(p)) {
                let from = if edges.contains(&(p, b)) {
                    back.get(&p)
                } else {
                    exits.get(&p)
                };
                if let Some(x) = from {
                    entry = Some(entry.map_or_else(|| x.clone(), |e| State::join(&e, x)));
                }
            }
            let Some(entry) = entry else { continue };
            let exit = self.run(g, b, &entry, &mut |key, here| {
                mine.insert(key, here);
                stronger(here, others.get(&key).copied().flatten())
            });
            exits.insert(b, exit);
        }
        let sources: BTreeSet<usize> = edges.iter().map(|e| e.0).collect();
        exits.retain(|b, _| sources.contains(b));
        (mine, exits)
    }

    /// The limits of F's iterates from "nothing lands":
    /// (odd, even), or none within 1000 rounds.
    pub fn bounds(&self) -> Option<(Decision, Decision)> {
        let mut seq = vec![Decision::new()];
        for _ in 0..1000 {
            let next = self.f(seq.last().expect("a start"));
            seq.push(next);
            let l = seq.len();
            if l >= 5 && seq[l - 1] == seq[l - 3] && seq[l - 2] == seq[l - 4] {
                let (a, b) = (seq[l - 1].clone(), seq[l - 2].clone());
                return Some(if (l - 1) % 2 == 1 { (a, b) } else { (b, a) });
            }
        }
        None
    }

    /// The counts: find, a few rounds of F, else bounds' odd.
    pub fn solve(&self) -> Result<Decision, Diag> {
        let mut d = self.find();
        for _ in 0..4 {
            let f = self.f(&d);
            if f == d {
                return Ok(d);
            }
            d = f;
        }
        self.bounds().map(|(odd, _)| odd).ok_or_else(no_fixpoint)
    }

    /// The counts when they must be unique.
    pub fn forced(&self) -> Result<Decision, Diag> {
        let (odd, even) = self.bounds().ok_or_else(no_fixpoint)?;
        if odd == even {
            return Ok(odd);
        }
        let found = self.find();
        let why = if self.f(&found) == found {
            "two sets of counts hold, each wait just enough"
        } else {
            "no counts are each just enough"
        };
        let keys: BTreeSet<Key> = odd.keys().chain(even.keys()).copied().collect();
        let open: Vec<Key> = keys
            .into_iter()
            .filter(|k| odd.get(k) != even.get(k))
            .take(8)
            .collect();
        let mut msg = vec![Part::Text(format!("wait counts not unique ({why}):"))];
        for key in open {
            let (k, c) = (key.index(), self.key_counter(key));
            let name = self.s.target().definition().counters[c].name;
            let (a, b) = (
                count_text(odd.get(&key).copied().flatten()),
                count_text(even.get(&key).copied().flatten()),
            );
            msg.push(Part::Text(format!("\n  a {name} wait at ")));
            msg.push(Part::Instr(k));
            msg.push(Part::Where(k));
            msg.push(Part::Text(format!(
                " ({}): {a} or {b}",
                self.label_before(k)
            )));
        }
        Err(Diag {
            class: DiagClass::Schedule,
            loc: 0,
            msg,
            notes: Vec::new(),
        })
    }

    fn key_counter(&self, key: Key) -> usize {
        match key {
            Key::Use(_, c) | Key::Pub(_, c) => c,
            Key::Marker(k) => self.s.requests().get(&k).map_or(0, |r| r.0),
        }
    }

    /// "after LABEL": the nearest label before k, or "the entry".
    fn label_before(&self, k: usize) -> String {
        (self.s.instrs()[..k].iter().rev())
            .find(|i| i.form().kind == Kind::Label)
            .and_then(|i| i.field_named("name").map(FieldValue::py_str))
            .map_or("after the entry".into(), |l| format!("after {l}"))
    }
}

/// The strongest count per key over decisions.
fn strongest<'d>(ds: impl Iterator<Item = &'d Decision>) -> Decision {
    let mut out = Decision::new();
    for d in ds {
        for (&k, &v) in d {
            let e = out.entry(k).or_insert(v);
            *e = stronger(*e, v);
        }
    }
    out
}

fn count_text(c: Count) -> String {
    c.map_or("no wait".into(), |v| v.to_string())
}

fn no_fixpoint() -> Diag {
    let msg = vec![Part::Text("the wait-count search found no fixpoint".into())];
    Diag {
        class: DiagClass::NoFixpoint,
        loc: 0,
        msg,
        notes: Vec::new(),
    }
}

/// A register's physical dwords.
fn dwords(s: &Snapshot, r: &thin_ir::Reg) -> Vec<Dword> {
    let p = s.phys(r);
    (p..p + r.n).map(|d| (r.file, d)).collect()
}

/// Whether an op is a coherent op: an FP_ATOMIC op.
fn coherent(i: &Instr) -> bool {
    i.form().flags.has(Flags::FP_ATOMIC)
}

/// Instruction k's act.
fn act(s: &Snapshot, k: usize) -> Act {
    let i = &s.instrs()[k];
    let f = i.form();
    if let Some((c, r)) = s.requests().get(&k) {
        let ablated = i.field_named("ablated").is_some_and(FieldValue::truthy);
        return if ablated {
            Act::Nothing
        } else {
            Act::Mark(*c, r.clone())
        };
    }
    if matches!(f.kind, Kind::Label | Kind::Comment) {
        return Act::Nothing;
    }
    if f.flags.has(Flags::WAITCNT) {
        let counters = &s.target().definition().counters;
        let counts = (counters.iter().enumerate())
            .filter_map(|(c, ct)| match i.field_named(ct.field) {
                Some(FieldValue::Int(m)) => Some((c, *m as u32)),
                _ => None,
            })
            .collect();
        return Act::Wait(counts);
    }
    let defs: Vec<Dword> = i.defs().flat_map(|r| dwords(s, r)).collect();
    let uses: Vec<Dword> = i.uses().flat_map(|r| dwords(s, r)).collect();
    match f.counter {
        Some(counter) => Act::Issue {
            counter,
            unordered: f.unordered,
            defs,
            uses,
        },
        None if defs.is_empty() && uses.is_empty() => Act::Nothing,
        None => Act::Other { defs, uses },
    }
}

/// The wait pass's result: the counts and the stream with the waits in
/// place, each placed instruction with the index it was placed for.
pub struct Placed {
    /// The counts.
    pub counts: Decision,
    /// The new stream.
    pub stream: Vec<(Arc<Instr>, Option<usize>)>,
}

/// The wait pass on a snapshot whose loops are `loops`.
pub fn pass(s: &Snapshot, loops: &[Loop], o: Options) -> Result<Placed, Diag> {
    let w = Waits::new(s);
    let mut counts = if o.unique { w.forced()? } else { w.solve()? };
    if o.ablation {
        counts.retain(|k, _| !k.implied());
    }
    if let Some((&key, &v)) = counts.iter().find(|(k, v)| k.implied() && v.is_some()) {
        return Err(implied_refusal(&w, key, v.expect("a count")));
    }
    w.steady(&counts, loops)?;
    if o.safe {
        counts.values_mut().for_each(|v| *v = v.map(|_| 0));
    }
    let stream = place(s, &counts);
    Ok(Placed { counts, stream })
}

/// The refusal of an implied key that still waits.
fn implied_refusal(w: &Waits<'_>, key: Key, v: u32) -> Diag {
    let k = key.index();
    let after = w.label_before(k);
    let text = match key {
        Key::Pub(..) => format!(
            " ({after}): a device-scope atomic with a coherent store or a data atomic still outstanding on some path \
             (it needs vmcnt({v})): the partials must be at the coherent level before the ticket (wait for them first)"
        ),
        _ => {
            let field = w.s.target().definition().counters[w.key_counter(key)].field;
            format!(
                " ({after}) touches a register while the load on it may be in flight on some path: it needs \
                 {field}({v}) - add a wait in the schedule"
            )
        }
    };
    Diag {
        class: DiagClass::Schedule,
        loc: 0,
        msg: vec![Part::Instr(k), Part::Text(text)],
        notes: Vec::new(),
    }
}

impl Waits<'_> {
    /// The steady-state check under decision d.
    fn steady(&self, d: &Decision, loops: &[Loop]) -> Result<(), Diag> {
        let mut findings: Vec<(Key, u32, Count, &Loop)> = Vec::new();
        for lp in loops.iter().filter(|l| !l.conservative) {
            for (_, g) in &self.classes {
                findings.extend(
                    self.steady_class(d, g, lp)
                        .into_iter()
                        .map(|(k, e, s)| (k, e, s, lp)),
                );
            }
        }
        findings.sort_by_key(|f| f.0.index());
        let Some((key, entry, steady, lp)) = findings.into_iter().next() else {
            return Ok(());
        };
        let c = self.key_counter(key);
        let ct = &self.s.target().definition().counters[c];
        let k = key.index();
        let mut msg = vec![Part::Text(format!("loop {}: ", lp.stem))];
        match steady {
            None => {
                msg.push(Part::Text(format!(
                    "the steady-state body has no {} wait at",
                    ct.name
                )));
                msg.push(Part::Where(k));
                msg.push(Part::Text(format!(
                    " where the loop entry needs {}({entry}) - something issued before the loop is still in flight \
                     there: wait for it before the loop, or peel (b.loop(peel=1)), or conservative=True (the entry's \
                     wait every iteration); or make the entry match the steady state: the ops of this counter a \
                     steady iteration has in flight here, the prologue issues too",
                    ct.field
                )));
            }
            Some(sv) => {
                msg.push(Part::Text(format!(
                    "the steady-state wait {}({sv}) at",
                    ct.field
                )));
                msg.push(Part::Where(k));
                msg.push(Part::Text(format!(
                    " is insufficient at the loop entry (needs {}({entry}): the entry has {} fewer {} ops in flight \
                     after the load than a steady iteration); wait for it before the loop, or peel \
                     (b.loop(peel=1)), or conservative=True",
                    ct.field,
                    sv.saturating_sub(entry),
                    ct.name
                )));
            }
        }
        Err(Diag {
            class: DiagClass::Schedule,
            loc: 0,
            msg,
            notes: Vec::new(),
        })
    }

    /// One class's findings for a loop: (key, entry count, steady count).
    fn steady_class(&self, d: &Decision, g: &Cfg, lp: &Loop) -> Vec<(Key, u32, Count)> {
        let rpo = g.rpo();
        let edges = g.back_edges(&rpo);
        let reached: BTreeSet<usize> = rpo.iter().copied().collect();
        let heads = edges
            .iter()
            .filter(|(q, t)| g.starts[*t] <= lp.first && g.end(*q) > lp.last);
        let Some(t) = heads.map(|e| e.1).max() else {
            return Vec::new();
        };
        let srcs: Vec<usize> = edges
            .iter()
            .filter(|e| e.1 == t && reached.contains(&e.0))
            .map(|e| e.0)
            .collect();
        if srcs.is_empty() {
            return Vec::new();
        }
        let hi = srcs.iter().map(|&q| g.end(q)).max().expect("a source");
        let flow = self.flow(d, g);
        let exit = |b: usize| {
            flow[b]
                .as_ref()
                .map(|st| self.run(g, b, st, &mut |key, _| d.get(&key).copied().flatten()))
        };
        let join_all = |bs: &mut dyn Iterator<Item = usize>| {
            bs.filter_map(&exit).reduce(|a, b| State::join(&a, &b))
        };
        let preds = g.block_preds[t]
            .iter()
            .copied()
            .filter(|p| reached.contains(p) && !edges.contains(&(*p, t)));
        let mut entry = join_all(&mut preds.into_iter());
        if t == 0 {
            entry = Some(entry.map_or_else(State::default, |e| State::join(&e, &State::default())));
        }
        let (Some(entry), Some(back)) = (entry, join_all(&mut srcs.iter().copied())) else {
            return Vec::new();
        };
        let lo = g.starts[t];
        let inside = |b: usize| (lo..hi).contains(&g.starts[b]);
        let evaluate = |start: State| {
            let mut seen: Vec<(Key, Count)> = Vec::new();
            let through = |st: &State, j: usize| {
                self.run(g, j, st, &mut |key, here| {
                    if (lp.first..=lp.last).contains(&key.index()) {
                        match seen.iter_mut().find(|e| e.0 == key) {
                            Some(e) => e.1 = stronger(e.1, here),
                            None => seen.push((key, here)),
                        }
                    }
                    d.get(&key).copied().flatten()
                })
            };
            solve_forward_on(g, vec![(t, start)], through, State::join, None, |_, to| {
                to != t && inside(to)
            });
            seen
        };
        let (at_entry, at_back) = (evaluate(entry), evaluate(back));
        let inf = |c: Count| c.unwrap_or(u32::MAX);
        (at_entry.into_iter())
            .filter_map(|(key, e)| {
                let b = at_back.iter().find(|x| x.0 == key).and_then(|x| x.1);
                (inf(e) < inf(b)).then(|| (key, e.expect("a count below the steady one"), b))
            })
            .collect()
    }
}

/// The stream with the counts in place.
pub fn place(s: &Snapshot, counts: &Decision) -> Vec<(Arc<Instr>, Option<usize>)> {
    let t = s.target();
    let mut items: Vec<(Arc<Instr>, Option<usize>)> = Vec::new();
    for (k, i) in s.instrs().iter().enumerate() {
        let mut implied: Vec<(&str, u32, usize)> = (counts.iter())
            .filter_map(|(key, v)| match key {
                Key::Use(q, c) | Key::Pub(q, c) if *q == k => {
                    Some((t.definition().counters[*c].name, (*v)?, *c))
                }
                _ => None,
            })
            .collect();
        implied.sort();
        for (_, v, c) in implied {
            items.extend(wait(t, c, v, "pass: a register it names").map(|w| (w, Some(k))));
        }
        match s.requests().get(&k) {
            Some((c, _)) => {
                if let Some(v) = counts.get(&Key::Marker(k)).copied().flatten() {
                    let said = i
                        .field_named("said")
                        .map(FieldValue::py_str)
                        .unwrap_or_default();
                    items.extend(wait(t, *c, v, &said).map(|w| (w, Some(k))));
                }
            }
            None => items.push((i.clone(), None)),
        }
    }
    merge(t, items)
}

/// The wait instruction of one count on a counter, with a comment.
fn wait(t: &'static Target, c: usize, v: u32, comment: &str) -> Option<Arc<Instr>> {
    let fields = t.wait_fields();
    let field = fields
        .iter()
        .position(|f| *f == t.definition().counters[c].field)?;
    let i = wait_instrs(t, &WaitCounts::one(fields.len(), field, v))
        .into_iter()
        .next()?;
    Some(Arc::new(
        i.with_attr(Attr::Comment(comment.into()))
            .expect("a comment"),
    ))
}

/// Whether an instruction is a count-less wait.
fn countless(i: &Instr) -> bool {
    i.form().wait_text.is_some() && i.ops().is_empty()
}

/// Merges runs of count-less waits into one instruction per target wait
/// instruction, per field the smallest count, comments joined (each
/// once: one wait's markers on several counters share its comment).
fn merge(
    t: &'static Target,
    items: Vec<(Arc<Instr>, Option<usize>)>,
) -> Vec<(Arc<Instr>, Option<usize>)> {
    let fields = t.wait_fields();
    let mut out = Vec::new();
    let mut run: Vec<(Arc<Instr>, Option<usize>)> = Vec::new();
    let flush = |run: &mut Vec<(Arc<Instr>, Option<usize>)>, out: &mut Vec<_>| {
        if run.len() <= 1 {
            out.append(run);
            return;
        }
        let mut counts = WaitCounts::empty(fields.len());
        let mut comments: Vec<String> = Vec::new();
        for (i, _) in run.iter() {
            for (id, v) in i.fields() {
                if let (Some(f), Some(v)) = (fields.iter().position(|f| *f == id.name()), v.int()) {
                    counts = counts.min(&WaitCounts::one(fields.len(), f, v as u32));
                }
            }
            for c in i.attrs().comment.iter() {
                if !comments.iter().any(|x| **x == **c) {
                    comments.push(c.to_string());
                }
            }
        }
        let origin = run.iter().find_map(|r| r.1);
        let comment: Arc<str> = comments.join("; ").into();
        for i in wait_instrs(t, &counts) {
            out.push((
                Arc::new(
                    i.with_attr(Attr::Comment(comment.clone()))
                        .expect("a comment"),
                ),
                origin,
            ));
        }
        run.clear();
    };
    for item in items {
        if countless(&item.0) {
            run.push(item);
        } else {
            flush(&mut run, &mut out);
            out.push(item);
        }
    }
    flush(&mut run, &mut out);
    out
}
