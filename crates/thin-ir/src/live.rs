//! Backward liveness of dense keys by call context and lane class: the one
//! solver of every analysis that keeps call sites apart.
//!
//! A live fact is a key, in a lane class, under a call context. Going back
//! over a row, the current EXEC's class becomes the class the row names,
//! the row's kills end keys, and its reads make keys live for every caller
//! (context none). Going back over an edge, each fact moves by the edge's
//! [`Flow`] ([`Flow::back`]) and loses the keys an edge claim ends on it.
//! [`solve`] gives the least fixpoint with contexts and lanes merged, per
//! row.
//!
//! Every row treats each context alike, and every step is a union of
//! per-fact moves, so the solver keeps contexts only between blocks: a
//! block's effect is computed once, as where each lane class goes and
//! what it loses, plus the facts the block makes live; a block's state is
//! a sparse set of (context, lane) slots over one arena of words; the
//! per-row results come from a single walk with contexts merged.

use std::collections::{BTreeMap, BinaryHeap, HashMap};
use std::ops::Range;
use std::rc::Rc;

use crate::control::{Context, Flow};
use crate::sets::Bits;

/// A lane class of the liveness: a known mask, a lane region's token, the
/// current EXEC, or every lane.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Lane {
    /// A known EXEC mask.
    Mask(u64),
    /// A lane region's token.
    Region(i128),
    /// The current EXEC.
    Cur,
    /// Every lane.
    Any,
}

/// What EXEC is known to be at a row.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Exec {
    /// A known mask.
    Mask(u64),
    /// A lane region's token.
    Region(i128),
    /// Not known.
    Unknown,
}

impl Exec {
    /// The lane class it is, unless unknown.
    fn lane(self) -> Option<Lane> {
        match self {
            Exec::Mask(m) => Some(Lane::Mask(m)),
            Exec::Region(r) => Some(Lane::Region(r)),
            Exec::Unknown => None,
        }
    }
}

/// Whether class `c` loses a lane kill made under EXEC `e`: the current
/// EXEC always; a known class when `e` covers it; every lane never.
fn lane_killed(c: Lane, e: Exec) -> bool {
    match (c, e) {
        (Lane::Cur, _) => true,
        (Lane::Mask(c), Exec::Mask(m)) => c & !m == 0,
        (Lane::Region(c), Exec::Region(r)) => c == r,
        _ => false,
    }
}

/// The facts a liveness reads, per row; keys are small ints.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Rows {
    /// Successors.
    pub succ: Vec<Vec<usize>>,
    /// Per successor edge, how it relates to calls (a row with none: all
    /// plain), so a key is live after a return only along the calls it
    /// crosses.
    pub flows: Vec<Vec<Flow>>,
    /// Keys an edge claim ends on one edge only: (the successor, the
    /// keys). Every claim on an edge ends its keys.
    pub edge_kills: Vec<Vec<(usize, Vec<usize>)>>,
    /// The keys read, in `read_lanes`.
    pub reads: Vec<Vec<usize>>,
    /// The lane class of the reads.
    pub read_lanes: Vec<Lane>,
    /// The keys read in every lane.
    pub every: Vec<Vec<usize>>,
    /// The keys the row ends in every lane.
    pub kills: Vec<Vec<usize>>,
    /// The keys the row ends in the lanes of its EXEC.
    pub lane_kills: Vec<Vec<usize>>,
    /// EXEC on entry.
    pub exec_in: Vec<Exec>,
    /// Whether the row writes EXEC.
    pub writes_exec: Vec<bool>,
    /// EXEC after an EXEC write.
    pub exec_out: Vec<Exec>,
    /// The keys live in every lane after a row with no successor.
    pub exit: Vec<usize>,
}

impl Rows {
    /// Rows with these edges that read, end and know nothing.
    pub fn new(succ: Vec<Vec<usize>>, flows: Vec<Vec<Flow>>) -> Rows {
        let n = succ.len();
        Rows {
            succ,
            flows,
            edge_kills: vec![Vec::new(); n],
            reads: vec![Vec::new(); n],
            read_lanes: vec![Lane::Cur; n],
            every: vec![Vec::new(); n],
            kills: vec![Vec::new(); n],
            lane_kills: vec![Vec::new(); n],
            exec_in: vec![Exec::Unknown; n],
            writes_exec: vec![false; n],
            exec_out: vec![Exec::Unknown; n],
            exit: Vec::new(),
        }
    }

    /// The flow of row k's edge e.
    fn flow(&self, k: usize, e: usize) -> Flow {
        (self.flows.get(k))
            .and_then(|f| f.get(e))
            .copied()
            .unwrap_or_default()
    }

    /// Whether row k's only edge goes on to the next row unchanged.
    fn falls_through(&self, k: usize) -> bool {
        self.succ[k][..] == [k + 1]
            && self.flow(k, 0) == Flow::Plain
            && self.edge_kills[k].is_empty()
    }
}

/// Per row, the keys live on entry and on exit.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Liveness {
    /// Live before each row.
    pub live_in: Vec<Bits>,
    /// Live after each row that does not fall through to the next; the
    /// others' is the next row's live-in.
    exits: Vec<Option<Bits>>,
}

impl Liveness {
    /// The keys live after row k.
    pub fn live_out(&self, k: usize) -> &Bits {
        self.exits[k]
            .as_ref()
            .unwrap_or_else(|| &self.live_in[k + 1])
    }
}

/// A lane class's dense id: the current EXEC and every lane first, then
/// the others in the order met.
type LaneId = u32;
const CUR: LaneId = 0;
const ANY: LaneId = 1;

/// Where facts are kept: a call context and a lane class.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Slot {
    ctx: Context,
    lane: LaneId,
}

impl Slot {
    /// A slot of context none, where reads put facts.
    fn top(lane: LaneId) -> Slot {
        Slot { ctx: None, lane }
    }
}

/// Adds `key` to a set of words.
fn set(words: &mut [u64], key: usize) {
    words[key / 64] |= 1 << (key % 64);
}

/// Removes `key` from a set of words.
fn unset(words: &mut [u64], key: usize) {
    words[key / 64] &= !(1 << (key % 64));
}

/// A set of live facts: per occupied slot, its keys as `width` words of
/// an arena. Slots are only added: a slot may hold no key.
#[derive(Clone, Debug)]
struct State {
    width: usize,
    /// The slots, ascending, with their words' offsets.
    index: Vec<(Slot, usize)>,
    words: Vec<u64>,
}

impl State {
    fn new(width: usize) -> State {
        State {
            width,
            index: Vec::new(),
            words: Vec::new(),
        }
    }

    /// Empties it, keeping its storage.
    fn clear(&mut self) {
        self.index.clear();
        self.words.clear();
    }

    /// The offset of `slot`'s words, an empty slot added if absent.
    fn offset(&mut self, slot: Slot) -> usize {
        match self.index.binary_search_by_key(&slot, |e| e.0) {
            Ok(i) => self.index[i].1,
            Err(i) => {
                let at = self.words.len();
                self.words.resize(at + self.width, 0);
                self.index.insert(i, (slot, at));
                at
            }
        }
    }

    /// The words of `slot`, an empty slot added if absent.
    fn slot(&mut self, slot: Slot) -> &mut [u64] {
        let at = self.offset(slot);
        &mut self.words[at..at + self.width]
    }

    /// Adds `keys` but `lost` to `slot`; whether a key was new.
    fn add(&mut self, slot: Slot, keys: &[u64], lost: Option<&[u64]>) -> bool {
        let kept = |w: usize| keys[w] & !lost.map_or(0, |l| l[w]);
        if self.index.binary_search_by_key(&slot, |e| e.0).is_err()
            && (0..self.width).all(|w| kept(w) == 0)
        {
            return false;
        }
        let mut new = 0;
        for (w, word) in self.slot(slot).iter_mut().enumerate() {
            new |= kept(w) & !*word;
            *word |= kept(w);
        }
        new != 0
    }

    /// [`State::add`], adding the keys that were new to `fresh` too.
    fn add_fresh(
        &mut self,
        fresh: &mut State,
        slot: Slot,
        keys: &[u64],
        lost: Option<&[u64]>,
    ) -> bool {
        let kept = |w: usize| keys[w] & !lost.map_or(0, |l| l[w]);
        let old = match self.index.binary_search_by_key(&slot, |e| e.0) {
            Ok(i) => &self.words[self.index[i].1..][..self.width],
            Err(_) => &[],
        };
        let old = |w: usize| old.get(w).copied().unwrap_or(0);
        if (0..self.width).all(|w| kept(w) & !old(w) == 0) {
            return false;
        }
        let (at, to) = (self.offset(slot), fresh.offset(slot));
        for w in 0..self.width {
            let new = kept(w) & !self.words[at + w];
            self.words[at + w] |= new;
            fresh.words[to + w] |= new;
        }
        true
    }

    /// The slots with their keys.
    fn iter(&self) -> impl Iterator<Item = (Slot, &[u64])> {
        (self.index.iter()).map(|&(s, at)| (s, &self.words[at..at + self.width]))
    }

    /// Every key of every slot.
    fn flat(&self) -> Bits {
        let mut out = vec![0; self.width];
        for (_, keys) in self.iter() {
            out.iter_mut().zip(keys).for_each(|(a, b)| *a |= b);
        }
        Bits::from_words(out)
    }
}

/// A block's effect on the facts after it, computed once: where each
/// lane class goes and the keys it loses, and the facts the block adds.
struct Effect {
    rows: Range<usize>,
    /// The facts its reads add, in context none.
    gen: State,
    /// The class the current EXEC's class ends in.
    cur: LaneId,
    /// The keys a class loses when not in `lost`.
    kill: Vec<u64>,
    /// The classes that lose more, ascending, with all the keys they
    /// lose; the current EXEC's along its way to `cur`.
    lost: Vec<(LaneId, Vec<u64>)>,
}

impl Effect {
    /// Where facts of class `lane` go, and the keys they lose.
    fn map(&self, lane: LaneId) -> (LaneId, &[u64]) {
        let to = if lane == CUR { self.cur } else { lane };
        match self.lost.binary_search_by_key(&lane, |e| e.0) {
            Ok(i) => (to, &self.lost[i].1),
            Err(_) => (to, &self.kill),
        }
    }

    /// `before` becomes the facts the block moves back from `after`, its
    /// reads' facts aside.
    fn apply(&self, after: &State, before: &mut State) {
        before.clear();
        for (slot, keys) in after.iter() {
            let (lane, lost) = self.map(slot.lane);
            before.add(Slot { lane, ..slot }, keys, Some(lost));
        }
    }
}

/// A block edge, moving facts back from its target into `block`.
struct Edge {
    block: usize,
    flow: Flow,
    /// The row it leaves.
    from: usize,
    /// The keys its edge claims end.
    lost: Option<Vec<u64>>,
}

impl Edge {
    /// Adds the facts `before` its target to those `after` its block,
    /// the new ones to `fresh` too; whether one was new.
    fn carry(&self, before: &State, after: &mut State, fresh: &mut State) -> bool {
        let mut new = false;
        for (slot, keys) in before.iter() {
            if let Some(ctx) = self.flow.back(self.from, slot.ctx) {
                let slot = Slot { ctx, ..slot };
                new |= after.add_fresh(fresh, slot, keys, self.lost.as_deref());
            }
        }
        new
    }
}

/// The rows with their lane classes interned.
struct Solver<'a> {
    rows: &'a Rows,
    width: usize,
    /// Per row, the class the current EXEC's class becomes going back.
    to: Vec<LaneId>,
    /// Per row, the class of its reads.
    read_lane: Vec<LaneId>,
    /// Per row with lane kills, the classes they end, ascending.
    cut: Vec<Rc<[LaneId]>>,
}

impl<'a> Solver<'a> {
    fn new(rows: &'a Rows) -> Solver<'a> {
        let n = rows.succ.len();
        let mut lanes: Vec<Lane> = vec![Lane::Cur, Lane::Any];
        let mut ids: HashMap<Lane, LaneId> = HashMap::from([(Lane::Cur, CUR), (Lane::Any, ANY)]);
        let mut intern = |l: Lane| {
            *ids.entry(l).or_insert_with(|| {
                lanes.push(l);
                (lanes.len() - 1) as LaneId
            })
        };
        let to: Vec<LaneId> = (0..n)
            .map(|k| {
                let (m, unknown) = match rows.writes_exec[k] {
                    true => (rows.exec_out[k], Lane::Any),
                    false => (rows.exec_in[k], Lane::Cur),
                };
                intern(m.lane().unwrap_or(unknown))
            })
            .collect();
        let read_lane = (0..n).map(|k| intern(rows.read_lanes[k])).collect();
        let mut cuts: HashMap<Exec, Rc<[LaneId]>> = HashMap::new();
        let cut = (0..n)
            .map(|k| {
                let e = rows.exec_in[k];
                let of = |(id, &c): (usize, &Lane)| {
                    let hit =
                        c == Lane::Cur || c != Lane::Any && e != Exec::Unknown && lane_killed(c, e);
                    hit.then_some(id as LaneId)
                };
                let set = || lanes.iter().enumerate().filter_map(of).collect();
                match rows.lane_kills[k].is_empty() {
                    true => Rc::from([]),
                    false => cuts.entry(e).or_insert_with(set).clone(),
                }
            })
            .collect();
        let keys = (rows.reads.iter().chain(&rows.every).chain(&rows.kills))
            .chain(&rows.lane_kills)
            .chain(rows.edge_kills.iter().flatten().map(|(_, keys)| keys))
            .chain([&rows.exit])
            .flatten();
        let width = keys.max().map_or(0, |&k| k / 64 + 1);
        Solver {
            rows,
            width,
            to,
            read_lane,
            cut,
        }
    }

    /// Lane-class facts of context none go back over row k.
    fn step(&self, k: usize, st: &mut State) {
        let r = self.rows;
        if self.to[k] != CUR {
            if let Ok(i) = st.index.binary_search_by_key(&Slot::top(CUR), |e| e.0) {
                let from = st.index[i].1;
                let to = st.offset(Slot::top(self.to[k]));
                for w in 0..self.width {
                    st.words[to + w] |= std::mem::take(&mut st.words[from + w]);
                }
            }
        }
        if !r.kills[k].is_empty() || !r.lane_kills[k].is_empty() {
            let (width, cut) = (self.width, &self.cut[k]);
            for &(slot, at) in &st.index {
                let keys = &mut st.words[at..at + width];
                r.kills[k].iter().for_each(|&key| unset(keys, key));
                if cut.binary_search(&slot.lane).is_ok() {
                    r.lane_kills[k].iter().for_each(|&key| unset(keys, key));
                }
            }
        }
        if !r.every[k].is_empty() {
            let keys = st.slot(Slot::top(ANY));
            r.every[k].iter().for_each(|&key| set(keys, key));
        }
        if !r.reads[k].is_empty() {
            let keys = st.slot(Slot::top(self.read_lane[k]));
            r.reads[k].iter().for_each(|&key| set(keys, key));
        }
    }

    /// The effect of the block of `rows`.
    fn effect(&self, rows: Range<usize>) -> Effect {
        let r = self.rows;
        let mut gen = State::new(self.width);
        let mut cur = CUR;
        let mut kill = vec![0; self.width];
        let mut lost: BTreeMap<LaneId, Vec<u64>> = BTreeMap::new();
        for k in rows.clone().rev() {
            self.step(k, &mut gen);
            if cur == CUR {
                cur = self.to[k];
            }
            r.kills[k].iter().for_each(|&key| set(&mut kill, key));
            // The current class ends a lane kill as the class it is by
            // now; every other class as itself.
            let cut = &self.cut[k];
            let hit = (cut.binary_search(&cur).is_ok().then_some(CUR))
                .into_iter()
                .chain(cut.iter().copied().filter(|&l| l != CUR));
            for lane in hit {
                let keys = lost.entry(lane).or_insert_with(|| vec![0; self.width]);
                r.lane_kills[k].iter().for_each(|&key| set(keys, key));
            }
        }
        let lost = (lost.into_iter())
            .map(|(lane, mut keys)| {
                keys.iter_mut().zip(&kill).for_each(|(a, b)| *a |= b);
                (lane, keys)
            })
            .collect();
        Effect {
            rows,
            gen,
            cur,
            kill,
            lost,
        }
    }

    /// The least fixpoint, on blocks of rows that fall through.
    fn solve(&self) -> Liveness {
        let r = self.rows;
        let n = r.succ.len();
        let mut preds = vec![0usize; n];
        r.succ.iter().flatten().for_each(|&t| preds[t] += 1);
        let firsts: Vec<usize> = (0..n)
            .filter(|&k| k == 0 || preds[k] != 1 || !r.falls_through(k - 1))
            .collect();
        let mut block_of = vec![usize::MAX; n];
        let effects: Vec<Effect> = (firsts.iter().enumerate())
            .map(|(j, &a)| {
                block_of[a] = j;
                self.effect(a..firsts.get(j + 1).copied().unwrap_or(n))
            })
            .collect();
        // Per block, the edges into it: a non-falling row ends its block,
        // so each of its successors starts one.
        let mut into: Vec<Vec<Edge>> = (0..effects.len()).map(|_| Vec::new()).collect();
        for (j, e) in effects.iter().enumerate() {
            let from = e.rows.end - 1;
            for (i, &t) in r.succ[from].iter().enumerate() {
                let claims: Vec<&Vec<usize>> = (r.edge_kills[from].iter())
                    .filter(|c| c.0 == t)
                    .map(|c| &c.1)
                    .collect();
                let lost = (!claims.is_empty()).then(|| {
                    let mut keys = vec![0; self.width];
                    claims
                        .iter()
                        .copied()
                        .flatten()
                        .for_each(|&key| set(&mut keys, key));
                    keys
                });
                into[block_of[t]].push(Edge {
                    block: j,
                    flow: r.flow(from, i),
                    from,
                    lost,
                });
            }
        }
        let mut after: Vec<State> = (effects.iter())
            .map(|e| {
                let mut st = State::new(self.width);
                if r.succ[e.rows.end - 1].is_empty() && !r.exit.is_empty() {
                    let keys = st.slot(Slot::top(ANY));
                    r.exit.iter().for_each(|&key| set(keys, key));
                }
                st
            })
            .collect();
        // Facts move semi-naively: each block's reads reach its
        // predecessors once, then a block moves back only the facts new
        // after it since its last visit. Every move is a union of per-fact
        // moves, so this reaches the least fixpoint.
        let mut fresh = after.clone();
        for (j, e) in effects.iter().enumerate() {
            for edge in &into[j] {
                edge.carry(&e.gen, &mut after[edge.block], &mut fresh[edge.block]);
            }
        }
        let mut queue: BinaryHeap<usize> = (0..effects.len()).collect();
        let mut queued = vec![true; effects.len()];
        let (mut moved, mut before) = (State::new(self.width), State::new(self.width));
        while let Some(j) = queue.pop() {
            queued[j] = false;
            std::mem::swap(&mut moved, &mut fresh[j]);
            fresh[j].clear();
            effects[j].apply(&moved, &mut before);
            for e in &into[j] {
                if e.carry(&before, &mut after[e.block], &mut fresh[e.block]) && !queued[e.block] {
                    queued[e.block] = true;
                    queue.push(e.block);
                }
            }
        }
        // Contexts merged: every row treats each context alike.
        let mut live = Liveness {
            live_in: vec![Bits::new(); n],
            exits: vec![None; n],
        };
        let mut lanes = State::new(self.width);
        for (e, after) in effects.iter().zip(&after) {
            lanes.clear();
            for (slot, keys) in after.iter() {
                lanes.add(Slot::top(slot.lane), keys, None);
            }
            live.exits[e.rows.end - 1] = Some(lanes.flat());
            for k in e.rows.clone().rev() {
                self.step(k, &mut lanes);
                live.live_in[k] = lanes.flat();
            }
        }
        live
    }
}

/// The liveness of `rows`: per row its live-in and live-out keys, the
/// least fixpoint over every context and lane class.
pub fn solve(rows: &Rows) -> Liveness {
    Solver::new(rows).solve()
}
