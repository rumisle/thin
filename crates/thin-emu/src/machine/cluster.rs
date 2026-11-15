//! Clusters of ranks: several ranks' launches over one
//! memory, in time slices, with parking, time-outs, deadlock and the hog.

use std::collections::{BTreeMap, VecDeque};

use super::lds::Lds;
use super::memory::Memory;
use super::run::{Machine, Yield};
use super::scope::Scope;
use super::wave::Wave;
use super::WgState;
use crate::tags::{Checker, Plain, Tables};
use crate::{Cause, EmuError};

/// The consecutive bounded-poll time-outs that are a deadlock.
pub const TIMEOUTS: u32 = 64;

/// A rank's dispatch order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Order {
    /// The grid's order.
    Grid,
    /// Reversed.
    Reverse,
    /// A seeded shuffle.
    Shuffle(u64),
}

/// An LDS carry record.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Carry {
    /// No carry: zeros.
    None,
    /// The last workgroup's final LDS.
    Prev,
    /// This workgroup's own from the previous launch, else the last's.
    Launch,
    /// A given (data, tags) every time.
    Fill(Vec<u8>, Vec<i64>),
    /// Zeros, and every workgroup's final LDS kept by (x, y).
    Record,
}

/// The cluster's options.
#[derive(Clone, Debug)]
pub struct Options {
    /// Workgroups a rank holds at once, per rank (0: all).
    pub resident: Vec<usize>,
    /// Instructions a wave runs per turn.
    pub quantum: u64,
    /// Dispatch order per rank.
    pub order: Vec<Order>,
    /// Turns each rank sits out before its first dispatch.
    pub skew: Vec<u64>,
    /// A factor on each rank's quantum.
    pub speed: Vec<f64>,
    /// A factor on a workgroup's quantum by x.
    pub wg_speed: BTreeMap<u32, f64>,
    /// A seed: each turn's quantum × a draw in [0.5, 1.5].
    pub jitter: Option<u64>,
    /// A seed: shuffle a workgroup's waves each turn, quantum × [0.25, 1.75].
    pub wave_jitter: Option<u64>,
    /// Rounds without progress before a deadlock.
    pub stall_rounds: u64,
    /// A rank starts launch L+1 only once every rank ended launch L.
    pub lockstep: bool,
    /// Per L2, the workgroups another kernel leaves room for.
    pub hog: Option<Vec<usize>>,
    /// Rounds without progress after which the hog ends.
    pub hog_stall: u64,
    /// LDS carry per rank.
    pub carry: Vec<Carry>,
    /// A Deadlock once all ranks together ran more instructions (0: none).
    pub max_ins: u64,
}

impl Default for Options {
    fn default() -> Options {
        Options {
            resident: Vec::new(),
            quantum: 64,
            order: Vec::new(),
            skew: Vec::new(),
            speed: Vec::new(),
            wg_speed: BTreeMap::new(),
            jitter: None,
            wave_jitter: None,
            stall_rounds: 200,
            lockstep: false,
            hog: None,
            hog_stall: 2000,
            carry: Vec::new(),
            max_ins: 0,
        }
    }
}

/// splitmix64: the seeded draws.
#[derive(Clone, Debug)]
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn shuffle<T>(&mut self, v: &mut [T]) {
        for i in (1..v.len()).rev() {
            let j = (self.next() % (i as u64 + 1)) as usize;
            v.swap(i, j);
        }
    }
}

/// Bytes written into a buffer as a launch starts (the runtime's staged
/// writes).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Staged {
    /// The buffer.
    pub buffer: usize,
    /// The offset.
    pub offset: u64,
    /// The bytes.
    pub data: Vec<u8>,
}

/// A live workgroup.
struct Live {
    x: u32,
    y: u32,
    waves: Vec<Wave>,
    lds: Lds,
    wg: WgState,
    l2: usize,
}

/// One rank's run state.
struct Rank {
    launch: usize,
    queue: VecDeque<(u32, u32)>,
    live: Vec<Live>,
    skew: u64,
    started: bool,
    done: bool,
    by: BTreeMap<(u32, u32), (Vec<u8>, Vec<i64>)>,
    prev_by: BTreeMap<(u32, u32), (Vec<u8>, Vec<i64>)>,
}

/// Several ranks' launches over one memory.
pub struct Cluster {
    /// One machine per rank.
    pub machines: Vec<Machine>,
    /// Per rank, its launches' kernarg addresses in order.
    pub launches: Vec<Vec<u64>>,
    /// The grid (x, y).
    pub grid: (u32, u32),
    /// Waves per workgroup.
    pub waves: usize,
    /// The options.
    pub options: Options,
    /// The checker, shared by the ranks.
    pub checker: Box<dyn Checker>,
    /// Per (rank, launch), the staged writes made as it starts.
    pub staged: BTreeMap<(usize, usize), Vec<Staged>>,
    /// Rounds run.
    pub rounds: u64,
    serial: usize,
}

fn deadlock(message: String) -> EmuError {
    EmuError {
        cause: Some(Cause::Deadlock),
        message,
    }
}

impl Cluster {
    /// A cluster of machines (one per rank) over launches' kernargs.
    pub fn new(
        machines: Vec<Machine>,
        launches: Vec<Vec<u64>>,
        grid: (u32, u32),
        waves: usize,
        options: Options,
    ) -> Cluster {
        Cluster {
            machines,
            launches,
            grid,
            waves,
            options,
            checker: Box::new(Plain),
            staged: BTreeMap::new(),
            rounds: 0,
            serial: 0,
        }
    }

    fn order(&self, r: usize) -> VecDeque<(u32, u32)> {
        let mut ids: Vec<(u32, u32)> = (0..self.grid.1)
            .flat_map(|y| (0..self.grid.0).map(move |x| (x, y)))
            .collect();
        match self.options.order.get(r).or(self.options.order.first()) {
            Some(Order::Reverse) => ids.reverse(),
            Some(Order::Shuffle(seed)) => Rng(*seed ^ r as u64).shuffle(&mut ids),
            _ => {}
        }
        ids.into()
    }

    fn resident(&self, r: usize) -> usize {
        let n = self
            .options
            .resident
            .get(r)
            .or(self.options.resident.first())
            .copied()
            .unwrap_or(0);
        if n == 0 {
            usize::MAX
        } else {
            n
        }
    }

    fn carry(&self, r: usize) -> &Carry {
        self.options
            .carry
            .get(r)
            .or(self.options.carry.first())
            .unwrap_or(&Carry::None)
    }

    /// Starts a rank's launch: its queue, its staged writes.
    fn start_launch(&mut self, mem: &mut Memory, ranks: &mut [Rank], r: usize) {
        let rank = &mut ranks[r];
        rank.queue = self.order(r);
        rank.started = true;
        if let Some(list) = self.staged.get(&(r, rank.launch)) {
            for s in list {
                if let Some(b) = mem.bufs.get_mut(s.buffer) {
                    let o = s.offset as usize;
                    let end = (o + s.data.len()).min(b.data.len());
                    b.data[o..end].copy_from_slice(&s.data[..end - o]);
                    b.version += 1;
                }
            }
        }
    }

    /// Dispatches workgroups up to residency (and the hog's room).
    fn dispatch(&mut self, mem: &mut Memory, ranks: &mut [Rank], r: usize, hog_on: bool) -> bool {
        let mut any = false;
        let xcds = self.machines[r].facts.xcds.max(1);
        while ranks[r].live.len() < self.resident(r) {
            let pick = if hog_on {
                let hog = self.options.hog.clone().unwrap_or_default();
                let room = |x: u32, live: &[Live]| {
                    let l2 = x as usize % xcds;
                    let on = live.iter().filter(|w| w.l2 == l2).count();
                    on < hog.get(l2).copied().unwrap_or(0)
                };
                let idx = ranks[r]
                    .queue
                    .iter()
                    .position(|(x, _)| room(*x, &ranks[r].live));
                idx.and_then(|i| ranks[r].queue.remove(i))
            } else {
                ranks[r].queue.pop_front()
            };
            let Some((x, y)) = pick else { break };
            let kernarg = self.launches[r].get(ranks[r].launch).copied().unwrap_or(0);
            let serial = self.serial;
            self.serial += 1;
            let l2 = x as usize % xcds;
            let m = &self.machines[r];
            let waves = m.make_waves(kernarg, (x, y), self.waves, serial, l2);
            let mut lds = m.new_lds(self.waves);
            let size = lds.size();
            let fill = |lds: &mut Lds, d: &(Vec<u8>, Vec<i64>)| {
                let n = size.min(d.0.len());
                lds.data[..n].copy_from_slice(&d.0[..n]);
                lds.tags[..n].copy_from_slice(&d.1[..n]);
            };
            match self.carry(r).clone() {
                Carry::Prev => {
                    if let Some(d) = &m.lds_last {
                        fill(&mut lds, d);
                    }
                }
                Carry::Launch => {
                    if let Some(d) = ranks[r].prev_by.get(&(x, y)).or(m.lds_last.as_ref()) {
                        fill(&mut lds, d);
                    }
                }
                Carry::Fill(d, t) => fill(&mut lds, &(d, t)),
                Carry::Record | Carry::None => {}
            }
            let marks = vec![-1; self.waves];
            ranks[r].live.push(Live {
                x,
                y,
                waves,
                lds,
                wg: WgState {
                    serial,
                    epoch: 0,
                    barrier_marks: marks,
                },
                l2,
            });
            any = true;
        }
        let _ = mem;
        any
    }

    fn end_launch(&mut self, mem: &mut Memory, r: usize) {
        if let Some(s) = &mut mem.scope {
            s.launch_ended(r);
        }
        mem.flight.end_launch(r);
    }

    /// `rank R wg X wave K at P `TEXT``: X the workgroup's x id, P the
    /// wave's pc, TEXT the text of instruction min(P, n − 1).
    fn place(&self, r: usize, live: &Live, w: &Wave) -> String {
        let texts = &self.machines[r].texts;
        let at = w.pc.min(texts.len().saturating_sub(1));
        let text = texts.get(at).cloned().unwrap_or_default();
        format!("rank {r} wg {} wave {} at {} `{text}`", live.x, w.wid, w.pc)
    }

    /// Runs every rank; the machines keep their reports.
    #[allow(clippy::too_many_lines)]
    pub fn run(&mut self, mem: &mut Memory, tables: &mut Tables) -> Result<(), EmuError> {
        let n = self.machines.len();
        let per_rank = self.machines.first().map_or(1, |m| m.facts.xcds.max(1));
        mem.scope = Some(Scope::new(per_rank));
        let mut ranks: Vec<Rank> = (0..n)
            .map(|r| Rank {
                launch: 0,
                queue: VecDeque::new(),
                live: Vec::new(),
                skew: self.options.skew.get(r).copied().unwrap_or(0),
                started: false,
                done: self.launches[r].is_empty(),
                by: BTreeMap::new(),
                prev_by: BTreeMap::new(),
            })
            .collect();
        let mut jitter = self.options.jitter.map(Rng);
        let mut wave_jitter = self.options.wave_jitter.map(Rng);
        let mut hog_on = self.options.hog.is_some();
        let mut stall = 0u64;
        let mut timeouts = 0u32;
        let ended: Vec<usize> = vec![0; n];
        let mut ended = ended;
        let snapshot = |ms: &[Machine]| -> Vec<u64> {
            ms.iter()
                .flat_map(|m| {
                    let s = &m.reports.stats;
                    [s.store_bytes, s.lds_write_bytes, s.matrix]
                })
                .collect()
        };
        loop {
            self.rounds += 1;
            let mut moved = false;
            let mut waiting = false;
            // 1. Lockstep-held and skewed ranks dispatch.
            for r in 0..n {
                if ranks[r].done {
                    continue;
                }
                if !ranks[r].started {
                    if ranks[r].skew > 0 {
                        ranks[r].skew -= 1;
                        waiting = true;
                        continue;
                    }
                    if self.options.lockstep && ended.iter().any(|e| *e < ranks[r].launch) {
                        waiting = true;
                        continue;
                    }
                    self.start_launch(mem, &mut ranks, r);
                    moved = true;
                }
                if self.dispatch(mem, &mut ranks, r, hog_on) {
                    moved = true;
                }
            }
            let before = snapshot(&self.machines);
            let mut ran = false;
            // 2. Each live workgroup's turn.
            for r in 0..n {
                let speed = self.options.speed.get(r).copied().unwrap_or(1.0);
                let mut i = 0;
                while i < ranks[r].live.len() {
                    let x = ranks[r].live[i].x;
                    let mut q = self.options.quantum as f64;
                    if let Some(j) = &mut jitter {
                        q *= 0.5 + j.unit();
                    }
                    q *= speed * self.options.wg_speed.get(&x).copied().unwrap_or(1.0);
                    let q = (q as u64).max(1);
                    let live = &mut ranks[r].live[i];
                    let mut order: Vec<usize> = (0..live.waves.len()).collect();
                    let mut quanta = vec![q; order.len()];
                    if let Some(j) = &mut wave_jitter {
                        j.shuffle(&mut order);
                        for qq in &mut quanta {
                            *qq = ((q as f64 * (0.25 + 1.5 * j.unit())) as u64).max(1);
                        }
                    }
                    let checker = std::mem::replace(&mut self.checker, Box::new(Plain));
                    let m = &mut self.machines[r];
                    let saved = std::mem::replace(&mut m.checker, checker);
                    let mut result = Ok(());
                    for (k, wi) in order.iter().enumerate() {
                        let w = &mut live.waves[*wi];
                        if w.done || w.at_barrier || !Machine::awake(w, mem) {
                            continue;
                        }
                        let before_ins = w.nins;
                        match m.run_wave(w, mem, tables, &mut live.lds, &mut live.wg, quanta[k]) {
                            Ok(Yield::Done) => moved = true,
                            Ok(_) => {}
                            Err(e) => {
                                result = Err(e);
                                break;
                            }
                        }
                        if w.nins > before_ins {
                            ran = true;
                        }
                    }
                    if result.is_ok() {
                        let all = live.waves.iter().all(|w| w.done || w.at_barrier);
                        let some = live.waves.iter().any(|w| w.at_barrier && !w.done);
                        if all && some {
                            result = m.release(&mut live.waves, &mut live.lds, &mut live.wg);
                            moved = true;
                        }
                    }
                    let finished = result.is_ok() && live.waves.iter().all(|w| w.done);
                    if finished {
                        result = m.check_uniform(&mut live.waves);
                    }
                    let checker = std::mem::replace(&mut m.checker, saved);
                    self.checker = checker;
                    result?;
                    if finished {
                        let l = ranks[r].live.remove(i);
                        let carry = self.carry(r).clone();
                        let last = (l.lds.data, l.lds.tags);
                        if !matches!(carry, Carry::Fill(..)) {
                            if matches!(carry, Carry::Launch | Carry::Record) {
                                ranks[r].by.insert((l.x, l.y), last.clone());
                            }
                            self.machines[r].lds_last = Some(last);
                        }
                        moved = true;
                        timeouts = 0;
                        self.dispatch(mem, &mut ranks, r, hog_on);
                        continue;
                    }
                    i += 1;
                }
                // A launch ends when all its workgroups ran.
                let rank = &mut ranks[r];
                if rank.started && rank.queue.is_empty() && rank.live.is_empty() && !rank.done {
                    self.end_launch(mem, r);
                    ended[r] = rank.launch + 1;
                    rank.prev_by = std::mem::take(&mut rank.by);
                    rank.launch += 1;
                    rank.started = false;
                    if rank.launch >= self.launches[r].len() {
                        rank.done = true;
                    }
                    moved = true;
                }
            }
            if ranks.iter().all(|r| r.done) {
                break;
            }
            // 3. The instruction cap.
            if self.options.max_ins > 0 {
                let total: u64 = self.machines.iter().map(|m| m.reports.stats.ins).sum();
                if total > self.options.max_ins {
                    return Err(deadlock(format!(
                        "ran past {} instructions (the run's cap: a wait that does not end in \
                         time - a bounded poll giving up round after round)",
                        self.options.max_ins
                    )));
                }
            }
            // 4. Progress.
            if snapshot(&self.machines) != before {
                moved = true;
            }
            if moved {
                stall = 0;
                continue;
            }
            let blocked = ranks.iter().all(|rk| {
                rk.live.iter().all(|l| {
                    l.waves
                        .iter()
                        .all(|w| w.done || w.at_barrier || w.parked.is_some())
                })
            });
            if !ran && !waiting && blocked {
                if hog_on {
                    hog_on = false;
                    continue;
                }
                let mut polls = false;
                for rk in &mut ranks {
                    for l in &mut rk.live {
                        for w in &mut l.waves {
                            polls |= Machine::time_out(w);
                        }
                    }
                }
                let why = self.why(&ranks, mem);
                if polls {
                    timeouts += 1;
                    if timeouts > TIMEOUTS {
                        return Err(deadlock(format!(
                            "bounded polls timed out {TIMEOUTS} times in a row, no workgroup ending \
                             (their steals or helps change nothing): {why}"
                        )));
                    }
                    continue;
                }
                return Err(deadlock(why));
            }
            stall += 1;
            if hog_on && stall >= self.options.stall_rounds.min(self.options.hog_stall) {
                hog_on = false;
            }
            if stall >= self.options.stall_rounds {
                let places: Vec<String> = ranks
                    .iter()
                    .enumerate()
                    .flat_map(|(r, rk)| {
                        rk.live.iter().flat_map(move |l| {
                            l.waves.iter().filter(|w| !w.done).map(move |w| (r, l, w))
                        })
                    })
                    .map(|(r, l, w)| self.place(r, l, w))
                    .collect();
                return Err(deadlock(format!(
                    "no progress for {} rounds: {}",
                    self.options.stall_rounds,
                    places.join("; ")
                )));
            }
        }
        Ok(())
    }

    fn why(&self, ranks: &[Rank], mem: &Memory) -> String {
        let mut parts = Vec::new();
        for (r, rk) in ranks.iter().enumerate() {
            for l in &rk.live {
                for w in l.waves.iter().filter(|w| !w.done) {
                    let place = self.place(r, l, w);
                    let tail = match &w.parked {
                        Some(on) => {
                            let mut bufs: Vec<usize> = on.iter().map(|(b, _)| *b).collect();
                            bufs.sort_unstable();
                            let names: Vec<String> = bufs
                                .iter()
                                .filter_map(|b| mem.bufs.get(*b).map(|x| x.name.clone()))
                                .collect();
                            format!(" waits on {}", names.join(", "))
                        }
                        None if w.at_barrier => " at a barrier".to_owned(),
                        None => String::new(),
                    };
                    parts.push(format!("{place}{tail}"));
                }
            }
        }
        format!("deadlock: no wave can run: {}", parts.join("; "))
    }
}
