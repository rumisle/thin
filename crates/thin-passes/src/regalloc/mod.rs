//! Register assignment: where each value is occupied, a
//! first-fit placement, and the placement check.

mod occupancy;

use std::collections::HashMap;

use thin_ir::cfg::{Cfg, Taken};
use thin_ir::live::{self, Exec, Lane};
use thin_ir::sets::{Bits, Runs};
use thin_ir::{Diag, DiagClass, ImmValue, Instr, Operand, Part, Reg, Snapshot};
use thin_target::{Flags, Kind, Pseudo};

pub use occupancy::{solve, Facts, Occupancy};

use crate::hazards::{Engine, Rule};

/// The pass's explicit inputs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Options {
    /// An instruction in a lane region with unknown EXEC uses the region's
    /// token (`regions`).
    pub regions: bool,
    /// The values a no-room error lists (`peak_list`).
    pub peak_list: usize,
    /// Append the live-dword profile to a no-room error (`pressure`).
    pub pressure: bool,
}

impl Default for Options {
    fn default() -> Options {
        Options {
            regions: true,
            peak_list: 24,
            pressure: false,
        }
    }
}

/// Whether allocation holds a load's registers while it may be in flight
///. Only a test's negative control turns it off: then the emulator's
/// in-flight check must catch the placements it allows.
pub static HOLD_LOADS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

/// The register files in order.
const FILES: [char; 3] = ['v', 's', 'a'];

/// A file's placement report.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Report {
    /// The file.
    pub file: char,
    /// The largest load of a row.
    pub peak: u32,
    /// The instruction of the first row reaching it.
    pub at_index: usize,
    /// The highest placed register + 1.
    pub used: u32,
    /// The file's values.
    pub values: usize,
    /// The names live at the peak, sorted.
    pub live: Vec<String>,
    /// (width, name, index) live at the peak, widest first.
    pub live_w: Vec<(u32, String, u32)>,
}

/// A placement: per program value its base (its canonical value's), and
/// a report per file placed.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Placement {
    /// Per value number, its first register, none in a skipped file.
    pub bases: Vec<Option<u32>>,
    /// Per file placed, its report.
    pub reports: Vec<Report>,
}

/// What an instruction does to keys, whatever the class.
#[derive(Clone, Debug, Default)]
struct Static {
    reads: Vec<usize>,
    writes: Vec<usize>,
    sgpr_writes: Vec<usize>,
    wait_names: Vec<usize>,
    first_lane: bool,
    const_lane: Option<u32>,
    cross: bool,
    covered: bool,
    late: Option<(Vec<usize>, u32)>,
    load: bool,
    ends: Vec<usize>,
    claim_reads: bool,
    /// The keys of a destination read only for the half the write keeps
    /// (`Instr::kept_dst`): live through it, but not a data read.
    kept: Vec<usize>,
    /// An edge claim's (branch, target, keys): ended on that edge only.
    edge_ends: Option<(usize, usize, Vec<usize>)>,
}

/// Whether an undefined claim is on its next branch's taken edge.
fn on_taken(i: &Instr) -> bool {
    matches!(i.field_named("on"), Some(thin_ir::FieldValue::Text(t)) if &**t == "taken")
}

/// The taken edge of the branch after instruction k, own ops skipped:
/// (the branch, its target label's index).
fn taken_edge(s: &Snapshot, k: usize) -> Option<(usize, usize)> {
    let instrs = s.instrs();
    let b = (k + 1..instrs.len()).find(|&j| {
        let f = instrs[j].form();
        f.kind == Kind::Label || !f.flags.has(Flags::PSEUDO)
    })?;
    let name = instrs[b].ops().iter().find_map(|o| match o {
        Operand::Block(n) => Some(n.clone()),
        _ => None,
    })?;
    let t = instrs.iter().position(|i| {
        i.form().kind == Kind::Label
            && matches!(i.field_named("name"), Some(thin_ir::FieldValue::Text(x)) if *x == name)
    })?;
    Some((b, t))
}

/// The analysis of one snapshot.
struct Assign<'a> {
    s: &'a Snapshot,
    canonical: Vec<usize>,
    key: Vec<Option<usize>>,
    width: Vec<u32>,
    align: Vec<u32>,
    order: Vec<usize>,
    nkeys: usize,
    statics: Vec<Static>,
    /// Per canonical value, the highest base its slots' register limits
    /// allow (`Slot::limit`), or none; below 0 when none fits.
    ceiling: Vec<Option<i64>>,
}

/// The end of a value's `same_as` chain.
fn canonical(s: &Snapshot, v: usize) -> usize {
    let mut at = v;
    for _ in 0..s.values().len() {
        match s.values()[at].same_as {
            Some(next) => at = next as usize,
            None => break,
        }
    }
    at
}

impl<'a> Assign<'a> {
    fn new(s: &'a Snapshot, rules: &[Rule]) -> Assign<'a> {
        let values = s.values();
        let mut a = Assign {
            s,
            canonical: (0..values.len()).map(|v| canonical(s, v)).collect(),
            key: vec![None; values.len()],
            width: values.iter().map(|v| v.n).collect(),
            align: vec![1; values.len()],
            order: Vec::new(),
            nkeys: 0,
            statics: Vec::new(),
            ceiling: vec![None; values.len()],
        };
        for f in FILES {
            for (v, info) in values
                .iter()
                .enumerate()
                .filter(|(v, i)| i.file == f && a.canonical[*v] == *v)
            {
                a.key[v] = Some(a.nkeys);
                a.nkeys += info.n as usize;
                a.align[v] = alignment(f, info.align, info.n);
                a.order.push(v);
            }
        }
        for i in s.instrs().iter() {
            for (k, op) in i.ops().iter().enumerate() {
                let (Some(r), Some(limit)) = (op.reg(), i.form().slot(k).and_then(|s| s.limit))
                else {
                    continue;
                };
                let Some(&v) = usize::try_from(r.value)
                    .ok()
                    .and_then(|v| a.canonical.get(v))
                else {
                    continue;
                };
                let top = i64::from(limit) - i64::from(r.lo + r.n);
                a.ceiling[v] = Some(a.ceiling[v].map_or(top, |c| c.min(top)));
            }
        }
        let engine = Engine::new(s, rules);
        a.statics = (0..s.instrs().len())
            .map(|k| a.static_of(&engine, k))
            .collect();
        a
    }

    /// A register's keys.
    fn keys(&self, r: &Reg) -> Vec<usize> {
        let Some(&v) = usize::try_from(r.value)
            .ok()
            .and_then(|v| self.canonical.get(v))
        else {
            return Vec::new();
        };
        let Some(base) = self.key[v] else {
            return Vec::new();
        };
        (r.lo..r.lo + r.n).map(|d| base + d as usize).collect()
    }

    fn keys_of<'r>(&self, regs: impl Iterator<Item = &'r Reg>) -> Vec<usize> {
        let mut out: Vec<usize> = regs.flat_map(|r| self.keys(r)).collect();
        out.sort_unstable();
        out.dedup();
        out
    }

    /// for instruction k.
    fn static_of(&self, engine: &Engine<'_>, k: usize) -> Static {
        let i = &self.s.instrs()[k];
        let f = i.form();
        let regs = |i: &Instr| {
            i.ops()
                .iter()
                .filter_map(Operand::reg)
                .copied()
                .collect::<Vec<Reg>>()
        };
        if f.pseudo == Some(Pseudo::Discard) {
            // A discard keeps its write alive for the dead-write check, so
            // the register stays occupied up to it as well.
            return Static {
                reads: self.keys_of(regs(i).iter()),
                claim_reads: true,
                ..Static::default()
            };
        }
        if matches!(f.kind, Kind::Label | Kind::Comment) {
            return match f.pseudo {
                // An edge claim on taken ends its keys on the branch's taken
                // edge only; one on fall stands on its edge by
                // its place, between the last instruction and the label.
                Some(Pseudo::Undefined) if on_taken(i) => Static {
                    edge_ends: taken_edge(self.s, k)
                        .map(|(b, t)| (b, t, self.keys_of(regs(i).iter()))),
                    ..Static::default()
                },
                Some(Pseudo::Undefined) => Static {
                    ends: self.keys_of(regs(i).iter()),
                    ..Static::default()
                },
                Some(Pseudo::Assume) => {
                    let first: Vec<Reg> = i
                        .ops()
                        .first()
                        .and_then(Operand::reg)
                        .copied()
                        .into_iter()
                        .collect();
                    Static {
                        reads: self.keys_of(first.iter()),
                        claim_reads: true,
                        ..Static::default()
                    }
                }
                _ => Static::default(),
            };
        }
        let printed = if f.flags.has(Flags::PSEUDO) {
            Vec::new()
        } else {
            regs(i)
        };
        let defs: Vec<Reg> = i.defs().copied().collect();
        let mut reads: Vec<Reg> = i.uses().copied().collect();
        reads.extend(printed.iter().filter(|r| !defs.contains(r)));
        let doubled = |d: &&Reg| {
            f.flags.has(Flags::READS_DST) || printed.iter().filter(|r| r == d).count() > 1
        };
        reads.extend(defs.iter().filter(doubled));
        let canon = f.canon.as_str();
        let const_lane = match i.ops().get(2) {
            Some(Operand::Imm(m))
                if canon == "v.readlane.b32" && i.ops().len() == 3 && m.text.is_empty() =>
            {
                match m.value {
                    ImmValue::Int(v) if v >= 0 => Some(v as u32),
                    _ => None,
                }
            }
            _ => None,
        };
        let cross = f.kind == Kind::Matrix
            || f.flags.any(Flags::CROSS_LANE | Flags::PERMLANE)
            || f.flags.has(Flags::READLANE) && canon.starts_with("v.readlane")
            || i.field_named("dpp").is_some();
        let covered = !defs.is_empty()
            && defs
                .iter()
                .all(|d| usize::try_from(d.value).is_ok_and(|v| self.s.values()[v].covered));
        let writes = self.keys_of(defs.iter());
        let late = engine
            .late_reads(i)
            .map(|(regs, states)| (self.keys_of(regs.iter()), states));
        let wait_uses: Vec<Reg> = i
            .attrs()
            .wait_uses
            .iter()
            .flat_map(|(_, r)| r.iter().copied())
            .collect();
        Static {
            reads: self.keys_of(reads.iter()),
            sgpr_writes: self.keys_of(defs.iter().filter(|d| d.file == 's')),
            wait_names: self.keys_of(wait_uses.iter()),
            first_lane: canon == "v.readfirstlane.b32",
            const_lane,
            cross,
            covered,
            late,
            load: !writes.is_empty() && f.counter.is_some(),
            writes,
            ends: Vec::new(),
            claim_reads: false,
            kept: self.keys_of(i.kept_dst().into_iter()),
            edge_ends: None,
        }
    }

    /// Whether instruction i's write keeps the old contents
    /// (`Instr::keeps_old`).
    fn merges(&self, i: &Instr, partial: bool) -> bool {
        i.keeps_old(partial)
    }

    /// The facts of one class, and per instruction its data reads.
    fn facts(
        &self,
        g: &Cfg,
        class: Option<&Taken>,
        o: Options,
    ) -> (Facts, Vec<Vec<usize>>, Vec<Vec<usize>>) {
        let s = self.s;
        let n = s.instrs().len();
        let exec = match class {
            None => s.exec().to_vec(),
            Some(_) => thin_ir::exec::exec(s, g),
        };
        let full = s.target().full_mask();
        let reached = g.reached();
        let exec_in: Vec<Exec> = (0..n)
            .map(|k| match (exec[k], s.instrs()[k].attrs().in_region) {
                (Some(m), _) => Exec::Mask(m),
                (None, Some(r)) if o.regions => Exec::Region(r),
                _ => Exec::Unknown,
            })
            .collect();
        let mut live = live::Rows::new(
            g.succ.clone(),
            thin_ir::control::flows(s.control_flow(), &g.succ),
        );
        live.exec_in.clone_from(&exec_in);
        live.writes_exec = s.instrs().iter().map(|i| i.writes_exec()).collect();
        let mut f = Facts {
            live,
            writes: vec![Vec::new(); n],
            pinned: (self.order.iter())
                .filter(|&&v| s.values()[v].pin.is_some())
                .flat_map(|&v| {
                    self.key[v]
                        .map(|k| k..k + self.width[v] as usize)
                        .into_iter()
                        .flatten()
                })
                .collect(),
            held: vec![Bits::new(); n],
            window_ends: (s.instrs().iter())
                .map(|i| i.form().kind == Kind::Label || i.form().flags.has(Flags::BRANCH))
                .collect(),
        };
        let mut data_reads = vec![Vec::new(); n];
        let mut exec_kills = vec![Vec::new(); n];
        for k in (0..n).filter(|&k| reached[k]) {
            if let Some((b, t, keys)) = &self.statics[k].edge_ends {
                f.live.edge_kills[*b].push((*t, keys.clone()));
            }
        }
        for k in (0..n).filter(|&k| reached[k]) {
            let i = &s.instrs()[k];
            let st = &self.statics[k];
            if f.live.writes_exec[k] && k + 1 < n && i.form().canon != "s.branch" {
                f.live.exec_out[k] = exec_in[k + 1];
            }
            let e = exec_in[k];
            let mut lane = match e {
                Exec::Mask(m) => Lane::Mask(m),
                Exec::Region(r) => Lane::Region(r),
                Exec::Unknown => Lane::Cur,
            };
            if st.ends.len() + st.reads.len() + st.writes.len() + st.wait_names.len() == 0 {
                continue;
            }
            f.live.kills[k].clone_from(&st.ends);
            if st.claim_reads {
                (f.live.reads[k], f.live.read_lanes[k], data_reads[k]) =
                    (st.reads.clone(), lane, st.reads.clone());
                continue;
            }
            if !st.ends.is_empty() {
                continue;
            }
            let mut every = st.wait_names.clone();
            match (st.first_lane, e, st.const_lane) {
                (true, Exec::Mask(m), _) if m != 0 => lane = Lane::Mask(m & m.wrapping_neg()),
                (_, _, Some(l)) if l < 64 => lane = Lane::Mask(1 << l),
                _ if st.cross => every.extend(&st.reads),
                _ => {}
            }
            if !st.cross || st.first_lane || st.const_lane.is_some() {
                f.live.reads[k].clone_from(&st.reads);
            }
            f.live.read_lanes[k] = lane;
            f.live.every[k] = every;
            data_reads[k] = (st.reads.iter())
                .filter(|r| !st.kept.contains(r))
                .copied()
                .collect();
            f.writes[k].clone_from(&st.writes);
            let whole = !self.merges(i, false);
            let partial = e != Exec::Mask(full);
            let merges = self.merges(i, partial);
            let mut kills = if merges && !(whole && st.covered) {
                Vec::new()
            } else {
                st.writes.clone()
            };
            if merges && whole && !st.covered {
                exec_kills[k].clone_from(&st.writes);
            }
            // An SGPR write is whole whatever EXEC is (`Instr::keeps_old_in`).
            kills.extend(&st.sgpr_writes);
            f.live.kills[k] = kills;
            if whole
                && matches!(
                    i.form().kind,
                    Kind::Valu | Kind::Vmem | Kind::Ds | Kind::Matrix
                )
            {
                f.live.lane_kills[k].clone_from(&st.writes);
            }
        }
        self.hold(g, &reached, &mut f);
        (f, data_reads, exec_kills)
    }

    /// Late reads and loads in flight hold their keys.
    fn hold(&self, g: &Cfg, reached: &[bool], f: &mut Facts) {
        let s = self.s;
        let n = s.instrs().len();
        for k in (0..n).filter(|&k| reached[k]) {
            let Some((keys, states)) = &self.statics[k].late else {
                continue;
            };
            let (mut q, mut left) = (k + 1, *states);
            while q < n && left > 0 {
                let i = &s.instrs()[q];
                if !matches!(i.form().kind, Kind::Label | Kind::Comment) {
                    left = left.saturating_sub(nop_states(i));
                }
                q += 1;
            }
            (k + 1..q)
                .filter(|_| q > k + 1)
                .for_each(|h| f.held[h].add(keys.iter().copied()));
        }
        if !HOLD_LOADS.load(std::sync::atomic::Ordering::Relaxed) {
            return; // the negative control: the allocator without load holds
        }
        let loads: Vec<usize> = (0..n).filter(|&k| self.statics[k].load).collect();
        for (q, flying) in thin_ir::flight::in_flight(s, g, &loads)
            .into_iter()
            .enumerate()
        {
            for j in flying {
                f.held[q].add(self.statics[loads[j]].writes.iter().copied());
            }
        }
    }
}

/// The wait states an instruction gives: `s_nop N` gives N + 1.
fn nop_states(i: &Instr) -> u32 {
    match (i.form().op.as_str(), i.ops().first()) {
        ("s_nop", Some(Operand::Imm(m))) => match m.value {
            ImmValue::Int(v) => v as u32 + 1,
            ImmValue::Float(_) => 1,
        },
        _ => 1,
    }
}

/// A value's alignment.
fn alignment(file: char, declared: u32, width: u32) -> u32 {
    let pow = if width == 0 {
        1
    } else {
        1 << (31 - width.leading_zeros())
    };
    let a = declared.min(pow).max(1);
    match (file, width) {
        (_, 0 | 1) => a,
        ('s', 2) => a.max(2),
        ('s', _) => a.max(4),
        _ => a.max(2),
    }
}

fn refusal(msg: Vec<Part>) -> Diag {
    Diag {
        class: DiagClass::Assign,
        loc: 0,
        msg,
        notes: Vec::new(),
    }
}

/// Every row's occupied keys, the unwritten keys and the rows' count.
struct Rows {
    n: usize,
    occupied: Vec<Bits>,
    /// Per key, the rows it is occupied at, as runs: placement asks
    /// whether keys overlap, not row by row.
    keys: Vec<Runs>,
    unwritten: Bits,
    exec_kills: Vec<Vec<usize>>,
    facts: Vec<Facts>,
}

impl Assign<'_> {
    /// The rows over the classes.
    fn rows(&self, o: Options) -> Rows {
        let n = self.s.instrs().len();
        let mut rows = Rows {
            n,
            occupied: Vec::new(),
            keys: vec![Runs::new(); self.nkeys],
            unwritten: Bits::new(),
            exec_kills: Vec::new(),
            facts: Vec::new(),
        };
        for class in self.s.wave_classes() {
            let g = self.s.cfg(class.as_deref());
            let (facts, data_reads, exec_kills) = self.facts(&g, class.as_deref(), o);
            let occ = solve(&facts);
            for (k, reads) in data_reads.iter().enumerate() {
                let mut r = Bits::new();
                r.add(reads.iter().copied());
                r.and_not(&occ.written[k]);
                rows.unwritten.or(&r);
            }
            let reached = g.reached();
            rows.occupied
                .extend(occ.occupied.into_iter().enumerate().map(|(k, b)| {
                    if reached[k] {
                        b
                    } else {
                        Bits::new()
                    }
                }));
            rows.exec_kills.extend(exec_kills);
            rows.facts.push(facts);
        }
        // A key's runs open and close where its occupancy changes.
        let mut open = vec![0; self.nkeys];
        let none = Bits::new();
        for r in 0..=rows.occupied.len() {
            let before = r.checked_sub(1).map_or(&none, |q| &rows.occupied[q]);
            let now = rows.occupied.get(r).unwrap_or(&none);
            for (key, opens) in before.changes(now) {
                match (rows.keys.get_mut(key), opens) {
                    (None, _) => {}
                    (Some(_), true) => open[key] = r as u32,
                    (Some(runs), false) => runs.push_run(open[key], r as u32),
                }
            }
        }
        rows
    }

    /// The rows a value occupies.
    fn value_rows(&self, rows: &Rows, v: usize) -> Runs {
        let mut out = Runs::new();
        self.key_rows(rows, v).into_iter().for_each(|r| out.add(r));
        out
    }

    /// The rows each dword j of a value occupies.
    fn key_rows<'r>(&self, rows: &'r Rows, v: usize) -> Vec<&'r Runs> {
        let Some(base) = self.key[v] else {
            return Vec::new();
        };
        (base..base + self.width[v] as usize)
            .map(|k| &rows.keys[k])
            .collect()
    }

    /// `NAME[I]` of a value.
    fn label(&self, v: usize) -> String {
        let info = &self.s.values()[v];
        format!("{}[{}]", self.s.name(info.name), info.index)
    }

    /// The read-before-written refusal.
    fn unwritten(&self, rows: &Rows) -> Result<(), Diag> {
        for &v in &self.order {
            let info = &self.s.values()[v];
            let Some(base) = self.key[v] else { continue };
            if info.pin.is_some()
                || !(base..base + self.width[v] as usize).any(|k| rows.unwritten.has(k))
            {
                continue;
            }
            let uses = |i: &Instr| {
                i.ops()
                    .iter()
                    .filter_map(Operand::reg)
                    .any(|r| usize::try_from(r.value).is_ok_and(|x| self.canonical[x] == v))
            };
            let first = self
                .s
                .instrs()
                .iter()
                .position(|i| !i.form().flags.has(Flags::PSEUDO) && uses(i))
                .unwrap_or(0);
            let msg = vec![
                Part::Text(format!("{}:{} is read where no path from the entry wrote it (first use: `", info.file, self.label(v))),
                Part::Instr(first),
                Part::Text(format!(
                    "`, instruction {first}): write it before (a register the hardware fills at entry is pinned: \
                     RegFile.entry, RegFile.pin)"
                )),
            ];
            return Err(refusal(msg));
        }
        Ok(())
    }
}

/// Assigns registers: per file in order its values first fit, under
/// `limits` (letter, registers), with the target's wait-state rules for
/// the late reads.
pub fn assign(
    s: &Snapshot,
    rules: &[Rule],
    limits: &[(char, u32)],
    o: Options,
) -> Result<Placement, Diag> {
    let a = Assign::new(s, rules);
    let rows = a.rows(o);
    a.unwritten(&rows)?;
    let mut bases: Vec<Option<u32>> = vec![None; s.values().len()];
    let mut reports = Vec::new();
    for f in FILES {
        let limit = limits.iter().find(|l| l.0 == f).map_or(0, |l| l.1);
        let values: Vec<usize> = a
            .order
            .iter()
            .copied()
            .filter(|&v| s.values()[v].file == f)
            .collect();
        if limit == 0 || values.is_empty() {
            continue;
        }
        reports.push(a.place_file(&rows, f, limit, &values, &mut bases, o)?);
    }
    for v in 0..s.values().len() {
        bases[v] = bases[a.canonical[v]];
    }
    Ok(Placement { bases, reports })
}

impl Assign<'_> {
    /// First fit of one file's values: pinned values at their pins
    /// only (a pin is never moved; `same_as` values take their canonical
    /// value's base), then the values a slot's register limit bounds,
    /// lowest bound first, then the rest, each at the lowest aligned base
    /// that fits.
    fn place_file(
        &self,
        rows: &Rows,
        f: char,
        limit: u32,
        values: &[usize],
        bases: &mut [Option<u32>],
        o: Options,
    ) -> Result<Report, Diag> {
        let info = |v: usize| &self.s.values()[v];
        let vrows: HashMap<usize, Runs> = values
            .iter()
            .map(|&v| (v, self.value_rows(rows, v)))
            .collect();
        let first = |v: usize| vrows[&v].first().map_or(1 << 30, |r| r as usize + 1);
        let mut order: Vec<usize> = values
            .iter()
            .copied()
            .filter(|&v| info(v).pin.is_some())
            .collect();
        let mut rest: Vec<usize> = values
            .iter()
            .copied()
            .filter(|&v| info(v).pin.is_none())
            .collect();
        // Values an encoding's register field limits go before the rest:
        // first fit would otherwise fill the registers they can reach.
        rest.sort_by_key(|&v| {
            let c = self.ceiling[v];
            (c.is_none(), c, first(v), info(v).group, info(v).index)
        });
        order.extend(rest);
        // Per register, the rows a placed value occupies it at.
        let mut busy: Vec<Runs> = vec![Runs::new(); limit as usize];
        for v in order {
            let rws = &vrows[&v];
            let w = self.width[v];
            // Register base + j is taken only where key j is occupied.
            let krows = self.key_rows(rows, v);
            let ceiling = self.ceiling[v];
            let fits = |base: u32, busy: &[Runs]| {
                base + w <= limit
                    && ceiling.is_none_or(|c| i64::from(base) <= c)
                    && (krows.iter().enumerate()).all(|(j, rs)| !busy[base as usize + j].meets(rs))
            };
            let at = match info(v).pin {
                Some(p) => Some(p).filter(|&p| fits(p, &busy)),
                None => (0..limit)
                    .step_by(self.align[v] as usize)
                    .find(|&b| fits(b, &busy)),
            };
            let Some(base) = at else {
                return Err(self.no_room(rows, f, limit, v, rws, values, o));
            };
            for (j, rs) in krows.iter().enumerate() {
                busy[base as usize + j].add(rs);
            }
            bases[v] = Some(base);
        }
        Ok(self.report(rows, f, values, bases))
    }

    /// The load of each row: the values' dwords occupied there.
    fn loads(&self, rows: &Rows, values: &[usize]) -> Vec<u32> {
        let mut delta = vec![0i64; rows.occupied.len() + 1];
        for &v in values {
            for rs in self.key_rows(rows, v) {
                for &(a, b) in rs.runs() {
                    delta[a as usize] += 1;
                    delta[b as usize] -= 1;
                }
            }
        }
        let mut at = 0;
        (delta[..rows.occupied.len()].iter())
            .map(|d| {
                at += d;
                at as u32
            })
            .collect()
    }

    /// The values occupying a row, widest first.
    fn live_at(&self, rows: &Rows, values: &[usize], row: usize) -> Vec<(u32, String, u32)> {
        let mut live: Vec<(u32, String, u32)> = (values.iter())
            .filter(|&&v| self.value_rows(rows, v).has(row as u32))
            .map(|&v| {
                (
                    self.width[v],
                    self.s.name(self.s.values()[v].name).to_owned(),
                    self.s.values()[v].index,
                )
            })
            .collect();
        live.sort_by(|a, b| b.cmp(a));
        live
    }

    fn report(&self, rows: &Rows, f: char, values: &[usize], bases: &[Option<u32>]) -> Report {
        let load = self.loads(rows, values);
        let peak = load.iter().copied().max().unwrap_or(0);
        let row = load.iter().position(|&l| l == peak).unwrap_or(0);
        let live_w = self.live_at(rows, values, row);
        let mut live: Vec<String> = live_w.iter().map(|(_, n, i)| format!("{n}[{i}]")).collect();
        live.sort();
        let used = values
            .iter()
            .filter_map(|&v| bases[v].map(|b| b + self.width[v]))
            .max()
            .unwrap_or(0);
        Report {
            file: f,
            peak,
            at_index: if rows.n == 0 { 0 } else { row % rows.n },
            used,
            values: values.len(),
            live,
            live_w,
        }
    }

    /// The no-room refusal.
    #[allow(clippy::too_many_arguments)]
    fn no_room(
        &self,
        rows: &Rows,
        f: char,
        limit: u32,
        v: usize,
        rws: &Runs,
        values: &[usize],
        o: Options,
    ) -> Diag {
        let n = rows.n.max(1);
        let info = &self.s.values()[v];
        let why = match info.pin {
            Some(p) if p + info.n > limit => {
                format!("pinned at {p}, past the file's {limit} registers")
            }
            Some(p) if self.ceiling[v].is_some_and(|c| i64::from(p) > c) => {
                format!("pinned at {p}, past what its encoding's register field reaches")
            }
            Some(p) => format!("pinned at {p}, which a pin before it holds"),
            None => match self.ceiling[v] {
                Some(c) => format!(
                    "every aligned place its encoding's register field reaches (bases up to {c}) is taken there"
                ),
                None => format!("every aligned place in {limit} is taken there"),
            },
        };
        let load = self.loads(rows, values);
        let peak = load.iter().copied().max().unwrap_or(0);
        let row = load.iter().position(|&l| l == peak).unwrap_or(0);
        let live = self.live_at(rows, values, row);
        let listed: Vec<String> = live
            .iter()
            .take(o.peak_list)
            .map(|(w, name, i)| format!("{name}[{i}] {w}"))
            .collect();
        let more = if live.len() > o.peak_list { " ..." } else { "" };
        let (x, y) = (
            rws.first().map_or(0, |r| r as usize % n),
            rws.last().map_or(0, |r| r as usize % n),
        );
        let mut msg = vec![
            Part::Text(format!(
                "{f}: no room for {} ({} dwords, align {}; occupied from `",
                self.label(v),
                self.width[v],
                self.align[v]
            )),
            Part::Instr(x),
            Part::Text("` to `".into()),
            Part::Instr(y),
            Part::Text(format!("`): {why}. The peak: {peak} dwords live at `")),
            Part::Instr(row % n),
            Part::Text(format!(
                "` (instruction {}); there: {}{more}",
                row % n,
                listed.join(", ")
            )),
        ];
        let partial = self.partial_rule(rows, values, row);
        if !partial.is_empty() {
            let d: u32 = partial.iter().map(|&p| self.width[p]).sum();
            let names: Vec<String> = partial
                .iter()
                .map(|&p| format!("{} {}", self.label(p), self.width[p]))
                .collect();
            msg.push(Part::Text(format!(
                ". Of those, held there only by writes under a partial EXEC ({d} dwords): {} - where such a write \
                 covers every lane its reads use, `V.lanes_covered(name)` ends the old value at the write",
                names.join(", ")
            )));
        }
        if o.pressure {
            msg.push(Part::Text(format!(
                ". Pressure: {}",
                self.pressure(&load, limit, n)
            )));
        }
        refusal(msg)
    }

    /// The values occupying a row only because of writes under a partial
    /// EXEC.
    fn partial_rule(&self, rows: &Rows, values: &[usize], row: usize) -> Vec<usize> {
        let (c, k) = (row / rows.n.max(1), row % rows.n.max(1));
        let Some(facts) = rows.facts.get(c) else {
            return Vec::new();
        };
        let mut whole = facts.clone();
        for (q, kills) in whole.live.kills.iter_mut().enumerate() {
            kills.extend(&rows.exec_kills[c * rows.n + q]);
        }
        let occ = solve(&whole);
        (values.iter().copied())
            .filter(|&v| {
                let base = self.key[v].unwrap_or(0);
                let keys = base..base + self.width[v] as usize;
                keys.clone().any(|key| rows.occupied[row].has(key))
                    && !keys.into_iter().any(|key| occ.occupied[k].has(key))
            })
            .collect()
    }

    /// A text profile of live dwords per stretch between labels.
    fn pressure(&self, load: &[u32], limit: u32, n: usize) -> String {
        let mut out = Vec::new();
        let mut start = 0;
        for k in 0..=n {
            let label = k == n || self.s.instrs()[k].form().kind == Kind::Label;
            if !label || k == start {
                continue;
            }
            let peak = (start..k)
                .map(|q| {
                    (0..load.len() / n)
                        .map(|c| load[c * n + q])
                        .max()
                        .unwrap_or(0)
                })
                .max()
                .unwrap_or(0);
            let bar = (peak as usize * 40).div_ceil(limit.max(1) as usize);
            let shown: String = if bar > 40 {
                "#".repeat(40) + "!"
            } else {
                "#".repeat(bar)
            };
            out.push(format!("[{start}, {k}) {peak} {shown}"));
            start = k;
        }
        out.join("; ")
    }
}

/// A register's (value, key) holders in order, and the rows any holds.
type Holders = (Vec<(usize, usize)>, Runs);

/// The placement check: no two values on one register share an
/// occupied row.
pub fn check(s: &Snapshot, rules: &[Rule], bases: &[Option<u32>], o: Options) -> Result<(), Diag> {
    let a = Assign::new(s, rules);
    for (v, ceiling) in a.ceiling.iter().enumerate() {
        if let (Some(c), Some(base)) = (ceiling, bases.get(v).copied().flatten()) {
            if i64::from(base) > *c {
                return Err(refusal(vec![Part::Text(format!(
                    "{}:{} at {base}: past what an encoding's register field reaches (`Slot::limit`)",
                    s.values()[v].file,
                    a.label(v)
                ))]));
            }
        }
    }
    let rows = a.rows(o);
    for f in FILES {
        let values: Vec<usize> = a
            .order
            .iter()
            .copied()
            .filter(|&v| s.values()[v].file == f && bases[v].is_some())
            .collect();
        // Per value, the rows of each of its keys.
        let rows_of: HashMap<usize, Vec<&Runs>> =
            values.iter().map(|&v| (v, a.key_rows(&rows, v))).collect();
        // Per register: the (value, key) on it in order, and the rows any holds.
        let mut on: HashMap<u32, Holders> = HashMap::new();
        for &v in &values {
            let base = bases[v].expect("a placed value");
            for (j, d) in (base..base + a.width[v]).enumerate() {
                let mine = rows_of[&v][j];
                let (holders, held) = on.entry(d).or_default();
                if held.meets(mine) {
                    for &(other, oj) in holders.iter() {
                        if let Some(row) = mine.first_common(rows_of[&other][oj]) {
                            let row = row as usize;
                            let k = row % rows.n.max(1);
                            return Err(refusal(vec![
                                Part::Text(format!(
                                    "{f}{d}: {} and {} both occupied at `",
                                    a.label(other),
                                    a.label(v)
                                )),
                                Part::Instr(k),
                                Part::Text(format!("` (instruction {k})")),
                            ]));
                        }
                    }
                }
                holders.push((v, j));
                held.add(mine);
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::alignment;

    #[test]
    fn alignment_follows_width_and_file() {
        assert_eq!(alignment('v', 4, 1), 1);
        assert_eq!(alignment('v', 1, 3), 2);
        assert_eq!(alignment('v', 8, 4), 4);
        assert_eq!(alignment('s', 1, 2), 2);
        assert_eq!(alignment('s', 1, 3), 4);
        assert_eq!(alignment('a', 1, 2), 2);
    }
}
