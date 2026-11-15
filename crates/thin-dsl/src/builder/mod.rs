//! The Builder: the kernel being built and its records.

mod claims;
mod emit;
mod make;
mod path;

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use thin_ir::{Attr, FieldId, FieldValue, Instr, Operand, Reg};
use thin_target::{Kind, Pseudo, Target};

use crate::error::{schedule, value, Result};
use crate::flow::Flow;
use crate::lanes::Exec;
use crate::lds::Layout;
use crate::values::{reg_repr, File, Placed, ValueRec};
use crate::waits::Requests;

pub use claims::ClaimReg;
pub use make::Make;
pub use path::{State, StateOp};

/// A row of the stream: an instruction, its identity and its location.
#[derive(Clone, Debug)]
pub struct Row {
    /// Its identity, stable while it is in the stream.
    pub id: u64,
    /// The instruction.
    pub ins: Arc<Instr>,
    /// Its source location, 0 for none.
    pub loc: u32,
    /// Whether its location is the hazard pass's: `loc` named "hazard"
    /// which the host's location table makes.
    pub hazard: bool,
}

/// Where a row of a new stream comes from (`Builder::set_stream`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Source {
    /// The row at this index of the current stream, kept with its id.
    Row(usize),
    /// A new row with this location, named "hazard" when `hazard`.
    New {
        /// Its location.
        loc: u32,
        /// Whether the hazard pass placed it.
        hazard: bool,
    },
}

/// The options the Builder reads; the others it only carries.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Options {
    /// A loop body is emitted twice and compared.
    pub audit: bool,
    /// `b.barrier` first waits for every counter at 0.
    pub safe_barrier: bool,
}

/// The kernel's needs: co-residency and same-grid arguments.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Needs {
    /// Why the kernel needs its workgroups co-resident.
    pub co_resident: Option<String>,
    /// Per argument, why it must be the same grid's.
    pub same_grid: BTreeMap<String, String>,
}

/// A record for the pipeline.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Deferred {
    /// A save-less region's exit that may be dead.
    Exit {
        /// The exit's row.
        exit: u64,
        /// The stream's last row when it was recorded.
        upto: u64,
        /// The chained region's id.
        region: Option<u32>,
    },
    /// A peeled loop.
    Peel {
        /// Its stem.
        stem: String,
        /// Its top label.
        label: String,
        /// The copies it may peel.
        max: u32,
        /// Each copy's first row.
        starts: Vec<Option<u64>>,
        /// The stream's last row.
        last: Option<u64>,
    },
}

/// A loop the Builder made.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LoopRec {
    /// Its stem.
    pub stem: String,
    /// The body's first row.
    pub first: u64,
    /// The body's last row.
    pub last: u64,
    /// Whether its waits may be the larger of entry and steady.
    pub conservative: bool,
}

/// A lane region the program records.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RegionRec {
    /// Its id.
    pub id: u32,
    /// Its entry's first row.
    pub entry: u64,
    /// The saved EXEC.
    pub save: Option<Reg>,
    /// Whether the render proves every lane on at its entry.
    pub checked: bool,
}

static BUILDERS: AtomicU64 = AtomicU64::new(1);

/// The kernel being built.
pub struct Builder {
    /// Its identity (a tree's values name it).
    pub id: u64,
    /// Its target.
    pub t: &'static Target,
    /// The register files v, s, a.
    pub files: [File; 3],
    /// Every file's values, in id order.
    pub values: Vec<ValueRec>,
    /// The stream.
    pub rows: Vec<Row>,
    next_row: u64,
    synced: usize,
    /// The options.
    pub options: Options,
    /// The ablation classes the build drops.
    pub ablate: BTreeSet<String>,
    open_abl: Vec<Vec<String>>,
    alias: Vec<Vec<(String, Option<String>)>>,
    /// Whether the passes have run.
    pub sealed: bool,
    /// The hazard instructions the pipeline placed.
    pub hazards: Option<usize>,
    /// Whether the waits pass has run.
    pub waits_placed: bool,
    labels: u32,
    names: HashMap<String, u32>,
    reachable: bool,
    placed_labels: BTreeSet<String>,
    forward: Vec<(String, State, bool)>,
    /// The current path state.
    pub state: State,
    states: Vec<path::Slot>,
    pending_forks: Vec<usize>,
    next_records: u64,
    /// What happened to the path states since the host last read it.
    pub state_ops: Vec<StateOp>,
    held: BTreeMap<(char, u32), Arc<str>>,
    /// The control constructs' state.
    pub flow: Flow,
    /// The lane regions' state.
    pub exec: Exec,
    /// The wait requests' state.
    pub requests: Requests,
    /// The LDS layout.
    pub layout: Option<Layout>,
    /// The entry.
    pub entered: Option<crate::entry::Entry>,
    /// The needs.
    pub needs: Needs,
    /// The SGPR holding the wave's index.
    pub wave_reg: Option<Reg>,
    /// The workgroup's waves.
    pub nwaves: u32,
    /// Whether the passes treat each wave class apart.
    pub by_wave: bool,
    /// The records for the pipeline.
    pub deferred: Vec<Deferred>,
    /// Undefined claims whose `on` waits for what follows them:
    /// (row, edge = "taken", the claim's registers' text).
    pub claims_open: Vec<(u64, bool, String)>,
    /// The loops.
    pub loops: Vec<LoopRec>,
    /// The lane regions.
    pub regions: Vec<RegionRec>,
    lit: Option<usize>,
    /// The location of the operation running (the host sets it per call).
    pub loc: u32,
    /// `short(loc)` of that location.
    pub where_text: String,
    /// The comment of the statement being lowered (its first instruction
    /// takes it).
    pub stmt_comment: Option<String>,
    /// The pipeline's records of the kernel (codegen owns their type).
    pub pipeline: Option<Box<dyn std::any::Any + Send + Sync>>,
}

/// The file sizes of a Builder; None takes the target's.
#[derive(Clone, Copy, Debug, Default)]
pub struct Sizes {
    /// VGPRs.
    pub vgprs: Option<u32>,
    /// SGPRs.
    pub sgprs: Option<u32>,
}

impl Builder {
    /// An empty kernel for the target; `placement`, when given,
    /// places each file's values as they are declared (the audit's second
    /// run).
    pub fn new(
        t: &'static Target,
        sizes: Sizes,
        options: Options,
        placement: Option<[Vec<Placed>; 3]>,
    ) -> Builder {
        let size =
            |f: char, given: Option<u32>| given.unwrap_or_else(|| t.file(f).map_or(0, |x| x.size));
        let [pv, ps, pa] =
            placement.map_or([None, None, None], |[v, s, a]| [Some(v), Some(s), Some(a)]);
        let ncounters = t.definition().counters.len();
        Builder {
            id: BUILDERS.fetch_add(1, Ordering::Relaxed),
            t,
            files: [
                File::new('v', size('v', sizes.vgprs), pv),
                File::new('s', size('s', sizes.sgprs), ps),
                File::new('a', 0, pa),
            ],
            values: Vec::new(),
            rows: Vec::new(),
            next_row: 1,
            synced: 0,
            options,
            ablate: BTreeSet::new(),
            open_abl: Vec::new(),
            alias: Vec::new(),
            sealed: false,
            hazards: None,
            waits_placed: false,
            labels: 0,
            names: HashMap::new(),
            reachable: true,
            placed_labels: BTreeSet::new(),
            forward: Vec::new(),
            state: State {
                counts: vec![0; ncounters],
                mem: BTreeMap::new(),
                records: 0,
            },
            states: Vec::new(),
            pending_forks: Vec::new(),
            next_records: 1,
            state_ops: Vec::new(),
            held: BTreeMap::new(),
            flow: Flow::default(),
            exec: Exec::default(),
            requests: Requests::default(),
            layout: None,
            entered: None,
            needs: Needs::default(),
            wave_reg: None,
            nwaves: 0,
            by_wave: false,
            deferred: Vec::new(),
            claims_open: Vec::new(),
            loops: Vec::new(),
            regions: Vec::new(),
            lit: None,
            loc: 0,
            where_text: String::new(),
            stmt_comment: None,
            pipeline: None,
        }
    }

    /// The file of a letter.
    pub fn file(&self, letter: char) -> &File {
        &self.files[file_index(letter)]
    }

    /// The file of a letter, to change it.
    pub fn file_mut(&mut self, letter: char) -> &mut File {
        &mut self.files[file_index(letter)]
    }

    /// The rows changed since the last call: the first changed index, and
    /// the rows from it.
    pub fn sync(&mut self) -> (usize, &[Row]) {
        let from = self.synced.min(self.rows.len());
        self.synced = self.rows.len();
        (from, &self.rows[from..])
    }

    /// Replaces the stream (the pipeline's transforms): each item keeps an
    /// old row (its id and location) or makes a new one; the host resyncs
    /// from the first changed row.
    pub fn set_stream(&mut self, items: Vec<(Arc<Instr>, Source)>) {
        let old = std::mem::take(&mut self.rows);
        let mut first = old.len().min(items.len());
        for (k, (ins, src)) in items.into_iter().enumerate() {
            let row = match src {
                Source::Row(j) if Arc::ptr_eq(&old[j].ins, &ins) => old[j].clone(),
                Source::Row(j) => Row {
                    ins,
                    ..old[j].clone()
                },
                Source::New { loc, hazard } => {
                    self.next_row += 1;
                    Row {
                        id: self.next_row - 1,
                        ins,
                        loc,
                        hazard,
                    }
                }
            };
            if !old
                .get(k)
                .is_some_and(|o| o.id == row.id && Arc::ptr_eq(&o.ins, &row.ins))
            {
                first = first.min(k);
            }
            self.rows.push(row);
        }
        self.touched(first);
    }

    /// Marks rows from k changed.
    pub(crate) fn touched(&mut self, k: usize) {
        self.synced = self.synced.min(k);
    }

    /// The row of an id.
    pub fn row_index(&self, id: u64) -> Option<usize> {
        self.rows.iter().rposition(|r| r.id == id)
    }

    /// A register's repr.
    pub fn repr(&self, r: &Reg) -> String {
        reg_repr(&self.values, r)
    }

    /// The register's number in its file.
    pub fn number(&self, r: &Reg) -> Result<u32> {
        self.file(r.file).number(&self.values, r, &self.repr(r))
    }

    /// The (file, number) of each dword of a register.
    pub fn dwords(&self, r: &Reg) -> Vec<(char, u32)> {
        match self.number(r) {
            Ok(n) => (n..n + r.n).map(|d| (r.file, d)).collect(),
            Err(_) => Vec::new(),
        }
    }

    /// An instruction's text: the printer's line with placeholder registers
    /// `{0}`, `{1}`….
    pub fn text(&self, i: &Instr) -> String {
        let regs: Vec<String> = (i.ops().iter())
            .filter_map(Operand::reg)
            .map(|r| crate::values::reg_text(&self.values, r))
            .collect();
        thin_ir::print::line(i, &regs).unwrap_or_default()
    }

    /// Refuses a sealed kernel.
    pub fn check_open(&self) -> Result<()> {
        match self.sealed {
            true => {
                schedule("the kernel's passes have run (render, derive): nothing more is emitted")
            }
            false => Ok(()),
        }
    }

    /// Refuses a loop left by `break`, then closes the open chain.
    pub fn settle(&mut self) -> Result<()> {
        if let Some(stem) = self.flow.broken.take() {
            return crate::error::flow(format!(
                "loop {stem}: left by break (a loop's body runs once a pass; its back edge was never emitted)"
            ));
        }
        crate::flow::close_chain(self)?;
        self.place_claims(false)
    }

    /// Settles, then marks the kernel sealed.
    pub fn seal(&mut self) -> Result<()> {
        self.settle()?;
        self.place_claims(true)?;
        self.sealed = true;
        Ok(())
    }

    /// The position: the stream's length minus the hazard instructions.
    pub fn pos(&mut self) -> Result<usize> {
        self.settle()?;
        Ok(self.rows.len() - self.hazards.unwrap_or(0))
    }

    /// The stream's indices between two positions.
    pub fn emitted(&mut self, start: i64, stop: Option<i64>) -> Result<std::ops::Range<usize>> {
        self.settle()?;
        let n = self.rows.len();
        let resolve = |p: i64| {
            if p < 0 {
                (p + n as i64).max(0) as usize
            } else {
                (p as usize).min(n)
            }
        };
        if self.hazards.is_none() {
            return Ok(resolve(start)..stop.map_or(n, resolve).max(resolve(start)));
        }
        let real: Vec<usize> = (0..n)
            .filter(|&k| !self.rows[k].ins.attrs().hazard)
            .collect();
        let map = |p: i64| match p {
            p if p <= 0 => 0,
            p if p as usize <= real.len() => real[p as usize - 1] + 1,
            _ => n,
        };
        let (a, b) = (map(start), stop.map_or(n, map));
        Ok(a..b.max(a))
    }

    /// The ops of a counter issued on the current path.
    pub fn issued(&self, counter: &str) -> Result<u32> {
        match self.t.counter_named(counter) {
            Some(k) => Ok(self.state.counts[k]),
            None => {
                let names: Vec<&str> = self
                    .t
                    .definition()
                    .counters
                    .iter()
                    .map(|c| c.name)
                    .collect();
                value(format!(
                    "issued(\"{counter}\"): one of {}",
                    names.join(", ")
                ))
            }
        }
    }

    /// Whether two registers' numbers overlap in one file.
    pub fn overlap(&self, a: &Reg, b: &Reg) -> bool {
        let (Ok(x), Ok(y)) = (self.number(a), self.number(b)) else {
            return false;
        };
        a.file == b.file && x < y + b.n && y < x + a.n
    }

    /// Pushes a row without the emission's steps.
    pub(crate) fn push(&mut self, ins: Instr, loc: u32) -> u64 {
        let id = self.next_row;
        self.next_row += 1;
        self.rows.push(Row {
            id,
            ins: Arc::new(ins),
            loc,
            hazard: false,
        });
        id
    }

    /// Drops the rows from k on.
    pub(crate) fn truncate(&mut self, k: usize) {
        self.rows.truncate(k);
        self.touched(k);
    }

    /// An own op of the Builder's target.
    pub fn own(&self, p: Pseudo) -> &'static thin_target::Form {
        self.t.pseudo(p)
    }

    /// `b.comment(text)`: an `ir.comment` line `"; " + text`.
    pub fn comment(&mut self, text: &str) -> Result<()> {
        self.check_open()?;
        self.settle()?;
        let i = Instr::new(
            self.own(Pseudo::Comment),
            Vec::new(),
            vec![field("text", format!("; {text}"))?],
            None,
        );
        self.push(i, 0);
        Ok(())
    }

    /// `.L<stem>_<k>`, k counting every label from 1.
    pub fn new_label(&mut self, stem: &str) -> String {
        self.labels += 1;
        format!(".L{stem}_{}", self.labels)
    }

    /// Whether a label is placed.
    pub fn placed(&self, name: &str) -> bool {
        self.placed_labels.contains(name)
    }

    /// Whether the next instruction is reached by falling through.
    pub fn reachable(&self) -> bool {
        self.reachable
    }

    /// `b.put_label(name, from_branches)`.
    pub fn put_label(&mut self, name: &str, from_branches: bool) -> Result<()> {
        self.settle()?;
        self.place_label(name, from_branches)
    }

    /// Apply emission checks and staging without settling deferred records.
    pub(crate) fn place_label(&mut self, name: &str, from_branches: bool) -> Result<()> {
        if from_branches && !self.reachable {
            if let Some(k) = self.forward.iter().rposition(|f| f.0 == name) {
                let st = self.forward[k].1.clone();
                self.state = self.copy_state(&st);
                self.reachable = true;
            }
        }
        let i = Instr::new(
            self.own(Pseudo::Label),
            Vec::new(),
            vec![field("name", name.to_owned())?],
            None,
        );
        self.push(i, 0);
        self.placed_labels.insert(name.to_owned());
        let (taken, rest): (Vec<_>, Vec<_>) = std::mem::take(&mut self.forward)
            .into_iter()
            .partition(|f| f.0 == name);
        self.forward = rest;
        let joined: Vec<State> = taken.into_iter().filter(|f| !f.2).map(|f| f.1).collect();
        if self.reachable && !joined.is_empty() {
            self.join(&joined);
        }
        self.reachable = true;
        Ok(())
    }

    /// The Builder's count of values made with a name base, plus one.
    pub(crate) fn next_name(&mut self, base: &str) -> String {
        let k = self.names.entry(base.to_owned()).or_insert(0);
        *k += 1;
        format!("{base}#{k}")
    }

    /// The counts per name base.
    pub fn name_counts(&self) -> HashMap<String, u32> {
        self.names.clone()
    }

    /// Rewinds the counts per name base.
    pub fn rewind_names(&mut self, counts: HashMap<String, u32>) {
        self.names = counts;
    }

    /// `b.nop(n)`: emits `s.nop n`.
    pub fn nop(&mut self, n: i128) -> Result<()> {
        let i = self.make_plain("s.nop", vec![crate::maker::Arg::Int(n)], "")?;
        self.emit(i, &[]).map(|_| ())
    }

    /// An instruction of the maker with no fields, carrying a comment.
    pub(crate) fn make_plain(
        &self,
        op: &str,
        args: Vec<crate::maker::Arg>,
        comment: &str,
    ) -> Result<Instr> {
        let notes = crate::maker::Notes {
            comment: comment.to_owned(),
            ..Default::default()
        };
        self.make(op, args, Vec::new(), notes)
    }

    /// The maker on the Builder's target.
    pub fn make(
        &self,
        op: &str,
        args: Vec<crate::maker::Arg>,
        fields: Vec<(String, crate::maker::FArg)>,
        notes: crate::maker::Notes,
    ) -> Result<Instr> {
        let c = crate::maker::Ctx {
            t: self.t,
            values: &self.values,
        };
        match crate::maker::op(&c, op, args, fields, notes)? {
            crate::maker::Made::Instr(i) => Ok(*i),
            crate::maker::Made::Lds(_) => {
                value(format!("{op}: an LDS address where an instruction is made"))
            }
        }
    }

    /// `b.barrier(comment, abl, fence)`.
    pub fn barrier(&mut self, comment: &str, abl: &[String], fence: bool) -> Result<()> {
        let comment = if fence && comment.is_empty() {
            crate::LDS_FENCE.to_owned()
        } else if fence {
            format!("{} {comment}", crate::LDS_FENCE)
        } else {
            comment.to_owned()
        };
        if self.options.safe_barrier {
            let counts: Vec<(usize, u32)> = (0..self.t.definition().counters.len())
                .map(|c| (c, 0))
                .collect();
            for i in crate::waits::wait_instrs(self.t, &counts, "(safe barrier)")? {
                self.emit(i, &[])?;
            }
        }
        let seq = self.t.definition().barrier.clone();
        let last = seq.len().saturating_sub(1);
        for (k, (m, ops)) in seq.iter().enumerate() {
            let args = ops
                .iter()
                .map(|&v| crate::maker::Arg::Int(i128::from(v)))
                .collect();
            let notes = crate::maker::Notes {
                comment: if k == last {
                    comment.clone()
                } else {
                    String::new()
                },
                abl: abl.to_vec(),
                ..Default::default()
            };
            let mut i = self.make(m, args, Vec::new(), notes)?;
            if fence && k == last {
                i = i.with_attr(Attr::Fence)?;
            }
            self.emit(i, &[])?;
        }
        Ok(())
    }

    /// `b.need(what, why, arg)`.
    pub fn need(&mut self, what: &str, why: &str, arg: Option<&str>) -> Result<()> {
        match (what, arg) {
            ("co_resident", _) => {
                self.needs.co_resident.get_or_insert_with(|| why.to_owned());
            }
            ("same_grid", Some(a)) if !a.is_empty() => {
                self.needs
                    .same_grid
                    .entry(a.to_owned())
                    .or_insert_with(|| why.to_owned());
            }
            _ => {
                let arg = arg.map_or("None".to_owned(), crate::maker::py_repr);
                return value(format!(
                    "need({}, arg={arg}): co_resident, or same_grid with its argument",
                    crate::maker::py_repr(what)
                ));
            }
        }
        Ok(())
    }

    /// Opens ablation classes (`b.ablatable`).
    pub fn open_ablatable(&mut self, classes: Vec<String>) {
        self.open_abl.push(classes);
    }

    /// Closes the innermost `ablatable` block.
    pub fn close_ablatable(&mut self) {
        self.open_abl.pop();
    }

    /// Whether an instruction of these classes would be dropped.
    pub fn ablated(&self, abl: &[String]) -> bool {
        !self.ablate.is_empty()
            && abl
                .iter()
                .chain(self.open_abl.iter().flatten())
                .any(|c| self.ablate.contains(c))
    }

    /// Opens alias pairs.
    pub fn open_alias(&mut self, reads: Vec<String>, over: Vec<String>, why: &str) -> Result<()> {
        if why.is_empty() {
            return value("alias_ok needs the argument (why)");
        }
        let pairs = (reads.iter())
            .flat_map(|r| match over.is_empty() {
                true => vec![(r.clone(), None)],
                false => over.iter().map(|o| (r.clone(), Some(o.clone()))).collect(),
            })
            .collect();
        self.alias.push(pairs);
        Ok(())
    }

    /// Closes the innermost `alias_ok` block.
    pub fn close_alias(&mut self) {
        self.alias.pop();
    }

    /// `b.hold(label, *regs)`.
    pub fn hold(&mut self, label: &str, regs: &[Reg]) -> Result<()> {
        for r in regs {
            if self.file(r.file).placed {
                continue;
            }
            for d in self.dwords(r) {
                if let Some(other) = self.held.get(&d) {
                    return schedule(format!(
                        "hold {label}: {} is already held by {other}",
                        self.repr(r)
                    ));
                }
            }
            for d in self.dwords(r) {
                self.held.insert(d, label.into());
            }
        }
        Ok(())
    }

    /// `b.release(label)`.
    pub fn release(&mut self, label: &str) {
        self.held.retain(|_, l| **l != *label);
    }

    /// The `_lit` value of the `s` file, made at its first use.
    pub(crate) fn lit(&mut self) -> Result<Reg> {
        let v = match self.lit {
            Some(v) => v,
            None => {
                let loc = self.loc;
                let g = self.files[1].alloc(&mut self.values, "_lit", 1, 1, 1, loc)?;
                let v = self.files[1].groups[g].values[0];
                self.lit = Some(v);
                v
            }
        };
        Ok(Reg::new(v as i32, 's', 0, 1))
    }

    /// The last row that is not a label or a comment.
    pub(crate) fn last_real(&self) -> Option<u64> {
        (self.rows.iter().rev())
            .find(|r| !matches!(r.ins.form().kind, Kind::Label | Kind::Comment))
            .map(|r| r.id)
    }

    /// `b.wave_index(reg, nwaves, by_wave)`.
    pub fn wave_index(&mut self, reg: Reg, nwaves: u32, by_wave: bool) {
        self.wave_reg = Some(reg);
        self.nwaves = nwaves;
        self.by_wave = by_wave;
    }
}

/// A field of an own op by name.
pub(crate) fn field(name: &str, text: String) -> Result<(FieldId, FieldValue)> {
    Ok((FieldId::named(name)?, FieldValue::Text(text.into())))
}

fn file_index(letter: char) -> usize {
    match letter {
        'v' => 0,
        's' => 1,
        _ => 2,
    }
}
