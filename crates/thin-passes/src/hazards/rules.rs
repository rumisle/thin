//! The generic wait-state engine.

use std::collections::HashMap;

use thin_ir::{FieldValue, Instr, Operand, Reg, Snapshot};
use thin_target::{Flags, Kind, Mem};

/// A test a side of a rule may require.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Test {
    /// The instruction is a VALU that writes an SGPR.
    SgprValuDef,
    /// M0 is among its writes.
    WritesM0,
    /// It writes only part of its destination.
    PartialDst,
    /// It has register defs.
    HasDefs,
    /// Whether it is a buffer op whose operand 3 is an SGPR register.
    SgprSoffset(bool),
}

/// The instructions a side of a rule selects.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Side {
    /// The kinds (empty: every kind).
    pub kinds: Vec<Kind>,
    /// Flags that must all be set.
    pub all: Flags,
    /// Flags of which one must be set, if any.
    pub any: Flags,
    /// Tests that must all hold.
    pub tests: Vec<Test>,
}

/// How the later instruction depends on the earlier.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dep {
    /// Always.
    None,
    /// A use of cur overlaps prev's defs.
    Raw,
    /// RAW, unless both have the same op and the use overlaps one of
    /// cur's defs.
    RawUnlessAccumulates,
    /// RAW, or a def of cur overlaps prev's defs.
    RawOrWaw,
    /// A def of cur overlaps prev's defs.
    Waw,
    /// prev's operand 3 (its SrcC) overlaps a def of cur.
    WritesSrcC,
    /// cur's SrcC, as a use, overlaps prev's defs.
    ReadsAsSrcC,
    /// A use of cur other than its SrcC overlaps prev's defs.
    ReadsNotAsSrcC,
    /// A use of prev wider than 2 dwords overlaps a def of cur.
    WritesWideData,
}

/// Pair- and instruction-dependent wait-state callbacks for a [`Rule`].
#[derive(Clone, Copy)]
pub struct DynamicStates {
    /// Upper bound on wait states this rule can return.
    pub max: u32,
    /// Wait states required between `(snapshot, prev, cur)`.
    pub pair: fn(&Snapshot, &Instr, &Instr) -> u32,
    /// Late-read wait states after `(snapshot, prev)` for register liveness.
    pub late: fn(&Snapshot, &Instr) -> u32,
}

impl std::fmt::Debug for DynamicStates {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DynamicStates")
            .field("max", &self.max)
            .finish()
    }
}

impl PartialEq for DynamicStates {
    fn eq(&self, other: &Self) -> bool {
        self.max == other.max
            && std::ptr::fn_addr_eq(self.pair, other.pair)
            && std::ptr::fn_addr_eq(self.late, other.late)
    }
}

impl Eq for DynamicStates {}

/// A rule's wait states.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum States {
    /// A constant.
    Const(u32),
    /// Computed from the matched instructions up to `max`.
    Dynamic(DynamicStates),
}

impl States {
    /// The most states it can ask.
    fn most(&self) -> u32 {
        match self {
            States::Const(s) => *s,
            States::Dynamic(d) => d.max,
        }
    }
}

/// A wait-state rule: `cur` must come at least its states after `prev`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rule {
    /// The earlier instruction.
    pub prev: Side,
    /// The later instruction.
    pub cur: Side,
    /// The dependence.
    pub dep: Dep,
    /// The states.
    pub states: States,
}

/// A register operand's physical extent: (file, first, count).
type Extent = (char, u32, u32);

/// Whether two extents share a dword of one file.
pub(crate) fn overlap(a: Extent, b: Extent) -> bool {
    a.0 == b.0 && a.1 < b.1 + b.2 && b.1 < a.1 + a.2
}

/// A register's physical extent.
pub(crate) fn extent(s: &Snapshot, r: &Reg) -> Extent {
    (r.file, s.phys(r), r.n)
}

/// The engine over a snapshot and a target's rules.
pub struct Engine<'a> {
    s: &'a Snapshot,
    rules: &'a [Rule],
}

impl<'a> Engine<'a> {
    /// The engine of `rules` on `s`.
    pub fn new(s: &'a Snapshot, rules: &'a [Rule]) -> Engine<'a> {
        Engine { s, rules }
    }

    /// Whether a side admits an instruction (tests checked with `tests`).
    fn admits(&self, side: &Side, i: &Instr, tests: bool) -> bool {
        let f = i.form();
        let kind = side.kinds.is_empty() || side.kinds.contains(&f.kind);
        let all = side.all.0 == 0 || f.flags.has(side.all);
        let any = side.any.0 == 0 || f.flags.any(side.any);
        kind && all && any && (!tests || side.tests.iter().all(|&t| self.test(t, i)))
    }

    /// One test on an instruction.
    fn test(&self, t: Test, i: &Instr) -> bool {
        match t {
            Test::SgprValuDef => i.sgpr_valu_def(),
            Test::WritesM0 => i.m0_write(),
            Test::PartialDst => i.partial_dst(),
            Test::HasDefs => i.defs().next().is_some(),
            Test::SgprSoffset(b) => {
                let sgpr = matches!(i.ops().get(3), Some(Operand::Reg(r)) if r.file == 's');
                (i.form().mem == Some(Mem::Buffer) && sgpr) == b
            }
        }
    }

    /// Named physical registers other than `exec` and `scc` as extents (`'p'`).
    fn phys_extents(list: &[usize]) -> impl Iterator<Item = Extent> + '_ {
        list.iter()
            .copied()
            .filter(|&p| !matches!(thin_target::PHYS[p].name, "exec" | "scc"))
            .map(|p| ('p', p as u32, 1))
    }

    /// The extents of an instruction's defs.
    fn defs(&self, i: &Instr) -> Vec<Extent> {
        i.defs()
            .map(|r| extent(self.s, r))
            .chain(Self::phys_extents(i.writes()))
            .collect()
    }

    /// Accumulator read (`SrcC`): operand 0 on a sparse matrix form (`rw`
    /// destination), operand 3 otherwise.
    fn src_c<'i>(&self, i: &'i Instr) -> Option<&'i Reg> {
        let idx = match self.s.target().hooks().matrix_form(&i.form().op) {
            Some(mf) if mf.sparse => 0,
            _ => 3,
        };
        i.ops().get(idx).and_then(Operand::reg)
    }

    /// Whether the dependence holds between prev and cur.
    fn depends(&self, dep: Dep, prev: &Instr, cur: &Instr) -> bool {
        let w = self.defs(prev);
        let hits = |e: Extent| w.iter().any(|&x| overlap(x, e));
        let uses: Vec<(Option<&Reg>, Extent)> = cur
            .uses()
            .map(|r| (Some(r), extent(self.s, r)))
            .chain(Self::phys_extents(cur.reads()).map(|e| (None, e)))
            .collect();
        let cur_defs = self.defs(cur);
        let src_c = self.src_c(cur);
        match dep {
            Dep::None => true,
            Dep::Raw => uses.iter().any(|u| hits(u.1)),
            Dep::RawUnlessAccumulates => uses.iter().any(|u| {
                hits(u.1)
                    && !(cur.form().op == prev.form().op
                        && cur_defs.iter().any(|&d| overlap(d, u.1)))
            }),
            Dep::RawOrWaw => uses.iter().any(|u| hits(u.1)) || cur_defs.iter().any(|&d| hits(d)),
            Dep::Waw => cur_defs.iter().any(|&d| hits(d)),
            Dep::WritesSrcC => self
                .src_c(prev)
                .is_some_and(|c| cur_defs.iter().any(|&d| overlap(d, extent(self.s, c)))),
            Dep::ReadsAsSrcC => uses
                .iter()
                .any(|u| u.0 == src_c && src_c.is_some() && hits(u.1)),
            Dep::ReadsNotAsSrcC => uses.iter().any(|u| u.0 != src_c && hits(u.1)),
            Dep::WritesWideData => prev
                .uses()
                .filter(|r| r.n > 2)
                .any(|r| cur_defs.iter().any(|&d| overlap(d, extent(self.s, r)))),
        }
    }

    /// A rule's states for a matching pair.
    fn pair_states(&self, states: &States, prev: &Instr, cur: &Instr) -> u32 {
        match states {
            States::Const(s) => *s,
            States::Dynamic(d) => (d.pair)(self.s, prev, cur),
        }
    }

    /// A rule's late-read states for `prev`.
    fn late_states(&self, states: &States, prev: &Instr) -> u32 {
        match states {
            States::Const(s) => *s,
            States::Dynamic(d) => (d.late)(self.s, prev),
        }
    }

    /// The most wait states any rule asks between prev and cur.
    pub fn wait_states(&self, prev: &Instr, cur: &Instr) -> u32 {
        (self.rules.iter())
            .filter(|r| self.admits(&r.prev, prev, true) && self.admits(&r.cur, cur, true))
            .filter(|r| self.depends(r.dep, prev, cur))
            .map(|r| self.pair_states(&r.states, prev, cur))
            .max()
            .unwrap_or(0)
    }

    /// The most states any rule admitting `kind` as cur can ask.
    pub fn reach(&self, kind: Kind) -> u32 {
        (self.rules.iter())
            .filter(|r| r.cur.kinds.is_empty() || r.cur.kinds.contains(&kind))
            .map(|r| r.states.most())
            .max()
            .unwrap_or(0)
    }

    /// The largest reach.
    pub fn window(&self) -> u32 {
        self.rules
            .iter()
            .map(|r| r.states.most())
            .max()
            .unwrap_or(0)
    }

    /// The most states of the rules of a dependence whose prev side's kinds
    /// and flags admit prev (tests ignored).
    pub fn late(&self, dep: Dep, prev: &Instr) -> u32 {
        (self.rules.iter())
            .filter(|r| r.dep == dep && self.admits(&r.prev, prev, false))
            .map(|r| self.late_states(&r.states, prev))
            .max()
            .unwrap_or(0)
    }

    /// The registers the hardware reads late after instruction i, and for
    /// how many states.
    pub fn late_reads(&self, i: &Instr) -> Option<(Vec<Reg>, u32)> {
        let wide: Vec<Reg> = i.uses().filter(|r| r.n > 2).copied().collect();
        let data = self.late(Dep::WritesWideData, i);
        if data > 0 && !wide.is_empty() {
            return Some((wide, data));
        }
        let c = self.late(Dep::WritesSrcC, i);
        let src = self.src_c(i).copied();
        (c > 0).then_some(())?;
        src.map(|r| (vec![r], c))
    }

    /// Per instruction, the wait states to place right before it.
    pub fn wait_state_needs(&self) -> Vec<u32> {
        Placer::new(self).run()
    }
}

/// An entry of the walk: an instruction, or a nop of N states placed for
/// the next instruction.
#[derive(Clone, Copy, Debug)]
enum Entry {
    Instr(usize),
    Nop(u32),
}

/// Walk the dependence rules over the instruction stream.
struct Placer<'e, 'a> {
    e: &'e Engine<'a>,
    entries: Vec<Entry>,
    labels: HashMap<String, usize>,
    forward: HashMap<String, Vec<usize>>,
}

impl<'e, 'a> Placer<'e, 'a> {
    fn new(e: &'e Engine<'a>) -> Placer<'e, 'a> {
        Placer {
            e,
            entries: Vec::new(),
            labels: HashMap::new(),
            forward: HashMap::new(),
        }
    }

    fn instr(&self, k: usize) -> &Instr {
        &self.e.s.instrs()[k]
    }

    fn kind(&self, entry: usize) -> Option<Kind> {
        match self.entries[entry] {
            Entry::Instr(k) => Some(self.instr(k).form().kind),
            Entry::Nop(_) => None,
        }
    }

    /// The wait states an entry gives.
    fn states(&self, entry: usize) -> u32 {
        match self.entries[entry] {
            Entry::Nop(n) => n,
            Entry::Instr(k) => {
                let i = self.instr(k);
                match i.form().kind {
                    Kind::Label | Kind::Comment => 0,
                    _ if i.form().op == "s_nop" => match i.ops().first() {
                        Some(Operand::Imm(thin_ir::Imm {
                            value: thin_ir::ImmValue::Int(n),
                            ..
                        })) => *n as u32 + 1,
                        _ => 1,
                    },
                    _ => 1,
                }
            }
        }
    }

    fn label_name(&self, entry: usize) -> Option<String> {
        let Entry::Instr(k) = self.entries[entry] else {
            return None;
        };
        let i = self.instr(k);
        (i.form().kind == Kind::Label).then(|| {
            i.field_named("name")
                .map(FieldValue::py_str)
                .unwrap_or_default()
        })
    }

    /// The predecessors of a position within `reach` states: (entry, states
    /// between), scanning back from entry k.
    fn preds(&self, mut k: isize, mut s: u32, reach: u32, out: &mut Vec<(usize, u32)>) {
        while k >= 0 && s < reach {
            let at = k as usize;
            if self.kind(at) == Some(Kind::Label) {
                let mut j = k;
                while j >= 0 && matches!(self.kind(j as usize), Some(Kind::Label | Kind::Comment)) {
                    j -= 1;
                }
                for e in (j + 1) as usize..=at {
                    let Some(name) = self.label_name(e) else {
                        continue;
                    };
                    for &br in self
                        .forward
                        .get(&name)
                        .into_iter()
                        .flatten()
                        .filter(|&&br| br < e)
                    {
                        self.preds(br as isize, s, reach, out);
                    }
                }
                let uncond = j >= 0
                    && matches!(self.entries[j as usize], Entry::Instr(q) if self.instr(q).form().flags.has(Flags::UNCOND));
                if uncond {
                    return;
                }
                k = j;
                continue;
            }
            out.push((at, s));
            s += self.states(at);
            k -= 1;
        }
    }

    /// The need of instruction i from its predecessors at the list's end.
    fn need(&self, i: &Instr) -> u32 {
        let reach = self.e.reach(i.form().kind);
        if reach == 0 {
            return 0;
        }
        let mut preds = Vec::new();
        self.preds(self.entries.len() as isize - 1, 0, reach, &mut preds);
        (preds.into_iter())
            .filter_map(|(q, s)| match self.entries[q] {
                Entry::Instr(qk) => self.e.wait_states(self.instr(qk), i).checked_sub(s),
                Entry::Nop(_) => None,
            })
            .max()
            .unwrap_or(0)
    }

    /// The need a back edge to entry `a` puts on the branch.
    fn back_edge_need(&self, a: usize) -> u32 {
        let window = self.e.window();
        let mut tail = Vec::new();
        self.preds(self.entries.len() as isize - 1, 1, window, &mut tail);
        let mut need = 0;
        let mut best = vec![window; self.entries.len()];
        let mut work = vec![(a + 1, 0u32)];
        while let Some((mut entry, mut s)) = work.pop() {
            while entry < self.entries.len() && s < window && best[entry] > s {
                best[entry] = s;
                if let Entry::Instr(ck) = self.entries[entry] {
                    let c = self.instr(ck);
                    for &(q, sq) in &tail {
                        if let Entry::Instr(qk) = self.entries[q] {
                            let w = self.e.wait_states(self.instr(qk), c);
                            if w > sq + s {
                                need = need.max(w - sq - s);
                            }
                        }
                    }
                    let flags = c.form().flags;
                    if flags.has(Flags::ENDPGM) {
                        break;
                    }
                    if flags.has(Flags::BRANCH) {
                        let t = c.block().and_then(|l| self.labels.get(l)).copied();
                        if let Some(t) = t.filter(|&t| t > entry) {
                            work.push((t + 1, s + self.states(entry)));
                        }
                        if flags.has(Flags::UNCOND) {
                            break;
                        }
                    }
                }
                s += self.states(entry);
                entry += 1;
            }
        }
        need
    }

    fn run(mut self) -> Vec<u32> {
        let n = self.e.s.instrs().len();
        let mut out = Vec::with_capacity(n);
        for k in 0..n {
            let i = self.instr(k).clone();
            match i.form().kind {
                Kind::Label => {
                    let name = i
                        .field_named("name")
                        .map(FieldValue::py_str)
                        .unwrap_or_default();
                    self.labels.insert(name, self.entries.len());
                    self.entries.push(Entry::Instr(k));
                    out.push(0);
                    continue;
                }
                Kind::Comment => {
                    self.entries.push(Entry::Instr(k));
                    out.push(0);
                    continue;
                }
                _ => {}
            }
            let mut need = self.need(&i);
            let branch = i.form().flags.has(Flags::BRANCH);
            let target = i.block().map(str::to_owned);
            if branch && self.e.window() > 0 {
                if let Some(&a) = target.as_ref().and_then(|t| self.labels.get(t)) {
                    need = need.max(self.back_edge_need(a));
                }
            }
            if need > 0 {
                self.entries.push(Entry::Nop(need));
            }
            out.push(need);
            if let (true, Some(t)) = (branch, target) {
                if !self.labels.contains_key(&t) {
                    self.forward.entry(t).or_default().push(self.entries.len());
                }
            }
            self.entries.push(Entry::Instr(k));
        }
        out
    }
}
