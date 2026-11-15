//! The alias check: a register read through one name that a
//! write through another name reaches.

use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::sync::Arc;

use thin_ir::cfg::{cfg_of, Cfg};
use thin_ir::reach::{Def, Defs};
use thin_ir::sets::Bits;
use thin_ir::solve::solve_forward;
use thin_ir::{Diag, FieldValue, Instr, Operand, Part, Reg, Snapshot};
use thin_target::{phys_named, Flags, Kind, Pseudo};

use crate::checks::refusal;

/// A physical dword: (file, number).
type Dword = (char, u32);

/// An item of the checked stream: an instruction of the snapshot, or the
/// hardware's write of the entry registers.
#[derive(Clone)]
struct Item {
    instr: Arc<Instr>,
    orig: Option<usize>,
}

/// A register access through a name.
#[derive(Clone, Copy, Debug)]
struct Access {
    reg: Reg,
    alias: u32,
    through: u32,
}

/// A site: an item's write of a tracked dword.
#[derive(Clone, Copy, Debug)]
struct Site {
    item: usize,
    dword: Dword,
    alias: u32,
    through: u32,
    partial: bool,
}

/// A finding before it is spelled.
struct Finding {
    item: usize,
    read: Access,
    dword: Dword,
    site: usize,
    more: usize,
    rule2: bool,
}

/// The check over one class's stream.
struct Check<'a> {
    s: &'a Snapshot,
    items: Vec<Item>,
    hardware: bool,
    g: Cfg,
    writes: Vec<Vec<Access>>,
    reads: Vec<Vec<Access>>,
    sites: Vec<Site>,
    by_item: Vec<Vec<usize>>,
    every: Defs,
    whole: Defs,
}

/// A register's physical dwords.
fn dwords(s: &Snapshot, r: &Reg) -> Vec<Dword> {
    let p = s.phys(r);
    (p..p + r.n).map(|d| (r.file, d)).collect()
}

/// The stream a class checks.
fn stream(s: &Snapshot, class: Option<&[(usize, bool)]>) -> (Vec<Item>, bool) {
    let t = s.target();
    let mut items = Vec::new();
    let hardware = !s.entry().is_empty();
    if hardware {
        let ops = s.entry().iter().map(|r| Operand::Reg(*r)).collect();
        items.push(Item {
            instr: Arc::new(Instr::bare(t.pseudo(Pseudo::Comment), ops)),
            orig: None,
        });
    }
    let g = class.map(|c| s.cfg(Some(c)));
    let reached = g.as_ref().map(|g| g.reached());
    for (k, i) in s.instrs().iter().enumerate() {
        let f = i.form();
        if f.kind == Kind::Comment && f.pseudo != Some(Pseudo::Undefined) {
            continue;
        }
        if reached.as_ref().is_some_and(|r| !r[k]) && f.kind != Kind::Label {
            continue;
        }
        let taken = class.and_then(|c| c.iter().find(|e| e.0 == k)).map(|e| e.1);
        let instr = match taken {
            Some(false) => continue,
            Some(true) => {
                let branch = t
                    .spell("s.branch")
                    .and_then(|m| t.form(&m, 1, false))
                    .expect("s.branch ");
                let target = i
                    .ops()
                    .iter()
                    .find(|o| matches!(o, Operand::Block(_)))
                    .cloned();
                Arc::new(Instr::new(
                    branch,
                    target.into_iter().collect(),
                    Vec::new(),
                    None,
                ))
            }
            None => i.clone(),
        };
        items.push(Item {
            instr,
            orig: Some(k),
        });
    }
    (on_edges(s, items), hardware)
}

/// An edge claim on taken stands on its branch's taken edge: the branch
/// goes to a block of its own at the end, holding the claim, which then
/// goes on to the target. (A claim on fall is on its edge by its place,
/// between the last instruction and the label.)
fn on_edges(s: &Snapshot, items: Vec<Item>) -> Vec<Item> {
    let t = s.target();
    let taken = |i: &Instr| {
        i.form().pseudo == Some(Pseudo::Undefined)
            && matches!(i.field_named("on"), Some(FieldValue::Text(x)) if &**x == "taken")
    };
    if !items.iter().any(|i| taken(&i.instr)) {
        return items;
    }
    let mut out: Vec<Item> = Vec::new();
    let mut tail: Vec<Item> = Vec::new();
    let mut pending: Vec<Item> = Vec::new();
    for item in items {
        if taken(&item.instr) {
            pending.push(item);
            continue;
        }
        let is_branch = item.instr.form().flags.has(Flags::BRANCH);
        if pending.is_empty() || !is_branch {
            out.push(item);
            continue;
        }
        let target = item.instr.ops().iter().find_map(|o| match o {
            Operand::Block(n) => Some(n.clone()),
            _ => None,
        });
        let Some(target) = target else {
            out.append(&mut pending);
            out.push(item);
            continue;
        };
        let edge: Arc<str> = format!("{target}.edge{}", out.len()).into();
        let ops = (item.instr.ops().iter())
            .map(|o| match o {
                Operand::Block(n) if *n == target => Operand::Block(edge.clone()),
                o => o.clone(),
            })
            .collect();
        out.push(Item {
            instr: Arc::new(item.instr.with_ops(ops)),
            orig: item.orig,
        });
        let name = thin_ir::FieldId::named("name").expect("a label's name field");
        tail.push(Item {
            instr: Arc::new(Instr::new(
                t.pseudo(Pseudo::Label),
                Vec::new(),
                vec![(name, FieldValue::Text(edge.to_string().into()))],
                None,
            )),
            orig: None,
        });
        tail.append(&mut pending);
        let branch = t
            .spell("s.branch")
            .and_then(|m| t.form(&m, 1, false))
            .expect("s.branch ");
        tail.push(Item {
            instr: Arc::new(Instr::new(
                branch,
                vec![Operand::Block(target)],
                Vec::new(),
                None,
            )),
            orig: None,
        });
    }
    out.append(&mut pending);
    if !tail.is_empty() {
        // The stream's end must not fall into the edge blocks.
        let end = t
            .spell("s.endpgm")
            .and_then(|m| t.form(&m, 0, false))
            .expect("s.endpgm ");
        out.push(Item {
            instr: Arc::new(Instr::new(end, Vec::new(), Vec::new(), None)),
            orig: None,
        });
    }
    out.append(&mut tail);
    out
}

impl<'a> Check<'a> {
    fn new(s: &'a Snapshot, class: Option<&[(usize, bool)]>) -> Check<'a> {
        let (items, hardware) = stream(s, class);
        let instrs: Vec<Arc<Instr>> = items.iter().map(|i| i.instr.clone()).collect();
        let g = cfg_of(&instrs, None, true);
        let mut c = Check {
            s,
            items,
            hardware,
            g,
            writes: Vec::new(),
            reads: Vec::new(),
            sites: Vec::new(),
            by_item: Vec::new(),
            every: Defs::default(),
            whole: Defs::default(),
        };
        for k in 0..c.items.len() {
            let (w, r) = c.accesses(k);
            c.writes.push(w);
            c.reads.push(r);
        }
        c.find_sites();
        let keys: BTreeMap<Dword, usize> = (c.sites.iter().map(|s| s.dword))
            .collect::<BTreeSet<_>>()
            .into_iter()
            .enumerate()
            .map(|(k, d)| (d, k))
            .collect();
        let defs = |whole_only: bool| {
            let sites = c.sites.iter().map(|s| Def {
                row: Some(s.item),
                key: keys[&s.dword],
                kills: !whole_only || !s.partial,
            });
            Defs::new(c.items.len(), sites.collect())
        };
        (c.every, c.whole) = (defs(false), defs(true));
        c
    }

    fn access(&self, r: &Reg) -> Access {
        Access {
            reg: *r,
            alias: self.s.alias(r),
            through: self.s.through(r),
        }
    }

    fn key(&self, a: &Access) -> (char, u32, u32, u32) {
        (a.reg.file, self.s.phys(&a.reg), a.reg.n, a.alias)
    }

    /// An item's writes and reads.
    fn accesses(&self, k: usize) -> (Vec<Access>, Vec<Access>) {
        let i = &self.items[k].instr;
        let mut writes: Vec<Access> = Vec::new();
        let regs: Vec<Reg> = if i.form().kind == Kind::Comment {
            i.ops().iter().filter_map(Operand::reg).copied().collect()
        } else {
            i.defs().copied().collect()
        };
        for r in &regs {
            let a = self.access(r);
            if !writes.iter().any(|w| self.key(w) == self.key(&a)) {
                writes.push(a);
            }
        }
        let mut reads: Vec<Access> = Vec::new();
        let push = |a: Access, reads: &mut Vec<Access>| {
            if !reads.iter().any(|x| self.key(x) == self.key(&a)) {
                reads.push(a);
            }
        };
        if i.form().kind != Kind::Comment {
            let kept = i.kept_dst();
            (i.uses().filter(|&r| Some(r) != kept)).for_each(|r| push(self.access(r), &mut reads));
            let mut destination: BTreeSet<(char, u32, u32, u32)> = BTreeSet::new();
            for r in i.ops().iter().filter_map(Operand::reg) {
                let a = self.access(r);
                let written = writes.iter().any(|w| self.key(w) == self.key(&a));
                if written && destination.insert(self.key(&a)) {
                    continue;
                }
                push(a, &mut reads);
            }
            if i.form().flags.has(Flags::ACCUMULATE) {
                writes.iter().for_each(|w| push(*w, &mut reads));
            }
        }
        (writes, reads)
    }

    /// Tracked dwords, then every item's sites on them.
    fn find_sites(&mut self) {
        let mut names: HashMap<Dword, BTreeSet<u32>> = HashMap::new();
        for a in self.writes.iter().chain(&self.reads).flatten() {
            dwords(self.s, &a.reg)
                .into_iter()
                .for_each(|d| _ = names.entry(d).or_default().insert(a.alias));
        }
        let tracked: BTreeSet<Dword> = names
            .into_iter()
            .filter(|(_, n)| n.len() > 1)
            .map(|(d, _)| d)
            .collect();
        let full = self.exec_states();
        for (k, &whole) in full.iter().enumerate() {
            let mut mine = Vec::new();
            let instr = &self.items[k].instr;
            for a in &self.writes[k] {
                for d in dwords(self.s, &a.reg)
                    .into_iter()
                    .filter(|d| tracked.contains(d))
                {
                    if mine.iter().any(|&s: &usize| self.sites[s].dword == d) {
                        continue;
                    }
                    let site = Site {
                        item: k,
                        dword: d,
                        alias: a.alias,
                        through: a.through,
                        partial: instr.keeps_old_in(d.0, !whole),
                    };
                    mine.push(self.sites.len());
                    self.sites.push(site);
                }
            }
            self.by_item.push(mine);
        }
        self.reads = self
            .reads
            .iter()
            .map(|r| {
                r.iter()
                    .copied()
                    .filter(|a| dwords(self.s, &a.reg).iter().any(|d| tracked.contains(d)))
                    .collect()
            })
            .collect();
    }

    /// Whether the item is VALU, VMEM or DS.
    fn lane_unit(&self, k: usize) -> bool {
        matches!(
            self.items[k].instr.form().kind,
            Kind::Valu | Kind::Vmem | Kind::Ds
        )
    }

    /// Per item, whether EXEC is whole on entry.
    fn exec_states(&self) -> Vec<bool> {
        let g = &self.g;
        let through = |st: &(bool, BTreeSet<Dword>), j: usize| {
            let mut st = st.clone();
            g.block(j).for_each(|k| self.exec_step(k, &mut st));
            st
        };
        let merge =
            |a: &(bool, BTreeSet<Dword>), b: &(bool, BTreeSet<Dword>)| (a.0 && b.0, &a.1 & &b.1);
        let mut out = vec![false; self.items.len()];
        if g.nblocks() == 0 {
            return out;
        }
        let entry = solve_forward(g, vec![(0, (true, BTreeSet::new()))], through, merge, None);
        for (j, st) in entry.iter().enumerate() {
            let Some(mut st) = st.clone() else { continue };
            for k in g.block(j) {
                out[k] = st.0;
                self.exec_step(k, &mut st);
            }
        }
        out
    }

    /// One item's effect on whether EXEC is whole and its saved copies.
    fn exec_step(&self, k: usize, st: &mut (bool, BTreeSet<Dword>)) {
        let i = &self.items[k].instr;
        let f = i.form();
        let t = self.s.target();
        let exec = phys_named("exec").expect("exec");
        let mask = t.mask_reg(exec).name;
        let defs: Vec<Dword> = i.defs().flat_map(|r| dwords(self.s, r)).collect();
        let names_exec = i
            .ops()
            .iter()
            .any(|o| o.whole().is_some_and(|w| w.number == exec.number));
        let movm = f.canon == t.masked("s.mov.mask");
        if f.kind == Kind::Salu && (names_exec || f.flags.has(Flags::SAVEEXEC) || !st.1.is_empty())
        {
            if f.flags.has(Flags::SAVEEXEC) {
                if st.0 {
                    st.1.extend(&defs);
                } else {
                    defs.iter().for_each(|d| _ = st.1.remove(d));
                }
                st.0 = false;
            } else if i
                .ops()
                .first()
                .and_then(Operand::whole)
                .is_some_and(|w| w.number == exec.number)
            {
                let to_mask = i.phys(0).is_some_and(|p| p.name == mask);
                let minus_one = matches!(i.ops().get(1), Some(Operand::Imm(m)) if match m.value {
                    thin_ir::ImmValue::Int(v) => v == -1,
                    thin_ir::ImmValue::Float(x) => x == -1.0,
                });
                let sources: Vec<Dword> = i.ops()[1..]
                    .iter()
                    .filter_map(Operand::reg)
                    .flat_map(|r| dwords(self.s, r))
                    .collect();
                st.0 = if to_mask && i.ops().len() == 2 && minus_one && i.fields().is_empty() {
                    true
                } else {
                    to_mask
                        && movm
                        && !sources.is_empty()
                        && sources.iter().all(|d| st.1.contains(d))
                };
            } else if !defs.is_empty() {
                let copy = movm
                    && i.ops()
                        .last()
                        .and_then(Operand::phys)
                        .is_some_and(|p| p.name == mask)
                    && st.0;
                if copy {
                    st.1.extend(&defs);
                } else {
                    defs.iter().for_each(|d| _ = st.1.remove(d));
                }
            }
        } else if f.kind == Kind::Valu && f.flags.has(Flags::CMPX) {
            st.0 = false;
        } else if !st.1.is_empty() && defs.iter().any(|d| st.1.contains(d)) {
            defs.iter().for_each(|d| _ = st.1.remove(d));
        }
    }

    /// The reaching definitions of one rule: every write ends its dwords'
    /// sites, or (whole only) only a write that is not partial does.
    fn defs(&self, whole_only: bool) -> &Defs {
        if whole_only {
            &self.whole
        } else {
            &self.every
        }
    }

    fn apply(&self, set: &mut Bits, k: usize, whole_only: bool) {
        self.defs(whole_only).step(set, k);
    }

    /// The reaching sites on entry to each block, for one rule.
    fn reaching(&self, whole_only: bool) -> Vec<Bits> {
        self.defs(whole_only).solve(&self.g)
    }

    /// The chains of partial sites to (dword, value) they write over.
    fn chains(&self, whole: &[Bits]) -> BTreeMap<usize, BTreeSet<(Dword, u32)>> {
        let mut out: BTreeMap<usize, BTreeSet<(Dword, u32)>> = BTreeMap::new();
        for (j, entry) in whole.iter().enumerate() {
            let mut st = entry.clone();
            for k in self.g.block(j) {
                for &p in &self.by_item[k] {
                    let ps = self.sites[p];
                    if !ps.partial {
                        continue;
                    }
                    for n in st.ones() {
                        let ns = self.sites[n];
                        if !ns.partial && ns.dword == ps.dword && ns.alias != ps.alias {
                            out.entry(p).or_default().insert((ps.dword, ns.alias));
                        }
                    }
                }
                self.apply(&mut st, k, true);
            }
        }
        out
    }

    /// The findings of this class's stream.
    fn findings(&self) -> Vec<Finding> {
        let every = self.reaching(false);
        let whole_sets = self.reaching(true);
        let chains = self.chains(&whole_sets);
        let full = self.exec_states();
        let mut out: Vec<Finding> = Vec::new();
        for j in 0..self.g.nblocks() {
            let (mut e, mut w) = (every[j].clone(), whole_sets[j].clone());
            for k in self.g.block(j) {
                let whole = full[k] || !self.lane_unit(k);
                for read in &self.reads[k] {
                    for d in dwords(self.s, &read.reg) {
                        let mut bad: BTreeSet<usize> = e
                            .ones()
                            .into_iter()
                            .filter(|&s| {
                                self.sites[s].dword == d && self.sites[s].alias != read.alias
                            })
                            .collect();
                        let rule1 = bad.clone();
                        if whole {
                            let chained = |s: &usize| {
                                chains.get(s).is_some_and(|c| c.contains(&(d, read.alias)))
                            };
                            bad.extend(w.ones().into_iter().filter(chained));
                        }
                        self.waive(k, read, &mut bad);
                        let Some(site) = self.reported(k, &bad) else {
                            continue;
                        };
                        let item = self.sites[site].item;
                        if out
                            .iter()
                            .any(|f| f.item == k && self.sites[f.site].item == item)
                        {
                            continue;
                        }
                        out.push(Finding {
                            item: k,
                            read: *read,
                            dword: d,
                            site,
                            more: bad.len() - 1,
                            rule2: !rule1.contains(&site),
                        });
                    }
                }
                self.apply(&mut e, k, false);
                self.apply(&mut w, k, true);
            }
        }
        out
    }

    /// Removes the sites an item's waivers cover for a read.
    fn waive(&self, k: usize, read: &Access, bad: &mut BTreeSet<usize>) {
        let name = self.s.name(read.through);
        for (r, writer) in &self.items[k].instr.attrs().alias_ok {
            if &**r != name {
                continue;
            }
            bad.retain(|&s| {
                writer
                    .as_deref()
                    .is_some_and(|w| w != self.s.name(self.first_writer(s)))
            });
        }
    }

    /// The name of the first write on a site's dword in its item.
    fn first_writer(&self, site: usize) -> u32 {
        let s = self.sites[site];
        let first = self.writes[s.item]
            .iter()
            .find(|a| dwords(self.s, &a.reg).contains(&s.dword));
        first.map_or(s.through, |a| a.through)
    }

    /// The site to report: the latest before k, else the first.
    fn reported(&self, k: usize, bad: &BTreeSet<usize>) -> Option<usize> {
        let before = bad
            .iter()
            .copied()
            .filter(|&s| self.sites[s].item < k)
            .max_by_key(|&s| self.sites[s].item);
        before.or_else(|| bad.iter().copied().min_by_key(|&s| self.sites[s].item))
    }

    /// An item's number in messages.
    fn number(&self, k: usize) -> usize {
        k - usize::from(self.hardware)
    }

    fn quoted(&self, k: usize) -> Vec<Part> {
        let orig = self.items[k].orig.expect("an instruction item");
        vec![
            Part::Text("`".into()),
            Part::Instr(orig),
            Part::Text(format!("` (instruction {})", self.number(k))),
        ]
    }

    /// The path label from a site's write to a read.
    fn path(&self, f: &Finding, whole_only: bool) -> Option<String> {
        let site = self.sites[f.site];
        let (wb, rb) = (self.g.block_of[site.item], self.g.block_of[f.item]);
        if wb == rb && site.item < f.item {
            return None;
        }
        let kills = |b: usize| {
            self.g
                .block(b)
                .any(|k| self.defs(whole_only).ends(k, f.site) && k != site.item)
        };
        let mut seen = BTreeSet::new();
        let mut queue: VecDeque<(usize, usize)> =
            self.g.block_succ[wb].iter().map(|&b| (b, b)).collect();
        while let Some((b, first)) = queue.pop_front() {
            if !seen.insert(b) {
                continue;
            }
            if b == rb {
                return Some(self.block_name(first));
            }
            if !kills(b) {
                queue.extend(self.g.block_succ[b].iter().map(|&n| (n, first)));
            }
        }
        Some("(a path not found)".into())
    }

    fn block_name(&self, b: usize) -> String {
        let start = self.g.starts[b];
        let i = &self.items[start].instr;
        match i.form().kind {
            Kind::Label => i
                .field_named("name")
                .map(FieldValue::py_str)
                .unwrap_or_default(),
            _ => format!("(the fall-through at {})", self.number(start)),
        }
    }

    /// A finding's message.
    fn message(&self, f: &Finding) -> Vec<Part> {
        let site = self.sites[f.site];
        let s = self.s;
        let n = s.name(f.read.through);
        let w = s.name(self.first_writer(f.site));
        let r = s.reg_text(&f.read.reg);
        let fd = format!("{}{}", f.dword.0, f.dword.1);
        let j = f.dword.1 - s.phys(&f.read.reg);
        let dword = if f.read.reg.n > 1 {
            format!(" (its dword {j})")
        } else {
            String::new()
        };
        let mut msg = self.quoted(f.item);
        let writer: Vec<Part> = match self.items[site.item].orig {
            Some(_) => self.quoted(site.item),
            None => vec![Part::Text("the hardware at entry".into())],
        };
        let path = self.path(f, f.rule2);
        if !f.rule2 {
            let through = path
                .as_ref()
                .map_or(String::new(), |p| format!(" (through {p})"));
            msg.push(Part::Text(format!(" reads {r} through `{n}`, but on a path to it{through} `{n}` is not written: {fd}{dword} holds `{w}` there, written by ")));
            msg.extend(writer);
            let more = if f.more > 0 {
                format!(" (and {} more write(s) of other values)", f.more)
            } else {
                String::new()
            };
            let hint = match &path {
                Some(p) if !s.by_wave() && p.starts_with(".Lwaves") => {
                    " (the path runs through a wave block: with by_wave, each wave class is checked on its own paths)"
                }
                _ => "",
            };
            msg.push(Part::Text(format!(
                "{more}. A derived file gives a value's registers to others before its first write: write `{n}` on every \
                 path to the read (all its dwords; a default before the branch), or - when that path never runs (the \
                 read's branch taken only when the write's was: one condition) - say so around the read: `with \
                 b.alias_ok(\"{n}\", why=..)`{hint}"
            )));
        } else {
            msg.push(Part::Text(format!(
                " reads {r} through `{n}`, but {fd} was written through `{w}` by "
            )));
            msg.extend(writer);
            let reaching = path.map_or(String::new(), |p| format!(" (reaching it through {p})"));
            let more = if f.more > 0 {
                format!(", and {} more write(s) through other names", f.more)
            } else {
                String::new()
            };
            msg.push(Part::Text(format!(
                " (a partial write over the whole value){reaching}{more}: `{w}` and `{n}` share the register; if they name \
                 one value, declare it (same_as), else the write clobbers a value still to be read"
            )));
        }
        msg
    }
}

/// Every finding of the alias check, each a message: the findings of each
/// wave class in turn, repeats dropped.
pub fn findings(s: &Snapshot) -> Vec<Vec<Part>> {
    let mut out: Vec<Vec<Part>> = Vec::new();
    for class in s.wave_classes() {
        let c = Check::new(s, class.as_deref());
        for f in c.findings() {
            let m = c.message(&f);
            if !out.contains(&m) {
                out.push(m);
            }
        }
    }
    out
}

/// The alias check: one ScheduleError naming the first 8 findings.
pub fn check(s: &Snapshot) -> Result<(), Diag> {
    let found = findings(s);
    if found.is_empty() {
        return Ok(());
    }
    let mut msg = vec![Part::Text(
        "register alias: a value read through another name than it was written through (in a derived file: a value \
         read where some path did not write it)"
            .into(),
    )];
    for f in found.iter().take(8) {
        msg.push(Part::Text("\n  ".into()));
        msg.extend(f.iter().cloned());
    }
    if found.len() > 8 {
        msg.push(Part::Text(format!("\n  ... and {} more", found.len() - 8)));
    }
    Err(refusal(msg))
}
