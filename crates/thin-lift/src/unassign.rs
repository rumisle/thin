//! Register unassignment, inverse of register assignment.
//!
//! A physical [`Program`] names registers; thin's IR before assignment
//! names values. Values are recovered from def-use chains over the whole
//! represented CFG (call and return edges included), one dword at a time:
//! every read joins the writes of its dword that reach it ([`thin_ir::reach`]),
//! so one physical register may hold several unrelated values and one value
//! keeps every write a later read may see. Then the dword webs one operand
//! names together (a 64-bit pair, a WMMA fragment, a load's tuple) become
//! one value, laid out as the registers were.
//!
//! A write that keeps part of its destination ([`Instr::keeps_old_in`]: a
//! partial EXEC, a writelane, a D16 or 16-bit half, op_sel's destination
//! half) and a read-modify-write slot (an accumulator, `s_cmov`, an
//! atomic's returned data) join the value they modify. A destination
//! naming exactly the registers of a source of the same instruction stays
//! on them (the source's value and the result are one value): in-place
//! updates such as a WMMA's C = D stay in place. A write over a load
//! that may still be in flight (by the schedule's waits) joins the load's
//! value: the program relies on that register being the same one.
//!
//! Some values keep their registers ([`Reason`]): hardware and ABI entry
//! inputs, reads no path from the entry wrote, values live on a call or
//! return edge (the convention the resolved call graph implies: the callee
//! shares the caller's registers, so whatever crosses the edge stays where
//! both sides expect it), every value of a file an M0-relative move
//! indexes, and everything when the control flow is incomplete. Named
//! registers (`exec`, `vcc`, `m0`, `scc`, TTMPs) are not numbered values
//! and stay physical operands.
//!
//! **Exact law:** [`Assignment::pin`] of the result is the input
//! program: the same instructions, operands, entry and control flow,
//! rendering to the same text and bytes. **Semantic law:** thin's own
//! register assignment of the virtual program, applied by [`place`], is a
//! valid program computing the same results (tested on the emulator).

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use thin_ir::control::{ControlFlow, EdgeKind, TransferKind};
use thin_ir::reach::{Def, Defs};
use thin_ir::sets::Bits;
use thin_ir::{Instr, Operand, Reg, Snapshot, ValueInfo, R, W};

use crate::program::{EntryBinding, EntryValue, Program, ProgramError};
use crate::schedule::Schedule;

/// Why a value keeps its original registers.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Reason {
    /// The hardware or the ABI fills it at entry.
    Entry(EntryValue),
    /// A read that no write on any path from the entry reaches.
    Undefined,
    /// Live on a call or return edge: the callee and its callers name it
    /// at this register.
    CallBoundary,
    /// Its file is indexed through M0 (`s_movrel*`, `v_movrel*`), so any
    /// of its registers may be read or written.
    RelativeIndexing,
    /// A transfer has no complete target set: unknown code may read it.
    UnresolvedControlFlow,
}

impl Reason {
    /// A stable coverage label.
    pub fn name(&self) -> &'static str {
        match self {
            Reason::Entry(_) => "entry input",
            Reason::Undefined => "read where no path wrote it",
            Reason::CallBoundary => "live across a call or return edge",
            Reason::RelativeIndexing => "file indexed through m0",
            Reason::UnresolvedControlFlow => "unresolved control flow",
        }
    }
}

/// The residue of unassignment: where each value was, and why the pinned
/// ones stay there.
#[derive(Clone, Debug)]
pub struct Assignment {
    /// Per virtual value, its original first register.
    pub bases: Vec<u32>,
    /// Per virtual value, why it is pinned; empty for a free value.
    pub reasons: Vec<Vec<Reason>>,
    /// The physical program's values (one backing value per file).
    physical: Vec<ValueInfo>,
}

impl Assignment {
    /// The values pinned at their original registers.
    pub fn pinned(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.bases.len()).filter(|&v| !self.reasons[v].is_empty())
    }

    /// The exact law's inverse: the virtual program on its original registers.
    pub fn pin(&self, program: &Program) -> Result<Program, ProgramError> {
        place_on(program, &self.physical, &self.bases)
    }
}

fn error(class: &'static str, detail: impl Into<String>) -> ProgramError {
    ProgramError {
        offset: None,
        class,
        detail: detail.into(),
    }
}

/// A union-find over dense ids.
struct Union(Vec<usize>);

impl Union {
    fn new(n: usize) -> Union {
        Union((0..n).collect())
    }

    fn find(&mut self, mut x: usize) -> usize {
        while self.0[x] != x {
            self.0[x] = self.0[self.0[x]];
            x = self.0[x];
        }
        x
    }

    fn join(&mut self, a: usize, b: usize) {
        let (a, b) = (self.find(a), self.find(b));
        if a != b {
            self.0[a.max(b)] = a.min(b);
        }
    }
}

/// The numbered register files of a physical program: per file its
/// backing value and its first key.
struct Keys {
    files: Vec<(char, usize, usize, u32)>,
    count: usize,
}

impl Keys {
    fn of(s: &Snapshot) -> Result<Keys, ProgramError> {
        let mut files = Vec::new();
        let mut count = 0;
        for (v, info) in s.values().iter().enumerate() {
            if info.phys != Some(0) || files.iter().any(|f: &(char, _, _, _)| f.0 == info.file) {
                return Err(error(
                    "physical values",
                    "unassignment needs one backing value per file, at register 0",
                ));
            }
            files.push((info.file, v, count, info.n));
            count += info.n as usize;
        }
        Ok(Keys { files, count })
    }

    /// A physical dword's key.
    fn key(&self, file: char, phys: u32) -> usize {
        let f = self.files.iter().find(|f| f.0 == file).expect("a file");
        f.2 + phys as usize
    }

    /// A key's (file, register).
    fn dword(&self, key: usize) -> (char, u32) {
        let f = (self.files.iter())
            .rev()
            .find(|f| f.2 <= key)
            .expect("a key's file");
        (f.0, (key - f.2) as u32)
    }
}

/// One register operand's dword at a row.
#[derive(Clone, Copy)]
struct Occurrence {
    row: usize,
    op: usize,
    key: usize,
    /// It reads the dword's old value: a source, a read-modify-write slot,
    /// or a destination keeping part of it.
    reads: bool,
    /// A read of the data, not only of the kept part of a destination.
    data: bool,
}

/// The rows whose transfers follow a call or return edge.
fn boundary_rows(control: Option<&ControlFlow>) -> Vec<usize> {
    control.map_or_else(Vec::new, |c| {
        (c.transfers.iter())
            .filter(|t| {
                t.edges
                    .iter()
                    .any(|e| matches!(e.kind, EdgeKind::Call | EdgeKind::Return { .. }))
            })
            .map(|t| t.instruction)
            .collect()
    })
}

/// Whether a form indexes its register file through M0.
fn relative(i: &Instr) -> Option<char> {
    let op = i.form().op.as_str();
    if op.starts_with("s_movrel") {
        Some('s')
    } else if op.starts_with("v_movrel") {
        Some('v')
    } else {
        None
    }
}

/// The analysis: occurrences, sites and the webs joining them.
struct Webs {
    keys: Keys,
    /// Node k < keys.count is key k's entry write; then the occurrences.
    occurrences: Vec<Occurrence>,
    union: Union,
    /// Per key, whether a data read sees only its entry write.
    undefined: Bits,
    /// Per node, whether a value crossing a call or return edge reaches it.
    crossing: Bits,
    /// Per occurrence, whether it writes its dword.
    writes: Bits,
}

impl Webs {
    fn node(&self, occurrence: usize) -> usize {
        self.keys.count + occurrence
    }

    fn build(program: &Program, schedule: &Schedule) -> Result<Webs, ProgramError> {
        let s = &program.snapshot;
        let keys = Keys::of(s)?;
        let full = s.target().full_mask();
        let exec = s.exec();
        let mut occurrences = Vec::new();
        let mut sites: Vec<Def> = (0..keys.count)
            .map(|key| Def {
                row: None,
                key,
                kills: true,
            })
            .collect();
        // The node of each site: the entry writes', then the occurrences'.
        let mut site_node: Vec<usize> = (0..keys.count).collect();
        let mut by_row: Vec<std::ops::Range<usize>> = Vec::new();
        let mut writes = Bits::new();
        for (row, i) in s.instrs().iter().enumerate() {
            let start = occurrences.len();
            let partial = exec[row] != Some(full);
            let kept = i.kept_dst().is_some();
            for (op, o) in i.ops().iter().enumerate() {
                let Some(r) = o.reg() else { continue };
                // A register in a slot of no role is taken as read.
                let role = match i.roles()[op] {
                    0 => R,
                    role => role,
                };
                let first = s.phys(r);
                let merges = role & W != 0 && (role & R != 0 || i.keeps_old_in(r.file, partial));
                for t in 0..r.n {
                    let key = keys.key(r.file, first + t);
                    if role & W != 0 {
                        writes.set(occurrences.len());
                        sites.push(Def {
                            row: Some(row),
                            key,
                            kills: !merges,
                        });
                        site_node.push(keys.count + occurrences.len());
                    }
                    occurrences.push(Occurrence {
                        row,
                        op,
                        key,
                        reads: role & R != 0 || merges,
                        data: role & R != 0 && !(op == 0 && kept),
                    });
                }
            }
            by_row.push(start..occurrences.len());
        }
        let n = s.instrs().len();
        let mut w = Webs {
            union: Union::new(keys.count + occurrences.len()),
            keys,
            occurrences,
            undefined: Bits::new(),
            crossing: Bits::new(),
            writes,
        };
        let g = s.cfg(None);
        let defs = Defs::new(n, sites.clone());
        let entries = defs.solve(&g);
        let mut joins = Vec::new();
        defs.walk(&g, &entries, |row, st| {
            for k in by_row[row].clone() {
                let o = w.occurrences[k];
                if !o.reads {
                    continue;
                }
                let reaching = defs.of_key(st, o.key);
                // Only the entry reaches a data read: no path wrote it
                //
                if o.data && reaching.iter().all(|&site| site < w.keys.count) {
                    w.undefined.set(o.key);
                }
                for site in reaching {
                    joins.push((w.keys.count + k, site_node[site]));
                }
            }
        });
        for (a, b) in joins {
            w.union.join(a, b);
        }
        w.tie(s, &by_row);
        w.in_flight(program, schedule, &by_row)?;
        let boundary = boundary_rows(s.control_flow());
        if !boundary.is_empty() {
            w.cross(n, &g, sites, &boundary, &by_row);
        }
        Ok(w)
    }

    /// A write over a load that may still be in flight joins the load's
    /// value. On hardware the load's late return would overwrite it, so
    /// the program relies on that path not running with the load
    /// outstanding (or on the race): the input registers are the
    /// witness that one register fits both, and keeping them one value
    /// keeps register assignment's in-flight rule
    /// from holding a second register for what the program never
    /// separates.
    fn in_flight(
        &mut self,
        program: &Program,
        schedule: &Schedule,
        by_row: &[std::ops::Range<usize>],
    ) -> Result<(), ProgramError> {
        let (s, row_of) = program.scheduled_rows(schedule)?;
        let back: HashMap<usize, usize> = row_of.iter().enumerate().map(|(k, &r)| (r, k)).collect();
        let loads: Vec<usize> = (0..s.instrs().len())
            .filter(|r| {
                let i = &s.instrs()[*r];
                i.form().counter.is_some() && i.defs().next().is_some()
            })
            .collect();
        if loads.is_empty() {
            return Ok(());
        }
        let flying = thin_ir::flight::in_flight(&s, &s.cfg(None), &loads);
        // Per load, the occurrence writing each of its dwords.
        let written: Vec<HashMap<usize, usize>> = (loads.iter())
            .map(|&r| {
                (by_row[back[&r]].clone())
                    .filter(|&l| self.roles_write(l))
                    .map(|l| (self.occurrences[l].key, l))
                    .collect()
            })
            .collect();
        let mut pairs = Vec::new();
        for (k, rows) in by_row.iter().enumerate() {
            let at = &flying[row_of[k]];
            if at.is_empty() {
                continue;
            }
            for o in rows.clone().filter(|&o| self.roles_write(o)) {
                let key = self.occurrences[o].key;
                pairs.extend(
                    at.iter()
                        .filter_map(|&j| written[j].get(&key))
                        .map(|&l| (o, l)),
                );
            }
        }
        for (a, b) in pairs {
            let (a, b) = (self.node(a), self.node(b));
            self.union.join(a, b);
        }
        Ok(())
    }

    /// Whether an occurrence writes its dword.
    fn roles_write(&self, o: usize) -> bool {
        self.writes.has(o)
    }

    /// A destination naming exactly a source's registers stays on them.
    fn tie(&mut self, s: &Snapshot, by_row: &[std::ops::Range<usize>]) {
        let mut pairs = Vec::new();
        for (row, i) in s.instrs().iter().enumerate() {
            let regs: Vec<(usize, Reg, u8)> = (i.ops().iter().enumerate())
                .filter_map(|(op, o)| o.reg().map(|r| (op, *r, i.roles()[op])))
                .collect();
            let occurrences = &self.occurrences;
            let of = |op: usize| -> Vec<usize> {
                (by_row[row].clone())
                    .filter(|&k| occurrences[k].op == op)
                    .collect()
            };
            for &(d, rd, role_d) in &regs {
                for &(u, ru, role_u) in &regs {
                    let same = rd.file == ru.file && s.phys(&rd) == s.phys(&ru) && rd.n == ru.n;
                    if d != u && role_d & W != 0 && role_u & R != 0 && same {
                        pairs.extend(of(d).into_iter().zip(of(u)));
                    }
                }
            }
        }
        for (a, b) in pairs {
            let (a, b) = (self.node(a), self.node(b));
            self.union.join(a, b);
        }
    }

    /// Marks the occurrences a value crossing a call or return edge reaches:
    /// the same sites, and per key one more, written at every boundary row
    /// after the row's own writes.
    fn cross(
        &mut self,
        n: usize,
        g: &thin_ir::cfg::Cfg,
        mut sites: Vec<Def>,
        boundary: &[usize],
        by_row: &[std::ops::Range<usize>],
    ) {
        let first = sites.len();
        let nkeys = self.keys.count;
        sites.extend((0..nkeys).map(|key| Def {
            row: Some(boundary[0]),
            key,
            kills: true,
        }));
        let mut defs = Defs::new(n, sites);
        for &row in &boundary[1..] {
            (first..first + nkeys).for_each(|site| defs.also(row, site));
        }
        let entries = defs.solve(g);
        let mut marks = Vec::new();
        defs.walk(g, &entries, |row, st| {
            for k in by_row[row].clone() {
                let o = self.occurrences[k];
                if o.reads && defs.of_key(st, o.key).iter().any(|&s| s >= first) {
                    marks.push(nkeys + k);
                }
            }
        });
        marks.into_iter().for_each(|m| self.crossing.set(m));
    }
}

/// Recovers values from a physical program with complete or explicitly
/// incomplete control flow, and its schedule (the waits say which loads
/// may be in flight): the virtual program and its residue.
pub fn unassign(
    program: &Program,
    schedule: &Schedule,
) -> Result<(Program, Assignment), ProgramError> {
    let s = &program.snapshot;
    let mut webs = Webs::build(program, schedule)?;
    let nkeys = webs.keys.count;
    let total = nkeys + webs.occurrences.len();
    // Values: webs one operand names together, and each entry binding.
    let mut values = Union::new(total);
    let roots: Vec<usize> = (0..total).map(|k| webs.union.find(k)).collect();
    let mut at = 0;
    while at < webs.occurrences.len() {
        let (row, op) = (webs.occurrences[at].row, webs.occurrences[at].op);
        let mut end = at + 1;
        while end < webs.occurrences.len()
            && (webs.occurrences[end].row, webs.occurrences[end].op) == (row, op)
        {
            values.join(roots[nkeys + at], roots[nkeys + end]);
            end += 1;
        }
        at = end;
    }
    let mut entry_keys: BTreeMap<usize, EntryValue> = BTreeMap::new();
    for b in &program.entry {
        let Operand::Reg(r) = &b.operand else {
            continue;
        };
        let first = s.phys(r);
        for t in 0..r.n {
            let key = webs.keys.key(r.file, first + t);
            entry_keys.insert(key, b.value.clone());
            values.join(roots[webs.keys.key(r.file, first)], roots[key]);
        }
    }
    let key_of = |node: usize| {
        if node < nkeys {
            node
        } else {
            webs.occurrences[node - nkeys].key
        }
    };
    // Entry writes reaching any node of a value make it an entry value.
    let mut entry_of_web: HashMap<usize, usize> = HashMap::new();
    for (key, &root) in roots.iter().enumerate().take(nkeys) {
        entry_of_web.entry(root).or_insert(key);
    }
    let relative: Vec<char> = s.instrs().iter().filter_map(|i| relative(i)).collect();
    let incomplete = !program.indirect.is_empty()
        || s.control_flow().is_some_and(|c| !c.complete())
        || s.instrs().iter().any(|i| {
            matches!(
                i.form().canon.as_str(),
                "s.setpc.b64" | "s.swappc.b64" | "s.call.b64"
            )
        }) && s.control_flow().is_none();
    // Why a value's nodes keep it where it is, and its registers.
    let describe = |nodes: &[usize]| {
        let (file, first) = webs.keys.dword(key_of(nodes[0]));
        let (mut lo, mut hi) = (first, first);
        let mut why: Vec<Reason> = Vec::new();
        let add = |r: Reason, why: &mut Vec<Reason>| {
            if !why.contains(&r) {
                why.push(r);
            }
        };
        for &node in nodes {
            let (f, p) = webs.keys.dword(key_of(node));
            debug_assert_eq!(f, file, "a value in one file");
            (lo, hi) = (lo.min(p), hi.max(p));
            if node >= nkeys && webs.crossing.has(node) {
                add(Reason::CallBoundary, &mut why);
            }
            if let Some(&key) = entry_of_web.get(&roots[node]) {
                if let Some(e) = entry_keys.get(&key) {
                    add(Reason::Entry(e.clone()), &mut why);
                } else if webs.undefined.has(key) {
                    add(Reason::Undefined, &mut why);
                }
            }
        }
        if relative.contains(&file) {
            add(Reason::RelativeIndexing, &mut why);
        }
        if incomplete {
            add(Reason::UnresolvedControlFlow, &mut why);
        }
        (file, lo, hi, why)
    };
    // Number the values by first appearance, then the entry-only ones.
    // Pinned values sharing a register are one value: they stay there
    // anyway, and register assignment's occupancy is context-insensitive
    // at returns, so two values its callers keep apart on one register
    // would both be occupied at the callee's return.
    let order: Vec<usize> = (nkeys..total).chain(entry_keys.keys().copied()).collect();
    let (number, described) = loop {
        let mut number: HashMap<usize, usize> = HashMap::new();
        let mut members: Vec<Vec<usize>> = Vec::new();
        for &node in &order {
            let v = values.find(roots[node]);
            let k = *number.entry(v).or_insert_with(|| {
                members.push(Vec::new());
                members.len() - 1
            });
            members[k].push(node);
        }
        let described: Vec<_> = members.iter().map(|m| describe(m)).collect();
        let mut owner: HashMap<(char, u32), usize> = HashMap::new();
        let mut joined = false;
        for (k, (file, lo, hi, why)) in described.iter().enumerate() {
            if why.is_empty() {
                continue;
            }
            let root = values.find(roots[members[k][0]]);
            for p in *lo..=*hi {
                match owner.get(&(*file, p)) {
                    Some(&other) if values.find(other) != values.find(root) => {
                        values.join(other, root);
                        joined = true;
                    }
                    Some(_) => {}
                    None => _ = owner.insert((*file, p), root),
                }
            }
        }
        if !joined {
            break (number, described);
        }
    };
    let mut bases = Vec::new();
    let mut reasons = Vec::new();
    let mut infos = Vec::new();
    let mut files = Vec::new();
    for (file, lo, hi, why) in described {
        bases.push(lo);
        files.push(file);
        let n = hi - lo + 1;
        let a64 = s.target().file(file).map_or(1, |f| f.align64.max(1));
        infos.push(ValueInfo {
            file,
            n,
            pin: (!why.is_empty()).then_some(lo),
            align: if n >= 2 { a64 } else { 1 },
            index: infos.len() as u32,
            ..ValueInfo::default()
        });
        reasons.push(why);
    }
    // A free value's provisional register is past its file, apart from
    // every other (as a DSL value's is before placement), so the
    // analyses that key by register see value identity.
    let mut cursor: HashMap<char, u32> = HashMap::new();
    for info in &mut infos {
        info.phys = match info.pin {
            Some(p) => Some(p),
            None => {
                let size = s.target().hooks().reg_count(info.file);
                let c = cursor.entry(info.file).or_insert(size);
                let at = *c;
                *c += info.n;
                Some(at)
            }
        };
    }
    let node_value: Vec<Option<usize>> = (0..total)
        .map(|node| number.get(&values.find(roots[node])).copied())
        .collect();
    let value_of_node = |node: usize| node_value[node].expect("a value of a used node");
    let mut by_operand: HashMap<(usize, usize), usize> = HashMap::new();
    for (k, o) in webs.occurrences.iter().enumerate() {
        by_operand
            .entry((o.row, o.op))
            .or_insert_with(|| value_of_node(nkeys + k));
    }
    let virtual_reg = |r: &Reg, v: usize| Reg::new(v as i32, r.file, s.phys(r) - bases[v], r.n);
    let rows: Vec<Arc<Instr>> = (s.instrs().iter().enumerate())
        .map(|(row, i)| {
            if !i.ops().iter().any(|o| o.reg().is_some()) {
                return i.clone();
            }
            let ops = (i.ops().iter().enumerate())
                .map(|(op, o)| match o {
                    Operand::Reg(r) => Operand::Reg(virtual_reg(r, by_operand[&(row, op)])),
                    o => o.clone(),
                })
                .collect();
            Arc::new(i.with_ops(ops))
        })
        .collect();
    let entry_reg = |r: &Reg| -> Reg {
        let key = webs.keys.key(r.file, s.phys(r));
        virtual_reg(r, value_of_node(key))
    };
    let entry: Vec<Reg> = s.entry().iter().map(entry_reg).collect();
    let bindings = (program.entry.iter())
        .map(|b| EntryBinding {
            value: b.value.clone(),
            operand: match &b.operand {
                Operand::Reg(r) => Operand::Reg(entry_reg(r)),
                o => o.clone(),
            },
        })
        .collect();
    let control = s.control_flow().map(|c| {
        remap_links(c, |row, link| {
            let i = &s.instrs()[row];
            let op = i.ops().iter().position(|o| o.reg() == Some(link))?;
            rows[row].ops()[op].reg().copied()
        })
    });
    let control = control.transpose()?;
    let virtual_snapshot = Snapshot::new(s.target(), infos, rows.into())
        .with_entry(entry)
        .and_then(|v| v.with_control_flow(control))
        .map_err(|e| error("virtual program", e.to_string()))?;
    let assignment = Assignment {
        bases,
        reasons,
        physical: s.values().to_vec(),
    };
    Ok((
        Program {
            snapshot: virtual_snapshot,
            entry: bindings,
            indirect: program.indirect.clone(),
            addresses: program.addresses.clone(),
            boundaries: program.boundaries.clone(),
        },
        assignment,
    ))
}

/// The same control flow with each link register mapped.
fn remap_links(
    c: &ControlFlow,
    mut map: impl FnMut(usize, &Reg) -> Option<Reg>,
) -> Result<ControlFlow, ProgramError> {
    let mut out = c.clone();
    for t in &mut out.transfers {
        let missing = || {
            error(
                "call graph",
                "a link register is not an operand of its transfer",
            )
        };
        t.kind = match t.kind {
            TransferKind::Call { link, continuation } => TransferKind::Call {
                link: map(t.instruction, &link).ok_or_else(missing)?,
                continuation,
            },
            TransferKind::Return { link } => TransferKind::Return {
                link: map(t.instruction, &link).ok_or_else(missing)?,
            },
            kind => kind,
        };
    }
    Ok(out)
}

/// A virtual program with every value placed at `bases` (as register
/// assignment's placement gives them): the physical program, with one
/// backing value per file.
pub fn place(program: &Program, bases: &[Option<u32>]) -> Result<Program, ProgramError> {
    let bases = (bases.iter().enumerate())
        .map(|(v, b)| b.ok_or_else(|| error("placement", format!("value {v} is not placed"))))
        .collect::<Result<Vec<u32>, _>>()?;
    let s = &program.snapshot;
    let physical = s
        .target()
        .definition()
        .files
        .iter()
        .map(|f| ValueInfo {
            file: f.letter,
            n: s.target().hooks().reg_count(f.letter),
            phys: Some(0),
            pin: Some(0),
            align: 1,
            ..ValueInfo::default()
        })
        .collect::<Vec<_>>();
    place_on(program, &physical, &bases)
}

fn place_on(
    program: &Program,
    physical: &[ValueInfo],
    bases: &[u32],
) -> Result<Program, ProgramError> {
    let s = &program.snapshot;
    if bases.len() != s.values().len() {
        return Err(error("placement", "one base per value"));
    }
    let backing = |file: char| physical.iter().position(|v| v.file == file);
    let map = |r: &Reg| -> Result<Reg, ProgramError> {
        let v = usize::try_from(r.value)
            .ok()
            .filter(|&v| v < bases.len())
            .ok_or_else(|| error("placement", "a register of no value"))?;
        let file = backing(r.file).ok_or_else(|| error("placement", "a file with no backing"))?;
        let at = bases[v] + r.lo;
        if at + r.n > physical[file].n {
            return Err(error(
                "placement",
                format!(
                    "{}{at}: past the file's {} registers",
                    r.file, physical[file].n
                ),
            ));
        }
        Ok(Reg::new(file as i32, r.file, at, r.n))
    };
    let rows = (s.instrs().iter())
        .map(|i| {
            if !i.ops().iter().any(|o| o.reg().is_some()) {
                return Ok(i.clone());
            }
            let ops = (i.ops().iter())
                .map(|o| match o {
                    Operand::Reg(r) => map(r).map(Operand::Reg),
                    o => Ok(o.clone()),
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(Arc::new(i.with_ops(ops)))
        })
        .collect::<Result<Vec<Arc<Instr>>, ProgramError>>()?;
    let entry = s.entry().iter().map(map).collect::<Result<Vec<_>, _>>()?;
    let bindings = (program.entry.iter())
        .map(|b| {
            Ok(EntryBinding {
                value: b.value.clone(),
                operand: match &b.operand {
                    Operand::Reg(r) => Operand::Reg(map(r)?),
                    o => o.clone(),
                },
            })
        })
        .collect::<Result<Vec<_>, ProgramError>>()?;
    let control = match s.control_flow() {
        Some(c) => Some(remap_links(c, |_, link| map(link).ok())?),
        None => None,
    };
    let snapshot = Snapshot::new(s.target(), physical.to_vec(), rows.into())
        .with_entry(entry)
        .and_then(|p| p.with_control_flow(control))
        .map_err(|e| error("placement", e.to_string()))?;
    Ok(Program {
        snapshot,
        entry: bindings,
        indirect: program.indirect.clone(),
        addresses: program.addresses.clone(),
        boundaries: program.boundaries.clone(),
    })
}
