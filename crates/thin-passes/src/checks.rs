//! The stream checks: dead writes, SCC readers, save-less
//! lane regions and the 64-bit shift rule. Each reads the final snapshot
//! and returns its first error.

use std::collections::BTreeMap;

use thin_ir::live::{self, Rows};
use thin_ir::solve::solve_forward;
use thin_ir::{Diag, DiagClass, Instr, Operand, Part, Reg, Snapshot};
use thin_target::{Flags, Kind, Pseudo};

/// A ScheduleError of message parts.
pub(crate) fn refusal(msg: Vec<Part>) -> Diag {
    Diag {
        class: DiagClass::Schedule,
        loc: 0,
        msg,
        notes: Vec::new(),
    }
}

/// `` `INSTR`WHERE `` of instruction k.
pub(crate) fn quoted(k: usize) -> Vec<Part> {
    vec![
        Part::Text("`".into()),
        Part::Instr(k),
        Part::Text("`".into()),
        Part::Where(k),
    ]
}

/// Whether EXEC is known to be the full mask on entry to k.
pub(crate) fn exec_full(s: &Snapshot, k: usize) -> bool {
    s.exec()[k] == Some(s.target().full_mask())
}

/// The value dwords a dead-write analysis keys by, numbered densely.
#[derive(Default)]
struct Keys {
    index: BTreeMap<(i32, u32), usize>,
}

impl Keys {
    fn of(&mut self, r: &Reg) -> Vec<usize> {
        (r.lo..r.lo + r.n)
            .map(|d| {
                let next = self.index.len();
                *self.index.entry((r.value, d)).or_insert(next)
            })
            .collect()
    }
}

/// What one instruction reads and writes of the V and A files.
#[derive(Default, Clone)]
struct Access {
    reads: Vec<usize>,
    writes: Vec<usize>,
}

/// The dead-write accesses of one instruction.
fn dead_access(s: &Snapshot, k: usize, keys: &mut Keys) -> Access {
    let i = &s.instrs()[k];
    let f = i.form();
    let vector = |r: &&Reg| matches!(r.file, 'v' | 'a');
    if f.pseudo == Some(Pseudo::Discard) {
        let reads = i
            .ops()
            .iter()
            .filter_map(Operand::reg)
            .filter(vector)
            .flat_map(|r| keys.of(r))
            .collect();
        return Access {
            reads,
            writes: Vec::new(),
        };
    }
    // A branch is an instruction like any other: whatever V or A
    // registers it reads or writes count (none on gfx1201 today).
    if f.kind == Kind::Label {
        return Access::default();
    }
    let defs: Vec<Reg> = i.defs().filter(vector).copied().collect();
    let printed: Vec<Reg> = if f.flags.has(Flags::PSEUDO) {
        Vec::new()
    } else {
        i.ops().iter().filter_map(Operand::reg).copied().collect()
    };
    let all_defs: Vec<Reg> = i.defs().copied().collect();
    let mut uses: Vec<Reg> = i.uses().copied().collect();
    uses.extend(printed.iter().filter(|r| !all_defs.contains(r)));
    uses.extend(
        all_defs
            .iter()
            .filter(|d| printed.iter().filter(|r| r == d).count() > 1),
    );
    uses.extend(
        i.attrs()
            .wait_uses
            .iter()
            .flat_map(|(_, regs)| regs.iter().copied()),
    );
    let mut reads: Vec<usize> = uses
        .iter()
        .filter(vector)
        .flat_map(|r| keys.of(r))
        .collect();
    let writes: Vec<usize> = defs.iter().flat_map(|r| keys.of(r)).collect();
    if i.keeps_old(!exec_full(s, k)) {
        reads.extend(&writes);
    }
    Access { reads, writes }
}

/// The text a dead key sorts by: `(%F<value id>:<name>, <dword>)`.
fn key_text(s: &Snapshot, (value, dword): (i32, u32)) -> String {
    let v = &s.values()[value as usize];
    format!("(%{}{value}:{}, {dword})", v.file, s.name(v.name))
}

/// Dead writes: a V or A write that every path writes again before
/// a read.
pub fn dead_writes(s: &Snapshot) -> Result<(), Diag> {
    dead_findings(s, true)
        .into_iter()
        .next()
        .map_or(Ok(()), |finding| Err(finding.diagnostic))
}

/// All dead-write instruction findings, one per instruction rather than
/// one per dword. Ordering and diagnostics match the existing first-error
/// check, allowing before/after CFG comparisons without masking later sites.
pub fn dead_writes_all(s: &Snapshot) -> Vec<Diag> {
    dead_findings(s, false)
        .into_iter()
        .map(|finding| finding.diagnostic)
        .collect()
}

/// One instruction's overwritten vector dwords, not an instruction-removal proof.
#[derive(Clone, Debug)]
pub struct DeadWriteFinding {
    /// The instruction defining these dwords.
    pub instruction: usize,
    /// Exactly the written dwords dead on every represented path, each n = 1.
    pub dead_dwords: Vec<Reg>,
    /// Other written vector dwords not proved dead, including exit live-outs.
    pub live_dwords: Vec<Reg>,
    /// The lint; partial results must not suggest dropping the whole instruction.
    pub diagnostic: Diag,
}

/// All dead-write findings with per-dword precision. Ordering is the same
/// as dead_writes_all; a partial finding retains its live written dwords.
pub fn dead_write_findings(s: &Snapshot) -> Vec<DeadWriteFinding> {
    dead_findings(s, false)
}

fn dead_findings(s: &Snapshot, first_only: bool) -> Vec<DeadWriteFinding> {
    let mut findings = Vec::new();
    let g = s.cfg(None);
    let mut keys = Keys::default();
    let acc: Vec<Access> = (0..s.instrs().len())
        .map(|k| dead_access(s, k, &mut keys))
        .collect();
    // A write is dead when no valid path reads it, a return going back
    // only to its own call's continuation (`thin_ir::control::Flow`);
    // every key is live at an exit.
    let mut rows = Rows::new(
        g.succ.clone(),
        thin_ir::control::flows(s.control_flow(), &g.succ),
    );
    for (k, a) in acc.iter().enumerate() {
        rows.every[k].clone_from(&a.reads);
        rows.kills[k].clone_from(&a.writes);
    }
    rows.exit = (0..keys.index.len()).collect();
    let live = live::solve(&rows);
    let names: BTreeMap<usize, (i32, u32)> = keys.index.iter().map(|(&k, &v)| (v, k)).collect();
    for j in 0..g.nblocks() {
        for k in g.block(j).rev() {
            let live = live.live_out(k);
            let dead: Vec<usize> = acc[k]
                .writes
                .iter()
                .copied()
                .filter(|&w| !live.has(w))
                .collect();
            if let Some(&key) = dead.iter().min_by_key(|&&w| key_text(s, names[&w])) {
                let mut written = acc[k].writes.clone();
                written.sort_unstable();
                written.dedup();
                let live_written: Vec<_> = written.into_iter().filter(|w| live.has(*w)).collect();
                let mut dead = dead;
                dead.sort_unstable();
                dead.dedup();
                let regs = |keys: &[usize]| {
                    keys.iter()
                        .map(|key| {
                            let (value, lo) = names[key];
                            Reg::new(value, s.values()[value as usize].file, lo, 1)
                        })
                        .collect()
                };
                let diagnostic = if live_written.is_empty() {
                    dead_write(&acc, k, key)
                } else {
                    let mut msg = quoted(k);
                    let words = dead
                        .iter()
                        .map(|key| key_text(s, names[key]))
                        .collect::<Vec<_>>()
                        .join(", ");
                    msg.push(Part::Text(format!(": destination dword(s) {words} are written over on every path before a read (partial dead write): other destination dwords remain live; preserve those results, do not drop the whole instruction")));
                    refusal(msg)
                };
                findings.push(DeadWriteFinding {
                    instruction: k,
                    dead_dwords: regs(&dead),
                    live_dwords: regs(&live_written),
                    diagnostic,
                });
                if first_only {
                    return findings;
                }
            }
        }
    }
    findings
}

/// The dead-write refusal at instruction k for a key.
fn dead_write(acc: &[Access], k: usize, key: usize) -> Diag {
    let mut msg = quoted(k);
    msg.push(Part::Text(
        ": its value is written over on every path before a read".into(),
    ));
    if let Some(next) = (k + 1..acc.len()).find(|&q| acc[q].writes.contains(&key)) {
        msg.push(Part::Text(" (".into()));
        msg.extend(quoted(next));
        msg.push(Part::Text(" writes it again)".into()));
    }
    msg.push(Part::Text(
        " (dead write): drop the write, or b.discard(reg, why) where it is not read; an ablation \
         variant that switched the reader off - b.ablate"
            .into(),
    ));
    refusal(msg)
}

/// What SCC holds: (kind, the op that wrote it, carries).
type Scc = (&'static str, String, bool);

/// SCC after instruction i.
fn scc_step(i: &Instr, st: &Scc) -> Scc {
    let f = i.form();
    if !matches!(f.kind, Kind::Salu | Kind::Branch) {
        return st.clone();
    }
    // The ops whose SCC is an unsigned carry or borrow (RDNA4 ISA guide,
    // scalar arithmetic instructions): the u32 adds and subtracts, with and
    // without a carry in, and S_LSHL{1,2,3,4}_ADD_U32. A signed add's SCC (s_add_co_i32,
    // s_addk_co_i32) is an overflow, not a carry.
    const CARRIES: [&str; 8] = [
        "s.add.u32",
        "s.addc.u32",
        "s.sub.u32",
        "s.subb.u32",
        "s.lshl1.add.u32",
        "s.lshl2.add.u32",
        "s.lshl3.add.u32",
        "s.lshl4.add.u32",
    ];
    let carries = CARRIES.contains(&f.canon.as_str());
    let scc = thin_target::phys_named("scc").expect("scc").number;
    if f.flags.has(Flags::SCC_CMP) {
        ("cmp", f.op.clone(), carries)
    } else if f.flags.has(Flags::PSEUDO) {
        ("unknown", f.op.clone(), carries)
    } else if i.writes().contains(&scc) {
        ("arith", f.op.clone(), carries)
    } else {
        st.clone()
    }
}

/// A reader of SCC subject to the compiler's authoring-policy checks.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum SccReader {
    /// A select expected to consume a comparison predicate.
    Select,
    /// A carry-in addition/subtraction expected to continue arithmetic.
    Carry,
}

/// One SCC policy finding, including its source instruction and last writer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SccFinding {
    /// Index of the reader in the snapshot.
    pub instruction: usize,
    /// The policy this reader violates.
    pub kind: SccReader,
    /// The last writer's opcode, or a join/entry description.
    pub writer: String,
    /// The existing diagnostic, unchanged from the first-error check.
    pub diagnostic: Diag,
}

/// SCC readers: a select reads a compare, a carry chain an add or
/// a sub.
pub fn scc_findings(s: &Snapshot) -> Vec<SccFinding> {
    if s.instrs().is_empty() {
        return Vec::new();
    }
    let mut findings = Vec::new();
    let g = s.cfg(None);
    let instrs = s.instrs();
    let through = |st: &Scc, j: usize| {
        g.block(j)
            .fold(st.clone(), |st, k| scc_step(&instrs[k], &st))
    };
    let merge = |a: &Scc, b: &Scc| {
        if a == b {
            a.clone()
        } else {
            ("unknown", "the paths into it".into(), false)
        }
    };
    let entry: Scc = ("unknown", "nothing".into(), false);
    let states = solve_forward(&g, vec![(0, entry)], through, merge, None);
    for (j, st) in states.iter().enumerate() {
        let Some(mut st) = st.clone() else { continue };
        for k in g.block(j) {
            let flags = instrs[k].form().flags;
            let mut msg = quoted(k);
            if flags.has(Flags::SCC_SELECT) && st.0 != "cmp" {
                msg.push(Part::Text(format!(
                    ": SCC was last written by {} ({}), not a compare",
                    st.1, st.0
                )));
                findings.push(SccFinding {
                    instruction: k,
                    kind: SccReader::Select,
                    writer: st.1.clone(),
                    diagnostic: refusal(msg.clone()),
                });
            }
            if !flags.has(Flags::SCC_SELECT)
                && flags.has(Flags::SCC_CARRY_IN)
                && !(st.0 == "arith" && st.2)
            {
                msg.push(Part::Text(format!(
                    ": carry chain needs an add / sub right before (SCC from {})",
                    st.1
                )));
                findings.push(SccFinding {
                    instruction: k,
                    kind: SccReader::Carry,
                    writer: st.1.clone(),
                    diagnostic: refusal(msg),
                });
            }
            st = scc_step(&instrs[k], &st);
        }
    }
    findings
}

/// SCC readers: the first policy failure in layout order.
/// The full findings are available without disabling either policy.
pub fn scc_readers(s: &Snapshot) -> Result<(), Diag> {
    scc_findings(s)
        .into_iter()
        .next()
        .map_or(Ok(()), |f| Err(f.diagnostic))
}

/// Save-less lane regions: EXEC is the full mask on entry to each.
pub fn saveless(s: &Snapshot, entries: &[usize]) -> Result<(), Diag> {
    for &k in entries {
        if exec_full(s, k) {
            continue;
        }
        let how = s.exec()[k].map_or("not one constant on every path".into(), |m| {
            format!("{m:#x}")
        });
        let mut msg = vec![Part::Text("`".into()), Part::Instr(k)];
        msg.push(Part::Text(format!(
            "` (instruction {k}) enters a lane region without a save register where exec is {how}: every lane \
             must be on (else pass save=)"
        )));
        return Err(refusal(msg));
    }
    Ok(())
}

/// Reads of SCC with no defined producer on every represented CFG path.
/// This is the ISA-level physical-program check, not the DSL policies that
/// require select-after-compare or carry-after-arithmetic. RDNA4 ISA guide
/// logic writes a nonzero predicate, comparisons write a predicate,
/// and ADD_CO_CI consumes the SCC bit regardless of which instruction set it.
/// Indirect-call edges must be resolved before claiming whole-program proof.
pub fn physical_scc(s: &Snapshot) -> Vec<Diag> {
    if s.instrs().is_empty() {
        return Vec::new();
    }
    let g = s.cfg(None);
    let scc = thin_target::phys_named("scc").expect("scc").number;
    let instrs = s.instrs();
    let step = |known: bool, i: &Instr| {
        if i.writes().contains(&scc) {
            !i.reads().contains(&scc) || known
        } else {
            known
        }
    };
    let through = |known: &bool, block: usize| {
        g.block(block)
            .fold(*known, |known, k| step(known, &instrs[k]))
    };
    let states = solve_forward(&g, vec![(0, false)], through, |a, b| *a && *b, None);
    let mut out = Vec::new();
    for (block, state) in states.iter().enumerate() {
        let Some(mut known) = *state else { continue };
        for k in g.block(block) {
            if instrs[k].reads().contains(&scc) && !known {
                let mut msg = quoted(k);
                msg.push(Part::Text(
                    ": SCC has no defined producer on every represented CFG path".into(),
                ));
                out.push(refusal(msg));
            }
            known = step(known, &instrs[k]);
        }
    }
    out
}
