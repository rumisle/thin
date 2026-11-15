//! gfx1201's hazard pass: the `s_wait_alu` counts of the
//! SGPR hazards, the VALU op of two hazards and the
//! delay hints. It has no wait-state rules.

use thin_ir::cfg::Cfg;
use thin_ir::solve::solve_forward_all;
use thin_ir::{FieldValue, Instr, Operand, Reg, Snapshot};
use thin_passes::hazards::HazardModel;
use thin_passes::{Need, Valu};
use thin_target::{Flags, Kind, WaitCounts, PHYS};

use super::{DEPCTR, DEPCTR_NONE};

/// gfx1201's hazard model.
pub struct Model;

impl HazardModel for Model {
    fn pass(&self, s: &Snapshot) -> Option<Vec<Need>> {
        Some(pass(s))
    }
}

/// An SGPR range in the scalar source space, and whether it is VCC.
#[derive(Clone, Copy, Debug)]
struct SReg {
    first: u32,
    n: u32,
    vcc: bool,
}

/// The scalar source-space number of a named register that counts here.
fn counted(p: usize) -> Option<SReg> {
    let name = PHYS[p].name;
    let (first, n) = match name {
        "vcc" => (106, 2),
        "vcc_lo" => (106, 1),
        "vcc_hi" => (107, 1),
        "ttmp7" => (115, 1),
        "ttmp9" => (117, 1),
        _ => return None,
    };
    Some(SReg {
        first,
        n,
        vcc: name.starts_with("vcc"),
    })
}

/// The SGPRs of registers and named registers.
fn sregs<'a>(s: &Snapshot, regs: impl Iterator<Item = &'a Reg>, named: &[usize]) -> Vec<SReg> {
    let numbered = regs.filter(|r| r.file == 's').map(|r| SReg {
        first: s.phys(r),
        n: r.n,
        vcc: false,
    });
    numbered
        .chain(named.iter().filter_map(|&p| counted(p)))
        .collect()
}

/// The units whose VCC write may be uncommitted.
const VALU: u8 = 1;
const SALU: u8 = 2;

/// Tracked pipeline dependence state.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Sgpr {
    tracked: u64,
    salu: u128,
    valu: u128,
    vcc: u8,
}

/// Waits required before issuing a dependent instruction.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Waits {
    va: bool,
    sa: bool,
    vcc: bool,
}

/// The bits of an SGPR range's numbers, capped below 128.
fn bits(r: SReg) -> u128 {
    (r.first..r.first + r.n)
        .filter(|&k| k < 128)
        .fold(0, |m, k| m | 1 << k)
}

/// The tracked-pair bit of a range (its first number's pair).
fn pair(r: SReg) -> u64 {
    1 << ((r.first >> 1) & 63)
}

/// An `s_wait_alu`'s packed value.
fn wait_alu_value(i: &Instr) -> u32 {
    if let Some(Operand::Imm(thin_ir::Imm {
        value: thin_ir::ImmValue::Int(v),
        ..
    })) = i.ops().first()
    {
        return *v as u32;
    }
    let mut value = DEPCTR_NONE;
    for &(name, shift, width) in &DEPCTR {
        if let Some(c) = i.field_named(name).and_then(FieldValue::int) {
            let mask = ((1u32 << width) - 1) << shift;
            value = (value & !mask) | ((c as u32) << shift & mask);
        }
    }
    value
}

/// Whether a dependency counter of an `s_wait_alu` value is 0.
fn zero(value: u32, field: &str) -> bool {
    let &(_, shift, width) = DEPCTR
        .iter()
        .find(|d| d.0 == field)
        .expect("a depctr field");
    value >> shift & ((1 << width) - 1) == 0
}

impl Sgpr {
    /// Commits the waits.
    fn commit(&mut self, w: Waits) {
        if w.vcc {
            self.vcc &= !VALU;
        }
        if w.sa {
            self.salu = 0;
            self.vcc &= !SALU;
        }
        if w.va {
            self.valu = 0;
        }
    }

    /// The waits a read of r by `unit` needs, tracking or committing.
    fn read(&mut self, r: SReg, unit: u8) -> Waits {
        let mut w = Waits::default();
        // A read of a pair no VALU has read commits nothing: conservative,
        // since any pending write then still gets its wait where it is read.
        if self.tracked & pair(r) == 0 {
            if unit == VALU {
                self.tracked |= pair(r);
            }
            return w;
        }
        // An SALU read commits the VALU writes in flight (VCC's when it reads
        // VCC). Only those: at a join VCC may also hold an SALU write from
        // another path, which still needs its sa_sdst wait below.
        if unit == SALU {
            if !r.vcc {
                self.valu = 0;
            } else {
                self.vcc &= !VALU;
            }
        }
        let b = bits(r);
        w.sa |= self.salu & b != 0;
        w.va |= unit == VALU && self.valu & b != 0;
        if r.vcc {
            w.sa |= self.vcc & SALU != 0;
            w.vcc |= self.vcc & VALU != 0;
        }
        w
    }

    /// One instruction's step; the waits it needs. `ds_nops` counts the
    /// consecutive `ds_nop`s of the block.
    fn step(&mut self, s: &Snapshot, i: &Instr, ds_nops: &mut u32) -> Waits {
        let f = i.form();
        if f.flags.has(Flags::PSEUDO) || matches!(f.kind, Kind::Label | Kind::Comment) {
            return Waits::default();
        }
        if f.op == "ds_nop" {
            *ds_nops += 1;
            if *ds_nops == 4 {
                self.tracked = 0;
            }
            return Waits::default();
        }
        *ds_nops = 0;
        // Memory instructions commit nothing: their
        // kinds are no unit below, so the state passes through.
        if f.op == "s_wait_alu" {
            let v = wait_alu_value(i);
            let w = Waits {
                vcc: zero(v, "depctr_va_vcc"),
                sa: zero(v, "depctr_sa_sdst"),
                va: zero(v, "depctr_va_sdst"),
            };
            self.commit(w);
            return Waits::default();
        }
        let unit = match f.kind {
            Kind::Valu | Kind::Matrix => VALU,
            Kind::Salu | Kind::Branch | Kind::Barrier | Kind::Wait | Kind::Nop => SALU,
            _ => return Waits::default(),
        };
        let call = matches!(f.op.as_str(), "s_setpc_b64" | "s_swappc_b64" | "s_call_b64");
        let w = if call {
            Waits {
                vcc: self.vcc & VALU != 0,
                sa: self.salu != 0 || self.vcc & SALU != 0,
                va: self.valu != 0,
            }
        } else {
            sregs(s, i.uses(), i.reads())
                .into_iter()
                .fold(Waits::default(), |a, r| {
                    let b = self.read(r, unit);
                    Waits {
                        va: a.va | b.va,
                        sa: a.sa | b.sa,
                        vcc: a.vcc | b.vcc,
                    }
                })
        };
        self.commit(w);
        if matches!(f.op.as_str(), "s_swappc_b64" | "s_call_b64") {
            self.tracked = u64::MAX;
        }
        for r in sregs(s, i.defs(), i.writes()) {
            if self.tracked & pair(r) == 0 {
                continue;
            }
            if r.vcc {
                self.vcc = unit;
            } else if unit == VALU {
                self.valu |= bits(r);
            } else {
                self.salu |= bits(r);
            }
        }
        w
    }

    fn join(a: &Sgpr, b: &Sgpr) -> Sgpr {
        Sgpr {
            tracked: a.tracked | b.tracked,
            salu: a.salu | b.salu,
            valu: a.valu | b.valu,
            vcc: a.vcc | b.vcc,
        }
    }
}

/// A VGPR range: (first, count).
type VRange = (u32, u32);

/// Tracked pipeline dependence state.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct Clear {
    cmpx: bool,
    wmma: Vec<VRange>,
}

/// Whether an op is a WMMA or an SWMMAC.
fn matrix(op: &str) -> bool {
    op.starts_with("v_wmma") || op.starts_with("v_swmmac")
}

/// Operand k's VGPR range.
fn vgprs(s: &Snapshot, i: &Instr, k: usize) -> Option<VRange> {
    i.ops()
        .get(k)?
        .reg()
        .filter(|r| r.file == 'v')
        .map(|r| (s.phys(r), r.n))
}

fn meets(a: VRange, b: VRange) -> bool {
    a.0 < b.0 + b.1 && b.0 < a.0 + a.1
}

impl Clear {
    /// Whether an instruction takes part: a VALU or matrix op, not an own op.
    fn takes_part(i: &Instr) -> bool {
        let f = i.form();
        matches!(f.kind, Kind::Valu | Kind::Matrix) && !f.flags.has(Flags::PSEUDO)
    }

    /// The VALU op instruction i needs before it.
    fn need(&self, s: &Snapshot, i: &Instr) -> Valu {
        let op = i.form().op.as_str();
        if op.starts_with("v_permlane") && self.cmpx {
            return Valu::Copy(1);
        }
        let sources: &[usize] = if op.starts_with("v_swmmac") {
            &[1, 2, 3]
        } else {
            &[1, 2]
        };
        let hit = matrix(op)
            && (sources.iter())
                .filter_map(|&k| vgprs(s, i, k))
                .any(|r| self.wmma.iter().any(|&w| meets(r, w)));
        if hit {
            Valu::Nop
        } else {
            Valu::None
        }
    }

    /// The placed op's effect, then i's.
    fn step(&mut self, s: &Snapshot, i: &Instr, placed: Valu) {
        match placed {
            Valu::None => {}
            Valu::Nop => self.wmma.clear(),
            Valu::Copy(_) => (self.cmpx, self.wmma) = (false, Vec::new()),
        }
        let op = i.form().op.as_str();
        if op.starts_with("v_cmpx") {
            self.cmpx = true;
        } else if op != "v_nop" {
            self.cmpx = false;
        }
        self.wmma = if matrix(op) {
            vgprs(s, i, 0).into_iter().collect()
        } else {
            Vec::new()
        };
    }

    fn join(a: &Clear, b: &Clear) -> Clear {
        let mut wmma: Vec<VRange> = a.wmma.iter().chain(&b.wmma).copied().collect();
        wmma.sort_unstable();
        wmma.dedup();
        Clear {
            cmpx: a.cmpx || b.cmpx,
            wmma,
        }
    }
}

/// A register range of: (is a VGPR, first, count).
type Range = (bool, u32, u32);

fn ranges<'a>(s: &Snapshot, regs: impl Iterator<Item = &'a Reg>, named: &[usize]) -> Vec<Range> {
    let regs: Vec<&Reg> = regs.collect();
    let v = regs
        .iter()
        .filter(|r| r.file == 'v')
        .map(|r| (true, s.phys(r), r.n));
    let sgprs = sregs(s, regs.iter().copied(), named)
        .into_iter()
        .map(|r| (false, r.first, r.n));
    v.chain(sgprs).collect()
}

fn overlaps(a: Range, b: Range) -> bool {
    a.0 == b.0 && a.1 < b.1 + b.2 && b.1 < a.1 + a.2
}

/// Whether an op is transcendental.
fn trans(op: &str) -> bool {
    [
        "v_exp_", "v_log_", "v_rcp_", "v_rsq_", "v_sqrt_", "v_sin_", "v_cos_",
    ]
    .iter()
    .any(|p| op.starts_with(p))
}

/// The delay hints of one block, given each instruction's VALU op:
/// in `out` each instruction's, in `copies` that of the copy placed before
/// it (a copy is a VALU op reading the register it copies).
fn delays(s: &Snapshot, g: &Cfg, j: usize, valus: &[Valu], out: &mut [u32], copies: &mut [u32]) {
    let instrs = s.instrs();
    let (mut valu, mut tr): (Vec<Vec<Range>>, Vec<Vec<Range>>) = (Vec::new(), Vec::new());
    let mut told = false;
    for k in g.block(j) {
        let i = &instrs[k];
        let f = i.form();
        if f.flags.has(Flags::PSEUDO) {
            continue;
        }
        if f.op == "s_delay_alu" {
            told = true;
            continue;
        }
        let writes = ranges(s, i.defs(), i.writes());
        if !matches!(f.kind, Kind::Valu | Kind::Matrix) {
            told = false;
            if f.op.starts_with("s_wait") && f.op != "s_wait_alu" {
                (valu, tr) = (Vec::new(), Vec::new());
            } else {
                for entry in valu.iter_mut().chain(tr.iter_mut()) {
                    entry.retain(|&r| !writes.iter().any(|&w| overlaps(r, w)));
                }
            }
            continue;
        }
        match valus[k] {
            Valu::None => {}
            Valu::Nop => valu.insert(0, Vec::new()),
            Valu::Copy(c) => {
                let copied: Vec<Range> = vgprs(s, i, c)
                    .map(|(a, n)| vec![(true, a, n)])
                    .unwrap_or_default();
                copies[k] = hint(&valu, &tr, &copied);
                for entry in valu.iter_mut().chain(tr.iter_mut()) {
                    entry.retain(|&r| !copied.iter().any(|&w| overlaps(r, w)));
                }
                valu.insert(0, copied);
            }
        }
        valu.truncate(4);
        let reads = ranges(s, i.uses(), i.reads());
        if !told {
            out[k] = hint(&valu, &tr, &reads);
        }
        told = false;
        // A later write ends the older producers' dependences on its ranges
        // a VALU's as any other.
        for entry in valu.iter_mut().chain(tr.iter_mut()) {
            entry.retain(|&r| !writes.iter().any(|&w| overlaps(r, w)));
        }
        let written = if matrix(&f.op) { Vec::new() } else { writes };
        if trans(&f.op) {
            tr.insert(0, written);
            tr.truncate(3);
            valu.insert(0, Vec::new());
        } else {
            valu.insert(0, written);
        }
        valu.truncate(4);
    }
}

/// The `s_delay_alu` operand for an instruction reading `reads`: the most
/// recent VALU and transcendental producers among the last writes, 0 for
/// none.
fn hint(valu: &[Vec<Range>], tr: &[Vec<Range>], reads: &[Range]) -> u32 {
    let first = |list: &[Vec<Range>]| {
        list.iter()
            .position(|e| e.iter().any(|&r| reads.iter().any(|&x| overlaps(r, x))))
            .map(|p| p as u32 + 1)
    };
    match (first(valu), first(tr)) {
        (Some(v), Some(t)) => v | (4 + t) << 7,
        (Some(v), None) => v,
        (None, Some(t)) => 4 + t,
        (None, None) => 0,
    }
}

/// gfx1201's own pass: one need per instruction.
pub fn pass(s: &Snapshot) -> Vec<Need> {
    let g = s.cfg(None);
    let instrs = s.instrs();
    let fields = s.target().wait_fields();
    let field = |name: &str| {
        fields
            .iter()
            .position(|f| *f == name)
            .expect("a wait field")
    };
    let n = instrs.len();
    let mut out: Vec<Need> = vec![Need::empty(fields.len()); n];
    let sgpr_through = |st: &Sgpr, j: usize| {
        let (mut st, mut ds) = (*st, 0);
        g.block(j).for_each(|k| _ = st.step(s, &instrs[k], &mut ds));
        st
    };
    let entry = solve_forward_all(
        &g,
        Sgpr::default(),
        vec![(0, Sgpr::default())],
        sgpr_through,
        Sgpr::join,
    );
    for (j, st) in entry.iter().enumerate() {
        let (mut st, mut ds) = (*st, 0);
        for k in g.block(j) {
            let w = st.step(s, &instrs[k], &mut ds);
            let mut waits = WaitCounts::empty(fields.len());
            for (on, name) in [
                (w.va, "depctr_va_sdst"),
                (w.sa, "depctr_sa_sdst"),
                (w.vcc, "depctr_va_vcc"),
            ] {
                if on {
                    waits.0[field(name)] = Some(0);
                }
            }
            out[k].waits = waits;
        }
    }
    let mut valus = vec![Valu::None; n];
    let clear_through = |st: &Clear, j: usize| {
        let mut st = st.clone();
        for k in g.block(j).filter(|&k| Clear::takes_part(&instrs[k])) {
            let need = st.need(s, &instrs[k]);
            st.step(s, &instrs[k], need);
        }
        st
    };
    let entry = solve_forward_all(
        &g,
        Clear::default(),
        vec![(0, Clear::default())],
        clear_through,
        Clear::join,
    );
    for (j, st) in entry.iter().enumerate() {
        let mut st = st.clone();
        for k in g.block(j).filter(|&k| Clear::takes_part(&instrs[k])) {
            valus[k] = st.need(s, &instrs[k]);
            st.step(s, &instrs[k], valus[k]);
        }
    }
    let (mut hints, mut copies) = (vec![0; n], vec![0; n]);
    (0..g.nblocks()).for_each(|j| delays(s, &g, j, &valus, &mut hints, &mut copies));
    for (k, need) in out.iter_mut().enumerate() {
        need.valu = valus[k];
        need.delay = hints[k];
        need.valu_delay = copies[k];
    }
    out
}
