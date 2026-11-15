//! EXEC's constant state on entry to each instruction.

use thin_target::{phys_named, Flags, Kind};

use crate::cfg::Cfg;
use crate::solve::{per_instr, solve_forward};
use crate::{Imm, ImmValue, Instr, Operand, Snapshot};

/// A saved copy of a known EXEC: the SGPRs starting at `.0` (one on a
/// 32-lane target, a pair on a 64-lane one) hold (lo, hi).
type Saved = (u32, u32, u32);

/// EXEC's halves where every path agrees, and the saved copies of it.
#[derive(Clone, Debug, PartialEq, Eq)]
struct State {
    lo: Option<u32>,
    hi: Option<u32>,
    saved: Vec<Saved>,
}

/// The facts of the target the analysis reads.
struct Facts {
    wave: u32,
    /// The SGPRs a saved copy takes: EXEC's mask width in dwords.
    copy: u32,
    mask: &'static str,
    movm: String,
}

/// Per instruction, EXEC on entry as a 64-bit mask where every path
/// agrees on a constant, else none; over the snapshot's CFG `g`.
pub fn exec(s: &Snapshot, g: &Cfg) -> Vec<Option<u64>> {
    let t = s.target();
    let wave = t.definition().wave;
    let exec = phys_named("exec").expect("exec");
    let facts = Facts {
        wave,
        copy: wave / 32,
        mask: t.mask_reg(exec).name,
        movm: t.masked("s.mov.mask"),
    };
    let entry = State {
        lo: Some(u32::MAX),
        hi: Some(if wave == 64 { u32::MAX } else { 0 }),
        saved: Vec::new(),
    };
    let instrs = s.instrs();
    let through = |st: &State, j: usize| {
        let mut st = st.clone();
        g.block(j)
            .for_each(|k| step(s, &facts, &instrs[k], &mut st));
        st
    };
    let entries = if g.nblocks() == 0 {
        Vec::new()
    } else {
        solve_forward(g, vec![(0, entry)], through, merge, None)
    };
    let result = |st: &State| Some(st.lo? as u64 | (st.hi? as u64) << 32);
    per_instr(
        g,
        &entries,
        |st, k| step(s, &facts, &instrs[k], st),
        result,
        None,
    )
}

/// Halves kept where equal; the copies both states hold, in a's order.
fn merge(a: &State, b: &State) -> State {
    State {
        lo: a.lo.filter(|_| a.lo == b.lo),
        hi: a.hi.filter(|_| a.hi == b.hi),
        saved: a
            .saved
            .iter()
            .filter(|t| b.saved.contains(t))
            .copied()
            .collect(),
    }
}

/// The value an integer operand gives a `bits`-bit mask. A 32-bit mask
/// takes any 32-bit pattern, signed or not; a 64-bit one takes a constant
/// sign-extended from 32 bits, so a value in 2^31.. 2^32 (whose
/// extension the operand alone does not say) is unknown.
fn literal(op: Option<&Operand>, bits: u32) -> Option<u64> {
    let Some(Operand::Imm(Imm {
        value: ImmValue::Int(v),
        ..
    })) = op
    else {
        return None;
    };
    let v = i64::try_from(*v).ok()?;
    match bits {
        32 if (-(1 << 31)..1 << 32).contains(&v) => Some(u64::from(v as u32)),
        64 if (-(1 << 31)..1 << 31).contains(&v) => Some(v as u64),
        _ => None,
    }
}

/// The physical number of an SGPR operand.
fn sgpr(s: &Snapshot, op: Option<&Operand>) -> Option<u32> {
    let r = op?.reg().filter(|r| r.file == 's')?;
    Some(s.phys(r))
}

/// Records a copy, replacing any copy at its register, keeping the set
/// sorted (so a register holds one copy and a lookup finds that one).
fn save(st: &mut State, copy: Saved) {
    st.saved.retain(|t| t.0 != copy.0);
    if let Err(at) = st.saved.binary_search(&copy) {
        st.saved.insert(at, copy);
    }
}

/// One instruction's effect on the state.
fn step(s: &Snapshot, f: &Facts, i: &Instr, st: &mut State) {
    let form = i.form();
    if matches!(form.kind, Kind::Label | Kind::Comment) {
        return;
    }
    for r in i.defs().filter(|r| r.file == 's') {
        let a = s.phys(r);
        st.saved
            .retain(|&(p, _, _)| p + f.copy <= a || a + r.n <= p);
    }
    let ops = i.ops();
    let known = st.lo.zip(st.hi);
    if !i.writes_exec() {
        let is_save = form.canon == f.movm
            && ops.len() == 2
            && ops[1].phys().is_some_and(|p| p.name == f.mask);
        if let (true, Some((lo, hi)), Some(p)) = (is_save, known, sgpr(s, ops.first())) {
            save(st, (p, lo, hi));
        }
        return;
    }
    if let (true, Some((lo, hi)), Some(p)) =
        (form.flags.has(Flags::SAVEEXEC), known, sgpr(s, ops.first()))
    {
        save(st, (p, lo, hi));
    }
    let to = ops.first().and_then(Operand::phys).map(|p| p.name);
    let src = ops.get(1);
    let mov32 = form.canon == "s.mov.b32";
    (st.lo, st.hi) = match to {
        Some(m) if m == f.mask && form.canon == f.movm => match literal(src, f.wave) {
            Some(v) => (Some(v as u32), Some((v >> 32) as u32)),
            None => (sgpr(s, src))
                .and_then(|p| st.saved.iter().find(|t| t.0 == p))
                .map_or((None, None), |&(_, lo, hi)| (Some(lo), Some(hi))),
        },
        Some("exec_lo") if mov32 => (literal(src, 32).map(|v| v as u32), st.hi),
        Some("exec_hi") if mov32 => (st.lo, literal(src, 32).map(|v| v as u32)),
        Some("exec_lo") => (None, st.hi),
        Some("exec_hi") => (st.lo, None),
        _ => (None, None),
    };
}
