//! Hazards: what each instruction needs before it, putting
//! the needs into the stream, and the clear check.

mod rules;

use std::sync::Arc;

use thin_ir::{Attr, Diag, DiagClass, FieldId, FieldValue, Imm, Instr, Operand, Part, Snapshot};
use thin_target::{Target, WaitCounts};

pub use rules::{Dep, DynamicStates, Engine, Rule, Side, States, Test};

use crate::need::{delay_text, Need, Valu};

/// A target's hazard model: wait-state rules and an own
/// pass. The passes define it; a target instance implements it.
pub trait HazardModel: Sync {
    /// The wait-state rules.
    fn rules(&self) -> &[Rule] {
        &[]
    }

    /// The target's own pass: one need per instruction, states 0;
    /// none for a target without one.
    fn pass(&self, s: &Snapshot) -> Option<Vec<Need>> {
        let _ = s;
        None
    }
}

/// The number of wait fields of a target.
pub fn fields(t: &Target) -> usize {
    t.wait_fields().len()
}

/// Per instruction, its need: the wait states of joined with the
/// target pass's need.
pub fn needs(s: &Snapshot, model: &dyn HazardModel) -> Vec<Need> {
    let n = fields(s.target());
    let states = Engine::new(s, model.rules()).wait_state_needs();
    let mut out: Vec<Need> = states
        .into_iter()
        .map(|states| Need {
            states,
            ..Need::empty(n)
        })
        .collect();
    if let Some(pass) = model.pass(s) {
        out.iter_mut().zip(&pass).for_each(|(a, b)| *a = a.join(b));
    }
    out
}

/// The stream without the instructions the hazard pass placed.
pub fn strip(instrs: &[Arc<Instr>]) -> Vec<Arc<Instr>> {
    instrs
        .iter()
        .filter(|i| !i.attrs().hazard)
        .cloned()
        .collect()
}

/// An instruction the hazard pass places, marked and commented.
fn placed(i: Instr, comment: String) -> Arc<Instr> {
    let i = i.with_attr(Attr::Hazard).expect("Hazard is put anywhere");
    Arc::new(
        i.with_attr(Attr::Comment(comment.into()))
            .expect("Comment is put anywhere"),
    )
}

/// The form `mnemonic` with `arity` operands, which every target has
///
fn form(t: &'static Target, mnemonic: &str, arity: usize) -> &'static thin_target::Form {
    (t.form(mnemonic, arity, false))
        .unwrap_or_else(|| panic!("{}: no {mnemonic}/{arity} ", t.name()))
}

/// The wait instructions of a count vector: one per target wait instruction
/// with a count among them, in the target's order.
pub fn wait_instrs(t: &'static Target, waits: &WaitCounts) -> Vec<Instr> {
    let names = t.wait_fields();
    let mut out = Vec::new();
    for w in &t.definition().waits {
        let fields: Vec<(FieldId, FieldValue)> = (w.fields.iter())
            .filter_map(|f| {
                let k = names.iter().position(|n| n == f)?;
                let v = waits.0[k]?;
                Some((
                    FieldId::named(f).expect("a count field"),
                    FieldValue::Int(v.into()),
                ))
            })
            .collect();
        if !fields.is_empty() {
            out.push(Instr::new(form(t, w.mnemonic, 0), Vec::new(), fields, None));
        }
    }
    out
}

/// The instructions that put a need before instruction `i`.
pub fn fill(t: &'static Target, i: &Instr, need: &Need) -> Vec<Arc<Instr>> {
    let mut out = Vec::new();
    let mark = |i: Instr| placed(i, "hazard".into());
    if need.states > 0 {
        let nop = form(t, "s_nop", 1);
        let mut rem = need.states;
        while rem > 0 {
            let chunk = rem.min(16);
            out.push(mark(Instr::new(
                nop,
                vec![Operand::Imm(Imm::int(chunk as i128 - 1))],
                Vec::new(),
                None,
            )));
            rem -= chunk;
        }
    }
    out.extend(wait_instrs(t, &need.waits).into_iter().map(mark));
    let spelled = |op: &str| {
        t.spell(op)
            .unwrap_or_else(|| panic!("{}: no {op} ", t.name()))
    };
    match need.valu {
        Valu::None => {}
        Valu::Nop => out.push(mark(Instr::new(
            form(t, &spelled("v.nop"), 0),
            Vec::new(),
            Vec::new(),
            None,
        ))),
        Valu::Copy(j) => {
            let r = i.ops()[j].clone();
            if need.valu_delay != 0 {
                out.push(delay_instr(t, need.valu_delay));
            }
            out.push(mark(Instr::new(
                form(t, &spelled("v.mov.b32"), 2),
                vec![r.clone(), r],
                Vec::new(),
                None,
            )));
        }
    }
    if need.delay != 0 {
        out.push(delay_instr(t, need.delay));
    }
    out
}

/// A placed delay hint of that operand, its comment the hint's text.
fn delay_instr(t: &'static Target, delay: u32) -> Arc<Instr> {
    let hint = t.definition().hazards.delay_hint;
    let form = form(
        t,
        hint.expect("a delay need on a target with a delay hint"),
        1,
    );
    let op = vec![Operand::Imm(Imm::int(delay.into()))];
    placed(
        Instr::new(form, op, Vec::new(), None),
        format!("hazard: {}", delay_text(delay)),
    )
}

/// The stream with each need put before its instruction: every
/// instruction, with for each placed one the index of the instruction it
/// was placed for. The stream must hold no hazard instruction (`strip`).
pub fn apply(s: &Snapshot, needs: &[Need]) -> Vec<(Arc<Instr>, Option<usize>)> {
    let mut out = Vec::new();
    for (k, (i, need)) in s.instrs().iter().zip(needs).enumerate() {
        out.extend(fill(s.target(), i, need).into_iter().map(|p| (p, Some(k))));
        out.push((i.clone(), None));
    }
    out
}

/// The text of a need in the clear check's message.
fn need_text(t: &Target, need: &Need) -> String {
    let names = t.wait_fields();
    let waits: Vec<String> = (need.waits.0.iter().zip(&names))
        .filter_map(|(c, f)| c.map(|c| format!("{f}({c})")))
        .collect();
    let valu = match need.valu {
        Valu::None => "none".to_owned(),
        Valu::Nop => "v_nop".to_owned(),
        Valu::Copy(j) => format!("copy of operand {j}"),
    };
    format!(
        "{} wait states, waits [{}], VALU op {valu} (delay {}), delay {}",
        need.states,
        waits.join(" "),
        need.valu_delay,
        need.delay
    )
}

/// The clear check: an internal error at the first instruction of
/// the final stream that still needs something.
pub fn clear(s: &Snapshot, model: &dyn HazardModel) -> Result<(), Diag> {
    let Some((k, need)) = needs(s, model)
        .into_iter()
        .enumerate()
        .find(|(_, n)| !n.is_empty())
    else {
        return Ok(());
    };
    let text = format!(
        ": hazards not clear after placement: {}",
        need_text(s.target(), &need)
    );
    Err(Diag {
        class: DiagClass::Schedule,
        loc: 0,
        msg: vec![
            Part::Text("internal error: `".into()),
            Part::Instr(k),
            Part::Text(format!("`{text}")),
        ],
        notes: Vec::new(),
    })
}

/// Whether instruction k provides something a later instruction needs: the
/// needs with k muted exceed the needs at some j > k.
pub fn gave(s: &Snapshot, model: &dyn HazardModel, k: usize) -> bool {
    let base = needs(s, model);
    let muted = needs(&s.muted(k), model);
    (k + 1..base.len()).any(|j| muted[j].exceeds(&base[j]))
}
