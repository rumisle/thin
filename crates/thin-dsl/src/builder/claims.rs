//! The claims.

use thin_ir::{FieldId, FieldValue, Instr, Operand};
use thin_target::{Flags, Kind, Pseudo};

use super::{field, Builder};
use crate::error::{schedule, value, Result};

/// A claim's register: a value's, or a named register.
#[derive(Clone, Debug)]
pub enum ClaimReg {
    /// A register.
    Reg(thin_ir::Reg),
    /// A named register.
    Phys(&'static thin_target::Phys),
}

impl ClaimReg {
    fn operand(&self) -> Operand {
        match self {
            ClaimReg::Reg(r) => Operand::Reg(*r),
            ClaimReg::Phys(p) => Operand::Phys(p),
        }
    }

    /// Its file: a register's; a named register is none of v, s and a, so
    /// the claims that bound or poison a value's register refuse it.
    fn file(&self) -> &'static str {
        match self {
            ClaimReg::Reg(r) => match r.file {
                'v' => "v",
                's' => "s",
                _ => "a",
            },
            ClaimReg::Phys(_) => "named",
        }
    }
}

impl Builder {
    /// Pushes a claim where it is written: with the operation's location,
    /// its lane region and the held-register check, as an instruction gets;
    /// not the pending first-use waits, since a claim reads nothing.
    fn push_claim(&mut self, i: Instr) -> Result<u64> {
        self.check_held(&i)?;
        let i = crate::lanes::check(self, i)?;
        let loc = self.loc;
        Ok(self.push(i, loc))
    }

    fn claim_text(&self, regs: &[ClaimReg]) -> String {
        let t: Vec<String> = (regs.iter())
            .map(|r| match r {
                ClaimReg::Reg(r) => self.repr(r),
                ClaimReg::Phys(p) => p.name.to_owned(),
            })
            .collect();
        t.join(", ")
    }

    /// `b.discard(reg, why)`: emits `ir.discard reg {why}`.
    pub fn discard(&mut self, reg: ClaimReg, why: &str) -> Result<()> {
        if why.is_empty() {
            return value("discard needs the reason (why)");
        }
        let i = Instr::new(
            self.own(Pseudo::Discard),
            vec![reg.operand()],
            vec![field("why", why.into())?],
            None,
        );
        self.emit(i, &[]).map(|_| ())
    }

    /// `b.uniform(*regs, why)`: pushes `ir.uniform`.
    pub fn uniform(&mut self, regs: &[ClaimReg], why: &str) -> Result<()> {
        if why.is_empty() {
            return schedule("uniform(): say why (why=)");
        }
        self.check_open()?;
        self.settle()?;
        let ops = regs.iter().map(ClaimReg::operand).collect();
        let i = Instr::new(
            self.own(Pseudo::Uniform),
            ops,
            vec![field("why", why.into())?],
            None,
        );
        self.push_claim(i)?;
        Ok(())
    }

    /// `b.undefined(*regs, why, edge=None)`: pushes `ir.undefined` and nothing
    /// else; its `on` is set when the Builder settles.
    pub fn undefined(&mut self, regs: &[ClaimReg], why: &str, edge: Option<&str>) -> Result<()> {
        if why.is_empty() {
            return schedule("undefined(): say why (why=)");
        }
        let taken = match edge {
            None => false,
            Some("taken") => true,
            Some(_) => return schedule("undefined(): edge is None or 'taken'"),
        };
        self.check_open()?;
        self.settle()?;
        if regs.iter().any(|r| !matches!(r.file(), "v" | "s" | "a")) {
            return schedule(format!(
                "undefined({}): registers of the v / s / a files",
                self.claim_text(regs)
            ));
        }
        let ops = regs.iter().map(ClaimReg::operand).collect();
        let i = Instr::new(
            self.own(Pseudo::Undefined),
            ops,
            vec![field("why", why.into())?],
            None,
        );
        let id = self.push_claim(i)?;
        let text = self.claim_text(regs);
        self.claims_open.push((id, taken, text));
        Ok(())
    }

    /// Sets each open undefined claim's `on` from what follows it, own ops
    /// skipped: a label makes it `fall`; `edge="taken"` needs a branch.
    /// At the seal (`last`), a claim followed by nothing stays `next`.
    pub(crate) fn place_claims(&mut self, last: bool) -> Result<()> {
        let open = std::mem::take(&mut self.claims_open);
        for (id, taken, text) in open {
            let Some(k) = self.row_index(id) else {
                continue;
            };
            let next = (k + 1..self.rows.len()).find(|&j| {
                let f = self.rows[j].ins.form();
                f.kind == Kind::Label || !f.flags.has(Flags::PSEUDO)
            });
            let on = match next.map(|j| self.rows[j].ins.form().kind) {
                None if !last => {
                    self.claims_open.push((id, taken, text));
                    continue;
                }
                Some(Kind::Branch) if taken => Some("taken"),
                None | Some(_) if taken => {
                    return schedule(format!(
                        "undefined({text}, edge='taken'): the next instruction is not a branch"
                    ));
                }
                Some(Kind::Label) => Some("fall"),
                _ => None,
            };
            if let Some(on) = on {
                let i = &self.rows[k].ins;
                let mut fields: Vec<(thin_ir::FieldId, FieldValue)> =
                    i.fields().iter().map(|(f, v)| (*f, v.clone())).collect();
                fields.push(field("on", on.to_owned())?);
                self.rows[k].ins = std::sync::Arc::new(i.with_fields(fields));
                self.touched(k);
            }
        }
        Ok(())
    }

    /// `b.assume(x, lo=, hi=, why=)`: pushes `ir.assume r {lo, hi, why}`;
    /// `written` is x as the author wrote it.
    pub fn assume(
        &mut self,
        reg: ClaimReg,
        written: &str,
        lo: i128,
        hi: i128,
        why: &str,
    ) -> Result<()> {
        if why.is_empty() {
            return schedule("assume(): say why (why=)");
        }
        if !(0 <= lo && lo <= hi && hi < 1 << 32) {
            return schedule(format!(
                "assume({written}, lo={lo}, hi={hi}): 0 <= lo <= hi < 2^32 (u32 values)"
            ));
        }
        self.check_open()?;
        self.settle()?;
        if !matches!(reg.file(), "v" | "s" | "a") {
            return schedule(format!(
                "assume({}): a register of the v / s / a files",
                self.claim_text(std::slice::from_ref(&reg))
            ));
        }
        let int = |name: &str, v: i128| {
            Ok::<_, crate::error::Error>((FieldId::named(name)?, FieldValue::Int(v)))
        };
        let fields = vec![int("lo", lo)?, int("hi", hi)?, field("why", why.into())?];
        let i = Instr::new(self.own(Pseudo::Assume), vec![reg.operand()], fields, None);
        self.push_claim(i)?;
        Ok(())
    }
}
