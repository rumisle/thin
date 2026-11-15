//! Conditions.

use thin_target::phys_named;

use super::lower::Opd;
use super::node::{const_of, value_of, Kind, Node, Num, Op, Tree, Val};
use super::text::top;
use super::vop::Vop;
use crate::builder::Builder;
use crate::error::{expr, Result};
use crate::types::Ty;
use crate::values::Typed;

/// Where a condition is.
#[derive(Clone, Debug, PartialEq)]
pub enum Where {
    /// In SCC.
    Scc,
    /// In VCC.
    Vcc,
    /// In a mask value.
    Mask(Typed),
    /// In a named register.
    Phys(&'static thin_target::Phys),
}

/// What `__asm_cond__` is asked for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Into {
    /// SCC.
    Scc,
    /// A fresh mask.
    Mask,
    /// Lanes: a mask type stays a mask.
    Lanes,
}

/// A junction's kind and parts, for a branch.
pub fn junction(n: &Tree) -> Option<(&'static str, [Tree; 2])> {
    match n.op {
        Op::Cand if n.file == Some('s') => Some(("and", [n.args[0].clone(), n.args[1].clone()])),
        Op::Cor if n.file == Some('s') => Some(("or", [n.args[0].clone(), n.args[1].clone()])),
        _ => None,
    }
}

impl Builder {
    fn vcc(&self) -> Opd {
        Opd::Phys(self.t.mask_reg(phys_named("vcc").expect("vcc")))
    }

    /// `cond.__asm_cond__(b, comment, into)`.
    pub fn asm_cond(&mut self, n: &Tree, comment: Option<String>, into: Into) -> Result<Where> {
        let text = top(n);
        if !n.ty.is_some_and(Ty::is_lanes) {
            return expr(format!("{text}: not a condition"));
        }
        if junction(n).is_some() {
            return expr(format!(
                "{text}: a junction of compares is a branch's (short-circuit branches), not one compare"
            ));
        }
        if into != Into::Scc {
            if let Some(v) = value_of(n) {
                return Ok(Where::Mask(v.clone()));
            }
            if let Val::Special(p) = n.val {
                return Ok(Where::Phys(p));
            }
        }
        self.stmt_comment = Some(comment.unwrap_or(text.clone()));
        let out = self.cond_into(n, into, &text);
        self.stmt_comment = None;
        out
    }

    fn cond_into(&mut self, n: &Tree, into: Into, text: &str) -> Result<Where> {
        let mask = n.ty.is_some_and(Ty::is_mask);
        if into == Into::Mask || mask && into == Into::Lanes {
            return self.into_mask(n, None).map(Where::Mask);
        }
        if into == Into::Scc || n.file == Some('s') {
            if into == Into::Scc && n.file == Some('v') {
                return expr(format!(
                    "{text}: a lane condition in a branch (wave-uniform): any(c) or all(c), or an exec region (b.lanes)"
                ));
            }
            self.into_scc(n)?;
            return Ok(Where::Scc);
        }
        self.into_vcc(n)?;
        Ok(Where::Vcc)
    }

    /// `cmp_name(compare, scalar)`.
    fn cmp_name(&self, n: &Node, scalar: bool) -> Result<String> {
        let Val::Cmp(kind, t, negated) = n.val else {
            unreachable!("a compare")
        };
        if scalar {
            if t == Ty::F32 {
                return expr("a float compare is VALU (the target has no scalar float)");
            }
            if t == Ty::U64 && !matches!(kind, Kind::Eq | Kind::Ne) {
                return expr(format!(
                    "a scalar u64 compares == or != (s_cmp_eq_u64 / s_cmp_lg_u64), not {}",
                    kind.symbol()
                ));
            }
            let k = if kind == Kind::Ne { "lg" } else { kind.name() };
            return Ok(format!("s.cmp.{k}.{}", t.name()));
        }
        if t == Ty::F32 {
            let k = match (negated, kind) {
                (false, Kind::Ne) => "neq",
                (false, k) => k.name(),
                (true, Kind::Lt) => "nlt",
                (true, Kind::Le) => "nle",
                (true, Kind::Gt) => "ngt",
                (true, Kind::Ge) => "nge",
                (true, Kind::Eq) => "neq",
                (true, Kind::Ne) => "eq",
            };
            return Ok(format!("v.cmp.{k}.f32"));
        }
        Ok(format!("v.cmp.{}.{}", kind.name(), t.name()))
    }

    /// A scalar condition into SCC.
    pub fn into_scc(&mut self, c: &Tree) -> Result<()> {
        match (&c.op, &c.val) {
            (Op::Cmp, Val::Cmp(kind, t, _)) => {
                let (a, b) = (&c.args[0], &c.args[1]);
                let zero = const_of(b).is_some_and(|z| z.is(0));
                if matches!(kind, Kind::Eq | Kind::Ne) && zero && a.op == Op::And && t.is_int() {
                    if let Some(k) = const_of(&a.args[1])
                        .and_then(Num::int)
                        .filter(|v| *v > 0 && v & (v - 1) == 0)
                    {
                        let x = self.operand(&a.args[0], Some(*t))?;
                        let op = if *kind == Kind::Eq {
                            "s.bitcmp0.b32"
                        } else {
                            "s.bitcmp1.b32"
                        };
                        let bit = Opd::Int(k.trailing_zeros().into());
                        return self.emit_op(op, None, &[x, bit], &[]);
                    }
                }
                let x = self.operand(a, Some(*t))?;
                let mut y = self.operand(b, Some(*t))?;
                if x.value().is_none()
                    && y.value().is_none()
                    && !matches!(x, Opd::Phys(_))
                    && !matches!(y, Opd::Phys(_))
                {
                    return expr(format!("{}: two constants (a Python bool)", top(c)));
                }
                if x.literal() && y.literal() {
                    y = self.sgpr_copy(&y, *t)?;
                }
                let name = self.cmp_name(c, true)?;
                self.emit_op(&name, None, &[x, y], &[])
            }
            (Op::Any | Op::All, _) => {
                let inner = &c.args[0];
                let m = match inner.op == Op::Cmp && inner.file == Some('v') {
                    true => {
                        self.into_vcc(inner)?;
                        self.vcc()
                    }
                    false => self.lanes_of(inner)?,
                };
                let exec = Opd::Phys(self.t.mask_reg(phys_named("exec").expect("exec")));
                match c.op {
                    Op::Any => self.emit_op(
                        &self.t.masked("s.cmp.lg.umask"),
                        None,
                        &[m, Opd::Int(0)],
                        &[],
                    ),
                    _ => self.emit_op(&self.t.masked("s.cmp.eq.umask"), None, &[m, exec], &[]),
                }
            }
            _ => expr(format!(
                "{}: not a scalar condition (a scalar compare, any(c), all(c))",
                top(c)
            )),
        }
    }

    /// A lane condition into VCC.
    pub fn into_vcc(&mut self, c: &Tree) -> Result<()> {
        let vcc = self.vcc();
        let Val::Cmp(kind, t, negated) = c.val else {
            let m = self.lanes_of(c)?;
            return self.emit_name(&self.spelled(&self.t.masked("s.mov.mask"))?, &[vcc, m], &[]);
        };
        let mut x = self.operand(&c.args[0], Some(t))?;
        let mut y = self.operand(&c.args[1], Some(t))?;
        let mut kind = kind;
        if !y.vgpr() && x.vgpr() {
            std::mem::swap(&mut x, &mut y);
            kind = kind.mirror();
        }
        let mirrored = Node {
            val: Val::Cmp(kind, t, negated),
            ..(**c).clone()
        };
        let name = self.cmp_name(&mirrored, false)?;
        if !y.vgpr() {
            return self.vop_to(&name, vcc, vec![x, y], t);
        }
        self.emit_name(&self.spelled(&name)?, &[vcc, x, y], &[])
    }

    /// VOP into a named register (a compare into VCC).
    fn vop_to(&mut self, op: &str, d: Opd, srcs: Vec<Opd>, t: Ty) -> Result<()> {
        let Opd::Phys(p) = d else {
            unreachable!("a named destination")
        };
        self.vop_phys(op, p, srcs, t, Vop::default())
    }

    /// A condition's lanes in a register.
    pub fn lanes_of(&mut self, c: &Tree) -> Result<Opd> {
        if let Some(v) = value_of(c).filter(|v| v.ty.is_mask()) {
            return Ok(Opd::Reg(v.clone(), 0));
        }
        if let Val::Special(p) = c.val {
            return Ok(Opd::Phys(self.t.mask_reg(p)));
        }
        self.into_mask(c, None).map(|m| Opd::Reg(m, 0))
    }

    /// A condition into the mask m, or a fresh one.
    pub fn into_mask(&mut self, c: &Tree, m: Option<Typed>) -> Result<Typed> {
        let m = match m {
            Some(m) => m,
            None => self.temp('s', Some(Ty::Mask))?,
        };
        let mov = self.t.masked("s.mov.mask");
        match (&c.op, &c.val) {
            (Op::Cmp, _) if c.file == Some('s') => {
                self.into_scc(c)?;
                let op = self.t.masked("s.cselect.mask");
                self.emit_op(&op, Some(&m), &[Opd::Int(-1), Opd::Int(0)], &[])?;
            }
            (Op::Cmp, Val::Cmp(_, t, _)) => {
                let srcs = self.operands(c, Some(*t))?;
                let name = format!("{}.e64", self.cmp_name(c, false)?);
                self.vop(&name, &m, srcs, *t, Vop::default())?;
            }
            (Op::Val, Val::Value(v)) => {
                self.emit_op(&mov, Some(&m), &[Opd::Reg(v.clone(), 0)], &[])?
            }
            (Op::Special, Val::Special(p)) => {
                self.emit_op(&mov, Some(&m), &[Opd::Phys(self.t.mask_reg(p))], &[])?
            }
            (Op::Cand | Op::Cor | Op::Cxor, _) => {
                let a = self.lanes_of(&c.args[0])?;
                let r = &c.args[1];
                let b = match r.op == Op::Cmp && r.file == Some('v') {
                    true => {
                        self.into_vcc(r)?;
                        self.vcc()
                    }
                    false => self.lanes_of(r)?,
                };
                let op = match c.op {
                    Op::Cand => "s.and.mask",
                    Op::Cor => "s.or.mask",
                    _ => "s.xor.mask",
                };
                self.emit_op(&self.t.masked(op), Some(&m), &[a, b], &[])?;
            }
            (Op::Cnot, _) => {
                let x = self.lanes_of(&c.args[0])?;
                let exec = Opd::Phys(self.t.mask_reg(phys_named("exec").expect("exec")));
                self.emit_op(&self.t.masked("s.andn2.mask"), Some(&m), &[exec, x], &[])?;
            }
            _ => return expr(format!("{}: not a lane condition", top(c))),
        }
        Ok(m)
    }
}
