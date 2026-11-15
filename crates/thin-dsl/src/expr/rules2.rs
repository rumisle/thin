//! The rules of storage formats, lane functions, across, scan, select and
//! any instruction.

use thin_target::phys_named;

use super::lower::Opd;
use super::node::{binop, constant, Num, Op, Tree, Val};
use super::text::top;
use super::vop::Vop;
use crate::builder::Builder;
use crate::error::{expr, Result};
use crate::maker::FArg;
use crate::types::Ty;
use crate::values::Typed;
use crate::waits::{Thing, Wait};

fn fp8(name: &str) -> &'static str {
    if name == "e4m3fn" {
        "fp8"
    } else {
        "bf8"
    }
}

/// `op_sel` with those bits, none when all are 0.
fn sel(bits: &[u32]) -> Vec<(String, FArg)> {
    match bits.iter().any(|&b| b != 0) {
        true => vec![(
            "op_sel".to_owned(),
            FArg::List(bits.iter().map(|&b| FArg::Int(b.into())).collect()),
        )],
        false => Vec::new(),
    }
}

fn dpp(step: &str) -> Vec<(String, FArg)> {
    vec![
        ("dpp".to_owned(), FArg::Str(step.to_owned())),
        ("row_mask".to_owned(), FArg::Int(15)),
        ("bank_mask".to_owned(), FArg::Int(15)),
    ]
}

impl Builder {
    pub(crate) fn storage(&mut self, n: &Tree, d: &Typed) -> Result<()> {
        let Val::Fmt(f) = &n.val else {
            unreachable!("a conversion holds its format")
        };
        let at = f.at;
        let plain = Vop::default();
        match n.op {
            Op::Unpack if f.name == "f16" => {
                let mut w = self.operand(&n.args[0], Some(Ty::U32))?;
                if at == 1 {
                    let t = self.temp('v', Some(Ty::U32))?;
                    self.vop(
                        "v.lshrrev.b32",
                        &t,
                        vec![Opd::Int(16), w],
                        Ty::U32,
                        plain.clone(),
                    )?;
                    w = Opd::Reg(t, 0);
                }
                self.vop("v.cvt.f32.f16", d, vec![w], Ty::U32, plain)
            }
            Op::Unpack => {
                let w = self.operand(&n.args[0], Some(Ty::U32))?;
                let k = fp8(&f.name);
                self.vop(&format!("v.cvt.f32.{k}"), d, vec![w], Ty::U32, plain)
            }
            Op::Unpack2 => {
                let w = self.operand(&n.args[0], Some(Ty::U32))?;
                let k = fp8(&f.name);
                self.vop(&format!("v.cvt.pk.f32.{k}"), d, vec![w], Ty::U32, plain)
            }
            _ => self.pack2(n, d),
        }
    }

    fn pack2(&mut self, n: &Tree, d: &Typed) -> Result<()> {
        let Val::Fmt(f) = &n.val else {
            unreachable!("a conversion holds its format")
        };
        let at = f.at;
        if f.name == "f16" {
            self.spelled("v.cvt.pk.f16.f32")?;
            let a = self.operand(&n.args[0], Some(Ty::F32))?;
            let b = self.operand(&n.args[1], Some(Ty::F32))?;
            return self.vop("v.cvt.pk.f16.f32", d, vec![a, b], Ty::F32, Vop::default());
        }
        let into = super::node::value_of(&n.args[2]);
        if !into.is_some_and(|v| v.value == d.value && v.sub == d.sub) {
            let w = top(&n.args[2]);
            return expr(format!(
                "{}: written into {w} itself ({w}[...] = pack2(a, b, {}, into={w}): its other bits kept)",
                top(n),
                f.name
            ));
        }
        let a = self.operand(&n.args[0], Some(Ty::F32))?;
        let b = self.operand(&n.args[1], Some(Ty::F32))?;
        let op = format!("v.cvt.pk.{}.f32", fp8(&f.name));
        let bits = vec![0, 0, at];
        let srcs = vec![a, b];
        self.vop(
            &op,
            d,
            srcs,
            Ty::F32,
            Vop {
                fields: sel(&bits),
                ..Vop::default()
            },
        )
    }

    pub(crate) fn lanes_rule(&mut self, n: &Tree, d: &Typed) -> Result<()> {
        match n.op {
            Op::FirstLane => {
                let x = self.operand(&n.args[0], n.args[0].ty)?;
                match x.vgpr() {
                    false => self.mov(d, &x),
                    true => self.emit_op("v.readfirstlane.b32", Some(d), &[x], &[]),
                }
            }
            Op::LaneRead => {
                let x = self.operand(&n.args[0], n.args[0].ty)?;
                let k = self.operand(&n.args[1], Some(Ty::U32))?;
                self.emit_op("v.readlane.b32", Some(d), &[x, k], &[])
            }
            Op::LaneWrite => {
                let same = super::node::value_of(&n.args[0])
                    .is_some_and(|v| v.value == d.value && v.sub == d.sub);
                if !same {
                    let x = top(&n.args[0]);
                    return expr(format!(
                        "{}: written into {x} itself ({x}[...] = lane_write({x}, k, s): the other lanes kept)",
                        top(n)
                    ));
                }
                let mut k = self.operand(&n.args[1], Some(Ty::U32))?;
                let s = self.operand(&n.args[2], n.args[2].ty)?;
                let on_k = matches!((&s, &k), (Opd::Reg(a, _), Opd::Reg(b, _)) if a.value == b.value && a.sub == b.sub);
                if k.sgpr() && (s.sgpr() && !on_k || s.literal()) {
                    let m0 = Opd::Phys(phys_named("m0").expect("m0"));
                    self.emit_name(&self.spelled("s.mov.b32")?, &[m0.clone(), k], &[])?;
                    k = m0;
                }
                self.emit_op("v.writelane.b32", Some(d), &[s, k], &[])
            }
            Op::Shuffle => {
                let mut x = self.operand(&n.args[0], n.args[0].ty)?;
                if !x.vgpr() {
                    x = self.vgpr_copy(&x, n.args[0].ty.unwrap_or(Ty::U32))?;
                }
                let a = self.temp('v', Some(Ty::U32))?;
                let shifted = binop(Op::Shl, n.args[1].clone(), constant(Num::Int(2)))?;
                self.lower(&shifted, Some(&a), Some(Ty::U32))?;
                self.emit_op("ds.bpermute.b32", Some(d), &[Opd::Reg(a, 0), x], &[])?;
                let w = Wait {
                    comment: "the shuffle".into(),
                    ..Wait::default()
                };
                self.wait(vec![Thing::Reg(d.reg())], w)
            }
            _ => {
                let m = self.lanes_of(&n.args[0])?;
                let base = match n.args.get(1) {
                    Some(b) => self.operand(b, Some(Ty::U32))?,
                    None => Opd::Int(0),
                };
                if self.t.definition().wave == 32 {
                    return self.emit_op("v.mbcnt.lo.u32.b32", Some(d), &[m, base], &[]);
                }
                let t = self.temp('v', Some(Ty::U32))?;
                let (lo, hi) = self.mask_halves(&m)?;
                self.emit_op("v.mbcnt.lo.u32.b32", Some(&t), &[lo, base], &[])?;
                self.emit_op("v.mbcnt.hi.u32.b32", Some(d), &[hi, Opd::Reg(t, 0)], &[])
            }
        }
    }

    /// A mask's halves: a value's `.lo`/`.hi`, a named mask's halves.
    fn mask_halves(&self, m: &Opd) -> Result<(Opd, Opd)> {
        match m {
            Opd::Reg(v, _) => Ok((Opd::Reg(v.half(0)?, 0), Opd::Reg(v.half(1)?, 0))),
            Opd::Phys(p) => {
                let stem = if p.name.starts_with("vcc") {
                    "vcc"
                } else {
                    "exec"
                };
                let h = |s: &str| Opd::Phys(phys_named(&format!("{stem}_{s}")).expect("a half"));
                Ok((h("lo"), h("hi")))
            }
            m => Ok((m.clone(), m.clone())),
        }
    }

    pub(crate) fn across(&mut self, n: &Tree, d: &Typed) -> Result<()> {
        let Val::Lanes(op, lanes) = n.val else {
            unreachable!("across holds its operator")
        };
        let name = self.spelled(&format!("v.{op}.f32"))?;
        let wave = self.t.definition().wave;
        if lanes > wave {
            return expr(format!("across over {lanes} lanes: a wave of {wave} lanes"));
        }
        let mut src = self.operand(&n.args[0], Some(Ty::F32))?;
        if !src.vgpr() {
            src = self.vgpr_copy(&src, Ty::F32)?;
        }
        let steps = [
            "quad_perm:[1,0,3,2]",
            "quad_perm:[2,3,0,1]",
            "row_half_mirror",
            "row_mirror",
        ];
        let k = match lanes {
            2 => 1,
            4 => 2,
            8 => 3,
            _ => 4,
        };
        let dd = Opd::Reg(d.clone(), 0);
        for step in &steps[..k] {
            self.emit_name(
                &format!("{name}_dpp"),
                &[dd.clone(), src.clone(), src.clone()],
                &dpp(step),
            )?;
            src = dd.clone();
        }
        if lanes < 32 {
            return Ok(());
        }
        let vc = Vop {
            commute: true,
            ..Vop::default()
        };
        if self.t.spell("v.permlane16.swap.b32").is_none() {
            if lanes == 32 && self.t.spell("v.permlanex16.b32").is_some() {
                let lo = self.temp('s', Some(Ty::U32))?;
                let hi = self.temp('s', Some(Ty::U32))?;
                self.emit_op("s.mov.b32", Some(&lo), &[Opd::Int(0x7654_3210)], &[])?;
                self.emit_op("s.mov.b32", Some(&hi), &[Opd::Int(0xFEDC_BA98)], &[])?;
                let t = self.temp('v', Some(Ty::F32))?;
                self.emit_op(
                    "v.permlanex16.b32",
                    Some(&t),
                    &[dd.clone(), Opd::Reg(lo, 0), Opd::Reg(hi, 0)],
                    &[],
                )?;
                return self.vop(
                    &format!("v.{op}.f32"),
                    d,
                    vec![dd, Opd::Reg(t, 0)],
                    Ty::F32,
                    vc,
                );
            }
            let lid = self.temp('v', Some(Ty::U32))?;
            if wave == 32 {
                self.emit_op(
                    "v.mbcnt.lo.u32.b32",
                    Some(&lid),
                    &[Opd::Int(-1), Opd::Int(0)],
                    &[],
                )?;
            } else {
                let lo = self.temp('v', Some(Ty::U32))?;
                self.emit_op(
                    "v.mbcnt.lo.u32.b32",
                    Some(&lo),
                    &[Opd::Int(-1), Opd::Int(0)],
                    &[],
                )?;
                self.emit_op(
                    "v.mbcnt.hi.u32.b32",
                    Some(&lid),
                    &[Opd::Int(-1), Opd::Reg(lo, 0)],
                    &[],
                )?;
            }
            let base = self.temp('v', Some(Ty::U32))?;
            self.emit_op(
                "v.lshlrev.b32",
                Some(&base),
                &[Opd::Int(2), Opd::Reg(lid, 0)],
                &[],
            )?;
            let steps: &[i128] = if lanes == 32 { &[64] } else { &[64, 128] };
            for &step in steps {
                let addr = self.temp('v', Some(Ty::U32))?;
                self.emit_op(
                    "v.xor.b32",
                    Some(&addr),
                    &[Opd::Int(step), Opd::Reg(base.clone(), 0)],
                    &[],
                )?;
                let t = self.temp('v', Some(Ty::F32))?;
                self.emit_op(
                    "ds.bpermute.b32",
                    Some(&t),
                    &[Opd::Reg(addr, 0), dd.clone()],
                    &[],
                )?;
                let w = Wait {
                    comment: "the across".into(),
                    ..Wait::default()
                };
                self.wait(vec![Thing::Reg(t.reg())], w)?;
                self.vop(
                    &format!("v.{op}.f32"),
                    d,
                    vec![dd.clone(), Opd::Reg(t, 0)],
                    Ty::F32,
                    vc.clone(),
                )?;
            }
            return Ok(());
        }
        let swaps: &[&str] = if lanes == 32 {
            &["v.permlane16.swap.b32"]
        } else {
            &["v.permlane16.swap.b32", "v.permlane32.swap.b32"]
        };
        for swap in swaps {
            let t = self.temp('v', Some(Ty::F32))?;
            self.mov(&t, &dd)?;
            self.emit_op(swap, Some(d), &[Opd::Reg(t.clone(), 0)], &[])?;
            self.vop(
                &format!("v.{op}.f32"),
                d,
                vec![dd.clone(), Opd::Reg(t, 0)],
                Ty::F32,
                vc.clone(),
            )?;
        }
        Ok(())
    }

    pub(crate) fn scan(&mut self, n: &Tree, d: &Typed) -> Result<()> {
        let Val::Lanes(op, lanes) = n.val else {
            unreachable!("scan holds its operator")
        };
        let name = self.spelled(&format!("v.{op}.f32"))?;
        let x = self.operand(&n.args[0], Some(Ty::F32))?;
        self.mov(d, &x)?;
        let dd = Opd::Reg(d.clone(), 0);
        for k in [1, 2, 4, 8] {
            self.emit_name(
                &format!("{name}_dpp"),
                &[dd.clone(), dd.clone(), dd.clone()],
                &dpp(&format!("row_shr:{k}")),
            )?;
        }
        // Rows of 16 lanes scan by DPP; each further row adds the last lane
        // of the row before. Lanes past the wave are refused; a 64-lane
        // wave's 32-lane scans are its two halves.
        let wave = self.t.definition().wave;
        if lanes > wave {
            return expr(format!("{}: {lanes} lanes on a {wave}-lane target", top(n)));
        }
        let rows: &[u32] = match (lanes, wave) {
            (16, _) => &[],
            (32, 32) => &[1],
            (32, _) => &[1, 3],
            _ => &[1, 2, 3],
        };
        let vc = Vop {
            commute: true,
            ..Vop::default()
        };
        for &r in rows {
            let t = self.temp('s', Some(Ty::F32))?;
            let v = self.temp('v', Some(Ty::F32))?;
            let m = self.temp('s', Some(Ty::Mask))?;
            self.emit_op(
                "v.readlane.b32",
                Some(&t),
                &[dd.clone(), Opd::Int(16 * i128::from(r) - 1)],
                &[],
            )?;
            self.vop(
                &format!("v.{op}.f32"),
                &v,
                vec![dd.clone(), Opd::Reg(t, 0)],
                Ty::F32,
                vc.clone(),
            )?;
            let bits: i128 = 0xFFFF << (16 * r);
            if wave == 32 {
                self.emit_op("s.mov.b32", Some(&m), &[Opd::Int(bits)], &[])?;
            } else {
                self.emit_op(
                    "s.mov.b32",
                    Some(&m.half(0)?),
                    &[Opd::Int(bits & 0xFFFF_FFFF)],
                    &[],
                )?;
                self.emit_op("s.mov.b32", Some(&m.half(1)?), &[Opd::Int(bits >> 32)], &[])?;
            }
            self.emit_op(
                "v.cndmask.b32.e64",
                Some(d),
                &[dd.clone(), Opd::Reg(v, 0), Opd::Reg(m, 0)],
                &[],
            )?;
        }
        Ok(())
    }

    pub(crate) fn select(&mut self, n: &Tree, d: &Typed) -> Result<()> {
        let t = n.ty.unwrap_or(Ty::U32);
        let arm = |b: &mut Builder, a: &Tree| -> Result<Opd> {
            match a.op {
                Op::Neg | Op::Abs => b.lower(a, None, Some(t)),
                _ => b.operand(a, Some(t)),
            }
        };
        let x = arm(self, &n.args[1])?;
        let y = arm(self, &n.args[2])?;
        let c = &n.args[0];
        if d.file == 's' {
            if c.file == Some('v') || c.ty.is_some_and(Ty::is_mask) {
                return expr(format!(
                    "{}: a lane condition chooses per lane: a vector select (a VGPR value)",
                    top(n)
                ));
            }
            self.into_scc(c)?;
            let op = if t.is_wide() {
                "s.cselect.b64"
            } else {
                "s.cselect.b32"
            };
            return self.sop(op, d, vec![x, y]);
        }
        let y = if !y.vgpr() && !y.inline() {
            self.vgpr_copy(&y, t)?
        } else {
            y
        };
        let x = if !x.vgpr() { self.vgpr_copy(&x, t)? } else { x };
        if c.op == Op::Cmp && c.file == Some('v') {
            self.into_vcc(c)?;
            let vcc = Opd::Phys(self.t.mask_reg(phys_named("vcc").expect("vcc")));
            return self.emit_op("v.cndmask.b32", Some(d), &[y, x, vcc], &[]);
        }
        let m = self.lanes_of(c)?;
        self.emit_op("v.cndmask.b32.e64", Some(d), &[y, x, m], &[])
    }

    pub(crate) fn instr(&mut self, n: &Tree, d: &Typed) -> Result<()> {
        let Val::Instr(given, fields) = &n.val else {
            unreachable!("op holds its name")
        };
        let mut ops = Vec::new();
        for a in &n.args {
            ops.push(self.operand(a, a.ty.or(n.ty))?);
        }
        let name = self.spelled(given)?;
        if name.starts_with("v_") && fields.is_empty() {
            return self.vop(&name, d, ops, n.ty.unwrap_or(Ty::U32), Vop::default());
        }
        let mut all = vec![Opd::Reg(d.clone(), 0)];
        all.extend(ops);
        let args = all.iter().map(|o| o.arg()).collect();
        let i = self.make(&name, args, fields.clone(), crate::maker::Notes::default())?;
        self.emit_lowered(i)
    }
}
