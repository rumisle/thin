//! The operators' rules.

use thin_target::phys_named;

use super::lower::{identity, Opd};
use super::node::{binop, const_of, constant, Divisor, Num, Op, Tree, Val};
use super::text::{num, top};
use super::vop::Vop;
use crate::builder::Builder;
use crate::error::{expr, Result};
use crate::maker::FArg;
use crate::types::Ty;
use crate::values::Typed;

/// The shift of a node: `x0 << k` with a constant k, `x0 * 2^k`, or
/// (`any`) `x0 << s` with any tree s.
enum Amount {
    Const(u32),
    Tree(Tree),
}

fn power(c: Num) -> Option<u32> {
    let v = c.int()?;
    (v > 0 && v & (v - 1) == 0).then(|| v.trailing_zeros())
}

fn shift_of(n: &Tree, any: bool) -> Option<(Tree, Amount)> {
    match n.op {
        Op::Shl => match const_of(&n.args[1]).and_then(Num::int) {
            Some(k) if (0..128).contains(&k) => Some((n.args[0].clone(), Amount::Const(k as u32))),
            _ if any => Some((n.args[0].clone(), Amount::Tree(n.args[1].clone()))),
            _ => None,
        },
        Op::Mul => {
            if let Some(k) = const_of(&n.args[1]).and_then(power) {
                return Some((n.args[0].clone(), Amount::Const(k)));
            }
            const_of(&n.args[0])
                .and_then(power)
                .map(|k| (n.args[1].clone(), Amount::Const(k)))
        }
        _ => None,
    }
}

fn int_node(c: i128) -> Tree {
    constant(Num::Int(c))
}

impl Builder {
    /// The rule of n's operator, writing into d.
    pub(crate) fn rule(&mut self, n: &Tree, d: &Typed) -> Result<()> {
        use Op::{
            Abs, Across, Add, All, And, Any, Bf16Hi, Bf16Lo, Bfe, Cand, Cast, Cmp, Cnot, Const,
            Cor, CvtF32, CvtInt, Cxor, Div, First, FirstLane, Fma, Instr, LaneRead, LaneWrite,
            Mad24, Max, Max3, Med3, Min, Min3, Mod, Mul, Mul24, MulWide, Mulhi, Neg, Not, Or,
            Pack2, Pair, PkBf16, Popcount, PopcountMask, Rank, Scan, Select, Shl, Shr, Shuffle,
            Special, Sub, Unpack, Unpack2, Xor, F1,
        };
        match n.op {
            Add | Sub | Mul | And | Or | Xor | Shl | Shr => self.arith(n, d),
            Not => {
                let x = self.operand(&n.args[0], n.ty)?;
                match d.file {
                    's' => self.sop("s.not.b32", d, vec![x]),
                    _ => self.vop(
                        "v.not.b32",
                        d,
                        vec![x],
                        n.ty.unwrap_or(Ty::U32),
                        Vop::default(),
                    ),
                }
            }
            Div | Mod => self.division(n, d),
            Mul24 | Mad24 | Mulhi | MulWide => self.multiply(n, d),
            Min | Max | Min3 | Max3 | Med3 => self.minmax(n, d),
            Bfe => match &n.val {
                Val::Bfe(o, w, signed) => {
                    let signed = signed.unwrap_or(n.ty == Some(Ty::I32));
                    self.bfe_rule(&n.args[0], *o, *w, signed, d)
                }
                _ => unreachable!("a bfe node holds its field"),
            },
            Popcount => {
                let x = self.operand(&n.args[0], Some(Ty::U32))?;
                match d.file {
                    's' => self.sop("s.bcnt1.i32.b32", d, vec![x]),
                    _ => self.vop(
                        "v.bcnt.u32.b32",
                        d,
                        vec![x, Opd::Int(0)],
                        Ty::U32,
                        Vop::default(),
                    ),
                }
            }
            PopcountMask | First => {
                let m = self.lanes_of(&n.args[0])?;
                let op = if n.op == First {
                    "s.ff1.i32.mask"
                } else {
                    "s.bcnt1.i32.mask"
                };
                let op = self.t.masked(op);
                self.emit_op(&op, Some(d), &[m], &[])
            }
            Fma | F1 | CvtF32 | CvtInt | Bf16Lo | Bf16Hi | PkBf16 => self.floats(n, d),
            Neg | Abs => self.sign_bits(n, d),
            Pair => self.pair(n, d),
            Unpack | Unpack2 | Pack2 => self.storage(n, d),
            FirstLane | LaneRead | LaneWrite | Shuffle | Rank => self.lanes_rule(n, d),
            Across => self.across(n, d),
            Scan => self.scan(n, d),
            Select => self.select(n, d),
            Instr => self.instr(n, d),
            Cmp | Cand | Cor | Cxor | Cnot | Any | All => {
                self.into_mask(n, Some(d.clone())).map(|_| ())
            }
            Op::Val | Special | Const | Cast => unreachable!("leaves lower before the rules"),
        }
    }

    fn arith(&mut self, n: &Tree, d: &Typed) -> Result<()> {
        let t = n.ty.unwrap_or(Ty::U32);
        if t == Ty::F32x2 {
            return self.packed(n, d);
        }
        if t.is_wide() {
            if let Some(kept) = identity(n) {
                let kept = kept.clone();
                return self.lower(&kept, Some(d), Some(t)).map(|_| ());
            }
            return self.carry(n, d);
        }
        if t != Ty::F32 {
            if let Some(r) = self.rewrite(n, t)? {
                return match r {
                    Rewrite::Tree(r) => self.lower(&r, Some(d), Some(t)).map(|_| ()),
                    Rewrite::Shift(x, amount, y, or) => self.shift_add(d, t, x, amount, y, or),
                    Rewrite::ScalarShift(x, k, y) => {
                        let (x, y) = (self.operand(&x, Some(t))?, self.operand(&y, Some(t))?);
                        self.sop(&format!("s.lshl{k}.add.u32"), d, vec![x, y])
                    }
                    Rewrite::Bfe(x, k, m) => self.bfe_rule(&x, k, m, false, d),
                };
            }
        }
        self.plain(n, t, d)
    }

    fn rewrite(&mut self, n: &Tree, t: Ty) -> Result<Option<Rewrite>> {
        let (a, b) = (&n.args[0], &n.args[1]);
        let int = t.is_int();
        if int && matches!(n.op, Op::Add | Op::Sub) && matches!(a.op, Op::Add | Op::Sub) {
            if let (Some(c2), Some(c1)) = (
                const_of(b).and_then(Num::int),
                const_of(&a.args[1]).and_then(Num::int),
            ) {
                let s1 = if a.op == Op::Add { c1 } else { -c1 };
                let s2 = if n.op == Op::Add { c2 } else { -c2 };
                let mut c = (s1 + s2).rem_euclid(1 << 32);
                if c > 1 << 31 {
                    c -= 1 << 32;
                }
                let x = a.args[0].clone();
                let r = if c >= 0 {
                    binop(Op::Add, x, int_node(c))?
                } else {
                    binop(Op::Sub, x, int_node(-c))?
                };
                return Ok(Some(Rewrite::Tree(r)));
            }
        }
        if int && n.op == Op::And && a.op == Op::Add {
            if let Some(m) = const_of(b).and_then(Num::int) {
                for (s, y) in [(&a.args[1], &a.args[0]), (&a.args[0], &a.args[1])] {
                    if let Some((_, Amount::Const(k))) = shift_of(s, false) {
                        if k < 127 && 0 <= m && m < 1i128 << k {
                            return Ok(Some(Rewrite::Tree(binop(Op::And, y.clone(), b.clone())?)));
                        }
                    }
                }
            }
        }
        if let Some(kept) = identity(n) {
            return Ok(Some(Rewrite::Tree(kept.clone())));
        }
        if n.op == Op::Mul {
            if let Some(k) = const_of(b).and_then(power) {
                return Ok(Some(Rewrite::Tree(binop(
                    Op::Shl,
                    a.clone(),
                    int_node(k.into()),
                )?)));
            }
            if let Some(k) = const_of(a).and_then(power) {
                return Ok(Some(Rewrite::Tree(binop(
                    Op::Shl,
                    b.clone(),
                    int_node(k.into()),
                )?)));
            }
        }
        let file = n.file.unwrap_or('s');
        if file == 'v' && matches!(n.op, Op::Add | Op::Or) {
            for (x, y) in [(b, a), (a, b)] {
                if let Some((x0, amount)) = shift_of(x, true) {
                    return Ok(Some(Rewrite::Shift(x0, amount, y.clone(), n.op == Op::Or)));
                }
            }
        }
        if file == 's' && n.op == Op::Add {
            for (x, y) in [(b, a), (a, b)] {
                if let Some((x0, Amount::Const(k))) = shift_of(x, false) {
                    if (1..=4).contains(&k) {
                        return Ok(Some(Rewrite::ScalarShift(x0, k, y.clone())));
                    }
                }
            }
        }
        if n.op == Op::And && a.op == Op::Shr {
            let k = const_of(&a.args[1]).and_then(Num::int);
            let c = const_of(b).and_then(Num::int);
            if let (Some(k), Some(c)) = (k, c) {
                let m = (c + 1).trailing_zeros();
                if c != 0 && (c + 1) & c == 0 && c > 0 && k >= 0 && k + i128::from(m) <= 32 {
                    return Ok(Some(Rewrite::Bfe(a.args[0].clone(), k as u32, m)));
                }
            }
        }
        Ok(None)
    }

    fn shift_add(
        &mut self,
        d: &Typed,
        t: Ty,
        x: Tree,
        amount: Amount,
        y: Tree,
        or: bool,
    ) -> Result<()> {
        let x = self.operand(&x, Some(t))?;
        let k = match amount {
            Amount::Const(k) => Opd::Int(k.into()),
            Amount::Tree(s) => self.operand(&s, Some(Ty::U32))?,
        };
        let y = self.operand(&y, Some(t))?;
        let op = if or {
            "v.lshl.or.b32"
        } else {
            "v.lshl.add.u32"
        };
        self.vop(op, d, vec![x, k, y], t, Vop::default())
    }

    fn plain(&mut self, n: &Tree, t: Ty, d: &Typed) -> Result<()> {
        let x = self.operand(&n.args[0], Some(t))?;
        let y = self.operand(&n.args[1], Some(t))?;
        if d.file == 's' {
            let op = match n.op {
                Op::Add => "s.add.u32",
                Op::Sub => "s.sub.u32",
                Op::Mul => "s.mul.i32",
                Op::And => "s.and.b32",
                Op::Or => "s.or.b32",
                Op::Xor => "s.xor.b32",
                Op::Shl => "s.lshl.b32",
                _ if t == Ty::I32 => "s.ashr.i32",
                _ => "s.lshr.b32",
            };
            return self.sop(op, d, vec![x, y]);
        }
        let float = t == Ty::F32;
        let (op, rev, commute) = match n.op {
            Op::Shl => return self.vop("v.lshlrev.b32", d, vec![y, x], t, Vop::default()),
            Op::Shr => {
                let op = if t == Ty::I32 {
                    "v.ashrrev.i32"
                } else {
                    "v.lshrrev.b32"
                };
                return self.vop(op, d, vec![y, x], t, Vop::default());
            }
            Op::Add if float => ("v.add.f32", None, true),
            Op::Sub if float => ("v.sub.f32", Some("v.subrev.f32"), false),
            Op::Mul if float => ("v.mul.f32", None, true),
            Op::Add => ("v.add.u32", None, true),
            Op::Sub => ("v.sub.u32", Some("v.subrev.u32"), false),
            Op::Mul => ("v.mul.lo.u32", None, true),
            Op::And => ("v.and.b32", None, true),
            Op::Or => ("v.or.b32", None, true),
            _ => ("v.xor.b32", None, true),
        };
        self.vop(
            op,
            d,
            vec![x, y],
            t,
            Vop {
                rev,
                commute,
                ..Vop::default()
            },
        )
    }

    /// The carry chain of a wide `+` or `-`.
    fn carry(&mut self, n: &Tree, d: &Typed) -> Result<()> {
        let (mut a, mut b) = (n.args[0].clone(), n.args[1].clone());
        if !a.ty.is_some_and(Ty::is_wide) {
            if n.op == Op::Sub {
                return expr(format!("{}: a u32 minus a u64", top(n)));
            }
            std::mem::swap(&mut a, &mut b);
        }
        let x = self.operand(&a, Some(Ty::U64))?;
        let (xl, xh) = halves(&x);
        let yt = b.ty.unwrap_or(Ty::U32);
        if let (Some(c), None) = (const_of(&b), b.ty) {
            let (yl, yh) = wide_constant(c, &b)?;
            return self.chain(n.op == Op::Add, d, (xl, xh), (yl, yh));
        }
        let y = self.operand(&b, Some(yt))?;
        let (yl, yh) = match (&y, yt) {
            (Opd::Reg(v, _), t) if t.is_wide() => {
                (Opd::Reg(v.half_dword(0), 0), Opd::Reg(v.half_dword(1), 0))
            }
            (y, Ty::I32) => {
                let h = self.temp(y.value().map_or('s', |v| v.file), Some(Ty::I32))?;
                match h.file {
                    's' => self.emit_op("s.ashr.i32", Some(&h), &[y.clone(), Opd::Int(31)], &[])?,
                    _ => {
                        self.emit_op("v.ashrrev.i32", Some(&h), &[Opd::Int(31), y.clone()], &[])?
                    }
                }
                (y.clone(), Opd::Reg(h, 0))
            }
            (y, _) => (y.clone(), Opd::Int(0)),
        };
        self.chain(n.op == Op::Add, d, (xl, xh), (yl, yh))
    }

    /// The two adds (or subtracts) of a wide `+` / `-`, the high one with the carry.
    fn chain(&mut self, add: bool, d: &Typed, x: (Opd, Opd), y: (Opd, Opd)) -> Result<()> {
        let ((xl, xh), (yl, yh)) = (x, y);
        let (dl, dh) = (d.half_dword(0), d.half_dword(1));
        if d.file == 's' {
            let (lo, hi) = if add {
                ("s.add.u32", "s.addc.u32")
            } else {
                ("s.sub.u32", "s.subb.u32")
            };
            self.sop(lo, &dl, vec![xl, yl])?;
            return self.sop(hi, &dh, vec![xh, yh]);
        }
        let (l0, l1) = self.src1(xl, yl, add)?;
        let l0 = if !l0.vgpr() && !l1.vgpr() {
            self.vgpr_copy(&l0, Ty::U32)?
        } else {
            l0
        };
        let xh = self.carry_in(xh)?;
        let yh = self.carry_in(yh)?;
        let (h0, h1) = self.src1(xh, yh, add)?;
        let vcc = Opd::Phys(self.t.mask_reg(phys_named("vcc").expect("vcc")));
        let (lo, hi) = if add {
            ("v.add.co.u32", "v.addc.co.u32")
        } else {
            ("v.sub.co.u32", "v.subb.co.u32")
        };
        self.emit_op(lo, Some(&dl), &[vcc.clone(), l0, l1], &[])?;
        self.emit_op(hi, Some(&dh), &[vcc.clone(), h0, h1, vcc], &[])
    }

    fn src1(&mut self, p: Opd, q: Opd, add: bool) -> Result<(Opd, Opd)> {
        if q.vgpr() {
            return Ok((p, q));
        }
        if p.vgpr() && add {
            return Ok((q, p));
        }
        let q = self.vgpr_copy(&q, Ty::U32)?;
        Ok((p, q))
    }

    fn carry_in(&mut self, z: Opd) -> Result<Opd> {
        match z.sgpr() || z.literal() {
            true => self.vgpr_copy(&z, Ty::U32),
            false => Ok(z),
        }
    }

    /// Division and remainder.
    fn division(&mut self, n: &Tree, d: &Typed) -> Result<()> {
        let Val::Divisor(div) = &n.val else {
            unreachable!("a division holds its divisor")
        };
        let rem = n.op == Op::Mod;
        let file = d.file;
        if let Divisor::Arg(dv) = div {
            return self.by_divisor(n, dv.clone(), d, rem);
        }
        let Divisor::Const(c) = *div else {
            unreachable!()
        };
        let dividend = &n.args[0];
        if c == 1 {
            let r = if rem { int_node(0) } else { dividend.clone() };
            return self.lower(&r, Some(d), Some(Ty::U32)).map(|_| ());
        }
        let x = self.operand(dividend, Some(Ty::U32))?;
        if x.value().is_none() {
            return expr(format!("{}: of a constant (compute it in Python)", top(n)));
        }
        if c & (c - 1) == 0 {
            let r = match rem {
                true => binop(Op::And, dividend.clone(), int_node(c - 1))?,
                false => binop(
                    Op::Shr,
                    dividend.clone(),
                    int_node(c.trailing_zeros().into()),
                )?,
            };
            return self.lower(&r, Some(d), Some(Ty::U32)).map(|_| ());
        }
        let (m, s, short) = magic_triple(c as u64);
        let q = if rem {
            self.temp(file, Some(Ty::U32))?
        } else {
            d.clone()
        };
        let m = Opd::Int(m.into());
        if file == 's' {
            self.sop("s.mul.hi.u32", &q, vec![x.clone(), m])?;
            if short {
                if s != 0 {
                    self.sop(
                        "s.lshr.b32",
                        &q,
                        vec![Opd::Reg(q.clone(), 0), Opd::Int(s.into())],
                    )?;
                }
            } else {
                let t = self.temp('s', Some(Ty::U32))?;
                let (tq, qq) = (Opd::Reg(t.clone(), 0), Opd::Reg(q.clone(), 0));
                self.sop("s.sub.u32", &t, vec![x.clone(), qq.clone()])?;
                self.sop("s.lshr.b32", &t, vec![tq.clone(), Opd::Int(1)])?;
                self.sop("s.add.u32", &t, vec![tq.clone(), qq])?;
                self.sop("s.lshr.b32", &q, vec![tq, Opd::Int(i128::from(s) - 1)])?;
            }
        } else {
            self.vop(
                "v.mul.hi.u32",
                &q,
                vec![x.clone(), m],
                Ty::U32,
                Vop::default(),
            )?;
            if short {
                if s != 0 {
                    let qq = Opd::Reg(q.clone(), 0);
                    self.vop(
                        "v.lshrrev.b32",
                        &q,
                        vec![Opd::Int(s.into()), qq],
                        Ty::U32,
                        Vop::default(),
                    )?;
                }
            } else {
                let t = self.temp('v', Some(Ty::U32))?;
                let (tq, qq) = (Opd::Reg(t.clone(), 0), Opd::Reg(q.clone(), 0));
                let sub = Vop {
                    rev: Some("v.subrev.u32"),
                    ..Vop::default()
                };
                self.vop("v.sub.u32", &t, vec![x.clone(), qq.clone()], Ty::U32, sub)?;
                self.vop(
                    "v.lshrrev.b32",
                    &t,
                    vec![Opd::Int(1), tq.clone()],
                    Ty::U32,
                    Vop::default(),
                )?;
                let add = Vop {
                    commute: true,
                    ..Vop::default()
                };
                self.vop("v.add.u32", &t, vec![tq.clone(), qq], Ty::U32, add)?;
                self.vop(
                    "v.lshrrev.b32",
                    &q,
                    vec![Opd::Int(i128::from(s) - 1), tq],
                    Ty::U32,
                    Vop::default(),
                )?;
            }
        }
        if rem {
            return self.remainder(dividend, &q, int_node(c), d);
        }
        Ok(())
    }

    /// `d = x - q·c` by
    fn remainder(&mut self, dividend: &Tree, q: &Typed, c: Tree, d: &Typed) -> Result<()> {
        let r = self.temp(d.file, Some(Ty::U32))?;
        let qn = super::node::leaf(super::node::Leafish::Value(q.clone()))?;
        let prod = binop(Op::Mul, qn, c)?;
        self.lower(&prod, Some(&r), Some(Ty::U32))?;
        let rn = super::node::leaf(super::node::Leafish::Value(r))?;
        let diff = binop(Op::Sub, dividend.clone(), rn)?;
        self.lower(&diff, Some(d), Some(Ty::U32)).map(|_| ())
    }

    fn by_divisor(&mut self, n: &Tree, dv: Typed, d: &Typed, rem: bool) -> Result<()> {
        let file = d.file;
        let x = self.operand(&n.args[0], Some(Ty::U32))?;
        let w = self.temp(file, Some(Ty::U64))?;
        let (wl, wh) = (w.half_dword(0), w.half_dword(1));
        let mul = Opd::Reg(dv.divisor_part(1), 0);
        let shr = Opd::Reg(dv.divisor_part(2), 0);
        let wlo = Opd::Reg(wl.clone(), 0);
        if file == 's' {
            self.sop("s.mul.hi.u32", &wl, vec![x.clone(), mul])?;
            self.sop("s.add.u32", &wl, vec![wlo.clone(), x.clone()])?;
            self.emit_op("s.addc.u32", Some(&wh), &[Opd::Int(0), Opd::Int(0)], &[])?;
            self.sop("s.lshr.b64", &w, vec![Opd::Reg(w.clone(), 0), shr])?;
        } else {
            self.vop(
                "v.mul.hi.u32",
                &wl,
                vec![x.clone(), mul],
                Ty::U32,
                Vop::default(),
            )?;
            let xv = if x.vgpr() {
                x.clone()
            } else {
                self.vgpr_copy(&x, Ty::U32)?
            };
            let vcc = Opd::Phys(self.t.mask_reg(phys_named("vcc").expect("vcc")));
            self.emit_op(
                "v.add.co.u32",
                Some(&wl),
                &[vcc.clone(), wlo.clone(), xv],
                &[],
            )?;
            self.emit_op(
                "v.addc.co.u32",
                Some(&wh),
                &[vcc.clone(), Opd::Int(0), Opd::Int(0), vcc],
                &[],
            )?;
            self.vop(
                "v.lshrrev.b64",
                &w,
                vec![shr, Opd::Reg(w.clone(), 0)],
                Ty::U64,
                Vop::default(),
            )?;
        }
        match rem {
            false => self.mov(d, &wlo),
            true => {
                let c = super::node::leaf(super::node::Leafish::Value(dv))?;
                self.remainder(&n.args[0], &wl, c, d)
            }
        }
    }

    fn multiply(&mut self, n: &Tree, d: &Typed) -> Result<()> {
        let ops = self.operands(n, Some(Ty::U32))?;
        let s = d.file == 's';
        let vc = Vop {
            commute: true,
            ..Vop::default()
        };
        match n.op {
            Op::Mul24 if s => self.sop("s.mul.i32", d, ops),
            Op::Mul24 => self.vop("v.mul.u32.u24", d, ops, Ty::U32, vc),
            Op::Mad24 if s => {
                self.sop("s.mul.i32", d, ops[..2].to_vec())?;
                self.sop("s.add.u32", d, vec![Opd::Reg(d.clone(), 0), ops[2].clone()])
            }
            Op::Mad24 => self.vop("v.mad.u32.u24", d, ops, Ty::U32, Vop::default()),
            Op::Mulhi if s => self.sop("s.mul.hi.u32", d, ops),
            Op::Mulhi => self.vop("v.mul.hi.u32", d, ops, Ty::U32, Vop::default()),
            _ => {
                let (lo, hi) = (d.half_dword(0), d.half_dword(1));
                if s {
                    self.sop("s.mul.i32", &lo, ops.clone())?;
                    self.sop("s.mul.hi.u32", &hi, ops)
                } else {
                    self.vop("v.mul.lo.u32", &lo, ops.clone(), Ty::U32, Vop::default())?;
                    self.vop("v.mul.hi.u32", &hi, ops, Ty::U32, Vop::default())
                }
            }
        }
    }

    fn minmax(&mut self, n: &Tree, d: &Typed) -> Result<()> {
        let t = n.ty.unwrap_or(Ty::U32);
        let ops = self.operands(n, Some(t))?;
        let name = n.op.name();
        if d.file == 's' {
            if ops.len() == 3 || t == Ty::F32 {
                return expr(format!(
                    "{}: min3 / max3 / med3 are VALU ops: write it into a V value (in SGPRs: two \
                     min / max)",
                    top(n)
                ));
            }
            return self.sop(&format!("s.{name}.{}", t.name()), d, ops);
        }
        let commute = ops.len() == 2;
        self.vop(
            &format!("v.{name}.{}", t.name()),
            d,
            ops,
            t,
            Vop {
                commute,
                ..Vop::default()
            },
        )
    }

    /// A bit-field extract.
    pub(crate) fn bfe_rule(
        &mut self,
        x: &Tree,
        offset: u32,
        width: u32,
        signed: bool,
        d: &Typed,
    ) -> Result<()> {
        let t = if signed { Ty::I32 } else { Ty::U32 };
        let x = self.operand(x, Some(t))?;
        let iu = if signed { "i" } else { "u" };
        match d.file {
            's' => {
                let f = i128::from(offset) | i128::from(width) << 16;
                self.sop(&format!("s.bfe.{iu}32"), d, vec![x, Opd::Int(f)])
            }
            _ => {
                let srcs = vec![x, Opd::Int(offset.into()), Opd::Int(width.into())];
                self.vop(&format!("v.bfe.{iu}32"), d, srcs, t, Vop::default())
            }
        }
    }

    fn floats(&mut self, n: &Tree, d: &Typed) -> Result<()> {
        let vc = Vop {
            commute: true,
            ..Vop::default()
        };
        match (&n.op, &n.val) {
            (Op::Fma, _) if n.ty == Some(Ty::F32x2) => self.packed(n, d),
            (Op::Fma, _) => {
                let ops = self.operands(n, Some(Ty::F32))?;
                self.vop("v.fma.f32", d, ops, Ty::F32, Vop::default())
            }
            (Op::F1, Val::F1(f)) => {
                let x = self.operand(&n.args[0], Some(Ty::F32))?;
                let m = match *f {
                    "exp2" => "v_exp_f32",
                    "log2" => "v_log_f32",
                    "rcp" => "v_rcp_f32",
                    "rsq" => "v_rsq_f32",
                    _ => "v_sqrt_f32",
                };
                self.vop(m, d, vec![x], Ty::F32, Vop::default())
            }
            (Op::CvtF32, _) => {
                let it = n.args[0].ty.unwrap_or(Ty::U32);
                let x = self.operand(&n.args[0], Some(it))?;
                let op = if it == Ty::I32 {
                    "v.cvt.f32.i32"
                } else {
                    "v.cvt.f32.u32"
                };
                self.vop(op, d, vec![x], it, Vop::default())
            }
            (Op::CvtInt, _) => {
                let x = self.operand(&n.args[0], Some(Ty::F32))?;
                let op = if n.ty == Some(Ty::I32) {
                    "v.cvt.i32.f32"
                } else {
                    "v.cvt.u32.f32"
                };
                self.vop(op, d, vec![x], Ty::F32, Vop::default())
            }
            (Op::Bf16Lo, _) => {
                let w = self.operand(&n.args[0], Some(Ty::U32))?;
                self.vop(
                    "v.lshlrev.b32",
                    d,
                    vec![Opd::Int(16), w],
                    Ty::U32,
                    Vop::default(),
                )
            }
            (Op::Bf16Hi, _) => {
                let w = self.operand(&n.args[0], Some(Ty::U32))?;
                self.vop("v.and.b32", d, vec![Opd::Int(0xFFFF_0000), w], Ty::U32, vc)
            }
            _ => {
                self.spelled("v.cvt.pk.bf16.f32")?;
                let ops = self.operands(n, Some(Ty::F32))?;
                self.vop("v.cvt.pk.bf16.f32", d, ops, Ty::F32, Vop::default())
            }
        }
    }

    /// A `neg`/`abs` f32 node lowered as a value.
    pub(crate) fn sign_bits(&mut self, n: &Tree, d: &Typed) -> Result<()> {
        if n.ty == Some(Ty::F32x2) {
            return self.packed_neg(n, d);
        }
        let (mut abs, mut neg) = (false, false);
        let mut inner = n.clone();
        while matches!(inner.op, Op::Neg | Op::Abs) {
            match inner.op {
                Op::Abs => abs = true,
                _ if !abs => neg = !neg,
                _ => {}
            }
            inner = inner.args[0].clone();
        }
        let x = self.operand(&inner, Some(Ty::F32))?;
        let vc = Vop {
            commute: true,
            ..Vop::default()
        };
        match (abs, neg) {
            (true, true) => self.vop("v.or.b32", d, vec![Opd::Int(0x8000_0000), x], Ty::U32, vc),
            (true, false) => self.vop("v.and.b32", d, vec![Opd::Int(0x7FFF_FFFF), x], Ty::U32, vc),
            (false, true) => self.vop("v.xor.b32", d, vec![Opd::Int(0x8000_0000), x], Ty::U32, vc),
            (false, false) => self.mov(d, &x),
        }
    }

    /// Packed f32x2 `+ - *` and `fma`.
    fn packed(&mut self, n: &Tree, d: &Typed) -> Result<()> {
        let op = match n.op {
            Op::Add | Op::Sub => "v.pk.add.f32",
            Op::Mul => "v.pk.mul.f32",
            _ => "v.pk.fma.f32",
        };
        let name = self.spelled(op)?;
        let mut negs = Vec::new();
        let mut srcs = Vec::new();
        for (k, a) in n.args.iter().enumerate() {
            let mut neg = n.op == Op::Sub && k == 1;
            let mut x = a.clone();
            while x.op == Op::Neg {
                neg = !neg;
                x = x.args[0].clone();
            }
            negs.push(i128::from(neg));
            srcs.push(self.lower(&x, None, Some(Ty::F32x2))?);
        }
        let mut fields = Vec::new();
        if negs.contains(&1) {
            let l = FArg::List(negs.iter().map(|&b| FArg::Int(b)).collect());
            fields = vec![("neg_lo".to_owned(), l.clone()), ("neg_hi".to_owned(), l)];
        }
        let mut all = vec![Opd::Reg(d.clone(), 0)];
        all.extend(srcs);
        self.emit_name(&name, &all, &fields)
    }

    fn packed_neg(&mut self, n: &Tree, d: &Typed) -> Result<()> {
        let mut count = 0;
        let mut x = n.clone();
        while x.op == Op::Neg {
            count += 1;
            x = x.args[0].clone();
        }
        let v = self.lower(&x, None, Some(Ty::F32x2))?;
        if count % 2 == 0 {
            return self.mov(d, &v);
        }
        let vc = Vop {
            commute: true,
            ..Vop::default()
        };
        for k in 0..2 {
            let half = match &v {
                Opd::Reg(t, _) => Opd::Reg(t.half_dword(k), 0),
                v => v.clone(),
            };
            self.vop(
                "v.xor.b32",
                &d.half_dword(k),
                vec![Opd::Int(0x8000_0000), half],
                Ty::U32,
                vc.clone(),
            )?;
        }
        Ok(())
    }

    fn pair(&mut self, n: &Tree, d: &Typed) -> Result<()> {
        let (a, b) = (&n.args[0], &n.args[1]);
        let (lo, hi) = (d.half_dword(0), d.half_dword(1));
        if self.reads(b, &lo) {
            let tmp = self.temp('v', Some(Ty::F32))?;
            self.lower(b, Some(&tmp), Some(Ty::F32))?;
            self.lower(a, Some(&lo), Some(Ty::F32))?;
            return self.mov(&hi, &Opd::Reg(tmp, 0));
        }
        self.lower(a, Some(&lo), Some(Ty::F32))?;
        self.lower(b, Some(&hi), Some(Ty::F32)).map(|_| ())
    }

    /// Whether any leaf of a tree is on registers overlapping r.
    fn reads(&self, n: &Tree, r: &Typed) -> bool {
        match super::node::value_of(n) {
            Some(v) => v.value == r.value && v.sub < r.sub + r.width() && r.sub < v.sub + v.width(),
            None => n.args.iter().any(|a| self.reads(a, r)),
        }
    }
}

enum Rewrite {
    Tree(Tree),
    Shift(Tree, Amount, Tree, bool),
    ScalarShift(Tree, u32, Tree),
    Bfe(Tree, u32, u32),
}

/// A constant of a wide `+` / `-`, taken whole:
/// its halves of v = c mod 2^64, each as its signed value when -16.. -1 (an
/// inline constant), else unsigned.
fn wide_constant(c: Num, b: &Tree) -> Result<(Opd, Opd)> {
    let Num::Int(c) = c else {
        return expr(format!("{}: a float constant where a u64 is", num(c)));
    };
    if !(-(1i128 << 63)..=(1i128 << 64) - 1).contains(&c) {
        return expr(format!(
            "{}: a constant outside the u64 range (-2^63 .. 2^64 - 1)",
            top(b)
        ));
    }
    let v = c.rem_euclid(1 << 64);
    let half = |h: i128| match h - (1 << 32) {
        s if (-16..=-1).contains(&s) => Opd::Int(s),
        _ => Opd::Int(h),
    };
    Ok((half(v & 0xFFFF_FFFF), half(v >> 32)))
}

fn halves(x: &Opd) -> (Opd, Opd) {
    match x {
        Opd::Reg(v, _) => (Opd::Reg(v.half_dword(0), 0), Opd::Reg(v.half_dword(1), 0)),
        Opd::Int(c) => (Opd::Int(c & 0xFFFF_FFFF), Opd::Int(c >> 32 & 0xFFFF_FFFF)),
        x => (x.clone(), Opd::Int(0)),
    }
}

/// magic(c): (m, s, short).
pub fn magic_triple(c: u64) -> (u64, u32, bool) {
    for s in 0..32u32 {
        let p = 1u128 << (32 + s);
        let m = p.div_ceil(u128::from(c));
        if m >= 1 << 32 {
            break;
        }
        if m * u128::from(c) - p <= 1u128 << s {
            return (m as u64, s, true);
        }
    }
    let s = 64 - (c - 1).leading_zeros();
    let m = ((1u128 << 32) * ((1u128 << s) - u128::from(c)) / u128::from(c)) + 1;
    (m as u64, s, false)
}
