//! Statements and the lowering of a tree.

use thin_ir::{Attr, FieldId, FieldValue, Imm, ImmValue, Instr, Operand};
use thin_target::Phys;

use super::node::{const_of, value_of, Node, Num, Op, Tree, Val};
use super::text::{num, top};
use crate::builder::Builder;
use crate::error::{err, expr, Class, Result};
use crate::maker::{Arg, FArg};
use crate::types::Ty;
use crate::values::Typed;

/// An instruction's source operand, or a lowered result.
#[derive(Clone, Debug, PartialEq)]
pub enum Opd {
    /// A register and its modifier byte (neg 1, abs 2; [`MODIFIED`]
    /// marks an operand a modifier was applied to, even one that cancels).
    Reg(Typed, u8),
    /// A named register.
    Phys(&'static Phys),
    /// An int.
    Int(i128),
    /// An inline float by its text.
    Text(&'static str),
    /// A raw float (an untyped constant with no type).
    Float(f64),
}

impl Opd {
    /// Whether it is a register of file `v`.
    pub fn vgpr(&self) -> bool {
        matches!(self, Opd::Reg(v, _) if v.file == 'v')
    }

    /// Whether it is a register of file `s`, or a named register.
    pub fn sgpr(&self) -> bool {
        matches!(self, Opd::Reg(v, _) if v.file == 's') || matches!(self, Opd::Phys(_))
    }

    /// Whether it is a literal for a VALU op.
    pub fn literal(&self) -> bool {
        match self {
            Opd::Int(v) => {
                let s = if *v >= 1 << 31 { *v - (1 << 32) } else { *v };
                !(-16..=64).contains(&s)
            }
            _ => false,
        }
    }

    /// Whether it is inline: an inline float's text, or a non-literal int.
    pub fn inline(&self) -> bool {
        matches!(self, Opd::Text(_)) || matches!(self, Opd::Int(_)) && !self.literal()
    }

    /// Its modifier byte.
    pub fn mods(&self) -> u8 {
        match self {
            Opd::Reg(_, m) => *m & 3,
            _ => 0,
        }
    }

    /// Whether it carries a neg or abs modifier.
    pub fn modified(&self) -> bool {
        // A modifier that cancels (`--f`) sets no neg or abs bit, so the
        // operand takes the short form as `f` would.
        self.mods() != 0
    }

    /// The typed value, if a register.
    pub fn value(&self) -> Option<&Typed> {
        match self {
            Opd::Reg(v, _) => Some(v),
            _ => None,
        }
    }

    /// The IR operand.
    pub fn operand(&self) -> Operand {
        match self {
            Opd::Reg(v, _) => Operand::Reg(v.reg()),
            Opd::Phys(p) => Operand::Phys(p),
            Opd::Int(v) => Operand::Imm(Imm::int(*v)),
            Opd::Text(t) => Operand::Imm(Imm {
                value: ImmValue::Float(t.parse().expect("an inline float")),
                text: "".into(),
            }),
            Opd::Float(x) => Operand::Imm(Imm {
                value: ImmValue::Float(*x),
                text: "".into(),
            }),
        }
    }

    /// As a maker argument (for the maker's refusal).
    pub(crate) fn arg(&self) -> Arg {
        match self {
            Opd::Reg(v, m) => Arg::Reg(v.reg(), *m & 3),
            Opd::Phys(p) => Arg::Phys(p, 0),
            Opd::Int(v) => Arg::Int(*v),
            Opd::Text(t) => Arg::Str((*t).to_owned()),
            Opd::Float(x) => Arg::Float(*x),
        }
    }
}

/// The bit of an operand's modifier byte that marks it modified.
pub const MODIFIED: u8 = 4;

/// The inline f32 constants by value and text.
const INLINE: [(f64, &str); 8] = [
    (0.5, "0.5"),
    (-0.5, "-0.5"),
    (1.0, "1.0"),
    (-1.0, "-1.0"),
    (2.0, "2.0"),
    (-2.0, "-2.0"),
    (4.0, "4.0"),
    (-4.0, "-4.0"),
];

/// A constant as an operand of type t; none: the raw constant.
pub fn const_operand(c: Num, t: Option<Ty>) -> Result<Opd> {
    match (t, c) {
        (Some(Ty::F32), c) => {
            let x = c.as_f64();
            if let Some((_, text)) = INLINE.iter().find(|(v, _)| *v == x) {
                return Ok(Opd::Text(text));
            }
            if x == 0.0 {
                // The sign by its bit: -0.0 is the literal 0x80000000 (no target has an
                // inline -0.0), never +0.
                return Ok(Opd::Int(if x.is_sign_negative() { 0x8000_0000 } else { 0 }));
            }
            let f = x as f32;
            if f.is_infinite() && x.is_finite() {
                return err(Class::Overflow, "float too large to pack with f format");
            }
            Ok(Opd::Int(i128::from(f.to_bits())))
        }
        (Some(t), Num::Float(_)) => {
            expr(format!("{}: a float constant where {} is", num(c), t.a()))
        }
        (Some(t), Num::Int(v)) if t.is_wide() => {
            let v = v.rem_euclid(1 << 64);
            match v {
                v if v < 1 << 31 => Ok(Opd::Int(v)),
                v if v >= (1 << 64) - 16 => Ok(Opd::Int(v - (1 << 64))),
                v => expr(format!(
                    "0x{v:x}: a {} constant past 2^31 has no 64-bit operand (its literal is 32 bits) - set its halves: \
                     x.lo, x.hi",
                    t.name()
                )),
            }
        }
        (Some(_), Num::Int(v)) => Ok(Opd::Int(v & 0xFFFF_FFFF)),
        (None, Num::Int(v)) => Ok(Opd::Int(v)),
        (None, Num::Float(x)) => Ok(Opd::Float(x)),
    }
}

/// The kept side of an identity, if n is one.
pub fn identity(n: &Node) -> Option<&Tree> {
    let t = n.ty?;
    if t.is_float() || n.args.len() != 2 {
        return None;
    }
    if t.is_wide() && !matches!(n.op, Op::Add | Op::Sub) {
        return None;
    }
    let (a, b) = (&n.args[0], &n.args[1]);
    // A wide x +- 0 is an identity only for a whole int 0: a float is refused.
    let is = |x: &Tree, v: i128| {
        const_of(x).is_some_and(|c| c.is(v) && !(t.is_wide() && matches!(c, Num::Float(_))))
    };
    match n.op {
        Op::Mul if is(b, 1) => Some(a),
        Op::Mul if is(a, 1) => Some(b),
        Op::And if t.is_int() && (is(b, -1) || is(b, 0xFFFF_FFFF)) => Some(a),
        Op::And if t.is_int() && (is(a, -1) || is(a, 0xFFFF_FFFF)) => Some(b),
        Op::And | Op::Mul => None,
        Op::Add | Op::Sub | Op::Or | Op::Xor | Op::Shl | Op::Shr if is(b, 0) => Some(a),
        Op::Add | Op::Or | Op::Xor if is(a, 0) => Some(b),
        _ => None,
    }
}

impl Builder {
    /// `assign(dst, tree, comment)`: a statement.
    pub fn assign(&mut self, dst: &Typed, tree: Tree, comment: Option<String>) -> Result<()> {
        if dst.builder != self.id {
            return expr(format!("{} = ..: a value of a Builder's files", dst.repr()));
        }
        let text = top(&tree);
        if dst.n != 1 {
            return expr(format!(
                "{} = {text}: a statement writes one element, not {} - each: for e in x: e[...] = ..",
                dst.repr(),
                dst.n
            ));
        }
        let lanes = tree.ty.is_some_and(Ty::is_lanes);
        if dst.ty.is_mask() && !lanes {
            return expr(format!("{} = {text}: a mask from a condition", dst.repr()));
        }
        if tree.ty == Some(Ty::Cond) && !dst.ty.is_mask() {
            return expr(format!(
                "{} = {text}: a condition is no {} (select(c, 1, 0); a mask: S.mask(c))",
                dst.repr(),
                dst.ty.name()
            ));
        }
        if let Some(tt) = tree.ty.filter(|_| !dst.ty.is_mask()) {
            if tt != dst.ty && !(tt.is_wide() && dst.ty.is_wide()) {
                return expr(format!(
                    "{} = {text}: {} into {} (cast: {}(x))",
                    dst.repr(),
                    tt.a(),
                    dst.ty.a(),
                    dst.ty.name()
                ));
            }
        }
        let who = dst.label.as_deref().unwrap_or(&dst.group).to_owned();
        let comment = comment.unwrap_or_else(|| match dst.ty.is_mask() {
            true => format!("{who} = {text}"),
            false => format!("{who}{} = {text}", dst.part),
        });
        self.stmt_comment = Some(comment);
        let out = match dst.ty.is_mask() {
            true => self.into_mask(&tree, Some(dst.clone())).map(|_| ()),
            false => self.lower(&tree, Some(dst), Some(dst.ty)).map(|_| ()),
        };
        self.stmt_comment = None;
        out
    }

    /// `lower(n, dst, t)`.
    pub fn lower(&mut self, n: &Tree, dst: Option<&Typed>, t: Option<Ty>) -> Result<Opd> {
        let t = n.ty.or(t);
        match (&n.op, &n.val) {
            (Op::Const, Val::Const(c)) => {
                let x = const_operand(*c, t)?;
                return self.land_in(dst, x);
            }
            (Op::Val, Val::Value(v)) => {
                if dst.is_some_and(|d| d.file == 's') && v.file == 'v' {
                    return expr(format!(
                        "{}: a vector value into an SGPR (v_readfirstlane is never implied: first_lane(x))",
                        top(n)
                    ));
                }
                let x = Opd::Reg(v.clone(), 0);
                return self.land_in(dst, x);
            }
            (Op::Special, Val::Special(p)) => return self.land_in(dst, Opd::Phys(p)),
            (Op::Cast, _) => return self.lower_cast(n, dst),
            _ => {}
        }
        let file = n.file.unwrap_or('s');
        match dst {
            Some(d) if d.file == 'v' && file == 's' => {
                let x = self.lower(n, None, t)?;
                self.mov(d, &x)?;
                return Ok(Opd::Reg(d.clone(), 0));
            }
            Some(d) if d.file == 's' && file == 'v' => {
                if !vector_leaf(n) {
                    return expr(format!(
                        "{}: of scalar operands, but the DSL computes it on the VALU (f32 arithmetic, \
                         conversions and lane functions are VALU ops): write it into a V value, \
                         and one lane of that into an SGPR with first_lane(x)",
                        top(n)
                    ));
                }
                return expr(format!(
                "{}: a vector value into an SGPR (v_readfirstlane is never implied: first_lane(x))",
                top(n)
            ));
            }
            _ => {}
        }
        let d = match dst {
            Some(d) => d.clone(),
            None => self.temp(
                file,
                Some(if n.ty == Some(Ty::Cond) {
                    Ty::Mask
                } else {
                    n.ty.unwrap_or(Ty::U32)
                }),
            )?,
        };
        self.rule(n, &d)?;
        Ok(Opd::Reg(d, 0))
    }

    fn land_in(&mut self, dst: Option<&Typed>, x: Opd) -> Result<Opd> {
        match dst {
            Some(d) => {
                self.mov(d, &x)?;
                Ok(Opd::Reg(d.clone(), 0))
            }
            None => Ok(x),
        }
    }

    fn lower_cast(&mut self, n: &Tree, dst: Option<&Typed>) -> Result<Opd> {
        let inner = &n.args[0];
        let cast_ty = n.ty.expect("a cast's type");
        if let (Some(d), Some(it)) = (dst, inner.ty) {
            if !matches!(inner.op, Op::Val | Op::Const | Op::Special) {
                let r = d.retyped(it);
                self.lower(inner, Some(&r), Some(it))?;
                return Ok(Opd::Reg(d.clone(), 0));
            }
        }
        let x = self.lower(inner, None, inner.ty)?;
        match x {
            Opd::Reg(v, m) => {
                let v = v.retyped(cast_ty);
                match dst {
                    Some(d) => {
                        self.mov(d, &Opd::Reg(v, m))?;
                        Ok(Opd::Reg(d.clone(), 0))
                    }
                    None => Ok(Opd::Reg(v, m)),
                }
            }
            x => Ok(x),
        }
    }

    /// `operand(n, t)`.
    pub fn operand(&mut self, n: &Tree, t: Option<Ty>) -> Result<Opd> {
        let t = n.ty.or(t);
        if matches!(n.op, Op::Neg | Op::Abs) && t == Some(Ty::F32) {
            let mut x = self.operand(&n.args[0], Some(Ty::F32))?;
            if !matches!(x, Opd::Reg(..)) {
                x = self.vgpr_copy(&x, Ty::F32)?;
            }
            let m = MODIFIED | if n.op == Op::Neg { x.mods() ^ 1 } else { 2 };
            return Ok(match x {
                Opd::Reg(v, _) => Opd::Reg(v, m),
                x => x,
            });
        }
        if let Some(c) = const_of(n) {
            return const_operand(c, t);
        }
        if let Some(kept) = identity(n) {
            let kept = kept.clone();
            return self.operand(&kept, t);
        }
        self.lower(n, None, t)
    }

    /// A copy of a modified or constant operand in a VGPR, without its
    /// modifier.
    pub(crate) fn vgpr_copy_keep(&mut self, x: &Opd, t: Ty) -> Result<Opd> {
        let plain = match x {
            Opd::Reg(v, _) => Opd::Reg(v.clone(), 0),
            x => x.clone(),
        };
        let tmp = self.temp('v', Some(t))?;
        self.mov(&tmp, &plain)?;
        let m = if let Opd::Reg(_, m) = x { *m } else { 0 };
        Ok(Opd::Reg(tmp, m))
    }

    /// Each operand node's operand, in order.
    pub fn operands(&mut self, n: &Node, t: Option<Ty>) -> Result<Vec<Opd>> {
        n.args.iter().map(|a| self.operand(a, t)).collect()
    }

    /// `move(dst, x)`.
    pub fn mov(&mut self, d: &Typed, x: &Opd) -> Result<()> {
        if let Opd::Reg(v, 0) = x {
            if v.value == d.value && v.sub == d.sub && v.builder == d.builder {
                return Ok(());
            }
        }
        if d.width() == 2 {
            if d.file == 's' {
                return self.emit_op("s.mov.b64", Some(d), std::slice::from_ref(x), &[]);
            }
            for k in 0..2 {
                let half = d.half_dword(k);
                let src = match x {
                    Opd::Reg(v, m) => Opd::Reg(v.half_dword(k), *m),
                    Opd::Int(c) => Opd::Int(c >> (32 * k) & 0xFFFF_FFFF),
                    x => x.clone(),
                };
                self.emit_op("v.mov.b32", Some(&half), &[src], &[])?;
            }
            return Ok(());
        }
        let op = if d.file == 'v' {
            "v.mov.b32"
        } else {
            "s.mov.b32"
        };
        self.emit_op(op, Some(d), std::slice::from_ref(x), &[])
    }

    /// `vgpr_copy(x, t)`: a fresh VGPR temporary holding x.
    pub fn vgpr_copy(&mut self, x: &Opd, t: Ty) -> Result<Opd> {
        let t = if t.is_lanes() { Ty::U32 } else { t };
        let tmp = self.temp('v', Some(t))?;
        self.mov(&tmp, x)?;
        Ok(Opd::Reg(tmp, 0))
    }

    /// `sgpr_copy(x, t)`: a fresh SGPR temporary holding x.
    pub fn sgpr_copy(&mut self, x: &Opd, t: Ty) -> Result<Opd> {
        let t = if t.is_lanes() { Ty::U32 } else { t };
        let tmp = self.temp('s', Some(t))?;
        self.mov(&tmp, x)?;
        Ok(Opd::Reg(tmp, 0))
    }

    /// Spells a dotted op; a missing spelling is an ExprError.
    pub fn spelled(&self, op: &str) -> Result<String> {
        match self.t.spell(op) {
            Some(m) => Ok(m),
            None => expr(format!("{op}: {} has no such op", self.t.name())),
        }
    }

    /// Emits `op dst, srcs` as is: from the form, or the maker's refusal
    /// when no form has that operand count.
    pub fn emit_op(
        &mut self,
        op: &str,
        dst: Option<&Typed>,
        srcs: &[Opd],
        fields: &[(String, FArg)],
    ) -> Result<()> {
        let name = self.spelled(op)?;
        let mut all: Vec<Opd> = dst.map(|d| Opd::Reg(d.clone(), 0)).into_iter().collect();
        all.extend(srcs.iter().cloned());
        self.emit_name(&name, &all, fields)
    }

    /// Emits a mnemonic over operands, the destination first.
    pub fn emit_name(&mut self, name: &str, all: &[Opd], fields: &[(String, FArg)]) -> Result<()> {
        let Some(form) = self.t.form(name, all.len(), false) else {
            let args = all.iter().map(Opd::arg).collect();
            self.make(name, args, fields.to_vec(), crate::maker::Notes::default())?;
            return expr(format!("{name}: no form"));
        };
        let ops = all.iter().map(Opd::operand).collect();
        let mods: Vec<u8> = all.iter().map(Opd::mods).collect();
        let mods = mods
            .iter()
            .any(|&m| m != 0)
            .then(|| mods.into_boxed_slice());
        let mut fv = Vec::new();
        for (k, v) in fields {
            let id = FieldId::named(k)?;
            fv.push((id, field_value(v)));
        }
        let i = Instr::new(form, ops, fv, mods);
        self.emit_lowered(i)
    }

    /// Emits a lowering's instruction: the statement's comment on its first.
    pub fn emit_lowered(&mut self, mut i: Instr) -> Result<()> {
        if let Some(c) = self.stmt_comment.take() {
            if i.attrs().comment.is_none() {
                i = i.with_attr(Attr::Comment(c.as_str().into()))?;
            }
        }
        self.emit(i, &[]).map(|_| ())
    }
}

fn field_value(v: &FArg) -> FieldValue {
    match v {
        FArg::Bool(b) => FieldValue::Bool(*b),
        FArg::Int(x) => FieldValue::Int(*x),
        FArg::Str(s) => FieldValue::Text(s.as_str().into()),
        FArg::List(xs) => FieldValue::Ints(
            xs.iter()
                .filter_map(|x| if let FArg::Int(v) = x { Some(*v) } else { None })
                .collect(),
        ),
        FArg::None => FieldValue::Bool(false),
    }
}

impl Typed {
    /// Dword k of a two-dword value, as a u32.
    pub fn half_dword(&self, k: u32) -> Typed {
        let ty = if self.ty == Ty::F32x2 {
            Ty::F32
        } else {
            Ty::U32
        };
        Typed {
            ty,
            n: 1,
            sub: self.sub + k,
            divisor: false,
            ..self.clone()
        }
    }
}

/// The value leaf of a tree, if it is one.
pub fn leaf_value(n: &Node) -> Option<&Typed> {
    value_of(n)
}

/// Whether a tree reads a VGPR value or a per-lane result.
fn vector_leaf(n: &Tree) -> bool {
    match (&n.op, &n.val) {
        (Op::Val, Val::Value(v)) => v.file == 'v',
        (Op::Shuffle | Op::Rank | Op::Across | Op::Scan, _) => true,
        _ => n.args.iter().any(vector_leaf),
    }
}
