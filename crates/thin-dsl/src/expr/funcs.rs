//! The expression functions and casts.

use super::node::{binop, constant, node, shared, Fmt, Num, Op, Tree, Val};
use super::text::top;
use crate::error::{expr, Result};
use crate::maker::FArg;
use crate::types::Ty;
use crate::values::Typed;

/// An integer operand: an int type or a constant.
fn int(f: &str, x: &Tree) -> Result<()> {
    match x.ty {
        None => Ok(()),
        Some(t) if t.is_int() => Ok(()),
        _ => expr(format!("{f}({}): an integer's (u32, i32)", top(x))),
    }
}

/// A float operand: f32 or a constant.
fn flt(f: &str, x: &Tree) -> Result<()> {
    match x.ty {
        None | Some(Ty::F32) => Ok(()),
        _ => expr(format!("{f}({}): an f32's", top(x))),
    }
}

/// What a cast gives: a node, or (`f32x2`) the same registers.
#[derive(Clone, Debug)]
pub enum Cast {
    /// A tree.
    Tree(Tree),
    /// A typed value.
    Value(Typed),
}

/// `f32x2(x)` of a typed value; `reg` is its register's number
/// (its group's first value's pin, or 0, plus `sub`).
pub fn cast_pair(x: &Typed, reg: u32) -> Result<Cast> {
    if x.ty == Ty::F32x2 {
        return Ok(Cast::Value(x.clone()));
    }
    if !(matches!(x.ty, Ty::U32 | Ty::I32 | Ty::F32) && x.width() == 2 && x.file == 'v') {
        return expr(format!(
            "f32x2({}): two dword elements of a VGPR value (f32x2(acc[2:4]))",
            x.repr()
        ));
    }
    if reg % 2 == 1 {
        return expr(format!(
            "f32x2({}): at an odd register (a pair is an even-aligned VGPR pair)",
            x.repr()
        ));
    }
    Ok(Cast::Value(Typed {
        ty: Ty::F32x2,
        n: 1,
        divisor: false,
        ..x.clone()
    }))
}

/// `T(x)` for a tree.
pub fn cast(t: Ty, x: Tree) -> Result<Tree> {
    if t == Ty::F32x2 {
        return expr(format!(
            "f32x2({}): two dword elements of a VGPR value (f32x2(acc[2:4]))",
            top(&x)
        ));
    }
    let xt = x.ty;
    if xt == Some(Ty::F32x2) {
        return expr(format!(
            "{}({}): an f32x2's halves are .lo and .hi",
            t.name(),
            top(&x)
        ));
    }
    let Some(xt) = xt else {
        let c = super::node::const_of(&x).expect("an untyped constant");
        let mut n = (*constant(c)).clone();
        n.ty = Some(t);
        return Ok(std::sync::Arc::new(n));
    };
    if xt == t {
        return Ok(x);
    }
    if xt.is_lanes() || t.is_mask() || xt.is_wide() != t.is_wide() {
        return expr(format!(
            "{}({}): {} is not {}'s width",
            t.name(),
            top(&x),
            xt.a(),
            t.a()
        ));
    }
    node(Op::Cast, Some(t), vec![x], Val::None)
}

/// `select(c, a, b)`.
pub fn select(c: Tree, a: Tree, b: Tree) -> Result<Tree> {
    if !c.ty.is_some_and(Ty::is_lanes) {
        return expr(format!("select({}, ..): a condition first", top(&c)));
    }
    let subject = || format!("select(.., {}, {})", top(&a), top(&b));
    let t = shared(a.ty, b.ty, &subject)?;
    if t == Some(Ty::F32x2) {
        return expr(format!(
            "{}: of dwords (an f32x2: .lo and .hi apart)",
            subject()
        ));
    }
    let t = t.or_else(|| {
        let float = [&a, &b]
            .iter()
            .any(|x| matches!(super::node::const_of(x), Some(Num::Float(_))));
        Some(if float { Ty::F32 } else { Ty::U32 })
    });
    node(Op::Select, t, vec![c, a, b], Val::None)
}

/// `mul24`, `mad24`, `mulhi`, `mul_wide`.
pub fn multiply(op: Op, args: Vec<Tree>) -> Result<Tree> {
    for a in &args {
        int(op.name(), a)?;
    }
    let t = if op == Op::MulWide { Ty::U64 } else { Ty::U32 };
    node(op, Some(t), args, Val::None)
}

/// `min`, `max` (any count), `min3`, `max3`, `med3`.
pub fn minmax(op: Op, args: Vec<Tree>) -> Result<Tree> {
    let mut t = None;
    for a in &args {
        let subject = || op.name().to_owned();
        t = shared(t, a.ty, &subject)?;
    }
    if !t.is_some_and(|t| t.is_int() || t == Ty::F32) {
        return expr(format!("{}: of numbers (u32, i32, f32)", op.name()));
    }
    // min / max of more than two: three as min3 / max3, more as a chain.
    if matches!(op, Op::Min | Op::Max) && args.len() > 2 {
        let three = if op == Op::Min { Op::Min3 } else { Op::Max3 };
        if args.len() == 3 {
            return node(three, t, args, Val::None);
        }
        let mut args = args;
        let rest = args.split_off(2);
        let mut acc = node(op, t, args, Val::None)?;
        for a in rest {
            acc = node(op, t, vec![acc, a], Val::None)?;
        }
        return Ok(acc);
    }
    node(op, t, args, Val::None)
}

/// `fma(a, b, c)`.
pub fn fma(args: Vec<Tree>) -> Result<Tree> {
    let pairs = args.iter().filter(|a| a.ty == Some(Ty::F32x2)).count();
    match pairs {
        3 => node(Op::Fma, Some(Ty::F32x2), args, Val::None),
        0 => {
            for a in &args {
                flt("fma", a)?;
            }
            node(Op::Fma, Some(Ty::F32), args, Val::None)
        }
        _ => {
            let t: Vec<String> = args.iter().map(|a| top(a)).collect();
            expr(format!("fma({}): three f32x2, or none", t.join(", ")))
        }
    }
}

/// `exp2`, `log2`, `rcp`, `rsq`, `sqrt`.
pub fn f1(name: &'static str, x: Tree) -> Result<Tree> {
    flt(name, &x)?;
    node(Op::F1, Some(Ty::F32), vec![x], Val::F1(name))
}

/// `cvt_f32`, `cvt_u32`, `cvt_i32`, `bf16_lo`, `bf16_hi`, `pk_bf16`, `pair`.
pub fn convert(f: &str, args: Vec<Tree>) -> Result<Tree> {
    let (op, ty, ints) = match f {
        "cvt_f32" => (Op::CvtF32, Ty::F32, true),
        "cvt_u32" => (Op::CvtInt, Ty::U32, false),
        "cvt_i32" => (Op::CvtInt, Ty::I32, false),
        "bf16_lo" => (Op::Bf16Lo, Ty::F32, true),
        "bf16_hi" => (Op::Bf16Hi, Ty::F32, true),
        "pk_bf16" => (Op::PkBf16, Ty::U32, false),
        _ => (Op::Pair, Ty::F32x2, false),
    };
    for a in &args {
        if ints {
            int(f, a)?;
        } else {
            flt(f, a)?;
        }
    }
    node(op, Some(ty), args, Val::None)
}

/// `popcount(x)`.
pub fn popcount(x: Tree) -> Result<Tree> {
    if x.ty.is_some_and(Ty::is_lanes) {
        return node(Op::PopcountMask, Some(Ty::U32), vec![x], Val::None);
    }
    int("popcount", &x)?;
    node(Op::Popcount, Some(Ty::U32), vec![x], Val::None)
}

/// `first(m)`.
pub fn first(m: Tree) -> Result<Tree> {
    if !m.ty.is_some_and(Ty::is_lanes) {
        return expr(format!("first({}): of a lane mask", top(&m)));
    }
    node(Op::First, Some(Ty::U32), vec![m], Val::None)
}

/// `any(c)`, `all(c)`.
pub fn any_all(op: Op, c: Tree) -> Result<Tree> {
    node(op, Some(Ty::Cond), vec![c], Val::None)
}

/// `first_lane`, `lane_read`, `lane_write`, `shuffle`, `rank`.
pub fn lane(op: Op, args: Vec<Tree>) -> Result<Tree> {
    if op == Op::Shuffle {
        int("shuffle", &args[1])?;
    }
    let ty = if op == Op::Rank {
        Some(Ty::U32)
    } else {
        args[0].ty
    };
    node(op, ty, args, Val::None)
}

/// `across(x, op, lanes)` and `scan(x, op, lanes)`.
pub fn lanes_op(scan: bool, x: Tree, op: &str, lanes: i128) -> Result<Tree> {
    let name: &'static str = match op {
        "max" => "max",
        "min" => "min",
        "add" => "add",
        _ => {
            let f = if scan { "scan" } else { "across" };
            return expr(format!(
                "{f}(.., {}): max, min, add",
                crate::maker::py_repr(op)
            ));
        }
    };
    let ok: &[i128] = if scan {
        &[16, 32, 64]
    } else {
        &[2, 4, 8, 16, 32, 64]
    };
    if !ok.contains(&lanes) {
        return match scan {
            true => expr(format!("scan(lanes={lanes}): 16, 32 or 64")),
            false => expr(format!("across(lanes={lanes}): 2, 4, 8, 16, 32 or 64")),
        };
    }
    flt(if scan { "scan" } else { "across" }, &x)?;
    let op = if scan { Op::Scan } else { Op::Across };
    node(op, Some(Ty::F32), vec![x], Val::Lanes(name, lanes as u32))
}

/// `bfe(a, offset, width)`.
pub fn bfe(a: Tree, offset: u32, width: u32, signed: Option<bool>) -> Result<Tree> {
    int("bfe", &a)?;
    let t = a.ty.or(Some(Ty::U32));
    node(Op::Bfe, t, vec![a], Val::Bfe(offset, width, signed))
}

/// A storage format argument: a name, or an object's `.name`.
pub struct FmtArg {
    /// Its name.
    pub name: String,
    /// `at=`: none, an exact int, or another value by its repr.
    pub at: Option<std::result::Result<i128, String>>,
}

fn check_fmt(f: &str, name: &str, takes: &[&str]) -> Result<()> {
    if name == "bf16" {
        return expr(format!(
            "{f}(.., bf16): bf16 is bf16_lo / bf16_hi / pk_bf16 (shifts, v_cvt_pk_bf16_f32)"
        ));
    }
    if !takes.contains(&name) {
        return expr(format!(
            "{f}(.., {}): the formats it converts are {}",
            crate::maker::py_repr(name),
            takes.join(", ")
        ));
    }
    Ok(())
}

/// The refusal of an element index `at` past `top_at`.
fn at_range(f: &str, at: &str, top_at: i128) -> String {
    if top_at == 0 {
        format!("{f}(at={at}): 0 only (the elements from the dword's low bits)")
    } else {
        format!("{f}(at={at}): the element of the dword, 0 .. {top_at}")
    }
}

fn check_at(f: &str, at: &Option<std::result::Result<i128, String>>, top_at: i128) -> Result<u32> {
    match at {
        None => Ok(0),
        Some(Ok(k)) if (0..=top_at).contains(k) => Ok(*k as u32),
        Some(Ok(k)) => expr(at_range(f, &k.to_string(), top_at)),
        Some(Err(r)) => expr(at_range(f, r, top_at)),
    }
}

/// `unpack(w, fmt, at=)` and `unpack2`.
pub fn unpack(pair: bool, w: Tree, fmt: FmtArg) -> Result<Tree> {
    let f = if pair { "unpack2" } else { "unpack" };
    let takes: &[&str] = if pair {
        &["e4m3fn", "e5m2"]
    } else {
        &["f16", "e4m3fn", "e5m2"]
    };
    check_fmt(f, &fmt.name, takes)?;
    // The 8-bit formats convert from the dword's low byte (unpack) or low
    // half (unpack2) only.
    let top_at = match (pair, fmt.name.as_str()) {
        (false, "f16") => 1,
        _ => 0,
    };
    let at = check_at(f, &fmt.at, top_at)?;
    int(f, &w)?;
    let args = vec![w];
    let ty = if pair { Ty::F32x2 } else { Ty::F32 };
    let val = Val::Fmt(Fmt {
        name: fmt.name,
        at,
        into: false,
    });
    node(
        if pair { Op::Unpack2 } else { Op::Unpack },
        Some(ty),
        args,
        val,
    )
}

/// `pack2(a, b, fmt, into=, at=)`.
pub fn pack2(a: Tree, b: Tree, fmt: FmtArg, into: Option<Tree>) -> Result<Tree> {
    check_fmt("pack2", &fmt.name, &["f16", "e4m3fn", "e5m2"])?;
    if fmt.name == "f16" {
        let at_set = matches!(fmt.at, Some(Ok(k)) if k != 0) || matches!(fmt.at, Some(Err(_)));
        if into.is_some() || at_set {
            return expr("pack2(.., f16): a whole dword (no into=, at=)");
        }
        flt("pack2", &a)?;
        flt("pack2", &b)?;
        return node(
            Op::Pack2,
            Some(Ty::U32),
            vec![a, b],
            Val::Fmt(Fmt {
                name: fmt.name,
                at: 0,
                into: false,
            }),
        );
    }
    let Some(into) = into else {
        return expr(format!(
            "pack2(.., {}): into= the dword it is written into (its other bits kept)",
            fmt.name
        ));
    };
    let at = check_at("pack2", &fmt.at, 1)?;
    flt("pack2", &a)?;
    flt("pack2", &b)?;
    int("pack2", &into)?;
    let ty = into.ty.unwrap_or(Ty::U32);
    let args = vec![a, b, into];
    let val = Val::Fmt(Fmt {
        name: fmt.name,
        at,
        into: true,
    });
    node(Op::Pack2, Some(ty), args, val)
}

/// `op(name, *args, ty=, **fields)`.
pub fn instr(
    name: &str,
    args: Vec<Tree>,
    ty: Option<Ty>,
    fields: Vec<(String, FArg)>,
) -> Result<Tree> {
    let ty = match ty {
        Some(t) => t,
        None => match name.rsplit('_').next().filter(|_| name.contains('_')) {
            Some("f32") => Ty::F32,
            Some("u32" | "b32") => Ty::U32,
            Some("i32") => Ty::I32,
            Some("b64" | "u64") => Ty::U64,
            _ => {
                return expr(format!(
                    "op({}): its result's type (ty=)",
                    crate::maker::py_repr(name)
                ))
            }
        },
    };
    node(
        Op::Instr,
        Some(ty),
        args,
        Val::Instr(name.to_owned(), fields),
    )
}

/// `bf16(x)` = `bf16_lo(pk_bf16(x, 0.0))`.
pub fn bf16(x: Tree) -> Result<Tree> {
    let pk = convert("pk_bf16", vec![x, constant(Num::Float(0.0))])?;
    convert("bf16_lo", vec![pk])
}

/// `x op= t`: the tree `x op t` (a statement writes it into x).
pub fn update(op: Op, x: Tree, t: Tree) -> Result<Tree> {
    binop(op, x, t)
}

/// The functions over operand lists by name: `mul24`, `mad24`, `mulhi`,
/// `mul_wide`, `min` … `med3`, `fma`, `popcount`, `first`, `any`, `all`,
/// the lane functions, `bf16`, the float functions and the conversions.
pub fn call(name: &str, a: Vec<Tree>) -> Result<Tree> {
    let one = |a: &[Tree]| a[0].clone();
    match name {
        "mul24" | "mad24" | "mulhi" | "mul_wide" => {
            multiply(Op::named(name).expect("an operator"), a)
        }
        "min" | "max" | "min3" | "max3" | "med3" => {
            minmax(Op::named(name).expect("an operator"), a)
        }
        "fma" => fma(a),
        "popcount" => popcount(one(&a)),
        "first" => first(one(&a)),
        "any" => any_all(Op::Any, one(&a)),
        "all" => any_all(Op::All, one(&a)),
        "first_lane" | "lane_read" | "lane_write" | "shuffle" | "rank" => {
            lane(Op::named(name).expect("an operator"), a)
        }
        "bf16" => bf16(one(&a)),
        "exp2" => f1("exp2", one(&a)),
        "log2" => f1("log2", one(&a)),
        "rcp" => f1("rcp", one(&a)),
        "rsq" => f1("rsq", one(&a)),
        "sqrt" => f1("sqrt", one(&a)),
        _ => convert(name, a),
    }
}
