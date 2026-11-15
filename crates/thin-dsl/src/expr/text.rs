//! A tree's text: how it was written.

use super::node::{Divisor, Node, Num, Op, Val};
use crate::maker::py_float;
use crate::values::Typed;

/// A constant's text: ints of magnitude 4096 or more in hex.
pub fn num(c: Num) -> String {
    match c {
        Num::Int(v) if v.unsigned_abs() >= 4096 => {
            if v < 0 {
                format!("-0x{:x}", v.unsigned_abs())
            } else {
                format!("0x{v:x}")
            }
        }
        Num::Int(v) => v.to_string(),
        Num::Float(x) => py_float(x),
    }
}

/// A typed value's text in a tree: its label and part, or its repr.
pub fn value(v: &Typed) -> String {
    match &v.label {
        Some(l) if *v.part == *format!(".{l}") => l.to_string(),
        Some(l) => format!("{l}{}", v.part),
        None => v.repr(),
    }
}

/// A node's text, parenthesized when infix.
pub fn inner(n: &Node) -> String {
    text(n, false)
}

/// A node's text at the top level: an infix node's outer parentheses
/// dropped.
pub fn top(n: &Node) -> String {
    text(n, true)
}

fn args(n: &Node) -> Vec<String> {
    n.args.iter().map(|a| top(a)).collect()
}

fn call(name: &str, n: &Node) -> String {
    format!("{name}({})", args(n).join(", "))
}

fn text(n: &Node, top_level: bool) -> String {
    let wrap = |s: String| if top_level { s } else { format!("({s})") };
    match (&n.op, &n.val) {
        (Op::Const, Val::Const(c)) => num(*c),
        (Op::Val, Val::Value(v)) => value(v),
        (Op::Special, Val::Special(p)) => {
            if p.name.starts_with("vcc") {
                "vcc".into()
            } else {
                "exec".into()
            }
        }
        (Op::Add | Op::Sub | Op::Mul | Op::And | Op::Or | Op::Xor | Op::Shl | Op::Shr, _)
        | (Op::Cand | Op::Cor | Op::Cxor, _) => wrap(format!(
            "{} {} {}",
            inner(&n.args[0]),
            n.op.symbol(),
            inner(&n.args[1])
        )),
        (Op::Cmp, Val::Cmp(k, _, negated)) => {
            let body = format!("{} {} {}", inner(&n.args[0]), k.symbol(), inner(&n.args[1]));
            if *negated {
                format!("~({body})")
            } else {
                wrap(body)
            }
        }
        (Op::Div | Op::Mod, Val::Divisor(d)) => {
            let d = match d {
                Divisor::Const(c) => num(Num::Int(*c)),
                Divisor::Arg(v) => value(v),
            };
            wrap(format!("{} {} {d}", inner(&n.args[0]), n.op.symbol()))
        }
        (Op::Not | Op::Cnot, _) => format!("~{}", inner(&n.args[0])),
        (Op::Neg, _) => format!("-{}", inner(&n.args[0])),
        (Op::Abs, _) => format!("|{}|", top(&n.args[0])),
        (Op::Cast, _) => format!("{}({})", n.ty.map_or("", |t| t.name()), top(&n.args[0])),
        (Op::Across, Val::Lanes(op, lanes)) => {
            format!("across({}, {op}, {lanes})", top(&n.args[0]))
        }
        (Op::Bfe, Val::Bfe(o, w, _)) => format!("bfe({}, {o}, {w})", top(&n.args[0])),
        (Op::Unpack | Op::Unpack2, Val::Fmt(f)) => {
            let mut s = format!("{}({}, {}", n.op.name(), top(&n.args[0]), f.name);
            if f.at != 0 {
                s.push_str(&format!(", at={}", f.at));
            }
            s + ")"
        }
        (Op::Pack2, Val::Fmt(f)) => {
            let mut s = format!("pack2({}, {}, {}", top(&n.args[0]), top(&n.args[1]), f.name);
            if f.into {
                s.push_str(&format!(", into={}", top(&n.args[2])));
            }
            if f.at != 0 {
                s.push_str(&format!(", at={}", f.at));
            }
            s + ")"
        }
        (Op::F1, Val::F1(name)) => call(name, n),
        (Op::Instr, Val::Instr(name, _)) => call(name, n),
        (Op::CvtInt, _) => call(&format!("cvt_{}", n.ty.map_or("", |t| t.name())), n),
        (Op::Scan, Val::Lanes(op, lanes)) => format!("scan({}, {op}, {lanes})", top(&n.args[0])),
        // A mask's popcount is written popcount(m), as for any value.
        (Op::PopcountMask, _) => call("popcount", n),
        (op, _) => call(op.name(), n),
    }
}
