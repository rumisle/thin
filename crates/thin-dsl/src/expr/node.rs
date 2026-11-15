//! Trees: nodes, leaves, files and typing.

use std::sync::Arc;

use thin_target::Phys;

use crate::error::{expr, Result};
use crate::maker::FArg;
use crate::types::Ty;
use crate::values::Typed;

/// A constant.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Num {
    /// An int.
    Int(i128),
    /// A float.
    Float(f64),
}

impl Num {
    /// Its value as a float.
    pub fn as_f64(self) -> f64 {
        match self {
            Num::Int(v) => v as f64,
            Num::Float(x) => x,
        }
    }

    /// Whether it equals an int, numerically (1.0 is 1).
    pub fn is(self, v: i128) -> bool {
        match self {
            Num::Int(x) => x == v,
            Num::Float(x) => x == v as f64,
        }
    }

    /// The int it is, if exact.
    pub fn int(self) -> Option<i128> {
        match self {
            Num::Int(x) => Some(x),
            Num::Float(x) if x.fract() == 0.0 && x.abs() < 1e38 => Some(x as i128),
            Num::Float(_) => None,
        }
    }
}

/// A compare's kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// `<`.
    Lt,
    /// `<=`.
    Le,
    /// `>`.
    Gt,
    /// `>=`.
    Ge,
    /// `==`.
    Eq,
    /// `!=`.
    Ne,
}

impl Kind {
    /// The kind of a name (`lt`, …).
    pub fn named(name: &str) -> Option<Kind> {
        Some(match name {
            "lt" => Kind::Lt,
            "le" => Kind::Le,
            "gt" => Kind::Gt,
            "ge" => Kind::Ge,
            "eq" => Kind::Eq,
            "ne" => Kind::Ne,
            _ => return None,
        })
    }

    /// Its name.
    pub fn name(self) -> &'static str {
        match self {
            Kind::Lt => "lt",
            Kind::Le => "le",
            Kind::Gt => "gt",
            Kind::Ge => "ge",
            Kind::Eq => "eq",
            Kind::Ne => "ne",
        }
    }

    /// Its operator.
    pub fn symbol(self) -> &'static str {
        match self {
            Kind::Lt => "<",
            Kind::Le => "<=",
            Kind::Gt => ">",
            Kind::Ge => ">=",
            Kind::Eq => "==",
            Kind::Ne => "!=",
        }
    }

    /// The complement (lt ↔ ge, le ↔ gt, eq ↔ ne).
    pub fn complement(self) -> Kind {
        match self {
            Kind::Lt => Kind::Ge,
            Kind::Ge => Kind::Lt,
            Kind::Le => Kind::Gt,
            Kind::Gt => Kind::Le,
            Kind::Eq => Kind::Ne,
            Kind::Ne => Kind::Eq,
        }
    }

    /// The kind with the operands swapped (< ↔ >, <= ↔ >=).
    pub fn mirror(self) -> Kind {
        match self {
            Kind::Lt => Kind::Gt,
            Kind::Gt => Kind::Lt,
            Kind::Le => Kind::Ge,
            Kind::Ge => Kind::Le,
            k => k,
        }
    }
}

/// A divisor: a constant or a divisor argument's value.
#[derive(Clone, Debug, PartialEq)]
pub enum Divisor {
    /// A positive constant.
    Const(i128),
    /// A divisor argument.
    Arg(Typed),
}

/// A node's operator.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(missing_docs)]
pub enum Op {
    Val,
    Special,
    Const,
    Add,
    Sub,
    Mul,
    And,
    Or,
    Xor,
    Shl,
    Shr,
    Cmp,
    Cand,
    Cor,
    Cxor,
    Cnot,
    Not,
    Neg,
    Abs,
    Cast,
    Select,
    Mul24,
    Mad24,
    Mulhi,
    MulWide,
    Min,
    Max,
    Min3,
    Max3,
    Med3,
    Popcount,
    PopcountMask,
    First,
    Any,
    All,
    Fma,
    F1,
    CvtF32,
    CvtInt,
    Bf16Lo,
    Bf16Hi,
    PkBf16,
    Pair,
    FirstLane,
    LaneRead,
    LaneWrite,
    Shuffle,
    Rank,
    Div,
    Mod,
    Bfe,
    Unpack,
    Unpack2,
    Pack2,
    Across,
    Scan,
    Instr,
}

impl Op {
    /// Every operator.
    pub const ALL: [Op; 57] = [
        Op::Val,
        Op::Special,
        Op::Const,
        Op::Add,
        Op::Sub,
        Op::Mul,
        Op::And,
        Op::Or,
        Op::Xor,
        Op::Shl,
        Op::Shr,
        Op::Cmp,
        Op::Cand,
        Op::Cor,
        Op::Cxor,
        Op::Cnot,
        Op::Not,
        Op::Neg,
        Op::Abs,
        Op::Cast,
        Op::Select,
        Op::Mul24,
        Op::Mad24,
        Op::Mulhi,
        Op::MulWide,
        Op::Min,
        Op::Max,
        Op::Min3,
        Op::Max3,
        Op::Med3,
        Op::Popcount,
        Op::PopcountMask,
        Op::First,
        Op::Any,
        Op::All,
        Op::Fma,
        Op::F1,
        Op::CvtF32,
        Op::CvtInt,
        Op::Bf16Lo,
        Op::Bf16Hi,
        Op::PkBf16,
        Op::Pair,
        Op::FirstLane,
        Op::LaneRead,
        Op::LaneWrite,
        Op::Shuffle,
        Op::Rank,
        Op::Div,
        Op::Mod,
        Op::Bfe,
        Op::Unpack,
        Op::Unpack2,
        Op::Pack2,
        Op::Across,
        Op::Scan,
        Op::Instr,
    ];

    /// The operator of a name.
    pub fn named(name: &str) -> Option<Op> {
        Op::ALL.into_iter().find(|o| o.name() == name)
    }

    /// Its name (Python's `op`).
    pub fn name(self) -> &'static str {
        use Op::*;
        match self {
            Val => "val",
            Special => "special",
            Const => "const",
            Add => "add",
            Sub => "sub",
            Mul => "mul",
            And => "and",
            Or => "or",
            Xor => "xor",
            Shl => "shl",
            Shr => "shr",
            Cmp => "cmp",
            Cand => "cand",
            Cor => "cor",
            Cxor => "cxor",
            Cnot => "cnot",
            Not => "not",
            Neg => "neg",
            Abs => "abs",
            Cast => "cast",
            Select => "select",
            Mul24 => "mul24",
            Mad24 => "mad24",
            Mulhi => "mulhi",
            MulWide => "mul_wide",
            Min => "min",
            Max => "max",
            Min3 => "min3",
            Max3 => "max3",
            Med3 => "med3",
            Popcount => "popcount",
            PopcountMask => "popcount_mask",
            First => "first",
            Any => "any",
            All => "all",
            Fma => "fma",
            F1 => "f1",
            CvtF32 => "cvt_f32",
            CvtInt => "cvt_int",
            Bf16Lo => "bf16_lo",
            Bf16Hi => "bf16_hi",
            PkBf16 => "pk_bf16",
            Pair => "pair",
            FirstLane => "first_lane",
            LaneRead => "lane_read",
            LaneWrite => "lane_write",
            Shuffle => "shuffle",
            Rank => "rank",
            Div => "div",
            Mod => "mod",
            Bfe => "bfe",
            Unpack => "unpack",
            Unpack2 => "unpack2",
            Pack2 => "pack2",
            Across => "across",
            Scan => "scan",
            Instr => "op",
        }
    }

    /// The binary operator of a name.
    pub fn binary(name: &str) -> Option<Op> {
        Some(match name {
            "add" => Op::Add,
            "sub" => Op::Sub,
            "mul" => Op::Mul,
            "and" => Op::And,
            "or" => Op::Or,
            "xor" => Op::Xor,
            "shl" => Op::Shl,
            "shr" => Op::Shr,
            _ => return None,
        })
    }

    /// The infix symbol of a binary operator.
    pub fn symbol(self) -> &'static str {
        match self {
            Op::Add => "+",
            Op::Sub => "-",
            Op::Mul => "*",
            Op::And | Op::Cand => "&",
            Op::Or | Op::Cor => "|",
            Op::Xor | Op::Cxor => "^",
            Op::Shl => "<<",
            Op::Shr => ">>",
            Op::Div => "//",
            Op::Mod => "%",
            _ => "?",
        }
    }
}

/// A storage format's conversion arguments.
#[derive(Clone, Debug, PartialEq)]
pub struct Fmt {
    /// The format's name.
    pub name: String,
    /// The element or word.
    pub at: u32,
    /// Whether `into=` was given (pack2: the third operand).
    pub into: bool,
}

/// A node's value (Python's `val`).
#[derive(Clone, Debug, PartialEq)]
pub enum Val {
    /// None.
    None,
    /// A typed value leaf.
    Value(Typed),
    /// VCC or EXEC.
    Special(&'static Phys),
    /// A constant.
    Const(Num),
    /// A compare: its kind, the compared type, whether negated.
    Cmp(Kind, Ty, bool),
    /// A divisor.
    Divisor(Divisor),
    /// A bit-field extract: offset, width, signedness when given.
    Bfe(u32, u32, Option<bool>),
    /// A one-operand float function's name.
    F1(&'static str),
    /// A storage format.
    Fmt(Fmt),
    /// A lane operator and its lanes.
    Lanes(&'static str, u32),
    /// An instruction's name as given and its fields.
    Instr(String, Vec<(String, FArg)>),
}

/// A node of a tree.
#[derive(Clone, Debug, PartialEq)]
pub struct Node {
    /// Its operator.
    pub op: Op,
    /// Its type, none for an untyped constant.
    pub ty: Option<Ty>,
    /// Its file, none for a constant.
    pub file: Option<char>,
    /// Its operands.
    pub args: Vec<Arc<Node>>,
    /// Its value.
    pub val: Val,
    /// The Builder its values belong to.
    pub builder: Option<u64>,
}

/// A tree.
pub type Tree = Arc<Node>;

/// What a tree's leaf is made from.
#[derive(Clone, Debug)]
pub enum Leafish {
    /// A node.
    Node(Tree),
    /// A typed value.
    Value(Typed),
    /// A named register.
    Phys(&'static Phys),
    /// A bool, by its repr.
    Bool(String),
    /// An int.
    Int(i128),
    /// An int past 128 bits, by its repr.
    Big(String),
    /// A float.
    Float(f64),
    /// Anything else, by its repr.
    Other(String),
}

/// The leaf of an operand.
pub fn leaf(x: Leafish) -> Result<Tree> {
    Ok(match x {
        Leafish::Node(n) => n,
        Leafish::Value(v) => {
            if v.n != 1 {
                return expr(format!(
                    "{}: {} elements in a formula (one: {}[j])",
                    v.repr(),
                    v.n,
                    v.repr()
                ));
            }
            let file = v.file;
            Arc::new(Node {
                op: Op::Val,
                ty: Some(v.ty),
                file: Some(file),
                args: Vec::new(),
                builder: Some(v.builder),
                val: Val::Value(v),
            })
        }
        // A mask register leaf is a mask of its own width: vcc_lo and
        // exec_lo (a 32-lane wave's) hold 32 lanes.
        Leafish::Phys(p) if matches!(p.name, "vcc" | "vcc_lo" | "exec" | "exec_lo") => {
            Arc::new(Node {
                op: Op::Special,
                ty: Some(Ty::mask_of(32 * p.dwords)),
                file: Some('s'),
                args: Vec::new(),
                builder: None,
                val: Val::Special(p),
            })
        }
        Leafish::Phys(p) => {
            return expr(format!(
                "{}: a lane mask (VCC, EXEC) is a leaf; {} is not",
                p.name, p.name
            ))
        }
        Leafish::Bool(r) => {
            return expr(format!(
                "{r}: a bool is not a constant of a type (0 / 1, or a compare)"
            ))
        }
        Leafish::Int(v) => constant(Num::Int(v)),
        Leafish::Big(r) => return expr(format!("{r}: a constant past 128 bits")),
        Leafish::Float(x) => constant(Num::Float(x)),
        Leafish::Other(r) => return expr(format!("{r}: not a value, a constant or an expression")),
    })
}

/// An untyped constant.
pub fn constant(c: Num) -> Tree {
    Arc::new(Node {
        op: Op::Const,
        ty: None,
        file: None,
        args: Vec::new(),
        val: Val::Const(c),
        builder: None,
    })
}

/// The constant of a node, if it is one.
pub fn const_of(n: &Node) -> Option<Num> {
    match (&n.op, &n.val) {
        (Op::Const, Val::Const(c)) => Some(*c),
        _ => None,
    }
}

/// The typed value of a value leaf.
pub fn value_of(n: &Node) -> Option<&Typed> {
    match &n.val {
        Val::Value(v) if n.op == Op::Val => Some(v),
        _ => None,
    }
}

/// The Builder of operands: one, or none; two → ExprError.
fn owner(args: &[Tree]) -> Result<Option<u64>> {
    let mut b = None;
    for a in args {
        match (b, a.builder) {
            (Some(x), Some(y)) if x != y => {
                return expr("values of two Builders' files in one expression")
            }
            (None, y) => b = y,
            _ => {}
        }
    }
    Ok(b)
}

/// The file of a new node.
fn file_of(op: Op, ty: Option<Ty>, args: &[Tree], val: &Val) -> char {
    use Op::{
        Abs, Across, All, Any, First, FirstLane, Instr, LaneRead, Neg, PopcountMask, Rank, Scan,
        Select, Shuffle, Special,
    };
    match op {
        Special | FirstLane | LaneRead | PopcountMask | First | Any | All => 's',
        Shuffle | Rank | Across | Scan => 'v',
        // A VALU op by mnemonic or dotted name.
        Instr => match val {
            Val::Instr(name, _) if name.starts_with("v_") || name.starts_with("v.") => 'v',
            _ => 's',
        },
        // A per-lane condition (a mask, or a vector compare) chooses per
        // lane, so the select is a vector one whatever its arms.
        Select => {
            let lanes = args[0].ty.is_some_and(Ty::is_mask);
            if lanes || args.iter().any(|a| a.file == Some('v')) {
                'v'
            } else {
                's'
            }
        }
        _ => {
            let float = ty.is_some_and(Ty::is_float)
                || matches!(op, Neg | Abs)
                || args
                    .iter()
                    .any(|a| a.ty.is_some_and(Ty::is_float) || a.file == Some('v'));
            if float {
                'v'
            } else {
                's'
            }
        }
    }
}

/// A new node over its operands.
pub fn node(op: Op, ty: Option<Ty>, args: Vec<Tree>, val: Val) -> Result<Tree> {
    let builder = owner(&args)?;
    let file = Some(file_of(op, ty, &args, &val));
    Ok(Arc::new(Node {
        op,
        ty,
        file,
        args,
        val,
        builder,
    }))
}

/// The shared type of two operand types.
pub fn shared(ta: Option<Ty>, tb: Option<Ty>, subject: &dyn Fn() -> String) -> Result<Option<Ty>> {
    Ok(match (ta, tb) {
        (None, t) | (t, None) => t,
        (Some(a), Some(b)) if a == b => Some(a),
        (Some(a), Some(b)) if a.is_lanes() && b.is_lanes() => Some(Ty::Cond),
        (Some(a), Some(b)) if a.is_wide() && (b.is_int() || b.is_wide()) => Some(a),
        (Some(a), Some(b)) if b.is_wide() && a.is_int() => Some(b),
        (Some(a), Some(b)) => {
            return expr(format!(
                "{}: {} with {} (types do not mix: cast one - u32(x), i32(x), f32(x) reinterpret, no instruction)",
                subject(),
                a.name(),
                b.name()
            ))
        }
    })
}

/// `a op b` for the arithmetic operators.
pub fn binop(op: Op, a: Tree, b: Tree) -> Result<Tree> {
    let subject = || {
        format!(
            "{} {} {}",
            super::text::top(&a),
            op.symbol(),
            super::text::top(&b)
        )
    };
    owner(&[a.clone(), b.clone()])?;
    let t = shared(a.ty, b.ty, &subject)?;
    let arith = matches!(op, Op::Add | Op::Sub | Op::Mul);
    match t {
        Some(Ty::F32) if !arith => {
            return expr(format!("{}: an f32 takes + - * (bits: u32(x))", subject()))
        }
        Some(Ty::F32x2) if !arith || a.ty != Some(Ty::F32x2) || b.ty != Some(Ty::F32x2) => {
            return expr(format!(
                "{}: an f32x2 takes + - * of two f32x2 (a broadcast: the scalar in a pair first)",
                subject()
            ))
        }
        Some(l) if l.is_lanes() => {
            let cop = match op {
                Op::And => Op::Cand,
                Op::Or => Op::Cor,
                Op::Xor => Op::Cxor,
                _ => {
                    return expr(format!(
                        "{} of lane masks / conditions: & | ^ ~",
                        op.symbol()
                    ))
                }
            };
            return node(cop, Some(Ty::Cond), vec![a, b], Val::None);
        }
        Some(w) if w.is_wide() && !matches!(op, Op::Add | Op::Sub) => {
            return expr(format!(
                "{}: a {} takes + and - (its halves: .lo / .hi)",
                subject(),
                w.name()
            ))
        }
        None => {
            return expr(format!(
                "{}: two constants (compute it in Python)",
                subject()
            ))
        }
        _ => {}
    }
    node(op, t, vec![a, b], Val::None)
}

/// `a <kind> b`.
pub fn compare(kind: Kind, a: Tree, b: Tree) -> Result<Tree> {
    let subject = || {
        format!(
            "{} {} {}",
            super::text::inner(&a),
            kind.symbol(),
            super::text::inner(&b)
        )
    };
    owner(&[a.clone(), b.clone()])?;
    let t = shared(a.ty, b.ty, &subject)?;
    match t {
        Some(t) if t.is_lanes() || t == Ty::F32x2 => expr(format!(
            "{}: a compare of numbers (an f32x2: of .lo, .hi)",
            subject()
        )),
        None => expr(format!(
            "{}: two constants (compute it in Python)",
            subject()
        )),
        Some(t) => {
            let t = if t == Ty::Ptr { Ty::U64 } else { t };
            node(
                Op::Cmp,
                Some(Ty::Cond),
                vec![a, b],
                Val::Cmp(kind, t, false),
            )
        }
    }
}

/// `x // d` and `x % d`.
pub fn divmod(op: Op, x: Tree, d: Option<Divisor>, d_text: &str) -> Result<Tree> {
    let sym = op.symbol();
    let d = match d {
        Some(Divisor::Const(c)) if c >= 1 => Divisor::Const(c),
        Some(Divisor::Arg(v)) if v.divisor => Divisor::Arg(v),
        _ => {
            return expr(format!(
                "{} {sym} {d_text}: by a positive constant or a divisor argument (u32.within(lo, hi).divisor())",
                super::text::inner(&x)
            ))
        }
    };
    if x.ty != Some(Ty::U32) {
        return expr(format!(
            "{} {sym} {d_text}: a u32's (cast: u32(x))",
            super::text::inner(&x)
        ));
    }
    let mut args = vec![x];
    if let Divisor::Arg(v) = &d {
        args.push(leaf(Leafish::Value(v.clone()))?);
    }
    node(op, Some(Ty::U32), args, Val::Divisor(d))
}

/// `~x`.
pub fn invert(x: Tree) -> Result<Tree> {
    if let (Op::Cmp, Val::Cmp(k, t, neg)) = (x.op, &x.val) {
        let val = if *t == Ty::F32 {
            Val::Cmp(*k, *t, !neg)
        } else {
            Val::Cmp(k.complement(), *t, *neg)
        };
        return node(Op::Cmp, Some(Ty::Cond), x.args.clone(), val);
    }
    match x.ty {
        Some(t) if t.is_lanes() => node(Op::Cnot, Some(Ty::Cond), vec![x], Val::None),
        Some(t) if t.is_int() => node(Op::Not, Some(t), vec![x], Val::None),
        _ => expr(format!(
            "~{}: an integer's bits or a condition",
            super::text::inner(&x)
        )),
    }
}

/// `-x` and `abs(x)`.
pub fn sign(op: Op, x: Tree) -> Result<Tree> {
    match x.ty {
        Some(Ty::F32) => node(op, Some(Ty::F32), vec![x], Val::None),
        Some(Ty::F32x2) if op == Op::Neg => node(op, Some(Ty::F32x2), vec![x], Val::None),
        _ => {
            if op == Op::Neg {
                let t = super::text::inner(&x);
                expr(format!("-{t}: a float's (an integer's negation: 0 - x)"))
            } else {
                let t = super::text::top(&x);
                expr(format!("|{t}|: a float's (an i32's: max(x, 0 - x))"))
            }
        }
    }
}
