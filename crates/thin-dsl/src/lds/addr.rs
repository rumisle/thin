//! LDS offsets, addresses, pointers, swizzles and address splitting.

use crate::builder::{Builder, Make};
use crate::error::{mem, Result};
use crate::expr::node::{binop, const_of, constant, value_of, Num, Op, Tree};
use crate::expr::text::{num, top};
use crate::types::Ty;
use crate::values::Typed;

use super::layout::{Layout, Region};

/// A byte offset: an int or a tree (a value is a value leaf).
#[derive(Clone, Debug, PartialEq)]
pub enum Off {
    /// An int.
    Int(i128),
    /// A tree.
    Tree(Tree),
}

impl Off {
    /// Its text.
    pub fn text(&self) -> String {
        match self {
            Off::Int(c) => num(Num::Int(*c)),
            Off::Tree(t) => top(t),
        }
    }

    fn tree(&self) -> Tree {
        match self {
            Off::Int(c) => constant(Num::Int(*c)),
            Off::Tree(t) => t.clone(),
        }
    }
}

/// An address: a region, a byte offset, a label.
#[derive(Clone, Debug, PartialEq)]
pub struct Address {
    /// The layout's Builder.
    pub builder: u64,
    /// The region's index.
    pub region: usize,
    /// The region's name.
    pub name: String,
    /// The byte offset into the region.
    pub off: Off,
    /// The label.
    pub label: Option<String>,
}

impl Address {
    /// Its text: the label, the region's name for offset 0, else
    /// `<name> + <offset text>`.
    pub fn text(&self) -> String {
        if let Some(l) = &self.label {
            return l.clone();
        }
        match &self.off {
            Off::Int(0) => self.name.clone(),
            off => format!("{} + {}", self.name, off.text()),
        }
    }

    /// `A + t` or `addr + t`.
    pub fn plus(&self, t: Off) -> Address {
        Address {
            off: sum(vec![self.off.clone(), t]),
            label: None,
            ..self.clone()
        }
    }

    /// `addr - c` for an int, `addr - t` for a tree.
    pub fn minus(&self, t: Off) -> Result<Address> {
        let off = match t {
            Off::Int(c) => sum(vec![self.off.clone(), Off::Int(-c)]),
            Off::Tree(t) => Off::Tree(binop(Op::Sub, self.off.tree(), t)?),
        };
        Ok(Address {
            off,
            label: None,
            ..self.clone()
        })
    }
}

/// The sum of terms in order, the constants folded into one at the end:
/// `t + c`, `t - |c|`, or the int alone.
pub fn sum(items: Vec<Off>) -> Off {
    let mut c = 0i128;
    let mut terms: Vec<Tree> = Vec::new();
    for it in items {
        match it {
            Off::Int(v) => c += v,
            Off::Tree(t) => match const_of(&t).and_then(Num::int) {
                Some(v) if t.ty.is_none() => c += v,
                _ => terms.push(t),
            },
        }
    }
    let Some(first) = terms.first().cloned() else {
        return Off::Int(c);
    };
    let mut acc = first;
    for t in &terms[1..] {
        acc = binop(Op::Add, acc, t.clone()).expect("a sum of offsets");
    }
    match c {
        0 => Off::Tree(acc),
        c if c > 0 => Off::Tree(binop(Op::Add, acc, constant(Num::Int(c))).expect("a sum")),
        c => Off::Tree(binop(Op::Sub, acc, constant(Num::Int(-c))).expect("a sum")),
    }
}

/// `split(off)` → (variable part, constant).
pub fn split(off: &Off) -> (Option<Tree>, i128) {
    match off {
        Off::Int(c) => (None, *c),
        Off::Tree(t) => split_tree(t),
    }
}

fn split_tree(t: &Tree) -> (Option<Tree>, i128) {
    if let Some(c) = const_of(t).and_then(Num::int) {
        return (None, c);
    }
    if value_of(t).is_some() {
        return (Some(t.clone()), 0);
    }
    let int = t.ty.is_some_and(Ty::is_int);
    match t.op {
        Op::Mul | Op::Shl if int => {
            if let Some(k) = const_of(&t.args[1]).and_then(Num::int) {
                let (va, ca) = split_tree(&t.args[0]);
                if ca == 0 {
                    return (Some(t.clone()), 0);
                }
                let v = va.map(|v| binop(t.op, v, constant(Num::Int(k))).expect("a scaled part"));
                let c = if t.op == Op::Mul { ca * k } else { ca << k };
                return (v, c);
            }
            (Some(t.clone()), 0)
        }
        Op::Add if int => {
            let (va, ca) = split_tree(&t.args[0]);
            let (vb, cb) = split_tree(&t.args[1]);
            let v = match (va, vb) {
                (Some(a), Some(b)) => Some(binop(Op::Add, a, b).expect("a sum")),
                (a, b) => a.or(b),
            };
            (v, ca + cb)
        }
        Op::Sub if int => match const_of(&t.args[1]).and_then(Num::int) {
            Some(c) => {
                let (va, ca) = split_tree(&t.args[0]);
                (va, ca - c)
            }
            None => (Some(t.clone()), 0),
        },
        _ => (Some(t.clone()), 0),
    }
}

/// A variable part folded with a stride: `x * c'` or `x << s` becomes x
/// with stride st × c' or st << s.
fn fold(v: Tree, st: i128) -> (Tree, i128) {
    match v.op {
        Op::Mul => match const_of(&v.args[1]).and_then(Num::int) {
            Some(k) => (v.args[0].clone(), st * k),
            None => (v, st),
        },
        Op::Shl => match const_of(&v.args[1]).and_then(Num::int) {
            Some(s) => (v.args[0].clone(), st << s),
            None => (v, st),
        },
        _ => (v, st),
    }
}

impl Region {
    /// `A.stored(*idx)`: the swizzle applied (idx unchanged unswizzled).
    pub fn stored(&self, idx: &[Off]) -> Result<Vec<Off>> {
        let Some(x) = self.swizzle else {
            return Ok(idx.to_vec());
        };
        let (by, into) = (x.by as usize, x.into as usize);
        if idx.len() <= by.max(into) {
            return Ok(idx.to_vec());
        }
        let mut out = idx.to_vec();
        let key = match &idx[by] {
            Off::Int(v) => Off::Int((v >> x.shift) & i128::from(x.mask)),
            Off::Tree(t) => {
                let shifted = match x.shift {
                    0 => t.clone(),
                    s => binop(Op::Shr, t.clone(), constant(Num::Int(s.into())))?,
                };
                Off::Tree(binop(Op::And, shifted, constant(Num::Int(x.mask.into())))?)
            }
        };
        out[into] = match (&idx[into], key) {
            (Off::Int(0), key) => key,
            (Off::Int(a), Off::Int(k)) => Off::Int(a ^ k),
            (a, k) => Off::Tree(binop(Op::Xor, a.tree(), k.tree())?),
        };
        Ok(out)
    }

    /// `A.offset(*idx, swizzled)`.
    pub fn offset(&self, idx: &[Off], swizzled: bool) -> Result<Off> {
        if idx.len() > self.shape.len() {
            let t: Vec<String> = idx.iter().map(Off::text).collect();
            return mem(format!(
                "{}[{}]: {} indices for {} dimensions",
                self.name,
                t.join(", "),
                idx.len(),
                self.shape.len()
            ));
        }
        let idx = match swizzled {
            true => self.stored(idx)?,
            false => idx.to_vec(),
        };
        let mut items = Vec::new();
        let mut c = 0i128;
        for (k, i) in idx.iter().enumerate() {
            let st = i128::from(self.strides[k]);
            let (v, ci) = split(i);
            c += ci * st;
            if let Some(v) = v {
                let (v, st) = fold(v, st);
                let term = match st & (st - 1) {
                    0 => binop(Op::Mul, v, constant(Num::Int(st)))?,
                    _ => crate::expr::funcs::multiply(Op::Mul24, vec![v, constant(Num::Int(st))])?,
                };
                items.push(Off::Tree(term));
            }
        }
        items.push(Off::Int(c));
        Ok(sum(items))
    }
}

/// A pointer: a value at an element of a region.
#[derive(Clone, Debug, PartialEq)]
pub struct Pointer {
    /// The region's index.
    pub region: usize,
    /// The value.
    pub value: Typed,
    /// The first index, if an int.
    pub first: Option<i128>,
    /// The indices given.
    pub depth: usize,
}

impl Builder {
    /// `A.pointer(*idx, name)`.
    pub fn pointer(
        &mut self,
        region: usize,
        idx: &[Off],
        name: Option<String>,
        where_: String,
    ) -> Result<Pointer> {
        let l = self.frozen_layout()?;
        let r = l.regions[region].clone();
        let off = sum(vec![Off::Int(r.base.into()), r.offset(idx, true)?]);
        let tree = off.tree();
        let file = tree.file.unwrap_or('v');
        let m = Make {
            n: Some(1),
            n_repr: "1".into(),
            name,
            where_,
            ..Make::default()
        };
        let v = self.make_value(file, Ty::U32, m)?;
        self.assign(&v, tree, None)?;
        let first = match idx.first() {
            Some(Off::Int(k)) => Some(*k),
            _ => None,
        };
        Ok(Pointer {
            region,
            value: v,
            first,
            depth: idx.len(),
        })
    }
}

impl Pointer {
    /// `p.at(*d)`.
    pub fn at(&self, l: &Layout, builder: u64, d: &[Off]) -> Result<Address> {
        let r = &l.regions[self.region];
        let name = self
            .value
            .label
            .as_deref()
            .unwrap_or(&self.value.group)
            .to_owned();
        let texts: Vec<String> = d.iter().map(Off::text).collect();
        let disp = r.offset(d, false)?;
        if let Some(x) = r.swizzle {
            let reach = x.by.max(x.into) as usize;
            match &disp {
                Off::Int(_) if self.depth > reach => {
                    let get = |k: usize| match d.get(k) {
                        Some(Off::Int(v)) => *v,
                        _ => 0,
                    };
                    let p = i128::from(x.mask + 1) << x.shift;
                    let m1 = i128::from(x.mask + 1);
                    if get(x.by as usize) % p != 0 || get(x.into as usize) % m1 != 0 {
                        return mem(format!(
                            "{name}.at({}): the move changes {}'s swizzle ({}): a move along [{}] by a multiple of {p}, \
                             along [{}] by a multiple of {m1} - else a pointer of its own",
                            texts.join(", "),
                            r.name,
                            x.text(),
                            x.by,
                            x.into
                        ));
                    }
                }
                _ if d.len() > reach => {
                    return mem(format!(
                        "{name}.at({}): {name} is a sub-array's start of {} ({} indices), the move an element of it, \
                         swizzled ({}): a pointer at the element",
                        texts.join(", "),
                        r.name,
                        self.depth,
                        x.text()
                    ))
                }
                _ => {}
            }
        }
        let Off::Int(disp) = disp else {
            return mem(format!(
                "{name}.at({}): a pointer moves by constant indices (a move by a formula: a pointer of its own, or {}.at(..))",
                texts.join(", "),
                r.name
            ));
        };
        let mut target = self.region;
        if let (Some(first), Some(Off::Int(d0))) = (self.first, d.first()) {
            let k = first + d0;
            if let Some(&p) = usize::try_from(k).ok().and_then(|k| r.parts.get(k)) {
                target = p;
            }
        }
        let tr = &l.regions[target];
        let leaf = crate::expr::node::leaf(crate::expr::node::Leafish::Value(self.value.clone()))?;
        let delta = disp - i128::from(tr.base);
        let off = sum(vec![Off::Tree(leaf), Off::Int(delta)]);
        Ok(Address {
            builder,
            region: target,
            name: tr.name.clone(),
            off,
            label: Some(format!("{}: {name} + {disp}", tr.name)),
        })
    }
}
