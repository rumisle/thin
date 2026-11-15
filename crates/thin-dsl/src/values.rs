//! Register files, groups and typed values.

use std::collections::HashMap;
use std::sync::Arc;

use thin_ir::Reg;

use crate::error::{err, schedule, value, Class, Result};
use crate::types::Ty;

/// A value of the program: an `Value` of type B32.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ValueRec {
    /// Its file letter.
    pub file: char,
    /// Its dwords.
    pub n: u32,
    /// Its alignment.
    pub align: u32,
    /// Its group's name, which is its IR name.
    pub name: Arc<str>,
    /// Its place in its group.
    pub index: u32,
    /// Its source location.
    pub loc: u32,
    /// Its pin.
    pub pin: Option<u32>,
    /// The value whose registers it shares.
    pub same_as: Option<usize>,
    /// Its first register: provisional, pinned or placed.
    pub first: u32,
    /// `short(loc)` of its location, "" when unknown (the name spelling's
    /// `@<where>` of a temporary).
    pub where_: Arc<str>,
}

/// A group: a name and its values.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Group {
    /// Its name.
    pub name: Arc<str>,
    /// Its values, by id.
    pub values: Vec<usize>,
}

/// One entry of a placement: (group name, index, dwords, register).
pub type Placed = (String, u32, u32, u32);

/// A register file.
#[derive(Clone, Debug)]
pub struct File {
    /// Its letter.
    pub letter: char,
    /// Its size.
    pub limit: u32,
    /// Its groups in declaration order.
    pub groups: Vec<Group>,
    by_name: HashMap<Arc<str>, usize>,
    /// The groups the hardware fills at entry, in order.
    pub entered: Vec<Arc<str>>,
    /// The pinned groups, in pin order.
    pub pinned: Vec<Arc<str>>,
    /// The lanes-covered names.
    pub lanes: Vec<String>,
    cursor: u32,
    /// The placement it was built with (the audit's second run), or applied.
    pub placement: Option<Vec<Placed>>,
    /// Whether its registers are final.
    pub placed: bool,
}

impl File {
    /// An empty file of that size.
    pub fn new(letter: char, limit: u32, placement: Option<Vec<Placed>>) -> File {
        File {
            letter,
            limit,
            groups: Vec::new(),
            by_name: HashMap::new(),
            entered: Vec::new(),
            pinned: Vec::new(),
            lanes: Vec::new(),
            cursor: limit.max(256),
            placed: placement.is_some(),
            placement,
        }
    }

    /// The group of that name.
    pub fn group(&self, name: &str) -> Option<&Group> {
        self.by_name.get(name).map(|&k| &self.groups[k])
    }

    /// Declares a group of `count` values of `width` dwords, appending
    /// them to the program's values.
    pub fn alloc(
        &mut self,
        values: &mut Vec<ValueRec>,
        name: &str,
        count: u32,
        width: u32,
        align: u32,
        loc: u32,
    ) -> Result<usize> {
        if self.by_name.contains_key(name) {
            return value(format!("{}:{name} allocated twice", self.letter));
        }
        // A file of no registers (the target has none of it, as gfx1201 has
        // no AGPRs) holds no value: refused where it is made.
        if self.limit == 0 {
            return value(format!(
                "{}:{name}: this target has no {} registers",
                self.letter, self.letter
            ));
        }
        let name: Arc<str> = name.into();
        let base = self.cursor.div_ceil(align.max(1)) * align.max(1);
        let mut ids = Vec::new();
        for index in 0..count {
            let id = values.len();
            let first = match &self.placement {
                Some(p) => self.taken(values, p, &name, index, width)?,
                None => base + index * width,
            };
            values.push(ValueRec {
                file: self.letter,
                n: width,
                align,
                name: name.clone(),
                index,
                loc,
                pin: None,
                same_as: None,
                first,
                where_: "".into(),
            });
            ids.push(id);
        }
        if self.placement.is_none() {
            self.cursor = base + count * width;
        }
        self.by_name.insert(name.clone(), self.groups.len());
        self.groups.push(Group { name, values: ids });
        Ok(self.groups.len() - 1)
    }

    /// The register of the next value under a placement: the entry of the
    /// same position must name it.
    fn taken(
        &self,
        values: &[ValueRec],
        p: &[Placed],
        name: &str,
        index: u32,
        width: u32,
    ) -> Result<u32> {
        let k = values.iter().filter(|v| v.file == self.letter).count();
        match p.get(k) {
            Some((n, i, w, r)) if n == name && *i == index && *w == width => Ok(*r),
            _ => value(format!(
                "{}:{name}: not in the placement (the second run declared a value the first did not: the generator \
                 is not deterministic)",
                self.letter
            )),
        }
    }

    /// Pins a group's values at `at`, `at + n0`, ….
    pub fn pin(&mut self, values: &mut [ValueRec], name: &str, at: u32) -> Result<()> {
        let g = self
            .group(name)
            .ok_or_else(|| missing(self.letter, name))?
            .clone();
        let mut next = at;
        for &v in &g.values {
            values[v].pin = Some(next);
            next += values[v].n;
        }
        if self.placed {
            let regs: Vec<u32> = g.values.iter().map(|&v| values[v].first).collect();
            if regs.first() != Some(&at) || !pins_match(values, &g.values) {
                return value(format!(
                    "{}:{name}: pinned at {at}, placed at {}",
                    self.letter,
                    list(&regs)
                ));
            }
        } else {
            g.values
                .iter()
                .for_each(|&v| values[v].first = values[v].pin.expect("a pin"));
        }
        if !self.pinned.iter().any(|p| **p == *name) {
            self.pinned.push(g.name.clone());
        }
        Ok(())
    }

    /// Puts a group on another's registers.
    pub fn same_as(&mut self, values: &mut [ValueRec], name: &str, other: &str) -> Result<()> {
        let a = self
            .group(name)
            .ok_or_else(|| missing(self.letter, name))?
            .clone();
        let b = self
            .group(other)
            .ok_or_else(|| missing(self.letter, other))?
            .clone();
        let shape = |g: &Group| (g.values.len(), g.values.first().map_or(0, |&v| values[v].n));
        if shape(&a) != shape(&b) {
            let ((c0, w0), (c1, w1)) = (shape(&a), shape(&b));
            return value(format!(
                "same_as({name}, {other}): shapes differ ({c0}, {w0}) vs ({c1}, {w1})"
            ));
        }
        if chain(values, b.values[0])
            .iter()
            .any(|&v| a.values.contains(&v))
        {
            return value(format!(
                "same_as({name}, {other}): {other} is on {name}'s registers already"
            ));
        }
        a.values
            .iter()
            .zip(&b.values)
            .for_each(|(&x, &y)| values[x].same_as = Some(y));
        Ok(())
    }

    /// The name of the group at the end of a group's `same_as` chain.
    pub fn value_of(&self, values: &[ValueRec], name: &str) -> Result<Arc<str>> {
        let g = self.group(name).ok_or_else(|| missing(self.letter, name))?;
        Ok(values[*chain(values, g.values[0]).last().expect("a value")]
            .name
            .clone())
    }

    /// The register `VReg(group[idx], sub, width)`.
    pub fn reg(
        &self,
        values: &[ValueRec],
        name: &str,
        idx: i64,
        width: Option<u32>,
        sub: u32,
    ) -> Result<Reg> {
        let g = self.group(name).ok_or_else(|| missing(self.letter, name))?;
        let n = g.values.len() as i64;
        let k = if idx < 0 { idx + n } else { idx };
        let v = *usize::try_from(k)
            .ok()
            .and_then(|k| g.values.get(k))
            .ok_or_else(|| crate::error::Error {
                class: Class::Index,
                msg: format!("{}:{name}[{idx}]: {n} value(s)", self.letter),
            })?;
        Ok(Reg::new(
            v as i32,
            self.letter,
            sub,
            width.unwrap_or(values[v].n),
        ))
    }

    /// The register's number: its value's first register (the end of its
    /// `same_as` chain's) plus `lo`.
    pub fn number(&self, values: &[ValueRec], r: &Reg, repr: &str) -> Result<u32> {
        let v = self.own(values, r, repr)?;
        if r.lo + r.n > values[v].n {
            return value(format!(
                "{repr}: sub {} + width {} exceeds declared width {}",
                r.lo, r.n, values[v].n
            ));
        }
        let end = *chain(values, v).last().expect("a value");
        Ok(values[end].first + r.lo)
    }

    fn own(&self, values: &[ValueRec], r: &Reg, repr: &str) -> Result<usize> {
        match usize::try_from(r.value)
            .ok()
            .filter(|&v| v < values.len() && values[v].file == self.letter)
        {
            Some(v) => Ok(v),
            None => schedule(format!(
                "{repr}: not a value of this file's program (another run's, another Builder's)"
            )),
        }
    }

    /// The register's number once final.
    pub fn phys(&self, values: &[ValueRec], r: &Reg, repr: &str) -> Result<u32> {
        let v = self.own(values, r, repr)?;
        let end = *chain(values, v).last().expect("a value");
        match self.placed || values[end].pin.is_some() {
            true => self.number(values, r, repr),
            false => schedule(format!(
                "{repr}: a register's number read before the placement makes it final (derive, compile_kernel)"
            )),
        }
    }

    /// The cursor before a placement; the highest placed register + 1 after.
    pub fn used(&self, values: &[ValueRec]) -> u32 {
        match self.placed {
            false => self.cursor,
            true => (values.iter())
                .filter(|v| v.file == self.letter)
                .map(|v| v.first + v.n)
                .max()
                .unwrap_or(0),
        }
    }

    /// Applies a placement: each value takes its register.
    pub fn apply(&mut self, values: &mut [ValueRec], placement: &[Placed]) -> Result<()> {
        let mine: Vec<usize> = (0..values.len())
            .filter(|&v| values[v].file == self.letter)
            .collect();
        for (k, &v) in mine.iter().enumerate() {
            let rec = &values[v];
            match placement.get(k) {
                Some((n, i, w, r)) if **n == *rec.name && *i == rec.index && *w == rec.n => values[v].first = *r,
                _ => {
                    return value(format!(
                        "{}:{}: not in the placement (the second run declared a value the first did not: the \
                         generator is not deterministic)",
                        self.letter, rec.name
                    ))
                }
            }
        }
        self.placed = true;
        self.placement = Some(placement.to_vec());
        for name in self.pinned.clone() {
            let g = self.group(&name).expect("a pinned group");
            if !pins_match(values, &g.values) {
                let regs: Vec<u32> = g.values.iter().map(|&v| values[v].first).collect();
                let at = values[g.values[0]].pin.expect("a pin");
                return schedule(format!(
                    "{}:{name}: pinned at {at}, placed at {}",
                    self.letter,
                    list(&regs)
                ));
            }
        }
        Ok(())
    }

    /// `F.lanes_covered(*names)`.
    pub fn lanes_covered(&mut self, names: &[String]) {
        self.lanes.extend(names.iter().cloned());
    }

    /// Declares the registers the hardware fills at entry: one group per
    /// (name, width), pinned from 0; returns their group indices.
    pub fn entry(
        &mut self,
        values: &mut Vec<ValueRec>,
        groups: &[(&str, u32)],
        loc: u32,
    ) -> Result<Vec<usize>> {
        let mut at = 0;
        let mut out = Vec::new();
        for &(name, w) in groups {
            out.push(self.alloc(values, name, 1, w, 1, loc)?);
            self.entered.push(name.into());
            self.pin(values, name, at)?;
            at += w;
        }
        Ok(out)
    }
}

fn pins_match(values: &[ValueRec], ids: &[usize]) -> bool {
    ids.iter().all(|&v| values[v].pin == Some(values[v].first))
}

/// `[a, b]`.
fn list(regs: &[u32]) -> String {
    let t: Vec<String> = regs.iter().map(u32::to_string).collect();
    format!("[{}]", t.join(", "))
}

fn missing(letter: char, name: &str) -> crate::error::Error {
    crate::error::Error {
        class: Class::Value,
        msg: format!("{letter}:{name}: no such group"),
    }
}

/// A value's `same_as` chain, itself first.
pub fn chain(values: &[ValueRec], v: usize) -> Vec<usize> {
    let mut out = vec![v];
    for _ in 0..values.len() {
        match values[*out.last().expect("a value")].same_as {
            Some(next) if !out.contains(&next) => out.push(next),
            _ => break,
        }
    }
    out
}

/// A register's repr: `<file>:<value name>[<index>]x<n>`, `+<lo>` when
/// lo ≠ 0.
pub fn reg_repr(values: &[ValueRec], r: &Reg) -> String {
    let at = if r.lo != 0 {
        format!("+{}", r.lo)
    } else {
        String::new()
    };
    match usize::try_from(r.value).ok().and_then(|v| values.get(v)) {
        Some(v) => format!("{}:{}[{}]x{}{at}", v.file, v.name, v.index, r.n),
        None => format!("{}:_lit[0]x{}{at}", r.file, r.n),
    }
}

/// A register as an instruction's text in a message names it:
/// `<file>:<value name>`, `[<index>]` when nonzero, `+<lo>` when lo ≠ 0 and
/// `x<n>` when wider than a dword (`v:acc#1`, `s:args#1+2x2`).
pub fn reg_text(values: &[ValueRec], r: &Reg) -> String {
    let at = if r.lo != 0 {
        format!("+{}", r.lo)
    } else {
        String::new()
    };
    let wide = if r.n > 1 {
        format!("x{}", r.n)
    } else {
        String::new()
    };
    match usize::try_from(r.value).ok().and_then(|v| values.get(v)) {
        Some(v) if v.index != 0 => format!("{}:{}[{}]{at}{wide}", v.file, v.name, v.index),
        Some(v) => format!("{}:{}{at}{wide}", v.file, v.name),
        None => format!("{}:_lit{at}{wide}", r.file),
    }
}

/// A typed value: the author's handle on some dwords of a group.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Typed {
    /// Its Builder's identity.
    pub builder: u64,
    /// Its file.
    pub file: char,
    /// Its group's first value.
    pub value: usize,
    /// Its group's name.
    pub group: Arc<str>,
    /// Its type.
    pub ty: Ty,
    /// Its elements.
    pub n: u32,
    /// Its dword offset into the group's first value.
    pub sub: u32,
    /// Its label.
    pub label: Option<Arc<str>>,
    /// The author's line.
    pub where_: Arc<str>,
    /// Its part: how it was taken from its value.
    pub part: Arc<str>,
    /// Whether it is a divisor argument.
    pub divisor: bool,
}

impl Typed {
    /// Its dwords.
    pub fn width(&self) -> u32 {
        self.n * self.ty.dwords()
    }

    /// Its register.
    pub fn reg(&self) -> Reg {
        Reg::new(self.value as i32, self.file, self.sub, self.width())
    }

    /// `<file>:<group><part>`.
    pub fn repr(&self) -> String {
        format!("{}:{}{}", self.file, self.group, self.part)
    }

    fn view(&self, ty: Ty, n: u32, sub: u32, part: String) -> Typed {
        Typed {
            ty,
            n,
            sub,
            part: part.into(),
            divisor: false,
            ..self.clone()
        }
    }

    /// Element j, negative from the end.
    pub fn element(&self, j: i64) -> Result<Typed> {
        let k = if j < 0 { j + i64::from(self.n) } else { j };
        if !(0..i64::from(self.n)).contains(&k) {
            return err(
                Class::Index,
                format!("{}[{j}]: {} element(s)", self.repr(), self.n),
            );
        }
        let part = format!("{}[{k}]", self.part);
        Ok(self.view(self.ty, 1, self.sub + k as u32 * self.ty.dwords(), part))
    }

    /// Elements a..b as Python slices them; `step` other than none or 1
    /// is refused.
    pub fn slice(
        &self,
        a: Option<i64>,
        b: Option<i64>,
        step: Option<i64>,
        written: &str,
    ) -> Result<Typed> {
        if !matches!(step, None | Some(1)) {
            return err(
                Class::Index,
                format!(
                    "{}[{written}]: elements are contiguous, no step",
                    self.repr()
                ),
            );
        }
        let n = i64::from(self.n);
        let clamp = |x: i64| if x < 0 { (x + n).max(0) } else { x.min(n) };
        let (a, b) = (clamp(a.unwrap_or(0)), clamp(b.unwrap_or(n)));
        if b <= a {
            return err(
                Class::Index,
                format!("{}[{written}]: no element", self.repr()),
            );
        }
        let part = format!("{}[{a}:{b}]", self.part);
        Ok(self.view(
            self.ty,
            (b - a) as u32,
            self.sub + a as u32 * self.ty.dwords(),
            part,
        ))
    }

    /// `.lo` (k = 0) or `.hi` (k = 1).
    pub fn half(&self, k: u32) -> Result<Typed> {
        let ty = match self.ty {
            Ty::U64 | Ty::Ptr | Ty::Mask if self.n == 1 => Ty::U32,
            Ty::F32x2 if self.n == 1 => Ty::F32,
            _ => {
                return err(
                    Class::Type,
                    format!(
                        "{}: .lo / .hi are a u64's, a mask's or an f32x2's halves",
                        self.repr()
                    ),
                )
            }
        };
        let part = format!("{}.{}", self.part, if k == 0 { "lo" } else { "hi" });
        Ok(self.view(ty, 1, self.sub + k, part))
    }

    /// A divisor argument's `mul` (k = 1) or `shr` (k = 2): the next
    /// dwords of its group.
    pub fn divisor_part(&self, k: u32) -> Typed {
        let part = format!("{}.{}", self.part, if k == 1 { "mul" } else { "shr" });
        Typed {
            ty: Ty::U32,
            n: 1,
            sub: self.sub + k,
            part: part.into(),
            divisor: false,
            ..self.clone()
        }
    }

    /// The same registers read as another type (a cast, or the lowering's
    /// retyping).
    pub fn retyped(&self, ty: Ty) -> Typed {
        Typed {
            ty,
            n: self.width() / ty.dwords().max(1),
            divisor: false,
            ..self.clone()
        }
    }
}
