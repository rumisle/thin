//! LDS groups, regions, placement and layout tables.

use std::collections::BTreeMap;

use crate::error::{mem, Result};
use crate::types::Ty;

/// A swizzle `Xor(by, into, mask, shift)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Xor {
    /// The dimension whose index keys the swizzle.
    pub by: i64,
    /// The dimension swizzled.
    pub into: i64,
    /// The key's mask.
    pub mask: u64,
    /// The key's shift.
    pub shift: u32,
}

impl Xor {
    /// Its text: `[into] ^= [by] & mask` or `[into] ^= ([by] >> shift) & mask`.
    pub fn text(&self) -> String {
        match self.shift {
            0 => format!("[{}] ^= [{}] & {}", self.into, self.by, self.mask),
            s => format!("[{}] ^= ([{}] >> {s}) & {}", self.into, self.by, self.mask),
        }
    }

    /// The same swizzle with both dimensions one less (a part's).
    fn inner(&self) -> Xor {
        Xor {
            by: self.by - 1,
            into: self.into - 1,
            ..*self
        }
    }
}

/// A region's element: its bytes and how it was declared.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Elem {
    /// Its bytes.
    pub bytes: u64,
    /// How it was declared.
    pub name: String,
}

/// The element names and their bytes.
pub const ELEMS: [(&str, u64); 18] = [
    ("f32", 4),
    ("u32", 4),
    ("i32", 4),
    ("b32", 4),
    ("bf16", 2),
    ("f16", 2),
    ("u16", 2),
    ("i16", 2),
    ("u8", 1),
    ("i8", 1),
    ("e4m3", 1),
    ("e5m2", 1),
    ("e2m1x2", 1),
    ("e8m0", 1),
    ("b8", 1),
    ("b64", 8),
    ("u64", 8),
    ("b128", 16),
];

fn names() -> String {
    ELEMS.iter().map(|e| e.0).collect::<Vec<_>>().join(", ")
}

/// An element given as a type.
pub fn elem_of_type(t: Ty) -> Result<Elem> {
    match t {
        Ty::U32 | Ty::I32 | Ty::F32 => Ok(Elem {
            bytes: 4,
            name: t.name().into(),
        }),
        Ty::U64 => Ok(Elem {
            bytes: 8,
            name: "u64".into(),
        }),
        t => mem(format!(
            "an element of type {}: u32, i32, f32, u64, or a name ({})",
            t.name(),
            names()
        )),
    }
}

/// An element given by name.
pub fn elem_of_name(n: &str) -> Result<Elem> {
    match ELEMS.iter().find(|e| e.0 == n) {
        Some(e) => Ok(Elem {
            bytes: e.1,
            name: n.into(),
        }),
        None => mem(format!(
            "element {}: a value type (f32, u32, ..) or one of {}",
            crate::maker::py_repr(n),
            names()
        )),
    }
}

/// A region of the layout.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Region {
    /// Its name.
    pub name: String,
    /// Its shape, outermost first.
    pub shape: Vec<u64>,
    /// Its element.
    pub elem: Elem,
    /// Its strides.
    pub strides: Vec<u64>,
    /// Its bytes.
    pub size: u64,
    /// Whether its size was given (`nbytes=`).
    pub nbytes: bool,
    /// Its alignment.
    pub align: u64,
    /// Its absolute offset, if given.
    pub at: Option<u64>,
    /// Its swizzle.
    pub swizzle: Option<Xor>,
    /// Its parts, by region index.
    pub parts: Vec<usize>,
    /// Its parent and index, for a part.
    pub parent: Option<(usize, u64)>,
    /// Its first byte, once placed.
    pub base: u64,
}

/// A member of a group.
/// A union's bytes and its arms: (first byte, end, per arm its name and
/// its leaves).
pub type UnionArms = (u64, u64, Vec<(String, Vec<usize>)>);

#[derive(Clone, Debug, PartialEq, Eq)]
enum Member {
    Region(usize),
    Group(usize),
}

/// A struct or a union.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Group {
    union: bool,
    name: Option<String>,
    align: Option<u64>,
    at: Option<u64>,
    members: Vec<Member>,
    off: u64,
    end: u64,
}

/// The kernel's one layout.
#[derive(Clone, Debug)]
pub struct Layout {
    /// Its Builder.
    pub builder: u64,
    /// Its budget.
    pub budget: Option<u64>,
    lds_max: u64,
    /// Its regions, parts included, by index.
    pub regions: Vec<Region>,
    /// Region index by name (regions and parts).
    pub by_name: BTreeMap<String, usize>,
    groups: Vec<Group>,
    open: Vec<usize>,
    /// Whether it is final.
    pub frozen: bool,
    /// Its end.
    pub end: u64,
}

/// What `L.region` is given.
#[derive(Clone, Debug, Default)]
pub struct RegionArgs {
    /// The shape.
    pub shape: Vec<i128>,
    /// The element.
    pub elem: Option<Elem>,
    /// `pitch=` as an int.
    pub pitch: Option<i128>,
    /// `pitch=` as a dict.
    pub pitch_dims: Option<Vec<(i64, i128)>>,
    /// `nbytes=`.
    pub nbytes: Option<i128>,
    /// `align=`.
    pub align: Option<u64>,
    /// `at=`.
    pub at: Option<u64>,
    /// `swizzle=`.
    pub swizzle: Option<Xor>,
}

impl Layout {
    /// A new layout (`b.lds(budget)`).
    pub fn new(builder: u64, budget: Option<i128>, lds_max: u64) -> Result<Layout> {
        if let Some(b) = budget {
            if !(1..=i128::from(lds_max)).contains(&b) {
                return mem(format!(
                    "budget {b}: 1 .. {lds_max} bytes (a workgroup's LDS)"
                ));
            }
        }
        let root = Group {
            union: false,
            name: None,
            align: None,
            at: None,
            members: Vec::new(),
            off: 0,
            end: 0,
        };
        Ok(Layout {
            builder,
            budget: budget.map(|b| b as u64),
            lds_max,
            regions: Vec::new(),
            by_name: BTreeMap::new(),
            groups: vec![root],
            open: vec![0],
            frozen: false,
            end: 0,
        })
    }

    /// Opens a struct or union in the current group.
    pub fn open_group(
        &mut self,
        union: bool,
        name: Option<String>,
        align: Option<u64>,
        at: Option<u64>,
    ) -> Result<()> {
        if self.frozen {
            return mem(format!(
                "a {}: the layout is final",
                if union { "union" } else { "struct" }
            ));
        }
        let g = Group {
            union,
            name,
            align,
            at,
            members: Vec::new(),
            off: 0,
            end: 0,
        };
        self.groups.push(g);
        let k = self.groups.len() - 1;
        let cur = *self.open.last().expect("the root");
        self.groups[cur].members.push(Member::Group(k));
        self.open.push(k);
        Ok(())
    }

    /// Closes the current group.
    pub fn close_group(&mut self) {
        if self.open.len() > 1 {
            self.open.pop();
        }
    }

    /// `L.region(name, shape, elem, …)`.
    pub fn region(&mut self, name: &str, a: RegionArgs) -> Result<usize> {
        if self.frozen {
            return mem(format!(
                "region {name}: the layout is final (its first base read, or the render)"
            ));
        }
        if a.shape.iter().any(|&d| d < 1) {
            let s: Vec<String> = a.shape.iter().map(i128::to_string).collect();
            return mem(format!(
                "region {name}: shape [{}], each dimension 1 or more",
                s.join(", ")
            ));
        }
        let shape: Vec<u64> = a.shape.iter().map(|&d| d as u64).collect();
        let elem = a.elem.clone().unwrap_or(Elem {
            bytes: 1,
            name: "u8".into(),
        });
        let strides = strides(name, &shape, elem.bytes, &a)?;
        let natural = shape.first().map(|&n0| n0 * strides[0]);
        let size = match (a.nbytes, natural) {
            (Some(nb), Some(s)) if nb < i128::from(s) => {
                return mem(format!("region {name}: nbytes {nb} below its shape's {s}"));
            }
            (Some(nb), _) if nb >= 1 => nb as u64,
            (None, Some(s)) if s >= 1 => s,
            _ => return mem(format!("region {name}: a shape or nbytes=")),
        };
        if let Some(x) = &a.swizzle {
            check_swizzle(name, &shape, x)?;
        }
        if self.by_name.contains_key(name) {
            return mem(format!("region {name}: declared twice"));
        }
        let r = Region {
            name: name.into(),
            shape,
            elem,
            strides,
            size,
            nbytes: a.nbytes.is_some(),
            align: a.align.unwrap_or(16),
            at: a.at,
            swizzle: a.swizzle.map(|x| normal(x, a.shape.len())),
            parts: Vec::new(),
            parent: None,
            base: 0,
        };
        self.regions.push(r);
        let k = self.regions.len() - 1;
        self.by_name.insert(name.into(), k);
        let cur = *self.open.last().expect("the root");
        self.groups[cur].members.push(Member::Region(k));
        Ok(k)
    }

    /// `A.parts(*names)`.
    pub fn parts(&mut self, a: usize, names: &[String]) -> Result<Vec<usize>> {
        let r = self.regions[a].clone();
        if names.len() as u64 != r.shape.first().copied().unwrap_or(0) {
            return mem(format!(
                "{}: {} parts for {} entries",
                r.name,
                names.len(),
                r.shape.first().copied().unwrap_or(0)
            ));
        }
        if let Some(x) = r.swizzle.filter(|x| x.by == 0 || x.into == 0) {
            return mem(format!(
                "{}: parts along dimension 0, which its swizzle ({}) uses",
                r.name,
                x.text()
            ));
        }
        let mut out = Vec::new();
        for (k, n) in names.iter().enumerate() {
            if self.by_name.contains_key(n) {
                return mem(format!("region {n}: declared twice"));
            }
            let p = Region {
                name: n.clone(),
                shape: r.shape[1..].to_vec(),
                elem: r.elem.clone(),
                strides: r.strides[1..].to_vec(),
                size: r.strides[0],
                nbytes: false,
                align: r.align,
                at: None,
                swizzle: r.swizzle.map(|x| x.inner()),
                parts: Vec::new(),
                parent: Some((a, k as u64)),
                base: 0,
            };
            self.regions.push(p);
            let idx = self.regions.len() - 1;
            self.by_name.insert(n.clone(), idx);
            out.push(idx);
        }
        self.regions[a].parts = out.clone();
        Ok(out)
    }

    /// Makes the layout final: places it and checks it.
    pub fn freeze(&mut self) -> Result<()> {
        if self.frozen {
            return Ok(());
        }
        self.frozen = true;
        let end = self.place(0, 0);
        self.end = end;
        self.place_parts();
        self.check_overlap()?;
        if let Some(b) = self.budget.filter(|&b| end > b) {
            return mem(format!(
                "the layout is {end} B, past its budget of {b} B:\n{}",
                self.table().join("\n")
            ));
        }
        if end > self.lds_max {
            return mem(format!(
                "the layout is {end} B, past the {} B a workgroup has",
                self.lds_max
            ));
        }
        Ok(())
    }

    fn group_align(&self, g: usize) -> u64 {
        let grp = &self.groups[g];
        grp.align.unwrap_or_else(|| {
            (grp.members.iter())
                .map(|m| match m {
                    Member::Region(r) => self.regions[*r].align,
                    Member::Group(h) => self.group_align(*h),
                })
                .max()
                .unwrap_or(1)
        })
    }

    /// Places group g at the cursor; returns its end.
    fn place(&mut self, g: usize, cursor: u64) -> u64 {
        let align = self.group_align(g).max(1);
        let off = self.groups[g]
            .at
            .unwrap_or_else(|| cursor.div_ceil(align) * align);
        let off = if g == 0 { 0 } else { off };
        self.groups[g].off = off;
        let (union, members) = (self.groups[g].union, self.groups[g].members.clone());
        let mut cur = off;
        let mut end = off;
        for m in members {
            let start = if union { off } else { cur };
            let e = match m {
                Member::Region(r) => {
                    let reg = &mut self.regions[r];
                    let a = reg.align.max(1);
                    reg.base = reg.at.unwrap_or_else(|| start.div_ceil(a) * a);
                    reg.base + reg.size
                }
                Member::Group(h) => self.place(h, start),
            };
            end = end.max(e);
            if !union {
                cur = cur.max(e);
            }
        }
        self.groups[g].end = end;
        end
    }

    fn place_parts(&mut self) {
        for k in 0..self.regions.len() {
            if let Some((p, i)) = self.regions[k].parent {
                let base = self.regions[p].base + i * self.regions[p].strides[0];
                self.regions[k].base = base;
            }
        }
    }

    /// The arms each leaf sits in: per union the arm index.
    fn arms(
        &self,
        g: usize,
        path: &mut Vec<(usize, usize)>,
        out: &mut Vec<(usize, Vec<(usize, usize)>)>,
    ) {
        let grp = &self.groups[g];
        for (k, m) in grp.members.iter().enumerate() {
            if grp.union {
                path.push((g, k));
            }
            match m {
                Member::Region(r) => out.push((*r, path.clone())),
                Member::Group(h) => self.arms(*h, path, out),
            }
            if grp.union {
                path.pop();
            }
        }
    }

    fn check_overlap(&self) -> Result<()> {
        let mut leaves = Vec::new();
        self.arms(0, &mut Vec::new(), &mut leaves);
        for (x, (a, pa)) in leaves.iter().enumerate() {
            for (b, pb) in &leaves[x + 1..] {
                let (ra, rb) = (&self.regions[*a], &self.regions[*b]);
                let overlap = ra.base < rb.base + rb.size && rb.base < ra.base + ra.size;
                let arms = pa
                    .iter()
                    .any(|(g, k)| pb.iter().any(|(h, j)| g == h && k != j));
                if overlap && !arms {
                    return mem(format!(
                        "{} [{}, {}) and {} [{}, {}) overlap and are not arms of one union",
                        ra.name,
                        ra.base,
                        ra.base + ra.size,
                        rb.name,
                        rb.base,
                        rb.base + rb.size
                    ));
                }
            }
        }
        Ok(())
    }

    /// The regions in layout order (no parts).
    pub fn leaves(&self) -> Vec<usize> {
        let mut out = Vec::new();
        self.arms(0, &mut Vec::new(), &mut out);
        out.into_iter().map(|x| x.0).collect()
    }

    /// The unions of more than one member, innermost first in layout
    /// order: their bytes and per arm its name and its leaves (positions in
    /// [`Layout::leaves`]), as the LDS proof reads them.
    pub fn unions(&self) -> Vec<UnionArms> {
        let leaves = self.leaves();
        let mut out = Vec::new();
        self.collect_unions(0, &leaves, &mut out);
        out
    }

    fn collect_unions(&self, g: usize, leaves: &[usize], out: &mut Vec<UnionArms>) {
        let grp = &self.groups[g];
        for m in &grp.members {
            if let Member::Group(h) = m {
                self.collect_unions(*h, leaves, out);
            }
        }
        if !grp.union || grp.members.len() < 2 {
            return;
        }
        let position = |r: usize| leaves.iter().position(|&x| x == r).expect("a leaf");
        let arm = |m: &Member| match m {
            Member::Region(r) => (self.regions[*r].name.clone(), vec![position(*r)]),
            Member::Group(h) => {
                let mut under = Vec::new();
                self.arms(*h, &mut Vec::new(), &mut under);
                let h = &self.groups[*h];
                let kind = if h.union { "union" } else { "struct" };
                let name = h.name.clone().unwrap_or_else(|| kind.to_owned());
                (name, under.into_iter().map(|x| position(x.0)).collect())
            }
        };
        out.push((grp.off, grp.end, grp.members.iter().map(arm).collect()));
    }

    /// The names an op on a region answers to: its own, its parent's, its
    /// parts'.
    pub fn tags(&self, r: usize) -> Vec<String> {
        let reg = &self.regions[r];
        let mut out = vec![reg.name.clone()];
        if let Some((p, _)) = reg.parent {
            out.push(self.regions[p].name.clone());
        }
        out.extend(reg.parts.iter().map(|&p| self.regions[p].name.clone()));
        out
    }

    /// `L.table()`: the layout's lines.
    pub fn table(&self) -> Vec<String> {
        let head = match self.budget {
            Some(b) => format!("lds: {} B of a {b} B budget", self.end),
            None => format!("lds: {} B", self.end),
        };
        let mut out = vec![
            head,
            format!(
                "{:>8} {:>7} {:>6}  {:<16} {:<20} strides",
                "offset", "end", "size", "region", "shape"
            ),
        ];
        self.table_group(0, 0, &mut out);
        out
    }

    fn table_group(&self, g: usize, depth: usize, out: &mut Vec<String>) {
        let grp = &self.groups[g];
        for (k, m) in grp.members.iter().enumerate() {
            let mark = match (grp.union, k) {
                (false, _) => "",
                (true, 0) => "+ ",
                (true, _) => "| ",
            };
            let prefix = format!("{}{mark}", "  ".repeat(depth));
            match m {
                Member::Region(r) => {
                    out.push(self.region_line(*r, &format!("{prefix}{}", self.regions[*r].name)));
                    for &p in &self.regions[*r].parts {
                        let name = format!("{}. {}", "  ".repeat(depth + 1), self.regions[p].name);
                        out.push(self.region_line(p, &name));
                    }
                }
                Member::Group(h) => {
                    let sub = &self.groups[*h];
                    let kind = if sub.union { "union" } else { "struct" };
                    let label = match &sub.name {
                        Some(n) => format!("{n} ({kind})"),
                        None => kind.to_owned(),
                    };
                    out.push(format!(
                        "{:>8} {:>7} {:>6}  {prefix}{label}",
                        sub.off,
                        sub.end,
                        sub.end - sub.off
                    ));
                    self.table_group(*h, depth + 1, out);
                }
            }
        }
    }

    fn region_line(&self, r: usize, name: &str) -> String {
        let reg = &self.regions[r];
        let shape = match reg.nbytes {
            true => format!("{} B", reg.size),
            false => {
                let dims: String = reg.shape.iter().map(|d| format!("[{d}]")).collect();
                format!("{dims} {}", reg.elem.name)
            }
        };
        let strides = match reg.nbytes {
            true => "-".to_owned(),
            false => reg
                .strides
                .iter()
                .map(u64::to_string)
                .collect::<Vec<_>>()
                .join(" "),
        };
        let mut line = format!(
            "{:>8} {:>7} {:>6}  {:<16} {:<20} {strides}",
            reg.base,
            reg.base + reg.size,
            reg.size,
            name,
            shape
        );
        if let Some(x) = &reg.swizzle {
            line.push_str(&format!("  swizzle {}", x.text()));
        }
        line
    }
}

/// The strides of a shape.
fn strides(name: &str, shape: &[u64], e: u64, a: &RegionArgs) -> Result<Vec<u64>> {
    let n = shape.len();
    if n < 2 && (a.pitch.is_some() || a.pitch_dims.is_some()) {
        return mem(format!(
            "region {name}: pitch is a row's bytes (two dimensions or more)"
        ));
    }
    let mut given: BTreeMap<usize, i128> = BTreeMap::new();
    if let Some(p) = a.pitch {
        given.insert(n - 2, p);
    }
    for (d, p) in a.pitch_dims.iter().flatten() {
        let k = if *d < 0 { *d + n as i64 } else { *d };
        given.insert(k as usize, *p);
    }
    let mut st = vec![0u64; n];
    if n > 0 {
        st[n - 1] = e;
    }
    for d in (0..n.saturating_sub(1)).rev() {
        let natural = shape[d + 1] * st[d + 1];
        st[d] = match given.get(&d) {
            Some(&p) if p >= i128::from(natural) && p % i128::from(e) == 0 => p as u64,
            Some(&p) if a.pitch.is_some() && d == n - 2 => {
                return mem(format!(
                    "region {name}: pitch {p}, a row of {} x {e} B or more, a multiple of {e}",
                    shape[d + 1]
                ));
            }
            Some(&p) => {
                return mem(format!(
                    "region {name}: pitch {p} for dimension {d}: an entry's {natural} B or more, a multiple of {e}"
                ));
            }
            None => natural,
        };
    }
    Ok(st)
}

/// A swizzle's dimensions counted from 0.
fn normal(x: Xor, n: usize) -> Xor {
    let k = |d: i64| if d < 0 { d + n as i64 } else { d };
    Xor {
        by: k(x.by),
        into: k(x.into),
        ..x
    }
}

fn check_swizzle(name: &str, shape: &[u64], x: &Xor) -> Result<()> {
    let n = shape.len() as i64;
    if n < 2 {
        return mem(format!(
            "region {name}: swizzle=Xor(by=, into=, mask=) on two dimensions or more"
        ));
    }
    let k = |d: i64| if d < 0 { d + n } else { d };
    let (by, into) = (k(x.by), k(x.into));
    if !(0..n).contains(&by) || !(0..n).contains(&into) || by == into {
        return mem(format!(
            "region {name}: swizzle dimensions {}, {} of {n}: two of its own",
            x.by, x.into
        ));
    }
    let size = shape[into as usize];
    let m = x.mask + 1;
    if x.mask < 1 || m & (m - 1) != 0 || size % m != 0 {
        return mem(format!(
            "region {name}: swizzle mask {}: one less than a power of two dividing dimension {into}'s {size}",
            x.mask
        ));
    }
    Ok(())
}
