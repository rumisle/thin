//! Value ranges: address offsets proven not to wrap past 32
//! bits, 24-bit multiply operands, and LDS bytes inside their regions.

use std::collections::BTreeMap;

use thin_ir::solve::{solve_forward, Widen};
use thin_ir::{Diag, DiagClass, FieldValue, ImmValue, Instr, Operand, Part, Reg, Snapshot};
use thin_target::{phys_named, Flags, Kind, Mem, Pseudo};

/// 2^32.
const P32: i128 = 1 << 32;

/// A set of integers lo..= hi the register holds modulo 2^32, with where
/// its u32 value was lost, why, and the instruction that computed it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Interval {
    /// The least.
    pub lo: i128,
    /// The greatest.
    pub hi: i128,
    /// The instruction where the u32 value was lost.
    pub wrap: Option<usize>,
    /// Why.
    pub why: String,
    /// The instruction that computed it.
    pub at: Option<usize>,
}

impl Interval {
    /// [lo, hi], computed nowhere.
    pub fn new(lo: i128, hi: i128) -> Interval {
        Interval {
            lo,
            hi,
            wrap: None,
            why: String::new(),
            at: None,
        }
    }

    fn at(lo: i128, hi: i128, k: usize) -> Interval {
        Interval {
            at: Some(k),
            ..Interval::new(lo, hi)
        }
    }

    /// `[lo, hi]`.
    fn text(&self) -> String {
        format!("[{}, {}]", self.lo, self.hi)
    }

    /// The u32 value: the interval shifted into one period, if it lies in
    /// one.
    pub fn u32(&self) -> Option<Interval> {
        let p = self.lo.div_euclid(P32);
        (self.hi.div_euclid(P32) == p).then(|| Interval {
            lo: self.lo - p * P32,
            hi: self.hi - p * P32,
            ..self.clone()
        })
    }

    /// Whether it is known not wrapped and inside [0, 2^32).
    fn plain_u32(&self) -> bool {
        self.wrap.is_none() && self.lo >= 0 && self.hi < P32
    }
}

/// Every u32 value, lost at `wrap` for `why`.
fn lost(wrap: Option<usize>, why: String, at: Option<usize>) -> Interval {
    Interval {
        lo: 0,
        hi: P32 - 1,
        wrap,
        why,
        at,
    }
}

/// The interval itself if it spans less than 2^32 integers, else every u32
/// value with where and why it was lost.
fn bounded(iv: Interval, k: Option<usize>, parts: &str) -> Interval {
    if iv.hi - iv.lo < P32 {
        return iv;
    }
    match k {
        Some(k) if iv.wrap.is_none() => lost(
            Some(k),
            format!("computes {parts}: {} .. {}, every u32 value", iv.lo, iv.hi),
            None,
        ),
        _ => {
            let why = if iv.why.is_empty() {
                "spans every u32 value".into()
            } else {
                iv.why.clone()
            };
            lost(iv.wrap.or(iv.at), why, None)
        }
    }
}

/// The interval an op computes on the integers.
fn ring(ops: &[&Interval], lo: i128, hi: i128, k: usize, parts: &str) -> Interval {
    match ops.iter().find(|o| o.wrap.is_some()) {
        Some(w) => lost(w.wrap, w.why.clone(), Some(k)),
        None => bounded(Interval::at(lo, hi, k), Some(k), parts),
    }
}

/// The operands' u32 values, or the result lost at k.
fn u32s(ops: &[&Interval], k: usize) -> Result<Vec<Interval>, Interval> {
    ops.iter()
        .map(|o| {
            o.u32().ok_or_else(|| {
                let why = format!(
                    "reads [{}, {}], across a multiple of 2^32 (its u32 value lost)",
                    o.lo, o.hi
                );
                lost(Some(k), why, Some(k))
            })
        })
        .collect()
}

/// [lo, hi] at k with the wrap of the first wrapped operand.
fn plain(ops: &[&Interval], lo: i128, hi: i128, k: usize) -> Interval {
    let w = ops.iter().find(|o| o.wrap.is_some());
    Interval {
        lo,
        hi,
        wrap: w.and_then(|w| w.wrap),
        why: w.map(|w| w.why.clone()).unwrap_or_default(),
        at: Some(k),
    }
}

/// A single value in 0..= 31: a shift amount.
fn shift(iv: &Interval) -> Option<u32> {
    (iv.lo == iv.hi && (0..32).contains(&iv.lo)).then_some(iv.lo as u32)
}

fn bitlen(v: i128) -> u32 {
    128 - v.max(0).leading_zeros()
}

/// The result of an op by its canon on operand intervals v(1), v(2), …
/// none when it is not followed.
pub fn transfer(canon: &str, v: &[Option<Interval>], k: usize) -> Option<Interval> {
    let get = |n: usize| v.get(n - 1).and_then(Option::as_ref);
    let all = |n: usize| (1..=n).map(get).collect::<Option<Vec<&Interval>>>();
    let r = match canon {
        "s.mov.b32" | "v.mov.b32" | "v.readfirstlane.b32" => get(1)?.clone(),
        "s.cselect.b32" | "v.cndmask.b32" => {
            let o = all(2)?;
            let parts = format!("{}, {}", o[0].text(), o[1].text());
            ring(&o, o[0].lo.min(o[1].lo), o[0].hi.max(o[1].hi), k, &parts)
        }
        "s.add.u32" | "v.add.u32" | "s.add.i32" => {
            let o = all(2)?;
            ring(
                &o,
                o[0].lo + o[1].lo,
                o[0].hi + o[1].hi,
                k,
                &format!("{} + {}", o[0].text(), o[1].text()),
            )
        }
        "s.sub.u32" | "v.sub.u32" | "s.sub.i32" | "v.subrev.u32" => {
            let o = all(2)?;
            let (a, c) = if canon == "v.subrev.u32" {
                (o[1], o[0])
            } else {
                (o[0], o[1])
            };
            ring(
                &o,
                a.lo - c.hi,
                a.hi - c.lo,
                k,
                &format!("{} - {}", a.text(), c.text()),
            )
        }
        "s.lshl.b32" | "v.lshlrev.b32" => {
            let o = all(2)?;
            let (a, s) = if canon == "s.lshl.b32" {
                (o[0], o[1])
            } else {
                (o[1], o[0])
            };
            let sh = shift(s)?;
            ring(
                &o,
                a.lo << sh,
                a.hi << sh,
                k,
                &format!("{} << {sh}", a.text()),
            )
        }
        c if c.starts_with("s.lshl") && c.ends_with(".add.u32") => {
            let n: u32 = c["s.lshl".len()..c.len() - ".add.u32".len()]
                .parse()
                .ok()
                .filter(|n| (1..=4).contains(n))?;
            let o = all(2)?;
            let parts = format!("({} << {n}) + {}", o[0].text(), o[1].text());
            ring(
                &o,
                (o[0].lo << n) + o[1].lo,
                (o[0].hi << n) + o[1].hi,
                k,
                &parts,
            )
        }
        "v.lshl.add.u32" => {
            let o = all(3)?;
            let sh = shift(o[1])?;
            let parts = format!("({} << {sh}) + {}", o[0].text(), o[2].text());
            ring(
                &o,
                (o[0].lo << sh) + o[2].lo,
                (o[0].hi << sh) + o[2].hi,
                k,
                &parts,
            )
        }
        "v.add.lshl.u32" => {
            let o = all(3)?;
            let sh = shift(o[2])?;
            let parts = format!("({} + {}) << {sh}", o[0].text(), o[1].text());
            ring(
                &o,
                (o[0].lo + o[1].lo) << sh,
                (o[0].hi + o[1].hi) << sh,
                k,
                &parts,
            )
        }
        "s.mul.i32" | "v.mul.lo.u32" => {
            let o = all(2)?;
            let corners = [
                o[0].lo * o[1].lo,
                o[0].lo * o[1].hi,
                o[0].hi * o[1].lo,
                o[0].hi * o[1].hi,
            ];
            let (lo, hi) = (*corners.iter().min()?, *corners.iter().max()?);
            ring(&o, lo, hi, k, &format!("{} * {}", o[0].text(), o[1].text()))
        }
        "v.mul.u32.u24" | "v.mad.u32.u24" => {
            let mad = canon == "v.mad.u32.u24";
            let o = all(if mad { 3 } else { 2 })?;
            let u = match u32s(&o[..2], k) {
                Ok(u) => u,
                Err(lost) => return Some(lost),
            };
            let (elo, ehi) = if mad { (o[2].lo, o[2].hi) } else { (0, 0) };
            let parts = if mad {
                format!("{} * {} + {}", u[0].text(), u[1].text(), o[2].text())
            } else {
                format!("{} * {}", u[0].text(), u[1].text())
            };
            ring(
                &o,
                u[0].lo * u[1].lo + elo,
                u[0].hi * u[1].hi + ehi,
                k,
                &parts,
            )
        }
        "s.lshr.b32" | "v.lshrrev.b32" => {
            let o = all(2)?;
            let u = match u32s(&o, k) {
                Ok(u) => u,
                Err(lost) => return Some(lost),
            };
            let (a, s) = if canon == "s.lshr.b32" {
                (&u[0], &u[1])
            } else {
                (&u[1], &u[0])
            };
            let shr = |x: i128, by: i128| if by >= 64 { 0 } else { x >> by };
            plain(&o, shr(a.lo, s.hi.min(31)), shr(a.hi, s.lo), k)
        }
        "s.and.b32" | "v.and.b32" => {
            let known: Vec<&Interval> = (1..=2).filter_map(get).collect();
            if known.is_empty() {
                return None;
            }
            let u = match u32s(&known, k) {
                Ok(u) => u,
                Err(lost) => return Some(lost),
            };
            plain(&known, 0, u.iter().map(|x| x.hi).min()?, k)
        }
        "s.or.b32" | "v.or.b32" | "s.xor.b32" | "v.xor.b32" => {
            let o = all(2)?;
            let u = match u32s(&o, k) {
                Ok(u) => u,
                Err(lost) => return Some(lost),
            };
            plain(&o, 0, (1i128 << bitlen(u[0].hi.max(u[1].hi))) - 1, k)
        }
        "v.lshl.or.b32" => {
            let o = all(3)?;
            let u = match u32s(&o, k) {
                Ok(u) => u,
                Err(lost) => return Some(lost),
            };
            let top = if u[0].hi == 0 {
                u[2].hi
            } else if u[1].hi >= 32 {
                return None;
            } else {
                (u[0].hi << u[1].hi) | u[2].hi
            };
            if top >= P32 {
                return None;
            }
            plain(&o, 0, (1i128 << bitlen(top)) - 1, k)
        }
        "s.min.u32" | "v.min.u32" | "s.max.u32" | "v.max.u32" => {
            let o = all(2)?;
            let u = match u32s(&o, k) {
                Ok(u) => u,
                Err(lost) => return Some(lost),
            };
            if canon.contains(".min.") {
                plain(&o, u[0].lo.min(u[1].lo), u[0].hi.min(u[1].hi), k)
            } else {
                plain(&o, u[0].lo.max(u[1].lo), u[0].hi.max(u[1].hi), k)
            }
        }
        "v.bfe.u32" | "s.bfe.u32" => bfe(canon == "s.bfe.u32", get(1)?, get(2), get(3), k)?,
        _ => return None,
    };
    Some(r)
}

/// A bit-field extract.
fn bfe(
    scalar: bool,
    x: &Interval,
    a: Option<&Interval>,
    b: Option<&Interval>,
    k: usize,
) -> Option<Interval> {
    let single = |i: Option<&Interval>| i.filter(|i| i.lo == i.hi).map(|i| i.lo);
    let (field, width) = if scalar {
        let c = single(a);
        (
            c.map(|c| (c & 31, (c >> 16) & 0x7F)),
            c.map(|c| (c >> 16) & 0x7F),
        )
    } else {
        let field = single(a).zip(single(b)).map(|(o, w)| (o & 31, w & 31));
        (field, b.map(|w| w.hi))
    };
    let xu = x.u32().filter(|u| u.lo == u.hi && x.wrap.is_none());
    if let (Some(xu), Some((off, w))) = (xu, field) {
        let v = xu.lo >> off;
        let v = if w >= 64 { v } else { v & ((1i128 << w) - 1) };
        return Some(Interval::at(v, v, k));
    }
    let w = width.filter(|&w| w <= 32)?;
    Some(Interval::at(0, (1i128 << w) - 1, k))
}

/// A key of the state.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Key {
    /// A physical dword.
    Dword(char, u32),
    /// s[0:1] still hold the kernel-argument pointer.
    Karg,
    /// The lanes of a mask: VCC (none), or an SGPR mask's first dword.
    Lanes(Option<u32>),
    /// A narrowed register.
    Narrowed(char, u32),
    /// A named register (M0, the workgroup ids).
    Named(usize),
}

/// An entry of the state.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Entry {
    Iv(Interval),
    Present,
    Lanes {
        reg: (char, u32),
        lo: i128,
        hi: i128,
        signed: bool,
    },
    Narrowed {
        before: Interval,
        after: Interval,
    },
}

type State = BTreeMap<Key, Entry>;

/// A kernel argument: offset, bytes, whether a u32, its declared
/// range, whether a divisor.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Arg {
    /// Its byte offset.
    pub offset: u32,
    /// Its bytes.
    pub bytes: u32,
    /// Whether it is a u32.
    pub is_u32: bool,
    /// Its declared least value.
    pub lo: Option<i128>,
    /// Its declared greatest value.
    pub hi: Option<i128>,
    /// Whether it is a divisor (its magic pair follows).
    pub divisor: bool,
}

/// What a launch tells the analysis.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Launch {
    /// The kernel arguments.
    pub args: Vec<Arg>,
    /// Lanes per workgroup.
    pub wgsize: u64,
    /// 1 for a 2-D grid.
    pub wgy: u32,
    /// The largest grid per dimension.
    pub max_grid: Vec<u64>,
    /// Argument dwords preloaded into SGPRs.
    pub preload: u32,
    /// One wave's thread ids, if the analysis is per wave.
    pub tid: Option<(i128, i128)>,
}

impl Launch {
    /// The argument dword at byte b, if known.
    fn arg_dword(&self, b: i128) -> Option<Interval> {
        let at = |a: &Arg| i128::from(a.offset) == b && (a.is_u32 && a.bytes == 4 || a.divisor);
        if let Some(a) = self.args.iter().find(|a| at(a)) {
            if let (Some(lo), Some(hi)) = (a.lo, a.hi) {
                return Some(Interval::new(lo, hi));
            }
        }
        self.args
            .iter()
            .any(|a| a.divisor && i128::from(a.offset) == b - 8)
            .then(|| Interval::new(0, 31))
    }
}

/// The report.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Report {
    /// Address offsets proven not to wrap.
    pub proven: u32,
    /// Address offsets not proven.
    pub unproven: u32,
    /// LDS accesses proven inside their regions.
    pub inside: u32,
    /// LDS accesses not proven.
    pub unknown: u32,
    /// Per proven LDS access, the bytes it may touch, [lo, hi): its
    /// address register's bounds plus its offset field, to its last byte.
    pub lds_bytes: BTreeMap<usize, (i128, i128)>,
}

impl Report {
    /// The report's line.
    pub fn line(&self) -> String {
        let mut out = format!(
            "ranges: {} address offsets proven not to wrap, {} unproven",
            self.proven, self.unproven
        );
        if self.inside != 0 || self.unknown != 0 {
            out.push_str(&format!(
                "; {} LDS accesses proven inside their regions, {} unproven",
                self.inside, self.unknown
            ));
        }
        out
    }
}

/// The analysis over a snapshot and a launch.
struct Ranges<'a> {
    s: &'a Snapshot,
    launch: &'a Launch,
}

fn dword_key(s: &Snapshot, r: &Reg) -> Option<Key> {
    (r.n == 1).then(|| Key::Dword(r.file, s.phys(r)))
}

impl<'a> Ranges<'a> {
    fn entry(&self) -> State {
        let l = self.launch;
        let mut st = State::new();
        st.insert(Key::Karg, Entry::Present);
        if l.wgsize >= 1 {
            let (lo, hi) = l.tid.unwrap_or((0, i128::from(l.wgsize) - 1));
            st.insert(Key::Dword('v', 0), Entry::Iv(Interval::new(lo, hi)));
        }
        for d in 0..l.preload {
            if let Some(iv) = l.arg_dword(4 * i128::from(d)) {
                st.insert(Key::Dword('s', 2 + d), Entry::Iv(iv));
            }
        }
        let wgids = &self.s.target().definition().chip.wgids;
        for d in 0..(1 + l.wgy as usize).min(l.max_grid.len()) {
            let iv = Interval::new(0, (l.max_grid[d] as i128 - 1).max(0));
            let key = match wgids.get(d).and_then(|n| phys_named(n)) {
                Some(p) => Key::Named(p.number),
                None => Key::Dword('s', 2 + l.preload + d as u32),
            };
            st.insert(key, Entry::Iv(iv));
        }
        st
    }

    fn iv(st: &State, key: Key) -> Option<&Interval> {
        match st.get(&key) {
            Some(Entry::Iv(iv)) => Some(iv),
            _ => None,
        }
    }

    /// Operand n's interval.
    fn operand(&self, st: &State, i: &Instr, n: usize) -> Option<Interval> {
        if i.modified(n) {
            return None;
        }
        match i.ops().get(n)? {
            Operand::Reg(r) => Ranges::iv(st, dword_key(self.s, r)?).cloned(),
            Operand::Imm(m) => match m.value {
                ImmValue::Int(v) if (-(1i128 << 31)..P32).contains(&v) => Some(Interval::new(v, v)),
                _ => None,
            },
            Operand::Phys(p) => Ranges::iv(st, Key::Named(p.number)).cloned(),
            Operand::Block(_) => None,
        }
    }

    fn mask(&self, name: &str) -> &'static str {
        self.s
            .target()
            .mask_reg(phys_named(name).expect("a lane mask"))
            .name
    }

    /// The lanes key of a mask operand: VCC as the target's mask register,
    /// or an SGPR value of the mask's width (one dword on a 32-lane
    /// target, two on a 64-lane one).
    fn lanes_key(&self, op: &Operand) -> Option<Key> {
        let dwords = self.s.target().definition().wave / 32;
        match op {
            Operand::Phys(p) if p.name == self.mask("vcc") => Some(Key::Lanes(None)),
            Operand::Reg(r) if r.file == 's' && r.n == dwords => {
                Some(Key::Lanes(Some(self.s.phys(r))))
            }
            _ => None,
        }
    }

    /// A compare's lanes entry.
    fn lanes(&self, st: &State, i: &Instr) -> Option<(Key, Entry)> {
        let key = self.lanes_key(i.ops().first()?)?;
        let words: Vec<&str> = i.form().canon.split('.').collect();
        let rel: &'static str = ["lt", "le", "gt", "ge", "eq", "ne"]
            .into_iter()
            .find(|r| words.get(2) == Some(r))?;
        let ty = *words.get(3)?;
        if i.ops().len() < 3 || !["u32", "i32"].contains(&ty) {
            return None;
        }
        let flip = |r: &'static str| -> &'static str {
            match r {
                "lt" => "gt",
                "gt" => "lt",
                "le" => "ge",
                "ge" => "le",
                "eq" => "eq",
                _ => "ne",
            }
        };
        for (n, other, rel) in [(1, 2, rel), (2, 1, flip(rel))] {
            let Some(reg) = i.ops()[n].reg().filter(|r| r.n == 1) else {
                continue;
            };
            let Some(o) = self.operand(st, i, other).filter(Interval::plain_u32) else {
                continue;
            };
            let (lo, hi) = match rel {
                "lt" => (0, o.hi - 1),
                "le" => (0, o.hi),
                "gt" => (o.lo + 1, P32 - 1),
                "ge" => (o.lo, P32 - 1),
                "eq" => (o.lo, o.hi),
                "ne" => (0, P32 - 1),
                _ => return None,
            };
            let signed = ty == "i32" && ["lt", "le"].contains(&rel);
            return Some((
                key,
                Entry::Lanes {
                    reg: (reg.file, self.s.phys(reg)),
                    lo,
                    hi,
                    signed,
                },
            ));
        }
        None
    }

    /// An EXEC write's restore and narrowing.
    fn exec_write(&self, st: &mut State, i: &Instr) {
        let narrowed: Vec<(Key, Entry)> = st
            .iter()
            .filter(|(k, _)| matches!(k, Key::Narrowed(..)))
            .map(|(k, e)| (*k, e.clone()))
            .collect();
        for (k, e) in narrowed {
            let (Key::Narrowed(f, d), Entry::Narrowed { before, after }) = (k, e) else {
                continue;
            };
            if Ranges::iv(st, Key::Dword(f, d)) == Some(&after) {
                st.insert(Key::Dword(f, d), Entry::Iv(before));
            }
            st.remove(&k);
        }
        let t = self.s.target();
        let canon = i.form().canon.as_str();
        let exec = self.mask("exec");
        let to_exec = i.phys(0).is_some_and(|p| p.name == exec);
        let source = if canon == t.masked("s.mov.mask") && to_exec
            || canon == t.masked("s.and.saveexec.mask")
        {
            i.ops().get(1)
        } else if canon == t.masked("s.and.mask")
            && to_exec
            && i.phys(1).is_some_and(|p| p.name == exec)
        {
            i.ops().get(2)
        } else {
            None
        };
        let Some(Entry::Lanes {
            reg,
            lo,
            hi,
            signed,
        }) = source
            .and_then(|o| self.lanes_key(o))
            .and_then(|k| st.get(&k))
            .cloned()
        else {
            return;
        };
        let key = Key::Dword(reg.0, reg.1);
        let Some(cur) = Ranges::iv(st, key).cloned() else {
            return;
        };
        if cur.wrap.is_some() || signed && cur.hi >= 1 << 31 {
            return;
        }
        let (nlo, nhi) = (cur.lo.max(lo), cur.hi.min(hi));
        if nlo <= nhi {
            let after = Interval {
                lo: nlo,
                hi: nhi,
                ..cur.clone()
            };
            st.insert(key, Entry::Iv(after.clone()));
            st.insert(
                Key::Narrowed(reg.0, reg.1),
                Entry::Narrowed { before: cur, after },
            );
        }
    }

    /// One instruction's step.
    fn step(&self, st: &mut State, k: usize) {
        let i = &self.s.instrs()[k];
        let f = i.form();
        if f.pseudo == Some(Pseudo::Assume) {
            self.assume(st, i, k);
            return;
        }
        if matches!(
            f.kind,
            Kind::Label | Kind::Comment | Kind::Nop | Kind::Wait | Kind::Branch | Kind::Barrier
        ) {
            return;
        }
        let ops: Vec<Option<Interval>> =
            (1..i.ops().len()).map(|n| self.operand(st, i, n)).collect();
        let m0 = phys_named("m0").expect("m0").number;
        if i.phys(0).is_some_and(|p| p.number == m0) {
            let result = (["s.mov.b32", "s.add.u32", "s.add.i32", "s.sub.u32"]
                .contains(&f.canon.as_str()))
            .then(|| transfer(&f.canon, &ops, k))
            .flatten();
            match result {
                Some(iv) => st.insert(Key::Named(m0), Entry::Iv(iv)),
                None => st.remove(&Key::Named(m0)),
            };
            return;
        }
        let op0_mask = i
            .phys(0)
            .is_some_and(|p| p.name == self.mask("vcc") || p.name == self.mask("exec"));
        if i.defs().next().is_none() && !op0_mask && !i.writes_exec() {
            return;
        }
        let compare = f.flags.has(Flags::CMP) && !f.flags.has(Flags::CMPX);
        let mut results: Vec<(Key, Entry)> = Vec::new();
        if compare {
            results.extend(self.lanes(st, i));
        } else if f.canon.starts_with("s.load.b") {
            results.extend(self.loads(st, i));
        } else if let Some(key) = i
            .ops()
            .first()
            .and_then(Operand::reg)
            .and_then(|r| dword_key(self.s, r))
        {
            results.extend(transfer(&f.canon, &ops, k).map(|iv| (key, Entry::Iv(iv))));
        }
        if i.writes_exec() {
            self.exec_write(st, i);
        }
        if i.phys(0).is_some_and(|p| p.name == self.mask("vcc")) || compare {
            st.remove(&Key::Lanes(None));
        }
        for r in i.defs() {
            let p = self.s.phys(r);
            for d in p..p + r.n {
                st.remove(&Key::Dword(r.file, d));
                st.remove(&Key::Narrowed(r.file, d));
                if r.file == 's' {
                    st.remove(&Key::Lanes(Some(d)));
                    if d < 2 {
                        st.remove(&Key::Karg);
                    }
                }
            }
        }
        st.extend(results);
    }

    /// A scalar load's argument dwords.
    fn loads(&self, st: &State, i: &Instr) -> Vec<(Key, Entry)> {
        let base = i
            .ops()
            .get(1)
            .and_then(Operand::reg)
            .is_some_and(|r| r.file == 's' && self.s.phys(r) == 0);
        let offset = match i.ops().get(2) {
            Some(Operand::Imm(m)) => match m.value {
                ImmValue::Int(v) => Some(v),
                ImmValue::Float(_) => None,
            },
            _ => None,
        };
        let (Some(off), true, Some(dst)) = (
            offset,
            base && st.contains_key(&Key::Karg),
            i.ops().first().and_then(Operand::reg),
        ) else {
            return Vec::new();
        };
        let p = self.s.phys(dst);
        (0..dst.n)
            .filter_map(|n| {
                self.launch
                    .arg_dword(off + 4 * i128::from(n))
                    .map(|iv| (Key::Dword('s', p + n), Entry::Iv(iv)))
            })
            .collect()
    }

    /// An assume claim: each dword of its operand holds [lo, hi].
    fn assume(&self, st: &mut State, i: &Instr, k: usize) {
        let int = |name| i.field_named(name).and_then(FieldValue::int);
        let (Some(lo), Some(hi), Some(r)) =
            (int("lo"), int("hi"), i.ops().first().and_then(Operand::reg))
        else {
            return;
        };
        let p = self.s.phys(r);
        for d in p..p + r.n {
            let key = Key::Dword(r.file, d);
            let mut iv = Interval::at(lo, hi, k);
            if let Some(known) =
                Ranges::iv(st, key).filter(|x| x.plain_u32() && x.lo <= hi && lo <= x.hi)
            {
                iv.lo = iv.lo.max(known.lo);
                iv.hi = iv.hi.min(known.hi);
            }
            st.insert(key, Entry::Iv(iv));
        }
    }
}

/// Where paths meet: the keys on both, intervals hulled.
fn merge(a: &State, b: &State) -> State {
    let mut out = State::new();
    for (k, x) in a {
        let Some(y) = b.get(k) else { continue };
        match (x, y) {
            (Entry::Iv(x), Entry::Iv(y)) => {
                let first = if x.wrap.is_some() { x } else { y };
                let hull = Interval {
                    lo: x.lo.min(y.lo),
                    hi: x.hi.max(y.hi),
                    wrap: first.wrap,
                    why: first.why.clone(),
                    at: if x.at == y.at { x.at } else { None },
                };
                out.insert(*k, Entry::Iv(bounded(hull, None, "")));
            }
            (x, y) if x == y => _ = out.insert(*k, x.clone()),
            _ => {}
        }
    }
    out
}

/// The analysis of a snapshot: the report, or the first RangeError.
pub fn ranges(s: &Snapshot, launch: &Launch) -> Result<Report, Diag> {
    let r = Ranges { s, launch };
    let g = s.cfg(None);
    let mut report = Report::default();
    if g.nblocks() == 0 {
        return Ok(report);
    }
    let through = |st: &State, j: usize| {
        let mut st = st.clone();
        g.block(j).for_each(|k| r.step(&mut st, k));
        st
    };
    let widen = |old: &State, new: &State| {
        old.iter()
            .filter(|(k, e)| new.get(k) == Some(e))
            .map(|(k, e)| (*k, e.clone()))
            .collect()
    };
    let entries = solve_forward(
        &g,
        vec![(0, r.entry())],
        through,
        merge,
        Some(Widen {
            widen: &widen,
            patience: 3,
        }),
    );
    for (j, st) in entries.iter().enumerate() {
        let Some(mut st) = st.clone() else { continue };
        for k in g.block(j) {
            r.check(&st, k, &mut report)?;
            r.step(&mut st, k);
        }
    }
    Ok(report)
}

impl Ranges<'_> {
    /// The address operand of an instruction.
    /// A global access with a scalar base has a 32-bit VGPR offset; a
    /// buffer access with `offen` has its offset VGPR after the data. The
    /// form's memory class says which, on any target.
    fn address(&self, i: &Instr) -> Option<usize> {
        let f = i.form();
        let truthy = |name: &str| i.field_named(name).is_some_and(FieldValue::truthy);
        let has_defs = i.defs().next().is_some();
        if f.mem == Some(Mem::Global)
            && i.ops()
                .iter()
                .any(|o| matches!(o, Operand::Reg(r) if r.file == 's' && r.n == 2))
        {
            return Some(usize::from(has_defs));
        }
        if f.mem == Some(Mem::Buffer) && truthy("offen") {
            return Some(usize::from(!f.flags.has(Flags::LDS_MEM)));
        }
        (f.kind == Kind::Ds && !f.flags.has(Flags::CROSS_LANE)).then_some(usize::from(has_defs))
    }

    fn quoted(k: usize) -> Vec<Part> {
        vec![
            Part::Text("`".into()),
            Part::Instr(k),
            Part::Text(format!("` (instruction {k}): ")),
        ]
    }

    fn refusal(msg: Vec<Part>) -> Diag {
        Diag {
            class: DiagClass::Range,
            loc: 0,
            msg,
            notes: Vec::new(),
        }
    }

    /// The checks of one instruction on the state before it.
    fn check(&self, st: &State, k: usize, report: &mut Report) -> Result<(), Diag> {
        let i = &self.s.instrs()[k];
        if ["v.mul.u32.u24", "v.mad.u32.u24"].contains(&i.form().canon.as_str()) {
            for n in 1..=2 {
                // An operand with no known bound may pass 24 bits as well.
                let Some(iv) = self.operand(st, i, n) else {
                    let mut msg = Ranges::quoted(k);
                    msg.push(Part::Text(format!(
                        "operand {n} has no known bound, so it may pass 24 bits (the instruction multiplies their low 24 bits)"
                    )));
                    return Err(Ranges::refusal(msg));
                };
                let u = iv.u32();
                if u.as_ref()
                    .is_none_or(|u| iv.wrap.is_some() || u.hi >= 1 << 24)
                {
                    let top = u.filter(|_| iv.wrap.is_none()).map_or(P32 - 1, |u| u.hi);
                    let mut msg = Ranges::quoted(k);
                    let wrapped = iv.wrap.map_or(String::new(), |w| {
                        format!(" (it may have wrapped at instruction {w})")
                    });
                    msg.push(Part::Text(format!(
                        "operand {n} may be up to {top}, past 24 bits (the instruction multiplies their low 24 bits){wrapped}"
                    )));
                    return Err(Ranges::refusal(msg));
                }
            }
        }
        let f = i.form();
        let vmem_lds = f.kind == Kind::Vmem && f.flags.has(Flags::LDS_MEM);
        let region = i
            .attrs()
            .lds_access
            .as_ref()
            .and_then(|acc| self.s.region(acc.region).map(|r| (acc, r)));
        if let Some(a) = self.address(i) {
            let ds_reg = (!vmem_lds)
                .then_some(region)
                .flatten()
                .map(|r| (r.0.field, r.1));
            self.offset(st, i, k, a, ds_reg, report)?;
        }
        if let (true, Some((_, reg))) = (vmem_lds, region) {
            self.dma_lds(st, k, f.mem_bytes, reg, report)?;
        }
        Ok(())
    }

    /// An M0-addressed VMEM-to-LDS DMA's destination range inside its LDS region.
    fn dma_lds(
        &self,
        st: &State,
        k: usize,
        mem_bytes: u32,
        reg: &thin_ir::Region,
        report: &mut Report,
    ) -> Result<(), Diag> {
        let m0 = phys_named("m0").expect("m0").number;
        let Some(iv) = Ranges::iv(st, Key::Named(m0)) else {
            report.unknown += 1;
            return Ok(());
        };
        if let Some(w) = iv.wrap {
            let mut msg = Ranges::quoted(k);
            msg.push(Part::Text("its M0 offset may wrap past 32 bits - `".into()));
            msg.extend([
                Part::Instr(w),
                Part::Text(format!("` (instruction {w}) {}", iv.why)),
            ]);
            return Err(Ranges::refusal(msg));
        }
        if iv.lo < 0 || iv.hi >= P32 {
            let mut msg = Ranges::quoted(k);
            msg.push(Part::Text(format!(
                "its M0 offset may be {} .. {}, outside 0 .. 2^32 - 1",
                iv.lo, iv.hi
            )));
            msg.extend(computed(iv));
            return Err(Ranges::refusal(msg));
        }
        let wave = u64::from(self.s.target().definition().wave);
        let lanes = match self.launch.tid {
            Some((lo, hi)) => (hi - lo + 1).max(0),
            None if self.launch.wgsize > 0 => i128::from(self.launch.wgsize.min(wave)),
            None => i128::from(wave),
        };
        let stride = if mem_bytes > 4 { 16 } else { 4 };
        let bytes = (iv.lo, iv.hi + stride * lanes);
        self.inside(k, report, bytes, iv, reg)
    }

    /// An address offset's checks.
    fn offset(
        &self,
        st: &State,
        i: &Instr,
        k: usize,
        a: usize,
        region: Option<(i128, &thin_ir::Region)>,
        report: &mut Report,
    ) -> Result<(), Diag> {
        let Some(iv) = self.operand(st, i, a) else {
            report.unproven += 1;
            report.unknown += u32::from(region.is_some());
            return Ok(());
        };
        if let Some(w) = iv.wrap {
            let mut msg = Ranges::quoted(k);
            msg.push(Part::Text(
                "its address offset may wrap past 32 bits - `".into(),
            ));
            msg.extend([
                Part::Instr(w),
                Part::Text(format!("` (instruction {w}) {}", iv.why)),
            ]);
            return Err(Ranges::refusal(msg));
        }
        if iv.lo < 0 || iv.hi >= P32 {
            let mut msg = Ranges::quoted(k);
            msg.push(Part::Text(format!(
                "its address offset may be {} .. {}, outside 0 .. 2^32 - 1",
                iv.lo, iv.hi
            )));
            msg.extend(computed(&iv));
            return Err(Ranges::refusal(msg));
        }
        report.proven += 1;
        if let Some((field, reg)) = region {
            let bytes = (
                iv.lo + field,
                iv.hi + field + i128::from(i.form().mem_bytes),
            );
            self.inside(k, report, bytes, &iv, reg)?;
        }
        Ok(())
    }

    /// LDS bytes inside their region.
    fn inside(
        &self,
        k: usize,
        report: &mut Report,
        bytes: (i128, i128),
        iv: &Interval,
        r: &thin_ir::Region,
    ) -> Result<(), Diag> {
        let (base, end) = (i128::from(r.base), i128::from(r.base + r.size));
        if bytes.0 < base || bytes.1 > end {
            let mut msg = Ranges::quoted(k);
            msg.push(Part::Text(format!(
                "its bytes may be [{}, {}), outside {} [{base}, {end})",
                bytes.0, bytes.1, r.name
            )));
            if iv.at.is_some() {
                msg.push(Part::Text(" - its address".into()));
                msg.extend(computed(iv).into_iter().skip(1));
            }
            return Err(Ranges::refusal(msg));
        }
        report.inside += 1;
        report.lds_bytes.insert(k, bytes);
        Ok(())
    }
}

/// " - computed by `AT` (instruction AT)" when it was computed.
fn computed(iv: &Interval) -> Vec<Part> {
    match iv.at {
        Some(at) => vec![
            Part::Text(" - computed by `".into()),
            Part::Instr(at),
            Part::Text(format!("` (instruction {at})")),
        ],
        None => Vec::new(),
    }
}

/// The per-wave analyses: one per wave of the target's width, each
/// with its thread ids.
pub fn per_wave(s: &Snapshot, launch: &Launch) -> Vec<Result<Report, Diag>> {
    let wave = u64::from(s.target().definition().wave);
    (0..launch.wgsize.div_ceil(wave))
        .map(|w| {
            let first = i128::from(w * wave);
            let last = (first + i128::from(wave)).min(i128::from(launch.wgsize)) - 1;
            let tid = (first, last);
            ranges(
                s,
                &Launch {
                    tid: Some(tid),
                    ..launch.clone()
                },
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounded_loses_wide_intervals() {
        let iv = bounded(Interval::at(0, P32, 3), Some(3), "x");
        assert_eq!((iv.lo, iv.hi, iv.wrap, iv.at), (0, P32 - 1, Some(3), None));
        assert_eq!(iv.why, "computes x: 0 .. 4294967296, every u32 value");
        let narrow = Interval::at(-5, 5, 1);
        assert_eq!(bounded(narrow.clone(), Some(1), ""), narrow);
        assert_eq!(
            Interval::new(P32 + 1, P32 + 3).u32(),
            Some(Interval::new(1, 3))
        );
        assert_eq!(Interval::new(-1, 1).u32(), None);
    }
}
