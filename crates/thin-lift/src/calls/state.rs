//! Bounded scalar constants and correlated, relocation-invariant PC pairs.

use super::Reason;
use crate::program::Program;
use std::sync::Arc;
use thin_ir::{Imm, ImmValue, Instr, Operand};

const SCALARS: usize = 64;
const ADDRESSES: usize = 4096;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum AddressRole {
    Code,
    Return(usize),
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct Address {
    pub offset: i64,
    pub role: AddressRole,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum Word {
    Unknown(Reason),
    Values(Arc<[u32]>),
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum Pair {
    Unknown(Reason),
    Values(Arc<[Address]>),
    Pending(Arc<[(Address, u32)]>),
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Carry {
    Unknown,
    Constant(bool),
    Pair(u32),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct State {
    words: Vec<Word>,
    pairs: Vec<Pair>,
    carry: Carry,
}

fn union<T: Ord + Clone>(a: &[T], b: &[T], limit: usize) -> Option<Arc<[T]>> {
    let mut values = a.to_vec();
    values.extend_from_slice(b);
    values.sort();
    values.dedup();
    (values.len() <= limit).then(|| values.into())
}
impl Word {
    fn constant(n: u32) -> Self {
        Self::Values(vec![n].into())
    }
    fn join(&self, other: &Self, back: bool) -> Self {
        if self == other {
            return self.clone();
        }
        match (self, other) {
            (Self::Unknown(a), Self::Unknown(b)) => Self::Unknown((*a).min(*b)),
            (Self::Unknown(r), _) | (_, Self::Unknown(r)) => Self::Unknown(*r),
            (Self::Values(_), Self::Values(_)) if back => Self::Unknown(Reason::LoopValue),
            (Self::Values(a), Self::Values(b)) => {
                union(a, b, SCALARS).map_or(Self::Unknown(Reason::ScalarLimit), Self::Values)
            }
        }
    }
}
impl Pair {
    fn join(&self, other: &Self) -> Self {
        if self == other {
            return self.clone();
        }
        match (self, other) {
            (Self::Unknown(a), Self::Unknown(b)) => Self::Unknown((*a).min(*b)),
            (Self::Unknown(r), _) | (_, Self::Unknown(r)) => Self::Unknown(*r),
            (Self::Values(a), Self::Values(b)) => {
                union(a, b, ADDRESSES).map_or(Self::Unknown(Reason::AddressLimit), Self::Values)
            }
            (Self::Pending(a), Self::Pending(b)) => {
                union(a, b, ADDRESSES).map_or(Self::Unknown(Reason::AddressLimit), Self::Pending)
            }
            _ => Self::Unknown(Reason::PartialAddress),
        }
    }
}
fn scalar(o: &Operand) -> Option<u32> {
    o.reg().filter(|r| r.file == 's' && r.n == 1).map(|r| r.lo)
}
fn pair(o: &Operand) -> Option<u32> {
    o.reg()
        .filter(|r| r.file == 's' && r.n == 2 && r.lo % 2 == 0)
        .map(|r| r.lo)
}
fn constant(o: &Operand) -> Option<i128> {
    match o {
        Operand::Imm(Imm {
            value: ImmValue::Int(n),
            ..
        }) => Some(*n),
        _ => None,
    }
}

fn integer64(o: &Operand) -> Option<i64> {
    // The decoder keeps signed inline integers negative and literal words
    // nonnegative. Preserve that distinction for unsigned/binary 64-bit
    // sources.
    constant(o).and_then(|n| i64::try_from(n).ok())
}

impl State {
    pub(super) fn new(n: u32) -> Self {
        Self {
            words: vec![Word::Unknown(Reason::Input); n as usize],
            pairs: vec![Pair::Unknown(Reason::Input); n.div_ceil(2) as usize],
            carry: Carry::Unknown,
        }
    }
    pub(super) fn address(&self, first: u32) -> &Pair {
        &self.pairs[first as usize / 2]
    }
    pub(super) fn join(&mut self, other: &Self, back: bool) -> bool {
        let mut changed = false;
        for (a, b) in self.words.iter_mut().zip(&other.words) {
            let next = a.join(b, back);
            changed |= *a != next;
            *a = next;
        }
        for (a, b) in self.pairs.iter_mut().zip(&other.pairs) {
            let next = a.join(b);
            changed |= *a != next;
            *a = next;
        }
        let carry = if self.carry == other.carry {
            self.carry
        } else {
            Carry::Unknown
        };
        changed |= carry != self.carry;
        self.carry = carry;
        changed
    }
    pub(super) fn without_link(&self, pair: u32) -> Self {
        let mut summary = self.clone();
        summary.set_pair(pair, Pair::Unknown(Reason::PartialAddress));
        summary
    }

    pub(super) fn restrict_return(&mut self, pair: u32, caller: usize) {
        if let Pair::Values(values) = self.address(pair) {
            let values: Vec<_> = values
                .iter()
                .copied()
                .filter(|a| a.role == AddressRole::Return(caller))
                .collect();
            self.set_pair(pair, Pair::Values(values.into()));
        }
    }

    fn word(&self, o: &Operand) -> Word {
        if let Some(n) = constant(o) {
            return Word::constant(n as u32);
        }
        if let Some(n) = scalar(o) {
            if self.words[n as usize] == Word::Unknown(Reason::PartialAddress) {
                if let Pair::Unknown(reason) = self.address(n & !1) {
                    return Word::Unknown(*reason);
                }
            }
            return self.words[n as usize].clone();
        }
        if o.phys().is_some_and(|p| p.name == "scc") {
            if let Carry::Constant(c) = self.carry {
                return Word::constant(u32::from(c));
            }
        }
        Word::Unknown(Reason::Operand)
    }
    fn set_word(&mut self, n: u32, word: Word) {
        self.pairs[n as usize / 2] = Pair::Unknown(match &word {
            Word::Unknown(r) => *r,
            _ if matches!(
                self.pairs[n as usize / 2],
                Pair::Values(_) | Pair::Pending(_)
            ) =>
            {
                Reason::PartialAddress
            }
            _ => Reason::NotPcRelative,
        });
        self.words[n as usize] = word;
    }
    fn set_pair(&mut self, first: u32, value: Pair) {
        let reason = match value {
            Pair::Unknown(r) => r,
            _ => Reason::PartialAddress,
        };
        self.words[first as usize] = Word::Unknown(reason);
        self.words[first as usize + 1] = Word::Unknown(reason);
        self.pairs[first as usize / 2] = value;
    }
    fn read_pair(&self, o: &Operand) -> Pair {
        match pair(o) {
            Some(n) => self.address(n).clone(),
            None => Pair::Unknown(Reason::Operand),
        }
    }
    fn kill(&mut self, i: &Instr, row: usize) {
        for r in i.defs().filter(|r| r.file == 's') {
            for n in r.lo..r.lo + r.n {
                self.set_word(n, Word::Unknown(Reason::Written(row)));
            }
        }
        if i.writes()
            .contains(&thin_target::phys_named("scc").expect("scc").number)
        {
            self.carry = Carry::Unknown;
        }
    }
    fn shifted(values: &[Address], delta: i64) -> Pair {
        let values: Option<Vec<_>> = values
            .iter()
            .map(|a| {
                Some(Address {
                    offset: a.offset.checked_add(delta)?,
                    role: if delta == 0 {
                        a.role
                    } else {
                        AddressRole::Code
                    },
                })
            })
            .collect();
        values.map_or(Pair::Unknown(Reason::AddressRange), |v| {
            Pair::Values(v.into())
        })
    }
    fn low_add(&mut self, dst: u32, source: &Operand, addend: &Operand) -> bool {
        if dst % 2 != 0 || scalar(source) != Some(dst) {
            return false;
        }
        let Pair::Values(addresses) = self.address(dst).clone() else {
            return false;
        };
        let constants = match self.word(addend) {
            Word::Values(values) => values,
            Word::Unknown(reason) => {
                self.set_word(dst, Word::Unknown(reason));
                self.carry = Carry::Unknown;
                return true;
            }
        };
        if addresses.len() > 1 && constants.len() > 1 {
            self.set_word(dst, Word::Unknown(Reason::Correlation));
            self.carry = Carry::Unknown;
            return true;
        }
        let mut terms = Vec::new();
        for a in addresses.iter() {
            for &b in constants.iter() {
                terms.push((*a, b));
            }
        }
        if terms.len() > ADDRESSES {
            self.set_word(dst, Word::Unknown(Reason::AddressLimit));
            self.carry = Carry::Unknown;
            return true;
        }
        terms.sort_unstable();
        terms.dedup();
        self.set_pair(dst, Pair::Pending(terms.into()));
        self.carry = Carry::Pair(dst);
        true
    }
    fn high_add(&mut self, dst: u32, source: &Operand, addend: &Operand) -> bool {
        if dst % 2 != 1 || scalar(source) != Some(dst) {
            return false;
        }
        let Pair::Pending(terms) = self.address(dst - 1).clone() else {
            return false;
        };
        let Word::Values(high) = self.word(addend) else {
            self.set_word(dst, Word::Unknown(Reason::Carry));
            self.carry = Carry::Unknown;
            return true;
        };
        if high.len() != 1 || self.carry != Carry::Pair(dst - 1) {
            self.set_word(dst, Word::Unknown(Reason::Carry));
            self.carry = Carry::Unknown;
            return true;
        }
        let values: Option<Vec<_>> = terms
            .iter()
            .map(|(a, low)| {
                let delta = (u64::from(*low) | u64::from(high[0]) << 32) as i64;
                Some(Address {
                    offset: a.offset.checked_add(delta)?,
                    role: if delta == 0 {
                        a.role
                    } else {
                        AddressRole::Code
                    },
                })
            })
            .collect();
        self.set_pair(
            dst - 1,
            values.map_or(Pair::Unknown(Reason::AddressRange), |mut v| {
                v.sort_unstable();
                v.dedup();
                Pair::Values(v.into())
            }),
        );
        self.carry = Carry::Unknown;
        true
    }
    fn binary(&mut self, i: &Instr, row: usize, op: &str) -> bool {
        let [dst, a, b] = i.ops() else { return false };
        let Some(dst) = scalar(dst) else { return false };
        if op == "s.add.u32" && (self.low_add(dst, a, b) || self.low_add(dst, b, a)) {
            return true;
        }
        if op == "s.addc.u32" && (self.high_add(dst, a, b) || self.high_add(dst, b, a)) {
            return true;
        }
        let (a_values, b_values) = match (self.word(a), self.word(b)) {
            (Word::Values(a), Word::Values(b)) => (a, b),
            (Word::Unknown(r), _) | (_, Word::Unknown(r)) => {
                self.set_word(dst, Word::Unknown(r));
                self.carry = Carry::Unknown;
                return true;
            }
        };
        if a_values.len() > 1 && b_values.len() > 1 && a != b {
            self.set_word(dst, Word::Unknown(Reason::Correlation));
            self.carry = Carry::Unknown;
            return true;
        }
        let carry_in = match self.carry {
            Carry::Constant(c) => u64::from(c),
            _ if op == "s.addc.u32" => {
                self.set_word(dst, Word::Unknown(Reason::Carry));
                self.carry = Carry::Unknown;
                return true;
            }
            _ => 0,
        };
        let mut values = Vec::new();
        let mut carries = Vec::new();
        for &x in a_values.iter() {
            for &y in b_values.iter() {
                if a == b && x != y {
                    continue;
                }
                let result = match op {
                    "s.add.u32" | "s.addc.u32" => {
                        let sum = u64::from(x) + u64::from(y) + carry_in;
                        (sum as u32, sum >> 32 != 0)
                    }
                    "s.add.co.i32" => {
                        let (v, c) = (x as i32).overflowing_add(y as i32);
                        (v as u32, c)
                    }
                    "s.and.b32" => (x & y, x & y != 0),
                    "s.or.b32" => (x | y, x | y != 0),
                    "s.xor.b32" => (x ^ y, x ^ y != 0),
                    _ => {
                        self.kill(i, row);
                        return true;
                    }
                };
                values.push(result.0);
                carries.push(result.1);
            }
        }
        values.sort_unstable();
        values.dedup();
        self.set_word(
            dst,
            if values.len() <= SCALARS {
                Word::Values(values.into())
            } else {
                Word::Unknown(Reason::ScalarLimit)
            },
        );
        self.carry = if carries.iter().all(|&c| c == carries[0]) {
            Carry::Constant(carries[0])
        } else {
            Carry::Unknown
        };
        true
    }
    pub(super) fn step(&mut self, program: &Program, row: usize) {
        let i = &program.snapshot.instrs()[row];
        let op = i.form().canon.as_str();
        match op {
            "s.getpc.b64" | "s.swappc.b64" => {
                if let Some(first) = i.ops().first().and_then(pair) {
                    self.set_pair(
                        first,
                        Pair::Values(
                            vec![Address {
                                offset: i64::from(program.addresses[row]) + 4,
                                role: if op == "s.swappc.b64" {
                                    AddressRole::Return(row)
                                } else {
                                    AddressRole::Code
                                },
                            }]
                            .into(),
                        ),
                    );
                } else {
                    self.kill(i, row);
                }
                return;
            }
            "s.mov.b64" => {
                if let [dst, src] = i.ops() {
                    if let Some(dst) = pair(dst) {
                        if let Some(n) = integer64(src) {
                            self.set_word(dst, Word::constant(n as u32));
                            self.set_word(dst + 1, Word::constant((n as u64 >> 32) as u32));
                        } else {
                            self.set_pair(dst, self.read_pair(src));
                        }
                        return;
                    }
                }
            }
            "s.mov.b32" => {
                if let [dst, src] = i.ops() {
                    if let Some(n) = scalar(dst) {
                        if scalar(src) != Some(n) {
                            self.set_word(n, self.word(src));
                        }
                        return;
                    }
                }
            }
            "s.add.nc.u64" => {
                if let [dst, a, b] = i.ops() {
                    if let Some(dst) = pair(dst) {
                        let (base, delta) = if let Some(n) = integer64(b) {
                            (self.read_pair(a), Some(n))
                        } else {
                            (self.read_pair(b), integer64(a))
                        };
                        if let (Pair::Values(values), Some(delta)) = (base, delta) {
                            self.set_pair(dst, Self::shifted(&values, delta));
                            return;
                        }
                    }
                }
            }
            "s.add.u32" | "s.addc.u32" | "s.add.co.i32" | "s.and.b32" | "s.or.b32"
            | "s.xor.b32" => {
                if self.binary(i, row, op) {
                    return;
                }
            }
            "s.cselect.b64" => {
                if let [dst, a, b] = i.ops() {
                    if let Some(dst) = pair(dst) {
                        let (a, b) = (self.read_pair(a), self.read_pair(b));
                        let value = match self.carry {
                            Carry::Constant(true) => a,
                            Carry::Constant(false) => b,
                            _ => a.join(&b),
                        };
                        self.set_pair(dst, value);
                        return;
                    }
                }
            }
            _ => {}
        }
        self.kill(i, row);
    }
}
