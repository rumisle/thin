//! The tag format and its tables.

use std::collections::{BTreeSet, HashMap};

/// A byte's provenance: a signed 64-bit word.
pub type Tag = i64;

/// The accumulator mark: a register whose provenance is in the accumulator state.
pub const ACC_MARK: Tag = -1;
/// The base of the encoded records.
pub const ACC_BASE: Tag = -(1 << 40);
/// The span of the encoded records below [`ACC_BASE`].
pub const ENC_SPAN: i64 = 1 << 60;
/// The base of the sum records.
pub const SUM_BASE: Tag = -(1 << 61);
/// The shift of a memory tag's buffer.
pub const TAG_SHIFT: u32 = 40;
/// A poison's m.
pub const POISON_M: i64 = -1;

const OFFSET_MASK: Tag = (1 << TAG_SHIFT) - 1;

/// The memory tag of byte `offset` of buffer `buffer`.
pub fn mem_tag(buffer: usize, offset: u64) -> Tag {
    ((buffer as i64 + 1) << TAG_SHIFT) | (offset as i64 & OFFSET_MASK)
}

/// A memory tag's buffer index.
pub fn tag_buf(tag: Tag) -> i64 {
    (tag >> TAG_SHIFT) - 1
}

/// A memory tag's byte offset.
pub fn tag_off(tag: Tag) -> u64 {
    (tag & OFFSET_MASK) as u64
}

/// Whether a tag is a record (encoded or from the table; poison included).
pub fn is_record(t: Tag) -> bool {
    t > SUM_BASE && t <= -2
}

/// Whether a tag is a sum record.
pub fn is_sum(t: Tag) -> bool {
    t <= SUM_BASE
}

/// Whether a tag is a record or a sum record (below the accumulator mark).
pub fn is_provenance_record(t: Tag) -> bool {
    t < -1
}

/// The column part of a record: one column, or an ordered tuple.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Cols {
    /// One D column.
    One(i64),
    /// A combination of columns, sorted.
    Many(Vec<i64>),
}

impl Cols {
    /// The columns as a sorted list.
    pub fn list(&self) -> Vec<i64> {
        match self {
            Cols::One(n) => vec![*n],
            Cols::Many(v) => v.clone(),
        }
    }

    /// The first column.
    pub fn first(&self) -> i64 {
        match self {
            Cols::One(n) => *n,
            Cols::Many(v) => v.first().copied().unwrap_or(-1),
        }
    }

    /// Python's repr of the column part: `3` or `(1, 2)`.
    pub fn text(&self) -> String {
        match self {
            Cols::One(n) => n.to_string(),
            Cols::Many(v) => py_tuple(v),
        }
    }
}

/// The K slot of a record: a set of k-blocks, or a poison's reason.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum KSet {
    /// k-blocks.
    Set(BTreeSet<i64>),
    /// A poison's reason.
    Reason(String),
}

impl KSet {
    /// The integer set, empty for a reason.
    pub fn set(&self) -> BTreeSet<i64> {
        match self {
            KSet::Set(s) => s.clone(),
            KSet::Reason(_) => BTreeSet::new(),
        }
    }
}

/// An accumulator record (m, n, K, stat, pid).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Record {
    /// The D row; −1 poison.
    pub m: i64,
    /// The D column(s).
    pub n: Cols,
    /// The k-blocks summed, or the reason.
    pub k: KSet,
    /// A statistic over the columns.
    pub stat: bool,
    /// The problem.
    pub pid: i64,
}

impl Record {
    /// A plain record of (m, n) over these k-blocks.
    pub fn new(m: i64, n: i64, k: impl IntoIterator<Item = i64>, pid: i64) -> Record {
        Record {
            m,
            n: Cols::One(n),
            k: KSet::Set(k.into_iter().collect()),
            stat: false,
            pid,
        }
    }

    /// Whether it is poison.
    pub fn is_poison(&self) -> bool {
        self.m == POISON_M
    }

    /// A poison's reason.
    pub fn reason(&self) -> &str {
        match &self.k {
            KSet::Reason(r) => r,
            KSet::Set(_) => "",
        }
    }

    /// The k-blocks as a set.
    pub fn blocks(&self) -> BTreeSet<i64> {
        self.k.set()
    }

    /// `(m, n)` as messages write it.
    pub fn cell(&self) -> String {
        format!("({}, {})", self.m, self.n.text())
    }
}

/// What the tables know of a problem: its K, N and the source of
/// its record operand. The GEMM checker registers the rest.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ProblemFacts {
    /// The reduction length.
    pub k: i64,
    /// The number of D columns, if set.
    pub n: Option<i64>,
    /// The pid of the problem whose D records this one takes as an operand.
    pub source: Option<i64>,
}

/// The append-only tables behind the record and sum tags.
#[derive(Clone, Debug, Default)]
pub struct Tables {
    ksets: Vec<BTreeSet<i64>>,
    kset_ids: HashMap<BTreeSet<i64>, usize>,
    records: Vec<Record>,
    record_ids: HashMap<Record, usize>,
    sums: Vec<(Tag, Tag)>,
    sum_ids: HashMap<(Tag, Tag), usize>,
    /// The registered problems' facts, by pid.
    pub problems: Vec<ProblemFacts>,
}

/// How a record tag was made: encoded with its k-set id, or the table's rid.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Form {
    /// The encoded form and its kid.
    Encoded(usize),
    /// The record table's index.
    Table(usize),
}

impl Tables {
    /// The k-set id of a set, assigned at its first use.
    pub fn kid(&mut self, k: &BTreeSet<i64>) -> usize {
        if let Some(&id) = self.kset_ids.get(k) {
            return id;
        }
        self.ksets.push(k.clone());
        self.kset_ids.insert(k.clone(), self.ksets.len() - 1);
        self.ksets.len() - 1
    }

    /// The canonical tag of a record and how it was made.
    pub fn record_form(&mut self, r: &Record) -> (Tag, Form) {
        let plain = !r.stat;
        if let (true, Cols::One(n), KSet::Set(k)) = (plain, &r.n, &r.k) {
            if (0..1 << 22).contains(&r.m) && (0..1 << 20).contains(n) && (0..256).contains(&r.pid)
            {
                let kid = self.kid(k);
                if kid < 1024 {
                    let c = ((((kid as i64) << 8 | r.pid) << 22 | r.m) << 20) | n;
                    return (ACC_BASE - c, Form::Encoded(kid));
                }
            }
        }
        let rid = match self.record_ids.get(r) {
            Some(&rid) => rid,
            None => {
                self.records.push(r.clone());
                self.record_ids.insert(r.clone(), self.records.len() - 1);
                self.records.len() - 1
            }
        };
        (-(rid as i64 + 2), Form::Table(rid))
    }

    /// The canonical tag of a record.
    pub fn record(&mut self, r: &Record) -> Tag {
        self.record_form(r).0
    }

    /// The record a record tag names.
    pub fn record_of(&self, tag: Tag) -> Record {
        if tag <= ACC_BASE && tag > ACC_BASE - ENC_SPAN {
            let c = ACC_BASE - tag;
            let kid = (c >> 50) as usize;
            return Record {
                m: (c >> 20) & ((1 << 22) - 1),
                n: Cols::One(c & ((1 << 20) - 1)),
                k: KSet::Set(self.ksets.get(kid).cloned().unwrap_or_default()),
                stat: false,
                pid: (c >> 42) & 0xFF,
            };
        }
        let rid = (-tag - 2) as usize;
        self.records.get(rid).cloned().unwrap_or_else(|| Record {
            m: POISON_M,
            n: Cols::One(-1),
            k: KSet::Reason(format!("an unknown record tag {tag}")),
            stat: false,
            pid: 0,
        })
    }

    /// The poison of a reason.
    pub fn poison(&mut self, reason: impl Into<String>) -> Tag {
        let r = Record {
            m: POISON_M,
            n: Cols::One(-1),
            k: KSet::Reason(reason.into()),
            stat: false,
            pid: 0,
        };
        self.record(&r)
    }

    /// The sum record of an add's two sources, in operand order.
    pub fn sum(&mut self, x: Tag, y: Tag) -> Tag {
        let sid = match self.sum_ids.get(&(x, y)) {
            Some(&sid) => sid,
            None => {
                self.sums.push((x, y));
                self.sum_ids.insert((x, y), self.sums.len() - 1);
                self.sums.len() - 1
            }
        };
        SUM_BASE - sid as i64
    }

    /// A sum record's two operands.
    pub fn sum_of(&self, tag: Tag) -> (Tag, Tag) {
        self.sums
            .get((SUM_BASE - tag) as usize)
            .copied()
            .unwrap_or((0, 0))
    }

    /// The leaves of a tag, left to right.
    pub fn sources(&self, tag: Tag) -> Vec<Tag> {
        let mut out = Vec::new();
        let mut stack = vec![tag];
        while let Some(t) = stack.pop() {
            if is_sum(t) {
                let (x, y) = self.sum_of(t);
                stack.push(y);
                stack.push(x);
            } else if t > 0 {
                out.push(t);
            }
        }
        out
    }

    /// Registers a problem's facts; returns its pid.
    pub fn register(&mut self, facts: ProblemFacts) -> i64 {
        self.problems.push(facts);
        self.problems.len() as i64 - 1
    }

    /// The facts of a pid.
    pub fn problem(&self, pid: i64) -> ProblemFacts {
        usize::try_from(pid)
            .ok()
            .and_then(|i| self.problems.get(i))
            .cloned()
            .unwrap_or_default()
    }
}

/// The dword provenance of 4 byte tags.
pub fn src(t: &[Tag]) -> Tag {
    let low = t.iter().copied().min().unwrap_or(0);
    if low < 0 {
        return low;
    }
    t.iter().copied().find(|&b| b > 0).unwrap_or(0)
}

/// Python's list repr of integers: `[1, 2]`.
pub fn py_list(v: impl IntoIterator<Item = i64>) -> String {
    let items: Vec<String> = v.into_iter().map(|x| x.to_string()).collect();
    format!("[{}]", items.join(", "))
}

/// Python's tuple repr of integers: `(1, 2)`, `(1,)`.
pub fn py_tuple(v: &[i64]) -> String {
    let items: Vec<String> = v.iter().map(|x| x.to_string()).collect();
    if items.len() == 1 {
        format!("({},)", items[0])
    } else {
        format!("({})", items.join(", "))
    }
}

/// A block list of at most 6, then `...`.
pub fn blocks_text(v: impl IntoIterator<Item = i64>) -> String {
    let all: Vec<i64> = v.into_iter().collect();
    let shown = py_list(all.iter().copied().take(6));
    if all.len() > 6 {
        format!("{shown}...")
    } else {
        shown
    }
}
