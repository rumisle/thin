//! The accumulator state and the matrix-form record the machine hands a
//! checker.

use std::collections::BTreeSet;

use super::checker::TagError;
use super::encoding::{is_record, py_list, src, Cols, KSet, Record, Tables, Tag};
use super::rules::Lane;

/// One operand format of a matrix form.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OperandFormat {
    /// The format's name, for messages.
    pub name: String,
    /// Bytes per lane.
    pub bytes: usize,
    /// Bytes per element (2 for a 16-bit form, else 1).
    pub elem_bytes: usize,
    /// `hwk[q][p]`: the hardware k of position p of lane group q.
    pub hwk: Vec<Vec<i64>>,
}

/// A target's matrix form.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MatrixForm {
    /// The wave width.
    pub w: usize,
    /// Rows per operand.
    pub r: usize,
    /// The hardware K.
    pub k_hw: i64,
    /// `d_map[e][l]` = (src0 row slot, src1 row slot) of D dword e, lane l.
    pub d_map: Vec<Vec<(usize, usize)>>,
}

impl MatrixForm {
    /// The lane groups G = W / R.
    pub fn groups(&self) -> usize {
        self.w / self.r
    }

    /// The D dwords per lane.
    pub fn d_dwords(&self) -> usize {
        self.d_map.len()
    }
}

/// One operand as the check sees it: its raw register tags `[W][bytes]` and
/// its format.
pub struct MatrixOperand<'a> {
    /// The raw tags, lane-major.
    pub tags: &'a [Vec<Tag>],
    /// The operand's format.
    pub format: &'a OperandFormat,
}

/// A matrix op handed to a checker.
pub struct MatrixOp<'a> {
    /// The form.
    pub form: &'a MatrixForm,
    /// The instruction's text.
    pub text: &'a str,
    /// The destination's name for messages (`a[0:3]`).
    pub dst: &'a str,
    /// The first accumulator entry of D.
    pub entry: usize,
    /// Per D dword, whether its register is marked.
    pub marked: Vec<bool>,
    /// Whether C is the literal 0.
    pub c_zero: bool,
    /// The two operands.
    pub src: [MatrixOperand<'a>; 2],
}

/// One accumulator entry.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct AccEntry {
    /// Per lane, the D row (−1 unset).
    pub m: Vec<i64>,
    /// Per lane, the D column (−1 unset).
    pub n: Vec<i64>,
    /// The k-block set.
    pub k: BTreeSet<i64>,
    /// The problem.
    pub pid: i64,
}

impl AccEntry {
    /// An unset entry of W lanes.
    pub fn new(w: usize) -> AccEntry {
        AccEntry {
            m: vec![-1; w],
            n: vec![-1; w],
            ..AccEntry::default()
        }
    }

    /// Resets every lane.
    pub fn reset(&mut self) {
        let w = self.m.len();
        *self = AccEntry::new(w);
    }
}

/// Every register's accumulator entry: accumulator-file register r at r, vector
/// register r at (accumulator registers) + r.
#[derive(Clone, Debug, Default)]
pub struct AccState {
    /// The entries.
    pub entries: Vec<AccEntry>,
    /// The wave width.
    pub w: usize,
}

impl AccState {
    /// The state of `count` entries of W lanes.
    pub fn new(count: usize, w: usize) -> AccState {
        AccState {
            entries: (0..count).map(|_| AccEntry::new(w)).collect(),
            w,
        }
    }

    /// The entry at index i, grown on demand.
    pub fn entry(&mut self, i: usize) -> &mut AccEntry {
        while self.entries.len() <= i {
            self.entries.push(AccEntry::new(self.w));
        }
        &mut self.entries[i]
    }

    /// Materializes a marked register's records.
    pub fn materialize(&mut self, tables: &mut Tables, i: usize) -> Vec<Lane> {
        let e = self.entry(i).clone();
        (0..e.m.len())
            .map(|l| {
                if e.m[l] >= 0 && e.n[l] >= 0 {
                    let r = Record {
                        m: e.m[l],
                        n: Cols::One(e.n[l]),
                        k: KSet::Set(e.k.clone()),
                        stat: false,
                        pid: e.pid,
                    };
                    [tables.record(&r); 4]
                } else {
                    [0; 4]
                }
            })
            .collect()
    }

    /// The accumulator-file write's reset and resume; returns whether
    /// the register is to be marked.
    pub fn resume(
        &mut self,
        tables: &mut Tables,
        i: usize,
        tags: &[Lane],
        text: &str,
    ) -> Result<bool, TagError> {
        let entry = self.entry(i);
        entry.reset();
        if !tags.iter().any(|l| l.iter().any(|t| is_record(*t))) {
            return Ok(false);
        }
        let mut first: Option<(BTreeSet<i64>, i64)> = None;
        let mut any = false;
        for (l, lane) in tags.iter().enumerate() {
            let t = src(lane);
            if !is_record(t) {
                continue;
            }
            let r = tables.record_of(t);
            let Cols::One(n) = r.n else { continue };
            if r.m < 0 || r.stat {
                continue;
            }
            let here = (r.blocks(), r.pid);
            match &first {
                None => first = Some(here),
                Some(f) if *f != here => {
                    return Err(TagError::fault(format!(
                        "{text}: lane {l} resumes k-blocks {} where other lanes hold {}",
                        py_list(here.0.iter().copied()),
                        py_list(f.0.iter().copied())
                    )));
                }
                Some(_) => {}
            }
            let entry = self.entry(i);
            entry.m[l] = r.m;
            entry.n[l] = n;
            any = true;
        }
        if let (true, Some((k, pid))) = (any, first) {
            let entry = self.entry(i);
            entry.k = k;
            entry.pid = pid;
        }
        Ok(any)
    }
}
