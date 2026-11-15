//! Problems: what a kernel is supposed to compute, as data.

use std::collections::{BTreeSet, HashMap};

use crate::tags::{Cols, TagError};

use super::layout::Layout;

/// `d_cols`: the B row each D cell must hold.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DCols {
    /// The same table [N] for every row.
    Row(Vec<i64>),
    /// A table [M][N].
    Grid(Vec<Vec<i64>>),
    /// A fused epilogue's tuple of such tables: the cell combines them.
    Tuple(Vec<DCols>),
}

impl DCols {
    fn one(&self, m: i64, n: i64) -> Result<i64, String> {
        let get = |v: &Vec<i64>, i: i64| {
            usize::try_from(i)
                .ok()
                .and_then(|i| v.get(i).copied())
                .ok_or_else(|| format!("d_cols indexed past its end at ({m}, {n})"))
        };
        match self {
            DCols::Row(v) => get(v, n),
            DCols::Grid(rows) => {
                let row = usize::try_from(m)
                    .ok()
                    .and_then(|i| rows.get(i))
                    .ok_or_else(|| format!("d_cols indexed past its end at ({m}, {n})"))?;
                get(row, n)
            }
            DCols::Tuple(_) => Err("a nested d_cols tuple".into()),
        }
    }

    /// The columns cell (m, n) must hold.
    pub fn at(&self, m: i64, n: i64) -> Result<Cols, String> {
        match self {
            DCols::Tuple(parts) => {
                let mut cols: Vec<i64> = parts
                    .iter()
                    .map(|p| p.one(m, n))
                    .collect::<Result<_, _>>()?;
                cols.sort_unstable();
                cols.dedup();
                Ok(Cols::Many(cols))
            }
            _ => Ok(Cols::One(self.one(m, n)?)),
        }
    }
}

/// An int64 table [M][N].
pub type Table = Vec<Vec<i64>>;

/// A `record_operand` link.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RecordOperand {
    /// 0 = A, 1 = B.
    pub side: i64,
    /// The source problem's pid.
    pub source: i64,
    /// (row[M][N], k[M][N]) giving (row, k) for (m, n); none is the identity.
    pub map: Option<(Table, Table)>,
}

/// A store-log cell: (m, n, A row or none).
pub type Cell = (i64, i64, Option<i64>);

/// One GEMM-like computation.
#[derive(Clone, Debug)]
pub struct Problem {
    /// Buffer name → layout (or none), in order.
    pub layouts: Vec<(String, Option<Layout>)>,
    /// The reduction length.
    pub k: i64,
    /// The A row each D row must hold.
    pub d_rows: Option<Vec<i64>>,
    /// The B row(s) each D cell must hold.
    pub d_cols: Option<DCols>,
    /// D row → its k-blocks.
    pub d_kblocks: Option<HashMap<i64, BTreeSet<i64>>>,
    /// The number of D columns.
    pub n: Option<i64>,
    /// Its operand that is another problem's D records.
    pub record_operand: Option<RecordOperand>,
    /// D row → the A rows added into it by atomics.
    pub scatter: Option<HashMap<i64, BTreeSet<i64>>>,
    /// Raise at the first error, else record it.
    pub fail_fast: bool,
    /// The registry's index, once registered (−1 before).
    pub pid: i64,
    /// The errors recorded without fail_fast.
    pub errors: Vec<String>,
    /// The store log, in order.
    pub store_log: Vec<Cell>,
    stores: Vec<(Cell, u64)>,
    store_index: HashMap<Cell, usize>,
}

impl Problem {
    /// A problem of these layouts and K, every other field at its default.
    pub fn new(layouts: Vec<(String, Option<Layout>)>, k: i64) -> Problem {
        Problem {
            layouts,
            k,
            d_rows: None,
            d_cols: None,
            d_kblocks: None,
            n: None,
            record_operand: None,
            scatter: None,
            fail_fast: true,
            pid: -1,
            errors: Vec::new(),
            store_log: Vec::new(),
            stores: Vec::new(),
            store_index: HashMap::new(),
        }
    }

    /// `kblocks()`: {0, …, ⌈K/32⌉ − 1}, or {0} below 32.
    pub fn kblocks(&self) -> BTreeSet<i64> {
        if self.k >= 32 {
            (0..(self.k + 31) / 32).collect()
        } else {
            [0].into()
        }
    }

    /// The layout of a buffer name.
    pub fn layout(&self, name: &str) -> Option<&Layout> {
        self.layouts
            .iter()
            .find(|(n, _)| n == name)
            .and_then(|(_, l)| l.as_ref())
    }

    /// Reports an error: raised with fail_fast, else recorded.
    pub fn err(&mut self, msg: String) -> Result<(), TagError> {
        if self.fail_fast {
            return Err(TagError::fault(msg));
        }
        self.errors.push(msg);
        Ok(())
    }

    /// Appends cells to the store log and its counts.
    pub fn log_stores(&mut self, cells: &[Cell]) {
        for c in cells {
            self.store_log.push(*c);
            match self.store_index.get(c) {
                Some(&i) => self.stores[i].1 += 1,
                None => {
                    self.store_index.insert(*c, self.stores.len());
                    self.stores.push((*c, 1));
                }
            }
        }
    }

    /// `stores`: cell → count, in order of first store.
    pub fn stores(&self) -> &[(Cell, u64)] {
        &self.stores
    }

    /// The A row D row m must hold.
    pub fn want_m(&self, m: i64) -> Result<i64, String> {
        match &self.d_rows {
            None => Ok(m),
            Some(t) => usize::try_from(m)
                .ok()
                .and_then(|i| t.get(i).copied())
                .ok_or_else(|| format!("d_rows indexed past its end at {m}")),
        }
    }

    /// The column(s) cell (m, n) must hold.
    pub fn want_n(&self, m: i64, n: i64) -> Result<Cols, String> {
        match &self.d_cols {
            None => Ok(Cols::One(n)),
            Some(d) => d.at(m, n),
        }
    }
}
