//! The checker's layouts: data, never a callback.

/// What a layout's bytes are.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Kind {
    /// An A operand.
    A,
    /// A B operand.
    B,
    /// A result.
    D,
    /// Anything else.
    Other,
}

impl Kind {
    /// The kind of a name (`A`, `B`, `D`, `other`).
    pub fn named(s: &str) -> Kind {
        match s {
            "A" => Kind::A,
            "B" => Kind::B,
            "D" => Kind::D,
            _ => Kind::Other,
        }
    }

    /// The name.
    pub fn name(self) -> &'static str {
        match self {
            Kind::A => "A",
            Kind::B => "B",
            Kind::D => "D",
            Kind::Other => "other",
        }
    }
}

/// How a layout decodes an offset.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Shape {
    /// `rowmajor` (the operands).
    RowMajor {
        /// Bytes per row.
        ld: i64,
        /// Bytes per element.
        elem: i64,
        /// Rows per batch.
        rows_per_batch: i64,
        /// Bytes per batch (0: no batches).
        batch_bytes: i64,
    },
    /// `rowmajor_d`.
    RowMajorD {
        /// Bytes per row.
        ld: i64,
        /// Bytes per element.
        elem: i64,
        /// Rows per batch.
        rows_per_batch: i64,
        /// Bytes per batch.
        batch_bytes: i64,
    },
}

/// A layout: kind and shape.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Layout {
    /// What its bytes are.
    pub kind: Kind,
    /// How offsets decode.
    pub shape: Shape,
    /// Informational text.
    pub note: String,
}

/// One offset's decoded fields: operands give row, k, sub; D gives m (row),
/// n (k), sub.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Decoded {
    /// The row (m for D).
    pub row: i64,
    /// k (n for D).
    pub k: i64,
    /// The byte within the element.
    pub sub: i64,
}

fn batch(off: i64, batch_bytes: i64) -> (i64, i64) {
    if batch_bytes > 0 {
        (off / batch_bytes, off - (off / batch_bytes) * batch_bytes)
    } else {
        (0, off)
    }
}

impl Layout {
    /// A layout of this kind and shape.
    pub fn new(kind: Kind, shape: Shape) -> Layout {
        Layout {
            kind,
            shape,
            note: String::new(),
        }
    }

    /// `elem_bytes`.
    pub fn elem_bytes(&self) -> i64 {
        match self.shape {
            Shape::RowMajor { elem, .. } | Shape::RowMajorD { elem, .. } => elem,
        }
    }

    /// Decodes one offset.
    pub fn decode(&self, off: u64) -> Decoded {
        let off = off as i64;
        let (ld, elem, rows_per_batch, batch_bytes) = match self.shape {
            Shape::RowMajor {
                ld,
                elem,
                rows_per_batch,
                batch_bytes,
            }
            | Shape::RowMajorD {
                ld,
                elem,
                rows_per_batch,
                batch_bytes,
            } => (ld, elem, rows_per_batch, batch_bytes),
        };
        let (b, o) = batch(off, batch_bytes);
        let col = o % ld;
        Decoded {
            row: o / ld + b * rows_per_batch,
            k: col / elem,
            sub: col % elem,
        }
    }
}
