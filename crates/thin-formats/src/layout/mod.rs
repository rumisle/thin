//! Layouts: how a stack of matrices `[stack][rows][cols]` is arranged in
//! bytes.
//!
//! A layout is a registered [`Definition`] and its arguments. Applied to a
//! dtype and a shape, a definition gives a [`Recipe`] (reshape, permute,
//! pad and repeat steps) or a [`Map`] (an offset decoder), and the
//! operations walk it over the buffer's logical bytes.

mod generic;
mod logical;
mod walk;

use std::borrow::Cow;
use std::fmt;

pub use logical::{pack, unpack};
pub use walk::CHUNK;

use crate::dtype::Dtype;
use crate::{Error, Result, Shape};
use walk::{MapWalk, RecipeWalk};

/// A layout made of reshape and permute steps.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Recipe {
    /// Rows per matrix, padded (at least the buffer's rows).
    pub rows: usize,
    /// Bytes per row, padded (at least the logical row's bytes).
    pub row_bytes: usize,
    /// The shape the padded logical bytes are reshaped to.
    pub dims: Vec<usize>,
    /// The transpose of `dims`.
    pub perm: Vec<usize>,
    /// The shape the transposed bytes are reshaped to; `None` keeps the
    /// permuted dims.
    pub out: Option<Vec<usize>>,
    /// (before, after) padding of each out dimension; empty for none.
    pub pad: Vec<(usize, usize)>,
    /// (out dimension of size 1, copies) of each repeated dimension.
    pub repeat: Vec<(usize, usize)>,
    /// The byte padding holds.
    pub fill: u8,
}

impl Recipe {
    /// A recipe of one reshape and transpose, with no padding beyond
    /// `rows` and `row_bytes`, no repeats and fill 0.
    pub fn new(rows: usize, row_bytes: usize, dims: Vec<usize>, perm: Vec<usize>) -> Recipe {
        Recipe {
            rows,
            row_bytes,
            dims,
            perm,
            out: None,
            pad: Vec::new(),
            repeat: Vec::new(),
            fill: 0,
        }
    }
}

/// The decoder of a [`Map`]: the (matrix, row, byte) of each offset.
pub type Decoder = Box<dyn Fn(&[u64], &mut [[usize; 3]]) + Send + Sync>;

/// A layout given by an offset decoder, for orders that are not one
/// permutation.
pub struct Map {
    /// Rows per matrix.
    pub rows: usize,
    /// Bytes per row.
    pub row_bytes: usize,
    /// The layout's bytes: stack · rows · row_bytes.
    pub size: usize,
    /// A bijection from offsets onto (matrix, row, byte).
    pub decode: Decoder,
    /// (P, shift): offset o + P holds the flat logical index of o plus
    /// shift, so only P offsets are decoded.
    pub period: Option<(usize, i64)>,
}

impl fmt::Debug for Map {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Map")
            .field("rows", &self.rows)
            .field("row_bytes", &self.row_bytes)
            .field("size", &self.size)
            .field("period", &self.period)
            .finish_non_exhaustive()
    }
}

/// What a definition gives for a dtype and shape.
#[derive(Debug)]
pub enum Arrangement {
    /// Reshape and permute steps.
    Recipe(Recipe),
    /// An offset decoder.
    Map(Map),
}

/// The builder of a definition: (dtype, shape, the parameters' values in
/// order) to an arrangement. With no dtype and a zero shape it runs only
/// the checks that need neither.
pub type Build = fn(Option<&Dtype>, Shape, &[i64]) -> Result<Arrangement>;

/// A registered layout definition.
pub struct Definition {
    /// The registered name (`<target>.<layout>` for a target's).
    pub name: &'static str,
    /// The parameters and their defaults, in order.
    pub params: &'static [(&'static str, i64)],
    /// The builder.
    pub build: Build,
}

impl fmt::Debug for Definition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Definition")
            .field("name", &self.name)
            .field("params", &self.params)
            .finish_non_exhaustive()
    }
}

/// The registry: the generic definitions. It is fixed at build
/// time; no target built in registers a layout yet.
static DEFINITIONS: [&Definition; 2] = [&generic::ROW_MAJOR, &generic::COL_MAJOR];

/// Every registered definition.
pub fn definitions() -> &'static [&'static Definition] {
    &DEFINITIONS
}

/// The definition of that name among some.
fn find(definitions: &[&'static Definition], name: &str) -> Result<&'static Definition> {
    definitions
        .iter()
        .copied()
        .find(|d| d.name == name)
        .ok_or_else(|| {
            let known: Vec<_> = definitions.iter().map(|d| d.name).collect();
            Error::Layout(format!(
                "{name:?} is not a layout (one of {})",
                known.join(", ")
            ))
        })
}

/// A layout: a definition and its arguments that differ from their
/// defaults, in parameter order.
#[derive(Clone)]
pub struct Layout {
    definition: &'static Definition,
    args: Vec<(&'static str, i64)>,
}

impl PartialEq for Layout {
    fn eq(&self, other: &Layout) -> bool {
        self.definition.name == other.definition.name && self.args == other.args
    }
}

impl Eq for Layout {}

impl fmt::Debug for Layout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.key())
    }
}

impl Layout {
    /// The layout of a registered definition with arguments given by
    /// name; runs the checks that need no dtype or shape.
    pub fn new(name: &str, args: &[(&str, i64)]) -> Result<Layout> {
        Layout::of(find(&DEFINITIONS, name)?, args)
    }

    /// The layout of a definition with arguments given by name.
    pub(crate) fn of(definition: &'static Definition, args: &[(&str, i64)]) -> Result<Layout> {
        let name = definition.name;
        let mut values: Vec<Option<i64>> = vec![None; definition.params.len()];
        for &(param, value) in args {
            let index = definition
                .params
                .iter()
                .position(|&(p, _)| p == param)
                .ok_or_else(|| Error::Layout(format!("{name} has no parameter {param:?}")))?;
            if values[index].replace(value).is_some() {
                return Err(Error::Layout(format!("{name}: {param} given twice")));
            }
        }
        let values: Vec<i64> = (values.iter().zip(definition.params))
            .map(|(v, &(_, default))| v.unwrap_or(default))
            .collect();
        (definition.build)(None, (0, 0, 0), &values)?;
        let args = (definition.params.iter().zip(&values))
            .filter(|(&(_, default), &value)| value != default)
            .map(|(&(param, _), &value)| (param, value))
            .collect();
        Ok(Layout { definition, args })
    }

    /// Rebuilds a layout from its key, refusing any key that is not
    /// exactly the key it rebuilds.
    pub fn parse(key: &str) -> Result<Layout> {
        Layout::parse_in(&DEFINITIONS, key)
    }

    /// [`Layout::parse`] among some definitions.
    pub(crate) fn parse_in(definitions: &[&'static Definition], key: &str) -> Result<Layout> {
        let refuse = |why: &str| Error::Layout(format!("layout key {key:?}: {why}"));
        let (name, inner) = match key.split_once('(') {
            None => (key, None),
            Some((name, rest)) => (
                name,
                Some(
                    rest.strip_suffix(')')
                        .ok_or_else(|| refuse("no closing ')'"))?,
                ),
            ),
        };
        let mut args = Vec::new();
        for part in inner.map(|inner| inner.split(',')).into_iter().flatten() {
            let (param, value) = part
                .split_once('=')
                .ok_or_else(|| refuse("an argument is not k=v"))?;
            let value = value
                .parse()
                .map_err(|_| refuse(&format!("{param}={value} is not an int")))?;
            args.push((param, value));
        }
        let layout = Layout::of(find(definitions, name)?, &args)?;
        if layout.key() != key {
            return Err(refuse(&format!("not canonical; it is {:?}", layout.key())));
        }
        Ok(layout)
    }

    /// The registered name.
    pub fn name(&self) -> &'static str {
        self.definition.name
    }

    /// The arguments that differ from their defaults, in parameter order.
    pub fn args(&self) -> &[(&'static str, i64)] {
        &self.args
    }

    /// `name`, or `name(k=v,k=v)`.
    pub fn key(&self) -> String {
        if self.args.is_empty() {
            return self.name().to_owned();
        }
        let args: Vec<String> = self.args.iter().map(|(k, v)| format!("{k}={v}")).collect();
        format!("{}({})", self.name(), args.join(","))
    }

    /// Every parameter's value, in order.
    fn values(&self) -> Vec<i64> {
        (self.definition.params.iter())
            .map(|&(param, default)| {
                self.args
                    .iter()
                    .find(|&&(p, _)| p == param)
                    .map_or(default, |&(_, v)| v)
            })
            .collect()
    }
}

/// How an offset's source is found.
enum Order {
    Recipe(RecipeWalk),
    Map(Map),
}

/// An arrangement applied to a dtype and shape: the geometry of its
/// padded logical buffer `[stack][rows][row_bytes]`, its walk, and the
/// operations of on it.
pub struct Arranged {
    name: String,
    dtype: &'static Dtype,
    shape: Shape,
    /// The buffer's rows and logical row bytes.
    real: (usize, usize),
    /// The padded rows and row bytes.
    rows: usize,
    row_bytes: usize,
    fill: u8,
    nbytes: usize,
    order: Order,
}

impl Arranged {
    /// A layout at a dtype and shape.
    pub fn of(layout: &Layout, dtype: &'static Dtype, shape: Shape) -> Result<Arranged> {
        logical::elements(shape)?;
        let arrangement = (layout.definition.build)(Some(dtype), shape, &layout.values())?;
        Arranged::new(arrangement, dtype, shape, &layout.key())
    }

    /// A definition's arrangement at a dtype and shape, checked as
    /// checks it; `name` names it in refusals.
    pub fn new(
        arrangement: Arrangement,
        dtype: &'static Dtype,
        shape: Shape,
        name: &str,
    ) -> Result<Arranged> {
        let elements = logical::elements(shape)?;
        let real_row_bytes = logical::row_bytes(dtype, shape.2)?;
        (elements / shape.2)
            .checked_mul(real_row_bytes)
            .ok_or_else(|| Error::Layout(format!("{name}: {shape:?} overflows")))?;
        let (rows, row_bytes) = match &arrangement {
            Arrangement::Recipe(r) => (r.rows, r.row_bytes),
            Arrangement::Map(m) => (m.rows, m.row_bytes),
        };
        let fits = match &arrangement {
            Arrangement::Recipe(_) => rows >= shape.1 && row_bytes >= real_row_bytes,
            Arrangement::Map(_) => (rows, row_bytes) == (shape.1, real_row_bytes),
        };
        if !fits {
            return Err(Error::Layout(format!(
                "{name}: rows {rows} of {row_bytes} bytes for {} rows of {real_row_bytes} \
                 (a Map has no padding, a Recipe no fewer)",
                shape.1
            )));
        }
        let padded = shape
            .0
            .checked_mul(rows)
            .and_then(|n| n.checked_mul(row_bytes))
            .ok_or_else(|| Error::Layout(format!("{name}: {shape:?} padded overflows")))?;
        let (order, nbytes, fill) = match arrangement {
            Arrangement::Recipe(recipe) => {
                let (walk, nbytes) = RecipeWalk::new(&recipe, padded, name)?;
                (Order::Recipe(walk), nbytes, recipe.fill)
            }
            Arrangement::Map(map) => {
                MapWalk::new(&map, padded, name)?;
                let size = map.size;
                (Order::Map(map), size, 0)
            }
        };
        Ok(Arranged {
            name: name.to_owned(),
            dtype,
            shape,
            real: (shape.1, real_row_bytes),
            rows,
            row_bytes,
            fill,
            nbytes,
            order,
        })
    }

    /// The bytes the arrangement occupies.
    pub fn nbytes(&self) -> usize {
        self.nbytes
    }

    /// Storage dimensions of a Recipe after padding/repetition; none for a Map.
    pub fn storage_shape(&self) -> Option<&[usize]> {
        match &self.order {
            Order::Recipe(walk) => Some(walk.storage_shape()),
            Order::Map(_) => None,
        }
    }

    /// Calls `visit(offset, source, copy)` for every offset.
    fn visit(&self, mut visit: impl FnMut(usize, Option<usize>, bool)) -> Result<()> {
        match &self.order {
            Order::Recipe(walk) => {
                walk.visit(visit);
                Ok(())
            }
            Order::Map(map) => MapWalk::new(map, self.nbytes, &self.name)?
                .visit(|offset, source| visit(offset, Some(source), false)),
        }
    }

    /// The (matrix, row, logical byte) of a padded-logical index, `None`
    /// for row or byte padding.
    fn place(&self, source: usize) -> Option<[usize; 3]> {
        let plane = self.rows * self.row_bytes;
        let (m, r, b) = (
            source / plane,
            source % plane / self.row_bytes,
            source % self.row_bytes,
        );
        (r < self.real.0 && b < self.real.1).then_some([m, r, b])
    }

    /// The padded logical buffer of logical bytes, borrowed when nothing pads.
    fn padded<'a>(&self, logical: &'a [u8]) -> Cow<'a, [u8]> {
        if (self.rows, self.row_bytes) == self.real {
            return Cow::Borrowed(logical);
        }
        let mut padded = vec![self.fill; self.shape.0 * self.rows * self.row_bytes];
        let real_rows = logical.chunks_exact(self.real.1);
        let padded_rows = padded
            .chunks_exact_mut(self.row_bytes * self.rows)
            .flat_map(|m| m.chunks_exact_mut(self.row_bytes).take(self.real.0));
        for (src, dst) in real_rows.zip(padded_rows) {
            dst[..src.len()].copy_from_slice(src);
        }
        Cow::Owned(padded)
    }

    /// The logical bytes of a padded logical buffer: rows and row bytes
    /// past the buffer's are dropped.
    fn unpadded(&self, padded: &[u8]) -> Vec<u8> {
        let mut logical = Vec::with_capacity(self.shape.0 * self.real.0 * self.real.1);
        for matrix in padded.chunks_exact(self.rows * self.row_bytes) {
            for row in matrix.chunks_exact(self.row_bytes).take(self.real.0) {
                logical.extend_from_slice(&row[..self.real.1]);
            }
        }
        logical
    }

    /// Lays out storage elements `[stack][rows][cols]` into `out` of
    /// exactly [`nbytes`](Self::nbytes) bytes.
    pub fn to_bytes(&self, elements: &[u8], out: &mut [u8]) -> Result<()> {
        let (dtype, shape) = (self.dtype, self.shape);
        if out.len() != self.nbytes {
            return Err(Error::Value(format!(
                "to_bytes: {} output bytes for the {} of {}",
                out.len(),
                self.nbytes,
                self.name
            )));
        }
        let logical = if dtype.width >= 8 {
            Cow::Borrowed(elements)
        } else {
            let mut packed = vec![0; shape.0 * shape.1 * self.real.1];
            pack(elements, dtype, shape, &mut packed)?;
            Cow::Owned(packed)
        };
        if logical.len() != shape.0 * self.real.0 * self.real.1 {
            return Err(Error::Value(format!(
                "to_bytes: {} bytes do not hold {shape:?} {} elements",
                elements.len(),
                dtype.name
            )));
        }
        let padded = self.padded(&logical);
        let fill = self.fill;
        self.visit(|offset, source, _| out[offset] = source.map_or(fill, |s| padded[s]))
    }

    /// The storage elements back from the arrangement's bytes: padding is
    /// ignored, and every copy of a repeated dimension must equal the first.
    pub fn from_bytes(&self, raw: &[u8], out: &mut [u8]) -> Result<()> {
        if raw.len() != self.nbytes {
            return Err(Error::Layout(format!(
                "from_bytes: {} bytes, but {} takes {} for {:?} {}",
                raw.len(),
                self.name,
                self.nbytes,
                self.shape,
                self.dtype.name
            )));
        }
        if let Order::Recipe(walk) = &self.order {
            if let Some(offset) = walk.first_differing_copy(raw) {
                return Err(Error::Layout(format!(
                    "from_bytes: {}: offset {offset} differs from the first copy",
                    self.name
                )));
            }
        }
        let mut padded = vec![self.fill; self.shape.0 * self.rows * self.row_bytes];
        self.visit(|offset, source, copy| {
            if let (Some(s), false) = (source, copy) {
                padded[s] = raw[offset];
            }
        })?;
        unpack(&self.unpadded(&padded), self.dtype, self.shape, out)
    }

    /// The (matrix, row, logical byte) of each offset, −1 for padding, into
    /// `out` laid out `[offsets][3]`.
    pub fn coords(&self, offsets: &[u64], out: &mut [i64]) -> Result<()> {
        if out.len() != offsets.len() * 3 {
            return Err(Error::Value(format!(
                "coords: {} outputs for {} offsets",
                out.len(),
                offsets.len()
            )));
        }
        if let Some(&o) = offsets.iter().find(|&&o| o >= self.nbytes as u64) {
            return Err(Error::Layout(format!(
                "coords: offset {o} is outside the {} bytes of {}",
                self.nbytes, self.name
            )));
        }
        let sources: Vec<Option<usize>> = match &self.order {
            Order::Recipe(walk) => offsets.iter().map(|&o| walk.source(o as usize).0).collect(),
            Order::Map(map) => {
                let mut flat = vec![0; offsets.len()];
                MapWalk::new(map, self.nbytes, &self.name)?.sources(offsets, &mut flat)?;
                flat.into_iter().map(Some).collect()
            }
        };
        for (source, out) in sources.into_iter().zip(out.chunks_exact_mut(3)) {
            let coords = source.and_then(|s| self.place(s));
            let as_i64 = |c: usize| i64::try_from(c).unwrap_or(i64::MAX);
            out.copy_from_slice(&coords.map_or([-1; 3], |c| c.map(as_i64)));
        }
        Ok(())
    }
}

/// The bytes a layout occupies for a dtype and shape.
pub fn nbytes(layout: &Layout, dtype: &'static Dtype, shape: Shape) -> Result<usize> {
    Ok(Arranged::of(layout, dtype, shape)?.nbytes)
}

/// The bytes of storage elements `[stack][rows][cols]` as the layout
/// arranges them. The elements are checked first, as checks
/// storage, so a sub-byte value out of range is a `ValueError` whatever
/// the layout.
pub fn to_bytes(
    elements: &[u8],
    layout: &Layout,
    dtype: &'static Dtype,
    shape: Shape,
) -> Result<Vec<u8>> {
    logical::check(elements, dtype, shape, "to_bytes")?;
    let arranged = Arranged::of(layout, dtype, shape)?;
    let mut out = vec![0; arranged.nbytes];
    arranged.to_bytes(elements, &mut out)?;
    Ok(out)
}

/// The storage elements back from a layout's bytes: padding is
/// ignored, and every copy of a repeated dimension must equal the first.
pub fn from_bytes(
    raw: &[u8],
    layout: &Layout,
    dtype: &'static Dtype,
    shape: Shape,
    out: &mut [u8],
) -> Result<()> {
    Arranged::of(layout, dtype, shape)?.from_bytes(raw, out)
}

/// The (matrix, row, logical byte) of each offset, −1 for padding, into
/// `out` laid out `[offsets][3]`.
pub fn coords(
    layout: &Layout,
    dtype: &'static Dtype,
    shape: Shape,
    offsets: &[u64],
    out: &mut [i64],
) -> Result<()> {
    Arranged::of(layout, dtype, shape)?.coords(offsets, out)
}

#[cfg(test)]
mod tests;
