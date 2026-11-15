//! Walking a Recipe or a Map: which byte of the padded logical buffer each
//! offset of a layout holds.

use super::{Map, Recipe};
use crate::{Error, Result};

/// The most offsets a Map decodes at once.
pub const CHUNK: usize = 1 << 24;

/// The product of sizes, refusing an overflow.
fn product(sizes: &[usize], what: &str) -> Result<usize> {
    sizes
        .iter()
        .try_fold(1usize, |n, &s| n.checked_mul(s))
        .ok_or_else(|| Error::Layout(format!("{what}: sizes {sizes:?} overflow")))
}

/// Row-major strides of a shape.
fn strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    for d in (0..shape.len().saturating_sub(1)).rev() {
        strides[d] = strides[d + 1] * shape[d + 1];
    }
    strides
}

/// A Recipe prepared for walking: its offsets index the padded out shape
/// whose unpadded part is the transposed buffer. A repeated
/// dimension is copied and never padded: its pad is ignored.
#[derive(Debug)]
pub(super) struct RecipeWalk {
    /// The out dimensions before padding.
    out: Vec<usize>,
    /// Row-major strides of `out`.
    out_strides: Vec<usize>,
    /// Padding before each out dimension.
    before: Vec<usize>,
    /// Each out dimension padded.
    padded: Vec<usize>,
    /// Each padded dimension times its copies: the offsets' shape.
    total: Vec<usize>,
    /// The transposed dimensions: `dims` permuted.
    tdims: Vec<usize>,
    /// The padded-logical stride of each transposed dimension.
    tstrides: Vec<usize>,
}

impl RecipeWalk {
    /// Storage dimensions after padding and repetition.
    pub(super) fn storage_shape(&self) -> &[usize] {
        &self.total
    }

    /// Checks a Recipe whose padded logical buffer has `logical` bytes and
    /// prepares its walk; returns it with the layout's byte count.
    pub(super) fn new(recipe: &Recipe, logical: usize, name: &str) -> Result<(RecipeWalk, usize)> {
        let n = recipe.dims.len();
        let mut seen = vec![false; n];
        let permutes = recipe.perm.len() == n
            && recipe
                .perm
                .iter()
                .all(|&p| p < n && !std::mem::replace(&mut seen[p], true));
        if n == 0 || !permutes {
            return Err(Error::Layout(format!(
                "{name}: perm {:?} does not permute dims {:?}",
                recipe.perm, recipe.dims
            )));
        }
        let tdims: Vec<usize> = recipe.perm.iter().map(|&p| recipe.dims[p]).collect();
        let out = recipe.out.clone().unwrap_or_else(|| tdims.clone());
        if out.is_empty()
            || product(&recipe.dims, name)? != logical
            || product(&out, name)? != logical
        {
            return Err(Error::Layout(format!(
                "{name}: dims {:?} and out {out:?} do not hold the {logical} padded logical bytes",
                recipe.dims
            )));
        }
        let mut pad = if recipe.pad.is_empty() {
            vec![(0, 0); out.len()]
        } else {
            recipe.pad.clone()
        };
        if pad.len() != out.len() {
            return Err(Error::Layout(format!(
                "{name}: {} pads for {} out dimensions",
                pad.len(),
                out.len()
            )));
        }
        let mut repeated = vec![None; out.len()];
        for &(d, copies) in &recipe.repeat {
            if d >= out.len() || out[d] != 1 || copies == 0 || repeated[d].is_some() {
                return Err(Error::Layout(format!(
                    "{name}: repeat ({d}, {copies}) needs an out dimension of size 1, once"
                )));
            }
            repeated[d] = Some(copies);
            pad[d] = (0, 0);
        }
        let padded: Vec<usize> = (out.iter().zip(&pad))
            .map(|(&o, &(b, a))| o.checked_add(b).and_then(|s| s.checked_add(a)))
            .collect::<Option<_>>()
            .ok_or_else(|| Error::Layout(format!("{name}: pad {pad:?} overflows")))?;
        let total: Vec<usize> = (padded.iter().zip(&repeated))
            .map(|(&p, r)| r.unwrap_or(p))
            .collect();
        let nbytes = product(&total, name)?;
        let dim_strides = strides(&recipe.dims);
        let walk = RecipeWalk {
            out_strides: strides(&out),
            before: pad.iter().map(|&(b, _)| b).collect(),
            tstrides: recipe.perm.iter().map(|&p| dim_strides[p]).collect(),
            out,
            padded,
            total,
            tdims,
        };
        Ok((walk, nbytes))
    }

    /// The first offset of `raw` whose byte differs from the same place in
    /// the first copy of a repeated dimension, padding included.
    pub(super) fn first_differing_copy(&self, raw: &[u8]) -> Option<usize> {
        for d in (0..self.total.len()).filter(|&d| self.total[d] != self.padded[d]) {
            let inner: usize = self.total[d + 1..].iter().product();
            let block = inner * self.total[d];
            for (b, outer) in raw.chunks_exact(block).enumerate() {
                let (first, rest) = outer.split_at(inner);
                for (k, copy) in rest.chunks_exact(inner).enumerate() {
                    if let Some(i) = copy.iter().zip(first).position(|(a, b)| a != b) {
                        return Some(b * block + (k + 1) * inner + i);
                    }
                }
            }
        }
        None
    }

    /// The unpadded out index of position `o` of dimension `d`, and whether
    /// the position is a copy past the first.
    #[inline]
    fn unpadded(&self, d: usize, o: usize) -> (Option<usize>, bool) {
        let i = o % self.padded[d];
        let u = i.checked_sub(self.before[d]).filter(|&u| u < self.out[d]);
        (u, o >= self.padded[d])
    }

    /// The padded-logical index of transposed flat index `f`, with the
    /// transposed multi-index left in `t`.
    fn locate(&self, f: usize, t: &mut [usize]) -> usize {
        let (mut rest, mut src) = (f, 0);
        for j in (0..t.len()).rev() {
            t[j] = rest % self.tdims[j];
            rest /= self.tdims[j];
            src += t[j] * self.tstrides[j];
        }
        src
    }

    /// The padded-logical index after `src`, whose multi-index is `t`.
    #[inline]
    fn step(&self, t: &mut [usize], mut src: usize) -> usize {
        for j in (0..t.len()).rev() {
            t[j] += 1;
            src += self.tstrides[j];
            if t[j] < self.tdims[j] {
                return src;
            }
            src -= self.tdims[j] * self.tstrides[j];
            t[j] = 0;
        }
        src
    }

    /// The source of one offset (`None` for padding) and whether it is a
    /// copy past the first.
    pub(super) fn source(&self, offset: usize) -> (Option<usize>, bool) {
        let (mut rest, mut f, mut padding, mut copy) = (offset, 0, false, false);
        for d in (0..self.total.len()).rev() {
            let (u, again) = self.unpadded(d, rest % self.total[d]);
            rest /= self.total[d];
            copy |= again;
            match u {
                Some(u) => f += u * self.out_strides[d],
                None => padding = true,
            }
        }
        let mut t = vec![0; self.tdims.len()];
        ((!padding).then(|| self.locate(f, &mut t)), copy)
    }

    /// Calls `visit(offset, source, copy)` for every offset in order.
    pub(super) fn visit(&self, mut visit: impl FnMut(usize, Option<usize>, bool)) {
        let last = self.total.len() - 1;
        let lines: usize = self.total[..last].iter().product();
        let mut outer = vec![0; last];
        let mut t = vec![0; self.tdims.len()];
        let (mut offset, mut next, mut src) = (0, usize::MAX, 0);
        for _ in 0..lines {
            let (mut base, mut padding, mut line_copy) = (0, false, false);
            for (d, &o) in outer.iter().enumerate() {
                let (u, again) = self.unpadded(d, o);
                line_copy |= again;
                match u {
                    Some(u) => base += u * self.out_strides[d],
                    None => padding = true,
                }
            }
            for o in 0..self.total[last] {
                let (u, again) = self.unpadded(last, o);
                match u.filter(|_| !padding) {
                    Some(u) => {
                        let f = base + u;
                        src = if f == next {
                            self.step(&mut t, src)
                        } else {
                            self.locate(f, &mut t)
                        };
                        next = f + 1;
                        visit(offset, Some(src), line_copy || again);
                    }
                    None => visit(offset, None, line_copy || again),
                }
                offset += 1;
            }
            for d in (0..last).rev() {
                outer[d] += 1;
                if outer[d] < self.total[d] {
                    break;
                }
                outer[d] = 0;
            }
        }
    }
}

/// A Map prepared for walking.
pub(super) struct MapWalk<'a> {
    map: &'a Map,
    name: &'a str,
    /// The period P, the whole size when there is none.
    period: usize,
    /// The flat shift per period.
    shift: i64,
}

impl<'a> MapWalk<'a> {
    /// Checks a Map whose padded logical buffer has `logical` bytes.
    pub(super) fn new(map: &'a Map, logical: usize, name: &'a str) -> Result<MapWalk<'a>> {
        if map.size != logical {
            return Err(Error::Layout(format!(
                "{name}: a map of {} bytes for {logical} logical bytes",
                map.size
            )));
        }
        let (period, shift) = map.period.unwrap_or((map.size, 0));
        if period == 0 || map.size % period != 0 {
            return Err(Error::Layout(format!(
                "{name}: period {period} does not divide the size {}",
                map.size
            )));
        }
        Ok(MapWalk {
            map,
            name,
            period,
            shift,
        })
    }

    /// The flat index of a decoded (matrix, row, byte), `periods` periods on.
    fn flat(&self, offset: u64, periods: usize, [m, r, b]: [usize; 3]) -> Result<usize> {
        let map = self.map;
        (r < map.rows && b < map.row_bytes)
            .then(|| {
                m.checked_mul(map.rows)?
                    .checked_add(r)?
                    .checked_mul(map.row_bytes)?
                    .checked_add(b)
            })
            .flatten()
            .and_then(|base| i64::try_from(base).ok())
            .and_then(|base| base.checked_add(self.shift.checked_mul(periods as i64)?))
            .and_then(|flat| usize::try_from(flat).ok())
            .filter(|&flat| flat < map.size)
            .ok_or_else(|| {
                Error::Layout(format!(
                    "{}: offset {offset} decodes outside the buffer",
                    self.name
                ))
            })
    }

    /// The flat source of each of `offsets` (each below the size), decoded
    /// in chunks of at most [`CHUNK`].
    pub(super) fn sources(&self, offsets: &[u64], flat: &mut [usize]) -> Result<()> {
        let period = self.period as u64;
        for (chunk, out) in offsets.chunks(CHUNK).zip(flat.chunks_mut(CHUNK)) {
            let reduced: Vec<u64> = chunk.iter().map(|&o| o % period).collect();
            let mut decoded = vec![[0; 3]; chunk.len()];
            (self.map.decode)(&reduced, &mut decoded);
            for ((&o, coords), f) in chunk.iter().zip(decoded).zip(out) {
                *f = self.flat(o, (o / period) as usize, coords)?;
            }
        }
        Ok(())
    }

    /// Calls `visit(offset, source)` for every offset, decoding only the
    /// first period's offsets.
    pub(super) fn visit(&self, mut visit: impl FnMut(usize, usize)) -> Result<()> {
        for start in (0..self.period).step_by(CHUNK) {
            let count = CHUNK.min(self.period - start);
            let offsets: Vec<u64> = (start..start + count).map(|o| o as u64).collect();
            let mut decoded = vec![[0; 3]; count];
            (self.map.decode)(&offsets, &mut decoded);
            for k in 0..self.map.size / self.period {
                for (&o, &coords) in offsets.iter().zip(&decoded) {
                    let offset = k * self.period + o as usize;
                    visit(offset, self.flat(offset as u64, k, coords)?);
                }
            }
        }
        Ok(())
    }
}
