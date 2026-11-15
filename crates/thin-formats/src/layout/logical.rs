//! Logical bytes: a buffer's rows as little-endian bytes, sub-byte
//! elements packed into a bit stream per row.

use crate::dtype::Dtype;
use crate::{Error, Result, Shape};

/// The bytes of one logical row: ceil(cols · width / 8).
pub fn row_bytes(dtype: &Dtype, cols: usize) -> Result<usize> {
    cols.checked_mul(dtype.width as usize)
        .map(|bits| bits.div_ceil(8))
        .ok_or_else(|| Error::Layout(format!("{cols} columns of {} overflow", dtype.name)))
}

/// The number of storage elements of a shape, refusing a size that is not
/// at least 1 in each dimension.
pub fn elements(shape: Shape) -> Result<usize> {
    let (stack, rows, cols) = shape;
    if stack == 0 || rows == 0 || cols == 0 {
        return Err(Error::Layout(format!(
            "shape {shape:?}: every size must be at least 1"
        )));
    }
    stack
        .checked_mul(rows)
        .and_then(|n| n.checked_mul(cols))
        .ok_or_else(|| Error::Layout(format!("shape {shape:?} overflows")))
}

/// Checks that `len` bytes hold the storage of a shape of a dtype.
fn check_storage(len: usize, dtype: &Dtype, shape: Shape, what: &str) -> Result<usize> {
    let n = elements(shape)?;
    if len != n * dtype.storage_bytes() {
        return Err(Error::Value(format!(
            "{what}: {len} bytes do not hold {shape:?} {} elements",
            dtype.name
        )));
    }
    Ok(n)
}

/// Checks storage elements `[stack][rows][cols]` as does: their
/// length, and no sub-byte value above 2^width − 1 (a `ValueError`).
pub fn check(elements: &[u8], dtype: &Dtype, shape: Shape, what: &str) -> Result<usize> {
    let n = check_storage(elements.len(), dtype, shape, what)?;
    if dtype.width < 8 {
        let limit = (1u8 << dtype.width) - 1;
        if let Some(i) = elements.iter().position(|&e| e > limit) {
            return Err(Error::Value(format!(
                "{what}: element {i} is {}, above {limit} for the {}-bit {}",
                elements[i], dtype.width, dtype.name
            )));
        }
    }
    Ok(n)
}

/// Packs storage elements `[stack][rows][cols]` into logical bytes
/// `[stack][rows][row_bytes]`, refusing a sub-byte value above its width.
pub fn pack(elements: &[u8], dtype: &Dtype, shape: Shape, out: &mut [u8]) -> Result<()> {
    let n = check(elements, dtype, shape, "pack")?;
    let cols = shape.2;
    let row = row_bytes(dtype, cols)?;
    if out.len() != n / cols * row {
        return Err(Error::Value(format!(
            "pack: {} output bytes for {shape:?} {}",
            out.len(),
            dtype.name
        )));
    }
    let width = dtype.width;
    if width >= 8 {
        out.copy_from_slice(elements);
        return Ok(());
    }
    for (src, dst) in elements.chunks_exact(cols).zip(out.chunks_exact_mut(row)) {
        dst.fill(0);
        for (i, &e) in src.iter().enumerate() {
            let bit = i * width as usize;
            let word = u16::from(e) << (bit % 8);
            dst[bit / 8] |= word as u8;
            if word > 0xFF {
                dst[bit / 8 + 1] |= (word >> 8) as u8;
            }
        }
    }
    Ok(())
}

/// Unpacks logical bytes into storage elements, the inverse of [`pack`];
/// a row's trailing partial bits are ignored.
pub fn unpack(logical: &[u8], dtype: &Dtype, shape: Shape, out: &mut [u8]) -> Result<()> {
    let n = check_storage(out.len(), dtype, shape, "unpack")?;
    let cols = shape.2;
    let row = row_bytes(dtype, cols)?;
    if logical.len() != n / cols * row {
        return Err(Error::Value(format!(
            "unpack: {} logical bytes for {shape:?} {}",
            logical.len(),
            dtype.name
        )));
    }
    let width = dtype.width;
    if width >= 8 {
        out.copy_from_slice(logical);
        return Ok(());
    }
    let mask = (1u16 << width) - 1;
    for (src, dst) in logical.chunks_exact(row).zip(out.chunks_exact_mut(cols)) {
        for (i, e) in dst.iter_mut().enumerate() {
            let bit = i * width as usize;
            let low = u16::from(src[bit / 8]);
            let high = src.get(bit / 8 + 1).map_or(0, |&b| u16::from(b));
            *e = ((low | high << 8) >> (bit % 8) & mask) as u8;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::dtype;

    #[test]
    fn six_bit_elements_pack_four_to_three_bytes() {
        let d = dtype("e3m2").unwrap();
        let elements = [0x3F, 0x01, 0x20, 0x15, 0x2A];
        let mut logical = [0; 4];
        pack(&elements, d, (1, 1, 5), &mut logical).unwrap();
        assert_eq!(logical, [0x7F, 0x00, 0x56, 0x2A]);
        let mut back = [0; 5];
        unpack(&logical, d, (1, 1, 5), &mut back).unwrap();
        assert_eq!(back, elements);
    }

    #[test]
    fn values_above_the_width_are_refused() {
        let d = dtype("e2m1").unwrap();
        let mut out = [0; 1];
        assert!(matches!(
            pack(&[3, 16], d, (1, 1, 2), &mut out),
            Err(Error::Value(_))
        ));
        assert!(matches!(
            pack(&[], d, (0, 1, 2), &mut out),
            Err(Error::Layout(_))
        ));
    }
}
