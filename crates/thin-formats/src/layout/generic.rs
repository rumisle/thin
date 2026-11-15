//! The generic layout definitions, `row_major` and `col_major`.

use super::{Arrangement, Definition, Recipe};
use crate::dtype::Dtype;
use crate::{Error, Result, Shape};

/// Each row after the previous one, `ld` elements apart, or the columns
/// rounded up to a multiple of `pad_to`.
pub static ROW_MAJOR: Definition = Definition {
    name: "row_major",
    params: &[("ld", 0), ("pad_to", 0)],
    build: row_major,
};

/// Each matrix stored transposed, element bytes kept together.
pub static COL_MAJOR: Definition = Definition {
    name: "col_major",
    params: &[],
    build: col_major,
};

fn row_major(dtype: Option<&Dtype>, shape: Shape, params: &[i64]) -> Result<Arrangement> {
    let refuse = |why: String| Err(Error::Layout(format!("row_major: {why}")));
    let (ld, pad_to) = (params[0], params[1]);
    if ld != 0 && pad_to != 0 {
        return refuse(format!("ld={ld} and pad_to={pad_to}: give one of them"));
    }
    if ld < 0 || pad_to < 0 {
        return refuse(format!("ld={ld}, pad_to={pad_to}: neither may be negative"));
    }
    let Some(dtype) = dtype else {
        return Ok(Arrangement::Recipe(Recipe::new(0, 0, vec![0], vec![0])));
    };
    let (stack, rows, cols) = shape;
    let (ld, pad_to) = (ld as usize, pad_to as usize);
    if 0 < ld && ld < cols {
        return refuse(format!("ld={ld} is below the {cols} columns"));
    }
    let count = match (ld, pad_to) {
        (0, 0) => Some(cols),
        (0, pad_to) => cols.div_ceil(pad_to).checked_mul(pad_to),
        (ld, _) => Some(ld),
    };
    let bits = count.and_then(|c| c.checked_mul(dtype.width as usize));
    let Some(bits) = bits else {
        return refuse(format!(
            "rows of {count:?} {} elements overflow",
            dtype.name
        ));
    };
    if bits % 8 != 0 {
        return refuse(format!(
            "a row of {} {} elements is {bits} bits, not whole bytes",
            bits / dtype.width as usize,
            dtype.name
        ));
    }
    let row_bytes = bits / 8;
    let flat = stack
        .checked_mul(rows)
        .and_then(|n| n.checked_mul(row_bytes));
    let Some(flat) = flat else {
        return refuse(format!(
            "{shape:?} with rows of {row_bytes} bytes overflows"
        ));
    };
    Ok(Arrangement::Recipe(Recipe::new(
        rows,
        row_bytes,
        vec![flat],
        vec![0],
    )))
}

fn col_major(dtype: Option<&Dtype>, shape: Shape, _: &[i64]) -> Result<Arrangement> {
    let Some(dtype) = dtype else {
        return Ok(Arrangement::Recipe(Recipe::new(0, 0, vec![0], vec![0])));
    };
    if dtype.width < 8 {
        return Err(Error::Layout(format!(
            "col_major: {} is a sub-byte dtype; its elements share bytes",
            dtype.name
        )));
    }
    let (stack, rows, cols) = shape;
    let element = dtype.storage_bytes();
    Ok(Arrangement::Recipe(Recipe::new(
        rows,
        cols * element,
        vec![stack, rows, cols, element],
        vec![0, 2, 1, 3],
    )))
}
