//! Layouts and problems from host data: the JSON a
//! binding sends, since checkers take data and never call back.

use std::collections::{BTreeSet, HashMap};

use serde_json::Value;

use super::layout::{Kind, Layout, Shape};
use super::problem::{DCols, Problem, RecordOperand};

fn int(v: &Value, k: &str, default: i64) -> i64 {
    v.get(k).and_then(Value::as_i64).unwrap_or(default)
}

fn ints(v: &Value) -> Vec<i64> {
    v.as_array()
        .into_iter()
        .flatten()
        .map(|x| x.as_i64().unwrap_or(0))
        .collect()
}

fn table(v: &Value) -> Vec<Vec<i64>> {
    v.as_array().into_iter().flatten().map(ints).collect()
}

/// A layout: `{shape: rowmajor | rowmajor_d, kind, ld_bytes, elem_bytes,
/// rows_per_batch, batch_bytes}`.
pub fn layout(v: &Value) -> Result<Layout, String> {
    let kind = Kind::named(v.get("kind").and_then(Value::as_str).unwrap_or("A"));
    let (ld, elem) = (int(v, "ld_bytes", 0), int(v, "elem_bytes", 1));
    let (rpb, bb) = (int(v, "rows_per_batch", 0), int(v, "batch_bytes", 0));
    let shape = match v.get("shape").and_then(Value::as_str).unwrap_or("") {
        "rowmajor" => Shape::RowMajor {
            ld,
            elem,
            rows_per_batch: rpb,
            batch_bytes: bb,
        },
        "rowmajor_d" => Shape::RowMajorD {
            ld,
            elem,
            rows_per_batch: rpb,
            batch_bytes: bb,
        },
        other => return Err(format!("layout {other:?}: rowmajor or rowmajor_d")),
    };
    let kind = if matches!(shape, Shape::RowMajorD { .. }) {
        Kind::D
    } else {
        kind
    };
    Ok(Layout::new(kind, shape))
}

fn sets(v: &Value) -> HashMap<i64, BTreeSet<i64>> {
    v.as_object()
        .into_iter()
        .flatten()
        .map(|(k, s)| (k.parse().unwrap_or(0), ints(s).into_iter().collect()))
        .collect()
}

fn d_cols(v: &Value) -> DCols {
    match v.get("tuple") {
        Some(t) => DCols::Tuple(t.as_array().into_iter().flatten().map(d_cols).collect()),
        None if v
            .as_array()
            .is_some_and(|a| a.first().is_some_and(Value::is_array)) =>
        {
            DCols::Grid(table(v))
        }
        None => DCols::Row(ints(v)),
    }
}

/// A problem: `{layouts: {name: layout | null}, K, d_cols, d_rows,
/// d_kblocks, N, record_operand: {side, source, rows, ks}, scatter, pid,
/// fail_fast}`.
pub fn problem(v: &Value) -> Result<Problem, String> {
    let mut layouts = Vec::new();
    for (name, l) in v
        .get("layouts")
        .and_then(Value::as_object)
        .into_iter()
        .flatten()
    {
        layouts.push((
            name.clone(),
            if l.is_null() { None } else { Some(layout(l)?) },
        ));
    }
    let mut p = Problem::new(layouts, int(v, "K", 0));
    p.d_rows = v.get("d_rows").filter(|x| !x.is_null()).map(ints);
    p.d_cols = v.get("d_cols").filter(|x| !x.is_null()).map(d_cols);
    p.d_kblocks = v.get("d_kblocks").filter(|x| !x.is_null()).map(sets);
    p.n = v.get("N").and_then(Value::as_i64);
    p.scatter = v.get("scatter").filter(|x| !x.is_null()).map(sets);
    p.record_operand = v
        .get("record_operand")
        .filter(|x| !x.is_null())
        .map(|r| RecordOperand {
            side: int(r, "side", 0),
            source: int(r, "source", -1),
            map: r
                .get("rows")
                .filter(|x| !x.is_null())
                .map(|rows| (table(rows), table(&r["ks"]))),
        });
    p.pid = int(v, "pid", -1);
    p.fail_fast = v.get("fail_fast").and_then(Value::as_bool).unwrap_or(true);
    Ok(p)
}

/// A layout from JSON text.
pub fn layout_text(text: &str) -> Result<Layout, String> {
    layout(&serde_json::from_str(text).map_err(|e| e.to_string())?)
}

/// Problems from the JSON text of a list.
pub fn problems_text(text: &str) -> Result<Vec<Problem>, String> {
    let v: Value = serde_json::from_str(text).map_err(|e| e.to_string())?;
    v.as_array().into_iter().flatten().map(problem).collect()
}
