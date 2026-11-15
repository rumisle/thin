//! Evaluate generator-authored IR programs through thin-ir's own analyses.
//!
//! Reads a JSON list of requests from stdin and writes one JSON list of
//! results to stdout. Each request names an analysis (`kind`) and carries a
//! program in the golden line notation; the result is serialized exactly as
//! the `ir` golden files record it. Nothing here reads a golden file.

use std::cell::RefCell;
use std::collections::BTreeSet;
use std::io::Read;
use std::sync::Arc;

use serde_json::{json, Value};
use thin_ir::cfg::{cfg_of, Cfg};
use thin_ir::solve::{solve_backward, solve_forward, solve_forward_all, solve_forward_on, Widen};
use thin_ir::{dpp_reads_old, FieldId, FieldValue, Instr, Snapshot};
use thin_target::PHYS;
use thin_test_support::adapters::default_target;
use thin_test_support::notation::{abstract_line, Lines};

/// The strings of a JSON list.
fn strs(v: &Value) -> Vec<&str> {
    v.as_array()
        .map_or(Vec::new(), |a| a.iter().filter_map(Value::as_str).collect())
}

/// The gfx1201 instructions of notation lines, and a bare snapshot of them.
fn snapshot(lines: &[&str]) -> Result<Snapshot, String> {
    let mut l = Lines::new(default_target());
    let instrs = l.read(lines)?;
    Ok(Snapshot::new(default_target(), l.values, instrs.into()))
}

/// An abstract CFG stream as gfx1201 instructions.
fn stream(v: &Value) -> Result<Vec<Arc<Instr>>, String> {
    let lines: Vec<String> = strs(v).into_iter().map(abstract_line).collect();
    Lines::new(default_target()).read(&lines.iter().map(String::as_str).collect::<Vec<_>>())
}

/// Every list of a graph.
fn graph(g: &Cfg) -> Value {
    let rpo = g.rpo();
    let back: Vec<[usize; 2]> = (g.back_edges(&rpo).into_iter())
        .map(|(q, t)| [q, t])
        .collect();
    json!({
        "succ": g.succ, "preds": g.preds, "starts": g.starts, "block_of": g.block_of,
        "block_succ": g.block_succ, "block_preds": g.block_preds, "rpo": rpo,
        "back_edges": back, "reached": g.reached(),
    })
}

/// Both `label_blocks` settings of a stream, under an optional wave class.
fn cfg(r: &Value) -> Result<Value, String> {
    let instrs = stream(&r["stream"])?;
    let waves: Option<Vec<(usize, bool)>> = r.get("waves").filter(|w| !w.is_null()).map(|w| {
        (w.as_array().into_iter().flatten())
            .map(|p| (p[0].as_u64().unwrap_or(0) as usize, p[1] == true))
            .collect()
    });
    Ok(json!({
        "cfg": graph(&cfg_of(&instrs, waves.as_deref(), false)),
        "cfg_label_blocks": graph(&cfg_of(&instrs, waves.as_deref(), true)),
    }))
}

/// The five solver modes on one stream: entry states and visit orders.
fn solvers(r: &Value) -> Result<Value, String> {
    let g = cfg_of(&stream(&r["stream"])?, None, false);
    let blocks: Vec<[usize; 2]> = (0..g.nblocks()).map(|j| [g.starts[j], g.end(j)]).collect();
    let visits = RefCell::new(Vec::new());
    let len = |j: usize| g.block(j).len() as u32;
    let capped = |s: &u32, j: usize| {
        visits.borrow_mut().push(j);
        (s + len(j)).min(10)
    };
    let max = |a: &u32, b: &u32| *a.max(b);
    let record = |key: &str, state: Value| json!({key: state, "visits": visits.take()});
    let a = record(
        "entry",
        json!(solve_forward(&g, vec![(0, 0)], capped, max, None)),
    );
    let plus = |s: &u32, j: usize| {
        visits.borrow_mut().push(j);
        (s + 1).min(1000)
    };
    let widen = |_: &u32, _: &u32| 1000;
    let widening = Widen {
        widen: &widen,
        patience: 2,
    };
    let b = solve_forward(&g, vec![(0, 0)], plus, max, Some(widening));
    let b = record("entry", json!(b));
    let c = solve_forward_on(&g, vec![(0, 0)], capped, max, None, |p, t| t > p);
    let c = record("entry", json!(c));
    let with = |s: &BTreeSet<usize>, j: usize| {
        visits.borrow_mut().push(j);
        let mut s = s.clone();
        s.insert(j);
        s
    };
    let union = |a: &BTreeSet<usize>, b: &BTreeSet<usize>| a | b;
    let d = solve_forward_all(&g, BTreeSet::new(), vec![(0, BTreeSet::new())], with, union);
    let d = record("entry", json!(d));
    let union_all = |l: &[&BTreeSet<usize>]| l.iter().fold(BTreeSet::new(), |a, b| &a | b);
    let e = record(
        "before",
        json!(solve_backward(&g, BTreeSet::new(), union_all, with)),
    );
    Ok(json!({
        "blocks": blocks, "block_succ": g.block_succ, "A_forward_capped": a,
        "B_forward_widened": b, "C_forward_follow": c, "D_forward_all": d, "E_backward": e,
    }))
}

/// EXEC on entry to each instruction.
fn exec(r: &Value) -> Result<Value, String> {
    let s = snapshot(&strs(&r["lines"]))?;
    let got: Vec<Option<String>> = (s.exec().iter())
        .map(|m| m.map(|m| format!("{m:#018x}")))
        .collect();
    Ok(json!(got))
}

/// The tracked ops in flight on entry to each instruction.
fn flight(r: &Value) -> Result<Value, String> {
    let s = snapshot(&strs(&r["lines"]))?;
    let loads: Vec<usize> = (r["loads"].as_array().into_iter().flatten())
        .filter_map(Value::as_u64)
        .map(|k| k as usize)
        .collect();
    Ok(json!(thin_ir::flight::in_flight(&s, &s.cfg(None), &loads)))
}

/// The redundant M0 writes.
fn m0(r: &Value) -> Result<Value, String> {
    let s = snapshot(&strs(&r["lines"]))?;
    Ok(json!(thin_ir::m0::redundant(&s)))
}

/// Each line's derived facts, with one value table across the list.
fn instrs(r: &Value) -> Result<Value, String> {
    let all = strs(&r["lines"]);
    let instrs = Lines::new(default_target()).read(&all)?;
    let regs = |it: &mut dyn Iterator<Item = &thin_ir::Reg>| -> Value {
        it.map(|r| json!([r.value, r.lo, r.n])).collect()
    };
    let names = |l: &[usize]| -> Value { l.iter().map(|&p| PHYS[p].name).collect() };
    let facts = (all.iter().zip(&instrs)).map(|(line, i)| {
        json!({
            "line": line, "roles": i.roles(), "defs": regs(&mut i.defs()),
            "uses": regs(&mut i.uses()), "writes": names(i.writes()), "reads": names(i.reads()),
            "sgpr_valu_def": i.sgpr_valu_def(), "writes_exec": i.writes_exec(),
            "m0_write": i.m0_write(),
        })
    });
    Ok(Value::Array(facts.collect()))
}

/// Whether a DPP move with these fields reads its old destination.
fn dpp(r: &Value) -> Result<Value, String> {
    let fields = (r["fields"].as_object().ok_or("fields")?.iter())
        .map(|(k, v)| {
            let value = match v {
                Value::Bool(b) => FieldValue::Bool(*b),
                Value::String(t) => FieldValue::Text(t.as_str().into()),
                n => FieldValue::Int(n.as_i64().ok_or("an int")? as i128),
            };
            Ok((FieldId::named(k).map_err(|e| e.to_string())?, value))
        })
        .collect::<Result<Vec<_>, String>>()?;
    Ok(json!(dpp_reads_old(&fields)))
}

/// Wave groups, classes and each class's instruction successors.
fn waves(r: &Value) -> Result<Value, String> {
    let s = snapshot(&strs(&r["lines"]))?;
    let s = s.with_waves(
        r["nwaves"].as_u64().ok_or("nwaves")? as u32,
        r["by_wave"] == true,
    );
    let groups: Vec<Value> = (s.wave_groups().into_iter())
        .map(|(waves, taken)| json!({"waves": waves, "taken": taken}))
        .collect();
    let classes = s.wave_classes();
    let succ: Vec<Value> = (classes.iter())
        .map(|c| json!(s.cfg(c.as_deref()).succ))
        .collect();
    Ok(json!({
        "nwaves": s.nwaves(), "by_wave": s.by_wave(), "wave_groups": groups,
        "wave_classes": classes, "class_cfg_succ": succ,
    }))
}

fn run(r: &Value) -> Result<Value, String> {
    match r["kind"].as_str().unwrap_or("") {
        "cfg" => cfg(r),
        "solvers" => solvers(r),
        "exec" => exec(r),
        "flight" => flight(r),
        "m0" => m0(r),
        "instrs" => instrs(r),
        "dpp" => dpp(r),
        "waves" => waves(r),
        other => Err(format!("unknown request kind {other:?}")),
    }
}

fn main() {
    let mut input = String::new();
    std::io::stdin()
        .read_to_string(&mut input)
        .expect("a request list on stdin");
    let requests: Vec<Value> = serde_json::from_str(&input).expect("a JSON request list");
    let results: Vec<Value> = (requests.iter())
        .map(|r| run(r).unwrap_or_else(|e| json!({"error": e})))
        .collect();
    println!("{}", Value::Array(results));
}
