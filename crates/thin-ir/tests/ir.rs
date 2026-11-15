//! Golden checks for instructions, CFGs, solvers, analyses and verification.

use std::cell::RefCell;
use std::collections::BTreeSet;
use std::sync::Arc;

use serde_json::{json, Value};
use thin_ir::cfg::{cfg_of, Cfg};
use thin_ir::sets::{Bits, Runs, Sparse};
use thin_ir::solve::{solve_backward, solve_forward, solve_forward_all, solve_forward_on, Widen};
use thin_ir::{dpp_reads_old, FieldId, FieldValue, Instr, Snapshot};
use thin_target::PHYS;
use thin_test_support::adapters::default_target;
use thin_test_support::golden_tests::expect_eq;
use thin_test_support::notation::{abstract_line, Lines};
use thin_test_support::{Case, Suite};

/// The strings of a JSON list.
fn strs(v: &Value) -> Vec<&str> {
    v.as_array()
        .map_or(Vec::new(), |a| a.iter().filter_map(Value::as_str).collect())
}

/// A case's gfx1201 lines as instructions, and a bare snapshot of them.
fn snapshot(lines: &[&str]) -> Result<Snapshot, String> {
    let mut l = Lines::new(default_target());
    let instrs = l.read(lines)?;
    Ok(Snapshot::new(default_target(), l.values, instrs.into()))
}

/// An abstract stream as gfx1201 instructions.
fn stream(case: &Value) -> Result<Vec<Arc<Instr>>, String> {
    let lines: Vec<String> = strs(&case["stream"])
        .into_iter()
        .map(abstract_line)
        .collect();
    Lines::new(default_target()).read(&lines.iter().map(String::as_str).collect::<Vec<_>>())
}

/// Every list of a graph, as `cfg.json` writes them.
fn cfg_json(g: &Cfg) -> Value {
    let rpo = g.rpo();
    let back: Vec<[usize; 2]> = g
        .back_edges(&rpo)
        .into_iter()
        .map(|(q, t)| [q, t])
        .collect();
    json!({
        "succ": g.succ, "preds": g.preds, "starts": g.starts, "block_of": g.block_of,
        "block_succ": g.block_succ, "block_preds": g.block_preds, "rpo": rpo,
        "back_edges": back, "reached": g.reached(),
    })
}

/// `cfg.json`: both `label_blocks` settings, with the case's wave class.
fn cfg_case(case: &Case<'_>) -> Result<(), String> {
    let instrs = stream(case.value)?;
    let waves: Option<Vec<(usize, bool)>> = case.value.get("waves").map(|w| {
        (w.as_array().into_iter().flatten())
            .map(|p| {
                (
                    p[0].as_u64().unwrap_or(0) as usize,
                    p[1].as_bool().unwrap_or(false),
                )
            })
            .collect()
    });
    for (key, label_blocks) in [("cfg", false), ("cfg_label_blocks", true)] {
        let g = cfg_of(&instrs, waves.as_deref(), label_blocks);
        expect_eq(cfg_json(&g), case.value[key].clone()).map_err(|e| format!("{key}: {e}"))?;
    }
    Ok(())
}

/// `solvers.json`: the five analyses of its notation, entry states and
/// visit orders.
fn solvers_case(case: &Case<'_>) -> Result<(), String> {
    let g = cfg_of(&stream(case.value)?, None, false);
    let blocks: Vec<[usize; 2]> = (0..g.nblocks()).map(|j| [g.starts[j], g.end(j)]).collect();
    expect_eq(json!(blocks), case.value["blocks"].clone())?;
    expect_eq(json!(g.block_succ), case.value["block_succ"].clone())?;
    let visits = RefCell::new(Vec::new());
    let len = |j: usize| g.block(j).len() as u32;
    let capped = |s: &u32, j: usize| {
        visits.borrow_mut().push(j);
        (s + len(j)).min(10)
    };
    let max = |a: &u32, b: &u32| *a.max(b);
    let check = |key: &str, entry: Value| {
        let got = json!({"entry": entry, "visits": visits.take()});
        let want = &case.value[key];
        let want =
            json!({"entry": want.get("entry").or(want.get("before")), "visits": want["visits"]});
        expect_eq(got, want).map_err(|e| format!("{key}: {e}"))
    };
    let a = solve_forward(&g, vec![(0, 0)], capped, max, None);
    check("A_forward_capped", json!(a))?;
    let plus = |s: &u32, j: usize| {
        visits.borrow_mut().push(j);
        (s + 1).min(1000)
    };
    let widen = |_: &u32, _: &u32| 1000;
    let b = solve_forward(
        &g,
        vec![(0, 0)],
        plus,
        max,
        Some(Widen {
            widen: &widen,
            patience: 2,
        }),
    );
    check("B_forward_widened", json!(b))?;
    let c = solve_forward_on(&g, vec![(0, 0)], capped, max, None, |p, t| t > p);
    check("C_forward_follow", json!(c))?;
    let with = |s: &BTreeSet<usize>, j: usize| {
        visits.borrow_mut().push(j);
        let mut s = s.clone();
        s.insert(j);
        s
    };
    let union = |a: &BTreeSet<usize>, b: &BTreeSet<usize>| a | b;
    let d = solve_forward_all(&g, BTreeSet::new(), vec![(0, BTreeSet::new())], with, union);
    check("D_forward_all", json!(d))?;
    let union_all = |l: &[&BTreeSet<usize>]| l.iter().fold(BTreeSet::new(), |a, b| &a | b);
    let e = solve_backward(&g, BTreeSet::new(), union_all, with);
    check("E_backward", json!(e))
}

/// `exec.json`: EXEC on entry to each instruction.
fn exec_case(case: &Case<'_>) -> Result<(), String> {
    let s = snapshot(&strs(&case.value["lines"]))?;
    let got: Vec<Option<String>> = s
        .exec()
        .iter()
        .map(|m| m.map(|m| format!("{m:#018x}")))
        .collect();
    expect_eq(json!(got), case.value["exec"].clone())
}

/// `flight.json`: the tracked ops in flight on entry to each instruction.
fn flight_case(case: &Case<'_>) -> Result<(), String> {
    let s = snapshot(&strs(&case.value["lines"]))?;
    let loads: Vec<usize> = (case.value["loads"].as_array().into_iter().flatten())
        .filter_map(Value::as_u64)
        .map(|k| k as usize)
        .collect();
    let got = thin_ir::flight::in_flight(&s, &s.cfg(None), &loads);
    expect_eq(json!(got), case.value["in_flight"].clone())
}

/// `m0.json`: the redundant M0 writes.
fn m0_case(case: &Case<'_>) -> Result<(), String> {
    let s = snapshot(&strs(&case.value["lines"]))?;
    expect_eq(
        json!(thin_ir::m0::redundant(&s)),
        case.value["redundant"].clone(),
    )
}

/// `instr.json` `instrs`: derived facts, with one value table across the
/// list.
fn instr_case(case: &Case<'_>) -> Result<(), String> {
    let mut lines = Lines::new(default_target());
    let k: usize = case
        .id
        .trim_start_matches("instrs[")
        .trim_end_matches(']')
        .parse()
        .map_err(|_| "an index")?;
    let all: Vec<&str> = (case.file["instrs"].as_array().into_iter().flatten())
        .map(|c| c["line"].as_str().unwrap_or(""))
        .collect();
    let instrs = lines.read(&all[..=k])?;
    let i = &instrs[k];
    let regs = |it: &mut dyn Iterator<Item = &thin_ir::Reg>| -> Value {
        it.map(|r| json!([r.value, r.lo, r.n])).collect()
    };
    let names = |l: &[usize]| -> Value { l.iter().map(|&p| PHYS[p].name).collect() };
    let got = json!({
        "line": all[k], "roles": i.roles(), "defs": regs(&mut i.defs()), "uses": regs(&mut i.uses()),
        "writes": names(i.writes()), "reads": names(i.reads()), "sgpr_valu_def": i.sgpr_valu_def(),
        "writes_exec": i.writes_exec(), "m0_write": i.m0_write(),
    });
    expect_eq(got, case.value.clone())
}

/// `instr.json` `dpp_reads_old`: the truth table.
fn dpp_case(case: &Case<'_>) -> Result<(), String> {
    let fields = (case.value["fields"].as_object().ok_or("fields")?.iter())
        .map(|(k, v)| {
            let value = match v {
                Value::Bool(b) => FieldValue::Bool(*b),
                Value::String(t) => FieldValue::Text(t.as_str().into()),
                n => FieldValue::Int(n.as_i64().ok_or("an int")? as i128),
            };
            Ok((FieldId::named(k).map_err(|e| e.to_string())?, value))
        })
        .collect::<Result<Vec<_>, String>>()?;
    expect_eq(
        json!(dpp_reads_old(&fields)),
        case.value["reads_old"].clone(),
    )
}

/// `waves.json`: groups, classes and each class's successors.
fn waves_case(case: &Case<'_>) -> Result<(), String> {
    let s = snapshot(&strs(&case.file["lines"]))?;
    let s = s.with_waves(
        case.value["nwaves"].as_u64().ok_or("nwaves")? as u32,
        case.value["by_wave"] == true,
    );
    let groups: Vec<Value> = (s.wave_groups().into_iter())
        .map(|(waves, taken)| json!({"waves": waves, "taken": taken}))
        .collect();
    let classes = s.wave_classes();
    let succ: Vec<Value> = classes
        .iter()
        .map(|c| json!(s.cfg(c.as_deref()).succ))
        .collect();
    let got = json!({
        "nwaves": s.nwaves(), "by_wave": s.by_wave(), "wave_groups": groups,
        "wave_classes": classes, "class_cfg_succ": succ,
    });
    let mut want = case.value.clone();
    want.as_object_mut().unwrap().remove("name");
    expect_eq(got, want)
}

/// A set given as a JSON list of members.
fn bits_of(members: &[usize]) -> Bits {
    let mut b = Bits::new();
    b.add(members.iter().copied());
    b
}

/// The ints of a JSON list.
fn ints(v: &Value) -> Vec<usize> {
    v.as_array()
        .into_iter()
        .flatten()
        .filter_map(Value::as_u64)
        .map(|k| k as usize)
        .collect()
}

/// `sets.json` `bits`: the whole script on one running set, up to the case.
fn bits_case(case: &Case<'_>) -> Result<(), String> {
    let k: usize = case
        .id
        .trim_start_matches("bits[")
        .trim_end_matches(']')
        .parse()
        .map_err(|_| "an index")?;
    let mut a = Bits::new();
    let script = case.file["bits"].as_array().ok_or("bits")?;
    for step in &script[..k] {
        bits_step(&mut a, step);
    }
    let (members, extra) = bits_step(&mut a, case.value);
    let mut got = json!({"op": case.value["op"], "args": case.value["args"], "members": members});
    if let (Some(map), Some((key, value))) = (got.as_object_mut(), extra) {
        map.insert(key.into(), value);
    }
    expect_eq(got, case.value.clone())
}

/// Applies one step of the bits script to A; the members it shows and the
/// extra key it reports.
fn bits_step(a: &mut Bits, step: &Value) -> (Vec<usize>, Option<(&'static str, Value)>) {
    let args = ints(&step["args"]);
    let other = bits_of(&[65, 300]);
    match step["op"].as_str().unwrap_or("") {
        "set" => a.set(args[0]),
        "unset" => a.unset(args[0]),
        "set_range" => a.set_range(args[0], args[1]),
        "clear_range" => a.clear_range(args[0], args[1]),
        "add" => a.add(args),
        "remove" => a.remove(args),
        "and_range" => return (a.and_range(args[0], args[1]).ones(), None),
        "changes" => return (a.ones(), Some(("changes", json!(a.changes(&other))))),
        "or_minus" => {
            let mut e = Bits::new();
            e.or_minus(a, &other);
            return (e.ones(), None);
        }
        "assign_and" => {
            let mut e = bits_of(&[1000]);
            e.assign_and(a, &other);
            return (e.ones(), None);
        }
        "eq" => {
            let mut e = bits_of(&[1000]);
            e.unset(1000);
            return (Vec::new(), Some(("equal", json!(e == Bits::new()))));
        }
        "intersects" => {
            let e = bits_of(&[5, 1000]);
            return (e.ones(), Some(("result", json!(e.intersects(a)))));
        }
        "range_but" => {
            let sp = Sparse::range_but(60, 140, (64, 128));
            let mut all = Bits::new();
            all.set_range(0, 200);
            all.and_sparse(&sp);
            return (
                all.ones(),
                Some(("meets_{5}", json!(bits_of(&[5]).meets(&sp)))),
            );
        }
        _ => {}
    }
    (a.ones(), None)
}

/// A set of runs as `sets.json` prints it.
fn runs_text(r: &Runs) -> String {
    let runs: Vec<String> = r
        .runs()
        .iter()
        .map(|(a, b)| format!("({a}, {b})"))
        .collect();
    format!("Runs([{}])", runs.join(", "))
}

/// `sets.json` `runs`: the script on one set, up to the case.
fn runs_case(case: &Case<'_>) -> Result<(), String> {
    let k: usize = case
        .id
        .trim_start_matches("runs[")
        .trim_end_matches(']')
        .parse()
        .map_err(|_| "an index")?;
    let script = case.file["runs"].as_array().ok_or("runs")?;
    let mut history = vec![Runs::new()];
    for step in &script[..=k] {
        let op = step["op"].as_str().unwrap_or("");
        let mut r = history.last().cloned().unwrap_or_default();
        if let Some(args) = op.strip_prefix("push_run ") {
            let ab: Vec<u32> = args.split(' ').filter_map(|x| x.parse().ok()).collect();
            r.push_run(ab[0], ab[1]);
        } else if op == "add [6,9) to previous" {
            let mut o = Runs::new();
            o.push_run(6, 9);
            r.add(&o);
        }
        history.push(r);
    }
    let op = case.value["op"].as_str().unwrap_or("");
    let got = if op.starts_with("first_common") {
        let before = &history[history.len() - 3];
        let mut o = Runs::new();
        o.push_run(5, 10);
        json!({"op": op, "result": before.first_common(&o), "meets": before.meets(&o)})
    } else {
        json!({"op": op, "runs": runs_text(history.last().ok_or("a set")?)})
    };
    expect_eq(got, case.value.clone())
}

/// `pseudo.json`: the own ops as data.
fn pseudo_case(case: &Case<'_>) -> Result<(), String> {
    let k: usize = case
        .id
        .trim_start_matches("ops[")
        .trim_end_matches(']')
        .parse()
        .map_err(|_| "an index")?;
    let f = default_target().own_ops().get(k).ok_or("an own op")?;
    let slots: Vec<Value> = (f.slots.iter())
        .map(|s| json!({"kind": s.kind.name(), "role": s.role, "dwords": s.dwords}))
        .collect();
    let fields: Vec<Value> = (f.fields.iter())
        .map(|d| {
            json!({"name": d.name, "kind": d.kind.name(), "bits": d.bits, "signed": d.signed,
                   "required": d.required, "choices": d.choices})
        })
        .collect();
    let got = json!({
        "name": f.name, "id": f.id, "kind": f.kind.name(), "flags": f.flags.names(),
        "variadic": f.variadic, "slots": slots, "fields": fields,
    });
    expect_eq(got, case.value.clone())
}

fn main() {
    let mut suite = Suite::new("ir");
    suite.section("cfg.json", "cases", cfg_case);
    suite.section("solvers.json", "cases", solvers_case);
    suite.section("exec.json", "cases", exec_case);
    suite.section("flight.json", "cases", flight_case);
    suite.section("m0.json", "cases", m0_case);
    suite.section("instr.json", "instrs", instr_case);
    suite.section("instr.json", "dpp_reads_old", dpp_case);
    suite.section("waves.json", "cases", waves_case);
    suite.section("sets.json", "bits", bits_case);
    suite.section("sets.json", "runs", runs_case);
    suite.section("pseudo.json", "ops", pseudo_case);
    suite.finish();
}
