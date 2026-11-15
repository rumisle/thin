//! Evaluate golden text-reader requests through the emulator's own reader and
//! claim reader. Reads a JSON request list from stdin, one result per request
//! on stdout. A request is `{"reading": text}` (the program as `reading.json`
//! records it, or its reading error) or `{"claims": text}` (each claim kind
//! as `claims.json` records it). Rows use the golden tests' own adapters.

use std::io::Read;

use serde_json::{json, Map, Value};
use thin_emu::machine::claims::{read, Claims, Kind};
use thin_emu::machine::facts::Facts;
use thin_test_support::adapters::{default_target, reading_row};

fn reading(text: &str) -> Value {
    let program = match thin_emu::read_program(text, default_target()) {
        Ok(p) => p,
        Err(e) => return json!({ "error": e.message }),
    };
    let labels: Map<String, Value> = (program.labels.iter())
        .map(|(name, i)| (name.clone(), Value::from(*i)))
        .collect();
    let sections: Vec<Value> = (program.sections.iter())
        .map(|(i, t, l)| json!([i, t, l]))
        .collect();
    json!({
        "ins": program.ins.iter().map(reading_row).collect::<Vec<_>>(),
        "labels": labels,
        "sections": sections,
        "user_sgprs": program.user_sgprs,
    })
}

fn claims(text: &str) -> Value {
    let program = match thin_emu::read_program(text, default_target()) {
        Ok(p) => p,
        Err(e) => return json!({ "reading_error": e.message }),
    };
    let facts = Facts::of(default_target());
    let reg = |r: &(String, u32, u32)| json!([r.0, r.1, r.2]);
    let mut out = Map::new();
    for (key, kind) in [
        ("uniform", Kind::Uniform),
        ("assume", Kind::Assume),
        ("undefined", Kind::Undefined),
    ] {
        let mut found = Claims::default();
        let got = match read(&program, &facts, kind, &mut found) {
            Err(e) => json!({ "error": e.message }),
            Ok(()) => {
                let mut m = Map::new();
                for (i, u) in &found.uniform {
                    m.insert(
                        i.to_string(),
                        json!([u.regs.iter().map(reg).collect::<Vec<_>>(), u.text]),
                    );
                }
                for (i, rows) in &found.assume {
                    let rows: Vec<Value> = (rows.iter())
                        .map(|a| json!([a.reg.0, a.reg.1, a.reg.2, a.lo, a.hi, a.text]))
                        .collect();
                    m.insert(i.to_string(), Value::Array(rows));
                }
                for (i, regs) in &found.undefined {
                    m.insert(i.to_string(), regs.iter().map(reg).collect());
                }
                for ((a, b), regs) in &found.edges {
                    m.insert(format!("edge:{a}->{b}"), regs.iter().map(reg).collect());
                }
                Value::Object(m)
            }
        };
        out.insert(key.to_owned(), got);
    }
    out.insert("user_sgprs".to_owned(), Value::from(program.user_sgprs));
    Value::Object(out)
}

fn main() {
    let mut input = String::new();
    std::io::stdin()
        .read_to_string(&mut input)
        .expect("stdin is readable");
    let requests: Vec<Value> = serde_json::from_str(&input).expect("a JSON request list");
    let results: Vec<Value> = (requests.iter())
        .map(|r| {
            if let Some(text) = r.get("reading").and_then(Value::as_str) {
                reading(text)
            } else if let Some(text) = r.get("claims").and_then(Value::as_str) {
                claims(text)
            } else {
                panic!("unknown request {r}")
            }
        })
        .collect();
    println!("{}", Value::Array(results));
}
