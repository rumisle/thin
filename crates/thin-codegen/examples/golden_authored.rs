//! Evaluate the codegen golden generator's authored requests with the
//! native core: the wait checker, manifests, the metadata reader, residency
//! and the budget check. Reads a JSON request list from stdin and writes one
//! result list to stdout; it never opens a golden file.

use std::io::Read;

use serde_json::{json, Value};
use thin_codegen::json::{self, Json};
use thin_codegen::{abi, manifest, meta, object};

/// A metadata value as JSON.
fn json_of(v: &thin_formats::Value) -> Value {
    use thin_formats::Value as V;
    match v {
        V::Nil => json!(null),
        V::Bool(b) => json!(b),
        V::Int(i) => json!(*i as i64),
        V::Str(s) => json!(s),
        V::Array(a) => json!(a.iter().map(json_of).collect::<Vec<_>>()),
        V::Map(m) => Value::Object(m.iter().map(|(k, x)| (k.clone(), json_of(x))).collect()),
    }
}

/// A record as the manifest line's JSON, parsed back for the golden file.
fn line_json(rec: &Json) -> Result<Value, String> {
    let line = manifest::line(rec);
    let text = line
        .strip_prefix(manifest::MARK)
        .ok_or("no mark")?
        .trim_end();
    serde_json::from_str(text).map_err(|e| e.to_string())
}

fn target() -> &'static thin_target::Target {
    thin_targets::get(thin_targets::default_name()).expect("registered default target")
}

fn n(v: &Value, k: &str) -> u64 {
    v[k].as_u64().unwrap_or_else(|| panic!("request field {k}"))
}

fn outcome(got: Result<Value, String>) -> Value {
    match got {
        Ok(v) => json!({ "ok": v }),
        Err(m) => json!({ "message": m }),
    }
}

fn evaluate(req: &Value) -> Value {
    match req["op"].as_str().expect("op") {
        "verifier" => {
            let got = thin_codegen::counter_waits(req["text"].as_str().expect("text"));
            let messages: Vec<&String> = got.iter().map(|(_, m)| m).collect();
            let records: Vec<Value> = got
                .iter()
                .map(|(r, _)| json!([r.0, r.1, r.2, r.3]))
                .collect();
            json!({ "counter_waits": messages, "counter_records": records })
        }
        "manifest_line" => outcome(
            json::parse(&req["record"].to_string())
                .and_then(manifest::record)
                .map(|rec| json!(manifest::line(&rec))),
        ),
        "manifest_split" => outcome(
            manifest::split(req["text"].as_str().expect("text")).and_then(|(body, rec)| {
                let rec = rec.as_ref().map(line_json).transpose()?;
                Ok(json!({ "body": body, "manifest": rec }))
            }),
        ),
        "metadata_yaml" => {
            let lines: Vec<String> = req["yaml"]
                .as_array()
                .expect("yaml")
                .iter()
                .map(|l| l.as_str().expect("a line").to_owned())
                .collect();
            outcome(meta::read(&lines).map(|v| json_of(&v)))
        }
        "metadata_s" => outcome(
            object::source(req["s"].as_str().expect("s"))
                .and_then(|f| object::metadata(&f))
                .map(|v| json_of(&v)),
        ),
        "residency" => {
            let r = abi::residency(
                target(),
                n(req, "lds"),
                n(req, "vgpr") as u32,
                n(req, "agpr") as u32,
                n(req, "wgsize") as u32,
            );
            json!({
                "resident": r.resident,
                "limiter": r.limiter,
                "limits": {"vgpr": r.limits.0, "lds": r.limits.1, "waves": r.limits.2},
            })
        }
        "budget" => {
            let got = abi::check_budget(
                target(),
                req["name"].as_str().expect("name"),
                n(req, "lds"),
                n(req, "vgpr") as u32,
                n(req, "agpr") as u32,
                n(req, "sgpr") as u32,
                n(req, "wgsize") as u32,
            );
            json!({ "message": got.err() })
        }
        other => panic!("unknown request {other}"),
    }
}

fn main() {
    let mut input = String::new();
    std::io::stdin()
        .read_to_string(&mut input)
        .expect("requests on stdin");
    let requests: Vec<Value> = serde_json::from_str(&input).expect("a JSON request list");
    let results: Vec<Value> = requests.iter().map(evaluate).collect();
    println!("{}", Value::Array(results));
}
