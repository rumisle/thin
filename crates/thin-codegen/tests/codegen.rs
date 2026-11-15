//! Golden checks for codegen ABI, manifests, metadata and wait validation.

use serde_json::json;

use thin_codegen::json::{self, Json};
use thin_codegen::{abi, manifest, meta, object};
use thin_formats::Value;
use thin_test_support::golden_tests::expect_eq;
use thin_test_support::{Case, Suite};

/// `verifier.json`: the independent wait checker's messages and records.
fn verifier_case(case: &Case<'_>) -> Result<(), String> {
    let text = case.value["text"].as_str().ok_or("no text")?;
    let got = thin_codegen::counter_waits(text);
    let messages: Vec<&String> = got.iter().map(|(_, m)| m).collect();
    expect_eq(json!(messages), case.value["counter_waits"].clone())?;
    let records: Vec<_> = got
        .iter()
        .map(|(r, _)| json!([r.0, r.1, r.2, r.3]))
        .collect();
    expect_eq(json!(records), case.value["counter_records"].clone())
}

/// `metadata.json`: the metadata reader's documents and refusals.
fn metadata_case(case: &Case<'_>) -> Result<(), String> {
    let v = case.value;
    let got = match (v.get("yaml"), v["s"].as_str()) {
        (Some(y), _) => {
            let lines: Vec<String> = y
                .as_array()
                .ok_or("no yaml")?
                .iter()
                .map(|l| l.as_str().unwrap_or("").to_owned())
                .collect();
            meta::read(&lines)
        }
        (None, Some(s)) => object::source(s).and_then(|f| object::metadata(&f)),
        (None, None) => return Err("no yaml or s".into()),
    };
    match (got, v.get("ok")) {
        (Ok(doc), Some(want)) => expect_eq(json_of(&doc), want.clone()),
        (Err(msg), None) => expect_eq(msg.as_str(), v["message"].as_str().unwrap_or("")),
        (Ok(doc), None) => Err(format!("accepted {:?}", json_of(&doc))),
        (Err(msg), Some(_)) => Err(format!("refused: {msg}")),
    }
}

/// A metadata value as JSON.
fn json_of(v: &Value) -> serde_json::Value {
    match v {
        Value::Nil => json!(null),
        Value::Bool(b) => json!(b),
        Value::Int(i) => json!(*i as i64),
        Value::Str(s) => json!(s),
        Value::Array(a) => json!(a.iter().map(json_of).collect::<Vec<_>>()),
        Value::Map(m) => {
            serde_json::Value::Object(m.iter().map(|(k, x)| (k.clone(), json_of(x))).collect())
        }
    }
}

/// A record as thin-codegen's JSON (via its text).
fn ours(v: &serde_json::Value) -> Result<Json, String> {
    json::parse(&v.to_string())
}

/// `manifest.json` `lines`: each line round trips through its record.
fn manifest_line(case: &Case<'_>) -> Result<(), String> {
    let want = case.value["manifest"].as_str().ok_or("no manifest")?;
    let text = want
        .strip_prefix(manifest::MARK)
        .ok_or("no mark")?
        .trim_end();
    let rec = manifest::record(json::parse(text)?)?;
    expect_eq(manifest::line(&rec).as_str(), want)
}

/// `manifest.json` `split`.
fn manifest_split(case: &Case<'_>) -> Result<(), String> {
    let c = case.value;
    let (body, rec) = manifest::split(c["text"].as_str().ok_or("no text")?)?;
    expect_eq(body.as_str(), c["body"].as_str().unwrap_or(""))?;
    let want = match &c["manifest"] {
        serde_json::Value::Null => None,
        m => Some(manifest::record(ours(m)?)?),
    };
    expect_eq(rec, want)
}

/// `abi.json` `residency`.
fn residency_case(case: &Case<'_>) -> Result<(), String> {
    let v = case.value;
    let n = |k: &str| v[k].as_u64().unwrap_or(0);
    let r = abi::residency(
        default_target(),
        n("lds"),
        n("vgpr") as u32,
        n("agpr") as u32,
        n("wgsize") as u32,
    );
    let l = &v["limits"];
    expect_eq(
        (
            u64::from(r.resident),
            r.limiter,
            r.limits.0.map(u64::from),
            r.limits.1.map(u64::from),
            u64::from(r.limits.2),
        ),
        (
            n("resident"),
            v["limiter"].as_str().unwrap_or(""),
            l["vgpr"].as_u64(),
            l["lds"].as_u64(),
            l["waves"].as_u64().unwrap_or(0),
        ),
    )
}

/// `abi.json` `budget`.
fn budget_case(case: &Case<'_>) -> Result<(), String> {
    let v = case.value;
    let n = |k: &str| v[k].as_u64().unwrap_or(0);
    let got = abi::check_budget(
        default_target(),
        "k",
        n("lds"),
        n("vgpr") as u32,
        n("agpr") as u32,
        n("sgpr") as u32,
        n("wgsize") as u32,
    );
    expect_eq(got.err(), v["message"].as_str().map(str::to_owned))
}

fn default_target() -> &'static thin_target::Target {
    thin_targets::get(thin_targets::default_name()).expect("registered default target")
}

fn main() {
    let mut suite = Suite::new("codegen");
    suite.section("verifier.json", "cases", verifier_case);
    suite.section("manifest.json", "lines", manifest_line);
    suite.section("manifest.json", "split", manifest_split);
    suite.section("metadata.json", "cases", metadata_case);
    suite.section("abi.json", "residency", residency_case);
    suite.section("abi.json", "budget", budget_case);
    suite.finish();
}
