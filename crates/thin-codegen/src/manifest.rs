//! The manifest: its record, its line and the split.

use std::collections::BTreeMap;

use crate::json::{parse, Json};

/// The line's mark.
pub const MARK: &str = "; thin.manifest ";

/// The record's fields and their defaults (empty when optional).
fn defaults() -> BTreeMap<String, Json> {
    let int = Json::Int(0);
    let list = Json::List(Vec::new());
    let map = Json::Map(BTreeMap::new());
    [
        ("symbol", Json::Str(String::new())),
        ("kernarg_bytes", int.clone()),
        ("lds", int.clone()),
        ("wgsize", int.clone()),
        ("wgy", int.clone()),
        ("vgpr", int.clone()),
        ("agpr", int.clone()),
        ("sgpr", int.clone()),
        ("args", list.clone()),
        ("build", map.clone()),
        ("resident", int),
        ("mix", map.clone()),
        ("notes", list.clone()),
        ("grid", list.clone()),
        ("phases", map.clone()),
        ("max_grid", list),
        ("needs", map),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_owned(), v))
    .collect()
}

/// The fields the line leaves out when empty.
const OPTIONAL: [&str; 3] = ["phases", "max_grid", "needs"];

/// An argument's defaults, left out of the record.
fn arg_default(key: &str, v: &Json) -> bool {
    match key {
        "lo" | "hi" | "owned" | "library" | "mem" | "ranks" => *v == Json::Null,
        "divisor" => *v == Json::Bool(false),
        "multiple" => *v == Json::Int(1),
        _ => false,
    }
}

/// An argument record without its defaults.
pub fn arg(a: &Json) -> Json {
    match a {
        Json::Map(m) => Json::Map(
            m.iter()
                .filter(|(k, v)| !arg_default(k, v))
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect(),
        ),
        other => other.clone(),
    }
}

/// The whole record of a JSON object: unknown keys refused, missing
/// fields at their defaults, arguments without theirs.
pub fn record(j: Json) -> Result<Json, String> {
    let Json::Map(given) = j else {
        return Err("a manifest is a JSON object".into());
    };
    let mut out = defaults();
    for (k, v) in given {
        if !out.contains_key(&k) {
            return Err(format!("manifest: unknown key {k:?}"));
        }
        out.insert(k, v);
    }
    if let Some(Json::List(args)) = out.get("args") {
        let args = Json::List(args.iter().map(arg).collect());
        out.insert("args".into(), args);
    }
    Ok(Json::Map(out))
}

/// The manifest line of a record.
pub fn line(record: &Json) -> String {
    let mut m = match record {
        Json::Map(m) => m.clone(),
        _ => BTreeMap::new(),
    };
    m.retain(|k, v| !(OPTIONAL.contains(&k.as_str()) && v.is_empty()));
    format!("{MARK}{}\n", Json::Map(m).dumps())
}

/// `split_manifest(text)`: the body and the record, if the last line is
/// a manifest.
pub fn split(text: &str) -> Result<(String, Option<Json>), String> {
    let trimmed = text.trim_end_matches('\n');
    let (head, last) = match trimmed.rfind('\n') {
        Some(k) => (&trimmed[..k], &trimmed[k + 1..]),
        None => ("", trimmed),
    };
    let Some(json) = last.strip_prefix(MARK) else {
        return Ok((text.to_owned(), None));
    };
    Ok((format!("{head}\n"), Some(record(parse(json)?)?)))
}
