//! Loading the checked-in golden.

use std::io::Read;
use std::path::PathBuf;

use serde_json::Value;

/// The repository root.
pub fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// Locate a golden file in its owning crate's tests/data directory.
pub fn path(component: &str, file: &str) -> PathBuf {
    let owner = match component {
        "emu-tags" => "emu",
        "target" => "targets",
        name => name,
    };
    repo_root()
        .join(format!("crates/thin-{owner}/tests/data"))
        .join(component)
        .join(file)
}

/// The raw bytes of a golden file.
pub fn bytes(component: &str, file: &str) -> Vec<u8> {
    let path = path(component, file);
    std::fs::read(&path).unwrap_or_else(|e| panic!("reading {}: {e}", path.display()))
}

/// A JSON golden file, gunzipped when its name ends in `.gz`.
pub fn json(component: &str, file: &str) -> Value {
    let raw = bytes(component, file);
    let text = if file.ends_with(".gz") {
        let mut text = String::new();
        flate2::read::GzDecoder::new(raw.as_slice())
            .read_to_string(&mut text)
            .unwrap_or_else(|e| panic!("gunzipping {component}/{file}: {e}"));
        text
    } else {
        String::from_utf8(raw).unwrap_or_else(|e| panic!("{component}/{file} is not UTF-8: {e}"))
    };
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("parsing {component}/{file}: {e}"))
}

/// Parses a hex string (`"0x3f80"` or `"3f80"`) as an integer.
pub fn hex(text: &str) -> u64 {
    let digits = text.trim_start_matches("0x");
    u64::from_str_radix(digits, 16).unwrap_or_else(|e| panic!("{text:?} is not hex: {e}"))
}

/// Parses a hex string of bytes.
pub fn hex_bytes(text: &str) -> Vec<u8> {
    (0..text.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&text[i..i + 2], 16).expect("hex byte"))
        .collect()
}

/// The section name and key of a case id `section[key]`.
pub fn split_id(id: &str) -> (&str, &str) {
    let open = id
        .find('[')
        .unwrap_or_else(|| panic!("case id {id:?} has no key"));
    (&id[..open], id[open + 1..].trim_end_matches(']'))
}

/// A JSON object's entries in key order.
fn sorted(map: &serde_json::Map<String, Value>) -> impl Iterator<Item = (&String, &Value)> {
    let mut entries: Vec<_> = map.iter().collect();
    entries.sort_by(|a, b| a.0.cmp(b.0));
    entries.into_iter()
}

/// The msgpack value a JSON value stands for, `None` for one with a float
/// (which the msgpack subset does not have).
pub fn msgpack_value(json: &serde_json::Value) -> Option<thin_formats::Value> {
    use thin_formats::Value as M;
    Some(match json {
        Value::Null => M::Nil,
        Value::Bool(b) => M::Bool(*b),
        Value::Number(n) => M::Int(
            n.as_i64()
                .map(i128::from)
                .or_else(|| n.as_u64().map(i128::from))
                .or_else(|| n.as_str().parse().ok())?,
        ),
        Value::String(s) => M::Str(s.clone()),
        Value::Array(items) => M::Array(items.iter().map(msgpack_value).collect::<Option<_>>()?),
        // Keys sorted: a JSON object compares regardless of key order.
        Value::Object(map) => M::Map(
            sorted(map)
                .map(|(k, v)| Some((k.clone(), msgpack_value(v)?)))
                .collect::<Option<_>>()?,
        ),
    })
}

/// The msgpack value a JSON value stands for.
pub fn msgpack_of(json: &serde_json::Value) -> thin_formats::Value {
    use thin_formats::Value as M;
    match json {
        Value::Null => M::Nil,
        Value::Bool(b) => M::Bool(*b),
        Value::Number(n) => M::Int(
            n.as_i64()
                .map(i128::from)
                .unwrap_or_else(|| i128::from(n.as_u64().expect("an integer msgpack value"))),
        ),
        Value::String(s) => M::Str(s.clone()),
        Value::Array(items) => M::Array(items.iter().map(msgpack_of).collect()),
        Value::Object(map) => M::Map(
            sorted(map)
                .map(|(k, v)| (k.clone(), msgpack_of(v)))
                .collect(),
        ),
    }
}
