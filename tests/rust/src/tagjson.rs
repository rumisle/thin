//! The emu-tags golden' tag notation: JSON to tags,
//! made through one set of tables so equal records are equal tags.

use serde_json::Value;
use thin_emu::tags::{mem_tag, Cols, KSet, Record, Tables, Tag};

fn ints(v: &Value) -> Vec<i64> {
    v.as_array()
        .map(|a| a.iter().filter_map(Value::as_i64).collect())
        .unwrap_or_default()
}

/// A record from the notation's `rec` object.
pub fn record(r: &Value) -> Record {
    let n = match &r["n"] {
        Value::Array(_) => Cols::Many(ints(&r["n"])),
        v => Cols::One(v.as_i64().unwrap_or(0)),
    };
    Record {
        m: r["m"].as_i64().unwrap_or(0),
        n,
        k: KSet::Set(ints(&r["k"]).into_iter().collect()),
        stat: r["stat"].as_bool().unwrap_or(false),
        pid: r["pid"].as_i64().unwrap_or(0),
    }
}

/// The tag a notation value names, made in `t`.
pub fn tag(t: &mut Tables, v: &Value) -> Tag {
    match v {
        Value::Number(n) => n.as_i64().unwrap_or(0),
        Value::String(s) if s == "acc_mark" => -1,
        Value::Object(o) => {
            if let Some(m) = o.get("mem") {
                let parts = ints(m);
                mem_tag(parts[0] as usize, parts[1] as u64)
            } else if let Some(r) = o.get("rec") {
                t.record(&record(r))
            } else if let Some(p) = o.get("poison") {
                t.poison(p.as_str().unwrap_or(""))
            } else if let Some(s) = o.get("sum") {
                let (x, y) = (tag(t, &s[0]), tag(t, &s[1]));
                t.sum(x, y)
            } else {
                0
            }
        }
        _ => 0,
    }
}

/// Four byte tags of one lane.
pub fn lane(t: &mut Tables, v: &Value) -> [Tag; 4] {
    let mut out = [0; 4];
    for (i, b) in v.as_array().into_iter().flatten().take(4).enumerate() {
        out[i] = tag(t, b);
    }
    out
}

/// A tag in the notation, for messages.
pub fn show(t: &Tables, tag: Tag) -> Value {
    use thin_emu::tags::{is_record, is_sum, tag_buf, tag_off};
    if tag == 0 {
        return Value::from(0);
    }
    if tag > 0 {
        return serde_json::json!({"mem": [tag_buf(tag), tag_off(tag)]});
    }
    if is_sum(tag) {
        let (x, y) = t.sum_of(tag);
        return serde_json::json!({"sum": [show(t, x), show(t, y)]});
    }
    if is_record(tag) {
        let r = t.record_of(tag);
        if r.is_poison() {
            return serde_json::json!({"poison": r.reason()});
        }
        return serde_json::json!({"rec": format!("{r:?}")});
    }
    Value::from(tag)
}
