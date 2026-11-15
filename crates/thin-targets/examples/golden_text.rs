//! Evaluate target-text golden requests through gfx1201's native syntax,
//! `.s` reader, assembler and hazard pass. Reads a JSON request list from
//! stdin and writes one result list to stdout. Each result is serialized
//! the way `tests/target.rs` reads `read_s.json`, `line_syntax.json` and
//! `hazards.json`; no expected value is an input.

use std::io::Read;

use serde_json::{json, Value};
use thin_target::{
    assemble, parse_line, parse_operand, read_s, CodeItem, ModValue, Operand, Target,
    Value as OperandValue, PHYS,
};

fn target() -> &'static Target {
    thin_targets::get(thin_targets::default_name()).expect("the default target")
}

fn text(v: &Value) -> &str {
    v.as_str().expect("a text request")
}

/// An operand as `line_syntax.json` writes it.
fn operand_json(o: &Operand) -> Value {
    let value = match &o.value {
        None => Value::Null,
        Some(OperandValue::Int(v)) => json!({"int": v.to_string()}),
        Some(OperandValue::Float(f)) => json!({"float": f}),
        Some(OperandValue::Label(l)) => json!({"label": l}),
    };
    json!({"kind": o.kind.name(), "n": o.n, "cnt": o.count, "neg": o.neg, "abs": o.abs,
        "val": value, "text": o.text})
}

fn classify(s: &str) -> Value {
    let reg = thin_target::reg_text(s).map(|(f, a, b)| json!([f.to_string(), a, b]));
    json!({"text": s, "int_text": thin_target::int_text(s),
        "float_text": thin_target::float_text(s), "label_text": thin_target::label_text(s),
        "reg_text": reg})
}

fn operand(t: &Target, s: &str) -> Value {
    match parse_operand(t.hooks(), s) {
        Ok(o) => json!({"text": s, "ok": operand_json(&o)}),
        Err(e) => json!({"text": s, "refused": e}),
    }
}

fn line(t: &Target, s: &str) -> Value {
    match parse_line(t.hooks(), s) {
        Ok(l) => {
            let modifiers: Vec<Value> = (l.modifiers.iter())
                .map(|(k, m)| match m {
                    ModValue::Int(i) => json!([k, i.to_string()]),
                    ModValue::List(v) => {
                        json!([k, v.iter().map(|i| i.to_string()).collect::<Vec<_>>()])
                    }
                })
                .collect();
            let operands: Vec<Value> = l.operands.iter().map(operand_json).collect();
            json!({"line": s, "mnemonic": l.mnemonic, "operands": operands,
                "modifiers": modifiers})
        }
        Err(e) => json!({"line": s, "refused": e}),
    }
}

fn ins(text: &str, line: usize) -> CodeItem {
    CodeItem::Ins {
        text: text.into(),
        comment: String::new(),
        line,
    }
}

/// Code items exactly as `tests/target.rs` builds them from a case: listed
/// items at lines 1.., or the far branch its description names.
fn items_of(case: &Value) -> Vec<CodeItem> {
    if let Some(items) = case["items"].as_array() {
        return (items.iter().enumerate())
            .flat_map(|(i, item)| {
                if item[0].as_str() == Some("repeat") {
                    let n = item[1].as_u64().expect("a repeat count") as usize;
                    return (0..n)
                        .map(|_| ins(text(&item[2]), i + 1))
                        .collect::<Vec<_>>();
                }
                vec![match item[0].as_str() {
                    Some("label") => CodeItem::Label {
                        name: text(&item[1]).into(),
                        line: i + 1,
                    },
                    Some("align") => CodeItem::Align {
                        p: item[1].as_u64().expect("a power") as u8,
                        line: i + 1,
                    },
                    _ => ins(text(&item[1]), i + 1),
                }]
            })
            .collect();
    }
    let described = text(&case["described"]);
    let n: usize = (described.split_whitespace())
        .find_map(|w| w.parse().ok())
        .expect("a count");
    let mut items = vec![ins("s_branch .Lfar", 1)];
    items.extend((0..n).map(|i| ins("s_nop 0", i + 2)));
    items.push(CodeItem::Label {
        name: ".Lfar".into(),
        line: n + 2,
    });
    if described.ends_with("s_endpgm") {
        items.push(ins("s_endpgm", n + 3));
    }
    items
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// The assembly of a case; with `summary`, its length and first word only.
fn assembled(t: &Target, case: &Value) -> Value {
    let mut out = case.clone();
    let fields = out.as_object_mut().expect("an assemble case");
    fields.remove("summary");
    match assemble(&items_of(case), t.hooks()) {
        Err(e) => _ = fields.insert("refused".into(), json!(e.to_string())),
        Ok(bytes) if case["summary"].as_bool() == Some(true) => {
            fields.insert("first_word_hex".into(), json!(hex(&bytes[..4])));
            fields.insert("bytes".into(), json!(bytes.len()));
        }
        Ok(bytes) => _ = fields.insert("hex".into(), json!(hex(&bytes))),
    }
    out
}

fn read(s: &str) -> Value {
    let file = match read_s(s, false) {
        Ok(f) => f,
        Err(e) => return json!({"text": s, "refused": e.to_string()}),
    };
    let code: Vec<Value> = (file.code.iter())
        .map(|item| match item {
            CodeItem::Ins { text, line, .. } => json!(["ins", text, line]),
            CodeItem::Label { name, line } => json!(["label", name, line]),
            CodeItem::Align { p, line } => json!(["align", p, line]),
            CodeItem::Note { text, line } => json!(["note", text, line]),
        })
        .collect();
    let pairs = |m: &std::collections::BTreeMap<String, (String, usize)>| -> Value {
        m.iter()
            .map(|(k, (v, l))| (k.clone(), json!([v, l])))
            .collect::<serde_json::Map<_, _>>()
            .into()
    };
    json!({"text": s, "code": code, "once": pairs(&file.once), "block": pairs(&file.block),
        "meta": file.meta})
}

/// A need as `hazards.json` writes it.
fn need_json(t: &Target, n: &thin_passes::Need) -> Value {
    let fields = t.wait_fields();
    let waits: Vec<String> = (n.waits.0.iter().zip(&fields))
        .filter_map(|(c, f)| c.map(|c| format!("{f}({c})")))
        .collect();
    let valu = match n.valu {
        thin_passes::Valu::None => Value::Null,
        thin_passes::Valu::Nop => json!("v_nop"),
        thin_passes::Valu::Copy(k) => json!({"copy": k}),
    };
    let text = (n.delay != 0).then(|| thin_passes::delay_text(n.delay));
    json!({"states": n.states, "waits": waits, "valu": valu, "delay": n.delay, "delay_text": text})
}

fn hazards(t: &'static Target, program: &Value) -> Value {
    let lines: Vec<&str> = (program.as_array().expect("lines").iter())
        .map(text)
        .collect();
    let instrs = thin_test_support::asm::read(t, &lines).expect("the program's lines");
    let s = thin_test_support::asm::snapshot(t, instrs);
    let model = thin_targets::hazards(t.name()).expect("the target's hazard model");
    let needs: Vec<Value> = (thin_passes::hazards::needs(&s, model).iter())
        .map(|n| need_json(t, n))
        .collect();
    json!({"lines": lines, "needs": needs})
}

/// The syntax facts generators build inputs from: register counts, named
/// registers, hwreg ids, the counted wait's names, modifier symbols.
fn vocab(t: &Target) -> Value {
    let h = t.hooks();
    let specials: Vec<Value> = (PHYS.iter())
        .filter_map(|p| h.special(p.name).map(|(n, w)| json!([p.name, n, w])))
        .collect();
    let symbols = |field: &str| -> Vec<String> {
        let mut found: Vec<String> = (t.forms().iter())
            .filter_map(|f| f.field(field))
            .flat_map(|x| x.choices.iter().cloned())
            .collect();
        found.sort();
        found.dedup();
        found
    };
    let mut flags: Vec<&str> = (t.forms().iter())
        .flat_map(|f| f.fields.iter().map(|x| x.name))
        .filter(|n| h.mod_flag(n))
        .collect();
    flags.sort_unstable();
    flags.dedup();
    let counted = |m: &str| h.counted(m).map(|c| json!(c));
    json!({
        "reg_count": {"v": h.reg_count('v'), "s": h.reg_count('s'), "a": h.reg_count('a')},
        "specials": specials,
        "hwreg": h.hwreg(),
        "counted": {"s_wait_alu": counted("s_wait_alu")},
        "th": symbols("th"),
        "scope": (symbols("scope").iter())
            .filter(|s| h.symbol("scope", s).is_some())
            .collect::<Vec<_>>(),
        "flags": flags,
        "wait_fields": t.wait_fields(),
        "nop_word": format!("{:08x}", h.nop_word()),
    })
}

fn main() {
    let mut input = String::new();
    std::io::stdin().read_to_string(&mut input).expect("stdin");
    let requests: Vec<Value> = serde_json::from_str(&input).expect("a JSON request list");
    let t = target();
    let results: Vec<Value> = (requests.iter())
        .map(|r| {
            let arg = &r["arg"];
            match r["op"].as_str().expect("an op") {
                "vocab" => vocab(t),
                "int" => json!([
                    text(arg),
                    thin_target::int_value(text(arg)).map(|i| i.to_string())
                ]),
                "classify" => classify(text(arg)),
                "operand" => operand(t, text(arg)),
                "line" => line(t, text(arg)),
                "assemble" => assembled(t, arg),
                "read_s" => read(text(arg)),
                "hazards" => hazards(t, arg),
                "delay_text" => {
                    let d = arg.as_u64().expect("a delay") as u32;
                    json!([d, thin_passes::delay_text(d)])
                }
                other => panic!("unknown op {other}"),
            }
        })
        .collect();
    println!("{}", serde_json::to_string(&results).expect("JSON"));
}
