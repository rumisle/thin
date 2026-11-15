//! Golden checks for target facts, instruction encoding and assembly syntax.

use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use thin_target::{
    assemble, descriptor, parse_line, parse_operand, read_s, Block, CodeItem, Encoded, Footer,
    Form, ModValue, Operand, Target, Value as OperandValue, PHYS,
};
use thin_test_support::adapters::default_target;
use thin_test_support::golden::{self, hex_bytes};
use thin_test_support::golden_tests::expect_eq;
use thin_test_support::{Case, Suite};

/// `tables_manifest.json`: the checked-in generated tables hash as pinned.
fn tables_manifest(case: &Case<'_>) -> Result<(), String> {
    let data = golden::repo_root().join("crates/thin-targets/data/gfx1201");
    for (file, key) in [
        ("isa_tables.json", "isa_tables"),
        ("operand_table.json", "operand_table"),
    ] {
        let bytes = std::fs::read(data.join(file))
            .map_err(|e| format!("{file}: {e} (run tools/isagen)"))?;
        let want = case.value[key]["sha256_canonical_json"]
            .as_str()
            .ok_or("no sha256")?;
        expect_eq(format!("{:x}", Sha256::digest(&bytes)), want.to_owned())?;
    }
    Ok(())
}

/// A checked-in table equals its vector.
fn same_table(case: &Case<'_>, data: &str) -> Result<(), String> {
    let path = golden::repo_root()
        .join("crates/thin-targets/data/gfx1201")
        .join(data);
    let text = std::fs::read_to_string(&path).map_err(|e| format!("{data}: {e}"))?;
    let checked_in: Value = serde_json::from_str(&text).map_err(|e| e.to_string())?;
    expect_eq(&checked_in, case.value)
}

/// `isa_tables.json.gz`: the checked-in ISA tables.
fn isa_tables(case: &Case<'_>) -> Result<(), String> {
    same_table(case, "isa_tables.json")
}

/// `operand_table.json.gz`: the checked-in operand table.
fn operand_table(case: &Case<'_>) -> Result<(), String> {
    same_table(case, "operand_table.json")
}

/// A form as `forms.json.gz` writes it.
fn form_json(t: &Target, f: &Form) -> Value {
    let names = |regs: &[usize]| regs.iter().map(|&r| PHYS[r].name).collect::<Vec<_>>();
    let slots: Vec<Value> = (f.slots.iter())
        .map(|s| {
            json!({
                "kind": s.kind.name(), "dwords": s.dwords, "role": s.role, "type": s.xml_type,
                "sgpr_widths": s.sgpr_widths, "modifiers": s.modifiers, "literal": s.literal,
                "widths": s.widths,
            })
        })
        .collect();
    let fields: Vec<Value> = (f.fields.iter())
        .map(|x| {
            json!({
                "name": x.name, "kind": x.kind.name(), "bits": x.bits, "signed": x.signed,
                "required": x.required, "choices": x.choices,
            })
        })
        .collect();
    let counters = &t.definition().counters;
    json!({
        "id": f.id, "name": f.name, "op": f.op, "base": f.base, "canon": f.canon,
        "kind": f.kind.name(), "counter": f.counter.map(|c| counters[c].name),
        "unordered": f.unordered, "flags": f.flags.names(), "encoding": f.encoding,
        "arity": f.arity, "returning": f.returning, "slots": slots, "variadic": f.variadic,
        "fields": fields, "implicit_defs": names(&f.implicit_defs),
        "implicit_uses": names(&f.implicit_uses),
        "mem_bytes": f.mem_bytes, "pseudo": f.pseudo.map_or("None", |p| p.name()),
        "dpp": f.dpp, "dst_sel": f.dst_sel.name(), "mem": f.mem.map(|m| m.name()),
        "wait_text": f.wait_text.map(|w| w.name()),
    })
}

/// `forms.json.gz`: every form in id order, then the own ops.
fn form_case(case: &Case<'_>) -> Result<(), String> {
    let t = default_target();
    let id = case.value["id"].as_i64().ok_or("no id")? as i32;
    let form = t.form_id(id).ok_or("no such form")?;
    expect_eq(&form_json(t, form), case.value)
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

fn line_syntax(case: &Case<'_>) -> Result<(), String> {
    let (section, _) = golden::split_id(&case.id);
    let t = default_target();
    let v = case.value;
    match section {
        "py_int" => expect_eq(
            thin_target::int_value(v[0].as_str().ok_or("no text")?).map(|i| i.to_string()),
            v[1].as_str().map(str::to_owned),
        ),
        "classify" => {
            let text = v["text"].as_str().ok_or("no text")?;
            let reg = thin_target::reg_text(text).map(|(f, a, b)| json!([f.to_string(), a, b]));
            expect_eq(
                json!({"text": text, "int_text": thin_target::int_text(text),
                    "float_text": thin_target::float_text(text),
                    "label_text": thin_target::label_text(text), "reg_text": reg}),
                v.clone(),
            )
        }
        "operands" => {
            let got = parse_operand(t.hooks(), v["text"].as_str().ok_or("no text")?);
            match (v.get("ok"), v["refused"].as_str()) {
                (Some(want), _) => expect_eq(&operand_json(&got?), want),
                (None, Some(message)) => expect_eq(got.err().as_deref(), Some(message)),
                (None, None) => expect_eq(got.is_err(), true),
            }
        }
        "lines" => {
            let got = parse_line(t.hooks(), v["line"].as_str().ok_or("no line")?);
            if let Some(refused) = v.get("refused") {
                return match refused.as_str() {
                    Some(message) => expect_eq(got.err().as_deref(), Some(message)),
                    None => expect_eq(got.is_err(), true),
                };
            }
            let line = got?;
            let modifiers: Vec<Value> = (line.modifiers.iter())
                .map(|(k, m)| match m {
                    ModValue::Int(i) => json!([k, i.to_string()]),
                    ModValue::List(l) => {
                        json!([k, l.iter().map(|i| i.to_string()).collect::<Vec<_>>()])
                    }
                })
                .collect();
            let operands: Vec<Value> = line.operands.iter().map(operand_json).collect();
            expect_eq(line.mnemonic.as_str(), v["mnemonic"].as_str().unwrap_or(""))?;
            expect_eq(&json!(operands), v.get("operands").unwrap_or(&json!([])))?;
            expect_eq(json!(modifiers), v["modifiers"].clone())
        }
        "assemble" => assemble_case(t, v),
        other => Err(format!("unknown section {other}")),
    }
}

/// `line_syntax.json` `assemble`: code items to bytes, and the two far
/// branches it describes.
fn assemble_case(t: &Target, v: &Value) -> Result<(), String> {
    let items: Vec<CodeItem> = match v["items"].as_array() {
        Some(items) => items
            .iter()
            .enumerate()
            .flat_map(|(i, item)| {
                let line = i + 1;
                if item[0].as_str() == Some("repeat") {
                    let n = item[1].as_u64().expect("a repeat count") as usize;
                    let text = item[2].as_str().expect("the repeated instruction");
                    return (0..n)
                        .map(|_| CodeItem::Ins {
                            text: text.into(),
                            comment: String::new(),
                            line,
                        })
                        .collect::<Vec<_>>();
                }
                vec![match item[0].as_str() {
                    Some("label") => CodeItem::Label {
                        name: item[1].as_str().unwrap_or("").into(),
                        line,
                    },
                    Some("align") => CodeItem::Align {
                        p: item[1].as_u64().unwrap_or(0) as u8,
                        line,
                    },
                    _ => CodeItem::Ins {
                        text: item[1].as_str().unwrap_or("").into(),
                        comment: String::new(),
                        line,
                    },
                }]
            })
            .collect(),
        None => far_branch(v["described"].as_str().ok_or("no items")?)?,
    };
    let got = assemble(&items, t.hooks());
    if let Some(refused) = v["refused"].as_str() {
        return expect_eq(got.err().map(|e| e.to_string()), Some(refused.to_owned()));
    }
    let bytes = got.map_err(|e| e.to_string())?;
    if let Some(first) = v["first_word_hex"].as_str() {
        expect_eq(bytes.len() as u64, v["bytes"].as_u64().unwrap_or(0))?;
        return expect_eq(hex(&bytes[..4]), first.to_owned());
    }
    expect_eq(hex(&bytes), v["hex"].as_str().unwrap_or("").to_owned())
}

/// The program a far-branch case describes: `s_branch` then n `s_nop 0`
/// then `.Lfar:`, optionally with an `s_endpgm`.
fn far_branch(described: &str) -> Result<Vec<CodeItem>, String> {
    let n: usize = described
        .split_whitespace()
        .find_map(|w| w.parse().ok())
        .ok_or("no count")?;
    let ins = |text: &str, line| CodeItem::Ins {
        text: text.into(),
        comment: String::new(),
        line,
    };
    let mut items = vec![ins("s_branch .Lfar", 1)];
    items.extend((0..n).map(|i| ins("s_nop 0", i + 2)));
    items.push(CodeItem::Label {
        name: ".Lfar".into(),
        line: n + 2,
    });
    if described.ends_with("s_endpgm") {
        items.push(ins("s_endpgm", n + 3));
    }
    Ok(items)
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// `encoder_lines.json.gz`: each probe line's bytes, or its refusal.
fn encoder_line(case: &Case<'_>) -> Result<(), String> {
    let line = case.value[0].as_str().ok_or("no line")?;
    let got = match default_target().hooks().encode(line) {
        Ok(Encoded::Bytes(bytes)) => Some(hex(&bytes)),
        Ok(Encoded::Branch { .. }) | Err(_) => None,
    };
    expect_eq(got, case.value[1].as_str().map(str::to_owned))
}

/// `read_s.json`: the reader's result, code items as [kind, text or name
/// or power, line] with the comment left out.
fn read_s_case(case: &Case<'_>) -> Result<(), String> {
    let got = read_s(case.value["text"].as_str().ok_or("no text")?, false);
    let Some(want) = case.value.get("code") else {
        return expect_eq(got.is_err(), true);
    };
    let file = got.map_err(|e| e.to_string())?;
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
    expect_eq(&Value::Array(code), want)?;
    expect_eq(pairs(&file.once), case.value["once"].clone())?;
    expect_eq(pairs(&file.block), case.value["block"].clone())?;
    expect_eq(&json!(file.meta), &case.value["meta"])
}

fn footer_of(p: &Value) -> Footer {
    let n = |k: &str| p[k].as_u64().unwrap_or(0) as u32;
    Footer {
        name: p["name"].as_str().unwrap_or("").into(),
        target_name: p["target_name"].as_str().unwrap_or("").into(),
        lds: n("lds"),
        kernarg: n("kernarg"),
        vgpr: n("vgpr"),
        agpr: n("agpr"),
        sgpr: n("sgpr"),
        wgsize: n("wgsize"),
        wgy: n("wgy"),
        preload: n("preload"),
        ends: false,
    }
}

/// `codeobject.json`: gfx1201's code-object facts.
fn codeobject(case: &Case<'_>) -> Result<(), String> {
    let (t, v) = (default_target(), case.value);
    let hooks = t.hooks();
    expect_eq(
        u64::from(hooks.e_flags()),
        v["e_flags"].as_u64().unwrap_or(0),
    )?;
    expect_eq(
        u64::from(hooks.granule()),
        v["granule"].as_u64().unwrap_or(0),
    )?;
    let mut directives = hooks.directives().to_vec();
    directives.sort_unstable();
    expect_eq(json!(directives), v["directives"].clone())?;
    let mut common = thin_target::COMMON.to_vec();
    common.sort_unstable();
    expect_eq(json!(common), v["common_directives"].clone())?;
    for f in v["footers"].as_array().ok_or("no footers")? {
        let text = hooks
            .footer(&footer_of(&f["params"]))
            .map_err(|e| e.to_string())?;
        expect_eq(text.as_str(), f["text"].as_str().unwrap_or(""))?;
    }
    for f in v["footer_refusals"].as_array().ok_or("no refusals")? {
        expect_eq(hooks.footer(&footer_of(&f["params"])).is_err(), true)?;
    }
    for r in v["resources"].as_array().ok_or("no resources")? {
        let block = Block(
            (r["block"].as_object().ok_or("no block")?)
                .iter()
                .map(|(k, x)| (k.clone(), x.as_u64().unwrap_or(0)))
                .collect(),
        );
        let got = hooks.resources(&block);
        match r.get("resources") {
            Some(want) => {
                let res = got.map_err(|e| e.to_string())?;
                expect_eq(
                    json!({"rsrc1": res.rsrc1, "rsrc3": res.rsrc3, "props": res.props,
                        "preload": res.preload, "user_sgprs": res.user_sgprs}),
                    want.clone(),
                )?;
            }
            None => expect_eq(got.is_err(), true)?,
        }
    }
    let kernel = &v["kernel"];
    let file = read_s(kernel["s"].as_str().ok_or("no .s")?, false).map_err(|e| e.to_string())?;
    let d = descriptor(&file, t).map_err(|e| e.to_string())?;
    expect_eq(
        d.to_vec(),
        hex_bytes(kernel["descriptor_hex"].as_str().unwrap_or("")),
    )?;
    let meta = thin_codegen::object::metadata(&file)?;
    expect_eq(json_of(&meta), kernel["metadata"].clone())
}

/// A metadata document as the JSON the vector writes it as.
fn json_of(v: &thin_formats::Value) -> Value {
    use thin_formats::Value as V;
    match v {
        V::Nil => Value::Null,
        V::Bool(b) => json!(b),
        V::Int(i) => json!(*i as i64),
        V::Str(s) => json!(s),
        V::Array(a) => Value::Array(a.iter().map(json_of).collect()),
        V::Map(m) => Value::Object(m.iter().map(|(k, x)| (k.clone(), json_of(x))).collect()),
    }
}

/// The target record's facts as `target_facts.json` `target` writes them.
fn target_fact(t: &Target, key: &str) -> Value {
    let d = t.definition();
    match key {
        "name" => json!(t.name()),
        "wave" => json!(d.wave),
        "lds_max" => json!(d.lds_max),
        "full_mask" => json!(t.full_mask()),
        "files" => json!(d
            .files
            .iter()
            .map(|f| json!({"name": f.letter.to_string(),
            "size": f.size, "granule": f.granule, "align64": f.align64}))
            .collect::<Vec<_>>()),
        "counters" => json!(d
            .counters
            .iter()
            .map(|c| json!({"name": c.name, "max": c.max,
            "field": c.field, "units": c.units.names()}))
            .collect::<Vec<_>>()),
        "waits" => json!(d
            .waits
            .iter()
            .map(|w| json!({"mnemonic": w.mnemonic,
            "fields": w.fields, "text": w.text.name()}))
            .collect::<Vec<_>>()),
        "wait_fields" => json!(t.wait_fields()),
        "operands" => json!({"vop3_literal": d.operands.vop3_literal,
            "constant_bus": d.operands.constant_bus,
            "valu_sgpr_write_waits_smem": d.operands.valu_sgpr_write_waits_smem}),
        "chip" => {
            let c = &d.chip;
            json!({"xcds": c.xcds, "cus": c.cus, "simds": c.simds, "lds": c.lds,
                "vgprs": c.vgprs, "vgprs_per_lane": c.vgprs_per_lane, "sgprs": c.sgprs,
                "user_sgprs": c.user_sgprs, "preload": c.preload, "wave_slots": c.wave_slots,
                "wgids": c.wgids})
        }
        "spellings" => json!(d.spellings),
        "barrier" => json!(d.barrier),
        "hazards" => json!({"rules": d.hazards.rules, "pass": d.hazards.pass,
            "window": d.hazards.window}),
        "forms" => json!(t.forms().len()),
        "pseudo_ops" => json!(t.own_ops().len()),
        "encodings" => json!(d.encodings.len()),
        _ => Value::Null,
    }
}

/// One query of `target_facts.json` `queries`: each [input…, output].
fn query(t: &Target, key: &str, q: &Value) -> Value {
    let s = |i: usize| q[i].as_str().unwrap_or("");
    match key {
        "spell" => json!([s(0), t.spell(s(0))]),
        "masked" => json!([s(0), t.masked(s(0))]),
        "canon" => json!([s(0), t.canon(s(0))]),
        "group" => {
            let names: Vec<&str> = t
                .group(s(0))
                .iter()
                .map(|&c| t.definition().counters[c].name)
                .collect();
            json!([s(0), names])
        }
        "mask_reg" => json!([
            s(0),
            thin_target::phys_named(s(0)).map(|p| t.mask_reg(p).name)
        ]),
        "unit_and_mem" => json!([
            s(0),
            t.unit_of(s(0)).map(|u| u.names()),
            t.hooks().mem_of(s(0)).map(|m| m.name())
        ]),
        "returning" => json!([s(0), q[1], t.hooks().returning(s(0), q[1].as_str())]),
        "form" => {
            let (arity, returning) = (
                q[1].as_u64().unwrap_or(0) as usize,
                q[2].as_bool().unwrap_or(false),
            );
            json!([
                s(0),
                q[1],
                q[2],
                t.form(s(0), arity, returning).map(|f| f.id)
            ])
        }
        _ => Value::Null,
    }
}

/// One generic rule of `target_facts.json` `generic`.
fn generic(key: &str, q: &Value) -> Value {
    let s = |i: usize| q[i].as_str().unwrap_or("");
    match key {
        "mem_of_encoding" => json!([s(0), thin_target::mem_of_encoding(s(0)).map(|m| m.name())]),
        "mem_bytes" => json!([s(0), thin_target::mem_bytes(s(0))]),
        "reads_dst" => json!([s(0), thin_target::reads_dst(s(0))]),
        "base" => json!([s(0), thin_target::base(s(0))]),
        "dst_sel" => {
            let flags = if q[1].as_bool() == Some(true) {
                thin_target::Flags::VALU
            } else {
                thin_target::Flags::NONE
            };
            json!([s(0), q[1], thin_target::dst_sel(flags, s(0)).name()])
        }
        _ => Value::Null,
    }
}

/// `target_facts.json`: the target record, the named registers, the flags,
/// the field names, the queries and the generic rules.
fn target_facts(case: &Case<'_>) -> Result<(), String> {
    let (section, key) = golden::split_id(&case.id);
    let t = default_target();
    let v = case.value;
    match section {
        "target" => expect_eq(&target_fact(t, key), v),
        "phys" => {
            let p = &PHYS[key.parse::<usize>().map_err(|e| e.to_string())?];
            expect_eq(
                json!({"name": p.name, "dwords": p.dwords, "file": p.space.name(),
                "whole": PHYS[p.whole].name}),
                v.clone(),
            )
        }
        "flags" => {
            let (name, bit) =
                thin_target::FLAG_BITS[key.parse::<usize>().map_err(|e| e.to_string())?];
            expect_eq(json!([name, bit]), v.clone())
        }
        "field_names" => {
            let i = key.parse::<usize>().map_err(|e| e.to_string())?;
            expect_eq(json!(thin_target::FIELD_NAMES[i]), v.clone())
        }
        "queries" => {
            let got: Vec<Value> = v
                .as_array()
                .ok_or("no queries")?
                .iter()
                .map(|q| query(t, key, q))
                .collect();
            expect_eq(&json!(got), v)
        }
        "generic" => {
            let got: Vec<Value> = v
                .as_array()
                .ok_or("no rules")?
                .iter()
                .map(|q| generic(key, q))
                .collect();
            expect_eq(&json!(got), v)
        }
        other => Err(format!("unknown section {other}")),
    }
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

/// `hazards.json` `cases`: gfx1201's need per line.
fn hazards_case(case: &Case<'_>) -> Result<(), String> {
    let t = default_target();
    let lines: Vec<&str> = case.value["lines"]
        .as_array()
        .ok_or("lines")?
        .iter()
        .filter_map(Value::as_str)
        .collect();
    let s = thin_test_support::asm::snapshot(t, thin_test_support::asm::read(t, &lines)?);
    let model = thin_targets::hazards(t.name()).map_err(|e| e.to_string())?;
    let needs: Vec<Value> = thin_passes::hazards::needs(&s, model)
        .iter()
        .map(|n| need_json(t, n))
        .collect();
    expect_eq(json!(needs), case.value["needs"].clone())
}

/// `hazards.json` `delay_text`: an `s_delay_alu` operand's text.
fn delay_text_case(case: &Case<'_>) -> Result<(), String> {
    let value = case.value[0].as_u64().ok_or("a delay")? as u32;
    expect_eq(json!(thin_passes::delay_text(value)), case.value[1].clone())
}

fn main() {
    let mut suite = Suite::new("target");
    suite.whole("tables_manifest.json", tables_manifest);
    suite.whole("isa_tables.json.gz", isa_tables);
    suite.whole("operand_table.json.gz", operand_table);
    suite.file("forms.json.gz", form_case);
    suite.file("target_facts.json", target_facts);
    suite.file("read_s.json", read_s_case);
    suite.file("line_syntax.json", line_syntax);
    suite.section("encoder_lines.json.gz", "rows", encoder_line);
    suite.section("hazards.json", "cases", hazards_case);
    suite.section("hazards.json", "delay_text", delay_text_case);
    suite.whole("codeobject.json", codeobject);
    suite.finish();
}
