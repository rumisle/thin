//! Golden checks for decoded execution, memory, tags and numerical operations.

use serde_json::Value;
use thin_test_support::adapters::{decoded_ins, default_target, reading_row};
use thin_test_support::golden_tests::expect_eq;
use thin_test_support::{Case, Suite};

/// `reading.json`: the target's one reader with gfx1201's decoding map.
fn reading_case(case: &Case<'_>) -> Result<(), String> {
    let text = case.value["text"].as_str().ok_or("no text")?;
    let got = thin_emu::read_program(text, default_target());
    let Some(want) = case.value.get("ins") else {
        return expect_eq(
            got.err().map(|e| e.message),
            case.value["error"].as_str().map(str::to_owned),
        );
    };
    let program = got.map_err(|e| e.message)?;
    expect_eq(
        &Value::Array(program.ins.iter().map(reading_row).collect()),
        want,
    )?;
    let labels: serde_json::Map<String, Value> = program
        .labels
        .iter()
        .map(|(name, i)| (name.clone(), Value::from(*i)))
        .collect();
    expect_eq(&Value::Object(labels), &case.value["labels"])?;
    let sections: Vec<Value> = program
        .sections
        .iter()
        .map(|(i, t, l)| serde_json::json!([i, t, l]))
        .collect();
    expect_eq(&Value::Array(sections), &case.value["sections"])?;
    expect_eq(
        u64::from(program.user_sgprs),
        case.value["user_sgprs"].as_u64().unwrap_or(2),
    )
}

use thin_emu::machine::facts::Facts;
use thin_emu::machine::memory::Memory;
use thin_emu::machine::run::Machine;
use thin_emu::machine::wave::Wave;
use thin_emu::machine::WgState;

/// One instruction on one wave at gfx1201's facts (W = 32): the machine
/// and its wave, set up by `init`, after the step.
fn one(ins: thin_emu::Ins, init: &mut dyn FnMut(&mut Wave)) -> Result<Wave, thin_emu::EmuError> {
    let program = thin_emu::Program {
        ins: vec![ins],
        user_sgprs: 2,
        ..Default::default()
    };
    let mut m = Machine::new(program, Facts::of(default_target()), 16, 0, 0)?;
    let mut mem = Memory::default();
    let mut tables = thin_emu::Tables::default();
    let mut w = m.make_waves(0, (0, 0), 1, 0, 0).remove(0);
    init(&mut w);
    let mut lds = m.new_lds(1);
    let mut wg = WgState::default();
    m.step(&mut w, &mut mem, &mut tables, &mut lds, &mut wg)?;
    Ok(w)
}

fn hex64(v: &Value) -> u64 {
    u64::from_str_radix(v.as_str().unwrap_or("0").trim_start_matches("0x"), 16).unwrap_or(0)
}

fn lanes16(v: &Value) -> Vec<u32> {
    v.as_array().into_iter().flatten().map(hex).collect()
}

fn mask16(v: &Value) -> u64 {
    let m = hex64(v) & 0xFFFF;
    m | m << 16
}

fn tag_of(t: &mut thin_emu::Tables, v: &Value) -> i64 {
    if let Some(a) = v.as_array() {
        let (x, y) = (tag_of(t, &a[1]), tag_of(t, &a[2]));
        return t.sum(x, y);
    }
    let s = v.as_str().unwrap_or("-");
    if s == "-" {
        return 0;
    }
    let (b, o) = s
        .trim_start_matches('m')
        .split_once(':')
        .unwrap_or(("0", "0"));
    thin_emu::tags::mem_tag(b.parse().unwrap_or(0), o.parse().unwrap_or(0))
}

fn tag_text(t: &thin_emu::Tables, tag: i64) -> Value {
    if tag == 0 {
        return Value::from("-");
    }
    if thin_emu::tags::is_sum(tag) {
        let (x, y) = t.sum_of(tag);
        return serde_json::json!(["sum", tag_text(t, x), tag_text(t, y)]);
    }
    if tag > 0 {
        return Value::from(format!(
            "m{}:{}",
            thin_emu::tags::tag_buf(tag),
            thin_emu::tags::tag_off(tag)
        ));
    }
    Value::from(format!("record {tag}"))
}

fn reg_of(name: &str) -> (char, usize) {
    let mut c = name.chars();
    let f = c.next().unwrap_or('v');
    (f, c.as_str().parse().unwrap_or(0))
}

/// `tags.json`: one instruction's tags on 32 lanes (lane l holds lane l
/// mod 4's values and tags).
fn tags_case(case: &Case<'_>) -> Result<(), String> {
    let c = case.value;
    let ins = decoded_ins(&c["ins"], 32);
    let program = thin_emu::Program {
        ins: vec![ins],
        user_sgprs: 2,
        ..Default::default()
    };
    let mut m =
        Machine::new(program, Facts::of(default_target()), 16, 0, 0).map_err(|e| e.message)?;
    m.sum_records = c["sum_records"].as_bool().unwrap_or(false);
    let mut mem = Memory::default();
    let mut tables = thin_emu::Tables::default();
    let mut w = m.make_waves(0, (0, 0), 1, 0, 0).remove(0);
    for (name, vals) in c["values"].as_object().into_iter().flatten() {
        let (f, r) = reg_of(name);
        if f == 's' {
            w.s[r] = hex(vals);
            continue;
        }
        let vals = lanes16(vals);
        for l in 0..32 {
            w.v[r * 32 + l] = vals[l % 4];
        }
    }
    for (name, lanes) in c["tags_in"].as_object().into_iter().flatten() {
        let (_, r) = reg_of(name);
        let lanes: Vec<[i64; 4]> = lanes
            .as_array()
            .into_iter()
            .flatten()
            .map(|l| {
                let b: Vec<i64> = l
                    .as_array()
                    .into_iter()
                    .flatten()
                    .map(|t| tag_of(&mut tables, t))
                    .collect();
                [b[0], b[1], b[2], b[3]]
            })
            .collect();
        let all: Vec<[i64; 4]> = (0..32).map(|l| lanes[l % 4]).collect();
        w.vt[r] = Some(all.into_boxed_slice());
    }
    if let Some(x) = c.get("exec") {
        let m4 = hex64(x) & 0xF;
        w.exec = (0..8).fold(0u64, |acc, k| acc | m4 << (4 * k));
    }
    let mut lds = m.new_lds(1);
    let mut wg = WgState::default();
    m.step(&mut w, &mut mem, &mut tables, &mut lds, &mut wg)
        .map_err(|e| e.message)?;
    for (name, want) in c["tags_out"].as_object().into_iter().flatten() {
        let (_, r) = reg_of(name);
        let got: Vec<Value> = (0..4)
            .map(|l| {
                let lane = w.vt[r].as_ref().map_or([0; 4], |t| t[l]);
                Value::Array(lane.iter().map(|t| tag_text(&tables, *t)).collect())
            })
            .collect();
        expect_eq((name.as_str(), &Value::Array(got)), (name.as_str(), want))?;
    }
    for (name, want) in c["values_out"].as_object().into_iter().flatten() {
        let (_, r) = reg_of(name);
        let got: Vec<u32> = (0..4).map(|l| w.v[r * 32 + l]).collect();
        expect_eq((name.as_str(), got), (name.as_str(), lanes16(want)))?;
    }
    Ok(())
}

/// A decoded instruction for a model script: its canon, kind, operands
/// (a compact spelling), fields and the op name as its text.
fn mk(
    canon: &str,
    kind: &str,
    ops: &[&str],
    fields: &[(&str, thin_emu::Field)],
    text: &str,
) -> thin_emu::Ins {
    let operand = |o: &str| {
        let base = if let Some(l) = o.strip_prefix('@') {
            thin_emu::Base::Label(l.to_owned())
        } else if o == "vcc" || o == "exec" {
            thin_emu::Base::Reg(o.to_owned(), 0, 2)
        } else if o == "ttmp9" {
            thin_emu::Base::Reg(o.to_owned(), 0, 1)
        } else if let Some(v) = o.strip_prefix('#') {
            thin_emu::Base::Literal(v.parse().unwrap_or(0), v.to_owned())
        } else {
            let (f, rest) = o.split_at(1);
            let (a, n) = match rest.strip_prefix('[') {
                Some(r) => {
                    let (a, b) = r
                        .trim_end_matches(']')
                        .split_once(':')
                        .unwrap_or(("0", "0"));
                    let (a, b): (u32, u32) = (a.parse().unwrap_or(0), b.parse().unwrap_or(0));
                    (a, b - a + 1)
                }
                None => (rest.parse().unwrap_or(0), 1),
            };
            thin_emu::Base::Reg(f.to_owned(), a, n)
        };
        thin_emu::Operand {
            base,
            neg: false,
            abs: false,
        }
    };
    thin_emu::Ins {
        canon: canon.to_owned(),
        kind: kind.to_owned(),
        operands: ops.iter().map(|o| operand(o)).collect(),
        fields: fields
            .iter()
            .map(|(k, v)| ((*k).to_owned(), v.clone()))
            .collect(),
        matrix: None,
        text: text.to_owned(),
        comment: String::new(),
        line: 0,
    }
}

fn int(v: i128) -> thin_emu::Field {
    thin_emu::Field::Int(v)
}

/// The decoded instructions of one `lds.json` op (gfx1201's, W = 32).
fn lds_op(op: &str, out: &mut Vec<thin_emu::Ins>) {
    let (name, at) = match op.split_once('@') {
        Some((n, a)) => (n, a.parse::<i128>().unwrap_or(0)),
        None => (op, 0),
    };
    let off = [("offset", int(at))];
    let off: &[(&str, thin_emu::Field)] = if at == 0 { &[] } else { &off };
    out.push(match name {
        // A read at an offset loads v4: its result lands at a wait, apart
        // from the plain read's v2.
        "read" => mk(
            "ds.load.b32",
            "ds",
            &[if at == 0 { "v2" } else { "v4" }, "v1"],
            off,
            op,
        ),
        "write" => mk("ds.store.b32", "ds", &["v1", "v3"], off, op),
        "atomic" => mk("ds.add.u32", "ds", &["v1", "v3"], off, op),
        "wait_ds" => mk("s.wait.dscnt", "wait", &["#0"], &[], op),
        "wait_ds(1)" => mk("s.wait.dscnt", "wait", &["#1"], &[], op),
        "wait_vm" => mk("s.wait.loadcnt", "wait", &["#0"], &[], op),
        "smem" => mk("s.load.b32", "smem", &["s20", "s[4:5]", "#0"], &[], op),
        "end" => mk("s.endpgm", "salu", &[], &[], op),
        "store" => mk("global.store.b32", "vmem", &["v1", "v2", "s[8:9]"], &[], op),
        "fence_sole" => {
            let mut b = mk("s.barrier.wait", "barrier", &[], &[], op);
            b.comment = "LDS_FENCE".into();
            b
        }
        _ => mk("s.barrier.wait", "barrier", &[], &[], op),
    });
}

/// `lds.json`: two waves of one workgroup run a script (W = 32).
fn lds_case(case: &Case<'_>) -> Result<(), String> {
    let c = case.value;
    let mut ins = Vec::new();
    let mut labels = Vec::new();
    for (k, step) in c["script"].as_array().into_iter().flatten().enumerate() {
        match step {
            Value::String(s) if s == "barrier" => {
                ins.push(mk("s.barrier.wait", "barrier", &[], &[], "barrier"))
            }
            Value::String(_) => {
                let mut b = mk("s.barrier.wait", "barrier", &[], &[], "fence");
                b.comment = "LDS_FENCE".into();
                ins.push(b);
            }
            Value::Array(a) => {
                let skip = format!("skip{k}");
                if let Some(w) = a[0].as_u64() {
                    ins.push(mk(
                        "s.cmp.eq.u32",
                        "salu",
                        &["s10", &format!("#{w}")],
                        &[],
                        "cmp",
                    ));
                    ins.push(mk(
                        "s.cbranch.scc0",
                        "branch",
                        &[&format!("@{skip}")],
                        &[],
                        "branch",
                    ));
                }
                for op in a[1].as_array().into_iter().flatten() {
                    lds_op(op.as_str().unwrap_or(""), &mut ins);
                }
                labels.push((skip, ins.len()));
            }
            _ => {}
        }
    }
    for counter in [
        "s.wait.loadcnt",
        "s.wait.storecnt",
        "s.wait.dscnt",
        "s.wait.kmcnt",
    ] {
        ins.push(mk(counter, "wait", &["#0"], &[], "wait"));
    }
    ins.push(mk("s.endpgm", "salu", &[], &[], "end"));
    let program = thin_emu::Program {
        ins,
        labels,
        user_sgprs: 2,
        ..Default::default()
    };
    let mut m =
        Machine::new(program, Facts::of(default_target()), 8, 0, 512).map_err(|e| e.message)?;
    let mut mem = Memory::default();
    let mut tables = thin_emu::Tables::default();
    let src = mem
        .alloc("src", 1024, true, 0, Default::default())
        .map_err(|e| e.message)?;
    let out = mem
        .alloc("out", 1024, true, 0, Default::default())
        .map_err(|e| e.message)?;
    let (sb, ob) = (mem.bufs[src].base, mem.bufs[out].base);
    let mut init = |w: &mut Wave| {
        w.s[10] = w.wid as u32;
        w.s[4] = sb as u32;
        w.s[5] = (sb >> 32) as u32;
        w.s[6] = 1024;
        w.s[7] = 0;
        w.s[8] = ob as u32;
        w.s[9] = (ob >> 32) as u32;
        w.m0 = 0;
        for l in 0..32 {
            w.v[32 + l] = 4 * l as u32;
            w.v[3 * 32 + l] = 7;
        }
    };
    let got = m.run_wg(&mut mem, &mut tables, 0, (0, 0), 2, 0, Some(&mut init));
    expect_eq(
        got.as_ref().err().map(|e| e.message.clone()),
        c["error"].as_str().map(str::to_owned),
    )?;
    if let Some(stored) = c.get("stored") {
        let d = &mem.bufs[out].data;
        let got: Vec<String> = (0..8)
            .map(|k| {
                format!(
                    "{:#010x}",
                    u32::from_le_bytes([d[4 * k], d[4 * k + 1], d[4 * k + 2], d[4 * k + 3]])
                )
            })
            .collect();
        let want: Vec<String> = stored
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap_or("").to_owned())
            .collect();
        expect_eq(got, want)?;
    }
    Ok(())
}

/// The decoded instructions of one `flight.json` op: d at s[20:21], f at
/// s[22:23], o at s[24:25]; v1 = 4 l, v3 = 7, v4 = 8, v5 = 1, v6 = 0.
fn flight_op(
    op: &str,
    value8: &mut bool,
    out: &mut Vec<thin_emu::Ins>,
    labels: &mut Vec<(String, usize)>,
) {
    let agent = [("scope", int(2))];
    let consume = |flag: i128,
                   retry: bool,
                   out: &mut Vec<thin_emu::Ins>,
                   labels: &mut Vec<(String, usize)>| {
        out.push(mk(
            "global.load.b32",
            "vmem",
            &["v7", "v6", "s[22:23]"],
            &[("offset", int(flag)), ("scope", int(2))],
            op,
        ));
        out.push(mk("s.wait.loadcnt", "wait", &["#0"], &[], "wait"));
        let top = format!("retry{}", out.len());
        labels.push((top.clone(), out.len()));
        out.push(mk(
            "global.load.b32",
            "vmem",
            &["v8", "v1", "s[20:21]"],
            &[],
            op,
        ));
        out.push(mk("s.wait.loadcnt", "wait", &["#0"], &[], "wait"));
        if retry {
            out.push(mk("v.cmp.ne.u32", "valu", &["vcc", "v8", "v3"], &[], "cmp"));
            out.push(mk(
                "s.cbranch.vccnz",
                "branch",
                &[&format!("@{top}")],
                &[],
                "branch",
            ));
        }
        out.push(mk(
            "global.store.b32",
            "vmem",
            &["v1", "v8", "s[24:25]"],
            &[],
            op,
        ));
    };
    match op {
        "data" => {
            let v = if *value8 { "v4" } else { "v3" };
            out.push(mk(
                "global.store.b32",
                "vmem",
                &["v1", v, "s[20:21]"],
                &[],
                op,
            ));
        }
        "value8" => *value8 = true,
        "wait" => out.push(mk("s.wait.storecnt", "wait", &["#0"], &[], op)),
        "flag_atomic" => out.push(mk(
            "global.atomic.add.u32",
            "vmem",
            &["v6", "v5", "s[22:23]"],
            &[],
            op,
        )),
        "flag_store" => out.push(mk(
            "global.store.b32",
            "vmem",
            &["v6", "v5", "s[22:23]"],
            &agent,
            op,
        )),
        "flag_plain" => out.push(mk(
            "global.store.b32",
            "vmem",
            &["v6", "v5", "s[22:23]"],
            &[],
            op,
        )),
        "flag2_store" => out.push(mk(
            "global.store.b32",
            "vmem",
            &["v6", "v5", "s[22:23]"],
            &[("offset", int(4)), ("scope", int(2))],
            op,
        )),
        "consume" => consume(0, false, out, labels),
        "consume_retry" => consume(0, true, out, labels),
        "consume2" => consume(4, false, out, labels),
        _ => out.push(mk("s.nop", "nop", &["#0"], &[], op)),
    }
}

/// `flight.json`: a producer workgroup, then a consumer, one wave each.
fn flight_case(case: &Case<'_>) -> Result<(), String> {
    let c = case.value;
    let mut ins = Vec::new();
    let mut labels = Vec::new();
    let mut value8 = false;
    for (wg, key) in [(0u32, "producer"), (1, "consumer")] {
        ins.push(mk(
            "s.cmp.eq.u32",
            "salu",
            &["ttmp9", &format!("#{wg}")],
            &[],
            "cmp",
        ));
        ins.push(mk(
            "s.cbranch.scc0",
            "branch",
            &[&format!("@skip{wg}")],
            &[],
            "branch",
        ));
        for op in c[key].as_array().into_iter().flatten() {
            flight_op(
                op.as_str().unwrap_or(""),
                &mut value8,
                &mut ins,
                &mut labels,
            );
        }
        labels.push((format!("skip{wg}"), ins.len()));
    }
    for counter in [
        "s.wait.loadcnt",
        "s.wait.storecnt",
        "s.wait.dscnt",
        "s.wait.kmcnt",
    ] {
        ins.push(mk(counter, "wait", &["#0"], &[], "wait"));
    }
    ins.push(mk("s.endpgm", "salu", &[], &[], "end"));
    let program = thin_emu::Program {
        ins,
        labels,
        user_sgprs: 2,
        ..Default::default()
    };
    let mut m =
        Machine::new(program, Facts::of(default_target()), 10, 0, 0).map_err(|e| e.message)?;
    let mut mem = Memory::default();
    let mut tables = thin_emu::Tables::default();
    let mut bases = Vec::new();
    for name in ["d", "f", "o"] {
        let b = mem
            .alloc(name, 256, true, 0, Default::default())
            .map_err(|e| e.message)?;
        bases.push(mem.bufs[b].base);
    }
    let mut init = |w: &mut Wave| {
        for (k, b) in bases.iter().enumerate() {
            w.s[20 + 2 * k] = *b as u32;
            w.s[21 + 2 * k] = (*b >> 32) as u32;
        }
        for l in 0..32 {
            w.v[32 + l] = 4 * l as u32;
            w.v[3 * 32 + l] = 7;
            w.v[4 * 32 + l] = 8;
            w.v[5 * 32 + l] = 1;
            w.v[6 * 32 + l] = 0;
        }
    };
    let grid = c["grid"].as_u64().unwrap_or(2) as u32;
    for x in 0..grid {
        m.run_wg(
            &mut mem,
            &mut tables,
            0,
            (x, 0),
            1,
            x as usize,
            Some(&mut init),
        )
        .map_err(|e| e.message)?;
    }
    let o = &mem.bufs[2].data;
    let mut stored: Vec<u64> = (0..32)
        .map(|k| {
            u64::from(u32::from_le_bytes([
                o[4 * k],
                o[4 * k + 1],
                o[4 * k + 2],
                o[4 * k + 3],
            ]))
        })
        .collect();
    stored.sort_unstable();
    stored.dedup();
    let want: Vec<u64> = c["consumer_stored"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap_or(0))
        .collect();
    expect_eq(("stored", stored), ("stored", want))?;
    expect_eq(
        ("events", mem.flight.events.len() as u64),
        ("events", c["events"].as_u64().unwrap_or(0)),
    )
}

/// The decoded instructions of one `scope.json` op: buf at s[20:21], v6 =
/// 0, v3 = 7.
fn scope_op(op: &str, out: &mut Vec<thin_emu::Ins>, labels: &mut Vec<(String, usize)>) {
    let sc1 = [("scope", int(2))];
    let sys = [("scope", int(3))];
    let nt = [("th", int(1))];
    let one = match op {
        "store" => mk(
            "global.store.b32",
            "vmem",
            &["v6", "v3", "s[20:21]"],
            &[],
            op,
        ),
        "store_agent" => mk(
            "global.store.b32",
            "vmem",
            &["v6", "v3", "s[20:21]"],
            &sc1,
            op,
        ),
        "load" => mk(
            "global.load.b32",
            "vmem",
            &["v7", "v6", "s[20:21]"],
            &[],
            op,
        ),
        "load_agent" => mk(
            "global.load.b32",
            "vmem",
            &["v7", "v6", "s[20:21]"],
            &sc1,
            op,
        ),
        "load_nt" => mk(
            "global.load.b32",
            "vmem",
            &["v7", "v6", "s[20:21]"],
            &nt,
            op,
        ),
        "atomic" => mk(
            "global.atomic.add.u32",
            "vmem",
            &["v6", "v3", "s[20:21]"],
            &[],
            op,
        ),
        "atomic_system" => mk(
            "global.atomic.add.u32",
            "vmem",
            &["v6", "v3", "s[20:21]"],
            &sys,
            op,
        ),
        "invalidate" => mk("global.inv", "vmem", &[], &sys, op),
        "writeback" => mk("global.wb", "vmem", &[], &sys, op),
        "wait" => {
            out.push(mk("s.wait.loadcnt", "wait", &["#0"], &[], op));
            mk("s.wait.storecnt", "wait", &["#0"], &[], op)
        }
        _ => {
            let top = format!("busy{}", out.len() + 1);
            out.push(mk("s.mov.b32", "salu", &["s30", "#1000"], &[], "busy"));
            labels.push((top.clone(), out.len()));
            out.push(mk("s.sub.u32", "salu", &["s30", "s30", "#1"], &[], "busy"));
            out.push(mk("s.cmp.lg.u32", "salu", &["s30", "#0"], &[], "busy"));
            mk(
                "s.cbranch.scc1",
                "branch",
                &[&format!("@{top}")],
                &[],
                "busy",
            )
        }
    };
    out.push(one);
}

/// `scope.json`: two ranks, one workgroup of one wave each, over one
/// memory with the scope model on (a cluster).
fn scope_case(case: &Case<'_>) -> Result<(), String> {
    use thin_emu::machine::cluster::{Cluster, Options};
    let c = case.value;
    let mut ins = vec![
        mk("s.load.b32", "smem", &["s10", "s[0:1]", "#0"], &[], "rank"),
        mk(
            "s.load.b64",
            "smem",
            &["s[20:21]", "s[0:1]", "#8"],
            &[],
            "buf",
        ),
        mk("s.wait.kmcnt", "wait", &["#0"], &[], "wait"),
        mk("v.mov.b32", "valu", &["v6", "#0"], &[], "zero"),
        mk("v.mov.b32", "valu", &["v3", "#7"], &[], "seven"),
    ];
    let mut labels = Vec::new();
    for r in 0..2u32 {
        ins.push(mk(
            "s.cmp.eq.u32",
            "salu",
            &["s10", &format!("#{r}")],
            &[],
            "cmp",
        ));
        ins.push(mk(
            "s.cbranch.scc0",
            "branch",
            &[&format!("@skip{r}")],
            &[],
            "branch",
        ));
        for op in c[format!("rank{r}")].as_array().into_iter().flatten() {
            scope_op(op.as_str().unwrap_or(""), &mut ins, &mut labels);
        }
        labels.push((format!("skip{r}"), ins.len()));
    }
    for counter in [
        "s.wait.loadcnt",
        "s.wait.storecnt",
        "s.wait.dscnt",
        "s.wait.kmcnt",
    ] {
        ins.push(mk(counter, "wait", &["#0"], &[], "wait"));
    }
    ins.push(mk("s.endpgm", "salu", &[], &[], "end"));
    let program = thin_emu::Program {
        ins,
        labels,
        user_sgprs: 2,
        ..Default::default()
    };
    let mut mem = Memory::default();
    let meta = thin_emu::machine::memory::Meta {
        ipc: c["buffer"]["ipc"].as_bool().unwrap_or(false),
        ..Default::default()
    };
    let buf = mem.alloc("buf", 64, true, 0, meta).map_err(|e| e.message)?;
    let base = mem.bufs[buf].base;
    let mut machines = Vec::new();
    let mut launches = Vec::new();
    for r in 0..2usize {
        let k = mem
            .alloc(&format!("kernarg{r}"), 16, false, 0, Default::default())
            .map_err(|e| e.message)?;
        mem.bufs[k].data[..4].copy_from_slice(&(r as u32).to_le_bytes());
        mem.bufs[k].data[8..16].copy_from_slice(&base.to_le_bytes());
        launches.push(vec![mem.bufs[k].base]);
        let mut m = Machine::new(program.clone(), Facts::of(default_target()), 8, 0, 0)
            .map_err(|e| e.message)?;
        m.rank = r;
        machines.push(m);
    }
    let options = Options {
        skew: c["skew"].as_array().map_or_else(Vec::new, |s| {
            s.iter().map(|v| v.as_u64().unwrap_or(0)).collect()
        }),
        ..Options::default()
    };
    let mut cluster = Cluster::new(machines, launches, (1, 1), 1, options);
    let mut tables = thin_emu::Tables::default();
    let got = cluster.run(&mut mem, &mut tables);
    let class = got.as_ref().err().map(|e| match e.cause {
        Some(thin_emu::Cause::Scope) => "ScopeModel".to_owned(),
        Some(thin_emu::Cause::Ipc) => "IpcPolicy".to_owned(),
        other => format!("{other:?}: {}", e.message),
    });
    expect_eq(class, c["error"].as_str().map(str::to_owned))?;
    expect_eq(
        got.err().map(|e| e.message),
        c["message"].as_str().map(str::to_owned),
    )
}

/// `alu.json` `vector`: one VALU op on 32 lanes (lane l holds lane l mod
/// 16's inputs).
fn vector_case(case: &Case<'_>) -> Result<(), String> {
    let c = case.value;
    let ins = decoded_ins(&c["ins"], 32);
    let w = one(ins, &mut |w: &mut Wave| {
        let set = |w: &mut Wave, r: usize, vals: &[u32]| {
            for l in 0..32 {
                w.v[r * 32 + l] = vals[l % vals.len().max(1)];
            }
        };
        for (k, (name, reg)) in [("x", 8usize), ("y", 10), ("z", 12)].iter().enumerate() {
            let v = &c["in"][*name];
            if v.is_array() {
                set(w, *reg, &lanes16(v));
            } else if let Some(s) = v.get("sgpr") {
                w.s[20 + k] = hex(s);
            } else if v.get("lo").is_some() {
                set(w, *reg, &lanes16(&v["lo"]));
                set(w, *reg + 1, &lanes16(&v["hi"]));
            }
        }
        set(w, 1, &[0xa5a5_a5a5]);
        set(w, 2, &[0xa5a5_a5a5]);
        for (d, old) in c["old"].as_array().into_iter().flatten().enumerate() {
            set(w, 1 + d, &lanes16(old));
        }
        if let Some(m) = c.get("mask_in") {
            w.vcc = mask16(m);
        }
        if let Some(m) = c.get("exec") {
            w.exec = mask16(m);
        }
    })
    .map_err(|e| e.message)?;
    for (key, reg) in [("d", 1usize), ("d2", 2)] {
        if let Some(want) = c["out"].get(key) {
            let want = lanes16(want);
            let got: Vec<u32> = (0..16).map(|l| w.v[reg * 32 + l]).collect();
            expect_eq((key, got), (key, want))?;
        }
    }
    if let Some(m) = c["out"].get("mask") {
        expect_eq(
            format!("{:#06x}", w.vcc & 0xFFFF),
            m.as_str().unwrap_or("").to_owned(),
        )?;
    }
    Ok(())
}

/// `alu.json` `scalar`: one SALU op on its sources and SCC.
fn scalar_case(case: &Case<'_>) -> Result<(), String> {
    let c = case.value;
    let ins = decoded_ins(&c["ins"], 32);
    let from = usize::from(c["out"].get("d").is_some());
    let srcs: Vec<(u32, u32)> = ins.operands[from..]
        .iter()
        .filter_map(|o| match &o.base {
            thin_emu::Base::Reg(f, r, n) if f == "s" => Some((*r, *n)),
            _ => None,
        })
        .collect();
    let dst = match &ins.operands[0].base {
        thin_emu::Base::Reg(_, r, n) => (*r as usize, *n),
        _ => (0, 1),
    };
    let w = one(ins, &mut |w: &mut Wave| {
        // "A destination starts as 0xa5a5a5a5" (the notation).
        if from == 1 {
            for i in 0..dst.1 as usize {
                w.s[dst.0 + i] = 0xa5a5_a5a5;
            }
        }
        for (k, v) in c["in"].as_array().into_iter().flatten().enumerate() {
            if let Some((r, n)) = srcs.get(k) {
                let v = hex64(v);
                w.s[*r as usize] = v as u32;
                if *n == 2 {
                    w.s[*r as usize + 1] = (v >> 32) as u32;
                }
            }
        }
        w.scc = c["scc_in"].as_u64().unwrap_or(0) != 0;
    })
    .map_err(|e| e.message)?;
    expect_eq(
        ("scc", u64::from(w.scc)),
        ("scc", c["out"]["scc"].as_u64().unwrap_or(0)),
    )?;
    if let Some(d) = c["out"].get("d") {
        let got = if dst.1 == 2 {
            u64::from(w.s[dst.0]) | u64::from(w.s[dst.0 + 1]) << 32
        } else {
            u64::from(w.s[dst.0])
        };
        expect_eq(got, hex64(d))?;
    }
    Ok(())
}

fn hex(v: &Value) -> u32 {
    u32::from_str_radix(v.as_str().unwrap_or("0").trim_start_matches("0x"), 16).unwrap_or(0)
}

/// Python's float.hex text (or nan/inf) as an f64.
fn float_hex(s: &str) -> f64 {
    match s {
        "nan" | "-nan" => return f64::NAN,
        "inf" => return f64::INFINITY,
        "-inf" => return f64::NEG_INFINITY,
        _ => {}
    }
    let (neg, s) = s.strip_prefix('-').map_or((false, s), |r| (true, r));
    let s = s.trim_start_matches("0x");
    let (mant, exp) = s.split_once('p').unwrap_or((s, "0"));
    let (int, frac) = mant.split_once('.').unwrap_or((mant, ""));
    let mut v = f64::from(u32::from_str_radix(int, 16).unwrap_or(0));
    for (i, c) in frac.chars().enumerate() {
        v += f64::from(c.to_digit(16).unwrap_or(0)) * 16f64.powi(-(i as i32 + 1));
    }
    let v = v * 2f64.powi(exp.parse().unwrap_or(0));
    if neg {
        -v
    } else {
        v
    }
}

/// `numbers.json.gz`: every function bit for bit.
fn numbers(case: &Case<'_>) -> Result<(), String> {
    use thin_emu::machine::numbers as n;
    let doc = case.value;
    type Bits = fn(u32) -> u32;
    let pairs: [(&str, Bits); 7] = [
        ("bf16_rne", n::bf16_rne),
        ("bf16_conv", n::bf16_conv),
        ("f16_of_f32", n::f16_of_f32),
        ("f16_rtz", n::f16_rtz),
        ("f32_of_f16", n::f32_of_f16),
        ("i32_of_f32", n::i32_of_f32),
        ("u32_of_f32", n::u32_of_f32),
    ];
    for (key, f) in pairs {
        for c in doc[key].as_array().into_iter().flatten() {
            let (x, want) = (hex(&c[0]), hex(&c[1]));
            if f(x) != want {
                return Err(format!("{key}({x:#x}) = {:#x}, want {want:#x}", f(x)));
            }
        }
    }
    for (fmt, codes) in doc["codes"].as_object().into_iter().flatten() {
        let f = n::Small::named(fmt).ok_or("a format")?;
        for (code, v) in codes.as_array().into_iter().flatten().enumerate() {
            let want = float_hex(v.as_str().unwrap_or(""));
            let got = f.value(code as u32);
            let same = (got.is_nan() && want.is_nan()) || got.to_bits() == want.to_bits();
            if !same {
                return Err(format!("{fmt} code {code:#x} = {got}, want {want}"));
            }
        }
    }
    for c in doc["down"].as_array().into_iter().flatten() {
        let f = n::Small::named(c[0].as_str().unwrap_or("")).ok_or("a format")?;
        let x = f64::from(f32::from_bits(hex(&c[1])));
        let got = n::down(x, hex(&c[2]), f);
        if got != hex(&c[3]) {
            return Err(format!("down({c}) = {got:#x}"));
        }
    }
    for c in doc["up"].as_array().into_iter().flatten() {
        let f = n::Small::named(c[0].as_str().unwrap_or("")).ok_or("a format")?;
        let got = n::up_bits(n::up(hex(&c[1]), hex(&c[2]), f));
        if got != (hex(&c[3]), hex(&c[4]), hex(&c[5])) {
            return Err(format!("up({c}) = {got:x?}"));
        }
    }
    Ok(())
}

/// `banks.json`: the bank count on abstract rules.
fn banks(case: &Case<'_>) -> Result<(), String> {
    use thin_emu::machine::banks::{bank_count, BankRule};
    let (doc, c) = (case.file, case.value);
    let r = &doc["rules"][c["rule"].as_str().unwrap_or("")];
    let rule = BankRule {
        banks: r["banks"].as_u64().unwrap_or(1),
        phase: r["phase"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| p.as_u64().unwrap_or(0) as u32)
            .collect(),
    };
    let addrs: Vec<u64> = doc["addr"][c["addr"].as_str().unwrap_or("")]
        .as_array()
        .unwrap()
        .iter()
        .map(|a| a.as_u64().unwrap_or(0))
        .collect();
    let active: Vec<bool> = doc["active"][c["active"].as_str().unwrap_or("")]
        .as_array()
        .unwrap()
        .iter()
        .map(|a| a.as_bool().unwrap_or(false))
        .collect();
    expect_eq(
        bank_count(&rule, &addrs, &active),
        (
            c["cycles"].as_u64().unwrap_or(0),
            c["conflicts"].as_u64().unwrap_or(0),
        ),
    )
}

/// `claims.json`: each kind's claims of a fragment, or its refusal.
fn claims_case(case: &Case<'_>) -> Result<(), String> {
    use thin_emu::machine::claims::{read, Claims, Kind};
    let c = case.value;
    let program = thin_emu::read_program(c["text"].as_str().unwrap_or(""), default_target())
        .map_err(|e| e.message)?;
    expect_eq(
        u64::from(program.user_sgprs),
        c["user_sgprs"].as_u64().unwrap_or(2),
    )?;
    let reg = |r: &(String, u32, u32)| serde_json::json!([r.0, r.1, r.2]);
    for (key, kind) in [
        ("uniform", Kind::Uniform),
        ("assume", Kind::Assume),
        ("undefined", Kind::Undefined),
    ] {
        let mut claims = Claims::default();
        let got = match read(
            &program,
            &thin_emu::machine::facts::Facts::of(default_target()),
            kind,
            &mut claims,
        ) {
            Err(e) => serde_json::json!({"error": e.message}),
            Ok(()) => {
                let mut m = serde_json::Map::new();
                for (i, u) in &claims.uniform {
                    m.insert(
                        i.to_string(),
                        serde_json::json!([u.regs.iter().map(reg).collect::<Vec<_>>(), u.text]),
                    );
                }
                for (i, a) in &claims.assume {
                    let rows: Vec<Value> = a
                        .iter()
                        .map(|a| serde_json::json!([a.reg.0, a.reg.1, a.reg.2, a.lo, a.hi, a.text]))
                        .collect();
                    m.insert(i.to_string(), Value::Array(rows));
                }
                for (i, u) in &claims.undefined {
                    m.insert(i.to_string(), u.iter().map(reg).collect());
                }
                for ((a, b), u) in &claims.edges {
                    m.insert(format!("edge:{a}->{b}"), u.iter().map(reg).collect());
                }
                Value::Object(m)
            }
        };
        expect_eq((key, &got), (key, &c[key]))?;
    }
    Ok(())
}

/// `wmma.json.gz`: one D element of the dot unit's model, bit for bit.
fn wmma_case(case: &Case<'_>) -> Result<(), String> {
    use thin_emu::machine::dot::{wmma_element, Format, Step, BF16, BF8, F16, F32, FP8};
    use thin_emu::machine::numbers::{f16_one, Small};
    let c = case.value;
    let name = c["form"].as_str().ok_or("form")?;
    // The form's (A, B, D) formats from its suffix: <D>_16x16x16_<A>[_<B>].
    let fmt = |s: &str| -> Result<Format, String> {
        Ok(match s {
            "f32" => F32,
            "f16" => F16,
            "bf16" => BF16,
            "fp8" => FP8,
            "bf8" => BF8,
            _ => return Err(format!("{name}: format {s}")),
        })
    };
    let parts: Vec<&str> = name.split('_').collect();
    let (d, a_fmt) = (fmt(parts[0])?, fmt(parts[2])?);
    let b_fmt = fmt(parts.get(3).copied().unwrap_or(parts[2]))?;
    // A value the format holds exactly, as its bits.
    let code = |x: f64, f: Format| -> Result<u32, String> {
        let small = |s: Small| {
            (0..256u32)
                .find(|&k| s.value(k).to_bits() == x.to_bits())
                .ok_or_else(|| format!("{name}: {x} not an 8-bit value"))
        };
        match f {
            F32 => Ok((x as f32).to_bits()),
            F16 => Ok(f16_one(x)),
            BF16 => Ok((x as f32).to_bits() >> 16),
            FP8 => small(Small::E4M3),
            _ => small(Small::E5M2),
        }
    };
    let codes = |key: &str, f: Format| -> Result<Vec<u32>, String> {
        c[key]
            .as_array()
            .into_iter()
            .flatten()
            .map(|v| code(v.as_f64().ok_or("number")?, f))
            .collect()
    };
    let (a, b) = (codes("a", a_fmt)?, codes("b", b_fmt)?);
    let cv = code(c["c"].as_f64().ok_or("c")?, d)?;
    let want = code(c["d"].as_f64().ok_or("d")?, d)?;
    let got = wmma_element(&a, &b, cv, Step::new(a_fmt, b_fmt, d));
    if got != want {
        return Err(format!(
            "{name} {}: {got:#x}, want {want:#x} ({})",
            c["probe"], c["d"]
        ));
    }
    Ok(())
}

fn main() {
    let mut suite = Suite::new("emu");
    suite.section("reading.json", "cases", reading_case);
    suite.section("claims.json", "cases", claims_case);
    suite.section("alu.json", "vector", vector_case);
    suite.section("alu.json", "scalar", scalar_case);
    suite.section("tags.json", "cases", tags_case);
    suite.section("lds.json", "cases", lds_case);
    suite.section("flight.json", "cases", flight_case);
    suite.section("scope.json", "cases", scope_case);
    suite.whole("numbers.json.gz", numbers);
    suite.section("banks.json", "cases", banks);
    suite.section("wmma.json.gz", "cases", wmma_case);
    suite.finish();
}
