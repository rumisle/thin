//! Golden checks for compiler passes on gfx1201 instruction streams.

use serde_json::Value;
use thin_ir::live::{Exec, Lane, Rows};
use thin_passes::{checks, delay_text, hazards, lds, ranges, regalloc, waits};
use thin_test_support::adapters::default_target;
use thin_test_support::asm;
use thin_test_support::golden_tests::expect_eq;
use thin_test_support::listing::Listing;
use thin_test_support::{Case, Suite};

/// The strings of a JSON list.
fn strs(v: &Value) -> Vec<&str> {
    v.as_array()
        .map_or(Vec::new(), |a| a.iter().filter_map(Value::as_str).collect())
}

/// `hazards.json`: applied to needs, and empty needs on a
/// re-run unless the case says otherwise.
fn hazards_case(case: &Case<'_>) -> Result<(), String> {
    let t = default_target();
    let model = thin_targets::hazards(t.name()).map_err(|e| e.to_string())?;
    let s = asm::snapshot(
        t,
        hazards::strip(&asm::read(t, &strs(&case.value["input"]))?),
    );
    let placed = hazards::apply(&s, &hazards::needs(&s, model));
    let lines: Vec<String> = (placed.iter())
        .map(|(i, origin)| {
            let line = asm::spell(&s, i);
            match (origin, i.form().op.as_str()) {
                (None, _) => line,
                (Some(_), "s_delay_alu") => {
                    format!("+ {line}    ; hazard: {}", delay_text(delay_of(i)))
                }
                (Some(_), _) => format!("+ {line}"),
            }
        })
        .collect();
    expect_eq(
        lines,
        strs(&case.value["output"])
            .into_iter()
            .map(str::to_owned)
            .collect(),
    )?;
    let rerun = asm::snapshot(t, placed.into_iter().map(|(i, _)| i).collect());
    let clean = hazards::needs(&rerun, model)
        .iter()
        .all(thin_passes::Need::is_empty);
    expect_eq(!clean, case.value["rerun_needs_more"] == true)
}

/// An `s_delay_alu`'s operand.
fn delay_of(i: &thin_ir::Instr) -> u32 {
    match i.ops().first() {
        Some(thin_ir::Operand::Imm(thin_ir::Imm {
            value: thin_ir::ImmValue::Int(v),
            ..
        })) => *v as u32,
        _ => 0,
    }
}

/// `checks.json`: one stream check's verdict.
fn checks_case(case: &Case<'_>) -> Result<(), String> {
    let t = default_target();
    let s = asm::snapshot(t, asm::read(t, &strs(&case.value["input"]))?);
    let entries: Vec<usize> = (case.value["entries"].as_array().into_iter().flatten())
        .filter_map(Value::as_u64)
        .map(|k| k as usize)
        .collect();
    let verdict = match case.value["check"].as_str().unwrap_or("") {
        "dead writes" => checks::dead_writes(&s),
        "SCC readers" => checks::scc_readers(&s),
        "save-less regions" => checks::saveless(&s, &entries),
        other => return Err(format!("no check {other}")),
    };
    let got = verdict.err().map(|d| asm::message(&s, &d.msg));
    expect_eq(got, case.value["error"].as_str().map(str::to_owned))
}

/// A wait count key as `waits.json` writes it.
fn key_text(t: &thin_target::Target, key: waits::Key) -> String {
    let name = |c: usize| t.definition().counters[c].name;
    match key {
        waits::Key::Marker(k) => k.to_string(),
        waits::Key::Use(k, c) => format!("use:{k}:{}", name(c)),
        waits::Key::Pub(k, c) => format!("pub:{k}:{}", name(c)),
    }
}

/// A source without its loop line, which names the loop's options.
fn unlooped<'a>(source: &[&'a str]) -> Vec<&'a str> {
    source
        .iter()
        .copied()
        .filter(|l| !l.starts_with("loop "))
        .collect()
}

/// A pipeline case's stream: its output's lines with each wait replaced
/// by the marker of the source's `@wait vN` in turn (taken from the
/// sibling case of the same source when this one is refused), and its
/// loop, the source's indented body.
fn pipeline_input(case: &Case<'_>) -> Result<(Vec<String>, waits::Loop), String> {
    let source = strs(&case.value["source"]);
    let output = match case.value.get("output") {
        Some(o) => strs(o),
        None => (case.file["cases"].as_array().ok_or("cases")?.iter())
            .find(|c| {
                c.get("output").is_some() && unlooped(&strs(&c["source"])) == unlooped(&source)
            })
            .map(|c| strs(&c["output"]))
            .ok_or("no sibling with an output")?,
    };
    let mut regs = source
        .iter()
        .filter_map(|l| l.trim().strip_prefix("@wait "));
    let lines: Vec<String> = (output.iter())
        .map(|l| match l.strip_prefix("s_wait_loadcnt ") {
            Some(_) => format!("@wait load regs {}", regs.next().unwrap_or("")),
            None => (*l).to_owned(),
        })
        .collect();
    let head = source
        .iter()
        .find(|l| l.starts_with("loop "))
        .ok_or("a loop")?;
    let body: Vec<&str> = source
        .iter()
        .filter(|l| l.starts_with("  "))
        .map(|l| l.trim())
        .collect();
    let first = lines
        .iter()
        .position(|l| l.starts_with("@wait"))
        .ok_or("a marker in the body")?;
    let end = *body.last().ok_or("a body")?;
    let last = lines
        .iter()
        .position(|l| l == end)
        .ok_or("the body's end")?;
    let loop_ = waits::Loop {
        stem: "k".into(),
        first,
        last,
        conservative: head.contains("conservative=True"),
    };
    Ok((lines, loop_))
}

/// `waits.json`: the counts, their uniqueness, and the stream or refusal.
fn waits_case(case: &Case<'_>) -> Result<(), String> {
    let t = default_target();
    let (lines, loops) = if case.value["pipeline"] == true {
        let (lines, l) = pipeline_input(case)?;
        (lines, vec![l])
    } else {
        (
            strs(&case.value["input"])
                .into_iter()
                .map(str::to_owned)
                .collect(),
            Vec::new(),
        )
    };
    let refs: Vec<&str> = lines.iter().map(String::as_str).collect();
    let s = asm::snapshot(t, asm::read(t, &refs)?);
    let w = waits::Waits::new(&s);
    if let Some(want) = case.value.get("counts") {
        let counts = w.solve().map_err(|d| asm::message(&s, &d.msg))?;
        let got: serde_json::Map<String, Value> = counts
            .iter()
            .map(|(&k, v)| (key_text(t, k), serde_json::json!(v)))
            .collect();
        expect_eq(Value::Object(got), want.clone())?;
        expect_eq(w.f(&counts) == counts, true).map_err(|_| "the counts are no fixpoint of F")?;
        let unique = w.bounds().is_some_and(|(a, b)| a == b);
        expect_eq(unique, case.value["unique"] == true)?;
    }
    // The pipeline cases spell a marker's source location as a placeholder.
    let place = if case.value["pipeline"] == true {
        " <the wait's source location>"
    } else {
        ""
    };
    match waits::pass(&s, &loops, waits::Options::default()) {
        Err(d) => {
            let line = |k: usize| asm::spell(&s, &s.instrs()[k]);
            let got = thin_ir::Diag::spell(&d.msg, &line, Some(&|_| place.to_owned()));
            expect_eq(Some(got), case.value["error"].as_str().map(str::to_owned))
        }
        Ok(placed) => {
            let mut stream: Vec<std::sync::Arc<thin_ir::Instr>> =
                placed.stream.into_iter().map(|p| p.0).collect();
            if case.value["pipeline"] == true {
                let model = thin_targets::hazards(t.name()).map_err(|e| e.to_string())?;
                let s2 = asm::snapshot(t, stream);
                stream = hazards::apply(&s2, &hazards::needs(&s2, model))
                    .into_iter()
                    .map(|p| p.0)
                    .collect();
            }
            let got: Vec<String> = stream.iter().map(|i| asm::spell(&s, i)).collect();
            let want: Vec<String> = strs(&case.value["output"])
                .into_iter()
                .map(str::to_owned)
                .collect();
            expect_eq(
                Some(got),
                (case.value.get("error").is_none()).then_some(want),
            )
        }
    }
}

/// A kernel of named groups (alias.json's notation): each group one value
/// at its first register, pinned unless it says otherwise; every plain
/// register spelling (`v40`, `s[0:1]`) one more pinned group; the entry
/// groups and the pinned ones written by the hardware at entry.
struct Groups {
    names: Vec<String>,
    gather: thin_ir::Gather,
}

impl Groups {
    fn new(case: &Value) -> Result<Groups, String> {
        let mut g = Groups {
            names: Vec::new(),
            gather: thin_ir::Gather::default(),
        };
        for (name, spec) in case["groups"].as_object().ok_or("groups")? {
            let file = spec[0]
                .as_str()
                .and_then(|f| f.chars().next())
                .ok_or("a file")?;
            let (first, n) = (
                spec[1].as_u64().ok_or("a register")? as u32,
                spec[2].as_u64().ok_or("dwords")? as u32,
            );
            g.add(
                name,
                file,
                first,
                n,
                spec.get(3) != Some(&Value::Bool(false)),
            );
        }
        for line in strs(&case["source"]) {
            for token in line.split([' ', ',']).filter(|t| !t.starts_with('%')) {
                if let Some(r) = asm::register(token) {
                    if !g.names.iter().any(|n| n == token) {
                        g.add(token, r.file, r.lo, r.n, true);
                    }
                }
            }
        }
        let entry: Vec<String> = strs(&case["entry"])
            .into_iter()
            .map(str::to_owned)
            .collect();
        for letter in ['v', 's'] {
            let of = |f: &thin_ir::GatherValue| f.file == letter;
            let pinned = g
                .gather
                .values
                .iter()
                .filter(|v| of(v) && v.pin.is_some())
                .map(|v| v.name.clone())
                .collect();
            let entered = entry
                .iter()
                .filter(|e| g.gather.values.iter().any(|v| of(v) && &v.name == *e))
                .cloned()
                .collect();
            g.gather.files.push(thin_ir::GatherFile {
                letter,
                entered,
                pinned,
                placed: true,
                ..Default::default()
            });
        }
        Ok(g)
    }

    fn add(&mut self, name: &str, file: char, first: u32, n: u32, pinned: bool) {
        self.names.push(name.to_owned());
        let value = thin_ir::GatherValue {
            file,
            n,
            align: 1,
            name: name.to_owned(),
            pin: pinned.then_some(first),
            register: Some(first),
            ..Default::default()
        };
        self.gather.values.push(value);
    }

    /// The register a token names: `%name` its group whole, a plain one its
    /// own group.
    fn register(&self, token: &str) -> Option<thin_ir::Reg> {
        let name = token.strip_prefix('%').unwrap_or(token);
        let k = self.names.iter().position(|n| n == name)?;
        let v = &self.gather.values[k];
        Some(thin_ir::Reg::new(k as i32, v.file, 0, v.n))
    }

    /// The source's instructions: `(alias_ok R[ over W]) ` before a line
    /// waives a read.
    fn read(
        &self,
        t: &'static thin_target::Target,
        source: &[&str],
    ) -> Result<Vec<std::sync::Arc<thin_ir::Instr>>, String> {
        let resolve = |tok: &str| self.register(tok);
        source
            .iter()
            .map(|line| {
                let (waiver, text) = match line
                    .strip_prefix("(alias_ok ")
                    .and_then(|r| r.split_once(") "))
                {
                    Some((w, text)) => (Some(w), text),
                    None => (None, *line),
                };
                let mut i = asm::line_with(t, text, &resolve)?;
                if let Some(w) = waiver {
                    let (r, writer) = w
                        .split_once(" over ")
                        .map_or((w, None), |(r, o)| (r, Some(o.into())));
                    i = i
                        .with_attr(thin_ir::Attr::AliasOk(vec![(r.into(), writer)]))
                        .map_err(|e| e.to_string())?;
                }
                Ok(std::sync::Arc::new(i))
            })
            .collect()
    }
}

/// `alias.json`: the findings, each spelled as its message.
fn alias_case(case: &Case<'_>) -> Result<(), String> {
    let t = default_target();
    let g = Groups::new(case.value)?;
    let instrs = g.read(t, &strs(&case.value["source"]))?;
    let s = thin_ir::Snapshot::of_kernel(t, &g.gather, instrs.into()).map_err(|e| e.to_string())?;
    let spelled: Vec<String> = s.instrs().iter().map(|i| asm::spell(&s, i)).collect();
    expect_eq(
        spelled.iter().map(String::as_str).collect(),
        strs(&case.value["input"]),
    )?;
    let got: Vec<String> = thin_passes::alias::findings(&s)
        .iter()
        .map(|m| asm::message(&s, m))
        .collect();
    expect_eq(
        got,
        strs(&case.value["findings"])
            .into_iter()
            .map(str::to_owned)
            .collect(),
    )
}

/// An interval as `ranges.json` writes it.
fn interval_json(iv: &ranges::Interval) -> Value {
    serde_json::json!({"lo": iv.lo as i64, "hi": iv.hi as i64, "wrap": iv.wrap, "why": iv.why, "at": iv.at})
}

/// `ranges.json` `transfer`: one op's result on its operands' intervals.
fn transfer_case(case: &Case<'_>) -> Result<(), String> {
    let op = case.value["op"].as_str().ok_or("an op")?;
    let form = default_target()
        .named(op)
        .ok_or_else(|| format!("no op {op}"))?;
    let ops: Vec<Option<ranges::Interval>> =
        (case.value["operands"].as_array().ok_or("operands")?.iter())
            .map(|o| {
                Some(ranges::Interval::new(
                    o[0].as_i64()? as i128,
                    o[1].as_i64()? as i128,
                ))
            })
            .collect();
    let k = case.value["k"].as_u64().ok_or("k")? as usize;
    let got = ranges::transfer(&form.canon, &ops, k).map_or(Value::Null, |iv| interval_json(&iv));
    expect_eq(got, case.value["result"].clone())
}

/// A launch as `ranges.json` writes it.
fn launch_of(v: &Value) -> ranges::Launch {
    let int = |x: &Value| x.as_i64().map(i128::from);
    let args = (v["args"].as_array().into_iter().flatten())
        .map(|a| ranges::Arg {
            offset: a[0].as_u64().unwrap_or(0) as u32,
            bytes: a[1].as_u64().unwrap_or(0) as u32,
            is_u32: a[2] == true,
            lo: int(&a[3]),
            hi: int(&a[4]),
            divisor: a[5] == true,
        })
        .collect();
    ranges::Launch {
        args,
        wgsize: v["wgsize"].as_u64().unwrap_or(0),
        wgy: v["wgy"].as_u64().unwrap_or(0) as u32,
        max_grid: v["max_grid"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(Value::as_u64)
            .collect(),
        preload: v["preload"].as_u64().unwrap_or(0) as u32,
        tid: v["tid"]
            .as_array()
            .map(|t| (int(&t[0]).unwrap_or(0), int(&t[1]).unwrap_or(0))),
    }
}

/// `ranges.json` `cases`: the report or the refusal.
fn ranges_case(case: &Case<'_>) -> Result<(), String> {
    let t = default_target();
    let s = asm::snapshot(t, asm::read(t, &strs(&case.value["input"]))?);
    match ranges::ranges(&s, &launch_of(&case.value["launch"])) {
        Ok(r) => {
            let bytes: serde_json::Map<String, Value> = r
                .lds_bytes
                .iter()
                .map(|(k, (lo, hi))| (k.to_string(), serde_json::json!([*lo as i64, *hi as i64])))
                .collect();
            let got = serde_json::json!({"proven": r.proven, "unproven": r.unproven, "inside": r.inside,
                "unknown": r.unknown, "lds_bytes": bytes});
            expect_eq(Some(got), case.value.get("report").cloned())
        }
        Err(d) => expect_eq(
            Some(asm::message(&s, &d.msg)),
            case.value["error"].as_str().map(str::to_owned),
        ),
    }
}

/// A lane class as `regalloc.json` writes it: an int mask, `["region", r]`,
/// `"cur"`, or null.
fn lane_of(v: &Value) -> Option<Lane> {
    match v {
        Value::Number(m) => m.as_u64().map(Lane::Mask),
        Value::Array(a) => a
            .get(1)
            .and_then(Value::as_i64)
            .map(|r| Lane::Region(r.into())),
        Value::String(c) if c == "cur" => Some(Lane::Cur),
        _ => None,
    }
}

/// EXEC as `regalloc.json` writes it.
fn exec_of(v: &Value) -> Exec {
    match lane_of(v) {
        Some(Lane::Mask(m)) => Exec::Mask(m),
        Some(Lane::Region(r)) => Exec::Region(r),
        _ => Exec::Unknown,
    }
}

/// A list of lists of keys.
fn key_lists(v: &Value) -> Vec<Vec<usize>> {
    (v.as_array().into_iter().flatten()).map(keys).collect()
}

fn keys(v: &Value) -> Vec<usize> {
    (v.as_array().into_iter().flatten())
        .filter_map(Value::as_u64)
        .map(|k| k as usize)
        .collect()
}

fn list<T>(v: &Value, f: impl Fn(&Value) -> T) -> Vec<T> {
    v.as_array().into_iter().flatten().map(f).collect()
}

/// `regalloc.json` `solve`: occupancy on facts given as lists.
fn solve_case(case: &Case<'_>) -> Result<(), String> {
    let v = case.value;
    let mut live = Rows::new(key_lists(&v["succ"]), Vec::new());
    live.reads = key_lists(&v["reads"]);
    live.read_lanes = list(&v["read_lanes"], |l| lane_of(l).unwrap_or(Lane::Cur));
    live.every = key_lists(&v["reads_every_lane"]);
    live.kills = key_lists(&v["kills"]);
    live.lane_kills = key_lists(&v["lane_kills"]);
    live.exec_in = list(&v["exec_in"], exec_of);
    live.writes_exec = list(&v["writes_exec"], |b| b == true);
    live.exec_out = list(&v["exec_out"], exec_of);
    let facts = regalloc::Facts {
        live,
        writes: key_lists(&v["writes"]),
        pinned: keys(&v["pinned"]),
        held: list(&v["held"], |h| {
            let mut b = thin_ir::sets::Bits::new();
            b.add(keys(h));
            b
        }),
        window_ends: list(&v["window_ends"], |b| b == true),
    };
    let got = regalloc::solve(&facts);
    let occupied: Vec<Vec<usize>> = got.occupied.iter().map(thin_ir::sets::Bits::ones).collect();
    expect_eq(occupied, key_lists(&v["occupied"]))?;
    expect_eq(got.unwritten.ones(), keys(&v["unwritten"]))
}

/// A derived kernel (`regalloc.json`'s notation): `tid` pinned at v0, the
/// declared groups (name, count, width, align) in V, each other `%name`
/// a pinned S value; values at distinct provisional registers.
struct Derived {
    gather: thin_ir::Gather,
}

impl Derived {
    fn new(case: &Value) -> Derived {
        let mut d = Derived {
            gather: thin_ir::Gather::default(),
        };
        d.add('v', "tid", 0, 1, 1, Some(0));
        for decl in case["decl"].as_array().into_iter().flatten() {
            let name = decl[0].as_str().unwrap_or("");
            let int = |n: usize| decl.get(n).and_then(Value::as_u64).unwrap_or(1) as u32;
            (0..int(1)).for_each(|i| d.add('v', name, i, int(2), int(3), None));
        }
        for line in strs(&case["input"]) {
            for token in line.split([' ', ',']).filter_map(|t| t.strip_prefix('%')) {
                let (name, _, n, _) = Derived::token(token);
                if !d.gather.values.iter().any(|v| v.name == name) {
                    let pin = d
                        .gather
                        .values
                        .iter()
                        .filter(|v| v.file == 's')
                        .map(|v| v.n)
                        .sum();
                    d.add('s', name, 0, n.unwrap_or(1), 1, Some(pin));
                }
            }
        }
        let lanes = strs(&case["lanes_covered"])
            .into_iter()
            .map(str::to_owned)
            .collect();
        d.gather.files.push(thin_ir::GatherFile {
            letter: 'v',
            lanes,
            ..Default::default()
        });
        d.gather.files.push(thin_ir::GatherFile {
            letter: 's',
            ..Default::default()
        });
        d
    }

    fn add(&mut self, file: char, name: &str, index: u32, n: u32, align: u32, pin: Option<u32>) {
        let past = 64
            + self
                .gather
                .values
                .iter()
                .filter(|v| v.file == file)
                .map(|v| v.n)
                .sum::<u32>();
        self.gather.values.push(thin_ir::GatherValue {
            file,
            n,
            align,
            name: name.to_owned(),
            pin,
            index,
            register: Some(pin.unwrap_or(past)),
            ..Default::default()
        });
    }

    /// `name[i]/w.d` → (name, i, w, d).
    fn token(t: &str) -> (&str, u32, Option<u32>, Option<u32>) {
        let (head, dword) = t
            .split_once('.')
            .map_or((t, None), |(h, d)| (h, d.parse().ok()));
        let (head, width) = head
            .split_once('/')
            .map_or((head, None), |(h, w)| (h, w.parse().ok()));
        let (name, index) = head
            .strip_suffix(']')
            .and_then(|h| h.split_once('['))
            .map_or((head, 0), |(n, i)| (n, i.parse().unwrap_or(0)));
        (name, index, width, dword)
    }

    fn register(&self, t: &str) -> Option<thin_ir::Reg> {
        let (name, index, _, dword) = Derived::token(t.strip_prefix('%')?);
        let k = self
            .gather
            .values
            .iter()
            .position(|v| v.name == name && v.index == index)?;
        let v = &self.gather.values[k];
        let (lo, n) = dword.map_or((0, v.n), |d| (d, 1));
        Some(thin_ir::Reg::new(k as i32, v.file, lo, n))
    }

    fn snapshot(&self, input: &[&str]) -> Result<thin_ir::Snapshot, String> {
        let t = default_target();
        let resolve = |tok: &str| self.register(tok);
        let instrs: Vec<std::sync::Arc<thin_ir::Instr>> = (input.iter().enumerate())
            .map(|(k, l)| {
                let wait = wait_line(input, k);
                asm::line_with(t, wait.as_deref().unwrap_or(l), &resolve).map(std::sync::Arc::new)
            })
            .collect::<Result<_, _>>()?;
        thin_ir::Snapshot::of_kernel(t, &self.gather, instrs.into()).map_err(|e| e.to_string())
    }
}

/// The wait a `regalloc.json` marker line stands for: `@wait` waits for every
/// VM load, `@wait %v` for v's load, so its count is the loads issued after
/// v's (straight-line input), the older ones retired with it in order.
fn wait_line(input: &[&str], k: usize) -> Option<String> {
    let line = input[k].trim();
    let load = |l: &str| -> Option<()> {
        let op = l.split_whitespace().next().unwrap_or("");
        (op.starts_with("buffer_load") || op.starts_with("global_load")).then_some(())
    };
    if line == "@wait" {
        return Some("s_wait_loadcnt 0".into());
    }
    let v = line.strip_prefix("@wait %")?;
    let writes_v = |l: &str| {
        load(l).is_some()
            && l.split_whitespace()
                .nth(1)
                .is_some_and(|d| d.trim_end_matches(',').trim_start_matches('%') == v)
    };
    let at = (0..k).rev().find(|&j| writes_v(input[j]))?;
    let younger = (at + 1..k).filter(|&j| load(input[j]).is_some()).count();
    Some(format!("s_wait_loadcnt {younger}"))
}

/// A message spelled with registers by name.
fn named_message(s: &thin_ir::Snapshot, parts: &[thin_ir::Part]) -> String {
    let spell = |k: usize| {
        let i = &s.instrs()[k];
        let regs: Vec<String> = (i.ops().iter().filter_map(thin_ir::Operand::reg))
            .map(|r| {
                let v = &s.values()[r.value as usize];
                let group = s
                    .values()
                    .iter()
                    .filter(|o| o.file == v.file && o.name == v.name)
                    .count();
                let named = thin_ir::print::Named {
                    file: v.file,
                    name: s.name(v.name),
                    index: v.index,
                    group,
                    lo: r.lo,
                    n: r.n,
                    value_n: v.n,
                    short: None,
                };
                thin_ir::print::name_text(&named)
            })
            .collect();
        thin_ir::print::line(i, &regs).expect("a well-formed line")
    };
    thin_ir::Diag::spell(parts, &spell, None)
}

/// `regalloc.json` `cases`: the placement and report, or the refusal.
fn regalloc_case(case: &Case<'_>) -> Result<(), String> {
    let t = default_target();
    let s = Derived::new(case.value).snapshot(&strs(&case.value["input"]))?;
    let rules = thin_targets::hazards(t.name())
        .map_err(|e| e.to_string())?
        .rules();
    let o = regalloc::Options::default();
    let p = match regalloc::assign(&s, rules, &[('v', 24)], o) {
        Ok(p) => p,
        Err(d) => {
            return expect_eq(
                Some(named_message(&s, &d.msg)),
                case.value["error"].as_str().map(str::to_owned),
            )
        }
    };
    let placement: serde_json::Map<String, Value> = (s.values().iter().zip(&p.bases))
        .filter(|(v, _)| v.file == 'v')
        .map(|(v, b)| {
            (
                format!("{}[{}]", s.name(v.name), v.index),
                serde_json::json!(b),
            )
        })
        .collect();
    let r = p
        .reports
        .iter()
        .find(|r| r.file == 'v')
        .ok_or("a V report")?;
    let got = serde_json::json!({"placement": placement, "vgprs_used": r.used,
        "report": {"peak": r.peak, "at_index": r.at_index, "used": r.used}});
    let want = serde_json::json!({"placement": case.value["placement"], "vgprs_used": case.value["vgprs_used"],
        "report": case.value["report"]});
    expect_eq(got, want)?;
    let bases = p.bases.clone();
    regalloc::check(&s, rules, &bases, o).map_err(|d| named_message(&s, &d.msg))
}

/// `lives.json` `cases`: the report and each leaf's epochs, or the
/// refusal, on the stream rebuilt from the listing.
fn lives_case(case: &Case<'_>) -> Result<(), String> {
    let l = Listing::new(
        &strs(&case.value["listing"]),
        &strs(&case.value["source"]),
        4,
        case.value["table_lines"].as_u64().map(|n| n as usize),
    )?;
    let o = lds::Options {
        ablate: false,
        wave_order: true,
    };
    let p = match lds::prove(&l.snapshot, &l.layout, o) {
        Ok(p) => p,
        Err(d) => {
            let got = asm::message(&l.snapshot, &d.msg);
            return expect_eq(Some(got), case.value["error"].as_str().map(str::to_owned));
        }
    };
    let epochs: serde_json::Map<String, Value> = (l.layout.leaves.iter().zip(&p.of_leaf))
        .map(|(leaf, es)| {
            let mut names: Vec<&str> = es.iter().map(|&e| p.epochs[e].as_str()).collect();
            names.sort_unstable();
            (leaf.name.clone(), serde_json::json!(names))
        })
        .collect();
    expect_eq(
        serde_json::json!({"report": p.report, "epochs": epochs}),
        serde_json::json!({"report": case.value["report"], "epochs": case.value["epochs"]}),
    )
}

/// `lives.json` `races`: the race check on each wave's bytes.
fn races_case(case: &Case<'_>) -> Result<(), String> {
    let v = case.value;
    let nwaves = (v["wgsize"].as_u64().ok_or("wgsize")?
        / v["wave_lanes"].as_u64().ok_or("wave_lanes")?) as u32;
    let l = Listing::new(&strs(&v["listing"]), &strs(&v["source"]), nwaves, None)?;
    let o = lds::Options {
        ablate: false,
        wave_order: true,
    };
    let p = lds::prove(&l.snapshot, &l.layout, o).map_err(|d| asm::message(&l.snapshot, &d.msg))?;
    let per_wave: Vec<lds::WaveBytes> = list(&v["per_wave_bytes"], |w| {
        (w.as_object().into_iter().flatten())
            .filter_map(|(k, b)| Some((k.parse().ok()?, (b[0].as_u64()?, b[1].as_u64()?))))
            .collect()
    });
    let got = lds::races(&l.snapshot, &p, &per_wave)
        .err()
        .map(|d| asm::message(&l.snapshot, &d.msg));
    expect_eq(got, v["error"].as_str().map(str::to_owned))
}

fn main() {
    let mut suite = Suite::new("passes");
    suite.section("hazards.json", "cases", hazards_case);
    suite.section("waits.json", "cases", waits_case);
    suite.section("checks.json", "cases", checks_case);
    suite.section("alias.json", "cases", alias_case);
    suite.section("ranges.json", "cases", ranges_case);
    suite.section("ranges.json", "transfer", transfer_case);
    suite.section("regalloc.json", "cases", regalloc_case);
    suite.section("regalloc.json", "solve", solve_case);
    suite.section("lives.json", "cases", lives_case);
    suite.section("lives.json", "races", races_case);
    suite.finish();
}
