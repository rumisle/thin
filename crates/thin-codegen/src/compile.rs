//! Compile built kernels into artifacts, assembly and resource summaries.

use std::collections::BTreeMap;
use std::sync::Arc;

use thin_dsl::args::Args;
use thin_dsl::Builder;
use thin_ir::Snapshot;
use thin_passes::{lds, ranges};
use thin_target::{Footer, Kind};

use crate::abi::{self, Line};
use crate::json::Json;
use crate::manifest;
use crate::pipeline::{self, records, refused, Options, FINISH, POST_PLACE, RENDER};
use crate::spell::{of_dsl, Form, Refusal, Result, Spell};

/// `finish(kernel, nops)`: FINISH, then POST_PLACE with `nops`.
pub fn finish(b: &mut Builder, o: &Options, nops: bool) -> Result<()> {
    b.seal().map_err(of_dsl)?;
    pipeline::run(b, &FINISH, o, false, false)?;
    if nops {
        pipeline::run(b, &POST_PLACE, o, false, false)?;
    }
    Ok(())
}

/// `rendered(kernel)`: the lines and the snapshot RENDER read.
pub fn rendered(b: &mut Builder, o: &Options) -> Result<(Vec<String>, Arc<Snapshot>)> {
    b.settle().map_err(of_dsl)?;
    if !crate::place::placed(b) {
        crate::place::place(b, o, false)?;
    }
    let snap = pipeline::run(b, &RENDER, o, true, true)?;
    let snap = match snap {
        Some(s) => s,
        None => pipeline::snapshot(b)?,
    };
    let mut lines: Vec<String> = match b.layout.as_mut() {
        Some(l) => {
            l.freeze().map_err(of_dsl)?;
            l.table().into_iter().map(|t| format!("; {t}")).collect()
        }
        None => Vec::new(),
    };
    lines.extend(Spell::new(b).body(snap.instrs())?);
    Ok((lines, snap))
}

/// What `compile_kernel` takes beside the kernel.
pub struct Call<'a> {
    /// The symbol.
    pub symbol: &'a str,
    /// Lanes per workgroup.
    pub wgsize: u32,
    /// The argument block, when the kernel has no entry.
    pub args: Option<Args>,
    /// The LDS bytes given.
    pub lds: u64,
    /// The caller's `build`.
    pub build: Json,
    /// The caller's `grid`.
    pub grid: Vec<u64>,
    /// The caller's `max_grid`.
    pub max_grid: Vec<u64>,
}

/// compile_kernel's result.
pub struct Artifact {
    /// The `.s`, its last line the manifest.
    pub text: String,
    /// The manifest record.
    pub manifest: Json,
    /// The summary.
    pub summary: String,
}

/// The ranges launch of a block.
fn launch(args: &Args, wgsize: u32, wgy: u32, max_grid: &[u64], preload: u32) -> ranges::Launch {
    let arg = |a: &thin_dsl::args::Arg| ranges::Arg {
        offset: a.offset as u32,
        bytes: a.bytes as u32,
        is_u32: a.kind == "u32",
        lo: a.lo,
        hi: a.hi,
        divisor: a.divisor,
    };
    ranges::Launch {
        args: args.args.iter().map(arg).collect(),
        wgsize: u64::from(wgsize),
        wgy,
        max_grid: max_grid.to_vec(),
        preload,
        tid: None,
    }
}

/// Value ranges, per wave ranges and races: the whole
/// workgroup's report.
fn ranges_of(b: &mut Builder, s: &Arc<Snapshot>, l: &ranges::Launch) -> Result<ranges::Report> {
    if s.instrs().is_empty() {
        return Ok(ranges::Report::default());
    }
    let report = pipeline::timed(b, "ranges", |b| {
        ranges::ranges(s, l).map_err(|d| refused(b, d, s, Form::Named))
    })?;
    // Races need two waves of the target's width.
    let wave = u64::from(s.target().definition().wave);
    if b.layout.is_none() || l.wgsize < 2 * wave {
        return Ok(report);
    }
    let waves = pipeline::timed(b, "ranges, each wave", |b| {
        (ranges::per_wave(s, l).into_iter())
            .map(|r| r.map_err(|d| refused(b, d, s, Form::Named)))
            .collect::<Result<Vec<_>>>()
    })?;
    let proof = records(b).proof.clone();
    let other = !proof
        .as_ref()
        .is_some_and(|(_, len)| *len == s.instrs().len());
    let skip = !b.ablate.is_empty() || waves.len() < 2 || other;
    if !skip {
        let bytes: Vec<lds::WaveBytes> = (waves.iter())
            .map(|w| {
                w.lds_bytes
                    .iter()
                    .map(|(&k, &(lo, hi))| (k, (lo as u64, hi as u64)))
                    .collect()
            })
            .collect();
        let (p, _) = proof.expect("a proof");
        pipeline::timed(b, "LDS races", |b| {
            lds::races(s, &p, &bytes).map_err(|d| refused(b, d, s, Form::Bare))
        })?;
    }
    Ok(report)
}

/// A file's report lines.
fn report_lines(b: &mut Builder) -> Vec<String> {
    let reports = records(b).reports.clone();
    (reports.iter())
        .filter(|r| r.report.values > 0)
        .map(|r| {
            let x = &r.report;
            format!(
                "{} derived: {} registers for a peak of {} live at `{}` ({} values; {:.2} s)",
                x.file, x.used, x.peak, r.at, x.values, r.seconds
            )
        })
        .collect()
}

/// The kernel name rule of the `.s`: the code-object step's.
fn check_name(name: &str) -> Result<()> {
    if crate::object::is_symbol(name) {
        return Ok(());
    }
    Err(Refusal::new(
        "Value",
        format!(
            "kernel name {}: the assembler takes [A-Za-z_.][A-Za-z0-9_.$]* only",
            thin_ir::pyfmt::repr_str(name)
        ),
    ))
}

/// What the `.s` file's footer and budget read.
pub struct Facts {
    /// LDS bytes.
    pub lds: u64,
    /// The block's size.
    pub kernarg: u64,
    /// Files' used.
    pub vgpr: u32,
    /// AGPRs.
    pub agpr: u32,
    /// SGPRs.
    pub sgpr: u32,
    /// Lanes per workgroup.
    pub wgsize: u32,
    /// Workgroup id y.
    pub wgy: u32,
    /// Preloaded dwords.
    pub preload: u32,
}

/// Whether a body already ends the program: its last
/// instruction line, after every comment, claim and empty line, is
/// `s_endpgm` or an unconditional `s_branch`, with no label line after it.
pub fn body_ends(body: &[String]) -> bool {
    for line in body.iter().rev() {
        let code = line.split("//").next().unwrap_or("").trim();
        if code.is_empty() || code.starts_with(';') {
            continue; // an empty line, a comment or a claim
        }
        if code.ends_with(':') {
            return false; // a label: a branch may reach the end
        }
        let op = code.split_whitespace().next().unwrap_or("");
        return op == "s_endpgm" || op == "s_branch";
    }
    false
}

/// `render_file`, of a kernel within budget.
pub fn render_file(b: &Builder, name: &str, body: &[String], f: &Facts) -> Result<String> {
    let t = b.t.name();
    let header = format!(
        "\t.amdgcn_target \"amdgcn-amd-amdhsa--{t}\"\n\t.amdhsa_code_object_version 6\n\t.text\n\t.protected\t{name}\n\
         \t.globl\t{name}\n\t.p2align\t8\n\t.type\t{name},@function\n{name}:\n"
    );
    let footer = Footer {
        name: name.to_owned(),
        target_name: format!("amdgcn-amd-amdhsa--{t}"),
        lds: f.lds as u32,
        kernarg: f.kernarg as u32,
        vgpr: f.vgpr,
        agpr: f.agpr,
        sgpr: f.sgpr,
        wgsize: f.wgsize,
        wgy: f.wgy,
        preload: f.preload,
        ends: body_ends(body),
    };
    let footer =
        b.t.hooks()
            .footer(&footer)
            .map_err(|e| Refusal::new("Value", e.to_string()))?;
    Ok(format!("{header}{}\n{footer}", body.join("\n")))
}

/// The final stream as the mix and the delay notes read it.
/// An instruction as the mix and the notes read it: (kind, mnemonic,
/// label name, branch target, delay hint).
type Read = (String, String, Option<String>, Option<String>, i128);

fn lines_of(b: &Builder) -> Vec<Read> {
    (b.rows.iter())
        .map(|r| {
            let i = &r.ins;
            let f = i.form();
            let op = b.text(i).split_whitespace().next().unwrap_or("").to_owned();
            let label = i
                .field_named("name")
                .and_then(|v| v.text())
                .filter(|_| f.kind == Kind::Label)
                .map(str::to_owned);
            let target = i.ops().iter().find_map(|o| match o {
                thin_ir::Operand::Block(name) => Some(name.to_string()),
                _ => None,
            });
            let delay = Some(op.as_str()) == b.t.definition().hazards.delay_hint;
            let hint = match i.ops().first().filter(|_| delay) {
                Some(thin_ir::Operand::Imm(thin_ir::Imm {
                    value: thin_ir::ImmValue::Int(v),
                    ..
                })) => *v,
                _ => 0,
            };
            (f.kind.name().to_owned(), op, label, target, hint)
        })
        .collect()
}

fn ints(xs: &[u64]) -> Json {
    Json::List(xs.iter().map(|&x| Json::Int(i128::from(x))).collect())
}

/// An argument's record.
fn arg_json(a: &thin_dsl::args::Arg) -> Json {
    let mut m = BTreeMap::new();
    let opt_str = |s: &Option<String>| s.clone().map_or(Json::Null, Json::Str);
    m.insert("name".into(), Json::Str(a.name.clone()));
    m.insert("offset".into(), Json::Int(i128::from(a.offset)));
    m.insert("bytes".into(), Json::Int(i128::from(a.bytes)));
    m.insert("kind".into(), Json::Str(a.kind.clone()));
    m.insert("lo".into(), a.lo.map_or(Json::Null, Json::Int));
    m.insert("hi".into(), a.hi.map_or(Json::Null, Json::Int));
    m.insert("divisor".into(), Json::Bool(a.divisor));
    m.insert("multiple".into(), Json::Int(a.multiple));
    m.insert("owned".into(), opt_str(&a.owned));
    m.insert("library".into(), opt_str(&a.library));
    m.insert("mem".into(), opt_str(&a.mem));
    m.insert(
        "ranks".into(),
        a.ranks.map_or(Json::Null, |r| Json::Int(i128::from(r))),
    );
    manifest::arg(&Json::Map(m))
}

/// The kernel's needs as the manifest writes them.
fn needs_json(b: &Builder) -> Json {
    let mut m = BTreeMap::new();
    if let Some(why) = &b.needs.co_resident {
        m.insert("co_resident".into(), Json::Str(why.clone()));
    }
    if !b.needs.same_grid.is_empty() {
        let g = b
            .needs
            .same_grid
            .iter()
            .map(|(k, v)| (k.clone(), Json::Str(v.clone())))
            .collect();
        m.insert("same_grid".into(), Json::Map(g));
    }
    Json::Map(m)
}

/// The phases field: per placed file with a nonzero `used`.
fn phases_json(b: &mut Builder) -> Json {
    let mut m = BTreeMap::new();
    for r in records(b).reports.iter().filter(|r| r.report.used > 0) {
        let mut p = BTreeMap::new();
        p.insert("phase".into(), Json::Str("(derived)".into()));
        p.insert("base".into(), Json::Int(0));
        p.insert("peak".into(), Json::Int(i128::from(r.report.peak)));
        p.insert("size".into(), Json::Null);
        p.insert("used".into(), Json::Int(i128::from(r.report.used)));
        p.insert("at".into(), Json::Str(r.at.clone()));
        m.insert(r.report.file.to_string(), Json::List(vec![Json::Map(p)]));
    }
    Json::Map(m)
}

/// `compile_kernel(kernel, …)`; the argument validation is the
/// caller's (it compares objects).
pub fn compile(b: &mut Builder, c: Call<'_>, o: &Options) -> Result<Artifact> {
    check_name(c.symbol)?;
    let entry = b
        .entered
        .as_ref()
        .map(|k| (k.args.clone(), k.preload, k.wg));
    let (args, preload, wgy) = match entry {
        Some((a, preload, wg)) => (a, preload, u32::from(wg == 2)),
        None => (c.args.clone().unwrap_or_default(), 0, 0),
    };
    let (body, snap) = rendered(b, o)?;
    let lds = match b.layout.as_ref() {
        Some(l) if c.lds != 0 && c.lds != l.end => {
            return Err(Refusal::new(
                "Schedule",
                format!(
                    "compile_kernel(lds={}): the kernel's LDS is its layout's {} B (b.lds)",
                    c.lds, l.end
                ),
            ))
        }
        Some(l) => l.end,
        None => c.lds,
    };
    let report = ranges_of(
        b,
        &snap,
        &launch(&args, c.wgsize, wgy, &c.max_grid, preload),
    )?;
    let used = |f: char| b.file(f).used(&b.values);
    let (vgpr, sgpr, agpr) = (used('v'), used('s'), used('a'));
    // The budget, once: with the derived files' peaks when placed.
    let lines = report_lines(b);
    if let Err(m) = abi::check_budget(b.t, c.symbol, lds, vgpr, agpr, sgpr, c.wgsize) {
        if lines.is_empty() {
            return Err(Refusal::new("Schedule", m));
        }
        let tail = "\n  (options.peak_report: the values live at the peak, widest first)";
        return Err(Refusal::new(
            "Schedule",
            format!("{m}\n  {}{tail}", lines.join("\n  ")),
        ));
    }
    let facts = Facts {
        lds,
        kernarg: args.size(),
        vgpr,
        agpr,
        sgpr,
        wgsize: c.wgsize,
        wgy,
        preload,
    };
    let file = render_file(b, c.symbol, &body, &facts)?;
    let res = abi::residency(b.t, lds, vgpr, agpr, c.wgsize);
    let stream = lines_of(b);
    let kinds: Vec<(&str, &str)> = stream
        .iter()
        .map(|x| (x.0.as_str(), x.1.as_str()))
        .collect();
    let mix = abi::mix(b.t, &kinds);
    let hint = b.t.definition().hazards.delay_hint;
    let notes_in: Vec<Line<'_>> = (stream.iter())
        .map(|x| Line {
            kind: &x.0,
            op: &x.1,
            label: x.2.as_deref(),
            target: x.3.as_deref(),
            hint: x.4,
            delay: Some(x.1.as_str()) == hint,
        })
        .collect();
    let notes = abi::notes(b.t, &res, &mix, &notes_in);
    let mut m = BTreeMap::new();
    let int = |x: u64| Json::Int(i128::from(x));
    m.insert("symbol".into(), Json::Str(c.symbol.to_owned()));
    m.insert("kernarg_bytes".into(), int(args.size()));
    m.insert("lds".into(), int(lds));
    m.insert("wgsize".into(), int(u64::from(c.wgsize)));
    m.insert("wgy".into(), int(u64::from(wgy)));
    m.insert("vgpr".into(), int(u64::from(vgpr)));
    m.insert("agpr".into(), int(u64::from(agpr)));
    m.insert("sgpr".into(), int(u64::from(sgpr)));
    m.insert(
        "args".into(),
        Json::List(args.args.iter().map(arg_json).collect()),
    );
    m.insert("build".into(), c.build.clone());
    m.insert("resident".into(), int(u64::from(res.resident)));
    m.insert(
        "mix".into(),
        Json::Map(mix.iter().map(|(k, v)| (k.clone(), int(*v))).collect()),
    );
    m.insert(
        "notes".into(),
        Json::List(notes.iter().cloned().map(Json::Str).collect()),
    );
    m.insert("grid".into(), ints(&c.grid));
    m.insert("phases".into(), phases_json(b));
    m.insert("max_grid".into(), ints(&c.max_grid));
    m.insert("needs".into(), needs_json(b));
    let record = Json::Map(m);
    let n: u64 = mix.values().sum();
    let wgs = if res.resident == 1 { "WG" } else { "WGs" };
    let lds_part = if lds > 0 {
        format!(", lds {lds}")
    } else {
        String::new()
    };
    let mut summary = vec![format!(
        "{}: {n} instructions, vgpr {vgpr} agpr {agpr} sgpr {sgpr}{lds_part}, {} {wgs} per CU",
        c.symbol, res.resident
    )];
    summary.extend(notes.iter().map(|x| format!("  note: {x}")));
    summary.extend(lines.iter().map(|x| format!("  {x}")));
    summary.extend(records(b).final_report.iter().map(|x| format!("  {x}")));
    summary.push(format!("  {}", report.line()));
    let text = format!("{file}{}", manifest::line(&record));
    Ok(Artifact {
        text,
        manifest: record,
        summary: summary.join("\n"),
    })
}
