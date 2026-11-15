//! The ops a corpus of gfx1201 code objects uses, and which of them thin's
//! emulator runs.
//!
//! cargo run --release -p thin-lift --example op_census -- [--json OUT] PATH...
//!
//! PATH: ELF code objects or directories of them (unbundle offload bundles
//! first). Every kernel is decoded; each instruction counts for its
//! mnemonic, and each mnemonic counts the kernels it appears in. An
//! instruction's shape is its mnemonic with its operands' kinds and widths
//! and its modifiers' keys; one instruction of each shape runs alone on the
//! emulator (one wave, zero inputs). A shape is supported when the run ends
//! or faults on its inputs; an emulator limit or a reading error makes it
//! unsupported, with the message. A kernel runs when all its shapes do.

#[path = "../../thin-targets/examples/corpus/mod.rs"]
mod corpus;

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::PathBuf;

use thin_emu::machine::facts::Facts;
use thin_emu::machine::memory::Memory;
use thin_emu::machine::run::Machine;
use thin_emu::{Cause, Tables};
use thin_targets::decode::{MVal, Opd};
use thin_targets::{decoder, Instruction};

/// An instruction's shape: mnemonic, operand kinds, modifier keys.
fn shape(i: &Instruction) -> String {
    let mut s = i.mnemonic.to_owned();
    for (k, o) in i.operands().iter().enumerate() {
        let (neg, abs) = i.signs(k);
        let sign = match (neg, abs) {
            (true, true) => "-|",
            (true, false) => "-",
            (false, true) => "|",
            _ => "",
        };
        let kind = match o {
            Opd::V { count, .. } => format!("v{count}"),
            Opd::A { count, .. } => format!("a{count}"),
            Opd::S { count, .. } => format!("s{count}"),
            Opd::Named(&(name, _, _)) => name.to_owned(),
            Opd::Int(v) if (-16..=64).contains(v) => "inline".into(),
            Opd::Int(_) => "literal".into(),
            Opd::Float(_) => "float".into(),
            Opd::Off => "off".into(),
            Opd::Label(_) => "label".into(),
            Opd::Hwreg { .. } => "hwreg".into(),
        };
        s.push_str(if k == 0 { " " } else { ", " });
        s.push_str(sign);
        s.push_str(&kind);
    }
    for m in i.modifiers() {
        s.push(' ');
        s.push_str(m.key);
        if let MVal::Symbol(name, _) = m.value {
            s.push(':');
            s.push_str(name);
        }
    }
    s
}

/// Whether the emulator runs one instruction: `Ok` or why not.
fn emulated(name: &str, i: &Instruction) -> Result<(), String> {
    let target = thin_targets::get(name).expect("a target");
    let mut text = format!("\t{i}\n");
    if let Some(t) = i.branch_target() {
        text.push_str(&format!("{}:\n", thin_targets::decode::label(t)));
    }
    text.push_str("\ts_endpgm\n");
    let program = thin_emu::read_program(&text, target).map_err(|e| format!("read: {e}"))?;
    let mut m = Machine::new(program, Facts::of(target), 256, 0, 65536)
        .map_err(|e| format!("read: {e}"))?;
    let (mut mem, mut tables) = (Memory::default(), Tables::default());
    match m.run_wg(&mut mem, &mut tables, 0, (0, 0), 1, 0, None) {
        Ok(_) => Ok(()),
        Err(e) if matches!(e.cause, Some(Cause::Emulator) | None) => Err(e.message),
        Err(_) => Ok(()),
    }
}

#[derive(Default)]
struct Op {
    instructions: u64,
    kernels: BTreeSet<usize>,
    shapes: BTreeMap<String, u64>,
}

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let json = args.iter().position(|a| a == "--json").map(|i| {
        let out = args[i + 1].clone();
        args.drain(i..i + 2);
        out
    });
    let mut ops: BTreeMap<&'static str, Op> = BTreeMap::new();
    let mut kernel_shapes: Vec<BTreeSet<String>> = Vec::new();
    let mut samples: HashMap<String, (&'static str, Instruction)> = HashMap::new();
    let mut undecoded = 0u64;
    for path in corpus::files(&args.iter().map(PathBuf::from).collect::<Vec<_>>()) {
        let img = corpus::image(&path, thin_targets::default_name())
            .expect("readable code object or bundle");
        let Ok(object) = thin_lift::object::read(&img) else {
            eprintln!("{}: not an ELF code object", path.display());
            continue;
        };
        let d = decoder(object.target).expect("a decoder");
        for k in corpus::kernels(&img).expect("kernel ranges") {
            let code = img.get(k.start..k.end).unwrap_or_else(|| {
                panic!("{}: {}: code range outside image", path.display(), k.name)
            });
            let id = kernel_shapes.len();
            let mut mine = BTreeSet::new();
            let mut at = 0;
            while at < code.len() {
                match d.decode_at(code, at) {
                    Ok(i) => {
                        let s = shape(&i);
                        let op = ops.entry(i.mnemonic).or_default();
                        op.instructions += 1;
                        op.kernels.insert(id);
                        *op.shapes.entry(s.clone()).or_default() += 1;
                        samples.entry(s.clone()).or_insert((object.target, i));
                        mine.insert(s);
                        at += usize::from(i.size);
                    }
                    Err(e) => {
                        undecoded += 1;
                        at += e.size.map_or(4, usize::from);
                    }
                }
            }
            kernel_shapes.push(mine);
        }
    }
    let verdicts: HashMap<&String, Result<(), String>> = samples
        .iter()
        .map(|(s, (t, i))| (s, emulated(t, i)))
        .collect();
    let ok = |s: &String| verdicts.get(s).is_some_and(Result::is_ok);
    let kernels = kernel_shapes.len();
    let runnable = kernel_shapes.iter().filter(|k| k.iter().all(ok)).count();
    // Impact: the kernels each missing op blocks, and those it alone blocks.
    let mut rows = Vec::new();
    for (m, op) in &ops {
        let bad: Vec<&String> = op.shapes.keys().filter(|s| !ok(s)).collect();
        if bad.is_empty() {
            continue;
        }
        let blocked: Vec<usize> = (op.kernels.iter())
            .copied()
            .filter(|&k| bad.iter().any(|s| kernel_shapes[k].contains(*s)))
            .collect();
        let alone = (blocked.iter())
            .filter(|&&k| {
                kernel_shapes[k]
                    .iter()
                    .filter(|s| !ok(s))
                    .all(|s| s.split(' ').next() == Some(m))
            })
            .count();
        let bad_ins: u64 = bad.iter().map(|s| op.shapes[*s]).sum();
        let why = verdicts[bad[0]].clone().err().unwrap_or_default();
        rows.push((
            blocked.len(),
            alone,
            *m,
            op.instructions,
            bad_ins,
            bad.len(),
            op.shapes.len(),
            why,
            bad[0].clone(),
        ));
    }
    rows.sort_by(|a, b| b.0.cmp(&a.0).then(b.3.cmp(&a.3)));
    let total: u64 = ops.values().map(|o| o.instructions).sum();
    println!(
        "kernels {kernels}  instructions {total} (undecoded {undecoded})  mnemonics {}  shapes {}",
        ops.len(),
        samples.len()
    );
    println!(
        "kernels whose every instruction the emulator runs: {runnable} ({:.2} %)",
        100.0 * runnable as f64 / kernels.max(1) as f64
    );
    println!("\nunsupported, by kernels blocked (cumulative: kernels that run once it and every op above it are supported):");
    println!("| mnemonic | kernels blocked | blocked by it alone | cumulative | instructions | unsupported instr. | shapes bad/all | why (first bad shape) |");
    println!("|---|---|---|---|---|---|---|---|");
    let mut fixed: BTreeSet<&str> = BTreeSet::new();
    for (blocked, alone, m, n, bad_ins, bad, all, why, s) in &rows {
        fixed.insert(m);
        let runs = |k: &BTreeSet<String>| {
            k.iter()
                .all(|s| ok(s) || fixed.contains(s.split(' ').next().unwrap_or("")))
        };
        let cumulative = kernel_shapes.iter().filter(|k| runs(k)).count();
        println!("| {m} | {blocked} | {alone} | {cumulative} | {n} | {bad_ins} | {bad}/{all} | `{s}`: {why} |");
    }
    println!("\nevery mnemonic: instructions, kernels, emulated");
    let mut all: Vec<_> = ops.iter().collect();
    all.sort_by_key(|(_, o)| std::cmp::Reverse(o.instructions));
    for (m, op) in &all {
        let good = op.shapes.keys().all(ok);
        println!(
            "{m:<36} {:>11} {:>6}  {}",
            op.instructions,
            op.kernels.len(),
            if good { "yes" } else { "NO" }
        );
    }
    if let Some(out) = json {
        let mut s = String::from("{\"mnemonics\":[");
        for (k, (m, op)) in ops.iter().enumerate() {
            let good = op.shapes.keys().all(ok);
            s.push_str(&format!(
                "{}{{\"mnemonic\":\"{m}\",\"instructions\":{},\"kernels\":{},\"emulated\":{good}}}",
                if k > 0 { "," } else { "" },
                op.instructions,
                op.kernels.len()
            ));
        }
        s.push_str(&format!(
            "],\"kernels\":{kernels},\"runnable\":{runnable}}}\n"
        ));
        std::fs::write(out, s).expect("writable");
    }
}
