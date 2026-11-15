//! Round-trips the code of gfx1201 ELF code objects through the decoder
//! and the encoder, and reports the coverage and the decoder's speed.
//!
//! cargo run --release -p thin-targets --example decode_corpus -- [--threads N] PATH...
//!
//! Each PATH is an ELF, an offload bundle, or a directory of them. Bundles
//! require clang-offload-bundler on PATH. Pass only the code objects to
//! measure, excluding duplicate extractions and unrelated helper ELFs.
//! Every function symbol in an executable section is a kernel; a file
//! without one is decoded whole.
//! For each kernel the decoder walks the instructions, stepping past a
//! refusal whose size it knows; each decoded instruction is printed,
//! parsed, compared with the decoded line, and re-encoded. Then the
//! kernel is decoded whole (branches checked) and assembled back.

mod corpus;

use corpus::{files, kernels, Kernel};
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use thin_target::parse_line;
use thin_targets::decode::DecodeError;
use thin_targets::{decoder, encode, encode_one, get, Decode, Instruction};

const T: &str = "gfx1201";

/// A failure class's tally.
#[derive(Default)]
struct Class {
    count: usize,
    example: String,
    mnemonics: HashMap<String, usize>,
}

#[derive(Default)]
struct Stats {
    kernels: usize,
    whole: usize,
    whole_exact: usize,
    instructions: usize,
    decoded: usize,
    exact: usize,
    parsed: usize,
    lost: usize,
    bytes: usize,
    classes: BTreeMap<String, Class>,
    mismatches: Vec<String>,
}

impl Stats {
    fn merge(&mut self, o: Stats) {
        self.kernels += o.kernels;
        self.whole += o.whole;
        self.whole_exact += o.whole_exact;
        self.instructions += o.instructions;
        self.decoded += o.decoded;
        self.exact += o.exact;
        self.parsed += o.parsed;
        self.lost += o.lost;
        self.bytes += o.bytes;
        for (k, c) in o.classes {
            let e = self.classes.entry(k).or_default();
            if e.count == 0 {
                e.example = c.example;
            }
            e.count += c.count;
            for (m, n) in c.mnemonics {
                *e.mnemonics.entry(m).or_default() += n;
            }
        }
        if self.mismatches.len() < 50 {
            self.mismatches.extend(o.mismatches);
        }
    }
}

/// Text with its numbers and register numbers written `N`.
fn norm(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = String::new();
    let mut i = 0;
    while i < b.len() {
        let word = i > 0 && (b[i - 1].is_ascii_alphanumeric() || b[i - 1] == b'_');
        let reg = matches!(b[i], b'v' | b's') && b.get(i + 1).is_some_and(u8::is_ascii_digit);
        if !word && (b[i].is_ascii_digit() || reg) {
            if reg {
                out.push(b[i] as char);
                i += 1;
            }
            let hex = b[i..].starts_with(b"0x");
            i += if hex { 2 } else { 0 };
            while i < b.len() && (b[i].is_ascii_digit() || (hex && b[i].is_ascii_hexdigit())) {
                i += 1;
            }
            out.push('N');
            continue;
        }
        out.push(b[i] as char);
        i += 1;
    }
    out
}

/// The class key of a refusal: its kind, class, field and its detail
/// with the numbers taken out.
fn key(e: &DecodeError) -> String {
    let field = e.field.unwrap_or("-");
    format!(
        "{} | {} | {field} | {}",
        e.why.name(),
        e.class,
        norm(&e.detail)
    )
}

fn round_trip(d: &dyn Decode, file: &str, img: &[u8], k: &Kernel, st: &mut Stats) {
    let code = &img[k.start..k.end];
    let syntax = get(T).unwrap().hooks();
    st.kernels += 1;
    st.bytes += code.len();
    let mut at = 0;
    while at < code.len() {
        st.instructions += 1;
        match d.decode_at(code, at) {
            Ok(i) => {
                st.decoded += 1;
                let mine = &code[at..at + usize::from(i.size)];
                let text = i.to_string();
                if parse_line(syntax, &text).is_ok_and(|l| l == i.line()) {
                    st.parsed += 1;
                }
                match encode_one(T, &i) {
                    Ok(b) if b == mine => st.exact += 1,
                    other => st
                        .mismatches
                        .push(format!("{file}:{}+{at:#x}: `{text}` -> {other:?}", k.name)),
                }
                at += usize::from(i.size);
            }
            Err(e) => {
                let c = st.classes.entry(key(&e)).or_default();
                if c.count == 0 {
                    c.example = format!("{} ({file}: {}+{at:#x})", e, k.name);
                }
                c.count += 1;
                *c.mnemonics
                    .entry(e.mnemonic.unwrap_or("?").to_owned())
                    .or_default() += 1;
                match e.size {
                    Some(n) => at += usize::from(n),
                    None => {
                        st.lost += 1;
                        at += 4;
                    }
                }
            }
        }
    }
    if let Ok(is) = d.decode(code) {
        st.whole += 1;
        if encode(T, &is).is_ok_and(|b| b == code) {
            st.whole_exact += 1;
        }
    }
}

/// Decodes every kernel, only decoding: instructions per second.
fn bench(d: &dyn Decode, codes: &[&[u8]], threads: usize) -> (usize, f64) {
    let next = AtomicUsize::new(0);
    let total = AtomicUsize::new(0);
    let t0 = Instant::now();
    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| {
                let mut n = 0;
                let mut i = Instruction::new(0, "");
                loop {
                    let k = next.fetch_add(1, Ordering::Relaxed);
                    let Some(code) = codes.get(k) else { break };
                    let mut at = 0;
                    while at < code.len() {
                        n += 1;
                        at += match d.decode_into(code, at, &mut i) {
                            Ok(()) => usize::from(i.size),
                            Err(e) => e.size.map_or(4, usize::from),
                        };
                    }
                }
                total.fetch_add(n, Ordering::Relaxed);
            });
        }
    });
    (total.into_inner(), t0.elapsed().as_secs_f64())
}

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    if let Some(i) = args.iter().position(|a| a == "--threads") {
        threads = args[i + 1].parse().expect("a thread count");
        args.drain(i..i + 2);
    }
    let list = args.iter().any(|a| a == "--list");
    let bench_only = args.iter().any(|a| a == "--bench");
    args.retain(|a| a != "--list" && a != "--bench");
    let paths = files(&args.iter().map(PathBuf::from).collect::<Vec<_>>());
    let d = decoder(T).expect("gfx1201's decoder");
    if list {
        // Every instruction's text by its section offset, or its refusal.
        for path in &paths {
            let img = corpus::image(path, T).expect("readable code object or bundle");
            for k in kernels(&img).expect("an ELF") {
                let code = &img[k.start..k.end];
                let mut at = 0;
                while at < code.len() {
                    match d.decode_at(code, at) {
                        Ok(i) => {
                            println!("{:x} {i}", k.start + at);
                            at += usize::from(i.size);
                        }
                        Err(e) => {
                            println!("{:x} ! {e}", k.start + at);
                            at += e.size.map_or(4, usize::from);
                        }
                    }
                }
            }
        }
        return;
    }
    let mut total = Stats::default();
    let (mut bench_n, mut t1, mut tn) = (0, 0.0, 0.0);
    for path in &paths {
        let img = corpus::image(path, T).expect("readable code object or bundle");
        let name = Path::new(path)
            .file_name()
            .unwrap()
            .to_string_lossy()
            .into_owned();
        let ks = match kernels(&img) {
            Ok(ks) => ks,
            Err(e) => {
                eprintln!("{name}: {e}");
                continue;
            }
        };
        let codes: Vec<&[u8]> = ks.iter().map(|k| &img[k.start..k.end]).collect();
        let (n, s1) = bench(d, &codes, 1);
        let (_, sn) = bench(d, &codes, threads);
        bench_n += n;
        t1 += s1;
        tn += sn;
        if bench_only {
            continue;
        }
        let next = AtomicUsize::new(0);
        let merged = Mutex::new(Stats::default());
        std::thread::scope(|s| {
            for _ in 0..threads {
                s.spawn(|| {
                    let mut st = Stats::default();
                    loop {
                        let k = next.fetch_add(1, Ordering::Relaxed);
                        let Some(k) = ks.get(k) else { break };
                        round_trip(d, &name, &img, k, &mut st);
                    }
                    merged.lock().unwrap().merge(st);
                });
            }
        });
        total.merge(merged.into_inner().unwrap());
    }
    let pct = |a: usize, b: usize| 100.0 * a as f64 / b.max(1) as f64;
    println!(
        "files {}  kernels {}  code {:.1} MB",
        paths.len(),
        total.kernels,
        total.bytes as f64 / 1e6
    );
    println!(
        "instructions {}  decoded {} ({:.3} %)  bit-exact re-encodes {} ({:.3} % of decoded)  parse(print) == line {}  unknown-size skips {}",
        total.instructions, total.decoded, pct(total.decoded, total.instructions), total.exact, pct(total.exact, total.decoded), total.parsed, total.lost
    );
    println!(
        "whole kernels decoded {} ({:.2} %), assembled back bit-exact {}",
        total.whole,
        pct(total.whole, total.kernels),
        total.whole_exact
    );
    println!(
        "decode only: {:.1} M instructions/s on 1 thread, {:.1} M/s on {threads} threads",
        bench_n as f64 / t1 / 1e6,
        bench_n as f64 / tn / 1e6
    );
    let mut classes: Vec<_> = total.classes.into_iter().collect();
    classes.sort_by_key(|a| std::cmp::Reverse(a.1.count));
    println!("\nrefusals by class ({} classes):", classes.len());
    for (k, c) in &classes {
        let mut ms: Vec<_> = c.mnemonics.iter().collect();
        ms.sort_by_key(|a| std::cmp::Reverse(*a.1));
        let top: Vec<String> = ms
            .iter()
            .take(10)
            .map(|(m, n)| format!("{m} {n}"))
            .collect();
        println!(
            "{:>10}  {k}\n            e.g. {}\n            top: {}",
            c.count,
            c.example,
            top.join(", ")
        );
    }
    if !total.mismatches.is_empty() {
        println!("\nre-encode mismatches:");
        for m in &total.mismatches {
            println!("  {m}");
        }
    }
}
