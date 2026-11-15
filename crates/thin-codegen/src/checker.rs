//! The independent wait checker: from a `.s` text alone, with its own
//! line handling, units, registers and control flow. It shares no code
//! with the compiler on purpose: gfx1201's split counters, read from the
//! mnemonics and wait instructions as the RDNA4 ISA guide names them.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

/// A violation: (line, op, register, the load's line).
pub type Record = (usize, String, String, usize);

/// An instruction of the text.
struct Ins {
    op: String,
    rest: String,
    line: usize,
}

/// The text as the checker reads it.
struct Prog {
    ins: Vec<Ins>,
    labels: HashMap<String, usize>,
    label_at: HashSet<usize>,
    skips: BTreeMap<usize, (BTreeSet<u32>, u32)>,
}

fn word(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_'
}

/// The listed waves and N of a `by_wave skip <list> of <N>` mark.
fn skip_mark(line: &str) -> Option<(BTreeSet<u32>, u32)> {
    let rest = &line[line.find("by_wave skip ")? + "by_wave skip ".len()..];
    let (list, n) = rest.split_once(" of ")?;
    let n: u32 = n
        .split(|c: char| !c.is_ascii_digit())
        .next()?
        .parse()
        .ok()?;
    let waves = list
        .split(',')
        .filter(|s| !s.trim().is_empty())
        .map(|s| s.trim().parse().ok());
    Some((waves.collect::<Option<_>>()?, n))
}

/// A label line: `^(\.?L?\w+):$`.
fn label(code: &str) -> Option<&str> {
    let name = code.strip_suffix(':')?;
    let inner = name.strip_prefix('.').unwrap_or(name);
    (!inner.is_empty() && inner.chars().all(word)).then_some(name)
}

/// The text up to its footer (the first `.section` or `.amdgpu_metadata`).
fn parse(text: &str) -> Prog {
    let mut p = Prog {
        ins: Vec::new(),
        labels: HashMap::new(),
        label_at: HashSet::new(),
        skips: BTreeMap::new(),
    };
    for (k, raw) in text.split('\n').enumerate() {
        let lead = raw.trim_start();
        if lead.starts_with(".section") || lead.starts_with(".amdgpu_metadata") {
            let w = lead
                .split(|c: char| !(word(c) || c == '.'))
                .next()
                .unwrap_or("");
            if w == ".section" || w == ".amdgpu_metadata" {
                break;
            }
        }
        if let Some(s) = skip_mark(raw) {
            p.skips.insert(p.ins.len(), s);
        }
        let code = raw.split("//").next().unwrap_or("");
        let code = code.split(';').next().unwrap_or("").trim();
        if code.is_empty() {
            continue;
        }
        if let Some(name) = label(code) {
            p.labels.insert(name.to_owned(), p.ins.len());
            p.label_at.insert(p.ins.len());
            continue;
        }
        if code.starts_with('.') {
            continue;
        }
        let (op, rest) = code.split_once(' ').unwrap_or((code, ""));
        p.ins.push(Ins {
            op: op.to_owned(),
            rest: rest.trim().to_owned(),
            line: k + 1,
        });
    }
    p
}

impl Prog {
    /// The successors of instruction i in a wave class.
    fn succ(&self, i: usize, class: &BTreeMap<usize, bool>) -> Vec<usize> {
        let target = |s: &str| {
            let name = s.split_whitespace().next().unwrap_or("");
            *self
                .labels
                .get(name)
                .unwrap_or_else(|| panic!("KeyError: {name}"))
        };
        let x = &self.ins[i];
        if let Some(&taken) = class.get(&i) {
            return vec![if taken { target(&x.rest) } else { i + 1 }];
        }
        match x.op.as_str() {
            "s_endpgm" => vec![],
            "s_branch" => vec![target(&x.rest)],
            op if op.starts_with("s_cbranch_") => vec![i + 1, target(&x.rest)],
            _ if i + 1 < self.ins.len() => vec![i + 1],
            _ => vec![],
        }
    }

    /// The wave classes: per class, each skip's bit.
    fn classes(&self) -> Vec<BTreeMap<usize, bool>> {
        let Some((_, n)) = self.skips.values().next() else {
            return vec![BTreeMap::new()];
        };
        let mut seen: Vec<Vec<bool>> = Vec::new();
        let mut out = Vec::new();
        for w in 0..*n {
            let sig: Vec<bool> = self
                .skips
                .values()
                .map(|(set, _)| set.contains(&w))
                .collect();
            if !seen.contains(&sig) {
                out.push(
                    self.skips
                        .keys()
                        .copied()
                        .zip(sig.iter().copied())
                        .collect(),
                );
                seen.push(sig);
            }
        }
        out
    }
}

/// A check's state and its step.
trait Check {
    type State: Clone + PartialEq + Default;
    fn step(&self, st: &mut Self::State, x: &Ins, found: &mut Found);
    fn merge(&self, old: &Self::State, new: &Self::State) -> Self::State;
}

/// The violations, keyed, the first met kept.
#[derive(Default)]
struct Found(BTreeMap<(usize, String), Record>);

impl Found {
    fn add(&mut self, key: (usize, String), r: Record) {
        self.0.entry(key).or_insert(r);
    }
}

fn walk<C: Check>(p: &Prog, c: &C, class: &BTreeMap<usize, bool>, found: &mut Found) {
    if p.ins.is_empty() {
        return;
    }
    let mut states: HashMap<usize, C::State> = HashMap::from([(0, C::State::default())]);
    let mut work = vec![0];
    while let Some(i) = work.pop() {
        let mut st = states[&i].clone();
        let mut j = i;
        let succ = loop {
            c.step(&mut st, &p.ins[j], found);
            let s = p.succ(j, class);
            if s == [j + 1] && !p.label_at.contains(&(j + 1)) && !p.skips.contains_key(&j) {
                j += 1;
                continue;
            }
            break s;
        };
        for s in succ.into_iter().filter(|&s| s < p.ins.len()) {
            let merged = match states.get(&s) {
                Some(old) => c.merge(old, &st),
                None => st.clone(),
            };
            if states.get(&s) != Some(&merged) {
                states.insert(s, merged);
                work.push(s);
            }
        }
    }
}

fn run<C: Check>(text: &str, c: &C) -> Vec<Record> {
    let p = parse(text);
    let mut found = Found::default();
    for class in p.classes() {
        walk(&p, c, &class, &mut found);
    }
    let mut out: Vec<Record> = found.0.into_values().collect();
    out.sort_by(|a, b| (a.0, &a.2).cmp(&(b.0, &b.2)));
    out
}

/// The registers of operand text, ranges expanded: files in `files`.
fn regs(text: &str, files: &str) -> Vec<String> {
    let b = text.as_bytes();
    let mut out = Vec::new();
    let mut k = 0;
    while k < b.len() {
        let c = b[k] as char;
        let bounded = k == 0 || !word(b[k - 1] as char);
        if !(bounded && files.contains(c)) {
            k += 1;
            continue;
        }
        let digits = |from: usize| {
            (from..b.len())
                .take_while(|&j| b[j].is_ascii_digit())
                .count()
        };
        let n = digits(k + 1);
        if n > 0 && (k + 1 + n == b.len() || !word(b[k + 1 + n] as char)) {
            out.push(text[k..k + 1 + n].to_owned());
            k += 1 + n;
            continue;
        }
        if let Some((lo, hi, len)) = range(&text[k + 1..]) {
            out.extend((lo..=hi).map(|r| format!("{c}{r}")));
            k += 1 + len;
            continue;
        }
        k += 1;
    }
    out
}

/// `[a:b]` at the start of a text: (a, b, its length).
fn range(t: &str) -> Option<(u32, u32, usize)> {
    let body = t.strip_prefix('[')?;
    let end = body.find(']')?;
    let (a, b) = body[..end].split_once(':')?;
    Some((a.trim().parse().ok()?, b.trim().parse().ok()?, end + 2))
}

fn operands(rest: &str) -> Vec<&str> {
    rest.split(',').map(str::trim).collect()
}

/// A violation's message.
pub fn message(r: &Record) -> String {
    format!(
        "line {}: {} touches {} while the load at line {} may be in flight",
        r.0, r.1, r.2, r.3
    )
}

/// Split and combined wait counters.
const COUNTERS: [&str; 6] = ["loadcnt", "storecnt", "dscnt", "kmcnt", "vmcnt", "lgkmcnt"];
const VM: u8 = 4;
const LGKM: u8 = 5;

/// Whether an LDS op returns data to a register.
fn lds_returns(op: &str) -> bool {
    let Some(r) = op.strip_prefix("ds_") else {
        return false;
    };
    [
        "load", "read", "bpermute", "permute", "swizzle", "consume", "append",
    ]
    .iter()
    .any(|k| r.starts_with(k))
        || r.find("_rtn_")
            .is_some_and(|k| k > 0 && r[..k].chars().all(word))
}

/// An op's counter, whether it returns, and whether it completes out of order.
fn op_counter(op: &str, rest: &str, unified: bool) -> Option<(u8, bool, bool)> {
    let unit = ["buffer_", "global_", "flat_", "scratch_"]
        .iter()
        .find_map(|u| op.strip_prefix(u).map(|r| (*u, r)));
    if let Some((u, r)) = unit {
        if r.starts_with("load") {
            let dma = u == "buffer_" && rest.split_whitespace().any(|t| t == "lds");
            return Some((if unified { VM } else { 0 }, !dma, false));
        }
        if r.starts_with("store") || r == "wbl2" || r == "inv" {
            return Some((if unified { VM } else { 1 }, false, false));
        }
        if r.starts_with("atomic") && u != "scratch_" {
            let has_sc0 = rest.split_whitespace().any(|t| t == "sc0");
            let returns = rest.contains("th:TH_ATOMIC_RETURN") || has_sc0;
            let c = if unified {
                VM
            } else if returns {
                0
            } else {
                1
            };
            return Some((c, returns, false));
        }
        return None;
    }
    if op.starts_with("ds_") {
        return Some((if unified { LGKM } else { 2 }, lds_returns(op), false));
    }
    // Scalar loads, clocks and returning messages land out of order.
    if op.starts_with("s_load")
        || op.starts_with("s_buffer_load")
        || op.starts_with("s_sendmsg_rtn")
        || op == "s_memtime"
        || op == "s_memrealtime"
    {
        return Some((if unified { LGKM } else { 3 }, true, true));
    }
    if unified
        && (op.starts_with("s_store")
            || op.starts_with("s_atomic")
            || op.starts_with("s_buffer_atomic"))
    {
        return Some((
            LGKM,
            op.contains("atomic") && rest.split_whitespace().any(|t| t == "glc"),
            true,
        ));
    }
    None
}

fn parse_u32(word: &str) -> Option<u32> {
    word.strip_prefix("0x")
        .map_or_else(|| word.parse().ok(), |h| u32::from_str_radix(h, 16).ok())
}

fn waitcnt_fields(rest: &str) -> Vec<(u8, u32)> {
    let trimmed = rest.trim();
    if trimmed.is_empty() {
        return Vec::new();
    }
    if let Some(simm) = parse_u32(trimmed) {
        let vm = (simm & 0xF) | (((simm >> 14) & 3) << 4);
        let lgkm = (simm >> 8) & 0xF;
        return vec![(VM, vm), (LGKM, lgkm)];
    }
    let mut out = Vec::new();
    for piece in trimmed.split(|c: char| c.is_whitespace() || c == '&' || c == ',') {
        let Some((name, num)) = piece.strip_suffix(')').and_then(|p| p.split_once('(')) else {
            continue;
        };
        let Some(n) = parse_u32(num.trim()) else {
            continue;
        };
        match name.trim() {
            "vmcnt" => out.push((VM, n)),
            "lgkmcnt" => out.push((LGKM, n)),
            _ => {}
        }
    }
    out
}

/// A pending register: (its load's counter, younger ops, unordered, the load's line).
type Split = BTreeMap<String, (u8, u32, bool, usize)>;

/// The counter check; the counter of each violation kept for its message.
struct Counters {
    unified: bool,
    seen: std::cell::RefCell<BTreeMap<(usize, String), u8>>,
}

impl Check for Counters {
    type State = Split;

    fn step(&self, st: &mut Split, x: &Ins, found: &mut Found) {
        if x.op == "s_waitcnt" {
            for (c, n) in waitcnt_fields(&x.rest) {
                st.retain(|_, e| !(e.0 == c && (n == 0 || (!e.2 && e.1 >= n))));
            }
            return;
        }
        if let Some(c) = x.op.strip_prefix("s_wait_") {
            let Some(c) = COUNTERS[..4].iter().position(|n| *n == c) else {
                return; // s_wait_alu and the like change nothing
            };
            let word = x.rest.split([' ', ',']).next().unwrap_or("");
            if let Some(n) = parse_u32(word) {
                let c = c as u8;
                st.retain(|_, e| !(e.0 == c && (n == 0 || (!e.2 && e.1 >= n))));
            }
            return;
        }
        for r in regs(&x.rest, "vsa") {
            if let Some(&(c, _, _, load)) = st.get(&r) {
                let key = (x.line, r.clone());
                if !found.0.contains_key(&key) {
                    self.seen.borrow_mut().insert(key.clone(), c);
                }
                found.add(key, (x.line, x.op.clone(), r, load));
            }
        }
        let Some((own, returns, unordered)) = op_counter(&x.op, &x.rest, self.unified) else {
            return;
        };
        if !unordered {
            st.values_mut()
                .filter(|e| e.0 == own && !e.2)
                .for_each(|e| e.1 = (e.1 + 1).min(64));
        }
        if returns {
            let ops = operands(&x.rest);
            for r in regs(ops[0], "vsa") {
                st.insert(r, (own, 0, unordered, x.line));
            }
        }
    }

    fn merge(&self, old: &Split, new: &Split) -> Split {
        let mut out = old.clone();
        for (r, &(c, n, u, line)) in new {
            out.entry(r.clone())
                .and_modify(|e| {
                    e.1 = e.1.min(n);
                    e.2 |= u;
                })
                .or_insert((c, n, u, line));
        }
        out
    }
}

/// Independent counter wait check: each violation's record and
/// message, sorted by (line, register).
pub fn counter_waits(text: &str) -> Vec<(Record, String)> {
    let unified = [
        "s_waitcnt",
        "s_load_dword",
        "s_buffer_load_dword",
        "global_load_dword",
        "global_load_ubyte",
        "global_load_sbyte",
        "global_load_ushort",
        "global_load_sshort",
        "buffer_load_dword",
        "ds_read_",
        "ds_write_",
        "s_memtime",
        "s_memrealtime",
    ]
    .iter()
    .any(|m| text.contains(m));
    let c = Counters {
        unified,
        seen: Default::default(),
    };
    let records = run(text, &c);
    let counters = c.seen.into_inner();
    let fallback = if unified { "vmcnt" } else { "loadcnt" };
    records
        .into_iter()
        .map(|r| {
            let cnt = counters
                .get(&(r.0, r.2.clone()))
                .map_or(fallback, |&k| COUNTERS[k as usize]);
            let m = format!("{} ({cnt})", message(&r));
            (r, m)
        })
        .collect()
}
