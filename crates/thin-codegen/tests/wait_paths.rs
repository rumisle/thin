//! Per-path wait counts on synthetic control-flow graphs.
//! A requested count must be safe on every represented path.

use std::collections::BTreeMap;

use thin_passes::waits::{self, Key};
use thin_test_support::adapters::default_target;
use thin_test_support::asm;

/// A tiny deterministic generator (xorshift).
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

#[derive(Clone, Copy, Debug)]
enum Item {
    Load(u32),
    Wait(u32),
    /// A request naming one to three registers.
    Marker(u32, u32, u32),
    /// A VALU overwriting a register: an implied request for its writer.
    Write(u32),
}

#[derive(Clone, Copy, Debug)]
enum End {
    Fall,
    Cond(usize),
    Jump(usize),
    Stop,
}

struct Prog {
    blocks: Vec<(Vec<Item>, End)>,
    /// Per block, the line index of each item.
    lines_of: Vec<Vec<usize>>,
    lines: Vec<String>,
}

fn make(rng: &mut Rng) -> Prog {
    let n = 1 + rng.below(6) as usize;
    let mut blocks: Vec<(Vec<Item>, End)> = Vec::new();
    for b in 0..n {
        let items: Vec<Item> = (0..rng.below(6))
            .map(|_| match rng.below(12) {
                0..=4 => Item::Load(1 + rng.below(6) as u32),
                5 => Item::Wait(rng.below(4) as u32),
                6 | 7 => Item::Write(1 + rng.below(6) as u32),
                _ => Item::Marker(
                    1 + rng.below(6) as u32,
                    rng.below(7) as u32,
                    rng.below(7) as u32,
                ),
            })
            .collect();
        let end = if b + 1 == n {
            End::Stop
        } else {
            match rng.below(4) {
                0 | 1 => End::Fall,
                2 => End::Cond(rng.below(n as u64) as usize),
                _ => End::Jump(rng.below(n as u64) as usize),
            }
        };
        blocks.push((items, end));
    }
    let mut lines = Vec::new();
    let mut lines_of = Vec::new();
    for (b, (items, end)) in blocks.iter().enumerate() {
        lines.push(format!(".L{b}:"));
        let mut at = Vec::new();
        for it in items {
            at.push(lines.len());
            lines.push(match *it {
                Item::Load(r) => format!("global_load_b32 v{r}, v0, s[0:1]"),
                Item::Wait(m) => format!("s_wait_loadcnt {m}"),
                Item::Marker(a, b, c) => {
                    let regs: Vec<String> = [a, b, c]
                        .iter()
                        .filter(|&&r| r != 0)
                        .map(|r| format!("v{r}"))
                        .collect();
                    format!("@wait load regs {}", regs.join(" "))
                }
                Item::Write(r) => format!("v_mov_b32 v{r}, 0"),
            });
        }
        lines_of.push(at);
        match end {
            End::Fall => {}
            End::Cond(t) => {
                lines.push("s_cmp_eq_u32 s2, 0".into());
                lines.push(format!("s_cbranch_scc0 .L{t}"));
            }
            End::Jump(t) => lines.push(format!("s_branch .L{t}")),
            End::Stop => lines.push("s_endpgm".into()),
        }
    }
    Prog {
        blocks,
        lines_of,
        lines,
    }
}

/// The queue on a path: (load line, register), oldest first.
#[derive(Clone, Default)]
struct Path {
    queue: Vec<(usize, u32)>,
    last: BTreeMap<u32, usize>,
}

impl Path {
    fn retire(&mut self, m: u32) {
        let keep = (m as usize).min(self.queue.len());
        let drop = self.queue.len() - keep;
        self.queue.drain(..drop);
    }
}

/// The brute-force count of every marker, given the decided counts.
fn brute(
    p: &Prog,
    decided: &BTreeMap<usize, Option<u32>>,
    largest: u32,
) -> BTreeMap<usize, Option<u32>> {
    let mut want: BTreeMap<usize, Option<u32>> = BTreeMap::new();
    for at in p.lines_of.iter().flatten() {
        if p.lines[*at].starts_with('@') {
            want.insert(*at, None);
        }
    }
    let mut visits = vec![0u32; p.blocks.len()];
    walk(
        p,
        0,
        Path::default(),
        &mut visits,
        decided,
        largest,
        &mut want,
    );
    want
}

fn walk(
    p: &Prog,
    b: usize,
    mut path: Path,
    visits: &mut Vec<u32>,
    decided: &BTreeMap<usize, Option<u32>>,
    largest: u32,
    want: &mut BTreeMap<usize, Option<u32>>,
) {
    if visits[b] == 3 {
        return;
    }
    visits[b] += 1;
    let (items, end) = &p.blocks[b];
    for (it, &line) in items.iter().zip(&p.lines_of[b]) {
        match *it {
            Item::Load(r) => {
                // A load over a load still in flight is an implied request
                // too, of its own counter.
                let named = path.last.get(&r).copied();
                if let Some(pos) =
                    named.and_then(|op| path.queue.iter().position(|&(l, _)| l == op))
                {
                    let m = ((path.queue.len() - pos - 1) as u32).min(64).min(largest);
                    let e = want.entry(line).or_insert(None);
                    *e = Some(e.map_or(m, |x| x.min(m)));
                }
                if let Some(Some(v)) = decided.get(&line) {
                    path.retire(*v);
                }
                path.queue.push((line, r));
                path.last.insert(r, line);
            }
            Item::Wait(m) => path.retire(m),
            Item::Marker(..) | Item::Write(_) => {
                let regs: Vec<u32> = match *it {
                    Item::Marker(a, b, c) => [a, b, c].into_iter().filter(|&r| r != 0).collect(),
                    Item::Write(r) => vec![r],
                    _ => unreachable!(),
                };
                for &r in &regs {
                    let named = path.last.get(&r).copied();
                    let pos = named.and_then(|op| path.queue.iter().position(|&(l, _)| l == op));
                    if let Some(pos) = pos {
                        let younger = (path.queue.len() - pos - 1) as u32;
                        let m = younger.min(64).min(largest);
                        let e = want.entry(line).or_insert(None);
                        *e = Some(e.map_or(m, |x| x.min(m)));
                    }
                }
                if let Some(Some(v)) = decided.get(&line) {
                    path.retire(*v);
                }
                if let Item::Write(r) = *it {
                    path.last.remove(&r);
                }
            }
        }
    }
    let succs: Vec<usize> = match *end {
        End::Fall => vec![b + 1],
        End::Cond(t) => vec![t, b + 1],
        End::Jump(t) => vec![t],
        End::Stop => vec![],
    };
    for s in succs {
        walk(p, s, path.clone(), visits, decided, largest, want);
    }
    visits[b] -= 1;
}

#[test]
fn every_request_counts_the_minimum_over_its_paths() {
    let t = default_target();
    let largest = t.definition().counters[0].max;
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let mut checked = 0;
    for case in 0..3000 {
        let p = make(&mut rng);
        let refs: Vec<&str> = p.lines.iter().map(String::as_str).collect();
        let s = asm::snapshot(t, asm::read(t, &refs).unwrap());
        let counts = waits::Waits::new(&s)
            .solve()
            .unwrap_or_else(|d| panic!("case {case}: {}", asm::message(&s, &d.msg)));
        // An implied wait is refusal: such programs never reach
        // placement, so the property is over the ones the pass accepts.
        if counts.iter().any(|(k, v)| k.implied() && v.is_some()) {
            continue;
        }
        let decided: BTreeMap<usize, Option<u32>> = counts
            .iter()
            .filter_map(|(k, v)| match k {
                Key::Marker(k) | Key::Use(k, 0) => Some((*k, *v)),
                _ => None,
            })
            .collect();
        let mut want = brute(&p, &decided, largest);
        let mut got: BTreeMap<usize, Option<u32>> = decided.clone();
        for k in want.keys() {
            got.entry(*k).or_insert(None);
        }
        for k in got.keys() {
            want.entry(*k).or_insert(None);
        }
        assert_eq!(got, want, "case {case}:\n{}", p.lines.join("\n"));
        checked += want.len();
    }
    assert!(checked > 1500, "{checked} requests");
}

/// The β epilogue of the lifted gfx1201 GEMM: on the K < 16 path 24 prefetch
/// loads are still in flight (the main loop that would consume them is
/// skipped); then 16 C loads, each waited for before its use.
fn epilogue() -> Vec<String> {
    let mut l: Vec<String> = vec![
        "s_cmp_lt_u32 s2, 16".into(),
        "s_cbranch_scc1 .Lshort".into(),
        ".Lloop:".into(),
    ];
    for k in 0..24 {
        l.push(format!(
            "buffer_load_b32 v{}, v70, s[8:11], off offen",
            100 + k
        ));
    }
    l.push("s_wait_loadcnt 0".into());
    for k in 0..24 {
        l.push(format!("v_add_f32 v{}, v{}, v99", 140 + k, 100 + k));
    }
    l.push("s_cmp_eq_u32 s3, 0".into());
    l.push("s_cbranch_scc0 .Lloop".into());
    l.push("s_branch .Lepi".into());
    l.push(".Lshort:".into());
    for k in 0..24 {
        l.push(format!(
            "buffer_load_b32 v{}, v70, s[8:11], off offen",
            100 + k
        ));
    }
    l.push(".Lepi:".into());
    for k in 0..16 {
        let r = 4 * k;
        l.push(format!(
            "buffer_load_b128 v[{}:{}], v71, s[12:15], off offen offset:{}",
            200 + r,
            203 + r,
            16 * k
        ));
    }
    for k in 0..16 {
        let r = 200 + 4 * k;
        l.push(format!("@wait load regs v[{r}:{}]", r + 3));
        for q in 0..4 {
            l.push(format!("v_fmac_f32_e64 v{}, v{}, s45", 4 * k + q, r + q));
        }
    }
    l.push("s_endpgm".into());
    l
}

#[test]
fn the_lifted_gemms_beta_epilogue_keeps_its_staircase() {
    let t = default_target();
    let lines = epilogue();
    let refs: Vec<&str> = lines.iter().map(String::as_str).collect();
    let s = asm::snapshot(t, asm::read(t, &refs).unwrap());
    let counts = waits::Waits::new(&s).solve().unwrap();
    let got: Vec<Option<u32>> = counts
        .iter()
        .filter_map(|(k, v)| matches!(k, Key::Marker(_)).then_some(*v))
        .collect();
    let want: Vec<Option<u32>> = (0..16).rev().map(Some).collect();
    assert_eq!(got, want);
    let placed = waits::pass(&s, &[], waits::Options::default()).unwrap();
    let text: Vec<String> = placed
        .stream
        .iter()
        .map(|p| asm::spell(&s, &p.0))
        .filter(|l| l.starts_with("s_wait_loadcnt"))
        .collect();
    let want: Vec<String> = std::iter::once("s_wait_loadcnt 0".to_string())
        .chain((0..16).rev().map(|n| format!("s_wait_loadcnt {n}")))
        .collect();
    assert_eq!(text, want);
}

/// The lifted shape: prefetch b128 loads into v[72:119] on the K < 16 path,
/// C loads into v[76:139] overwriting some of them (the lift's WAW markers
/// before each), and the first hint naming C0 and the prefetch registers
/// still in flight there.
fn epilogue_lifted() -> Vec<String> {
    let mut l: Vec<String> = vec![
        "s_cmp_lt_u32 s2, 16".into(),
        "s_cbranch_scc1 .Lshort".into(),
    ];
    l.push("s_branch .Lepi".into());
    l.push(".Lshort:".into());
    for k in 0..12 {
        let r = 72 + 4 * k;
        l.push(format!(
            "buffer_load_b128 v[{r}:{}], v64, s[48:51], off offen",
            r + 3
        ));
    }
    l.push(".Lepi:".into());
    for k in 0..16 {
        let r = 76 + 4 * k;
        if r <= 119 {
            l.push(format!("@wait load regs v[{r}:{}]", r + 3));
        }
        l.push(format!(
            "buffer_load_b128 v[{r}:{}], v72, s[20:23], off offen",
            r + 3
        ));
    }
    for k in 0..16 {
        let r = 76 + 4 * k;
        if k == 0 {
            l.push(format!("@wait load regs v[{r}:{}] v[72:75]", r + 3));
        } else {
            l.push(format!("@wait load regs v[{r}:{}]", r + 3));
        }
        for q in 0..4 {
            l.push(format!("v_fmac_f32_e64 v{}, v{}, s45", 4 * k + q, r + q));
        }
    }
    l.push("s_endpgm".into());
    l
}

#[test]
fn the_staircase_holds_with_older_prefetch_loads_named() {
    let t = default_target();
    let lines = epilogue_lifted();
    let refs: Vec<&str> = lines.iter().map(String::as_str).collect();
    let s = asm::snapshot(t, asm::read(t, &refs).unwrap());
    let counts = waits::Waits::new(&s).solve().unwrap();
    let got: Vec<(String, Option<u32>)> = counts
        .iter()
        .filter_map(|(k, v)| match k {
            Key::Marker(k) => Some((lines[*k].clone(), *v)),
            _ => None,
        })
        .collect();
    let stair: Vec<Option<u32>> = got.iter().rev().take(16).rev().map(|x| x.1).collect();
    assert_eq!(stair, (0..16).rev().map(Some).collect::<Vec<_>>());
}
