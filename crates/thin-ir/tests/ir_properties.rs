//! Dataflow and CFG properties on synthetic instruction streams.

use std::collections::{BTreeSet, HashMap};

use thin_ir::Snapshot;
use thin_test_support::adapters::default_target;
use thin_test_support::notation::Lines;

/// A small deterministic generator (xorshift64*).
struct Rng(u64);

impl Rng {
    fn below(&mut self, n: u64) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d) % n
    }

    fn pick<'a>(&mut self, items: &[&'a str]) -> &'a str {
        items[self.below(items.len() as u64) as usize]
    }
}

fn snapshot(lines: &[String]) -> Snapshot {
    let mut l = Lines::new(default_target());
    let refs: Vec<&str> = lines.iter().map(String::as_str).collect();
    let instrs = l.read(&refs).expect("lines of the notation");
    Snapshot::new(default_target(), l.values, instrs.into())
}

/// A line that reads or writes EXEC or an SGPR; with `literal`, only
/// literal moves, saves and other ops.
fn exec_line(rng: &mut Rng, literal: bool) -> String {
    let reg = rng.pick(&["s4", "s5", "s6", "s7"]);
    let imm = rng.pick(&["-1", "0", "3", "0x7fffffff", "0x80000000", "-2"]);
    let kind = if literal {
        [0, 2, 5, 6][rng.below(4) as usize]
    } else {
        rng.below(7)
    };
    match kind {
        0 => format!("s_mov_b32 exec_lo, {imm}"),
        1 => format!("s_mov_b32 exec_lo, {reg}"),
        2 => format!("s_mov_b32 {reg}, exec_lo"),
        3 => format!("s_and_saveexec_b32 {reg}, s8"),
        4 => "v_cmpx_lt_u32 v1, v2".into(),
        5 => format!("s_mov_b32 {reg}, {imm}"),
        _ => "v_mov_b32 v0, 0".into(),
    }
}

/// Runs straight-line lines on concrete registers, EXEC on wave32; ops
/// whose result the analysis cannot know take values from `noise`. EXEC
/// before each line.
fn simulate(lines: &[String], noise: &mut Rng) -> Vec<u64> {
    let mut exec: u64 = 0xffff_ffff;
    let mut s: HashMap<String, u64> = HashMap::new();
    let mut out = Vec::new();
    let literal = |t: &str| -> u64 {
        let v = t.strip_prefix("0x").map_or_else(
            || t.parse::<i64>().unwrap(),
            |h| i64::from_str_radix(h, 16).unwrap(),
        );
        v as u32 as u64
    };
    for line in lines {
        out.push(exec);
        let (op, rest) = line.split_once(' ').unwrap_or((line, ""));
        let ops: Vec<&str> = rest.split(", ").collect();
        let read = |s: &HashMap<String, u64>, t: &str, noise: &mut Rng| -> u64 {
            if t.starts_with('s') {
                *s.get(t).unwrap_or(&noise.below(1 << 32))
            } else {
                literal(t)
            }
        };
        match (op, ops[0]) {
            ("s_mov_b32", "exec_lo") => exec = read(&s, ops[1], noise),
            ("s_mov_b32", dst) if ops[1] == "exec_lo" => _ = s.insert(dst.into(), exec),
            ("s_mov_b32", dst) => _ = s.insert(dst.into(), literal(ops[1])),
            ("s_and_saveexec_b32", dst) => {
                s.insert(dst.into(), exec);
                exec = noise.below(1 << 32);
            }
            ("v_cmpx_lt_u32", _) => exec = noise.below(1 << 32),
            _ => {}
        }
    }
    out
}

#[test]
fn exec_agrees_with_a_simulator_where_it_claims_to_know() {
    let mut rng = Rng(0x1234_5678);
    for round in 0..400 {
        let literal_only = round % 2 == 1;
        let mut lines: Vec<String> = (0..12).map(|_| exec_line(&mut rng, literal_only)).collect();
        lines.push("s_endpgm".into());
        let got = snapshot(&lines).exec();
        let runs: Vec<Vec<u64>> = (1..4)
            .map(|seed| simulate(&lines, &mut Rng(seed * 7919)))
            .collect();
        for (k, known) in got.iter().enumerate() {
            if let Some(v) = known {
                assert!(runs.iter().all(|r| r[k] == *v), "{lines:?} at {k}: {v:#x}");
            }
        }
        if literal_only {
            // Every 32-bit literal is a known EXEC on a 32-lane target.
            assert!(got.iter().all(Option::is_some), "{lines:?}");
        }
    }
}

/// A load or wait line and, for a load, its counter and whether it is
/// unordered.
fn flight_line(rng: &mut Rng) -> (String, Option<(&'static str, bool)>) {
    let m = rng.below(3);
    match rng.below(8) {
        0 | 1 => (
            "global_load_b32 v1, v[2:3], off".into(),
            Some(("load", false)),
        ),
        2 => (
            "global_store_b32 v[2:3], v1, off".into(),
            Some(("store", false)),
        ),
        3 => ("ds_load_b32 v4, v6".into(), Some(("ds", false))),
        4 => ("s_load_b32 s4, s[0:1], 0".into(), Some(("km", true))),
        5 => (format!("s_wait_loadcnt {{loadcnt={m}}}"), None),
        6 => (format!("s_wait_dscnt {{dscnt={m}}}"), None),
        _ => (format!("s_wait_kmcnt {{kmcnt={m}}}"), None),
    }
}

#[test]
fn in_flight_agrees_with_a_queue_per_counter() {
    let mut rng = Rng(99);
    for _ in 0..400 {
        let mut lines = Vec::new();
        let mut kinds = Vec::new();
        for _ in 0..14 {
            let (line, kind) = flight_line(&mut rng);
            lines.push(line);
            kinds.push(kind);
        }
        lines.push("s_endpgm".into());
        let loads: Vec<usize> = (0..kinds.len()).filter(|&k| kinds[k].is_some()).collect();
        let s = snapshot(&lines);
        let got = thin_ir::flight::in_flight(&s, &s.cfg(None), &loads);
        let mut queues: HashMap<&str, Vec<usize>> = HashMap::new();
        for (k, line) in lines.iter().enumerate() {
            let outstanding: BTreeSet<usize> = queues
                .values()
                .flatten()
                .map(|&k| loads.iter().position(|&l| l == k).unwrap())
                .collect();
            assert_eq!(
                got[k].iter().copied().collect::<BTreeSet<_>>(),
                outstanding,
                "{lines:?} at {k}"
            );
            if let Some((counter, _)) = kinds.get(k).copied().flatten() {
                queues.entry(counter).or_default().push(k);
            } else if let Some(rest) = line.strip_prefix("s_wait_") {
                let (field, m) = rest.split_once(" {").unwrap();
                let counter = &field[..field.len() - 3];
                let m: usize = m
                    .trim_end_matches('}')
                    .split('=')
                    .nth(1)
                    .unwrap()
                    .parse()
                    .unwrap();
                let q = queues.entry(counter).or_default();
                let keep = if counter == "km" && m > 0 {
                    q.len()
                } else {
                    m.min(q.len())
                };
                q.drain(..q.len() - keep);
            }
        }
    }
}

/// A random acyclic program of M0 writes and forward conditional branches.
fn m0_program(rng: &mut Rng) -> Vec<String> {
    let mut lines = Vec::new();
    let mut open: Vec<usize> = Vec::new();
    for k in 0..10 {
        let c = 1 + rng.below(2);
        lines.push(match rng.below(6) {
            0 => format!("s_mov_b32 m0, {c} ; m0_set={c}"),
            1 => format!("s_mov_b32 m0, {c} ; m0_set={c}"),
            2 => "s_mov_b32 m0, s4".into(),
            3 => format!("s_mov_b32 m0, {c}"),
            4 => {
                open.push(k);
                format!("s_cbranch_scc1 @L{k}")
            }
            _ => "v_mov_b32 v0, 0".into(),
        });
        if rng.below(3) == 0 {
            if let Some(label) = open.pop() {
                lines.push(format!("L{label}:"));
            }
        }
    }
    lines.extend(open.into_iter().map(|label| format!("L{label}:")));
    lines.push("s_endpgm".into());
    lines
}

#[test]
fn redundant_m0_writes_hold_on_every_path() {
    let mut rng = Rng(4242);
    for _ in 0..300 {
        let lines = m0_program(&mut rng);
        let s = snapshot(&lines);
        let g = s.cfg(None);
        let mut arrivals: Vec<BTreeSet<Option<i128>>> = vec![BTreeSet::new(); lines.len()];
        let mut stack = vec![(0usize, None::<i128>)];
        while let Some((k, m0)) = stack.pop() {
            arrivals[k].insert(m0);
            let i = &s.instrs()[k];
            let after = if i.m0_write() { i.attrs().m0_set } else { m0 };
            stack.extend(g.succ[k].iter().map(|&t| (t, after)));
        }
        let want: Vec<usize> = (0..lines.len())
            .filter(|&k| {
                let set = s.instrs()[k].attrs().m0_set;
                set.is_some() && arrivals[k].len() == 1 && arrivals[k].contains(&set)
            })
            .collect();
        assert_eq!(thin_ir::m0::redundant(&s), want, "{lines:?}");
    }
}

/// A gathered value of file `file` named `name`.
fn gathered(file: char, name: &str, n: u32, register: Option<u32>) -> thin_ir::GatherValue {
    thin_ir::GatherValue {
        file,
        n,
        align: 1,
        name: name.into(),
        register,
        ..Default::default()
    }
}

#[test]
fn a_kernel_snapshot_gathers_as_section_8_2_says() {
    use thin_ir::{Gather, GatherFile, GatherRegion, Reg};
    let mut values = vec![
        gathered('v', "x#1", 1, Some(300)),
        gathered('s', "args", 2, Some(0)),
        gathered('v', "q", 1, Some(310)),
        gathered('v', "q", 1, Some(311)),
        gathered('v', "y", 1, Some(320)),
    ];
    values[4].same_as = Some(2);
    values[0].parts = vec!["lo".into()];
    let file = |letter, placed| GatherFile {
        letter,
        lanes: vec!["x".into()],
        entered: vec!["q".into()],
        pinned: vec!["q".into(), "args".into()],
        placed,
    };
    let region = GatherRegion {
        key: 9,
        base: 64,
        size: 16,
        name: "ring".into(),
        parent: Some("lds".into()),
        on_layout: true,
        tags: vec!["a".into()],
    };
    let g = Gather {
        values,
        files: vec![file('v', false), file('s', true)],
        regions: vec![region],
        ..Default::default()
    };
    let s = Snapshot::of_kernel(default_target(), &g, Vec::new().into()).unwrap();
    let v = s.values();
    assert_eq!(
        v.iter().map(|v| v.group).collect::<Vec<_>>(),
        [0, 0, 1, 1, 2]
    );
    assert_eq!(
        v.iter().map(|v| v.covered).collect::<Vec<_>>(),
        [true, false, false, false, false]
    );
    assert_eq!(
        v.iter().map(|v| v.phys).collect::<Vec<_>>(),
        [Some(300), Some(0), Some(310), Some(311), Some(310)]
    );
    let through = |value, part| {
        let r = Reg {
            part,
            ..Reg::new(value, 'v', 0, 1)
        };
        (
            s.name(s.through(&r)).to_owned(),
            s.name(s.alias(&r)).to_owned(),
        )
    };
    assert_eq!(through(0, 1), ("x#1.lo".into(), "x#1.lo".into()));
    assert_eq!(through(4, 0), ("y".into(), "q".into()));
    assert_eq!(through(2, 0), ("q".into(), "q".into()));
    let entry: Vec<(i32, u32)> = s.entry().iter().map(|r| (r.value, r.n)).collect();
    assert_eq!(entry, [(2, 1), (3, 1), (1, 2)]);
    assert_eq!(&*s.region(9).unwrap().leaf, "lds");
    let bad = Gather {
        wave_reg: Some(Reg::new(7, 's', 0, 1)),
        ..Default::default()
    };
    let refused = Snapshot::of_kernel(default_target(), &bad, Vec::new().into()).unwrap_err();
    assert_eq!(
        refused,
        thin_ir::Error::Value("a register of a value not in the program".into())
    );
}

#[test]
fn an_alias_names_the_same_as_root_of_the_groups_first_value() {
    let mut values = vec![
        gathered('v', "a", 1, Some(256)),
        gathered('v', "b", 1, Some(257)),
    ];
    values[1].same_as = Some(0);
    let file = thin_ir::GatherFile {
        letter: 'v',
        ..Default::default()
    };
    let g = thin_ir::Gather {
        values,
        files: vec![file],
        ..Default::default()
    };
    let s = Snapshot::of_kernel(default_target(), &g, Vec::new().into()).unwrap();
    let r = thin_ir::Reg::new(1, 'v', 0, 1);
    assert_eq!((s.name(s.through(&r)), s.name(s.alias(&r))), ("b", "a"));
    assert_eq!(s.values()[1].phys, Some(256));
}

/// A muted snapshot still verifies: the instruction is replaced by a
/// comment with its text, not a bare one.
#[test]
fn a_muted_snapshot_verifies() {
    let lines: Vec<String> = ["v_mov_b32 v0, 0", "s_endpgm"].map(String::from).to_vec();
    let s = snapshot(&lines).muted(0);
    let values = HashMap::new();
    assert_eq!(
        thin_ir::verify::verify(s.target(), s.instrs(), &values, false),
        None
    );
}
