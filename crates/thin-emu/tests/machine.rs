//! Machine execution tests using gfx1201 target facts.

use thin_emu::machine::cluster::{Cluster, Options};
use thin_emu::machine::facts::Facts;
use thin_emu::machine::memory::{first_boundary, placement, Memory};
use thin_emu::machine::run::Machine;
use thin_emu::machine::wave::Wave;
use thin_emu::machine::POISON;
use thin_emu::{Base, Cause, EmuError, Field, Ins, Operand, Program, Tables};

/// A decoded instruction: operands `v1`, `s[4:7]`, `vcc`, `#7` (a
/// literal), `@L` (a label).
fn mk(canon: &str, kind: &str, ops: &[&str], fields: &[(&str, Field)]) -> Ins {
    let operand = |o: &str| {
        let base = if let Some(l) = o.strip_prefix('@') {
            Base::Label(l.to_owned())
        } else if let Some(v) = o.strip_prefix('#') {
            Base::Literal(v.parse().unwrap_or(0), v.to_owned())
        } else if o == "vcc" || o == "exec" || o == "off" {
            Base::Reg(o.to_owned(), 0, 2)
        } else if o == "ttmp9" || o == "exec_lo" {
            Base::Reg(o.trim_end_matches("_lo").to_owned(), 0, 1)
        } else {
            let (f, rest) = o.split_at(1);
            let (a, n) = match rest.strip_prefix('[') {
                Some(r) => {
                    let (a, b) = r.trim_end_matches(']').split_once(':').unwrap();
                    let (a, b): (u32, u32) = (a.parse().unwrap(), b.parse().unwrap());
                    (a, b - a + 1)
                }
                None => (rest.parse().unwrap(), 1),
            };
            Base::Reg(f.to_owned(), a, n)
        };
        Operand {
            base,
            neg: false,
            abs: false,
        }
    };
    let mut ins = Ins {
        canon: canon.to_owned(),
        kind: kind.to_owned(),
        operands: ops.iter().map(|o| operand(o)).collect(),
        fields: fields
            .iter()
            .map(|(k, v)| ((*k).to_owned(), v.clone()))
            .collect(),
        matrix: None,
        text: String::new(),
        comment: String::new(),
        line: 0,
    };
    ins.text = ins.default_text();
    ins
}

/// The default target's facts (gfx1201's).
fn facts() -> Facts {
    Facts::of(thin_targets::get(thin_targets::default_name()).expect("the default target"))
}

fn program(ins: Vec<Ins>, labels: &[(&str, usize)]) -> Program {
    Program {
        ins,
        labels: labels.iter().map(|(l, i)| ((*l).to_owned(), *i)).collect(),
        user_sgprs: 2,
        ..Program::default()
    }
}

fn one_wave(p: Program, init: &mut dyn FnMut(&mut Wave)) -> Result<Wave, EmuError> {
    let mut m = Machine::new(p, facts(), 16, 0, 0)?;
    let mut mem = Memory::default();
    let mut tables = Tables::default();
    let (mut ws, _) = m.run_wg(&mut mem, &mut tables, 0, (3, 4), 1, 0, Some(init))?;
    Ok(ws.remove(0))
}

#[test]
fn an_output_modifier_and_an_unknown_op_are_limits() {
    let clamp = mk(
        "v.add.f32",
        "valu",
        &["v1", "v2", "v3"],
        &[("clamp", Field::True)],
    );
    let e = one_wave(program(vec![clamp], &[]), &mut |_| {})
        .map(|_| ())
        .unwrap_err();
    assert_eq!(e.cause, Some(Cause::Emulator));
    assert!(e.message.contains("output modifier"), "{}", e.message);
    let unknown = mk("v.frobnicate.b32", "valu", &["v1"], &[]);
    let e = one_wave(program(vec![unknown], &[]), &mut |_| {})
        .map(|_| ())
        .unwrap_err();
    assert_eq!(e.cause, Some(Cause::Emulator));
    assert_eq!(e.message, "instruction not emulated: v.frobnicate.b32 v1");
}

#[test]
fn an_error_without_the_text_is_wrapped_and_keeps_its_class() {
    let read = mk("global.load.b32", "vmem", &["v1", "v[2:3]", "off"], &[]);
    let e = one_wave(program(vec![read], &[]), &mut |w| {
        for l in 0..32 {
            w.v[2 * 32 + l] = 0x10;
            w.v[3 * 32 + l] = 0;
        }
    })
    .map(|_| ())
    .unwrap_err();
    assert_eq!(e.cause, Some(Cause::Fault));
    assert_eq!(
        e.message,
        "wave 0 at 0 `global.load.b32 v1, v[2:3], off`: load from unmapped address 0x10"
    );
}

#[test]
fn a_wave_starts_poisoned_but_for_its_entry_registers() {
    let w = one_wave(
        program(vec![mk("s.nop", "nop", &["#0"], &[])], &[]),
        &mut |_| {},
    )
    .unwrap();
    // The kernarg pointer in s[0:1]; the workgroup ids in ttmp9 and ttmp7.
    assert_eq!((w.s[0], w.s[1], w.ttmp9, w.ttmp7), (0, 0, 3, 4));
    assert!(w.s[2..].iter().all(|s| *s == POISON));
    assert!((0..32).all(|l| w.v[l] == l as u32));
    assert!(w.v[32..].iter().all(|v| *v == POISON));
    assert_eq!((w.exec, w.m0, w.scc), (u64::from(u32::MAX), POISON, false));
    assert_eq!((w.vcc, w.vcc_hi), (u64::from(POISON), POISON));
}

#[test]
fn a_buffer_load_through_a_poison_descriptor_faults() {
    let load = mk(
        "buffer.load.b32",
        "vmem",
        &["v1", "v0", "s[8:11]", "#0"],
        &[("offen", Field::True)],
    );
    let e = one_wave(program(vec![load], &[]), &mut |_| {})
        .map(|_| ())
        .unwrap_err();
    assert_eq!(
        e.message,
        "wave 0 buffer.load.b32 v1, v0, s[8:11], 0: its descriptor s[8:11] holds a register no \
         instruction wrote (on the GPU: what the last kernel left)"
    );
}

#[test]
fn buffers_straddle_4_gib_boundaries() {
    let mut next = first_boundary(1 << 32);
    let mut bases = Vec::new();
    for size in [64u64, 1000, 1 << 20, 6 << 30] {
        let (base, n) = placement(next, size);
        bases.push(base);
        next = n;
    }
    assert_eq!(
        bases,
        [
            0x1000_FFFF_FF00,
            0x1002_FFFF_FE00,
            0x1004_FFF8_0000,
            0x1006_8000_0000
        ]
    );
    assert_eq!(placement(first_boundary(1 << 31), 4096).0, 0x1001_7FFF_F800);
    let mut mem = Memory::default();
    let b = mem.alloc("x", 64, true, 0, Default::default()).unwrap();
    assert_eq!(mem.bufs[b].base, 0x1000_FFFF_FF00);
}

#[test]
fn unmapped_and_past_the_end_accesses_fault() {
    let mut mem = Memory::default();
    let b = mem.alloc("x", 64, true, 0, Default::default()).unwrap();
    let base = mem.bufs[b].base;
    assert_eq!(mem.access(base + 60, 4, false).unwrap(), (b, 60));
    assert_eq!(
        mem.access(base + 62, 4, false).unwrap_err().message,
        "load past the end of x"
    );
    assert_eq!(
        mem.access(base + 64, 4, true).unwrap_err().message,
        format!("store to unmapped address {:#x}", base + 64)
    );
    assert_eq!(
        mem.locate(7).unwrap_err().message,
        "address 0x7 is in no buffer"
    );
}

#[test]
fn a_buffer_load_reads_zero_past_num_records() {
    let mut m = Machine::new(
        program(
            vec![mk(
                "buffer.load.b32",
                "vmem",
                &["v1", "v0", "s[8:11]", "#0"],
                &[("offen", Field::True)],
            )],
            &[],
        ),
        facts(),
        4,
        0,
        0,
    )
    .unwrap();
    let mut mem = Memory::default();
    let b = mem.alloc("x", 256, true, 7, Default::default()).unwrap();
    let base = mem.bufs[b].base;
    let mut tables = Tables::default();
    let mut init = |w: &mut Wave| {
        w.s[8] = base as u32;
        w.s[9] = (base >> 32) as u32;
        w.s[10] = 10;
        w.s[11] = 0;
        for (l, v) in w.v[..32].iter_mut().enumerate() {
            *v = l as u32 * 4;
        }
    };
    let (ws, _) = m
        .run_wg(&mut mem, &mut tables, 0, (0, 0), 1, 0, Some(&mut init))
        .unwrap();
    let got: Vec<u32> = (0..4).map(|l| ws[0].v[32 + l]).collect();
    assert_eq!(got, [0x0707_0707, 0x0707_0707, 0, 0]);
}

/// The poll program: the wave whose kernarg dword 0 is 0 polls f[0] until
/// it is 1, the other sets it (or, without `set`, does not).
fn poll_program(set: bool, spin: bool, bound: bool) -> Program {
    let mut ins = vec![
        mk("s.load.b32", "smem", &["s10", "s[0:1]", "#0"], &[]),
        mk("s.load.b64", "smem", &["s[20:21]", "s[0:1]", "#8"], &[]),
        mk("s.wait.kmcnt", "wait", &["#0"], &[]),
        mk("v.mov.b32", "valu", &["v6", "#0"], &[]),
        mk("v.mov.b32", "valu", &["v3", "#1"], &[]),
        mk("s.mov.b32", "salu", &["s30", "#0"], &[]),
        mk("s.cmp.eq.u32", "salu", &["s10", "#0"], &[]),
        mk("s.cbranch.scc0", "branch", &["@setter"], &[]),
    ];
    let top = ins.len();
    if spin {
        let mut count = mk("s.add.u32", "salu", &["s30", "s30", "#1"], &[]);
        count.comment = "SPIN".into();
        ins.push(count);
    }
    ins.push(mk(
        "global.load.b32",
        "vmem",
        &["v7", "v6", "s[20:21]"],
        &[("scope", Field::Int(2))],
    ));
    ins.push(mk("s.wait.loadcnt", "wait", &["#0"], &[]));
    if bound {
        ins.push(mk("s.cmp.gt.u32", "salu", &["s30", "#1000000"], &[]));
        ins.push(mk("s.cbranch.scc1", "branch", &["@end"], &[]));
    }
    ins.push(mk("v.cmp.eq.u32", "valu", &["vcc", "v7", "#0"], &[]));
    ins.push(mk("s.cbranch.vccnz", "branch", &["@top"], &[]));
    ins.push(mk("s.branch", "branch", &["@end"], &[]));
    let setter = ins.len();
    if set {
        ins.push(mk(
            "global.store.b32",
            "vmem",
            &["v6", "v3", "s[20:21]"],
            &[("scope", Field::Int(2))],
        ));
    }
    let end = ins.len();
    ins.push(mk("s.wait.loadcnt", "wait", &["#0"], &[]));
    ins.push(mk("s.endpgm", "salu", &[], &[]));
    program(ins, &[("top", top), ("setter", setter), ("end", end)])
}

/// A two-rank cluster over a flag: rank 0 polls f[0] until it is 1, rank
/// 1 sets it (after `delay` turns of skew, or never).
fn poll_cluster(set: bool, spin: bool, bound: bool, skew: u64) -> Result<Cluster, EmuError> {
    let p = poll_program(set, spin, bound);
    let mut mem = Memory::default();
    let flag = mem.alloc("flag", 64, true, 0, Default::default())?;
    let fb = mem.bufs[flag].base;
    let mut machines = Vec::new();
    let mut launches = Vec::new();
    for r in 0..2usize {
        let k = mem.alloc(&format!("k{r}"), 16, false, 0, Default::default())?;
        mem.bufs[k].data[..4].copy_from_slice(&(r as u32).to_le_bytes());
        mem.bufs[k].data[8..16].copy_from_slice(&fb.to_le_bytes());
        launches.push(vec![mem.bufs[k].base]);
        let mut m = Machine::new(p.clone(), facts(), 8, 0, 0)?;
        m.rank = r;
        machines.push(m);
    }
    let options = Options {
        skew: vec![0, skew],
        stall_rounds: 200,
        ..Options::default()
    };
    let mut c = Cluster::new(machines, launches, (1, 1), 1, options);
    let mut tables = Tables::default();
    c.run(&mut mem, &mut tables)?;
    Ok(c)
}

#[test]
fn a_poll_parks_and_wakes_on_the_peers_store() {
    let c = poll_cluster(true, false, false, 30).unwrap();
    assert!(c.rounds > 30);
}

#[test]
fn a_poll_nobody_answers_deadlocks_naming_the_parked_wave() {
    let e = poll_cluster(false, false, false, 0)
        .map(|_| ())
        .unwrap_err();
    assert_eq!(e.cause, Some(Cause::Deadlock));
    assert!(
        e.message
            .starts_with("deadlock: no wave can run: rank 0 wg 0 wave 0 at "),
        "{}",
        e.message
    );
    assert!(e.message.ends_with(" waits on flag"), "{}", e.message);
}

/// One workgroup of the poll program on the single-machine runner.
fn poll_alone(spin: bool, bound: bool) -> Result<(), EmuError> {
    let mut mem = Memory::default();
    let flag = mem.alloc("flag", 64, true, 0, Default::default())?;
    let fb = mem.bufs[flag].base;
    let k = mem.alloc("k", 16, false, 0, Default::default())?;
    mem.bufs[k].data[8..16].copy_from_slice(&fb.to_le_bytes());
    let base = mem.bufs[k].base;
    let mut m = Machine::new(poll_program(false, spin, bound), facts(), 8, 0, 0)?;
    let mut tables = Tables::default();
    m.run_wg(&mut mem, &mut tables, base, (0, 0), 1, 0, None)?;
    Ok(())
}

#[test]
fn the_single_machine_runner_finds_a_poll_nobody_answers() {
    let e = poll_alone(false, false).unwrap_err();
    assert_eq!(e.cause, Some(Cause::Deadlock));
    assert_eq!(e.message, "deadlock: no wave can run: wave 0 waits on flag");
    // A bounded poll times out and leaves; one that never leaves deadlocks.
    poll_alone(true, true).unwrap();
    let e = poll_alone(true, false).unwrap_err();
    assert_eq!(e.cause, Some(Cause::Deadlock));
    assert!(
        e.message
            .starts_with("bounded polls timed out 64 times in a row"),
        "{}",
        e.message
    );
}

#[test]
fn a_bounded_poll_times_out_and_too_many_time_outs_deadlock() {
    // The poll's round counter is pushed past its bound: the wave leaves
    // its loop through the bound check and the run ends.
    poll_cluster(false, true, true, 0).unwrap();
    // A poll that never leaves: time-outs in a row are a deadlock.
    let e = poll_cluster(false, true, false, 0).map(|_| ()).unwrap_err();
    assert_eq!(e.cause, Some(Cause::Deadlock));
    assert!(
        e.message
            .starts_with("bounded polls timed out 64 times in a row"),
        "{}",
        e.message
    );
}

#[test]
fn stall_rounds_end_a_run_without_progress() {
    let ins = vec![
        mk("s.mov.b32", "salu", &["s5", "#0"], &[]),
        mk("s.add.u32", "salu", &["s5", "s5", "#1"], &[]),
        mk("s.branch", "branch", &["@top"], &[]),
    ];
    let p = program(ins, &[("top", 1)]);
    let m = Machine::new(p, facts(), 4, 0, 0).unwrap();
    let options = Options {
        stall_rounds: 7,
        ..Options::default()
    };
    let mut c = Cluster::new(vec![m], vec![vec![0]], (1, 1), 1, options);
    let mut mem = Memory::default();
    let e = c.run(&mut mem, &mut Tables::default()).unwrap_err();
    assert_eq!(e.cause, Some(Cause::Deadlock));
    assert!(
        e.message
            .starts_with("no progress for 7 rounds: rank 0 wg 0 wave 0 at "),
        "{}",
        e.message
    );
}

/// The order in which four waves' LDS atomics land: each wave adds 1 to
/// LDS word 0 and stores the old value at out[4 k] (lane 0's thread id
/// 32 k, shifted right by 3).
fn atomic_ranks(wave_jitter: Option<u64>) -> Vec<u32> {
    let ins = vec![
        mk("v.mov.b32", "valu", &["v1", "#0"], &[]),
        mk("v.mov.b32", "valu", &["v2", "#1"], &[]),
        mk("v.lshrrev.b32", "valu", &["v4", "#3", "v0"], &[]),
        mk("s.mov.b64", "salu", &["exec", "#1"], &[]),
        mk("ds.add.rtn.u32", "ds", &["v3", "v1", "v2"], &[]),
        mk("s.wait.dscnt", "wait", &["#0"], &[]),
        mk("global.store.b32", "vmem", &["v4", "v3", "s[0:1]"], &[]),
        mk("s.wait.loadcnt", "wait", &["#0"], &[]),
        mk("s.endpgm", "salu", &[], &[]),
    ];
    let m = Machine::new(program(ins, &[]), facts(), 8, 0, 64).unwrap();
    let mut mem = Memory::default();
    let out = mem.alloc("out", 64, true, 0, Default::default()).unwrap();
    let options = Options {
        wave_jitter,
        ..Options::default()
    };
    let mut c = Cluster::new(vec![m], vec![vec![mem.bufs[out].base]], (1, 1), 4, options);
    c.run(&mut mem, &mut Tables::default()).unwrap();
    let d = &mem.bufs[out].data;
    (0..4)
        .map(|k| u32::from_le_bytes(d[4 * k..4 * k + 4].try_into().unwrap()))
        .collect()
}

#[test]
fn wave_jitter_changes_lds_atomic_ranks_across_seeds() {
    assert_eq!(atomic_ranks(None), [0, 1, 2, 3]);
    let runs: std::collections::BTreeSet<Vec<u32>> =
        (1..6).map(|s| atomic_ranks(Some(s))).collect();
    assert!(runs.len() > 1, "{runs:?}");
}

/// The order workgroups run in: after a delay loop of `spin` rounds, each
/// takes a ticket from a global atomic and stores its tag (rank << 16 |
/// launch << 8 | wg) at out[ticket]. `launches` per rank, `grid`
/// workgroups a launch, over `xcds` L2s; returns the tags in ticket order.
fn ticket_order(
    spin: &[u32],
    launches: usize,
    grid: u32,
    xcds: usize,
    options: Options,
) -> Vec<u32> {
    let ranks = spin.len();
    let ins = vec![
        mk("s.load.b32", "smem", &["s10", "s[0:1]", "#0"], &[]),
        mk("s.load.b32", "smem", &["s11", "s[0:1]", "#4"], &[]),
        mk("s.load.b64", "smem", &["s[20:21]", "s[0:1]", "#8"], &[]),
        mk("s.load.b64", "smem", &["s[22:23]", "s[0:1]", "#16"], &[]),
        mk("s.load.b32", "smem", &["s30", "s[0:1]", "#24"], &[]),
        mk("s.wait.kmcnt", "wait", &["#0"], &[]),
        // top (6): the delay loop
        mk("s.sub.u32", "salu", &["s30", "s30", "#1"], &[]),
        mk("s.cmp.lg.u32", "salu", &["s30", "#0"], &[]),
        mk("s.cbranch.scc1", "branch", &["@top"], &[]),
        mk("v.mov.b32", "valu", &["v6", "#0"], &[]),
        mk("v.mov.b32", "valu", &["v3", "#1"], &[]),
        mk("s.mov.b64", "salu", &["exec", "#1"], &[]),
        mk(
            "global.atomic.add.u32",
            "vmem",
            &["v7", "v6", "v3", "s[20:21]"],
            &[],
        ),
        mk("s.wait.loadcnt", "wait", &["#0"], &[]),
        mk("v.lshlrev.b32", "valu", &["v9", "#2", "v7"], &[]),
        mk("s.lshl.b32", "salu", &["s14", "s10", "#16"], &[]),
        mk("s.lshl.b32", "salu", &["s15", "s11", "#8"], &[]),
        mk("s.or.b32", "salu", &["s14", "s14", "s15"], &[]),
        mk("s.or.b32", "salu", &["s14", "s14", "ttmp9"], &[]),
        mk("v.mov.b32", "valu", &["v10", "s14"], &[]),
        mk("global.store.b32", "vmem", &["v9", "v10", "s[22:23]"], &[]),
        mk("s.wait.loadcnt", "wait", &["#0"], &[]),
        mk("s.endpgm", "salu", &[], &[]),
    ];
    let p = program(ins, &[("top", 6)]);
    let total = ranks * launches * grid as usize;
    let mut mem = Memory::default();
    let counter = mem
        .alloc("counter", 64, true, 0, Default::default())
        .unwrap();
    let out = mem
        .alloc("out", 4 * total as u64, true, 0, Default::default())
        .unwrap();
    let (cb, ob) = (mem.bufs[counter].base, mem.bufs[out].base);
    let mut machines = Vec::new();
    let mut blocks = Vec::new();
    for (r, &delay) in spin.iter().enumerate() {
        let mut mine = Vec::new();
        for l in 0..launches {
            let k = mem
                .alloc(&format!("k{r}.{l}"), 32, false, 0, Default::default())
                .unwrap();
            let d = &mut mem.bufs[k].data;
            d[..4].copy_from_slice(&(r as u32).to_le_bytes());
            d[4..8].copy_from_slice(&(l as u32).to_le_bytes());
            d[8..16].copy_from_slice(&cb.to_le_bytes());
            d[16..24].copy_from_slice(&ob.to_le_bytes());
            d[24..28].copy_from_slice(&delay.max(1).to_le_bytes());
            mine.push(mem.bufs[k].base);
        }
        blocks.push(mine);
        let many = Facts { xcds, ..facts() };
        let mut m = Machine::new(p.clone(), many, 16, 0, 0).unwrap();
        m.rank = r;
        machines.push(m);
    }
    let mut c = Cluster::new(machines, blocks, (grid, 1), 1, options);
    c.run(&mut mem, &mut Tables::default()).unwrap();
    let d = &mem.bufs[out].data;
    (0..total)
        .map(|i| u32::from_le_bytes(d[4 * i..4 * i + 4].try_into().unwrap()))
        .collect()
}

fn tag(rank: u32, launch: u32, wg: u32) -> u32 {
    rank << 16 | launch << 8 | wg
}

#[test]
fn skew_holds_a_rank_back_and_speed_slows_it() {
    let skewed = Options {
        skew: vec![50, 0],
        ..Options::default()
    };
    assert_eq!(
        ticket_order(&[1, 1], 1, 1, 1, skewed),
        [tag(1, 0, 0), tag(0, 0, 0)]
    );
    assert_eq!(
        ticket_order(&[1, 1], 1, 1, 1, Options::default()),
        [tag(0, 0, 0), tag(1, 0, 0)]
    );
    // Equal delay loops: at a sixteenth of the quantum, rank 0 arrives last.
    let slow = Options {
        speed: vec![1.0 / 16.0, 1.0],
        ..Options::default()
    };
    assert_eq!(
        ticket_order(&[100, 100], 1, 1, 1, slow),
        [tag(1, 0, 0), tag(0, 0, 0)]
    );
}

#[test]
fn lockstep_holds_a_rank_at_the_next_launch_until_every_rank_ended() {
    let free = Options {
        skew: vec![0, 200],
        ..Options::default()
    };
    // Without lockstep rank 0 runs both launches before the skewed rank 1.
    assert_eq!(
        ticket_order(&[1, 1], 2, 1, 1, free),
        [tag(0, 0, 0), tag(0, 1, 0), tag(1, 0, 0), tag(1, 1, 0)]
    );
    let lockstep = Options {
        skew: vec![0, 200],
        lockstep: true,
        ..Options::default()
    };
    assert_eq!(
        ticket_order(&[1, 1], 2, 1, 1, lockstep),
        [tag(0, 0, 0), tag(1, 0, 0), tag(0, 1, 0), tag(1, 1, 0)]
    );
}

#[test]
fn a_hog_runs_l2_0_first_and_ends_when_nothing_else_can_dispatch() {
    // 16 workgroups over 8 L2s (x mod 8; gfx1201 has one): with room for one
    // workgroup on L2 0 and none elsewhere, 0 and 8 run first, one at a
    // time; then nothing can dispatch, the hog ends and the rest run.
    let hog = Options {
        hog: Some(vec![1]),
        ..Options::default()
    };
    let order = ticket_order(&[1], 1, 16, 8, hog);
    assert_eq!(order[..2], [tag(0, 0, 0), tag(0, 0, 8)]);
    let mut rest: Vec<u32> = order[2..].to_vec();
    rest.sort_unstable();
    let want: Vec<u32> = (0..16)
        .filter(|x| x % 8 != 0)
        .map(|x| tag(0, 0, x))
        .collect();
    assert_eq!(rest, want);
    // Without the hog, the grid order.
    let plain = ticket_order(&[1], 1, 16, 8, Options::default());
    assert_eq!(plain, (0..16).map(|x| tag(0, 0, x)).collect::<Vec<_>>());
}

#[test]
fn fnuz_fp8_conversions_follow_target_hook() {
    let fnuz_target = thin_targets::names()
        .iter()
        .map(|n| thin_targets::get(n).unwrap())
        .find(|t| t.hooks().fnuz_fp8())
        .expect("a target with fnuz_fp8");
    let fnuz_facts = Facts::of(fnuz_target);
    assert!(fnuz_facts.fnuz_fp8);
    assert!(!facts().fnuz_fp8);

    let run = |f: Facts| {
        let p = program(
            vec![
                mk("v.cvt.pk.fp8.f32", "valu", &["v0", "v1", "v2"], &[]),
                mk(
                    "v.cvt.pk.fp8.f32",
                    "valu",
                    &["v0", "v3", "v4"],
                    &[("op_sel", Field::List(vec![0, 0, 1]))],
                ),
                mk("v.cvt.f32.fp8", "valu", &["v5", "v0"], &[]),
                mk("v.cvt.pk.bf8.f32", "valu", &["v6", "v7", "v8"], &[]),
            ],
            &[],
        );
        let mut m = Machine::new(p, f, 16, 0, 0).unwrap();
        let mut mem = Memory::default();
        let mut tables = Tables::default();
        let (mut ws, _) = m
            .run_wg(
                &mut mem,
                &mut tables,
                0,
                (0, 0),
                1,
                0,
                Some(&mut |w| {
                    w.v[w.w] = (-0.0f32).to_bits(); // v1 lane 0: -0.0
                    w.v[2 * w.w] = 240.0f32.to_bits(); // v2 lane 0: 240.0
                    w.v[3 * w.w] = 248.0f32.to_bits(); // v3 lane 0: 248.0 (overflow on FNUZ)
                    w.v[4 * w.w] = f32::INFINITY.to_bits(); // v4 lane 0: +inf
                    w.v[7 * w.w] = f32::INFINITY.to_bits(); // v7 lane 0: +inf
                    w.v[8 * w.w] = (-0.0f32).to_bits(); // v8 lane 0: -0.0
                }),
            )
            .unwrap();
        let w = ws.remove(0);
        (w.v[0], w.v[5 * w.w], w.v[6 * w.w] & 0xFFFF)
    };

    // FNUZ: [-0.0, 240.0, 248.0, +inf] -> [0x00, 0x7F, 0x80, 0x80]; bf8 [+inf, -0.0] -> [0x80, 0x00].
    assert_eq!(run(fnuz_facts), (0x8080_7F00, 0.0f32.to_bits(), 0x0080));
    // OCP:  [-0.0, 240.0, 248.0, +inf] -> [0x80, 0x77, 0x78, 0x7F]; bf8 [+inf, -0.0] -> [0x7C, 0x80].
    assert_eq!(run(facts()), (0x7F78_7780, (-0.0f32).to_bits(), 0x807C));
}

#[test]
fn wave64_all_matrix_forms_execute_in_agpr_and_vgpr() {
    use thin_emu::machine::program::read_program;
    let w64_targets: Vec<_> = thin_targets::names()
        .iter()
        .map(|n| thin_targets::get(n).unwrap())
        .filter(|t| t.definition().wave == 64)
        .collect();
    assert!(!w64_targets.is_empty(), "at least one wave64 target");

    let reg = |file: char, first: u32, n: u32| -> String {
        if n == 1 {
            format!("{file}{first}")
        } else {
            format!("{file}[{first}:{}]", first + n - 1)
        }
    };

    for w64_target in w64_targets {
        let f = Facts::of(w64_target);
        assert!(f.matrix.len() >= 46);

        for (idx, (mnemonic, form)) in f.matrix.iter().enumerate() {
            let use_agpr = idx % 2 == 0;
            let cd_file = if use_agpr { 'a' } else { 'v' };
            let dst = reg(cd_file, 0, form.dwords('d'));
            let src0 = reg('v', 32, form.dwords('a'));
            let src1 = reg('a', 32, form.dwords('b'));
            let src2 = if form.sparse {
                reg('v', 48, 1)
            } else {
                dst.clone()
            };
            let asm = if form.scaled {
                format!("\t{mnemonic} {dst}, {src0}, {src1}, {src2}, v56, v57\n\ts_endpgm\n")
            } else {
                format!("\t{mnemonic} {dst}, {src0}, {src1}, {src2}\n\ts_endpgm\n")
            };
            let prog = read_program(&asm, w64_target).expect(&asm);
            let mut m = Machine::new(prog, f.clone(), 64, 64, 0).unwrap();
            let mut mem = Memory::default();
            let mut tables = Tables::default();
            let elem_word = |e: thin_target::Elem| -> u32 {
                match e {
                    thin_target::Elem::F16 => 0x3C00_3C00,
                    thin_target::Elem::Bf16 => 0x3F80_3F80,
                    thin_target::Elem::F32 | thin_target::Elem::Xf32 => 1.0f32.to_bits(),
                    thin_target::Elem::F64 => 0,
                    thin_target::Elem::I8 => 0x0201_0201,
                    thin_target::Elem::Fp8 | thin_target::Elem::F8F6F4 => 0x3838_3838,
                    thin_target::Elem::Bf8 => 0x3C3C_3C3C,
                    thin_target::Elem::Fp8Fnuz | thin_target::Elem::Bf8Fnuz => 0x4040_4040,
                    _ => 1,
                }
            };
            let unit_word = elem_word(form.a);
            let unit_b = elem_word(form.b);
            let (mut ws, _) = m
                .run_wg(
                    &mut mem,
                    &mut tables,
                    0,
                    (0, 0),
                    1,
                    0,
                    Some(&mut |w| {
                        w.v.fill(0);
                        w.a.fill(0);
                        for d in 0..form.dwords('a') as usize {
                            let start = (32 + d) * 64;
                            let val = if form.a == thin_target::Elem::F64 && d % 2 == 1 {
                                0x3FF0_0000
                            } else {
                                unit_word
                            };
                            w.v[start..start + 64].fill(val);
                        }
                        for d in 0..form.dwords('b') as usize {
                            let start = (32 + d) * 64;
                            let val = if form.b == thin_target::Elem::F64 && d % 2 == 1 {
                                0x4000_0000
                            } else {
                                unit_b
                            };
                            w.a[start..start + 64].fill(val);
                        }
                        if form.sparse {
                            w.v[48 * 64..49 * 64].fill(0xDDDD_DDDD); // nibbles 0b11_01 -> idx0=1, idx1=3
                        }
                        if form.scaled {
                            w.v[56 * 64..57 * 64].fill(0x7F7F_7F7F); // 1.0 E8M0 scale
                            w.v[57 * 64..58 * 64].fill(0x7F7F_7F7F);
                        }
                    }),
                )
                .unwrap_or_else(|e| panic!("{mnemonic} failed: {e:?}"));
            assert_eq!(m.reports.stats.matrix, 1);
            let w = ws.remove(0);
            let out_slice = if use_agpr {
                &w.a[..form.dwords('d') as usize * 64]
            } else {
                &w.v[..form.dwords('d') as usize * 64]
            };
            assert!(
                out_slice.iter().any(|&x| x != 0),
                "{mnemonic} produced all-zero output"
            );
        }
    }
}
