//! gfx1201 instruction semantics on small synthetic programs.

use thin_emu::machine::cluster::{Cluster, Options};
use thin_emu::machine::facts::Facts;
use thin_emu::machine::memory::{Memory, Meta};
use thin_emu::machine::run::Machine;
use thin_emu::{Cause, EmuError};

const HEAD: &str = "s_load_b64 s[4:5], s[0:1], 0x0\ns_wait_kmcnt 0\nv_lshlrev_b32 v9, 2, v0\n";

/// Runs `body` on one 32-lane wave with `x` (`words` dwords, IPC when
/// `ipc`) at s[4:5] and v9 = lane · 4; returns x.
fn run_ipc(body: &str, words: usize, ipc: bool) -> Result<Vec<u32>, EmuError> {
    let t = thin_targets::get(thin_targets::default_name()).expect("registered default target");
    let text = format!("{HEAD}{body}\ns_endpgm\n");
    let program = thin_emu::read_program(&text, t)?;
    let mut mem = Memory::default();
    let meta = Meta {
        ipc,
        ..Default::default()
    };
    let x = mem.alloc("x", 4 * words as u64, true, 0, meta)?;
    let kb = mem.alloc("kernarg", 8, false, 0, Default::default())?;
    let base = mem.bufs[x].base;
    mem.bufs[kb].data[..8].copy_from_slice(&base.to_le_bytes());
    let machine = Machine::new(program, Facts::of(t), 64, 0, 256)?;
    let mut c = Cluster::new(
        vec![machine],
        vec![vec![mem.bufs[kb].base]],
        (1, 1),
        1,
        Options::default(),
    );
    c.run(&mut mem, &mut thin_emu::Tables::default())?;
    Ok((0..words)
        .map(|i| u32::from_le_bytes(mem.bufs[x].data[4 * i..4 * i + 4].try_into().unwrap()))
        .collect())
}

fn run(body: &str) -> Vec<u32> {
    run_ipc(body, 32, false).unwrap()
}

#[test]
fn cmpx_writes_exec_only() {
    // EXEC lanes 0..7; the compare is true in every lane: EXEC stays 0xff
    // (inactive lanes 0), and VCC keeps its value.
    let x = run("s_mov_b32 vcc_lo, 0x1234\ns_mov_b32 exec_lo, 0xff\n\
         v_cmpx_le_u32 0, v0\ns_mov_b32 s6, exec_lo\ns_mov_b32 s7, vcc_lo\n\
         s_mov_b32 exec_lo, -1\nv_mov_b32 v1, s6\nv_mov_b32 v2, s7\n\
         v_cmp_eq_u32 vcc_lo, 1, v0\nv_cndmask_b32 v1, v1, v2, vcc_lo\n\
         global_store_b32 v9, v1, s[4:5]");
    assert_eq!(x[0], 0xff);
    assert_eq!(x[1], 0x1234);
    // Lanes 0..15 by a compare: the store runs in those only.
    let x = run("v_mov_b32 v1, 7\nv_cmpx_gt_u32 16, v0\nglobal_store_b32 v9, v1, s[4:5]");
    assert_eq!(
        x,
        (0..32)
            .map(|l| if l < 16 { 7 } else { 0 })
            .collect::<Vec<_>>()
    );
}

#[test]
fn permlane16_selects_by_nibble_and_skips_inactive_sources() {
    // lanesel s7:s6 = 0xfedcba98_76543210 reversed: lane i takes 15 − i.
    let body = "s_mov_b32 s6, 0x89abcdef\ns_mov_b32 s7, 0x01234567\n\
         v_add_nc_u32 v1, 100, v0\nv_mov_b32 v2, 1\n\
         s_mov_b32 exec_lo, 0xfffffffe\nv_permlane16_b32 v2, v1, s6, s7{}\n\
         s_mov_b32 exec_lo, -1\nglobal_store_b32 v9, v2, s[4:5]";
    let src = |l: usize| l - l % 16 + 15 - l % 16;
    for (sel, at15) in [("", 1), (" op_sel:[1,0]", 100), (" op_sel:[0,1]", 0)] {
        let x = run(&body.replace("{}", sel));
        for (l, v) in x.iter().enumerate() {
            let want = match l {
                0 => 1,     // inactive: unwritten
                15 => at15, // its source lane 0 is inactive
                _ => 100 + src(l) as u32,
            };
            assert_eq!(*v, want, "{sel:?} lane {l}");
        }
    }
}

fn bits(text: &[&str]) -> Vec<u32> {
    text.iter()
        .map(|t| match *t {
            "qnan" => 0x7FC0_0000,
            "snan" => 0x7F80_0001,
            "-0" => 0x8000_0000,
            v => v.parse::<f32>().unwrap().to_bits(),
        })
        .collect()
}

/// One three-operand op on lane 0's (x, y, z).
fn three(op: &str, x: u32, y: u32, z: u32) -> u32 {
    let body = format!(
        "v_mov_b32 v1, {x:#x}\nv_mov_b32 v2, {y:#x}\nv_mov_b32 v3, {z:#x}\n\
         {op} v4, v1, v2, v3\nglobal_store_b32 v9, v4, s[4:5]"
    );
    run(&body)[0]
}

#[test]
fn three_operand_min_max_med() {
    let q = 0x7FC0_0000;
    let cases: [(&str, [&str; 3], u32); 12] = [
        ("v_med3_num_f32", ["3", "1", "2"], 2.0f32.to_bits()),
        ("v_med3_num_f32", ["1", "qnan", "2"], 1.0f32.to_bits()),
        ("v_med3_num_f32", ["-0", "0", "0"], 0),
        ("v_med3_num_f32", ["0", "-0", "-0"], 0x8000_0000),
        ("v_max3_num_f32", ["qnan", "qnan", "1"], 1.0f32.to_bits()),
        ("v_min3_num_f32", ["0", "-0", "1"], 0x8000_0000),
        ("v_minimum3_f32", ["1", "qnan", "0"], q),
        ("v_minimum3_f32", ["snan", "1", "2"], 0x7FC0_0001), // quieted, payload kept (the GPU)
        ("v_minimum3_f32", ["0", "-0", "1"], 0x8000_0000),
        ("v_maximum3_f32", ["-0", "0", "-1"], 0),
        ("v_maximum3_f32", ["1", "3", "2"], 3.0f32.to_bits()),
        ("v_minimum3_f32", ["2", "1", "3"], 1.0f32.to_bits()),
    ];
    for (op, args, want) in cases {
        let v = bits(&args);
        assert_eq!(three(op, v[0], v[1], v[2]), want, "{op} {args:?}");
    }
}

#[test]
fn ipc_needs_device_scope_or_a_non_temporal_hint() {
    let store = |fields: &str| {
        run_ipc(
            &format!("v_mov_b32 v1, 5\nglobal_store_b32 v9, v1, s[4:5]{fields}"),
            32,
            true,
        )
    };
    for ok in [
        " scope:SCOPE_DEV",
        " scope:SCOPE_SYS",
        " th:TH_STORE_NT",
        " th:TH_STORE_NT_WB",
    ] {
        assert_eq!(store(ok).expect(ok)[3], 5, "{ok}");
    }
    for bad in ["", " scope:SCOPE_SE", " th:TH_STORE_HT"] {
        let e = store(bad).expect_err(bad);
        assert_eq!(e.cause, Some(Cause::Ipc), "{bad}: {}", e.message);
        // message with reading of its policy phrase.
        assert_eq!(
            e.message,
            format!(
                "global_store_b32 v9, v1, s[4:5]{bad}: accesses the IPC buffer x without \
                 scope ≥ SCOPE_DEV or a non-temporal th (another rank's view)"
            )
        );
    }
}

#[test]
fn a_scalar_load_of_an_ipc_buffer_needs_device_scope() {
    let load = |fields: &str| {
        run_ipc(
            &format!("s_load_b32 s8, s[4:5], 0x0{fields}\ns_wait_kmcnt 0"),
            32,
            true,
        )
    };
    load(" scope:SCOPE_DEV").expect("device scope");
    let e = load("").expect_err("no scope");
    assert_eq!(e.cause, Some(Cause::Ipc), "{}", e.message);
}

#[test]
fn returning_atomics_count_on_load_and_cmpstore_takes_the_new_value_first() {
    // A returning atomic: the old value comes back after the load wait.
    let x = run_ipc(
        "v_mov_b32 v1, 3\nglobal_store_b32 v9, v1, s[4:5]\ns_wait_storecnt 0\n\
         v_mov_b32 v2, 4\nglobal_atomic_add_u32 v3, v9, v2, s[4:5] th:TH_ATOMIC_RETURN\n\
         s_wait_loadcnt 0\nglobal_store_b32 v9, v3, s[4:5] offset:128",
        64,
        false,
    )
    .unwrap();
    assert_eq!((x[0], x[32]), (7, 3));
    // ds_cmpstore_rtn_b32 d, addr, data, cmp: data the new value.
    let x = run(
        "v_mov_b32 v1, 0\nv_mov_b32 v2, 9\nds_store_b32 v9, v2\ns_wait_dscnt 0\n\
         v_mov_b32 v3, 11\nv_mov_b32 v4, 9\nds_cmpstore_rtn_b32 v5, v9, v3, v4\n\
         s_wait_dscnt 0\nds_load_b32 v6, v9\ns_wait_dscnt 0\n\
         v_lshl_or_b32 v7, v5, 8, v6\nglobal_store_b32 v9, v7, s[4:5]",
    );
    assert_eq!(x[0], 9 << 8 | 11);
}

/// Two ranks over one memory (the scope model on): rank 0 stores 7 at
/// SCOPE_CU, runs `wb` and stays busy; rank 1, later, loads at SCOPE_DEV. Only a
/// SCOPE_SYS write-back reaches the L2.
fn two_ranks(wb: &str) -> Result<(), EmuError> {
    let t = thin_targets::get(thin_targets::default_name()).expect("registered default target");
    let text = format!(
        "s_load_b32 s10, s[0:1], 0x0\ns_load_b64 s[4:5], s[0:1], 0x8\ns_wait_kmcnt 0\n\
         v_mov_b32 v6, 0\nv_mov_b32 v3, 7\ns_cmp_eq_u32 s10, 0\ns_cbranch_scc0 rank1\n\
         global_store_b32 v6, v3, s[4:5]\n{wb}\ns_wait_storecnt 0\ns_mov_b32 s11, 1000\n\
         busy:\ns_sub_co_u32 s11, s11, 1\ns_cmp_lg_u32 s11, 0\ns_cbranch_scc1 busy\ns_endpgm\n\
         rank1:\nglobal_load_b32 v4, v6, s[4:5] scope:SCOPE_DEV\ns_wait_loadcnt 0\ns_endpgm\n"
    );
    let program = thin_emu::read_program(&text, t)?;
    let mut mem = Memory::default();
    let buf = mem.alloc("buf", 64, true, 0, Meta::default())?;
    let base = mem.bufs[buf].base;
    let (mut machines, mut launches) = (Vec::new(), Vec::new());
    for r in 0..2usize {
        let k = mem.alloc(&format!("kernarg{r}"), 16, false, 0, Meta::default())?;
        mem.bufs[k].data[..4].copy_from_slice(&(r as u32).to_le_bytes());
        mem.bufs[k].data[8..16].copy_from_slice(&base.to_le_bytes());
        launches.push(vec![mem.bufs[k].base]);
        let mut m = Machine::new(program.clone(), Facts::of(t), 8, 0, 0)?;
        m.rank = r;
        machines.push(m);
    }
    let options = Options {
        skew: vec![0, 5],
        ..Options::default()
    };
    let mut c = Cluster::new(machines, launches, (1, 1), 1, options);
    c.run(&mut mem, &mut thin_emu::Tables::default())
}

#[test]
fn only_a_system_scope_write_back_reaches_the_l2() {
    two_ranks("global_wb scope:SCOPE_SYS").expect("written back");
    for wb in ["global_wb scope:SCOPE_DEV", "global_wb", "s_nop 0"] {
        let e = two_ranks(wb).expect_err(wb);
        assert_eq!(e.cause, Some(Cause::Scope), "{wb}: {}", e.message);
    }
}

/// Lane l's dword d of an operand: a hash of (l, d, salt).
fn word(l: usize, d: usize, salt: u32) -> u32 {
    let mut h = (l as u32).wrapping_mul(0x9E37_79B9) ^ (d as u32).wrapping_mul(0x85EB_CA6B) ^ salt;
    h ^= h >> 15;
    h = h.wrapping_mul(0x2C1B_3C6D);
    h ^ h >> 12
}

/// Runs one matrix op with A (v[10:..]) and B (v[20:..]) made by `a` and
/// `b` (through x: A's dwords, then B's at 2048, lane-major, 8 dwords a
/// lane); returns D's dwords, lane-major.
fn wmma_out(
    op: &str,
    ad: usize,
    bd: usize,
    dd: usize,
    a: &dyn Fn(usize, usize) -> u32,
    b: &dyn Fn(usize, usize) -> u32,
) -> Vec<u32> {
    let t = thin_targets::get(thin_targets::default_name()).expect("registered default target");
    let text = format!(
        "s_load_b64 s[4:5], s[0:1], 0x0\ns_wait_kmcnt 0\nv_lshlrev_b32 v9, 5, v0\n\
         global_load_b128 v[10:13], v9, s[4:5]\nglobal_load_b128 v[14:17], v9, s[4:5] offset:16\n\
         global_load_b128 v[20:23], v9, s[4:5] offset:2048\n\
         global_load_b128 v[24:27], v9, s[4:5] offset:2064\ns_wait_loadcnt 0\n\
         {op}\n\
         global_store_b128 v9, v[30:33], s[4:5] offset:4096\n\
         global_store_b128 v9, v[34:37], s[4:5] offset:4112\ns_wait_storecnt 0\ns_endpgm\n",
        op = op
            .replace("A", &format!("v[10:{}]", 9 + ad))
            .replace("B", &format!("v[20:{}]", 19 + bd))
            .replace("D", &format!("v[30:{}]", 29 + dd))
    );
    let program = thin_emu::read_program(&text, t).unwrap();
    let mut mem = Memory::default();
    let x = mem.alloc("x", 8192, true, 0, Meta::default()).unwrap();
    for l in 0..32 {
        for d in 0..8 {
            let at = 32 * l + 4 * d;
            mem.bufs[x].data[at..at + 4].copy_from_slice(&a(l, d).to_le_bytes());
            let bt = 2048 + at;
            mem.bufs[x].data[bt..bt + 4].copy_from_slice(&b(l, d).to_le_bytes());
        }
    }
    let kb = mem.alloc("kernarg", 8, false, 0, Meta::default()).unwrap();
    let base = mem.bufs[x].base;
    mem.bufs[kb].data[..8].copy_from_slice(&base.to_le_bytes());
    let machine = Machine::new(program, Facts::of(t), 64, 0, 0).unwrap();
    let mut c = Cluster::new(
        vec![machine],
        vec![vec![mem.bufs[kb].base]],
        (1, 1),
        1,
        Options::default(),
    );
    c.run(&mut mem, &mut thin_emu::Tables::default()).unwrap();
    (0..32 * 8)
        .map(|i| {
            let at = 4096 + 4 * i;
            u32::from_le_bytes(mem.bufs[x].data[at..at + 4].try_into().unwrap())
        })
        .collect()
}

/// Element e (of `bits`) of lane l's dwords of an operand made by `f`.
fn elem(f: &dyn Fn(usize, usize) -> u32, l: usize, e: usize, bits: usize) -> u32 {
    (f(l, e * bits / 32) >> (e * bits % 32)) & (u32::MAX >> (32 - bits))
}

#[test]
fn wmma_iu8_is_exact_and_signed_by_neg_lo() {
    let a = |l: usize, d: usize| word(l, d, 5);
    let b = |l: usize, d: usize| word(l, d, 77);
    for (mods, signed) in [("", false), (" neg_lo:[1,1,0]", true)] {
        let out = wmma_out(
            &format!("v_wmma_i32_16x16x16_iu8 D, A, B, 0{mods}"),
            2,
            2,
            8,
            &a,
            &b,
        );
        let v = |x: u32| {
            if signed {
                i64::from(x as u8 as i8)
            } else {
                i64::from(x)
            }
        };
        for l in 0..32 {
            let (j, h) = (l % 16, l / 16);
            for e in 0..8 {
                let i = 8 * h + e;
                let sum: i64 = (0..16)
                    .map(|k| {
                        let (kh, ke) = (k / 8, k % 8);
                        v(elem(&a, i + 16 * kh, ke, 8)) * v(elem(&b, j + 16 * kh, ke, 8))
                    })
                    .sum();
                assert_eq!(out[8 * l + e], sum as u32, "{mods} D[{i}][{j}]");
            }
        }
    }
}

/// The f16 value of a pattern (finite inputs only here).
fn f16v(h: u32) -> f64 {
    let (s, ex, m) = ((h >> 15) & 1, (h >> 10) & 31, f64::from(h & 1023));
    let v = if ex == 0 {
        m * 2f64.powi(-24)
    } else {
        (1.0 + m / 1024.0) * 2f64.powi(ex as i32 - 15)
    };
    if s == 1 {
        -v
    } else {
        v
    }
}

#[test]
fn wmma_f32_places_every_element_by_the_dot_unit() {
    // Finite f16 inputs (exponent field below 31), sums that round in f32:
    // each D element of the whole WMMA is the dot unit's four steps over its
    // row and column (k in 0..4, 8..12, 4..8, 12..16; measured on gfx1201,
    // tests/hw/test_wmma.py test_random_f16_and_bf16_sums_are_the_gpus).
    use thin_emu::machine::dot::step;
    let a = |l: usize, d: usize| word(l, d, 5) & 0xBBFF_BBFF;
    let b = |l: usize, d: usize| word(l, d, 77) & 0xBBFF_BBFF;
    let out = wmma_out("v_wmma_f32_16x16x16_f16 D, A, B, 0", 4, 4, 8, &a, &b);
    let mut ordered_differs = 0;
    for l in 0..32 {
        let (j, h) = (l % 16, l / 16);
        for e in 0..8 {
            let i = 8 * h + e;
            let (mut ab, mut bb) = (vec![], vec![]);
            for k in 0..16 {
                let (kh, ke) = (k / 8, k % 8);
                ab.push(elem(&a, i + 16 * kh, ke, 16));
                bb.push(elem(&b, j + 16 * kh, ke, 16));
            }
            let mut want = 0;
            for s0 in [0, 8, 4, 12] {
                want = step(&ab[s0..s0 + 4], &bb[s0..s0 + 4], want, false);
            }
            let got = out[8 * l + e];
            assert_eq!(got, want, "D[{i}][{j}]: {got:#x} want {want:#x}");
            let ordered: f64 = ab.iter().zip(&bb).map(|(x, y)| f16v(*x) * f16v(*y)).sum();
            ordered_differs += usize::from((ordered as f32).to_bits() != got);
        }
    }
    // The inputs' sums round: the dot unit is not the ordered sum here.
    assert!(ordered_differs > 0);
}
