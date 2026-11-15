//! Execute thin-compiled gfx1201 kernels with the CPU machine.

use std::path::PathBuf;

use serde_json::Value;
use thin_emu::machine::cluster::{Cluster, Options};
use thin_emu::machine::facts::Facts;
use thin_emu::machine::memory::Memory;
use thin_emu::machine::run::Machine;
use thin_emu::EmuError;

fn kernel(name: &str) -> Value {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../crates/thin-codegen/tests/data/codegen/kernels.json");
    let all: Value =
        serde_json::from_str(&std::fs::read_to_string(p).expect("kernels.json")).expect("json");
    all["kernels"]
        .as_array()
        .unwrap()
        .iter()
        .find(|k| k["name"] == name)
        .cloned()
        .expect("a kernel")
}

fn lds_of(s: &str) -> usize {
    s.lines()
        .find_map(|l| l.trim().strip_prefix(".amdhsa_group_segment_fixed_size "))
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(0)
}

/// Runs a kernel over buffers (name, bytes) with kernarg bytes built by
/// `kernarg` from the buffers' addresses; returns the memory.
fn run(
    name: &str,
    bufs: &[(&str, Vec<u8>)],
    grid: (u32, u32),
    kernarg: &dyn Fn(&[u64]) -> Vec<u8>,
) -> Result<(Memory, Vec<usize>), EmuError> {
    let k = kernel(name);
    let s = k["s"].as_str().unwrap();
    let t = thin_targets::get(thin_targets::default_name()).expect("registered default target");
    let program = thin_emu::read_program(s, t)?;
    let mut mem = Memory::default();
    let mut ids = Vec::new();
    for (n, data) in bufs {
        let b = mem.alloc(n, data.len() as u64, true, 0, Default::default())?;
        mem.bufs[b].data[..data.len()].copy_from_slice(data);
        ids.push(b);
    }
    let bases: Vec<u64> = ids.iter().map(|b| mem.bufs[*b].base).collect();
    let ka = kernarg(&bases);
    let kb = mem.alloc("kernarg", ka.len() as u64, false, 0, Default::default())?;
    mem.bufs[kb].data[..ka.len()].copy_from_slice(&ka);
    let machine = Machine::new(
        program,
        Facts::of(
            thin_targets::get(thin_targets::default_name()).expect("registered default target"),
        ),
        256,
        0,
        lds_of(s),
    )?;
    let waves = k["wgsize"].as_u64().unwrap() as usize / 32;
    let mut c = Cluster::new(
        vec![machine],
        vec![vec![mem.bufs[kb].base]],
        grid,
        waves,
        Options::default(),
    );
    let mut tables = thin_emu::Tables::default();
    c.run(&mut mem, &mut tables)?;
    Ok((mem, ids))
}

fn words(v: &[u32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn read(mem: &Memory, b: usize, n: usize) -> Vec<u32> {
    (0..n)
        .map(|i| u32::from_le_bytes(mem.bufs[b].data[4 * i..4 * i + 4].try_into().unwrap()))
        .collect()
}

#[test]
fn scaled_merge_combines() {
    let n = 1000usize;
    let x: Vec<u32> = (0..n).map(|i| (i as f32).to_bits()).collect();
    let y: Vec<u32> = (0..n).map(|i| (2.0 * i as f32).to_bits()).collect();
    let bufs = [("x", words(&x)), ("y", words(&y)), ("z", vec![0; 4 * n])];
    let (mem, ids) = run("scaled_merge_b128", &bufs, (8, 1), &|b| {
        let mut k = Vec::new();
        for a in &b[..3] {
            k.extend_from_slice(&a.to_le_bytes());
        }
        k.extend_from_slice(&(n as u32).to_le_bytes());
        k
    })
    .unwrap();
    let z = read(&mem, ids[2], n);
    assert!(z
        .iter()
        .enumerate()
        .all(|(i, v)| f32::from_bits(*v) == 3.25 * i as f32));
}

#[test]
fn scalar_reuse_stores_its_mix() {
    let n = 100u32;
    let (mem, ids) = run("scalar_reuse_b64", &[("x", vec![0; 512])], (2, 1), &|b| {
        let mut k = b[0].to_le_bytes().to_vec();
        k.extend_from_slice(&n.to_le_bytes());
        k
    })
    .unwrap();
    let x = read(&mem, ids[0], 128);
    for (i, v) in x.iter().enumerate() {
        let want = if (i as u32) < n {
            ((i as u32 + 18 * n + 9) ^ (17 * n + 1)) + 3 * n + 11
        } else {
            0
        };
        assert_eq!(*v, want, "x[{i}]");
    }
}

#[test]
fn filter_loops() {
    let m = 5u32;
    let x: Vec<u32> = (0..128).map(|i| (i as f32 * 0.5).to_bits()).collect();
    let bufs = [("x", words(&x)), ("y", vec![0; 512])];
    let (mem, ids) = run("loop_filter_b64", &bufs, (2, 1), &|b| {
        let mut k = b[0].to_le_bytes().to_vec();
        k.extend_from_slice(&b[1].to_le_bytes());
        k.extend_from_slice(&m.to_le_bytes());
        k
    })
    .unwrap();
    let y = read(&mem, ids[1], 128);
    for (i, v) in y.iter().enumerate() {
        let a = i as f32 * 0.5;
        let mut acc = 1.0f32;
        for _ in 0..m {
            acc = (acc - a) * 0.5 + a;
        }
        assert_eq!(f32::from_bits(*v), acc, "y[{i}]");
    }
}

#[test]
fn lds_exchange_rotates_and_combines() {
    let (mem, ids) = run("lds_exchange_b96", &[("x", vec![0; 1152])], (3, 1), &|b| {
        b[0].to_le_bytes().to_vec()
    })
    .unwrap();
    let x = read(&mem, ids[0], 288);
    for (i, v) in x.iter().enumerate() {
        let lane = i as u32 % 96;
        let peer = (lane + 48) % 96;
        assert_eq!(*v, (peer ^ 0x55) + 17 + lane, "x[{i}]");
    }
}

#[test]
fn grid2_places_sparse_rows() {
    let w = 2u32;
    let (mem, ids) = run(
        "plane_store_b32",
        &[("x", vec![0; 4 * 2176])],
        (2, 3),
        &|b| {
            let mut k = b[0].to_le_bytes().to_vec();
            k.extend_from_slice(&w.to_le_bytes());
            k
        },
    )
    .unwrap();
    let x = read(&mem, ids[0], 2176);
    let mut expected = vec![0; 2176];
    for y in 0..3 {
        for column in 0..2 {
            for lane in 0..32 {
                let index = (y * 32 + column * 2 + 1) * 32 + lane;
                expected[index] = index as u32 ^ w;
            }
        }
    }
    assert_eq!(x, expected);
}

/// v_cmpx stops global lane 5. The shuffle selects a nontrivial permutation
/// within each half-row; lane 13 selects inactive lane 5. Its unwritten result
/// starts as POISON, then the integer add incorporates its destination lane.
#[test]
fn permute_selects_lanes() {
    let n = 64u32;
    let (mem, ids) = run(
        "guarded_permute_b64",
        &[("x", vec![0; 256])],
        (1, 1),
        &|b| {
            let mut k = b[0].to_le_bytes().to_vec();
            k.extend_from_slice(&n.to_le_bytes());
            k
        },
    )
    .unwrap();
    let x = read(&mem, ids[0], 64);
    for (t, v) in x.iter().enumerate() {
        let selector = (0x60517243_u32 >> (4 * (t % 8))) & 15;
        let src = (t - t % 16) as u32 + selector;
        let want = if t == 5 {
            0
        } else if src == 5 {
            thin_emu::machine::POISON.wrapping_add(t as u32)
        } else {
            (src ^ 5) + t as u32
        };
        assert_eq!(*v, want, "x[{t}]");
    }
}

/// f16 halves of a dword.
fn f16_pair(lo: u16, hi: u16) -> u32 {
    u32::from(lo) | u32::from(hi) << 16
}

/// A small exact f16: i / 4 for |i| < 64.
fn f16_of(i: i32) -> u16 {
    if i == 0 {
        return 0;
    }
    let x = f64::from(i) / 4.0;
    let (sign, a) = (u16::from(x < 0.0) << 15, x.abs());
    let e = a.log2().floor() as i32;
    let man = ((a / 2f64.powi(e) - 1.0) * 1024.0).round() as u16;
    sign | ((e + 15) as u16) << 10 | man
}

/// The two WMMAs on small exact f16
/// inputs: D = A·B, then E = D[0:4]·B − D (neg_lo[2] negates C). Every sum
/// is exact, so the model's bits are the exact values.
#[test]
fn wmma_multiplies() {
    let av = |r: usize, k: usize| ((r * 3 + k * 5) % 9) as i32 - 4;
    let bv = |k: usize, c: usize| ((k * 7 + c * 2) % 7) as i32 - 3;
    // Lane l = r + 16h, dword d holds elements 2d, 2d + 1 of k = 8h + e.
    let pack = |f: &dyn Fn(usize, usize) -> i32| -> Vec<u8> {
        let mut out = Vec::new();
        for l in 0..32 {
            let (r, h) = (l % 16, l / 16);
            for d in 0..4 {
                let (k0, k1) = (8 * h + 2 * d, 8 * h + 2 * d + 1);
                out.extend_from_slice(&f16_pair(f16_of(f(r, k0)), f16_of(f(r, k1))).to_le_bytes());
            }
        }
        out
    };
    let a = pack(&|r, k| av(r, k));
    let mut b = vec![0; 16];
    b.extend(pack(&|c, k| bv(k, c)));
    let bufs = [("a", a), ("b", b), ("d", vec![0; 2048])];
    let (mem, ids) = run("matrix_recurrence_b32", &bufs, (1, 1), &|b| {
        b.iter().flat_map(|a| a.to_le_bytes()).collect()
    })
    .unwrap();
    let out = read(&mem, ids[2], 512);
    let d = |i: usize, j: usize| -> f64 {
        (0..16)
            .map(|k| f64::from(av(i, k)) * f64::from(bv(k, j)))
            .sum::<f64>()
            / 16.0
    };
    // D's lanes as the next A: lane l = r + 16h, its dwords 0..4 of D are
    // rows 8h.. 8h + 3 of column r; as f32 bits they are not f16 pairs,
    // so the second WMMA reads A's f16 halves from them.
    for l in 0..32 {
        let (j, h) = (l % 16, l / 16);
        for e in 0..8 {
            let position = if e < 4 { 12 + e } else { e };
            let got = f32::from_bits(out[16 * l + position]);
            assert_eq!(f64::from(got), d(8 * h + e, j), "D lane {l} element {e}");
        }
    }
    // E = A'·B − D with A' the f16 halves of D's dwords 0..3.
    let half = |u: u32, hi: bool| -> f64 {
        let h = if hi { u >> 16 } else { u & 0xFFFF };
        let (s, ex, m) = ((h >> 15) & 1, (h >> 10) & 31, f64::from(h & 1023));
        let v = match ex {
            0 => m * 2f64.powi(-24),
            31 => f64::NAN,
            _ => (1.0 + m / 1024.0) * 2f64.powi(ex as i32 - 15),
        };
        if s == 1 {
            -v
        } else {
            v
        }
    };
    let a2 = |r: usize, k: usize| -> f64 {
        let (h, e) = (k / 8, k % 8);
        half(out[16 * (r + 16 * h) + 12 + e / 2], e % 2 == 1)
    };
    for l in 0..32 {
        let (j, h) = (l % 16, l / 16);
        for e in 0..8 {
            let i = 8 * h + e;
            let mut acc = 0.0f64;
            for k in 0..16 {
                acc += a2(i, k) * f64::from(bv(k, j)) / 4.0;
            }
            let want = (acc - d(i, j)) as f32;
            let position = if e < 4 { 8 + e } else { e - 4 };
            let got = f32::from_bits(out[16 * l + position]);
            assert!(
                got.to_bits() == want.to_bits() || (got.is_nan() && want.is_nan()),
                "E lane {l} element {e}: {got} want {want}"
            );
        }
    }
}

/// Throughput: scaled merge over 2^18 elements (run with `--ignored --nocapture`
/// in release; docs/dev/perf.md records the number).
#[test]
#[ignore]
fn throughput() {
    let n = 1usize
        << std::env::var("THIN_EMU_LOG2N")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(18);
    let x: Vec<u32> = (0..n).map(|i| (i as f32).to_bits()).collect();
    let bufs = [("x", words(&x)), ("y", words(&x)), ("z", vec![0; 4 * n])];
    let t = std::time::Instant::now();
    let k = kernel("scaled_merge_b128");
    let s = k["s"].as_str().unwrap();
    let program =
        thin_emu::read_program(s, thin_targets::get(thin_targets::default_name()).unwrap())
            .unwrap();
    let mut mem = Memory::default();
    let mut ids = Vec::new();
    for (name, data) in &bufs {
        let b = mem
            .alloc(name, data.len() as u64, true, 0, Default::default())
            .unwrap();
        mem.bufs[b].data.copy_from_slice(data);
        ids.push(b);
    }
    let mut ka = Vec::new();
    for b in &ids {
        ka.extend_from_slice(&mem.bufs[*b].base.to_le_bytes());
    }
    ka.extend_from_slice(&(n as u32).to_le_bytes());
    let kb = mem
        .alloc("kernarg", 28, false, 0, Default::default())
        .unwrap();
    mem.bufs[kb].data[..28].copy_from_slice(&ka);
    let machine = Machine::new(
        program,
        Facts::of(
            thin_targets::get(thin_targets::default_name()).expect("registered default target"),
        ),
        3,
        0,
        0,
    )
    .unwrap();
    let mut c = Cluster::new(
        vec![machine],
        vec![vec![mem.bufs[kb].base]],
        ((n / 128) as u32, 1),
        4,
        Options::default(),
    );
    let mut tables = thin_emu::Tables::default();
    c.run(&mut mem, &mut tables).unwrap();
    let dt = t.elapsed().as_secs_f64();
    let ins = c.machines[0].reports.stats.ins;
    println!(
        "{ins} instructions in {dt:.3} s: {:.2} M instructions/s, {:.1} M lane-ops/s",
        ins as f64 / dt / 1e6,
        ins as f64 * 32.0 / dt / 1e6
    );
}
