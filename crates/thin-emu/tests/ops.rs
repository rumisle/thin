//! The default target's (gfx1201's) ops of shipped kernels (see
//! docs/dev/lift.md) and its measured number rules, from the RDNA4 ISA guide,
//! on small programs: each stores what it computes, one dword a lane, into
//! buffer `x` (s[4:5]); buffer `y`
//! (s[6:7]) holds the inputs. thin-targets is a dev-dependency only.

use thin_emu::machine::cluster::{Cluster, Options};
use thin_emu::machine::facts::Facts;
use thin_emu::machine::memory::{Memory, Meta};
use thin_emu::machine::run::Machine;
use thin_emu::machine::wave::Wave;
use thin_emu::tags::{mem_tag, Tag};
use thin_emu::{Cause, EmuError};

const HEAD: &str = "s_load_b64 s[4:5], s[0:1], 0x0\ns_load_b64 s[6:7], s[0:1], 0x8\n\
                    s_wait_kmcnt 0\nv_lshlrev_b32 v9, 2, v0\n";

/// Runs `body` on one 32-lane wave with x (`words` dwords) at s[4:5], y
/// (holding `input`) at s[6:7], v9 = lane · 4; returns x.
fn run_with(body: &str, input: &[u32], words: usize) -> Result<Vec<u32>, EmuError> {
    let t = thin_targets::get(thin_targets::default_name()).expect("the default target");
    let text = format!("{HEAD}{body}\ns_endpgm\n");
    let program = thin_emu::read_program(&text, t)?;
    let mut mem = Memory::default();
    let x = mem.alloc("x", 4 * words as u64, true, 0, Meta::default())?;
    let y = mem.alloc("y", 4 * input.len().max(1) as u64, true, 0, Meta::default())?;
    for (i, v) in input.iter().enumerate() {
        mem.bufs[y].data[4 * i..4 * i + 4].copy_from_slice(&v.to_le_bytes());
    }
    let kb = mem.alloc("kernarg", 16, false, 0, Meta::default())?;
    let (xb, yb) = (mem.bufs[x].base, mem.bufs[y].base);
    mem.bufs[kb].data[..8].copy_from_slice(&xb.to_le_bytes());
    mem.bufs[kb].data[8..16].copy_from_slice(&yb.to_le_bytes());
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

/// Runs `body` on one wave as [`run_with`] does (x buffer 0, y buffer 1,
/// both tagged); returns the wave at its end.
fn wave(body: &str, input: &[u32]) -> Wave {
    let t = thin_targets::get(thin_targets::default_name()).expect("the default target");
    let text = format!("{HEAD}{body}\ns_endpgm\n");
    let program = thin_emu::read_program(&text, t).unwrap();
    let mut mem = Memory::default();
    let x = mem.alloc("x", 128, true, 0, Meta::default()).unwrap();
    let y = mem.alloc("y", 4 * input.len().max(1) as u64, true, 0, Meta::default());
    let y = y.unwrap();
    for (i, v) in input.iter().enumerate() {
        mem.bufs[y].data[4 * i..4 * i + 4].copy_from_slice(&v.to_le_bytes());
    }
    let kb = mem.alloc("kernarg", 16, false, 0, Meta::default()).unwrap();
    let (xb, yb) = (mem.bufs[x].base, mem.bufs[y].base);
    mem.bufs[kb].data[..8].copy_from_slice(&xb.to_le_bytes());
    mem.bufs[kb].data[8..16].copy_from_slice(&yb.to_le_bytes());
    let kernarg = mem.bufs[kb].base;
    let mut m = Machine::new(program, Facts::of(t), 64, 0, 256).unwrap();
    let mut tables = thin_emu::Tables::default();
    let (mut ws, _) = m
        .run_wg(&mut mem, &mut tables, kernarg, (0, 0), 1, 0, None)
        .unwrap();
    ws.remove(0)
}

/// Register v`r`'s lane `l`.
fn vreg(w: &Wave, r: usize, l: usize) -> u32 {
    w.v[r * w.w + l]
}

/// Register v`r`'s lane `l`'s byte tags.
fn vtag(w: &Wave, r: usize, l: usize) -> [Tag; 4] {
    w.vt[r].as_ref().map_or([0; 4], |t| t[l])
}

/// y's (buffer 1's) tag of byte `o`.
fn ytag(o: u64) -> Tag {
    mem_tag(1, o)
}

fn run(body: &str) -> Vec<u32> {
    run_with(body, &[], 32).unwrap()
}

/// The error of a body that fails.
fn refused(body: &str) -> EmuError {
    run_with(body, &[], 32).expect_err(body)
}

// ------------------------------------------------------------------ exec

#[test]
fn a_64_bit_exec_write_keeps_exec_hi_beside_the_wave32_mask() {
    // s_mov_b64 exec: lanes 0..15 active, EXEC_HI 0x12345678; read back
    // as a pair and as exec_hi; the store runs in lanes 0..15 only.
    let x = run("s_mov_b32 s10, 0xffff\ns_mov_b32 s11, 0x12345678\n\
         s_mov_b64 exec, s[10:11]\ns_mov_b64 s[12:13], exec\ns_mov_b32 s14, exec_hi\n\
         v_mov_b32 v1, s12\nv_mov_b32 v2, s13\nv_mov_b32 v3, s14\n\
         v_cmp_eq_u32 vcc_lo, 1, v0\nv_cndmask_b32 v1, v1, v2, vcc_lo\n\
         v_cmp_eq_u32 vcc_lo, 2, v0\nv_cndmask_b32 v1, v1, v3, vcc_lo\n\
         global_store_b32 v9, v1, s[4:5]");
    let want: Vec<u32> = (0..32)
        .map(|l| match l {
            1 | 2 => 0x1234_5678,
            l if l < 16 => 0xffff,
            _ => 0,
        })
        .collect();
    assert_eq!(x, want);
}

#[test]
fn exec_hi_starts_0_and_a_32_bit_write_of_it_leaves_the_mask() {
    let x = run(
        "s_mov_b64 s[12:13], exec\nv_mov_b32 v1, s13\nv_mov_b32 v2, s12\n\
         s_mov_b32 exec_hi, 0\nv_add_nc_u32 v1, v1, v2\nglobal_store_b32 v9, v1, s[4:5]",
    );
    assert_eq!(x, vec![0xffff_ffff; 32]);
    // The lanes are EXEC_LO's alone: a nonzero EXEC_HI enables none.
    let x = run(
        "v_mov_b32 v1, 5\ns_mov_b32 exec_hi, -1\ns_mov_b32 exec_lo, 1\n\
         global_store_b32 v9, v1, s[4:5]",
    );
    assert_eq!(x[..2], [5, 0]);
}

#[test]
fn exec_hi_as_a_lane_mask_is_refused() {
    let e = refused("v_cndmask_b32 v1, 1, 2, exec_hi");
    assert_eq!(e.cause, Some(Cause::Fault));
    assert_eq!(
        e.message,
        "v_cndmask_b32 v1, 1, 2, exec_hi: exec_hi is not a lane mask on wave32"
    );
}

#[test]
fn a_wave32_mask_into_one_sgpr_leaves_the_next() {
    // A compare and a carry-out into s8: s9 keeps its value.
    for op in ["v_cmp_eq_u32_e64 s8, v0, v0", "v_add_co_u32 v1, s8, v0, v0"] {
        let x = run(&format!(
            "s_mov_b32 s9, 0x1234\n{op}\nv_mov_b32 v1, s9\nv_mov_b32 v2, s8\n\
             v_cmp_eq_u32 vcc_lo, 0, v0\nv_cndmask_b32 v1, v1, v2, vcc_lo\n\
             global_store_b32 v9, v1, s[4:5]"
        ));
        let mask = if op.starts_with("v_cmp") { u32::MAX } else { 0 };
        assert_eq!((x[0], x[1]), (mask, 0x1234), "{op}");
    }
}

// -------------------------------------------------------------- saveexec

/// After `setup` and `op` (writing s8 from EXEC): lane 0 the saved s8,
/// lane 1 the new EXEC_LO, lane 2 SCC, lane 3 EXEC_HI.
fn saveexec(setup: &str, op: &str) -> [u32; 4] {
    let x = run(&format!(
        "{setup}\n{op}\ns_cselect_b32 s12, 1, 0\ns_mov_b32 s13, exec_lo\ns_mov_b32 s14, exec_hi\n\
         s_mov_b32 exec_lo, -1\nv_mov_b32 v1, s8\n\
         v_cmp_eq_u32 vcc_lo, 1, v0\nv_cndmask_b32 v1, v1, s13, vcc_lo\n\
         v_cmp_eq_u32 vcc_lo, 2, v0\nv_cndmask_b32 v1, v1, s12, vcc_lo\n\
         v_cmp_eq_u32 vcc_lo, 3, v0\nv_cndmask_b32 v1, v1, s14, vcc_lo\n\
         global_store_b32 v9, v1, s[4:5]"
    ));
    [x[0], x[1], x[2], x[3]]
}

#[test]
fn or_saveexec_b32_ors_exec_lo_and_keeps_exec_hi() {
    let setup = "s_mov_b32 exec_hi, 7\ns_mov_b32 exec_lo, 0xf0\ns_mov_b32 s10, 0x0f00";
    assert_eq!(
        saveexec(setup, "s_or_saveexec_b32 s8, s10"),
        [0xf0, 0xff0, 1, 7]
    );
    // EXEC 0 or 0: SCC 0.
    let zero = "s_mov_b32 exec_lo, 0\ns_mov_b32 s10, 0";
    assert_eq!(saveexec(zero, "s_or_saveexec_b32 s8, s10"), [0, 0, 0, 0]);
    // The and form, beside it: the same rule with and.
    assert_eq!(
        saveexec(setup, "s_and_saveexec_b32 s8, s10"),
        [0xf0, 0, 0, 7]
    );
}

#[test]
fn the_b64_saveexec_forms_take_exec_hi_on_wave32() {
    let setup = "s_mov_b32 exec_hi, 7\ns_mov_b32 exec_lo, 0xf0\n\
                 s_mov_b32 s10, 0x0f00\ns_mov_b32 s11, 0x30";
    // s[8:9] saves EXEC whole; s8 is its low half. EXEC_HI = 7 | 0x30.
    assert_eq!(
        saveexec(setup, "s_or_saveexec_b64 s[8:9], s[10:11]"),
        [0xf0, 0xff0, 1, 0x37]
    );
    // EXEC_LO = 0, EXEC_HI = 7 & 0x30 = 0: SCC 0; with s11 = 3, EXEC_HI 3
    // and SCC 1, though no lane is active (EXEC.u64 != 0).
    assert_eq!(
        saveexec(setup, "s_and_saveexec_b64 s[8:9], s[10:11]"),
        [0xf0, 0, 0, 0]
    );
    let setup3 = setup.replace("0x30", "3");
    assert_eq!(
        saveexec(&setup3, "s_and_saveexec_b64 s[8:9], s[10:11]"),
        [0xf0, 0, 1, 3]
    );
}

// ------------------------------------------------------------- d16 loads

/// y's dword `i`: distinct bytes, sign bits set in some.
fn yword(i: usize) -> u32 {
    let b = |k: usize| ((i * 4 + k) as u32 * 37 + 0x5B) & 0xFF;
    b(0) | b(1) << 8 | b(2) << 16 | b(3) << 24
}

/// v1 = 0xaabbccdd, then `load v1, v9, s[16:19], null offen` from y
/// (num_records `num` bytes) under EXEC_LO `exec`; the wave at the end.
fn d16(load: &str, num: u32, exec: u32) -> Wave {
    let y: Vec<u32> = (0..64).map(yword).collect();
    wave(
        &format!(
            "s_mov_b32 s16, s6\ns_and_b32 s17, s7, 0xffff\ns_mov_b32 s18, {num}\n\
             s_mov_b32 s19, 0x30020000\nv_mov_b32 v1, 0xaabbccdd\ns_mov_b32 exec_lo, {exec:#x}\n\
             {load} v1, v9, s[16:19], null offen\ns_wait_loadcnt 0\ns_mov_b32 exec_lo, -1"
        ),
        &y,
    )
}

#[test]
fn a_d16_load_writes_its_half_and_keeps_the_other() {
    let ext = |v: u32, signed: bool| {
        if signed {
            v as u8 as i8 as i16 as u16 as u32
        } else {
            v
        }
    };
    for (op, hi, bytes, signed) in [
        ("buffer_load_d16_b16", false, 2, false),
        ("buffer_load_d16_hi_b16", true, 2, false),
        ("buffer_load_d16_u8", false, 1, false),
        ("buffer_load_d16_hi_u8", true, 1, false),
        ("buffer_load_d16_i8", false, 1, true),
        ("buffer_load_d16_hi_i8", true, 1, true),
    ] {
        let w = d16(op, 256, u32::MAX);
        for l in 0..32 {
            let m = if bytes == 2 { 0xFFFF } else { 0xFF };
            let v = ext(yword(l) & m, signed);
            let want = if hi {
                0xccdd | v << 16
            } else {
                0xaabb_0000 | v
            };
            assert_eq!(vreg(&w, 1, l), want, "{op} lane {l}");
        }
    }
}

#[test]
fn a_d16_load_leaves_inactive_lanes_and_zeroes_its_half_out_of_range() {
    // Lanes 0, 2, 4,... active; num_records 64: lanes 16 on out of range.
    for (op, hi) in [
        ("buffer_load_d16_b16", false),
        ("buffer_load_d16_hi_u8", true),
    ] {
        let w = d16(op, 64, 0x5555_5555);
        for l in 0..32 {
            let v = if l >= 16 {
                0
            } else if hi {
                yword(l) & 0xFF
            } else {
                yword(l) & 0xFFFF
            };
            let want = match (l % 2, hi) {
                (1, _) => 0xaabb_ccdd,
                (_, true) => 0xccdd | v << 16,
                _ => 0xaabb_0000 | v,
            };
            assert_eq!(vreg(&w, 1, l), want, "{op} lane {l}");
        }
    }
}

#[test]
fn a_d16_load_tags_its_half_and_keeps_the_other_halfs_tags() {
    // v1 loaded whole from y[l + 32], then a d16 load from y[l] over it.
    let y: Vec<u32> = (0..64).map(yword).collect();
    for (op, hi, two) in [
        ("buffer_load_d16_hi_b16", true, true),
        ("buffer_load_d16_u8", false, false),
    ] {
        let w = wave(
            &format!(
                "s_mov_b32 s16, s6\ns_and_b32 s17, s7, 0xffff\ns_mov_b32 s18, 256\n\
                 s_mov_b32 s19, 0x30020000\n\
                 buffer_load_b32 v1, v9, s[16:19], null offen offset:128\ns_wait_loadcnt 0\n\
                 {op} v1, v9, s[16:19], null offen\ns_wait_loadcnt 0"
            ),
            &y,
        );
        for l in 0..32u64 {
            let old = |k: u64| ytag(128 + 4 * l + k);
            let new = |k: u64| if k == 0 || two { ytag(4 * l + k) } else { 0 };
            let want = if hi {
                [old(0), old(1), new(0), new(1)]
            } else {
                [new(0), new(1), old(2), old(3)]
            };
            assert_eq!(vtag(&w, 1, l as usize), want, "{op} lane {l}");
        }
    }
}

#[test]
fn a_global_d16_load_merges_as_a_buffer_one_does() {
    let y: Vec<u32> = (0..64).map(yword).collect();
    let w = wave(
        "v_mov_b32 v1, 0xaabbccdd\nglobal_load_d16_hi_b16 v1, v9, s[6:7]\ns_wait_loadcnt 0",
        &y,
    );
    for l in 0..32 {
        assert_eq!(
            vreg(&w, 1, l),
            0xccdd | (yword(l) & 0xFFFF) << 16,
            "lane {l}"
        );
    }
}

// ---------------------------------------------------------- d16_hi stores

#[test]
fn a_d16_hi_ds_store_stores_the_high_half() {
    // Lane l stores the high byte or short of y[l] at LDS 4 l, over
    // 0x77777777; the dword is read back. EXEC_LO 0x5555: odd lanes keep it.
    let y: Vec<u32> = (0..32).map(yword).collect();
    for (op, mask) in [
        ("ds_store_b16_d16_hi", 0xFFFF),
        ("ds_store_b8_d16_hi", 0xFF),
    ] {
        let w = wave(
            &format!(
                "global_load_b32 v1, v9, s[6:7]\nv_mov_b32 v2, 0x77777777\n\
                 ds_store_b32 v9, v2\ns_wait_dscnt 0\ns_wait_loadcnt 0\n\
                 s_mov_b32 exec_lo, 0x55555555\n{op} v9, v1\ns_mov_b32 exec_lo, -1\n\
                 s_wait_dscnt 0\nds_load_b32 v3, v9\ns_wait_dscnt 0"
            ),
            &y,
        );
        for l in 0..32 {
            let want = if l % 2 == 1 {
                0x7777_7777
            } else {
                0x7777_7777 & !mask | (yword(l) >> 16) & mask
            };
            assert_eq!(vreg(&w, 3, l), want, "{op} lane {l}");
            if l % 2 == 0 {
                // The stored bytes keep y's tags of bytes 2 (and 3).
                let t = vtag(&w, 3, l);
                let l = l as u64;
                assert_eq!(t[0], ytag(4 * l + 2), "{op} lane {l}");
                assert_eq!(t[1], if mask == 0xFFFF { ytag(4 * l + 3) } else { 0 });
            }
        }
    }
}

// ------------------------------------------------------------ cmp_class

/// The ten classes' representatives, in mask-bit order (RDNA4 ISA guide
/// V_CMP_CLASS_F32), two of each.
const CLASSES: [[u32; 2]; 10] = [
    [0x7F80_0001, 0xFFBF_FFFF], // signaling NaN
    [0x7FC0_0000, 0xFFC0_0001], // quiet NaN
    [0xFF80_0000, 0xFF80_0000], // -inf
    [0xBF80_0000, 0xFF7F_FFFF], // -normal
    [0x8000_0001, 0x807F_FFFF], // -denormal
    [0x8000_0000, 0x8000_0000], // -0
    [0x0000_0000, 0x0000_0000], // +0
    [0x0000_0001, 0x007F_FFFF], // +denormal
    [0x0080_0000, 0x3F80_0000], // +normal
    [0x7F80_0000, 0x7F80_0000], // +inf
];

#[test]
fn cmp_class_reports_each_class_by_its_mask_bit() {
    // Lane l: x = class l / 2's representative l % 2 (l < 20); the mask
    // in lane l's y; the result mask into s8 (e64) or vcc_lo (e32).
    let mut y = vec![0u32; 64];
    for l in 0..20 {
        y[l] = CLASSES[l / 2][l % 2];
    }
    for mask in [0x3FFu32, 0x001, 0x002, 0x204, 0x0A8, 0x150, 0x060, 0x000] {
        for (op, d) in [
            ("v_cmp_class_f32_e64 s8", "s8"),
            ("v_cmp_class_f32 vcc_lo", "vcc_lo"),
        ] {
            let w = wave(
                &format!(
                    "global_load_b32 v1, v9, s[6:7]\ns_wait_loadcnt 0\nv_mov_b32 v2, {mask:#x}\n\
                     s_mov_b32 exec_lo, 0xfffff\n{op}, v1, v2\ns_mov_b32 s10, {d}"
                ),
                &y,
            );
            let want: u32 = (0..20)
                .filter(|l| mask >> (l / 2) & 1 != 0)
                .map(|l| 1 << l)
                .sum();
            assert_eq!(w.s[10], want, "{op} mask {mask:#x}");
        }
    }
}

#[test]
fn cmpx_class_writes_exec() {
    let w = wave(
        "v_cvt_f32_i32 v1, v0\nv_sub_f32 v1, v1, 3.0\nv_cmpx_class_f32 v1, 0x60\n\
         s_mov_b32 s10, exec_lo\ns_mov_b32 exec_lo, -1",
        &[],
    );
    assert_eq!(w.s[10], 1 << 3, "only lane 3's 0.0 is a zero");
}

// ------------------------------------------------------------ integer ops

/// y = the lanes' (x, y, z) triples (dwords 3 l, 3 l + 1, 3 l + 2); runs
/// `op v4, v1, v2, v3` (or two sources) on them; the wave at the end.
fn three_ops(op: &str, triples: &[[u32; 3]]) -> Wave {
    let mut y = vec![0u32; 96];
    for (l, t) in triples.iter().enumerate() {
        y[3 * l..3 * l + 3].copy_from_slice(t);
    }
    let srcs = if op.contains("_mul_") {
        "v1, v2"
    } else {
        "v1, v2, v3"
    };
    wave(
        &format!(
            "v_mul_u32_u24 v8, 12, v0\nglobal_load_b32 v1, v8, s[6:7]\n\
             global_load_b32 v2, v8, s[6:7] offset:4\nglobal_load_b32 v3, v8, s[6:7] offset:8\n\
             s_wait_loadcnt 0\n{op} v4, {srcs}"
        ),
        &y,
    )
}

#[test]
fn bfe_i32_sign_extends_its_field() {
    let cases: [([u32; 3], u32); 10] = [
        ([0x0000_0F00, 8, 4], 0xFFFF_FFFF),    // field 0xF: -1
        ([0x0000_0700, 8, 4], 7),              // field 0x7: positive
        ([0x8000_0000, 31, 1], 0xFFFF_FFFF),   // the sign bit alone
        ([0x1234_5678, 0, 0], 0),              // width 0
        ([0x1234_5678, 32 + 4, 32 + 8], 0x67), // offset and width mod 32
        ([0xF000_0000, 28, 8], 0xFFFF_FFFF),   // past bit 31: S0's sign shifted in
        ([0x7000_0000, 28, 8], 7),
        ([0xDEAD_BEEF, 0, 31], 0xDEAD_BEEF), // width 31: bit 30 the sign
        ([0x3EAD_BEEF, 0, 31], 0x3EAD_BEEF),
        ([0x0080_0000, 16, 8], 0xFFFF_FF80),
    ];
    let w = three_ops("v_bfe_i32", &cases.map(|c| c.0));
    for (l, (c, want)) in cases.iter().enumerate() {
        assert_eq!(vreg(&w, 4, l), *want, "{c:x?}");
    }
    // Tags: the field's bytes keep their source bytes' tags; with offset 8
    // and width 12 bytes 0-1 come from bytes 1-2, the sign bytes untagged.
    let w = three_ops("v_bfe_i32", &[[0x000F_0000, 8, 12]]);
    assert_eq!(vtag(&w, 4, 0), [ytag(1), ytag(2), 0, 0]);
    assert_eq!(vreg(&w, 4, 0), 0xFFFF_FF00);
}

#[test]
fn mul_i32_i24_sign_extends_bit_23_and_ignores_the_high_byte() {
    let cases: [([u32; 3], u32); 7] = [
        ([3, 5, 0], 15),
        ([0x00FF_FFFF, 5, 0], (-5i32) as u32), // -1 * 5
        ([0xAB00_0002, 0x7F00_0003, 0], 6),    // high bytes ignored
        ([0x0080_0000, 0x0080_0000, 0], 0),    // (-2^23)^2 = 2^46: low 32 bits 0
        (
            [0x007F_FFFF, 0x007F_FFFF, 0],
            0x7F_FFFFu32.wrapping_mul(0x7F_FFFF),
        ),
        ([0x0080_0000, 1, 0], 0xFF80_0000),
        ([0xFFFF_FFFF, 0xFFFF_FFFF, 0], 1),
    ];
    let w = three_ops("v_mul_i32_i24", &cases.map(|c| c.0));
    for (l, (c, want)) in cases.iter().enumerate() {
        assert_eq!(vreg(&w, 4, l), *want, "{c:x?}");
    }
    let w = three_ops("v_mul_i32_i24_e64", &cases.map(|c| c.0));
    assert_eq!(vreg(&w, 4, 1), (-5i32) as u32);
}

#[test]
fn the_u24_multiplies_ignore_the_high_byte_as_the_gpu_does() {
    // Measured on gfx1201: no fault; the low 24 bits' product (and z for
    // mad).
    let cases = [
        [0xFF00_0003u32, 0x0100_0005, 7],
        [0x00FF_FFFF, 0x80FF_FFFF, 1],
    ];
    let lo = |v: u32| u64::from(v & 0xFF_FFFF);
    let w = three_ops("v_mul_u32_u24", &cases);
    let w2 = three_ops("v_mad_u32_u24", &cases);
    for (l, [x, y, z]) in cases.iter().enumerate() {
        let p = (lo(*x) * lo(*y)) as u32;
        assert_eq!(vreg(&w, 4, l), p, "mul lane {l}");
        assert_eq!(vreg(&w2, 4, l), p.wrapping_add(*z), "mad lane {l}");
    }
}

#[test]
fn and_or_ands_then_ors_and_moves_tags_by_byte() {
    let cases: [[u32; 3]; 3] = [
        [0x1234_5678, 0xFF00_FF00, 0x0000_0001],
        [0xFFFF_0000, 0x00FF_FF00, 0x0000_00AB],
        [0, 0, 0xDEAD_BEEF],
    ];
    let w = three_ops("v_and_or_b32", &cases);
    for (l, [a, b, c]) in cases.iter().enumerate() {
        assert_eq!(vreg(&w, 4, l), (a & b) | c, "lane {l}");
    }
    // Lane 0, then: the and keeps x's bytes 1 and 3 (y's are 0xFF)
    // and leaves bytes 0 and 2 untagged (neither byte 0xFF); the or takes
    // z's tags there, and x's where z's byte is 0.
    let o = |k: u64| ytag(k);
    assert_eq!(vtag(&w, 4, 0), [o(8), o(1), o(10), o(3)]);
}

// ---------------------------------------------------------------- 16-bit

/// Lane l: v1 = y[2 l], v2 = y[2 l + 1], v4 = 0xa5a5_0000 | l (the old
/// destination), then `op` (writing v4); the wave at the end.
fn two16(op: &str, pairs: &[[u32; 2]]) -> Wave {
    let mut y = vec![0u32; 64];
    for (l, p) in pairs.iter().enumerate() {
        y[2 * l..2 * l + 2].copy_from_slice(p);
    }
    wave(
        &format!(
            "v_lshlrev_b32 v8, 3, v0\nglobal_load_b32 v1, v8, s[6:7]\n\
             global_load_b32 v2, v8, s[6:7] offset:4\nv_or_b32 v4, 0xa5a50000, v0\n\
             s_wait_loadcnt 0\n{op}"
        ),
        &y,
    )
}

#[test]
fn max_num_f16_follows_maximum_number_and_keeps_the_high_half() {
    // (x, y) low halves; the high halves are junk the op ignores.
    let cases: [(u32, u32, u32); 12] = [
        (0x3C00, 0x4000, 0x4000), // 1 < 2
        (0xC000, 0xBC00, 0xBC00), // -2 < -1
        (0x7E05, 0x3C00, 0x3C00), // quiet NaN: the number
        (0x3C00, 0x7C05, 0x3C00), // signaling NaN: the number
        (0x7E05, 0xFE07, 0x7E05), // both NaN: S0 quieted (the GPU's)
        (0x7C05, 0x7C07, 0x7E05),
        (0x0000, 0x8000, 0x0000), // +0 above -0, either order
        (0x8000, 0x0000, 0x0000),
        (0x0001, 0x0000, 0x0001), // a denormal above 0
        (0x8001, 0x0000, 0x0000),
        (0xFC00, 0x7C00, 0x7C00), // -inf, +inf
        (0xFC00, 0x7E00, 0xFC00),
    ];
    let pairs: Vec<[u32; 2]> = cases
        .iter()
        .map(|c| [c.0 | 0x1234_0000, c.1 | 0x5678_0000])
        .collect();
    for op in ["v_max_num_f16 v4, v1, v2", "v_max_num_f16_e64 v4, v1, v2"] {
        let w = two16(op, &pairs);
        for (l, c) in cases.iter().enumerate() {
            assert_eq!(vreg(&w, 4, l), 0xa5a5_0000 | c.2, "{op} {c:x?}");
        }
    }
    // Source modifiers act on bit 15: max(|-2|, -(1)) = 2; max(-(-0), 0) = +0.
    let w = two16(
        "v_max_num_f16_e64 v4, |v1|, -v2",
        &[[0xC000, 0x3C00], [0x0000, 0x0000]],
    );
    assert_eq!((vreg(&w, 4, 0), vreg(&w, 4, 1)), (0xa5a5_4000, 0xa5a5_0000));
}

#[test]
fn a_16_bit_op_writes_active_lanes_only_and_refuses_op_sel() {
    let w = two16(
        "s_mov_b32 exec_lo, 0x5\nv_max_num_f16 v4, v1, v2\ns_mov_b32 exec_lo, -1",
        &[[0x3C00, 0], [0x3C00, 0], [0x3C00, 0]],
    );
    assert_eq!(
        (0..3).map(|l| vreg(&w, 4, l)).collect::<Vec<_>>(),
        [0xA5A5_3C00, 0xA5A5_0001, 0xA5A5_3C00]
    );
    // The decoded form only: gfx1201's encoder has no op_sel for these.
    use thin_emu::{Base, Field, Ins, Operand};
    let reg = |n| Operand {
        base: Base::Reg("v".into(), n, 1),
        neg: false,
        abs: false,
    };
    let ins = Ins {
        canon: "v.max.num.f16".into(),
        kind: "valu".into(),
        operands: vec![reg(4), reg(1), reg(2)],
        fields: vec![("op_sel".into(), Field::List(vec![1, 0, 0]))],
        matrix: None,
        text: "v_max_num_f16 v4, v1, v2 op_sel:[1,0,0]".into(),
        comment: String::new(),
        line: 0,
    };
    let t = thin_targets::get(thin_targets::default_name()).expect("the default target");
    let program = thin_emu::Program {
        ins: vec![ins],
        user_sgprs: 2,
        ..Default::default()
    };
    let mut m = Machine::new(program, Facts::of(t), 8, 0, 0).unwrap();
    let e = m
        .run_wg(
            &mut Memory::default(),
            &mut thin_emu::Tables::default(),
            0,
            (0, 0),
            1,
            0,
            None,
        )
        .map(|_| ())
        .unwrap_err();
    assert_eq!(e.cause, Some(Cause::Emulator));
    assert!(
        e.message.ends_with(&format!(
            "op_sel is not modelled (no {} form of v.max.num.f16 takes it)",
            thin_targets::default_name()
        )),
        "{}",
        e.message
    );
}

#[test]
fn lshlrev_b16_shifts_the_low_half_by_s0_mod_16() {
    let cases: [(u32, u32); 7] = [
        (0, 0xABCD),
        (1, 0x579A),
        (8, 0xCD00),
        (15, 0x8000),
        (16, 0xABCD),
        (17, 0x579A),
        (0xFFFF_0004, 0xBCD0),
    ];
    let pairs: Vec<[u32; 2]> = cases.iter().map(|c| [c.0, 0x1234_ABCD]).collect();
    let w = two16("v_lshlrev_b16 v4, v1, v2", &pairs);
    for (l, c) in cases.iter().enumerate() {
        assert_eq!(vreg(&w, 4, l), 0xa5a5_0000 | c.1, "{c:x?}");
    }
    // Tags (within the half): by 0 both bytes move, by 8 the low byte
    // to the high one, by 1 none; the high half keeps v4's (untagged).
    let o = |k: u64| ytag(k);
    assert_eq!(vtag(&w, 4, 0), [o(4), o(5), 0, 0]);
    assert_eq!(vtag(&w, 4, 2), [0, o(20), 0, 0]);
    assert_eq!(vtag(&w, 4, 1), [0; 4]);
}

#[test]
fn pack_b32_f16_packs_low_halves_quieting_nans() {
    let cases: [([u32; 2], u32); 6] = [
        ([0x1111_3C00, 0x2222_4000], 0x4000_3C00),
        ([0x7C05, 0x3C00], 0x3C00_7E05), // a signaling NaN: quieted
        ([0xFE07, 0x7E01], 0x7E01_FE07), // quiet NaNs: kept
        ([0x0001, 0x8000], 0x8000_0001), // a denormal and -0 kept
        ([0xFC00, 0x7C00], 0x7C00_FC00),
        ([0x8001, 0x03FF], 0x03FF_8001),
    ];
    let w = two16("v_pack_b32_f16 v4, v1, v2", &cases.map(|c| c.0));
    for (l, c) in cases.iter().enumerate() {
        assert_eq!(vreg(&w, 4, l), c.1, "{c:x?}");
    }
    // Tags move with the halves: x's bytes 0-1, then y's bytes 0-1.
    let o = |k: u64| ytag(k);
    assert_eq!(vtag(&w, 4, 0), [o(0), o(1), o(4), o(5)]);
    let w = two16("v_pack_b32_f16 v4, -v1, |v2|", &[[0x3C00, 0xC000]]);
    assert_eq!(vreg(&w, 4, 0), 0x4000_BC00);
}

// ---------------------------------------------------------------- fma_mix

#[test]
fn fma_mix_selects_f32_or_either_f16_half_per_source() {
    // v1 = 2.0 (f32), v2 = {hi 3.0, lo 0.5} (f16), v3 = {hi -1.0, lo 4.0}.
    let body = |mods: &str| {
        format!(
            "v_mov_b32 v1, 2.0\nv_mov_b32 v2, 0x42003800\nv_mov_b32 v3, 0xbc004400\n\
             v_fma_mix_f32 v4, v1, v2, v3{mods}"
        )
    };
    let f = |x: f32| x.to_bits();
    for (mods, want) in [
        (" op_sel_hi:[0,1,1]", f(2.0 * 0.5 + 4.0)),
        (" op_sel:[0,1,1] op_sel_hi:[0,1,1]", f(2.0 * 3.0 - 1.0)),
        (" op_sel:[0,1,0] op_sel_hi:[0,1,1]", f(2.0 * 3.0 + 4.0)),
        // All f32: v2, v3 as their f32 patterns (op_sel ignored there).
        (
            " op_sel:[1,1,1]",
            f(2.0 * f32::from_bits(0x4200_3800) + f32::from_bits(0xbc00_4400)),
        ),
        (
            "",
            f(2.0 * f32::from_bits(0x4200_3800) + f32::from_bits(0xbc00_4400)),
        ),
    ] {
        let w = wave(&body(mods), &[]);
        assert_eq!(vreg(&w, 4, 0), want, "{mods:?}");
    }
    // abs and neg on the converted values: -|v2.hi| * 2 + |v3.hi| = -6 + 1.
    let w = wave(
        "v_mov_b32 v1, 2.0\nv_mov_b32 v2, 0xc2003800\nv_mov_b32 v3, 0xbc004400\n\
         v_fma_mix_f32 v4, v1, -|v2|, |v3| op_sel:[0,1,1] op_sel_hi:[0,1,1]",
        &[],
    );
    assert_eq!(vreg(&w, 4, 0), f(-5.0));
}

#[test]
fn fma_mix_converts_f16_specials_exactly() {
    let run_mix = |a: u32, b: u32, c: u32| {
        let w = wave(
            &format!(
                "v_mov_b32 v1, {a:#x}\nv_mov_b32 v2, {b:#x}\nv_mov_b32 v3, {c:#x}\n\
                 v_fma_mix_f32 v4, v1, v2, v3 op_sel_hi:[1,1,0]"
            ),
            &[],
        );
        vreg(&w, 4, 0)
    };
    // The least f16 denormal times 1 plus 0: 2^-24 exactly.
    assert_eq!(run_mix(0x0001, 0x3C00, 0), 0x3380_0000);
    // f16 infinities convert to f32's; inf - inf is the default NaN.
    assert_eq!(run_mix(0x7C00, 0x3C00, 0), 0x7F80_0000);
    assert_eq!(run_mix(0x7C00, 0x3C00, 0xFF80_0000), 0xFFC0_0000);
    // An f16 NaN (signaling, payload 5): converted with its payload, quieted.
    assert_eq!(run_mix(0x7C05, 0x3C00, 0), 0x7FC0_A000);
    // An f32 denormal addend is kept: 0 * 1 + 2^-149.
    assert_eq!(run_mix(0, 0x3C00, 1), 1);
    // -0 * 1 + -0 = -0.
    assert_eq!(run_mix(0x8000, 0x3C00, 0x8000_0000), 0x8000_0000);
}

// -------------------------------------------------------------------- f64

/// `op v[10:11],...` on lane 0's 64-bit sources v[2:3], v[4:5], v[6:7]
/// (`args` spells the sources); D's 64 bits.
fn f64_op(op: &str, args: &str, x: u64, y: u64, z: u64) -> u64 {
    let mut body = String::new();
    for (r, v) in [(2, x), (4, y), (6, z)] {
        body += &format!(
            "v_mov_b32 v{r}, {:#x}\nv_mov_b32 v{}, {:#x}\n",
            v as u32,
            r + 1,
            (v >> 32) as u32
        );
    }
    body += &format!("{op} v[10:11], {args}");
    let w = wave(&body, &[]);
    u64::from(vreg(&w, 10, 0)) | u64::from(vreg(&w, 11, 0)) << 32
}

#[test]
fn mul_f64_rounds_once_keeps_denormals_and_propagates_nans() {
    let mul = |x: f64, y: f64| f64_op("v_mul_f64", "v[2:3], v[4:5]", x.to_bits(), y.to_bits(), 0);
    let tiny = f64::from_bits(1);
    assert_eq!(mul(1.5, -2.0), (-3.0f64).to_bits());
    assert_eq!(mul(0.1, 3.0), (0.1f64 * 3.0).to_bits());
    assert_eq!(mul(tiny, 0.5), 0, "2^-1075 ties to even: +0");
    assert_eq!(mul(f64::from_bits(3), 0.5), 2, "1.5 * 2^-1074 ties to 2");
    assert_eq!(
        mul(f64::MIN_POSITIVE, 0.5),
        f64::MIN_POSITIVE.to_bits() >> 1
    );
    assert_eq!(mul(-0.0, 0.0), 0x8000_0000_0000_0000);
    assert_eq!(
        mul(f64::INFINITY, 0.0),
        0xFFF8_0000_0000_0000,
        "the default NaN"
    );
    let snan = f64::from_bits(0x7FF4_0000_0000_0001);
    assert_eq!(mul(snan, 1.0), 0x7FFC_0000_0000_0001, "a NaN: quieted");
    assert_eq!(mul(f64::MAX, 2.0), f64::INFINITY.to_bits());
    // Modifiers on bit 63, and an integer constant sign-extended to 64 bits
    // (1: the least denormal; -1: all ones, a NaN).
    let (two, three) = (2.0f64.to_bits(), 3.0f64.to_bits());
    assert_eq!(
        f64_op(
            "v_mul_f64",
            "-v[2:3], |v[4:5]|",
            two,
            (-3.0f64).to_bits(),
            0
        ),
        (-6.0f64).to_bits()
    );
    assert_eq!(f64_op("v_mul_f64", "v[2:3], 1", two, 0, 0), 2);
    assert_eq!(
        f64_op("v_mul_f64", "v[2:3], -1", three, 0, 0),
        u64::MAX,
        "-1 is all ones: a NaN, quieted already"
    );
}

#[test]
fn fma_f64_is_fused() {
    let fma = |x: f64, y: f64, z: f64| {
        f64_op(
            "v_fma_f64",
            "v[2:3], v[4:5], v[6:7]",
            x.to_bits(),
            y.to_bits(),
            z.to_bits(),
        )
    };
    let e = f64::EPSILON;
    // (1 + e)(1 - e) - 1 = -e^2 exactly; unfused it is 0.
    assert_eq!(fma(1.0 + e, 1.0 - e, -1.0), (-(e * e)).to_bits());
    assert_eq!(fma(2.0, 3.0, 1.0), 7.0f64.to_bits());
    assert_eq!(fma(f64::from_bits(1), 0.5, 0.0), 0);
    assert_eq!(fma(-0.0, 1.0, -0.0), 0x8000_0000_0000_0000);
    assert_eq!(fma(0.0, 1.0, -0.0), 0);
    assert_eq!(
        fma(f64::INFINITY, 1.0, f64::NEG_INFINITY),
        0xFFF8_0000_0000_0000
    );
    assert_eq!(
        fma(f64::MAX, 2.0, f64::NEG_INFINITY),
        f64::NEG_INFINITY.to_bits()
    );
}

#[test]
fn fma_mix_rounds_once_as_the_gpu_does() {
    // Measured on gfx1201: an f32 times an f16 whose product lies on a
    // binary32 midpoint, plus a small addend: rounding the sum in binary64
    // first would lose the addend and tie to even; the GPU rounds once.
    for (x, y, z, want) in [
        (0xe0a5_bc00u32, 0x4f05u32, 0x4104u32, 0xe311_6c15u32),
        (0x7106_b68c, 0x1d80, 0x33d4, 0x6d39_3b01),
    ] {
        let w = wave(
            &format!(
                "v_mov_b32 v1, {x:#x}\nv_mov_b32 v2, {y:#x}\nv_mov_b32 v3, {:#x}\n\
                 v_fma_mix_f32 v4, v1, v2, v3 op_sel:[0,0,1] op_sel_hi:[0,1,1]",
                z << 16
            ),
            &[],
        );
        assert_eq!(vreg(&w, 4, 0), want, "{x:#x} {y:#x} {z:#x}");
    }
}

// ------------------------------------------------------- the GPU's numbers

/// `op v4, v1[, v2[, v3]]` on lane 0's patterns; v4 starts as 0xa5a50000.
fn op1(op: &str, srcs: &[u32]) -> u32 {
    let mut body = String::from("v_mov_b32 v4, 0xa5a50000\n");
    for (k, v) in srcs.iter().enumerate() {
        body += &format!("v_mov_b32 v{}, {v:#x}\n", k + 1);
    }
    let names: Vec<String> = (1..=srcs.len()).map(|k| format!("v{k}")).collect();
    body += &format!("{op} v4, {}", names.join(", "));
    vreg(&wave(&body, &[]), 4, 0)
}

const QP: u32 = 0x7FC1_2345; // quiet NaN, payload
const QN: u32 = 0xFFC5_4321; // quiet NaN, negative
const SP: u32 = 0x7F80_1234; // signaling NaN
const ONE: u32 = 0x3F80_0000;
const INF: u32 = 0x7F80_0000;

#[test]
fn an_f32_nan_is_the_first_nan_operand_quieted_or_the_default_nan() {
    let cases: &[(&str, &[u32], u32)] = &[
        ("v_add_f32", &[QP, ONE], QP),
        ("v_add_f32", &[ONE, SP], 0x7FC0_1234),
        ("v_add_f32", &[QN, SP], QN),
        ("v_add_f32", &[SP, QN], 0x7FC0_1234),
        ("v_add_f32", &[INF, INF | 1 << 31], 0xFFC0_0000), // inf - inf
        ("v_mul_f32", &[INF, 0], 0xFFC0_0000),
        ("v_sub_f32", &[ONE, QP], QP | 1 << 31), // the subtrahend negated
        ("v_subrev_f32", &[QP, ONE], QP | 1 << 31),
        ("v_fma_f32", &[INF, 0, QP], 0xFFC0_0000), // the product's NaN first
        ("v_fma_f32", &[ONE, ONE, SP], 0x7FC0_1234),
        ("v_max_num_f32", &[SP, QN], 0x7FC0_1234), // both NaN: S0 quieted
        ("v_max_num_f32", &[QP, ONE], ONE),
        ("v_minimum3_f32", &[QP, SP, ONE], QP), // the first, signaling or not
        ("v_ldexp_f32", &[SP, 3], 0x7FC0_1234),
        ("v_sqrt_f32", &[0xBF80_0000], 0xFFC0_0000),
        ("v_log_f32", &[0xBF80_0000], 0xFFC0_0000),
        ("v_rcp_f32", &[QN], QN),
        ("v_ceil_f32", &[SP], 0x7FC0_1234),
        // The transcendental unit flushes denormals, in and out.
        ("v_sqrt_f32", &[3], 0),
        ("v_rcp_f32", &[0x8000_0003], 0xFF80_0000),
        ("v_exp_f32", &[0xC310_0000], 0), // 2^-144 flushed
        // med3_num compares bits: -0 is not +0.
        (
            "v_med3_num_f32",
            &[0xC080_0000, 0x8000_0000, 0],
            0x8000_0000,
        ),
    ];
    for (op, srcs, want) in cases {
        assert_eq!(op1(op, srcs), *want, "{op} {srcs:x?}");
    }
}

#[test]
fn f16_and_f64_nans_follow_the_same_rule() {
    // pk_mul_f16 per half; the invalid default 0xFE00.
    assert_eq!(
        op1("v_pk_mul_f16", &[0x7C00_FD01, 0x0000_3C00]),
        0xFE00_FF01
    );
    // v_cvt_f16_f32 keeps the destination's high half on the GPU.
    assert_eq!(op1("v_cvt_f16_f32", &[QP]), 0xA5A5_7E09);
    assert_eq!(op1("v_cvt_f16_f32", &[ONE]), 0xA5A5_3C00);
    assert_eq!(op1("v_cvt_f16_f32_e64", &[ONE]), 0xA5A5_3C00, "e64 too");
}

#[test]
fn dot2_bf16_signs_a_nan_input_and_sums_on_its_grid() {
    // A NaN bf16 source: 0xFFC00A3D, whatever z; an invalid product: the
    // default NaN, before z's NaN; z's NaN alone quieted.
    assert_eq!(
        op1("v_dot2_f32_bf16", &[0x3F80_7FC1, 0x3F80_3F80, QP]),
        0xFFC0_0A3D
    );
    assert_eq!(
        op1("v_dot2_f32_bf16", &[0x7F80_3F80, 0x0000_3F80, QP]),
        0xFFC0_0000
    );
    assert_eq!(
        op1("v_dot2_f32_bf16", &[0x3F80_3F80, 0x3F80_3F80, SP]),
        0x7FC0_1234
    );
    // Measured lanes (gfx1201) where one RNE rounding of the exact sum is off.
    for (x, y, z, want) in [
        (
            0x4040_4040u32,
            0x9EFA_5221u32,
            0x4D2A_D711u32,
            0x52F1_955Au32,
        ),
        (0x3F80_0003, 0x3F80_C020, 0x0000_0003, 0x3F7F_FFFF),
        (0x0003_2701, 0x7094_3F80, 0, 0x2EDE_0100),
        (0xC020_8000, 0x0000_51D7, 0x8000_0000, 0), // a zero is +0
    ] {
        assert_eq!(
            op1("v_dot2_f32_bf16", &[x, y, z]),
            want,
            "{x:#x} {y:#x} {z:#x}"
        );
    }
}

#[test]
fn fma_f32_rounds_once() {
    // A binary32 midpoint product plus an addend a binary64 rounding loses:
    // rounding twice would tie to even (0xe3116c16), the GPU rounds once.
    assert_eq!(
        op1("v_fma_f32", &[0xE0A5_BC00, 0x41E0_A000, 0x4020_8000]),
        0xE311_6C15
    );
}

#[test]
fn an_f16_transcendental_rounds_once_and_keeps_the_high_half() {
    // Measured on gfx1201 (every binary16 input): one RNE rounding of the
    // exact result, denormals kept, NaNs quieted, invalid ones 0xFE00.
    for (op, x, want) in [
        ("v_exp_f16", 0x3C00u32, 0x4000u32),
        ("v_log_f16", 0xBC00, 0xFE00),
        ("v_rsq_f16", 0x8000, 0xFC00),
        ("v_rcp_f16", 0x0001, 0x7C00),
        ("v_sqrt_f16", 0x0001, 0x0C00),
        ("v_sqrt_f16", 0x7C01, 0x7E01),
        ("v_exp_f16_e64", 0xCE00, 0x0001),
    ] {
        assert_eq!(op1(op, &[x]), 0xA5A5_0000 | want, "{op} {x:#x}");
    }
}

#[test]
fn bad_accesses_fault_and_unmodelled_forms_are_limits() {
    let e = run_with(
        "s_load_b128 s[8:11], s[4:5], 0x8\ns_wait_kmcnt 0",
        &[0; 4],
        4,
    )
    .unwrap_err();
    assert_eq!(e.cause, Some(Cause::Fault), "{}", e.message);
    assert!(
        e.message.ends_with("a scalar load past the end of x"),
        "{}",
        e.message
    );
    let e = run_with("v_pk_mul_f16 v1, v0, v0 neg_lo:[1,0]", &[0; 4], 4).unwrap_err();
    assert_eq!(e.cause, Some(Cause::Emulator), "{}", e.message);
}

#[test]
fn float_compares_order_nans_as_the_guide_does() {
    // Lane 0 compares NaN with 1.0, lane 1 1.0 with 1.0, lane 2 1.0 with
    // 2.0; each compare's mask is stored as a dword (1 a lane).
    let nan = 0x7fc0_0000;
    let one = 1.0f32.to_bits();
    let two = 2.0f32.to_bits();
    let want = [
        ("lg", [0, 0, 1]),
        ("neq", [1, 0, 1]),
        ("nlg", [1, 1, 0]),
        ("nlt", [1, 1, 0]),
        ("nge", [1, 0, 1]),
        ("o", [0, 1, 1]),
        ("u", [1, 0, 0]),
    ];
    for (kind, mask) in want {
        let body = format!(
            "global_load_b32 v1, v9, s[6:7]\nglobal_load_b32 v2, v9, s[6:7] offset:12\n\
             s_wait_loadcnt 0\nv_cmp_{kind}_f32 vcc_lo, v1, v2\nv_cndmask_b32 v3, 0, 1, vcc_lo\n\
             global_store_b32 v9, v3, s[4:5]"
        );
        let mut input = vec![0; 35];
        input[..6].copy_from_slice(&[nan, one, one, one, one, two]);
        let x = run_with(&body, &input, 32).unwrap();
        assert_eq!(x[..3], mask, "v_cmp_{kind}_f32");
    }
}

#[test]
fn a_reverse_borrow_out_is_0_in_inactive_lanes() {
    // v_subrev_co_ci_u32 v1, s8, v0, 0, vcc_lo with vcc_lo = 0: 0 - lane
    // borrows in lanes 1..; only lanes 0..15 are active.
    let x = run("s_mov_b32 vcc_lo, 0\ns_mov_b32 exec_lo, 0xffff\n\
         v_subrev_co_ci_u32_e64 v1, s8, v0, 0, vcc_lo\ns_mov_b32 exec_lo, -1\n\
         v_mov_b32 v2, s8\nglobal_store_b32 v9, v2, s[4:5]");
    assert_eq!(x, vec![0xfffe; 32]);
}

#[test]
fn an_inline_integer_is_64_bits_wide_and_an_inline_float_is_a_limit() {
    let x = run("v_lshlrev_b64 v[2:3], 0, -1\nglobal_store_b32 v9, v3, s[4:5]");
    assert_eq!(x, vec![u32::MAX; 32]);
    let x = run("v_lshlrev_b64 v[2:3], 0, 7\nglobal_store_b32 v9, v3, s[4:5]");
    assert_eq!(x, vec![0; 32]);
    let e = refused("v_lshlrev_b64 v[2:3], 0, 1.0");
    assert_eq!(e.cause, Some(Cause::Emulator), "{}", e.message);
    assert!(
        e.message.contains("64-bit vector source is not modelled"),
        "{}",
        e.message
    );
}

#[test]
fn bpermute_adds_its_offset_and_idxen_is_a_limit() {
    // Lane l reads lane (4 l + 8) / 4 = l + 2 (mod 32) of v1 = 100 + l.
    let x = run(
        "v_add_nc_u32 v1, 100, v0\nds_bpermute_b32 v2, v9, v1 offset:8\n\
         s_wait_dscnt 0\nglobal_store_b32 v9, v2, s[4:5]",
    );
    let want: Vec<u32> = (0..32).map(|l| 100 + (l + 2) % 32).collect();
    assert_eq!(x, want);
    let e = refused(
        "s_mov_b32 s8, 0\ns_mov_b32 s9, 0\ns_mov_b32 s10, 64\ns_mov_b32 s11, 0\n\
         buffer_load_b32 v1, v0, s[8:11], null idxen",
    );
    assert_eq!(e.cause, Some(Cause::Emulator), "{}", e.message);
}

#[test]
fn a_packed_bf16_atomic_of_loaded_data_runs_under_the_plain_checker() {
    // x as a raw buffer: base s[4:5], 128 bytes; each lane adds its loaded
    // y word (two bf16 1.0s) to x[l].
    let x = run_with(
        "s_mov_b32 s8, s4\ns_and_b32 s9, s5, 0xffff\ns_mov_b32 s10, 128\n\
         s_mov_b32 s11, 0x30020000\nglobal_load_b32 v1, v9, s[6:7]\ns_wait_loadcnt 0\n\
         buffer_atomic_pk_add_bf16 v1, v9, s[8:11], null offen\ns_wait_storecnt 0",
        &[0x3f80_3f80; 32],
        32,
    )
    .unwrap();
    assert_eq!(x, vec![0x3f80_3f80; 32]);
}

/// A checker that takes every atomic as checked, counting them.
struct Counting(std::rc::Rc<std::cell::Cell<usize>>);

impl thin_emu::tags::checker::Checker for Counting {
    fn atomic(
        &mut self,
        _tables: &mut thin_emu::Tables,
        _call: &thin_emu::tags::checker::StoreCall<'_>,
    ) -> Result<thin_emu::tags::checker::AtomicAnswer, thin_emu::tags::TagError> {
        self.0.set(self.0.get() + 1);
        Ok(thin_emu::tags::checker::AtomicAnswer::Checked)
    }
}

#[test]
fn the_f32_global_atomic_asks_the_checker_as_the_packed_bf16_one_does() {
    let t = thin_targets::get(thin_targets::default_name()).unwrap();
    let text = format!(
        "{HEAD}v_mov_b32 v1, 1.0\nglobal_atomic_add_f32 v9, v1, s[4:5]\ns_wait_storecnt 0\ns_endpgm\n"
    );
    let program = thin_emu::read_program(&text, t).unwrap();
    let mut mem = Memory::default();
    let x = mem.alloc("x", 128, true, 0, Meta::default()).unwrap();
    let kb = mem.alloc("kernarg", 16, false, 0, Meta::default()).unwrap();
    let xb = mem.bufs[x].base;
    mem.bufs[kb].data[..8].copy_from_slice(&xb.to_le_bytes());
    mem.bufs[kb].data[8..16].copy_from_slice(&xb.to_le_bytes());
    let machine = Machine::new(program, Facts::of(t), 64, 0, 256).unwrap();
    let mut c = Cluster::new(
        vec![machine],
        vec![vec![mem.bufs[kb].base]],
        (1, 1),
        1,
        Options::default(),
    );
    let calls = std::rc::Rc::new(std::cell::Cell::new(0));
    c.checker = Box::new(Counting(calls.clone()));
    c.run(&mut mem, &mut thin_emu::Tables::default()).unwrap();
    assert_eq!(calls.get(), 1);
}

/// RDNA4 ISA guide, Table 40: VGPR moves indexed by M0, and the scalar
/// ones, including the guide's own examples.
#[test]
fn movrel_moves_index_registers_through_m0() {
    let body = "v_mov_b32 v17, 11\nv_mov_b32 v7, 22\nv_mov_b32 v12, 33\ns_mov_b32 m0, 10\n\
                v_movrels_b32 v5, v7\nv_movreld_b32 v5, v12\nv_movrelsd_b32 v6, v7\n\
                s_mov_b32 m0, 0x20003\nv_movrelsd_2_b32 v30, v14\n\
                s_mov_b32 s17, 44\ns_mov_b32 m0, 10\ns_movrels_b32 s20, s7\ns_movreld_b32 s15, s20\n\
                v_mov_b32 v1, s25\n\
                global_store_b32 v9, v5, s[4:5]\nglobal_store_b32 v9, v15, s[4:5] offset:128\n\
                global_store_b32 v9, v16, s[4:5] offset:256\nglobal_store_b32 v9, v32, s[4:5] offset:384\n\
                global_store_b32 v9, v1, s[4:5] offset:512";
    let x = run_with(body, &[0], 160).unwrap();
    for l in 0..32 {
        assert_eq!(x[l], 11, "v5 = v[7 + 10]");
        assert_eq!(x[32 + l], 33, "v[5 + 10] = v12");
        assert_eq!(x[64 + l], 11, "v[6 + 10] = v[7 + 10]");
        assert_eq!(x[96 + l], 11, "v[30 + 2] = v[14 + 3]");
        assert_eq!(x[128 + l], 44, "s[15 + 10] = s20 = s[7 + 10]");
    }
    // An index past the wave's file is the program's fault.
    for body in [
        "s_mov_b32 m0, 300\nv_movrels_b32 v1, v7",
        "s_mov_b32 m0, 200\ns_movreld_b32 s1, s2",
    ] {
        assert_eq!(
            run_with(body, &[0], 1).unwrap_err().cause,
            Some(Cause::Fault),
            "{body}"
        );
    }
    let odd = run_with("s_mov_b32 m0, 1\ns_movrels_b64 s[10:11], s[2:3]", &[0], 1);
    assert_eq!(odd.unwrap_err().cause, Some(Cause::Fault));
}
