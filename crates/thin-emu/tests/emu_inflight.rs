//! gfx1201's pending register writes: load, ds and km results
//! land at the wait that retires them; touching a pending register before
//! is `RegisterInFlight`.

use thin_emu::machine::cluster::{Cluster, Options};
use thin_emu::machine::facts::Facts;
use thin_emu::machine::memory::{Memory, Meta};
use thin_emu::machine::run::Machine;
use thin_emu::{Cause, EmuError};

const HEAD: &str = "s_load_b64 s[4:5], s[0:1], 0x0\ns_wait_kmcnt 0\nv_lshlrev_b32 v9, 2, v0\n";

/// Runs `body` on one 32-lane wave with `x` (32 dwords, x[i] = 1000 + i) at
/// s[4:5] and v9 = lane · 4; returns x.
fn run(body: &str) -> Result<Vec<u32>, EmuError> {
    let t = thin_targets::get(thin_targets::default_name()).expect("registered default target");
    let text = format!("{HEAD}{body}\ns_endpgm\n");
    let program = thin_emu::read_program(&text, t)?;
    let mut mem = Memory::default();
    let x = mem.alloc("x", 128, true, 0, Meta::default())?;
    for i in 0..32 {
        mem.bufs[x].data[4 * i..4 * i + 4].copy_from_slice(&(1000 + i as u32).to_le_bytes());
    }
    let kb = mem.alloc("kernarg", 8, false, 0, Meta::default())?;
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
    Ok((0..32)
        .map(|i| u32::from_le_bytes(mem.bufs[x].data[4 * i..4 * i + 4].try_into().unwrap()))
        .collect())
}

fn in_flight(body: &str) -> String {
    let e = run(body).expect_err("RegisterInFlight");
    assert_eq!(e.cause, Some(Cause::InFlight), "{}", e.message);
    e.message
}

const LOAD: &str = "global_load_b32 v1, v9, s[4:5]";

#[test]
fn a_load_read_before_its_wait_is_in_flight_and_after_it_is_loaded() {
    let m = in_flight(&format!(
        "{LOAD}\nv_add_nc_u32 v2, 1, v1\nglobal_store_b32 v9, v2, s[4:5]"
    ));
    assert!(
        m.contains("v_add_nc_u32 v2, 1, v1: v1 is still being loaded by `global_load_b32 v1, v9, s[4:5]` (instruction 3): a wait must retire it first"),
        "{m}"
    );
    let x = run(&format!(
        "{LOAD}\ns_wait_loadcnt 0\nv_add_nc_u32 v2, 1, v1\nglobal_store_b32 v9, v2, s[4:5]"
    ))
    .unwrap();
    assert_eq!(x, (0..32).map(|i| 1001 + i).collect::<Vec<u32>>());
}

#[test]
fn a_wait_retires_in_issue_order() {
    let two = "global_load_b32 v1, v9, s[4:5]\nglobal_load_b32 v2, v9, s[4:5]\ns_wait_loadcnt 1\n";
    run(&format!(
        "{two}v_mov_b32 v3, v1\ns_wait_loadcnt 0\nglobal_store_b32 v9, v3, s[4:5]"
    ))
    .unwrap();
    let m = in_flight(&format!(
        "{two}v_mov_b32 v3, v2\nglobal_store_b32 v9, v3, s[4:5]"
    ));
    assert!(m.contains(": v2 is still being loaded"), "{m}");
}

#[test]
fn a_scalar_load_lands_only_at_kmcnt_0() {
    let load = "s_load_b32 s6, s[4:5], 0x0\n";
    let m = in_flight(&format!("{load}s_wait_kmcnt 1\ns_add_co_u32 s7, s6, 1"));
    assert!(
        m.contains(": s6 is still being loaded by `s_load_b32 s6, s[4:5], 0x0`"),
        "{m}"
    );
    let x = run(&format!(
        "{load}s_wait_kmcnt 0\ns_add_co_u32 s7, s6, 1\nv_mov_b32 v1, s7\nglobal_store_b32 v9, v1, s[4:5]"
    ))
    .unwrap();
    assert!(x.iter().all(|&v| v == 1001));
}

#[test]
fn an_lds_read_lands_at_its_dscnt_wait() {
    let lds = "v_mov_b32 v3, 5\nds_store_b32 v9, v3\ns_wait_dscnt 0\nds_load_b32 v1, v9\n";
    let m = in_flight(&format!("{lds}v_add_nc_u32 v2, 1, v1"));
    assert!(
        m.contains(": v1 is still being loaded by `ds_load_b32 v1, v9`"),
        "{m}"
    );
    let x = run(&format!(
        "{lds}s_wait_dscnt 0\nv_add_nc_u32 v2, 1, v1\nglobal_store_b32 v9, v2, s[4:5]"
    ))
    .unwrap();
    assert!(x.iter().all(|&v| v == 6));
}

#[test]
fn a_second_load_into_a_pending_register_is_in_flight() {
    let m = in_flight(&format!("{LOAD}\nglobal_load_b32 v1, v9, s[4:5] offset:4"));
    assert!(
        m.contains("global_load_b32 v1, v9, s[4:5] offset:4: v1 is still being loaded"),
        "{m}"
    );
}

#[test]
fn a_lane_outside_the_loads_exec_is_checked_too() {
    let m = in_flight(&format!(
        "s_mov_b32 exec_lo, 0xffff\n{LOAD}\ns_mov_b32 exec_lo, 0xffff0000\nv_mov_b32 v2, v1"
    ));
    assert!(m.contains(": v1 is still being loaded"), "{m}");
    // Landing writes the load's lanes only (EXEC at issue).
    let x = run(&format!(
        "v_mov_b32 v1, 7\ns_mov_b32 exec_lo, 0xffff\n{LOAD}\ns_mov_b32 exec_lo, -1\n\
         s_wait_loadcnt 0\nglobal_store_b32 v9, v1, s[4:5]"
    ))
    .unwrap();
    for (l, v) in x.iter().enumerate() {
        assert_eq!(*v, if l < 16 { 1000 + l as u32 } else { 7 }, "lane {l}");
    }
}
