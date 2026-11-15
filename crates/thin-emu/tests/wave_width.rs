//! The per-wave analyses split a workgroup into waves of the target's
//! width: 32 lanes on gfx1201.

use thin_passes::ranges;
use thin_test_support::adapters::default_target;
use thin_test_support::asm;

#[test]
fn per_wave_ranges_take_the_target_s_wave_width() {
    let t = default_target();
    // tid << 27 fits 32 bits for tid < 32 only.
    let lines = [
        "v_lshlrev_b32 v2, 27, v0",
        "ds_load_b32 v3, v2",
        "s_wait_dscnt 0",
        "s_endpgm",
    ];
    let s = asm::snapshot(t, asm::read(t, &lines).unwrap());
    let launch = ranges::Launch {
        args: Vec::new(),
        wgsize: 80,
        wgy: 0,
        max_grid: Vec::new(),
        preload: 0,
        tid: None,
    };
    let waves = ranges::per_wave(&s, &launch);
    // 80 lanes: two whole waves of 32 and one of 16.
    assert_eq!(waves.len(), 3);
    assert!(waves[0].is_ok(), "{:?}", waves[0]);
    assert!(waves[1].is_err() && waves[2].is_err());
}
