//! Each refusal of the LDS proof once, on hand-written
//! listings in `lives.json`'s notation (a layout of a union of G and W,
//! then X; the wave index in s0, four waves).

use thin_passes::lds;
use thin_test_support::asm;
use thin_test_support::listing::Listing;

const LAYOUT: [&str; 6] = [
    "; lds: 512 B of a 4096 B budget",
    ";   offset     end   size  region           shape                strides",
    ";        0     256    256  union",
    ";        0     256    256    + G            [64] u32             4",
    ";        0     256    256    | W            [64] u32             4",
    ";      256     512    256  X                [64] u32             4",
];

/// The proof's refusal of a listing's body, spelled.
fn refusal(body: &[&str]) -> String {
    let lines: Vec<&str> = LAYOUT.iter().chain(body).copied().collect();
    let l = Listing::new(&lines, &[], 4, None).expect("a listing");
    let o = lds::Options {
        ablate: false,
        wave_order: true,
    };
    let d = lds::prove(&l.snapshot, &l.layout, o).expect_err("a refusal");
    asm::message(&l.snapshot, &d.msg)
}

#[test]
fn an_lds_fence_with_an_op_in_flight() {
    let got = refusal(&[
        "ds_load_b32 v5, v1 // G + lane * 4",
        "s_barrier_signal -1",
        "s_barrier_wait 65535 // [lds fence] end",
    ]);
    assert_eq!(
        got,
        "the LDS fence 2 `s_barrier_wait` with `ds_load_b32 G + lane * 4` (instruction 0) in flight: an LDS fence \
         needs every wave's LDS ops and DMAs done (a wait) before it"
    );
}

#[test]
fn a_write_in_flight_at_a_barrier() {
    let got = refusal(&[
        "ds_store_b32 v1, v4 offset:256 // X + lane * 4",
        "s_barrier_signal -1",
        "s_barrier_wait 65535 // written",
        "ds_load_b32 v5, v1 offset:256 // X + lane * 4",
        "s_wait_dscnt 0",
    ]);
    assert_eq!(
        got,
        "`ds_store_b32 X + lane * 4` (instruction 0) is in flight at the barrier 'written' (instruction 2), and \
         `ds_load_b32 X + lane * 4` (instruction 3) accesses its bytes after it: another wave may run it before the \
         op lands (wait before the barrier: b.wait(X))"
    );
}

#[test]
fn union_arms_live_together() {
    let got = refusal(&[
        "ds_store_b32 v1, v4 // G + lane * 4",
        "ds_store_b32 v1, v4 // W + lane * 4",
        "s_wait_dscnt 0",
    ]);
    assert_eq!(
        got,
        "G [0, 256) and W [0, 256) share bytes and are both live in the epoch 'entry': G accessed 0 `ds_store_b32 G + \
         lane * 4` .. 0 `ds_store_b32 G + lane * 4`, in it at 0 `ds_store_b32 G + lane * 4`; W accessed 1 \
         `ds_store_b32 W + lane * 4` .. 1 `ds_store_b32 W + lane * 4`, in it at 1 `ds_store_b32 W + lane * 4`"
    );
}

#[test]
fn a_wave_dependent_branch_over_a_barrier() {
    let got = refusal(&[
        "s_cmp_eq_u32 s0, 0",
        "s_cbranch_scc0 .Lif_1",
        "s_barrier_signal -1",
        "s_barrier_wait 65535 // wave 0 only",
        ".Lif_1:",
        "s_barrier_signal -1",
        "s_barrier_wait 65535 // every wave",
    ]);
    assert_eq!(
        got,
        "the wave-dependent branch `s_cbranch_scc0 .Lif_1` (instruction 1) rejoins at .Lif_1 (instruction 4) over \
         paths passing 0 and 1 s_barrier: barriers pair by count, so the epochs between them would not be every \
         wave's - give every path the same barriers"
    );
}

#[test]
fn a_wave_dependent_branch_into_a_loop_with_a_barrier() {
    let got = refusal(&[
        "s_cmp_eq_u32 s0, 0",
        "s_cbranch_scc0 .Lif_1",
        "s_mov_b32 s2, 0",
        ".Lloop_2:",
        "s_barrier_signal -1",
        "s_barrier_wait 65535 // in wave 0",
        "s_add_co_u32 s2, s2, 1",
        "s_cmp_lt_u32 s2, 2",
        "s_cbranch_scc1 .Lloop_2",
        ".Lif_1:",
        "s_endpgm",
    ]);
    assert_eq!(
        got,
        "the wave-dependent branch `s_cbranch_scc0 .Lif_1` (instruction 1) leads into a loop holding s_barrier before \
         its join (instruction 9): waves would pass different numbers of barriers (barriers pair by count)"
    );
}

#[test]
fn a_wave_dependent_branch_that_never_ends() {
    let got = refusal(&[
        "s_cmp_eq_u32 s0, 0",
        ".Lspin_1:",
        "s_cbranch_scc0 .Lspin_1",
        ".Lstop_2:",
        "s_branch .Lstop_2",
    ]);
    assert_eq!(got, "`s_cbranch_scc0 .Lspin_1` (instruction 2): a wave-dependent branch on a path that never ends");
}

#[test]
fn a_race_between_waves() {
    let lines: Vec<&str> = LAYOUT
        .iter()
        .copied()
        .chain([
            "ds_store_b32 v1, v4 offset:256 // X + lane * 4",
            "s_wait_dscnt 0",
        ])
        .collect();
    let l = Listing::new(&lines, &[], 2, None).expect("a listing");
    let p = lds::prove(&l.snapshot, &l.layout, lds::Options::default()).expect("a proof");
    let bytes = |lo, hi| lds::WaveBytes::from([(0, (lo, hi))]);
    assert!(lds::races(&l.snapshot, &p, &[bytes(256, 384), bytes(384, 512)]).is_ok());
    let d = lds::races(&l.snapshot, &p, &[bytes(256, 384), bytes(380, 512)]).expect_err("a race");
    assert!(asm::message(&l.snapshot, &d.msg).starts_with(
        "`ds_store_b32 X + lane * 4` (instruction 0) in wave 0 and `ds_store_b32 X + lane * 4` (instruction 0) in \
         wave 1 may touch the same bytes ([256, 384) and [380, 512)) in the epoch 'entry'"
    ));
}

#[test]
fn the_proof_is_deterministic_and_reports_the_union() {
    let body = ["ds_store_b32 v1, v4 // G + lane * 4", "s_wait_dscnt 0"];
    let lines: Vec<&str> = LAYOUT.iter().chain(&body).copied().collect();
    let l = Listing::new(&lines, &[], 4, None).expect("a listing");
    let p = lds::prove(&l.snapshot, &l.layout, lds::Options::default()).expect("a proof");
    assert_eq!(
        p,
        lds::prove(&l.snapshot, &l.layout, lds::Options::default()).expect("a proof")
    );
    assert_eq!(
        p.report,
        [
            "lds: union [0, 256) proven by the lives: G in 'entry'; W in never accessed",
            "lds: never accessed: W, X"
        ]
    );
}
