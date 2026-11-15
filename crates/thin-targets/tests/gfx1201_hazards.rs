//! Properties of the gfx1201 hazard insertion pass.

use thin_test_support::adapters::default_target;
use thin_test_support::asm;

/// The s_wait_alu fields each line needs, as `name(count)`.
fn waits(lines: &[&str]) -> Vec<Vec<String>> {
    let t = default_target();
    let s = asm::snapshot(t, asm::read(t, lines).unwrap());
    let model = thin_targets::hazards(t.name()).unwrap();
    let fields = t.wait_fields();
    thin_passes::hazards::needs(&s, model)
        .iter()
        .map(|n| {
            (n.waits.0.iter().zip(&fields))
                .filter_map(|(c, f)| c.map(|c| format!("{f}({c})")))
                .collect()
        })
        .collect()
}

const MEMORY: [&str; 6] = [
    "s_load_b32 s20, s[0:1], 0x0",
    "s_load_b512 s[24:39], s[0:1], 0x0",
    "buffer_load_b32 v3, off, s[4:7], s8",
    "global_load_b32 v1, v0, s[4:5]",
    "ds_load_b32 v1, v2",
    "buffer_store_b32 v3, off, s[4:7], s8",
];

#[test]
fn an_salu_write_stays_pending_across_every_memory_kind() {
    for mem in MEMORY {
        let lines = [
            "v_add_nc_u32 v5, s2, v5",
            "s_mov_b32 s2, 7",
            mem,
            "s_add_co_u32 s9, s2, 1",
        ];
        let got = waits(&lines);
        assert_eq!(got[3], ["depctr_sa_sdst(0)"], "{mem}: {got:?}");
        assert!(got[..3].iter().all(Vec::is_empty), "{mem}: {got:?}");
    }
}

#[test]
fn a_valu_write_stays_pending_across_every_memory_kind() {
    for mem in MEMORY {
        let lines = [
            "v_add_nc_u32 v5, s2, v5",
            "v_readfirstlane_b32 s2, v6",
            mem,
            // a VALU read: an SALU read commits VALU writes
            "v_add_nc_u32 v7, s2, v7",
        ];
        let got = waits(&lines);
        assert_eq!(got[3], ["depctr_va_sdst(0)"], "{mem}: {got:?}");
    }
}

/// The delay hint of each line.
fn delays(lines: &[&str]) -> Vec<u32> {
    let t = default_target();
    let s = asm::snapshot(t, asm::read(t, lines).unwrap());
    let model = thin_targets::hazards(t.name()).unwrap();
    thin_passes::hazards::needs(&s, model)
        .iter()
        .map(|n| n.delay)
        .collect()
}

#[test]
fn a_reader_after_an_overwrite_names_only_the_newer_producer() {
    let trans = ["v_rcp_f32 v6, v8", "v_sqrt_f32 v6, v8", "v_exp_f32 v6, v8"];
    for t in trans {
        // trans, then a plain VALU: VALU_DEP_1 alone (1), no TRANS32_DEP.
        let got = delays(&[t, "v_mul_f32 v6, v6, v7", "v_cvt_u32_f32 v9, v6"]);
        assert_eq!(got[2], 1, "{t} then a VALU: {got:?}");
        // a plain VALU, then trans: TRANS32_DEP_1 alone (5), no VALU_DEP.
        let got = delays(&["v_mul_f32 v6, v8, v7", t, "v_cvt_u32_f32 v9, v6"]);
        assert_eq!(got[2], 5, "a VALU then {t}: {got:?}");
    }
}
