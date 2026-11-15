//! Register-range constraints: authoring forms need none, because the
//! encoder chooses an encoding that fits whatever registers a placement
//! gives; only encoding-fixed physical forms carry `Slot::limit`.

use thin_targets::decode::{Instruction, Opd};

fn bytes(line: &str) -> Result<usize, String> {
    let t = thin_targets::get(thin_targets::default_name()).expect("the default target");
    match t.hooks().encode(line)? {
        thin_target::Encoded::Bytes(b) => Ok(b.len()),
        thin_target::Encoded::Branch { .. } => Ok(4),
    }
}

/// A 16-bit short op past v127 takes the long encoding by itself.
#[test]
fn an_authoring_16_bit_op_past_v127_takes_the_long_encoding() {
    assert_eq!(bytes("v_max_num_f16 v1, v2, v3"), Ok(4));
    assert_eq!(bytes("v_max_num_f16 v130, v2, v3"), Ok(8));
    assert_eq!(bytes("v_max_num_f16 v1, v2, v200"), Ok(8));
    assert!(bytes("v_max_num_f16_e32 v130, v2, v3").is_err());
}

/// Every short authoring form has a long counterpart, but for these,
/// none of which is a 16-bit op the encoder takes: so no authoring form
/// needs a register limit. A new short-only 16-bit form fails here.
#[test]
fn every_short_authoring_form_but_known_32_bit_ones_has_a_long_form() {
    let t = thin_targets::get(thin_targets::default_name()).expect("the default target");
    let d = t.definition();
    let mut short_only: Vec<&str> = (d.forms.iter())
        .filter(|f| matches!(f.encoding.as_str(), "ENC_VOP1" | "ENC_VOP2" | "ENC_VOPC"))
        .filter(|f| {
            let long = format!("{}_e64", f.name);
            !d.encodings.contains(&long) && t.form(&long, f.slots.len(), false).is_none()
        })
        .map(|f| f.name.as_str())
        .collect();
    short_only.sort_unstable();
    assert_eq!(
        short_only,
        [
            "v_permlane64_b32",
            "v_pk_fmac_f16",
            "v_swap_b16",
            "v_swap_b32",
            "v_swaprel_b32"
        ]
    );
    assert!(
        bytes("v_swap_b16 v1, v2").is_err(),
        "not encoded on this target"
    );
    assert_eq!(
        bytes("v_pk_fmac_f16 v200, v201, v202"),
        Ok(4),
        "packed 32-bit fields"
    );
}

/// A decoded short form's 16-bit fields are limited; its long form's and
/// a 32-bit short form's are not.
#[test]
fn physical_short_16_bit_forms_carry_the_halves_limit() {
    let name = thin_targets::default_name();
    let limits = |line: &str| {
        let t = thin_targets::get(name).expect("the default target");
        let thin_target::Encoded::Bytes(b) = t.hooks().encode(line).unwrap() else {
            panic!("bytes")
        };
        let i: Instruction = thin_targets::decode(name, &b).unwrap()[0];
        let f = thin_targets::physical_form(name, &i).unwrap();
        (i.operands().iter().enumerate())
            .filter(|(_, o)| matches!(o, Opd::V { .. }))
            .map(|(k, _)| f.slots[k].limit)
            .collect::<Vec<_>>()
    };
    assert_eq!(limits("v_max_num_f16 v1, v2, v3"), [Some(128); 3]);
    assert_eq!(limits("v_max_num_f16 v130, v2, v3"), [None; 3]);
    assert_eq!(limits("v_max_num_f32 v1, v2, v3"), [None; 3]);
}
