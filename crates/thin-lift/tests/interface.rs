//! Interface laws on thin's own compiled kernels and descriptor variations.

use thin_emu::machine::{facts::Facts, run::Machine, POISON};
use thin_formats::Value;
use thin_lift::interface::{read_interface, ArgumentKind, UserValue};
use thin_target::{descriptor, read_s, SFile};

fn kernels() -> Vec<String> {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../crates/thin-codegen/tests/data/codegen/kernels.json"
    );
    let doc: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    doc["kernels"]
        .as_array()
        .unwrap()
        .iter()
        .map(|k| k["s"].as_str().unwrap().to_owned())
        .collect()
}

fn put(file: &mut SFile, key: &str, value: u32) {
    file.block.insert(key.into(), (value.to_string(), 0));
}

#[test]
fn compiled_kernel_interfaces_reproduce_the_descriptor() {
    let target = thin_targets::get(thin_targets::default_name()).unwrap();
    let inputs = kernels();
    assert_eq!(inputs.len(), 7);
    for text in inputs {
        let bytes = thin_codegen::object::code_object(&text).unwrap();
        let object = thin_lift::object::read(&bytes).unwrap();
        assert_eq!(object.kernels.len(), 1);
        let k = &object.kernels[0];
        let original = k.descriptor.unwrap();
        let (interface, residue) = read_interface(&original, k.metadata.as_ref().unwrap()).unwrap();
        assert_eq!(interface.wave_size, 32);
        assert_eq!(interface.user_sgpr_count, 2);
        assert_eq!(interface.user_sgprs[0].value, UserValue::KernargPointer);
        assert_eq!(interface.arguments[0].kind, ArgumentKind::ByValue);
        let mut file = read_s(&text, false).unwrap();
        // Destroy the template's interface first: it cannot supply the law.
        for value in file.block.values_mut() {
            value.0 = "0".into();
        }
        // Resource fields are chosen independently; only interface fields
        // are claimed by the law, the rest are restored from the residue.
        put(&mut file, "next_free_vgpr", 1);
        put(&mut file, "reserve_vcc", 1);
        assert_eq!(residue.write(&interface, &file, target).unwrap(), original);
    }
}

#[test]
fn every_id_enable_combination_and_user_layout_round_trips() {
    let target = thin_targets::get(thin_targets::default_name()).unwrap();
    let base = read_s(&kernels()[0], false).unwrap();
    let empty = Value::Map(Vec::new());
    for enables in 0..8 {
        for user_flags in 0..32 {
            let mut file = base.clone();
            let keys = [
                "user_sgpr_dispatch_ptr",
                "user_sgpr_queue_ptr",
                "user_sgpr_kernarg_segment_ptr",
                "user_sgpr_dispatch_id",
                "user_sgpr_private_segment_size",
            ];
            let mut count = 0;
            for (n, key) in keys.iter().enumerate() {
                let on = (user_flags >> n) & 1;
                put(&mut file, key, on);
                count += on * if n == 4 { 1 } else { 2 };
            }
            put(&mut file, "user_sgpr_count", count);
            for (n, axis) in ["x", "y", "z"].iter().enumerate() {
                put(
                    &mut file,
                    &format!("system_sgpr_workgroup_id_{axis}"),
                    (enables >> n) & 1,
                );
            }
            put(&mut file, "private_segment_fixed_size", 128);
            put(&mut file, "enable_private_segment", 1);
            put(&mut file, "uses_dynamic_stack", 1);
            let original = descriptor(&file, target).unwrap();
            let (interface, residue) = read_interface(&original, &empty).unwrap();
            assert_eq!(interface.private_segment.size, 128);
            assert!(interface.private_segment.enabled);
            assert!(interface.private_segment.dynamic_stack);
            assert_eq!(residue.write(&interface, &base, target).unwrap(), original);
            for mode in 0..3 {
                let mut changed = interface.clone();
                changed.workitem_id_components = mode;
                let rewritten = residue.write(&changed, &base, target).unwrap();
                assert_eq!(read_interface(&rewritten, &empty).unwrap().0, changed);
            }
        }
    }
}

#[test]
fn descriptor_dispatch_packs_only_enabled_ids_and_includes_z() {
    let target = thin_targets::get(thin_targets::default_name()).unwrap();
    for count in [0, 2, 5, 16] {
        for mask in 0..8 {
            let text = format!("s_endpgm\n.amdhsa_kernel test\n.amdhsa_user_sgpr_count {count}\n.amdhsa_system_sgpr_workgroup_id_x {}\n.amdhsa_system_sgpr_workgroup_id_y {}\n.amdhsa_system_sgpr_workgroup_id_z {}\n.end_amdhsa_kernel\n", mask & 1, (mask >> 1) & 1, (mask >> 2) & 1);
            let program = thin_emu::read_program(&text, target).unwrap();
            let m = Machine::new(program, Facts::of(target), 1, 0, 0).unwrap();
            let waves = m.make_waves(0x1234_5678_90ab_cdef, (17, 29, 43), 2, 0, 0);
            for (k, w) in waves.iter().enumerate() {
                let mut next = count as usize;
                for (n, id) in [17, 29, 43].into_iter().enumerate() {
                    if mask & (1 << n) != 0 {
                        assert_eq!(w.s[next], id);
                        next += 1;
                    }
                }
                assert_eq!(w.s[next], POISON);
                assert_eq!(w.ttmp9, 17);
                assert_eq!(w.ttmp7, 29 | 43 << 16);
                assert_eq!(w.v[0], (k * 32) as u32);
            }
        }
    }
}

#[test]
fn kernarg_pointer_follows_enabled_user_values() {
    let target = thin_targets::get(thin_targets::default_name()).unwrap();
    let text = "s_endpgm\n.amdhsa_kernel test\n.amdhsa_user_sgpr_count 6\n.amdhsa_user_sgpr_dispatch_ptr 1\n.amdhsa_user_sgpr_queue_ptr 1\n.amdhsa_user_sgpr_kernarg_segment_ptr 1\n.end_amdhsa_kernel\n";
    let program = thin_emu::read_program(text, target).unwrap();
    let m = Machine::new(program, Facts::of(target), 1, 0, 0).unwrap();
    let ws = m.make_waves(0x1234_5678_90ab_cdef, (0, 0), 1, 0, 0);
    assert_eq!(ws[0].s[0], POISON);
    assert_eq!(ws[0].s[4], 0x90ab_cdef);
    assert_eq!(ws[0].s[5], 0x1234_5678);
}

#[test]
fn argument_metadata_is_typed_without_discarding_extension_fields() {
    let target = thin_targets::get(thin_targets::default_name()).unwrap();
    let file = read_s(&kernels()[0], false).unwrap();
    let d = descriptor(&file, target).unwrap();
    let arg = Value::Map(vec![
        (".name".into(), Value::Str("x".into())),
        (".offset".into(), Value::Int(0)),
        (".size".into(), Value::Int(8)),
        (".value_kind".into(), Value::Str("global_buffer".into())),
        (".address_space".into(), Value::Str("global".into())),
    ]);
    let meta = Value::Map(vec![(".args".into(), Value::Array(vec![arg]))]);
    let (interface, residue) = read_interface(&d, &meta).unwrap();
    assert_eq!(interface.arguments[0].name.as_deref(), Some("x"));
    assert_eq!(interface.arguments[0].offset, 0);
    assert_eq!(interface.arguments[0].size, 8);
    assert_eq!(interface.arguments[0].kind, ArgumentKind::GlobalBuffer);
    assert!(residue.argument_metadata[0].get(".address_space").is_some());
    let mut bad = d;
    bad[8..12].copy_from_slice(&4u32.to_le_bytes());
    assert!(read_interface(&bad, &meta).is_err());
    assert!(read_interface(&d, &Value::Nil).is_err());
}

#[test]
fn metadata_argument_size_and_descriptor_word_are_independent() {
    let target = thin_targets::get(thin_targets::default_name()).unwrap();
    let text = &kernels()[0];
    let bytes = thin_codegen::object::code_object(text).unwrap();
    let object = thin_lift::object::read(&bytes).unwrap();
    let kernel = &object.kernels[0];
    let mut original = kernel.descriptor.unwrap();
    original[8..12].fill(0);
    let (interface, residue) =
        read_interface(&original, kernel.metadata.as_ref().unwrap()).unwrap();
    assert_eq!(interface.kernarg_size, 28);
    assert_eq!(interface.descriptor_kernarg_size, 0);
    assert_eq!(interface.arguments[0].size, 28);
    let template = read_s(text, false).unwrap();
    assert_eq!(
        residue.write(&interface, &template, target).unwrap(),
        original
    );
}

fn note(metadata: &Value) -> Vec<u8> {
    let encoded = thin_formats::msgpack_write(metadata).unwrap();
    let mut out = Vec::new();
    out.extend_from_slice(&7u32.to_le_bytes());
    out.extend_from_slice(&(encoded.len() as u32).to_le_bytes());
    out.extend_from_slice(&32u32.to_le_bytes());
    out.extend_from_slice(b"AMDGPU\0\0");
    out.extend(encoded);
    out.resize(out.len().div_ceil(4) * 4, 0);
    out
}

fn with_notes(mut bytes: Vec<u8>, notes: &[Value]) -> Vec<u8> {
    let object = thin_lift::object::read(&bytes).unwrap();
    let index = object.sections.iter().position(|s| s.kind == 7).unwrap();
    let table = u64::from_le_bytes(bytes[0x28..0x30].try_into().unwrap()) as usize;
    let stride = u16::from_le_bytes(bytes[0x3a..0x3c].try_into().unwrap()) as usize;
    bytes.resize(bytes.len().div_ceil(4) * 4, 0);
    let offset = bytes.len() as u64;
    let combined: Vec<_> = notes.iter().flat_map(note).collect();
    let size = combined.len() as u64;
    bytes.extend(combined);
    let header = table + index * stride;
    bytes[header + 0x18..header + 0x20].copy_from_slice(&offset.to_le_bytes());
    bytes[header + 0x20..header + 0x28].copy_from_slice(&size.to_le_bytes());
    bytes
}

#[test]
fn every_metadata_note_contributes_its_kernel_entry() {
    let bytes = thin_codegen::object::code_object(&kernels()[0]).unwrap();
    let metadata = thin_lift::object::read(&bytes).unwrap().metadata.unwrap();
    let other = Value::Map(vec![(
        "amdhsa.kernels".into(),
        Value::Array(vec![Value::Map(vec![(
            ".name".into(),
            Value::Str("other".into()),
        )])]),
    )]);
    let image = with_notes(bytes, &[other, metadata]);
    let object = thin_lift::object::read(&image).unwrap();
    assert_eq!(object.metadata_notes.len(), 2);
    assert_eq!(
        object.kernels[0].metadata.as_ref().unwrap().get(".name"),
        Some(&Value::Str(object.kernels[0].name.clone()))
    );
}

#[test]
fn distinct_metadata_note_extension_properties_are_preserved() {
    let bytes = thin_codegen::object::code_object(&kernels()[0]).unwrap();
    let a = Value::Map(vec![("custom.config".into(), Value::Int(1))]);
    let b = Value::Map(vec![("custom.config".into(), Value::Int(2))]);
    let image = with_notes(bytes, &[a, b]);
    let object = thin_lift::object::read(&image).unwrap();
    assert_eq!(
        object.metadata_notes[0].get("custom.config"),
        Some(&Value::Int(1))
    );
    assert_eq!(
        object.metadata_notes[1].get("custom.config"),
        Some(&Value::Int(2))
    );
}

#[test]
fn interface_field_writer_covers_both_wave_sizes_without_allocating_registers() {
    let target = thin_targets::get(thin_targets::default_name()).unwrap();
    let file = read_s(&kernels()[0], false).unwrap();
    let original = descriptor(&file, target).unwrap();
    let (mut interface, _) = read_interface(&original, &Value::Map(Vec::new())).unwrap();
    for wave in [32, 64] {
        interface.wave_size = wave;
        let mut rewritten = file.clone();
        interface.apply_to(&mut rewritten);
        let encoded = thin_target::interface_descriptor(&rewritten, target).unwrap();
        assert_eq!(encoded[56] & 8, original[56] & 8);
        assert_eq!((encoded[57] >> 2) & 1, u8::from(wave == 32));
        assert_eq!(&encoded[..12], &original[..12]);
        assert_eq!(&encoded[52..56], &original[52..56]);
        if wave == 64 {
            // The full kernel writer's resource policy is deliberately
            // unchanged; this is an entry-field encoder, not wave64 codegen.
            assert!(descriptor(&rewritten, target).is_err());
        }
    }
}
