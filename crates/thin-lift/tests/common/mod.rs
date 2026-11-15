//! Inputs and exact decoded-instruction comparisons shared by lift laws.

use thin_targets::Instruction;

pub fn kernels() -> Vec<String> {
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

pub fn instructions_equal(actual: &[Instruction], expected: &[Instruction]) {
    assert_eq!(actual.len(), expected.len());
    for (a, b) in actual.iter().zip(expected) {
        assert_eq!(a.offset, b.offset);
        assert_eq!(a.size, b.size);
        assert_eq!(a.mnemonic, b.mnemonic);
        assert_eq!(a.counted, b.counted);
        assert_eq!(a.operands(), b.operands());
        assert_eq!(a.modifiers(), b.modifiers());
        for k in 0..a.operands().len() {
            assert_eq!(a.signs(k), b.signs(k));
        }
    }
}
