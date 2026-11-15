//! Golden checks for constant-division lowering.

use thin_test_support::golden_tests::expect_eq;
use thin_test_support::{Case, Suite};

/// `magic.json`: (m, s, short) of each divisor.
fn magic_case(case: &Case<'_>) -> Result<(), String> {
    let d = case.value["d"].as_u64().ok_or("no d")? as u32;
    let want = (
        case.value["m"].as_u64().ok_or("no m")?,
        case.value["s"].as_u64().ok_or("no s")?,
        case.value["short"].as_bool().ok_or("no short")?,
    );
    let (m, s, short) = thin_dsl::magic(d)?;
    expect_eq((m, u64::from(s), short), want)
}

fn main() {
    let mut suite = Suite::new("dsl");
    suite.section("magic.json", "cases", magic_case);
    suite.finish();
}
