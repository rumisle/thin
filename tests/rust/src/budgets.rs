//! The budgets of `tests/budgets.toml` and what a process used.
//!
//! The file is a flat subset of TOML, `[section]` and `key = number` lines,
//! read here without a TOML dependency.
//!
//! The time budgets (every key ending in `seconds`) are multiplied by
//! `THIN_BUDGET_SCALE` where a slower machine sets it, as `tests/budgets.py`
//! does (docs/dev/ci.md); memory budgets never are.

use std::collections::BTreeMap;

use crate::golden;

/// The variable a slower machine sets to scale the time budgets.
pub const SCALE_VARIABLE: &str = "THIN_BUDGET_SCALE";
/// The largest factor: past it the budgets no longer guard anything.
pub const MAX_SCALE: f64 = 4.0;

/// The factor `raw` (the variable's value, if set) says, 1 when unset; an
/// unreadable or out-of-range one is an error, never ignored.
pub fn parse_scale(raw: Option<&str>) -> Result<f64, String> {
    let Some(raw) = raw else { return Ok(1.0) };
    let scale: f64 = raw
        .trim()
        .parse()
        .map_err(|_| format!("{SCALE_VARIABLE}={raw:?}: not a number"))?;
    if (1.0..=MAX_SCALE).contains(&scale) {
        Ok(scale)
    } else {
        Err(format!("{SCALE_VARIABLE}={raw:?}: not in [1, {MAX_SCALE}]"))
    }
}

/// This process's time-budget factor. A malformed one ends the process
/// with the reason (a golden harness silences panics while it runs cases).
pub fn time_scale() -> f64 {
    let raw = std::env::var(SCALE_VARIABLE).ok();
    parse_scale(raw.as_deref()).unwrap_or_else(|e| {
        eprintln!("tests/budgets.toml: {e}");
        std::process::exit(2)
    })
}

/// One section's budgets of `tests/budgets.toml`, by key, the time ones
/// multiplied by [`time_scale`].
pub fn section(name: &str) -> BTreeMap<String, f64> {
    let scale = time_scale();
    let path = golden::repo_root().join("tests/budgets.toml");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("reading {}: {e}", path.display()));
    let mut current = String::new();
    let mut budgets = BTreeMap::new();
    for (number, raw) in text.lines().enumerate() {
        let line = raw.split('#').next().unwrap_or("").trim();
        if let Some(header) = line.strip_prefix('[').and_then(|l| l.strip_suffix(']')) {
            current = header.trim().to_owned();
        } else if !line.is_empty() && current == name {
            let parsed = line
                .split_once('=')
                .and_then(|(k, v)| Some((k.trim().to_owned(), v.trim().parse::<f64>().ok()?)));
            let (key, value) = parsed.unwrap_or_else(|| {
                panic!(
                    "{}:{}: not `key = number`: {raw:?}",
                    path.display(),
                    number + 1
                )
            });
            let factor = if key.ends_with("seconds") { scale } else { 1.0 };
            budgets.insert(key, value * factor);
        }
    }
    budgets
}

/// The peak resident set of this process in MB (VmHWM), where the
/// platform reports it without a dependency (Linux).
pub fn peak_rss_mb() -> Option<f64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let line = status.lines().find(|l| l.starts_with("VmHWM:"))?;
    let kb: f64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kb / 1024.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unset_is_one_and_a_factor_in_range_is_read() {
        assert_eq!(parse_scale(None), Ok(1.0));
        for (raw, scale) in [("1", 1.0), ("2", 2.0), ("2.5", 2.5), ("4", 4.0)] {
            assert_eq!(parse_scale(Some(raw)), Ok(scale));
        }
    }

    #[test]
    fn an_unreadable_or_out_of_range_factor_is_an_error() {
        for raw in ["", "fast", "0", "0.5", "-2", "4.5", "inf", "NaN"] {
            let err = parse_scale(Some(raw)).unwrap_err();
            assert!(err.contains(SCALE_VARIABLE), "{raw:?}: {err}");
        }
    }
}
