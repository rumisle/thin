//! Exact golden assertions and test-resource budgets.

use std::panic::{self, AssertUnwindSafe};
use std::time::Instant;

use serde_json::Value;

use crate::{budgets, golden};

/// One case of a golden file.
pub struct Case<'a> {
    /// Where it sits in the file (`cases[3]`, `bf16[0x3f80]`).
    pub id: String,
    /// Its JSON.
    pub value: &'a Value,
    /// The whole file, for cases that read shared tables.
    pub file: &'a Value,
}

/// How a check of one case ended.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Outcome {
    /// It matched.
    Pass,
    /// An assertion failed or the tested operation panicked.
    Fail(String),
}

/// A check of one case: `Ok` when the result equals the golden expectation exactly.
pub type GoldenCheck = fn(&Case<'_>) -> Result<(), String>;

/// The result of one golden file.
struct FileResult {
    label: String,
    passed: usize,
    total: usize,
    first_failure: Option<String>,
}

/// The checks of one component.
pub struct Suite {
    component: &'static str,
    results: Vec<FileResult>,
    started: Instant,
}

impl Suite {
    /// Exact golden checks for one component.
    pub fn new(component: &'static str) -> Suite {
        // Capture assertion panics as named golden-case failures.
        panic::set_hook(Box::new(|_| {}));
        Suite {
            component,
            results: Vec::new(),
            started: Instant::now(),
        }
    }

    /// Checks every case of a golden JSON file, the cases found by [`cases`].
    pub fn file(&mut self, name: &str, check: GoldenCheck) {
        let doc = golden::json(self.component, name);
        let cases = cases(&doc);
        self.record(name, &doc, cases, check);
    }

    /// Checks the items of one list-valued key of a golden JSON file.
    pub fn section(&mut self, name: &str, key: &str, check: GoldenCheck) {
        let doc = golden::json(self.component, name);
        let items = doc[key]
            .as_array()
            .unwrap_or_else(|| panic!("{name}: no list {key:?}"));
        let cases = items
            .iter()
            .enumerate()
            .map(|(i, v)| (format!("{key}[{i}]"), v))
            .collect();
        self.record(&format!("{name} {key}"), &doc, cases, check);
    }

    /// Checks a golden file as one case (files whose parts check together).
    pub fn whole(&mut self, name: &str, check: GoldenCheck) {
        let doc = golden::json(self.component, name);
        let cases = vec![("file".to_owned(), &doc)];
        self.record(name, &doc, cases, check);
    }

    fn record(
        &mut self,
        label: &str,
        doc: &Value,
        cases: Vec<(String, &Value)>,
        check: GoldenCheck,
    ) {
        let mut passed = 0;
        let mut first_failure = None;
        let total = cases.len();
        for (id, value) in &cases {
            let case = Case {
                id: id.clone(),
                value,
                file: doc,
            };
            match run_one(check, &case) {
                Outcome::Pass => passed += 1,
                Outcome::Fail(why) => {
                    first_failure.get_or_insert_with(|| format!("{id}: {why}"));
                }
            }
        }
        self.results.push(FileResult {
            label: format!("{}/{label}", self.component),
            passed,
            total,
            first_failure,
        });
    }

    /// Prints the per-file table and exits non-zero when any file has a
    /// failing case or a budget is exceeded.
    pub fn finish(self) {
        let mut failures = 0;
        println!("golden tests: {}", self.component);
        for r in &self.results {
            let green = r.passed == r.total;
            let status = if green {
                "pass"
            } else {
                failures += 1;
                "FAIL"
            };
            println!("  {:<44} {:>6}/{:<6} {status}", r.label, r.passed, r.total);
            if !green {
                if let Some(why) = &r.first_failure {
                    println!("    first failure: {why}");
                }
            }
        }
        let green = self.results.iter().filter(|r| r.passed == r.total).count();
        println!("  {green}/{} checks green", self.results.len());
        failures += self.report_cost();
        if failures > 0 {
            std::process::exit(1);
        }
    }
}

impl Suite {
    /// Prints the binary's wall time and peak RSS against their budgets;
    /// returns how many were exceeded.
    fn report_cost(&self) -> usize {
        let budget = budgets::section("cargo");
        let seconds = self.started.elapsed().as_secs_f64();
        let rss = budgets::peak_rss_mb();
        let limit = |key: &str| budget.get(key).copied().unwrap_or(f64::INFINITY);
        let (max_seconds, max_rss) = (limit("binary_seconds"), limit("peak_rss_mb"));
        let rss_text = rss.map_or("n/a".to_owned(), |mb| format!("{mb:.0} MB"));
        let scale = budgets::time_scale();
        let scaled = if scale == 1.0 {
            String::new()
        } else {
            let written = max_seconds / scale;
            format!(
                ": {written} s scaled by {}={scale}",
                budgets::SCALE_VARIABLE
            )
        };
        println!(
            "  cost of {}: {seconds:.2} s (budget {max_seconds} s{scaled}), peak RSS {rss_text} (budget {max_rss} MB)",
            self.component
        );
        let over = [seconds > max_seconds, rss.is_some_and(|mb| mb > max_rss)];
        for (exceeded, what) in over.iter().zip(["wall time", "peak RSS"]) {
            if *exceeded {
                println!("  OVER BUDGET: {} {what}", self.component);
            }
        }
        over.iter().filter(|&&e| e).count()
    }
}

fn run_one(check: GoldenCheck, case: &Case<'_>) -> Outcome {
    match panic::catch_unwind(AssertUnwindSafe(|| check(case))) {
        Ok(Ok(())) => Outcome::Pass,
        Ok(Err(why)) => Outcome::Fail(why),
        Err(payload) => {
            let why = payload
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| payload.downcast_ref::<&str>().map(|s| (*s).to_owned()))
                .unwrap_or_else(|| "panicked".to_owned());
            Outcome::Fail(why)
        }
    }
}

/// The cases of a golden file: the items of a top-level list; else each
/// item of each list-valued key, and each entry of each object-valued key,
/// `notation` keys and scalars left out.
pub fn cases(doc: &Value) -> Vec<(String, &Value)> {
    match doc {
        Value::Array(items) => items
            .iter()
            .enumerate()
            .map(|(i, v)| (format!("[{i}]"), v))
            .collect(),
        Value::Object(map) => map
            .iter()
            .filter(|(key, _)| !key.contains("notation"))
            .flat_map(|(key, value)| match value {
                Value::Array(items) => items
                    .iter()
                    .enumerate()
                    .map(|(i, v)| (format!("{key}[{i}]"), v))
                    .collect(),
                Value::Object(entries) => entries
                    .iter()
                    .map(|(k, v)| (format!("{key}[{k}]"), v))
                    .collect(),
                _ => Vec::new(),
            })
            .collect(),
        _ => Vec::new(),
    }
}

/// `Ok` when `got` equals `want`, else a message showing both.
pub fn expect_eq<T: PartialEq + std::fmt::Debug>(got: T, want: T) -> Result<(), String> {
    if got == want {
        Ok(())
    } else {
        Err(format!("got {got:?}, want {want:?}"))
    }
}
