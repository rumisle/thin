//! Physical-program coverage, existing checks and refusal examples.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use thin_ir::verify::{verify, ValueFacts};
use thin_ir::{Diag, Part};
use thin_lift::interface::Interface;
use thin_lift::program::{unrender, Program};
use thin_lift::schedule::Schedule;
use thin_passes::checks::{self, SccReader};
use thin_target::Target;
use thin_targets::Instruction;

#[derive(Default)]
pub struct Census {
    attempted: u64,
    converted: u64,
    checks: BTreeMap<&'static str, u64>,
    failures: BTreeMap<String, (u64, u64, u64)>,
    examples: BTreeMap<String, String>,
    indirect: u64,
    resolved_programs: u64,
    unresolved_sites: u64,
    transfers: BTreeMap<String, u64>,
    call_examples: BTreeMap<String, String>,
    before_dead: u64,
    after_dead: u64,
    dead_sites: BTreeMap<&'static str, u64>,
    dead_classes: BTreeMap<String, u64>,
    dead_examples: BTreeMap<String, String>,
    dead_word_classes: BTreeMap<String, u64>,
    edge_set_sizes: BTreeMap<&'static str, BTreeMap<usize, u64>>,
    unassign: crate::unassign_census::Census,
}

impl Census {
    fn failure(&mut self, class: String, example: String) {
        let count = self.failures.entry(class.clone()).or_default();
        count.0 += 1;
        if count.2 != self.attempted {
            count.1 += 1;
            count.2 = self.attempted;
        }
        self.examples.entry(class).or_insert(example);
    }

    fn diagnostic(
        &mut self,
        check: &'static str,
        program: &Program,
        result: Result<(), Diag>,
        context: &str,
    ) {
        match result {
            Ok(()) => *self.checks.entry(check).or_default() += 1,
            Err(d) => {
                let opcode = d
                    .msg
                    .iter()
                    .find_map(|p| {
                        if let Part::Instr(k) = p {
                            Some(program.snapshot.instrs()[*k].form().op.as_str())
                        } else {
                            None
                        }
                    })
                    .unwrap_or("unknown");
                let class = format!("{check}: {opcode}");
                let line = |k: usize| {
                    format!(
                        "{}+{:#x} {}",
                        context,
                        program.addresses[k],
                        program.line(k).expect("physical line")
                    )
                };
                self.failure(class, Diag::spell(&d.msg, &line, None));
            }
        }
    }

    pub fn check(
        &mut self,
        target: &'static Target,
        instructions: &[Instruction],
        schedule: &Schedule,
        interface: &Interface,
        context: &str,
    ) {
        self.attempted += 1;
        let (program, rendering) = match unrender(target, instructions, schedule, interface) {
            Ok(p) => p,
            Err(e) => {
                self.failure(
                    format!("conversion {}: {}", e.class, e.detail),
                    format!("{context}+{:?}: {e}", e.offset),
                );
                return;
            }
        };
        self.converted += 1;
        self.indirect += u64::from(!program.indirect.is_empty());
        let before = checks::dead_writes_all(&program.snapshot);
        let row = |d: &Diag| {
            d.msg
                .iter()
                .find_map(|p| {
                    if let Part::Instr(k) = p {
                        Some(*k)
                    } else {
                        None
                    }
                })
                .expect("dead-write row")
        };
        let before_rows: BTreeSet<_> = before.iter().map(row).collect();
        self.before_dead += u64::from(!before.is_empty());
        let (program, calls) = match thin_lift::calls::resolve_calls(program) {
            Ok(result) => result,
            Err(error) => {
                self.failure(format!("call graph: {error}"), context.to_owned());
                return;
            }
        };
        self.resolved_programs += u64::from(calls.complete());
        for site in &calls.sites {
            self.unresolved_sites += u64::from(site.unresolved.is_some());
            let kind = match site.kind {
                thin_ir::control::TransferKind::Call { .. }
                | thin_ir::control::TransferKind::UnresolvedCall { .. } => "call",
                thin_ir::control::TransferKind::Return { .. } => "return",
                thin_ir::control::TransferKind::Jump => "jump",
            };
            let outcome = site.unresolved.map_or("resolved", |r| r.name());
            let class = format!("{kind}: {outcome}");
            *self.transfers.entry(class.clone()).or_default() += 1;
            *self
                .edge_set_sizes
                .entry(kind)
                .or_default()
                .entry(site.edges.len())
                .or_default() += 1;
            self.call_examples.entry(class).or_insert_with(|| {
                format!(
                    "{context}+{:#x}: {}; {} edges; first destinations {:?}",
                    program.addresses[site.instruction],
                    program.line(site.instruction).expect("line"),
                    site.edges.len(),
                    site.edges
                        .iter()
                        .take(4)
                        .map(|e| match e.target {
                            thin_ir::control::Destination::Instruction(k) =>
                                format!("+{:#x} {:?}", program.addresses[k], e.kind),
                            thin_ir::control::Destination::Exit => "exit".into(),
                        })
                        .collect::<Vec<_>>()
                )
            });
        }
        self.unassign
            .check(&program, &rendering, schedule, instructions, context);
        let s = &program.snapshot;
        let precise = checks::dead_write_findings(s);
        for finding in &precise {
            let opcode = s.instrs()[finding.instruction].form().op.as_str();
            let extent = if finding.live_dwords.is_empty() {
                "all vector outputs dead"
            } else {
                "partial outputs dead"
            };
            *self
                .dead_word_classes
                .entry(format!("{extent}: {opcode}"))
                .or_default() += 1;
            *self
                .dead_word_classes
                .entry(format!("dead dwords: {opcode}"))
                .or_default() += finding.dead_dwords.len() as u64;
        }
        let after: Vec<_> = precise
            .into_iter()
            .map(|finding| finding.diagnostic)
            .collect();
        let after_rows: BTreeSet<_> = after.iter().map(row).collect();
        self.after_dead += u64::from(!after.is_empty());
        let complete = calls.complete();
        for (name, rows) in [
            ("before", before_rows.clone()),
            ("after", after_rows.clone()),
            (
                "removed",
                before_rows.difference(&after_rows).copied().collect(),
            ),
            (
                "persistent",
                before_rows.intersection(&after_rows).copied().collect(),
            ),
            (
                "introduced",
                after_rows.difference(&before_rows).copied().collect(),
            ),
            (
                if complete {
                    "remaining on complete CFG"
                } else {
                    "remaining with unresolved CFG"
                },
                after_rows.clone(),
            ),
        ] {
            *self.dead_sites.entry(name).or_default() += rows.len() as u64;
            for k in rows {
                let class = format!("{name}: {}", s.instrs()[k].form().op);
                *self.dead_classes.entry(class.clone()).or_default() += 1;
                self.dead_examples.entry(class).or_insert_with(|| {
                    format!(
                        "{context}+{:#x}: {}",
                        program.addresses[k],
                        program.line(k).expect("physical line")
                    )
                });
            }
        }
        if let Some(first) = before.first().map(row) {
            *self
                .dead_sites
                .entry("original first persisted")
                .or_default() += u64::from(after_rows.contains(&first));
            *self.dead_sites.entry("original first removed").or_default() +=
                u64::from(!after_rows.contains(&first));
        }
        let values: HashMap<_, _> = s
            .values()
            .iter()
            .enumerate()
            .map(|(k, v)| {
                (
                    k as i32,
                    ValueFacts {
                        repr: format!("physical {}", v.file),
                        n: v.n,
                        align: v.align,
                        pinned: v.pin.is_some(),
                    },
                )
            })
            .collect();
        match verify(target, s.instrs(), &values, true) {
            None => *self.checks.entry("verify").or_default() += 1,
            Some((k, msg)) => self.failure(
                format!("verify: {msg}"),
                format!("{context}+{:#x}: {msg}", program.addresses[k]),
            ),
        }
        let findings = checks::scc_findings(s);
        for (name, kind) in [
            ("SCC selects", SccReader::Select),
            ("carry chains", SccReader::Carry),
        ] {
            let mut passed = true;
            for finding in findings.iter().filter(|f| f.kind == kind) {
                passed = false;
                let k = finding.instruction;
                let class = format!(
                    "{name}: {} after {}",
                    s.instrs()[k].form().op,
                    finding.writer
                );
                let line = |k: usize| {
                    format!(
                        "{}+{:#x} {}",
                        context,
                        program.addresses[k],
                        program.line(k).expect("physical line")
                    )
                };
                let mut example = Diag::spell(&finding.diagnostic.msg, &line, None);
                if let Some(writer) = (0..k)
                    .rev()
                    .find(|&j| s.instrs()[j].form().op == finding.writer)
                {
                    example.push_str(&format!(
                        "; nearby writer +{:#x}: {}",
                        program.addresses[writer],
                        program.line(writer).expect("physical line")
                    ));
                }
                self.failure(class, example);
            }
            if passed {
                *self.checks.entry(name).or_default() += 1;
            }
        }
        let physical_scc = checks::physical_scc(s);
        if physical_scc.is_empty() {
            *self.checks.entry("physical SCC definedness").or_default() += 1;
        }
        for diagnostic in physical_scc {
            self.diagnostic(
                "physical SCC definedness",
                &program,
                Err(diagnostic),
                context,
            );
        }
        self.diagnostic(
            "dead writes",
            &program,
            after.into_iter().next().map_or(Ok(()), Err),
            context,
        );
        self.diagnostic("alias", &program, thin_passes::alias::check(s), context);
    }

    pub fn json(&self) -> serde_json::Value {
        let (unassign, unassign_examples) = self.unassign.json();
        serde_json::json!({"unassign": unassign, "unassign_examples": unassign_examples,"attempted": self.attempted, "converted": self.converted,
            "programs_with_indirect": self.indirect, "checks": self.checks,
            "resolved_programs":self.resolved_programs,"unresolved_sites":self.unresolved_sites,"transfers":self.transfers,"transfer_examples":self.call_examples,
            "dead_kernels_before":self.before_dead,"dead_kernels_after":self.after_dead,
            "dead_sites":self.dead_sites,"dead_classes":self.dead_classes,"dead_examples":self.dead_examples,
            "edge_set_sizes":self.edge_set_sizes,"dead_word_classes":self.dead_word_classes,
            "refusals": self.failures.iter().map(|(class,(sites,kernels,_))| serde_json::json!({
                "class":class,"sites":sites,"kernels":kernels,"example":self.examples[class]})).collect::<Vec<_>>()})
    }

    pub fn report(&self) {
        println!(
            "physical Programs {}; with physical indirect transfers {}",
            self.converted, self.indirect
        );
        println!(
            "call graph complete Programs {}; before dead kernels {}; after dead kernels {}",
            self.resolved_programs, self.before_dead, self.after_dead
        );
        for (class, n) in &self.transfers {
            println!("call graph {class}: {n}");
            println!("  example: {}", self.call_examples[class]);
        }
        for (class, n) in &self.dead_sites {
            println!("dead-write sites {class}: {n}");
        }
        for (check, n) in &self.checks {
            println!("physical check {check}: passed {n}");
        }
        for (class, (sites, kernels, _)) in &self.failures {
            println!("physical refusal sites {sites}; kernels {kernels}: {class}");
            println!("  example: {}", self.examples[class]);
        }
        self.unassign.report();
    }
}
