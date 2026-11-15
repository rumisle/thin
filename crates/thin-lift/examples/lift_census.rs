//! Coverage counts and interface writer laws on user-supplied code objects.
//!
//! `cargo run --release -p thin-lift --example lift_census -- PATH...`

#[path = "../../thin-targets/examples/corpus/mod.rs"]
mod corpus;
mod interface_law;
mod physical_census;
mod unassign_census;

use std::collections::BTreeMap;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args: Vec<_> = std::env::args_os().skip(1).collect();
    let json = if let Some(at) = args.iter().position(|a| a == "--physical-json") {
        if at + 1 >= args.len() {
            return Err("--physical-json requires a path".into());
        }
        let path = PathBuf::from(args.remove(at + 1));
        args.remove(at);
        Some(path)
    } else {
        None
    };
    let paths = corpus::files(&args.into_iter().map(PathBuf::from).collect::<Vec<_>>());
    let (mut objects, mut kernels, mut interfaces, mut z_ids) = (0, 0, 0, 0);
    let mut refused = BTreeMap::new();
    let (mut decoded, mut undecoded, mut scheduled, mut scheduled_kernels) =
        (0u64, 0u64, 0u64, 0u64);
    let mut schedule_ops = BTreeMap::new();
    let (mut complete_kernels, mut branch_rewrites, mut rewritten_kernels) = (0u64, 0u64, 0u64);
    let (mut getpc_count, mut getpc_kernels, mut long_jumps) = (0u64, 0u64, 0u64);
    let mut branch_outcomes = BTreeMap::new();
    let mut branch_examples = BTreeMap::new();
    let mut branch_forms = BTreeMap::new();
    let mut different_forms = BTreeMap::new();
    let mut interface_law = interface_law::Law::default();
    let mut physical = physical_census::Census::default();
    for path in paths {
        let bytes = corpus::image(&path, thin_targets::default_name())?;
        let object = thin_lift::object::read(&bytes)?;
        let ranges = corpus::kernels(&bytes)?;
        objects += 1;
        for range in ranges {
            let k = object
                .kernels
                .iter()
                .find(|k| {
                    k.code.start == range.start && k.code.end == range.end && k.name == range.name
                })
                .ok_or("object reader disagrees with decoder corpus walker")?;
            kernels += 1;
            let decoder = thin_targets::decoder(object.target)?;
            let code = &bytes[k.code.clone()];
            let mut at = 0;
            let mut has_schedule = false;
            let mut complete = true;
            let mut instructions = Vec::new();
            while at < code.len() {
                match decoder.decode_at(code, at) {
                    Ok(i) => {
                        decoded += 1;
                        instructions.push(i);
                        if thin_lift::schedule::is_scheduling(&i) {
                            scheduled += 1;
                            has_schedule = true;
                            *schedule_ops.entry(i.mnemonic).or_insert(0u64) += 1;
                        }
                        at += usize::from(i.size);
                    }
                    Err(e) => {
                        undecoded += 1;
                        complete = false;
                        at += e.size.map_or(4, usize::from);
                    }
                }
            }
            scheduled_kernels += u64::from(has_schedule);
            // Do not recognize sequences across decoder refusals. Counts
            // for branch rewriting use only completely decoded kernels.
            if complete {
                complete_kernels += 1;
                let (body, schedule) = thin_lift::schedule::unschedule(&instructions);
                let analysis = thin_lift::branches::inspect_branches(&body, Some(&schedule));
                getpc_count += analysis.len() as u64;
                getpc_kernels += u64::from(!analysis.is_empty());
                for a in analysis {
                    if a.outcome == thin_lift::branches::BranchOutcome::DifferentForm {
                        let end = (a.position + 5).min(body.len());
                        let shape = body[a.position..end]
                            .iter()
                            .map(|i| i.mnemonic)
                            .collect::<Vec<_>>()
                            .join(" / ");
                        *different_forms.entry(shape).or_insert(0u64) += 1;
                    }
                    let is_jump = a.sequence_len != 0
                        && a.outcome != thin_lift::branches::BranchOutcome::PcSetup;
                    long_jumps += u64::from(is_jump);
                    if is_jump {
                        *branch_forms.entry(a.sequence_len).or_insert(0u64) += 1;
                    }
                    *branch_outcomes.entry(a.outcome).or_insert(0u64) += 1;
                    branch_examples.entry(a.outcome).or_insert_with(|| {
                        let end = (a.position + 6).min(body.len());
                        format!(
                            "{}:{}+{:#x}: {}",
                            path.display(),
                            range.name,
                            body[a.position].offset,
                            body[a.position..end]
                                .iter()
                                .map(ToString::to_string)
                                .collect::<Vec<_>>()
                                .join("; ")
                        )
                    });
                }
                let (_, records) = thin_lift::branches::unrelax_branches(&body, Some(&schedule));
                branch_rewrites += records.entries.len() as u64;
                rewritten_kernels += u64::from(!records.entries.is_empty());
            }
            match (k.descriptor, k.metadata.as_ref()) {
                (Some(d), Some(m)) => match thin_lift::interface::read_interface(&d, m) {
                    Ok((interface, _)) => {
                        interfaces += 1;
                        interface_law.check(
                            &interface,
                            &d,
                            thin_targets::get(object.target)?,
                            &format!("{}:{}", path.display(), k.name),
                        );
                        z_ids += usize::from(interface.workgroup_ids[2]);
                        if complete {
                            let (body, schedule) = thin_lift::schedule::unschedule(&instructions);
                            physical.check(
                                thin_targets::get(object.target)?,
                                &body,
                                &schedule,
                                &interface,
                                &format!("{}:{}", path.display(), k.name),
                            );
                        }
                    }
                    Err(e) => *refused.entry(e.to_string()).or_insert(0usize) += 1,
                },
                _ => {
                    *refused
                        .entry("missing descriptor or metadata".into())
                        .or_insert(0) += 1
                }
            }
        }
    }
    println!("objects {objects}; kernels {kernels}; interfaces {interfaces}; z enabled {z_ids}");
    println!("decoded {decoded}; undecoded {undecoded}; scheduled {scheduled}; kernels with schedules {scheduled_kernels}");
    println!("completely decoded kernels {complete_kernels}; branch rewrites {branch_rewrites}; kernels with rewrites {rewritten_kernels}");
    println!("getpc {getpc_count}; kernels with getpc {getpc_kernels}; recognized long jumps {long_jumps}");
    for (len, n) in branch_forms {
        println!("long-jump form {len} instructions: {n}");
    }
    for (outcome, n) in branch_outcomes {
        println!("branch {}: {n}", outcome.name());
        println!("  example: {}", branch_examples[&outcome]);
    }
    for (form, n) in different_forms {
        println!("different form {n}: {form}");
    }
    for (op, n) in schedule_ops {
        println!("schedule {op}: {n}");
    }
    for (why, n) in refused {
        println!("interface refusal {n}: {why}");
    }
    physical.report();
    if let Some(path) = json {
        let mut report = physical.json();
        report["corpus"] = serde_json::json!({
            "objects":objects, "kernels":kernels, "interfaces":interfaces,
            "z_enabled":z_ids, "decoded":decoded, "undecoded":undecoded,
            "scheduled":scheduled, "scheduled_kernels":scheduled_kernels,
            "complete_kernels":complete_kernels, "branch_rewrites":branch_rewrites,
            "rewritten_kernels":rewritten_kernels, "getpc":getpc_count,
            "getpc_kernels":getpc_kernels, "long_jumps":long_jumps,
        });
        std::fs::write(path, serde_json::to_string_pretty(&report)?)?;
    }
    interface_law.report();
    if !interface_law.passes() {
        return Err("interface writer law failed".into());
    }
    Ok(())
}
