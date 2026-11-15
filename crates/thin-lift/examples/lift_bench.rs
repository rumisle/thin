//! Per-pass timings of the lift and thin's passes on the largest kernels
//! of a corpus: per-stage timings for user-supplied code objects.
//!
//! `cargo run --release -p thin-lift --example lift_bench -- [--top N | --every K] PATH...`
//!
//! The N largest kernels by code size (default 10), or every K-th kernel
//! in name order (a sample of typical ones), run each step once:
//! decode, unschedule, unrender, call resolution, unassign, the scheduled
//! view, register assignment and its check, the dead-write, SCC and alias
//! checks, and rendering the placed program.
//!
//! Each kernel's line carries a digest of its results (the placement or
//! its refusal, the check's verdict and the dead-write findings), and the
//! total line one of all of them: a change that should keep the results
//! the same can be compared before and after. A kernel the lift refuses
//! is counted, not timed.

#[path = "../../thin-targets/examples/corpus/mod.rs"]
mod corpus;

use std::hash::{DefaultHasher, Hash, Hasher};
use std::path::{Path, PathBuf};
use std::time::Instant;

/// The timed steps, in order.
const STEPS: [&str; 12] = [
    "decode",
    "unschedule",
    "unrender",
    "calls",
    "unassign",
    "scheduled",
    "assign",
    "check",
    "dead",
    "scc",
    "alias",
    "render",
];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut option = |name: &str| -> Result<Option<usize>, Box<dyn std::error::Error>> {
        match args.iter().position(|a| a == name) {
            Some(at) => {
                let n = args[at + 1].parse()?;
                args.drain(at..at + 2);
                Ok(Some(n))
            }
            None => Ok(None),
        }
    };
    let (top, every) = (option("--top")?.unwrap_or(10), option("--every")?);
    let mut kernels = Vec::new();
    for path in corpus::files(&args.iter().map(PathBuf::from).collect::<Vec<_>>()) {
        let bytes = corpus::image(&path, thin_targets::default_name())?;
        for k in corpus::kernels(&bytes)? {
            kernels.push((k.end - k.start, path.clone(), k.name));
        }
    }
    match every {
        Some(k) => {
            kernels.sort_by(|a, b| a.2.cmp(&b.2));
            kernels = kernels.into_iter().step_by(k.max(1)).collect();
        }
        None => {
            kernels.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.2.cmp(&b.2)));
            kernels.truncate(top);
        }
    }
    let mut total = vec![0.0f64; STEPS.len()];
    let mut all = DefaultHasher::new();
    let mut refused = 0;
    println!("bytes rows {} digest name", STEPS.join(" "));
    for (size, path, name) in &kernels {
        let short = &name[..name.len().min(60)];
        let run = match bench(path, name) {
            Ok(run) => run,
            Err(e) => {
                // A kernel the lift refuses (an unsupported form, say) is
                // counted, not timed.
                refused += 1;
                if every.is_none() {
                    println!("{size} refused {short}: {e}");
                }
                continue;
            }
        };
        run.digest.hash(&mut all);
        for (a, b) in total.iter_mut().zip(&run.times) {
            *a += b;
        }
        if every.is_some() {
            continue;
        }
        let cells: Vec<String> = run.times.iter().map(|x| format!("{x:.3}")).collect();
        println!(
            "{size} {} {} {:016x} {short}{}",
            run.rows,
            cells.join(" "),
            run.digest,
            if run.placed { "" } else { " (no room)" }
        );
    }
    let cells: Vec<String> = total.iter().map(|x| format!("{x:.3}")).collect();
    println!(
        "total {} = {:.3} s, {} kernels, {refused} refused, digest {:016x}",
        cells.join(" "),
        total.iter().sum::<f64>(),
        kernels.len(),
        all.finish()
    );
    Ok(())
}

/// One kernel's timings and results.
struct Run {
    /// Seconds per step of [`STEPS`].
    times: Vec<f64>,
    /// A hash of the placement, the check's verdict and the dead writes.
    digest: u64,
    /// The physical program's rows.
    rows: usize,
    /// Whether assignment found room.
    placed: bool,
}

/// Lifts, places and checks the kernel `name` of `path` once.
fn bench(path: &Path, name: &str) -> Result<Run, Box<dyn std::error::Error>> {
    let bytes = corpus::image(path, thin_targets::default_name())?;
    let object = thin_lift::object::read(&bytes)?;
    let k = (object.kernels.iter())
        .find(|k| k.name == name)
        .ok_or("kernel")?;
    let target = thin_targets::get(object.target)?;
    let mut times = Vec::new();
    let mut at = Instant::now();
    let mut lap = |t: &mut Vec<f64>| {
        t.push(at.elapsed().as_secs_f64());
        at = Instant::now();
    };
    let input = thin_targets::decode(object.target, &bytes[k.code.clone()])?;
    lap(&mut times);
    let (body, schedule) = thin_lift::schedule::unschedule(&input);
    lap(&mut times);
    let interface = thin_lift::interface::read_interface(
        &k.descriptor.ok_or("descriptor")?,
        k.metadata.as_ref().ok_or("metadata")?,
    )?
    .0;
    let (program, rendering) = thin_lift::program::unrender(target, &body, &schedule, &interface)?;
    lap(&mut times);
    let (program, _) = thin_lift::calls::resolve_calls(program)?;
    lap(&mut times);
    let (virt, _) = thin_lift::unassign::unassign(&program, &schedule)?;
    lap(&mut times);
    let s = virt.scheduled(&schedule)?;
    lap(&mut times);
    let rules = thin_targets::hazards(target.name())?.rules();
    let limits: Vec<(char, u32)> = ['v', 's']
        .iter()
        .map(|&f| (f, target.hooks().reg_count(f)))
        .collect();
    let o = thin_passes::regalloc::Options::default();
    let placement = thin_passes::regalloc::assign(&s, rules, &limits, o);
    lap(&mut times);
    let mut digest = DefaultHasher::new();
    format!("{:?}", placement.as_ref().map(|p| &p.bases)).hash(&mut digest);
    if let Ok(p) = &placement {
        let verdict = thin_passes::regalloc::check(&s, rules, &p.bases, o);
        format!("{verdict:?}").hash(&mut digest);
    }
    lap(&mut times);
    let p = &program.snapshot;
    let dead = thin_passes::checks::dead_write_findings(p);
    lap(&mut times);
    format!("{dead:?}").hash(&mut digest);
    let _ = thin_passes::checks::physical_scc(p);
    lap(&mut times);
    let _ = thin_passes::alias::check(p);
    lap(&mut times);
    if let Ok(pl) = &placement {
        let placed = thin_lift::unassign::place(&virt, &pl.bases)?;
        let _ = rendering.render(&placed);
    }
    lap(&mut times);
    Ok(Run {
        times,
        digest: digest.finish(),
        rows: p.instrs().len(),
        placed: placement.is_ok(),
    })
}
