//! Register placement: place and the audit.

use std::sync::Arc;
use std::time::Instant;

use thin_dsl::values::Placed;
use thin_dsl::Builder;
use thin_ir::{Operand, Snapshot};
use thin_passes::regalloc::{self, Placement};
use thin_target::Kind;

use crate::pipeline::{self, records, refused, FileReport, Options, POST_PLACE, PRE_PLACE};
use crate::spell::{Form, Refusal, Result, Spell};

/// The first run's placement, held while the generator runs again.
pub struct Pending {
    placement: Placement,
    snapshot: Arc<Snapshot>,
    reports: Vec<FileReport>,
    start: Instant,
    first: Vec<Item>,
}

/// An instruction as the audit compares it: its placeholder text and its
/// registers by name (file, value name, index, n, lo), or a named one.
type Item = (String, Vec<(char, String, u32, u32, u32)>);

const FILES: [char; 3] = ['v', 's', 'a'];

/// Whether the Builder's values are placed: any of its files is.
pub fn placed(b: &Builder) -> bool {
    FILES.iter().any(|&f| b.file(f).placed)
}

fn assign_err(r: Refusal) -> Refusal {
    Refusal {
        class: "Assign",
        ..r
    }
}

fn ropts(o: &Options) -> regalloc::Options {
    regalloc::Options {
        regions: o.regions,
        peak_list: o.peak_list,
        pressure: o.pressure,
    }
}

/// Each file's placement entries: (group name, index, dwords, register).
fn entries(b: &Builder, p: &Placement) -> [Vec<Placed>; 3] {
    let of = |f: char| -> Vec<Placed> {
        (b.values.iter().enumerate())
            .filter(|(_, v)| v.file == f)
            .map(|(k, v)| {
                (
                    v.name.to_string(),
                    v.index,
                    v.n,
                    p.bases[k].unwrap_or(v.first),
                )
            })
            .collect()
    };
    [of('v'), of('s'), of('a')]
}

/// First fit of every file and its reports.
fn first_fit(b: &mut Builder, o: &Options) -> Result<(Placement, Arc<Snapshot>, Vec<FileReport>)> {
    let s = pipeline::snapshot(b)?;
    let rules = pipeline::model(b).rules();
    let limits: Vec<(char, u32)> = FILES.iter().map(|&f| (f, b.file(f).limit)).collect();
    let p = regalloc::assign(&s, rules, &limits, ropts(o))
        .map_err(|d| assign_err(refused(b, d, &s, Form::Named)))?;
    let n = s.instrs().len();
    let spell = Spell::new(b);
    let reports = (p.reports.iter())
        .map(|r| {
            let at = if n == 0 {
                String::new()
            } else {
                spell.named(&s.instrs()[r.at_index % n])
            };
            FileReport {
                report: r.clone(),
                at,
                seconds: 0.0,
                first_run_seconds: 0.0,
            }
        })
        .collect();
    Ok((p, s, reports))
}

/// Applies a placement to the files in order v, s, a; a pinned group
/// placed elsewhere is an AssignError of `F.apply`'s text.
fn apply(b: &mut Builder, p: &Placement) -> Result<()> {
    let all = entries(b, p);
    for (file, placed) in b.files.iter_mut().zip(&all) {
        file.apply(&mut b.values, placed)
            .map_err(|e| Refusal::new("Assign", e.msg))?;
    }
    Ok(())
}

/// The placement check, instructions by their named
/// lines.
fn check(b: &Builder, s: &Snapshot, p: &Placement, o: &Options) -> Result<()> {
    let rules = pipeline::model(b).rules();
    regalloc::check(s, rules, &p.bases, ropts(o))
        .map_err(|d| assign_err(refused(b, d, s, Form::Named)))
}

/// `place(kernel)` up to the audit: with `audit`, the placement
/// is returned for the generator's second run and held on the kernel;
/// else it is applied, checked and the hazards placed (none returned).
pub fn place(b: &mut Builder, o: &Options, audit: bool) -> Result<Option<[Vec<Placed>; 3]>> {
    let start = Instant::now();
    if placed(b) {
        return Err(Refusal::new(
            "Assign",
            "place: this Builder's values are placed already",
        ));
    }
    b.seal().map_err(crate::spell::of_dsl)?;
    pipeline::run(b, &PRE_PLACE, o, false, false)?;
    let fit = pipeline::timed(b, "regalloc", |b| first_fit(b, o))?;
    let (p, s, mut reports) = fit;
    let first_run = start.elapsed().as_secs_f64();
    if audit {
        let placed = entries(b, &p);
        let first = items(b);
        records(b).pending = Some(Pending {
            placement: p,
            snapshot: s,
            reports,
            start,
            first,
        });
        return Ok(Some(placed));
    }
    pipeline::timed(b, "regalloc", |b| {
        apply(b, &p).and_then(|()| check(b, &s, &p, o))
    })?;
    pipeline::run(b, &POST_PLACE, o, false, false)?;
    let seconds = start.elapsed().as_secs_f64();
    reports
        .iter_mut()
        .for_each(|r| (r.seconds, r.first_run_seconds) = (seconds, first_run));
    records(b).reports = reports;
    Ok(None)
}

/// The audit's list of a kernel.
fn items(b: &Builder) -> Vec<Item> {
    let spell = Spell::new(b);
    let reg = |o: &Operand| match o {
        Operand::Reg(r) => {
            let v = usize::try_from(r.value).ok().and_then(|k| b.values.get(k));
            Some(v.map_or((r.file, "_lit".to_owned(), 0, r.n, r.lo), |v| {
                (r.file, v.name.to_string(), v.index, r.n, r.lo)
            }))
        }
        Operand::Phys(p) => Some(('p', p.name.to_owned(), 0, 0, 0)),
        _ => None,
    };
    // The second run is compared after finish: the instructions the
    // hazard pass placed are not the generator's, and the first run has none.
    (b.rows.iter())
        .filter(|r| !matches!(r.ins.form().kind, Kind::Nop | Kind::Wait | Kind::Comment))
        .filter(|r| !r.ins.attrs().hazard)
        .map(|r| {
            (
                spell.placeholders(&r.ins),
                r.ins.ops().iter().filter_map(reg).collect(),
            )
        })
        .collect()
}

/// The audit of the first run's kernel against the second's, made
/// with the placement offered: finish the second, compare, check the
/// placement, and give the second its reports.
pub fn audit(first: &mut Builder, second: &mut Builder, o: &Options) -> Result<()> {
    let pending = records(first)
        .pending
        .take()
        .expect("a first run's placement");
    crate::compile::finish(second, o, true)?;
    let theirs = items(second);
    let ours = &pending.first;
    if let Some(k) = (0..ours.len().max(theirs.len())).find(|&k| ours.get(k) != theirs.get(k)) {
        let text = |x: Option<&Item>| x.map_or("(end)".to_owned(), |x| x.0.clone());
        return Err(Refusal::new(
            "Assign",
            format!(
                "the two runs differ at instruction {k} (waits and nops aside): `{}` vs `{}` - the generator decides on \
                 a register number or a register count",
                text(ours.get(k)),
                text(theirs.get(k))
            ),
        ));
    }
    check(first, &pending.snapshot, &pending.placement, o)?;
    let seconds = pending.start.elapsed().as_secs_f64();
    let reports = (pending.reports.into_iter())
        .map(|r| FileReport { seconds, ..r })
        .collect();
    records(second).reports = reports;
    Ok(())
}

/// Turns the hold of in-flight loads on or off, for a test's negative
/// control only; returns the previous setting.
pub fn hold_loads(on: bool) -> bool {
    thin_passes::regalloc::HOLD_LOADS.swap(on, std::sync::atomic::Ordering::Relaxed)
}
