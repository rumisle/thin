//! Compilation phases, stream transforms and snapshot-based pass execution.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use thin_dsl::builder::{Deferred, Source};
use thin_dsl::Builder;
use thin_ir::{FieldId, FieldValue, Instr, Snapshot};
use thin_passes::hazards::{self, HazardModel};
use thin_passes::{checks, lds, waits};
use thin_target::{Kind, Pseudo};

use crate::spell::{of_dsl, Form, Refusal, Result, Spell};

/// The build options the pipeline reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Options {
    /// Emit loop bodies twice; audit the placement.
    pub audit: bool,
    /// Skip the dead-write check.
    pub keep_dead: bool,
    /// Refuse counts that are not the only fixpoint.
    pub waits_unique: bool,
    /// Every placed count 0.
    pub safe_waits: bool,
    /// Lane regions' EXEC tokens in placement.
    pub regions: bool,
    /// Values a pressure error lists.
    pub peak_list: usize,
    /// The values at the peak.
    pub peak_report: bool,
    /// Live dwords along the program.
    pub pressure: bool,
    /// One wave's program order may separate LDS regions.
    pub lds_wave_order: bool,
}

impl Default for Options {
    fn default() -> Options {
        Options {
            audit: false,
            keep_dead: false,
            waits_unique: false,
            safe_waits: false,
            regions: true,
            peak_list: 24,
            peak_report: false,
            pressure: false,
            lds_wave_order: true,
        }
    }
}

/// A file's derive report.
#[derive(Clone, Debug)]
pub struct FileReport {
    /// First fit's report.
    pub report: thin_passes::regalloc::Report,
    /// The named line of the peak's instruction.
    pub at: String,
    /// Seconds since placement started.
    pub seconds: f64,
    /// Seconds of the first run.
    pub first_run_seconds: f64,
}

/// What the pipeline records on a kernel.
#[derive(Default)]
pub struct Records {
    /// Seconds per pass, in the order first recorded.
    pub pass_times: Vec<(String, f64)>,
    /// The LDS proof's report and the ablation line.
    pub final_report: Vec<String>,
    /// The files' derive reports, in v, s, a order (those placed).
    pub reports: Vec<FileReport>,
    /// The render's proof and the length of the stream it read.
    pub proof: Option<(lds::Proof, usize)>,
    /// The first run's placement, held for the audit.
    pub pending: Option<crate::place::Pending>,
}

impl Records {
    /// Adds seconds under a name.
    pub fn time(&mut self, name: &str, seconds: f64) {
        match self.pass_times.iter_mut().find(|(n, _)| n == name) {
            Some(e) => e.1 += seconds,
            None => self.pass_times.push((name.to_owned(), seconds)),
        }
    }
}

/// The kernel's records, made at first use.
pub fn records(b: &mut Builder) -> &mut Records {
    if !b.pipeline.as_ref().is_some_and(|p| p.is::<Records>()) {
        b.pipeline = Some(Box::<Records>::default());
    }
    b.pipeline
        .as_mut()
        .and_then(|p| p.downcast_mut())
        .expect("the records")
}

/// Times `f` into the kernel's records under `name`, also when it fails.
pub fn timed<T>(
    b: &mut Builder,
    name: &str,
    f: impl FnOnce(&mut Builder) -> Result<T>,
) -> Result<T> {
    let start = Instant::now();
    let out = f(b);
    records(b).time(name, start.elapsed().as_secs_f64());
    out
}

/// A pass of a phase.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pass {
    /// The entry moves nothing reads deleted.
    DropEntry,
    /// The deferred transforms.
    Deferred,
    /// Redundant M0 writes dropped.
    DropM0,
    /// The waits pass.
    Waits,
    /// The hazard instructions placed.
    Hazards,
    /// The LDS proof.
    Proof,
    /// The save-less regions check.
    Saveless,
    /// The SCC readers check.
    Scc,
    /// The dead-write check.
    Dead,
    /// The alias check.
    Aliases,
    /// The hazards clear check.
    Clear,
}

/// FINISH.
pub const FINISH: [Pass; 3] = [Pass::DropEntry, Pass::Deferred, Pass::Waits];
/// PRE_PLACE.
pub const PRE_PLACE: [Pass; 4] = [Pass::DropEntry, Pass::Deferred, Pass::DropM0, Pass::Waits];
/// POST_PLACE.
pub const POST_PLACE: [Pass; 1] = [Pass::Hazards];
/// RENDER.
pub const RENDER: [Pass; 6] = [
    Pass::Proof,
    Pass::Saveless,
    Pass::Scc,
    Pass::Dead,
    Pass::Aliases,
    Pass::Clear,
];

impl Pass {
    /// Its timing key.
    pub fn name(self) -> &'static str {
        match self {
            Pass::DropEntry => "drop_entry",
            Pass::Deferred => "deferred",
            Pass::DropM0 => "drop_m0",
            Pass::Waits => "waits",
            Pass::Hazards => "hazards",
            Pass::Proof => "LDS proof",
            Pass::Saveless => "save-less regions",
            Pass::Scc => "SCC readers",
            Pass::Dead => "dead writes",
            Pass::Aliases => "aliases",
            Pass::Clear => "hazards clear",
        }
    }

    fn check(self) -> bool {
        !matches!(
            self,
            Pass::DropEntry | Pass::Deferred | Pass::DropM0 | Pass::Waits | Pass::Hazards
        )
    }

    fn when(self, b: &Builder, o: &Options) -> bool {
        match self {
            Pass::Deferred => !b.deferred.is_empty(),
            Pass::DropM0 => b.layout.is_some(),
            Pass::Dead => !o.keep_dead && b.ablate.is_empty(),
            Pass::Aliases => b.ablate.is_empty(),
            Pass::Clear => b.hazards.is_some(),
            _ => true,
        }
    }
}

/// The target's hazard model.
pub fn model(b: &Builder) -> &'static dyn HazardModel {
    thin_targets::hazards(b.t.name()).expect("a target's hazard model")
}

/// A snapshot of the kernel as it stands.
pub fn snapshot(b: &mut Builder) -> Result<Arc<Snapshot>> {
    b.snapshot().map(Arc::new).map_err(of_dsl)
}

/// The rows' locations, for a Diag's `Where` parts.
pub fn locs(b: &Builder) -> Vec<u32> {
    b.rows.iter().map(|r| r.loc).collect()
}

/// A pass's Diag over the kernel's current stream.
pub fn refused(b: &Builder, d: thin_ir::Diag, s: &Snapshot, f: Form) -> Refusal {
    Spell::new(b).diag(d, s.instrs(), &locs(b), f)
}

/// The pass runner over a kernel: checks between transforms
/// share one snapshot; returns the one the last checks read.
pub fn run(
    b: &mut Builder,
    passes: &[Pass],
    o: &Options,
    verify: bool,
    printable: bool,
) -> Result<Option<Arc<Snapshot>>> {
    let mut snap: Option<Arc<Snapshot>> = None;
    for &p in passes {
        if !p.when(b, o) {
            continue;
        }
        let start = Instant::now();
        let out = match p.check() {
            true => {
                if snap.is_none() {
                    snap = Some(snapshot(b)?);
                }
                check(b, p, snap.as_ref().expect("a snapshot"), o)
            }
            false => {
                snap = None;
                transform(b, p, o)
            }
        };
        records(b).time(p.name(), start.elapsed().as_secs_f64());
        out?;
    }
    if verify {
        timed(b, "verify", |b| verify_kernel(b, printable))?;
    }
    Ok(snap)
}

fn transform(b: &mut Builder, p: Pass, o: &Options) -> Result<()> {
    match p {
        Pass::DropEntry => drop_entry(b),
        Pass::Deferred => deferred(b),
        Pass::DropM0 => drop_m0(b),
        Pass::Waits => place_waits(b, o),
        _ => place_hazards(b),
    }
}

fn check(b: &mut Builder, p: Pass, s: &Arc<Snapshot>, o: &Options) -> Result<()> {
    let r = match p {
        Pass::Proof => return proof(b, s, o),
        Pass::Saveless => {
            let entries: Vec<usize> = (b.regions.iter())
                .filter(|r| r.checked)
                .filter_map(|r| b.row_index(r.entry))
                .collect();
            if entries.is_empty() {
                return Ok(());
            }
            checks::saveless(s, &entries)
        }
        Pass::Scc => checks::scc_readers(s),
        Pass::Dead => checks::dead_writes(s),
        Pass::Aliases => thin_passes::alias::check(s),
        _ => {
            // The registers are placed by now: the line names them.
            return hazards::clear(s, model(b)).map_err(|d| {
                let mut r = refused(b, d, s, Form::Bare);
                r.class = "Assertion";
                r
            });
        }
    };
    r.map_err(|d| refused(b, d, s, Form::Bare))
}

/// The LDS proof: its report, and the ablation line.
fn proof(b: &mut Builder, s: &Arc<Snapshot>, o: &Options) -> Result<()> {
    let ablate = !b.ablate.is_empty();
    let mut report = Vec::new();
    if let Some(l) = b.layout.as_mut() {
        l.freeze().map_err(of_dsl)?;
        let l = b.layout.as_ref().expect("a layout");
        let leaves: Vec<lds::Leaf> = (l.leaves().into_iter())
            .map(|r| lds::Leaf {
                name: l.regions[r].name.clone(),
                base: l.regions[r].base,
                size: l.regions[r].size,
            })
            .collect();
        let unions = (l.unions().into_iter())
            .map(|(base, end, arms)| lds::Union { base, end, arms })
            .collect();
        let layout = lds::Layout { leaves, unions };
        let opts = lds::Options {
            ablate,
            wave_order: o.lds_wave_order,
        };
        let p = lds::prove(s, &layout, opts).map_err(|d| refused(b, d, s, Form::Bare))?;
        report.clone_from(&p.report);
        records(b).proof = Some((p, s.instrs().len()));
    }
    if ablate {
        let classes: Vec<String> = b
            .ablate
            .iter()
            .map(|c| thin_ir::pyfmt::repr_str(c))
            .collect();
        report.push(format!(
            "ablate [{}]: an ablation variant - the dead-write, alias, landed, wait-names-a-load and LDS race checks off",
            classes.join(", ")
        ));
    }
    records(b).final_report = report;
    Ok(())
}

/// The IR verifier over the kernel's stream.
fn verify_kernel(b: &mut Builder, printable: bool) -> Result<()> {
    let mut values = HashMap::new();
    for r in b
        .rows
        .iter()
        .flat_map(|r| r.ins.ops().iter().filter_map(|o| o.reg()))
    {
        if let Some(v) = usize::try_from(r.value).ok().and_then(|k| b.values.get(k)) {
            let repr = format!("%{}{}:{}", v.file, r.value, v.name);
            values.insert(
                r.value,
                thin_ir::verify::ValueFacts {
                    repr,
                    n: v.n,
                    align: v.align,
                    pinned: v.pin.is_some(),
                },
            );
        }
    }
    let instrs: Vec<Arc<Instr>> = b.rows.iter().map(|r| r.ins.clone()).collect();
    match thin_ir::verify::verify(b.t, &instrs, &values, printable) {
        None => Ok(()),
        Some((k, msg)) => Err(Refusal {
            loc: b.rows[k].loc,
            ..Refusal::new("Diag", msg)
        }),
    }
}

/// Puts a pass's new stream in place: originals by identity keep their
/// rows; each placed instruction takes the location of the one it was
/// placed for.
fn put(
    b: &mut Builder,
    items: Vec<(Arc<Instr>, Option<usize>)>,
    hazard: bool,
    locs_of: &[Arc<Instr>],
) {
    let index: HashMap<*const Instr, usize> = (b.rows.iter().enumerate())
        .map(|(k, r)| (Arc::as_ptr(&r.ins), k))
        .collect();
    let stream = (items.into_iter())
        .map(|(ins, src)| {
            let source = match (index.get(&Arc::as_ptr(&ins)), src) {
                (Some(&k), None) => Source::Row(k),
                (_, Some(k)) => {
                    let loc = index
                        .get(&Arc::as_ptr(&locs_of[k]))
                        .map_or(0, |&j| b.rows[j].loc);
                    Source::New { loc, hazard }
                }
                (None, None) => Source::New {
                    loc: 0,
                    hazard: false,
                },
            };
            (ins, source)
        })
        .collect();
    b.set_stream(stream);
}

/// The waits pass.
fn place_waits(b: &mut Builder, o: &Options) -> Result<()> {
    if b.waits_placed {
        return Ok(());
    }
    let s = snapshot(b)?;
    let loops: Vec<waits::Loop> = (b.loops.iter())
        .filter(|l| !l.conservative)
        .filter_map(|l| {
            let (first, last) = (b.row_index(l.first)?, b.row_index(l.last)?);
            Some(waits::Loop {
                stem: l.stem.clone(),
                first,
                last,
                conservative: false,
            })
        })
        .collect();
    let opts = waits::Options {
        unique: o.waits_unique,
        safe: o.safe_waits,
        ablation: !b.ablate.is_empty(),
    };
    let placed = waits::pass(&s, &loops, opts).map_err(|d| refused(b, d, &s, Form::Bare))?;
    put(b, placed.stream, false, s.instrs());
    b.waits_placed = true;
    Ok(())
}

/// The hazard pass: the old hazard instructions out, the needs of
/// a fresh snapshot in.
fn place_hazards(b: &mut Builder) -> Result<()> {
    let kept: Vec<(Arc<Instr>, Source)> = (b.rows.iter().enumerate())
        .filter(|(_, r)| !r.ins.attrs().hazard)
        .map(|(k, r)| (r.ins.clone(), Source::Row(k)))
        .collect();
    b.set_stream(kept);
    let s = snapshot(b)?;
    let needs = hazards::needs(&s, model(b));
    let items = hazards::apply(&s, &needs);
    let added = items.iter().filter(|(_, src)| src.is_some()).count();
    put(b, items, true, s.instrs());
    b.hazards = Some(added);
    Ok(())
}

/// Deletes each entry move whose value no other instruction of the stream
/// reads, by any operand, a claim's included.
fn drop_entry(b: &mut Builder) -> Result<()> {
    let Some(k) = b.entered.as_ref() else {
        return Ok(());
    };
    let rows: Vec<(usize, i32)> = (k.entry_moves.iter())
        .filter_map(|(id, v)| Some((b.row_index(*id)?, v.reg().value)))
        .collect();
    let read = |row: usize, value: i32| {
        b.rows.iter().enumerate().any(|(j, r)| {
            j != row
                && r.ins
                    .ops()
                    .iter()
                    .any(|o| o.reg().is_some_and(|x| x.value == value))
        })
    };
    let gone: Vec<usize> = (rows.iter())
        .filter(|&&(row, value)| !read(row, value))
        .map(|&(row, _)| row)
        .collect();
    if gone.is_empty() {
        return Ok(());
    }
    let kept = (b.rows.iter().enumerate())
        .filter(|(j, _)| !gone.contains(j))
        .map(|(j, r)| (r.ins.clone(), Source::Row(j)))
        .collect();
    b.set_stream(kept);
    Ok(())
}

/// Deletes the redundant M0 writes.
fn drop_m0(b: &mut Builder) -> Result<()> {
    let s = snapshot(b)?;
    let gone = thin_ir::m0::redundant(&s);
    let kept = (b.rows.iter().enumerate())
        .filter(|(k, _)| !gone.contains(k))
        .map(|(k, r)| (r.ins.clone(), Source::Row(k)))
        .collect();
    b.set_stream(kept);
    Ok(())
}

/// An own op of the kernel's target with fields.
fn own(b: &Builder, p: Pseudo, fields: Vec<(&str, String)>) -> Arc<Instr> {
    let f = (fields.into_iter())
        .map(|(k, v)| {
            (
                FieldId::named(k).expect("an own op's field"),
                FieldValue::Text(v.into()),
            )
        })
        .collect();
    Arc::new(Instr::new(b.own(p), Vec::new(), f, None))
}

/// The deferred transforms, in order, then the records dropped.
fn deferred(b: &mut Builder) -> Result<()> {
    for d in std::mem::take(&mut b.deferred) {
        match d {
            Deferred::Exit { exit, upto, region } => dead_exit(b, exit, upto, region)?,
            Deferred::Peel {
                stem,
                label,
                max,
                starts,
                last,
            } => peel(b, &stem, &label, max, &starts, last)?,
        }
    }
    Ok(())
}

/// a dead exit kept when the hazard pass needs it, else a comment.
fn dead_exit(b: &mut Builder, exit: u64, upto: u64, region: Option<u32>) -> Result<()> {
    let (Some(k), Some(u)) = (b.row_index(exit), b.row_index(upto)) else {
        return Ok(());
    };
    let instrs: Arc<[Arc<Instr>]> = b.rows[..=u].iter().map(|r| r.ins.clone()).collect();
    let s = Snapshot::of_kernel(b.t, &b.gather(), instrs)
        .map_err(|e| Refusal::new("Value", e.to_string()))?;
    if hazards::gave(&s, model(b), k) {
        if let Some(r) = region.and_then(|id| b.regions.iter_mut().find(|r| r.id == id)) {
            r.checked = true;
        }
        return Ok(());
    }
    let bare = Spell::new(b).bare(&b.rows[k].ins)?;
    let c = own(
        b,
        Pseudo::Comment,
        vec![(
            "text",
            format!("; {bare}: dead (exec set again before any read)"),
        )],
    );
    let stream = (b.rows.iter().enumerate())
        .map(|(j, r)| {
            (
                if j == k { c.clone() } else { r.ins.clone() },
                Source::Row(j),
            )
        })
        .collect();
    b.set_stream(stream);
    Ok(())
}

/// The items around index k, two before to two after, as a list.
fn excerpt(xs: &[String], k: usize) -> String {
    let (a, z) = (k.saturating_sub(2).min(xs.len()), (k + 3).min(xs.len()));
    format!("[{}]", xs[a..z].join(", "))
}

/// a peeled loop keeps its copies up to the first repeat.
fn peel(
    b: &mut Builder,
    stem: &str,
    label: &str,
    max: u32,
    starts: &[Option<u64>],
    last: Option<u64>,
) -> Result<()> {
    let Some(lp) = last.and_then(|l| b.row_index(l)) else {
        return Ok(());
    };
    let mut pos: Vec<usize> = vec![lp + 1; starts.len() + 1];
    for j in (0..starts.len()).rev() {
        pos[j] = starts[j].and_then(|s| b.row_index(s)).unwrap_or(pos[j + 1]);
    }
    let s = snapshot(b)?;
    let counts = waits::Waits::new(&s)
        .solve()
        .map_err(|d| refused(b, d, &s, Form::Bare))?;
    let item = |k: usize| -> Option<String> {
        let i = &b.rows[k].ins;
        if i.is_wait() {
            let ablated = i.field_named("ablated").is_some_and(FieldValue::truthy);
            let counter = i
                .field_named("counter")
                .map(FieldValue::py_str)
                .unwrap_or_default();
            let count = counts.get(&waits::Key::Marker(k)).copied().flatten();
            let count = count.map_or("None".to_owned(), |c| c.to_string());
            return (!ablated)
                .then(|| format!("('wait', {}, {count})", thin_ir::pyfmt::repr_str(&counter)));
        }
        let kind = i.form().kind;
        (!matches!(kind, Kind::Label | Kind::Comment | Kind::Wait))
            .then(|| thin_ir::pyfmt::repr_str(&b.passed(i)))
    };
    let texts: Vec<Vec<String>> = (0..starts.len())
        .map(|j| (pos[j]..pos[j + 1]).filter_map(item).collect())
        .collect();
    let Some(fix) = (1..texts.len()).find(|&j| texts[j] == texts[j - 1]) else {
        let (a, c) = (&texts[texts.len() - 2], &texts[texts.len() - 1]);
        let k = (0..a.len().min(c.len()))
            .find(|&k| a[k] != c[k])
            .unwrap_or(a.len().min(c.len()));
        return Err(Refusal::new(
            "Schedule",
            format!(
                "loop body {stem} has no steady state after {max} peeled iterations; difference at {k}: {} vs {}",
                excerpt(a, k),
                excerpt(c, k)
            ),
        ));
    };
    let top = own(b, Pseudo::Label, vec![("name", label.to_owned())]);
    let mut stream: Vec<(Arc<Instr>, Source)> = Vec::new();
    for (j, r) in b.rows.iter().enumerate() {
        if j == pos[fix - 1] {
            stream.push((
                top.clone(),
                Source::New {
                    loc: 0,
                    hazard: false,
                },
            ));
        }
        if !(pos[fix]..=lp).contains(&j) {
            stream.push((r.ins.clone(), Source::Row(j)));
        }
    }
    b.set_stream(stream);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::excerpt;

    #[test]
    fn an_excerpt_is_the_items_around_its_index() {
        let xs: Vec<String> = (0..6).map(|k| k.to_string()).collect();
        assert_eq!(excerpt(&xs, 0), "[0, 1, 2]");
        assert_eq!(excerpt(&xs, 1), "[0, 1, 2, 3]");
        assert_eq!(excerpt(&xs, 3), "[1, 2, 3, 4, 5]");
        assert_eq!(excerpt(&xs, 6), "[4, 5]");
        assert_eq!(excerpt(&[], 0), "[]");
    }
}
