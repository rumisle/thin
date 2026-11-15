//! Lane regions and wave blocks.

use std::collections::BTreeSet;

use thin_ir::{Attr, Instr, Operand, Reg};
use thin_target::{phys_named, Flags, Kind};

use crate::builder::{Builder, Deferred, RegionRec};
use crate::error::{schedule, value, Result};
use crate::expr::cond::{Into, Where};
use crate::expr::lower::Opd;
use crate::expr::node::Tree;
use crate::maker::Arg;
use crate::types::Ty;
use crate::values::Typed;

/// A region's condition as the author gives it.
#[derive(Clone, Debug)]
pub enum LaneCond {
    /// None.
    None,
    /// A tree.
    Tree(Tree),
    /// An instruction.
    Instr(Box<Instr>),
    /// An int.
    Int(i128),
    /// A string.
    Str(String),
    /// A mask value.
    Mask(Typed),
}

/// `save=`.
#[derive(Clone, Debug)]
pub enum Save {
    /// None or False.
    None,
    /// True: a fresh mask.
    Fresh,
    /// A register.
    Reg(Reg, String),
}

/// A region's mask.
#[derive(Clone, Debug, PartialEq)]
enum Mask {
    Vcc,
    Scc,
    Reg(Opd),
    Int(i128),
}

/// An open region.
#[derive(Clone, Debug)]
struct Region {
    id: u32,
    /// A wave block's text; a lane region's is its masks and save.
    lane: bool,
    text: String,
    masks: Vec<String>,
    save: Option<Reg>,
    n0: usize,
    skip: Option<String>,
    entry_ones: (bool, bool),
}

/// The lane regions' state.
#[derive(Clone, Debug)]
pub struct Exec {
    open: Vec<Region>,
    next: u32,
    raw: Vec<String>,
    last_exit: Option<(u64, bool)>,
    ones: (bool, bool),
    own: bool,
}

impl Default for Exec {
    fn default() -> Exec {
        Exec {
            open: Vec::new(),
            next: 0,
            raw: Vec::new(),
            last_exit: None,
            ones: (true, true),
            own: false,
        }
    }
}

impl Exec {
    fn take_id(&mut self) -> u32 {
        self.next += 1;
        self.next
    }

    fn innermost_lane(&mut self) -> Option<&mut Region> {
        self.open.iter_mut().rev().find(|r| r.lane)
    }
}

fn region_text(b: &Builder, r: &Region) -> String {
    if !r.lane {
        return r.text.clone();
    }
    let text = r.masks.join(" & ");
    match &r.save {
        Some(sv) => format!("{text}, saved in {}", b.repr(sv)),
        None => text,
    }
}

fn mask_text(b: &Builder, m: &Mask) -> String {
    match m {
        Mask::Vcc => "vcc".into(),
        Mask::Scc => "scc".into(),
        Mask::Int(v) => v.to_string(),
        Mask::Reg(Opd::Reg(v, _)) => b.repr(&v.reg()),
        Mask::Reg(Opd::Phys(p)) => p.name.to_owned(),
        Mask::Reg(_) => "?".into(),
    }
}

impl Builder {
    /// Where code is being emitted: the open arms' identities and the open
    /// lane regions', innermost last. A value made here is written only on
    /// the paths through all of them.
    pub fn scope(&self) -> (Vec<u64>, Vec<u32>) {
        let regions = self.exec.open.iter().filter(|r| r.lane).map(|r| r.id);
        (self.arm_ids(), regions.collect())
    }

    /// Whether a lane region or wave block is open.
    pub fn exec_open(&self) -> bool {
        !self.exec.open.is_empty()
    }

    fn exec_phys(&self) -> &'static thin_target::Phys {
        self.t.mask_reg(phys_named("exec").expect("exec"))
    }

    fn mask_opd(&self, m: &Mask) -> Opd {
        match m {
            Mask::Vcc => Opd::Phys(self.t.mask_reg(phys_named("vcc").expect("vcc"))),
            Mask::Scc => Opd::Phys(phys_named("scc").expect("scc")),
            Mask::Reg(o) => o.clone(),
            Mask::Int(v) => Opd::Int(*v),
        }
    }

    /// Emits an instruction as the region's own entry or exit (not checked
    /// as inside it).
    fn own_emit(&mut self, i: Instr) -> Result<Option<u64>> {
        self.exec.own = true;
        let r = self.emit(i, &[]);
        self.exec.own = false;
        r
    }

    fn own_op(&mut self, op: &str, args: Vec<Opd>, comment: &mut Option<String>) -> Result<u64> {
        let name = self.spelled(op)?;
        let args = args.iter().map(Opd::arg).collect();
        let i = self.make_plain(&name, args, comment.take().as_deref().unwrap_or(""))?;
        Ok(self.own_emit(i)?.expect("an entry row"))
    }

    /// Opens `b.lanes(*conds, save=, skip=, comment=)`; returns
    /// whether a region opened. `display` names the conditions as written.
    pub fn lanes_begin(
        &mut self,
        conds: Vec<LaneCond>,
        display: &[String],
        save: Save,
        skip: bool,
        comment: &str,
    ) -> Result<bool> {
        // No condition, or a constant holding every lane of the wave, is no
        // region.
        let full = i128::from(self.t.full_mask());
        let keep: Vec<(LaneCond, &String)> = (conds.into_iter().zip(display))
            .filter(|(c, _)| {
                !matches!(c, LaneCond::None) && !matches!(c, LaneCond::Int(v) if v & full == full)
            })
            .collect();
        if keep.is_empty() {
            return Ok(false);
        }
        let shown = format!(
            "lanes[{}]",
            keep.iter()
                .map(|k| k.1.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        );
        let own = keep.len() == 1 && matches!(&keep[0].0, LaneCond::Instr(i) if i.writes_exec());
        let lane_open = self.exec.open.iter().any(|r| r.lane);
        let mut save = save;
        if matches!(save, Save::None) && lane_open {
            if own {
                let id = self
                    .exec
                    .open
                    .iter()
                    .rev()
                    .find(|r| r.lane)
                    .expect("a lane region")
                    .id;
                return schedule(format!(
                    "{shown} inside the lane region {id} without a save register: pass save= (the entry exec is \
                     restored from it)"
                ));
            }
            save = Save::Fresh;
        }
        let save = match save {
            Save::None => None,
            Save::Fresh => {
                let m = crate::builder::Make {
                    n: Some(1),
                    n_repr: "1".into(),
                    name: Some("save".into()),
                    where_: self.where_text.clone(),
                    ..Default::default()
                };
                Some(self.make_value('s', Ty::Mask, m)?.reg())
            }
            Save::Reg(r, repr) => {
                let w = self.t.definition().wave / 32;
                if r.file != 's' || r.n != w {
                    return schedule(format!(
                        "lanes(save={repr}): the entry exec is saved in a lane mask's SGPRs (width {w}, as s_and_saveexec \
                         and s_mov of exec take them)"
                    ));
                }
                Some(r)
            }
        };
        if own && save.is_some() {
            let LaneCond::Instr(i) = &keep[0].0 else {
                unreachable!()
            };
            return schedule(format!(
                "`{}` as a region's entry: save-less only (it writes exec itself)",
                self.text(i)
            ));
        }
        let masks = self.masks(keep.iter().map(|k| k.0.clone()).collect(), own)?;
        if masks.contains(&Mask::Scc) && (save.is_some() || masks.len() > 2) {
            return schedule(format!(
                "{shown}: a wave-uniform condition (SCC) after one mask at most, without a save register"
            ));
        }
        let id = self.exec.take_id();
        let entry_ones = self.exec.ones;
        let mut comment = (!comment.is_empty()).then(|| comment.to_owned());
        let before = if save.is_none() {
            self.exec.last_exit.take()
        } else {
            None
        };
        let first = match (&save, own) {
            (_, true) => {
                let LaneCond::Instr(i) = keep.into_iter().next().expect("one").0 else {
                    unreachable!()
                };
                self.own_emit(*i)?.expect("an entry row")
            }
            (Some(sv), _) => self.saved_entry(*sv, &masks, &mut comment)?,
            (None, _) => self.plain_entry(&masks, &mut comment)?,
        };
        let mut checked = save.is_none();
        if save.is_none() {
            if let Some((exit, true)) = before {
                if self.dead_exit(exit, first, false) {
                    self.deferred.push(Deferred::Exit {
                        exit,
                        upto: self.rows.last().expect("a row").id,
                        region: Some(id),
                    });
                    checked = false;
                }
            }
        }
        self.regions.push(RegionRec {
            id,
            entry: first,
            save,
            checked,
        });
        let texts: Vec<String> = masks.iter().map(|m| mask_text(self, m)).collect();
        let region = Region {
            id,
            lane: true,
            text: String::new(),
            masks: texts,
            save,
            n0: self.rows.len(),
            skip: None,
            entry_ones,
        };
        self.exec.open.push(region);
        if skip {
            let over = self.new_label("lanes_none");
            let i = self.make_plain("s.cbranch.execz", vec![Arg::Str(over.clone())], "")?;
            self.emit(i, &[])?;
            self.exec.open.last_mut().expect("the region").skip = Some(over);
        }
        Ok(true)
    }

    fn masks(&mut self, conds: Vec<LaneCond>, own: bool) -> Result<Vec<Mask>> {
        if own {
            return Ok(Vec::new());
        }
        let n = conds.len();
        let mut out = Vec::new();
        for (k, c) in conds.into_iter().enumerate() {
            let last = k + 1 == n;
            out.push(self.mask_of(c, last)?);
        }
        Ok(out)
    }

    fn mask_of(&mut self, c: LaneCond, last: bool) -> Result<Mask> {
        Ok(match c {
            LaneCond::Instr(i) => {
                let defs: Vec<&str> = i.writes().iter().map(|&k| thin_target::PHYS[k].name).collect();
                let vcc = defs.iter().any(|d| d.starts_with("vcc"));
                let scc = defs.contains(&"scc");
                if !(vcc || scc) || !last {
                    return schedule(format!(
                        "`{}` as a region's condition: a compare into vcc or SCC, the last condition",
                        self.text(&i)
                    ));
                }
                self.emit(*i, &[])?;
                if vcc {
                    Mask::Vcc
                } else {
                    Mask::Scc
                }
            }
            LaneCond::Tree(n) => match self.asm_cond(&n, None, if last { Into::Lanes } else { Into::Mask })? {
                Where::Scc => Mask::Scc,
                Where::Vcc => Mask::Vcc,
                Where::Mask(v) => Mask::Reg(Opd::Reg(v, 0)),
                Where::Phys(p) => Mask::Reg(Opd::Phys(self.t.mask_reg(p))),
            },
            LaneCond::Int(v) => Mask::Int(v),
            LaneCond::Str(s) => {
                return schedule(format!(
                    "lanes({}): write the condition (an expr compare, a compare instruction), not the register it \
                     is left in",
                    crate::maker::py_repr(&s)
                ))
            }
            LaneCond::Mask(v) => Mask::Reg(Opd::Reg(v, 0)),
            LaneCond::None => unreachable!("dropped"),
        })
    }

    fn saved_entry(
        &mut self,
        sv: Reg,
        masks: &[Mask],
        comment: &mut Option<String>,
    ) -> Result<u64> {
        let save = self.typed_reg(sv);
        let first = self.own_op(
            "s.and_saveexec.mask",
            vec![save, self.mask_opd(&masks[0])],
            comment,
        )?;
        let exec = Opd::Phys(self.exec_phys());
        for m in &masks[1..] {
            self.own_op(
                "s.and.mask",
                vec![exec.clone(), exec.clone(), self.mask_opd(m)],
                comment,
            )?;
        }
        Ok(first)
    }

    fn plain_entry(&mut self, masks: &[Mask], comment: &mut Option<String>) -> Result<u64> {
        let exec = Opd::Phys(self.exec_phys());
        if masks.last() == Some(&Mask::Scc) {
            let a = if masks.len() == 2 {
                self.mask_opd(&masks[0])
            } else {
                Opd::Int(-1)
            };
            return self.own_op("s.cselect.mask", vec![exec, a, Opd::Int(0)], comment);
        }
        if let [Mask::Int(v)] = masks {
            let wide = self.t.definition().wave == 64;
            let fits = (-16..=64).contains(v) || (0..1 << 31).contains(v) || *v == -1;
            if wide && !fits {
                let lo = Opd::Int(i128::from(*v as i32));
                let hi = Opd::Int(i128::from((*v >> 32) as i32));
                let first = self.own_op(
                    "s.mov.b32",
                    vec![Opd::Phys(phys_named("exec_lo").expect("exec_lo")), lo],
                    comment,
                )?;
                self.own_op(
                    "s.mov.b32",
                    vec![Opd::Phys(phys_named("exec_hi").expect("exec_hi")), hi],
                    comment,
                )?;
                return Ok(first);
            }
        }
        if masks.len() == 1 {
            return self.own_op("s.mov.mask", vec![exec, self.mask_opd(&masks[0])], comment);
        }
        let first = self.own_op(
            "s.and.mask",
            vec![
                exec.clone(),
                self.mask_opd(&masks[0]),
                self.mask_opd(&masks[1]),
            ],
            comment,
        )?;
        for m in &masks[2..] {
            self.own_op(
                "s.and.mask",
                vec![exec.clone(), exec.clone(), self.mask_opd(m)],
                comment,
            )?;
        }
        Ok(first)
    }

    /// A register as an operand, with its value's type.
    fn typed_reg(&self, r: Reg) -> Opd {
        let v = &self.values[r.value as usize];
        let t = crate::values::Typed {
            builder: self.id,
            file: r.file,
            value: r.value as usize,
            group: v.name.clone(),
            ty: Ty::mask_of(self.t.definition().wave),
            n: 1,
            sub: r.lo,
            label: None,
            where_: "".into(),
            part: "".into(),
            divisor: false,
        };
        Opd::Reg(t, 0)
    }

    /// Closes the innermost region; `error` when its
    /// block raised.
    pub fn lanes_end(&mut self, error: bool) -> Result<()> {
        if error {
            self.exec.open.pop();
            return Ok(());
        }
        let skip = self.exec.open.last().and_then(|r| r.skip.clone());
        if let Some(over) = skip {
            self.put_label(&over, false)?;
        }
        self.settle()?;
        let r = self.exec.open.pop().expect("an open region");
        self.check_branches_out(&r)?;
        let exec = Opd::Phys(self.exec_phys());
        let src = match &r.save {
            Some(sv) => self.typed_reg(*sv),
            None => Opd::Int(-1),
        };
        let id = self.own_op("s.mov.mask", vec![exec, src], &mut None)?;
        self.exec.last_exit = Some((id, r.save.is_none()));
        self.exec.ones = r.entry_ones;
        Ok(())
    }

    /// a branch out of the region.
    fn check_branches_out(&self, r: &Region) -> Result<()> {
        let inside: BTreeSet<String> = (self.rows[r.n0.min(self.rows.len())..].iter())
            .filter(|row| row.ins.form().kind == Kind::Label)
            .filter_map(|row| label_name(&row.ins))
            .collect();
        for row in &self.rows[r.n0.min(self.rows.len())..] {
            let i = &row.ins;
            if i.form().kind != Kind::Branch {
                continue;
            }
            let Some(t) = i.ops().iter().find_map(|o| {
                if let Operand::Block(b) = o {
                    Some(b.to_string())
                } else {
                    None
                }
            }) else {
                continue;
            };
            if !inside.contains(&t) {
                let kind = if r.lane { "lanes" } else { "waves" };
                return schedule(format!(
                    "`{}` leaves the {kind} region {} ({}) to {t}, outside it: exec there would be the region's (end \
                     the region first)",
                    self.text(i),
                    r.id,
                    region_text(self, r)
                ));
            }
        }
        Ok(())
    }

    /// `b.narrow(cond, comment)`.
    pub fn narrow(&mut self, cond: LaneCond, display: &str, comment: &str) -> Result<()> {
        if !self.exec.open.iter().any(|r| r.lane) {
            return schedule(format!("narrow({display}) outside a lane region"));
        }
        let m = self.mask_of(cond, true)?;
        if m == Mask::Scc {
            return schedule("narrow: a vector condition or a mask (SCC is no lane mask)");
        }
        let exec = Opd::Phys(self.exec_phys());
        let mut c = (!comment.is_empty()).then(|| comment.to_owned());
        self.own_op(
            "s.and.mask",
            vec![exec.clone(), exec, self.mask_opd(&m)],
            &mut c,
        )?;
        let text = mask_text(self, &m);
        let id = self.exec.take_id();
        let r = self.exec.innermost_lane().expect("a lane region");
        r.masks.push(text);
        r.id = id;
        Ok(())
    }

    /// Opens `b.raw_exec(why)`.
    pub fn raw_exec_begin(&mut self, why: &str) -> Result<()> {
        if why.is_empty() {
            return value("raw_exec needs the argument (why)");
        }
        self.exec.raw.push(why.to_owned());
        Ok(())
    }

    /// Closes `b.raw_exec`.
    pub fn raw_exec_end(&mut self) {
        self.exec.raw.pop();
    }

    /// Opens `b.waves(waves)`.
    pub fn waves_begin(&mut self, waves: &[i128]) -> Result<()> {
        let Some(wave) = self.wave_reg else {
            return schedule("waves(): declare the wave index first (Builder.wave_index)");
        };
        let n = i128::from(self.nwaves);
        let (lo, hi) = (
            waves.first().copied().unwrap_or(0),
            waves.last().copied().unwrap_or(-1),
        );
        let contiguous = !waves.is_empty() && waves.iter().copied().eq(lo..=hi);
        if !contiguous || lo < 0 || hi >= n {
            let w: Vec<String> = waves.iter().map(i128::to_string).collect();
            return value(format!(
                "waves([{}]) of {n}: a contiguous range of the workgroup's waves",
                w.join(", ")
            ));
        }
        self.settle()?;
        let id = self.exec.take_id();
        let skip = self.new_label("waves");
        let from = self.forward_len();
        let w = Arg::Reg(wave, 0);
        let mut branches: Vec<(&str, i128, Vec<i128>)> = Vec::new();
        if lo == hi {
            branches.push(("s.cmp.lg.u32", lo, (0..n).filter(|&x| x != lo).collect()));
        } else {
            if lo > 0 {
                branches.push(("s.cmp.lt.u32", lo, (0..lo).collect()));
            }
            if hi < n - 1 {
                branches.push(("s.cmp.gt.u32", hi, (hi + 1..n).collect()));
            }
        }
        for (cmp, v, taken) in branches {
            let c = self.make_plain(cmp, vec![w.clone(), Arg::Int(v)], "")?;
            self.emit(c, &[])?;
            let comment = match self.by_wave {
                true => {
                    let t: Vec<String> = taken.iter().map(i128::to_string).collect();
                    format!("by_wave skip {} of {n}", t.join(","))
                }
                false => String::new(),
            };
            let mut i =
                self.make_plain("s.cbranch.scc1", vec![Arg::Str(skip.clone())], &comment)?;
            i = i.with_attr(Attr::WavesTaken(taken))?;
            self.emit(i, &[])?;
        }
        if self.by_wave {
            self.mark_other_waves(&skip, from);
        }
        let w: Vec<String> = waves.iter().map(i128::to_string).collect();
        let region = Region {
            id,
            lane: false,
            text: format!("waves [{}]", w.join(", ")),
            masks: Vec::new(),
            save: None,
            n0: self.rows.len(),
            skip: Some(skip),
            entry_ones: self.exec.ones,
        };
        self.exec.open.push(region);
        Ok(())
    }

    /// Closes `b.waves`.
    pub fn waves_end(&mut self, error: bool) -> Result<()> {
        if error {
            self.exec.open.pop();
            return Ok(());
        }
        self.settle()?;
        let r = self.exec.open.pop().expect("an open wave block");
        self.check_branches_out(&r)?;
        self.put_label(r.skip.as_deref().expect("a skip label"), false)
    }

    /// The instructions between two rows are quiet.
    fn dead_exit(&self, exit: u64, upto: u64, labels: bool) -> bool {
        let (Some(a), Some(b)) = (self.row_index(exit), self.row_index(upto)) else {
            return false;
        };
        self.rows[a + 1..b]
            .iter()
            .all(|r| self.quiet(&r.ins, labels))
    }

    fn quiet(&self, i: &Instr, labels: bool) -> bool {
        let f = i.form();
        if f.op == "v_readlane_b32" {
            return true;
        }
        if labels && f.kind == Kind::Label {
            return true;
        }
        if !matches!(
            f.kind,
            Kind::Salu | Kind::Smem | Kind::Wait | Kind::Nop | Kind::Comment
        ) {
            return false;
        }
        let names_exec = i
            .ops()
            .iter()
            .any(|o| matches!(o, Operand::Phys(p) if p.name.starts_with("exec")))
            || f.flags.has(Flags::SAVEEXEC)
            || f.kind == Kind::Comment
                && thin_ir::print::line(i, &[]).is_ok_and(|l| l.contains("exec"));
        !names_exec
    }
}

fn label_name(i: &Instr) -> Option<String> {
    i.field_named("name").map(|v| v.py_str())
}

/// the EXEC tracker and the region check.
pub(crate) fn check(b: &mut Builder, mut i: Instr) -> Result<Instr> {
    if i.writes_exec() {
        track(b, &i);
    }
    if b.exec.open.is_empty() || b.exec.own {
        return Ok(i);
    }
    let Some(lane) = b.exec.open.iter().rev().find(|r| r.lane).cloned() else {
        return Ok(i);
    };
    i = i.with_attr(Attr::InRegion(lane.id.into()))?;
    if i.writes_exec() {
        if b.exec.raw.is_empty() {
            return schedule(format!(
                "`{}` writes exec inside the lane region {} ({}): name it (Builder.raw_exec(why)) or make it a region",
                b.text(&i),
                lane.id,
                region_text(b, &lane)
            ));
        }
        let id = b.exec.take_id();
        b.exec.innermost_lane().expect("a lane region").id = id;
    }
    let placed = |f: char| b.file(f).placed;
    for d in i.defs().filter(|d| !placed(d.file)) {
        for r in b.exec.open.iter().filter(|r| r.lane) {
            if let Some(sv) = &r.save {
                if b.overlap(d, sv) {
                    return schedule(format!(
                        "`{}` writes {}, the saved exec of the lane region {} ({})",
                        b.text(&i),
                        b.repr(sv),
                        r.id,
                        region_text(b, r)
                    ));
                }
            }
        }
    }
    Ok(i)
}

/// whether each half of EXEC has all lanes on.
fn track(b: &mut Builder, i: &Instr) {
    let t = i.phys(0).map(|p| p.name);
    let ones = i.ops().len() == 2
        && i.fields().is_empty()
        && matches!(i.ops().get(1), Some(Operand::Imm(m)) if m.text.is_empty() && match m.value {
            thin_ir::ImmValue::Int(v) => v == -1,
            thin_ir::ImmValue::Float(x) => x == -1.0,
        });
    match t {
        Some("exec_lo") => b.exec.ones.0 = ones,
        Some("exec_hi") => b.exec.ones.1 = ones,
        _ => {
            let both = t == Some("exec") && ones;
            b.exec.ones = (both, both);
        }
    }
}

/// the dead-exit test at an ENDPGM form.
pub(crate) fn dead_exit_at_end(b: &mut Builder) {
    let Some((exit, _)) = b.exec.last_exit.take() else {
        return;
    };
    let end = b.rows.last().expect("the ENDPGM row").id;
    if b.dead_exit(exit, end, true) {
        b.deferred.push(Deferred::Exit {
            exit,
            upto: end,
            region: None,
        });
    }
}

/// A loop body's start and end cut the exit.
pub(crate) fn cut_exit(b: &mut Builder) {
    b.exec.last_exit = None;
}
