//! The machine: dispatch of waves, one step, the semantics'
//! dispatch, branches and waits, barriers and the
//! single-machine runner, the claims' checks and spins
//!

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::hash::{Hash, Hasher};

use super::claims::{self, Claims};
use super::facts::{role, Facts, Ids};
use super::lds::Lds;
use super::memory::Memory;
use super::wave::Wave;
use super::{alu, memops, Exec, Reports, WgState, POISON};
use crate::tags::{Checker, Plain, Tables};
use crate::{Base, Cause, EmuError, Ins, Program};

fn fault(message: String) -> EmuError {
    EmuError {
        cause: Some(Cause::Fault),
        message,
    }
}

fn limit(message: String) -> EmuError {
    EmuError {
        cause: Some(Cause::Emulator),
        message,
    }
}

/// The waves' recorded values of one claim occurrence.
type Held = Vec<(usize, Vec<Vec<u32>>)>;

/// What a step did to the wave's run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Yield {
    /// Keep running.
    Go,
    /// Stopped at a barrier.
    Barrier,
    /// Ended.
    Done,
    /// Parked on buffers it read.
    Parked,
}

/// How an instruction is dispatched, decided once per program.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Class {
    Barrier,
    End,
    Other,
}

/// One rank's interpreter of one kernel.
pub struct Machine {
    /// The target facts.
    pub facts: Facts,
    /// The program.
    pub program: Program,
    /// Instruction texts, for messages.
    pub texts: Vec<String>,
    classes: Vec<Class>,
    /// Labels → instruction index.
    pub labels: HashMap<String, usize>,
    /// The claims by index.
    pub claims: Claims,
    /// VGPRs, AGPRs and LDS bytes of the kernel.
    pub nv: usize,
    /// AGPRs.
    pub na: usize,
    /// LDS bytes.
    pub lds_bytes: usize,
    /// u of: where the workgroup ids go (reference instance).
    pub wg_sgpr: u32,
    /// The checker.
    pub checker: Box<dyn Checker>,
    /// Sum mode.
    pub sum_records: bool,
    /// The IPC policy (on by default).
    pub ipc_policy: bool,
    /// The rank.
    pub rank: usize,
    /// The reports.
    pub reports: Reports,
    /// The clock the clock reads advance.
    pub clock: u64,
    /// The bounded polls' round counters: instruction index → SGPR.
    pub spin_at: BTreeMap<usize, u32>,
    /// Print each instruction.
    pub trace: bool,
    /// The last LDS a workgroup left.
    pub lds_last: Option<(Vec<u8>, Vec<i64>)>,
    serial: u64,
}

/// A wave initialiser (tests): called on each new wave.
pub type Init<'a> = &'a mut dyn FnMut(&mut Wave);

impl Machine {
    /// A machine over a decoded program, with decoded claims.
    pub fn new(
        program: Program,
        facts: Facts,
        nv: usize,
        na: usize,
        lds_bytes: usize,
    ) -> Result<Machine, EmuError> {
        if program.user_sgprs as usize + 3 > facts.ns
            || program
                .kernarg_sgpr
                .is_some_and(|n| n as usize + 2 > facts.ns)
        {
            return Err(limit("entry registers exceed the SGPR file".into()));
        }
        if let Some(layout) = &program.pc_layout {
            layout.validate(program.ins.len())?;
        }
        let claims = claims::all(&program, &facts)?;
        let texts: Vec<String> = program.ins.iter().map(|i| i.text.clone()).collect();
        let classes = program
            .ins
            .iter()
            .map(|i| {
                if i.canon == "s.endpgm" {
                    Class::End
                } else if i.canon == facts.barrier {
                    Class::Barrier
                } else {
                    Class::Other
                }
            })
            .collect();
        let labels = program.labels.iter().cloned().collect();
        let mut spin_at = BTreeMap::new();
        for (k, ins) in program.ins.iter().enumerate() {
            if ins.comment.starts_with("SPIN") || ins.comment.starts_with("[spin]") {
                if let Some(crate::Operand {
                    base: Base::Reg(f, r, _),
                    ..
                }) = ins.operands.first()
                {
                    if f == "s" {
                        spin_at.insert(k, *r);
                    }
                }
            }
        }
        Ok(Machine {
            wg_sgpr: program.user_sgprs,
            facts,
            texts,
            classes,
            labels,
            claims,
            nv,
            na,
            lds_bytes,
            program,
            checker: Box::new(Plain),
            sum_records: false,
            ipc_policy: true,
            rank: 0,
            reports: Reports::default(),
            clock: 0,
            spin_at,
            trace: false,
            lds_last: None,
            serial: 0,
        })
    }

    /// The waves of a workgroup at dispatch.
    pub fn make_waves(
        &self,
        kernarg: u64,
        wgid: impl Into<crate::WorkgroupId>,
        waves: usize,
        wg: usize,
        xcd: usize,
    ) -> Vec<Wave> {
        let [x, y, z] = wgid.into().0;
        let w = self.facts.w;
        (0..waves)
            .map(|k| {
                let mut wave = Wave::new(k, &self.facts, self.nv.max(1), self.na);
                // Bodies retain the historical s[0:1] ABI. A descriptor
                // can disable or relocate the kernarg pointer.
                let kernarg_sgpr = self
                    .program
                    .kernarg_sgpr
                    .or_else(|| self.program.workgroup_ids.is_none().then_some(0));
                if let Some(n) = kernarg_sgpr {
                    wave.s[n as usize] = kernarg as u32;
                    wave.s[n as usize + 1] = (kernarg >> 32) as u32;
                }
                // RDNA4 ISA guide: enabled inputs are packed with
                // no alignment holes. The descriptor ABI orders x, y, z.
                if let Some(enabled) = self.program.workgroup_ids {
                    let mut n = self.wg_sgpr as usize;
                    for (id, enabled) in [x, y, z].into_iter().zip(enabled) {
                        if enabled {
                            wave.s[n] = id;
                            n += 1;
                        }
                    }
                }
                match self.facts.ids {
                    Ids::AfterUser if self.program.workgroup_ids.is_none() => {
                        let u = self.wg_sgpr as usize;
                        if u + 1 < wave.s.len() {
                            wave.s[u] = x;
                            wave.s[u + 1] = y;
                        }
                    }
                    Ids::Ttmp => {
                        wave.ttmp9 = x;
                        wave.ttmp7 = (y & 0xFFFF) | (z & 0xFFFF) << 16;
                    }
                    Ids::AfterUser => {}
                }
                for l in 0..w {
                    wave.v[l] = (l + w * k) as u32;
                }
                wave.wg = wg;
                wave.wgx = x;
                wave.xcd = xcd;
                wave
            })
            .collect()
    }

    /// A fresh LDS for a workgroup.
    pub fn new_lds(&self, waves: usize) -> Lds {
        Lds::new(self.lds_bytes, waves)
    }

    fn exec<'a>(
        &'a mut self,
        mem: &'a mut Memory,
        tables: &'a mut Tables,
        lds: &'a mut Lds,
        wg: &'a mut WgState,
        pc: usize,
    ) -> (Exec<'a>, &'a Program) {
        self.serial += 1;
        (
            Exec {
                facts: &self.facts,
                tables,
                checker: &mut *self.checker,
                sum_mode: self.sum_records,
                mem,
                lds,
                epoch: wg.epoch,
                rank: self.rank,
                reports: &mut self.reports,
                texts: &self.texts,
                wg,
                ipc_policy: self.ipc_policy,
                serial: self.serial,
                pc,
                clock: &mut self.clock,
            },
            &self.program,
        )
    }

    /// One step of a wave.
    pub fn step(
        &mut self,
        w: &mut Wave,
        mem: &mut Memory,
        tables: &mut Tables,
        lds: &mut Lds,
        wg: &mut WgState,
    ) -> Result<Yield, EmuError> {
        let pc = w.pc;
        if pc >= self.program.ins.len() {
            self.wave_end(w, mem, lds, wg);
            return Ok(Yield::Done);
        }
        if let Some(regs) = self.claims.undefined.get(&pc) {
            for (f, a, n) in regs {
                w.poison(f, *a, *n);
            }
        }
        if let Some(list) = self.claims.assume.get(&pc) {
            for a in list {
                check_assume(w, a, pc)?;
            }
        }
        if let Some(u) = self.claims.uniform.get(&pc) {
            let vals = u
                .regs
                .iter()
                .map(|(f, a, n)| {
                    if f == "s" {
                        (*a..*a + *n)
                            .map(|r| w.s.get(r as usize).copied().unwrap_or(POISON))
                            .collect()
                    } else {
                        // A named register: its dwords as an instruction reads them.
                        let o = crate::Operand {
                            base: Base::Reg(f.clone(), *a, *n),
                            neg: false,
                            abs: false,
                        };
                        let v = w.sread64(&o).unwrap_or(u64::from(POISON));
                        (0..*n).map(|k| (v >> (32 * k)) as u32).collect()
                    }
                })
                .collect();
            w.uniform.push((pc, vals));
        }
        w.pc += 1;
        w.nins += 1;
        self.reports.stats.ins += 1;
        if self.trace {
            println!("w{} {pc} {}", w.wid, self.texts[pc]);
        }
        match self.classes[pc] {
            Class::Barrier => {
                w.at_barrier = true;
                w.at_fence = self.program.ins[pc].comment.starts_with("LDS_FENCE")
                    || self.program.ins[pc].comment.starts_with("[lds fence]");
                return Ok(Yield::Barrier);
            }
            Class::End => {
                self.wave_end(w, mem, lds, wg);
                return Ok(Yield::Done);
            }
            Class::Other => {}
        }
        let kind = self.program.ins[pc].kind.as_str();
        match kind {
            "salu" | "smem" | "branch" | "wait" | "nop" => self.reports.stats.salu += 1,
            "valu" => self.reports.stats.valu += 1,
            _ => {}
        }
        // a register a pending load writes is touched by no other
        // instruction (the waits excepted); a landing op's write is kept.
        let landing = {
            let ins = &self.program.ins[pc];
            if !ins.canon.starts_with("s.wait") {
                super::pending::check(w, ins).map_err(|err| wrap(err, w.wid, pc, ins))?;
            }
            super::pending::landing(&self.facts, ins).map(|(c, unordered, dest)| {
                let seq = w.seq.get(c).copied().unwrap_or(0);
                (super::pending::before(w, dest), c, seq, unordered)
            })
        };
        let (mut e, program) = self.exec(mem, tables, lds, wg, pc);
        let ins = &program.ins[pc];
        let taken = match semantics(&mut e, w, ins, &program.labels, program.pc_layout.as_ref()) {
            Ok(t) => t,
            Err(err) => return Err(wrap(err, w.wid, pc, ins)),
        };
        if let Some((before, c, seq, unordered)) = landing {
            super::pending::record(w, before, pc, &ins.text, c, seq, unordered);
        }
        // An edge claim poisons its registers when its edge is taken.
        if let Some(regs) = self.claims.edges.get(&(pc, w.pc)) {
            for (f, a, n) in regs {
                w.poison(f, *a, *n);
            }
        }
        if let Some(target) = taken {
            if target <= pc {
                return Ok(self.spin_check(w, mem, pc, target));
            }
        }
        Ok(Yield::Go)
    }

    /// The spin check at a taken backward branch.
    fn spin_check(&mut self, w: &mut Wave, mem: &Memory, pc: usize, target: usize) -> Yield {
        let zeroed: BTreeSet<u32> = self.spin_at.range(target..=pc).map(|(_, r)| *r).collect();
        let mut h = std::collections::hash_map::DefaultHasher::new();
        (target, w.exec, w.exec_hi, w.vcc, w.vcc_hi, w.scc, w.m0).hash(&mut h);
        for (i, s) in w.s.iter().enumerate() {
            if zeroed.contains(&(i as u32)) {
                0u32.hash(&mut h);
            } else {
                s.hash(&mut h);
            }
        }
        w.v.hash(&mut h);
        w.a.hash(&mut h);
        let hash = h.finish();
        if let Some((prev, progress)) = w.head {
            if prev == hash && progress == w.progress && !w.watch.is_empty() {
                w.parked = Some(w.watch.iter().map(|(b, v)| (*b, *v)).collect());
                w.spin_regs = zeroed.into_iter().collect();
                let _ = mem;
                return Yield::Parked;
            }
        }
        w.head = Some((hash, w.progress));
        w.watch.clear();
        Yield::Go
    }

    /// Times out a parked wave's bounded poll: its spin counters
    /// are set to run out and it runs again. Whether it had any.
    pub fn time_out(w: &mut Wave) -> bool {
        if w.parked.is_none() || w.spin_regs.is_empty() {
            return false;
        }
        for reg in std::mem::take(&mut w.spin_regs) {
            if let Some(s) = w.s.get_mut(reg as usize) {
                *s = 0x7FFF_FFFF;
            }
        }
        w.parked = None;
        w.head = None;
        w.watch.clear();
        true
    }

    /// Whether a parked wave may run again: one of its buffers changed.
    pub fn awake(w: &mut Wave, mem: &Memory) -> bool {
        let Some(on) = &w.parked else { return true };
        let woke = on
            .iter()
            .any(|(b, v)| mem.bufs.get(*b).is_some_and(|x| x.version != *v));
        if woke {
            w.parked = None;
            w.head = None;
            w.watch.clear();
        }
        woke
    }

    /// A wave's end.
    pub fn wave_end(&mut self, w: &mut Wave, mem: &mut Memory, lds: &mut Lds, wg: &WgState) {
        if w.done {
            return;
        }
        w.done = true;
        w.at_barrier = false;
        super::pending::retire_all(w); // Every pending write lands.
        self.reports.stats.wave_ins_max = self.reports.stats.wave_ins_max.max(w.nins);
        if w.wid < lds.waves.len() {
            lds.wave_end(w.wid, wg.epoch);
        }
        let pending: Vec<usize> = w
            .vm_ops
            .iter()
            .flat_map(|o| o.records.iter().copied())
            .collect();
        mem.flight.wave_end(&pending);
    }

    /// Runs a wave until it stops at a barrier, ends or parks; with a
    /// budget, at most that many instructions.
    pub fn run_wave(
        &mut self,
        w: &mut Wave,
        mem: &mut Memory,
        tables: &mut Tables,
        lds: &mut Lds,
        wg: &mut WgState,
        budget: u64,
    ) -> Result<Yield, EmuError> {
        let mut n = 0;
        loop {
            if w.done {
                return Ok(Yield::Done);
            }
            let y = self.step(w, mem, tables, lds, wg)?;
            if y != Yield::Go {
                return Ok(y);
            }
            n += 1;
            if budget > 0 && n >= budget {
                return Ok(Yield::Go);
            }
        }
    }

    /// Releases a workgroup's barrier.
    pub fn release(
        &mut self,
        waves: &mut [Wave],
        lds: &mut Lds,
        wg: &mut WgState,
    ) -> Result<(), EmuError> {
        let waiting: Vec<(usize, bool)> = waves
            .iter()
            .filter(|w| w.at_barrier && !w.done)
            .map(|w| (w.wid, w.at_fence))
            .collect();
        let fence = waiting.iter().any(|(_, f)| *f);
        if fence {
            lds.fence(wg.epoch + 1, &waiting, &self.texts)?;
        }
        for w in waves.iter_mut() {
            w.at_barrier = false;
            w.at_fence = false;
        }
        wg.epoch += 1;
        wg.barrier_marks = waves.iter().map(|w| w.mark).collect();
        self.check_uniform(waves)
    }

    /// The uniform claims' check: the k-th record of every wave
    /// equals the first wave's.
    pub fn check_uniform(&mut self, waves: &mut [Wave]) -> Result<(), EmuError> {
        let mut by: BTreeMap<(usize, usize), Held> = BTreeMap::new();
        for w in waves.iter() {
            let mut count: HashMap<usize, usize> = HashMap::new();
            for (pc, vals) in &w.uniform {
                let k = count.entry(*pc).or_default();
                by.entry((*pc, *k)).or_default().push((w.wid, vals.clone()));
                *k += 1;
            }
        }
        for w in waves.iter_mut() {
            w.uniform.clear();
        }
        for ((pc, _), list) in by {
            let (a, first) = &list[0];
            for (b, vals) in &list[1..] {
                if vals != first {
                    let text = &self.claims.uniform[&pc].text;
                    return Err(fault(format!(
                        "`; {text}` before instruction {pc} is false: wave {a} holds {}, wave {b} {} (Builder.uniform's claim: alike in every wave)",
                        claims::held_text(first),
                        claims::held_text(vals)
                    )));
                }
            }
        }
        Ok(())
    }

    /// Runs one workgroup on the single-machine runner.
    #[allow(clippy::too_many_arguments)]
    pub fn run_wg(
        &mut self,
        mem: &mut Memory,
        tables: &mut Tables,
        kernarg: u64,
        wgid: impl Into<crate::WorkgroupId>,
        waves: usize,
        wg_serial: usize,
        init: Option<Init<'_>>,
    ) -> Result<(Vec<Wave>, Lds), EmuError> {
        let mut ws = self.make_waves(kernarg, wgid, waves, wg_serial, 0);
        if let Some(f) = init {
            for w in &mut ws {
                f(w);
            }
        }
        let mut lds = self.new_lds(waves);
        let mut wg = WgState {
            serial: wg_serial,
            epoch: 0,
            barrier_marks: vec![-1; waves],
        };
        let mut timeouts = 0;
        loop {
            let mut ran = false;
            for i in 0..ws.len() {
                if ws[i].done || ws[i].at_barrier || !Machine::awake(&mut ws[i], mem) {
                    continue;
                }
                let mut w = std::mem::replace(&mut ws[i], Wave::new(0, &self.facts, 0, 0));
                let r = self.run_wave(&mut w, mem, tables, &mut lds, &mut wg, 0);
                ws[i] = w;
                if r? == Yield::Done {
                    timeouts = 0;
                }
                ran = true;
            }
            if ws.iter().all(|w| w.done) {
                break;
            }
            if ws.iter().all(|w| w.done || w.at_barrier) {
                self.release(&mut ws, &mut lds, &mut wg)?;
                continue;
            }
            // Every wave left is parked on memory only this workgroup could
            // change: its bounded polls time out, else nothing can run.
            if !ran {
                if !ws
                    .iter_mut()
                    .fold(false, |any, w| Machine::time_out(w) | any)
                {
                    return Err(self.deadlock(&ws, mem));
                }
                timeouts += 1;
                if timeouts > super::cluster::TIMEOUTS {
                    return Err(EmuError {
                        cause: Some(Cause::Deadlock),
                        message: format!(
                            "bounded polls timed out {} times in a row, no wave ending",
                            super::cluster::TIMEOUTS
                        ),
                    });
                }
            }
        }
        self.check_uniform(&mut ws)?;
        self.lds_last = Some((lds.data.clone(), lds.tags.clone()));
        Ok((ws, lds))
    }
}

impl Machine {
    /// The single-machine runner's deadlock: no wave of the workgroup can run.
    fn deadlock(&self, ws: &[Wave], mem: &Memory) -> EmuError {
        let parts: Vec<String> = (ws.iter().filter(|w| !w.done))
            .map(|w| match &w.parked {
                Some(on) => {
                    let mut bufs: Vec<usize> = on.iter().map(|(b, _)| *b).collect();
                    bufs.sort_unstable();
                    bufs.dedup();
                    let names: Vec<&str> = (bufs.iter())
                        .filter_map(|b| mem.bufs.get(*b).map(|x| x.name.as_str()))
                        .collect();
                    format!("wave {} waits on {}", w.wid, names.join(", "))
                }
                None => format!("wave {} at a barrier", w.wid),
            })
            .collect();
        EmuError {
            cause: Some(Cause::Deadlock),
            message: format!("deadlock: no wave can run: {}", parts.join("; ")),
        }
    }
}

fn check_assume(w: &Wave, a: &claims::Assume, pc: usize) -> Result<(), EmuError> {
    let (f, first, count) = (&a.reg.0, a.reg.1, a.reg.2);
    if f == "s" {
        let v = (0..count).fold(0u64, |acc, k| {
            acc | u64::from(w.s.get((first + k) as usize).copied().unwrap_or(POISON)) << (32 * k)
        });
        let v = if count == 1 { v & 0xFFFF_FFFF } else { v };
        if v < a.lo || v > a.hi {
            return Err(fault(claims::assume_false(&a.text, pc, w.wid, None, v)));
        }
        return Ok(());
    }
    let wn = w.w;
    for l in 0..wn {
        if w.exec >> l & 1 == 0 {
            continue;
        }
        let r = first as usize;
        let file = if f == "a" { &w.a } else { &w.v };
        let v = u64::from(file.get(r * wn + l).copied().unwrap_or(POISON));
        if v < a.lo || v > a.hi {
            return Err(fault(claims::assume_false(&a.text, pc, w.wid, Some(l), v)));
        }
    }
    Ok(())
}

/// Wrap a native execution error with instruction context.
fn wrap(err: EmuError, wid: usize, pc: usize, ins: &Ins) -> EmuError {
    if err.message.contains(&ins.text) {
        return err;
    }
    EmuError {
        cause: err.cause,
        message: format!("wave {wid} at {pc} `{}`: {}", ins.text, err.message),
    }
}

/// The semantics of one instruction: `Some(target)` for a taken
/// branch.
#[allow(clippy::too_many_lines)]
fn semantics(
    e: &mut Exec<'_>,
    w: &mut Wave,
    ins: &Ins,
    labels: &[(String, usize)],
    pc_layout: Option<&super::pc::Layout>,
) -> Result<Option<usize>, EmuError> {
    for omod in ["clamp", "mul", "div", "omod"] {
        let allowed_clamp = omod == "clamp"
            && (ins.canon.starts_with("v.pk.")
                || ins.canon == "v.dot2.f32.bf16"
                || ins.canon.starts_with("v.wmma.i32."));
        if ins.field(omod).is_some() && !allowed_clamp {
            return Err(limit(format!(
                "`{}`: its output modifier (clamp, mul, div) is not modelled for {}",
                ins.text, ins.canon
            )));
        }
    }
    let canon = ins.canon.trim_end_matches(".e64").trim_end_matches(".e32");
    if super::pc::is_pc(canon) {
        return super::pc::execute(canon, w, ins, pc_layout, e.pc);
    }
    let label = |l: &str| -> Result<usize, EmuError> {
        labels
            .iter()
            .rev()
            .find(|(n, _)| n == l)
            .map(|(_, i)| *i)
            .ok_or_else(|| {
                limit(format!(
                    "{}: a branch to a missing label {l} (internal)",
                    ins.text
                ))
            })
    };
    let target = || -> Result<usize, EmuError> {
        match ins.operands.first().map(|o| &o.base) {
            Some(Base::Label(l)) => label(l),
            _ => Err(fault(format!("{}: no label", ins.text))),
        }
    };
    if let Some(wt) = e.facts.waits.iter().find(|x| x.canon == canon) {
        // A wait: each count retires its counters' ops past it; scalar
        // memory is unordered, so only a count of 0 retires it.
        for (field, roles) in &wt.counts {
            let n = if wt.bare {
                match ins.operands.first().map(|o| &o.base) {
                    Some(Base::Literal(v, _)) => u64::from(*v),
                    _ => alu::int_field(ins, "simm16", 0) as u64,
                }
            } else {
                match ins.field(field) {
                    Some(crate::Field::Int(v)) => *v as u64,
                    Some(_) => 0,
                    None => continue,
                }
            };
            for &r in roles {
                if r == role::SMEM {
                    super::pending::retire(w, r, n, 0);
                } else {
                    wait(e, w, r, n);
                    let issued = w.seq[r];
                    super::pending::retire(w, r, n, issued);
                }
            }
        }
        return Ok(None);
    }
    match canon {
        "s.branch" => {
            let t = target()?;
            w.pc = t;
            return Ok(Some(t));
        }
        c if c.starts_with("s.cbranch.") => {
            let cond = match &c[10..] {
                "scc0" => !w.scc,
                "scc1" => w.scc,
                "execz" => w.exec == 0,
                "execnz" => w.exec != 0,
                "vccz" => w.vcc == 0,
                "vccnz" => w.vcc != 0,
                _ => return Err(limit(format!("instruction not emulated: {}", ins.text))),
            };
            if cond {
                let t = target()?;
                w.pc = t;
                return Ok(Some(t));
            }
            return Ok(None);
        }
        c if e.facts.no_ops.iter().any(|n| n == c) => return Ok(None),
        "v.nop" | "s.nop" | "s.sleep" | "s.setprio" | "s.wait.idle" | "s.wait.expcnt" => {
            return Ok(None);
        }
        "s.getreg.b32" => {
            // The core cycle counter (the clock hooks' halves): a read of
            // the low half moves it on, as a clock read does; the high half
            // reads it as it is. No other hardware register is modelled.
            let imm = match ins.operands.get(1).map(|o| &o.base) {
                Some(Base::Literal(v, _)) => *v,
                _ => 0,
            };
            let (id, off, size) = (imm & 63, (imm >> 6) & 31, ((imm >> 11) & 31) + 1);
            let v = match e.facts.clocks.cycles {
                Some((lo, _)) if id == lo => {
                    *e.clock += 100;
                    *e.clock as u32
                }
                Some((_, hi)) if id == hi => (*e.clock >> 32) as u32,
                _ => {
                    return Err(limit(format!(
                        "{}: hwreg {id} not modelled (only the cycle counter)",
                        ins.text
                    )));
                }
            };
            let mask = if size >= 32 {
                u32::MAX
            } else {
                (1 << size) - 1
            };
            w.swrite(&ins.operands[0], u64::from((v >> off) & mask))?;
            return Ok(None);
        }
        "s.sendmsg.rtn.b64" | "s.sendmsg.rtn.b32" => {
            // The constant clock, at a twentieth of the cycle counter's
            // rate; its result lands at the KMcnt wait (pending.rs).
            let msg = match ins.operands.get(1).map(|o| &o.base) {
                Some(Base::Literal(v, _)) => *v,
                _ => 0,
            };
            if e.facts.clocks.realtime != Some(msg) {
                return Err(limit(format!(
                    "{}: message {msg} not modelled (only the constant clock)",
                    ins.text
                )));
            }
            *e.clock += 100;
            let now = *e.clock / 20;
            if canon.ends_with("b64") {
                w.swrite64(&ins.operands[0], now)?;
            } else {
                w.swrite(&ins.operands[0], now & 0xffff_ffff)?;
            }
            return Ok(None);
        }
        "s.memtime" if e.facts.clocks.memtime => {
            *e.clock += 100;
            w.swrite64(&ins.operands[0], *e.clock)?;
            return Ok(None);
        }
        "s.memrealtime" if e.facts.clocks.memrealtime => {
            *e.clock += 100;
            w.swrite64(&ins.operands[0], *e.clock / 20)?;
            return Ok(None);
        }
        _ => {}
    }
    if let Some(name) = &ins.matrix {
        e.reports.stats.matrix += 1;
        w.progress += 1;
        let form = e.facts.matrix_form(name);
        return match form {
            Some(form) if e.checker.products() => {
                super::wmma::wmma(w, ins, form)?;
                Ok(None)
            }
            Some(form) => {
                super::mfma::check_matrix(e, w, ins, form)?;
                Ok(None)
            }
            None => Err(limit(format!(
                "{}: matrix op values are not modelled on {} ",
                ins.text, e.facts.name
            ))),
        };
    }

    if canon == "v.accvgpr.mov.b32" && w.exec == w.all() && ins.operands.len() >= 2 {
        if let (Base::Reg(df, dr, _), Base::Reg(sf, sr, _)) =
            (&ins.operands[0].base, &ins.operands[1].base)
        {
            if df == "a" && sf == "a" && w.is_acc_marked("a", *sr as usize) {
                let s = w.vread(&ins.operands[1], 0)?;
                if let (Some(di), Some(si)) = (
                    w.acc_index("a", *dr as usize),
                    w.acc_index("a", *sr as usize),
                ) {
                    if di != si {
                        let cloned = w.acc.entry(si).clone();
                        *w.acc.entry(di) = cloned;
                    }
                    let marks = vec![[crate::tags::ACC_MARK; 4]; w.w];
                    w.vwrite(&ins.operands[0], 0, &s.v, Some(&marks), w.all())?;
                    return Ok(None);
                }
            }
        }
    }

    if canon.starts_with("v.movrel") {
        w.materialize_file(e.tables, "v");
    } else if !canon.starts_with("s.") {
        for op in &ins.operands {
            w.materialize_operand(e.tables, op);
        }
    }

    let handled = if canon.starts_with("s.") {
        match alu::salu(w, ins, canon) {
            Some(r) => {
                r?;
                true
            }
            None => memops::smem(e, w, ins, canon)?,
        }
    } else if canon.starts_with("v.") {
        match alu::valu(e, w, ins, canon) {
            Some(r) => {
                r?;
                if canon == "v.accvgpr.write.b32" && w.exec == w.all() {
                    if let Some(dst_op) = ins.operands.first() {
                        if let Base::Reg(f, r, _) = &dst_op.base {
                            if let Some(acc_idx) = w.acc_index(f, *r as usize) {
                                let dst_src = w.vread(dst_op, 0)?;
                                let lanes = dst_src.tags(w.w);
                                if lanes.iter().flatten().any(|&t| crate::tags::is_record(t)) {
                                    let resumed = w
                                        .acc
                                        .resume(e.tables, acc_idx, lanes, &ins.text)
                                        .map_err(alu::tag_error)?;
                                    if resumed {
                                        let marks = vec![[crate::tags::ACC_MARK; 4]; w.w];
                                        w.vwrite(dst_op, 0, &dst_src.v, Some(&marks), w.all())?;
                                    }
                                }
                            }
                        }
                    }
                }
                true
            }
            None => false,
        }
    } else if canon.starts_with("ds.") {
        memops::ds(e, w, ins, canon)?
    } else if canon.starts_with("global.") || canon.starts_with("buffer.") {
        memops::vmem(e, w, ins, canon)?
    } else {
        false
    };
    if !handled {
        return Err(limit(format!("instruction not emulated: {}", ins.text)));
    }
    Ok(None)
}

/// A wait with count n on a counter.
fn wait(e: &mut Exec<'_>, w: &mut Wave, counter: usize, n: u64) {
    let seq = w.seq[counter];
    let done = seq.saturating_sub(n);
    if counter == role::DS {
        e.lds_cover(w, done);
        return;
    }
    if counter == role::LOAD && w.wid < e.lds.waves.len() {
        e.lds.vm_cover(w.wid, e.epoch, done);
    }
    let (cov, keep): (Vec<_>, Vec<_>) = std::mem::take(&mut w.vm_ops)
        .into_iter()
        .partition(|o| o.counter == counter && o.seq < done);
    w.vm_ops = keep;
    if cov.is_empty() {
        return;
    }
    let entry = e.reports.vmw.entry(e.pc).or_insert([0; 6]);
    entry[0] += 1;
    let loads = cov.iter().filter(|o| !o.store).count() as u64;
    let stores = cov.iter().filter(|o| o.store).count() as u64;
    entry[1] += loads;
    entry[2] += stores;
    let youngest_load = cov.last().is_some_and(|o| !o.store);
    if youngest_load && stores > 0 {
        entry[3] += 1;
        entry[4] += stores;
    }
    let ids: Vec<usize> = cov.iter().flat_map(|o| o.records.iter().copied()).collect();
    let newest = e.mem.flight.perform(&ids);
    w.mark = w.mark.max(newest);
}

impl Exec<'_> {
    fn lds_cover(&mut self, w: &Wave, done: u64) {
        if w.wid < self.lds.waves.len() {
            self.lds.ds_cover(w.wid, self.epoch, done);
        }
    }
}
