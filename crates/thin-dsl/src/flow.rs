//! Control flow: branch conditions, if-chains, loops and exits.

use std::collections::HashMap;

use thin_ir::Instr;
use thin_target::{Flags, Kind};

use crate::builder::{Builder, LoopRec, State};
use crate::error::{flow, schedule, Result};
use crate::expr::cond::{junction, Into, Where};
use crate::expr::node::Tree;
use crate::maker::Arg;
use crate::types::Ty;
use crate::values::Typed;

/// A branch condition.
#[derive(Clone, Debug)]
pub enum Cond {
    /// A tree.
    Tree(Tree),
    /// An instruction (a scalar compare).
    Instr(Box<Instr>),
    /// Anything else, by its repr.
    Other(String),
}

/// An open if / elif / else construct.
#[derive(Clone, Debug)]
pub struct Chain {
    name: String,
    end_given: Option<String>,
    skip: Option<String>,
    skip_state: Option<State>,
    end: Option<String>,
    arm_ends: Vec<State>,
}

/// An open arm: its identity and its chain.
#[derive(Clone, Debug)]
pub struct Arm {
    /// Its identity (the entry's loads compare the open arms).
    pub id: u64,
    chain: Chain,
    kind: ArmKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ArmKind {
    When,
    Otherwise,
}

/// The constructs' state.
#[derive(Clone, Debug, Default)]
pub struct Flow {
    /// The loop a Python `break` left (the next settle refuses it).
    pub broken: Option<String>,
    /// The chain kept open by the arm before.
    pub chain: Option<Chain>,
    /// The chain an `otherwise` arm just closed, until the next statement.
    otherwise_closed: bool,
    /// The open arms, innermost last.
    pub arms: Vec<Arm>,
    next_arm: u64,
    loops: Vec<LoopState>,
}

/// A loop being emitted.
#[derive(Clone, Debug)]
pub struct LoopState {
    name: String,
    kind: LoopKind,
    i: Option<Typed>,
    n: Option<Arg>,
    until: Option<Tree>,
    peel: u32,
    end_given: Option<String>,
    end: Option<String>,
    top: String,
    skip: Option<String>,
    out: Option<String>,
    conservative: bool,
    names: HashMap<String, u32>,
    n0: usize,
    pass_start: usize,
    starts: Vec<Option<u64>>,
    audit: Option<(Vec<String>, State)>,
    copies: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LoopKind {
    Counted,
    Until,
    Repeat,
}

/// The `<what>` of a construct as written.
fn what(c: &str, display: &str) -> String {
    format!("{c}({display})")
}

/// Closes the chain an arm kept open.
pub(crate) fn close_chain(b: &mut Builder) -> Result<()> {
    b.flow.otherwise_closed = false;
    let Some(chain) = b.flow.chain.take() else {
        return Ok(());
    };
    b.close(chain)
}

impl Builder {
    fn close(&mut self, chain: Chain) -> Result<()> {
        if let Some(skip) = &chain.skip {
            let was = self.reachable();
            self.place_label(skip, false)?;
            if !was {
                if let Some(s) = &chain.skip_state {
                    self.state = self.copy_state(&s.clone());
                }
            }
        }
        if let Some(end) = &chain.end {
            let mut ends = chain.arm_ends.clone();
            if !self.reachable() && !ends.is_empty() {
                self.state = ends.remove(0);
            }
            if !ends.is_empty() {
                self.join(&ends);
            }
            self.place_label(end, false)?;
        }
        Ok(())
    }

    /// The open arms' identities, innermost last.
    pub fn arm_ids(&self) -> Vec<u64> {
        self.flow.arms.iter().map(|a| a.id).collect()
    }

    /// `branch_if(b, cond, taken, target, what, comment)`.
    pub fn branch_if(
        &mut self,
        cond: &Cond,
        taken: bool,
        target: &str,
        what: &str,
        comment: Option<String>,
    ) -> Result<()> {
        if let Cond::Tree(n) = cond {
            if let Some((op, [p1, p2])) = junction(n) {
                let any_decides = op == "and" && !taken || op == "or" && taken;
                if any_decides {
                    self.branch_if(&Cond::Tree(p1), taken, target, what, None)?;
                    return self.branch_if(&Cond::Tree(p2), taken, target, what, None);
                }
                let o = self.new_label("j");
                self.branch_if(&Cond::Tree(p1), !taken, &o, what, None)?;
                self.branch_if(&Cond::Tree(p2), taken, target, what, None)?;
                return self.put_label(&o, false);
            }
        }
        self.scalar_compare(cond, what, comment)?;
        let op = if taken {
            "s.cbranch.scc1"
        } else {
            "s.cbranch.scc0"
        };
        let i = self.make_plain(op, vec![Arg::Str(target.to_owned())], "")?;
        self.emit(i, &[]).map(|_| ())
    }

    fn scalar_compare(&mut self, cond: &Cond, what: &str, comment: Option<String>) -> Result<()> {
        match cond {
            Cond::Instr(i) => {
                let f = i.form();
                if f.flags.has(Flags::CMP) {
                    return flow(format!(
                        "{what}: `{}` is a lane compare: a lane condition is an exec region (b.lanes), not a branch",
                        self.text(i)
                    ));
                }
                if f.kind != Kind::Salu
                    || !(f.flags.has(Flags::SCC_CMP) || f.op.starts_with("s_cmpk"))
                {
                    return flow(format!(
                        "{what}: `{}` is not a scalar compare (s_cmp* / s_bitcmp*)",
                        self.text(i)
                    ));
                }
                self.emit((**i).clone(), &[]).map(|_| ())
            }
            Cond::Other(r) => flow(format!(
                "{what}: {r} is not a condition (an expr compare, or an s_cmp / s_bitcmp Instr)"
            )),
            Cond::Tree(n) => match self.asm_cond(n, comment, Into::Scc)? {
                Where::Scc => Ok(()),
                w => {
                    let at = match w {
                        Where::Vcc => "vcc".to_owned(),
                        Where::Mask(v) => v.repr(),
                        Where::Phys(p) => p.name.to_owned(),
                        Where::Scc => unreachable!(),
                    };
                    flow(format!(
                        "{what}: {} is a lane condition (in {at}): an exec region (b.lanes), not a branch",
                        crate::expr::text::top(n)
                    ))
                }
            },
        }
    }

    /// Enters `when`/`unless`.
    pub fn when(
        &mut self,
        cond: &Cond,
        display: &str,
        unless: bool,
        name: &str,
        comment: Option<String>,
        end: Option<String>,
    ) -> Result<()> {
        self.settle()?;
        let chain = Chain {
            name: name.to_owned(),
            end_given: end,
            skip: None,
            skip_state: None,
            end: None,
            arm_ends: Vec::new(),
        };
        let c = if unless { "unless" } else { "when" };
        self.enter_arm(chain, Some((cond, display)), unless, c, comment)
    }

    /// Enters `elif_` (a condition) or `otherwise` (none).
    pub fn elif(&mut self, cond: Option<(&Cond, &str)>, comment: Option<String>) -> Result<()> {
        let c = if cond.is_some() { "elif_" } else { "otherwise" };
        let Some(mut chain) = self.flow.chain.take() else {
            let w = what(c, cond.map_or("", |c| c.1));
            if std::mem::take(&mut self.flow.otherwise_closed) {
                return flow(format!(
                    "{w}: right after otherwise, which ends the chain (otherwise is its last arm)"
                ));
            }
            return flow(format!(
                "{w}: no when / elif_ arm right before it (a comment, a label or an instruction closes the chain)"
            ));
        };
        if chain.end.is_none() {
            let end = match &chain.end_given {
                Some(e) => e.clone(),
                None => self.new_label(&format!("{}_end", chain.name)),
            };
            chain.end = Some(end);
        }
        if self.reachable() {
            let end = chain.end.clone().expect("an end label");
            let i = self.make_plain("s_branch", vec![Arg::Str(end)], "")?;
            self.emit(i, &[])?;
            let st = self.copy_current();
            chain.arm_ends.push(st);
        }
        if let Some(skip) = chain.skip.take() {
            self.place_label(&skip, false)?;
        }
        if let Some(s) = chain.skip_state.take() {
            self.state = self.copy_state(&s);
        }
        self.enter_arm(chain, cond, false, c, comment)
    }

    fn enter_arm(
        &mut self,
        mut chain: Chain,
        cond: Option<(&Cond, &str)>,
        unless: bool,
        c: &str,
        comment: Option<String>,
    ) -> Result<()> {
        let kind = if cond.is_some() {
            ArmKind::When
        } else {
            ArmKind::Otherwise
        };
        if let Some((cond, display)) = cond {
            let skip = self.new_label(&chain.name);
            self.branch_if(cond, unless, &skip, &what(c, display), comment)?;
            chain.skip = Some(skip);
            chain.skip_state = Some(self.copy_current());
        }
        self.flow.next_arm += 1;
        let id = self.flow.next_arm;
        self.flow.arms.push(Arm { id, chain, kind });
        Ok(())
    }

    /// Leaves the innermost arm; `error` when it is left by an
    /// exception.
    pub fn leave_arm(&mut self, error: bool) -> Result<()> {
        let Some(arm) = self.flow.arms.pop() else {
            return Ok(());
        };
        if error {
            return Ok(());
        }
        self.settle()?;
        match arm.kind {
            ArmKind::Otherwise => {
                self.close(arm.chain)?;
                self.flow.otherwise_closed = true;
                Ok(())
            }
            ArmKind::When => {
                self.flow.chain = Some(arm.chain);
                Ok(())
            }
        }
    }

    /// `b.exit_when(cond, to, comment)`.
    pub fn exit_when(
        &mut self,
        cond: &Cond,
        display: &str,
        to: &str,
        comment: Option<String>,
    ) -> Result<()> {
        self.settle()?;
        self.branch_if(cond, true, to, &what("exit_when", display), comment)
    }
}

/// A loop's arguments.
#[derive(Clone, Debug)]
pub struct LoopArgs {
    /// The count: a register or an int.
    pub n: Option<Arg>,
    /// The condition and its display.
    pub until: Option<(Tree, String)>,
    /// The given index.
    pub i: Option<Typed>,
    /// The start: an int, or (`start_given`) another value.
    pub start: Arg,
    /// Whether start was given other than 0.
    pub start_given: bool,
    /// The copies to peel.
    pub peel: u32,
    /// The end label.
    pub end: Option<String>,
    /// The name.
    pub name: Option<String>,
    /// Whether the waits may be the larger of entry and steady.
    pub conservative: bool,
}

impl Builder {
    /// Opens a loop: the refusals, the arguments landed, the
    /// prologue. Returns the copies of the body the author writes.
    pub fn loop_begin(&mut self, a: LoopArgs) -> Result<u32> {
        let kind = match (&a.n, &a.until) {
            (Some(_), None) => LoopKind::Counted,
            (None, Some(_)) => LoopKind::Until,
            _ => LoopKind::Repeat,
        };
        let name = a.name.clone().unwrap_or_else(|| match kind {
            LoopKind::Counted => "loop".into(),
            LoopKind::Until => "until".into(),
            LoopKind::Repeat => "repeat".into(),
        });
        let name = if a.n.is_some() && a.until.is_some() && a.name.is_none() {
            "loop".to_owned()
        } else {
            name
        };
        if a.n.is_some() && a.until.is_some() {
            return flow(format!("loop ({name}): a count or a condition, not both"));
        }
        if a.n.is_none() && (a.i.is_some() || a.start_given) {
            return flow(format!(
                "loop ({name}): i= and start= are a counted loop's (n=)"
            ));
        }
        if a.peel > 0 && a.until.is_some() {
            return flow(format!(
                "loop ({name}): peel= is a counted loop's or one left by its breaks (no until=)"
            ));
        }
        self.land_arguments(&format!("the kernel arguments, before loop {name}"))?;
        if let (Some(Arg::Int(n)), Arg::Int(s)) = (&a.n, &a.start) {
            if n <= s {
                return flow(format!(
                    "loop ({name}): {} trips (n {n}, start {s}): write nothing",
                    n - s
                ));
            }
        }
        self.settle()?;
        let mut l = LoopState {
            name: name.clone(),
            kind,
            i: None,
            n: a.n.clone(),
            until: a.until.map(|u| u.0),
            peel: a.peel,
            end_given: a.end,
            end: None,
            top: String::new(),
            skip: None,
            out: None,
            conservative: a.conservative,
            names: HashMap::new(),
            n0: 0,
            pass_start: 0,
            starts: Vec::new(),
            audit: None,
            copies: 0,
        };
        if kind == LoopKind::Counted {
            let i = match a.i {
                Some(i) => i,
                None => {
                    let m = crate::builder::Make {
                        n: Some(1),
                        n_repr: "1".into(),
                        name: Some(format!("{name}.i")),
                        where_: self.where_text.clone(),
                        ..Default::default()
                    };
                    self.make_value('s', Ty::U32, m)?
                }
            };
            self.emit_plain("s.mov.b32", vec![reg(&i), a.start.clone()])?;
            l.i = Some(i);
        }
        l.top = self.new_label(&name);
        let copies = if a.peel > 0 {
            l.out = Some(self.new_label(&format!("{name}_end")));
            a.peel + 2
        } else {
            if kind == LoopKind::Counted
                && !matches!((&a.n, &a.start), (Some(Arg::Int(_)), Arg::Int(_)))
            {
                let skip = self.new_label(&format!("{name}_end"));
                let i = reg(l.i.as_ref().expect("an index"));
                self.emit_plain("s.cmp.lt.u32", vec![i, a.n.clone().expect("a count")])?;
                self.emit_plain("s.cbranch.scc0", vec![Arg::Str(skip.clone())])?;
                l.skip = Some(skip);
            }
            self.place_label(&l.top.clone(), false)?;
            if self.options.audit {
                2
            } else {
                1
            }
        };
        l.names = self.name_counts();
        l.n0 = self.rows.len();
        l.copies = copies;
        self.flow.loops.push(l);
        Ok(copies)
    }

    fn emit_plain(&mut self, op: &str, args: Vec<Arg>) -> Result<()> {
        let i = self.make_plain(op, args, "")?;
        self.emit(i, &[]).map(|_| ())
    }

    /// The loop's index, if counted.
    pub fn loop_index(&self) -> Option<Typed> {
        self.flow.loops.last().and_then(|l| l.i.clone())
    }

    /// Starts copy k of the body.
    pub fn loop_copy_begin(&mut self, k: u32) -> Result<()> {
        let l = self.flow.loops.last().expect("an open loop").clone();
        self.rewind_names(l.names.clone());
        crate::lanes::cut_exit(self);
        if l.peel > 0 {
            self.flow.loops.last_mut().expect("a loop").pass_start = self.rows.len();
            if l.kind == LoopKind::Counted {
                let out = l.out.clone().expect("an out label");
                let i = reg(l.i.as_ref().expect("an index"));
                self.emit_plain("s.cmp.lt.u32", vec![i, l.n.clone().expect("a count")])?;
                self.emit_plain("s.cbranch.scc0", vec![Arg::Str(out)])?;
            }
            return Ok(());
        }
        if k == 1 {
            let st = self.copy_current();
            let first = self.flow.loops.last().expect("a loop").pass_start;
            let lines = self.audit_lines(first, self.rows.len());
            self.flow.loops.last_mut().expect("a loop").audit = Some((lines, st));
        }
        self.open_body(&l.name);
        let start = self.rows.len();
        self.flow.loops.last_mut().expect("a loop").pass_start = start;
        Ok(())
    }

    /// Ends copy k of the body.
    pub fn loop_copy_end(&mut self, k: u32) -> Result<()> {
        let l = self.flow.loops.last().expect("an open loop").clone();
        if l.peel > 0 {
            if l.kind == LoopKind::Counted {
                let i = reg(l.i.as_ref().expect("an index"));
                self.emit_plain("s.add.u32", vec![i.clone(), i, Arg::Int(1)])?;
            }
            self.settle()?;
            crate::lanes::cut_exit(self);
            let first = (l.pass_start < self.rows.len()).then(|| self.rows[l.pass_start].id);
            self.flow
                .loops
                .last_mut()
                .expect("a loop")
                .starts
                .push(first);
            return Ok(());
        }
        self.settle()?;
        crate::lanes::cut_exit(self);
        self.close_body(false)?;
        if k == 0 {
            if l.pass_start < self.rows.len() {
                let first = self.rows[l.pass_start].id;
                let last = self.rows.last().expect("a row").id;
                self.loops.push(LoopRec {
                    stem: l.name.clone(),
                    first,
                    last,
                    conservative: l.conservative,
                });
            }
            return Ok(());
        }
        let (first_lines, st) = l.audit.clone().expect("the first pass's lines");
        let second = self.audit_lines(l.pass_start, self.rows.len());
        self.truncate(l.pass_start);
        self.state = st;
        if first_lines != second {
            let k = (0..first_lines.len().max(second.len()))
                .find(|&k| first_lines.get(k) != second.get(k))
                .unwrap_or(0);
            let window = |v: &[String]| -> String {
                let lo = k.saturating_sub(2);
                let hi = (k + 3).min(v.len());
                let items: Vec<String> = v
                    .get(lo..hi)
                    .unwrap_or(&[])
                    .iter()
                    .map(|s| crate::maker::py_repr(s))
                    .collect();
                format!("[{}]", items.join(", "))
            };
            return schedule(format!(
                "loop body {} has no steady state: its second iteration differs at {k}: {} vs {}",
                l.name,
                window(&first_lines),
                window(&second)
            ));
        }
        Ok(())
    }

    /// The body left by a Python `break`.
    pub fn loop_broken(&mut self) -> Result<()> {
        let l = self.flow.loops.pop().expect("an open loop");
        self.flow.broken = Some(l.name);
        self.close_body(true)
    }

    /// Closes a loop: its epilogue and its break label.
    pub fn loop_end(&mut self) -> Result<()> {
        let l = self.flow.loops.pop().expect("an open loop");
        let top = l.top.clone();
        match (l.peel > 0, l.kind) {
            (true, _) => {
                // `last` is the last copy's last instruction,
                // so the peel keeps the back branch and the end label.
                let last = self.rows.last().map(|r| r.id);
                self.emit_plain("s.branch", vec![Arg::Str(top.clone())])?;
                self.place_label(&l.out.clone().expect("an out label"), false)?;
                if !l.starts.iter().all(Option::is_none) {
                    self.deferred.push(crate::builder::Deferred::Peel {
                        stem: l.name.clone(),
                        label: top,
                        max: l.peel,
                        starts: l.starts.clone(),
                        last,
                    });
                }
            }
            (false, LoopKind::Counted) => {
                let i = reg(l.i.as_ref().expect("an index"));
                self.emit_plain("s.add.u32", vec![i.clone(), i.clone(), Arg::Int(1)])?;
                self.emit_plain("s.cmp.lt.u32", vec![i, l.n.clone().expect("a count")])?;
                self.emit_plain("s.cbranch.scc1", vec![Arg::Str(top)])?;
                if let Some(skip) = &l.skip {
                    self.place_label(skip, false)?;
                }
            }
            (false, LoopKind::Until) => {
                let c = Cond::Tree(l.until.clone().expect("a condition"));
                let display = crate::expr::text::top(l.until.as_ref().expect("a condition"));
                self.branch_if(&c, false, &top, &format!("loop(until={display})"), None)?;
            }
            (false, LoopKind::Repeat) => self.emit_plain("s.branch", vec![Arg::Str(top)])?,
        }
        if let Some(end) = &l.end {
            self.put_label(end, true)?;
        }
        Ok(())
    }

    /// `lp.break_when(c, comment)`.
    pub fn loop_break(
        &mut self,
        cond: &Cond,
        display: &str,
        comment: Option<String>,
    ) -> Result<()> {
        let l = self.flow.loops.last().expect("an open loop").clone();
        let end = match &l.end {
            Some(e) => e.clone(),
            None => {
                let e = l
                    .end_given
                    .clone()
                    .unwrap_or_else(|| self.new_label(&format!("{}_end", l.name)));
                self.flow.loops.last_mut().expect("a loop").end = Some(e.clone());
                e
            }
        };
        self.settle()?;
        self.branch_if(cond, true, &end, &what("break_when", display), comment)
    }

    /// `b.steady_state_check(body, stem, conservative)` for a hand-written
    /// loop: opens one pass of the body protocol.
    pub fn steady_begin(&mut self, stem: &str, conservative: bool) -> Result<u32> {
        self.settle()?;
        let l = LoopState {
            name: stem.to_owned(),
            kind: LoopKind::Repeat,
            i: None,
            n: None,
            until: None,
            peel: 0,
            end_given: None,
            end: None,
            top: String::new(),
            skip: None,
            out: None,
            conservative,
            names: self.name_counts(),
            n0: self.rows.len(),
            pass_start: self.rows.len(),
            starts: Vec::new(),
            audit: None,
            copies: if self.options.audit { 2 } else { 1 },
        };
        let copies = l.copies;
        self.flow.loops.push(l);
        Ok(copies)
    }

    /// Ends a hand-written loop's protocol.
    pub fn steady_end(&mut self) {
        self.flow.loops.pop();
    }

    /// The audit's lines of rows a..b.
    fn audit_lines(&self, a: usize, b: usize) -> Vec<String> {
        let mut out = Vec::new();
        for row in &self.rows[a..b.min(self.rows.len())] {
            let i = &row.ins;
            let f = i.form();
            if f.kind == Kind::Label || f.kind == Kind::Nop || f.kind == Kind::Wait {
                continue;
            }
            if f.kind == Kind::Comment && !i.is_wait() {
                continue;
            }
            if i.is_wait() {
                out.push(format!("wait for {}", self.request_text(i)));
                continue;
            }
            out.push(self.passed(i));
        }
        out
    }

    /// A marker's request as the audit spells it.
    fn request_text(&self, i: &Instr) -> String {
        let regs: Vec<String> = i
            .ops()
            .iter()
            .filter_map(|o| o.reg())
            .map(|r| self.repr(r))
            .collect();
        let field = |n: &str| i.field_named(n).map_or(String::new(), |v| v.py_str());
        format!(
            "{} {} {} {}",
            field("counter"),
            i.form().name,
            regs.join(", "),
            field("regions")
        )
    }

    /// `passed(i)`: the line by names, label serials masked, a counter
    /// form's answers after `@`.
    pub fn passed(&self, i: &Instr) -> String {
        let regs: Vec<String> = i
            .ops()
            .iter()
            .filter_map(|o| o.reg())
            .map(|r| self.name_text(r))
            .collect();
        let line = thin_ir::print::line(i, &regs).unwrap_or_default();
        let line = thin_ir::print::mask_serials(&line);
        let mut answers: Vec<String> = i.attrs().places.iter().map(|p| p.to_string()).collect();
        if let Some(a) = &i.attrs().lds_access {
            answers.extend(crate::lds::tags_of(self, a.region));
        }
        answers.sort();
        answers.dedup();
        match (i.form().counter.is_some(), answers.is_empty()) {
            (true, false) => format!("{line} @{}", answers.join(",")),
            _ => line,
        }
    }

    /// A register by name.
    pub fn name_text(&self, r: &thin_ir::Reg) -> String {
        let Some(v) = usize::try_from(r.value)
            .ok()
            .and_then(|v| self.values.get(v))
        else {
            return self.repr(r);
        };
        let group = self
            .file(v.file)
            .group(&v.name)
            .map_or(1, |g| g.values.len());
        let named = thin_ir::print::Named {
            file: v.file,
            name: &v.name,
            index: v.index,
            group,
            lo: r.lo,
            n: r.n,
            value_n: v.n,
            short: (!v.where_.is_empty()).then_some(&*v.where_),
        };
        thin_ir::print::name_text(&named)
    }
}

fn reg(v: &Typed) -> Arg {
    Arg::Reg(v.reg(), 0)
}
