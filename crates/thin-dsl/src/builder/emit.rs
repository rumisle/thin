//! Emission: the steps every instruction passes.

use thin_ir::{Attr, Instr, Operand};
use thin_target::{Flags, Kind};

use super::Builder;
use crate::error::{mem, schedule, Result};
use crate::maker::Stage;

impl Builder {
    /// `b.emit(x, abl)` for an instruction; returns its row, or
    /// none when it was ablated.
    pub fn emit(&mut self, i: Instr, abl: &[String]) -> Result<Option<u64>> {
        self.emit_at(i, abl, 0)
    }

    /// Emission with the instruction's own location (0 takes the
    /// operation's).
    pub fn emit_at(&mut self, mut i: Instr, abl: &[String], loc: u32) -> Result<Option<u64>> {
        self.check_open()?;
        self.settle()?;
        i = self.notes(i, abl)?;
        let classes: Vec<String> = i.attrs().ablation.iter().map(|c| c.to_string()).collect();
        if self.ablated(&classes) {
            return self.drop_ablated(&i, &classes).map(|_| None);
        }
        if i.attrs().staged.is_some() {
            i = self.stage(i)?;
        }
        self.widths(&i)?;
        self.check_held(&i)?;
        i = crate::waits::before(self, i)?;
        i = crate::lanes::check(self, i)?;
        self.lds_names(&i)?;
        let loc = if loc == 0 { self.loc } else { loc };
        let kind = i.form().kind;
        let id = self.push(i, loc);
        if !matches!(kind, Kind::Label | Kind::Comment) {
            self.close_forks(id);
        }
        let i = self.rows.last().expect("the row").ins.clone();
        crate::waits::after(self, &i);
        if let Some(c) = i.form().counter {
            self.state.counts[c] += 1;
        }
        if kind == Kind::Branch {
            self.branch_rule(&i);
        } else if i.form().flags.has(Flags::ENDPGM) {
            self.unreachable();
            crate::lanes::dead_exit_at_end(self);
        }
        Ok(Some(id))
    }

    /// the ablation classes and the open alias pairs.
    fn notes(&self, mut i: Instr, abl: &[String]) -> Result<Instr> {
        // The classes of the open ablatable() blocks are the instruction's
        // own too: a variant that keeps it still records what could drop it.
        let open = self.open_abl.iter().flatten();
        if !abl.is_empty() || self.open_abl.iter().any(|c| !c.is_empty()) {
            let mut all: Vec<String> = i.attrs().ablation.iter().map(|c| c.to_string()).collect();
            all.extend(abl.iter().cloned());
            all.extend(open.cloned());
            all.sort();
            all.dedup();
            i = i.with_attr(Attr::Ablation(
                all.iter().map(|c| c.as_str().into()).collect(),
            ))?;
        }
        let open: Vec<(String, Option<String>)> = self.alias.iter().flatten().cloned().collect();
        if !open.is_empty() {
            let mut pairs = i.attrs().alias_ok.clone();
            pairs.extend(open.into_iter().map(|(r, o)| (r.into(), o.map(Into::into))));
            i = i.with_attr(Attr::AliasOk(pairs))?;
        }
        Ok(i)
    }

    /// a dropped instruction (control flow refused).
    fn drop_ablated(&mut self, i: &Instr, classes: &[String]) -> Result<()> {
        let f = i.form();
        if matches!(f.kind, Kind::Branch | Kind::Label) {
            let mut all: Vec<String> = classes
                .iter()
                .chain(self.open_abl.iter().flatten())
                .cloned()
                .collect();
            all.sort();
            all.dedup();
            let quoted: Vec<String> = all.iter().map(|c| crate::maker::py_repr(c)).collect();
            return schedule(format!(
                "{}: ablated ([{}]) - control flow is not ablatable: mark the instructions to drop, not the block \
                 around them",
                self.text(i),
                quoted.join(", ")
            ));
        }
        if f.kind == Kind::Vmem {
            if let Some(tag) = &i.attrs().tag {
                self.requests.issue_tag(tag.clone());
            }
        }
        Ok(())
    }

    /// the staging moves.
    fn stage(&mut self, i: Instr) -> Result<Instr> {
        let stage = crate::maker::families::staged_of(&i).expect("a native stage");
        let lit = self.lit()?;
        let (op, args, comment) = match stage {
            Stage::Mov(off) => (
                "s.mov.b32",
                vec![
                    crate::maker::Arg::Reg(lit, 0),
                    crate::maker::Arg::Str(format!("0x{:x}", off & 0xFFFF_FFFF)),
                ],
                "(staged: no room in the encoding)",
            ),
            Stage::Add(r, off) => (
                "s.add.u32",
                vec![
                    crate::maker::Arg::Reg(lit, 0),
                    crate::maker::Arg::Reg(r, 0),
                    crate::maker::Arg::Str(format!("0x{off:x}")),
                ],
                "(staged: offset past the buffer immediate)",
            ),
        };
        let mov = self.make_plain(op, args, comment)?;
        self.emit(mov, &[])?;
        let ops = (i.ops().iter())
            .map(|o| match o {
                Operand::Reg(r) if r.value == crate::maker::families::LIT => Operand::Reg(lit),
                o => o.clone(),
            })
            .collect();
        Ok(i.with_ops(ops))
    }

    /// registers the slots' SGPR widths exclude.
    fn widths(&self, i: &Instr) -> Result<()> {
        let errors = i.width_errors(|k| i.ops().get(k).map_or(String::new(), |o| self.op_repr(o)));
        if errors.is_empty() {
            return Ok(());
        }
        let regs: Vec<String> = i
            .ops()
            .iter()
            .filter_map(Operand::reg)
            .map(|r| self.repr(r))
            .collect();
        schedule(format!(
            "`{}` ({}): {} (the widths of the role table's forms, the target's operand table: a 64-bit operand is \
             an SGPR pair, allocated aligned)",
            self.text(i),
            regs.join(", "),
            errors.join("; ")
        ))
    }

    /// An operand's repr: a register's, else its text.
    fn op_repr(&self, o: &Operand) -> String {
        match o {
            Operand::Reg(r) => self.repr(r),
            Operand::Phys(p) => p.name.to_owned(),
            Operand::Imm(m) => match (&m.value, m.text.is_empty()) {
                (_, false) => m.text.to_string(),
                (thin_ir::ImmValue::Int(v), true) => v.to_string(),
                (thin_ir::ImmValue::Float(x), true) => crate::maker::py_float(*x),
            },
            Operand::Block(b) => b.to_string(),
        }
    }

    /// a def on held dwords.
    pub(super) fn check_held(&self, i: &Instr) -> Result<()> {
        if self.held.is_empty() {
            return Ok(());
        }
        for r in i.defs() {
            if let Some(label) = self.dwords(r).iter().find_map(|d| self.held.get(d)) {
                return schedule(format!(
                    "{}: writes {}, held by {label}",
                    self.text(i),
                    self.repr(r)
                ));
            }
        }
        Ok(())
    }

    /// an LDS access names its region in a kernel with a layout.
    fn lds_names(&self, i: &Instr) -> Result<()> {
        if self.layout.is_none() || i.attrs().lds_access.is_some() {
            return Ok(());
        }
        if i.form().flags.has(Flags::LDS_MEM) {
            return mem(format!(
                "`{}`: an LDS access names its region in a kernel with a layout - its address a region's \
                 (O.ds_read(d, A + t))",
                self.text(i)
            ));
        }
        Ok(())
    }
}
