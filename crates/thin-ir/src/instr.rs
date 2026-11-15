//! Instructions: immutable data with facts derived once.

use std::sync::Arc;

use thin_target::{DstHalf, DstSel, Flags, Form, Kind, Phys, Pseudo, Target, PHYS};

use crate::attr::{Attr, AttrClass, Attrs};
use crate::field::{FieldId, FieldValue, Tag};
use crate::operand::{Operand, Reg};
use crate::Error;

/// The read role bit.
pub const R: u8 = 1;
/// The write role bit.
pub const W: u8 = 2;

/// Named-register numbers the IR tests for.
const EXEC: usize = 0;
const VCC: usize = 3;
const M0: usize = 6;

/// What a wait marker asks for.
#[derive(Clone, Debug, PartialEq)]
pub enum Request {
    /// The loads of these registers, and the ops on these regions.
    Regs {
        /// The registers, in operand order.
        regs: Vec<Reg>,
        /// The region names.
        regions: Vec<Arc<str>>,
    },
    /// The VMEM ops that carry one of these tags.
    Tags(Vec<Tag>),
    /// At most this many ops outstanding.
    Left(i128),
    /// Every scalar memory op.
    Smem,
    /// Every op of the counter.
    All,
}

/// An instruction: its form, operands, modifiers, fields and attributes,
/// and the facts derived from them when it was made.
#[derive(Clone, Debug)]
pub struct Instr {
    form: &'static Form,
    ops: Vec<Operand>,
    mods: Option<Box<[u8]>>,
    fields: Vec<(FieldId, FieldValue)>,
    attrs: Attrs,
    roles: Vec<u8>,
    defs: Vec<u16>,
    uses: Vec<u16>,
    writes: Vec<usize>,
    reads: Vec<usize>,
    sgpr_valu_def: bool,
    kept: bool,
}

impl Instr {
    /// An instruction of form `form`, its facts derived.
    pub fn new(
        form: &'static Form,
        ops: Vec<Operand>,
        fields: Vec<(FieldId, FieldValue)>,
        mods: Option<Box<[u8]>>,
    ) -> Instr {
        let mut i = Instr::bare(form, ops);
        i.fields = fields;
        i.mods = mods;
        i.derive();
        i
    }

    /// An instruction with no fields, no attributes and no derived facts,
    /// as a pass makes one.
    pub fn bare(form: &'static Form, ops: Vec<Operand>) -> Instr {
        Instr {
            form,
            ops,
            mods: None,
            fields: Vec::new(),
            attrs: Attrs::default(),
            roles: Vec::new(),
            defs: Vec::new(),
            uses: Vec::new(),
            writes: Vec::new(),
            reads: Vec::new(),
            sgpr_valu_def: false,
            kept: false,
        }
    }

    /// The same instruction with other operands, its fields, modifiers and
    /// attributes kept and its facts derived again.
    pub fn with_ops(&self, ops: Vec<Operand>) -> Instr {
        let mut i = Instr {
            ops,
            ..self.clone()
        };
        i.derive();
        i
    }

    /// The same instruction with other fields, its operands, modifiers and
    /// attributes kept.
    pub fn with_fields(&self, fields: Vec<(FieldId, FieldValue)>) -> Instr {
        let mut i = Instr {
            fields,
            ..self.clone()
        };
        i.derive();
        i
    }

    /// The same instruction with `attr` put.
    pub fn with_attr(&self, attr: Attr) -> Result<Instr, Error> {
        let mut i = self.clone();
        i.attrs.put(attr)?;
        Ok(i)
    }

    /// The same instruction without its attribute of that class.
    pub fn without_attr(&self, class: AttrClass) -> Instr {
        let mut i = self.clone();
        i.attrs.drop(class);
        i
    }

    /// Roles, defs and uses, writes and reads, and `sgpr_valu_def`. A
    /// write of half the destination (`Form::dst_half`) reads it too: its
    /// first operand's role gains the read bit, as an accumulator's has.
    fn derive(&mut self) {
        let form = self.form;
        self.roles = (0..self.ops.len()).map(|k| form.role(k)).collect();
        let whole_dpp = form.dpp && !dpp_reads_old(&self.fields);
        if whole_dpp && self.roles.first() == Some(&(R | W)) {
            self.roles[0] = W;
        }
        self.kept = false;
        if let Some(role) = self.roles.first_mut().filter(|_| form.dst_half.is_some()) {
            self.kept = *role == W;
            if *role & W != 0 {
                *role |= R;
            }
        }
        (self.defs, self.uses) = (Vec::new(), Vec::new());
        for (k, op) in self.ops.iter().enumerate() {
            let Operand::Reg(r) = op else { continue };
            let role = self.roles[k];
            for (bit, list) in [(W, &mut self.defs), (R, &mut self.uses)] {
                if role & bit != 0 && !list.iter().any(|&j| self.ops[j as usize].reg() == Some(r)) {
                    list.push(k as u16);
                }
            }
        }
        // Named registers as their whole register, implicit or explicit
        // alike: a v_cmpx writes `exec` just as `s_mov_b32 exec_lo,..`
        // does, so one test of `writes` sees both.
        let whole = |list: &[usize]| list.iter().map(|&p| PHYS[p].whole).collect::<Vec<_>>();
        self.writes = whole(&form.implicit_defs);
        self.reads = whole(&form.implicit_uses);
        for (k, op) in self.ops.iter().enumerate() {
            let Some(whole) = op.whole() else { continue };
            if self.roles[k] & W != 0 {
                self.writes.push(whole.number);
            }
            if self.roles[k] & R != 0 {
                self.reads.push(whole.number);
            }
        }
        for list in [&mut self.writes, &mut self.reads] {
            list.sort_unstable();
            list.dedup();
        }
        self.sgpr_valu_def = self.derive_sgpr_valu_def();
    }

    /// A VALU that writes an SGPR: a def in `s`, or a write of vcc or exec
    /// with no vector def.
    fn derive_sgpr_valu_def(&self) -> bool {
        let def_in = |files: &str| self.defs().any(|r| files.contains(r.file));
        let mask_write = (self.writes.iter()).any(|&p| [VCC, EXEC].contains(&PHYS[p].whole));
        self.form.kind == Kind::Valu && (def_in("s") || mask_write && !def_in("va"))
    }

    /// Its form.
    pub fn form(&self) -> &'static Form {
        self.form
    }

    /// Its operands, one per explicit slot.
    pub fn ops(&self) -> &[Operand] {
        &self.ops
    }

    /// Its modifier bytes, one per slot, or none.
    pub fn mods(&self) -> Option<&[u8]> {
        self.mods.as_deref()
    }

    /// Its fields, in the order given.
    pub fn fields(&self) -> &[(FieldId, FieldValue)] {
        &self.fields
    }

    /// Its attributes.
    pub fn attrs(&self) -> &Attrs {
        &self.attrs
    }

    /// Each operand's role bits.
    pub fn roles(&self) -> &[u8] {
        &self.roles
    }

    /// The operand indices of its register defs: the first operand of each
    /// distinct written register.
    pub fn def_ops(&self) -> &[u16] {
        &self.defs
    }

    /// The operand indices of its register uses.
    pub fn use_ops(&self) -> &[u16] {
        &self.uses
    }

    /// Its register defs, in order.
    pub fn defs(&self) -> impl Iterator<Item = &Reg> + '_ {
        self.defs.iter().filter_map(|&k| self.ops[k as usize].reg())
    }

    /// Its register uses, in order.
    pub fn uses(&self) -> impl Iterator<Item = &Reg> + '_ {
        self.uses.iter().filter_map(|&k| self.ops[k as usize].reg())
    }

    /// The named registers it writes, by number, ascending.
    pub fn writes(&self) -> &[usize] {
        &self.writes
    }

    /// The named registers it reads, by number, ascending.
    pub fn reads(&self) -> &[usize] {
        &self.reads
    }

    /// Whether it is a VALU that writes an SGPR.
    pub fn sgpr_valu_def(&self) -> bool {
        self.sgpr_valu_def
    }

    /// The field's value, if it has the field.
    pub fn field(&self, id: FieldId) -> Option<&FieldValue> {
        self.fields.iter().find(|(f, _)| *f == id).map(|(_, v)| v)
    }

    /// The field of that name's value.
    pub fn field_named(&self, name: &str) -> Option<&FieldValue> {
        FieldId::named(name).ok().and_then(|id| self.field(id))
    }

    /// Whether operand n has a nonzero modifier byte.
    pub fn modified(&self, n: usize) -> bool {
        self.mods()
            .is_some_and(|m| m.get(n).is_some_and(|&b| b != 0))
    }

    /// Operand n's named register, if it is one.
    pub fn phys(&self, n: usize) -> Option<&'static Phys> {
        self.ops.get(n).and_then(Operand::phys)
    }

    /// Operand n as a plain register: none for a constant, a named register,
    /// a block or a modified operand.
    pub fn reg(&self, n: usize) -> Option<&Reg> {
        self.ops
            .get(n)
            .and_then(Operand::reg)
            .filter(|_| !self.modified(n))
    }

    /// The first block operand's name.
    pub fn block(&self) -> Option<&str> {
        self.ops.iter().find_map(|op| match op {
            Operand::Block(name) => Some(&**name),
            _ => None,
        })
    }

    /// Whether it writes EXEC: a SALU or VALU with CMPX or SAVEEXEC, or
    /// whose operand 0 is a half of exec.
    pub fn writes_exec(&self) -> bool {
        let f = self.form;
        if !matches!(f.kind, Kind::Salu | Kind::Valu) {
            return false;
        }
        f.flags.any(Flags::CMPX | Flags::SAVEEXEC)
            || self
                .ops
                .first()
                .and_then(Operand::whole)
                .is_some_and(|p| p.number == EXEC)
    }

    /// Whether it writes M0.
    pub fn m0_write(&self) -> bool {
        self.writes.contains(&M0)
    }

    /// Whether a VALU writes only part of its destination: by the form's
    /// `dst_sel`, or by `op_sel` past its second source.
    pub fn partial_dst(&self) -> bool {
        match self.form.dst_sel {
            DstSel::Always => true,
            DstSel::Never => false,
            DstSel::OpSel => (self.field(FieldId::of("op_sel")))
                .and_then(FieldValue::bits)
                .is_some_and(|m| m >> 2 != 0),
        }
    }

    /// Whether its write keeps part of its destination's old contents, so
    /// the write reads them too: the one partial-write rule every pass
    /// uses (the dead-write check, the alias check and register
    /// assignment). Lanes EXEC leaves off keep theirs when `exec_partial`
    /// and the op works per lane (VALU, matrix, VMEM, DS); a writelane
    /// writes one lane; a D16 load, a destination select (`dst_sel`,
    /// op_sel's destination bit) or a 16-bit VALU result (`dst_half`)
    /// writes half of each dword.
    pub fn keeps_old(&self, exec_partial: bool) -> bool {
        let f = self.form;
        let lanes = matches!(f.kind, Kind::Valu | Kind::Matrix | Kind::Vmem | Kind::Ds);
        exec_partial && lanes
            || f.flags.any(Flags::WRITELANE | Flags::D16)
            || f.kind == Kind::Valu && self.partial_dst()
            || f.dst_half.is_some()
    }

    /// Whether its write of a destination in register file `file` keeps
    /// part of that destination's old contents: [`Instr::keeps_old`] for
    /// the vector files, never for an SGPR. A VALU writes a whole SGPR
    /// whatever EXEC is (a lane mask's inactive lanes are written 0, a
    /// readlane writes every bit), and no form writes one lane or one half
    /// of an SGPR; a conditional SGPR write (`s_cmov`) reads its
    /// destination through its role instead.
    pub fn keeps_old_in(&self, file: char, exec_partial: bool) -> bool {
        file != 's' && self.keeps_old(exec_partial)
    }

    /// The half of each destination dword it writes: its form's `dst_half`,
    /// with `op_sel`'s destination bit (its last) read. It keeps the other
    /// half, so it reads its destination: its first operand's role is read
    /// and write.
    pub fn dst_half(&self) -> Option<DstHalf> {
        let hi = || match self.field(FieldId::of("op_sel")) {
            Some(FieldValue::Ints(v)) => v.last().is_some_and(|x| x & 1 != 0),
            _ => false,
        };
        match self.form.dst_half? {
            DstHalf::OpSel if hi() => Some(DstHalf::Hi),
            DstHalf::OpSel => Some(DstHalf::Lo),
            half => Some(half),
        }
    }

    /// The destination it reads only for the half it keeps (`dst_half`):
    /// its first operand when its slot only writes it (not an accumulator,
    /// nor a DPP's old value), unless another operand reads it as a
    /// source. That read keeps the old contents live (liveness, def-use,
    /// the dead-write check, waits, hazards) but only carries them
    /// through: a kept half never written is no read-before-write
    /// and no alias read, as with the lanes a
    /// partial EXEC keeps.
    pub fn kept_dst(&self) -> Option<&Reg> {
        let r = self.ops.first().filter(|_| self.kept)?.reg()?;
        let source = (self.ops.iter().zip(&self.roles).skip(1))
            .any(|(op, &role)| role & R != 0 && op.reg() == Some(r));
        (!source).then_some(r)
    }

    /// Whether it is one of the IR's own wait markers.
    pub fn is_wait(&self) -> bool {
        self.form.is_wait()
    }

    /// For a wait marker, its counter and request; a missing or unknown
    /// counter is a `ValueError`.
    pub fn request(&self, target: &Target) -> Result<Option<(usize, Request)>, Error> {
        if !self.is_wait() {
            return Ok(None);
        }
        let name = self
            .field(FieldId::of("counter"))
            .and_then(FieldValue::text);
        let counter = (name.and_then(|n| target.counter_named(n)))
            .ok_or_else(|| Error::Value(format!("`{}`: no counter {name:?}", self.form.name)))?;
        let request = match self.form.pseudo {
            Some(Pseudo::WaitRegs) => Request::Regs {
                regs: self.ops.iter().filter_map(Operand::reg).copied().collect(),
                regions: match self.field(FieldId::of("regions")) {
                    Some(FieldValue::Names(n)) => n.to_vec(),
                    _ => Vec::new(),
                },
            },
            Some(Pseudo::WaitTags) => Request::Tags(match self.field(FieldId::of("tags")) {
                Some(FieldValue::Tags(t)) => t.to_vec(),
                _ => Vec::new(),
            }),
            Some(Pseudo::WaitLeft) => Request::Left(
                self.field(FieldId::of("left"))
                    .and_then(FieldValue::int)
                    .unwrap_or(0),
            ),
            Some(Pseudo::WaitSmem) => Request::Smem,
            _ => Request::All,
        };
        Ok(Some((counter, request)))
    }

    /// One message per scalar register operand whose width its slot's SGPR
    /// widths exclude, on forms that check widths; `repr`
    /// spells operand k.
    pub fn width_errors(&self, repr: impl Fn(usize) -> String) -> Vec<String> {
        if !self.form.checks_widths() {
            return Vec::new();
        }
        let mut out = Vec::new();
        for (k, op) in self.ops.iter().enumerate() {
            let (Some(r), Some(slot)) = (op.reg(), self.form.slot(k)) else {
                continue;
            };
            let widths = &slot.sgpr_widths;
            if r.file != 's' || widths.is_empty() || widths.contains(&r.n) {
                continue;
            }
            let s = if r.n > 1 { "s" } else { "" };
            let takes: Vec<String> = widths.iter().map(u32::to_string).collect();
            let mut msg = format!(
                "{} is {} SGPR{s} where the instruction takes {}",
                repr(k),
                r.n,
                takes.join(" or ")
            );
            if r.n == 2 && widths.iter().max() == Some(&1) {
                msg.push_str(WIDE_HINT);
            }
            out.push(msg);
        }
        out
    }
}

/// The hint after a pair in a one-SGPR slot.
const WIDE_HINT: &str = " (a value an expression made 64-bit - a sum or product whose range \
    passes 2^32 is a u64: tighten a leaf's hi, or take the low dword, x.dword(0))";

/// Whether a DPP control reads the old destination in lanes it does not
/// write.
pub fn dpp_reads_old(fields: &[(FieldId, FieldValue)]) -> bool {
    let get = |name: &str| {
        fields
            .iter()
            .find(|(f, _)| f.name() == name)
            .map(|(_, v)| v)
    };
    let fifteen = |name| matches!(get(name), Some(FieldValue::Int(15)));
    if !fifteen("row_mask") || !fifteen("bank_mask") {
        return true;
    }
    const IN_ROW: [&str; 6] = [
        "quad_perm:",
        "row_mirror",
        "row_half_mirror",
        "row_ror:",
        "row_share:",
        "row_xmask:",
    ];
    if let Some(FieldValue::Text(t)) = get("dpp") {
        if IN_ROW.iter().any(|p| t.starts_with(p)) {
            return false;
        }
    }
    !get("bound_ctrl").is_some_and(FieldValue::truthy)
}
