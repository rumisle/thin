//! LDS accesses, address splitting and region wait tracking.

use thin_ir::{Attr, Datum, HostObject, LdsAccess, Reg};
use thin_target::Mem;

use super::addr::{split, sum, Address, Off};
use crate::builder::{Builder, Make};
use crate::error::{mem, value, Result};
use crate::expr::lower::Opd;
use crate::expr::node::{binop, constant, value_of, Num, Op, Tree};
use crate::expr::text::top;
use crate::maker::families::{Family, Opts, Val};
use crate::maker::{Arg, FArg, Made, Notes};
use crate::types::Ty;

/// An LDS access the maker defers.
#[derive(Clone, Debug)]
pub struct Access {
    name: String,
    args: Vec<Arg>,
    at: usize,
    addr: Address,
    extra: i128,
    ordered: String,
    fields: Vec<(String, FArg)>,
    notes: Notes,
}

/// The maker's LDS access: the first address operand, its `offset` and
/// `ordered` fields taken.
pub fn access(
    name: &str,
    args: Vec<Arg>,
    at: usize,
    fields: Vec<(String, FArg)>,
    notes: Notes,
) -> Result<Access> {
    let Arg::Addr(addr) = args[at].clone() else {
        unreachable!("an address operand")
    };
    let mut extra = 0;
    let mut ordered = String::new();
    let mut rest = Vec::new();
    for (k, v) in fields {
        match (k.as_str(), &v) {
            ("offset", FArg::Int(c)) => extra = *c,
            ("ordered", FArg::Str(s)) => ordered = s.clone(),
            ("offset" | "ordered", FArg::None) => {}
            _ => rest.push((k, v)),
        }
    }
    Ok(Access {
        name: name.to_owned(),
        args,
        at,
        addr,
        extra,
        ordered,
        fields: rest,
        notes,
    })
}

/// The address register and the offset field.
fn addr_reg(b: &mut Builder, mem: Mem, var: Option<Tree>, c: i128) -> Result<(Opd, i128)> {
    // The offset field takes what the target's forms of the class take;
    // the rest goes into the address register.
    let range = b.t.offset_range(mem).unwrap_or(0..=0);
    let (tree, field) = if !range.contains(&c) {
        let t = match var {
            None => constant(Num::Int(c)),
            Some(v) => binop(Op::Add, v, constant(Num::Int(c)))?,
        };
        (t, 0)
    } else {
        if let Some(v) = var.as_ref().and_then(|v| value_of(v)) {
            if v.file == 'v' && v.n == 1 && v.width() == 1 && v.ty.is_int() {
                return Ok((Opd::Reg(v.clone(), 0), c));
            }
        }
        (var.unwrap_or_else(|| constant(Num::Int(0))), c)
    };
    let tree = match tree.ty {
        Some(Ty::I32) => crate::expr::funcs::cast(Ty::U32, tree)?,
        _ => tree,
    };
    let m = Make {
        n: Some(1),
        n_repr: "1".into(),
        name: Some("adr".into()),
        where_: b.where_text.clone(),
        ..Make::default()
    };
    let r = b.make_value('v', Ty::U32, m)?;
    let comment = format!("adr = {}", top(&tree));
    b.assign(&r, tree, Some(comment))?;
    Ok((Opd::Reg(r, 0), field))
}

fn access_object(
    b: &Builder,
    region: usize,
    what: &str,
    var: bool,
    field: i128,
    ordered: &str,
) -> LdsAccess {
    let name = b
        .layout
        .as_ref()
        .map_or(String::new(), |l| l.regions[region].name.clone());
    let d = Datum::Tuple(vec![
        Datum::Int(region as i128),
        Datum::Str(what.into()),
        Datum::Int(field),
        Datum::Int(i128::from(var)),
        Datum::Str(ordered.into()),
    ]);
    let repr = format!("LdsAccess({name}, {what:?})");
    LdsAccess {
        object: HostObject::of_datum(d, repr),
        region: region as u64,
        field,
        what: what.into(),
        ordered: ordered.into(),
    }
}

impl Builder {
    fn own_layout(&self, a: &Address) -> Result<()> {
        match self.layout.as_ref() {
            Some(l) if l.builder == a.builder => Ok(()),
            _ => mem(format!(
                "{}: a region of another Builder's layout",
                a.text()
            )),
        }
    }

    /// Emits an LDS access.
    pub fn emit_access(&mut self, x: Access, abl: &[String]) -> Result<Option<u64>> {
        self.own_layout(&x.addr)?;
        self.frozen_layout()?;
        let base = self.layout.as_ref().expect("a layout").regions[x.addr.region].base;
        let (var, c) = split(&x.addr.off);
        let c = c + i128::from(base) + x.extra;
        let mut text = x.addr.text();
        if x.extra != 0 {
            text = format!("{text} + {}", x.extra);
        }
        let has_var = var.is_some();
        let (reg, c) = addr_reg(self, Mem::Ds, var, c)?;
        let mut args = x.args.clone();
        args[x.at] = reg.arg();
        let mut fields = x.fields.clone();
        if c != 0 {
            fields.push(("offset".into(), FArg::Int(c)));
        }
        let mut notes = x.notes.clone();
        notes.comment = match notes.comment.is_empty() {
            true => text.clone(),
            false => format!("{text}: {}", notes.comment),
        };
        let mut i = self.make(&x.name, args, fields, notes)?;
        let what = format!("{} {text}", i.form().op);
        i = i.with_attr(Attr::LdsAccess(access_object(
            self,
            x.addr.region,
            &what,
            has_var,
            c,
            &x.ordered,
        )))?;
        let counted = i.form().counter;
        let id = self.emit(i, abl)?;
        if let (Some(id), Some(ctr)) = (id, counted) {
            if self.rows.last().map(|r| r.id) == Some(id) {
                self.record_region_op(x.addr.region, ctr);
            }
        }
        Ok(id)
    }

    /// The region-op record: per tag, the count at its last op.
    fn record_region_op(&mut self, region: usize, counter: usize) {
        let tags = self.layout.as_ref().expect("a layout").tags(region);
        let n = self.state.counts[counter];
        for t in tags {
            self.state.mem.insert((counter, t.into()), n);
        }
    }
}

/// `b.wait(A)`: the region's records on the current path.
pub fn wait_region(b: &mut Builder, name: &str) -> Result<()> {
    let any = b.state.mem.keys().any(|(_, r)| **r == *name);
    if !any && b.ablate.is_empty() {
        return mem(format!(
            "wait({name}): no op on it issued on any path to here (nothing to wait for)"
        ));
    }
    Ok(())
}

/// Which family a split object came from.
#[derive(Clone, Debug)]
pub enum SplitKind {
    /// A global load or store, with its base when given.
    Global {
        /// A store.
        store: bool,
        /// The base given.
        base: Option<Reg>,
    },
    /// A buffer load or store.
    Buffer {
        /// A store.
        store: bool,
        /// The resource.
        desc: Reg,
        /// The scalar offset.
        soffset: Val,
    },
    /// A scalar load or store.
    Scalar,
}

/// An address-split object.
#[derive(Clone, Debug)]
pub struct Split {
    kind: SplitKind,
    name: String,
    data: Reg,
    addr: Val,
    offset: Val,
    opts: Opts,
}

impl Split {
    /// A split of a family's operation.
    pub fn new(
        kind: SplitKind,
        name: String,
        data: Reg,
        addr: Val,
        offset: Val,
        opts: Opts,
    ) -> Split {
        Split {
            kind,
            name,
            data,
            addr,
            offset,
            opts,
        }
    }
}

/// `pointer_sum(addr)`: one scalar pointer value and u32 terms.
fn pointer_sum(t: &Tree) -> Option<(crate::values::Typed, Off)> {
    let mut terms = Vec::new();
    flatten(t, &mut terms);
    let ptrs: Vec<usize> = (0..terms.len())
        .filter(|&k| {
            value_of(&terms[k]).is_some_and(|v| v.ty.is_wide() && v.file == 's' && v.n == 1)
        })
        .collect();
    let [p] = ptrs.as_slice() else { return None };
    let ptr = value_of(&terms[*p]).expect("a pointer").clone();
    let rest: Vec<Off> = (terms.iter().enumerate())
        .filter(|(k, _)| k != p)
        .map(
            |(_, t)| match crate::expr::node::const_of(t).and_then(Num::int) {
                Some(c) if t.ty.is_none() => Off::Int(c),
                _ => Off::Tree(t.clone()),
            },
        )
        .collect();
    if rest
        .iter()
        .any(|o| matches!(o, Off::Tree(t) if !t.ty.is_none_or(|t| t == Ty::U32)))
    {
        return None;
    }
    Some((ptr, sum(rest)))
}

fn flatten(t: &Tree, out: &mut Vec<Tree>) {
    if t.op == Op::Add && t.ty.is_some_and(Ty::is_wide) {
        flatten(&t.args[0], out);
        flatten(&t.args[1], out);
    } else {
        out.push(t.clone());
    }
}

fn off_of(v: &Val) -> Option<Off> {
    match v {
        Val::Int(c) => Some(Off::Int(*c)),
        Val::Tree(t) => Some(Off::Tree(t.clone())),
        Val::Reg(_) | Val::None | Val::Str(_) | Val::Addr(_) => None,
    }
}

impl Builder {
    /// Emits a split object.
    pub fn emit_split(&mut self, s: Split, abl: &[String]) -> Result<Option<u64>> {
        match s.kind.clone() {
            SplitKind::Global { store, base } => self.split_global(s, store, base, abl),
            SplitKind::Buffer {
                store,
                desc,
                soffset,
            } => self.split_buffer(s, store, desc, soffset, abl),
            SplitKind::Scalar => self.split_scalar(s, abl),
        }
    }

    fn split_comment(&self, label: Option<&str>, off: &Off, comment: &str) -> String {
        let text = match label {
            Some(l) => format!("{l} + {}", off.text()),
            None => off.text(),
        };
        match comment.is_empty() {
            true => text,
            false => format!("{text}: {comment}"),
        }
    }

    fn split_global(
        &mut self,
        s: Split,
        store: bool,
        base: Option<Reg>,
        abl: &[String],
    ) -> Result<Option<u64>> {
        let (ptr, label, addr) = match base {
            Some(b) => (b, None, off_of(&s.addr).expect("a global address")),
            // A bare scalar pointer: the base, at offset 0.
            None if matches!(&s.addr, Val::Reg(r) if r.file == 's') => {
                let Val::Reg(r) = &s.addr else {
                    unreachable!("a register address")
                };
                (*r, None, Off::Int(0))
            }
            None => {
                let Val::Tree(t) = &s.addr else {
                    unreachable!("a tree address")
                };
                match pointer_sum(t) {
                    Some((p, off)) => (p.reg(), p.label.as_deref().map(str::to_owned), off),
                    None => {
                        return value(format!(
                            "{}: a global address is a scalar pointer plus u32 terms",
                            top(t)
                        ))
                    }
                }
            }
        };
        let off = match &s.offset {
            Val::Int(0) => addr,
            o => sum(vec![addr, off_of(o).unwrap_or(Off::Int(0))]),
        };
        let (var, c) = split(&off);
        let (reg, c) = addr_reg(self, Mem::Global, var, c)?;
        let comment = self.split_comment(label.as_deref(), &off, &s.opts.comment);
        let args = match store {
            true => vec![reg.arg(), Arg::Reg(s.data, 0), Arg::Reg(ptr, 0)],
            false => vec![Arg::Reg(s.data, 0), reg.arg(), Arg::Reg(ptr, 0)],
        };
        let mut fields = Vec::new();
        if c != 0 {
            fields.push(("offset".to_owned(), FArg::Int(c)));
        }
        fields.extend(s.opts.cache.fields());
        let notes = Notes {
            comment,
            tag: s.opts.tag.clone(),
            ..Notes::default()
        };
        let i = self.make(&s.name, args, fields, notes)?;
        self.emit(i, abl)
    }

    fn split_buffer(
        &mut self,
        s: Split,
        store: bool,
        desc: Reg,
        soffset: Val,
        abl: &[String],
    ) -> Result<Option<u64>> {
        let addr = off_of(&s.addr).expect("a buffer offset");
        let off = match &s.offset {
            Val::Int(0) => addr,
            o => sum(vec![addr, off_of(o).unwrap_or(Off::Int(0))]),
        };
        let (var, c) = split(&off);
        let (reg, c) = addr_reg(self, Mem::Buffer, var, c)?;
        let reg = match reg {
            Opd::Reg(v, _) => v.reg(),
            _ => unreachable!("a register"),
        };
        let opts = Opts {
            offset: Val::Int(c),
            comment: self.split_comment(None, &off, &s.opts.comment),
            ..s.opts.clone()
        };
        let c = crate::maker::Ctx {
            t: self.t,
            values: &self.values,
        };
        let made = crate::maker::families::buffer_op(
            &c,
            &s.name,
            store,
            &s.data,
            Val::Reg(reg),
            Val::Reg(desc),
            soffset,
            opts,
        )?;
        let Made::Instr(i) = made else {
            unreachable!("a buffer op")
        };
        self.emit(*i, abl)
    }

    fn split_scalar(&mut self, s: Split, abl: &[String]) -> Result<Option<u64>> {
        let (base, label, off) = match &s.addr {
            Val::Tree(t) => {
                let Some((p, t)) = pointer_sum(t) else {
                    return value(format!(
                        "{}: a scalar address is a scalar pointer plus u32 terms",
                        top(t)
                    ));
                };
                let off = match &s.offset {
                    Val::Int(0) => t,
                    o => sum(vec![t, off_of(o).unwrap_or(Off::Int(0))]),
                };
                (p.reg(), p.label.as_deref().map(str::to_owned), off)
            }
            Val::Reg(r) => (*r, None, off_of(&s.offset).unwrap_or(Off::Int(0))),
            _ => unreachable!("a scalar base"),
        };
        let (var, c) = split(&off);
        let operand = match (&var, c) {
            (None, c)
                if c >= 0
                    && self
                        .t
                        .offset_range(Mem::Smem)
                        .is_some_and(|r| r.contains(&c)) =>
            {
                Arg::Str(format!("0x{c:x}"))
            }
            (Some(v), 0) if value_of(v).is_some_and(|x| x.file == 's' && x.width() == 1) => {
                Arg::Reg(value_of(v).expect("a value").reg(), 0)
            }
            _ => {
                let tree = match off {
                    Off::Tree(ref t) if t.ty == Some(Ty::I32) => {
                        Off::Tree(crate::expr::funcs::cast(Ty::U32, t.clone())?)
                    }
                    ref o => o.clone(),
                };
                let tree = match tree {
                    Off::Int(c) => constant(Num::Int(c)),
                    Off::Tree(t) => t,
                };
                let m = Make {
                    n: Some(1),
                    n_repr: "1".into(),
                    name: Some("soff".into()),
                    where_: self.where_text.clone(),
                    ..Make::default()
                };
                let r = self.make_value('s', Ty::U32, m)?;
                let comment = format!("soff = {}", top(&tree));
                self.assign(&r, tree, Some(comment))?;
                Arg::Reg(r.reg(), 0)
            }
        };
        let comment = self.split_comment(label.as_deref(), &off, &s.opts.comment);
        let fields = s.opts.cache.fields();
        let notes = Notes {
            comment,
            tag: s.opts.tag.clone(),
            ..Notes::default()
        };
        let i = self.make(
            &s.name,
            vec![Arg::Reg(s.data, 0), Arg::Reg(base, 0), operand],
            fields,
            notes,
        )?;
        self.emit(i, abl)
    }

    /// Emits what a family made.
    pub fn emit_family(&mut self, f: Family, abl: &[String]) -> Result<Option<u64>> {
        match f {
            Family::Made(Made::Instr(i)) => self.emit(*i, abl),
            Family::Made(Made::Lds(a)) => self.emit_access(*a, abl),
            Family::Split(s) => self.emit_split(*s, abl),
            Family::Dma(d) => self.emit_dma(*d, abl),
        }
    }
}

/// Which VMEM family an LDS DMA came from.
#[derive(Clone, Debug)]
pub enum DmaKind {
    /// `global_load(lds, addr, base=...)`.
    Global {
        /// Scalar base SGPR pair or `Val::None`.
        base: Val,
    },
    /// `buffer_load(lds, voffset, desc, soffset=...)`.
    Buffer {
        /// Buffer resource descriptor (4 SGPRs).
        desc: Val,
        /// Scalar byte offset.
        soffset: Val,
    },
}

/// A VMEM-to-LDS DMA deferred until emission so `M0`, address splitting,
/// `LdsAccess`, and `vmcnt` region tracking are emitted together.
#[derive(Clone, Debug)]
pub struct Dma {
    kind: DmaKind,
    op_name: String,
    lds: Address,
    addr: Val,
    offset: Val,
    opts: Opts,
}

impl Dma {
    /// Creates a deferred VMEM-to-LDS DMA.
    pub fn new(
        kind: DmaKind,
        op_name: String,
        lds: Address,
        addr: Val,
        offset: Val,
        opts: Opts,
    ) -> Dma {
        Dma {
            kind,
            op_name,
            lds,
            addr,
            offset,
            opts,
        }
    }
}

fn has_lane_value(t: &Tree) -> bool {
    value_of(t).is_some_and(|v| matches!(v.file, 'v' | 'a')) || t.args.iter().any(has_lane_value)
}

fn reg_of_val(v: &Val) -> Option<Reg> {
    match v {
        Val::Reg(r) => Some(*r),
        _ => None,
    }
}

impl Builder {
    /// Emits an M0 setup followed by a global- or buffer-to-LDS DMA.
    pub fn emit_dma(&mut self, d: Dma, abl: &[String]) -> Result<Option<u64>> {
        self.own_layout(&d.lds)?;
        self.frozen_layout()?;
        let base = self.layout.as_ref().expect("a layout").regions[d.lds.region].base;
        let (lds_var, lds_c) = split(&d.lds.off);
        let lds_text = d.lds.text();
        if lds_var.as_ref().is_some_and(has_lane_value) {
            return mem(format!(
                "{lds_text}: an LDS DMA destination is wave-uniform (set in M0; each lane writes M0 + 4*lane), \
                 not lane-varying"
            ));
        }
        let m0_const = i128::from(base) + lds_c;
        if lds_var.is_none() {
            if m0_const < 0 {
                return mem(format!("{lds_text}: negative LDS byte offset {m0_const}"));
            }
            if m0_const % 4 != 0 {
                return mem(format!(
                    "{lds_text}: LDS DMA destination offset {m0_const} is not 4-byte aligned"
                ));
            }
        }
        if self.ablated(abl) {
            if let Some(tag) = &d.opts.tag {
                self.requests.issue_tag(tag.clone());
            }
            return Ok(None);
        }
        self.emit_dma_m0(&lds_text, lds_var.clone(), m0_const, abl)?;
        let mut i = match d.kind.clone() {
            DmaKind::Global { base } => self.make_global_dma(&d, &lds_text, base)?,
            DmaKind::Buffer { desc, soffset } => {
                self.make_buffer_dma(&d, &lds_text, desc, soffset)?
            }
        };
        let what = format!("{} {lds_text}", i.form().op);
        i = i.with_attr(Attr::LdsAccess(access_object(
            self,
            d.lds.region,
            &what,
            lds_var.is_some(),
            m0_const,
            &d.opts.ordered,
        )))?;
        let counted = i.form().counter;
        let id = self.emit(i, abl)?;
        if let (Some(id), Some(ctr)) = (id, counted) {
            if self.rows.last().map(|r| r.id) == Some(id) {
                self.record_region_op(d.lds.region, ctr);
            }
        }
        Ok(id)
    }

    fn emit_dma_m0(
        &mut self,
        lds_text: &str,
        lds_var: Option<Tree>,
        m0_const: i128,
        abl: &[String],
    ) -> Result<()> {
        let m0 = Arg::Phys(thin_target::phys_named("m0").expect("m0"), 0);
        let comment = format!("m0 = {lds_text}");
        let i = match lds_var {
            None => {
                let mov = self.make_plain(
                    "s.mov.b32",
                    vec![m0, Arg::Str(format!("0x{m0_const:x}"))],
                    &comment,
                )?;
                mov.with_attr(Attr::M0Set(m0_const))?
            }
            Some(var) => {
                let sreg = match value_of(&var)
                    .filter(|v| v.file == 's' && v.n == 1 && v.width() == 1 && v.ty.is_int())
                {
                    Some(v) => v.reg(),
                    None => {
                        let tree_text = top(&var);
                        let tree = match var.ty {
                            Some(Ty::I32) => crate::expr::funcs::cast(Ty::U32, var)?,
                            _ => var,
                        };
                        let m = Make {
                            n: Some(1),
                            n_repr: "1".into(),
                            name: Some("m0off".into()),
                            where_: self.where_text.clone(),
                            ..Make::default()
                        };
                        let r = self.make_value('s', Ty::U32, m)?;
                        self.assign(&r, tree, Some(format!("m0off = {tree_text}")))?;
                        r.reg()
                    }
                };
                if m0_const == 0 {
                    self.make_plain("s.mov.b32", vec![m0, Arg::Reg(sreg, 0)], &comment)?
                } else if m0_const > 0 {
                    self.make_plain(
                        "s.add.u32",
                        vec![m0, Arg::Reg(sreg, 0), Arg::Str(format!("0x{m0_const:x}"))],
                        &comment,
                    )?
                } else {
                    let abs = -m0_const;
                    self.make_plain(
                        "s.sub.u32",
                        vec![m0, Arg::Reg(sreg, 0), Arg::Str(format!("0x{abs:x}"))],
                        &comment,
                    )?
                }
            }
        };
        self.emit(i, abl)?;
        Ok(())
    }

    fn make_global_dma(&mut self, d: &Dma, lds_text: &str, base: Val) -> Result<thin_ir::Instr> {
        let pointer = matches!(&d.addr, Val::Reg(r) if r.file == 's' && r.n == 2);
        let do_split = matches!(d.addr, Val::Tree(_))
            || pointer && matches!(base, Val::None)
            || matches!(base, Val::Reg(_)) && matches!(d.addr, Val::Int(_));
        if do_split {
            let (ptr, label, addr) = match reg_of_val(&base) {
                Some(b) => (b, None, off_of(&d.addr).expect("a global address")),
                None if matches!(&d.addr, Val::Reg(r) if r.file == 's') => {
                    let Val::Reg(r) = &d.addr else {
                        unreachable!("a register address")
                    };
                    (*r, None, Off::Int(0))
                }
                None => {
                    let Val::Tree(t) = &d.addr else {
                        unreachable!("a tree address")
                    };
                    match pointer_sum(t) {
                        Some((p, off)) => (p.reg(), p.label.as_deref().map(str::to_owned), off),
                        None => {
                            return value(format!(
                                "{}: a global address is a scalar pointer plus u32 terms",
                                top(t)
                            ))
                        }
                    }
                }
            };
            let off = match &d.offset {
                Val::Int(0) => addr,
                o => sum(vec![addr, off_of(o).unwrap_or(Off::Int(0))]),
            };
            let (var, c) = split(&off);
            let (reg, c) = addr_reg(self, Mem::Global, var, c)?;
            let src_comment = self.split_comment(label.as_deref(), &off, &d.opts.comment);
            let comment = format!("{lds_text} <- {src_comment}");
            let mut fields = Vec::new();
            if c != 0 {
                fields.push(("offset".to_owned(), FArg::Int(c)));
            }
            fields.extend(d.opts.cache.fields());
            let notes = Notes {
                comment,
                tag: d.opts.tag.clone(),
                ..Notes::default()
            };
            return self.make(&d.op_name, vec![reg.arg(), Arg::Reg(ptr, 0)], fields, notes);
        }
        let saddr = match base {
            Val::None => Arg::Str("off".into()),
            b => b.arg(),
        };
        let mut fields = match &d.offset {
            Val::Int(0) | Val::None => Vec::new(),
            Val::Int(v) => vec![("offset".to_owned(), FArg::Int(*v))],
            _ => Vec::new(),
        };
        fields.extend(d.opts.cache.fields());
        let comment = if d.opts.comment.is_empty() {
            lds_text.to_owned()
        } else {
            format!("{lds_text}: {}", d.opts.comment)
        };
        let notes = Notes {
            comment,
            tag: d.opts.tag.clone(),
            ..Notes::default()
        };
        self.make(&d.op_name, vec![d.addr.arg(), saddr], fields, notes)
    }

    fn make_buffer_dma(
        &mut self,
        d: &Dma,
        lds_text: &str,
        desc: Val,
        soffset: Val,
    ) -> Result<thin_ir::Instr> {
        let (voff, opts) = if matches!(d.addr, Val::Tree(_) | Val::Int(_)) {
            let addr = off_of(&d.addr).expect("a buffer offset");
            let off = match &d.offset {
                Val::Int(0) => addr,
                o => sum(vec![addr, off_of(o).unwrap_or(Off::Int(0))]),
            };
            let (var, c) = split(&off);
            let (reg, c) = addr_reg(self, Mem::Buffer, var, c)?;
            let reg = match reg {
                Opd::Reg(v, _) => v.reg(),
                _ => unreachable!("a register"),
            };
            let src_comment = self.split_comment(None, &off, &d.opts.comment);
            let opts = Opts {
                offset: Val::Int(c),
                comment: format!("{lds_text} <- {src_comment}"),
                ..d.opts.clone()
            };
            (Val::Reg(reg), opts)
        } else {
            let comment = if d.opts.comment.is_empty() {
                lds_text.to_owned()
            } else {
                format!("{lds_text}: {}", d.opts.comment)
            };
            let opts = Opts {
                comment,
                ..d.opts.clone()
            };
            (d.addr.clone(), opts)
        };
        let c = crate::maker::Ctx {
            t: self.t,
            values: &self.values,
        };
        let made =
            crate::maker::families::buffer_dma_op(&c, &d.op_name, voff, desc, soffset, opts)?;
        let Made::Instr(i) = made else {
            unreachable!("a buffer DMA op")
        };
        Ok(*i)
    }
}
