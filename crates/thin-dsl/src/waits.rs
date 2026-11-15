//! Wait requests.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use thin_ir::{Attr, Datum, FieldId, FieldValue, HostObject, Instr, Operand, Reg, Tag};
use thin_target::{Kind, Pseudo, Target};

use crate::builder::Builder;
use crate::error::{err, schedule, Class, Result};

/// The wait requests' state.
#[derive(Clone, Debug, Default)]
pub struct Requests {
    /// The dwords some counted op's def wrote (any path).
    loaded: BTreeSet<(char, u32)>,
    /// The tags issued.
    tags: Vec<Tag>,
    /// The VMEM ops emitted with a tag, by row.
    tagged: Vec<(Tag, u64)>,
    /// The registers the waits since the last instruction named.
    pending: Vec<Reg>,
    /// Per pending argument dword, its comment.
    args: BTreeMap<(char, u32), String>,
    /// The pending argument registers, in order.
    arg_regs: Vec<Reg>,
    /// The argument dwords read.
    read: BTreeSet<(char, u32)>,
    /// The pending dwords `b.wait_on_use` named: any load's, where an
    /// argument's are a scalar load's.
    uses: BTreeSet<(char, u32)>,
    /// Per open loop body, its stem and the registers whose load check
    /// waits for its end (with their comments).
    bodies: Vec<(String, Vec<(Reg, String)>)>,
}

impl Requests {
    /// Records a tag as issued.
    pub fn issue_tag(&mut self, tag: Tag) {
        if !self.tags.contains(&tag) {
            self.tags.push(tag);
        }
    }
}

/// What a wait names.
#[derive(Clone, Debug)]
pub enum Thing {
    /// A counter or a counter group, by name.
    Counter(String),
    /// A plain string.
    Str(String),
    /// A register.
    Reg(Reg),
    /// An LDS region, by its name.
    Region(String),
    /// A list or tuple.
    List(Vec<Thing>),
    /// Anything else, by its repr.
    Other(String),
}

/// A wait's request.
#[derive(Clone, Debug)]
pub enum Request {
    /// Registers and region names.
    Regs(Vec<Reg>, Vec<String>),
    /// Tags.
    Tags(Vec<Tag>),
    /// Everything.
    All,
    /// All but n.
    Left(i128),
    /// The SMEM hazard.
    Smem,
}

/// A wait's options beside its things.
#[derive(Clone, Debug, Default)]
pub struct Wait {
    /// `left=`.
    pub left: Option<i128>,
    /// `tags=`.
    pub tags: Vec<Tag>,
    /// `written=`.
    pub written: bool,
    /// `comment=`.
    pub comment: String,
    /// `abl=`.
    pub abl: Vec<String>,
}

fn field(name: &str, v: FieldValue) -> Result<(FieldId, FieldValue)> {
    Ok((FieldId::named(name)?, v))
}

/// A marker on a counter.
pub fn marker(
    t: &'static Target,
    counter: usize,
    req: &Request,
    said: &str,
    ablated: bool,
) -> Result<Instr> {
    let name = t.definition().counters[counter].name;
    let mut fields = vec![
        field("counter", FieldValue::Text(name.into()))?,
        field("said", FieldValue::Text(said.into()))?,
    ];
    if ablated {
        fields.push(field("ablated", FieldValue::Bool(true))?);
    }
    let (p, ops) = match req {
        Request::Regs(regs, regions) => {
            fields.push(field(
                "regions",
                FieldValue::Names(regions.iter().map(|r| r.as_str().into()).collect()),
            )?);
            (
                Pseudo::WaitRegs,
                regs.iter().map(|r| Operand::Reg(*r)).collect(),
            )
        }
        Request::Tags(tags) => {
            fields.push(field("tags", FieldValue::Tags(tags.clone().into()))?);
            (Pseudo::WaitTags, Vec::new())
        }
        Request::All => (Pseudo::WaitAll, Vec::new()),
        Request::Left(n) => {
            fields.push(field("left", FieldValue::Int(*n))?);
            (Pseudo::WaitLeft, Vec::new())
        }
        Request::Smem => (Pseudo::WaitSmem, Vec::new()),
    };
    Ok(Instr::new(t.pseudo(p), ops, fields, None))
}

/// The target's wait instructions for these counts.
pub fn wait_instrs(
    t: &'static Target,
    counts: &[(usize, u32)],
    comment: &str,
) -> Result<Vec<Instr>> {
    let mut out = Vec::new();
    for w in &t.definition().waits {
        let mut fields = Vec::new();
        for &(c, n) in counts {
            let f = t.definition().counters[c].field;
            if w.fields.contains(&f) {
                fields.push(field(f, FieldValue::Int(n.into()))?);
            }
        }
        if !fields.is_empty() {
            let form = t.form(w.mnemonic, 0, false).expect("a wait form");
            let mut i = Instr::new(form, Vec::new(), fields, None);
            if !comment.is_empty() {
                i = i.with_attr(Attr::Comment(comment.into()))?;
            }
            out.push(i);
        }
    }
    Ok(out)
}

/// A registers' host object for WaitUses.
fn uses_object(b: &Builder, regs: &[Reg]) -> HostObject {
    let reprs: Vec<String> = regs.iter().map(|r| b.repr(r)).collect();
    HostObject::of_datum(
        Datum::Tuple(regs.iter().map(|r| Datum::Reg(*r)).collect()),
        format!("[{}]", reprs.join(", ")),
    )
}

impl Builder {
    /// Pushes one marker per counter.
    fn markers(
        &mut self,
        counters: &[usize],
        req: &Request,
        said: &str,
        ablated: bool,
    ) -> Result<()> {
        for &c in counters {
            let i = marker(self.t, c, req, said, ablated)?;
            let loc = self.loc;
            self.push(i, loc);
        }
        Ok(())
    }

    fn all_counters(&self) -> Vec<usize> {
        (0..self.t.definition().counters.len()).collect()
    }

    /// `b.wait(*what, left=, tags=, written=, comment=, abl=)`.
    pub fn wait(&mut self, mut what: Vec<Thing>, mut w: Wait) -> Result<()> {
        self.check_open()?;
        self.settle()?;
        // A trailing plain string is the comment, unless it names a counter
        // or a counter group (`b.wait("load")`); a second comment is refused.
        if let Some(Thing::Str(s)) = what.last().cloned() {
            what.pop();
            if !self.t.group(&s).is_empty() {
                what.push(Thing::Counter(s));
            } else if w.comment.is_empty() {
                w.comment = s;
            } else {
                return schedule(format!(
                    "wait(..., {s:?}, comment={:?}): two comments; give one",
                    w.comment
                ));
            }
        }
        let (counters, things): (Vec<Thing>, Vec<Thing>) = what
            .into_iter()
            .partition(|t| matches!(t, Thing::Counter(_)));
        if w.left.is_some() && counters.len() != 1 {
            return schedule("wait(left=n): of one counter (VM or LGKM)");
        }
        let mut named = Vec::new();
        for c in &counters {
            let Thing::Counter(name) = c else { continue };
            let group = self.t.group(name);
            if group.is_empty() {
                let mut names: Vec<&str> = self
                    .t
                    .definition()
                    .counters
                    .iter()
                    .map(|c| c.name)
                    .collect();
                names.extend(["vm", "lgkm", "smem"]);
                return schedule(format!(
                    "wait({name}): {} counts on {}",
                    self.t.name(),
                    names.join(", ")
                ));
            }
            named.push(group);
        }
        if things.is_empty() && counters.is_empty() && w.tags.is_empty() {
            return schedule("wait(): what to wait for (values, regions, VM / LGKM, tags=)");
        }
        let ablated = self.ablated(&w.abl);
        for group in named {
            let req = w.left.map_or(Request::All, Request::Left);
            self.markers(&group, &req, &w.comment, ablated)?;
        }
        if !w.tags.is_empty() {
            self.wait_tags(&w, ablated)?;
        }
        if !things.is_empty() {
            self.wait_things(things, &w.comment, ablated)?;
        }
        Ok(())
    }

    fn wait_tags(&mut self, w: &Wait, ablated: bool) -> Result<()> {
        let missing: Vec<&Tag> = w
            .tags
            .iter()
            .filter(|t| !self.requests.tags.contains(t))
            .collect();
        if !missing.is_empty() {
            let shown: Vec<String> = missing
                .iter()
                .take(3)
                .map(|t| thin_ir::tag_repr(t))
                .collect();
            let more = if missing.len() > 3 { "[...]" } else { "" };
            return schedule(format!(
                "wait for DMA tags never issued: [{}]{more}",
                shown.join(", ")
            ));
        }
        let vm = self.t.group("vm");
        if !w.written {
            return self.markers(&vm, &Request::Tags(w.tags.clone()), &w.comment, ablated);
        }
        let ops: Vec<u64> = (self.requests.tagged.iter())
            .filter(|(t, _)| w.tags.contains(t))
            .map(|x| x.1)
            .collect();
        let mut regs: Vec<Reg> = Vec::new();
        let mut names: BTreeSet<String> = BTreeSet::new();
        let mut last_writer = None;
        for (k, id) in ops.iter().enumerate() {
            let i = self.rows[self.row_index(*id).expect("a tagged row")]
                .ins
                .clone();
            if i.defs().next().is_some() {
                last_writer = Some(k);
            }
        }
        for (k, id) in ops.iter().enumerate() {
            let i = self.rows[self.row_index(*id).expect("a tagged row")]
                .ins
                .clone();
            if i.defs().next().is_none() {
                let mut placed: Vec<String> =
                    i.attrs().places.iter().map(|n| n.to_string()).collect();
                if let Some(acc) = &i.attrs().lds_access {
                    placed.extend(crate::lds::tags_of(self, acc.region));
                }
                if placed.is_empty() && last_writer.is_none_or(|l| k > l) {
                    return err(
                        Class::Schedule,
                        format!(
                            "wait(tags=, written=True): `{}` writes no register and places no \
                             name, so it has nothing written to wait for (wait without written=, \
                             or place what it writes)",
                            i.form().name
                        ),
                    );
                }
                names.extend(placed);
            }
            for r in i.defs() {
                if !regs.contains(r) {
                    regs.push(*r);
                }
            }
        }
        self.markers(
            &vm,
            &Request::Regs(regs, names.into_iter().collect()),
            &w.comment,
            ablated,
        )
    }

    fn wait_things(&mut self, things: Vec<Thing>, comment: &str, ablated: bool) -> Result<()> {
        let mut flat = Vec::new();
        flatten(things, &mut flat);
        let mut regs = Vec::new();
        let mut regions = Vec::new();
        for t in flat {
            match t {
                Thing::Reg(r) => regs.push(r),
                Thing::Region(name) => {
                    crate::lds::wait_region(self, &name)?;
                    regions.push(name);
                }
                Thing::Other(repr) | Thing::Str(repr) => {
                    return schedule(format!("wait({repr}): not a register, a value or a region"));
                }
                Thing::Counter(_) | Thing::List(_) => {}
            }
        }
        if self.ablate.is_empty() {
            for r in &regs {
                self.check_loaded(r, comment)?;
            }
        }
        self.requests.pending.extend(regs.iter().copied());
        let all = self.all_counters();
        self.markers(&all, &Request::Regs(regs, regions), comment, ablated)
    }

    /// A register some counted op loaded, or deferred to the open body.
    fn check_loaded(&mut self, r: &Reg, comment: &str) -> Result<()> {
        if self
            .dwords(r)
            .iter()
            .any(|d| self.requests.loaded.contains(d))
        {
            return Ok(());
        }
        let said = if comment.is_empty() {
            String::new()
        } else {
            format!(" ({comment})")
        };
        match self.requests.bodies.last_mut() {
            Some((_, deferred)) => {
                deferred.push((*r, said));
                Ok(())
            }
            None => schedule(format!(
                "wait: {}{said} was written by no load",
                self.repr(r)
            )),
        }
    }

    /// Opens a loop body's deferred load checks.
    pub fn open_body(&mut self, stem: &str) {
        self.requests.bodies.push((stem.to_owned(), Vec::new()));
    }

    /// Closes a loop body: its deferred registers still unloaded pass to
    /// the enclosing body, or are refused; `dropped` drops them (a body
    /// left by break).
    pub fn close_body(&mut self, dropped: bool) -> Result<()> {
        let Some((stem, deferred)) = self.requests.bodies.pop() else {
            return Ok(());
        };
        if dropped {
            return Ok(());
        }
        for (r, said) in deferred {
            if self
                .dwords(&r)
                .iter()
                .any(|d| self.requests.loaded.contains(d))
            {
                continue;
            }
            match self.requests.bodies.last_mut() {
                Some((_, outer)) => outer.push((r, said)),
                None => {
                    return schedule(format!(
                        "wait: {}{said} was written by no load (nor by one later in the loop {stem} it is in)",
                        self.repr(&r)
                    ))
                }
            }
        }
        Ok(())
    }

    /// `b.wait_on_use(*regs, comment)`.
    pub fn wait_on_use(&mut self, regs: &[Reg], comment: &str) {
        self.on_use(regs, comment, true);
    }

    /// The kernel arguments' wait at their first read: their scalar loads'.
    pub(crate) fn wait_arguments_on_use(&mut self, regs: &[Reg], comment: &str) {
        self.on_use(regs, comment, false);
    }

    /// Waits for `regs` at their first read: on every counter (`any`), or on
    /// the scalar-memory ones.
    fn on_use(&mut self, regs: &[Reg], comment: &str, any: bool) {
        for r in regs {
            for d in self.dwords(r) {
                self.requests.args.insert(d, comment.to_owned());
                if any {
                    self.requests.uses.insert(d);
                } else {
                    self.requests.uses.remove(&d);
                }
            }
            if !self.requests.arg_regs.contains(r) {
                self.requests.arg_regs.push(*r);
            }
        }
    }

    /// `b.land_arguments(why)`.
    pub fn land_arguments(&mut self, why: &str) -> Result<()> {
        let unread: Vec<Reg> = (self.requests.arg_regs.iter())
            .filter(|r| {
                let ds = self.dwords(r);
                ds.iter().any(|d| self.requests.args.contains_key(d))
                    && !ds.iter().any(|d| self.requests.read.contains(d))
            })
            .copied()
            .collect();
        if unread.is_empty() {
            return Ok(());
        }
        for r in &unread {
            for d in self.dwords(r) {
                self.requests.args.remove(&d);
                self.requests.uses.remove(&d);
            }
        }
        self.requests.arg_regs.retain(|r| !unread.contains(r));
        self.settle()?;
        self.wait_things(unread.into_iter().map(Thing::Reg).collect(), why, false)
    }
}

fn flatten(things: Vec<Thing>, out: &mut Vec<Thing>) {
    for t in things {
        match t {
            Thing::List(xs) => flatten(xs, out),
            t => out.push(t),
        }
    }
}

/// the argument requests, the SMEM hazard, the pending uses.
pub(crate) fn before(b: &mut Builder, mut i: Instr) -> Result<Instr> {
    let is_wait = i.form().kind == Kind::Wait;
    if !is_wait && !b.requests.args.is_empty() {
        let named: Vec<Reg> = (i.uses())
            .filter(|r| b.dwords(r).iter().any(|d| b.requests.args.contains_key(d)))
            .copied()
            .collect();
        if let Some(first) = named.first() {
            let d = *b
                .dwords(first)
                .iter()
                .find(|d| b.requests.args.contains_key(d))
                .expect("a pending dword");
            let said = b.requests.args[&d].clone();
            for r in &named {
                let ds = b.dwords(r);
                b.requests.read.extend(ds);
            }
            let any_load =
                (named.iter()).any(|r| b.dwords(r).iter().any(|d| b.requests.uses.contains(d)));
            let counters = match any_load {
                true => b.all_counters(),
                false => b.t.group("lgkm"),
            };
            b.markers(&counters, &Request::Regs(named, Vec::new()), &said, false)?;
        }
        let defs: Vec<Reg> = i.defs().copied().collect();
        for r in defs {
            for d in b.dwords(&r) {
                b.requests.args.remove(&d);
                b.requests.uses.remove(&d);
            }
        }
    }
    if b.t.definition().operands.valu_sgpr_write_waits_smem && i.sgpr_valu_def() {
        let smem = b.t.group("smem");
        b.markers(
            &smem,
            &Request::Smem,
            "hazard: VALU SGPR write with SMEM outstanding",
            false,
        )?;
    }
    if !is_wait && !b.requests.pending.is_empty() {
        let regs = std::mem::take(&mut b.requests.pending);
        i = i.with_attr(Attr::WaitUses(uses_object(b, &regs), regs))?;
    }
    Ok(i)
}

/// a VMEM op's tag; the dwords of a counted op's defs loaded.
pub(crate) fn after(b: &mut Builder, i: &Arc<Instr>) {
    let id = b.rows.last().expect("the row").id;
    if i.form().counter.is_none() {
        return;
    }
    if i.form().kind == Kind::Vmem {
        if let Some(tag) = &i.attrs().tag {
            b.requests.issue_tag(tag.clone());
            b.requests.tagged.push((tag.clone(), id));
        }
    }
    let defs: Vec<Reg> = i.defs().copied().collect();
    for r in defs {
        let ds = b.dwords(&r);
        b.requests.loaded.extend(ds);
    }
}
