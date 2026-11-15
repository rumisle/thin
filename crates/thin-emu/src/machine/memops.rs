//! The memory semantics: scalar, global and
//! buffer memory, LDS ops, with the flight and scope models and the tag
//! rules.

use super::alu::{int_field, tag_error};
use super::facts::role;
use super::flight::{Write, Writer};
use super::lds::Lds;
use super::numbers::{bf16_rne, f32_of, nan32};
use super::scope::{Level, Site};
use super::wave::{Src, VmOp, Wave, MAX_W};
use super::{Exec, POISON};
use crate::tags::rules::{self, Ctx as TagCtx};
use crate::tags::{AtomicAnswer, Lane, StoreCall, Tag};
use crate::{Base, Cause, EmuError, Field, Ins, Operand};

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

fn ipc_error(message: String) -> EmuError {
    EmuError {
        cause: Some(Cause::Ipc),
        message,
    }
}

fn active(exec: u64, w: usize) -> impl Iterator<Item = usize> {
    (0..w).filter(move |l| exec >> l & 1 != 0)
}

/// The bytes a memory mnemonic moves per lane, and whether a sub-dword
/// load sign-extends (by the canon's last part).
pub fn width(last: &str) -> Option<(u32, bool)> {
    Some(match last {
        "b32" | "u32" | "i32" | "f32" => (4, false),
        "b64" => (8, false),
        "b96" => (12, false),
        "b128" => (16, false),
        "b256" => (32, false),
        "b512" => (64, false),
        "u16" | "b16" | "d16" => (2, false),
        "i16" => (2, true),
        "u8" | "b8" => (1, false),
        "i8" => (1, true),
        _ => return None,
    })
}

/// The fields of an instruction as the scope rule reads them: a flag
/// reads 1, an integer its value.
fn field_value(ins: &Ins) -> impl Fn(&str) -> Option<i64> + '_ {
    |n: &str| match ins.field(n)? {
        Field::Int(v) => Some(*v as i64),
        _ => Some(1),
    }
}

/// The scope of an access, by the target's scope rule.
fn level(e: &Exec<'_>, ins: &Ins, atomic: bool, uncached: bool) -> Level {
    if uncached {
        return Level::System;
    }
    let rule = &e.facts.scope;
    rule.scope(rule.code_of(&field_value(ins)), atomic)
}

/// Whether the policy lets an access touch an IPC buffer: the
/// scope its fields name (an atomic's as a load's or store's, not raised)
/// reaches the rule's IPC scope, or a load or store is non-temporal.
fn ipc_ok(e: &Exec<'_>, ins: &Ins, store: Option<bool>) -> bool {
    let rule = &e.facts.scope;
    let fields = field_value(ins);
    let nt = store.is_some_and(|st| rule.nontemporal(&fields, st));
    nt || rule.scope(rule.code_of(&fields), false) >= rule.ipc
}

impl Exec<'_> {
    fn site(&self, w: &Wave) -> Site {
        Site {
            rank: self.rank,
            l2: w.xcd,
        }
    }

    fn writer(&self, w: &Wave) -> Writer {
        Writer {
            wg: self.wg.serial,
            wave: w.wid,
            rank: self.rank,
            wgx: w.wgx,
        }
    }

    fn frontier(&self, w: &Wave) -> Vec<i64> {
        let mut f = self.wg.barrier_marks.clone();
        if f.len() <= w.wid {
            f.resize(w.wid + 1, -1);
        }
        f[w.wid] = w.mark;
        f
    }

    /// The counter of a VMEM op.
    fn vm_counter(&self, store: bool) -> usize {
        if store {
            self.facts.store_counter
        } else {
            role::LOAD
        }
    }

    fn vm_issue(&mut self, w: &mut Wave, store: bool, records: Vec<usize>) -> u64 {
        let counter = self.vm_counter(store);
        let seq = w.seq[counter];
        w.seq[counter] += 1;
        w.vm_ops.push(VmOp {
            counter,
            seq,
            store,
            records,
        });
        seq
    }

    /// The IPC policy of a load (`Some(false)`), a store
    /// (`Some(true)`) or an atomic (`None`).
    fn check_ipc(&self, ins: &Ins, bufs: &[usize], store: Option<bool>) -> Result<(), EmuError> {
        if !self.ipc_policy || ipc_ok(self, ins, store) {
            return Ok(());
        }
        for &b in bufs {
            if self.mem.bufs[b].meta.ipc {
                let policy = self.facts.scope.ipc_text;
                return Err(ipc_error(format!(
                    "{}: accesses the IPC buffer {} without {policy} (another rank's view)",
                    ins.text, self.mem.bufs[b].name
                )));
            }
        }
        Ok(())
    }

    /// A read of global bytes by this workgroup: watch, traffic, the flight
    /// model's stale bytes and observation. Returns the bytes and tags read.
    fn read_global(
        &mut self,
        w: &mut Wave,
        ins: &Ins,
        buf: usize,
        offs: &[u64],
    ) -> (Vec<u8>, Vec<Tag>) {
        let b = &self.mem.bufs[buf];
        w.watch.entry(buf).or_insert(b.version);
        let mut data: Vec<u8> = offs.iter().map(|o| b.data[*o as usize]).collect();
        let mut tags: Vec<Tag> = offs.iter().map(|o| b.tags[*o as usize]).collect();
        if self.mem.flight.active() {
            let reader = self.wg.serial;
            let current = &self.mem.bufs[buf].data;
            if let Some((served, from)) = self.mem.flight.serve(reader, buf, offs, current) {
                for (off, byte, tag) in &served.bytes {
                    for (k, o) in offs.iter().enumerate() {
                        if o == off {
                            data[k] = *byte;
                            tags[k] = *tag;
                        }
                    }
                }
                let r = &self.mem.flight.records[&from];
                let first = served.bytes[0].0;
                let observed = self.mem.flight.observed_text(reader, r.who.wg);
                // The write's counter: the one stores count on.
                let counter = &self.facts.counter_names[self.facts.store_counter];
                let msg = format!(
                    "wave {} of WG {} (rank {}) `{}` read {}[{first}] stale: wave {} of WG {} (rank {})'s `{}` was not waited for ({counter}; and a barrier, from another wave) before its workgroup's `{observed}`, which this workgroup observed - the flag may be performed first, the data late",
                    w.wid, w.wgx, self.rank, ins.text, self.mem.bufs[buf].name, r.who.wave, r.who.wgx, r.who.rank, r.text
                );
                self.mem.flight.events.push(msg);
                w.progress += 1;
            }
            self.mem.flight.observe(reader, buf, offs);
        }
        (data, tags)
    }

    /// Records a global write before its bytes change.
    #[allow(clippy::too_many_arguments)]
    fn record_write(
        &mut self,
        w: &Wave,
        ins: &Ins,
        buf: usize,
        offs: Vec<u64>,
        atomic: bool,
        publisher: bool,
        performed: bool,
    ) -> usize {
        let who = self.writer(w);
        let frontier = self.frontier(w);
        let b = &self.mem.bufs[buf];
        let write = Write {
            who,
            frontier,
            buf,
            bytes: offs,
            data: &b.data,
            tags: &b.tags,
            text: &ins.text,
            atomic,
            publisher,
            performed,
            serial: self.serial,
        };
        self.mem.flight.write(write)
    }
}

/// The per-lane global addresses of a global op: (vaddr operand, saddr
/// operand) plus the signed `offset`.
fn global_addrs(
    w: &Wave,
    ins: &Ins,
    vaddr: &Operand,
    saddr: Option<&Operand>,
) -> Result<[u64; MAX_W], EmuError> {
    let off = int_field(ins, "offset", 0) as i64;
    let mut out = [0u64; MAX_W];
    let lo = w.vread(vaddr, 0)?;
    let base = match saddr {
        Some(o) if !matches!(&o.base, Base::Reg(f, _, _) if f == "off" || f == "null") => {
            Some(w.sread64(o)?)
        }
        _ => None,
    };
    match base {
        Some(b) => {
            for l in 0..w.w {
                out[l] = b.wrapping_add(u64::from(lo.v[l])).wrapping_add(off as u64);
            }
        }
        None => {
            let hi = w.vread(vaddr, 1)?;
            for l in 0..w.w {
                let a = u64::from(lo.v[l]) | u64::from(hi.v[l]) << 32;
                out[l] = a.wrapping_add(off as u64);
            }
        }
    }
    Ok(out)
}

/// A buffer descriptor's (base, num_records) and the per-lane offsets.
fn buffer_addrs(
    w: &Wave,
    ins: &Ins,
    voff: &Operand,
    desc: &Operand,
    soff: Option<&Operand>,
) -> Result<(u64, u64, [u64; MAX_W]), EmuError> {
    let Base::Reg(f, r, _) = &desc.base else {
        return Err(fault(format!(
            "{}: a buffer descriptor must be SGPRs",
            ins.text
        )));
    };
    if f != "s" {
        return Err(fault(format!(
            "{}: a buffer descriptor must be SGPRs",
            ins.text
        )));
    }
    // `off` and `null` as soffset both read 0: no scalar read.
    let soff = soff.filter(|o| !matches!(&o.base, Base::Reg(f, _, _) if f == "off" || f == "null"));
    let r = *r as usize;
    let d: Vec<u32> = (0..4)
        .map(|k| w.s.get(r + k).copied().unwrap_or(POISON))
        .collect();
    // An index term (idxen) needs the descriptor's stride and its index
    // range checks: not modelled.
    if ins
        .field("idxen")
        .is_some_and(|f| !matches!(f, crate::Field::Int(0)))
    {
        return Err(limit(format!("{}: idxen is not modelled", ins.text)));
    }
    let offen = ins.field("offen").is_some();
    let v = if offen { Some(w.vread(voff, 0)?) } else { None };
    let any = w.exec != 0;
    if any {
        let poison_desc = d.contains(&POISON);
        let so = match soff {
            Some(o) => w.sread(o)? as u32,
            None => 0,
        };
        let poison_soff =
            matches!(soff.map(|o| &o.base), Some(Base::Reg(f, _, _)) if f == "s") && so == POISON;
        let poison_lane = v
            .as_ref()
            .is_some_and(|v| (0..w.w).any(|l| w.exec >> l & 1 != 0 && v.v[l] == POISON));
        if poison_desc || poison_soff || poison_lane {
            let which = if poison_desc {
                format!("its descriptor s[{r}:{}]", r + 3)
            } else if poison_soff {
                "its soffset".to_owned()
            } else {
                "a lane's voffset".to_owned()
            };
            return Err(fault(format!(
                "wave {} {}: {which} holds a register no instruction wrote (on the GPU: what the last kernel left)",
                w.wid, ins.text
            )));
        }
    }
    let base = u64::from(d[0]) | u64::from(d[1] & 0xFFFF) << 32;
    let num = u64::from(d[2]);
    let so = match soff {
        Some(o) => u64::from(w.sread(o)? as u32),
        None => 0,
    };
    let fo = int_field(ins, "offset", 0) as u64;
    let mut out = [0u64; MAX_W];
    for (l, o) in out.iter_mut().enumerate().take(w.w) {
        *o = v.as_ref().map_or(0, |v| u64::from(v.v[l])) + fo + so;
    }
    Ok((base, num, out))
}

fn bufs_of(lanes: &[Option<(usize, u64)>]) -> Vec<usize> {
    let mut b: Vec<usize> = lanes.iter().flatten().map(|x| x.0).collect();
    b.sort_unstable();
    b.dedup();
    b
}

/// A vector-memory op: loads, stores, atomics, cache control. `Ok(false)`
/// when `canon` is none of them.
#[allow(clippy::too_many_lines)]
pub fn vmem(e: &mut Exec<'_>, w: &mut Wave, ins: &Ins, canon: &str) -> Result<bool, EmuError> {
    let mut parts: Vec<&str> = canon.split('.').collect();
    let ops = &ins.operands;
    let d16 = d16_load(&mut parts);
    let old = match d16 {
        Some(_) => Some(w.vread(&ops[0], 0)?),
        None => None,
    };
    match parts.as_slice() {
        ["global", "load", last] => {
            let Some((n, signed)) = width(last) else {
                return Ok(false);
            };
            let addrs = global_addrs(w, ins, &ops[1], ops.get(2))?;
            load(e, w, ins, &addrs, n, signed, None, &ops[0])?;
        }
        ["buffer", "load", last] => {
            let Some((n, signed)) = width(last) else {
                return Ok(false);
            };
            let (base, num, offs) = buffer_addrs(w, ins, &ops[1], &ops[2], ops.get(3))?;
            let mut addrs = [0u64; MAX_W];
            for l in 0..w.w {
                addrs[l] = base.wrapping_add(offs[l]);
            }
            load(e, w, ins, &addrs, n, signed, Some((num, offs)), &ops[0])?;
        }
        ["global", "store", last] => {
            let Some((n, _)) = width(last) else {
                return Ok(false);
            };
            let addrs = global_addrs(w, ins, &ops[0], ops.get(2))?;
            store(e, w, ins, &addrs, n, None, &ops[1])?;
        }
        ["buffer", "store", last] => {
            let Some((n, _)) = width(last) else {
                return Ok(false);
            };
            let (base, num, offs) = buffer_addrs(w, ins, &ops[1], &ops[2], ops.get(3))?;
            let mut addrs = [0u64; MAX_W];
            for l in 0..w.w {
                addrs[l] = base.wrapping_add(offs[l]);
            }
            store(e, w, ins, &addrs, n, Some((num, offs)), &ops[0])?;
        }
        ["global", "atomic", rest @ ..] => {
            let op = rest.join(".");
            atomic(e, w, ins, &op)?;
        }
        ["buffer", "atomic", "pk", "add", "bf16"] => pk_add_bf16(e, w, ins)?,
        ["global", "wb"] => {
            // A write-back reaches the L2 from the rule's code; it counts
            // as a store where the rule says so.
            let rule = e.facts.scope;
            if rule.maintenance_counted {
                e.vm_issue(w, true, Vec::new());
            }
            let code = rule.code_of(&field_value(ins));
            let l2 = rule.writeback_l2.is_none_or(|from| code >= from);
            let site = e.site(w);
            if let (true, Some(s)) = (l2, &mut e.mem.scope) {
                s.writeback(site);
            }
        }
        ["global", "inv"] => {
            let rule = e.facts.scope;
            if rule.maintenance_counted {
                e.vm_issue(w, false, Vec::new());
            }
            let l2 = rule.code_of(&field_value(ins)) >= rule.invalidate_l2;
            let site = e.site(w);
            if let (true, Some(s)) = (l2, &mut e.mem.scope) {
                s.invalidate(site);
            }
        }
        _ => return Ok(false),
    }
    if let (Some(hi), Some(old)) = (d16, old) {
        d16_merge(w, &ops[0], &old, hi)?;
    }
    Ok(true)
}

/// A d16 load's half (true: the high one), its `d16` and `hi` parts taken
/// out of `parts` so the plain load's arm runs it; none for any other op.
fn d16_load(parts: &mut Vec<&str>) -> Option<bool> {
    if parts.len() < 4 || parts[1] != "load" || parts[2] != "d16" {
        return None;
    }
    parts.remove(2);
    let hi = parts[2] == "hi";
    if hi {
        parts.remove(2);
    }
    (parts.len() == 3 && matches!(parts[2], "b16" | "u8" | "i8")).then_some(hi)
}

/// The d16 merge of a load's result (RDNA4 ISA guide
/// BUFFER_LOAD_D16_{B16,U8,I8} and their `_HI` forms, Table 51 "writes
/// only 16 bits"): the load wrote its 16-bit value (zero- or sign-extended
/// from a byte) into the low half; it moves to half `hi`, and the other half
/// keeps the register's value and tags from before the load. Out-of-range
/// lanes load 0 into their half; inactive lanes are unwritten.
fn d16_merge(w: &mut Wave, dst: &Operand, old: &Src, hi: bool) -> Result<(), EmuError> {
    let new = w.vread(dst, 0)?;
    let mut out = [0u32; MAX_W];
    let mut tags = vec![[0 as Tag; 4]; w.w];
    for l in 0..w.w {
        let (v, nt, ot) = (new.v[l] & 0xFFFF, new.tag(l), old.tag(l));
        (out[l], tags[l]) = if hi {
            (old.v[l] & 0xFFFF | v << 16, [ot[0], ot[1], nt[0], nt[1]])
        } else {
            (old.v[l] & 0xFFFF_0000 | v, [nt[0], nt[1], ot[2], ot[3]])
        };
    }
    let tagged = old.tagged || new.tagged;
    let exec = w.exec;
    w.vwrite(dst, 0, &out, tagged.then_some(&tags), exec)
}

#[allow(clippy::too_many_arguments)]
fn load(
    e: &mut Exec<'_>,
    w: &mut Wave,
    ins: &Ins,
    addrs: &[u64],
    n: u32,
    signed: bool,
    range: Option<(u64, [u64; MAX_W])>,
    dst: &Operand,
) -> Result<(), EmuError> {
    let wn = w.w;
    let exec = w.exec;
    let dwords = n.div_ceil(4) as usize;
    let mut vals = vec![[0u32; MAX_W]; dwords];
    let mut tags = vec![[[0 as Tag; 4]; MAX_W]; dwords];
    let mut tagged = false;
    // Per lane, the in-range elements (per dword, per element for sub-dword).
    let elem = n.min(4) as u64;
    let in_range = |l: usize, k: u64| -> bool {
        match &range {
            Some((num, offs)) => offs[l] + k + elem <= *num,
            None => true,
        }
    };
    let mut lanes: Vec<Option<(usize, u64)>> = vec![None; wn];
    for l in active(exec, wn) {
        let any = (0..n as u64).step_by(elem as usize).any(|k| in_range(l, k));
        if any {
            lanes[l] = Some(e.mem.access(addrs[l], n as u64, false)?);
        }
    }
    let bufs = bufs_of(&lanes);
    if range.is_none() {
        e.check_ipc(ins, &bufs, Some(false))?;
    }
    for &b in &bufs {
        let mut offs: Vec<u64> = Vec::new();
        for l in 0..wn {
            if let Some((bb, o)) = lanes[l] {
                if bb == b {
                    for k in 0..n as u64 {
                        if in_range(l, k - k % elem) {
                            offs.push(o + k);
                        }
                    }
                }
            }
        }
        let uncached = e.mem.bufs[b].uncached();
        let lvl = level(e, ins, false, uncached);
        let site = e.site(w);
        let size = e.mem.bufs[b].data.len();
        let name = e.mem.bufs[b].name.clone();
        if let (false, Some(s)) = (uncached, &mut e.mem.scope) {
            s.load(b, size, &name, &offs, lvl, site, &ins.text)?;
        }
        e.mem.count(
            b,
            offs.first().copied().unwrap_or(0),
            offs.len() as u64,
            0,
            e.rank,
        );
        let (data, t) = e.read_global(w, ins, b, &offs);
        let mut k = 0;
        for l in 0..wn {
            if let Some((bb, _)) = lanes[l] {
                if bb != b {
                    continue;
                }
                for j in 0..n as usize {
                    if !in_range(l, (j as u64) - (j as u64) % elem) {
                        continue;
                    }
                    let (d, byte) = (j / 4, j % 4);
                    vals[d][l] |= u32::from(data[k]) << (8 * byte);
                    tags[d][l][byte] = t[k];
                    tagged |= t[k] != 0;
                    k += 1;
                }
                if n < 4 && signed {
                    let bits = 8 * n;
                    let v = vals[0][l];
                    vals[0][l] = (((v << (32 - bits)) as i32) >> (32 - bits)) as u32;
                }
            }
        }
    }
    for d in 0..dwords {
        wn_write(
            w,
            dst,
            d as u32,
            &vals[d],
            tagged.then_some(&tags[d][..wn]),
            exec,
        )?;
    }
    let count = active(exec, wn).count() as u64;
    e.reports.stats.vmem_load_bytes += count * u64::from(n);
    e.vm_issue(w, false, Vec::new());
    Ok(())
}

fn wn_write(
    w: &mut Wave,
    dst: &Operand,
    d: u32,
    vals: &[u32],
    tags: Option<&[Lane]>,
    mask: u64,
) -> Result<(), EmuError> {
    w.vwrite(dst, d, vals, tags, mask)
}

fn store(
    e: &mut Exec<'_>,
    w: &mut Wave,
    ins: &Ins,
    addrs: &[u64],
    n: u32,
    range: Option<(u64, [u64; MAX_W])>,
    data: &Operand,
) -> Result<(), EmuError> {
    let wn = w.w;
    let exec = w.exec;
    let dwords = n.div_ceil(4) as usize;
    let mut srcs: Vec<Src> = Vec::with_capacity(dwords);
    for d in 0..dwords {
        srcs.push(w.vread(data, d as u32)?);
    }
    let elem = n.min(4) as u64;
    let in_range = |l: usize, k: u64| -> bool {
        match &range {
            Some((num, offs)) => offs[l] + k + elem <= *num,
            None => true,
        }
    };
    let mut lanes: Vec<Option<(usize, u64)>> = vec![None; wn];
    for l in active(exec, wn) {
        if (0..n as u64).step_by(elem as usize).any(|k| in_range(l, k)) {
            lanes[l] = Some(e.mem.access(addrs[l], n as u64, true)?);
        }
    }
    let bufs = bufs_of(&lanes);
    if range.is_none() {
        e.check_ipc(ins, &bufs, Some(true))?;
    }
    let mut records = Vec::new();
    for &b in &bufs {
        // Flat bytes of the buffer's lanes; `ends` closes each lane's run.
        let mut offs = Vec::with_capacity(wn * n as usize);
        let mut bytes = Vec::with_capacity(wn * n as usize);
        let mut btags = Vec::with_capacity(wn * n as usize);
        let mut ends = Vec::with_capacity(wn);
        for l in 0..wn {
            let Some((bb, o)) = lanes[l] else { continue };
            if bb != b {
                continue;
            }
            for j in 0..n as usize {
                if !in_range(l, (j as u64) - (j as u64) % elem) {
                    continue;
                }
                let (d, byte) = (j / 4, j % 4);
                offs.push(o + j as u64);
                bytes.push((srcs[d].v[l] >> (8 * byte)) as u8);
                btags.push(srcs[d].tag(l)[byte]);
            }
            ends.push(offs.len());
        }
        let uncached = e.mem.bufs[b].uncached();
        let lvl = level(e, ins, false, uncached);
        let site = e.site(w);
        let size = e.mem.bufs[b].data.len();
        if let (false, Some(s)) = (uncached, &mut e.mem.scope) {
            s.store(b, size, &offs, lvl, site);
        }
        let publisher = lvl >= Level::Agent;
        let id = e.record_write(w, ins, b, offs.clone(), false, publisher, false);
        records.push(id);
        let buf = &mut e.mem.bufs[b];
        for k in 0..offs.len() {
            buf.data[offs[k] as usize] = bytes[k];
            buf.tags[offs[k] as usize] = btags[k];
        }
        buf.version += 1;
        w.progress += 1;
        if btags.iter().any(|x| *x != 0) {
            let name = buf.name.clone();
            let split = |v: &[u64]| -> Vec<Vec<u64>> {
                let mut start = 0;
                ends.iter()
                    .map(|&e| {
                        let r = v[start..e].to_vec();
                        start = e;
                        r
                    })
                    .collect()
            };
            let o = split(&offs);
            let mut start = 0;
            let d: Vec<Vec<u8>> = ends
                .iter()
                .map(|&e| {
                    let r = bytes[start..e].to_vec();
                    start = e;
                    r
                })
                .collect();
            start = 0;
            let t: Vec<Vec<Tag>> = ends
                .iter()
                .map(|&e| {
                    let r = btags[start..e].to_vec();
                    start = e;
                    r
                })
                .collect();
            let call = StoreCall {
                buffer: b,
                name: &name,
                offsets: &o,
                data: &d,
                tags: &t,
                wave: w.wid,
            };
            e.checker.store(e.tables, &call).map_err(tag_error)?;
        }
        e.mem.count(
            b,
            offs.first().copied().unwrap_or(0),
            offs.len() as u64,
            1,
            e.rank,
        );
        e.reports.stats.store_bytes += offs.len() as u64;
    }
    e.vm_issue(w, true, records);
    Ok(())
}

#[allow(clippy::too_many_lines)]
fn atomic(e: &mut Exec<'_>, w: &mut Wave, ins: &Ins, op: &str) -> Result<(), EmuError> {
    let wn = w.w;
    let exec = w.exec;
    let ops = &ins.operands;
    let wide = op.ends_with("b64")
        || op.ends_with("u64")
        || op.ends_with("i64")
        || op.ends_with("f64")
        || op.ends_with("x2");
    let returning_form = ops.len() >= 4;
    let returns = returning_form && e.facts.scope.returns.is_none_or(|f| ins.field(f).is_some());
    if wide && returning_form {
        return Err(limit(format!(
            "{}: only the non-returning form is modelled",
            ins.text
        )));
    }
    let (vaddr, data, saddr) = if returning_form {
        (&ops[1], &ops[2], ops.get(3))
    } else {
        (&ops[0], &ops[1], ops.get(2))
    };
    let addrs = global_addrs(w, ins, vaddr, saddr)?;
    let n: u64 = if wide { 8 } else { 4 };
    let x = w.vread(data, 0)?;
    let x2 = if wide { Some(w.vread(data, 1)?) } else { None };
    let mut lanes = vec![None; wn];
    for l in active(exec, wn) {
        if wide && addrs[l] % 8 != 0 {
            return Err(fault(format!(
                "{}: lane {l}'s address {:#x} is not 8-aligned",
                ins.text, addrs[l]
            )));
        }
        lanes[l] = Some(e.mem.access(addrs[l], n, true)?);
    }
    let bufs = bufs_of(&lanes);
    e.check_ipc(ins, &bufs, None)?;
    let kind = op.split('.').next().unwrap_or("");
    let float = op == "add.f32";
    // f32 tags: the checker sees the atomic first, as it sees the packed-bf16
    // one; R3c applies to the buffers it leaves to the machine.
    let mut checked = std::collections::BTreeSet::new();
    if float {
        for &b in &bufs {
            let lane_ids: Vec<usize> = active(exec, wn)
                .filter(|l| lanes[*l].is_some_and(|x| x.0 == b))
                .collect();
            let name = e.mem.bufs[b].name.clone();
            let o: Vec<Vec<u64>> = (lane_ids.iter())
                .map(|l| (0..4).map(|k| lanes[*l].map_or(0, |x| x.1) + k).collect())
                .collect();
            let d: Vec<Vec<u8>> = lane_ids
                .iter()
                .map(|l| x.v[*l].to_le_bytes().to_vec())
                .collect();
            let t: Vec<Vec<Tag>> = lane_ids.iter().map(|l| x.tag(*l).to_vec()).collect();
            let call = StoreCall {
                buffer: b,
                name: &name,
                offsets: &o,
                data: &d,
                tags: &t,
                wave: w.wid,
            };
            if e.checker.atomic(e.tables, &call).map_err(tag_error)? == AtomicAnswer::Checked {
                checked.insert(b);
            }
        }
    }
    let mut new_tags: Option<Vec<(usize, u64, Lane)>> = None;
    if float {
        let rec = |t: Tag| t < -1;
        let any = active(exec, wn).any(|l| {
            let (b, o) = lanes[l].unwrap_or((0, 0));
            !checked.contains(&b)
                && (x.tag(l).iter().any(|t| rec(*t))
                    || (0..4).any(|k| rec(e.mem.bufs[b].tags[(o + k) as usize])))
        });
        if any {
            let mut words: Vec<(usize, u64)> = Vec::new();
            for l in active(exec, wn) {
                let word = lanes[l].unwrap_or((0, 0));
                if words.contains(&word) {
                    return Err(limit(format!(
                        "{}: lanes adding records into one word (not modelled)",
                        ins.text
                    )));
                }
                words.push(word);
            }
            let mut out = Vec::new();
            for l in active(exec, wn) {
                let (b, o) = lanes[l].unwrap_or((0, 0));
                if checked.contains(&b) {
                    continue;
                }
                let mt: Lane = std::array::from_fn(|k| e.mem.bufs[b].tags[(o + k as u64) as usize]);
                let mut c = TagCtx {
                    tables: e.tables,
                    sum_mode: e.sum_mode,
                };
                let r = rules::float(&mut c, &[mt], &[x.tag(l)], false, true).map_err(tag_error)?;
                out.push((b, o, r[0]));
            }
            new_tags = Some(out);
        }
    }
    let site = e.site(w);
    for &b in &bufs {
        let offs: Vec<u64> = (0..wn)
            .filter_map(|l| lanes[l].filter(|x| x.0 == b))
            .flat_map(|(_, o)| (0..n).map(move |k| o + k))
            .collect();
        let size = e.mem.bufs[b].data.len();
        let name = e.mem.bufs[b].name.clone();
        let uncached = e.mem.bufs[b].uncached();
        if let (false, Some(s)) = (uncached, &mut e.mem.scope) {
            s.atomic(b, size, &name, &offs, site, &ins.text)?;
        }
        if returns {
            let _ = e.read_global(w, ins, b, &offs);
        }
    }
    let mut records = Vec::new();
    for &b in &bufs {
        let offs: Vec<u64> = (0..wn)
            .filter_map(|l| lanes[l].filter(|x| x.0 == b))
            .flat_map(|(_, o)| (0..n).map(move |k| o + k))
            .collect();
        records.push(e.record_write(w, ins, b, offs.clone(), true, true, false));
        e.mem.count(
            b,
            offs.first().copied().unwrap_or(0),
            offs.len() as u64,
            2,
            e.rank,
        );
    }
    let mut old = [0u32; MAX_W];
    for l in active(exec, wn) {
        let (b, o) = lanes[l].unwrap_or((0, 0));
        let buf = &mut e.mem.bufs[b];
        let o = o as usize;
        if wide {
            let cur = u64::from_le_bytes(buf.data[o..o + 8].try_into().unwrap_or([0; 8]));
            let d = u64::from(x.v[l]) | u64::from(x2.as_ref().map_or(0, |s| s.v[l])) << 32;
            let new = match kind {
                "umax" | "max" => cur.max(d),
                _ => cur.wrapping_add(d),
            };
            buf.data[o..o + 8].copy_from_slice(&new.to_le_bytes());
        } else {
            let cur = u32::from_le_bytes(buf.data[o..o + 4].try_into().unwrap_or([0; 4]));
            let d = x.v[l];
            let new = match op {
                "add" | "add.u32" => cur.wrapping_add(d),
                "or" | "or.b32" => cur | d,
                "umin" | "min.u32" => cur.min(d),
                "umax" | "max.u32" => cur.max(d),
                "swap" | "swap.b32" => d,
                // A NaN by `nan32`, the data first (measured on gfx1201).
                "add.f32" => nan32(f32_of(cur) + f32_of(d), &[d, cur]),
                _ => return Err(limit(format!("instruction not emulated: {}", ins.text))),
            };
            buf.data[o..o + 4].copy_from_slice(&new.to_le_bytes());
            old[l] = cur;
        }
        buf.version += 1;
    }
    if let Some(t) = new_tags {
        for (b, o, lane) in t {
            for k in 0..4 {
                e.mem.bufs[b].tags[(o + k) as usize] = lane[k as usize];
            }
        }
    }
    w.progress += 1;
    if returns {
        w.vwrite(&ops[0], 0, &old, None, exec)?;
    }
    e.vm_issue(w, !returning_form, records);
    Ok(())
}

fn pk_add_bf16(e: &mut Exec<'_>, w: &mut Wave, ins: &Ins) -> Result<(), EmuError> {
    let wn = w.w;
    let exec = w.exec;
    let ops = &ins.operands;
    let x = w.vread(&ops[0], 0)?;
    let (base, num, offs) = buffer_addrs(w, ins, &ops[1], &ops[2], ops.get(3))?;
    let mut lanes = vec![None; wn];
    for l in active(exec, wn) {
        if offs[l] + 4 <= num {
            lanes[l] = Some(e.mem.access(base.wrapping_add(offs[l]), 4, true)?);
        }
    }
    let bufs = bufs_of(&lanes);
    e.check_ipc(ins, &bufs, None)?;
    let site = e.site(w);
    let mut records = Vec::new();
    for &b in &bufs {
        let lane_ids: Vec<usize> = (0..wn)
            .filter(|l| lanes[*l].is_some_and(|x| x.0 == b))
            .collect();
        let offs: Vec<u64> = lane_ids
            .iter()
            .flat_map(|l| {
                let o = lanes[*l].map_or(0, |x| x.1);
                (0..4).map(move |k| o + k)
            })
            .collect();
        let size = e.mem.bufs[b].data.len();
        let name = e.mem.bufs[b].name.clone();
        let uncached = e.mem.bufs[b].uncached();
        if let (false, Some(s)) = (uncached, &mut e.mem.scope) {
            s.atomic(b, size, &name, &offs, site, &ins.text)?;
        }
        records.push(e.record_write(w, ins, b, offs.clone(), true, true, false));
        let o: Vec<Vec<u64>> = lane_ids
            .iter()
            .map(|l| {
                let o = lanes[*l].map_or(0, |x| x.1);
                (0..4).map(|k| o + k).collect()
            })
            .collect();
        let d: Vec<Vec<u8>> = lane_ids
            .iter()
            .map(|l| x.v[*l].to_le_bytes().to_vec())
            .collect();
        let t: Vec<Vec<Tag>> = lane_ids.iter().map(|l| x.tag(*l).to_vec()).collect();
        let call = StoreCall {
            buffer: b,
            name: &name,
            offsets: &o,
            data: &d,
            tags: &t,
            wave: w.wid,
        };
        let answer = e.checker.atomic(e.tables, &call).map_err(tag_error)?;
        let tagged = t.iter().flatten().any(|v| *v < -1)
            || o.iter()
                .flatten()
                .any(|off| e.mem.bufs[b].tags[*off as usize] < -1);
        if answer == AtomicAnswer::Scratch && tagged {
            let mut words: Vec<u64> = Vec::new();
            for l in &lane_ids {
                let word = lanes[*l].map_or(0, |x| x.1);
                if words.contains(&word) {
                    return Err(limit(format!(
                        "{}: lanes adding records into one word (not modelled)",
                        ins.text
                    )));
                }
                words.push(word);
            }
            for l in &lane_ids {
                let off = lanes[*l].map_or(0, |x| x.1) as usize;
                for h in 0..2 {
                    let mt = [
                        e.mem.bufs[b].tags[off + 2 * h],
                        e.mem.bufs[b].tags[off + 2 * h + 1],
                    ];
                    let xt = [x.tag(*l)[2 * h], x.tag(*l)[2 * h + 1]];
                    let mut c = TagCtx {
                        tables: e.tables,
                        sum_mode: e.sum_mode,
                    };
                    let r = rules::float(
                        &mut c,
                        &[[mt[0], mt[1], mt[0], mt[1]]],
                        &[[xt[0], xt[1], xt[0], xt[1]]],
                        false,
                        true,
                    )
                    .map_err(tag_error)?;
                    e.mem.bufs[b].tags[off + 2 * h] = r[0][0];
                    e.mem.bufs[b].tags[off + 2 * h + 1] = r[0][1];
                }
            }
        }
        for l in &lane_ids {
            let off = lanes[*l].map_or(0, |x| x.1) as usize;
            let buf = &mut e.mem.bufs[b];
            let cur = u32::from_le_bytes(buf.data[off..off + 4].try_into().unwrap_or([0; 4]));
            let mut new = 0u32;
            for h in 0..2 {
                let (ah, ch) = ((cur >> (16 * h)) & 0xFFFF, (x.v[*l] >> (16 * h)) & 0xFFFF);
                let r = f32_of(ah << 16) + f32_of(ch << 16);
                // A NaN (measured on gfx1201): the data's, then memory's,
                // quieted, or 0xFFC0; else one RNE rounding to bf16.
                let v = if r.is_nan() {
                    [ch, ah]
                        .iter()
                        .find(|u| **u & 0x7FFF > 0x7F80)
                        .map_or(0xFFC0, |u| u | 0x40)
                } else {
                    bf16_rne(r.to_bits())
                };
                new |= v << (16 * h);
            }
            buf.data[off..off + 4].copy_from_slice(&new.to_le_bytes());
            buf.version += 1;
        }
    }
    w.progress += 1;
    e.vm_issue(w, true, records);
    Ok(())
}

/// Scalar memory. `Ok(false)` when `canon` is none of it.
#[allow(clippy::too_many_lines)]
pub fn smem(e: &mut Exec<'_>, w: &mut Wave, ins: &Ins, canon: &str) -> Result<bool, EmuError> {
    let parts: Vec<&str> = canon.split('.').collect();
    let ops = &ins.operands;
    let addr = |w: &Wave| -> Result<u64, EmuError> {
        let base = w.sread64(&ops[1])?;
        let off = match ops.get(2) {
            Some(o) => w.sread(o)? as u32 as i32 as i64,
            None => 0,
        } + int_field(ins, "offset", 0) as i64;
        Ok(base.wrapping_add(off as u64))
    };
    // A scalar load at device scope or wider reads past the scalar cache:
    // the poll report counts it.
    let polls = level(e, ins, false, false) >= Level::Agent;
    match parts.as_slice() {
        ["s", "load", last] | ["s", "buffer", "load", last] => {
            let Some((n, _)) = width(last) else {
                return Ok(false);
            };
            let a = addr(w)?;
            let Some((b, off)) = e.mem.resolve(a) else {
                return Err(fault(format!("load from unmapped address {a:#x}")));
            };
            e.check_ipc(ins, &[b], Some(false))?;
            let size = e.mem.bufs[b].size();
            if off + u64::from(n) > size {
                return Err(fault(format!(
                    "{}: a scalar load past the end of {}",
                    ins.text, e.mem.bufs[b].name
                )));
            }
            if polls {
                let line = (a / 128) * 128;
                let entry = e.reports.spoll.entry((e.pc, line)).or_default();
                entry.0 += 1;
                entry.1.insert((e.rank, w.wgx, w.wid));
            }
            let offs: Vec<u64> = (0..u64::from(n)).map(|k| off + k).collect();
            e.mem.count(b, off, u64::from(n), 0, e.rank);
            let (data, _) = e.read_global(w, ins, b, &offs);
            let Base::Reg(f, r, _) = &ops[0].base else {
                return Err(fault(format!("scalar dest {}", ops[0].spelling())));
            };
            if f != "s" {
                return Err(fault(format!("scalar dest {}", ops[0].spelling())));
            }
            for d in 0..(n / 4) as usize {
                let v = u32::from_le_bytes([
                    data[4 * d],
                    data[4 * d + 1],
                    data[4 * d + 2],
                    data[4 * d + 3],
                ]);
                let i = *r as usize + d;
                if i < w.s.len() {
                    w.s[i] = v;
                }
            }
        }
        _ => return Ok(false),
    }
    Ok(true)
}

/// LDS ops. `Ok(false)` when `canon` is none of them.
#[allow(clippy::too_many_lines)]
pub fn ds(e: &mut Exec<'_>, w: &mut Wave, ins: &Ins, canon: &str) -> Result<bool, EmuError> {
    let parts: Vec<&str> = canon.split('.').collect();
    let ops = &ins.operands;
    let wn = w.w;
    let exec = w.exec;
    let off = int_field(ins, "offset", 0) as u64;
    let pc = e.pc;
    let size = e.lds.size() as u64;
    match parts.as_slice() {
        ["ds", "bpermute", "b32"] => {
            let (a, x) = (w.vread(&ops[1], 0)?, w.vread(&ops[2], 0)?);
            let mut out = [0u32; MAX_W];
            let mut t = vec![[0; 4]; wn];
            // The source lane is (address + offset) / 4, within the wave;
            // an inactive one gives 0 (measured on gfx1201).
            for l in 0..wn {
                let s = ((u64::from(a.v[l]) + off) >> 2) as usize % wn;
                if exec >> s & 1 != 0 {
                    out[l] = x.v[s];
                    t[l] = x.tag(s);
                }
            }
            w.vwrite(&ops[0], 0, &out, x.tagged.then_some(t.as_slice()), exec)?;
            w.seq[2] += 1;
        }
        ["ds", "load", rest @ ..] => {
            let last = rest.last().copied().unwrap_or("");
            let Some((n, signed)) = width(last) else {
                return Ok(false);
            };
            let a = w.vread(&ops[1], 0)?;
            let mut addrs = vec![0u64; wn];
            for l in active(exec, wn) {
                addrs[l] = u64::from(a.v[l]) + off;
                if addrs[l] + u64::from(n) > size {
                    return Err(fault(format!(
                        "wave {} {}: LDS read out of range (max {})",
                        w.wid, ins.text, size
                    )));
                }
            }
            count_banks(e, true, n, &addrs, exec, wn);
            let bytes: Vec<u32> = active(exec, wn)
                .flat_map(|l| {
                    let a = addrs[l];
                    (0..u64::from(n)).map(move |k| (a + k) as u32)
                })
                .collect();
            let racers = e
                .lds
                .check_read(w.wid, e.epoch, pc, &bytes, e.texts, &ins.text)?;
            if !racers.is_empty() {
                let list: Vec<String> = racers.iter().map(ToString::to_string).collect();
                return Err(fault(format!(
                    "wave {} {}: reads LDS bytes written by wave(s) [{}] since the last barrier",
                    w.wid,
                    ins.text,
                    list.join(", ")
                )));
            }
            let dwords = n.div_ceil(4) as usize;
            let mut vals = vec![[0u32; MAX_W]; dwords];
            let mut tags = vec![vec![[0 as Tag; 4]; wn]; dwords];
            let mut tagged = false;
            for l in active(exec, wn) {
                for k in 0..n as usize {
                    let j = addrs[l] as usize + k;
                    vals[k / 4][l] |= u32::from(e.lds.data[j]) << (8 * (k % 4));
                    tags[k / 4][l][k % 4] = e.lds.tags[j];
                    tagged |= e.lds.tags[j] != 0;
                }
                if n < 4 && signed {
                    let bits = 8 * n;
                    let v = vals[0][l];
                    vals[0][l] = (((v << (32 - bits)) as i32) >> (32 - bits)) as u32;
                }
            }
            for d in 0..dwords {
                w.vwrite(
                    &ops[0],
                    d as u32,
                    &vals[d],
                    tagged.then_some(tags[d].as_slice()),
                    exec,
                )?;
            }
            let seq = w.seq[2];
            w.seq[2] += 1;
            e.reports.stats.ds_read_bytes += bytes.len() as u64;
            e.lds.record_read(w.wid, e.epoch, pc, seq, bytes);
        }
        // DS_STORE_B8_D16_HI / DS_STORE_B16_D16_HI:
        // MEM[ADDR] = DATA[23:16] / DATA[31:16], the data from byte 2.
        ["ds", "store", last] | ["ds", "store", last @ ("b8" | "b16"), "d16", "hi"] => {
            let Some((n, _)) = width(last) else {
                return Ok(false);
            };
            let from = if parts.len() == 5 { 2 } else { 0 };
            let a = w.vread(&ops[0], 0)?;
            let dwords = n.div_ceil(4) as usize;
            let mut srcs = Vec::with_capacity(dwords);
            for d in 0..dwords {
                srcs.push(w.vread(&ops[1], d as u32)?);
            }
            let mut addrs = vec![0u64; wn];
            for l in active(exec, wn) {
                addrs[l] = u64::from(a.v[l]) + off;
                if addrs[l] + u64::from(n) > size {
                    return Err(fault(format!(
                        "wave {} {}: LDS write out of range (max {}, LDS {})",
                        w.wid,
                        ins.text,
                        addrs[l] + u64::from(n),
                        size
                    )));
                }
            }
            count_banks(e, false, n, &addrs, exec, wn);
            let bytes: Vec<u32> = active(exec, wn)
                .flat_map(|l| {
                    let a = addrs[l];
                    (0..u64::from(n)).map(move |k| (a + k) as u32)
                })
                .collect();
            e.lds
                .check_write(w.wid, e.epoch, pc, &bytes, false, e.texts, &ins.text)?;
            for l in active(exec, wn) {
                for k in 0..n as usize {
                    let (j, b) = (addrs[l] as usize + k, k + from);
                    e.lds.data[j] = (srcs[b / 4].v[l] >> (8 * (b % 4))) as u8;
                    e.lds.tags[j] = srcs[b / 4].tag(l)[b % 4];
                }
            }
            let seq = w.seq[2];
            w.seq[2] += 1;
            e.reports.stats.lds_write_bytes += bytes.len() as u64;
            w.progress += 1;
            e.lds.record_write(w.wid, e.epoch, pc, seq, bytes, false);
        }
        ["ds", op @ ("add" | "or" | "cmpstore" | "max" | "min"), rest @ ..] => {
            let rtn = rest.first() == Some(&"rtn");
            let ty = rest.last().copied().unwrap_or("u32");
            let base = usize::from(rtn);
            let a = w.vread(&ops[base], 0)?;
            let x = w.vread(&ops[base + 1], 0)?;
            let y = if *op == "cmpstore" {
                Some(w.vread(&ops[base + 2], 0)?)
            } else {
                None
            };
            let mut words = Vec::new();
            for l in active(exec, wn) {
                let addr = u64::from(a.v[l]) + off;
                if addr % 4 != 0 || addr + 4 > size {
                    return Err(fault(format!(
                        "wave {} {}: an LDS atomic at {addr} (out of range or not 4-aligned)",
                        w.wid, ins.text
                    )));
                }
                words.push((l, addr as usize));
            }
            let bytes: Vec<u32> = words
                .iter()
                .flat_map(|(_, a)| (0..4).map(move |k| (*a + k) as u32))
                .collect();
            for &j in &bytes {
                if e.lds.wrace[j as usize] >= 0 {
                    e.lds
                        .check_read(w.wid, e.epoch, pc, &[j], e.texts, &ins.text)?;
                }
            }
            let fresh = bytes
                .iter()
                .filter(|j| e.lds.writer[**j as usize] < 0)
                .count();
            if fresh > 0 {
                e.lds.unwritten.push((pc, fresh as u64));
            }
            e.lds
                .check_write(w.wid, e.epoch, pc, &bytes, true, e.texts, &ins.text)?;
            let mut old = [0u32; MAX_W];
            for (l, addr) in &words {
                let cur =
                    u32::from_le_bytes(e.lds.data[*addr..*addr + 4].try_into().unwrap_or([0; 4]));
                let v = x.v[*l];
                let new = match (*op, ty) {
                    ("add", "f32") => nan32(f32_of(cur) + f32_of(v), &[v, cur]),
                    ("add", _) => cur.wrapping_add(v),
                    ("or", _) => cur | v,
                    ("max", "u32") => cur.max(v),
                    ("min", "u32") => cur.min(v),
                    ("max", _) => (cur as i32).max(v as i32) as u32,
                    ("min", _) => (cur as i32).min(v as i32) as u32,
                    _ => {
                        // ds_cmpstore: data0 (v) is stored where data1
                        // (src) equals the old value.
                        let (cmp, src) = (y.as_ref().map_or(0, |y| y.v[*l]), v);
                        if cur == cmp {
                            src
                        } else {
                            cur
                        }
                    }
                };
                e.lds.data[*addr..*addr + 4].copy_from_slice(&new.to_le_bytes());
                for k in 0..4 {
                    e.lds.tags[*addr + k] = 0;
                    e.lds.epoch[*addr + k] = e.epoch;
                }
                old[*l] = cur;
            }
            let seq = w.seq[2];
            w.seq[2] += 1;
            if rtn {
                w.vwrite(&ops[0], 0, &old, None, exec)?;
            }
            e.reports.stats.lds_write_bytes += bytes.len() as u64;
            w.progress += 1;
            e.lds.record_write(w.wid, e.epoch, pc, seq, bytes, true);
        }
        _ => return Ok(false),
    }
    Ok(true)
}

fn count_banks(e: &mut Exec<'_>, read: bool, n: u32, addrs: &[u64], exec: u64, wn: usize) {
    if exec == 0 {
        return;
    }
    let Some((banks, phase)) = e.facts.bank_rule(read, n).cloned() else {
        return;
    };
    let rule = super::banks::BankRule { banks, phase };
    let act: Vec<bool> = (0..wn).map(|l| exec >> l & 1 != 0).collect();
    let (cycles, conflicts) = super::banks::bank_count(&rule, addrs, &act);
    let entry = e.reports.banks.entry(e.pc).or_insert([0; 3]);
    entry[0] += 1;
    entry[1] += cycles;
    entry[2] += conflicts;
}

/// Whether a field is set.
pub fn has(ins: &Ins, name: &str) -> bool {
    matches!(
        ins.field(name),
        Some(Field::True | Field::Int(_) | Field::List(_))
    )
}

/// An LDS for a caller that has none (ALU-only runs).
pub fn empty_lds() -> Lds {
    Lds::new(0, 1)
}
