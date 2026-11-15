//! A wave's state and operand access.

use std::collections::BTreeMap;

use super::facts::Facts;
use super::POISON;
use crate::tags::{AccState, Lane, Tables, Tag, ACC_MARK};
use crate::{Base, Cause, EmuError, Operand};

/// The most lanes a wave has.
pub const MAX_W: usize = 64;

/// Every lane untagged.
static UNTAGGED: [Lane; MAX_W] = [[0; 4]; MAX_W];

/// One operand's per-lane values and tags (empty: untagged).
#[derive(Clone)]
pub struct Src {
    /// The values, lane by lane (W of them used).
    pub v: [u32; MAX_W],
    /// The tags, lane by lane; empty when every lane is untagged.
    pub t: Vec<Lane>,
    /// Whether any tag may be nonzero.
    pub tagged: bool,
}

impl Src {
    /// Every lane `v`, untagged.
    pub fn splat(v: u32) -> Src {
        Src {
            v: [v; MAX_W],
            t: Vec::new(),
            tagged: false,
        }
    }

    /// The tags of the first `w` lanes.
    pub fn tags(&self, w: usize) -> &[Lane] {
        if self.t.is_empty() {
            &UNTAGGED[..w]
        } else {
            &self.t[..w]
        }
    }

    /// The tags of lane `l`.
    #[inline]
    pub fn tag(&self, l: usize) -> Lane {
        self.t.get(l).copied().unwrap_or([0; 4])
    }

    /// The tags made writable (filled with zeros when untagged).
    pub fn tags_mut(&mut self) -> &mut Vec<Lane> {
        if self.t.is_empty() {
            self.t = vec![[0; 4]; MAX_W];
        }
        self.tagged = true;
        &mut self.t
    }
}

/// A register file's tags: one optional lane array per register (none:
/// untagged in every lane).
pub type FileTags = Vec<Option<Box<[Lane]>>>;

/// A VMEM op not yet covered by a wait.
#[derive(Clone, Debug)]
pub struct VmOp {
    /// Its counter.
    pub counter: usize,
    /// Its sequence number on that counter.
    pub seq: u64,
    /// A store or atomic (else a load).
    pub store: bool,
    /// The write records it made.
    pub records: Vec<usize>,
}

/// One wave's state.
#[derive(Clone)]
pub struct Wave {
    /// Its id within its workgroup.
    pub wid: usize,
    /// W.
    pub w: usize,
    /// Vector registers `[reg][lane]`.
    pub v: Vec<u32>,
    /// Their tags.
    pub vt: FileTags,
    /// Accumulator registers.
    pub a: Vec<u32>,
    /// Their tags.
    pub at: FileTags,
    /// Per-register GEMM provenance accumulator entries (`a` then `v`).
    pub acc: AccState,
    /// Scalar registers.
    pub s: Vec<u32>,
    /// EXEC.
    pub exec: u64,
    /// VCC.
    pub vcc: u64,
    /// VCC_HI on wave32: a plain scalar, never part of a mask.
    pub vcc_hi: u32,
    /// EXEC_HI on wave32: EXEC[63:32], kept but never acting on a lane
    /// at dispatch it is zero because only the low 32 lanes are active.
    pub exec_hi: u32,
    /// M0.
    pub m0: u32,
    /// SCC.
    pub scc: bool,
    /// ttmp7 (where a target puts workgroup id y, bits 0–15).
    pub ttmp7: u32,
    /// ttmp9 (where a target puts workgroup id x).
    pub ttmp9: u32,
    /// The next instruction's index.
    pub pc: usize,
    /// Ended.
    pub done: bool,
    /// Stopped at a barrier.
    pub at_barrier: bool,
    /// The barrier it stopped at is an LDS fence.
    pub at_fence: bool,
    /// Issue counts per counter (0: VMEM or load, 1: store, 2: DS).
    pub seq: [u64; 3],
    /// VMEM ops not covered, in issue order.
    pub vm_ops: Vec<VmOp>,
    /// The newest of its write records a wait covered, or −1.
    pub mark: i64,
    /// The uniform-claim values recorded since the last barrier.
    pub uniform: Vec<(usize, Vec<Vec<u32>>)>,
    /// Register writes not yet landed, in issue order.
    pub pending: Vec<super::pending::Pending>,
    /// Progress: stores, LDS writes, matrix ops, atomics, stale reads.
    pub progress: u64,
    /// Instructions run.
    pub nins: u64,
    /// Its workgroup's serial in the memory.
    pub wg: usize,
    /// Its workgroup's x id.
    pub wgx: u32,
    /// Its L2's index on its rank.
    pub xcd: usize,
    /// The previous loop-head hash and the progress then.
    pub head: Option<(u64, u64)>,
    /// Buffers read in this iteration → their version at the first read.
    pub watch: BTreeMap<usize, u64>,
    /// Parked on these (buffer, version) pairs.
    pub parked: Option<Vec<(usize, u64)>>,
    /// The round-counter SGPRs of a parked bounded poll.
    pub spin_regs: Vec<u32>,
}

fn fault(message: String) -> EmuError {
    EmuError {
        cause: Some(Cause::Fault),
        message,
    }
}

/// A register file's letter for messages.
fn reg_name(file: &str, r: u32, n: u32) -> String {
    if n == 1 {
        format!("{file}{r}")
    } else {
        format!("{file}[{r}:{}]", r + n - 1)
    }
}

impl Wave {
    /// A wave at dispatch: every register POISON and untagged, EXEC all
    /// lanes, SCC 0.
    pub fn new(wid: usize, facts: &Facts, nv: usize, na: usize) -> Wave {
        let w = facts.w;
        Wave {
            wid,
            w,
            v: vec![POISON; nv * w],
            vt: vec![None; nv],
            a: vec![POISON; na * w],
            at: vec![None; na],
            acc: AccState::new(na + nv, w),
            s: vec![POISON; facts.ns],
            exec: facts.all_lanes(),
            vcc: if w == 64 {
                u64::from(POISON) | u64::from(POISON) << 32
            } else {
                u64::from(POISON)
            },
            vcc_hi: POISON,
            exec_hi: 0,
            m0: POISON,
            scc: false,
            ttmp7: POISON,
            ttmp9: POISON,
            pc: 0,
            done: false,
            at_barrier: false,
            at_fence: false,
            seq: [0; 3],
            vm_ops: Vec::new(),
            mark: -1,
            uniform: Vec::new(),
            pending: Vec::new(),
            progress: 0,
            nins: 0,
            wg: 0,
            wgx: 0,
            xcd: 0,
            head: None,
            watch: BTreeMap::new(),
            parked: None,
            spin_regs: Vec::new(),
        }
    }

    /// The lane mask of every lane.
    pub fn all(&self) -> u64 {
        if self.w == 64 {
            u64::MAX
        } else {
            (1u64 << self.w) - 1
        }
    }

    /// The number of VGPRs.
    pub fn nv(&self) -> usize {
        self.vt.len()
    }

    /// The `AccState` entry index for `(file, reg)`: `a` at `0..na`, `v` at `na..na+nv`.
    pub fn acc_index(&self, file: &str, reg: usize) -> Option<usize> {
        match file {
            "a" => (reg < self.at.len()).then_some(reg),
            "v" => (reg < self.vt.len()).then_some(self.at.len() + reg),
            _ => None,
        }
    }

    /// Whether register `(file, reg)` currently carries `ACC_MARK` (`-1`).
    pub fn is_acc_marked(&self, file: &str, reg: usize) -> bool {
        self.vfile(file)
            .and_then(|(_, ftags)| ftags.get(reg))
            .and_then(|slot| slot.as_ref())
            .is_some_and(|lanes| lanes.iter().any(|l| l.contains(&ACC_MARK)))
    }

    /// Materializes `ACC_MARK` in register `(file, reg)` into concrete `Record` tags.
    pub fn materialize_reg(&mut self, tables: &mut Tables, file: &str, reg: usize) {
        if !self.is_acc_marked(file, reg) {
            return;
        }
        let Some(idx) = self.acc_index(file, reg) else {
            return;
        };
        let lanes = self.acc.materialize(tables, idx);
        let any = lanes.iter().any(|l| *l != [0; 4]);
        if let Some((_, ftags)) = self.vfile_mut(file) {
            if let Some(slot) = ftags.get_mut(reg) {
                *slot = any.then(|| lanes.into_boxed_slice());
            }
        }
    }

    /// Materializes any `ACC_MARK` registers referenced by an operand.
    pub fn materialize_operand(&mut self, tables: &mut Tables, op: &Operand) {
        if let Base::Reg(f, r, n) = &op.base {
            if f == "v" || f == "a" {
                for d in 0..*n {
                    self.materialize_reg(tables, f, (*r + d) as usize);
                }
            }
        }
    }

    /// Materializes every marked register in `file` (`"v"` or `"a"`).
    pub fn materialize_file(&mut self, tables: &mut Tables, file: &str) {
        let len = match file {
            "v" => self.vt.len(),
            "a" => self.at.len(),
            _ => 0,
        };
        for reg in 0..len {
            self.materialize_reg(tables, file, reg);
        }
    }

    fn vfile(&self, file: &str) -> Option<(&Vec<u32>, &FileTags)> {
        match file {
            "v" => Some((&self.v, &self.vt)),
            "a" => Some((&self.a, &self.at)),
            _ => None,
        }
    }

    fn vfile_mut(&mut self, file: &str) -> Option<(&mut Vec<u32>, &mut FileTags)> {
        match file {
            "v" => Some((&mut self.v, &mut self.vt)),
            "a" => Some((&mut self.a, &mut self.at)),
            _ => None,
        }
    }

    /// Whether vcc's part `r` is wave32's VCC_HI, a plain scalar.
    fn hi(&self, r: u32) -> bool {
        self.w == 32 && r == 1
    }

    /// EXEC as the 64-bit register: on wave32 EXEC_HI above the mask
    ///
    pub fn exec64(&self) -> u64 {
        if self.w == 32 {
            self.exec | u64::from(self.exec_hi) << 32
        } else {
            self.exec
        }
    }

    /// Writes EXEC as the 64-bit register: the lanes' mask, and on wave32
    /// the upper half into EXEC_HI.
    pub fn set_exec64(&mut self, v: u64) {
        self.exec = v & self.all();
        if self.w == 32 {
            self.exec_hi = (v >> 32) as u32;
        }
    }

    /// A scalar read of one dword; the whole of a two-dword mask is
    /// not truncated.
    pub fn sread(&self, o: &Operand) -> Result<u64, EmuError> {
        let v = match &o.base {
            Base::Literal(v, _) => u64::from(*v),
            Base::Reg(f, r, n) => match f.as_str() {
                "s" => u64::from(*self.s.get(*r as usize).ok_or_else(|| past("s", *r))?),
                "exec" => mask_part(self.exec64(), *r, *n),
                "vcc" if self.hi(*r) => u64::from(self.vcc_hi),
                "vcc" => mask_part(self.vcc, *r, *n),
                "m0" => u64::from(self.m0),
                "scc" => u64::from(self.scc),
                "null" => 0,
                "ttmp7" => u64::from(self.ttmp7),
                "ttmp9" => u64::from(self.ttmp9),
                "off" => return Err(fault("scalar read of off".to_owned())),
                other => return Err(fault(format!("scalar read of {}", reg_name(other, *r, *n)))),
            },
            Base::Label(l) => return Err(fault(format!("scalar read of label {l}"))),
        };
        Ok(if o.neg {
            u64::from((v as u32).wrapping_neg())
        } else {
            v
        })
    }

    /// A scalar read of 64 bits.
    pub fn sread64(&self, o: &Operand) -> Result<u64, EmuError> {
        match &o.base {
            Base::Literal(v, _) => Ok(if *v >= 0x8000_0000 {
                u64::from(*v) | 0xFFFF_FFFF_0000_0000
            } else {
                u64::from(*v)
            }),
            Base::Reg(f, r, _) if f == "s" => {
                let r = *r as usize;
                let lo = *self.s.get(r).ok_or_else(|| past("s", r as u32))?;
                let hi = *self.s.get(r + 1).ok_or_else(|| past("s", r as u32 + 1))?;
                Ok(u64::from(lo) | u64::from(hi) << 32)
            }
            Base::Reg(f, r, n) if f == "exec" && self.w == 32 => match (r, n) {
                (0, 1) => Ok(self.exec),
                (0, _) => Ok(self.exec64()),
                _ => Err(not_mask_exec()),
            },
            Base::Reg(f, _, _) if f == "exec" => Ok(self.exec),
            Base::Reg(f, r, _) if f == "vcc" && self.hi(*r) => Err(not_mask()),
            Base::Reg(f, _, _) if f == "vcc" => Ok(self.vcc),
            Base::Reg(f, _, _) if f == "null" => Ok(0),
            Base::Reg(f, r, n) => Err(fault(format!(
                "64-bit scalar read of {}",
                reg_name(f, *r, *n)
            ))),
            Base::Label(l) => Err(fault(format!("64-bit scalar read of label {l}"))),
        }
    }

    /// A scalar write of one dword.
    pub fn swrite(&mut self, o: &Operand, v: u64) -> Result<(), EmuError> {
        let Base::Reg(f, r, n) = &o.base else {
            return Err(fault(format!("scalar dest {}", o.spelling())));
        };
        let (r, n) = (*r, *n);
        match f.as_str() {
            "s" => {
                let i = r as usize;
                *self.s.get_mut(i).ok_or_else(|| past("s", r))? = v as u32;
            }
            "exec" => {
                let wide = set_mask_part(self.exec64(), r, n, v);
                self.set_exec64(wide);
            }
            "vcc" if self.hi(r) => self.vcc_hi = v as u32,
            "vcc" => self.vcc = set_mask_part(self.vcc, r, n, v) & self.all(),
            "m0" => self.m0 = v as u32,
            "null" => {}
            "scc" => self.scc = v & 1 != 0,
            _ => return Err(fault(format!("scalar dest {}", o.spelling()))),
        }
        Ok(())
    }

    /// A 64-bit scalar write: an SGPR pair, or exec / vcc whole; on wave32
    /// `exec` (two dwords) writes EXEC_HI with the high half, and a lane
    /// mask into one SGPR writes that SGPR alone (RDNA4 ISA guide:
    /// "Wave32 writes only the low 32 bits of VCC, EXEC or a single SGPR").
    pub fn swrite64(&mut self, o: &Operand, v: u64) -> Result<(), EmuError> {
        let Base::Reg(f, r, n) = &o.base else {
            return Err(fault(format!("scalar dest {}", o.spelling())));
        };
        match f.as_str() {
            "s" if *n == 1 && self.w == 32 => {
                let i = *r as usize;
                *self.s.get_mut(i).ok_or_else(|| past("s", *r))? = v as u32;
            }
            "s" => {
                let i = *r as usize;
                if i + 1 >= self.s.len() {
                    return Err(past("s", *r + 1));
                }
                self.s[i] = v as u32;
                self.s[i + 1] = (v >> 32) as u32;
            }
            "exec" if self.w == 32 => match (r, n) {
                (0, 1) => self.exec = v & self.all(),
                (0, _) => self.set_exec64(v),
                _ => return Err(not_mask_exec()),
            },
            "exec" => self.exec = v & self.all(),
            "vcc" if self.hi(*r) => return Err(not_mask()),
            "vcc" => self.vcc = v & self.all(),
            "null" => {}
            _ => return Err(fault(format!("scalar dest {}", o.spelling()))),
        }
        Ok(())
    }

    /// The vector read of dword `d` of an operand: values and tags
    /// per lane, then abs and neg on bit 31.
    pub fn vread(&self, o: &Operand, d: u32) -> Result<Src, EmuError> {
        let w = self.w;
        let mut src = match &o.base {
            // An inline integer is its 64-bit value; a literal or an inline
            // float fills a 64-bit operand by the op's type (the high dword
            // of an f64, an extended integer): not modelled.
            Base::Literal(v, text) if d > 0 => match text.parse::<i64>() {
                Ok(k) if (-16..=64).contains(&k) => Src::splat(if (*v as i32) < 0 { u32::MAX } else { 0 }),
                _ => {
                    return Err(EmuError {
                        cause: Some(Cause::Emulator),
                        message: format!(
                            "{}: a literal or an inline float as a 64-bit vector source is not modelled",
                            o.spelling()
                        ),
                    })
                }
            },
            Base::Literal(v, _) => Src::splat(*v),
            Base::Reg(f, r, n) if f == "v" || f == "a" => {
                let (vals, tags) = self
                    .vfile(f)
                    .ok_or_else(|| fault(format!("vector read of {f}")))?;
                let reg = (*r + d) as usize;
                if reg >= tags.len() {
                    return Err(past(f, *r + d.min(n.saturating_sub(1))));
                }
                let mut s = Src::splat(0);
                s.v[..w].copy_from_slice(&vals[reg * w..reg * w + w]);
                if let Some(t) = &tags[reg] {
                    s.t = t.to_vec();
                    s.tagged = true;
                }
                s
            }
            Base::Reg(f, r, n) => {
                let v = if *n >= 2 && (f == "s" || f == "exec" || f == "vcc") {
                    let wide = self.sread64(o)?;
                    (wide >> (32 * (d & 1))) as u32
                } else {
                    let plain = Operand {
                        base: Base::Reg(f.clone(), *r, *n),
                        neg: false,
                        abs: false,
                    };
                    self.sread(&plain)? as u32
                };
                Src::splat(v)
            }
            Base::Label(l) => return Err(fault(format!("vector read of label {l}"))),
        };
        if o.abs {
            for v in &mut src.v[..w] {
                *v &= 0x7FFF_FFFF;
            }
        }
        if o.neg {
            for v in &mut src.v[..w] {
                *v ^= 0x8000_0000;
            }
        }
        Ok(src)
    }

    /// The vector write of dword `d`: only the lanes of `mask`
    /// change, values and tags (none: untagged).
    pub fn vwrite(
        &mut self,
        o: &Operand,
        d: u32,
        vals: &[u32],
        tags: Option<&[Lane]>,
        mask: u64,
    ) -> Result<(), EmuError> {
        let w = self.w;
        let Base::Reg(f, r, _) = &o.base else {
            return Err(fault(format!("vector dest {}", o.spelling())));
        };
        let (file, ftags) = self
            .vfile_mut(f)
            .ok_or_else(|| fault(format!("vector dest {}", o.spelling())))?;
        let reg = (*r + d) as usize;
        if reg >= ftags.len() {
            return Err(past(f, *r + d));
        }
        let full = mask & lanes(w) == lanes(w);
        let row = &mut file[reg * w..reg * w + w];
        if full {
            row.copy_from_slice(&vals[..w]);
        } else {
            for (l, slot) in row.iter_mut().enumerate() {
                if mask >> l & 1 != 0 {
                    *slot = vals[l];
                }
            }
        }
        let slot = &mut ftags[reg];
        match tags {
            Some(t) if t[..w].iter().any(|l| *l != [0; 4]) => {
                if full {
                    *slot = Some(t[..w].to_vec().into_boxed_slice());
                } else {
                    let dst = slot.get_or_insert_with(|| vec![[0; 4]; w].into_boxed_slice());
                    for (l, lane) in dst.iter_mut().enumerate() {
                        if mask >> l & 1 != 0 {
                            *lane = t[l];
                        }
                    }
                }
            }
            _ => {
                if full {
                    *slot = None;
                } else if let Some(dst) = slot.as_mut() {
                    for (l, lane) in dst.iter_mut().enumerate() {
                        if mask >> l & 1 != 0 {
                            *lane = [0; 4];
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// The tags of one register byte (`file` `v` or `a`).
    pub fn tag_at(&self, file: &str, reg: usize, lane: usize, byte: usize) -> Tag {
        self.vfile(file)
            .and_then(|(_, t)| t.get(reg))
            .and_then(|t| t.as_ref())
            .map_or(0, |t| t[lane][byte])
    }

    /// Sets a register range POISON with no tags in every lane, or an
    /// SGPR range POISON.
    pub fn poison(&mut self, file: &str, first: u32, count: u32) {
        let w = self.w;
        for r in first..first + count {
            let r = r as usize;
            match file {
                "s" => {
                    if r < self.s.len() {
                        self.s[r] = POISON;
                    }
                }
                f => {
                    if let Some((vals, tags)) = self.vfile_mut(f) {
                        if r < tags.len() {
                            vals[r * w..r * w + w].fill(POISON);
                            tags[r] = None;
                        }
                    }
                }
            }
        }
    }
}

fn lanes(w: usize) -> u64 {
    if w == 64 {
        u64::MAX
    } else {
        (1u64 << w) - 1
    }
}

/// The message of vcc_hi used as a lane mask on wave32; the VALU
/// step prefixes the instruction's text.
pub const NOT_MASK: &str = "vcc_hi is not a lane mask on wave32";

fn not_mask() -> EmuError {
    fault(NOT_MASK.to_owned())
}

/// The message of exec_hi used as a lane mask on wave32 (RDNA4 ISA guide
/// wave32 may not use EXEC_HI for mask operations).
pub const NOT_MASK_EXEC: &str = "exec_hi is not a lane mask on wave32";

fn not_mask_exec() -> EmuError {
    fault(NOT_MASK_EXEC.to_owned())
}

fn mask_part(mask: u64, part: u32, n: u32) -> u64 {
    if n >= 2 {
        mask
    } else {
        (mask >> (32 * part)) & 0xFFFF_FFFF
    }
}

fn set_mask_part(mask: u64, part: u32, n: u32, v: u64) -> u64 {
    if n >= 2 {
        v
    } else {
        let shift = 32 * part;
        (mask & !(0xFFFF_FFFFu64 << shift)) | ((v & 0xFFFF_FFFF) << shift)
    }
}

/// A register past its file.
fn past(file: &str, r: u32) -> EmuError {
    EmuError {
        cause: Some(Cause::Emulator),
        message: format!("register {file}{r} is past its file (internal)"),
    }
}
