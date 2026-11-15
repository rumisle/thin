//! A workgroup's LDS with its per-byte race state.

use crate::tags::Tag;
use crate::{Cause, EmuError};

/// An LDS op of one wave not yet covered.
#[derive(Clone, Debug)]
pub struct LdsOp {
    /// Its sequence number on its completion counter (`ds` or `vm`).
    pub seq: u64,
    /// The bytes it touched.
    pub bytes: Vec<u32>,
    /// Its instruction.
    pub pc: usize,
    /// Its issue epoch.
    pub epoch: i64,
    /// An atomic.
    pub atomic: bool,
    /// Retired by `vmcnt` (a VMEM-to-LDS DMA) rather than `lgkmcnt`/`dscnt`.
    pub vm: bool,
}

/// One wave's LDS ops in flight.
#[derive(Clone, Debug, Default)]
pub struct WaveOps {
    /// Uncovered reads, in issue order.
    pub reads: Vec<LdsOp>,
    /// Uncovered writes and atomics.
    pub writes: Vec<LdsOp>,
}

/// A workgroup's LDS.
#[derive(Clone, Debug, Default)]
pub struct Lds {
    /// The bytes.
    pub data: Vec<u8>,
    /// One tag per byte.
    pub tags: Vec<Tag>,
    /// The last writer wave, or −1.
    pub writer: Vec<i8>,
    /// The last write's epoch.
    pub epoch: Vec<i64>,
    /// 2·seq (+1 for an atomic) of an uncovered ds/vm write, or −1.
    pub wq: Vec<i64>,
    /// Whether the uncovered write in `wq` counts on `vm` (DMA) rather than `ds`.
    pub wvm: Vec<bool>,
    /// The waves with an uncovered read.
    pub rmask: Vec<u32>,
    /// The waves whose reads were covered in epoch `rcov_ep`.
    pub rcov: Vec<u32>,
    /// That epoch.
    pub rcov_ep: Vec<i64>,
    /// The last covered read's instruction, or −1.
    pub rcpc: Vec<i32>,
    /// Its wave.
    pub rcw: Vec<i8>,
    /// It was issued in an earlier epoch than its cover.
    pub rcx: Vec<bool>,
    /// The last write was in flight at a barrier and covered after it: 1
    /// ds write, 2 atomic.
    pub wcx: Vec<u8>,
    /// The last write's instruction.
    pub wcpc: Vec<i32>,
    /// A racy pair of writes: the earlier write's instruction, or −1.
    pub wrace: Vec<i32>,
    /// Its wave.
    pub wrwho: Vec<i8>,
    /// Per wave, its ops in flight.
    pub waves: Vec<WaveOps>,
    /// Bytes read that the workgroup had not written, per instruction.
    pub unwritten: Vec<(usize, u64)>,
}

fn fault(message: String) -> EmuError {
    EmuError {
        cause: Some(Cause::Fault),
        message,
    }
}

/// The instruction texts of the program, for messages.
pub type Texts<'a> = &'a [String];

fn text(texts: Texts<'_>, pc: i32) -> &str {
    usize::try_from(pc)
        .ok()
        .and_then(|p| texts.get(p))
        .map_or("ds_read", String::as_str)
}

impl Lds {
    /// An LDS of `size` bytes, all zero and untagged, for `waves` waves.
    pub fn new(size: usize, waves: usize) -> Lds {
        Lds {
            data: vec![0; size],
            tags: vec![0; size],
            writer: vec![-1; size],
            epoch: vec![0; size],
            wq: vec![-1; size],
            wvm: vec![false; size],
            rmask: vec![0; size],
            rcov: vec![0; size],
            rcov_ep: vec![-1; size],
            rcpc: vec![-1; size],
            rcw: vec![-1; size],
            rcx: vec![false; size],
            wcx: vec![0; size],
            wcpc: vec![-1; size],
            wrace: vec![-1; size],
            wrwho: vec![-1; size],
            waves: vec![WaveOps::default(); waves],
            unwritten: Vec::new(),
        }
    }

    /// Its size.
    pub fn size(&self) -> usize {
        self.data.len()
    }

    /// A read of `bytes` (lane-major) by wave `w` in epoch `e`: the
    /// track synchronization and return racing writers for the checker.
    pub fn check_read(
        &mut self,
        w: usize,
        e: i64,
        pc: usize,
        bytes: &[u32],
        texts: Texts<'_>,
        wtext: &str,
    ) -> Result<Vec<usize>, EmuError> {
        for &j in bytes {
            let j = j as usize;
            if self.wrace[j] >= 0 {
                return Err(fault(format!(
                    "wave {w} {wtext}: reads LDS byte {j}, which wave {}'s `{}` and wave {}'s later write both wrote, the first in flight at a barrier between them (a barrier waits for no LDS op; waited for only after it, or not yet): the two writes may land in either order",
                    self.wrwho[j],
                    text(texts, self.wrace[j]),
                    self.writer[j]
                )));
            }
        }
        let fresh = bytes
            .iter()
            .filter(|j| self.writer[**j as usize] < 0)
            .count();
        if fresh > 0 {
            self.unwritten.push((pc, fresh as u64));
        }
        for &j in bytes {
            let j = j as usize;
            let a = self.writer[j];
            if self.wq[j] >= 0 && a >= 0 && a as usize != w && self.epoch[j] != e {
                let wait_kind = if self.wvm[j] { "vm wait" } else { "ds wait" };
                return Err(fault(format!(
                    "wave {w} {wtext}: reads LDS byte {j} of wave {a}'s `{}` still in flight: it did not wait for it ({wait_kind}) before the barrier - a barrier waits for no LDS op",
                    text(texts, self.wcpc[j])
                )));
            }
        }
        let mut racers: Vec<usize> = Vec::new();
        for &j in bytes {
            let j = j as usize;
            let a = self.writer[j];
            if self.epoch[j] == e && a >= 0 && a as usize != w && !racers.contains(&(a as usize)) {
                racers.push(a as usize);
            }
        }
        racers.sort_unstable();
        Ok(racers)
    }

    /// Track the in-flight read until its wait retires it.
    pub fn record_read(&mut self, w: usize, e: i64, pc: usize, seq: u64, bytes: Vec<u32>) {
        for &j in &bytes {
            self.rmask[j as usize] |= 1 << w;
        }
        self.waves[w].reads.push(LdsOp {
            seq,
            bytes,
            pc,
            epoch: e,
            atomic: false,
            vm: false,
        });
    }

    /// The write rules of for wave `w` in epoch `e`; marks late and
    /// pending bytes.
    #[allow(clippy::too_many_arguments)]
    pub fn check_write(
        &mut self,
        w: usize,
        e: i64,
        pc: usize,
        bytes: &[u32],
        atomic: bool,
        texts: Texts<'_>,
        wtext: &str,
    ) -> Result<(), EmuError> {
        let me = 1u32 << w;
        for &j in bytes {
            let j = j as usize;
            let others = self.rmask[j] & !me;
            let covered = if self.rcov_ep[j] == e {
                self.rcov[j] & !me
            } else {
                0
            };
            if others != 0 {
                let a = others.trailing_zeros() as usize;
                let op = self.waves[a]
                    .reads
                    .iter()
                    .find(|r| r.bytes.contains(&(j as u32)))
                    .map_or("ds_read", |r| {
                        texts.get(r.pc).map_or("ds_read", String::as_str)
                    });
                return Err(fault(format!(
                    "wave {w} {wtext}: writes LDS byte {j} that wave {a}'s `{op}` reads, still in flight (not covered by its ds wait; a barrier waits for no LDS op): the write may land before the read (a WAR race)"
                )));
            }
            if covered != 0 {
                let rcw = self.rcw[j];
                let (a, op) = if rcw >= 0 && covered >> rcw & 1 != 0 {
                    (rcw as usize, text(texts, self.rcpc[j]).to_owned())
                } else {
                    (covered.trailing_zeros() as usize, "ds_read".to_owned())
                };
                let when = if self.rcx[j] {
                    "in flight at a barrier since, waited for only after it"
                } else {
                    "since the last barrier (none between the read and the write)"
                };
                return Err(fault(format!(
                    "wave {w} {wtext}: writes LDS byte {j} that wave {a}'s `{op}` read, {when}: the write may land before the read (a WAR race)"
                )));
            }
            let a = self.writer[j];
            if self.wq[j] >= 0 && a >= 0 && a as usize != w && self.epoch[j] != e {
                let theirs_atomic = self.wq[j] & 1 == 1;
                if !(atomic && theirs_atomic) {
                    let wait_kind = if self.wvm[j] { "vm wait" } else { "ds wait" };
                    return Err(fault(format!(
                        "wave {w} {wtext}: writes LDS byte {j} that wave {a}'s `{}` writes, still in flight at a barrier since (not covered by its {wait_kind}): the two writes may land in either order",
                        text(texts, self.wcpc[j])
                    )));
                }
            }
        }
        for &j in bytes {
            let j = j as usize;
            let a = self.writer[j];
            let other = a >= 0 && a as usize != w;
            let late =
                self.wcx[j] > 0 && other && self.epoch[j] == e && !(atomic && self.wcx[j] == 2);
            if late {
                self.wrace[j] = self.wcpc[j];
                self.wrwho[j] = a;
            } else {
                self.wrace[j] = -1;
                self.wrwho[j] = -1;
            }
            self.writer[j] = w as i8;
            self.wcx[j] = 0;
            self.wcpc[j] = pc as i32;
        }
        Ok(())
    }

    /// A ds write's state after its data and tags are written.
    pub fn record_write(
        &mut self,
        w: usize,
        e: i64,
        pc: usize,
        seq: u64,
        bytes: Vec<u32>,
        atomic: bool,
    ) {
        self.record_write_on(w, e, pc, seq, bytes, atomic, false);
    }

    /// A VMEM-to-LDS DMA write's state after its data and tags are written.
    pub fn record_vm_write(&mut self, w: usize, e: i64, pc: usize, seq: u64, bytes: Vec<u32>) {
        self.record_write_on(w, e, pc, seq, bytes, false, true);
    }

    #[allow(clippy::too_many_arguments)]
    fn record_write_on(
        &mut self,
        w: usize,
        e: i64,
        pc: usize,
        seq: u64,
        bytes: Vec<u32>,
        atomic: bool,
        vm: bool,
    ) {
        let q = 2 * seq as i64 + i64::from(atomic);
        for &j in &bytes {
            let j = j as usize;
            self.epoch[j] = e;
            self.wq[j] = q;
            self.wvm[j] = vm;
        }
        self.waves[w].writes.push(LdsOp {
            seq,
            bytes,
            pc,
            epoch: e,
            atomic,
            vm,
        });
    }

    /// The ds cover of wave `w` in epoch `e` of its ops below `done`.
    pub fn ds_cover(&mut self, w: usize, e: i64, done: u64) {
        let me = 1u32 << w;
        let reads = std::mem::take(&mut self.waves[w].reads);
        let (cov, keep): (Vec<LdsOp>, Vec<LdsOp>) = reads.into_iter().partition(|r| r.seq < done);
        for r in &cov {
            for &j in &r.bytes {
                let j = j as usize;
                self.rmask[j] &= !me;
                if self.rcov_ep[j] != e {
                    self.rcov[j] = 0;
                }
                self.rcov_ep[j] = e;
                self.rcov[j] |= me;
                self.rcpc[j] = r.pc as i32;
                self.rcw[j] = w as i8;
                self.rcx[j] = r.epoch != e;
            }
        }
        for r in &keep {
            for &j in &r.bytes {
                self.rmask[j as usize] |= me;
            }
        }
        self.waves[w].reads = keep;
        self.cover_writes(w, e, done, false);
    }

    /// The vm cover of wave `w` in epoch `e` of its VMEM-to-LDS DMAs below `done`.
    pub fn vm_cover(&mut self, w: usize, e: i64, done: u64) {
        self.cover_writes(w, e, done, true);
    }

    fn cover_writes(&mut self, w: usize, e: i64, done: u64, vm: bool) {
        let writes = std::mem::take(&mut self.waves[w].writes);
        let (cov, keep): (Vec<LdsOp>, Vec<LdsOp>) =
            writes.into_iter().partition(|x| x.vm == vm && x.seq < done);
        for x in &cov {
            let q = 2 * x.seq as i64 + i64::from(x.atomic);
            for &j in &x.bytes {
                let j = j as usize;
                if self.wq[j] == q && self.wvm[j] == vm && self.writer[j] == w as i8 {
                    self.epoch[j] = e;
                    self.wcx[j] = if x.epoch != e {
                        1 + u8::from(x.atomic)
                    } else {
                        0
                    };
                    self.wcpc[j] = x.pc as i32;
                    self.wq[j] = -1;
                    self.wvm[j] = false;
                }
            }
        }
        self.waves[w].writes = keep;
    }

    /// A wave's end: its uncovered reads are dropped, every write covered.
    pub fn wave_end(&mut self, w: usize, e: i64) {
        let me = 1u32 << w;
        for r in std::mem::take(&mut self.waves[w].reads) {
            for j in r.bytes {
                self.rmask[j as usize] &= !me;
            }
        }
        self.ds_cover(w, e, u64::MAX);
        self.vm_cover(w, e, u64::MAX);
    }

    /// The LDS fence at a barrier release into epoch `e2`.
    pub fn fence(
        &mut self,
        e2: i64,
        waiting: &[(usize, bool)],
        texts: Texts<'_>,
    ) -> Result<(), EmuError> {
        if waiting.iter().any(|(_, f)| !f) {
            return Err(fault(
                "waves released from an LDS fence and a plain barrier together".to_owned(),
            ));
        }
        for (w, _) in waiting {
            let ops = &self.waves[*w];
            let first = ops.reads.first().or(ops.writes.first());
            if let Some(op) = first {
                let kind_text = if op.vm { "an LDS DMA" } else { "an LDS op" };
                return Err(fault(format!(
                    "wave {w}: `{}` ({kind_text}) in flight at the LDS fence",
                    texts.get(op.pc).map_or("", String::as_str)
                )));
            }
        }
        for (k, (b, t)) in self.data.iter_mut().zip(self.tags.iter_mut()).enumerate() {
            let v = (k as u32)
                .wrapping_mul(0x9E37_79B1)
                .wrapping_add((e2 as u32).wrapping_mul(0x85EB_CA77));
            *b = (v >> 24) as u8;
            *t = 0;
        }
        Ok(())
    }
}
