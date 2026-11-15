//! Global memory: buffers straddling 4 GiB boundaries,
//! resolution, faults, versions and traffic. The flight model and
//! the scope model live beside it.

use super::flight::Flight;
use super::scope::Scope;
use crate::tags::{mem_tag, Tag};
use crate::{Cause, EmuError};

/// A buffer's metadata the core reads.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Meta {
    /// Shared with another rank.
    pub ipc: bool,
    /// An ipc buffer that is cached anyway.
    pub cached: bool,
    /// Bypasses the L2.
    pub uncached: bool,
}

/// One buffer.
#[derive(Clone, Debug)]
pub struct Buffer {
    /// Its name (not necessarily unique).
    pub name: String,
    /// Its base address.
    pub base: u64,
    /// Its bytes.
    pub data: Vec<u8>,
    /// One tag per byte (all 0 for an untagged buffer).
    pub tags: Vec<Tag>,
    /// Whether its bytes started as their own memory tags.
    pub tagged: bool,
    /// Its metadata.
    pub meta: Meta,
    /// Its store count.
    pub version: u64,
    /// Released: its range is unmapped.
    pub released: bool,
}

impl Buffer {
    /// Its size in bytes.
    pub fn size(&self) -> u64 {
        self.data.len() as u64
    }

    /// Whether the scope model treats it as uncached.
    pub fn uncached(&self) -> bool {
        self.meta.uncached || (self.meta.ipc && !self.meta.cached)
    }
}

/// Per-buffer traffic: per 64-byte segment, bit 0 loaded, bit 1 stored.
#[derive(Clone, Debug, Default)]
pub struct Traffic {
    /// Segment bits per buffer.
    pub segments: Vec<Vec<u8>>,
    /// Per rank (15 and above in row 15): bytes loaded, stored, moved by atomics.
    pub bytes: [[u64; 3]; 16],
}

/// The memory of a run, shared by every rank.
#[derive(Debug)]
pub struct Memory {
    /// The buffers in allocation (and address) order.
    pub bufs: Vec<Buffer>,
    /// The next 4 GiB boundary a buffer straddles.
    next: u64,
    /// Traffic, when tracked.
    pub traffic: Option<Traffic>,
    /// The flight model.
    pub flight: Flight,
    /// The scope model, when on (clusters).
    pub scope: Option<Scope>,
}

impl Default for Memory {
    fn default() -> Memory {
        Memory::new(1 << 32)
    }
}

fn fault(message: String) -> EmuError {
    EmuError {
        cause: Some(Cause::Fault),
        message,
    }
}

/// Where a buffer of `size` bytes goes below the boundary `next`:
/// its base and the next boundary.
pub fn placement(next: u64, size: u64) -> (u64, u64) {
    let size = size.max(16);
    let below = ((size / 2).div_ceil(256) * 256).clamp(256, 1 << 31);
    (next - below, next + ((size.div_ceil(1 << 32) + 1) << 32))
}

/// The first boundary of a memory with this straddle.
pub fn first_boundary(straddle: u64) -> u64 {
    (1 << 44) + (1 << 32) + straddle % (1 << 32)
}

impl Memory {
    /// Empty memory with the given straddle.
    pub fn new(straddle: u64) -> Memory {
        Memory {
            bufs: Vec::new(),
            next: first_boundary(straddle),
            traffic: None,
            flight: Flight::default(),
            scope: None,
        }
    }

    /// Allocates a buffer: size at least 16, straddling the next
    /// 4 GiB boundary. Returns its index.
    pub fn alloc(
        &mut self,
        name: &str,
        size: u64,
        tagged: bool,
        fill: u8,
        meta: Meta,
    ) -> Result<usize, EmuError> {
        let size = size.max(16);
        let (base, next) = placement(self.next, size);
        self.next = next;
        let index = self.bufs.len();
        let tags = if tagged {
            (0..size).map(|o| mem_tag(index, o)).collect()
        } else {
            vec![0; size as usize]
        };
        self.bufs.push(Buffer {
            name: name.to_owned(),
            base,
            data: vec![fill; size as usize],
            tags,
            tagged,
            meta,
            version: 0,
            released: false,
        });
        if let Some(t) = &mut self.traffic {
            t.segments.push(vec![0; size.div_ceil(64) as usize]);
        }
        Ok(index)
    }

    /// Frees a buffer's bytes and unmaps its range.
    pub fn release(&mut self, index: usize) {
        if let Some(b) = self.bufs.get_mut(index) {
            b.released = true;
            b.data = Vec::new();
            b.tags = Vec::new();
        }
    }

    /// The index of the newest buffer of a name (names need not be
    /// unique).
    pub fn by_name(&self, name: &str) -> Option<usize> {
        self.bufs.iter().rposition(|b| b.name == name)
    }

    /// The buffer an address belongs to and the offset in it.
    pub fn resolve(&self, addr: u64) -> Option<(usize, u64)> {
        let i = self.bufs.partition_point(|b| b.base <= addr);
        let i = i.checked_sub(1)?;
        let b = &self.bufs[i];
        (!b.released && addr < b.base + b.size()).then(|| (i, addr - b.base))
    }

    /// Resolve an address to its buffer and byte offset, or report an invalid access.
    pub fn locate(&self, addr: u64) -> Result<(usize, u64), EmuError> {
        self.resolve(addr)
            .ok_or_else(|| fault(format!("address {addr:#x} is in no buffer")))
    }

    /// (buffer, offset) of many addresses; buffer −1 for none.
    pub fn resolve_many(&self, addrs: &[u64]) -> Vec<(i64, u64)> {
        addrs
            .iter()
            .map(|a| self.resolve(*a).map_or((-1, 0), |(b, o)| (b as i64, o)))
            .collect()
    }

    /// Reads `n` bytes of a buffer at `off` (clipped to its end).
    pub fn read(&self, b: usize, off: usize, n: usize) -> Vec<u8> {
        let data = self.bufs.get(b).map_or(&[][..], |x| &x.data[..]);
        let start = off.min(data.len());
        data[start..(off + n).min(data.len())].to_vec()
    }

    /// Writes bytes into a buffer at `off` (clipped to its end); its
    /// version counts the write.
    pub fn write(&mut self, b: usize, off: usize, bytes: &[u8]) {
        if let Some(x) = self.bufs.get_mut(b) {
            let start = off.min(x.data.len());
            let end = (off + bytes.len()).min(x.data.len());
            x.data[start..end].copy_from_slice(&bytes[..end - start]);
            x.version += 1;
        }
    }

    /// Whether any buffer is shared with another rank.
    pub fn ipc(&self) -> bool {
        self.bufs.iter().any(|b| b.meta.ipc)
    }

    /// Resolves an access of `n` bytes for a load or a store.
    pub fn access(&self, addr: u64, n: u64, store: bool) -> Result<(usize, u64), EmuError> {
        let what = if store { "store" } else { "load" };
        let Some((i, off)) = self.resolve(addr) else {
            let dir = if store { "to" } else { "from" };
            return Err(fault(format!("{what} {dir} unmapped address {addr:#x}")));
        };
        if off + n > self.bufs[i].size() {
            return Err(fault(format!(
                "{what} past the end of {}",
                self.bufs[i].name
            )));
        }
        Ok((i, off))
    }

    /// Starts tracking traffic.
    pub fn track_traffic(&mut self) {
        let segments = self
            .bufs
            .iter()
            .map(|b| vec![0; b.size().div_ceil(64) as usize])
            .collect();
        self.traffic = Some(Traffic {
            segments,
            bytes: [[0; 3]; 16],
        });
    }

    /// Counts traffic of `n` bytes at `off` of buffer `b` (kind 0 load, 1
    /// store, 2 atomic) for a rank.
    pub fn count(&mut self, b: usize, off: u64, n: u64, kind: usize, rank: usize) {
        let Some(t) = &mut self.traffic else { return };
        if let Some(seg) = t.segments.get_mut(b) {
            for s in off / 64..(off + n).div_ceil(64) {
                if let Some(x) = seg.get_mut(s as usize) {
                    *x |= match kind {
                        0 => 1,
                        1 => 2,
                        _ => 3,
                    };
                }
            }
        }
        t.bytes[rank.min(15)][kind] += n;
    }
}
