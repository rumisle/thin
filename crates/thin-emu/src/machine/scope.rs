//! The scope model: stores below agent scope held in one L2,
//! kept lines that go stale, write-back and invalidate.

use std::collections::HashMap;

use crate::{Cause, EmuError};

/// An access's scope: the target crate's.
pub use thin_target::Scope as Level;

/// One cached buffer's state.
#[derive(Clone, Debug, Default)]
struct BufState {
    /// Per byte: the L2 holding a store below agent scope, or −1.
    held: Vec<i32>,
    /// Per byte: the memory's write count at its last write.
    clock: Vec<u64>,
    /// Per L2: per 128-byte line, the write count when it was read in, or
    /// −1 (not kept).
    kept: HashMap<usize, Vec<i64>>,
}

/// The scope model of one memory (on in a cluster).
#[derive(Clone, Debug, Default)]
pub struct Scope {
    bufs: HashMap<usize, BufState>,
    writes: u64,
    /// L2s per rank.
    pub per_rank: usize,
}

fn scope_error(message: String) -> EmuError {
    EmuError {
        cause: Some(Cause::Scope),
        message,
    }
}

/// Where an access comes from, for messages: (rank, L2 index on the rank).
#[derive(Clone, Copy, Debug)]
pub struct Site {
    /// The rank.
    pub rank: usize,
    /// The L2's index on its rank.
    pub l2: usize,
}

impl Scope {
    /// A scope model with `per_rank` L2s on each rank.
    pub fn new(per_rank: usize) -> Scope {
        Scope {
            per_rank,
            ..Scope::default()
        }
    }

    fn l2(&self, s: Site) -> usize {
        s.rank * self.per_rank.max(1) + s.l2
    }

    fn site_of(&self, id: usize) -> (usize, usize) {
        let l = self.per_rank.max(1);
        (id / l, id % l)
    }

    fn state(&mut self, buf: usize, size: usize) -> &mut BufState {
        let st = self.bufs.entry(buf).or_default();
        if st.held.len() < size {
            st.held.resize(size, -1);
            st.clock.resize(size, 0);
        }
        st
    }

    /// A store's rule.
    pub fn store(&mut self, buf: usize, size: usize, bytes: &[u64], level: Level, at: Site) {
        self.writes += 1;
        let now = self.writes;
        let me = self.l2(at);
        let st = self.state(buf, size);
        let below = level < Level::Agent;
        for &b in bytes {
            let b = b as usize;
            st.clock[b] = now;
            st.held[b] = if below { me as i32 } else { -1 };
        }
        if let Some(lines) = st.kept.get_mut(&me) {
            for &b in bytes {
                if let Some(x) = lines.get_mut(b as usize / 128) {
                    if *x >= 0 {
                        *x = now as i64;
                    }
                }
            }
        }
    }

    /// A load's rule: refusals, then the lines kept.
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        &mut self,
        buf: usize,
        size: usize,
        name: &str,
        bytes: &[u64],
        level: Level,
        at: Site,
        text: &str,
    ) -> Result<(), EmuError> {
        let me = self.l2(at);
        let now = self.writes;
        let per_rank = self.per_rank;
        let st = self.state(buf, size);
        for &b in bytes {
            let h = st.held[b as usize];
            if h >= 0 && h as usize != me {
                let l = per_rank.max(1);
                let (r2, x2) = (h as usize / l, h as usize % l);
                return Err(scope_error(format!(
                    "{text}: rank {} L2 {} reads bytes of {name} that rank {r2} L2 {x2} stored below agent scope in this launch: they are in its L2, not in memory (a peer sees them after an agent-scope store, a write-back or the launch's end)",
                    at.rank, at.l2
                )));
            }
        }
        let lines = st
            .kept
            .entry(me)
            .or_insert_with(|| vec![-1; size.div_ceil(128)]);
        if level < Level::Agent {
            for &b in bytes {
                let line = lines[b as usize / 128];
                if line >= 0 && st.clock[b as usize] as i64 > line {
                    return Err(scope_error(format!(
                        "{text}: rank {} L2 {} reads byte {b} of {name} below agent scope from the line its L2 kept since an earlier read in this launch - another L2 wrote it since: the read is stale (an acquire at agent scope: an agent-scope load, or an invalidate before it)",
                        at.rank, at.l2
                    )));
                }
            }
        }
        for &b in bytes {
            let x = &mut lines[b as usize / 128];
            if *x < 0 || level >= Level::Agent {
                *x = now as i64;
            }
        }
        Ok(())
    }

    /// An atomic's rule (performed at the L2 or beyond).
    pub fn atomic(
        &mut self,
        buf: usize,
        size: usize,
        name: &str,
        bytes: &[u64],
        at: Site,
        text: &str,
    ) -> Result<(), EmuError> {
        let me = self.l2(at);
        let per_rank = self.per_rank;
        self.writes += 1;
        let now = self.writes;
        let st = self.state(buf, size);
        for &b in bytes {
            let h = st.held[b as usize];
            if h >= 0 && h as usize != me {
                let l = per_rank.max(1);
                let (r2, x2) = (h as usize / l, h as usize % l);
                return Err(scope_error(format!(
                    "{text}: an atomic of rank {} L2 {} on bytes of {name} that rank {r2} L2 {x2} stored below agent scope in this launch: they are in its L2, not in memory, and the atomic misses them (store at agent scope, or write back after the store)",
                    at.rank, at.l2
                )));
            }
        }
        for &b in bytes {
            st.held[b as usize] = -1;
            st.clock[b as usize] = now;
        }
        Ok(())
    }

    /// A scalar store or atomic: held none, write clock recorded.
    pub fn scalar_write(&mut self, buf: usize, size: usize, bytes: &[u64]) {
        self.writes += 1;
        let now = self.writes;
        let st = self.state(buf, size);
        for &b in bytes {
            st.held[b as usize] = -1;
            st.clock[b as usize] = now;
        }
    }

    /// Writes back an L2: its held bytes go to memory.
    pub fn writeback(&mut self, at: Site) {
        let me = self.l2(at) as i32;
        for st in self.bufs.values_mut() {
            for h in &mut st.held {
                if *h == me {
                    *h = -1;
                }
            }
        }
    }

    /// Invalidates an L2: its kept lines are dropped.
    pub fn invalidate(&mut self, at: Site) {
        let me = self.l2(at);
        for st in self.bufs.values_mut() {
            st.kept.remove(&me);
        }
    }

    /// The end of a rank's launch: write back and invalidate its L2s.
    pub fn launch_ended(&mut self, rank: usize) {
        for l2 in 0..self.per_rank.max(1) {
            let at = Site { rank, l2 };
            self.writeback(at);
            self.invalidate(at);
        }
        let _ = self.site_of(0);
    }
}
