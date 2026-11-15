//! Writes in flight and stale reads: the flight model plays
//! an adversary that may perform a flag first and the data late.

use std::collections::{BTreeMap, HashMap, HashSet};

use crate::tags::Tag;

/// A workgroup's observation of a producer workgroup: the newest record
/// seen, and per wave the frontier it carried.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Obs {
    /// The newest record id seen.
    pub newest: i64,
    /// Per wave of the producer, the newest performed record known.
    pub frontier: Vec<i64>,
}

impl Obs {
    fn join(&mut self, newest: i64, frontier: &[i64]) {
        self.newest = self.newest.max(newest);
        if self.frontier.len() < frontier.len() {
            self.frontier.resize(frontier.len(), -1);
        }
        for (a, b) in self.frontier.iter_mut().zip(frontier) {
            *a = (*a).max(*b);
        }
    }
}

/// Who wrote: a workgroup and its wave, for messages and frontiers.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Writer {
    /// The workgroup's serial in the memory (unique over ranks and launches).
    pub wg: usize,
    /// The wave within it.
    pub wave: usize,
    /// The rank.
    pub rank: usize,
    /// The workgroup's x id.
    pub wgx: u32,
}

/// One write record.
#[derive(Clone, Debug)]
pub struct Record {
    /// Its id.
    pub id: usize,
    /// The writer.
    pub who: Writer,
    /// Its frontier: per wave of the writer's workgroup.
    pub frontier: Vec<i64>,
    /// Released by a publisher of its workgroup.
    pub exposed: bool,
    /// Not yet performed (no wait covered it).
    pub pending: bool,
    /// The buffer.
    pub buf: usize,
    /// The bytes written, sorted and unique.
    pub bytes: Vec<u64>,
    /// The previous bytes and tags, while kept.
    pub prev: Option<(Vec<u8>, Vec<Tag>)>,
    /// Per byte, the record that last wrote it before this one.
    pub before: Vec<Option<usize>>,
    /// The instruction text.
    pub text: String,
    /// An atomic.
    pub atomic: bool,
    /// For an atomic: the atomic records that last wrote its bytes.
    pub release: Vec<usize>,
    /// A publisher's snapshot of its workgroup's observations.
    pub obs: Option<HashMap<usize, Obs>>,
    /// The instruction's serial, so its chunks merge.
    pub serial: u64,
}

/// The flight model of one memory.
#[derive(Clone, Debug, Default)]
pub struct Flight {
    /// The live records by id.
    pub records: BTreeMap<usize, Record>,
    next: usize,
    /// The last writer record of each byte, per buffer (`NONE`: none).
    last: Vec<Vec<u32>>,
    /// Per reader workgroup, its observation of each producer workgroup.
    pub obs: HashMap<usize, HashMap<usize, Obs>>,
    folded: HashSet<(usize, usize)>,
    served: HashSet<(usize, usize)>,
    /// Per workgroup, its pending records.
    pending: HashMap<usize, Vec<usize>>,
    /// The stale reads served.
    pub events: Vec<String>,
    /// Exposed records still keeping their previous bytes: none means
    /// nothing can be served.
    exposed_kept: usize,
}

/// No record.
const NONE: u32 = u32::MAX;

/// A write about to happen (made before the bytes change).
pub struct Write<'a> {
    /// The writer.
    pub who: Writer,
    /// Its frontier.
    pub frontier: Vec<i64>,
    /// The buffer and its name.
    pub buf: usize,
    /// The byte offsets.
    pub bytes: Vec<u64>,
    /// The buffer's bytes and tags now (for the previous ones).
    pub data: &'a [u8],
    /// Its tags now.
    pub tags: &'a [Tag],
    /// The text.
    pub text: &'a str,
    /// An atomic.
    pub atomic: bool,
    /// A publisher (an atomic, or a store at agent scope or above).
    pub publisher: bool,
    /// Performed at issue (a scalar store or atomic).
    pub performed: bool,
    /// The instruction's serial.
    pub serial: u64,
}

/// A stale read served: which bytes read which previous values.
pub struct Served {
    /// (byte offset, previous byte, previous tag).
    pub bytes: Vec<(u64, u8, Tag)>,
}

impl Flight {
    /// Whether any record is live.
    pub fn active(&self) -> bool {
        !self.records.is_empty()
    }

    fn last_of(&self, buf: usize, b: u64) -> Option<usize> {
        let id = *self.last.get(buf)?.get(b as usize)?;
        (id != NONE).then_some(id as usize)
    }

    fn set_last(&mut self, buf: usize, b: u64, id: usize, size: usize) {
        if self.last.len() <= buf {
            self.last.resize_with(buf + 1, Vec::new);
        }
        let per = &mut self.last[buf];
        if per.len() < size {
            per.resize(size, NONE);
        }
        per[b as usize] = id as u32;
    }

    /// Records a write and returns its id.
    pub fn write(&mut self, w: Write<'_>) -> usize {
        let mut bytes = w.bytes;
        bytes.sort_unstable();
        bytes.dedup();
        // Merge into the same instruction's record on this buffer.
        let merge = self
            .records
            .values()
            .next_back()
            .filter(|r| r.serial == w.serial && r.buf == w.buf && r.who == w.who)
            .map(|r| r.id);
        let id = match merge {
            Some(id) => id,
            None => {
                let id = self.next;
                self.next += 1;
                id
            }
        };
        let mut prev_bytes = Vec::with_capacity(bytes.len());
        let mut prev_tags = Vec::with_capacity(bytes.len());
        let mut before = Vec::with_capacity(bytes.len());
        let mut release = Vec::new();
        for &b in &bytes {
            prev_bytes.push(w.data.get(b as usize).copied().unwrap_or(0));
            prev_tags.push(w.tags.get(b as usize).copied().unwrap_or(0));
            let last = self.last_of(w.buf, b).filter(|l| *l != id);
            before.push(last);
            if let Some(l) = last {
                if w.atomic
                    && self.records.get(&l).is_some_and(|r| r.atomic)
                    && !release.contains(&l)
                {
                    release.push(l);
                }
            }
            self.set_last(w.buf, b, id, w.data.len());
        }
        let obs = w
            .publisher
            .then(|| self.obs.get(&w.who.wg).cloned().unwrap_or_default());
        if w.publisher {
            if let Some(p) = self.pending.get(&w.who.wg) {
                for r in p {
                    if let Some(r) = self.records.get_mut(r) {
                        if !r.exposed && r.prev.is_some() {
                            self.exposed_kept += 1;
                        }
                        r.exposed = true;
                    }
                }
            }
        }
        let pending = !w.performed;
        if let Some(r) = self.records.get_mut(&id) {
            for (k, b) in bytes.iter().enumerate() {
                if !r.bytes.contains(b) {
                    r.bytes.push(*b);
                    if let Some((pb, pt)) = &mut r.prev {
                        pb.push(prev_bytes[k]);
                        pt.push(prev_tags[k]);
                    }
                    r.before.push(before[k]);
                }
            }
            return id;
        }
        let prev = (!w.performed).then_some((prev_bytes, prev_tags));
        self.records.insert(
            id,
            Record {
                id,
                who: w.who.clone(),
                frontier: w.frontier,
                exposed: false,
                pending,
                buf: w.buf,
                bytes,
                prev,
                before,
                text: w.text.to_owned(),
                atomic: w.atomic,
                release,
                obs,
                serial: w.serial,
            },
        );
        if pending {
            self.pending.entry(w.who.wg).or_default().push(id);
        }
        id
    }

    /// A wait covered these records: they are performed. Returns the
    /// newest id among them.
    pub fn perform(&mut self, ids: &[usize]) -> i64 {
        let mut newest = -1;
        for id in ids {
            newest = newest.max(*id as i64);
            if let Some(r) = self.records.get_mut(id) {
                r.pending = false;
                if !r.exposed {
                    r.prev = None;
                }
                if let Some(p) = self.pending.get_mut(&r.who.wg) {
                    p.retain(|x| x != id);
                }
            }
        }
        newest
    }

    /// A wave ended: its unexposed pending records drop their previous
    /// bytes (they stay pending).
    pub fn wave_end(&mut self, ids: &[usize]) {
        for id in ids {
            if let Some(r) = self.records.get_mut(id) {
                if !r.exposed {
                    r.prev = None;
                }
            }
        }
    }

    /// The end of a rank's launch: its records are dropped.
    pub fn end_launch(&mut self, rank: usize) {
        self.records.retain(|_, r| r.who.rank != rank);
        self.exposed_kept = self
            .records
            .values()
            .filter(|r| r.exposed && r.prev.is_some())
            .count();
        for p in self.pending.values_mut() {
            p.clear();
        }
        if self.records.is_empty() {
            self.last.clear();
        } else {
            let records = &self.records;
            for per in &mut self.last {
                for id in per.iter_mut() {
                    if *id != NONE && !records.contains_key(&(*id as usize)) {
                        *id = NONE;
                    }
                }
            }
        }
    }

    fn fold(&mut self, reader: usize, id: usize) {
        if !self.folded.insert((reader, id)) {
            return;
        }
        let Some(r) = self.records.get(&id) else {
            return;
        };
        let (wg, newest, frontier, snapshot, release) = (
            r.who.wg,
            r.id as i64,
            r.frontier.clone(),
            r.obs.clone(),
            r.release.clone(),
        );
        let mine = self.obs.entry(reader).or_default();
        if wg != reader {
            mine.entry(wg).or_default().join(newest, &frontier);
        }
        if let Some(s) = snapshot {
            for (p, o) in s {
                if p != reader {
                    mine.entry(p).or_default().join(o.newest, &o.frontier);
                }
            }
        }
        if r_atomic_chain(&release) {
            for rel in release {
                self.fold(reader, rel);
            }
        }
    }

    /// A workgroup reads bytes: it observes their last writers.
    pub fn observe(&mut self, reader: usize, buf: usize, bytes: &[u64]) {
        if self.records.is_empty() {
            return;
        }
        let mut seen: Vec<usize> = Vec::new();
        for b in bytes {
            if let Some(id) = self.last_of(buf, *b) {
                if !seen.contains(&id) {
                    seen.push(id);
                }
            }
        }
        for id in seen {
            self.fold(reader, id);
        }
    }

    fn servable(&self, reader: usize, id: usize) -> bool {
        let Some(r) = self.records.get(&id) else {
            return false;
        };
        if r.who.wg == reader
            || !r.exposed
            || r.prev.is_none()
            || self.served.contains(&(reader, id))
        {
            return false;
        }
        let Some(o) = self.obs.get(&reader).and_then(|m| m.get(&r.who.wg)) else {
            return false;
        };
        let f = o.frontier.get(r.who.wave).copied().unwrap_or(-1);
        o.newest > id as i64 && f < id as i64
    }

    /// A load by workgroup `reader` of bytes of a buffer whose current
    /// values are `data`: the bytes it reads stale, if any. The
    /// caller records the event text.
    pub fn serve(
        &mut self,
        reader: usize,
        buf: usize,
        bytes: &[u64],
        data: &[u8],
    ) -> Option<(Served, usize)> {
        if self.records.is_empty() {
            return None;
        }
        let mut out = Vec::new();
        let mut chosen: Vec<usize> = Vec::new();
        for &b in bytes {
            let Some(id) = self.last_of(buf, b) else {
                continue;
            };
            if !self.servable(reader, id) {
                continue;
            }
            let r = &self.records[&id];
            let Some(k) = r.bytes.iter().position(|x| *x == b) else {
                continue;
            };
            let Some((pb, pt)) = &r.prev else { continue };
            let (mut byte, mut tag, mut from) = (pb[k], pt[k], id);
            let cur = data.get(b as usize).copied().unwrap_or(0);
            if byte == cur {
                // An idempotent rewrite: walk back to the record before.
                if let Some(Some(back)) = r.before.get(k) {
                    if self.servable(reader, *back) {
                        let rb = &self.records[back];
                        if let (Some(kb), Some((bb, bt))) =
                            (rb.bytes.iter().position(|x| *x == b), &rb.prev)
                        {
                            if bb[kb] != cur {
                                byte = bb[kb];
                                tag = bt[kb];
                                from = *back;
                            }
                        }
                    }
                }
                if byte == cur {
                    continue;
                }
            }
            if !chosen.contains(&from) {
                chosen.push(from);
            }
            out.push((b, byte, tag));
        }
        if out.is_empty() {
            return None;
        }
        for id in &chosen {
            self.served.insert((reader, *id));
        }
        Some((Served { bytes: out }, chosen[0]))
    }

    /// The text of the newest record `reader` observed from `wg`.
    pub fn observed_text(&self, reader: usize, wg: usize) -> String {
        let newest = self
            .obs
            .get(&reader)
            .and_then(|m| m.get(&wg))
            .map_or(-1, |o| o.newest);
        usize::try_from(newest)
            .ok()
            .and_then(|n| self.records.get(&n))
            .map_or_else(String::new, |r| r.text.clone())
    }
}

fn r_atomic_chain(release: &[usize]) -> bool {
    !release.is_empty()
}
