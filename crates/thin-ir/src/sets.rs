//! The set types the passes use: native only.

use std::cmp::Ordering;

/// A dense set of small integers: member k is bit k mod 64 of word k / 64.
/// Storage grows on demand; trailing zero words do not matter to equality.
#[derive(Clone, Debug, Default)]
pub struct Bits {
    words: Vec<u64>,
}

impl PartialEq for Bits {
    fn eq(&self, other: &Bits) -> bool {
        let n = self.words.len().max(other.words.len());
        (0..n).all(|w| self.word(w) == other.word(w))
    }
}

impl Eq for Bits {}

impl Bits {
    /// The empty set.
    pub fn new() -> Bits {
        Bits::default()
    }

    /// The set whose member k is bit k mod 64 of `words[k / 64]`.
    pub fn from_words(words: Vec<u64>) -> Bits {
        Bits { words }
    }

    /// Word w, 0 past the storage.
    fn word(&self, w: usize) -> u64 {
        self.words.get(w).copied().unwrap_or(0)
    }

    /// Grows the storage to hold word w.
    fn grow(&mut self, w: usize) {
        if self.words.len() <= w {
            self.words.resize(w + 1, 0);
        }
    }

    /// Adds k.
    pub fn set(&mut self, k: usize) {
        self.grow(k / 64);
        self.words[k / 64] |= 1 << (k % 64);
    }

    /// Removes k; a no-op past the storage.
    pub fn unset(&mut self, k: usize) {
        if let Some(w) = self.words.get_mut(k / 64) {
            *w &= !(1 << (k % 64));
        }
    }

    /// Whether k is a member.
    pub fn has(&self, k: usize) -> bool {
        self.word(k / 64) >> (k % 64) & 1 == 1
    }

    /// Whether it has no member.
    pub fn is_empty(&self) -> bool {
        self.words.iter().all(|&w| w == 0)
    }

    /// Adds o's members.
    pub fn or(&mut self, o: &Bits) {
        self.grow(o.words.len().saturating_sub(1));
        self.words
            .iter_mut()
            .zip(&o.words)
            .for_each(|(a, b)| *a |= b);
    }

    /// Removes o's members.
    pub fn and_not(&mut self, o: &Bits) {
        self.words
            .iter_mut()
            .zip(&o.words)
            .for_each(|(a, b)| *a &= !b);
    }

    /// The members of both, as a new set.
    #[must_use]
    pub fn and(&self, o: &Bits) -> Bits {
        Bits {
            words: self
                .words
                .iter()
                .zip(&o.words)
                .map(|(a, b)| a & b)
                .collect(),
        }
    }

    /// Whether it shares a member with o.
    pub fn intersects(&self, o: &Bits) -> bool {
        self.words.iter().zip(&o.words).any(|(a, b)| a & b != 0)
    }

    /// The members, ascending.
    pub fn ones(&self) -> Vec<usize> {
        let mut out = Vec::new();
        for (w, &word) in self.words.iter().enumerate() {
            let mut m = word;
            while m != 0 {
                out.push(w * 64 + m.trailing_zeros() as usize);
                m &= m - 1;
            }
        }
        out
    }

    /// The mask of bits `lo.. hi` (0 ≤ lo ≤ hi ≤ 64) of a word.
    fn span(lo: usize, hi: usize) -> u64 {
        let upto = |n: usize| if n >= 64 { u64::MAX } else { (1 << n) - 1 };
        upto(hi) & !upto(lo)
    }

    /// Applies `f(word index, mask)` to each word `a.. b` meets.
    fn each_word(a: usize, b: usize, mut f: impl FnMut(usize, u64)) {
        if a >= b {
            return;
        }
        for w in a / 64..=(b - 1) / 64 {
            let lo = a.max(w * 64) - w * 64;
            let hi = b.min(w * 64 + 64) - w * 64;
            f(w, Bits::span(lo, hi));
        }
    }

    /// Adds `a.. b`; a no-op if a ≥ b.
    pub fn set_range(&mut self, a: usize, b: usize) {
        if a < b {
            self.grow((b - 1) / 64);
        }
        Bits::each_word(a, b, |w, m| self.words[w] |= m);
    }

    /// Removes `a.. b`, b clamped to the storage.
    pub fn clear_range(&mut self, a: usize, b: usize) {
        let b = b.min(self.words.len() * 64);
        Bits::each_word(a, b, |w, m| self.words[w] &= !m);
    }

    /// The members in `a.. b`, as a new set.
    #[must_use]
    pub fn and_range(&self, a: usize, b: usize) -> Bits {
        let mut out = Bits::new();
        let b = b.min(self.words.len() * 64);
        if a < b {
            out.grow((b - 1) / 64);
        }
        Bits::each_word(a, b, |w, m| out.words[w] = self.words[w] & m);
        out
    }

    /// Adds each key.
    pub fn add(&mut self, keys: impl IntoIterator<Item = usize>) {
        keys.into_iter().for_each(|k| self.set(k));
    }

    /// Removes each key.
    pub fn remove(&mut self, keys: impl IntoIterator<Item = usize>) {
        keys.into_iter().for_each(|k| self.unset(k));
    }

    /// Ascending, each k in exactly one of the two sets with whether o has
    /// it.
    pub fn changes(&self, o: &Bits) -> Vec<(usize, bool)> {
        let mut diff = Bits::new();
        let n = self.words.len().max(o.words.len());
        diff.words = (0..n).map(|w| self.word(w) ^ o.word(w)).collect();
        diff.ones().into_iter().map(|k| (k, o.has(k))).collect()
    }

    /// Empties it, keeping its storage.
    pub fn clear(&mut self) {
        self.words.fill(0);
    }

    /// Adds the members of a that b lacks.
    pub fn or_minus(&mut self, a: &Bits, b: &Bits) {
        self.grow(a.words.len().saturating_sub(1));
        for (w, &x) in a.words.iter().enumerate() {
            self.words[w] |= x & !b.word(w);
        }
    }

    /// Becomes the members of both a and b.
    pub fn assign_and(&mut self, a: &Bits, b: &Bits) {
        self.words.clear();
        self.words
            .extend(a.words.iter().zip(&b.words).map(|(x, y)| x & y));
    }

    /// The same set, sparse.
    pub fn sparse(&self) -> Sparse {
        Sparse {
            words: (self.words.iter().enumerate())
                .filter(|(_, &w)| w != 0)
                .map(|(i, &w)| (i as u32, w))
                .collect(),
        }
    }

    /// Keeps only the members a sparse set has.
    pub fn and_sparse(&mut self, sp: &Sparse) {
        let mut at = sp.words.iter().peekable();
        for (w, word) in self.words.iter_mut().enumerate() {
            while at.next_if(|(i, _)| (*i as usize) < w).is_some() {}
            *word &= at
                .peek()
                .filter(|(i, _)| *i as usize == w)
                .map_or(0, |(_, m)| *m);
        }
    }

    /// Removes a sparse set's members; costs only the sparse set's words.
    pub fn and_not_sparse(&mut self, sp: &Sparse) {
        for &(i, m) in &sp.words {
            if let Some(w) = self.words.get_mut(i as usize) {
                *w &= !m;
            }
        }
    }

    /// Its members that a sparse set has, ascending; costs only the sparse
    /// set's words.
    pub fn ones_in(&self, sp: &Sparse) -> Vec<usize> {
        let mut out = Vec::new();
        for &(i, m) in &sp.words {
            let mut w = self.word(i as usize) & m;
            while w != 0 {
                out.push(i as usize * 64 + w.trailing_zeros() as usize);
                w &= w - 1;
            }
        }
        out
    }

    /// Whether it shares a member with a sparse set; costs only the sparse
    /// set's words.
    pub fn meets(&self, sp: &Sparse) -> bool {
        sp.words
            .iter()
            .any(|&(i, m)| self.word(i as usize) & m != 0)
    }
}

/// A set stored as its nonzero 64-bit words with their indices, ascending.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Sparse {
    words: Vec<(u32, u64)>,
}

impl Sparse {
    /// The members of `a.. b` that are not in `c.. d`.
    pub fn range_but(a: usize, b: usize, (c, d): (usize, usize)) -> Sparse {
        let mut bits = Bits::new();
        bits.set_range(a, b);
        bits.clear_range(c, d);
        bits.sparse()
    }

    /// The members, ascending.
    pub fn ones(&self) -> Vec<usize> {
        let bits = Bits {
            words: self.words.last().map_or(Vec::new(), |&(last, _)| {
                let mut v = vec![0; last as usize + 1];
                self.words.iter().for_each(|&(i, m)| v[i as usize] = m);
                v
            }),
        };
        bits.ones()
    }
}

/// A set of rows as sorted, disjoint, half-open ranges, no two adjacent.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Runs {
    runs: Vec<(u32, u32)>,
}

impl Runs {
    /// The empty set.
    pub fn new() -> Runs {
        Runs::default()
    }

    /// The runs, ascending.
    pub fn runs(&self) -> &[(u32, u32)] {
        &self.runs
    }

    /// Adds `a.. b` (a ≤ b) past every row in the set: extending the last
    /// run when it ends at a; an empty run is ignored.
    ///
    /// # Panics
    /// If the last run ends after a.
    pub fn push_run(&mut self, a: u32, b: u32) {
        match self.runs.last_mut() {
            Some(last) if last.1 == a => last.1 = b,
            Some(last) if last.1 > a => {
                panic!("push_run({a}, {b}) after a run ending at {}", last.1)
            }
            _ if a < b => self.runs.push((a, b)),
            _ => {}
        }
    }

    /// Whether row r is in the set.
    pub fn has(&self, r: u32) -> bool {
        self.find(r).is_ok()
    }

    /// The run holding r, or where a run holding it would go.
    fn find(&self, r: u32) -> Result<usize, usize> {
        self.runs.binary_search_by(|&(a, b)| match () {
            _ if b <= r => Ordering::Less,
            _ if a > r => Ordering::Greater,
            _ => Ordering::Equal,
        })
    }

    /// The first row.
    pub fn first(&self) -> Option<u32> {
        self.runs.first().map(|r| r.0)
    }

    /// The last row (an inclusive bound).
    pub fn last(&self) -> Option<u32> {
        self.runs.last().map(|r| r.1 - 1)
    }

    /// The smallest row in both sets.
    pub fn first_common(&self, o: &Runs) -> Option<u32> {
        let (mut i, mut j) = (0, 0);
        while let (Some(&(a, b)), Some(&(c, d))) = (self.runs.get(i), o.runs.get(j)) {
            if a.max(c) < b.min(d) {
                return Some(a.max(c));
            }
            if b <= d {
                i += 1;
            } else {
                j += 1;
            }
        }
        None
    }

    /// Whether the sets share a row; looks up each of o's runs, so pass the
    /// smaller set as o.
    pub fn meets(&self, o: &Runs) -> bool {
        o.runs.iter().any(|&(c, d)| {
            let at = self.find(c).unwrap_or_else(|i| i);
            self.runs.get(at).is_some_and(|&(a, b)| a < d && c < b)
        })
    }

    /// Adds o's rows; overlapping and touching runs coalesce, and an empty
    /// run (rows a..a) adds nothing, so the result stays canonical.
    pub fn add(&mut self, o: &Runs) {
        let mut all: Vec<(u32, u32)> = (self.runs.iter().chain(&o.runs))
            .filter(|(a, b)| a < b)
            .copied()
            .collect();
        all.sort_unstable();
        let mut out: Vec<(u32, u32)> = Vec::with_capacity(all.len());
        for (a, b) in all {
            match out.last_mut() {
                Some(last) if a <= last.1 => last.1 = last.1.max(b),
                _ => out.push((a, b)),
            }
        }
        self.runs = out;
    }
}
