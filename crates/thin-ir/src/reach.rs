//! Reaching definitions of keyed sites, on the forward solver of
//!
//!
//! A site is one row's write of one key (a register dword, as the caller
//! numbers them). A site reaches a point when some path from it to the
//! point has no killing write of its key. A write that keeps part of the
//! old contents (a partial write) is a site that does not kill: the older
//! sites of its key reach past it too. The callers decide what is a key,
//! which rows write which keys and which writes kill; this module only
//! solves. The alias check and the lift's register
//! unassignment both use it.

use crate::cfg::Cfg;
use crate::sets::{Bits, Sparse};
use crate::solve::solve_forward_all;

/// One row's write of one key.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Def {
    /// The row writing it; none for a write at entry, which reaches the
    /// first row.
    pub row: Option<usize>,
    /// The key written.
    pub key: usize,
    /// Whether the write ends the older sites of its key.
    pub kills: bool,
}

/// The sites of a stream, indexed by row and by key. Site numbers are the
/// positions in the list given to [`Defs::new`].
#[derive(Clone, Debug, Default)]
pub struct Defs {
    entry: Bits,
    sites: Vec<Def>,
    by_key: Vec<Sparse>,
    gens: Vec<Vec<usize>>,
    kills: Vec<Vec<usize>>,
}

impl Defs {
    /// The sites `sites` over `rows` rows, those of no row at entry. Several sites of one row may
    /// write one key (a caller numbering per-name sites); each is a site.
    ///
    /// # Panics
    /// If a site's row is not below `rows`.
    pub fn new(rows: usize, sites: Vec<Def>) -> Defs {
        let nkeys = sites.iter().map(|d| d.key + 1).max().unwrap_or(0);
        let mut by_key = vec![Bits::new(); nkeys];
        let mut gens = vec![Vec::new(); rows];
        let mut kills: Vec<Vec<usize>> = vec![Vec::new(); rows];
        let mut entry = Bits::new();
        for (k, d) in sites.iter().enumerate() {
            by_key[d.key].set(k);
            let Some(row) = d.row else {
                entry.set(k);
                continue;
            };
            gens[row].push(k);
            if d.kills && !kills[row].contains(&d.key) {
                kills[row].push(d.key);
            }
        }
        Defs {
            entry,
            sites,
            by_key: by_key.iter().map(Bits::sparse).collect(),
            gens,
            kills,
        }
    }

    /// Writes site `site` at one more row too: one site standing for
    /// equivalent writes (a caller asking only whether any of them
    /// reaches), with the site's kill.
    pub fn also(&mut self, row: usize, site: usize) {
        let d = self.sites[site];
        self.gens[row].push(site);
        if d.kills && !self.kills[row].contains(&d.key) {
            self.kills[row].push(d.key);
        }
    }

    /// The sites, by number.
    pub fn sites(&self) -> &[Def] {
        &self.sites
    }

    /// The sites a row writes.
    pub fn at(&self, row: usize) -> &[usize] {
        &self.gens[row]
    }

    /// Whether a row's write ends a site.
    pub fn ends(&self, row: usize, site: usize) -> bool {
        self.kills[row].contains(&self.sites[site].key)
    }

    /// The state after a row from the state before it: its killing writes
    /// end their keys' sites, then all its sites reach.
    pub fn step(&self, state: &mut Bits, row: usize) {
        for &key in &self.kills[row] {
            state.and_not_sparse(&self.by_key[key]);
        }
        state.add(self.gens[row].iter().copied());
    }

    /// Per block of `g`, the sites reaching its entry, the entry sites
    /// reaching block 0 (an unreached block's state is empty).
    pub fn solve(&self, g: &Cfg) -> Vec<Bits> {
        if g.nblocks() == 0 {
            return Vec::new();
        }
        let through = |st: &Bits, j: usize| {
            let mut st = st.clone();
            g.block(j).for_each(|k| self.step(&mut st, k));
            st
        };
        let union = |a: &Bits, b: &Bits| {
            let mut m = a.clone();
            m.or(b);
            m
        };
        solve_forward_all(
            g,
            Bits::new(),
            vec![(0, self.entry.clone())],
            through,
            union,
        )
    }

    /// The sites of `key` in a state, ascending.
    pub fn of_key(&self, state: &Bits, key: usize) -> Vec<usize> {
        self.by_key
            .get(key)
            .map_or_else(Vec::new, |sp| state.ones_in(sp))
    }

    /// Per row, `visit(row, state before it)`, block by block from the
    /// solved block entries.
    pub fn walk(&self, g: &Cfg, entries: &[Bits], mut visit: impl FnMut(usize, &Bits)) {
        for (j, entry) in entries.iter().enumerate() {
            let mut st = entry.clone();
            for k in g.block(j) {
                visit(k, &st);
                self.step(&mut st, k);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 0: def k0; 1: branch to 3 or fall; 2: def k0; 3: partial def k0;
    /// 4: end.
    fn diamond() -> (Cfg, Defs) {
        let g = Cfg::from_succ_only(vec![vec![1], vec![3, 2], vec![3], vec![4], vec![]]);
        let d = |row, kills| Def {
            row: Some(row),
            key: 0,
            kills,
        };
        (g, Defs::new(5, vec![d(0, true), d(2, true), d(3, false)]))
    }

    #[test]
    fn a_join_reaches_both_arms_and_a_partial_write_kills_nothing() {
        let (g, defs) = diamond();
        let entries = defs.solve(&g);
        let mut before = vec![Vec::new(); 5];
        defs.walk(&g, &entries, |k, st| before[k] = defs.of_key(st, 0));
        assert_eq!(before[2], [0]);
        assert_eq!(before[3], [0, 1]);
        assert_eq!(before[4], [0, 1, 2]);
    }

    #[test]
    fn an_entry_state_reaches_until_a_kill() {
        let g = Cfg::from_succ_only(vec![vec![1], vec![]]);
        let defs = Defs::new(
            2,
            vec![
                Def {
                    row: None,
                    key: 3,
                    kills: true,
                },
                Def {
                    row: None,
                    key: 4,
                    kills: true,
                },
                Def {
                    row: Some(1),
                    key: 3,
                    kills: true,
                },
            ],
        );
        let entries = defs.solve(&g);
        let mut st = entries[0].clone();
        assert_eq!(st.ones(), [0, 1]);
        defs.step(&mut st, 0);
        defs.step(&mut st, 1);
        assert_eq!(st.ones(), [1, 2], "the row-1 write ends key 3's entry site");
    }

    #[test]
    fn a_site_written_at_two_rows_reaches_from_the_later() {
        let g = Cfg::from_succ_only(vec![vec![1], vec![2], vec![3], vec![]]);
        let d = |row, kills| Def {
            row: Some(row),
            key: 0,
            kills,
        };
        let mut defs = Defs::new(4, vec![d(0, true), d(1, true)]);
        defs.also(2, 0);
        let entries = defs.solve(&g);
        let mut before = vec![Vec::new(); 4];
        defs.walk(&g, &entries, |k, st| before[k] = defs.of_key(st, 0));
        assert_eq!(before, [vec![], vec![0], vec![1], vec![0]]);
    }
}
