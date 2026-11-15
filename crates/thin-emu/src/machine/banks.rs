//! The LDS bank model's count.

use std::collections::{BTreeMap, BTreeSet};

/// A target's bank rule for one LDS access form: banks B and each lane's phase.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BankRule {
    /// Banks.
    pub banks: u64,
    /// The phase of each lane.
    pub phase: Vec<u32>,
}

/// (cycles, conflict cycles) of one single-dword execution: per phase with an
/// active lane, as many cycles as its busiest bank has distinct dwords.
pub fn bank_count(rule: &BankRule, addrs: &[u64], active: &[bool]) -> (u64, u64) {
    bank_count_bytes(rule, 4, addrs, active)
}

/// (cycles, conflict cycles) of an `n`-byte-per-lane execution: each active
/// lane accesses every 4-byte dword spanned by `[addr, addr + n)`.
pub fn bank_count_bytes(rule: &BankRule, bytes: u32, addrs: &[u64], active: &[bool]) -> (u64, u64) {
    let n_dw = u64::from(bytes.max(1).div_ceil(4));
    let mut phases: BTreeMap<u32, BTreeMap<u64, BTreeSet<u64>>> = BTreeMap::new();
    for (l, (a, on)) in addrs.iter().zip(active).enumerate() {
        if !on {
            continue;
        }
        let first_dw = a >> 2;
        let phase = rule.phase.get(l).copied().unwrap_or(0);
        let bank_map = phases.entry(phase).or_default();
        for dword in first_dw..first_dw + n_dw {
            bank_map
                .entry(dword % rule.banks)
                .or_default()
                .insert(dword);
        }
    }
    let cycles: u64 = phases
        .values()
        .map(|banks| banks.values().map(|d| d.len() as u64).max().unwrap_or(0))
        .sum();
    (cycles, cycles - phases.len() as u64)
}
