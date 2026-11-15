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

/// (cycles, conflict cycles) of one execution: per phase with an active lane,
/// as many cycles as its busiest bank has distinct dwords.
pub fn bank_count(rule: &BankRule, addrs: &[u64], active: &[bool]) -> (u64, u64) {
    let mut phases: BTreeMap<u32, BTreeMap<u64, BTreeSet<u64>>> = BTreeMap::new();
    for (l, (a, on)) in addrs.iter().zip(active).enumerate() {
        if !on {
            continue;
        }
        let dword = a >> 2;
        let phase = rule.phase.get(l).copied().unwrap_or(0);
        phases
            .entry(phase)
            .or_default()
            .entry(dword % rule.banks)
            .or_default()
            .insert(dword);
    }
    let cycles: u64 = phases
        .values()
        .map(|banks| banks.values().map(|d| d.len() as u64).max().unwrap_or(0))
        .sum();
    (cycles, cycles - phases.len() as u64)
}
