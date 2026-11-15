//! The machine: numbers, memory, waves, the LDS, waits,
//! the semantics and clusters. It uses `tags` and never `gemm`.

// Lane loops index several per-lane arrays by the lane number.
#![allow(clippy::needless_range_loop)]

use std::collections::{BTreeMap, BTreeSet};

pub mod alu;
pub mod banks;
pub mod claims;
pub mod cluster;
pub mod cvt;
pub mod dot;
pub mod facts;
pub mod flight;
pub mod lds;
pub mod memops;
pub mod memory;
pub mod mfma;
pub mod numbers;
pub mod pc;
pub mod pending;
pub mod program;
pub mod run;
pub mod scope;
pub mod trans32;
pub mod wave;
pub mod wmma;

use crate::tags::{Checker, Tables};

/// The value every register starts as: a bf16 NaN in both halves,
/// an f32 NaN, an address far out of range.
pub const POISON: u32 = 0x7FC1_7FC1;

/// The machine's counts.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Stats {
    /// Instructions.
    pub ins: u64,
    /// Matrix ops.
    pub matrix: u64,
    /// SALU issues (by form kind).
    pub salu: u64,
    /// VALU issues.
    pub valu: u64,
    /// LDS write bytes (ds writes, LDS atomics).
    pub lds_write_bytes: u64,
    /// LDS read bytes.
    pub ds_read_bytes: u64,
    /// VMEM load bytes (active lanes × bytes).
    pub vmem_load_bytes: u64,
    /// Global store bytes.
    pub store_bytes: u64,
    /// The most instructions one wave issued.
    pub wave_ins_max: u64,
}

/// The (rank, WG x, wave)s of a scalar poll.
pub type Pollers = BTreeSet<(usize, u32, usize)>;

/// A machine's reports after a run.
#[derive(Clone, Debug, Default)]
pub struct Reports {
    /// The counts.
    pub stats: Stats,
    /// Instruction → [executions, cycles, conflict cycles].
    pub banks: BTreeMap<usize, [u64; 3]>,
    /// VMEM-wait index → [executions, loads retired, stores retired,
    /// coupled executions, stores those retired, coupled past the first].
    pub vmw: BTreeMap<usize, [u64; 6]>,
    /// Instruction → [executions, bytes] of LDS reads of unwritten bytes.
    pub ldsu: BTreeMap<usize, [u64; 2]>,
    /// (instruction, 128-byte line) → (executions, the (rank, WG x, wave)s).
    pub spoll: BTreeMap<(usize, u64), (u64, Pollers)>,
}

/// A workgroup's shared state: its serial, epoch and barrier frontier.
#[derive(Clone, Debug, Default)]
pub struct WgState {
    /// The workgroup's serial in the memory (unique over ranks).
    pub serial: usize,
    /// Its epoch: the barriers passed.
    pub epoch: i64,
    /// Every wave's mark at the last barrier.
    pub barrier_marks: Vec<i64>,
}

/// What one instruction's semantics reach.
pub struct Exec<'a> {
    /// The target facts.
    pub facts: &'a facts::Facts,
    /// The tag tables.
    pub tables: &'a mut Tables,
    /// The checker.
    pub checker: &'a mut dyn Checker,
    /// Sum mode.
    pub sum_mode: bool,
    /// The memory.
    pub mem: &'a mut memory::Memory,
    /// The workgroup's LDS.
    pub lds: &'a mut lds::Lds,
    /// The workgroup's epoch.
    pub epoch: i64,
    /// The rank.
    pub rank: usize,
    /// The machine's reports.
    pub reports: &'a mut Reports,
    /// The instruction texts.
    pub texts: &'a [String],
    /// The workgroup.
    pub wg: &'a mut WgState,
    /// The IPC policy.
    pub ipc_policy: bool,
    /// The instruction's serial (so one instruction's chunks merge).
    pub serial: u64,
    /// The instruction's index.
    pub pc: usize,
    /// The machine's clock.
    pub clock: &'a mut u64,
}
