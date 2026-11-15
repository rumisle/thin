//! Recognizing PC-relative long jumps, the inverse of branch relaxation.
//!
//! Offsets stay in the input address space, as with unscheduling. Only
//! statically resolved, in-range instruction targets (or the end) become
//! branches. Original scratch-register and SCC effects remain explicit in
//! the rewrite record: this is not a proof that those effects are dead.

use std::collections::BTreeSet;
use std::ops::Bound::{Excluded, Unbounded};

use thin_targets::decode::Opd;
use thin_targets::Instruction;

use crate::schedule::Schedule;

/// One long jump replaced by a short branch.
#[derive(Clone)]
pub struct BranchRewrite {
    /// Instruction index of the replacement in the resulting sequence.
    pub position: usize,
    /// The complete original sequence, including offsets and literal sizes.
    pub original: Vec<Instruction>,
    /// The generated short branch, used to validate reversal.
    pub replacement: Instruction,
    /// First SGPR of the scratch pair written by the input sequence.
    pub scratch_sgpr: u16,
    /// Extra SGPR holding a folded displacement in the five-instruction form.
    pub displacement_sgpr: Option<u16>,
    /// The displacement materialization itself writes SCC; the following
    /// unsigned low add overwrites it before the high add consumes carry.
    pub displacement_writes_scc: bool,
    /// Whether the input addition writes SCC.
    pub writes_scc: bool,
}

/// Rewrites in strictly increasing result-position order.
#[derive(Clone, Default)]
pub struct BranchRewrites {
    /// Every replaced sequence, with its original instructions.
    pub entries: Vec<BranchRewrite>,
}

/// A rewrite record does not describe the supplied result sequence.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RewriteError {
    /// Positions are unordered, duplicated or outside the result.
    Position,
    /// The replacement instruction was edited after the rewrite.
    Replacement,
    /// A record cannot restore an empty original sequence.
    EmptyOriginal,
    /// The restored instruction count overflows usize.
    Length,
}

impl std::fmt::Display for RewriteError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Position => "branch rewrite positions are invalid",
            Self::Replacement => "branch rewrite replacement no longer matches",
            Self::EmptyOriginal => "branch rewrite has an empty original sequence",
            Self::Length => "restored branch sequence length overflows usize",
        })
    }
}

impl std::error::Error for RewriteError {}

/// Why a getpc sequence was rewritten or left unchanged.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum BranchOutcome {
    /// A recognized static long jump became a short branch.
    Rewritten,
    /// Not one of the supported getpc/add/setpc forms.
    DifferentForm,
    /// The arithmetic computes a PC value, but s_branch transfers control
    /// elsewhere rather than immediately executing setpc with that value.
    PcSetup,
    /// The resolved target cannot be represented in the decoded address space.
    TargetOutsideAddressSpace,
    /// The target is not dword aligned relative to the branch's following PC.
    UnalignedTarget,
    /// The target is neither an instruction start nor the program's end.
    UnknownTarget,
    /// The displacement does not fit a signed 16-bit dword offset.
    OutOfRange,
    /// A known branch enters the input sequence after its first instruction.
    InteriorEntry,
}

impl BranchOutcome {
    /// A stable diagnostic label suitable for coverage counts.
    pub fn name(self) -> &'static str {
        match self {
            Self::Rewritten => "rewritten",
            Self::DifferentForm => "different form",
            Self::PcSetup => "PC setup ends in s_branch, not immediate setpc",
            Self::TargetOutsideAddressSpace => "target outside decoded address space",
            Self::UnalignedTarget => "unaligned target",
            Self::UnknownTarget => "target is not an instruction or end boundary",
            Self::OutOfRange => "outside signed-16-dword range",
            Self::InteriorEntry => "known incoming edge to sequence interior",
        }
    }
}

/// The disposition of one getpc instruction, including non-jump uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BranchAnalysis {
    /// Index of the getpc instruction in the input.
    pub position: usize,
    /// Length of a recognized PC-relative setup, or zero for a different form.
    pub sequence_len: usize,
    /// Resolved input-address-space PC value, when representable.
    pub target: Option<u32>,
    /// The rewrite decision or primary refusal reason.
    pub outcome: BranchOutcome,
}

#[derive(Clone, Copy)]
struct Candidate {
    start: usize,
    len: usize,
    target: i128,
    scratch: u16,
    displacement_sgpr: Option<u16>,
    writes_scc: bool,
    pc_setup: bool,
}

fn plain(i: &Instruction) -> bool {
    i.modifiers().is_empty() && (0..i.operands().len()).all(|k| i.signs(k) == (false, false))
}

fn sum_sources(i: &Instruction, mnemonic: &str, dst: Opd, left: Opd, right: Opd) -> bool {
    i.mnemonic == mnemonic
        && plain(i)
        && matches!(i.operands(), [d, a, b]
        if *d == dst && ((*a == left && *b == right) || (*a == right && *b == left)))
}

/// An addition to the same register, with one integer constant.
fn addend(i: &Instruction, mnemonic: &str, register: Opd) -> Option<i64> {
    if i.mnemonic != mnemonic || !plain(i) {
        return None;
    }
    match i.operands() {
        [dst, src, Opd::Int(n)] | [dst, Opd::Int(n), src]
            if *dst == register && *src == register =>
        {
            Some(*n)
        }
        _ => None,
    }
}

fn constant_u32(n: i64) -> Option<u32> {
    // Negative inline integers are their two's-complement u32 bit patterns.
    (-16..=i64::from(u32::MAX)).contains(&n).then_some(n as u32)
}

/// This signed add is constant materialization, not the PC's low-word add.
/// Its result is the wrapped low dword; its overflow SCC is overwritten.
fn folded_displacement(i: &Instruction) -> Option<(u16, u32)> {
    if i.mnemonic != "s_add_co_i32" || !plain(i) {
        return None;
    }
    let &[Opd::S { n, count: 1 }, Opd::Int(a), Opd::Int(b)] = i.operands() else {
        return None;
    };
    Some((n, constant_u32(a)?.wrapping_add(constant_u32(b)?)))
}

fn candidate(instructions: &[Instruction], start: usize) -> Option<Candidate> {
    let tail = &instructions[start..];
    let get = tail.first()?;
    if get.mnemonic != "s_getpc_b64" || get.size != 4 || !plain(get) {
        return None;
    }
    let &[pair @ Opd::S { n, count: 2 }] = get.operands() else {
        return None;
    };
    let low_register = Opd::S { n, count: 1 };
    let high_register = Opd::S {
        n: n.checked_add(1)?,
        count: 1,
    };
    let (len, displacement, displacement_sgpr, writes_scc) =
        if let Some(delta) = addend(tail.get(1)?, "s_add_nc_u64", pair) {
            (3, delta, None, false)
        } else {
            let (low, high_index, temporary) =
                if let Some((temp, low)) = folded_displacement(tail.get(1)?) {
                    if temp == n
                        || temp == n.checked_add(1)?
                        || !sum_sources(
                            tail.get(2)?,
                            "s_add_co_u32",
                            low_register,
                            low_register,
                            Opd::S { n: temp, count: 1 },
                        )
                    {
                        return None;
                    }
                    (low, 3, Some(temp))
                } else {
                    (
                        constant_u32(addend(tail.get(1)?, "s_add_co_u32", low_register)?)?,
                        2,
                        None,
                    )
                };
            let high = constant_u32(addend(
                tail.get(high_index)?,
                "s_add_co_ci_u32",
                high_register,
            )?)?;
            let bits = u64::from(low) | u64::from(high) << 32;
            (high_index + 2, bits as i64, temporary, true)
        };
    let set = tail.get(len - 1)?;
    let pc_setup = if set.mnemonic == "s_setpc_b64" && set.operands() == [pair] && plain(set) {
        false
    } else if set.mnemonic == "s_branch" && set.branch_target().is_some() && plain(set) {
        true
    } else {
        return None;
    };
    // RDNA4 ISA guide S_GETPC_B64 saves the next instruction's PC.
    Some(Candidate {
        start,
        len,
        target: i128::from(get.offset) + 4 + i128::from(displacement),
        scratch: n,
        displacement_sgpr,
        writes_scc,
        pc_setup,
    })
}

fn disposition(
    c: &Candidate,
    instructions: &[Instruction],
    targets: &BTreeSet<u32>,
    boundaries: &BTreeSet<u32>,
) -> BranchOutcome {
    if c.pc_setup {
        return BranchOutcome::PcSetup;
    }
    let Ok(target) = u32::try_from(c.target) else {
        return BranchOutcome::TargetOutsideAddressSpace;
    };
    let first = &instructions[c.start];
    let last = &instructions[c.start + c.len - 1];
    let delta = i64::from(target) - (i64::from(first.offset) + 4);
    if delta % 4 != 0 {
        return BranchOutcome::UnalignedTarget;
    }
    if !boundaries.contains(&target) {
        return BranchOutcome::UnknownTarget;
    }
    if !(-32768..32768).contains(&(delta / 4)) {
        return BranchOutcome::OutOfRange;
    }
    let end = u64::from(last.offset) + u64::from(last.size);
    if targets
        .range((Excluded(first.offset), Unbounded))
        .next()
        .is_some_and(|&t| u64::from(t) < end)
    {
        return BranchOutcome::InteriorEntry;
    }
    BranchOutcome::Rewritten
}

/// Classifies every getpc use with the same recognizer and eligibility
/// checks used by unrelaxation, so corpus refusals cannot be silently lost.
/// If unscheduled, supply the Schedule to retain removed target boundaries.
pub fn inspect_branches(
    instructions: &[Instruction],
    schedule: Option<&Schedule>,
) -> Vec<BranchAnalysis> {
    let candidates: Vec<_> = (0..instructions.len())
        .filter_map(|start| candidate(instructions, start))
        .collect();
    let targets: BTreeSet<_> = instructions
        .iter()
        .filter_map(Instruction::branch_target)
        .chain(
            candidates
                .iter()
                .filter_map(|c| u32::try_from(c.target).ok()),
        )
        .collect();
    let end = instructions
        .last()
        .into_iter()
        .chain(
            schedule
                .into_iter()
                .flat_map(|s| s.entries.iter().map(|e| &e.instruction)),
        )
        .filter_map(|i| i.offset.checked_add(u32::from(i.size)))
        .max();
    let boundaries: BTreeSet<_> = instructions
        .iter()
        .map(|i| i.offset)
        .chain(
            schedule
                .into_iter()
                .flat_map(|s| s.entries.iter().map(|e| e.instruction.offset)),
        )
        .chain(end)
        .collect();
    let mut next = candidates.iter().peekable();
    instructions
        .iter()
        .enumerate()
        .filter(|(_, i)| i.mnemonic == "s_getpc_b64")
        .map(|(position, _)| {
            if let Some(c) = next.next_if(|c| c.start == position) {
                BranchAnalysis {
                    position,
                    sequence_len: c.len,
                    target: u32::try_from(c.target).ok(),
                    outcome: disposition(c, instructions, &targets, &boundaries),
                }
            } else {
                BranchAnalysis {
                    position,
                    sequence_len: 0,
                    target: None,
                    outcome: BranchOutcome::DifferentForm,
                }
            }
        })
        .collect()
}

/// Replaces in-range static long jumps with branches, retaining their
/// complete sequences and clobbers. Supports unsigned low/carry addition,
/// inline 64-bit addition, and signed constant materialization followed by
/// unsigned low/carry addition. Signed arithmetic on the PC remains refused.
/// Known entry points into a removed sequence also prevent rewriting.
/// Supply the Schedule for an unscheduled input: targets at removed waits
/// retain their original labels and can still be recognized as boundaries.
/// PC setups ending in s_branch, rather than setpc, are not rewritten.
pub fn unrelax_branches(
    instructions: &[Instruction],
    schedule: Option<&Schedule>,
) -> (Vec<Instruction>, BranchRewrites) {
    let analysis = inspect_branches(instructions, schedule);
    let mut next = analysis
        .iter()
        .filter(|a| a.outcome == BranchOutcome::Rewritten)
        .peekable();
    let mut out = Vec::with_capacity(instructions.len());
    let mut entries = Vec::new();
    let mut at = 0;
    while at < instructions.len() {
        if let Some(a) = next.next_if(|a| a.position == at) {
            let c = candidate(instructions, at).expect("recognized long jump");
            let mut replacement = Instruction::new(instructions[at].offset, "s_branch");
            replacement.size = 4;
            replacement.push(Opd::Label(a.target.expect("representable branch target")));
            entries.push(BranchRewrite {
                position: out.len(),
                original: instructions[at..at + c.len].to_vec(),
                replacement,
                scratch_sgpr: c.scratch,
                displacement_sgpr: c.displacement_sgpr,
                displacement_writes_scc: c.displacement_sgpr.is_some(),
                writes_scc: c.writes_scc,
            });
            out.push(replacement);
            at += c.len;
        } else {
            out.push(instructions[at]);
            at += 1;
        }
    }
    (out, BranchRewrites { entries })
}

fn same(a: &Instruction, b: &Instruction) -> bool {
    a.offset == b.offset
        && a.size == b.size
        && a.mnemonic == b.mnemonic
        && a.counted == b.counted
        && a.operands() == b.operands()
        && a.modifiers() == b.modifiers()
        && (0..a.operands().len()).all(|k| a.signs(k) == b.signs(k))
}

impl BranchRewrites {
    /// Reverses the records, refusing invalid indices or changed replacement
    /// instructions rather than silently restoring into the wrong program.
    pub fn reverse(&self, instructions: &[Instruction]) -> Result<Vec<Instruction>, RewriteError> {
        let mut previous = None;
        let mut len = instructions.len();
        for entry in &self.entries {
            if entry.position >= instructions.len() || previous.is_some_and(|p| p >= entry.position)
            {
                return Err(RewriteError::Position);
            }
            if !same(&entry.replacement, &instructions[entry.position]) {
                return Err(RewriteError::Replacement);
            }
            if entry.original.is_empty() {
                return Err(RewriteError::EmptyOriginal);
            }
            len = len
                .checked_add(entry.original.len() - 1)
                .ok_or(RewriteError::Length)?;
            previous = Some(entry.position);
        }
        let mut out = Vec::with_capacity(len);
        let mut entries = self.entries.iter().peekable();
        for (position, &instruction) in instructions.iter().enumerate() {
            if entries
                .peek()
                .is_some_and(|entry| entry.position == position)
            {
                out.extend_from_slice(&entries.next().expect("entry at this position").original);
            } else {
                out.push(instruction);
            }
        }
        Ok(out)
    }
}
