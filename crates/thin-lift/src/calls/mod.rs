//! Resolving PC-relative indirect calls without changing physical instructions.
//!
//! A finite SGPR constant propagation joins complete address pairs, not
//! independent low/high words. The low-add/carry-add relation preserves
//! relocation invariance. Unknown values and bounded-domain widening are
//! explicit refusals, never guessed addresses.

mod state;

use std::collections::{BTreeSet, VecDeque};

use thin_ir::control::{ControlFlow, Destination, Edge, EdgeKind, Transfer, TransferKind};
use thin_ir::{Operand, Reg};
use thin_target::Flags;

use crate::program::{Program, ProgramError};
use state::{AddressRole, Pair, State};

/// Why a transfer has no complete finite target set.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Reason {
    /// The SGPR value originates in dispatch data or another unknown input.
    Input,
    /// A load or unsupported instruction overwrote an address dependency.
    Written(usize),
    /// A scalar value set exceeded the finite domain.
    ScalarLimit,
    /// An address or return-link set exceeded the finite domain.
    AddressLimit,
    /// A scalar changed across a backward branch and was widened.
    LoopValue,
    /// Independent alternatives cannot be combined without losing correlation.
    Correlation,
    /// Only part of a PC-relative pair was copied or overwritten.
    PartialAddress,
    /// Carry no longer belongs to the low-word address addition.
    Carry,
    /// The value is a literal, not a relocated code address.
    NotPcRelative,
    /// A source or link is not a numbered SGPR pair.
    Operand,
    /// The resolved byte address is not a known instruction boundary.
    Boundary,
    /// The relative address overflows the representable range.
    AddressRange,
    /// No currently known path reaches the transfer.
    Unreached,
    /// The opcode has no supported PC-transfer rule.
    Opcode,
}

impl Reason {
    /// A stable coverage label; a Written refusal also retains its row.
    pub fn name(self) -> &'static str {
        match self {
            Self::Input => "nonconstant input",
            Self::Written(_) => "dependency overwritten by an unmodeled instruction",
            Self::ScalarLimit => "scalar alternative limit",
            Self::AddressLimit => "address alternative limit",
            Self::LoopValue => "scalar widening on a backward branch",
            Self::Correlation => "lost alternative correlation",
            Self::PartialAddress => "partial PC-relative address",
            Self::Carry => "lost low-add carry relation",
            Self::NotPcRelative => "not a PC-relative address",
            Self::Operand => "not a numbered SGPR pair",
            Self::Boundary => "target is not an instruction boundary",
            Self::AddressRange => "relative address outside the program address range",
            Self::Unreached => "not reached on the resolved CFG",
            Self::Opcode => "unsupported PC-transfer opcode",
        }
    }
}

/// One physical transfer's finite target set or explicit refusal.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Site {
    /// IR row of the call, jump or return.
    pub instruction: usize,
    /// The typed operation and link binding.
    pub kind: TransferKind,
    /// Known typed edges. Complete sets have no refusal reason.
    pub edges: Vec<Edge>,
    /// Why the target set is incomplete, if it is.
    pub unresolved: Option<Reason>,
}

/// Reversible control-flow annotations, with no saved instruction operands.
#[derive(Clone, Debug)]
pub struct Calls {
    /// Every indirect transfer, including unresolved and unreachable sites.
    pub sites: Vec<Site>,
    original: Option<ControlFlow>,
    original_indirect: Vec<usize>,
    resolved: ControlFlow,
}

#[derive(Default)]
struct ReturnSummary {
    state: Option<State>,
    callers: BTreeSet<usize>,
}

fn pair(operand: Option<&Operand>) -> Option<Reg> {
    operand
        .and_then(Operand::reg)
        .copied()
        .filter(|r| r.file == 's' && r.n == 2 && r.lo % 2 == 0)
}

fn destination(program: &Program, address: i64) -> Result<Destination, Reason> {
    let address = u32::try_from(address).map_err(|_| Reason::AddressRange)?;
    program
        .boundaries
        .get(&address)
        .copied()
        .ok_or(Reason::Boundary)
}

fn site(program: &Program, row: usize, state: Option<&State>) -> Site {
    let i = &program.snapshot.instrs()[row];
    let canon = i.form().canon.as_str();
    let is_call = canon == "s.swappc.b64";
    let source = pair(i.ops().get(usize::from(is_call)));
    let link = pair(i.ops().first());
    let continuation = destination(program, i64::from(program.addresses[row]) + 4);
    let kind = if is_call {
        match (link, continuation) {
            (Some(link), Ok(continuation)) => TransferKind::Call { link, continuation },
            (_, continuation) => TransferKind::UnresolvedCall {
                continuation: continuation.ok(),
            },
        }
    } else {
        TransferKind::Jump
    };
    let mut out = Site {
        instruction: row,
        kind,
        edges: Vec::new(),
        unresolved: None,
    };
    if is_call && continuation.is_err() {
        out.unresolved = continuation.err();
        return out;
    }
    if !matches!(canon, "s.setpc.b64" | "s.swappc.b64") {
        out.unresolved = Some(Reason::Opcode);
        return out;
    }
    let Some(source) = source.filter(|_| !is_call || link.is_some()) else {
        out.unresolved = Some(Reason::Operand);
        return out;
    };
    let Some(state) = state else {
        out.unresolved = Some(Reason::Unreached);
        return out;
    };
    let addresses = match state.address(source.lo) {
        Pair::Values(values) => values,
        Pair::Unknown(reason) => {
            out.unresolved = Some(*reason);
            return out;
        }
        Pair::Pending(_) => {
            out.unresolved = Some(Reason::PartialAddress);
            return out;
        }
    };
    let mut returns = !is_call;
    for address in addresses.iter() {
        let target = match destination(program, address.offset) {
            Ok(target) => target,
            Err(reason) => {
                out.unresolved = Some(reason);
                continue;
            }
        };
        let edge_kind = if is_call {
            EdgeKind::Call
        } else {
            match address.role {
                AddressRole::Code => {
                    returns = false;
                    EdgeKind::Branch
                }
                AddressRole::Return(caller) => EdgeKind::Return { caller },
            }
        };
        out.edges.push(Edge {
            target,
            kind: edge_kind,
        });
    }
    out.edges.sort_unstable();
    out.edges.dedup();
    if returns && !out.edges.is_empty() {
        out.kind = TransferKind::Return { link: source };
    }
    out
}

/// Resolves indirect targets by finite constant propagation on a compacted
/// CFG. Vector-only stretches share a propagation node; every SGPR/SCC
/// writer and branch remains explicit. Edges discovered at calls propagate
/// the newly written link pair into callees; propagated return links add
/// caller-specific continuation edges and resume the caller.
pub fn resolve_calls(mut program: Program) -> Result<(Program, Calls), ProgramError> {
    let snapshot = &program.snapshot;
    let n = snapshot.instrs().len();
    let indirect: BTreeSet<_> = snapshot
        .instrs()
        .iter()
        .enumerate()
        .filter(|(_, i)| {
            matches!(
                i.form().canon.as_str(),
                "s.setpc.b64" | "s.swappc.b64" | "s.call.b64"
            )
        })
        .map(|(row, _)| row)
        .collect();
    let scc = thin_target::phys_named("scc").expect("scc").number;
    let nodes: Vec<_> = snapshot
        .instrs()
        .iter()
        .enumerate()
        .filter(|(row, i)| {
            *row == 0
                || indirect.contains(row)
                || i.form().flags.any(Flags::BRANCH | Flags::ENDPGM)
                || i.defs().any(|r| r.file == 's')
                || i.writes().contains(&scc)
                || i.reads().contains(&scc)
        })
        .map(|(row, _)| row)
        .collect();
    let mut node_of = vec![None; n + 1];
    let mut next = None;
    let mut node = nodes.len();
    for row in (0..n).rev() {
        if node > 0 && nodes[node - 1] == row {
            node -= 1;
            next = Some(node);
        }
        node_of[row] = next;
    }
    let cfg = snapshot.cfg(None);
    let mut states: Vec<Option<State>> = vec![None; nodes.len()];
    let mut queued = vec![false; nodes.len()];
    let mut returns: Vec<ReturnSummary> =
        (0..nodes.len()).map(|_| ReturnSummary::default()).collect();
    let mut queue = VecDeque::new();
    if !nodes.is_empty() {
        states[0] = Some(State::new(snapshot.target().hooks().reg_count('s')));
        queue.push_back(0);
        queued[0] = true;
    }
    while let Some(at) = queue.pop_front() {
        queued[at] = false;
        let row = nodes[at];
        let before = states[at].as_ref().expect("reached propagation node");
        let mut successors = Vec::new();
        let mut return_pair = None;
        if indirect.contains(&row) {
            let resolved = site(&program, row, Some(before));
            if let TransferKind::Return { link } = resolved.kind {
                return_pair = Some(link.lo);
            }
            for edge in &resolved.edges {
                if let Destination::Instruction(target) = edge.target {
                    if let Some(target) = node_of[target] {
                        let restriction = match edge.kind {
                            EdgeKind::Return { caller } => {
                                pair(snapshot.instrs()[row].ops().first()).map(|r| (r.lo, caller))
                            }
                            _ => None,
                        };
                        successors.push((
                            target,
                            restriction,
                            matches!(edge.kind, EdgeKind::Branch),
                        ));
                    }
                }
            }
        } else {
            successors.extend(
                cfg.succ[row]
                    .iter()
                    .filter_map(|&row| node_of[row])
                    .map(|target| (target, None, true)),
            );
        }
        successors.sort_unstable();
        successors.dedup();
        let mut after = before.clone();
        after.step(&program, row);
        // A new caller grows the return-link set, not the subroutine's
        // other output facts. Repropagate existing caller edges only when
        // those other facts change; each edge narrows the link to its caller.
        if let Some(pair) = return_pair {
            let summary = after.without_link(pair);
            if returns[at].state.as_ref() != Some(&summary) {
                returns[at].state = Some(summary);
                returns[at].callers.clear();
            }
        }
        for (target, restriction, branch) in successors {
            if let Some((_, caller)) = restriction {
                if return_pair.is_some() && !returns[at].callers.insert(caller) {
                    continue;
                }
            }
            let mut outgoing = after.clone();
            if let Some((pair, caller)) = restriction {
                outgoing.restrict_return(pair, caller);
            }
            let changed = match &mut states[target] {
                None => {
                    states[target] = Some(outgoing);
                    true
                }
                Some(state) => state.join(&outgoing, branch && nodes[target] <= row),
            };
            if changed && !queued[target] {
                queue.push_back(target);
                queued[target] = true;
            }
        }
    }
    let sites: Vec<_> = indirect
        .iter()
        .map(|&row| {
            site(
                &program,
                row,
                node_of[row].and_then(|node| states[node].as_ref()),
            )
        })
        .collect();
    let resolved = ControlFlow {
        transfers: sites
            .iter()
            .map(|s| Transfer {
                instruction: s.instruction,
                kind: s.kind,
                edges: s.edges.clone(),
                complete: s.unresolved.is_none(),
            })
            .collect(),
    };
    let original = program.snapshot.control_flow().cloned();
    let original_indirect = program.indirect.clone();
    program.indirect = sites
        .iter()
        .filter(|s| s.unresolved.is_some())
        .map(|s| s.instruction)
        .collect();
    program.snapshot = program
        .snapshot
        .with_control_flow(Some(resolved.clone()))
        .map_err(|e| ProgramError {
            offset: None,
            class: "call graph",
            detail: e.to_string(),
        })?;
    Ok((
        program,
        Calls {
            sites,
            original,
            original_indirect,
            resolved,
        },
    ))
}

impl Calls {
    /// Whether every transfer has a complete finite target set.
    pub fn complete(&self) -> bool {
        self.sites.iter().all(|s| s.unresolved.is_none())
    }

    /// Removes precisely this analysis's annotations, preserving all physical
    /// instructions. Refuses to overwrite a subsequently edited call graph.
    pub fn forget(self, mut program: Program) -> Result<Program, ProgramError> {
        if program.snapshot.control_flow() != Some(&self.resolved) {
            return Err(ProgramError {
                offset: None,
                class: "call graph",
                detail: "call annotations were edited".into(),
            });
        }
        program.snapshot = program
            .snapshot
            .with_control_flow(self.original)
            .map_err(|e| ProgramError {
                offset: None,
                class: "call graph",
                detail: e.to_string(),
            })?;
        program.indirect = self.original_indirect;
        Ok(program)
    }
}
