//! Typed physical control-flow edges supplementing branch-label operands.

use crate::Reg;

/// An instruction boundary or the end of the program.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Destination {
    /// An IR row, including a label row.
    Instruction(usize),
    /// Control leaves the instruction sequence.
    Exit,
}

/// Why a physical transfer follows an edge.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum EdgeKind {
    /// An indirect branch whose address has been resolved.
    Branch,
    /// Entry to a called subroutine.
    Call,
    /// Return to this caller's continuation.
    Return {
        /// The caller's IR row.
        caller: usize,
    },
}

/// A typed edge of a physical transfer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Edge {
    /// Its destination, not a string label.
    pub target: Destination,
    /// Its call/return role.
    pub kind: EdgeKind,
}

/// The operation at a physical control-transfer site.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TransferKind {
    /// A PC assignment, including a return if its edges carry caller identities.
    Jump,
    /// A call that writes the following instruction's address to a link pair.
    Call {
        /// The pinned link pair.
        link: Reg,
        /// The caller's fall-through destination.
        continuation: Destination,
    },
    /// A call whose link or continuation is not representable by this analysis.
    UnresolvedCall {
        /// Its continuation when the input byte boundary is known.
        continuation: Option<Destination>,
    },
    /// A return through a propagated link pair.
    Return {
        /// The pinned register pair holding return addresses.
        link: Reg,
    },
}

/// All currently known edges of one physical transfer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Transfer {
    /// Its IR row.
    pub instruction: usize,
    /// Its operation and typed register binding.
    pub kind: TransferKind,
    /// Exact edges when complete; a known subset otherwise.
    pub edges: Vec<Edge>,
    /// Whether every incoming represented path has a resolved target.
    pub complete: bool,
}

/// Physical control flow. Missing edges remain explicit, never fall-through.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ControlFlow {
    /// Transfers in strictly increasing IR-row order.
    pub transfers: Vec<Transfer>,
}

impl ControlFlow {
    /// Whether all physical transfers have complete target sets.
    pub fn complete(&self) -> bool {
        self.transfers.iter().all(|t| t.complete)
    }
}

/// How one CFG edge relates to calls, for analyses that keep call sites
/// apart ([`crate::live`]): a callee's code is shared, but a return goes
/// only to the continuation of the call that entered it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Flow {
    /// An ordinary edge.
    #[default]
    Plain,
    /// Entry to a callee from the call at the edge's source.
    Call,
    /// A return to the continuation of the call at row `caller`.
    Return {
        /// The call's row.
        caller: usize,
    },
}

/// A backward state's call context: the call whose continuation it is
/// live at (it came back through a return to that call), or none when it
/// holds for every caller.
pub type Context = Option<usize>;

impl Flow {
    /// The context a state with context `ctx` after this edge has before
    /// it, the edge going out of row `from`; none when no valid path
    /// carries it (a state of one call's continuation entering a callee
    /// from another call). A return replaces an outer context, so nested
    /// calls keep only the innermost: a state that lost its outer call
    /// holds for every caller, which is never less than the truth.
    pub fn back(self, from: usize, ctx: Context) -> Option<Context> {
        match (self, ctx) {
            (Flow::Plain, ctx) => Some(ctx),
            (Flow::Return { caller }, _) => Some(Some(caller)),
            (Flow::Call, None) => Some(None),
            (Flow::Call, Some(call)) => (call == from).then_some(None),
        }
    }
}

/// Per row, the flow of each of `succ[row]`'s edges: the typed call and
/// return edges of `control`, ordinary edges elsewhere. An edge that is a
/// call or return edge of a transfer is that kind; a transfer's own edge
/// listed twice keeps the first kind.
pub fn flows(control: Option<&ControlFlow>, succ: &[Vec<usize>]) -> Vec<Vec<Flow>> {
    let mut out: Vec<Vec<Flow>> = succ.iter().map(|s| vec![Flow::Plain; s.len()]).collect();
    let Some(control) = control else {
        return out;
    };
    for t in &control.transfers {
        let Some(row) = out.get_mut(t.instruction) else {
            continue;
        };
        for e in &t.edges {
            let Destination::Instruction(to) = e.target else {
                continue;
            };
            let flow = match e.kind {
                EdgeKind::Call => Flow::Call,
                EdgeKind::Return { caller } => Flow::Return { caller },
                EdgeKind::Branch => continue,
            };
            if let Some(k) = succ[t.instruction].iter().position(|&s| s == to) {
                if row[k] == Flow::Plain {
                    row[k] = flow;
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_return_tags_and_only_its_call_untags() {
        let ret = Flow::Return { caller: 7 };
        assert_eq!(ret.back(20, None), Some(Some(7)));
        assert_eq!(ret.back(20, Some(3)), Some(Some(7)), "the innermost call");
        assert_eq!(Flow::Call.back(7, Some(7)), Some(None));
        assert_eq!(
            Flow::Call.back(9, Some(7)),
            None,
            "another call's continuation"
        );
        assert_eq!(Flow::Call.back(9, None), Some(None));
        assert_eq!(Flow::Plain.back(9, Some(7)), Some(Some(7)));
    }
}
