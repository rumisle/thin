//! The lift: shipped AMDHSA code objects (see docs/dev/lift.md) back into
//! thin's terms, each step the inverse of one of thin's lowering steps.
//!
//! **Owns** reading a multi-kernel code object ([`object`]) and its typed
//! entry interface ([`interface`]), separating its schedule ([`schedule`]),
//! reversing long-branch relaxation ([`branches`]), constructing a pinned
//! physical IR Program ([`program`]), resolving its calls ([`calls`]) and
//! recovering its values from its registers ([`unassign`]).
//!
//! **Depends on** thin-formats, thin-target, thin-targets (the decoder),
//! thin-emu, thin-ir and thin-passes.
//!
//! **Invariants.** Each step states its law: the lowering step it inverts,
//! applied to its result, gives back its input.
//!
//! See docs/dev/lift.md for each step, its law and its typed residue/refusals.

pub mod branches;
pub mod interface;
pub mod object;
pub mod schedule;

pub mod program;

pub mod calls;

pub mod unassign;
