//! Experimental, test-only GEMM provenance checks; not a correctness certificate.
//!
//! Problems as data, the
//! matrix-op check and the store checks.
//!
//! **The module rule.** This module uses only `tags`; the machine never
//! uses it. It implements the checker interface [`crate::tags::Checker`].

pub mod data;
pub mod layout;
pub mod matrix;
pub mod problem;
pub mod store;

use std::collections::HashMap;

use crate::tags::{
    mem_tag, AccState, AtomicAnswer, Checker, MatrixOp, MatrixOutcome, ProblemFacts, StoreCall,
    Tables, Tag, TagError,
};

pub use layout::{Decoded, Kind, Layout, Shape};
pub use problem::{Cell, DCols, Problem, RecordOperand};

/// `GemmHooks(problems, memory)`: the checker over these problems and
/// the memory's buffer names (by index).
#[derive(Clone, Debug)]
pub struct GemmHooks {
    /// The problems, in the order given.
    pub probs: Vec<Problem>,
    /// The current problem's index in `probs`.
    pub current: usize,
    /// The memory's buffer names, by index.
    pub names: Vec<String>,
    owner: HashMap<usize, usize>,
}

impl GemmHooks {
    /// Registers the problems that have no pid and maps buffers to problems.
    pub fn new(tables: &mut Tables, mut probs: Vec<Problem>, names: Vec<String>) -> GemmHooks {
        for p in &mut probs {
            if p.pid < 0 {
                p.pid = tables.register(ProblemFacts {
                    k: p.k,
                    n: p.n,
                    source: None,
                });
            }
        }
        for p in &probs {
            let source = p.record_operand.as_ref().map(|r| r.source);
            let slot = usize::try_from(p.pid).ok();
            if let Some(f) = slot.and_then(|i| tables.problems.get_mut(i)) {
                f.source = source;
                f.k = p.k;
                f.n = p.n;
            }
        }
        let mut owner = HashMap::new();
        for (bi, name) in names.iter().enumerate() {
            for (pi, p) in probs.iter().enumerate() {
                if p.layouts.iter().any(|(n, _)| n == name) {
                    owner.insert(bi, pi);
                }
            }
        }
        GemmHooks {
            probs,
            current: 0,
            names,
            owner,
        }
    }

    /// The owning problem's index of a buffer.
    pub fn owner_of(&self, buffer: usize) -> Option<usize> {
        self.owner.get(&buffer).copied()
    }

    /// The layout of a buffer, switching the current problem to its owner.
    pub fn layout_of(&mut self, buffer: usize) -> Option<Layout> {
        let pi = self.owner_of(buffer)?;
        self.current = pi;
        let name = self.names.get(buffer)?;
        self.probs[pi].layout(name).cloned()
    }

    /// The name of a buffer index.
    pub fn name(&self, buffer: i64) -> String {
        usize::try_from(buffer)
            .ok()
            .and_then(|b| self.names.get(b).cloned())
            .unwrap_or_else(|| "no buffer".to_owned())
    }
}

impl Checker for GemmHooks {
    fn products(&self) -> bool {
        false
    }

    fn store(&mut self, tables: &mut Tables, call: &StoreCall<'_>) -> Result<(), TagError> {
        store::store(self, tables, call)
    }

    fn atomic(
        &mut self,
        tables: &mut Tables,
        call: &StoreCall<'_>,
    ) -> Result<AtomicAnswer, TagError> {
        store::atomic(self, tables, call)
    }

    fn matrix(
        &mut self,
        tables: &mut Tables,
        op: &MatrixOp<'_>,
        acc: &mut AccState,
    ) -> Result<MatrixOutcome, TagError> {
        matrix::check(self, tables, op, acc)
    }

    fn as_gemm(&mut self) -> Option<&mut dyn std::any::Any> {
        Some(self)
    }
}

/// The memory tag of a buffer's byte, for callers that build operand tags.
pub fn tag_of(buffer: usize, offset: u64) -> Tag {
    mem_tag(buffer, offset)
}
