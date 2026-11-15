//! The Builder: construction, positions, lines, claims and emission
//!

use pyo3::prelude::*;
use pyo3::types::PyTuple;
use thin_dsl::builder::{ClaimReg, Options, Sizes};
use thin_dsl::values::Placed;
use thin_dsl::Builder;

use super::convert::{ok, phys, reg};
use super::maker::DslMade;
use crate::ir::Ins;
use crate::target::get;

/// The wave index: (register, nwaves, by_wave).
type WaveInfo = (Option<(i32, char, u32, u32)>, u32, bool);

/// The kernel being built.
#[pyclass(module = "thin._native")]
pub(crate) struct DslBuilder {
    pub(crate) b: Builder,
}

/// A claim's register: `("r", (value, file, lo, n))` or `("p", name)`.
fn claim(t: &Bound<'_, PyTuple>) -> PyResult<ClaimReg> {
    let tag: String = t.get_item(0)?.extract()?;
    match tag.as_str() {
        "r" => Ok(ClaimReg::Reg(reg(t.get_item(1)?.extract()?))),
        _ => Ok(ClaimReg::Phys(phys(&t.get_item(1)?.extract::<String>()?)?)),
    }
}

#[pymethods]
impl DslBuilder {
    #[new]
    #[pyo3(signature = (target, vgprs=None, sgprs=None, placement=None))]
    fn new(
        py: Python<'_>,
        target: &str,
        vgprs: Option<u32>,
        sgprs: Option<u32>,
        placement: Option<[Vec<Placed>; 3]>,
    ) -> PyResult<DslBuilder> {
        let sizes = Sizes { vgprs, sgprs };
        Ok(DslBuilder {
            b: Builder::new(get(py, target)?, sizes, Options::default(), placement),
        })
    }

    /// Its identity.
    #[getter]
    fn id(&self) -> u64 {
        self.b.id
    }

    /// Sets the location of the operation about to run.
    fn at(&mut self, loc: u32, where_: String) {
        self.b.loc = loc;
        self.b.where_text = where_;
    }

    /// Sets the options the Builder reads.
    fn set_options(&mut self, audit: bool, safe_barrier: bool) {
        self.b.options = Options {
            audit,
            safe_barrier,
        };
    }

    /// Sets the ablation classes.
    fn set_ablate(&mut self, classes: Vec<String>) {
        self.b.ablate = classes.into_iter().collect();
    }

    fn settle(&mut self) -> PyResult<()> {
        ok(self.b.settle())
    }

    fn seal(&mut self) -> PyResult<()> {
        ok(self.b.seal())
    }

    #[getter]
    fn sealed(&self) -> bool {
        self.b.sealed
    }

    fn check_open(&self) -> PyResult<()> {
        ok(self.b.check_open())
    }

    #[getter]
    fn hazards(&self) -> Option<usize> {
        self.b.hazards
    }

    #[setter]
    fn set_hazards(&mut self, n: Option<usize>) {
        self.b.hazards = n;
    }

    #[getter]
    fn waits_placed(&self) -> bool {
        self.b.waits_placed
    }

    #[setter]
    fn set_waits_placed(&mut self, v: bool) {
        self.b.waits_placed = v;
    }

    fn pos(&mut self) -> PyResult<usize> {
        ok(self.b.pos())
    }

    /// The stream's indices between two positions.
    #[pyo3(signature = (start=0, stop=None))]
    fn emitted(&mut self, start: i64, stop: Option<i64>) -> PyResult<(usize, usize)> {
        ok(self.b.emitted(start, stop)).map(|r| (r.start, r.end))
    }

    fn issued(&self, counter: &str) -> PyResult<u32> {
        ok(self.b.issued(counter))
    }

    fn comment(&mut self, text: &str) -> PyResult<()> {
        ok(self.b.comment(text))
    }

    fn put_label(&mut self, name: &str, from_branches: bool) -> PyResult<()> {
        ok(self.b.put_label(name, from_branches))
    }

    fn new_label(&mut self, stem: &str) -> String {
        self.b.new_label(stem)
    }

    fn discard(&mut self, r: Bound<'_, PyTuple>, why: &str) -> PyResult<()> {
        ok(self.b.discard(claim(&r)?, why))
    }

    fn uniform(&mut self, regs: Vec<Bound<'_, PyTuple>>, why: &str) -> PyResult<()> {
        let regs = regs.iter().map(claim).collect::<PyResult<Vec<_>>>()?;
        ok(self.b.uniform(&regs, why))
    }

    #[pyo3(signature = (regs, why, edge=None))]
    fn undefined(
        &mut self,
        regs: Vec<Bound<'_, PyTuple>>,
        why: &str,
        edge: Option<&str>,
    ) -> PyResult<()> {
        let regs = regs.iter().map(claim).collect::<PyResult<Vec<_>>>()?;
        ok(self.b.undefined(&regs, why, edge))
    }

    fn assume(
        &mut self,
        r: Bound<'_, PyTuple>,
        written: &str,
        lo: i128,
        hi: i128,
        why: &str,
    ) -> PyResult<()> {
        ok(self.b.assume(claim(&r)?, written, lo, hi, why))
    }

    fn nop(&mut self, n: i128) -> PyResult<()> {
        ok(self.b.nop(n))
    }

    fn barrier(&mut self, comment: &str, abl: Vec<String>, fence: bool) -> PyResult<()> {
        ok(self.b.barrier(comment, &abl, fence))
    }

    fn need(&mut self, what: &str, why: &str, arg: Option<String>) -> PyResult<()> {
        ok(self.b.need(what, why, arg.as_deref()))
    }

    /// The open arms and lane regions (`Builder::scope`).
    fn scope(&self) -> (Vec<u64>, Vec<u32>) {
        self.b.scope()
    }

    /// The needs: (co_resident, same_grid pairs).
    fn needs(&self) -> (Option<String>, Vec<(String, String)>) {
        let n = &self.b.needs;
        (
            n.co_resident.clone(),
            n.same_grid.clone().into_iter().collect(),
        )
    }

    fn open_ablatable(&mut self, classes: Vec<String>) {
        self.b.open_ablatable(classes);
    }

    fn close_ablatable(&mut self) {
        self.b.close_ablatable();
    }

    fn ablated(&self, abl: Vec<String>) -> bool {
        self.b.ablated(&abl)
    }

    fn open_alias(&mut self, reads: Vec<String>, over: Vec<String>, why: &str) -> PyResult<()> {
        ok(self.b.open_alias(reads, over, why))
    }

    fn close_alias(&mut self) {
        self.b.close_alias();
    }

    fn hold(&mut self, label: &str, regs: Vec<(i32, char, u32, u32)>) -> PyResult<()> {
        let regs: Vec<_> = regs.into_iter().map(reg).collect();
        ok(self.b.hold(label, &regs))
    }

    fn release(&mut self, label: &str) {
        self.b.release(label);
    }

    fn fork(&mut self) -> PyResult<usize> {
        ok(self.b.fork())
    }

    fn cnt(&mut self) -> PyResult<usize> {
        ok(self.b.cnt())
    }

    fn set_cnt(&mut self, slot: usize) -> PyResult<()> {
        ok(self.b.set_cnt(slot))
    }

    /// The current state's records number.
    fn records(&self) -> u64 {
        self.b.state.records
    }

    /// What happened to the records since the last call: ("copy", from,
    /// to) and ("join", into, froms).
    fn state_ops(&mut self) -> Vec<(&'static str, u64, Vec<u64>)> {
        use thin_dsl::builder::StateOp;
        let row = |op: StateOp| match op {
            StateOp::Copy { from, to } => ("copy", from, vec![to]),
            StateOp::Join { into, from } => ("join", into, from),
        };
        std::mem::take(&mut self.b.state_ops)
            .into_iter()
            .map(row)
            .collect()
    }

    /// `b.emit(i)`: its row, or None when ablated.
    fn emit(&mut self, i: PyRef<'_, Ins>, abl: Vec<String>, loc: u32) -> PyResult<Option<u64>> {
        let ins = (**i.instr()).clone();
        ok(self.b.emit_at(ins, &abl, loc))
    }

    /// Emits what a family or the maker made (an access, a split, a DMA).
    fn emit_made(&mut self, m: PyRef<'_, DslMade>, abl: Vec<String>) -> PyResult<Option<u64>> {
        ok(self.b.emit_family(m.f.clone(), &abl))
    }

    fn wave_index(&mut self, r: (i32, char, u32, u32), nwaves: u32, by_wave: bool) {
        self.b.wave_index(reg(r), nwaves, by_wave);
    }

    /// The wave index: (register, nwaves, by_wave).
    fn wave_info(&self) -> WaveInfo {
        (
            self.b.wave_reg.map(|r| super::convert::reg_out(&r)),
            self.b.nwaves,
            self.b.by_wave,
        )
    }

    /// The rows' ids in stream order.
    fn row_ids(&self) -> Vec<u64> {
        self.b.rows.iter().map(|r| r.id).collect()
    }

    /// `b.wait_on_use(regs, comment)`.
    fn wait_on_use(&mut self, regs: Vec<(i32, char, u32, u32)>, comment: &str) {
        let regs: Vec<_> = regs.into_iter().map(reg).collect();
        self.b.wait_on_use(&regs, comment);
    }

    /// The snapshot of the kernel.
    fn snapshot(&mut self) -> PyResult<crate::ir::Snap> {
        Ok(crate::ir::Snap::of(ok(self.b.snapshot())?))
    }
}
