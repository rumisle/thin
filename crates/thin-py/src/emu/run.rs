//! Machines and clusters.

use std::collections::BTreeMap;

use numpy::PyArray1;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use super::memory::EmuMemory;
use super::{emu_error, tables};
use crate::emu_tags::EmuGemm;
use thin_emu::api::{self, Kernel};
use thin_emu::gemm::GemmHooks;
use thin_emu::machine::cluster::{Cluster, Options, Staged};
use thin_emu::machine::run::Machine;
use thin_emu::machine::wave::Wave;
use thin_emu::machine::Reports;
use thin_emu::tags::Plain;
use thin_emu::Program;

/// A target by name.
fn target_of(py: Python<'_>, name: &str) -> PyResult<&'static thin_target::Target> {
    thin_targets::get(name).map_err(|e| crate::target::target_error(py, e))
}

/// A program read off a `.s` for a target.
fn read(py: Python<'_>, text: &str, t: &thin_target::Target) -> PyResult<Program> {
    thin_emu::read_program(text, t).map_err(|e| emu_error(py, e))
}

/// A program from the decoded encoding's JSON at a target's W.
fn decoded(py: Python<'_>, json: &str, t: &thin_target::Target) -> PyResult<Program> {
    api::program_of(json, api::facts(t).w).map_err(|e| emu_error(py, e))
}

fn machine(
    py: Python<'_>,
    p: Program,
    t: &thin_target::Target,
    counts: (usize, usize, usize),
) -> PyResult<EmuMachine> {
    let inner = Machine::new(p, api::facts(t), counts.0, counts.1, counts.2)
        .map_err(|e| emu_error(py, e))?;
    Ok(EmuMachine { inner, serial: 0 })
}

/// A machine's reports as a dict: stats, banks, vmw, ldsu.
fn reports<'py>(py: Python<'py>, r: &Reports) -> PyResult<Bound<'py, PyDict>> {
    let s = &r.stats;
    let d = PyDict::new(py);
    let stats = [
        ("ins", s.ins),
        ("matrix", s.matrix),
        ("salu", s.salu),
        ("valu", s.valu),
        ("lds_write_bytes", s.lds_write_bytes),
        ("ds_read_bytes", s.ds_read_bytes),
        ("vmem_load_bytes", s.vmem_load_bytes),
        ("store_bytes", s.store_bytes),
        ("wave_ins_max", s.wave_ins_max),
    ];
    d.set_item("stats", stats.into_iter().collect::<BTreeMap<_, _>>())?;
    d.set_item("banks", r.banks.clone())?;
    d.set_item("vmw", r.vmw.clone())?;
    let ldsu: BTreeMap<usize, [u64; 2]> = r.ldsu.clone();
    d.set_item("ldsu", ldsu)?;
    Ok(d)
}

/// A wave as host values: (wid, V [nv·W], S, exec, vcc, scc, m0, pc, done).
type WaveRow<'py> = (
    usize,
    Bound<'py, PyArray1<u32>>,
    Vec<u32>,
    u64,
    u64,
    bool,
    u32,
    usize,
    bool,
);

fn wave_row<'py>(py: Python<'py>, w: &Wave) -> WaveRow<'py> {
    (
        w.wid,
        PyArray1::from_slice(py, &w.v),
        w.s.clone(),
        w.exec,
        w.vcc,
        w.scc,
        w.m0,
        w.pc,
        w.done,
    )
}

/// One rank's interpreter of one kernel.
#[pyclass(module = "thin._native", unsendable)]
struct EmuMachine {
    inner: Machine,
    serial: usize,
}

#[pymethods]
impl EmuMachine {
    /// A machine of a `.s`: counts (nv, na, lds bytes).
    #[staticmethod]
    fn from_text(
        py: Python<'_>,
        text: &str,
        target: &str,
        counts: (usize, usize, usize),
    ) -> PyResult<EmuMachine> {
        let t = target_of(py, target)?;
        machine(py, read(py, text, t)?, t, counts)
    }

    /// A machine of a decoded program (decoded JSON).
    #[staticmethod]
    fn from_json(
        py: Python<'_>,
        json: &str,
        target: &str,
        counts: (usize, usize, usize),
    ) -> PyResult<EmuMachine> {
        let t = target_of(py, target)?;
        machine(py, decoded(py, json, t)?, t, counts)
    }

    /// Sets sum mode, the IPC policy, the rank and tracing.
    fn configure(&mut self, sum_records: bool, ipc_policy: bool, rank: usize, trace: bool) {
        self.inner.sum_records = sum_records;
        self.inner.ipc_policy = ipc_policy;
        self.inner.rank = rank;
        self.inner.trace = trace;
    }

    /// Runs one workgroup on the single-machine runner; its waves after.
    fn run_wg<'py>(
        &mut self,
        py: Python<'py>,
        mem: &mut EmuMemory,
        kernarg: u64,
        wgid: Vec<u32>,
        waves: usize,
    ) -> PyResult<Vec<WaveRow<'py>>> {
        self.serial += 1;
        let (ws, _) = self
            .inner
            .run_wg(
                &mut mem.inner,
                &mut tables(),
                kernarg,
                thin_emu::WorkgroupId::try_from(wgid).map_err(|e| emu_error(py, e))?,
                waves,
                self.serial,
                None,
            )
            .map_err(|e| emu_error(py, e))?;
        Ok(ws.iter().map(|w| wave_row(py, w)).collect())
    }

    fn reports<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        reports(py, &self.inner.reports)
    }

    fn user_sgprs(&self) -> u32 {
        self.inner.wg_sgpr
    }
}

/// Several ranks' launches over one memory.
#[pyclass(module = "thin._native", unsendable)]
struct EmuCluster {
    inner: Cluster,
}

/// The cluster's options as host values, normalised by the caller.
#[derive(FromPyObject)]
struct HostOptions {
    resident: Vec<usize>,
    quantum: u64,
    order: Vec<String>,
    skew: Vec<u64>,
    speed: Vec<f64>,
    wg_speed: BTreeMap<u32, f64>,
    jitter: Option<u64>,
    wave_jitter: Option<u64>,
    stall_rounds: u64,
    lockstep: bool,
    hog: Option<Vec<usize>>,
    hog_stall: u64,
    carry: Vec<(String, Vec<u8>, Vec<i64>)>,
    max_ins: u64,
}

fn options(py: Python<'_>, o: HostOptions) -> PyResult<Options> {
    let order = o
        .order
        .iter()
        .map(|t| api::order(t))
        .collect::<Result<_, _>>();
    let carry = o
        .carry
        .into_iter()
        .map(|(m, d, t)| api::carry(&m, d, t))
        .collect::<Result<_, _>>();
    Ok(Options {
        resident: o.resident,
        quantum: o.quantum,
        order: order.map_err(|e| emu_error(py, e))?,
        skew: o.skew,
        speed: o.speed,
        wg_speed: o.wg_speed,
        jitter: o.jitter,
        wave_jitter: o.wave_jitter,
        stall_rounds: o.stall_rounds,
        lockstep: o.lockstep,
        hog: o.hog,
        hog_stall: o.hog_stall,
        carry: carry.map_err(|e| emu_error(py, e))?,
        max_ins: o.max_ins,
    })
}

#[pymethods]
impl EmuCluster {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        text: &str,
        target: &str,
        counts: (usize, usize, usize, usize),
        launches: Vec<Vec<u64>>,
        grid: (u32, u32),
        opts: HostOptions,
        flags: (bool, bool),
    ) -> PyResult<EmuCluster> {
        let t = target_of(py, target)?;
        let p = read(py, text, t)?;
        let (nv, na, lds, wgsize) = counts;
        let kernel = Kernel {
            program: p,
            target: t,
            nv,
            na,
            lds,
            wgsize,
        };
        let o = options(py, opts)?;
        let inner = api::cluster(&kernel, launches, grid, o, flags.0, flags.1)
            .map_err(|e| emu_error(py, e))?;
        Ok(EmuCluster { inner })
    }

    /// Runs every rank; `staged` (rank, launch, buffer, offset, bytes)
    /// are written as that launch starts; a GEMM checker is lent to the
    /// run and comes back with its results.
    #[pyo3(signature = (mem, staged, gemm=None))]
    fn run(
        &mut self,
        py: Python<'_>,
        mem: &mut EmuMemory,
        staged: Vec<(usize, usize, usize, u64, Vec<u8>)>,
        gemm: Option<PyRefMut<'_, EmuGemm>>,
    ) -> PyResult<()> {
        let mut by: BTreeMap<(usize, usize), Vec<Staged>> = BTreeMap::new();
        staged.into_iter().for_each(|(r, l, buffer, offset, data)| {
            by.entry((r, l)).or_default().push(Staged {
                buffer,
                offset,
                data,
            });
        });
        let lent = gemm.as_ref().and_then(|g| g.inner.clone());
        if let Some(g) = lent {
            self.inner.checker = Box::new(g);
        }
        let result = api::run(&mut self.inner, &mut mem.inner, &mut tables(), by);
        let mut back = std::mem::replace(&mut self.inner.checker, Box::new(Plain));
        let hooks = back
            .as_gemm()
            .and_then(|a| a.downcast_mut::<GemmHooks>())
            .cloned();
        if let Some(mut g) = gemm {
            g.inner = hooks.or(g.inner.take());
        }
        result.map_err(|e| emu_error(py, e))
    }

    fn reports<'py>(&self, py: Python<'py>, rank: usize) -> PyResult<Option<Bound<'py, PyDict>>> {
        self.inner
            .machines
            .get(rank)
            .map(|m| reports(py, &m.reports))
            .transpose()
    }

    /// The machines' instruction texts with their lines and comments.
    fn texts(&self) -> Vec<(String, usize, String)> {
        let p = self
            .inner
            .machines
            .first()
            .map(|m| &m.program.ins[..])
            .unwrap_or(&[]);
        p.iter()
            .map(|i| (i.text.clone(), i.line, i.comment.clone()))
            .collect()
    }

    fn rounds(&self) -> u64 {
        self.inner.rounds
    }
}

/// Adds this module's classes.
pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<EmuMachine>()?;
    m.add_class::<EmuCluster>()
}
