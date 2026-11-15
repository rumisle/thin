//! Memory and buffers.

use numpy::{PyArray1, PyReadonlyArrayDyn};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use super::emu_error;
use crate::input;
use thin_emu::machine::memory::{Memory, Meta};

/// The memory of a run, shared by every rank.
#[pyclass(module = "thin._native")]
pub(crate) struct EmuMemory {
    /// The core memory.
    pub(crate) inner: Memory,
}

#[pymethods]
impl EmuMemory {
    #[new]
    fn new(straddle: u64) -> EmuMemory {
        EmuMemory {
            inner: Memory::new(straddle),
        }
    }

    /// Allocates a buffer; returns its index.
    #[allow(clippy::too_many_arguments)]
    fn alloc(
        &mut self,
        py: Python<'_>,
        name: &str,
        size: u64,
        tagged: bool,
        fill: u8,
        ipc: bool,
        cached: bool,
        uncached: bool,
    ) -> PyResult<usize> {
        let meta = Meta {
            ipc,
            cached,
            uncached,
        };
        self.inner
            .alloc(name, size, tagged, fill, meta)
            .map_err(|e| emu_error(py, e))
    }

    fn release(&mut self, index: usize) {
        self.inner.release(index);
    }

    fn by_name(&self, name: &str) -> Option<usize> {
        self.inner.by_name(name)
    }

    fn count(&self) -> usize {
        self.inner.bufs.len()
    }

    /// (name, base, size, version, released) of a buffer.
    fn info(&self, index: usize) -> Option<(String, u64, u64, u64, bool)> {
        let b = self.inner.bufs.get(index)?;
        Some((b.name.clone(), b.base, b.size(), b.version, b.released))
    }

    fn versions(&self) -> Vec<u64> {
        self.inner.bufs.iter().map(|b| b.version).collect()
    }

    fn resolve(&self, py: Python<'_>, addr: u64) -> PyResult<(usize, u64)> {
        self.inner.locate(addr).map_err(|e| emu_error(py, e))
    }

    fn resolve_many(&self, addrs: Vec<u64>) -> Vec<(i64, u64)> {
        self.inner.resolve_many(&addrs)
    }

    fn read<'py>(
        &self,
        py: Python<'py>,
        index: usize,
        off: usize,
        n: usize,
    ) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.inner.read(index, off, n))
    }

    fn write(&mut self, index: usize, off: usize, data: &[u8]) {
        self.inner.write(index, off, data);
    }

    /// A copy of every byte's tag.
    fn tags<'py>(&self, py: Python<'py>, index: usize) -> Bound<'py, PyArray1<i64>> {
        let t = self.inner.bufs.get(index).map_or(&[][..], |b| &b.tags[..]);
        PyArray1::from_slice(py, t)
    }

    /// Replaces the tags of these offsets.
    fn set_tags(
        &mut self,
        index: usize,
        offsets: PyReadonlyArrayDyn<'_, i64>,
        tags: PyReadonlyArrayDyn<'_, i64>,
    ) -> PyResult<()> {
        let (o, t) = (input(&offsets)?, input(&tags)?);
        if let Some(b) = self.inner.bufs.get_mut(index) {
            o.iter().zip(t).for_each(|(off, tag)| {
                if let Some(slot) = b.tags.get_mut(*off as usize) {
                    *slot = *tag;
                }
            });
        }
        Ok(())
    }

    fn track_traffic(&mut self) {
        self.inner.track_traffic();
    }

    /// The stale reads served (flight events).
    fn events(&self) -> Vec<String> {
        self.inner.flight.events.clone()
    }

    fn end_launch(&mut self, rank: usize) {
        self.inner.flight.end_launch(rank);
    }

    fn scope_model(&self) -> bool {
        self.inner.scope.is_some()
    }

    fn ipc(&self) -> bool {
        self.inner.ipc()
    }
}

/// Adds this module's classes.
pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<EmuMemory>()
}
