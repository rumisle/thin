//! The kernel as the snapshot reads it.

use std::sync::Arc;
use thin_ir::{Gather, GatherFile, GatherRegion, GatherValue, Instr, Snapshot};

use crate::builder::Builder;
use crate::error::Result;

impl Builder {
    /// What the snapshot gathers of the kernel.
    pub fn gather(&self) -> Gather {
        let mut g = Gather::default();
        for v in &self.values {
            g.values.push(GatherValue {
                file: v.file,
                n: v.n,
                align: v.align,
                name: v.name.to_string(),
                pin: v.pin,
                same_as: v.same_as.map(|s| s as u32),
                index: v.index,
                parts: Vec::new(),
                register: Some(v.first),
            });
        }
        for f in &self.files {
            g.files.push(GatherFile {
                letter: f.letter,
                lanes: f.lanes.clone(),
                entered: f.entered.iter().map(|s| s.to_string()).collect(),
                pinned: f.pinned.iter().map(|s| s.to_string()).collect(),
                placed: f.placed,
            });
        }
        if let Some(l) = &self.layout {
            for (k, r) in l.regions.iter().enumerate() {
                g.regions.push(GatherRegion {
                    key: k as u64,
                    base: r.base,
                    size: r.size,
                    name: r.name.clone(),
                    parent: r.parent.map(|(p, _)| l.regions[p].name.clone()),
                    on_layout: true,
                    tags: l.tags(k),
                });
            }
        }
        g.wave_reg = self.wave_reg;
        g.nwaves = self.nwaves;
        g.by_wave = self.by_wave;
        g
    }

    /// The snapshot of the kernel's stream (`program(kernel)`).
    pub fn snapshot(&mut self) -> Result<Snapshot> {
        self.settle()?;
        let instrs: Arc<[Arc<Instr>]> = self.rows.iter().map(|r| r.ins.clone()).collect();
        Ok(Snapshot::of_kernel(self.t, &self.gather(), instrs)?)
    }
}
