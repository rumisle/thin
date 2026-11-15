//! The LDS layout and addresses.

mod access;
mod addr;
pub mod layout;

pub use access::{access, wait_region, Access, Split, SplitKind};
pub use addr::{split, sum, Address, Off, Pointer};
pub use layout::{elem_of_name, elem_of_type, Layout, Region, RegionArgs, Xor};

use crate::builder::Builder;
use crate::error::{mem, Result};

impl Builder {
    /// `b.lds(budget)`.
    pub fn lds(&mut self, budget: Option<i128>) -> Result<()> {
        if self.layout.is_some() {
            return mem("lds(): the kernel's layout is made already (a kernel has one)");
        }
        self.layout = Some(Layout::new(
            self.id,
            budget,
            u64::from(self.t.definition().lds_max),
        )?);
        Ok(())
    }

    /// The layout, frozen (its first base read, `L.size`, the table).
    pub fn frozen_layout(&mut self) -> Result<&Layout> {
        let l = self.layout.as_mut().expect("a layout");
        l.freeze()?;
        Ok(self.layout.as_ref().expect("a layout"))
    }
}

/// The names an op on a region of the Builder's layout answers to.
pub fn tags_of(b: &Builder, region: u64) -> Vec<String> {
    b.layout
        .as_ref()
        .map_or_else(Vec::new, |l| match l.regions.get(region as usize) {
            Some(_) => l.tags(region as usize),
            None => Vec::new(),
        })
}
