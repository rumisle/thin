//! The build options record, as the native consumers
//! read it.

/// The build options; none is on in a kernel that ships. The host keeps
/// the current record and its overrides; a kernel takes a copy when it is
/// made.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Options {
    /// Emit loop bodies twice and audit the placement.
    pub audit: bool,
    /// Skip the dead-writes check.
    pub keep_dead: bool,
    /// Wait for every counter at 0 before each barrier.
    pub safe_barrier: bool,
    /// Refuse forced counts.
    pub waits_unique: bool,
    /// Place every count at 0.
    pub safe_waits: bool,
    /// Use lane regions' EXEC tokens in the analysis.
    pub regions: bool,
    /// The values a pressure error lists.
    pub peak_list: u32,
    /// Report the values at the peak and the partial-rule values.
    pub peak_report: bool,
    /// Report live dwords along the program.
    pub pressure: bool,
    /// Let one wave's program order separate LDS regions in the LDS proof.
    pub lds_wave_order: bool,
}

impl Default for Options {
    /// The defaults of table.
    fn default() -> Options {
        Options {
            audit: false,
            keep_dead: false,
            safe_barrier: false,
            waits_unique: false,
            safe_waits: false,
            regions: true,
            peak_list: 24,
            peak_report: false,
            pressure: false,
            lds_wave_order: true,
        }
    }
}
