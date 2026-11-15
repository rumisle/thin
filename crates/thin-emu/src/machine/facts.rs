//! The target facts the core reads, each read off the
//! target's definition and hooks: nothing here names a target.

use thin_target::{Flags, MatrixForm, Scope, ScopeRule, Target, WaitText};

/// The emulator's counters, by the ops they count: vector-memory
/// loads, vector-memory stores, LDS ops, scalar memory. A target counter
/// covers one or more of them.
pub mod role {
    /// Vector-memory loads, returning atomics and cache invalidates.
    pub const LOAD: usize = 0;
    /// Vector-memory stores, non-returning atomics and write-backs.
    pub const STORE: usize = 1;
    /// LDS ops.
    pub const DS: usize = 2;
    /// Scalar memory (unordered: only a count of 0 retires it).
    pub const SMEM: usize = 3;
}

/// Where the workgroup ids go at dispatch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ids {
    /// They follow the user SGPRs.
    AfterUser,
    /// In the target's named registers: x in the first (ttmp9), y and z
    /// in the halves of the second (ttmp7).
    Ttmp,
}

/// A wait instruction: its canonical op, how its line gives its counts,
/// and per count field the emulator counters it waits on.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Wait {
    /// The canonical op name.
    pub canon: String,
    /// Whether its count is its one operand (else named fields).
    pub bare: bool,
    /// (count field, the counters it waits on).
    pub counts: Vec<(String, Vec<usize>)>,
}

/// An LDS access's bank rule: banks and each lane's phase.
pub type BankRule = (u64, Vec<u32>);

/// The facts of one target.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Facts {
    /// The target's name.
    pub name: String,
    /// W, the lanes of a wave and the lane-mask width.
    pub w: usize,
    /// SGPRs.
    pub ns: usize,
    /// Whether there is an `a` file.
    pub has_a: bool,
    /// Whether `fp8` / `bf8` ops use FNUZ (`e4m3fnuz` / `e5m2fnuz`).
    pub fnuz_fp8: bool,
    /// The counter vector-memory stores count on: their own, or the
    /// loads' when one counter counts both.
    pub store_counter: usize,
    /// The wait instructions.
    pub waits: Vec<Wait>,
    /// The canonical op of the instruction a wave waits at the workgroup
    /// barrier with (the last of the target's barrier sequence).
    pub barrier: String,
    /// The canonical ops the machine runs as no-ops: the barrier
    /// sequence's others and the target's issue hint.
    pub no_ops: Vec<String>,
    /// Where the workgroup ids go.
    pub ids: Ids,
    /// L2s per rank.
    pub xcds: usize,
    /// How accesses name their scope and temporal policy.
    pub scope: ScopeRule,
    /// The clocks the target reads (its core and real clock hooks).
    pub clocks: Clocks,
    /// The LDS bank rules per (read, bytes a lane), where the target has
    /// them: the bank model reports conflicts by them.
    pub banks: Vec<((bool, u32), BankRule)>,
    /// The matrix forms the machine computes, by mnemonic.
    pub matrix: Vec<(String, MatrixForm)>,
    /// Per emulator counter, whether its ops' register results land at a
    /// wait.
    pub lands: [bool; 4],
    /// Per emulator counter, the count field of the target counter covering it
    /// (empty for none), for messages.
    pub counter_names: [String; 4],
}

/// Every access at system scope, no temporal policy: for a target with no
/// scope rule.
const NO_SCOPE: ScopeRule = ScopeRule {
    code: &[],
    plain: &[Scope::System],
    atomic: &[Scope::System],
    writeback_l2: None,
    invalidate_l2: 0,
    maintenance_counted: false,
    nontemporal: ("", [0, 0]),
    ipc: Scope::Wave,
    ipc_text: "",
    returns: None,
};

/// The emulator counters a target counter covers, by its units: its first
/// VMEM counter counts loads (and stores, when it is the only one), the
/// second stores; DS and SMEM their ops.
fn roles_of(def: &thin_target::Definition, c: &thin_target::Counter) -> Vec<usize> {
    let vmem: Vec<&str> = (def.counters.iter())
        .filter(|c| c.units.has(Flags::VMEM))
        .map(|c| c.name)
        .collect();
    let mut roles = Vec::new();
    match vmem.iter().position(|n| *n == c.name) {
        Some(0) if vmem.len() == 1 => roles.extend([role::LOAD, role::STORE]),
        Some(0) => roles.push(role::LOAD),
        Some(_) => roles.push(role::STORE),
        None => {}
    }
    if c.units.has(Flags::DS) {
        roles.push(role::DS);
    }
    if c.units.has(Flags::SMEM) {
        roles.push(role::SMEM);
    }
    roles
}

impl Facts {
    /// The facts of a target, read off its definition: W,
    /// its files, its counters and wait instructions, where its workgroup
    /// ids go, its L2s, its scope rule, its LDS bank rules, its clocks and
    /// its matrix forms.
    pub fn of(target: &Target) -> Facts {
        let def = target.definition();
        let hooks = target.hooks();
        let mut matrix: Vec<(String, MatrixForm)> = Vec::new();
        for f in target.forms() {
            if let Some(m) = hooks.matrix_form(&f.name) {
                if !matrix.iter().any(|(n, _)| *n == f.name) {
                    matrix.push((f.name.clone(), m));
                }
            }
        }
        let mut waits = Vec::new();
        for wi in &def.waits {
            let Some(form) = target.forms().iter().find(|f| f.name == wi.mnemonic) else {
                continue;
            };
            let counts = (wi.fields.iter())
                .filter_map(|field| {
                    let c = def.counters.iter().find(|c| c.field == *field)?;
                    Some(((*field).to_owned(), roles_of(def, c)))
                })
                .collect::<Vec<_>>();
            // A wait whose fields count no memory (an ALU dependency wait)
            // has no counts: the machine runs it as a no-op.
            waits.push(Wait {
                canon: form.canon.clone(),
                bare: wi.text == WaitText::Bare,
                counts,
            });
        }
        let mut lands = [false; 4];
        let mut counter_names: [String; 4] = Default::default();
        let mut store_counter = role::STORE;
        for c in &def.counters {
            let roles = roles_of(def, c);
            if roles.contains(&role::LOAD) && roles.contains(&role::STORE) {
                store_counter = role::LOAD;
            }
            for r in roles {
                lands[r] = c.lands == thin_target::Lands::Wait;
                counter_names[r] = c.field.to_owned();
            }
        }
        let banks = (0..2)
            .flat_map(|read| [1, 2, 4, 8, 12, 16].map(move |n| (read == 1, n)))
            .filter_map(|(read, n)| Some(((read, n), hooks.lds_banks(read, n)?)))
            .collect();
        let canon_of = |mnemonic: &str| {
            (target.forms().iter())
                .find(|f| f.name == mnemonic)
                .map(|f| f.canon.clone())
        };
        let barrier = (def.barrier.last())
            .and_then(|(m, _)| canon_of(m))
            .unwrap_or_default();
        let lead = def.barrier.len().saturating_sub(1);
        let no_ops = (def.barrier[..lead].iter().map(|(m, _)| *m))
            .chain(def.hazards.delay_hint)
            .filter_map(canon_of)
            .collect();
        let ns = target
            .file('s')
            .map_or(def.chip.sgprs, |f| f.size.max(def.chip.sgprs));
        Facts {
            name: target.name().to_owned(),
            w: def.wave as usize,
            ns: ns as usize,
            has_a: target.file('a').is_some(),
            fnuz_fp8: hooks.fnuz_fp8(),
            store_counter,
            waits,
            barrier,
            no_ops,
            ids: if def.chip.wgids.is_empty() {
                Ids::AfterUser
            } else {
                Ids::Ttmp
            },
            xcds: def.chip.xcds.max(1) as usize,
            scope: hooks.scope_rule().unwrap_or(NO_SCOPE),
            clocks: Clocks::of(&hooks.lib_hooks()),
            banks,
            matrix,
            lands,
            counter_names,
        }
    }

    /// The decoded register of a named register: (name,
    /// part, count), or none for one the core does not model.
    pub fn named_reg(&self, name: &str) -> Option<(&'static str, u32, u32)> {
        // On a 32-lane wave VCC is vcc_lo and vcc_hi is an SGPR of its
        // own; on a 64-lane one VCC is the pair.
        let wide = self.w == 64;
        Some(match name {
            "exec_lo" => ("exec", 0, 1),
            "exec_hi" => ("exec", 1, 1),
            "exec" => ("exec", 0, 2),
            "vcc_lo" => ("vcc", 0, 1),
            "vcc_hi" if !wide => ("vcc", 1, 1),
            "vcc" if wide => ("vcc", 0, 2),
            "m0" => ("m0", 0, 1),
            "scc" => ("scc", 0, 1),
            "null" => ("null", 0, 1),
            "ttmp7" => ("ttmp7", 0, 1),
            "ttmp9" => ("ttmp9", 0, 1),
            _ => return None,
        })
    }

    /// The matrix form of a mnemonic, if the machine computes it.
    pub fn matrix_form(&self, mnemonic: &str) -> Option<&MatrixForm> {
        self.matrix
            .iter()
            .find(|(n, _)| n == mnemonic)
            .map(|(_, m)| m)
    }

    /// The bank rule of an LDS access, if the target has one.
    pub fn bank_rule(&self, read: bool, bytes: u32) -> Option<&BankRule> {
        (self.banks.iter())
            .find(|(k, _)| *k == (read, bytes))
            .map(|(_, r)| r)
    }

    /// The lane mask of every lane.
    pub fn all_lanes(&self) -> u64 {
        if self.w == 64 {
            u64::MAX
        } else {
            (1u64 << self.w) - 1
        }
    }
}

/// How a program reads the clocks: the hardware-register ids of the core
/// cycle counter's halves, and the message id that returns the constant
/// clock.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Clocks {
    /// The `s_getreg` ids of the cycle counter's low and high halves.
    pub cycles: Option<(u32, u32)>,
    /// The `s_sendmsg_rtn` message that returns the constant clock.
    pub realtime: Option<u32>,
    /// Whether `s_memtime` reads the cycle counter.
    pub memtime: bool,
    /// Whether `s_memrealtime` reads the constant clock.
    pub memrealtime: bool,
}

impl Clocks {
    fn of(hooks: &thin_target::LibHooks) -> Clocks {
        use thin_target::ClockRead;
        let mut c = Clocks::default();
        for read in [hooks.core_clock, hooks.real_clock].into_iter().flatten() {
            match read {
                ClockRead::Halves { op, lo, hi } if op.starts_with("s_getreg") => {
                    c.cycles = Some((lo & 63, hi & 63));
                }
                ClockRead::Pair { op, imm: Some(m) } if op.starts_with("s_sendmsg_rtn") => {
                    c.realtime = Some(m);
                }
                ClockRead::Pair {
                    op: "s_memtime",
                    imm: None,
                } => {
                    c.memtime = true;
                }
                ClockRead::Pair {
                    op: "s_memrealtime",
                    imm: None,
                } => {
                    c.memrealtime = true;
                }
                _ => {}
            }
        }
        c
    }
}
