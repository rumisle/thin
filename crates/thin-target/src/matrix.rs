//! Hooks of for the emulator and the kernel library: the
//! matrix forms and their summation model, and the library's hooks.

/// An element type of a matrix operand.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Elem {
    /// IEEE binary16.
    F16,
    /// bfloat16.
    Bf16,
    /// IEEE binary32.
    F32,
    /// A 32-bit integer.
    I32,
    /// An 8-bit integer, unsigned or signed by the op's modifiers.
    Iu8,
    /// A 4-bit integer, unsigned or signed by the op's modifiers.
    Iu4,
    /// The 8-bit float with 4 exponent bits (e4m3).
    Fp8,
    /// The 8-bit float with 5 exponent bits (e5m2).
    Bf8,
}

impl Elem {
    /// Its bits.
    pub fn bits(self) -> u32 {
        match self {
            Elem::F16 | Elem::Bf16 => 16,
            Elem::F32 | Elem::I32 => 32,
            Elem::Iu8 | Elem::Fp8 | Elem::Bf8 => 8,
            Elem::Iu4 => 4,
        }
    }

    /// Whether it is an integer type.
    pub fn integer(self) -> bool {
        matches!(self, Elem::I32 | Elem::Iu8 | Elem::Iu4)
    }
}

/// How a matrix op sums its products.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Summation {
    /// Products exact in binary64, summed in k order with one rounding
    /// per add in binary64, then C, rounded once to D's type; integer
    /// forms exact, wrapped to 32 bits.
    Ordered,
    /// The measured grouped model: products in groups of
    /// `group_half` (f16, bf16 inputs) or `group_byte` (fp8, bf8) consecutive
    /// k, in the A operand's half-interleaved order; each group aligned to
    /// its largest product's reference (the inputs' exponent sum), the
    /// accumulator (from C) and the products floored `fraction` bits below
    /// it, summed exactly and rounded to D's format after every group (f32
    /// and f16 ties to even, bf16 toward zero).
    Grouped {
        /// G for 16-bit inputs.
        group_half: u32,
        /// G for 8-bit inputs.
        group_byte: u32,
        /// The fraction bits kept below a group's top reference exponent.
        fraction: i32,
    },
}

/// One matrix form: D (16×16) = A (16×K) · B (K×16) + C, with the
/// lane layout of a 32-lane wave in two halves: lane
/// l = r + 16h holds row r of A and column r of B, element e of half h
/// being k = (K/2)·h + e; C and D's lane l holds column r, element e
/// being row 8h + e.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MatrixForm {
    /// A's element type.
    pub a: Elem,
    /// B's element type.
    pub b: Elem,
    /// C's and D's element type.
    pub d: Elem,
    /// The reduction depth K.
    pub k: u32,
    /// The summation model.
    pub sum: Summation,
    /// Whether the model is provisional (not measured on hardware).
    pub provisional: bool,
}

impl MatrixForm {
    /// Dwords per lane of A (`a`), B or C/D.
    pub fn dwords(&self, operand: char) -> u32 {
        match operand {
            'a' => self.k / 2 * self.a.bits() / 32,
            'b' => self.k / 2 * self.b.bits() / 32,
            _ => 8 * self.d.bits() / 32,
        }
    }
}

/// How a clock is read into a scalar pair.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClockRead {
    /// One op writes the pair asynchronously, on its counter; its
    /// immediate operand, if any, follows the destination.
    Pair {
        /// The op.
        op: &'static str,
        /// Its immediate (a message id), if it takes one.
        imm: Option<u32>,
    },
    /// One op per 32-bit half, each synchronous, with the half's immediate
    /// (a hardware-register selector). The high half is read on both sides
    /// of the low one, so a carry between the reads is seen.
    Halves {
        /// The op.
        op: &'static str,
        /// The low half's immediate.
        lo: u32,
        /// The high half's immediate.
        hi: u32,
    },
}

impl ClockRead {
    /// The text the library reads: `pair OP [IMM]` or `halves OP LO HI`.
    pub fn text(&self) -> String {
        match *self {
            ClockRead::Pair { op, imm: None } => format!("pair {op}"),
            ClockRead::Pair { op, imm: Some(i) } => format!("pair {op} {i}"),
            ClockRead::Halves { op, lo, hi } => format!("halves {op} {lo} {hi}"),
        }
    }
}

/// The kernel library's hooks: each the
/// target's value, or none where it supplies none yet ("not yet").
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LibHooks {
    /// The lanes a wave is counted by when a library sizes per-wave memory.
    pub lane_granule: Option<u32>,
    /// The read of the core's cycle counter.
    pub core_clock: Option<ClockRead>,
    /// The read of the constant-rate clock.
    pub real_clock: Option<ClockRead>,
    /// The constant-rate clock's MHz.
    pub real_clock_mhz: Option<u32>,
    /// The scope that makes a memory op coherent across workgroups.
    pub coherent: Option<&'static str>,
    /// The scope of memory other devices map (uncached).
    pub system: Option<&'static str>,
    /// The 32-bit global atomic add that returns the old value.
    pub returning_atomic_add: Option<&'static str>,
    /// The temporal hint an atomic returns its old value with.
    pub atomic_return: Option<&'static str>,
    /// The non-carry scalar u32 add (a poll's round counter).
    pub scalar_add: Option<&'static str>,
    /// The right shift of a lane mask.
    pub mask_shr: Option<&'static str>,
    /// The short sleep.
    pub sleep: Option<&'static str>,
}

impl LibHooks {
    /// The hooks by name, in order.
    pub fn named(&self) -> [(&'static str, Option<String>); 11] {
        let s = |v: Option<&str>| v.map(str::to_owned);
        [
            ("lane_granule", self.lane_granule.map(|g| g.to_string())),
            ("core_clock", self.core_clock.map(|c| c.text())),
            ("real_clock", self.real_clock.map(|c| c.text())),
            ("real_clock_mhz", self.real_clock_mhz.map(|m| m.to_string())),
            ("coherent", s(self.coherent)),
            ("system", s(self.system)),
            ("returning_atomic_add", s(self.returning_atomic_add)),
            ("atomic_return", s(self.atomic_return)),
            ("scalar_add", s(self.scalar_add)),
            ("mask_shr", s(self.mask_shr)),
            ("sleep", s(self.sleep)),
        ]
    }
}

/// The instruction-cache invalidate a timing harness launches: the
/// instruction, the no-ops it needs after it, and how many workgroups per
/// compute unit reach every unit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IcacheInvalidate {
    /// The invalidate instruction.
    pub instruction: &'static str,
    /// The no-op that follows it.
    pub nop: &'static str,
    /// How many of them.
    pub nops: u32,
    /// Workgroups per compute unit.
    pub per_cu: u32,
}

/// A memory access's coherence scope, narrowest first.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Scope {
    /// The wave.
    Wave,
    /// The workgroup.
    Workgroup,
    /// The agent (the device).
    Agent,
    /// The system.
    System,
}

/// How a target's memory instructions name their scope and temporal policy
/// as data the emulator reads.
///
/// An access's scope code is the sum of its `code` fields' values, each
/// shifted (a flag field reads 1, an absent one 0); `plain` and `atomic`
/// give the scope of each code. For example a `scope` field of four values
/// is `code: [("scope", 0)]`, and two scope bits are
/// `code: [("sc0", 0), ("sc1", 1)]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScopeRule {
    /// The fields the scope code is read from, with their shifts.
    pub code: &'static [(&'static str, u32)],
    /// The scope of a load or store per code.
    pub plain: &'static [Scope],
    /// The scope of an atomic per code.
    pub atomic: &'static [Scope],
    /// The code from which a cache write-back reaches the L2 (none: every
    /// write-back does).
    pub writeback_l2: Option<u32>,
    /// The code from which a cache invalidate reaches the L2.
    pub invalidate_l2: u32,
    /// Whether write-backs and invalidates count on the vector-memory
    /// counters (a write-back as a store, an invalidate as a load).
    pub maintenance_counted: bool,
    /// The temporal-hint field and, for loads (`[0]`) and stores (`[1]`),
    /// the bit set of its values that are non-temporal; an empty set with
    /// a flag field reads the flag.
    pub nontemporal: (&'static str, [u8; 2]),
    /// The narrowest scope at which an access may touch another rank's
    /// (IPC) memory, besides a non-temporal load or store.
    pub ipc: Scope,
    /// That policy as the IPC refusal names it.
    pub ipc_text: &'static str,
    /// The flag field that makes an atomic return its old value; none when
    /// every returning form returns.
    pub returns: Option<&'static str>,
}

impl ScopeRule {
    /// The scope code of an access whose fields `field` reads (a value,
    /// or none when absent).
    pub fn code_of(&self, field: &dyn Fn(&str) -> Option<i64>) -> u32 {
        self.code
            .iter()
            .map(|(f, shift)| (field(f).unwrap_or(0) as u32) << shift)
            .sum()
    }

    /// The scope of an access of that code.
    pub fn scope(&self, code: u32, atomic: bool) -> Scope {
        let table = if atomic { self.atomic } else { self.plain };
        table
            .get(code as usize)
            .or(table.last())
            .copied()
            .unwrap_or(Scope::System)
    }

    /// Whether a load (`store` false) or store with these fields is
    /// non-temporal.
    pub fn nontemporal(&self, field: &dyn Fn(&str) -> Option<i64>, store: bool) -> bool {
        let (name, hints) = self.nontemporal;
        let set = hints[usize::from(store)];
        match field(name) {
            None => false,
            Some(v) if set == 0 => v != 0,
            Some(v) => set >> (v & 7) & 1 != 0,
        }
    }
}
