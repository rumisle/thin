//! The target record, how it is built and what generic code asks of it
//!

use std::collections::{BTreeSet, HashMap};

use crate::code_object::{Block, Footer, Resources};
use crate::form::{Form, Pseudo};
use crate::own::own_ops;
use crate::rules::{unit_of, WaitInsn};
use crate::syntax::Syntax;
use crate::vocab::{phys_named, Flags, Mem, Phys, PHYS};
use crate::Error;

/// A register file: (letter, registers per wave, allocation granule,
/// alignment of a 64-bit or wider range).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct File {
    /// `v`, `s` or `a`.
    pub letter: char,
    /// Registers per wave.
    pub size: u32,
    /// The allocation granule in registers.
    pub granule: u32,
    /// The alignment of a 64-bit or wider range.
    pub align64: u32,
}

/// A wait counter.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Counter {
    /// How a wait request names it.
    pub name: &'static str,
    /// Its largest count.
    pub max: u32,
    /// The count field of a wait instruction that holds its count.
    pub field: &'static str,
    /// The units whose ops it counts (VMEM, DS, SMEM).
    pub units: Flags,
    /// When its ops' register results are written in the emulator: at
    /// issue, or once a wait retires the op.
    pub lands: Lands,
}

/// When a counter's ops write their register results.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Lands {
    /// As the op issues.
    Issue,
    /// When a wait retires the op (in issue order; an unordered op only by
    /// a count of 0).
    Wait,
}

/// The operand rules the instruction maker enforces.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Operands {
    /// A VOP3/e64 form takes a 32-bit literal.
    pub vop3_literal: bool,
    /// The most scalar registers plus literals a VALU op reads.
    pub constant_bus: u32,
    /// A VALU write of a scalar register first waits for scalar loads.
    pub valu_sgpr_write_waits_smem: bool,
    /// The bits of a scalar load's immediate offset operand (signed).
    pub smem_offset_bits: u32,
}

/// Sizes and limits.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Chip {
    /// XCDs, each with its own L2.
    pub xcds: u32,
    /// Compute units that hold a workgroup.
    pub cus: u32,
    /// SIMDs per such unit.
    pub simds: u32,
    /// That unit's LDS bytes.
    pub lds: u32,
    /// Vector registers a wave addresses.
    pub vgprs: u32,
    /// A SIMD lane's register file.
    pub vgprs_per_lane: u32,
    /// Numbered scalar registers a kernel may allocate.
    pub sgprs: u32,
    /// The most user SGPRs a descriptor may ask for.
    pub user_sgprs: u32,
    /// The most kernel-argument dwords preloaded into user SGPRs.
    pub preload: u32,
    /// Waves a SIMD holds at once.
    pub wave_slots: u32,
    /// The named registers holding workgroup ids x and y, where they do.
    pub wgids: &'static [&'static str],
}

/// A target's hazard model as data: the passes own the rules' engine
/// and the target's own pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Hazards {
    /// The number of wait-state rules.
    pub rules: usize,
    /// Whether the target has a hazard pass of its own.
    pub pass: bool,
    /// The rules' window in instructions.
    pub window: u32,
    /// The VALU dependency hint's mnemonic, if the target has one: an
    /// instruction that tells the issue logic to wait, not a hazard.
    pub delay_hint: Option<&'static str>,
}

/// What an instruction line encodes to.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Encoded {
    /// Bytes to append.
    Bytes(Vec<u8>),
    /// A branch: its raw word with a zero offset field, and its label.
    Branch {
        /// The word.
        raw: u32,
        /// The label.
        label: String,
    },
}

/// The hooks a target module supplies besides its data: the mnemonic hooks
/// the line syntax, the line encoder and the
/// code-object module.
pub trait Hooks: Syntax + Send + Sync {
    /// The memory class of a mnemonic, or none.
    fn mem_of(&self, mnemonic: &str) -> Option<Mem>;
    /// Whether an atomic with this temporal hint (`th`) is its returning
    /// form, for targets whose returning atomics take the same operands.
    fn returning(&self, mnemonic: &str, th: Option<&str>) -> bool;
    /// The word an alignment pads with: the target's `s_nop 0`.
    fn nop_word(&self) -> u32;
    /// One instruction line, or the refusal's message.
    fn encode(&self, line: &str) -> Result<Encoded, String>;
    /// The ELF e_flags of its code objects.
    fn e_flags(&self) -> u32;
    /// The descriptor's VGPR block size.
    fn granule(&self) -> u32;
    /// The `.amdhsa_*` names its block takes beyond the common set.
    fn directives(&self) -> &[&'static str];
    /// The .s text after a kernel's body.
    fn footer(&self, footer: &Footer) -> Result<String, Error>;
    /// Its own descriptor words from a block.
    fn resources(&self, block: &Block) -> Result<Resources, Error>;
    /// Interface properties without register-allocation policy. Defaults to
    /// the full resources hook for targets that do not separate the two.
    fn interface_resources(&self, block: &Block) -> Result<Resources, Error> {
        self.resources(block)
    }
    /// The matrix form of a mnemonic the emulator computes, or
    /// none.
    fn matrix_form(&self, _mnemonic: &str) -> Option<crate::MatrixForm> {
        None
    }
    /// How its memory instructions name their scope and temporal policy
    /// or none when the emulator's scope model and
    /// IPC policy do not apply to it.
    fn scope_rule(&self) -> Option<crate::ScopeRule> {
        None
    }
    /// The LDS bank rule of a read (`read`) or write of `bytes` a lane:
    /// the banks and each lane's phase, for the emulator's bank-conflict
    /// report; none by default (no report).
    fn lds_banks(&self, _read: bool, _bytes: u32) -> Option<(u64, Vec<u32>)> {
        None
    }
    /// The kernel library's hooks; none by default.
    fn lib_hooks(&self) -> crate::LibHooks {
        crate::LibHooks::default()
    }
    /// The instruction-cache invalidate, or none.
    fn icache_invalidate(&self) -> Option<crate::IcacheInvalidate> {
        None
    }
    /// Whether FP8 (`fp8` / `bf8`) instructions use FNUZ (`e4m3fnuz` / `e5m2fnuz`)
    /// rather than OCP (`e4m3fn` / `e5m2`).
    fn fnuz_fp8(&self) -> bool {
        false
    }
}

/// What a target module supplies, from which the target is built.
pub struct Definition {
    /// The target's name.
    pub name: &'static str,
    /// Its forms, each with id equal to its index.
    pub forms: Vec<Form>,
    /// Its register files.
    pub files: Vec<File>,
    /// The mnemonics its encoder has an opcode for.
    pub encodings: BTreeSet<String>,
    /// Lanes per wave, 32 or 64.
    pub wave: u32,
    /// LDS bytes a workgroup may use.
    pub lds_max: u32,
    /// Its hazard model.
    pub hazards: Hazards,
    /// Its wait counters, ordered.
    pub counters: Vec<Counter>,
    /// The counters' names in the same order.
    pub counter_names: Vec<&'static str>,
    /// Its wait instructions, ordered.
    pub waits: Vec<WaitInsn>,
    /// The matrix and scalar-memory prefixes of `unit_of`.
    pub unit_prefixes: (&'static [&'static str], &'static [&'static str]),
    /// The operand rules.
    pub operands: Operands,
    /// Sizes and limits.
    pub chip: Chip,
    /// (dotted op, mnemonic) pairs where the default spelling does not apply.
    pub spellings: Vec<(&'static str, &'static str)>,
    /// The workgroup barrier sequence: (mnemonic, constant operands).
    pub barrier: Vec<(&'static str, Vec<i64>)>,
    /// The hooks.
    pub hooks: Box<dyn Hooks>,
}

/// A group of counters a wait may name, whatever the target.
pub static GROUPS: [(&str, Flags); 3] = [
    ("vm", Flags::VMEM),
    ("lgkm", Flags(Flags::DS.0 | Flags::SMEM.0)),
    ("smem", Flags::SMEM),
];

/// A target, built from its definition.
pub struct Target {
    def: Definition,
    own: Vec<Form>,
    keys: HashMap<(String, usize, bool), usize>,
    ops: BTreeSet<String>,
    line_fields: Vec<&'static str>,
}

impl std::fmt::Debug for Target {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Target({})", self.def.name)
    }
}

impl Target {
    /// Builds a target: canon, the lookup map, the build-time checks
    /// and the compiler's own operations.
    pub fn build(mut def: Definition) -> Result<Target, Error> {
        check_waits(&def)?;
        let spellings = def.spellings.clone();
        for form in &mut def.forms {
            form.canon = canon_in(&spellings, &form.op);
        }
        let mut keys = HashMap::new();
        for (i, form) in def.forms.iter().enumerate() {
            if form.id != i as i32 {
                return Err(Error::Definition(format!(
                    "form {} has id {}",
                    form.name, form.id
                )));
            }
            keys.insert((form.op.clone(), form.arity, form.returning), i);
        }
        let names: Vec<String> = def.counter_names.iter().map(|n| n.to_string()).collect();
        let own = own_ops(&names)?;
        let ops = keys.keys().map(|(op, _, _)| op.clone()).collect();
        let present: BTreeSet<&str> = def
            .forms
            .iter()
            .flat_map(|f| f.fields.iter().map(|x| x.name))
            .collect();
        let line_fields = crate::vocab::LINE_FIELD_NAMES
            .iter()
            .copied()
            .filter(|name| present.contains(name) || *name == "omod")
            .collect();
        Ok(Target {
            def,
            own,
            keys,
            ops,
            line_fields,
        })
    }

    /// The line modifier fields that appear on at least one form of this target.
    pub fn line_fields(&self) -> &[&'static str] {
        &self.line_fields
    }

    /// The definition the target was built from.
    pub fn definition(&self) -> &Definition {
        &self.def
    }

    /// The target's name.
    pub fn name(&self) -> &str {
        self.def.name
    }

    /// The hooks of its module.
    pub fn hooks(&self) -> &dyn Hooks {
        self.def.hooks.as_ref()
    }

    /// Its forms in id order, own ops excluded.
    pub fn forms(&self) -> &[Form] {
        &self.def.forms
    }

    /// Its own ops in id order −1, −2, …
    pub fn own_ops(&self) -> &[Form] {
        &self.own
    }

    /// The form of a key (op, arity, returning).
    pub fn form(&self, op: &str, arity: usize, returning: bool) -> Option<&Form> {
        let key = (op.to_owned(), arity, returning);
        self.keys.get(&key).map(|&i| &self.def.forms[i])
    }

    /// The form of an id, own ops included.
    pub fn form_id(&self, id: i32) -> Option<&Form> {
        if id >= 0 {
            self.def.forms.get(id as usize)
        } else {
            self.own.get((-1 - id) as usize)
        }
    }

    /// The first form with that op that is not an own op.
    pub fn named(&self, op: &str) -> Option<&Form> {
        self.def.forms.iter().find(|f| f.op == op)
    }

    /// An own op by kind.
    pub fn pseudo(&self, kind: Pseudo) -> &Form {
        self.own
            .iter()
            .find(|f| f.pseudo == Some(kind))
            .expect("every target has every own op")
    }

    /// An own op by name.
    pub fn pseudo_named(&self, name: &str) -> Option<&Form> {
        self.own.iter().find(|f| f.name == name)
    }

    /// The register file of a letter.
    pub fn file(&self, letter: char) -> Option<&File> {
        self.def.files.iter().find(|f| f.letter == letter)
    }

    /// Whether its encoder has an opcode for a mnemonic.
    pub fn encodable(&self, mnemonic: &str) -> bool {
        self.def.encodings.contains(mnemonic)
    }

    /// The wait fields: every wait instruction's count fields in order.
    pub fn wait_fields(&self) -> Vec<&'static str> {
        self.def
            .waits
            .iter()
            .flat_map(|w| w.fields.iter().copied())
            .collect()
    }

    /// The counter a wait instruction of one count waits on when the count
    /// is written as its operand (`s_wait_loadcnt 0`) rather than its
    /// field: the counter whose field is that wait's only field.
    pub fn operand_wait_counter(&self, mnemonic: &str) -> Option<usize> {
        let w = self.def.waits.iter().find(|w| w.mnemonic == mnemonic)?;
        let [field] = w.fields else { return None };
        self.def.counters.iter().position(|c| c.field == *field)
    }

    /// The immediate offsets a memory class's forms take, from their
    /// `offset` field (its bits and sign), or none if its forms have none.
    pub fn offset_range(&self, mem: Mem) -> Option<std::ops::RangeInclusive<i128>> {
        if mem == Mem::Smem {
            let bits = self.def.operands.smem_offset_bits;
            return (bits > 0).then(|| -(1i128 << (bits - 1))..=(1i128 << (bits - 1)) - 1);
        }
        let f = (self.def.forms.iter())
            .filter(|f| f.mem == Some(mem))
            .find_map(|f| f.field("offset"))?;
        let bits = f.bits.min(64);
        Some(if f.signed {
            -(1i128 << (bits - 1))..=(1i128 << (bits - 1)) - 1
        } else {
            0..=(1i128 << bits) - 1
        })
    }

    /// The index of the counter of that name.
    pub fn counter_named(&self, name: &str) -> Option<usize> {
        self.def.counters.iter().position(|c| c.name == name)
    }

    /// The counter named so, else every counter of a group's units.
    pub fn group(&self, name: &str) -> Vec<usize> {
        if let Some(c) = self.counter_named(name) {
            return vec![c];
        }
        let Some(&(_, units)) = GROUPS.iter().find(|(g, _)| *g == name) else {
            return Vec::new();
        };
        (0..self.def.counters.len())
            .filter(|&c| self.def.counters[c].units.any(units))
            .collect()
    }

    /// The lane-mask half of a named register: exec and vcc's low halves
    /// on a 32-lane target, else the register itself.
    pub fn mask_reg(&self, phys: &'static Phys) -> &'static Phys {
        match (self.def.wave, phys.name) {
            (32, "exec") => phys_named("exec_lo").expect("exec_lo"),
            (32, "vcc") => phys_named("vcc_lo").expect("vcc_lo"),
            _ => phys,
        }
    }

    /// All ones over the wave's lanes.
    pub fn full_mask(&self) -> u64 {
        if self.def.wave == 64 {
            u64::MAX
        } else {
            (1 << self.def.wave) - 1
        }
    }

    /// The whole register a named register is a half of.
    pub fn whole(&self, phys: &Phys) -> &'static Phys {
        &PHYS[phys.whole]
    }

    /// `mask` → `b<wave>` and `umask` → `u<wave>` in each word.
    pub fn masked(&self, op: &str) -> String {
        let wave = self.def.wave;
        op.split('.')
            .map(|w| match w {
                "mask" => format!("b{wave}"),
                "umask" => format!("u{wave}"),
                other => other.to_owned(),
            })
            .collect::<Vec<_>>()
            .join(".")
    }

    /// The mnemonic of a dotted op, if the target has it.
    pub fn spell(&self, op: &str) -> Option<String> {
        if !op.contains('.') {
            return Some(op.to_owned());
        }
        let masked = self.masked(op);
        let m = (self.def.spellings.iter())
            .find(|(dotted, _)| *dotted == masked)
            .map_or_else(|| masked.replace('.', "_"), |(_, m)| m.to_string());
        (self.def.encodings.contains(&m) || self.ops.contains(&m)).then_some(m)
    }

    /// The dotted op whose spelling is a mnemonic.
    pub fn canon(&self, mnemonic: &str) -> String {
        canon_in(&self.def.spellings, mnemonic)
    }

    /// The unit flag of a mnemonic, or none.
    pub fn unit_of(&self, mnemonic: &str) -> Option<Flags> {
        let (matrix, smem) = self.def.unit_prefixes;
        unit_of(mnemonic, matrix, smem)
    }

    /// The vector registers allocated for (vgpr, agpr).
    pub fn allocated_vgprs(&self, vgpr: u32, agpr: u32) -> u32 {
        let total = if self.file('a').is_some() {
            vgpr.div_ceil(4) * 4 + agpr
        } else {
            vgpr
        };
        let granule = self.file('v').map_or(1, |f| f.granule.max(1));
        total.div_ceil(granule) * granule
    }

    /// The matrix form of a mnemonic, if modelled on this target.
    pub fn matrix_form(&self, mnemonic: &str) -> Option<crate::MatrixForm> {
        self.def.hooks.matrix_form(mnemonic)
    }

    /// Whether this target's `fp8` / `bf8` instructions use FNUZ (`e4m3fnuz` / `e5m2fnuz`)
    /// rather than OCP (`e4m3fn` / `e5m2`).
    pub fn fnuz_fp8(&self) -> bool {
        self.def.hooks.fnuz_fp8()
    }

    /// The `thin-formats` dtype name for this target's `fp8` (E4M3) encoding.
    pub fn fp8_dtype(&self) -> &'static str {
        if self.fnuz_fp8() {
            "e4m3fnuz"
        } else {
            "e4m3fn"
        }
    }

    /// The `thin-formats` dtype name for this target's `bf8` (E5M2) encoding.
    pub fn bf8_dtype(&self) -> &'static str {
        if self.fnuz_fp8() {
            "e5m2fnuz"
        } else {
            "e5m2"
        }
    }
}

fn canon_in(spellings: &[(&str, &str)], mnemonic: &str) -> String {
    let spelled = |m: &str| {
        spellings
            .iter()
            .find(|(_, s)| *s == m)
            .map(|(d, _)| d.to_string())
    };
    if let Some(dotted) = spelled(mnemonic) {
        return dotted;
    }
    // An encoding variant of a spelled op is that op's: v_max_num_f32_dpp
    // is v.max.f32.dpp as v_max_num_f32 is v.max.f32.
    for suffix in ["_e32", "_e64", "_dpp", "_sdwa"] {
        if let Some(dotted) = mnemonic.strip_suffix(suffix).and_then(spelled) {
            return format!("{dotted}.{}", &suffix[1..]);
        }
    }
    mnemonic.replace('_', ".")
}

/// Validate the target definition before publishing it.
fn check_waits(def: &Definition) -> Result<(), Error> {
    let fail = |why: String| Err(Error::Definition(format!("{}: {why}", def.name)));
    if def.counters.len() > 8 {
        return fail(format!("{} counters, more than 8", def.counters.len()));
    }
    let names: Vec<&str> = def.counters.iter().map(|c| c.name).collect();
    if names != def.counter_names {
        return fail(format!(
            "counters {names:?}, but counter names {:?}",
            def.counter_names
        ));
    }
    let fields: Vec<&str> = def
        .waits
        .iter()
        .flat_map(|w| w.fields.iter().copied())
        .collect();
    if fields.len() > 12 {
        return fail(format!("{} count fields, more than 12", fields.len()));
    }
    for counter in &def.counters {
        if !fields.contains(&counter.field) {
            return fail(format!(
                "counter {}'s field {} is no wait instruction's",
                counter.name, counter.field
            ));
        }
    }
    for form in &def.forms {
        if let Some(mf) = def.hooks.matrix_form(&form.name) {
            if mf.wave != def.wave {
                return fail(format!(
                    "{}: matrix form wave {} != target wave {}",
                    form.name, mf.wave, def.wave
                ));
            }
            let want_slots = if mf.scaled {
                vec![
                    mf.dwords('d'),
                    mf.dwords('a'),
                    mf.dwords('b'),
                    mf.dwords('c'),
                    1,
                    1,
                ]
            } else {
                vec![
                    mf.dwords('d'),
                    mf.dwords('a'),
                    mf.dwords('b'),
                    mf.dwords('c'),
                ]
            };
            if form.slots.len() == want_slots.len() {
                let got: Vec<u32> = form.slots.iter().map(|s| s.dwords).collect();
                if got != want_slots {
                    return fail(format!(
                        "{}: matrix form dwords {want_slots:?} != ISA slots {got:?}",
                        form.name
                    ));
                }
            }
        }
    }
    Ok(())
}

/// A wait count vector: one optional count per wait field.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct WaitCounts(pub Vec<Option<u32>>);

impl WaitCounts {
    /// No field present, over `n` wait fields.
    pub fn empty(n: usize) -> WaitCounts {
        WaitCounts(vec![None; n])
    }

    /// Only field `field` set to `v`.
    pub fn one(n: usize, field: usize, v: u32) -> WaitCounts {
        let mut counts = WaitCounts::empty(n);
        counts.0[field] = Some(v);
        counts
    }

    /// Per field, the smaller of two present counts, else whichever is.
    pub fn min(&self, other: &WaitCounts) -> WaitCounts {
        let pick = |(a, b): (&Option<u32>, &Option<u32>)| match (a, b) {
            (Some(a), Some(b)) => Some(*a.min(b)),
            _ => a.or(*b),
        };
        WaitCounts(self.0.iter().zip(&other.0).map(pick).collect())
    }

    /// Whether no field is present.
    pub fn is_empty(&self) -> bool {
        self.0.iter().all(Option::is_none)
    }

    /// Whether some field has a count `other` lacks, or a smaller one.
    pub fn exceeds(&self, other: &WaitCounts) -> bool {
        self.0.iter().zip(&other.0).any(|(a, b)| match (a, b) {
            (Some(_), None) => true,
            (Some(a), Some(b)) => a < b,
            _ => false,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::code_object::{Block, Footer, Resources};
    use crate::form::WaitText;
    use crate::rules::{FormRules, Row};
    use crate::syntax::Syntax;

    /// A target module with no syntax, encoder or code objects.
    struct Bare;

    impl Syntax for Bare {
        fn reg_count(&self, _: char) -> u32 {
            0
        }
        fn special(&self, _: &str) -> Option<(u32, u32)> {
            None
        }
        fn hwreg(&self) -> &[(&'static str, u32)] {
            &[]
        }
        fn mod_flag(&self, _: &str) -> bool {
            false
        }
        fn counted(&self, _: &str) -> Option<Vec<(&'static str, u64)>> {
            None
        }
        fn symbol(&self, _: &str, _: &str) -> Option<i128> {
            None
        }
    }

    impl Hooks for Bare {
        fn mem_of(&self, _: &str) -> Option<Mem> {
            None
        }
        fn returning(&self, _: &str, _: Option<&str>) -> bool {
            false
        }
        fn nop_word(&self) -> u32 {
            0
        }
        fn encode(&self, _: &str) -> Result<Encoded, String> {
            Err("no encoder".into())
        }
        fn e_flags(&self) -> u32 {
            0
        }
        fn granule(&self) -> u32 {
            1
        }
        fn directives(&self) -> &[&'static str] {
            &[]
        }
        fn footer(&self, _: &Footer) -> Result<String, Error> {
            Err(Error::Value("no footer".into()))
        }
        fn resources(&self, _: &Block) -> Result<Resources, Error> {
            Ok(Resources::default())
        }
    }

    static ONE: [&str; 1] = ["loadcnt"];
    static THIRTEEN: [&str; 13] = ["loadcnt"; 13];

    fn definition(counters: usize, fields: &'static [&'static str]) -> Definition {
        let counter = |i: usize| Counter {
            name: ["a", "b", "c", "d", "e", "f", "g", "h", "i"][i],
            max: 1,
            field: "loadcnt",
            units: Flags::VMEM,
            lands: Lands::Issue,
        };
        let counters: Vec<Counter> = (0..counters).map(counter).collect();
        Definition {
            name: "synthetic",
            forms: Vec::new(),
            files: Vec::new(),
            encodings: BTreeSet::new(),
            wave: 32,
            lds_max: 0,
            hazards: Hazards {
                rules: 0,
                pass: false,
                window: 0,
                delay_hint: None,
            },
            counter_names: counters.iter().map(|c| c.name).collect(),
            counters,
            waits: vec![WaitInsn {
                mnemonic: "s_wait",
                fields,
                text: WaitText::Bare,
            }],
            unit_prefixes: (&[], &[]),
            operands: Operands {
                vop3_literal: false,
                constant_bus: 1,
                valu_sgpr_write_waits_smem: false,
                smem_offset_bits: 0,
            },
            chip: Chip {
                xcds: 1,
                cus: 1,
                simds: 1,
                lds: 0,
                vgprs: 0,
                vgprs_per_lane: 0,
                sgprs: 0,
                user_sgprs: 0,
                preload: 0,
                wave_slots: 1,
                wgids: &[],
            },
            spellings: Vec::new(),
            barrier: Vec::new(),
            hooks: Box::new(Bare),
        }
    }

    fn fails(def: Definition) -> bool {
        matches!(Target::build(def), Err(Error::Definition(_)))
    }

    #[test]
    fn wait_checks_fail_the_build() {
        assert!(!fails(definition(2, &ONE)));
        assert!(fails(definition(9, &ONE)));
        assert!(fails(definition(1, &THIRTEEN)));
        let mut no_field = definition(1, &ONE);
        no_field.counters[0].field = "kmcnt";
        assert!(fails(no_field));
        let mut misnamed = definition(1, &ONE);
        misnamed.counter_names = vec!["z"];
        assert!(fails(misnamed));
    }

    fn rules() -> FormRules {
        FormRules {
            matrix: &[],
            smem: &[],
            trans: &[],
            waits: &[],
            barrier: "",
            carry_in: &[],
            wave_varying: &[],
            counter: |_, _, _| None,
            fields: |_, _, _| Ok(Vec::new()),
            signs: |_| Vec::new(),
            widths: |_, _| Vec::new(),
        }
    }

    fn row(slot: &str, uses: &[&str]) -> Row {
        Row {
            mnemonic: "v_x".into(),
            count: 1,
            roles: vec!["r".into()],
            defs: Vec::new(),
            uses: uses.iter().map(|u| u.to_string()).collect(),
            sgpr_widths: vec![Vec::new()],
            encoding: "ENC_VOP1".into(),
            slots: vec![(slot.into(), 32)],
        }
    }

    #[test]
    fn form_rules_refuse_unknown_types_and_registers() {
        assert!(rules().forms(&[row("OPR_VGPR", &["exec_lo"])], &[]).is_ok());
        assert!(rules().forms(&[row("OPR_NEW", &[])], &[]).is_err());
        assert!(rules().forms(&[row("OPR_VGPR", &["ttmp3"])], &[]).is_err());
        let mut no_unit = row("OPR_VGPR", &[]);
        no_unit.mnemonic = "x_y".into();
        assert!(rules().forms(&[no_unit], &[]).is_err());
    }

    #[test]
    fn wait_counts_combine() {
        let a = WaitCounts(vec![Some(3), None, Some(1)]);
        let b = WaitCounts(vec![Some(1), Some(2), None]);
        assert_eq!(a.min(&b), WaitCounts(vec![Some(1), Some(2), Some(1)]));
        assert!(a.exceeds(&WaitCounts(vec![Some(3), None, None])));
        assert!(!WaitCounts(vec![Some(3), None, None]).exceeds(&a));
        assert!(WaitCounts::empty(3).is_empty() && !WaitCounts::one(3, 1, 0).is_empty());
    }
}
