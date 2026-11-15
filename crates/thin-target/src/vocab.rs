//! The generic vocabulary a target is described in.

use std::fmt;

/// The encoding space of a named register.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Space {
    /// The scalar source space.
    S,
    /// SCC's own space.
    Scc,
}

impl Space {
    /// The space's name, `s` or `scc`.
    pub fn name(self) -> &'static str {
        match self {
            Space::S => "s",
            Space::Scc => "scc",
        }
    }
}

/// A named register: a row of the one table every target shares.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Phys {
    /// Its number, the index in [`PHYS`].
    pub number: usize,
    /// Its name.
    pub name: &'static str,
    /// Its width in dwords.
    pub dwords: u32,
    /// Its encoding space.
    pub space: Space,
    /// The number of the register it is a half of (its own if whole).
    pub whole: usize,
}

const fn phys(number: usize, name: &'static str, dwords: u32, space: Space, whole: usize) -> Phys {
    Phys {
        number,
        name,
        dwords,
        space,
        whole,
    }
}

/// The named registers of every target; the order is normative.
pub static PHYS: [Phys; 12] = [
    phys(0, "exec", 2, Space::S, 0),
    phys(1, "exec_lo", 1, Space::S, 0),
    phys(2, "exec_hi", 1, Space::S, 0),
    phys(3, "vcc", 2, Space::S, 3),
    phys(4, "vcc_lo", 1, Space::S, 3),
    phys(5, "vcc_hi", 1, Space::S, 3),
    phys(6, "m0", 1, Space::S, 6),
    phys(7, "scc", 1, Space::Scc, 7),
    phys(8, "off", 1, Space::S, 8),
    phys(9, "ttmp7", 1, Space::S, 9),
    phys(10, "ttmp9", 1, Space::S, 10),
    phys(11, "null", 1, Space::S, 11),
];

/// The named register of that name.
pub fn phys_named(name: &str) -> Option<&'static Phys> {
    PHYS.iter().find(|p| p.name == name)
}

/// A form's flags: a bit set with normative bit positions.
#[derive(Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Flags(pub u64);

macro_rules! flags {
    ($($name:ident = $bit:expr),* $(,)?) => {
        impl Flags {
            $(
                #[doc = concat!("The `", stringify!($name), "` flag.")]
                pub const $name: Flags = Flags(1 << $bit);
            )*
        }

        /// Every flag's (name, bit), in table order.
        pub static FLAG_BITS: &[(&str, u32)] = &[$((stringify!($name), $bit)),*];
    };
}

flags! {
    SALU = 0, VALU = 1, MATRIX = 2, SMEM = 3, VMEM = 4, DS = 5, BRANCH = 6, UNCOND = 7,
    ENDPGM = 8, BARRIER = 9, WAITCNT = 10, NOP = 11, LOAD = 12, STORE = 13, ATOMIC = 14,
    TRANS = 15, DOT = 16, DPP = 17, PERMLANE = 19, READLANE = 20, WRITELANE = 21,
    CMP = 22, CMPX = 23, CROSS_LANE = 24, D16 = 25, SCC_CMP = 26, SCC_SELECT = 27,
    SCC_CARRY_IN = 28, SCC_CARRY_OUT = 29, PSEUDO = 30, SAVEEXEC = 31, FP_ATOMIC = 32,
    WAVE_VARYING = 33, ACCUMULATE = 34, READS_DST = 35, LDS_MEM = 36,
}

impl Flags {
    /// No flag.
    pub const NONE: Flags = Flags(0);

    /// Whether every flag of `other` is set.
    pub fn has(self, other: Flags) -> bool {
        self.0 & other.0 == other.0 && other.0 != 0
    }

    /// Whether any flag of `other` is set.
    pub fn any(self, other: Flags) -> bool {
        self.0 & other.0 != 0
    }

    /// The names of the set flags, in table order.
    pub fn names(self) -> Vec<&'static str> {
        FLAG_BITS
            .iter()
            .filter(|(_, bit)| self.0 >> bit & 1 == 1)
            .map(|(name, _)| *name)
            .collect()
    }
}

impl std::ops::BitOr for Flags {
    type Output = Flags;
    fn bitor(self, other: Flags) -> Flags {
        Flags(self.0 | other.0)
    }
}

impl std::ops::BitOrAssign for Flags {
    fn bitor_assign(&mut self, other: Flags) {
        self.0 |= other.0;
    }
}

impl fmt::Debug for Flags {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let names = self.names();
        f.write_str(&if names.is_empty() {
            "NONE".into()
        } else {
            names.join("|")
        })
    }
}

/// Declares a fieldless enum whose variants have the given names, in order.
macro_rules! named {
    ($(#[$meta:meta])* $enum:ident { $($(#[$vmeta:meta])* $variant:ident = $name:literal),* $(,)? }) => {
        $(#[$meta])*
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
        pub enum $enum {
            $($(#[$vmeta])* $variant),*
        }

        impl $enum {
            /// Every value, in order.
            pub const ALL: &'static [$enum] = &[$($enum::$variant),*];

            /// Its name, as the Python view spells it.
            pub fn name(self) -> &'static str {
                match self {
                    $($enum::$variant => $name),*
                }
            }

            /// The value of that name.
            pub fn named(name: &str) -> Option<$enum> {
                $enum::ALL.iter().copied().find(|v| v.name() == name)
            }
        }
    };
}
pub(crate) use named;

named! {
    /// A form's class, the one the passes switch on; the order is
    /// normative where a table is indexed by kind.
    Kind {
        /// A vector ALU op.
        Valu = "valu",
        /// A scalar ALU op.
        Salu = "salu",
        /// A scalar memory op.
        Smem = "smem",
        /// A vector memory op.
        Vmem = "vmem",
        /// An LDS op.
        Ds = "ds",
        /// A matrix-unit op.
        Matrix = "matrix",
        /// A branch.
        Branch = "branch",
        /// A wait instruction.
        Wait = "wait",
        /// A no-op.
        Nop = "nop",
        /// The barrier.
        Barrier = "barrier",
        /// A comment (and the IR's own ops but the label).
        Comment = "comment",
        /// A label.
        Label = "label",
    }
}

named! {
    /// What may fill an explicit operand slot.
    SlotKind {
        /// A vector register.
        Vgpr = "vgpr",
        /// An accumulator register.
        Agpr = "agpr",
        /// Either.
        Av = "av",
        /// Either, or a constant.
        Avc = "avc",
        /// A scalar register.
        Sgpr = "sgpr",
        /// A scalar or named register, an inline constant or a literal.
        Ssrc = "ssrc",
        /// A vector, scalar or named register, an inline constant or a literal.
        Vsrc = "vsrc",
        /// An immediate written as an operand.
        Imm = "imm",
        /// A branch target.
        Label = "label",
        /// One named register only (VCC in a carry slot).
        Special = "special",
        /// Any register or named register: the IR's own ops only.
        Any = "any",
    }
}

impl SlotKind {
    /// The register files a register operand in the slot may come from.
    pub fn files(self) -> &'static str {
        match self {
            SlotKind::Vgpr => "v",
            SlotKind::Agpr => "a",
            SlotKind::Av | SlotKind::Avc => "va",
            SlotKind::Sgpr | SlotKind::Ssrc => "s",
            SlotKind::Vsrc => "vs",
            SlotKind::Any => "vsa",
            SlotKind::Imm | SlotKind::Label | SlotKind::Special => "",
        }
    }
}

named! {
    /// The kind of a form's field, and so of the value it holds.
    FieldKind {
        /// A bool.
        Flag = "flag",
        /// An integer of `bits` bits.
        Int = "int",
        /// A tuple of 0/1, one per operand.
        Bits = "bits",
        /// A DPP control's text.
        Dpp = "dpp",
        /// A string (own ops only).
        Text = "text",
        /// One of the field's choices.
        Choice = "choice",
        /// A tuple of names, tags or ints (own ops only).
        Tuple = "tuple",
    }
}

/// Every field name, in the normative order; the first 17, indices 40..49,
/// and index 52 are the fields a target's line spells.
pub static FIELD_NAMES: [&str; 53] = [
    "dpp",
    "row_mask",
    "bank_mask",
    "bound_ctrl",
    "offen",
    "idxen",
    "offset",
    "offset0",
    "offset1",
    "th",
    "scope",
    "op_sel",
    "op_sel_hi",
    "neg_lo",
    "neg_hi",
    "clamp",
    "omod",
    "name",
    "text",
    "why",
    "counter",
    "said",
    "regions",
    "tags",
    "left",
    "ablated",
    "lo",
    "hi",
    "loadcnt",
    "storecnt",
    "dscnt",
    "kmcnt",
    "depctr_hold_cnt",
    "depctr_sa_sdst",
    "depctr_va_vdst",
    "depctr_va_sdst",
    "depctr_va_ssrc",
    "depctr_va_vcc",
    "depctr_vm_vsrc",
    // ir.undefined's place, appended.
    "on",
    // Appended line fields and wait-count fields (preserving indices 0..40).
    "sc0",
    "sc1",
    "nt",
    "glc",
    "lds",
    "gds",
    "cbsz",
    "abid",
    "blgp",
    "vmcnt",
    "lgkmcnt",
    "expcnt",
    "neg",
];

/// How many leading entries of [`FIELD_NAMES`] a target's line spells.
pub const LINE_FIELDS: usize = 17;

/// Every field name a target's line may spell.
pub static LINE_FIELD_NAMES: [&str; 27] = [
    "dpp",
    "row_mask",
    "bank_mask",
    "bound_ctrl",
    "offen",
    "idxen",
    "offset",
    "offset0",
    "offset1",
    "th",
    "scope",
    "op_sel",
    "op_sel_hi",
    "neg_lo",
    "neg_hi",
    "clamp",
    "omod",
    "sc0",
    "sc1",
    "nt",
    "glc",
    "lds",
    "gds",
    "cbsz",
    "abid",
    "blgp",
    "neg",
];

/// Whether a [`FIELD_NAMES`] index is spelled on an instruction line.
pub fn is_line_field(index: usize) -> bool {
    index < LINE_FIELDS || (40..49).contains(&index) || index == 52
}

named! {
    /// A memory form's class, from its encoding name.
    Mem {
        /// LDS.
        Ds = "ds",
        /// Flat.
        Flat = "flat",
        /// Global.
        Global = "global",
        /// Scratch.
        Scratch = "scratch",
        /// Buffer.
        Buffer = "buffer",
        /// Scalar memory.
        Smem = "smem",
    }
}

/// The memory class of an XML encoding name.
pub fn mem_of_encoding(encoding: &str) -> Option<Mem> {
    match encoding {
        "ENC_VDS" | "ENC_DS" => Some(Mem::Ds),
        "ENC_VFLAT" | "ENC_FLAT" => Some(Mem::Flat),
        "ENC_VGLOBAL" | "ENC_FLAT_GLBL" => Some(Mem::Global),
        "ENC_VSCRATCH" | "ENC_FLAT_SCRATCH" => Some(Mem::Scratch),
        "ENC_VBUFFER" | "ENC_MUBUF" | "ENC_MUBUF lds" | "ENC_MTBUF" => Some(Mem::Buffer),
        "ENC_SMEM" => Some(Mem::Smem),
        _ => None,
    }
}

named! {
    /// Whether a VALU form writes only part of its destination.
    DstSel {
        /// Always part of it.
        Always = "Always",
        /// Never.
        Never = "Never",
        /// As the instruction's `op_sel` says.
        OpSel = "OpSel",
    }
}

named! {
    /// Which half of each destination dword a form writes; it keeps the
    /// other half, so it reads its destination too.
    DstHalf {
        /// The low half.
        Lo = "Lo",
        /// The high half.
        Hi = "Hi",
        /// The half `op_sel`'s destination bit names (low when clear).
        OpSel = "OpSel",
    }
}

/// The half rule on a form's flags and spelling, the width in bits of
/// its first slot and whether that slot is a written VGPR: a D16 load
/// writes the half its `_hi` names; a VALU op of a 16-bit VGPR destination
/// (not a compare) writes the half its destination select gives, the low
/// one when it has none.
pub fn dst_half(flags: Flags, spelling: &str, bits: u32, vgpr_dst: bool) -> Option<DstHalf> {
    let hi = spelling.contains("_hi") || spelling.contains("mixhi");
    let side = if hi { DstHalf::Hi } else { DstHalf::Lo };
    if !vgpr_dst {
        None
    } else if flags.has(Flags::D16) && flags.has(Flags::LOAD) {
        Some(side)
    } else if !flags.has(Flags::VALU) || flags.has(Flags::CMP) || bits != 16 {
        None
    } else if dst_sel(flags, spelling) == DstSel::OpSel {
        Some(DstHalf::OpSel)
    } else {
        Some(side)
    }
}

/// The `dst_sel` rule on a form's flags and spelling.
pub fn dst_sel(flags: Flags, spelling: &str) -> DstSel {
    let starts = |prefixes: &[&str]| prefixes.iter().any(|p| spelling.starts_with(p));
    if !flags.has(Flags::VALU) {
        DstSel::Never
    } else if starts(&["v_fma_mixlo", "v_fma_mixhi"]) {
        DstSel::Always
    } else if starts(&["v_pk_", "v_dot"]) || spelling.contains("_mix") {
        DstSel::Never
    } else {
        DstSel::OpSel
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flags_have_their_bits() {
        assert_eq!(Flags::LDS_MEM.0, 1 << 36);
        assert_eq!(FLAG_BITS.len(), 36);
        assert_eq!(format!("{:?}", Flags::VALU | Flags::DPP), "VALU|DPP");
        assert_eq!(format!("{:?}", Flags::NONE), "NONE");
    }

    #[test]
    fn halves_follow_the_spelling_and_the_width() {
        let valu = Flags::VALU;
        let load = Flags::VMEM | Flags::LOAD | Flags::D16;
        let cases = [
            (load, "global_load_d16_b16", 32, Some(DstHalf::Lo)),
            (load, "global_load_d16_hi_b16", 32, Some(DstHalf::Hi)),
            (
                Flags::VMEM | Flags::STORE | Flags::D16,
                "global_store_d16_hi_b16",
                32,
                None,
            ),
            (valu, "v_lshlrev_b16", 16, Some(DstHalf::OpSel)),
            (valu, "v_fma_mixhi_f16", 16, Some(DstHalf::Hi)),
            (valu, "v_dot2_f16_f16", 16, Some(DstHalf::Lo)),
            (valu, "v_add_f32", 32, None),
            (valu | Flags::CMP, "v_cmpx_eq_f16_dpp", 16, None),
            (Flags::SALU, "s_add_f16", 16, None),
        ];
        for (flags, name, bits, half) in cases {
            assert_eq!(dst_half(flags, name, bits, true), half, "{name}");
        }
        assert_eq!(dst_half(valu, "v_lshlrev_b16", 16, false), None);
    }

    #[test]
    fn named_registers_keep_their_numbers() {
        assert_eq!(phys_named("ttmp9").unwrap().number, 10);
        assert_eq!(PHYS[phys_named("vcc_hi").unwrap().whole].name, "vcc");
    }
}
