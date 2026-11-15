//! Forms: one record per instruction spelling and operand count.

use crate::vocab::{
    dst_sel, named, DstHalf, DstSel, FieldKind, Flags, Kind, Mem, SlotKind, FIELD_NAMES,
};
use crate::Error;

/// An explicit operand slot.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Slot {
    /// What may fill it.
    pub kind: SlotKind,
    /// The operand's width in dwords (0: any, for an own op's `any` slot).
    pub dwords: u32,
    /// 1 read, 2 written, 3 both, 0 neither.
    pub role: u8,
    /// The XML operand type, for messages (empty for own ops).
    pub xml_type: String,
    /// The scalar register widths it takes, in dwords, sorted.
    pub sgpr_widths: Vec<u32>,
    /// Whether it takes neg/abs.
    pub modifiers: bool,
    /// Whether a source in it may be a literal.
    pub literal: bool,
    /// The widths it takes when more than one, else empty.
    pub widths: Vec<u32>,
    /// A register in it must end below this register number, when its
    /// encoding's register field is narrower than its file (an
    /// encoding-fixed form's 16-bit VGPR operand: v0..v127). Authoring
    /// forms have none: their encoder chooses an encoding that fits.
    pub limit: Option<u32>,
}

/// A field a form takes: an instruction's named modifier.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Field {
    /// Its name, from [`FIELD_NAMES`].
    pub name: &'static str,
    /// Its kind.
    pub kind: FieldKind,
    /// Its width in bits (0 where the kind has none).
    pub bits: u32,
    /// Whether an int field is signed.
    pub signed: bool,
    /// Whether the field must be given (own ops only).
    pub required: bool,
    /// A choice field's choices.
    pub choices: Vec<String>,
}

impl Field {
    /// A field of a name from the vocabulary, refusing any other.
    pub fn new(name: &str, kind: FieldKind, bits: u32, signed: bool) -> Result<Field, Error> {
        let name = FIELD_NAMES
            .iter()
            .find(|&&n| n == name)
            .ok_or_else(|| Error::Definition(format!("field {name:?} is not in the vocabulary")))?;
        Ok(Field {
            name,
            kind,
            bits,
            signed,
            required: false,
            choices: Vec::new(),
        })
    }

    /// The same field, required.
    pub fn required(mut self) -> Field {
        self.required = true;
        self
    }

    /// The same field with these choices.
    pub fn choices<S: ToString>(mut self, choices: &[S]) -> Field {
        self.choices = choices.iter().map(ToString::to_string).collect();
        self
    }
}

named! {
    /// Which of the IR's own ops a form is.
    Pseudo {
        /// `ir.wait_regs`.
        WaitRegs = "WaitRegs",
        /// `ir.wait_tags`.
        WaitTags = "WaitTags",
        /// `ir.wait_all`.
        WaitAll = "WaitAll",
        /// `ir.wait_left`.
        WaitLeft = "WaitLeft",
        /// `ir.wait_smem`.
        WaitSmem = "WaitSmem",
        /// `ir.discard`.
        Discard = "Discard",
        /// `ir.assume`.
        Assume = "Assume",
        /// `ir.uniform`.
        Uniform = "Uniform",
        /// `ir.undefined`.
        Undefined = "Undefined",
        /// `ir.label`.
        Label = "Label",
        /// `ir.comment`.
        Comment = "Comment",
    }
}

named! {
    /// How a wait instruction's line spells its counts.
    WaitText {
        /// `mnemonic f(n) g(m)`.
        Named = "Named",
        /// `mnemonic n`.
        Bare = "Bare",
    }
}

/// A form: what every instruction of one spelling and operand count
/// shares.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Form {
    /// Its index in the target's form list; own ops are −1, −2, …
    pub id: i32,
    /// The spelling, `+rtn` appended for a returning atomic keyed apart.
    pub name: String,
    /// The mnemonic as written: `name` without `+rtn`.
    pub op: String,
    /// The operation its encodings share.
    pub base: String,
    /// The op as a target-neutral dotted name.
    pub canon: String,
    /// Its class.
    pub kind: Kind,
    /// The index of the counter its ops count on.
    pub counter: Option<usize>,
    /// Whether its ops complete out of order on their counter.
    pub unordered: bool,
    /// Its flags.
    pub flags: Flags,
    /// The XML encoding name it came from (empty for own ops).
    pub encoding: String,
    /// The number of explicit operands.
    pub arity: usize,
    /// Whether it is an atomic's returning form keyed with `+rtn`.
    pub returning: bool,
    /// Its explicit operand slots, in text order.
    pub slots: Vec<Slot>,
    /// Whether the last slot repeats (own ops only).
    pub variadic: bool,
    /// The fields it takes, in a fixed order.
    pub fields: Vec<Field>,
    /// Named registers written implicitly, by number.
    pub implicit_defs: Vec<usize>,
    /// Named registers read implicitly, by number.
    pub implicit_uses: Vec<usize>,
    /// A memory form's bytes per lane, else 0.
    pub mem_bytes: u32,
    /// Which own op it is, if one.
    pub pseudo: Option<Pseudo>,
    /// Whether it is a DPP form.
    pub dpp: bool,
    /// Whether it writes part of its destination.
    pub dst_sel: DstSel,
    /// The half of each destination dword it writes, keeping the other:
    /// an instruction of it reads its first operand as well as writing it.
    /// Its slots keep the operand table's roles.
    pub dst_half: Option<DstHalf>,
    /// Its memory class.
    pub mem: Option<Mem>,
    /// How the count-less form of a wait instruction prints its counts.
    pub wait_text: Option<WaitText>,
}

impl Form {
    /// The name without `+rtn`.
    pub fn printed(&self) -> &str {
        &self.op
    }

    /// Whether it is one of the five wait own ops.
    pub fn is_wait(&self) -> bool {
        use Pseudo::*;
        matches!(
            self.pseudo,
            Some(WaitRegs | WaitTags | WaitAll | WaitLeft | WaitSmem)
        )
    }

    /// The field of that name.
    pub fn field(&self, name: &str) -> Option<&Field> {
        self.fields.iter().find(|f| f.name == name)
    }

    /// Whether the instruction maker checks scalar widths against the slots.
    pub fn checks_widths(&self) -> bool {
        !self.flags.has(Flags::PSEUDO)
            && !matches!(
                self.kind,
                Kind::Label | Kind::Comment | Kind::Wait | Kind::Nop | Kind::Branch
            )
    }

    /// Whether it loads from a buffer (a DMA when its `lds` field is set).
    pub fn loads_buffer(&self) -> bool {
        self.mem == Some(Mem::Buffer) && self.flags.has(Flags::LOAD)
    }

    /// Whether the compiler spells its line itself.
    pub fn own_line(&self) -> bool {
        self.flags.has(Flags::PSEUDO) || matches!(self.kind, Kind::Label | Kind::Comment)
    }

    /// Operand k's slot: `slots[k]`, or the last one past the end.
    pub fn slot(&self, k: usize) -> Option<&Slot> {
        self.slots.get(k).or(self.slots.last())
    }

    /// Operand k's role, 0 when there are no slots.
    pub fn role(&self, k: usize) -> u8 {
        self.slot(k).map_or(0, |s| s.role)
    }

    /// A form with op = base = name and no slots, fields, implicit
    /// registers or counter.
    pub fn plain(id: i32, name: &str, kind: Kind, flags: Flags) -> Form {
        Form {
            id,
            name: name.to_owned(),
            op: name.to_owned(),
            base: name.to_owned(),
            canon: name.replace('_', "."),
            kind,
            counter: None,
            unordered: false,
            flags,
            encoding: String::new(),
            arity: 0,
            returning: false,
            slots: Vec::new(),
            variadic: false,
            fields: Vec::new(),
            implicit_defs: Vec::new(),
            implicit_uses: Vec::new(),
            mem_bytes: 0,
            pseudo: None,
            dpp: false,
            dst_sel: dst_sel(flags, name),
            dst_half: None,
            mem: None,
            wait_text: None,
        }
    }
}

/// The operation an op's encodings share: one trailing `_e64` or `_dpp`
/// removed, the first that matches.
pub fn base(op: &str) -> &str {
    let op = op.strip_suffix("+rtn").unwrap_or(op);
    ["_e64", "_dpp"]
        .iter()
        .find_map(|s| op.strip_suffix(s))
        .unwrap_or(op)
}

/// A memory form's bytes per lane, from its name.
pub fn mem_bytes(name: &str) -> u32 {
    let last = name.rsplit('_').next().unwrap_or(name);
    let sized = last
        .strip_prefix(['b', 'i', 'u', 'f'])
        .and_then(|n| n.parse::<u32>().ok())
        .filter(|n| [8, 16, 32, 64, 96, 128].contains(n));
    match (sized, last) {
        (Some(n), _) if name.len() > last.len() => n / 8,
        _ => 4,
    }
}

/// Whether a form reads its destination.
pub fn reads_dst(name: &str) -> bool {
    let Some(rest) = name.strip_prefix("v_") else {
        return false;
    };
    if ["fmac_", "pk_fmac_"].iter().any(|p| rest.starts_with(p)) {
        return true;
    }
    rest.strip_prefix("dot").is_some_and(|d| d.contains("acc_"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bases_drop_one_suffix() {
        assert_eq!(base("v_add_co_u32_e64"), "v_add_co_u32");
        assert_eq!(base("buffer_atomic_add_u32+rtn"), "buffer_atomic_add_u32");
        assert_eq!(base("v_add_f32_e32"), "v_add_f32_e32");
    }

    #[test]
    fn bytes_per_lane() {
        assert_eq!(mem_bytes("global_load_d16_hi_b16"), 2);
        assert_eq!(mem_bytes("b32"), 4);
        assert_eq!(mem_bytes("s_load_b512"), 4);
        assert_eq!(mem_bytes("global_load_b96"), 12);
    }

    #[test]
    fn destinations_read() {
        assert!(reads_dst("v_fmac_f32") && reads_dst("v_dot2acc_f32_f16"));
        assert!(!reads_dst("v_dot2_f32_f16") && !reads_dst("s_fmac_f32"));
    }
}
