//! Concrete target instances and their registry.
//!
//! The gfx1201 instance owns its instruction tables, encoder/decoder, line
//! syntax, code-object facts, hazard model, WMMA layouts and emulator
//! hooks. It is the only crate that names target instances.
//!
//! The registry pairs a target with its hardware analyses. Instances are
//! fixed at build time. Every successful decode must re-encode byte for
//! byte, and every encoded instruction must decode.
//!
//! The registry exposes names, default_name, get, hazards and machine
//! numbers; physical encoding/decoding and instruction data are also
//! available to lifting.

use thin_formats::Machine;
use thin_target::{assemble, CodeItem, Error, Target};

pub use decode::{Decode, DecodeError, Instruction};

/// The ELF machine number of each target. gfx1201's
/// code objects use e_flags 0x4E exactly: code object v6, no xnack or
/// sramecc bits.
pub static MACHINES: [Machine; 1] = [Machine {
    number: 0x4E,
    target: gfx1201::NAME,
}];

pub mod decode;
mod gfx1201;
mod tables;

use std::sync::OnceLock;

/// The hazard model of the target of that name: its rules
/// and its own pass, which the generic passes call through the hook.
pub fn hazards(name: &str) -> Result<&'static dyn thin_passes::hazards::HazardModel, Error> {
    match name {
        gfx1201::NAME => Ok(&gfx1201::hazards::Model),
        _ => get(name).map(|_| unreachable!("every registered target has a hazard model")),
    }
}

/// The known target names, in the registry's order. The first is
/// the default.
pub fn names() -> &'static [&'static str] {
    &[gfx1201::NAME]
}

/// The default target's name.
pub fn default_name() -> &'static str {
    names()[0]
}

/// The target of that name, built on first use and shared; an unknown
/// name is `"<name>: not a target (one of <names>)"`. A target
/// that fails to build is a bug of its module, so that panics.
pub fn get(name: &str) -> Result<&'static Target, Error> {
    static GFX1201: OnceLock<Target> = OnceLock::new();
    match name {
        gfx1201::NAME => Ok(GFX1201.get_or_init(|| {
            gfx1201::build().unwrap_or_else(|e| panic!("gfx1201 does not build: {e}"))
        })),
        _ => Err(Error::Value(format!(
            "{name}: not a target (one of {})",
            names().join(", ")
        ))),
    }
}

/// The physical IR form of a decoded instruction. Encoder-proved fields
/// and source modifiers may extend the target's narrower DSL authoring form.
/// Input must come from this target's decoder; edited IR is still verified
/// and checked by the encoder when rendered.
pub fn physical_form(
    name: &str,
    instruction: &Instruction,
) -> Result<&'static thin_target::Form, Error> {
    match name {
        gfx1201::NAME => gfx1201::decoder::physical_form(instruction),
        _ => get(name).map(|_| unreachable!("registered target has physical forms")),
    }
}

/// The decoder of the target of that name, built on first use and shared.
pub fn decoder(name: &str) -> Result<&'static dyn Decode, Error> {
    match name {
        gfx1201::NAME => Ok(gfx1201::decoder::decoder()?),
        _ => get(name).map(|_| unreachable!("every registered target has a decoder")),
    }
}

/// Decodes a word sequence of the target of that name (see [`Decode`]);
/// an unknown target is a refusal at offset 0.
pub fn decode(name: &str, bytes: &[u8]) -> Result<Vec<Instruction>, DecodeError> {
    match decoder(name) {
        Ok(d) => d.decode(bytes),
        Err(e) => Err(DecodeError {
            offset: 0,
            bits: 0,
            len: 0,
            size: None,
            why: decode::Why::NoEncoding,
            class: "",
            mnemonic: None,
            field: None,
            detail: e.to_string(),
        }),
    }
}

/// The bytes of decoded instructions, through the target's line encoder
/// and the assembler: each instruction's text, with a label `.L<offset>`
/// at every branch target.
pub fn encode(name: &str, instructions: &[Instruction]) -> Result<Vec<u8>, Error> {
    let target = get(name)?;
    let mut targets: Vec<u32> = instructions
        .iter()
        .filter_map(Instruction::branch_target)
        .collect();
    targets.sort_unstable();
    targets.dedup();
    let mut items = Vec::with_capacity(instructions.len() + targets.len());
    let mut next = targets.iter().peekable();
    for (k, i) in instructions.iter().enumerate() {
        while let Some(&&t) = next.peek() {
            if t > i.offset {
                break;
            }
            items.push(CodeItem::Label {
                name: decode::label(t),
                line: k + 1,
            });
            next.next();
        }
        items.push(CodeItem::Ins {
            text: i.to_string(),
            comment: String::new(),
            line: k + 1,
        });
    }
    for &t in next {
        items.push(CodeItem::Label {
            name: decode::label(t),
            line: instructions.len() + 1,
        });
    }
    assemble(&items, target.hooks())
}

/// The bytes of one decoded instruction through the target's line
/// encoder; a branch is placed at the instruction's own offset, so its
/// target need not be among the bytes.
pub fn encode_one(name: &str, i: &Instruction) -> Result<Vec<u8>, String> {
    let target = get(name).map_err(|e| e.to_string())?;
    match target.hooks().encode(&i.to_string())? {
        thin_target::Encoded::Bytes(bytes) => Ok(bytes),
        thin_target::Encoded::Branch { raw, label } => {
            let to = i.branch_target().filter(|&t| decode::label(t) == label);
            let to = to.ok_or_else(|| format!("{i}: the branch names {label}"))?;
            let d = (i64::from(to) - (i64::from(i.offset) + 4)).div_euclid(4);
            Ok((raw | (d as u32 & 0xFFFF)).to_le_bytes().to_vec())
        }
    }
}
