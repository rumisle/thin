//! Assembling a text section.

use std::collections::HashMap;
use std::ops::Range;

use crate::read_s::CodeItem;
use crate::target::{Encoded, Hooks};
use crate::Error;

/// The bytes of code items with a target's line encoder: labels resolved,
/// alignments padded with its nop word, branches fixed up. Notes
/// are skipped and comments ignored.
pub fn assemble(items: &[CodeItem], hooks: &dyn Hooks) -> Result<Vec<u8>, Error> {
    encode(items, hooks, None)
}

/// Text bytes with the range of each explicit instruction. Alignment gaps
/// contain the target's nop words and are not explicit instruction rows.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Assembly {
    /// Encoded text, with labels, branch displacements and padding resolved.
    pub bytes: Vec<u8>,
    /// Byte ranges in CodeItem::Ins order, including literal extensions.
    pub instructions: Vec<Range<usize>>,
}

/// Assembles through the same layout driver while recording byte ranges.
pub fn assemble_with_layout(items: &[CodeItem], hooks: &dyn Hooks) -> Result<Assembly, Error> {
    let mut instructions = Vec::new();
    let bytes = encode(items, hooks, Some(&mut instructions))?;
    Ok(Assembly {
        bytes,
        instructions,
    })
}

fn encode(
    items: &[CodeItem],
    hooks: &dyn Hooks,
    mut ranges: Option<&mut Vec<Range<usize>>>,
) -> Result<Vec<u8>, Error> {
    let mut out = Vec::new();
    let mut labels: HashMap<&str, usize> = HashMap::new();
    let mut branches = Vec::new();
    for item in items {
        match item {
            CodeItem::Label { name, line } => {
                if labels.insert(name, out.len()).is_some() {
                    return Err(Error::Asm(format!(
                        "line {line}: label {name} defined twice"
                    )));
                }
            }
            CodeItem::Align { p, .. } => {
                let nop = hooks.nop_word().to_le_bytes();
                while out.len() % (1 << p) != 0 {
                    out.extend_from_slice(&nop);
                }
            }
            CodeItem::Ins { text, line, .. } => {
                let start = out.len();
                match hooks.encode(text) {
                    Ok(Encoded::Bytes(bytes)) => out.extend(bytes),
                    Ok(Encoded::Branch { raw, label }) => {
                        branches.push((out.len(), raw, label, text));
                        out.extend([0; 4]);
                    }
                    Err(why) => return Err(Error::Asm(format!("line {line}: {text}: {why}"))),
                }
                if let Some(ranges) = ranges.as_deref_mut() {
                    ranges.push(start..out.len());
                }
            }
            CodeItem::Note { .. } => {}
        }
    }
    for (at, raw, label, text) in branches {
        let target = *labels
            .get(label.as_str())
            .ok_or_else(|| Error::Asm(format!("undefined label {label}")))?;
        let d = (target as i64 - (at as i64 + 4)).div_euclid(4);
        if !(-32768..32768).contains(&d) {
            return Err(Error::Asm(format!("branch out of range: {text}")));
        }
        out[at..at + 4].copy_from_slice(&(raw | (d as u32 & 0xFFFF)).to_le_bytes());
    }
    Ok(out)
}
