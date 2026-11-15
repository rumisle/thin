//! The one `.s` reader.

use std::collections::BTreeMap;

use crate::Error;

/// One code item of a `.s`, in line order, as states the result.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CodeItem {
    /// An instruction: its trimmed code text and its comment (`""` if none).
    Ins {
        /// The text, comment stripped.
        text: String,
        /// The comment, trimmed.
        comment: String,
        /// The 1-based line.
        line: usize,
    },
    /// A label, without its `:`, dots kept.
    Label {
        /// The name.
        name: String,
        /// The 1-based line.
        line: usize,
    },
    /// A `.p2align p` while code is on.
    Align {
        /// The power, 0 to 12.
        p: u8,
        /// The 1-based line.
        line: usize,
    },
    /// A column-0 `;` line while code is on (claims, the LDS table).
    Note {
        /// The text after the `;`, trimmed.
        text: String,
        /// The 1-based line.
        line: usize,
    },
}

/// The reader's result, exactly (code, once, block, meta).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SFile {
    /// The code items in line order.
    pub code: Vec<CodeItem>,
    /// The four directives written once: word → (argument, line).
    pub once: BTreeMap<String, (String, usize)>,
    /// The `.amdhsa_kernel` block: entry (after `.amdhsa_`) → (value, line).
    pub block: BTreeMap<String, (String, usize)>,
    /// The metadata lines between `---` and `...`, verbatim.
    pub meta: Vec<String>,
}

/// The directives a `.s` may have outside the descriptor block.
const DIRECTIVES: [&str; 13] = [
    ".protected",
    ".globl",
    ".type",
    ".size",
    ".amdgcn_target",
    ".amdhsa_code_object_version",
    ".text",
    ".p2align",
    ".section",
    ".amdhsa_kernel",
    ".end_amdhsa_kernel",
    ".amdgpu_metadata",
    ".end_amdgpu_metadata",
];

/// The directives recorded once.
const ONCE: [&str; 4] = [
    ".amdgcn_target",
    ".amdhsa_code_object_version",
    ".amdhsa_kernel",
    ".amdgpu_metadata",
];

/// Lines as Python's `str.splitlines()` splits them.
fn lines(text: &str) -> Vec<&str> {
    let breaks = |c: char| {
        matches!(
            c,
            '\n' | '\r'
                | '\u{0b}'
                | '\u{0c}'
                | '\u{1c}'
                | '\u{1d}'
                | '\u{1e}'
                | '\u{85}'
                | '\u{2028}'
                | '\u{2029}'
        )
    };
    let mut out = Vec::new();
    let mut start = 0;
    let mut chars = text.char_indices().peekable();
    while let Some((i, c)) = chars.next() {
        if breaks(c) {
            out.push(&text[start..i]);
            let mut next = i + c.len_utf8();
            if c == '\r' {
                if let Some(&(j, '\n')) = chars.peek() {
                    chars.next();
                    next = j + 1;
                }
            }
            start = next;
        }
    }
    if start < text.len() {
        out.push(&text[start..]);
    }
    out
}

/// The text before the comment: up to the first `//`, then the first `;`.
fn strip_comment(line: &str) -> &str {
    let line = line.split("//").next().unwrap_or(line);
    line.split(';').next().unwrap_or(line)
}

/// A line's comment: after its first `//` if that is non-empty, else after
/// the first `;` of the part before the `//`; trimmed.
fn comment_of(line: &str) -> String {
    if let Some((before, after)) = line.split_once("//") {
        if !after.is_empty() {
            return after.trim().to_owned();
        }
        return before
            .split_once(';')
            .map_or("", |(_, c)| c)
            .trim()
            .to_owned();
    }
    line.split_once(';')
        .map_or("", |(_, c)| c)
        .trim()
        .to_owned()
}

/// A label line: ending in `:` with no space.
fn label(code: &str) -> Option<&str> {
    code.strip_suffix(':').filter(|_| !code.contains(' '))
}

/// The reader's state.
struct Reader {
    file: SFile,
    code_on: bool,
    in_block: bool,
}

impl Reader {
    fn directive(&mut self, code: &str, number: usize) -> Result<(), Error> {
        let refuse = |why: &str| Err(Error::Asm(format!("line {number}: {code}: {why}")));
        let (word, argument) = match code.split_once(char::is_whitespace) {
            Some((w, a)) => (w, a.trim_start()),
            None => (code, ""),
        };
        if self.in_block && word.starts_with(".amdhsa_") && word != ".amdhsa_kernel" {
            let entry = &word[".amdhsa_".len()..];
            if self.file.block.contains_key(entry) {
                return refuse("given twice");
            }
            self.file
                .block
                .insert(entry.to_owned(), (argument.trim().to_owned(), number));
            return Ok(());
        }
        if !DIRECTIVES.contains(&word) {
            return refuse("not a directive this encoder takes");
        }
        if ONCE.contains(&word) {
            if self.file.once.contains_key(word) {
                return refuse("given twice");
            }
            self.file
                .once
                .insert(word.to_owned(), (argument.trim().to_owned(), number));
        }
        match word {
            ".text" => self.code_on = true,
            ".section" => self.code_on = false,
            ".amdhsa_kernel" => {
                self.code_on = false;
                self.in_block = true;
            }
            ".end_amdhsa_kernel" => self.in_block = false,
            ".p2align" if self.code_on => {
                let digits = !argument.is_empty() && argument.bytes().all(|b| b.is_ascii_digit());
                match argument.parse::<u8>() {
                    Ok(p) if digits && p <= 12 => {
                        self.file.code.push(CodeItem::Align { p, line: number })
                    }
                    _ => return refuse("an alignment is a power of two up to 2^12 with no fill"),
                }
            }
            _ => {}
        }
        Ok(())
    }
}

/// Reads a `.s` text. With `body`, code is on from the first line,
/// so a body or a fragment reads without `.text`.
pub fn read_s(text: &str, body: bool) -> Result<SFile, Error> {
    let mut reader = Reader {
        file: SFile::default(),
        code_on: body,
        in_block: false,
    };
    let all = lines(text);
    let mut i = 0;
    while i < all.len() {
        let (line, number) = (all[i], i + 1);
        i += 1;
        if reader.code_on && line.starts_with(';') {
            let text = line[1..].trim().to_owned();
            reader.file.code.push(CodeItem::Note { text, line: number });
            continue;
        }
        let code = strip_comment(line).trim();
        if code.is_empty() {
            continue;
        }
        if code.starts_with('.') {
            if let Some(name) = label(code) {
                reader.file.code.push(CodeItem::Label {
                    name: name.to_owned(),
                    line: number,
                });
                continue;
            }
            reader.directive(code, number)?;
            if code.split_whitespace().next() == Some(".amdgpu_metadata") {
                read_metadata(&all, &mut i, &mut reader.file)?;
            }
            continue;
        }
        if !reader.code_on {
            return Err(Error::Asm(format!("line {number}: {code}: outside .text")));
        }
        if let Some(name) = label(code) {
            reader.file.code.push(CodeItem::Label {
                name: name.to_owned(),
                line: number,
            });
        } else {
            reader.file.code.push(CodeItem::Ins {
                text: code.to_owned(),
                comment: comment_of(line),
                line: number,
            });
        }
    }
    Ok(reader.file)
}

/// The metadata block after `.amdgpu_metadata`: its lines up to one whose
/// trimmed text starts `.end_amdgpu_metadata`, the first `---` and the last
/// `...`.
fn read_metadata(all: &[&str], i: &mut usize, file: &mut SFile) -> Result<(), Error> {
    let start = *i;
    let end = (start..all.len())
        .find(|&j| all[j].trim().starts_with(".end_amdgpu_metadata"))
        .ok_or_else(|| Error::Asm(".amdgpu_metadata without its end".into()))?;
    let kept = &all[start..end];
    let framed = kept.len() >= 2 && kept[0].trim() == "---" && kept[kept.len() - 1].trim() == "...";
    if !framed {
        return Err(Error::Asm(format!(
            "line {}: .amdgpu_metadata: a YAML document, --- to ...",
            end + 1
        )));
    }
    file.meta = kept[1..kept.len() - 1]
        .iter()
        .map(|l| l.to_string())
        .collect();
    *i = end + 1;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_descriptor_block_ends_at_its_end_directive() {
        let text = ".amdhsa_kernel k\n.amdhsa_next_free_vgpr 8\n.end_amdhsa_kernel\n\
                    .amdhsa_code_object_version 6\n";
        let file = read_s(text, false).unwrap();
        assert_eq!(file.block.keys().collect::<Vec<_>>(), ["next_free_vgpr"]);
        assert_eq!(file.once[".amdhsa_code_object_version"].0, "6");
        // Past the block, an entry is no directive of its own.
        let after = ".amdhsa_kernel k\n.end_amdhsa_kernel\n.amdhsa_next_free_vgpr 8\n";
        assert!(read_s(after, false).is_err());
    }
}
