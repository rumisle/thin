//! msgpack, the subset the AMDGPU metadata note uses, and the canonical
//! JSON of its values.

use std::fmt::Write as _;

use crate::{Error, Result};

/// A value of the msgpack subset the AMDGPU metadata note uses.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    /// nil.
    Nil,
    /// false or true.
    Bool(bool),
    /// An integer in the uint64 or int64 range.
    Int(i128),
    /// A UTF-8 string.
    Str(String),
    /// An array.
    Array(Vec<Value>),
    /// A map with string keys (written sorted bytewise).
    Map(Vec<(String, Value)>),
}

impl Value {
    /// The value of `key` in a map, `None` for another value or a missing key.
    pub fn get(&self, key: &str) -> Option<&Value> {
        match self {
            Value::Map(entries) => entries.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }
}

/// How deep values may nest, so that a malformed note cannot exhaust the stack.
const MAX_DEPTH: usize = 256;

fn refuse(why: String) -> Error {
    Error::CodeObject(format!("msgpack: {why}"))
}

/// The smallest msgpack encoding of a value.
pub fn msgpack_write(value: &Value) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    write_value(value, &mut out, 0)?;
    Ok(out)
}

/// Writes a length with the fix form's limit and the 8/16/32-bit forms'
/// type bytes (no 8-bit form for arrays and maps).
fn write_len(len: usize, fix: (u8, usize), wide: [Option<u8>; 3], out: &mut Vec<u8>) -> Result<()> {
    if len < fix.1 {
        out.push(fix.0 | len as u8);
    } else if let (Some(t), Ok(n)) = (wide[0], u8::try_from(len)) {
        out.extend([t, n]);
    } else if let (Some(t), Ok(n)) = (wide[1], u16::try_from(len)) {
        out.push(t);
        out.extend(n.to_be_bytes());
    } else if let (Some(t), Ok(n)) = (wide[2], u32::try_from(len)) {
        out.push(t);
        out.extend(n.to_be_bytes());
    } else {
        return Err(refuse(format!("a length of {len} does not fit 32 bits")));
    }
    Ok(())
}

fn write_int(i: i128, out: &mut Vec<u8>) -> Result<()> {
    if (0..=127).contains(&i) {
        out.push(i as u8);
    } else if (-32..0).contains(&i) {
        out.push(i as i8 as u8);
    } else if let Ok(n) = u8::try_from(i) {
        out.extend([0xCC, n]);
    } else if let Ok(n) = u16::try_from(i) {
        out.push(0xCD);
        out.extend(n.to_be_bytes());
    } else if let Ok(n) = u32::try_from(i) {
        out.push(0xCE);
        out.extend(n.to_be_bytes());
    } else if let Ok(n) = u64::try_from(i) {
        out.push(0xCF);
        out.extend(n.to_be_bytes());
    } else if let Ok(n) = i8::try_from(i) {
        out.extend([0xD0, n as u8]);
    } else if let Ok(n) = i16::try_from(i) {
        out.push(0xD1);
        out.extend(n.to_be_bytes());
    } else if let Ok(n) = i32::try_from(i) {
        out.push(0xD2);
        out.extend(n.to_be_bytes());
    } else if let Ok(n) = i64::try_from(i) {
        out.push(0xD3);
        out.extend(n.to_be_bytes());
    } else {
        return Err(refuse(format!("{i} does not fit 64 bits")));
    }
    Ok(())
}

fn write_str(s: &str, out: &mut Vec<u8>) -> Result<()> {
    write_len(
        s.len(),
        (0xA0, 32),
        [Some(0xD9), Some(0xDA), Some(0xDB)],
        out,
    )?;
    out.extend_from_slice(s.as_bytes());
    Ok(())
}

fn write_value(value: &Value, out: &mut Vec<u8>, depth: usize) -> Result<()> {
    if depth > MAX_DEPTH {
        return Err(refuse(format!("values nest deeper than {MAX_DEPTH}")));
    }
    match value {
        Value::Nil => out.push(0xC0),
        Value::Bool(b) => out.push(0xC2 | u8::from(*b)),
        Value::Int(i) => write_int(*i, out)?,
        Value::Str(s) => write_str(s, out)?,
        Value::Array(items) => {
            write_len(items.len(), (0x90, 16), [None, Some(0xDC), Some(0xDD)], out)?;
            for item in items {
                write_value(item, out, depth + 1)?;
            }
        }
        Value::Map(entries) => {
            let mut sorted: Vec<&(String, Value)> = entries.iter().collect();
            sorted.sort_by(|a, b| a.0.as_bytes().cmp(b.0.as_bytes()));
            if let Some(pair) = sorted.windows(2).find(|w| w[0].0 == w[1].0) {
                return Err(refuse(format!("the key {:?} twice", pair[0].0)));
            }
            write_len(
                sorted.len(),
                (0x80, 16),
                [None, Some(0xDE), Some(0xDF)],
                out,
            )?;
            for (key, value) in sorted {
                if !key.is_ascii() {
                    return Err(refuse(format!("the key {key:?} is not ASCII")));
                }
                write_str(key, out)?;
                write_value(value, out, depth + 1)?;
            }
        }
    }
    Ok(())
}

/// A bounds-checked reader over msgpack bytes.
struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl Reader<'_> {
    fn take(&mut self, n: usize, what: &str) -> Result<&[u8]> {
        let end = self
            .at
            .checked_add(n)
            .filter(|&end| end <= self.bytes.len());
        let end = end.ok_or_else(|| {
            refuse(format!(
                "{what} at byte {} runs past the {} bytes",
                self.at,
                self.bytes.len()
            ))
        })?;
        let taken = &self.bytes[self.at..end];
        self.at = end;
        Ok(taken)
    }

    fn uint(&mut self, n: usize, what: &str) -> Result<u64> {
        let bytes = self.take(n, what)?;
        Ok(bytes.iter().fold(0, |acc, &b| acc << 8 | u64::from(b)))
    }

    fn int(&mut self, n: usize) -> Result<i128> {
        let unsigned = self.uint(n, "an int")?;
        let shift = 64 - 8 * n as u32;
        Ok(i128::from((unsigned << shift) as i64 >> shift))
    }

    fn str(&mut self, len: usize) -> Result<String> {
        let at = self.at;
        let bytes = self.take(len, "a string")?;
        String::from_utf8(bytes.to_vec())
            .map_err(|_| refuse(format!("the string at byte {at} is not UTF-8")))
    }

    fn value(&mut self, depth: usize) -> Result<Value> {
        if depth > MAX_DEPTH {
            return Err(refuse(format!("values nest deeper than {MAX_DEPTH}")));
        }
        let at = self.at;
        let t = self.take(1, "a type byte")?[0];
        Ok(match t {
            0x00..=0x7F => Value::Int(i128::from(t)),
            0xE0..=0xFF => Value::Int(i128::from(t as i8)),
            0xC0 => Value::Nil,
            0xC2 => Value::Bool(false),
            0xC3 => Value::Bool(true),
            0xCC..=0xCF => Value::Int(i128::from(self.uint(1 << (t - 0xCC), "a uint")?)),
            0xD0..=0xD3 => Value::Int(self.int(1 << (t - 0xD0))?),
            0xA0..=0xBF => Value::Str(self.str(usize::from(t & 0x1F))?),
            0xD9..=0xDB => {
                let len = self.uint(1 << (t - 0xD9), "a string length")?;
                Value::Str(self.str(len as usize)?)
            }
            0x90..=0x9F => self.array(usize::from(t & 0x0F), depth)?,
            0xDC | 0xDD => {
                let len = self.uint(if t == 0xDC { 2 } else { 4 }, "an array length")?;
                self.array(len as usize, depth)?
            }
            0x80..=0x8F => self.map(usize::from(t & 0x0F), depth)?,
            0xDE | 0xDF => {
                let len = self.uint(if t == 0xDE { 2 } else { 4 }, "a map length")?;
                self.map(len as usize, depth)?
            }
            _ => {
                return Err(refuse(format!(
                    "type byte {t:#04x} at byte {at} is not in the subset"
                )))
            }
        })
    }

    fn array(&mut self, len: usize, depth: usize) -> Result<Value> {
        let mut items = Vec::with_capacity(len.min(self.bytes.len()));
        for _ in 0..len {
            items.push(self.value(depth + 1)?);
        }
        Ok(Value::Array(items))
    }

    fn map(&mut self, len: usize, depth: usize) -> Result<Value> {
        let mut entries = Vec::with_capacity(len.min(self.bytes.len()));
        for _ in 0..len {
            let at = self.at;
            let Value::Str(key) = self.value(depth + 1)? else {
                return Err(refuse(format!("the map key at byte {at} is not a string")));
            };
            entries.push((key, self.value(depth + 1)?));
        }
        Ok(Value::Map(entries))
    }
}

/// The value at the start of msgpack bytes, and its length.
pub fn msgpack_read(bytes: &[u8]) -> Result<(Value, usize)> {
    let mut reader = Reader { bytes, at: 0 };
    let value = reader.value(0)?;
    Ok((value, reader.at))
}

/// Canonical JSON: keys sorted, no spaces, everything outside printable
/// ASCII escaped as `\uXXXX`.
pub fn canonical_json(value: &Value) -> String {
    let mut out = String::new();
    write_json(value, &mut out);
    out
}

fn write_json(value: &Value, out: &mut String) {
    match value {
        Value::Nil => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Int(i) => {
            let _ = write!(out, "{i}");
        }
        Value::Str(s) => write_json_str(s, out),
        Value::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_json(item, out);
            }
            out.push(']');
        }
        Value::Map(entries) => {
            let mut sorted: Vec<_> = entries.iter().collect();
            sorted.sort_by(|a, b| a.0.cmp(&b.0));
            out.push('{');
            for (i, (key, value)) in sorted.into_iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_json_str(key, out);
                out.push(':');
                write_json(value, out);
            }
            out.push('}');
        }
    }
}

fn write_json_str(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            ' '..='~' => out.push(c),
            _ => {
                let mut units = [0; 2];
                for unit in c.encode_utf16(&mut units) {
                    let _ = write!(out, "\\u{unit:04x}");
                }
            }
        }
    }
    out.push('"');
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integers_take_their_smallest_form() {
        let hex = |i: i128| {
            msgpack_write(&Value::Int(i))
                .unwrap()
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect::<String>()
        };
        assert_eq!(hex(-33), "d0df");
        assert_eq!(hex(255), "ccff");
        assert_eq!(hex(-129), "d1ff7f");
        assert_eq!(hex(-(1 << 63)), "d38000000000000000");
        assert!(msgpack_write(&Value::Int(1 << 64)).is_err());
        assert!(msgpack_write(&Value::Int(-(1 << 63) - 1)).is_err());
    }

    #[test]
    fn maps_sort_and_refuse_non_ascii_keys() {
        let map = Value::Map(vec![("b".into(), Value::Nil), ("a".into(), Value::Nil)]);
        assert_eq!(
            msgpack_write(&map).unwrap(),
            [0x82, 0xA1, b'a', 0xC0, 0xA1, b'b', 0xC0]
        );
        let bad = Value::Map(vec![("é".into(), Value::Nil)]);
        assert!(matches!(msgpack_write(&bad), Err(Error::CodeObject(_))));
    }

    #[test]
    fn reading_refuses_what_is_outside_the_subset() {
        assert!(msgpack_read(&[0xC1]).is_err());
        assert!(msgpack_read(&[0xCA, 0, 0, 0, 0]).is_err());
        assert!(msgpack_read(&[0xA2, b'a']).is_err());
        assert!(msgpack_read(&[0xA1, 0xFF]).is_err());
        assert!(msgpack_read(&[0x81, 0x01, 0xC0]).is_err());
        assert_eq!(
            msgpack_read(&[0xD0, 0x80, 0]).unwrap(),
            (Value::Int(-128), 2)
        );
        assert!(msgpack_read(&[0x91; 1000]).is_err());
    }

    #[test]
    fn json_escapes_like_the_reference() {
        let s = Value::Str("a\"\\\n\u{7f}é😀".into());
        assert_eq!(canonical_json(&s), r#""a\"\\\n\u007f\u00e9\ud83d\ude00""#);
        let map = Value::Map(vec![
            ("b".into(), Value::Int(-1)),
            ("a".into(), Value::Array(vec![])),
        ]);
        assert_eq!(canonical_json(&map), r#"{"a":[],"b":-1}"#);
    }
}
