//! JSON as the manifest writes and reads it: Python's
//! `json.dumps(d, sort_keys=True)` bytes, and a strict parser.

use std::collections::BTreeMap;

/// A JSON value; maps keep their keys sorted.
#[derive(Clone, Debug, PartialEq)]
pub enum Json {
    /// `null`.
    Null,
    /// `true` or `false`.
    Bool(bool),
    /// An integer.
    Int(i128),
    /// A float.
    Float(f64),
    /// A string.
    Str(String),
    /// A list.
    List(Vec<Json>),
    /// An object.
    Map(BTreeMap<String, Json>),
}

impl Json {
    /// Python's `json.dumps(self, sort_keys=True)`: separators `", "` and
    /// `": "`, non-ASCII escaped.
    pub fn dumps(&self) -> String {
        let mut out = String::new();
        self.write(&mut out);
        out
    }

    fn write(&self, out: &mut String) {
        match self {
            Json::Null => out.push_str("null"),
            Json::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            Json::Int(i) => out.push_str(&i.to_string()),
            Json::Float(x) => out.push_str(&float(*x)),
            Json::Str(s) => string(s, out),
            Json::List(items) => {
                out.push('[');
                for (k, x) in items.iter().enumerate() {
                    if k > 0 {
                        out.push_str(", ");
                    }
                    x.write(out);
                }
                out.push(']');
            }
            Json::Map(m) => {
                out.push('{');
                for (k, (key, x)) in m.iter().enumerate() {
                    if k > 0 {
                        out.push_str(", ");
                    }
                    string(key, out);
                    out.push_str(": ");
                    x.write(out);
                }
                out.push('}');
            }
        }
    }

    /// The value of a key in a map.
    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Map(m) => m.get(key),
            _ => None,
        }
    }

    /// Whether it is an empty list or map.
    pub fn is_empty(&self) -> bool {
        matches!(self, Json::List(v) if v.is_empty())
            || matches!(self, Json::Map(m) if m.is_empty())
    }
}

/// Python's float repr in JSON (`NaN`, `Infinity` as Python writes them).
fn float(x: f64) -> String {
    match x {
        x if x.is_nan() => "NaN".into(),
        x if x == f64::INFINITY => "Infinity".into(),
        x if x == f64::NEG_INFINITY => "-Infinity".into(),
        x => thin_ir::pyfmt::repr_float(x),
    }
}

fn string(s: &str, out: &mut String) {
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
            c if (c as u32) < 0x20 || (c as u32) > 0x7f => {
                let mut buf = [0u16; 2];
                for u in c.encode_utf16(&mut buf) {
                    out.push_str(&format!("\\u{u:04x}"));
                }
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Parses a JSON text (with Python's `NaN`/`Infinity`).
pub fn parse(text: &str) -> Result<Json, String> {
    let mut p = Parser {
        s: text.as_bytes(),
        at: 0,
    };
    let v = p.value()?;
    p.ws();
    match p.at == p.s.len() {
        true => Ok(v),
        false => Err(format!("JSON: extra data at {}", p.at)),
    }
}

struct Parser<'a> {
    s: &'a [u8],
    at: usize,
}

impl Parser<'_> {
    fn ws(&mut self) {
        while self.s.get(self.at).is_some_and(|b| b" \t\n\r".contains(b)) {
            self.at += 1;
        }
    }

    fn eat(&mut self, word: &str) -> bool {
        let hit = self.s[self.at..].starts_with(word.as_bytes());
        if hit {
            self.at += word.len();
        }
        hit
    }

    fn err<T>(&self) -> Result<T, String> {
        Err(format!("JSON: unexpected input at {}", self.at))
    }

    fn value(&mut self) -> Result<Json, String> {
        self.ws();
        let words = [
            ("null", Json::Null),
            ("true", Json::Bool(true)),
            ("false", Json::Bool(false)),
        ];
        for (w, v) in words {
            if self.eat(w) {
                return Ok(v);
            }
        }
        for (w, x) in [
            ("NaN", f64::NAN),
            ("Infinity", f64::INFINITY),
            ("-Infinity", f64::NEG_INFINITY),
        ] {
            if self.eat(w) {
                return Ok(Json::Float(x));
            }
        }
        match self.s.get(self.at) {
            Some(b'"') => self.string().map(Json::Str),
            Some(b'[') => self.list(),
            Some(b'{') => self.map(),
            Some(b) if b.is_ascii_digit() || *b == b'-' => self.number(),
            _ => self.err(),
        }
    }

    fn list(&mut self) -> Result<Json, String> {
        self.at += 1;
        let mut out = Vec::new();
        self.ws();
        if self.eat("]") {
            return Ok(Json::List(out));
        }
        loop {
            out.push(self.value()?);
            self.ws();
            if self.eat("]") {
                return Ok(Json::List(out));
            }
            if !self.eat(",") {
                return self.err();
            }
        }
    }

    fn map(&mut self) -> Result<Json, String> {
        self.at += 1;
        let mut out = BTreeMap::new();
        self.ws();
        if self.eat("}") {
            return Ok(Json::Map(out));
        }
        loop {
            self.ws();
            if self.s.get(self.at) != Some(&b'"') {
                return self.err();
            }
            let k = self.string()?;
            self.ws();
            if !self.eat(":") {
                return self.err();
            }
            out.insert(k, self.value()?);
            self.ws();
            if self.eat("}") {
                return Ok(Json::Map(out));
            }
            if !self.eat(",") {
                return self.err();
            }
        }
    }

    fn number(&mut self) -> Result<Json, String> {
        let start = self.at;
        while self
            .s
            .get(self.at)
            .is_some_and(|b| b.is_ascii_digit() || b"+-.eE".contains(b))
        {
            self.at += 1;
        }
        let t = std::str::from_utf8(&self.s[start..self.at]).map_err(|e| e.to_string())?;
        match t.contains(['.', 'e', 'E']) {
            true => t
                .parse()
                .map(Json::Float)
                .map_err(|_| format!("JSON: number {t}")),
            false => t
                .parse()
                .map(Json::Int)
                .map_err(|_| format!("JSON: number {t}")),
        }
    }

    fn string(&mut self) -> Result<String, String> {
        self.at += 1;
        let mut units: Vec<u16> = Vec::new();
        let mut out = String::new();
        let flush = |units: &mut Vec<u16>, out: &mut String| {
            out.push_str(&String::from_utf16_lossy(units));
            units.clear();
        };
        loop {
            let Some(&b) = self.s.get(self.at) else {
                return self.err();
            };
            self.at += 1;
            match b {
                b'"' => {
                    flush(&mut units, &mut out);
                    return Ok(out);
                }
                b'\\' => {
                    let e = *self.s.get(self.at).ok_or("JSON: an escape at the end")?;
                    self.at += 1;
                    let simple = match e {
                        b'"' => Some('"'),
                        b'\\' => Some('\\'),
                        b'/' => Some('/'),
                        b'n' => Some('\n'),
                        b'r' => Some('\r'),
                        b't' => Some('\t'),
                        b'b' => Some('\u{8}'),
                        b'f' => Some('\u{c}'),
                        _ => None,
                    };
                    match (simple, e) {
                        (Some(c), _) => {
                            flush(&mut units, &mut out);
                            out.push(c);
                        }
                        (None, b'u') => {
                            let hex = std::str::from_utf8(
                                self.s.get(self.at..self.at + 4).ok_or("JSON: \\u")?,
                            )
                            .map_err(|e| e.to_string())?;
                            units.push(u16::from_str_radix(hex, 16).map_err(|e| e.to_string())?);
                            self.at += 4;
                        }
                        _ => return self.err(),
                    }
                }
                _ => {
                    flush(&mut units, &mut out);
                    let start = self.at - 1;
                    let len = utf8_len(b);
                    let piece = std::str::from_utf8(&self.s[start..start + len])
                        .map_err(|e| e.to_string())?;
                    out.push_str(piece);
                    self.at = start + len;
                }
            }
        }
    }
}

fn utf8_len(b: u8) -> usize {
    match b {
        b if b < 0x80 => 1,
        b if b >> 5 == 0b110 => 2,
        b if b >> 4 == 0b1110 => 3,
        _ => 4,
    }
}
