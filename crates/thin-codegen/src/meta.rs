//! The metadata reader: the YAML subset the footer writes.

use thin_formats::Value;

/// A refusal of the reader, an `AsmError` message.
fn refuse<T>(msg: impl Into<String>) -> Result<T, String> {
    Err(msg.into())
}

/// The YAML words a scalar may not be.
const WORDS: [&str; 28] = [
    "null", "Null", "NULL", "~", "true", "True", "TRUE", "false", "False", "FALSE", "yes", "Yes",
    "YES", "no", "No", "NO", "on", "On", "ON", "off", "Off", "OFF", "y", "Y", "n", "N", ".inf",
    ".nan",
];

/// The `.language` values LLVM takes.
const LANGUAGES: [&str; 6] = [
    "OpenCL C",
    "OpenCL C++",
    "HCC",
    "HIP",
    "OpenMP",
    "Assembler",
];

/// A line: its indent and its trimmed content.
struct Line {
    indent: usize,
    content: String,
}

struct Reader {
    lines: Vec<Line>,
    at: usize,
}

/// Python's `repr` of a string, as the messages quote it.
fn repr(s: &str) -> String {
    match (s.contains('\''), s.contains('"')) {
        (true, false) => format!("\"{s}\""),
        _ => format!("'{}'", s.replace('\\', "\\\\").replace('\'', "\\'")),
    }
}

/// A scalar: an int, a bool or a plain string.
fn scalar(s: &str) -> Result<Value, String> {
    let digits = |t: &str| !t.is_empty() && t.bytes().all(|b| b.is_ascii_digit());
    let int = s == "0"
        || (s
            .strip_prefix('-')
            .unwrap_or(s)
            .starts_with(|c: char| ('1'..='9').contains(&c))
            && digits(s.strip_prefix('-').unwrap_or(s)));
    if int {
        return s.parse().map(Value::Int).map_err(|_| {
            format!(
                ".amdgpu_metadata: {} is not a value this reader takes",
                repr(s)
            )
        });
    }
    match s {
        "true" => return Ok(Value::Bool(true)),
        "false" => return Ok(Value::Bool(false)),
        _ => {}
    }
    let mut chars = s.chars();
    let first = chars
        .next()
        .is_some_and(|c| c.is_ascii_alphabetic() || c == '_');
    let rest = chars.all(|c| c.is_ascii_alphanumeric() || "_.- ".contains(c));
    if first && rest && s.trim() == s && !WORDS.contains(&s) {
        return Ok(Value::Str(s.to_owned()));
    }
    refuse(format!(
        ".amdgpu_metadata: {} is not a value this reader takes",
        repr(s)
    ))
}

/// Whether an item starts with a key (`[.\w]+:`).
fn keyed(item: &str) -> bool {
    let key = item.split(':').next().unwrap_or("");
    item.contains(':')
        && !key.is_empty()
        && key
            .chars()
            .all(|c| c == '.' || c == '_' || c.is_alphanumeric())
}

impl Reader {
    fn peek(&self) -> Option<&Line> {
        self.lines.get(self.at)
    }

    fn block(&mut self, indent: usize) -> Result<Value, String> {
        match self.peek() {
            Some(l) if l.content.starts_with("- ") => self.list(indent),
            _ => self.map(indent),
        }
    }

    fn list(&mut self, indent: usize) -> Result<Value, String> {
        let mut out = Vec::new();
        while let Some(l) = self
            .peek()
            .filter(|l| l.indent == indent && l.content.starts_with("- "))
        {
            let item = l.content[2..].trim_start().to_owned();
            if keyed(&item) {
                self.lines[self.at] = Line {
                    indent: indent + 2,
                    content: item,
                };
                out.push(self.map(indent + 2)?);
            } else {
                self.at += 1;
                out.push(scalar(&item)?);
            }
        }
        Ok(Value::Array(out))
    }

    fn map(&mut self, indent: usize) -> Result<Value, String> {
        let mut out: Vec<(String, Value)> = Vec::new();
        while let Some(l) = self
            .peek()
            .filter(|l| l.indent == indent && !l.content.starts_with("- "))
        {
            let content = l.content.clone();
            let Some((key, value)) = content.split_once(':') else {
                return refuse(format!(".amdgpu_metadata: {}", repr(&content)));
            };
            let key = key.trim().to_owned();
            if out.iter().any(|(k, _)| *k == key) {
                return refuse(format!(".amdgpu_metadata: {}", repr(&content)));
            }
            self.at += 1;
            let value = value.trim();
            let v = match (value.is_empty(), self.peek()) {
                (false, _) => scalar(value)?,
                (true, Some(n)) if n.indent > indent => {
                    let deeper = n.indent;
                    self.block(deeper)?
                }
                (true, Some(n)) if n.indent == indent && n.content.starts_with("- ") => {
                    self.list(indent)?
                }
                (true, _) => return refuse(format!(".amdgpu_metadata: {key} has no value")),
            };
            out.push((key, v));
        }
        Ok(Value::Map(out))
    }
}

/// Python's `repr` of a scalar value, for LLVM's type check.
fn value_repr(v: &Value) -> String {
    match v {
        Value::Str(s) => repr(s),
        Value::Int(i) => i.to_string(),
        Value::Bool(true) => "True".into(),
        Value::Bool(false) => "False".into(),
        _ => "...".into(),
    }
}

/// LLVM's type check of each map of `amdhsa.kernels`.
fn check_kernels(doc: &Value) -> Result<(), String> {
    let Some(Value::Array(kernels)) = doc.get("amdhsa.kernels") else {
        return Ok(());
    };
    for k in kernels {
        let Value::Map(entries) = k else { continue };
        for (key, v) in entries {
            let ok = match key.as_str() {
                ".uses_dynamic_stack" => matches!(v, Value::Bool(_)),
                ".language" => matches!(v, Value::Str(s) if LANGUAGES.contains(&s.as_str())),
                ".max_flat_workgroup_size" | ".kernarg_segment_size" | ".wavefront_size" => {
                    matches!(v, Value::Int(_))
                }
                _ => true,
            };
            if !ok {
                return refuse(format!(
                    ".amdgpu_metadata: {key}: {} is not a value LLVM takes there",
                    value_repr(v)
                ));
            }
        }
    }
    Ok(())
}

/// The metadata document of the lines between `---` and `...`.
pub fn read(lines: &[String]) -> Result<Value, String> {
    let lines = lines
        .iter()
        .filter(|l| !l.trim().is_empty())
        .map(|l| Line {
            indent: l.len() - l.trim_start_matches(' ').len(),
            content: l.trim().to_owned(),
        })
        .collect();
    let mut r = Reader { lines, at: 0 };
    let doc = r.block(0)?;
    if let Some(l) = r.peek() {
        return refuse(format!(".amdgpu_metadata: {} left over", repr(&l.content)));
    }
    check_kernels(&doc)?;
    Ok(doc)
}
