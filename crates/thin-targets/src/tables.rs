//! A target's generated tables, read from their checked-in
//! canonical JSON.

use std::collections::{BTreeMap, HashMap};

use serde_json::Value;
use thin_target::{Error, Row};

/// A field of an encoding class.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EncField {
    /// Its name.
    pub name: String,
    /// Its highest bit.
    pub hi: u32,
    /// Its lowest bit.
    pub lo: u32,
    /// Its kind: fixed, op, vgpr, sgpr, ssrc, src, sbase, srsrc, signed, bits.
    pub kind: String,
    /// A fixed field's value, or a field's default.
    pub value: u64,
}

impl EncField {
    /// Its width in bits.
    pub fn bits(&self) -> u32 {
        self.hi - self.lo + 1
    }
}

/// An encoding class: its bytes and fields.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Encoding {
    /// The bytes of its word.
    pub bytes: usize,
    /// Its fields, highest first.
    pub fields: Vec<EncField>,
}

impl Encoding {
    /// The field of that name.
    pub fn field(&self, name: &str) -> Option<&EncField> {
        self.fields.iter().find(|f| f.name == name)
    }
}

/// An operand of an op, by its encoding field.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OpOperand {
    /// The encoding field it is written to.
    pub field: String,
    /// Its data format.
    pub format: String,
    /// Its size in bits.
    pub bits: u32,
    /// Its XML operand type.
    pub xml_type: String,
}

/// An op: its class, opcode and operands.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Op {
    /// Its encoding class.
    pub class: String,
    /// Its opcode.
    pub opcode: u64,
    /// The operand of each field, by field name.
    pub operands: Vec<OpOperand>,
}

impl Op {
    /// The operand written to a field.
    pub fn operand(&self, field: &str) -> Option<&OpOperand> {
        self.operands.iter().find(|o| o.field == field)
    }
}

/// The ISA tables: encoding classes and ops by mnemonic.
#[derive(Debug, Default)]
pub struct IsaTables {
    /// The classes by name.
    pub encodings: BTreeMap<String, Encoding>,
    /// The ops by mnemonic.
    pub ops: HashMap<String, Op>,
}

fn bad(what: &str) -> Error {
    Error::Definition(format!("the tables: {what}"))
}

fn str_of(v: &Value, key: &str) -> Result<String, Error> {
    v[key].as_str().map(str::to_owned).ok_or_else(|| bad(key))
}

fn u64_of(v: &Value, key: &str) -> Result<u64, Error> {
    v[key].as_u64().ok_or_else(|| bad(key))
}

fn list<'v>(v: &'v Value, key: &str) -> Result<&'v Vec<Value>, Error> {
    v[key].as_array().ok_or_else(|| bad(key))
}

/// Reads the ISA tables' JSON.
pub fn isa_tables(json: &str) -> Result<IsaTables, Error> {
    let doc: Value = serde_json::from_str(json).map_err(|e| bad(&e.to_string()))?;
    let mut tables = IsaTables::default();
    for e in list(&doc, "encodings")? {
        let fields = list(e, "fields")?
            .iter()
            .map(|f| {
                Ok(EncField {
                    name: str_of(f, "name")?,
                    hi: u64_of(f, "hi")? as u32,
                    lo: u64_of(f, "lo")? as u32,
                    kind: str_of(f, "kind")?,
                    value: u64_of(f, "value")?,
                })
            })
            .collect::<Result<_, Error>>()?;
        let bytes = u64_of(e, "bytes")? as usize;
        tables
            .encodings
            .insert(str_of(e, "class")?, Encoding { bytes, fields });
    }
    for o in list(&doc, "ops")? {
        let operands = list(o, "operands")?
            .iter()
            .map(|x| {
                Ok(OpOperand {
                    field: str_of(x, "field")?,
                    format: str_of(x, "format")?,
                    bits: u64_of(x, "bits")? as u32,
                    xml_type: str_of(x, "type")?,
                })
            })
            .collect::<Result<_, Error>>()?;
        let op = Op {
            class: str_of(o, "class")?,
            opcode: u64_of(o, "opcode")?,
            operands,
        };
        tables.ops.insert(str_of(o, "mnemonic")?, op);
    }
    Ok(tables)
}

/// Reads the operand table's JSON.
pub fn operand_table(json: &str) -> Result<Vec<Row>, Error> {
    let doc: Value = serde_json::from_str(json).map_err(|e| bad(&e.to_string()))?;
    let strings = |v: &Value| -> Result<Vec<String>, Error> {
        v.as_array()
            .ok_or_else(|| bad("a list"))?
            .iter()
            .map(|s| s.as_str().map(str::to_owned).ok_or_else(|| bad("a string")))
            .collect()
    };
    doc.as_array()
        .ok_or_else(|| bad("rows"))?
        .iter()
        .map(|r| {
            let widths = list(r, "sgpr_widths")?
                .iter()
                .map(|w| {
                    w.as_array()
                        .ok_or_else(|| bad("widths"))?
                        .iter()
                        .map(|n| n.as_u64().map(|n| n as u32).ok_or_else(|| bad("width")))
                        .collect()
                })
                .collect::<Result<_, Error>>()?;
            let slots = list(r, "slots")?
                .iter()
                .map(|s| {
                    let kind = s[0].as_str().ok_or_else(|| bad("slot type"))?.to_owned();
                    Ok((kind, s[1].as_u64().ok_or_else(|| bad("slot size"))? as u32))
                })
                .collect::<Result<_, Error>>()?;
            Ok(Row {
                mnemonic: str_of(r, "mnemonic")?,
                count: u64_of(r, "count")? as usize,
                roles: strings(&r["roles"])?,
                defs: strings(&r["defs"])?,
                uses: strings(&r["uses"])?,
                sgpr_widths: widths,
                encoding: str_of(r, "encoding")?,
                slots,
            })
        })
        .collect()
}
