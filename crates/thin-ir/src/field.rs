//! Field values and field names.

use std::any::Any;
use std::fmt;
use std::sync::Arc;

use thin_target::FIELD_NAMES;

use crate::{pyfmt, Error};

/// A field name: its number in the global registry.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct FieldId(pub u8);

impl FieldId {
    /// The field of that name; a name no form has is a `ValueError`.
    pub fn named(name: &str) -> Result<FieldId, Error> {
        FIELD_NAMES
            .iter()
            .position(|n| *n == name)
            .map(|k| FieldId(k as u8))
            .ok_or_else(|| Error::Value(format!("no field {name} (no form has one)")))
    }

    /// The field of a name the IR reads, which the registry holds.
    pub(crate) fn of(name: &str) -> FieldId {
        FieldId::named(name).expect("a registered field name")
    }

    /// Its name.
    pub fn name(self) -> &'static str {
        FIELD_NAMES[self.0 as usize]
    }
}

/// An atom of a tag: an int or a str.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Atom {
    /// An integer.
    Int(i128),
    /// A string.
    Str(Arc<str>),
}

/// A tag: a tuple of atoms, compared by value.
pub type Tag = Arc<[Atom]>;

/// What the IR reads of a host object, captured once when it is attached.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ObjectFacts {
    /// Its truthiness.
    pub truthy: bool,
    /// Its `str()`.
    pub str: String,
    /// Its `repr()`.
    pub repr: String,
    /// Whether it is an int (of any int class).
    pub is_int: bool,
    /// Its value, when it is an int that fits i128.
    pub int: Option<i128>,
    /// Whether it is a str (of any str class).
    pub is_str: bool,
    /// Whether it is a tuple (of any tuple class).
    pub is_tuple: bool,
    /// Whether it is the host's none.
    pub is_none: bool,
}

/// A host object kept by identity behind an opaque handle, with the facts
/// the IR reads of it. Holding it keeps the object alive.
#[derive(Clone)]
pub struct HostObject {
    /// The host's handle; the IR never looks inside.
    pub handle: Arc<dyn Any + Send + Sync>,
    /// What the IR reads of it.
    pub facts: Arc<ObjectFacts>,
}

impl fmt::Debug for HostObject {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "HostObject({})", self.facts.repr)
    }
}

/// Plain data a native writer (the Builder) keeps behind a host object's
/// handle in place of a host value: the host reads it back as nested
/// tuples.
#[derive(Clone, Debug, PartialEq)]
pub enum Datum {
    /// None.
    None,
    /// An integer.
    Int(i128),
    /// A string.
    Str(Arc<str>),
    /// A register.
    Reg(crate::Reg),
    /// A tuple.
    Tuple(Vec<Datum>),
}

impl HostObject {
    /// A host object of native data, with the facts a host value of that
    /// shape would give.
    pub fn of_datum(d: Datum, repr: String) -> HostObject {
        let facts = ObjectFacts {
            truthy: !matches!(&d, Datum::None | Datum::Int(0))
                && !matches!(&d, Datum::Tuple(t) if t.is_empty()),
            str: repr.clone(),
            repr,
            is_int: matches!(d, Datum::Int(_)),
            int: match d {
                Datum::Int(v) => Some(v),
                _ => None,
            },
            is_str: matches!(d, Datum::Str(_)),
            is_tuple: matches!(d, Datum::Tuple(_)),
            is_none: matches!(d, Datum::None),
        };
        HostObject {
            handle: Arc::new(d),
            facts: Arc::new(facts),
        }
    }

    /// The native data behind the handle, if it holds some.
    pub fn datum(&self) -> Option<&Datum> {
        self.handle.downcast_ref::<Datum>()
    }
}

/// A field's value, held losslessly.
#[derive(Clone, Debug)]
pub enum FieldValue {
    /// An integer.
    Int(i128),
    /// A bool.
    Bool(bool),
    /// A string: also a choice or a DPP control.
    Text(Arc<str>),
    /// A tuple of ints; the empty tuple is this variant.
    Ints(Arc<[i128]>),
    /// A tuple of strs.
    Names(Arc<[Arc<str>]>),
    /// A tuple of tags.
    Tags(Arc<[Tag]>),
    /// Any other host value, kept by identity.
    Object(HostObject),
}

impl FieldValue {
    /// Int ≠ 0, the Bool itself, a non-empty Text, Ints, Names or Tags, or
    /// an Object's captured truthiness.
    pub fn truthy(&self) -> bool {
        match self {
            FieldValue::Int(v) => *v != 0,
            FieldValue::Bool(b) => *b,
            FieldValue::Text(t) => !t.is_empty(),
            FieldValue::Ints(v) => !v.is_empty(),
            FieldValue::Names(v) => !v.is_empty(),
            FieldValue::Tags(v) => !v.is_empty(),
            FieldValue::Object(o) => o.facts.truthy,
        }
    }

    /// An Int, or a Bool as 0/1.
    pub fn int(&self) -> Option<i128> {
        match self {
            FieldValue::Int(v) => Some(*v),
            FieldValue::Bool(b) => Some(*b as i128),
            _ => None,
        }
    }

    /// For Ints, the mask with bit k = element k & 1.
    pub fn bits(&self) -> Option<u64> {
        match self {
            FieldValue::Ints(v) => Some(
                (v.iter().enumerate())
                    .filter(|(k, _)| *k < 64)
                    .fold(0, |m, (k, x)| m | ((x & 1) as u64) << k),
            ),
            _ => None,
        }
    }

    /// The text, of a Text or of an Object that is a str.
    pub fn text(&self) -> Option<&str> {
        match self {
            FieldValue::Text(t) => Some(t),
            FieldValue::Object(o) if o.facts.is_str => Some(&o.facts.str),
            _ => None,
        }
    }

    /// Whether it is an Object holding the host's none.
    pub fn is_none(&self) -> bool {
        matches!(self, FieldValue::Object(o) if o.facts.is_none)
    }

    /// The host's `str()` of it.
    pub fn py_str(&self) -> String {
        match self {
            FieldValue::Text(t) => t.to_string(),
            FieldValue::Object(o) => o.facts.str.clone(),
            other => other.py_repr(),
        }
    }

    /// The host's `repr()` of it.
    pub fn py_repr(&self) -> String {
        match self {
            FieldValue::Int(v) => v.to_string(),
            FieldValue::Bool(b) => pyfmt::bool(*b).into(),
            FieldValue::Text(t) => pyfmt::repr_str(t),
            FieldValue::Ints(v) => pyfmt::tuple(&v.iter().map(i128::to_string).collect::<Vec<_>>()),
            FieldValue::Names(v) => {
                pyfmt::tuple(&v.iter().map(|n| pyfmt::repr_str(n)).collect::<Vec<_>>())
            }
            FieldValue::Tags(v) => pyfmt::tuple(&v.iter().map(|t| tag_repr(t)).collect::<Vec<_>>()),
            FieldValue::Object(o) => o.facts.repr.clone(),
        }
    }
}

/// The host's `repr()` of a tag.
pub fn tag_repr(tag: &[Atom]) -> String {
    let atoms: Vec<String> = (tag.iter())
        .map(|a| match a {
            Atom::Int(v) => v.to_string(),
            Atom::Str(s) => pyfmt::repr_str(s),
        })
        .collect();
    pyfmt::tuple(&atoms)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn helpers_read_their_variants() {
        let ints = FieldValue::Ints(Arc::from([1, 0, 3]));
        assert_eq!(ints.bits(), Some(0b101));
        assert!(!FieldValue::Ints(Arc::from([])).truthy());
        assert_eq!(FieldValue::Bool(true).int(), Some(1));
        assert_eq!(FieldValue::Text("x".into()).int(), None);
    }

    #[test]
    fn values_spell_as_python() {
        let tags = FieldValue::Tags(Arc::from([
            Tag::from([Atom::Str("dma".into()), Atom::Int(3), Atom::Int(1)]),
            Tag::from([Atom::Str("noorder".into())]),
        ]));
        assert_eq!(tags.py_str(), "(('dma', 3, 1), ('noorder',))");
        assert_eq!(FieldValue::Ints(Arc::from([0, 2])).py_str(), "(0, 2)");
        assert_eq!(FieldValue::Text("vm".into()).py_repr(), "'vm'");
        assert_eq!(FieldValue::Text("vm".into()).py_str(), "vm");
    }

    #[test]
    fn unknown_field_names_are_refused() {
        assert_eq!(FieldId::named("offset").unwrap().name(), "offset");
        let Err(Error::Value(msg)) = FieldId::named("nosuch") else {
            panic!("an unknown field passed")
        };
        assert_eq!(msg, "no field nosuch (no form has one)");
    }
}
