//! The argument block.

use std::collections::BTreeMap;

use crate::error::{value, Result};
use crate::types::Ty;

/// An argument's kind.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Kind {
    /// A type, optionally ranged.
    Type {
        /// The type.
        ty: Ty,
        /// The range.
        range: Option<(i128, i128)>,
        /// A divisor (its magic pair follows).
        divisor: bool,
        /// The multiple, 1 for none.
        multiple: i128,
    },
    /// Own memory: `scratch`, `kept`, `carried`, `gens`, `mapped(w)`.
    Own {
        /// The kind's name.
        owned: String,
        /// Whether uncached.
        uncached: bool,
        /// A mapped group's ranks.
        ranks: Option<u32>,
    },
    /// `peers(w)`.
    Peers(u32),
    /// `lib_arg(kind, library)`.
    Lib(Box<Kind>, String),
}

impl Kind {
    /// Its text as written (`u32.within(1, 4)`, `kept.uncached`).
    pub fn text(&self) -> String {
        match self {
            Kind::Type {
                ty,
                range,
                divisor,
                multiple,
            } => {
                let mut s = ty.name().to_owned();
                if let Some((lo, hi)) = range {
                    s.push_str(&format!(".within({lo}, {hi})"));
                }
                if *multiple != 1 {
                    s.push_str(&format!(".multiple_of({multiple})"));
                }
                if *divisor {
                    s.push_str(".divisor()");
                }
                s
            }
            Kind::Own {
                owned,
                uncached,
                ranks,
            } => {
                let mut s = match ranks {
                    Some(w) => format!("{owned}({w})"),
                    None => owned.clone(),
                };
                if *uncached {
                    s.push_str(".uncached");
                }
                s
            }
            Kind::Peers(w) => format!("peers({w})"),
            Kind::Lib(k, l) => format!("lib_arg({}, {l:?})", k.text()),
        }
    }

    /// `T.within(lo, hi)`.
    pub fn within(ty: Ty, lo: i128, hi: i128) -> Result<Kind> {
        if lo > hi {
            return value(format!("{}.within({lo}, {hi}): an empty range", ty.name()));
        }
        Ok(Kind::Type {
            ty,
            range: Some((lo, hi)),
            divisor: false,
            multiple: 1,
        })
    }

    /// `.divisor()`.
    pub fn divisor(&self) -> Result<Kind> {
        match self {
            Kind::Type {
                ty: Ty::U32,
                range: Some((lo, hi)),
                multiple,
                ..
            } if *lo >= 1 && *hi <= 1 << 31 => Ok(Kind::Type {
                ty: Ty::U32,
                range: Some((*lo, *hi)),
                divisor: true,
                multiple: *multiple,
            }),
            _ => value(format!("{}.divisor(): a u32 within 1 .. 2^31", self.text())),
        }
    }

    /// `.multiple_of(m)`; `m` is None for a non-int.
    pub fn multiple_of(&self, m: Option<i128>, written: &str) -> Result<Kind> {
        let Some(m) = m.filter(|&m| m >= 2) else {
            return value(format!("multiple_of({written}): an int, 2 or more"));
        };
        let Kind::Type {
            ty,
            range,
            divisor,
            multiple,
        } = self
        else {
            return value(format!("multiple_of({written}): an int, 2 or more"));
        };
        if *multiple != 1 {
            return value(format!(
                "multiple_of({m}): already a multiple of {multiple} (state one: the lcm)"
            ));
        }
        if let Some((lo, hi)) = range {
            let first = lo.div_euclid(m) * m + if lo.rem_euclid(m) == 0 { 0 } else { m };
            if first > *hi {
                return value(format!(
                    "{}.multiple_of({m}): no multiple of {m} in it",
                    self.text()
                ));
            }
        }
        Ok(Kind::Type {
            ty: *ty,
            range: *range,
            divisor: *divisor,
            multiple: m,
        })
    }

    /// `.uncached`.
    pub fn uncached(&self) -> Result<Kind> {
        match self {
            Kind::Own {
                owned,
                uncached: false,
                ranks: None,
            } if matches!(owned.as_str(), "kept" | "carried" | "gens") => Ok(Kind::Own {
                owned: owned.clone(),
                uncached: true,
                ranks: None,
            }),
            _ => value(format!(
                "{}.uncached: kept, carried or gens memory",
                self.text()
            )),
        }
    }

    /// `mapped(w)` or `peers(w)`; `w` None for a non-int.
    pub fn ranked(peers: bool, w: Option<i128>, written: &str) -> Result<Kind> {
        let f = if peers { "peers" } else { "mapped" };
        match w.filter(|w| (2..=8).contains(w)) {
            Some(w) if peers => Ok(Kind::Peers(w as u32)),
            Some(w) => Ok(Kind::Own {
                owned: "mapped".into(),
                uncached: false,
                ranks: Some(w as u32),
            }),
            None => value(format!("{f}({written}): a group of 2 .. 8 ranks")),
        }
    }
}

/// An argument's record.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Arg {
    /// Its name.
    pub name: String,
    /// Its byte offset.
    pub offset: u64,
    /// Its bytes.
    pub bytes: u64,
    /// The type's name (`ptr` for own memory and peers).
    pub kind: String,
    /// Its range.
    pub lo: Option<i128>,
    /// Its range's top.
    pub hi: Option<i128>,
    /// Whether a divisor.
    pub divisor: bool,
    /// Its multiple, 1 if none.
    pub multiple: i128,
    /// The own-memory kind.
    pub owned: Option<String>,
    /// The library that adds it.
    pub library: Option<String>,
    /// `"uncached"` or none.
    pub mem: Option<String>,
    /// A mapped or peers argument's ranks.
    pub ranks: Option<u32>,
}

/// The argument block.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Args {
    /// The records, in keyword order.
    pub args: Vec<Arg>,
}

/// An argument as given: a kind, or (by its repr) anything else.
pub type Given = std::result::Result<Kind, String>;

impl Args {
    /// `Args(at=, **kinds)`.
    pub fn new(kinds: Vec<(String, Given)>, at: &BTreeMap<String, i128>) -> Result<Args> {
        let unknown: Vec<String> = at
            .keys()
            .filter(|k| !kinds.iter().any(|(n, _)| n == *k))
            .cloned()
            .collect();
        let mut out = Vec::new();
        let mut cursor = 0u64;
        for (name, given) in kinds {
            let kind = given.map_err(|r| crate::error::Error {
                class: crate::error::Class::Value,
                msg: format!("argument {name}: {r} is not one of ptr, u32, i32, f32, u64"),
            })?;
            let (inner, library) = match kind {
                Kind::Lib(k, l) => (*k, Some(l)),
                k => (k, None),
            };
            let mut a = record(&name, &inner)?;
            a.library = library;
            let align = a.bytes.min(8);
            let off = match at.get(&name) {
                Some(&o) => o as u64,
                None => cursor.div_ceil(align) * align,
            };
            if off % align != 0 {
                return value(format!(
                    "argument {name}: at byte {off}, a {} is {align}-aligned",
                    a.kind
                ));
            }
            a.offset = off;
            cursor = off + a.bytes;
            out.push(a);
        }
        // In order: after every argument's own refusals.
        if !unknown.is_empty() {
            return value(format!("at= names no argument: {}", unknown.join(", ")));
        }
        let mut sorted: Vec<&Arg> = out.iter().collect();
        sorted.sort_by_key(|a| a.offset);
        for w in sorted.windows(2) {
            if w[1].offset < w[0].offset + w[0].bytes {
                return value(format!(
                    "arguments {} and {} overlap (bytes {}..{}, {}..)",
                    w[0].name,
                    w[1].name,
                    w[0].offset,
                    w[0].offset + w[0].bytes - 1,
                    w[1].offset
                ));
            }
        }
        Ok(Args { args: out })
    }

    /// The end of the last argument.
    pub fn size(&self) -> u64 {
        self.args
            .iter()
            .map(|a| a.offset + a.bytes)
            .max()
            .unwrap_or(0)
    }

    /// An argument by name.
    pub fn get(&self, name: &str) -> Option<&Arg> {
        self.args.iter().find(|a| a.name == name)
    }

    /// The names without a library.
    pub fn own(&self) -> Vec<String> {
        self.args
            .iter()
            .filter(|a| a.library.is_none())
            .map(|a| a.name.clone())
            .collect()
    }
}

fn record(name: &str, k: &Kind) -> Result<Arg> {
    let mut a = Arg {
        name: name.to_owned(),
        offset: 0,
        bytes: 0,
        kind: String::new(),
        lo: None,
        hi: None,
        divisor: false,
        multiple: 1,
        owned: None,
        library: None,
        mem: None,
        ranks: None,
    };
    match k {
        Kind::Type {
            ty,
            range,
            divisor,
            multiple,
        } => {
            if !matches!(ty, Ty::Ptr | Ty::U32 | Ty::I32 | Ty::F32 | Ty::U64) {
                return value(format!(
                    "argument {name}: {} is not one of ptr, u32, i32, f32, u64",
                    ty.name()
                ));
            }
            if range.is_some() && !ty.is_int() {
                return value(format!(
                    "argument {name}: a range on an integer (u32, i32), not {}",
                    ty.name()
                ));
            }
            a.kind = ty.name().into();
            a.lo = range.map(|r| r.0);
            a.hi = range.map(|r| r.1);
            a.divisor = *divisor;
            a.multiple = *multiple;
            a.bytes = if *divisor {
                16
            } else {
                4 * u64::from(ty.dwords())
            };
        }
        Kind::Own {
            owned,
            uncached,
            ranks,
        } => {
            a.kind = "ptr".into();
            a.owned = Some(owned.clone());
            a.mem = uncached.then(|| "uncached".into());
            a.ranks = *ranks;
            a.bytes = 8 * ranks.map_or(1, |w| u64::from(w) + 1);
        }
        Kind::Peers(w) => {
            a.kind = "ptr".into();
            a.ranks = Some(*w);
            a.bytes = 8 * (u64::from(*w) + 1);
        }
        Kind::Lib(..) => unreachable!("a library's kind is unwrapped"),
    }
    Ok(a)
}
