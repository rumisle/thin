//! The dtype table and the facts of each float encoding.

use crate::{Error, Result};

/// How an encoding spells NaN.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NanKind {
    /// An all-ones exponent with a non-zero mantissa; a zero mantissa is ±inf.
    Ieee,
    /// Only the all-ones magnitude, of either sign.
    AllOnes,
    /// Only the sign-bit-only pattern (0x80 for 8-bit); 0x00 is unsigned zero.
    NegZero,
    /// The encoding has no NaN.
    None,
}

/// A float encoding: field widths and the special values it has.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Encoding {
    /// Exponent bits.
    pub exp: u32,
    /// Mantissa bits.
    pub man: u32,
    /// Whether it has ±inf.
    pub inf: bool,
    /// How it spells NaN.
    pub nan: NanKind,
    /// Whether it has a sign bit.
    pub signed: bool,
    /// Whether exponent field 0 holds subnormals.
    pub subnormals: bool,
}

impl Encoding {
    /// The exponent bias: 2^(exp−1) for FNUZ (`NegZero`), 2^(exp−1) − 1 otherwise.
    pub const fn bias(&self) -> i32 {
        let half = 1 << (self.exp - 1);
        if matches!(self.nan, NanKind::NegZero) {
            half
        } else {
            half - 1
        }
    }

    /// The magnitude pattern with every exponent and mantissa bit set.
    pub const fn ones(&self) -> u32 {
        (1 << (self.exp + self.man)) - 1
    }

    /// The magnitude pattern of the largest finite value.
    pub const fn max_bits(&self) -> u32 {
        let ones = self.ones();
        match self.nan {
            NanKind::Ieee => ones - (1 << self.man),
            _ if self.inf => ones - (1 << self.man),
            NanKind::AllOnes => ones - 1,
            NanKind::NegZero | NanKind::None => ones,
        }
    }

    /// The largest finite value: [`max_bits`](Self::max_bits) read as a
    /// normal number.
    pub fn max(&self) -> f64 {
        let bits = self.max_bits();
        let exponent = (bits >> self.man) as i32 - self.bias();
        let fraction = f64::from(bits & ((1 << self.man) - 1)) / f64::from(1u32 << self.man);
        (1.0 + fraction) * 2f64.powi(exponent)
    }

    /// The exponent of the largest finite value (e2m1: 2, e4m3fn: 8).
    pub const fn emax(&self) -> i32 {
        (self.max_bits() >> self.man) as i32 - self.bias()
    }

    /// Whether the encoding has neither NaN nor inf, so an encode takes
    /// the caller's patterns for them.
    pub const fn has_no_specials(&self) -> bool {
        !self.inf && matches!(self.nan, NanKind::None)
    }
}

/// An element dtype. Its identity is its name.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Dtype {
    /// Its identity, used in serialized forms.
    pub name: &'static str,
    /// Bits per element.
    pub width: u32,
    /// The name of the numpy storage type.
    pub numpy: &'static str,
    /// The name of the torch dtype of a tensor of it.
    pub torch: &'static str,
    /// The DLPack (type code, bits).
    pub dlpack: (u8, u8),
    /// The float encoding, or `None` for an integer dtype.
    pub encoding: Option<Encoding>,
}

impl Dtype {
    /// Bytes per element in an array: one for a sub-byte dtype.
    pub const fn storage_bytes(&self) -> usize {
        if self.width <= 8 {
            1
        } else {
            (self.width / 8) as usize
        }
    }

    /// The float encoding, refusing an integer dtype for `what`.
    pub fn float_encoding(&self, what: &str) -> Result<&Encoding> {
        self.encoding
            .as_ref()
            .ok_or_else(|| Error::Value(format!("{what}: {} is an integer dtype", self.name)))
    }
}

const fn float(
    name: &'static str,
    width: u32,
    numpy: &'static str,
    torch: &'static str,
    dlpack: (u8, u8),
    encoding: Encoding,
) -> Dtype {
    Dtype {
        name,
        width,
        numpy,
        torch,
        dlpack,
        encoding: Some(encoding),
    }
}

const fn integer(name: &'static str, width: u32, numpy: &'static str, dlpack: (u8, u8)) -> Dtype {
    Dtype {
        name,
        width,
        numpy,
        torch: numpy,
        dlpack,
        encoding: None,
    }
}

const fn signed(exp: u32, man: u32, inf: bool, nan: NanKind) -> Encoding {
    Encoding {
        exp,
        man,
        inf,
        nan,
        signed: true,
        subnormals: true,
    }
}

/// Every dtype, in the order of table.
pub static DTYPES: [Dtype; 15] = [
    float(
        "f32",
        32,
        "float32",
        "float32",
        (2, 32),
        signed(8, 23, true, NanKind::Ieee),
    ),
    float(
        "bf16",
        16,
        "uint16",
        "bfloat16",
        (4, 16),
        signed(8, 7, true, NanKind::Ieee),
    ),
    float(
        "f16",
        16,
        "uint16",
        "float16",
        (2, 16),
        signed(5, 10, true, NanKind::Ieee),
    ),
    float(
        "e4m3fn",
        8,
        "uint8",
        "float8_e4m3fn",
        (10, 8),
        signed(4, 3, false, NanKind::AllOnes),
    ),
    float(
        "e5m2",
        8,
        "uint8",
        "float8_e5m2",
        (12, 8),
        signed(5, 2, true, NanKind::Ieee),
    ),
    float(
        "e4m3fnuz",
        8,
        "uint8",
        "float8_e4m3fnuz",
        (11, 8),
        signed(4, 3, false, NanKind::NegZero),
    ),
    float(
        "e5m2fnuz",
        8,
        "uint8",
        "float8_e5m2fnuz",
        (13, 8),
        signed(5, 2, false, NanKind::NegZero),
    ),
    float(
        "e2m1",
        4,
        "uint8",
        "uint8",
        (1, 8),
        signed(2, 1, false, NanKind::None),
    ),
    float(
        "e2m3",
        6,
        "uint8",
        "uint8",
        (1, 8),
        signed(2, 3, false, NanKind::None),
    ),
    float(
        "e3m2",
        6,
        "uint8",
        "uint8",
        (1, 8),
        signed(3, 2, false, NanKind::None),
    ),
    float(
        "e8m0",
        8,
        "uint8",
        "uint8",
        (1, 8),
        Encoding {
            exp: 8,
            man: 0,
            inf: false,
            nan: NanKind::AllOnes,
            signed: false,
            subnormals: false,
        },
    ),
    integer("u8", 8, "uint8", (1, 8)),
    integer("i8", 8, "int8", (0, 8)),
    integer("u32", 32, "uint32", (1, 32)),
    integer("u64", 64, "uint64", (1, 64)),
];

/// The dtype of that name; an unknown name is a `ValueError` listing the known ones.
pub fn dtype(name: &str) -> Result<&'static Dtype> {
    DTYPES.iter().find(|d| d.name == name).ok_or_else(|| {
        let known: Vec<_> = DTYPES.iter().map(|d| d.name).collect();
        Error::Value(format!(
            "{name:?} is not a dtype (one of {})",
            known.join(", ")
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn enc(name: &str) -> Encoding {
        dtype(name).unwrap().encoding.unwrap()
    }

    #[test]
    fn max_follows_max_bits() {
        assert_eq!(enc("e4m3fn").max(), 448.0);
        assert_eq!(enc("e4m3fnuz").max(), 240.0);
        assert_eq!(enc("e4m3fnuz").bias(), 8);
        assert_eq!(enc("e4m3fnuz").max_bits(), 0x7F);
        assert_eq!(enc("e2m1").max(), 6.0);
        assert_eq!(enc("e5m2").max(), 57344.0);
        assert_eq!(enc("e5m2fnuz").max(), 57344.0);
        assert_eq!(enc("e5m2fnuz").bias(), 16);
        assert_eq!(enc("e5m2fnuz").max_bits(), 0x7F);
        assert_eq!(enc("f32").max(), f64::from(f32::MAX));
        assert_eq!(enc("f16").max(), 65504.0);
        assert_eq!(enc("e8m0").max(), 2f64.powi(127));
    }

    #[test]
    fn emax_of_the_mx_elements() {
        assert_eq!(enc("e2m1").emax(), 2);
        assert_eq!(enc("e4m3fn").emax(), 8);
    }

    #[test]
    fn unknown_names_are_refused() {
        assert!(matches!(dtype("fp8"), Err(Error::Value(_))));
    }
}
