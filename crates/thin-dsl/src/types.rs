//! The value types.

/// A type: one object per type, compared by identity in the language.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Ty {
    /// A 32-bit unsigned integer.
    U32,
    /// A 32-bit signed integer.
    I32,
    /// An IEEE single.
    F32,
    /// A 64-bit integer (a register pair).
    U64,
    /// A u64 that is an address.
    Ptr,
    /// A lane mask of a 64-lane wave.
    Mask,
    /// A lane mask of a 32-lane wave.
    Mask32,
    /// An f32 pair on an aligned VGPR pair.
    F32x2,
    /// A condition: in SCC or as lanes, never in a value.
    Cond,
}

impl Ty {
    /// Every type, in the table's order.
    pub const ALL: [Ty; 9] = [
        Ty::U32,
        Ty::I32,
        Ty::F32,
        Ty::U64,
        Ty::Ptr,
        Ty::Mask,
        Ty::Mask32,
        Ty::F32x2,
        Ty::Cond,
    ];

    /// Its name.
    pub fn name(self) -> &'static str {
        match self {
            Ty::U32 => "u32",
            Ty::I32 => "i32",
            Ty::F32 => "f32",
            Ty::U64 => "u64",
            Ty::Ptr => "ptr",
            Ty::Mask => "mask",
            Ty::Mask32 => "mask32",
            Ty::F32x2 => "f32x2",
            Ty::Cond => "cond",
        }
    }

    /// Its name with its article: `an i32`, `an f32`, `a u32`.
    pub fn a(self) -> String {
        let n = self.name();
        let an = n.starts_with('i') || n.starts_with('f');
        format!("{} {n}", if an { "an" } else { "a" })
    }

    /// The type of a name.
    pub fn named(name: &str) -> Option<Ty> {
        Ty::ALL.into_iter().find(|t| t.name() == name)
    }

    /// Its dwords per element.
    pub fn dwords(self) -> u32 {
        match self {
            Ty::U32 | Ty::I32 | Ty::F32 | Ty::Mask32 => 1,
            Ty::U64 | Ty::Ptr | Ty::Mask | Ty::F32x2 => 2,
            Ty::Cond => 0,
        }
    }

    /// u32 or i32.
    pub fn is_int(self) -> bool {
        matches!(self, Ty::U32 | Ty::I32)
    }

    /// u64 or ptr.
    pub fn is_wide(self) -> bool {
        matches!(self, Ty::U64 | Ty::Ptr)
    }

    /// f32 or f32x2.
    pub fn is_float(self) -> bool {
        matches!(self, Ty::F32 | Ty::F32x2)
    }

    /// mask, mask32 or cond.
    pub fn is_lanes(self) -> bool {
        matches!(self, Ty::Mask | Ty::Mask32 | Ty::Cond)
    }

    /// mask or mask32.
    pub fn is_mask(self) -> bool {
        matches!(self, Ty::Mask | Ty::Mask32)
    }

    /// The mask of a wave of that many lanes.
    pub fn mask_of(wave: u32) -> Ty {
        if wave == 32 {
            Ty::Mask32
        } else {
            Ty::Mask
        }
    }
}
