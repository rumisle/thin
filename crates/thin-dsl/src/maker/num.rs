//! The numbers the maker reads as Python reads them.

/// `int(s, 0)`: an optional sign, then `0x`/`0o`/`0b` digits or a decimal
/// without leading zeros, underscores between digits; None if Python
/// refuses it.
pub fn int0(s: &str) -> Option<i128> {
    let t = s.trim();
    let (neg, body) = match t.as_bytes().first() {
        Some(b'-') => (true, &t[1..]),
        Some(b'+') => (false, &t[1..]),
        _ => (false, t),
    };
    let lower = body.to_ascii_lowercase();
    let (radix, digits) = match lower.get(..2) {
        Some("0x") => (16, &lower[2..]),
        Some("0o") => (8, &lower[2..]),
        Some("0b") => (2, &lower[2..]),
        _ => (10, lower.as_str()),
    };
    let digits = digits
        .strip_prefix('_')
        .filter(|_| radix != 10)
        .unwrap_or(digits);
    if digits.is_empty()
        || digits.starts_with('_')
        || digits.ends_with('_')
        || digits.contains("__")
    {
        return None;
    }
    let clean: String = digits.chars().filter(|&c| c != '_').collect();
    if radix == 10 && clean.len() > 1 && clean.starts_with('0') && clean.chars().any(|c| c != '0') {
        return None;
    }
    let v = i128::from_str_radix(&clean, radix).ok()?;
    Some(if neg { -v } else { v })
}

/// A float as Python's `repr` writes it.
pub fn py_float(x: f64) -> String {
    if x.is_nan() {
        return "nan".into();
    }
    if x.is_infinite() {
        return if x > 0.0 { "inf" } else { "-inf" }.into();
    }
    let exp = if x == 0.0 {
        0
    } else {
        x.abs().log10().floor() as i32
    };
    if (-5..16).contains(&exp) {
        let s = format!("{x}");
        if s.contains('.') || s.contains('e') {
            s
        } else {
            format!("{s}.0")
        }
    } else {
        let s = format!("{x:e}");
        let (m, e) = s.split_once('e').expect("an exponent");
        let e: i32 = e.parse().expect("an int");
        let sign = if e < 0 { '-' } else { '+' };
        format!("{m}e{sign}{:02}", e.abs())
    }
}

/// The bits of the f32 inline constants ±0.5, ±1, ±2, ±4, 1/(2π).
pub const INLINE_F32: [u32; 9] = [
    0x3F00_0000,
    0xBF00_0000,
    0x3F80_0000,
    0xBF80_0000,
    0x4000_0000,
    0xC000_0000,
    0x4080_0000,
    0xC080_0000,
    0x3E22_F983,
];

/// Whether an int is a 32-bit literal: its low 32 bits, read signed, are
/// outside −16.. 64 and are not an f32 inline constant's bits.
pub fn is_literal32(v: i128) -> bool {
    let low = (v & 0xFFFF_FFFF) as u32;
    let signed = low as i32;
    !(-16..=64).contains(&signed) && !INLINE_F32.contains(&low)
}

/// A string of decimal or `0x` hex digits with an optional `-`.
pub fn int_text(s: &str) -> Option<i128> {
    let body = s.strip_prefix('-').unwrap_or(s);
    let ok = match body.strip_prefix("0x").or_else(|| body.strip_prefix("0X")) {
        Some(h) => !h.is_empty() && h.chars().all(|c| c.is_ascii_hexdigit()),
        None => !body.is_empty() && body.chars().all(|c| c.is_ascii_digit()),
    };
    if ok {
        int0(s).or_else(|| {
            body.parse::<i128>()
                .ok()
                .map(|v| if s.starts_with('-') { -v } else { v })
        })
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn python_numbers() {
        assert_eq!(int0("0x10"), Some(16));
        assert_eq!(int0("-1"), Some(-1));
        assert_eq!(int0("010"), None);
        assert_eq!(int0("1_000"), Some(1000));
        assert_eq!(int0("1.5"), None);
        assert_eq!(py_float(1.5), "1.5");
        assert_eq!(py_float(2.0), "2.0");
        assert_eq!(py_float(1e30), "1e+30");
        assert_eq!(py_float(1e-7), "1e-07");
        assert!(is_literal32(1000));
        assert!(!is_literal32(-16));
        assert!(!is_literal32(0x3F80_0000));
    }
}
