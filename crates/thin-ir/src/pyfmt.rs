//! Python's spellings of plain values, which the IR's messages and the
//! printer reproduce.

/// Python's `repr()` of a str: single quotes unless the text holds a single
/// quote and no double one; backslash, the quote, tab, newline and return
/// escaped, other control characters as `\xNN`, `\uNNNN` or `\UNNNNNNNN`.
/// Outside ASCII a character is kept unless it is a control character.
pub fn repr_str(s: &str) -> String {
    let quote = if s.contains('\'') && !s.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut out = String::with_capacity(s.len() + 2);
    out.push(quote);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\t' => out.push_str("\\t"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c if c.is_control() => {
                let n = c as u32;
                out.push_str(&match n {
                    0..=0xff => format!("\\x{n:02x}"),
                    0x100..=0xffff => format!("\\u{n:04x}"),
                    _ => format!("\\U{n:08x}"),
                });
            }
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}

/// Python's `repr()` of a float: the shortest digits that read back, in
/// fixed notation for decimal exponents −4 ≤ e < 16 (always with a
/// fraction, `2.0`), else `d.ddde±XX` (`1e+16`, `1e-05`).
pub fn repr_float(x: f64) -> String {
    if x.is_nan() {
        return "nan".into();
    }
    if x.is_infinite() {
        return if x > 0.0 { "inf" } else { "-inf" }.into();
    }
    let sci = format!("{:e}", x.abs());
    let (mantissa, exp) = sci.split_once('e').expect("a float spelled with e");
    let exp: i32 = exp.parse().expect("a decimal exponent");
    let digits: String = mantissa.chars().filter(|c| *c != '.').collect();
    let sign = if x.is_sign_negative() { "-" } else { "" };
    if (-4..16).contains(&exp) {
        format!("{sign}{}", fixed(&digits, exp))
    } else {
        let exp_sign = if exp < 0 { '-' } else { '+' };
        format!("{sign}{mantissa}e{exp_sign}{:02}", exp.unsigned_abs())
    }
}

/// Digits `d₀d₁…` with decimal exponent `exp` (the value d₀.d₁… × 10^exp)
/// in fixed notation, with at least one digit after the point.
fn fixed(digits: &str, exp: i32) -> String {
    if exp < 0 {
        return format!("0.{}{digits}", "0".repeat((-exp - 1) as usize));
    }
    let point = exp as usize + 1;
    if digits.len() <= point {
        format!("{digits}{}.0", "0".repeat(point - digits.len()))
    } else {
        format!("{}.{}", &digits[..point], &digits[point..])
    }
}

/// Python's spelling of a tuple whose elements are already spelled:
/// `()`, `(a,)`, `(a, b)`.
pub fn tuple(items: &[String]) -> String {
    match items {
        [one] => format!("({one},)"),
        _ => format!("({})", items.join(", ")),
    }
}

/// Python's spelling of a bool.
pub fn bool(b: bool) -> &'static str {
    if b {
        "True"
    } else {
        "False"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn floats_spell_as_python_repr() {
        let cases = [
            (0.1, "0.1"),
            (2.0, "2.0"),
            (1e16, "1e+16"),
            (1.5e16, "1.5e+16"),
            (1e-5, "1e-05"),
            (1e-4, "0.0001"),
            (-0.0, "-0.0"),
            (123456789.0, "123456789.0"),
            (9999999999999998.0, "9999999999999998.0"),
            (1.5, "1.5"),
            (f64::INFINITY, "inf"),
            (f64::NEG_INFINITY, "-inf"),
            (f64::NAN, "nan"),
            (5e-324, "5e-324"),
            (1.7976931348623157e308, "1.7976931348623157e+308"),
        ];
        for (x, text) in cases {
            assert_eq!(repr_float(x), text, "{x:e}");
        }
    }

    #[test]
    fn strs_spell_as_python_repr() {
        assert_eq!(repr_str("vm"), "'vm'");
        assert_eq!(repr_str("it's"), "\"it's\"");
        assert_eq!(repr_str("'\""), "'\\'\"'");
        assert_eq!(repr_str("a\\b\n\x01é"), "'a\\\\b\\n\\x01é'");
    }

    #[test]
    fn tuples_spell_as_python() {
        assert_eq!(tuple(&[]), "()");
        assert_eq!(tuple(&["1".into()]), "(1,)");
        assert_eq!(tuple(&["1".into(), "2".into()]), "(1, 2)");
    }
}
