// SPDX-License-Identifier: MIT OR Apache-2.0

//! An exact decimal parser for the fixed-point types (ADR-0065 item 3):
//! `Fx32::from_decimal` and `Fx64::from_decimal`. Integer arithmetic only,
//! no floating point, so a config value given as a decimal string means
//! the same raw value on every platform.
//!
//! Grammar: an optional sign (`-` or `+`), one or more ASCII digits, and an
//! optional `.` followed by one or more ASCII digits. No exponent,
//! separators or whitespace. A value is accepted only when it is exactly
//! representable: `value * 2^frac` is an integer that fits the raw type.

use core::fmt;

/// Why a decimal string is not an exact fixed-point value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DecimalError {
    /// Not a decimal of the accepted grammar.
    Syntax,
    /// Outside the type's range.
    OutOfRange,
    /// In range, but not a multiple of `2^-FRAC_BITS`.
    Inexact,
}

impl fmt::Display for DecimalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Syntax => "not a decimal number (digits, an optional sign and fraction)",
            Self::OutOfRange => "outside the fixed-point range",
            Self::Inexact => "not exactly representable in the fixed-point format",
        })
    }
}

impl core::error::Error for DecimalError {}

/// Most integer digits read before the value is out of range for any
/// fixed-point type (`Fx64` holds below 2^31, ten digits).
const MAX_INT_DIGITS: usize = 30;

/// The raw value `s * 2^frac`, exact, of a type with `bits` raw bits
/// (two's complement) and `frac` fractional bits (`frac < bits <= 64`).
///
/// Precedence: [`DecimalError::Syntax`] first, then
/// [`DecimalError::OutOfRange`] for a value outside the type's range,
/// exact or not (decided from the digits, before exactness), then
/// [`DecimalError::Inexact`] for an in-range value that is not a multiple
/// of `2^-frac`.
pub(crate) fn parse(s: &str, frac: u32, bits: u32) -> Result<i128, DecimalError> {
    let (negative, digits) = match s.as_bytes() {
        [b'-', rest @ ..] => (true, rest),
        [b'+', rest @ ..] | rest => (false, rest),
    };
    let (int, fraction) = match digits.iter().position(|&b| b == b'.') {
        Some(dot) => {
            let (int, fraction) = digits.split_at(dot);
            match fraction {
                [_, rest @ ..] if !rest.is_empty() => (int, rest),
                _ => return Err(DecimalError::Syntax),
            }
        }
        None => (digits, &[][..]),
    };
    if int.is_empty() || !int.iter().chain(fraction).all(u8::is_ascii_digit) {
        return Err(DecimalError::Syntax);
    }
    // Leading zeros of the integer part and trailing zeros of the fraction
    // change nothing.
    let int = match int.iter().position(|&b| b != b'0') {
        Some(first) => int.get(first..).unwrap_or_default(),
        None => &[],
    };
    let fraction = match fraction.iter().rposition(|&b| b != b'0') {
        Some(last) => fraction.get(..=last).unwrap_or_default(),
        None => &[],
    };
    if int.len() > MAX_INT_DIGITS {
        return Err(DecimalError::OutOfRange);
    }
    let int = int
        .iter()
        .try_fold(0i128, |acc, &d| {
            acc.checked_mul(10)?.checked_add(i128::from(d - b'0'))
        })
        .ok_or(DecimalError::OutOfRange)?;
    if out_of_range(negative, int, fraction, frac, bits)? {
        return Err(DecimalError::OutOfRange);
    }
    // After trimming, the last fraction digit is not 0, so the fraction
    // `F / 10^k` reduces to an odd numerator over `2^k` at best: exact only
    // if `k <= frac`.
    let k = u32::try_from(fraction.len()).map_err(|_| DecimalError::Inexact)?;
    if k > frac {
        return Err(DecimalError::Inexact);
    }
    // `F < 10^k <= 10^32 < 2^107`: no overflow.
    let f = fraction
        .iter()
        .try_fold(0u128, |acc, &d| {
            acc.checked_mul(10)?.checked_add(u128::from(d - b'0'))
        })
        .ok_or(DecimalError::Inexact)?;
    // `F / 10^k * 2^frac = (F / 5^k) * 2^(frac - k)`: exact iff `5^k | F`.
    let five = 5u128.checked_pow(k).ok_or(DecimalError::Inexact)?;
    if f % five != 0 {
        return Err(DecimalError::Inexact);
    }
    let frac_raw = (f / five)
        .checked_shl(frac - k)
        .and_then(|v| i128::try_from(v).ok())
        .ok_or(DecimalError::Inexact)?;
    // In range (checked above), so these cannot overflow.
    let magnitude = int
        .checked_shl(frac)
        .filter(|v| v >> frac == int)
        .and_then(|v| v.checked_add(frac_raw))
        .ok_or(DecimalError::OutOfRange)?;
    Ok(if negative { -magnitude } else { magnitude })
}

/// Whether `±int.fraction` lies outside `[-2^(bits-1), 2^(bits-1) - 1]`
/// raw units of `2^-frac`, exactly, from the digits (no rounding).
///
/// The largest value is `2^(bits-1-frac) - 2^-frac`: an integer part below
/// `2^(bits-1-frac)` and, at the integer part `2^(bits-1-frac) - 1`, a
/// fraction of at most `1 - 2^-frac`, whose decimal digits are finite. The
/// smallest is `-2^(bits-1-frac)`, an integer.
fn out_of_range(
    negative: bool,
    int: i128,
    fraction: &[u8],
    frac: u32,
    bits: u32,
) -> Result<bool, DecimalError> {
    let whole = bits
        .checked_sub(1)
        .and_then(|b| b.checked_sub(frac))
        .and_then(|e| 1i128.checked_shl(e))
        .ok_or(DecimalError::OutOfRange)?;
    if negative {
        return Ok(int > whole || (int == whole && !fraction.is_empty()));
    }
    if int < whole - 1 {
        return Ok(false);
    }
    if int > whole - 1 {
        return Ok(true);
    }
    // `1 - 2^-frac = (2^frac - 1) * 5^frac / 10^frac`: `frac` digits.
    let top = (1u128 << frac)
        .checked_sub(1)
        .and_then(|m| m.checked_mul(5u128.checked_pow(frac)?))
        .ok_or(DecimalError::OutOfRange)?;
    let width = usize::try_from(frac).map_err(|_| DecimalError::OutOfRange)?;
    let top = format!("{top:0width$}");
    // Compare the fractions digit by digit, the shorter padded with zeros.
    let len = top.len().max(fraction.len());
    let digit = |s: &[u8], i: usize| s.get(i).copied().unwrap_or(b'0');
    for i in 0..len {
        let (a, b) = (digit(fraction, i), digit(top.as_bytes(), i));
        if a != b {
            return Ok(a > b);
        }
    }
    Ok(false)
}

#[cfg(test)]
mod tests {
    use super::{DecimalError, parse};

    #[test]
    fn test_decimal_parse_exact_values() {
        assert_eq!(parse("0", 16, 32), Ok(0));
        assert_eq!(parse("-0", 16, 32), Ok(0));
        assert_eq!(parse("+1", 16, 32), Ok(1 << 16));
        assert_eq!(parse("1.5", 16, 32), Ok(3 << 15));
        assert_eq!(parse("-2.25", 16, 32), Ok(-(9 << 14)));
        assert_eq!(parse("007.50000", 16, 32), Ok(15 << 15));
        // 2^-16 needs 16 fraction digits.
        assert_eq!(parse("0.0000152587890625", 16, 32), Ok(1));
        assert_eq!(parse("0.00000000023283064365386962890625", 32, 64), Ok(1));
    }

    #[test]
    fn test_decimal_parse_inexact_values() {
        assert_eq!(parse("0.1", 16, 32), Err(DecimalError::Inexact));
        assert_eq!(
            parse("0.00000762939453125", 16, 32),
            Err(DecimalError::Inexact)
        );
        assert_eq!(
            parse("0.000000000116415321826934814453125", 32, 64),
            Err(DecimalError::Inexact)
        );
    }

    #[test]
    fn test_decimal_parse_syntax_errors() {
        for bad in [
            "", "-", "+", ".5", "1.", "1e3", "1_000", " 1", "1 ", "--1", "0x10", "1.2.3", "١",
        ] {
            assert_eq!(parse(bad, 16, 32), Err(DecimalError::Syntax), "{bad:?}");
        }
    }

    #[test]
    fn test_decimal_parse_huge_integer_is_out_of_range() {
        assert_eq!(
            parse(&"9".repeat(31), 16, 32),
            Err(DecimalError::OutOfRange)
        );
        assert_eq!(
            parse(&format!("1{}", "0".repeat(29)), 32, 64),
            Err(DecimalError::OutOfRange)
        );
    }

    #[test]
    fn test_decimal_parse_trailing_zeros_are_trimmed() {
        // Seventeen fraction digits, all but one zeros: exactly 0.5.
        assert_eq!(parse("0.50000000000000000", 16, 32), Ok(1 << 15));
        assert_eq!(
            parse("-2.2500000000000000000000000000000000000", 32, 64),
            Ok(-(9 << 30))
        );
    }

    #[test]
    fn test_decimal_parse_out_of_range_takes_precedence_over_inexact() {
        // Q16.16: [-32768, 32767.9999847412109375].
        assert_eq!(
            parse("32767.9999847412109375", 16, 32),
            Ok(i128::from(i32::MAX))
        );
        assert_eq!(parse("32767.9999", 16, 32), Err(DecimalError::Inexact));
        assert_eq!(parse("32767.99999", 16, 32), Err(DecimalError::OutOfRange));
        assert_eq!(
            parse("32767.99998474121093751", 16, 32),
            Err(DecimalError::OutOfRange)
        );
        assert_eq!(parse("32768.1", 16, 32), Err(DecimalError::OutOfRange));
        assert_eq!(parse("40000.1", 16, 32), Err(DecimalError::OutOfRange));
        assert_eq!(parse("-32768", 16, 32), Ok(i128::from(i32::MIN)));
        assert_eq!(
            parse("-32768.0000000001", 16, 32),
            Err(DecimalError::OutOfRange)
        );
        assert_eq!(parse("-32767.1", 16, 32), Err(DecimalError::Inexact));
        // Syntax first.
        assert_eq!(parse("99999999.x", 16, 32), Err(DecimalError::Syntax));
    }
}
