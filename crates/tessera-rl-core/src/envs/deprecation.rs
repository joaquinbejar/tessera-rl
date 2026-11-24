// SPDX-License-Identifier: MIT OR Apache-2.0

//! Deprecation of a registered id (SPEC 16.1, ADR-0080 item 4).

/// A deprecated id's metadata, set by
/// [`register!`](crate::register)`(..., deprecated(since = "...", note = "..."))`.
///
/// Metadata only: the environment, its goldens and its replays are
/// unchanged and stay available within the major version (SPEC 16.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct Deprecation {
    since: &'static str,
    note: &'static str,
}

impl Deprecation {
    /// A deprecation since `since` (a `MAJOR.MINOR.PATCH` version, checked
    /// by [`register!`](crate::register) with [`is_semver`]) with `note`.
    #[must_use]
    pub const fn new(since: &'static str, note: &'static str) -> Self {
        Self { since, note }
    }

    /// The release that deprecated the id, `MAJOR.MINOR.PATCH`.
    #[must_use]
    pub const fn since(&self) -> &'static str {
        self.since
    }

    /// What to use instead, or why.
    #[must_use]
    pub const fn note(&self) -> &'static str {
        self.note
    }
}

/// Whether `s` is `MAJOR.MINOR.PATCH`: three dot-separated decimal numbers
/// without leading zeros (a lone `0` is allowed) and no pre-release or build
/// suffix. Usable in a `const` assertion.
#[must_use]
pub const fn is_semver(s: &str) -> bool {
    let mut rest = s.as_bytes();
    // Dots seen so far (at most two), whether the current part has a digit,
    // and whether it starts with `0`.
    let (mut one_dot, mut two_dots) = (false, false);
    let (mut has_digit, mut leading_zero) = (false, false);
    while let Some((&c, tail)) = rest.split_first() {
        if c.is_ascii_digit() {
            if leading_zero {
                return false;
            }
            leading_zero = !has_digit && c == b'0';
            has_digit = true;
        } else if c == b'.' {
            if !has_digit || two_dots {
                return false;
            }
            if one_dot {
                two_dots = true;
            } else {
                one_dot = true;
            }
            has_digit = false;
            leading_zero = false;
        } else {
            return false;
        }
        rest = tail;
    }
    two_dots && has_digit
}

#[cfg(test)]
mod tests {
    use super::is_semver;

    #[test]
    fn test_is_semver_plain_versions_true() {
        for v in ["0.4.0", "1.0.0", "10.20.30", "0.0.0"] {
            assert!(is_semver(v), "{v}");
        }
    }

    #[test]
    fn test_is_semver_malformed_false() {
        for v in [
            "",
            "1",
            "1.2",
            "1.2.3.4",
            "01.2.3",
            "1.02.3",
            "1.2.03",
            "1.2.3-rc1",
            "1.2.3+b",
            "v1.2.3",
            "1..3",
            "1.2.",
            ".1.2",
        ] {
            assert!(!is_semver(v), "{v}");
        }
    }
}
