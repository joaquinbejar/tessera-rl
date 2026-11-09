// SPDX-License-Identifier: MIT OR Apache-2.0

//! Level detection and the `match` that enters a kernel.

#![expect(
    unsafe_code,
    reason = "calls target_feature entry points after detection"
)]

use core::fmt;

use tessera_rl_num::math::lane::F64Lanes;
use thiserror::Error;

use crate::lanes;

/// An instruction-set level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Level {
    /// Plain arrays: every platform.
    Scalar,
    /// SSE4.2, x86-64.
    Sse42,
    /// AVX2, x86-64.
    Avx2,
    /// NEON, aarch64.
    Neon,
}

impl Level {
    /// The name used by `TESSERA_SIMD`.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Scalar => "scalar",
            Self::Sse42 => "sse42",
            Self::Avx2 => "avx2",
            Self::Neon => "neon",
        }
    }

    fn present(self) -> bool {
        match self {
            Self::Scalar => true,
            #[cfg(target_arch = "x86_64")]
            Self::Sse42 => std::arch::is_x86_feature_detected!("sse4.2"),
            #[cfg(target_arch = "x86_64")]
            Self::Avx2 => {
                std::arch::is_x86_feature_detected!("avx")
                    && std::arch::is_x86_feature_detected!("avx2")
            }
            #[cfg(target_arch = "aarch64")]
            Self::Neon => std::arch::is_aarch64_feature_detected!("neon"),
            #[cfg(not(target_arch = "x86_64"))]
            Self::Sse42 | Self::Avx2 => false,
            #[cfg(not(target_arch = "aarch64"))]
            Self::Neon => false,
        }
    }
}

impl fmt::Display for Level {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// A level that cannot be selected.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum LevelError {
    /// This CPU lacks the level; there is no fallback.
    #[error("this CPU does not have SIMD level {0}")]
    Unavailable(Level),
    /// Not a level name, or a level without native testing such as `avx512`.
    #[error("{0:?} is not a selectable SIMD level")]
    Unknown(String),
}

/// A kernel generic over the lane type.
pub trait Kernel {
    /// The kernel's result.
    type Output;
    /// Runs the kernel. Implementations must be `#[inline(always)]`, or the
    /// lane operations leave the `target_feature` context.
    fn run<L: F64Lanes>(self) -> Self::Output;
}

/// A level whose features were checked on this CPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Resolved(Level);

impl Resolved {
    /// The widest level this CPU has.
    #[must_use]
    pub fn best() -> Self {
        let order = [Level::Avx2, Level::Sse42, Level::Neon];
        Self(
            order
                .into_iter()
                .find(|l| l.present())
                .unwrap_or(Level::Scalar),
        )
    }

    /// Exactly `level`, or an error if this CPU lacks it.
    ///
    /// # Errors
    /// [`LevelError::Unavailable`] without a fallback.
    pub fn force(level: Level) -> Result<Self, LevelError> {
        if level.present() {
            Ok(Self(level))
        } else {
            Err(LevelError::Unavailable(level))
        }
    }

    /// The level a `TESSERA_SIMD` value names; [`Self::best`] for `None`.
    ///
    /// # Errors
    /// [`LevelError`] for an unknown name or a level this CPU lacks.
    pub fn from_name(name: Option<&str>) -> Result<Self, LevelError> {
        let level = match name {
            None => return Ok(Self::best()),
            Some("scalar") => Level::Scalar,
            Some("sse42") => Level::Sse42,
            Some("avx2") => Level::Avx2,
            Some("neon") => Level::Neon,
            Some(other) => return Err(LevelError::Unknown(other.to_owned())),
        };
        Self::force(level)
    }

    /// Every level this CPU has, for tests and benches.
    pub fn available() -> impl Iterator<Item = Self> {
        [Level::Scalar, Level::Sse42, Level::Avx2, Level::Neon]
            .into_iter()
            .filter(|l| l.present())
            .map(Self)
    }

    /// The level.
    #[must_use]
    pub const fn level(self) -> Level {
        self.0
    }

    /// Runs `kernel` at this level. A `match`, without dynamic dispatch.
    #[inline]
    pub fn dispatch<K: Kernel>(self, kernel: K) -> K::Output {
        match self.0 {
            #[cfg(target_arch = "x86_64")]
            // SAFETY: `Resolved` holds this level only after `present`.
            Level::Avx2 => unsafe { lanes::run_avx2(kernel) },
            #[cfg(target_arch = "x86_64")]
            // SAFETY: `Resolved` holds this level only after `present`.
            Level::Sse42 => unsafe { lanes::run_sse42(kernel) },
            #[cfg(target_arch = "aarch64")]
            // SAFETY: `Resolved` holds this level only after `present`.
            Level::Neon => unsafe { lanes::run_neon(kernel) },
            _ => lanes::run_scalar(kernel),
        }
    }
}
