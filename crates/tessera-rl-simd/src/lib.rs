// SPDX-License-Identifier: MIT OR Apache-2.0

//! Audited lane types and once-resolved dispatch (ADR-0015).
//!
//! A kernel implements [`Kernel`] generically over
//! [`F64Lanes`](tessera_rl_num::math::lane::F64Lanes); [`Resolved::dispatch`]
//! runs it inside a `#[target_feature]` function of the level. The level types
//! are private, so safe code can only hold one inside [`Kernel::run`], which
//! is reached only after the level's features were detected.
//!
//! Levels: scalar, SSE4.2, AVX2 and NEON. AVX-512 has no native test runner
//! and is not selectable. No level enables `fma`.

mod dispatch;
mod lanes;

pub use dispatch::{Kernel, Level, LevelError, Resolved};
