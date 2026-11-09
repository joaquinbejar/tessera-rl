<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->

# tessera-rl-simd

Eight-lane `f64` types over `core::arch`, one per instruction set, selected
once at run time. This crate is the audited home of Tessera's SIMD `unsafe`.
It makes no performance claim, and nothing in the engine uses it yet.
