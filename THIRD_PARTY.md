<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->

# Third-party code

Code from other projects that is vendored into this repository (ADR-0034).

## libm 0.2.16

- **Where:** `crates/tessera-rl-num/src/math/vendor/`
- **Upstream:** <https://github.com/rust-lang/compiler-builtins>, path `libm`,
  commit `dfd2203a4d6110820ad7bb65cafe1bf331a03a3d` (crate `libm` 0.2.16 on
  crates.io).
- **License:** MIT (copyright (c) 2018 Jorge Aparicio and the libm
  contributors). Several files also carry notices from FreeBSD msun and
  musl, which derive from Sun Microsystems' fdlibm ("Permission to use,
  copy, modify, and distribute this software is freely granted, provided
  that this notice is preserved"). Those notices are kept in each file.
- **Use:** math version 1 (ADR-0008). Kept verbatim except for the
  mechanical changes listed in each file's header. `scalbn.rs` follows the
  musl algorithm used by earlier `libm` releases and is checked against
  `libm` 0.2.16.

The full upstream license file, including the MIT text and the musl and
fdlibm notices, is kept verbatim at
`crates/tessera-rl-num/src/math/vendor/LICENSE.txt`.
