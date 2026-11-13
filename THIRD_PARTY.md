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

## CleanRL `ppo.py`

- **Where:** `tools/reference_training.py` (the PPO: network, rollout,
  GAE and clipped update).
- **Upstream:** <https://github.com/vwxyzjn/cleanrl>, `cleanrl/ppo.py`.
- **License:** MIT, text below.
- **Use:** adapted, not vendored verbatim: the structure, the update and
  the CartPole hyperparameters follow `ppo.py`. The changes are the two
  environment backends, the greedy evaluation, the JSON report and the
  gate, and the removal of the logging and the command-line flags of the
  original. A validation tool only (ADR-0018): Tessera ships no trainer.

```text
MIT License

Copyright (c) 2019 CleanRL developers

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## sha2 0.10 (crates.io dependency, not vendored)

- **Where:** a dependency of `crates/tessera-rl-onnx` behind its optional
  `onnx` feature, with `default-features = false`; nothing is copied into
  this repository.
- **Upstream:** <https://github.com/RustCrypto/hashes>, crate `sha2` on
  crates.io.
- **License:** MIT OR Apache-2.0.
- **Use:** the content sha256 of the ONNX Runtime library and of the model
  bytes, the identity and provenance of ADR-0057 sections 3 and 4.
