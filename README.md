<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->

# Tessera

Tessera is a Rust engine for writing and running reinforcement learning
environments, with cheap snapshot, restore and fork, bit-exact determinism
and a native multi-agent model. It is exposed to Python through Gymnasium,
PettingZoo and PufferLib compatible interfaces.

This crate is the facade: it re-exports the workspace crates, following
the layered workspace layout of SPEC section 4.1 and ADR-0021.

Most authors need only the `prelude`:

```rust
use tessera_rl::prelude::*;
```

# Features

- `debug-checks`: strict checks at the validation boundary (SPEC 5.5),
  forwarded to the engine; `engine::DEBUG_CHECKS` reports whether it
  is on.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.
