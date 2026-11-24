// SPDX-License-Identifier: MIT OR Apache-2.0

// The `deprecated(...)` form compiles with a version and a note (ADR-0080).
tessera_rl::register!(
    tessera_rl::classic::Grid,
    id = "tessera/Grid-v1",
    deprecated(since = "0.4.0", note = "use tessera/Grid-v2"),
);

fn main() {}
