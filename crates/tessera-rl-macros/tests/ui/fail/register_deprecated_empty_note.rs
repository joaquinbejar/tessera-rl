// SPDX-License-Identifier: MIT OR Apache-2.0

// A deprecation needs a note (ADR-0080 item 4).
tessera_rl::register!(
    tessera_rl::classic::Grid,
    id = "tessera/Grid-v1",
    deprecated(since = "0.4.0", note = ""),
);

fn main() {}
