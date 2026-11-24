// SPDX-License-Identifier: MIT OR Apache-2.0

// A deprecation's `since` must be MAJOR.MINOR.PATCH (ADR-0080 item 4).
tessera_rl::register!(
    tessera_rl::classic::Grid,
    id = "tessera/Grid-v1",
    deprecated(since = "0.4", note = "use tessera/Grid-v2"),
);

fn main() {}
