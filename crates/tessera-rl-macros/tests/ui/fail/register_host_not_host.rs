// SPDX-License-Identifier: MIT OR Apache-2.0

// `register_host!` requires `HostEnvironment`; a first-party `Environment`
// is not one.
tessera_rl::register_host!(tessera_rl::classic::Grid, id = "example/Grid-v1");

fn main() {}
