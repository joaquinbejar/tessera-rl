// SPDX-License-Identifier: MIT OR Apache-2.0

// `register!` requires `Persist`.
struct NotAnEnv;

tessera_rl::register!(NotAnEnv, id = "tessera/NotAnEnv-v1");

fn main() {}
