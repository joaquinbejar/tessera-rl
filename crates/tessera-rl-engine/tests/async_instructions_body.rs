// SPDX-License-Identifier: MIT OR Apache-2.0

//! Runs the `async_instructions` bench body natively (issue #83): the bench
//! only runs under Valgrind on Linux, so its setup and round are checked here.

#![expect(clippy::unwrap_used, reason = "test setup unwraps")]

mod async_round;
mod common;

#[test]
fn test_async_instructions_body_round_receives_every_id() {
    let (pool, ids, actions) = async_round::round(async_round::round_256());
    assert_eq!(ids.len(), 256);
    // A second round on the same input works too (the bench returns it).
    let (_pool, _, _) = async_round::round((pool, ids, actions));
}
