// SPDX-License-Identifier: MIT OR Apache-2.0

//! The engine holds the `TESSERA_BUILD_ID` of this very build. A stale value
//! would be a wrong identity with no error. Driven with changing ids by
//! `scripts/check_build_id.sh`; ignored otherwise, since it needs that script's
//! `EXPECT_BUILD_ID`.

#[test]
#[ignore = "run by scripts/check_build_id.sh"]
fn test_persist_build_id_is_the_one_this_build_was_given() {
    let expected = std::env::var("EXPECT_BUILD_ID").ok();
    assert_eq!(tessera_rl_engine::BUILD_ID, expected.as_deref());
}
