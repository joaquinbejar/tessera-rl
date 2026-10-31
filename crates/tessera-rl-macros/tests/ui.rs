// SPDX-License-Identifier: MIT OR Apache-2.0

//! Compile-pass and compile-fail tests for the derives (ADR-0029).

#[test]
fn test_derives_ui_cases() {
    let cases = trybuild::TestCases::new();
    cases.pass("tests/ui/pass/*.rs");
    cases.compile_fail("tests/ui/fail/*.rs");
}
