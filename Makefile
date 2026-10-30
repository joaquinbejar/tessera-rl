# SPDX-License-Identifier: MIT OR Apache-2.0

.PHONY: fmt fmt-check clippy test doc readme readme-check pre-push deny

fmt:
	cargo fmt --all

fmt-check:
	cargo fmt --all -- --check

clippy:
	cargo clippy --workspace --all-targets --all-features --locked -- -D warnings

test:
	cargo test --workspace --all-features --locked

doc:
	RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --all-features --locked

readme:
	cargo readme -r crates/tessera-rl -t ../../README.tpl --no-title --no-license --no-indent-headings > README.md

readme-check:
	cargo readme -r crates/tessera-rl -t ../../README.tpl --no-title --no-license --no-indent-headings | diff -u README.md -

pre-push: fmt-check clippy test doc readme-check

deny:
	cargo deny check
