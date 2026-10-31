# SPDX-License-Identifier: MIT OR Apache-2.0

.PHONY: fmt fmt-check clippy test doc readme readme-check pre-push deny check-targets

fmt:
	cargo fmt --all

fmt-check:
	cargo fmt --all -- --check

clippy:
	cargo clippy --workspace --all-targets --all-features --locked -- -D warnings

test:
	cargo test --workspace --all-features --locked
	cargo test --workspace --locked
	# Release with every feature: strict checks without debug assertions.
	cargo test --workspace --release --all-features --locked

doc:
	RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --all-features --locked

readme:
	cargo readme -r crates/tessera-rl -t ../../README.tpl --no-title --no-license --no-indent-headings > README.md

readme-check:
	cargo readme -r crates/tessera-rl -t ../../README.tpl --no-title --no-license --no-indent-headings | diff -u README.md -

pre-push: fmt-check clippy test doc readme-check

deny:
	cargo deny check

# The Python tool environment (uv, Python 3.11) matches compat/targets.toml.
# ENV=main (default) or ENV=pufferlib (its own uv project in compat/pufferlib).
ENV ?= main
check-targets:
ifeq ($(ENV),main)
	uv sync --locked --python 3.11 --all-groups
	uv run --locked --python 3.11 python scripts/check_targets.py --require-all main
else
	uv sync --locked --python 3.11 --project compat/$(ENV)
	uv run --locked --python 3.11 --project compat/$(ENV) python scripts/check_targets.py --require-all $(ENV)
endif
