# SPDX-License-Identifier: MIT OR Apache-2.0

# Wheels target the macOS support floor (SPEC 7.1: macOS 13), not
# maturin's default of macOS 11. Ignored on other platforms.
export MACOSX_DEPLOYMENT_TARGET := 13.0

.PHONY: fmt fmt-check clippy test doc readme readme-check pre-push deny check-targets compat-test py-entries py-callouts py-test py-test-hooks py-test-torch pre-push-torch gym-equivalence bench bench-quick bench-kernels gungraun gungraun-bless

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

pre-push: fmt-check clippy test doc readme-check compat-test py-entries py-callouts py-test py-test-hooks

deny:
	cargo deny check

# The Python tool environment matches compat/targets.toml for a profile
# (b1, test, torch, conformance or pufferlib; default conformance).
PROFILE ?= conformance
check-targets:
ifeq ($(PROFILE),pufferlib)
	uv sync --locked --python 3.11 --project compat/pufferlib
	uv run --locked --python 3.11 --project compat/pufferlib python scripts/check_targets.py pufferlib
else ifeq ($(PROFILE),b1)
	uv sync --locked --python 3.11 --no-default-groups
	uv run --locked --python 3.11 --no-default-groups python scripts/check_targets.py b1
else ifeq ($(PROFILE),test)
	uv sync --locked --python 3.11 --only-group test
	uv run --locked --python 3.11 --only-group test python scripts/check_targets.py test
else ifeq ($(PROFILE),torch)
	uv sync --locked --python 3.11 --group test --group torch
	uv run --locked --python 3.11 --group test --group torch python scripts/check_targets.py torch
else
	uv sync --locked --python 3.11 --all-groups
	uv run --locked --python 3.11 --all-groups python scripts/check_targets.py $(PROFILE)
endif

# Tests of the compat tooling (the target checker).
compat-test:
	uv run --locked --python 3.11 --only-group test pytest -q tests/compat

# ADR-0041: no raw PyO3 registrations in tessera-rl-py.
py-entries:
	python3 scripts/check_py_entries.py

# ADR-0041 callouts: only ffi::capi calls the C API, and every call that
# can run Python code is a callout.
py-callouts:
	python3 scripts/check_py_callouts.py

# Builds the native package into the uv environment and runs the Python tests.
py-test:
	uv sync --locked --python 3.11
	uv run --locked --python 3.11 pytest -q tests/python

# The ADR-0041 unwind tests against a wheel built with the `test-hooks`
# feature (failure injection), then the default build again.
py-test-hooks:
	MATURIN_PEP517_ARGS="--features test-hooks" uv sync --locked --python 3.11 --reinstall-package tessera-rl
	uv run --locked --python 3.11 pytest -q tests/python/test_fp_boundary.py
	uv sync --locked --python 3.11 --reinstall-package tessera-rl

# The Python tests with PyTorch (marker `torch`), then the default
# environment again.
py-test-torch:
	uv sync --locked --python 3.11 --group test --group torch
	uv run --locked --python 3.11 --group test --group torch pytest -q tests/python
	uv sync --locked --python 3.11

# SPEC 18 CartPole equivalence with the pinned Gymnasium: JSON report
# (the gate itself runs in py-test, tests/python/test_gym_equivalence.py).
gym-equivalence:
	uv sync --locked --python 3.11
	uv run --locked --python 3.11 python tools/gym_equivalence.py

# Benchmark harness (ADR-0025, tools/bench): the full method, a JSON
# report per machine; `MACHINE` names a compat/machines.toml entry.
MACHINE ?=
RUNS ?= 1
bench:
	uv sync --locked --python 3.11
	mkdir -p target/bench
	uv run --locked --python 3.11 python tools/bench/bench.py $(if $(MACHINE),--machine $(MACHINE)) --runs $(RUNS) --out target/bench/report.json

# A short smoke run of every configuration (never valid for claims).
bench-quick:
	uv sync --locked --python 3.11
	uv run --locked --python 3.11 python tools/bench/bench.py --quick

# Criterion benches of the kernels.
bench-kernels:
	cargo bench -p tessera-rl-classic -p tessera-rl-engine --locked

# Instruction-count gate (ADR-0025 level 1, gungraun): compares with the
# committed baseline in benches/gungraun; Linux only (skipped elsewhere).
gungraun:
	scripts/gungraun.sh check

# Records a new instruction-count baseline (benches/gungraun/README.md says
# when and who approves).
gungraun-bless:
	scripts/gungraun.sh bless

# pre-push plus the PyTorch tests (downloads PyTorch on first use).
pre-push-torch: pre-push py-test-torch
