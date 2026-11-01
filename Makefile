# SPDX-License-Identifier: MIT OR Apache-2.0

# Wheels target the macOS support floor (SPEC 7.1: macOS 13), not
# maturin's default of macOS 11. Ignored on other platforms.
export MACOSX_DEPLOYMENT_TARGET := 13.0

.PHONY: fmt fmt-check clippy test doc readme readme-check pre-push deny check-targets compat-test py-entries py-callouts py-test py-test-hooks py-test-torch pre-push-torch gym-equivalence reference-training learning-test learning-calibrate bench bench-quick bench-kernels gungraun gungraun-bless connect4-calibration

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
	cargo readme -r crates/tessera-rl -t $(CURDIR)/README.tpl --no-title --no-license --no-indent-headings > README.md

readme-check:
	cargo readme -r crates/tessera-rl -t $(CURDIR)/README.tpl --no-title --no-license --no-indent-headings | diff -u README.md -

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
	MATURIN_PEP517_ARGS="--features test-hooks" uv sync --locked --python 3.11 --reinstall-package tessera-rl; \
	status=$$?; \
	if [ $$status -eq 0 ]; then uv run --locked --python 3.11 pytest -q tests/python/test_fp_boundary.py; status=$$?; fi; \
	uv sync --locked --python 3.11 --reinstall-package tessera-rl || status=1; \
	exit $$status

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

# SPEC 14.2 reference training: PPO on tessera/CartPole-v1 through the
# Gymnasium adapter, Gymnasium's CartPole-v1 as the control and broken
# adapters as negative controls (nightly CI,
# about 15 minutes; not in pre-push). JSON report in reference-training.json.
reference-training:
	uv sync --locked --python 3.11 --group test --group torch
	uv run --locked --python 3.11 --group test --group torch python tools/reference_training.py --out reference-training.json

# SPEC 18 learning test (nightly): the reference PPO on seeds 1-5 against
# the calibrated threshold (compat/learning/cartpole.json); red only after
# two consecutive failing nights (PREVIOUS names last night's result).
PREVIOUS ?=
learning-test:
	uv sync --locked --python 3.11 --group test --group torch
	uv run --locked --python 3.11 --group test --group torch python tools/learning_test.py night --calibration compat/learning/cartpole.json $(if $(PREVIOUS),--previous $(PREVIOUS)) --out learning-test.json

# Re-calibrates the learning test (a reviewed re-bless, like gungraun):
# seeds 101-140 in 4 single-threaded shards, on Linux x86-64.
learning-calibrate:
	uv sync --locked --python 3.11 --group test --group torch
	rm -f learning-cal-*.json
	pids=""; for k in 0 1 2 3; do a=$$((101 + k * 10)); \
	  uv run --locked --python 3.11 --group test --group torch python tools/learning_test.py scores --total-steps 150000 --seeds $$(seq $$a $$((a + 9))) --out learning-cal-$$k.json & \
	  pids="$$pids $$!"; \
	done; status=0; for p in $$pids; do wait $$p || status=1; done; \
	if [ $$status -ne 0 ]; then echo "a calibration shard failed" >&2; exit 1; fi
	uv run --locked --python 3.11 --group test --group torch python tools/learning_test.py calibrate --out compat/learning/cartpole.json learning-cal-0.json learning-cal-1.json learning-cal-2.json learning-cal-3.json

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

# The full ADR-0042 calibration of the ported P2-G1 bot, including the
# games against the perfect solver (about a minute with 14 threads,
# a few minutes on xaskar). Not in pre-push; run it on any change to
# tessera-rl-games/src/connect4/bot.rs or board.rs.
connect4-calibration:
	cargo test --release --locked -p tessera-rl-games --test bot -- --include-ignored

# Criterion benches of the kernels.
bench-kernels:
	cargo bench -p tessera-rl-classic -p tessera-rl-games -p tessera-rl-engine --locked

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
