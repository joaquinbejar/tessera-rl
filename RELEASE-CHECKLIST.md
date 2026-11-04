<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->

# Release checklist: v0.1.0

The maintainer runs these steps in order. A step that fails stops the
release. The tag, the published packages and the release all identify one
reviewed tree: the frozen SHA.

## 1. Freeze

- [ ] All release PRs are merged: the facade prelude and `debug-checks`, the
      README, the CI fork guard, and this CHANGELOG and checklist.
- [ ] The CHANGELOG heading carries the actual release date
      (`## [0.1.0] - 2026-11-04`). If the release slips, the last PR before
      the freeze corrects it. Nothing changes the tree after the freeze.
- [ ] Record the frozen `main` SHA. Nothing merges after it until the tag.
- [ ] `make pre-push` is green on a clean checkout of that SHA.
- [ ] The full CI matrix is green on that SHA.
- [ ] Issue #40's evidence (the downstream test selection on an immutable
      commit) is recorded against that SHA.
- [ ] External consumer check: a throwaway crate outside the repository,
      with `tessera-rl = { git = "<repo>", rev = "<frozen SHA>" }` and no
      lockfile, builds and runs the downstream toy environment.

## 2. Scans, on the frozen SHA, before any publication

- [ ] Secret scan of the full history of every ref, including the pull
      request refs, which become public with the repository:
      `git clone --mirror <repo>`, then
      `git fetch origin '+refs/pull/*:refs/remotes/pull/*'`, then gitleaks
      over all refs. No findings. (The earlier audit scanned the local and
      `origin` refs only.)
- [ ] Scan of the tracked files and the history for internal hostnames, IP
      addresses, local paths and private service names, each finding
      classified. Decided for 0.1.0: the short runner and machine names
      (`xaskar`, `swarmer02`, `m4-linux-arm64`, `m4`, `Studio`) stay; they
      carry no domain or address. Anything else found stops the release.

## 3. Packages

The eight crates depend on each other at 0.1.0, which is not on crates.io
yet, so a single crate cannot be packaged or dry-run on its own before the
ones it depends on are published (`cargo package -p tessera-rl-core` alone
fails: `tessera-rl-num ^0.1.0` is not on the registry). They are packaged,
verified and published together, in one cargo invocation: cargo orders them
by dependency (`tessera-rl-num`, `tessera-rl-core`, `tessera-rl-macros`,
`tessera-rl-engine`, `tessera-rl-classic`, `tessera-rl-games`,
`tessera-rl-search`, `tessera-rl`), verifies each against the others
through a local overlay registry, and on a real publish waits for each to
appear in the index before the next.

Always from a fresh clone of the frozen SHA, with a fresh target directory.
A `target/package/tmp-registry` left by an earlier packaging run of another
revision breaks verification with a cargo internal error ("no hash listed
for tessera-rl-num v0.1.0"); that is what failed in one review run, while a
clean run passed.

```text
git clone <repo> release && cd release && git checkout <frozen SHA>
export CARGO_TARGET_DIR="$PWD/../release-target"
cargo package --locked \
  -p tessera-rl-num -p tessera-rl-macros -p tessera-rl-core \
  -p tessera-rl-engine -p tessera-rl-classic -p tessera-rl-games \
  -p tessera-rl-search -p tessera-rl
cargo publish --dry-run --locked <the same eight -p flags>
```

- [ ] `cargo package` packages all eight and prints `Verifying` for each,
      with no error (not `--no-verify`).
- [ ] `cargo publish --dry-run` ends with "aborting upload due to dry run"
      for each of the eight.
- [ ] Package contents (`tar -tzf
      $CARGO_TARGET_DIR/package/<crate>-0.1.0.crate`): the crate's `src/`,
      `tests/`, `benches/`, `README.md`, the normalized `Cargo.toml`,
      `Cargo.toml.orig`, `Cargo.lock` and `.cargo_vcs_info.json`, plus the
      expected assets: the golden and reference data under `tests/` of
      `tessera-rl-classic` and `tessera-rl-games`, the trybuild cases under
      `tests/ui/` of `tessera-rl-macros`, and the vendored math source and
      its licence in `tessera-rl-num`. Nothing from `examples/`,
      `results/`, `tools/`, `docs/`, `compat/` or `.github/`.
- [ ] Path-only dev-dependencies (`tessera-rl-connect4-ref`, and the facade
      and core in `tessera-rl-macros`) are absent from the normalized
      manifests.
- [ ] The facade's package `README.md` is byte-identical to the repository
      README.
- [ ] The same `cargo publish` command without `--dry-run` publishes the
      eight crates. The six crates already on crates.io at 0.0.1 are owned
      by the maintainer; `tessera-rl-games` and `tessera-rl-search` are new
      names.
- [ ] After publishing: a throwaway crate with `tessera-rl = "0.1.0"`
      resolves from crates.io and builds the downstream toy environment.

## 4. Tag and release

- [ ] Create the annotated tag `v0.1.0` on the frozen SHA and push it.
- [ ] Create the GitHub release from the CHANGELOG section.
- [ ] docs.rs builds every crate.

## 5. Public repository

Before the repository is made public:

- [ ] The CI guard is merged: the self-hosted jobs are skipped for a pull
      request from a fork that leaves the workflow unchanged (see the CI
      note below).
- [ ] The persistence and host credentials of the three self-hosted
      runners (xaskar, m4-linux-arm64, m4) have been reviewed: what a job
      can reach on each host, and what it can leave behind.
- [ ] Decided for 0.1.0: the specification, the ADRs and the agent notes
      stay out of the repository. The CHANGELOG states that code, comments
      and documents cite them.

The flip, by the coordinator, in this order:

1. [ ] Disable GitHub Actions on the repository, and confirm that no
       workflow run is queued or running.
2. [ ] Make the repository public.
3. [ ] Set workflow approval to **all outside contributors** (not only
       first-time contributors).
4. [ ] Read the setting back and confirm it.
5. [ ] Re-enable GitHub Actions.
6. [ ] If a fork is available, test once with an unmodified fork pull
       request: no self-hosted job may run.

Fallback, if GitHub refuses step 3 while Actions are disabled: take the
three self-hosted runners offline, enable Actions, set the approval to all
outside contributors, read it back, and only then bring the runners back
online.

Approver rule: by default a workflow run from a fork is never approved. A
maintainer reads the contribution and pushes it to a branch of this
repository, where CI runs.

CI note: every job of `ci.yml` runs on self-hosted runners. The guard only
skips them for a fork pull request that leaves the workflow unchanged: a
fork's pull request runs the workflow files of its own head, so it can edit
the guard or add a workflow. The approval setting is the control that must
stay on.

## 6. After the release

- [ ] Announce the release with its stated limits (CHANGELOG, "Not
      included" and "Known deviations").
- [ ] Open the next development section in the CHANGELOG.
- [ ] Follow-up: a GitHub-hosted CI job for pull requests from forks, once
      hosted runners can be tested on the public repository.
