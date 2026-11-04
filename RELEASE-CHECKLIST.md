<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->

# Release checklist: v0.1.0

The maintainer runs these steps in order. A step that fails stops the
release.

## 1. Freeze

- [ ] All release PRs are merged: the facade prelude and `debug-checks`, the
      README, and this CHANGELOG and checklist.
- [ ] Record the frozen `main` SHA. Nothing merges after it until the tag.
- [ ] `make pre-push` is green on a clean checkout of that SHA.
- [ ] The full CI matrix is green on that SHA.
- [ ] Issue #40's evidence (the downstream test selection on an immutable
      commit) is recorded against that SHA.
- [ ] External consumer check: a throwaway crate outside the repository,
      with `tessera-rl = { git = "<repo>", rev = "<frozen SHA>" }` and no
      lockfile, builds and runs the downstream toy environment.

## 2. Packages

Publish order, each depending only on the ones before it:
`tessera-rl-num`, `tessera-rl-macros`, `tessera-rl-core`,
`tessera-rl-engine`, `tessera-rl-classic`, `tessera-rl-games`,
`tessera-rl-search`, `tessera-rl`.

- [ ] `cargo package --locked -p <each of the 8>` packages and verifies
      every crate on the frozen SHA.
- [ ] Each package contains only its `src/`, `tests/`, `benches/`,
      `README.md` and manifest. Path-only dev-dependencies
      (`tessera-rl-connect4-ref`, and the facade and core in
      `tessera-rl-macros`) are stripped from the published manifests.
- [ ] The facade's package README is the full repository README.
- [ ] `cargo publish -p <crate>` in the order above. The six crates already
      on crates.io at 0.0.1 are owned by the maintainer; `tessera-rl-games`
      and `tessera-rl-search` are new names.
- [ ] After publishing: a throwaway crate with `tessera-rl = "0.1.0"`
      resolves from crates.io and builds the downstream toy environment.

## 3. Tag and release

- [ ] Set the CHANGELOG date for 0.1.0.
- [ ] Create the annotated tag `v0.1.0` on the frozen SHA and push it.
- [ ] Create the GitHub release from the CHANGELOG section.
- [ ] docs.rs builds every crate.

## 4. Public repository

Before the repository is made public:

- [ ] Secret scan of the full history, all refs: no findings.
- [ ] Scan of tracked files and history for internal hostnames, IP
      addresses, local paths and private service names, each finding
      classified.
- [ ] Self-hosted runners cannot run code from outside contributors (see
      the CI note below).
- [ ] Decide whether the specification and the ADRs are published. They
      are not tracked in git, so a public reader cannot open the
      `SPEC` and `ADR` references in code comments.

CI note: every job of `ci.yml` runs on self-hosted runners and is triggered
by `pull_request`. On a public repository, a pull request from a fork would
run its code on those machines. Close this before the repository is public.

## 5. After the release

- [ ] Announce the release with its stated limits (CHANGELOG, "Not
      included" and "Known deviations").
- [ ] Open the next development section in the CHANGELOG.
