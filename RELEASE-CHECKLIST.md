<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->

# Release checklist: v0.1.0

The maintainer runs these steps in order. A step that fails stops the
release.

## 1. Freeze

- [ ] All release PRs are merged: the facade prelude and `debug-checks`, the
      README, and this CHANGELOG and checklist.
- [ ] The last PR before the freeze sets the release date in the CHANGELOG
      heading (`## [0.1.0] - YYYY-MM-DD`). Nothing changes the tree after
      the freeze: the tag, the packages and the release are all built from
      the frozen SHA.
- [ ] Record the frozen `main` SHA. Nothing merges after it until the tag.
- [ ] `make pre-push` is green on a clean checkout of that SHA.
- [ ] The full CI matrix is green on that SHA.
- [ ] Issue #40's evidence (the downstream test selection on an immutable
      commit) is recorded against that SHA.
- [ ] External consumer check: a throwaway crate outside the repository,
      with `tessera-rl = { git = "<repo>", rev = "<frozen SHA>" }` and no
      lockfile, builds and runs the downstream toy environment.

## 2. Packages

The eight crates depend on each other at 0.1.0, which is not on crates.io
yet, so a single crate cannot be packaged or dry-run on its own before the
ones it depends on are published. Package, verify and publish them
together, in one cargo command: cargo orders them by dependency
(`tessera-rl-num`, `tessera-rl-core`, `tessera-rl-macros`,
`tessera-rl-engine`, `tessera-rl-classic`, `tessera-rl-games`,
`tessera-rl-search`, `tessera-rl`), verifies each against the others
through a local overlay, and on a real publish waits for each to appear in
the index before the next.

From a fresh clone of the frozen SHA, with a fresh target directory (a
stale `target/package` from another revision can break verification):

```text
git clone <repo> release && cd release && git checkout <frozen SHA>
export CARGO_TARGET_DIR="$PWD/../release-target"
cargo publish --dry-run --locked \
  -p tessera-rl-num -p tessera-rl-macros -p tessera-rl-core \
  -p tessera-rl-engine -p tessera-rl-classic -p tessera-rl-games \
  -p tessera-rl-search -p tessera-rl
```

- [ ] The dry run packages and verifies all eight crates and ends with
      "aborting upload due to dry run" for each.
- [ ] Each package contains only its `src/`, `tests/`, `benches/`,
      `README.md` and manifest (`tar -tzf
      $CARGO_TARGET_DIR/package/<crate>-0.1.0.crate`). Path-only
      dev-dependencies (`tessera-rl-connect4-ref`, and the facade and core
      in `tessera-rl-macros`) are absent from the published manifests.
- [ ] The facade's package README is the full repository README.
- [ ] The same command without `--dry-run` publishes the eight crates. The
      six crates already on crates.io at 0.0.1 are owned by the maintainer;
      `tessera-rl-games` and `tessera-rl-search` are new names.
- [ ] After publishing: a throwaway crate with `tessera-rl = "0.1.0"`
      resolves from crates.io and builds the downstream toy environment.

## 3. Tag and release

- [ ] Create the annotated tag `v0.1.0` on the frozen SHA and push it.
- [ ] Create the GitHub release from the CHANGELOG section.
- [ ] docs.rs builds every crate.

## 4. Public repository

Before the repository is made public:

- [ ] Secret scan of the full history, all refs: no findings. The audit
      for this release scanned the local and `origin` refs only. Before the
      flip, fetch the pull request refs too (`git fetch origin
      '+refs/pull/*:refs/remotes/pull/*'`) and rescan, since they become
      public with the repository.
- [ ] Scan of tracked files and history for internal hostnames, IP
      addresses, local paths and private service names, each finding
      classified.
- [ ] The CI guard is merged: no self-hosted job runs for a pull request
      from a fork (see the CI note below).
- [ ] The persistence and host credentials of the three self-hosted
      runners (xaskar, m4-linux-arm64, m4) have been reviewed: what a job
      can reach on each host, and what it can leave behind.

The flip, by the coordinator, in this order:

1. [ ] Disable GitHub Actions on the repository.
2. [ ] Make the repository public.
3. [ ] Set workflow approval to **all outside contributors** (not only
       first-time contributors).
4. [ ] Read the setting back and confirm it.
5. [ ] Re-enable GitHub Actions.
6. [ ] If a fork is available, test once with an unmodified fork pull
       request: no self-hosted job may run.

Approver rule: by default a workflow run from a fork is never approved. A
maintainer reads the contribution and pushes it to a branch of this
repository, where CI runs.
- [ ] The specification, the ADRs and the agent notes stay out of the
      repository (decided for 0.1.0); the CHANGELOG states that code
      comments cite them.

CI note: every job of `ci.yml` runs on self-hosted runners. The guard
only skips them for a fork pull request that leaves the workflow unchanged:
a fork's pull request runs the workflow files of its own head, so it can
edit the guard or add a workflow. The approval setting is the control that
must stay on.

## 5. After the release

- [ ] Announce the release with its stated limits (CHANGELOG, "Not
      included" and "Known deviations").
- [ ] Open the next development section in the CHANGELOG.
- [ ] Follow-up: a GitHub-hosted CI job for pull requests from forks, once
      hosted runners can be tested on the public repository.
