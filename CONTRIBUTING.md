<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->

# Contributing

Contributions are welcome.

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes and ensure that the project still builds and all tests
   pass: `make pre-push` is the local gate (formatting, clippy with
   `-D warnings`, the Rust and Python tests, documentation and the README
   check).
4. Commit your changes and push your branch to your fork.
5. Submit a pull request to the main repository.

CI does not run for pull requests from forks: it runs on the maintainer's
own machines. A maintainer reviews the contribution and pushes it to a
branch of this repository, where CI runs.

Before you start:

- Open an issue before a design change, so it can be discussed first.
- Code, comments, docs, commits and pull requests are in English.
- Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/)
  (`feat:`, `fix:`, `docs:`, `test:`, `ci:`, `chore:`, with a scope where it
  helps).
- Keep a pull request to one concern.
- Every source file starts with the SPDX header
  `SPDX-License-Identifier: MIT OR Apache-2.0` in its comment syntax.
- After adding or changing a dependency, run `make deny` (cargo-deny:
  licences, advisories and sources).
- The README is generated from the facade crate's documentation and
  `README.tpl` with `cargo-readme` (`make readme`); edit those, not
  `README.md`.

Contributions are dual licensed under MIT OR Apache-2.0, as described in the
README.
