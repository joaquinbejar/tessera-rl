# API stability

Tessera is before 1.0. Any 0.x release may change the API, and the
CHANGELOG says what changed. **Promises come with 1.0.** Until then, the
public API is tracked, so that every change to it is visible and reviewed.

## What is tracked

The Rust public API of the `tessera-rl` facade: every public item you can
reach from `tessera_rl` and `tessera_rl::prelude` with the default features,
plus what the `onnx` feature adds. Through the facade this covers the
member crates it re-exports (`tessera_rl::core`, `::engine`, `::num` and the
others).

Not tracked:

- items hidden from the docs, and names starting with `_`;
- test-only features;
- the host-environment Rust API (`HostEnvironment`, `register_host!` and
  their types), which stays outside the stable surface after 1.0 (see
  "Host environments" below);
- the tool, example and integration crates.

The Python API of the `tessera_rl` package: the names in `tessera_rl.__all__`,
and the modules `tessera_rl.gym`, `tessera_rl.pz`, `tessera_rl.puffer`,
`tessera_rl.replay` and `tessera_rl.render`, with the signatures of their
shipped type stubs (`.pyi`). The stubs are checked against the built module
(`make py-typecheck`), so the signatures tracked are those of the code.

Not tracked in Python:

- names starting with `_`, and private modules such as `tessera_rl._native`;
- the other public modules (`tessera_rl.eval`, `jax`, `masks`, `search`,
  `transforms` and `viewer`), whose place in the stable surface is still
  open;
- the host-environment wrappers `tessera_rl.HostEnv` and
  `tessera_rl.HostStep`, not tracked yet.

## How it is tracked

The repository keeps a text snapshot of each API:

- Rust: `api/tessera-rl.api`, one line per item, with its signature, fields,
  variants and trait implementations, including `Send` and `Sync`.
  `make facade-api` rebuilds it from rustdoc and fails on any difference.
- Python: `api/tessera-rl-py.api`, one line per class, method, attribute,
  function and constant, with its signature from the stubs. `make py-api`
  rebuilds it (it reads the stubs and builds nothing) and fails on any
  difference.

A change is accepted only by regenerating the file (`make facade-api-bless`
or `make py-api-bless`) in a reviewed commit, so the diff of the snapshot
shows each API change.

## Breaking changes between releases

Each release checks what changed in the API since the previous one: `git
diff <previous tag> -- api/tessera-rl.api api/tessera-rl-py.api`. A removed or changed line is a
candidate breaking change, and the CHANGELOG names it. Before 1.0 such
changes may ship in any release. From 1.0, a breaking change to the stable
surface ships only in a major release (ADR-0094 item 1), and the release
checklist confirms it.

## Deprecation

From 1.0, a stable item is deprecated for at least one minor release before
it is removed in the next major release (ADR-0094 item 5). In Rust it carries
`#[deprecated(since, note)]`, and in Python it raises a `DeprecationWarning`;
the note names the replacement. `EnvAssets`, superseded by `Asset<T>` config
fields (#418), is the example. Deprecated environment ids follow ADR-0080.

## Rust and Python versions

The minimum supported Rust version is the workspace's `rust-version`,
checked by CI's `msrv` job. From 1.0 it is at most 4 stable Rust releases
behind the newest stable at the time of each minor release, and it is raised
only in a minor release, with a CHANGELOG entry, never in a patch (ADR-0094
item 6). The Python floor (CPython 3.11, abi3 wheels) is raised the same
way: only in a minor release, with a CHANGELOG entry.

## Host environments

The host protocols that a host wheel speaks to `tessera_rl`, protocols 1 and
2 (SPEC 14.4), are stable from 1.0: a host wheel built against a protocol
works with every 1.x release that supports it, and 1.0 supports both
(ADR-0094 items 3 and 4). The Rust host API (`HostEnvironment`, `HostPool`,
`HostSnapshot`, `export_host_env!`, `register_host!`) and the
`tessera-rl-pyffi` crate are
outside the stable surface: a minor release may change them, with a
CHANGELOG entry, so a host crate that rebuilds against a newer Tessera may
need source changes. `tessera_rl.jax.ffi_handle` is not affected by this.
`cargo tessera new --kind env` scaffolds an environment on the stable
surface; `--kind host` scaffolds a host environment.
