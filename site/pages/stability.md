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
- the host-environment API (`HostEnvironment`, `register_host!` and their
  types), whose place in the stable surface is still open;
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
  `tessera_rl.HostStep`, for the same open question as the Rust host API.

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
changes may ship in any release. What 1.0 will promise is still being
decided; this page will state it then.
