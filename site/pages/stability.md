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

The Python API is pinned separately: its type stubs are checked against the
built module (`make py-typecheck`).

## How it is tracked

The repository keeps a text snapshot of that API, `api/tessera-rl.api`: one
line per item, with its signature, fields, variants and trait
implementations, including `Send` and `Sync`. `make facade-api` rebuilds it
from rustdoc and fails on any difference. A change is accepted only by
regenerating the file (`make facade-api-bless`) in a reviewed commit, so
the diff of `api/tessera-rl.api` shows each API change.

## Breaking changes between releases

Each release checks what changed in the API since the previous one: `git
diff <previous tag> -- api/tessera-rl.api`. A removed or changed line is a
candidate breaking change, and the CHANGELOG names it. Before 1.0 such
changes may ship in any release. What 1.0 will promise is still being
decided; this page will state it then.
