# {{crate_name}}: `{{env_id}}`

A [Tessera](https://github.com/joaquinbejar/tessera-rl) host environment,
scaffolded by `cargo tessera new` {{tool_version}}. It starts as a counter
game ({{agents}} agent(s), {{turns}} turns); replace it with your
environment in `src/lib.rs`.

## Layout

| File | Role |
| --- | --- |
| `Cargo.toml` | Its own package and workspace. The `python` feature builds the extension module. |
| `src/lib.rs` | The environment (`HostEnvironment`), `register_host!` and `export_host_env!`. |
| `tests/check.rs` | The Rust checks and the golden hashes (`tests/golden.json`). |
| `tests/test_adapters.py` | The Python adapter checks. |
| `benches/step.rs` | The criterion bench. |
| `pyproject.toml` | A maturin project with the `tessera_rl.envs` entry point. |
| `python/{{py_package}}/` | The Python package that holds the compiled `_tessera_host` module. |

## Dependencies

{{#path_deps}}
The Tessera crates come from a local checkout, by path (`--tessera-path`).
Before publishing, depend on a released version instead.
{{/path_deps}}
{{^path_deps}}
The Tessera crates are pinned to exactly `={{tool_version}}`, the version
of the `cargo-tessera` that scaffolded this crate. The host environment Rust
API is excluded from the 1.x stability promise (ADR-0094 item 3) and a minor
release may change it (host protocols 1 and 2, which the wheel speaks, are
stable), so move every `tessera-rl*` dependency to a new version together, and
run `cargo tessera check` after.
{{/path_deps}}

## Checks

```sh
cargo tessera check            # determinism, continuation, spaces, invariants, goldens
cargo tessera check --bless    # record tests/golden.json (commit it)
cargo tessera check --python   # also the Gymnasium and PettingZoo adapter tests
cargo tessera bench --quick    # steps per second on this machine
```

`check --python` needs `maturin` on the PATH (or `uv` with `--uv`). It
installs the wheel into the active virtual environment when `VIRTUAL_ENV`
is set, otherwise into a new one under `target/` that sees the system
site-packages: `tessera-rl`, `pytest`, `gymnasium` and `pettingzoo` must be
installed there.

The goldens hash every output of every call of a seeded run. When a change
to the outputs is intended, bump the version in the id and `ENV_VERSION`,
then re-bless.

## Use

```python
import tessera_rl

env = tessera_rl.make("{{env_id}}", num_envs=8, seed=1)
step = env.reset()
```

## Choosing an id

The namespace `tessera/` is reserved for first-party environments, and the
registry refuses an id registered twice, by any wheel.
