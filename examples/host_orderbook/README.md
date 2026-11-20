<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->

# Example: a host environment wheel (`example/OrderBook-v1`)

This example shows how a crate outside Tessera ships a host environment. A host environment keeps state on the heap and implements `HostEnvironment` (ADR-0062 and ADR-0065). Its wheel is found by `tessera_rl.make` through an entry point.

The environment is a one-trader order book:

- Each step, the trader holds, buys or sells at a price level.
- Resting orders live in a `Vec`.
- Prices are integer ticks, so the same seed and actions give the same outputs. CI runs every check here twice and compares hashes.

## Layout

| File | Role |
| --- | --- |
| `Cargo.toml` | Its own package and its own workspace. It depends on `tessera-rl-core`, `tessera-rl-macros` and `tessera-rl-pyffi` (feature `host`), here by path. A published crate depends on their releases. |
| `src/lib.rs` | `OrderBook: HostEnvironment`, then `register_host!` and `export_host_env!`. |
| `pyproject.toml` | A maturin project, built abi3 for Python 3.11 and later. It declares the `tessera_rl.envs` entry point. |
| `python/tessera_example_orderbook/` | The Python package that holds the compiled `_tessera_host` module. |

## Build and use

From the repository root:

```sh
make py-test-orderbook
```

It builds the first-party wheel and this one into one environment, then runs `tests/python/test_host_orderbook.py`.

In Python:

```python
import tessera_rl

env = tessera_rl.make("example/OrderBook-v1", num_envs=8, seed=1, episode=32)
step = env.reset()
```

The same id also works with the adapters:

- `tessera_rl.gym.make` and `tessera_rl.gym.TesseraEnv` (one agent);
- `tessera_rl.pz.parallel_env` and `tessera_rl.pz.aec_env`.

## Choosing an id

The namespace `example/` is used only by this example. An environment you publish needs a namespace of its own, and `tessera/` is reserved. The registry refuses an id registered twice, by any wheel.
