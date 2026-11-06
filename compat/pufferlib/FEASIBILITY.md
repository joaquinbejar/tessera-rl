<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->

# PufferLib native adapter: feasibility record (phase 2 M0(c))

**Decision (plan v2):** P2-M8 builds the native `PufferEnv` adapter in this existing profile, at the pins of `compat/targets.toml` `[env.pufferlib]`:

| Package | Pin |
| --- | --- |
| PufferLib | 3.0.0 |
| Gymnasium | 0.29.1 |
| PettingZoo | 1.24.1 |
| NumPy | 1.26.4 |

There is no new pin and no ADR-0027 change. Any upgrade is its own compatibility PR.

## Probe

`probe_native.py` (this directory) runs a minimal native `pufferlib.PufferEnv` over `tessera/CartPole-v1`.

To run it, build the wheel and use it from this profile:

```
uvx --from 'maturin>=1.16,<2' maturin build --release -m crates/tessera-rl-py/Cargo.toml -o <dir>
cd compat/pufferlib
uv run --locked --python 3.11 --with <dir>/<wheel> python probe_native.py
```

Result on macOS arm64, 2026-11-01, main `988b1a4` plus this PR:

```json
{"pufferlib": 3.0, "gymnasium": "0.29.1", "numpy": "1.26.4", "standalone_steps": true, "standalone_shares_memory": false, "serial_steps": true, "serial_agents": 8, "serial_obs_shape": [8, 4], "serial_hands_buffers": true}
```

## Findings

1. **Compatibility.** The Tessera abi3 wheel imports and runs next to NumPy 1.26.4 and Gymnasium 0.29.1.
2. **Construction.** A native `PufferEnv` with Tessera's layout constructs and steps, both standalone and under `pufferlib.vector.Serial`. The layout is:
   - observations: a single `Box` leaf;
   - actions: `Discrete`;
   - one agent per instance, so `num_agents = N * A`.
3. **Buffer ownership.** In PufferLib 3.0.0 the env writes into buffers that PufferLib owns:
   - `PufferEnv.__init__(buf)` calls `set_buffers`;
   - the vector backends always pass `buf` (`serial_hands_buffers: true`);
   - otherwise the env allocates its own.

   Tessera's outputs live in engine-owned storage (ADR-0003). So today each step costs one copy of observations, rewards and flags into PufferLib's buffers (`standalone_shares_memory: false`). "Zero copy" in P2-M8 therefore needs one of:
   - (a) the engine writing its outputs into caller-provided buffers. This is an output target that ADR-0003 does not cover yet and would need its own ADR before M8;
   - (b) a measured, documented single copy per step, with zero copy claimed only where the engine's own buffers are handed to PufferLib.

   M8 must pick one, with a benchmark.
4. **Multi-leaf observations.** PufferLib's native env requires `Box` observations, so Dict or Tuple observation spaces need a flattening adapter (ADR-0014 leaves are contiguous only when their dtypes match).

## Outcome (P2-M8)

Option (b): `tessera_rl.puffer` shares the engine's buffers on the native backend and copies once per step on `vector.Serial` and `vector.Multiprocessing`, which pass buffers they own. `bench_adapter.py` measures this; no performance claim rests on a development machine.
