# Tessera

Tessera (`tessera-rl`) is an open-source Rust engine for writing and running
reinforcement learning environments, exposed to Python through Gymnasium,
PettingZoo and PufferLib compatible interfaces.

- **Snapshot, restore and fork** of environment state, cheap enough for search
  on the real environment.
- **Bit-exact determinism** across platforms and thread counts.
- **Native multi-agent**: agent slots, partial observability, action masks,
  simultaneous and sequential turns.
- **Performance** from batch-first storage, no allocation per step and
  zero-copy outputs.

## Install

```
pip install tessera-rl
```

Wheels are abi3 for CPython 3.11 and newer.

## Where next

- [Getting started](getting-started.md): make a pool, step it, snapshot it.
- Tutorials: [writing an environment](tutorials/writing-an-environment.md),
  [writing an `Environment`-trait environment](tutorials/writing-an-environment-trait.md),
  [snapshot, restore and fork](tutorials/snapshot-fork.md),
  [multi-agent](tutorials/multi-agent.md).
- [API reference](api.md): the Python API, and the Rust API on docs.rs.
- [Benchmarks](benchmarks.md): the benchmark dashboard.
