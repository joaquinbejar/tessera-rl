# Getting started

A pool holds `num_envs` instances of one environment and steps them together.
Outputs are arrays with one row per instance. They are borrowed by default:
read-only views that stay valid until the next call on the pool.

<!-- snippet: getting_started.py -->

The code above is `site/snippets/getting_started.py` in the repository, and the
test suite runs it.

## The adapters

- `tessera_rl.gym.make(env_id, num_envs)` gives a Gymnasium `VectorEnv`
  (owned outputs by default).
- `tessera_rl.pz.parallel_env(env_id)` gives a PettingZoo `ParallelEnv` for
  multi-agent environments.
- `tessera_rl.puffer` gives a PufferLib environment.
