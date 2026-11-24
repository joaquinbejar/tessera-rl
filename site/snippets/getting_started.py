# SPDX-License-Identifier: MIT OR Apache-2.0
import numpy as np

import tessera_rl

env = tessera_rl.make("tessera/CartPole-v1", num_envs=8, seed=1, workers=0)
out = env.reset()
print(out.observations.keys(), out.rewards.shape)  # (num_envs, agents)

rng = np.random.default_rng(0)
for _ in range(10):
    actions = rng.integers(0, env.action_choices, size=(env.num_envs, env.agents), dtype=np.int32)
    out = env.step(actions)

# Snapshot the whole pool, step on, then restore: the state comes back exactly.
before = env.state_hash(0)
snap = env.snapshot_all()
env.step(np.zeros((env.num_envs, env.agents), dtype=np.int32))
env.restore_pool(snap)
assert env.state_hash(0) == before
env.close()
