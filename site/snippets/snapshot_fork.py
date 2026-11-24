# SPDX-License-Identifier: MIT OR Apache-2.0
import numpy as np

import tessera_rl

# One instance to search from, plus one copy per action.
env = tessera_rl.make("tessera/CartPole-v1", num_envs=3, seed=7, workers=0, outputs="owned")
env.reset()

# Try each action on its own copy of instance 0 (CartPole has 2 actions).
env.fork(0, [1, 2])
actions = np.array([[0], [0], [1]], dtype=np.int32)  # instance 0 stays as the reference
out = env.step(actions)
print("reward per first action:", {a: float(out.rewards[1 + a, 0]) for a in range(2)})

# A snapshot restores the whole pool exactly.
snap = env.snapshot_all()
hashes = [env.state_hash(i) for i in range(env.num_envs)]
for _ in range(5):
    env.step(np.zeros((env.num_envs, env.agents), dtype=np.int32))
env.restore_pool(snap)
assert [env.state_hash(i) for i in range(env.num_envs)] == hashes
env.close()
