# SPDX-License-Identifier: MIT OR Apache-2.0
from tessera_rl import pz

env = pz.parallel_env("tessera/Tag-v1", max_episode_steps=50)
observations, infos = env.reset(seed=1)
print("agents:", env.agents)

steps = 0
while env.agents:
    # Each agent samples a legal action from its own mask.
    actions = {a: env.action_space(a).sample(observations[a]["action_mask"]) for a in env.agents}
    observations, rewards, terminations, truncations, infos = env.step(actions)
    steps += 1
print("episode length:", steps)
env.close()
