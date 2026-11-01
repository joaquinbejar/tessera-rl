# SPDX-License-Identifier: MIT OR Apache-2.0
"""Phase 2 M0(c) feasibility probe: a native ``pufferlib.PufferEnv`` over
``tessera/CartPole-v1`` in the pinned PufferLib 3.0.0 profile.

It checks, and prints as JSON:

- the pinned versions;
- that a native ``PufferEnv`` subclass constructs and steps with Tessera's
  layout (Box observations, Discrete actions, one agent per instance);
- that ``pufferlib.vector.Serial`` drives it and hands it buffers it owns
  (``buf`` is not None), which is where the engine's outputs must land;
- whether the observations PufferLib reads share memory with Tessera's
  borrowed views (zero-copy) or are a copy.

Run: ``uv run --locked --with <tessera wheel> python probe_native.py``.
"""

import json
import sys

import gymnasium
import numpy as np
import pufferlib
import pufferlib.vector

import tessera_rl._native as native


class TesseraCartPole(pufferlib.PufferEnv):
    def __init__(self, num_envs=1, buf=None, seed=0):
        self.single_observation_space = gymnasium.spaces.Box(-np.inf, np.inf, (4,), np.float32)
        self.single_action_space = gymnasium.spaces.Discrete(2)
        self.num_agents = num_envs
        self.got_buffers = buf is not None
        self._env = native.make_cartpole(num_envs=num_envs, seed=seed, outputs="borrowed")
        super().__init__(buf)

    def _views(self):
        out = self._env.outputs()
        return np.asarray(out["observations"][0]).reshape(self.num_agents, 4)

    def reset(self, seed=None):
        self._env.reset()
        obs = self._views()
        self.observations[:] = obs  # one copy into PufferLib's buffer
        return self.observations, [{}]

    def step(self, actions):
        self._env.step(np.asarray(actions, dtype="<i4").reshape(-1, 1).tobytes())
        out = self._env.outputs()
        self.observations[:] = np.asarray(out["observations"][0]).reshape(self.num_agents, 4)
        self.rewards[:] = np.asarray(out["rewards"]).reshape(-1)
        self.terminals[:] = np.asarray(out["episode_terminated"]).reshape(-1)
        self.truncations[:] = np.asarray(out["episode_truncated"]).reshape(-1)
        return self.observations, self.rewards, self.terminals, self.truncations, [{}]

    def close(self):
        self._env.close()


def main():
    report = {
        "pufferlib": getattr(pufferlib, "__version__", "?"),
        "gymnasium": gymnasium.__version__,
        "numpy": np.__version__,
    }
    env = TesseraCartPole(num_envs=4)
    obs, _ = env.reset()
    for _ in range(10):
        env.step(np.ones(4, dtype=np.int32))
    report["standalone_steps"] = True
    report["standalone_shares_memory"] = bool(np.shares_memory(env.observations, env._views()))
    env.close()

    vec = pufferlib.vector.make(TesseraCartPole, num_envs=2, backend=pufferlib.vector.Serial, env_kwargs={"num_envs": 4})
    obs, _ = vec.reset()
    for _ in range(10):
        obs, rew, term, trunc, info = vec.step(np.ones(vec.num_agents, dtype=np.int32))
    report["serial_steps"] = True
    report["serial_agents"] = int(vec.num_agents)
    report["serial_obs_shape"] = list(obs.shape)
    report["serial_hands_buffers"] = all(e.got_buffers for e in vec.envs)
    vec.close()
    sys.stdout.write(json.dumps(report) + "\n")


if __name__ == "__main__":
    main()
