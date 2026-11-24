# Multi-agent

Multi-agent environments have agent slots, per-agent observations and
rewards, and action masks. Through the PettingZoo adapter they are a
standard `ParallelEnv`: every live agent acts each step, and each agent's
observation carries its `action_mask`.

<!-- snippet: multi_agent.py -->

`tessera/Tag-v1` is a pursuit game on a grid: when pursuers catch an evader,
the evader gets -1 and is removed, and every pursuer on its cell gets +1.
