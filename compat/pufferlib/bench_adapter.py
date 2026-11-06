# SPDX-License-Identifier: MIT OR Apache-2.0
"""Steps per second, for the record FEASIBILITY.md requires. A number from a
development machine is indicative only; no claim rests on it."""

import json
import multiprocessing
import sys
import time

import numpy as np
import pufferlib.vector

import tessera_rl as t
from tessera_rl.puffer import make_puffer


def rate(step, agents, choices, seconds=1.5):
    acts = np.random.default_rng(1).integers(0, choices, (64, agents), dtype=np.int32)
    done, i = 0, 0
    start = time.perf_counter()
    while time.perf_counter() - start < seconds:
        step(acts[i % 64])
        done += agents
        i += 1
    return round(done / (time.perf_counter() - start))


def main():
    multiprocessing.set_start_method("spawn")
    rows = []
    for env_id, choices, kws in [("tessera/CartPole-v1", 2, {}), ("tessera/Tag-v1", 5, {})]:
        for n in (1, 64, 4096):
            raw = t.make(env_id, num_envs=n, auto_reset="same_step")
            raw.reset(1)
            row = {"env": env_id, "n": n, "raw": rate(raw.step, n * raw.agents, choices)}
            raw.close()
            native = pufferlib.vector.make(make_puffer(env_id, num_envs=n))
            native.reset(seed=1)
            row["native"] = rate(native.step, native.num_agents, choices)
            native.close()
            serial = pufferlib.vector.make(make_puffer(env_id, num_envs=n // 2 or 1), backend=pufferlib.vector.Serial, num_envs=2)
            serial.reset(seed=1)
            row["serial_2"] = rate(serial.step, serial.num_agents, choices)
            serial.close()
            multi = pufferlib.vector.make(
                make_puffer(env_id, num_envs=n // 2 or 1), backend=pufferlib.vector.Multiprocessing, num_envs=2, num_workers=2
            )
            multi.reset(seed=1)
            row["multiprocessing_2"] = rate(multi.step, multi.num_agents, choices)
            multi.close()
            rows.append(row)
            sys.stdout.write(json.dumps(row) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
