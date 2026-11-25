# SPDX-License-Identifier: MIT OR Apache-2.0
"""Tier T1s learning protocols (#595): T1's training, unchanged, with a
sampled (stochastic) evaluation instead of T1's greedy one, for protocols
such as Gather whose sampling policy learns while the argmax of a not yet
sharp policy locks into loops (#593).

- Training is ``reference_training.train`` through
  ``learning_protocols._factories``: the same vector environment, SAME_STEP,
  flattening, hyperparameters and seeding as T1. No training code is copied.
- Only the evaluation differs. For the duration of one call
  ``reference_training.evaluate`` is replaced by ``evaluate_sampled`` and
  restored in ``finally``. ``reference_training.train`` calls ``evaluate``
  as a module global, looked up at call time, so the replacement is the one
  it runs. A module guard refuses a nested or concurrent swap.
- ``evaluate_sampled`` uses T1's evaluation environment and reset seeds
  (``EVAL_SEED_BASE + i``) and episode end (terminated or truncated), and
  samples each action from the policy's ``Categorical`` with a
  ``torch.Generator`` seeded per evaluation episode (``episode_seed``), apart
  from training's RNG. Scores are raw environment returns.

Fingerprint: a T1s protocol hashes ``reference_training``, T1's training
path in ``learning_protocols`` and this module. So any edit to T1's training
moves the T1s fingerprints too (T1s runs that training); editing this module
moves only T1s.
"""

from __future__ import annotations

import contextlib
import threading

import numpy as np

#: Held while ``reference_training.evaluate`` is swapped: one swap at a time
#: in a process (the factories of ``_factories`` are module globals too).
_SWAP = threading.Lock()


def episode_seed(seed: int, eval_seed_base: int, episode: int) -> int:
    """The sampling generator's seed of one evaluation episode: a function of
    the training seed and the episode, distinct for distinct pairs."""
    return (eval_seed_base + episode) * 1_000_003 + seed


def evaluate_sampled(rt, agent, episodes: int, seed: int) -> list[float]:
    """``episodes`` episodes of the protocol's evaluation environment
    (``rt.make_single``, which ``_factories`` points at the protocol), reset
    on ``EVAL_SEED_BASE + i``, each action sampled from the policy."""
    import torch  # noqa: PLC0415
    from torch.distributions.categorical import Categorical  # noqa: PLC0415

    env = rt.make_single("tessera")
    out = []
    for i in range(episodes):
        gen = torch.Generator().manual_seed(episode_seed(seed, rt.EVAL_SEED_BASE, i))
        o, _ = env.reset(seed=rt.EVAL_SEED_BASE + i)
        total, done = 0.0, False
        while not done:
            with torch.no_grad():
                logits = agent.actor(torch.tensor(np.asarray(o), dtype=torch.float32))
            probs = Categorical(logits=logits).probs
            action = int(torch.multinomial(probs, 1, generator=gen))
            o, r, term, trunc, _ = env.step(action)
            total += float(r)
            done = term or trunc
        out.append(total)
    env.close()
    return out


@contextlib.contextmanager
def _sampled_evaluation(rt, seed: int):
    """``rt.evaluate`` replaced by the sampled evaluator for ``seed``,
    restored on exit, also when the run raises. Refuses a nested or
    concurrent swap."""
    if not _SWAP.acquire(blocking=False):
        raise RuntimeError("a sampled evaluation is already in progress in this process")
    saved = rt.evaluate
    try:
        rt.evaluate = lambda backend, agent, episodes: evaluate_sampled(rt, agent, episodes, seed)
        yield rt
    finally:
        rt.evaluate = saved
        _SWAP.release()


def train(env_id: str, seed: int, h, eval_episodes: int) -> dict:
    """One T1 training run of ``env_id`` with the sampled evaluation; the
    same fields as ``reference_training.train``."""
    import learning_protocols as lp  # noqa: PLC0415

    with lp._factories(env_id) as rt, _sampled_evaluation(rt, seed):
        return rt.train("tessera", seed, h, eval_episodes)
