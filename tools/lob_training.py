#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""The reference PPO protocol for ``tessera/LimitOrderBook-v1`` (#381;
SPEC 16.1; design ``docs/reviews/381/DESIGN.md`` section 5; protocol table
``compat/learning/protocols.toml`` entry ``lob``), adapted from CleanRL's
``ppo.py`` as ``tools/reference_training.py`` is (MIT License, Copyright
(c) 2019 CleanRL developers; entry in ``THIRD_PARTY.md``).

Setup: one PettingZoo Parallel environment (``tessera_rl.pz``). Slot 0 is
the learner. Slots 1 to 7 are ADR-0079 heuristic slots (#470): the engine
plays the environment's own baseline mix there (``baseline_of(slot)``:
market makers, noise and momentum traders, the design's section 5 mix),
from each slot's information state. ``agents="learner"`` hides them, so the
learner's dicts hold only ``agent_0``.

Policy: an MLP over the learner's observation (every leaf flattened, each
value ``sign(x) * log1p(|x|)``) with four masked heads sampled in order:
``kind``; ``offset`` under the sampled kind; ``size`` under kind and
offset; ``cancel`` under kind. The log-probability and entropy are the sums
over the four heads at the sampled parents.

Score: the mean episode return of slot 0 over ``eval_episodes`` episodes on
fixed evaluation seeds, against the same in-engine opponents. Two baselines
on the same seeds, both reported:

- (a) the **pass reference**: a noise trader in slot 0, played here in
  Python (with probability 1/4 a legal action drawn head by head through
  the conditional masks, otherwise the no-op), as in #381;
- (b) for information only: the engine's own baseline in slot 0 (a market
  maker), from a pool where every slot is a heuristic.

**Pass:** over the protocol's seeds, the lower
bound of a bootstrap 95% interval of ``mean(trained) - baseline`` is above
0. A truncation (the last step) is treated as the episode's end.

This script is runnable (``--total-steps`` overrides the budget for a
smoke run), but no CI job runs the full protocol, as for the RTS: the
budget is a reviewed, owner-approved run.

Usage::

    python tools/lob_training.py --seeds 5 --total-steps 2000000 --eval-episodes 10
"""

from __future__ import annotations

import argparse
import json
import pathlib
import tomllib
from dataclasses import asdict, dataclass

import numpy as np

ENV_ID = "tessera/LimitOrderBook-v1"
PROTOCOLS_FILE = pathlib.Path(__file__).resolve().parents[1] / "compat" / "learning" / "protocols.toml"
#: Head sizes and the reshapes of each head's conditional mask.
NVEC = (6, 9, 4, 8)
SHAPES = ((6,), (6, 9), (6, 9, 4), (6, 8))
LEARNER = "agent_0"
#: Fixed evaluation seeds, apart from every training seed.
EVAL_SEED = 1_000_003


@dataclass(frozen=True)
class Hyper:
    """CleanRL ``ppo.py`` defaults, one environment of ``num_steps``."""

    total_steps: int = 2_000_000
    num_steps: int = 512
    learning_rate: float = 2.5e-4
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden: int = 128


def protocol() -> dict:
    """The ``lob`` table of the protocol file."""
    return tomllib.loads(PROTOCOLS_FILE.read_text())["lob"]


#: The trading agents. It must equal ``LobConfig``'s default (8): with
#: ``agents="learner"``, ``pz.parallel_env``'s ``agents`` argument is the
#: learner filter, so the environment's own ``agents`` key cannot be passed
#: alongside it, and the pool runs the default count.
AGENTS = 8
#: Slots 1 to 7 are in-engine heuristic opponents (ADR-0079, #470); a dict
#: map leaves the other slots learners.
OPPONENTS = {s: "heuristic" for s in range(1, AGENTS)}


def _config() -> dict:
    cfg = dict(protocol()["config"])
    if cfg.pop("agents", AGENTS) != AGENTS:
        raise ValueError(f"the lob protocol runs LobConfig's default {AGENTS} agents (see AGENTS)")
    return cfg


def make_env(seed: int):
    """The learner's environment: slot 0 visible, slots 1 to 7 played by the
    engine."""
    from tessera_rl import pz  # noqa: PLC0415

    return pz.parallel_env(ENV_ID, seed=seed, policy_map=OPPONENTS, agents="learner", **_config())


def make_all_heuristic_env(seed: int):
    """Every slot played by the engine, slot 0 included (baseline (b))."""
    from tessera_rl import pz  # noqa: PLC0415

    return pz.parallel_env(ENV_ID, seed=seed, policy_map={s: "heuristic" for s in range(AGENTS)}, agents=AGENTS, **_config())


def masks_of(info) -> tuple[np.ndarray, ...]:
    """The four conditional masks of one agent, as bool arrays in
    ``SHAPES``."""
    return tuple(np.asarray(m, dtype=bool).reshape(shape) for m, shape in zip(info["action_masks"], SHAPES, strict=True))


def legal_action(rng: np.random.Generator, masks) -> np.ndarray:
    """A uniformly drawn legal action, head by head."""
    kind, offset, size, cancel = masks
    k = int(rng.choice(np.flatnonzero(kind)))
    o = int(rng.choice(np.flatnonzero(offset[k])))
    s = int(rng.choice(np.flatnonzero(size[k, o])))
    c = int(rng.choice(np.flatnonzero(cancel[k])))
    return np.asarray([k, o, s, c], dtype=np.int64)


def noise_action(rng: np.random.Generator, masks) -> np.ndarray:
    """The noise trader: a legal action one step in four, else the no-op."""
    if rng.random() < 0.25:
        return legal_action(rng, masks)
    return np.zeros(4, dtype=np.int64)


def features(obs: dict) -> np.ndarray:
    """Every leaf of one agent's observation, flattened in key order and
    squashed with ``sign(x) * log1p(|x|)``."""
    flat = np.concatenate([np.asarray(obs[k], dtype=np.float64).reshape(-1) for k in sorted(obs)])
    return (np.sign(flat) * np.log1p(np.abs(flat))).astype(np.float32)


def build_agent(obs_dim: int, hidden: int):
    import torch  # noqa: PLC0415
    from torch import nn  # noqa: PLC0415

    class Agent(nn.Module):
        def __init__(self):
            super().__init__()
            self.body = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())
            self.value = nn.Linear(hidden, 1)
            self.kind = nn.Linear(hidden, 6)
            self.offset = nn.Linear(hidden, 6 * 9)
            self.size = nn.Linear(hidden, 6 * 9 * 4)
            self.cancel = nn.Linear(hidden, 6 * 8)

        def heads(self, x, masks, action=None):
            """Samples (or scores ``action``) through the four masked
            heads; returns the action, the summed log-prob and entropy."""
            h = self.body(x)
            n = x.shape[0]
            rows = torch.arange(n)
            kind_m, off_m, size_m, can_m = masks

            def dist(logits, mask):
                logits = torch.where(mask, logits, torch.full_like(logits, -1e8))
                return torch.distributions.Categorical(logits=logits)

            d0 = dist(self.kind(h), kind_m)
            k = d0.sample() if action is None else action[:, 0]
            d1 = dist(self.offset(h).view(n, 6, 9)[rows, k], off_m[rows, k])
            o = d1.sample() if action is None else action[:, 1]
            d2 = dist(self.size(h).view(n, 6, 9, 4)[rows, k, o], size_m[rows, k, o])
            s = d2.sample() if action is None else action[:, 2]
            d3 = dist(self.cancel(h).view(n, 6, 8)[rows, k], can_m[rows, k])
            c = d3.sample() if action is None else action[:, 3]
            act = torch.stack([k, o, s, c], dim=1)
            logp = d0.log_prob(k) + d1.log_prob(o) + d2.log_prob(s) + d3.log_prob(c)
            ent = d0.entropy() + d1.entropy() + d2.entropy() + d3.entropy()
            return act, logp, ent, self.value(h).squeeze(-1)

    return Agent()


def _masks_tensor(masks_list):
    import torch  # noqa: PLC0415

    return tuple(torch.as_tensor(np.stack([m[i] for m in masks_list])) for i in range(4))


def episode_scores(policy, episodes: int, seed: int) -> list[float]:
    """Slot 0's episode returns on ``episodes`` evaluation episodes:
    ``policy(obs, masks, rng)`` gives its action."""
    scores = []
    for ep in range(episodes):
        env = make_env(seed + ep)
        rng = np.random.default_rng(seed + ep)
        obs, infos = env.reset()
        total = 0.0
        while LEARNER in env.agents:
            acts = {LEARNER: policy(obs[LEARNER], masks_of(infos[LEARNER]), rng)}
            obs, rewards, _, _, infos = env.step(acts)
            total += float(rewards.get(LEARNER, 0.0))
        env.close()
        scores.append(total)
    return scores


def baseline_scores(episodes: int) -> list[float]:
    """Baseline (a), the pass reference: the noise trader's scores in slot 0
    on the evaluation seeds, against the in-engine opponents."""
    return episode_scores(lambda _obs, masks, rng: noise_action(rng, masks), episodes, EVAL_SEED)


def engine_baseline_scores(episodes: int) -> list[float]:
    """Baseline (b), for information: slot 0 played by the engine's own
    baseline (a market maker) on the evaluation seeds."""
    scores = []
    for ep in range(episodes):
        env = make_all_heuristic_env(EVAL_SEED + ep)
        env.reset()
        total = 0.0
        while LEARNER in env.agents:
            # Every slot is a heuristic: the engine ignores these rows.
            _, rewards, _, _, _ = env.step({a: np.zeros(4, dtype=np.int64) for a in env.agents})
            total += float(rewards.get(LEARNER, 0.0))
        env.close()
        scores.append(total)
    return scores


def train(seed: int, hyper: Hyper, eval_episodes: int) -> dict:
    """One protocol run: PPO for slot 0 against the in-engine opponents, then the
    evaluation scores of the trained policy."""
    import torch  # noqa: PLC0415

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    env = make_env(seed)
    obs, infos = env.reset()
    agent = build_agent(features(obs[LEARNER]).shape[0], hyper.hidden)
    optimizer = torch.optim.Adam(agent.parameters(), lr=hyper.learning_rate, eps=1e-5)
    updates = max(1, hyper.total_steps // hyper.num_steps)
    episode = 0
    for update in range(updates):
        if hyper.anneal_lr:
            optimizer.param_groups[0]["lr"] = (1.0 - update / updates) * hyper.learning_rate
        xs, ms, acts, logps, rews, dones, vals = [], [], [], [], [], [], []
        for _ in range(hyper.num_steps):
            x = torch.as_tensor(features(obs[LEARNER]))[None]
            m = masks_of(infos[LEARNER])
            with torch.no_grad():
                a, lp, _, v = agent.heads(x, _masks_tensor([m]))
            step_acts = {LEARNER: a[0].numpy().astype(np.int64)}
            obs, rewards, terms, truncs, infos = env.step(step_acts)
            done = LEARNER not in env.agents or terms.get(LEARNER, False) or truncs.get(LEARNER, False)
            xs.append(x[0])
            ms.append(m)
            acts.append(a[0])
            logps.append(lp[0])
            vals.append(v[0])
            rews.append(float(rewards.get(LEARNER, 0.0)))
            dones.append(float(done))
            if done:
                env.close()
                episode += 1
                env = make_env(seed * 1_000_003 + episode)
                obs, infos = env.reset()
        with torch.no_grad():
            next_v = agent.heads(torch.as_tensor(features(obs[LEARNER]))[None], _masks_tensor([masks_of(infos[LEARNER])]))[3][0]
        values = torch.stack(vals)
        rewards_t = torch.as_tensor(rews)
        dones_t = torch.as_tensor(dones)
        adv = torch.zeros_like(rewards_t)
        last = 0.0
        for t in reversed(range(hyper.num_steps)):
            nv = next_v if t == hyper.num_steps - 1 else values[t + 1]
            nonterminal = 1.0 - dones_t[t]
            delta = rewards_t[t] + hyper.gamma * nv * nonterminal - values[t]
            last = delta + hyper.gamma * hyper.gae_lambda * nonterminal * last
            adv[t] = last
        returns = adv + values
        b_x, b_a, b_lp = torch.stack(xs), torch.stack(acts), torch.stack(logps)
        b_m = _masks_tensor(ms)
        idx = np.arange(hyper.num_steps)
        mb = hyper.num_steps // hyper.num_minibatches
        for _ in range(hyper.update_epochs):
            rng.shuffle(idx)
            for start in range(0, hyper.num_steps, mb):
                j = torch.as_tensor(idx[start : start + mb])
                _, new_lp, ent, new_v = agent.heads(b_x[j], tuple(t[j] for t in b_m), b_a[j])
                ratio = (new_lp - b_lp[j]).exp()
                a_j = adv[j]
                a_j = (a_j - a_j.mean()) / (a_j.std() + 1e-8)
                pg = torch.max(-a_j * ratio, -a_j * ratio.clamp(1 - hyper.clip_coef, 1 + hyper.clip_coef)).mean()
                v_loss = 0.5 * ((new_v - returns[j]) ** 2).mean()
                loss = pg - hyper.ent_coef * ent.mean() + hyper.vf_coef * v_loss
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), hyper.max_grad_norm)
                optimizer.step()
    env.close()

    def policy(o, masks, _rng):
        with torch.no_grad():
            a, _, _, _ = agent.heads(torch.as_tensor(features(o))[None], _masks_tensor([masks]))
        return a[0].numpy().astype(np.int64)

    return {"seed": seed, "scores": episode_scores(policy, eval_episodes, EVAL_SEED)}


def passes(trained: list[float], baseline: float, resamples: int = 10_000) -> dict:
    """The pass criterion: the lower bound of a bootstrap 95% interval of
    ``mean(trained) - baseline`` over seeds is above 0."""
    rng = np.random.default_rng(0)
    arr = np.asarray(trained, dtype=np.float64)
    means = rng.choice(arr, size=(resamples, arr.size), replace=True).mean(axis=1) - baseline
    low = float(np.quantile(means, 0.025))
    return {"effect": float(arr.mean() - baseline), "low": low, "pass": low > 0.0}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--seeds", type=int, default=protocol()["seeds"])
    p.add_argument("--total-steps", type=int, default=protocol()["total_steps"])
    p.add_argument("--eval-episodes", type=int, default=protocol()["eval_episodes"])
    args = p.parse_args(argv)
    hyper = Hyper(**{**protocol()["hyper"], "total_steps": args.total_steps})
    runs = [train(seed, hyper, args.eval_episodes) for seed in range(args.seeds)]
    baseline = float(np.mean(baseline_scores(args.eval_episodes)))
    engine_baseline = float(np.mean(engine_baseline_scores(args.eval_episodes)))
    trained = [float(np.mean(r["scores"])) for r in runs]
    report = {
        "env": ENV_ID,
        "opponents": "ADR-0079 heuristic slots 1 to 7 (#470)",
        "hyper": asdict(hyper),
        "runs": runs,
        "baseline": baseline,
        "engine_baseline": engine_baseline,
        **passes(trained, baseline),
    }
    print(json.dumps(report, indent=2))  # noqa: T201
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
