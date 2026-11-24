# SPDX-License-Identifier: MIT OR Apache-2.0
"""Tier T3a of the learning protocols (#370, design
docs/reviews/learning-370/DESIGN-T3-r1.md): PPO with action masks, one
learner, one discrete head.

The PPO is CleanRL's discrete ``ppo.py`` structure (MIT, ``THIRD_PARTY.md``),
as in ``reference_training`` (left untouched: its source is in other
protocols' fingerprints), with masked logits. Each step's mask comes from
``info["action_mask"]`` (SPEC 11.3; an environment without a mask is all
legal) and is stored with the step, so the update evaluates the same masked
distribution the action was sampled from. Rev's masking rules
(REVIEW-T3-design-r1):

1. Illegal logits are set to ``finfo.min``, never ``-inf``, and the entropy
   sums over legal actions only, so neither is NaN.
2. A row whose mask is all false (no legal action: a dead or pending slot)
   samples action 0 (the engine's illegal-action policy applies) and is left
   out of the loss.
3. Conditional heads (a child mask chosen by the sampled parent) are not in
   this tier: one head only. RTS comes with them, later.
4. Evaluation is greedy: the legal action with the largest logit, ties to
   the lowest index.

Observations that are not a 1-D box are flattened as in T1.
"""

from __future__ import annotations

import pathlib
import random
import sys
import time
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from learning_continuous import bootstrap_rewards  # noqa: E402


@dataclass
class MaskedHyper:
    """CleanRL ``ppo.py`` defaults (as ``reference_training.Hyper``)."""

    total_steps: int = 500_000
    num_envs: int = 4
    num_steps: int = 128
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


def masked_logits(logits, mask):
    """``logits`` with illegal entries at the dtype's minimum (rule 1)."""
    import torch  # noqa: PLC0415

    return torch.where(mask, logits, torch.full_like(logits, torch.finfo(logits.dtype).min))


def legal_entropy(logits, mask):
    """The entropy of the masked distribution, summed over legal actions
    only (rule 1): finite also where illegal probabilities are exactly 0."""
    import torch  # noqa: PLC0415

    logp = torch.log_softmax(masked_logits(logits, mask), dim=-1)
    return -(torch.where(mask, logp.exp() * logp, torch.zeros_like(logp))).sum(-1)


def greedy(logits, mask):
    """The legal action with the largest logit; ties to the lowest index
    (rule 4: ``argmax`` returns the first maximum). An all-false mask gives
    action 0."""
    import torch  # noqa: PLC0415

    return torch.argmax(masked_logits(logits, mask), dim=-1)


def _agent_class():
    import torch  # noqa: PLC0415
    from torch import nn  # noqa: PLC0415
    from torch.distributions.categorical import Categorical  # noqa: PLC0415

    def layer(linear, std=float(np.sqrt(2)), bias=0.0):
        nn.init.orthogonal_(linear.weight, std)
        nn.init.constant_(linear.bias, bias)
        return linear

    class MaskedAgent(nn.Module):
        def __init__(self, obs_dim: int, actions: int):
            super().__init__()
            self.critic = nn.Sequential(
                layer(nn.Linear(obs_dim, 64)), nn.Tanh(),
                layer(nn.Linear(64, 64)), nn.Tanh(),
                layer(nn.Linear(64, 1), std=1.0),
            )  # fmt: skip
            self.actor = nn.Sequential(
                layer(nn.Linear(obs_dim, 64)), nn.Tanh(),
                layer(nn.Linear(64, 64)), nn.Tanh(),
                layer(nn.Linear(64, actions), std=0.01),
            )  # fmt: skip

        def value(self, x):
            return self.critic(x)

        def act(self, x, mask, action=None):
            logits = self.actor(x)
            dist = Categorical(logits=masked_logits(logits, mask))
            if action is None:
                action = dist.sample()
            return action, dist.log_prob(action), legal_entropy(logits, mask), self.critic(x)

    return MaskedAgent


def live_rows(idx, masks):
    """The rows of ``idx`` that have at least one legal action (rule 2):
    the others are left out of the loss."""
    return idx[masks[idx].any(-1)]


def mask_of(info: dict, n: int, actions: int) -> np.ndarray:
    """The step's masks ``[n, actions]`` as bool; all legal without a mask."""
    if "action_mask" in info:
        return np.asarray(info["action_mask"], dtype=bool).reshape(n, actions)
    return np.ones((n, actions), dtype=bool)


def train(env_id: str, kwargs: dict, seed: int, h: MaskedHyper, eval_episodes: int, eval_seed_base: int) -> dict:
    """Trains one masked PPO agent on ``env_id`` (Tessera backend, extra
    ``kwargs`` for the environment, such as ``obs``) and evaluates it
    greedily; returns the same fields as ``reference_training.train``."""
    import gymnasium  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from torch import nn  # noqa: PLC0415

    import learning_protocols as lp  # noqa: PLC0415
    import tessera_rl.gym  # noqa: PLC0415

    if h.total_steps < h.num_envs * h.num_steps:
        raise ValueError(f"total_steps {h.total_steps} is less than one rollout ({h.num_envs * h.num_steps} steps)")
    if eval_episodes <= 0:
        raise ValueError(f"eval_episodes must be positive, got {eval_episodes}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    envs = tessera_rl.gym.make(env_id, num_envs=h.num_envs, seed=seed, autoreset_mode=gymnasium.vector.AutoresetMode.SAME_STEP, **kwargs)
    if lp._needs_flat(envs.single_observation_space):
        envs = lp._FlatVector(envs)
    envs = gymnasium.wrappers.vector.RecordEpisodeStatistics(envs)
    obs_dim = int(np.prod(envs.single_observation_space.shape))
    n_actions = int(envs.single_action_space.n)
    agent = _agent_class()(obs_dim, n_actions)
    optimizer = torch.optim.Adam(agent.parameters(), lr=h.learning_rate, eps=1e-5)

    batch = h.num_envs * h.num_steps
    minibatch = batch // h.num_minibatches
    updates = h.total_steps // batch
    obs = torch.zeros((h.num_steps, h.num_envs, obs_dim))
    masks = torch.zeros((h.num_steps, h.num_envs, n_actions), dtype=torch.bool)
    actions = torch.zeros((h.num_steps, h.num_envs))
    logprobs = torch.zeros((h.num_steps, h.num_envs))
    rewards = torch.zeros((h.num_steps, h.num_envs))
    dones = torch.zeros((h.num_steps, h.num_envs))
    values = torch.zeros((h.num_steps, h.num_envs))

    started = time.perf_counter()
    raw, info = envs.reset(seed=seed)
    next_obs = torch.tensor(np.asarray(raw, dtype=np.float32).reshape(h.num_envs, -1))
    next_mask = torch.tensor(mask_of(info, h.num_envs, n_actions))
    next_done = torch.zeros(h.num_envs)
    episodes = 0
    curve = []
    rng = np.random.default_rng(seed)

    def value_of(final):
        with torch.no_grad():
            return agent.value(torch.tensor(np.asarray(final, dtype=np.float32).reshape(len(final), -1))).flatten().numpy()

    for update in range(1, updates + 1):
        if h.anneal_lr:
            optimizer.param_groups[0]["lr"] = (1.0 - (update - 1.0) / updates) * h.learning_rate
        finished = []
        for step in range(h.num_steps):
            obs[step] = next_obs
            masks[step] = next_mask
            dones[step] = next_done
            with torch.no_grad():
                action, logprob, _, value = agent.act(next_obs, next_mask)
            # Rule 2: no legal action samples action 0.
            action = torch.where(next_mask.any(-1), action, torch.zeros_like(action))
            values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob
            raw, reward, term, trunc, info = envs.step(action.numpy())
            rewards[step] = torch.tensor(bootstrap_rewards(reward, term, trunc, info, value_of, h.gamma))
            next_obs = torch.tensor(np.asarray(raw, dtype=np.float32).reshape(h.num_envs, -1))
            next_mask = torch.tensor(mask_of(info, h.num_envs, n_actions))
            next_done = torch.tensor(np.logical_or(term, trunc), dtype=torch.float32)
            if "episode" in info:
                done_mask = info["_episode"]
                finished.extend(float(r) for r in np.asarray(info["episode"]["r"])[done_mask])
        episodes += len(finished)
        curve.append({"update": update, "steps": update * batch, "episodes": len(finished), "mean_return": float(np.mean(finished)) if finished else None})

        with torch.no_grad():
            next_value = agent.value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards)
            last = 0
            for t in reversed(range(h.num_steps)):
                if t == h.num_steps - 1:
                    nonterminal, next_values = 1.0 - next_done, next_value
                else:
                    nonterminal, next_values = 1.0 - dones[t + 1], values[t + 1]
                delta = rewards[t] + h.gamma * next_values * nonterminal - values[t]
                advantages[t] = last = delta + h.gamma * h.gae_lambda * nonterminal * last
            returns = advantages + values

        b_obs = obs.reshape((-1, obs_dim))
        b_masks = masks.reshape((-1, n_actions))
        b_actions = actions.reshape(-1)
        b_logprobs, b_advantages = logprobs.reshape(-1), advantages.reshape(-1)
        b_returns, b_values = returns.reshape(-1), values.reshape(-1)
        for _ in range(h.update_epochs):
            order = rng.permutation(batch)
            for start in range(0, batch, minibatch):
                idx = torch.as_tensor(order[start : start + minibatch])
                idx = live_rows(idx, b_masks)
                if idx.numel() == 0:
                    continue
                _, newlogprob, entropy, newvalue = agent.act(b_obs[idx], b_masks[idx], b_actions.long()[idx])
                ratio = (newlogprob - b_logprobs[idx]).exp()
                adv = b_advantages[idx]
                adv = (adv - adv.mean()) / (adv.std() + 1e-8) if adv.numel() > 1 else adv
                pg_loss = torch.max(-adv * ratio, -adv * torch.clamp(ratio, 1 - h.clip_coef, 1 + h.clip_coef)).mean()
                newvalue = newvalue.view(-1)
                v_unclipped = (newvalue - b_returns[idx]) ** 2
                v_clipped = b_values[idx] + torch.clamp(newvalue - b_values[idx], -h.clip_coef, h.clip_coef)
                v_loss = 0.5 * torch.max(v_unclipped, (v_clipped - b_returns[idx]) ** 2).mean()
                loss = pg_loss - h.ent_coef * entropy.mean() + v_loss * h.vf_coef
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), h.max_grad_norm)
                optimizer.step()
    envs.close()
    train_seconds = time.perf_counter() - started

    env = gymnasium.make(env_id, **kwargs)
    flat = lp._needs_flat(env.observation_space)
    if flat:
        env = gymnasium.wrappers.FlattenObservation(env)
    evaluation = []
    for i in range(eval_episodes):
        o, info = env.reset(seed=eval_seed_base + i)
        total, done = 0.0, False
        while not done:
            mask = torch.tensor(mask_of(info, 1, n_actions))
            with torch.no_grad():
                logits = agent.actor(torch.tensor(np.asarray(o, dtype=np.float32).reshape(1, -1)))
            o, r, term, trunc, info = env.step(int(greedy(logits, mask)[0]))
            total += float(r)
            done = term or trunc
        evaluation.append(total)
    env.close()
    return {
        "backend": "tessera",
        "seed": seed,
        "episodes": episodes,
        "train_seconds": round(train_seconds, 2),
        "eval_returns": evaluation,
        "eval_mean": float(np.mean(evaluation)),
        "curve": curve,
    }
