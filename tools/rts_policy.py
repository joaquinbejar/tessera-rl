# SPDX-License-Identifier: MIT OR Apache-2.0
"""The reference learner's action distribution for ``tessera/RTS-v1``
(stage 5b, plan r3 section 3, Q4): the full conditional masks of SPEC 11.3.

Per unit slot the action is three heads, ``(type, target, product)``. The
masks come as the head views of ``tessera_rl.masks.head_views``:
``type [B, U, T]``, ``target [B, U, T, W]``, ``product [B, U, T, P]``.
Sampling is conditional, as the masks are:

1. ``type`` under the type mask;
2. ``target`` given the chosen type, under ``target[..., type, :]``;
3. ``product`` given the chosen type, under ``product[..., type, :]``.

The log-probability of a joint action is the sum over slots and heads of
the three conditional log-probabilities. The entropy is the type head's
plus the two conditional heads' at the chosen type (the usual estimate for
autoregressive heads; it is what the PPO entropy bonus uses).

A mask row with no legal choice (a head that does not apply, or a dead
slot whose rows are all false) is treated as the single choice 0 with
probability 1: it contributes 0 to the log-probability and the entropy.
The environment's no-op is ``(0, 0, 0)`` (stage 4, R1), so an empty slot
always plays it. Root-mask-only sampling with the no-op fallback is not
used anywhere here (plan r3, Q4).

This module holds no training loop and runs no training: the protocol
(``docs/reviews/ppo-protocols/rts.md``) and the owner's authorization come
first.
"""

from __future__ import annotations

import torch
from torch import nn

NEG = -1e9


def _masked(logits: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Logits with illegal choices at ``NEG``; rows with no legal choice
    get choice 0 only. Returns the logits and whether each row was empty."""
    mask = mask.bool()
    empty = ~mask.any(dim=-1)
    if empty.any():
        mask = mask.clone()
        mask[..., 0] |= empty
    return logits.masked_fill(~mask, NEG), empty


def _gather(x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """``x[..., index, :]`` over the type axis (second to last)."""
    idx = index[..., None, None].expand(*index.shape, 1, x.shape[-1])
    return x.gather(-2, idx).squeeze(-2)


class Conditional:
    """The joint distribution of one batch of unit-slot actions."""

    def __init__(self, type_logits, target_logits, product_logits, type_mask, target_mask, product_mask):
        self.type_logits, self.type_empty = _masked(type_logits, type_mask)
        self.target_logits = target_logits
        self.product_logits = product_logits
        self.target_mask = target_mask
        self.product_mask = product_mask

    def _heads(self, kind: torch.Tensor):
        t_logits, t_empty = _masked(_gather(self.target_logits, kind), _gather(self.target_mask, kind))
        p_logits, p_empty = _masked(_gather(self.product_logits, kind), _gather(self.product_mask, kind))
        return t_logits, t_empty, p_logits, p_empty

    def sample(self, greedy: bool = False) -> torch.Tensor:
        """Actions ``[B, U, 3]``, each legal under the conditional masks."""

        def draw(logits):
            if greedy:
                return logits.argmax(dim=-1)
            return torch.distributions.Categorical(logits=logits).sample()

        kind = draw(self.type_logits)
        t_logits, _, p_logits, _ = self._heads(kind)
        return torch.stack([kind, draw(t_logits), draw(p_logits)], dim=-1)

    def log_prob_entropy(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Per batch row: the joint log-probability of ``actions`` and the
        entropy (both summed over slots and heads)."""
        kind, target, product = actions[..., 0], actions[..., 1], actions[..., 2]
        t_logits, t_empty, p_logits, p_empty = self._heads(kind)
        lp = torch.zeros(kind.shape, dtype=self.type_logits.dtype)
        ent = torch.zeros_like(lp)
        for logits, empty, choice in (
            (self.type_logits, self.type_empty, kind),
            (t_logits, t_empty, target),
            (p_logits, p_empty, product),
        ):
            dist = torch.distributions.Categorical(logits=logits)
            # An empty row's masked logits leave only choice 0: its
            # log-probability is 0 there and about NEG elsewhere, so a
            # different choice has probability 0, as in `sample`.
            lp = lp + dist.log_prob(choice)
            ent = ent + torch.where(empty, torch.zeros_like(ent), dist.entropy())
        return lp.sum(dim=-1), ent.sum(dim=-1)


def layer(module: nn.Module, std: float = 2**0.5, bias: float = 0.0) -> nn.Module:
    nn.init.orthogonal_(module.weight, std)
    nn.init.constant_(module.bias, bias)
    return module


class Agent(nn.Module):
    """A convolutional torso over the planes plus the globals, a value head,
    and per-slot logits for the three heads, all for every type."""

    def __init__(self, planes: int, size: int, globals_: int, units: int, types: int, window: int, products: int):
        super().__init__()
        self.units, self.types, self.window, self.products = units, types, window, products
        self.torso = nn.Sequential(
            layer(nn.Conv2d(planes, 32, 3, padding=1)), nn.ReLU(),
            layer(nn.Conv2d(32, 32, 3, padding=1)), nn.ReLU(),
            nn.Flatten(),
        )
        hidden = 32 * size * size + globals_
        self.trunk = nn.Sequential(layer(nn.Linear(hidden, 256)), nn.ReLU())
        self.critic = layer(nn.Linear(256, 1), std=1.0)
        self.heads = layer(nn.Linear(256, units * types * (1 + window + products)), std=0.01)

    def features(self, planes: torch.Tensor, globals_: torch.Tensor) -> torch.Tensor:
        x = self.torso(planes.float() / 255.0)
        return self.trunk(torch.cat([x, globals_.float() / 65535.0], dim=-1))

    def value(self, planes, globals_):
        return self.critic(self.features(planes, globals_)).squeeze(-1)

    def distribution(self, planes, globals_, type_mask, target_mask, product_mask) -> tuple[Conditional, torch.Tensor]:
        """The conditional distribution for this batch and its values.

        Masks are the head views reshaped to ``[B, U, ...]`` (one agent per
        batch row). Type logits come from the first block of each slot; the
        target and product logits are given for every type and selected by
        the sampled type."""
        h = self.features(planes, globals_)
        b, u, t = h.shape[0], self.units, self.types
        out = self.heads(h).view(b, u, t * (1 + self.window + self.products))
        type_logits = out[..., :t]
        target_logits = out[..., t : t + t * self.window].reshape(b, u, t, self.window)
        product_logits = out[..., t + t * self.window :].reshape(b, u, t, self.products)
        dist = Conditional(
            type_logits,
            target_logits,
            product_logits,
            torch.as_tensor(type_mask),
            torch.as_tensor(target_mask),
            torch.as_tensor(product_mask),
        )
        return dist, self.critic(h).squeeze(-1)


def flat_actions(actions: torch.Tensor) -> torch.Tensor:
    """``[B, U, 3]`` to the environment's ``MultiDiscrete`` row ``[B, U * 3]``
    (``action_nvec`` is ``(types, window, products) * max_units``)."""
    return actions.reshape(actions.shape[0], -1)
