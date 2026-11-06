# SPDX-License-Identifier: MIT OR Apache-2.0
"""Runs in a subprocess under a process-group timeout, as Multiprocessing hangs
on a raise and the trainer has hung at exit. Prints one JSON line."""

import json
import multiprocessing
import os
import sys
import traceback

import numpy as np
import pufferlib.vector

import tessera_rl as t
from tessera_rl.puffer import make_puffer, mix_seed


def multiprocessing_identity(start, workers_inner, envs, procs, reset_seed, formula="true"):
    """Every row against a pool built from the seeds the creator received."""
    n_inst = 3
    multiprocessing.set_start_method(start)
    per = envs // procs
    creator = make_puffer("tessera/CartPole-v1", num_envs=n_inst, workers=workers_inner, log_interval=None)
    vec = pufferlib.vector.make(creator, backend=pufferlib.vector.Multiprocessing, num_envs=envs, num_workers=procs)
    driver_has_pool = vec.driver_env._env is not None
    refs, slots = [], []
    for w in range(procs):
        for j in range(per):
            # "spike" is the wrong reference of the throwaway spike: a control.
            ctor = w * per + j if formula == "true" else (3 + w) * envs + j
            base = mix_seed(ctor, reset_seed + w + j)
            refs.append(t.make("tessera/CartPole-v1", num_envs=n_inst, seed=base, auto_reset="same_step"))
            slots.append(base)
    total = envs * n_inst
    obs, _ = vec.reset(seed=reset_seed)
    want = [r.reset(b) for r, b in zip(refs, slots, strict=True)]
    diffs = 0
    rng = np.random.default_rng(5)
    distinct = len({bytes(row) for row in np.asarray(obs)})

    def compare(o, rw, d, tr):
        nonlocal diffs
        eo = np.concatenate([s.observations["obs"].reshape(n_inst, -1) for s in want])
        diffs += int(eo.tobytes() != np.asarray(o).reshape(total, -1).tobytes())
        if rw is not None:
            diffs += int(np.concatenate([s.rewards.ravel() for s in want]).tobytes() != np.asarray(rw).tobytes())
            diffs += int(np.concatenate([s.agent_terminated.ravel() for s in want]).tobytes() != np.asarray(d).tobytes())
            diffs += int(np.concatenate([s.agent_truncated.ravel() for s in want]).tobytes() != np.asarray(tr).tobytes())

    compare(obs, None, None, None)
    for _ in range(300):
        acts = rng.integers(0, 2, total, dtype=np.int32)
        o, rw, d, tr, _i = vec.step(acts)
        want = [r.step(acts[k * n_inst : (k + 1) * n_inst]) for k, r in enumerate(refs)]
        compare(o, rw, d, tr)
    vec.close()
    for r in refs:
        r.close()
    return {"diffs": diffs, "distinct": distinct, "total": total, "driver_has_pool": driver_has_pool}


def lifecycle(start):
    multiprocessing.set_start_method(start)
    vec = pufferlib.vector.make(
        make_puffer("tessera/CartPole-v1", num_envs=2, workers=2),
        backend=pufferlib.vector.Multiprocessing, num_envs=4, num_workers=2,
    )  # fmt: skip
    driver_has_pool = vec.driver_env._env is not None
    vec.reset(seed=1)
    for _ in range(50):
        vec.step(np.zeros(vec.num_agents, dtype=np.int32))
    vec.close()
    return {"driver_has_pool": driver_has_pool, "closed": True}


def root_above_range(start, root):
    """A root that a child refuses: the parent's reset blocks. One-off probe."""
    multiprocessing.set_start_method(start)
    vec = pufferlib.vector.make(
        make_puffer("tessera/CartPole-v1", num_envs=2),
        backend=pufferlib.vector.Multiprocessing, num_envs=4, num_workers=2,
    )  # fmt: skip
    vec.reset(seed=root)
    return {"returned": True}


def _train(env_id, use_rnn, num_envs):
    from pufferlib import models, pufferl

    vec = pufferlib.vector.make(make_puffer(env_id, num_envs=num_envs))
    config = dict(
        env=env_id, seed=1, torch_deterministic=True, cpu_offload=False, device="cpu", optimizer="adam", anneal_lr=True,
        precision="float32", total_timesteps=8192, learning_rate=0.001, gamma=0.99, gae_lambda=0.9, update_epochs=1,
        clip_coef=0.2, vf_coef=2.0, vf_clip_coef=0.2, max_grad_norm=1.5, ent_coef=0.001, adam_beta1=0.95,
        adam_beta2=0.999, adam_eps=1e-12, data_dir="experiments", checkpoint_interval=200, batch_size="auto",
        minibatch_size=256, max_minibatch_size=256, bptt_horizon=16, compile=False,
        compile_mode="max-autotune-no-cudagraphs", compile_fullgraph=True, vtrace_rho_clip=1.0, vtrace_c_clip=1.0,
        prio_alpha=0.8, prio_beta0=0.2, use_rnn=use_rnn,
    )
    policy = models.Default(vec, hidden_size=32)
    if use_rnn:
        policy = models.LSTMWrapper(vec, policy, input_size=32, hidden_size=32)
    trainer = pufferl.PuffeRL(config, vec, policy)
    losses, checks = [], 0
    while trainer.global_step < config["total_timesteps"]:
        trainer.evaluate()
        # Whatever the update does, it must not write into the engine's rows.
        before = vec.observations.tobytes(), vec.rewards.tobytes()
        trainer.train()
        checks += 1
        assert (vec.observations.tobytes(), vec.rewards.tobytes()) == before, "trainer wrote into a view"
        losses.extend(float(v) for v in trainer.losses.values())
    finite = bool(losses) and bool(np.isfinite(np.asarray(losses)).all())
    return {"global_step": int(trainer.global_step), "finite": finite, "checks": checks}


def smoke(env_id, use_rnn):
    num_envs = {"tessera/CartPole-v1": 64, "tessera/Tag-v1": 8}[env_id]
    return _train(env_id, use_rnn == "rnn", num_envs)


def main():
    name, *args = sys.argv[1:]
    conv = [int(a) if a.lstrip("-").isdigit() else a for a in args]
    try:
        out = globals()[name](*conv)
    except BaseException:
        # PuffeRL starts a non-daemon thread, so a normal exit would hang.
        traceback.print_exc()
        sys.stderr.flush()
        os._exit(1)
    sys.stdout.write("\nRESULT " + json.dumps(out) + "\n")
    sys.stdout.flush()
    os._exit(0)  # The trainer's and PufferLib's threads must not delay the exit.


if __name__ == "__main__":
    main()
