# Reference training scripts (SPEC 14.2, #368)

Each script trains a library's own algorithm on a Tessera environment
through the Gymnasium adapter, with Gymnasium's environment as the control.
They validate the adapters with unmodified third-party trainers; they are
not the SPEC 18 learning tests. Each library has its own uv project and
lock under `compat/training/<library>/`, so none of them enters the root
lock, `make py-test` or `make pre-push`.

| Library | Script | Run |
| --- | --- | --- |
| CleanRL | `tools/reference_training.py` ("CleanRL ppo.py, adapted"; see `THIRD_PARTY.md`) | `make reference-training` |
| Stable-Baselines3 | `sb3_cartpole.py` | `make reference-sb3` |
| Tianshou | `tianshou_cartpole.py` | `make reference-tianshou` |
| RLlib | `rllib_cartpole.py` (validated against Gymnasium 1.2.2, RLlib's pin; its own `[env.rllib]`) | `make reference-rllib`; multi-agent wiring smoke on Tag (PettingZoo adapter, no learning claim): `rllib_tag_smoke.py`, `make reference-rllib-tag` |

Pass rule, the defaults of `tools/reference_training.py`: the median across
seeds of the Tessera greedy evaluation mean is at least 475 (CartPole-v1's
reward threshold), and the Tessera minus Gymnasium difference of the
training-curve means is at most 20 in absolute value. SB3 and Tianshou
fail the nightly when they miss it (#501); RLlib and the Tag smoke stay
report-only until they have 7 consecutive green nights (#579). They run nightly in
`.github/workflows/reference-libraries.yml` (CleanRL in
`reference-training.yml`).
