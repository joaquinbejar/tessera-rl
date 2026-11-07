# PuffeRL split harness for the GPU run

This is the harness of `docs/reviews/gpu-split-plan.md` r5, sha256 `52b0ac29`. It is reviewed before the RTX 4090 host is available.

- **Scope:** exploratory, CartPole only, no gate claim.
- **Not authorized:** no run on the owner's machine, and nothing installed anywhere. The plan's WSL2 route needs the owner's approval through the coordinator, and this directory changes nothing about that.

## Files

| File | Role |
| --- | --- |
| `split.py` | One run: one case, one mode, prints one JSON line. |
| `sync.py` | Builds the synchronised scratch copy of `pufferl.py` for mode `sync`. The installed file is never modified. |
| `pufferl-sync.patch` | The published instrumentation diff. It enables the two commented `torch.cuda.synchronize()` blocks in `Profile.__call__` and `Profile.end`, and nothing else. The tests check that the copy equals it. |
| `run.py` | The driver: rounds with a fixed shuffle, one process group per run, and the timeouts. |
| `../tests/test_gpu_split.py` | Local CPU tests, run by `make puffer-test`. |

## Cases

| Case | Vector environment |
| --- | --- |
| `ocean-serial` | 1 x 8192 agents, Serial |
| `ocean-mp` | 2 x 4096 agents, Multiprocessing (fork by default on Linux) |
| `tessera-w1` | 8192 agents, 1 worker |
| `tessera-w2` | 8192 agents, 2 workers |

The configuration is PufferLib's `cartpole.ini` over `default.ini`:

- bptt 64, batch auto (524,288 at 8192 agents), minibatch 32,768;
- `models.Default` hidden 128, no RNN;
- seed 1, `compile=False`, torch threads 4;
- 11 epochs, the first dropped.

## Modes, reported separately

- **`e2e`:** end to end, uninstrumented. The profiler never fires in the timed window. `torch.cuda.synchronize()` runs only immediately before and after the window, so queued learner work at the end is counted. Throughput only.
- **`sync`:** synchronised phase costs. This mode runs the trainer from the patched copy, profiled every epoch, with flat exclusive attribution:
  - every interval between two profiler calls is charged to the one name most recently started;
  - the interval after `end()` is charged to no name;
  - so the categories add up to at most the run's wall, and the rest is `unattributed_s`.

  This is needed because PufferLib's own `Profile` is not exclusive. Inside the minibatch loop `train_misc` is pushed with `nest=True` without popping the entry before it, so its numbers overlap. They are kept as `pufferlib_profile_s_not_additive`, for reference only. The flat attribution hooks the same synchronize points as the patch: it calls the patched `Profile` first, then reads `perf_counter`.

  Synchronising changes overlap and scheduling, so `sync` is a cost breakdown of a different run, not a share of `e2e`.
- **`env`:** environment only. The same vector environment and agent count, random actions, no policy, 300 warm-up steps and then the timed steps. It is CPU work. It measures env throughput and is not a bound: random actions change trajectories, resets and cache behaviour, and drop the policy transfers.

No environment share of the training critical path is inferred from these modes.

## Process rules, as the plan states them

- **CUDA order:** CUDA is never initialised before the vector environment and its fork workers exist. `split.py` checks `torch.cuda.is_initialized()` before building it and refuses the run otherwise (exit 4). Without CUDA, a `cuda` run stops with exit 3.
- **Cleanup in `try`/`finally`:**
  - every PufferLib `Utilization` monitor thread is stopped and joined, including one left behind by a trainer whose construction failed;
  - the vector environment is closed;
  - the run's own child processes are listed before and after, terminated and reaped.

  PufferLib's Multiprocessing `close()` terminates its workers without joining them; the tests show them still listed after `close()` and reaped afterwards. `PuffeRL.close()` is not called, because it saves a checkpoint. There is no `os._exit`.
- **The driver:**
  - runs each attempt in its own temporary directory and process group;
  - kills the group after every run, whether it succeeded or timed out, and checks that no process of the group is left (zombies awaiting reaping excepted);
  - stops the session if one is left.

  The timeouts are 100 s per run and 300 s for the session's first run.
- **Recorded per run:**
  - the advantage backend: `cpu-fallback` unless PufferLib found `nvcc`;
  - the device name;
  - the versions;
  - for `sync`, the SHA-256 of the installed `pufferl.py` and of its synchronised copy.

## Local check

```sh
make puffer-test            # builds the wheel; runs compat/pufferlib/tests, this harness's included
```

The tests run on CPU at `--scale small` (1024 agents, minibatch 8192). That scale is for the tests only.
