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
| `run.py` | The driver: rounds with the case order rotated per round, one process group per run, the timeouts, and with a manifest the ADR-0061 attempt and its summary row. |
| `manifest.py` | The ADR-0061 item 9 manifest: commit, wheel sha256 (checked against the file), CUDA toolkit (checked against `nvcc --version`) and the order of every round. It prints its own sha256. |
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
- 31 epochs, the first dropped (ADR-0061 item 2: only the timed window grows).

## Modes, reported separately

- **`e2e`:** end to end, uninstrumented. The profiler never fires in the timed window. `torch.cuda.synchronize()` runs only immediately before and after the window, so queued learner work at the end is counted. Throughput only.
- **`sync`:** synchronised phase costs. This mode runs the trainer from the patched copy, profiled every epoch, with flat exclusive attribution:
  - every interval between two profiler calls is charged to the one name most recently started;
  - the interval after `end()` is charged to no name;
  - the intervals charged to a parent, from `eval` or `train` starting to its first child, are reported apart as `parent_overhead_s`, not as phases;
  - so the child categories plus that overhead add up to at most the run's wall, and the rest is `unattributed_s`.

  The synchronize wait at each call fires before `perf_counter` is read, so it is charged to the phase before the call, the one that queued the GPU work. `parent_overhead_s` holds only the small gaps before a parent's first child, near zero by construction; it does not measure the `eval` or `train` phase. Profiling runs every epoch (frequency 1, recorded as `profile_frequency`), so the per-epoch cost of the synchronize points and the timer reads is part of this run's wall.

  This flat attribution is an explicit instrumentation amendment to the plan's wording ("exclusive categories, children only"). It keeps the plan's sync points.

  This is needed because PufferLib's own `Profile` is not exclusive. Inside the minibatch loop `train_misc` is pushed with `nest=True` without popping the entry before it, so its numbers overlap. They are kept as `pufferlib_profile_s_not_additive`, for reference only. The flat attribution hooks the same synchronize points as the patch: it calls the patched `Profile` first, then reads `perf_counter`.

  Synchronising changes overlap and scheduling, so `sync` is a cost breakdown of a different run, not a share of `e2e`.
- **`env`:** environment only. The same vector environment and agent count, random actions and no policy:
  - the reset and 300 warm-up steps are received before timing starts, so nothing is pending when it does;
  - each timed step is a `send` followed by its matching `recv`, so the last transition completes inside the window, asynchronous Multiprocessing included. It is CPU work. It measures env throughput and is not a bound: random actions change trajectories, resets and cache behaviour, and drop the policy transfers.

No environment share of the training critical path is inferred from these modes.

## Process rules, as the plan states them

- **CUDA order:** no CUDA call of any kind happens before the vector environment and its fork workers exist, since even `torch.cuda.is_available()` may initialise the driver. Before building it, `split.py` reads only `torch.cuda.is_initialized()`, which is torch's own flag, and refuses the run if it is set (exit 4). `is_available()` and the device name come after the workers exist. Without CUDA, a `cuda` run cleans up and stops with exit 3. A test checks the call order.
- **Cleanup in `try`/`finally`:**
  - every PufferLib `Utilization` monitor thread is stopped and joined, including one left behind by a trainer whose construction failed;
  - the vector environment is closed;
  - the run's own child processes are listed before and after, terminated and reaped.

  PufferLib's Multiprocessing `close()` terminates its workers without joining them; the tests show them still listed after `close()` and reaped afterwards. `PuffeRL.close()` is not called, because it saves a checkpoint. There is no `os._exit`.
- **The driver:**
  - runs each attempt in its own temporary directory and process group;
  - kills the group after every run, whether it succeeded or timed out, and checks that no process of the group is left (zombies awaiting reaping excepted);
  - stops the session if one is left.

  The timeouts are 100 s per run. Only the session's first CUDA trainer run (`e2e` or `sync` on `cuda`) gets 300 s, as it includes context creation and cuDNN autotune; env-only runs never get it. Each row records the timeout it had.
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

## The ADR-0061 gate attempt

ADR-0061 (Accepted) fixes the phase 3 gate protocol; this harness implements it.

1. Before the timed rounds: `python manifest.py --out manifest.json --commit SHA --wheel WHEEL --wheel-sha256 HEX --wheel-source pypi-release|main-wheels-run --toolkit X.Y.Z`. It refuses (exit 2) when the wheel or `nvcc` disagree, and prints the manifest sha256, which is sent to the reviewer.
2. The attempt: `python run.py --python PY --out results.jsonl --device cuda --manifest manifest.json --manifest-sha256 HEX --wheel WHEEL`. It refuses (exit 7) a manifest or wheel whose sha256 differs, runs the manifest's order (5 rounds, cases rotated by one per round, interleaved mode by mode), stamps every row with the manifest sha256, and always ends `results.jsonl` with a summary row: per-cell (n, median, min, max) of `sps`, `attempt` (`complete` or `failed, no result`, item 8) and `gate_eligible`, true only for a complete cuda attempt with the CUDA advantage kernel on every trainer run (item 6).

The pass rule of item 7 (each side's better case by `e2e` median) is applied in the record, from the summary row.
