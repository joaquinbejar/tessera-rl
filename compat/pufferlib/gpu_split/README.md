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
| `run.py` | The driver: rounds with the case order rotated per round, one process group per run, the timeouts, and with a manifest the ADR-0061 attempt with sealed rows (ADR-0063). |
| `manifest.py` | The ADR-0061 item 9 manifest: commit, wheel sha256 (checked against the file), CUDA toolkit (checked against `nvcc --version`), the order of every round, and the ADR-0063 clock fields. It prints its own sha256. |
| `clock.py` | ADR-0063: the guest clock sampler (`sample`) and the pure clock verdict (`decide`). |
| `clock_ref.py` | ADR-0063 item 2: the external reference client, run on the reference host. It talks TCP through one `ssh -L` tunnel to a PowerShell listener on the box's 127.0.0.1 (`--port`, explicit); the ssh session's stdin is unused, because the box's DefaultShell (PowerShell `-c`) takes it. The listener exits on `end`, on disconnect, after `--idle` seconds without a request, or if no client connects within 60 s. |
| `summarise.py` | ADR-0063 item 6: the attempt's summary, computed from the sealed rows after the clock verdict was sent. |
| `../tests/test_gpu_split.py`, `../tests/test_gpu_split_clock.py` | Local CPU tests, run by `make puffer-test`. |

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

  The synchronize wait at each call fires before the wall clock (`wall_now()`, `CLOCK_MONOTONIC_RAW`) is read, so it is charged to the phase before the call, the one that queued the GPU work. `parent_overhead_s` holds only the small gaps before a parent's first child, near zero by construction; it does not measure the `eval` or `train` phase. Profiling runs every epoch (frequency 1, recorded as `profile_frequency`), so the per-epoch cost of the synchronize points and the timer reads is part of this run's wall.

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

**Wall clock (ADR-0064).** Every timed boundary in `split.py` reads `CLOCK_MONOTONIC_RAW` through `wall_now()`, with no fallback, and every result row says so in `wall_clock`. The guest's time agents (timesyncd, and chrony in WSL's system distro) slew `perf_counter`, not the raw counter. `run.py` stamps the timed span on raw (`timed_start_raw`, `timed_end_raw`); `clock.py` brackets each host read on raw and binds (b) and (d2) on raw against QPC, with `perf_counter`, `monotonic` and `realtime` reported only; `summarise.py` refuses a manifest, a clock verdict or any row not on raw.

ADR-0061 (Accepted) fixes the phase 3 gate protocol; this harness implements it.

ADR-0063 (Accepted) amends it with the clock validation of the session; the order of a session is in `docs/reviews/152-adr0063-harness-plan.md` §2.6, and a run plan gives the exact commands.

1. Before the timed rounds: `python manifest.py --out manifest.json --commit SHA --wheel WHEEL --wheel-sha256 HEX --wheel-source pypi-release|main-wheels-run --toolkit X.Y.Z --clock-ref-host NAME --clock-ref-command CMD [--clock-cpu N] [--rerun-of-manifest PATH --rerun-of-sha256 HEX]`. It refuses (exit 2) when the wheel or `nvcc` disagree, and prints the manifest sha256, which is sent to the reviewer.
2. The pre-check: `clock.py sample` and `clock_ref.py` run for at least 300 s, then `clock.py decide --scope pre-check` must print `pre-check: go`.
   - The sampler's tag is the content of `CONTROL/tag` (`pre`, `session`, `post`). Change it atomically: write `CONTROL/tag.tmp`, then rename it to `CONTROL/tag`. The sampler accepts only those three values and only forwards; an empty, unreadable or backward tag keeps the current one, and `decide` fails the instrumentation on samples out of order.
   - Before starting the driver, set the tag to `session` and wait until the `pre -> session` context row is in the samples file. The post-check starts the same way, with `session -> post`.
   - `clock_ref.py --host` is the ssh target (the box); its header records the reference host's own name, which must be the manifest's `--clock-ref-host`. `--port` must be free on both hosts; ssh exits (`ExitOnForwardFailure`) if the local forward cannot bind.
3. The attempt: `python run.py --python PY --rows rows.jsonl --status status.json --device cuda --manifest manifest.json --manifest-sha256 HEX --wheel WHEEL`. It refuses (exit 7) a manifest or wheel whose sha256 differs, any `--split-args`, a `--wheel` path absent from `--python`, a `git rev-parse HEAD` other than the manifest's commit, and an existing `--rows` or `--status`; runs the manifest's order (5 rounds, cases rotated by one per round, interleaved mode by mode), and writes every row, stamped with the manifest sha256, to `rows.jsonl` only. Stdout carries `started manifest=…`, `alive` heartbeats and `rows sealed: <sha256>`, nothing per run. The rows file is not read until step 5.
4. The post-check (at least 300 s), then `clock.py decide --scope session --status status.json`, whose sha256 is sent to the reviewer. Besides the sampler's CPU share over the timed span (budget 1%), it reports the host reader's share from its first to its last context row (start of pre to the stop) and the reference loop's share from its first to its last `cpu` row; those two are reported, not budgeted.
5. Only then `python summarise.py`, with the rows, manifest and decide sha256 and the send time. Its summary row has per-cell (n, median, min, max) of `sps`, `attempt` (`complete`, `failed, no result`, or `void: clock-invalid`, ADR-0061 item 8 as amended by ADR-0063 item 5), the clock outcome, and `gate_eligible`, true only for a complete, clock-valid cuda attempt with the CUDA advantage kernel on every trainer run (item 6). A wrong instrument (a `clock.py` or `clock_ref.py` self-hash mismatch, or a reference host, reference command or sampler core other than the manifest's) is never a void, even when a clock fault was also measured: the attempt is `failed, no result`.

`gate_eligible` also requires 30 timed epochs at full scale and, in every Tessera run, the native module whose sha256 the manifest took from the wheel. Only an eligible summary carries item 7: `ocean_score` and `tessera_score`, each the larger `e2e` median of the side's two cases, with `ocean_case` and `tessera_case`, and `pass` = `tessera_score >= ocean_score`.
