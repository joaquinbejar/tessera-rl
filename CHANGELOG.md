<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->

# Changelog

All notable changes to this project are documented in this file. The
format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project uses [Semantic Versioning](https://semver.org/). Before 1.0
the API is not stable, and any 0.x release may break it.

## [Unreleased]

### Changed

- Behaviour change: `tessera_rl.make()` without `workers` now starts one worker per physical core this process may use, at most one per chunk (`tessera_rl_engine::default_workers`, SPEC 6.1), in Sync, Pipelined and Async modes. Before, Sync stepped inline and the split-phase modes used one worker. Pass `workers=0` for the old Sync behaviour. On an x86-64 host with 8 physical cores, the default ran CartPole and BrickWall pool steps in 0.28 to 0.44 of the inline time at 1024 and 8192 instances (#356). The PufferLib backends keep their explicit worker counts.

### Added

- Per-agent refusal for host environments (ADR-0070, #337): `HostEnvironment::REFUSES_AGENTS` and `refuse(&self, &Actions, &mut Refusals)` in `tessera-rl-core`, with `RefuseCode`, `Refusals` and `Actions::is_refused`/`refuse_code`. `refuse` runs per instance after `validate` accepted the call and before `step`; refused agents are reported in the new `refused` and `refuse_code` outputs (`[N, A]`, refusing host pools only) while every other agent and instance steps. Refusing a slot that is not acting poisons the pool (`EngineError::RefusedNotActing`). Host protocol 2 (protocol 1 plus those outputs and the `refuses_agents` layout key) is spoken by a wheel whose environment refuses agents; `tessera_rl` accepts protocols 1 and 2. `HostStep.refused`/`refuse_code`, Gymnasium `info["refused"]`/`info["refuse_code"]` and PettingZoo `infos[agent]["refused"]`/`["refuse_code"]` report refusals, and both adapters' metadata carry `refuses_agents`. Refusal rows in replay files (ADR-0070 item 9) follow in #339.
- `tessera_rl_engine::physical_cores()` and Python `tessera_rl.physical_cores()`: physical cores this process may use (sysfs topology within the affinity mask on Linux, `hw.physicalcpu` on macOS, processor core records on Windows), capped by `available_parallelism()` so CPU quotas apply. Efficiency cores of hybrid CPUs count as cores (#356, SPEC 6.1).
- `Default` for `SyncOptions` and `PipelinedOptions`: one worker per physical core. A Rust pool still steps inline until it is given workers. `default_workers()` adds the per-chunk cap that Python's `make()` uses.
- Output buffer alignment (SPEC 5.2, #355): the output allocation and every observation and terminal observation leaf start on a 16 KiB boundary (a multiple of the page size: 4 KiB on x86-64, 16 KiB on Apple silicon; `tessera_rl_engine::OUTPUT_ALIGN`); every other output buffer starts on 64 bytes (`OUTPUT_MIN_ALIGN`). Before, buffers were 8-byte aligned. Step times are unchanged within noise on CartPole and Tag at 1024 and 8192 instances.

- Host snapshots (ADR-0073, #344): `HostSnapshot` and `SnapshotReject` in `tessera-rl-core`, and on host pools of such environments `snapshot`, `snapshot_to`, `snapshot_all`, `restore`, `import`, `restore_all`, `fork`, `state_hash`, `encode` and `decode` over `HostRecord`s (identity digest, host header, `SNAPSHOT_VERSION` and the environment's bytes). Every record is checked before any instance changes: identity, version, length, the host header rules of ADR-0068 (r2h) and `check_snapshot`. A poisoned host pool refuses all of them, `restore_all` included; a restored pending instance is not observed; there is no `fork_branched`. Host files use their own magic, `TESHOST\x1a`, under the shared ADR-0011 header. New `EngineError::SnapshotRejected` and `SnapshotTooLong`, `DecodeLimits::max_host_record_bytes`, and `StateHasher::add_bytes`. Through the type-erased pool and Python: `HostPool::with_dyn_snapshots`, `register_host_factory!(Env, snapshots)` and `export_host_env!(Env, module = .., snapshots)` make `DynHostPool`'s snapshot methods (`snapshot_records`, `restore_records`, `restore_all_records`, `fork`, `state_hash`, `encode`, `import_file`; `EngineError::NoSnapshots` otherwise) and the host object's (`snapshot`, `restore`, `restore_all`, `fork`, `state_hash`, `encode`, `import_records`) work, and add the layout key `snapshots` (host protocol 1, an addition). `HostEnv.snapshots`, `snapshot`, `restore`, `restore_all`, `import_record`, `fork`, `state_hash`, `encode` and `load` follow; host modules bind `RecordError` and `PersistError`. The engine feature `testing` adds `host_testing::check_snapshot_contract` for environment crates' tests. The `host_orderbook` example implements `HostSnapshot`, and the fuzz target `host_decode` covers host files and records.

### Changed

- **Breaking (experimental host API, ADR-0062 item 9):** `HostPool::new` takes the configuration by value and keeps it, as `EnvPool::new` does (ADR-0073 needs it after construction). Replace `HostPool::new(&cfg, options)` with `HostPool::new(cfg, options)`. A type-erased pool (`DynHostPool`, `make_host`, `register_host_factory!`, `export_host_env!`) now needs a `Send` configuration. `DecodeLimits` gains the field `max_host_record_bytes`, so struct literals need it (or `..DecodeLimits::default()`).

## [0.3.0] - 2026-11-23

### Added

- Host environments (ADR-0062, experimental until 1.0): the `HostEnvironment` trait, `HostSeed`, `HostOut` and `HostError` in `tessera-rl-core`, for environments whose state lives on the heap (`&mut self`, no `Copy`, no snapshot, fork or `Persist`); `register_host!` and `registered_host()`; the public `Space::heads()`; and `HostPool` in `tessera-rl-engine` (feature `host`), in Inline and Sync modes, with ADR-0065's aborted episodes, joint-action validation and runtime agent count. `HostEnvironment::new` is called once per instance, in index order, on the thread that builds the pool, before any method on an instance (#285).
- `tessera-rl-pyffi` (new crate, experimental): the CPython boundary shared by Tessera's Python extensions, a typed safe FFI layer and, behind its `host` feature, `export_host_env!` (host protocol 1). The first-party extension is built on it.
- `examples/host_orderbook`: an out-of-tree host environment wheel (`example/OrderBook-v1`), its own Cargo workspace and maturin project, built and checked next to the first-party wheel: on Linux x86-64 in CI (`make py-test-orderbook`), and on Linux aarch64 and macOS arm64 in the wheels workflow (#313).
- Host environments from Rust by id (ADR-0065): `register_host_factory!`, `make_host`, `MakeOptions`, `FactoryError` and the type-erased `DynHostPool` in `tessera-rl-engine` (feature `host`).
- Host environments from Python (SPEC 14.4 draft, ADR-0065): `tessera_rl.make` finds the `tessera_rl.envs` entry points of installed host wheels (read once per process: restart after installing one) and returns a `HostEnv`, whose `HostStep` is `Step` followed by `aborted`, `abort_code` and `pending_reset`; `CapabilityError` for what a host environment does not support (snapshots, split-phase modes, PufferLib, JAX); `is_stale` and `set_fp_policy` reach host views and objects; the Gymnasium adapter reports `aborted`, `abort_code` and `pending_reset` infos, and PettingZoo's `agents` is empty after an aborted reset, with Mixed host environments through `parallel_env` only; both adapters report `metadata["tessera_kind"]` (ADR-0062).
- Host protocol 1 for third-party host environment wheels (SPEC 14.4 draft, ADR-0062, ADR-0065): `export_host_env!` in `tessera-rl-pyffi` (feature `host`) exports `make(**kwargs)` and the host object, with `layout()`, `reset(seed)`, `step(actions)` returning read-only output views (including `aborted`, `abort_code` and `pending_reset`), `close()`, `is_poisoned()`, the generation and FP-policy hooks; `HostPool::layout` and `HostLayout` in `tessera-rl-engine`; read-only attributes (`AttributeTable`) in `tessera-rl-pyffi`'s typed layer.
- `slot_generations` (issue #273): each slot's occupant generation, `uint32` `[N, agents]`, in the first-party `_native.Env` outputs (borrowed and owned; Sync, Pipelined and Async) and as the tenth field of `tessera_rl.Step` and `tessera_rl.jax.Step`, with the same name, dtype and shape as host protocol 1. Unpacking a `Step` into 9 names now fails; access by name is unchanged.
- Config decoding (ADR-0065): `ConfigSource`, `ConfigItem`, `ConfigDecode` and `ConfigDecodeError` in `tessera-rl-core`, generated by `#[derive(EnvConfig)]`; `Fx32::from_decimal` and `Fx64::from_decimal`, exact decimal parsing.

### Changed

- **PettingZoo agent names for spawning environments (#71, ADR-0067):** an agent is named per (slot, generation): `agent_<slot>` for generation 0, unchanged for every existing environment, and `agent_<slot>_g<g>` after that. For a spawning environment `possible_agents` grows as agents spawn and is rebuilt at `reset`. A respawned agent is a new agent, as PettingZoo's API tests require; native AEC refuses a spawning environment with `CapabilityError`.
- **PettingZoo multi-head masks (#311):** for an environment with several masked discrete heads, `tessera_rl.pz`'s `infos[agent]["action_mask"]` is now a tuple of `int8` masks, one per `MultiDiscrete` component in `nvec` order, which `MultiDiscrete.sample` accepts (it was the root head's array). A conditional head's entry is the union over its parents, so a sampled action can still be illegal; the exact masks stay in `infos[agent]["action_masks"]`. The Gymnasium adapter's `info["action_mask"]` is unchanged.
- `OutputView.__dlpack__` raises `TypeError` for an unknown keyword argument (it was ignored); its argument rules are now shared with host views (#272).

### Performance

- The plain Sync and Inline step validates and copies actions in one pass over the action buffer while every instance steps (none pending a reset, every agent alive and acting) and no counter is near its limit (#152).
- The step and observation RNG keys are derived on their first draw, not for every instance and step (#152).
- Output rows are cleared once per chunk (#266).
- The Python step passes an exact `int32` C-contiguous action array for a single discrete head to the native step as a memoryview, with no Python-side pass over it, and the native binding builds its output view set once per environment (#152 E2, E3).
- The PufferLib adapter logs episodes from the engine's episode statistics (#271).

### Fixed

- An Async worker that dies between the two reads of a failure check is reported as `WorkerDied`, not as a step's panic (#303).
- A host instance whose `same_step` reset aborts keeps the ended episode's end in its record header (#290).
- The release and PGO build scripts refuse an ambient `CARGO_PROFILE_*` or `CARGO_TARGET_*` variable every time: the check missed a set variable in some runs (#320). The instruction-count gate's profile check reports a crate built without `-C codegen-units=1` or with LTO, also for large `rustc` lines and for `lto = true` (#323).
- A host environment object made before `os.fork()` now raises `ForkError` in the child, as SPEC 14.4.2 specifies: the host module registers its own copy of the fork guard, since a host wheel links its own `tessera-rl-pyffi`. Before, the call returned with `workers=0` and hung with workers (#329).
- PettingZoo AEC by cycle buffering (`tessera_rl.pz.aec_env` over a Simultaneous environment) accepts a multi-head action: its action check used `action_space(agent).n`, which a `MultiDiscrete` space does not have (#311).

## [0.2.0] - 2026-11-14

### Changed

- **`RtsConfig` gains `opponent`** (RTS stage 5c): 0 keeps the two-agent game; 1 to 3 put a scripted Rush,
  Prospector or Random-legal opponent in agent slot 1, played inside the step from its own side's view, and the pool
  has one agent. Gymnasium ids `tessera/RTS-vsRush-v1`, `tessera/RTS-vsProspector-v1` and `tessera/RTS-vsRandom-v1`
  fix it. The new field changes every RTS config hash, so an RTS file persisted before this change is refused
  (`PersistError::Incompatible { field: "config_hash" }`), never misread; outputs with `opponent = 0` are unchanged and
  `ENV_VERSION` stays 2 (RTS has been in no release).
- **`EngineError` is `#[non_exhaustive]` and has two new variants**, `NanDetected` and `AllocationInChunkStep`
  (ADR-0055); a breaking change for downstream exhaustive matches, for the next incompatible version. Both are
  declared in every build and constructed only with `debug-checks`.
- **`debug-checks` traps NaN and allocations** (SPEC 5.5, ADR-0055): a NaN in a reward or in a state field not
  annotated `#[state(allow_nan)]` poisons the pool and fails the call with `NanDetected`, in every mode. With
  `engine::debug::TrackingAllocator` installed as a test binary's global allocator, an allocation inside a chunk step
  poisons the pool with `AllocationInChunkStep`, counted per thread and per pool. `Canonical` gains a provided
  `first_nan` method, generated by `#[derive(EnvState)]`; a hand-written impl for a type that can hold a float must
  override it. Python raises both as `PoisonedError`.
- **`EnvPool::outputs` returns `Result<&Outputs, EngineError>`** and refuses a poisoned pool with `Poisoned`
  (ADR-0028, SPEC 5.4); a source-breaking change. Python's `outputs()` refuses too, in borrowed and owned modes.

- **Observation leaves are zero on entry to `observe`** (ADR-0058, #243): the engine already cleared every leaf of
  the agent's row before `observe`, on every path (step, reset, both auto-reset modes, re-observation after
  restore, fork, import and load); it is now a stated contract. An environment need not clear its leaves.
- **The Linux x86-64 wheel builds with the `wheel` profile** (#253): `lto = "fat"`, `codegen-units = 1`,
  `panic = "unwind"`, the W1 candidate of the PGO ladder (11.7% fewer instructions on its scenarios). The wheel job
  also builds the default-profile wheel from the same commit and fails on any determinism difference between the
  two. aarch64 Linux and macOS wheels keep the release profile. No PGO wheel ships.

### Added

- **`tessera_rl::register!`** (SPEC 10, ADR-0032, ADR-0054): `register!(Env, id = "...")` requires `Persist`,
  checks at compile time that the id equals `ENV_ID`, and records the id, version, agent count, tier and
  information kind. `tessera_rl::envs::registered()` lists the records sorted by id and returns
  `RegistryError::DuplicateId` at first lookup. Every shipped environment is registered;
  `tessera_rl._native.registered_envs()` lists them in Python. Adds the `inventory` dependency.
  Still deferred (#63): the record's capabilities and constructor, and `tessera_rl.make(id)` through the
  `tessera_rl.envs` entry points.
- **Snake and paddle duel in the Gymnasium adapter** with one agent: `tessera/Snake-v1` and
  `tessera/PaddleDuel-v1` are registered with `agents=1` (time limits 1000 and 5000) and read slot 0 of
  the native rows; `agents=2` is refused in favour of PettingZoo. Snake's reverse action is a no-op
  there, as the adapter exposes no mask. `layout()` gains `initial_agents`, the initial population.
- **Persisted snapshots in every mode**: `encode`, `decode`, `check_records` and `import_all` on `PipelinedPool` and `AsyncPool`, and `Env.encode` and `Env.load` in Python, with a new `PersistError`. Files are mode-independent and hold full state, so they are privileged. A file holds no cycle state.
- Persisted snapshots (ADR-0011): `EnvPool::encode` and `decode` for the custom flat format, on Sync pools, with the `Persist` and `PersistMask` traits. Decoded records stay untrusted until `import`. Same-platform files need an explicit `TESSERA_BUILD_ID`. Files hold full state and are privileged.
- **The P2-G1 gate under ADR-0048** (`examples/alphazero_connect4/gate.py`): at least 815 of 1,000 wins, held-out
  eligibility, an attempts journal across loop roots, and a solver comparison step (`solver_slots`), refused until pinned.
- **Pipelined in the PufferLib adapter** (`make_puffer(..., mode="pipelined")`): native backend only, equal halves. No speed gain has been measured.
- **Connect Four best-response measurement** (`tools/connect4-ref`): an open-loop search per held-out slot against
  the pinned bot and an independent replay of its lines. A reference tool, not part of any gate.
- **PufferLib adapter** (`tessera_rl.puffer`): a native `PufferEnv` in Sync mode for CartPole and Tag at PufferLib 3.0.0, zero copy on the native backend and one copy per step on the vector backends. Limits, including hangs under `Multiprocessing` when a worker raises, are in the module docs.
- **Threaded Sync instruction counts** (`make gungraun-threaded`, #76): caller and job-body counts of a step,
  a report outside the gate.

- **Async mode** (`tessera-rl-engine`): `EnvPool::async_mode` sends any
  subsets of instances with their actions and receives them in batches of
  `batch_size`, the first ready, as a `BatchToken` whose views borrow the
  token. Workers claim instances one at a time from a submission ring and
  report them on a completion ring. An instance cannot be sent again while
  a token covers it; `send_batch` resends a whole token. `reset(ids)` and
  `restore(ids)` deliver their instances through the next `recv`. Writers
  refuse while a token is alive or a sent instance still runs. Every
  instance's trajectory is bit-identical to Sync with the same actions, for
  any worker count and send order; batch membership depends on timing.
  It is currently much slower per round than Sync (indicative, on a loaded
  Mac: about 183 µs on park and 58 µs on native for a round of 256
  instances in batches of 64, against 15 µs and 9 µs for one Sync step),
  because workers claim one instance at a time and waits poll; batch
  claiming and wake-based waits are tracked in #83. Rust only for now; the
  Python binding follows.
- **Pipelined mode** (`tessera-rl-engine`): `EnvPool::pipelined` moves a
  pool into two fixed halves stepped by its own workers. `send` consumes a
  `HalfToken` with that half's actions and starts its step; `recv`,
  `try_recv` and `recv_timeout` return the oldest sent half as a new token,
  whose views borrow the token, so one half's outputs stay readable while
  the other runs. A half may not run two cycles ahead (`CycleAhead`);
  writers (`reset_all`, `restore_all`, `into_pool`) refuse while a token is
  alive or work is in flight. Every half batch is bit-identical to the same
  rows of a Sync step with the same actions, for any worker count. Rust
  only for now; the Python binding follows.
- **PettingZoo adapters** (`tessera_rl.pz`, with the `pettingzoo` extra):
  `parallel_env` and `aec_env`. Each environment declares the interfaces
  it supports and any other request is refused at construction: Tag is
  Parallel natively and AEC through cycle buffering, Connect Four is AEC
  only. They pass the official PettingZoo API tests, and tests cover
  terminal reward timing, agent death, a conditional second-player
  decision and equality with the native pool. `turn_masked_env` is
  declared by no environment yet. Not yet tested: agent respawn, which
  ADR-0013 requires, because no shipped environment spawns agents; the
  test lands with the first one that does (#71).
- **Search** (`tessera-rl-search`): player-perspective search on
  imperfect-information environments. `SearchPool::new_determinized` and
  `set_root_determinized` root the tree at a state sampled by
  `Determinize` from the searching agent's information state, with a
  caller's sampler seed and sample index. The root's hidden state and RNG
  lineage never reach the search: the sampler and the rollout draw with
  their own identities (the `SAMPLER` and `ROLLOUT` branch labels), the
  header keeps only the step clock, `alive`, `generation` and adaptive
  values, and the record passes the engine's import checks. A
  determinized tree is never re-rooted. `SearchPool::new_privileged`
  searches the full record and is labelled by `is_privileged`;
  `SearchPool::new` still refuses imperfect information.
- **Leduc hold'em** (`tessera/Leduc-v1`, `tessera-rl-games`): a
  two-player poker game with a hidden card per player and a public card
  dealt at step time from `StepRng`; `Discrete(3)` (fold, check or call,
  raise) with masks, sequential turns, rewards in units of the largest
  stake so they lie in `[-1, 1]`. It implements `ImperfectInformation`
  and `Determinize` (the opponent's card drawn from the cards the
  player cannot see) and passes the eight determinized search tests;
  golden fixtures, a scripted baseline and a step bench. In Python it is
  available through `make` and the PettingZoo `aec_env`.
- **Othello** (`tessera/Othello-v1`, `tessera-rl-games`): the 8 x 8 game
  with `Discrete(65)` (one action per square and a forced pass), masks,
  sequential turns and rewards of +1, -1 or 0 by disc count. Import
  validation, golden fixtures, continuation tests, perft against an
  independent reference, a scripted baseline and a step bench. In Python it
  is available through `make` and the PettingZoo `aec_env`.
- **Go 9x9** (`tessera/Go9-v1`, `tessera-rl-games`): Tromp-Taylor area
  scoring and positional superko, with suicide illegal, a scored cap of 162
  plies and a configurable komi, default 7.5. `Discrete(82)` with masks and
  sequential turns; rewards of +1, -1 or 0. The state holds exact past
  positions, about 3.9 KB. Import validation replays the history; golden
  fixtures, continuation tests, an independent reference, a scripted baseline
  and step and fork benches. In Python it is available through `make` and the
  PettingZoo `aec_env`.
- **Reference grid world** (`tessera/Grid-v1`, `tessera-rl-classic`): the
  example of SPEC section 10, with its 16-byte state; one agent, a goal drawn
  at reset, `Discrete(4)` and a full view. Golden fixtures, continuation tests,
  a test reference, a scripted baseline and a step bench. In Python it is
  available through `make` and the Gymnasium adapter.
- **Paddle duel** (`tessera/PaddleDuel-v1`, `tessera-rl-classic`): two
  paddles and a ball in `Fx32`, one agent against a scripted paddle or two
  agents, with original rules and rendering: a 30-frame serve, crossing-based
  hits that take spin from the paddle's move, games to `max_score`, `frameskip`
  and a `bounce` reward in config, slot 1 seeing a mirrored world, and an `obs`
  choice of 84 x 84 grayscale pixels (the default) or a 9-field state vector.
  Golden fixtures, continuation, a test reference in raw integer arithmetic and
  a reference rasterizer, a scripted baseline and step benches. In Python it is
  available through `make` and the PettingZoo adapters.
- **Pendulum** (`tessera/Pendulum-v1`, `tessera-rl-classic`): Gymnasium
  `Pendulum-v1` dynamics and rewards in the reference's mixed f32 and f64,
  continuous torque, a 200-step limit, and the frozen `math::fmod`. Rewards are
  compared with Gymnasium within the derived bound of ADR-0050, not exactly
  (`tools/gym_equivalence_pendulum.py`).
- **Acrobot** (`tessera/Acrobot-v1`, `tessera-rl-classic`): Gymnasium `Acrobot-v1`
  book dynamics with one RK4 step, in f64 with frozen math, cross-platform tier, a
  500-step limit. One-step equivalence with the pinned Gymnasium 1.4.0 within SPEC 18
  tolerances over 100,000 rollout states and a boundary set
  (`tools/gym_equivalence_acrobot.py`). The tolerance is not met over the whole
  domain: the tool also reports a uniform set with failures at near-zero velocities.
  Golden fixtures, continuation, import validation, a scripted baseline, a step bench
  and a persist fuzz target. In Python it is available through `make` and the
  Gymnasium adapter.
- **Mountain car** (`tessera/MountainCar-v1`, `tessera-rl-classic`): Gymnasium
  `MountainCar-v0` dynamics, rewards and termination in f64 with frozen math,
  cross-platform tier, a 200-step limit. One-step equivalence with the pinned
  Gymnasium 1.4.0 within SPEC 18 tolerances over 100,000 sampled states and goal,
  wall and speed-clip edges (`tools/gym_equivalence_mountaincar.py`). Golden
  fixtures, continuation, import validation by phase and end flags, a scripted
  baseline, a step bench and a persist fuzz target. In Python it is available
  through `make` and the Gymnasium adapter. `HeaderView` gains `terminated` and
  `truncated`, the end flags of a pending-reset record.
- **Brick wall** (`tessera/BrickWall-v1`, `tessera-rl-classic`): a paddle, a
  ball and a wall of 108 bricks in `Fx32`, with original rules, scoring and
  rendering: a 30-frame serve, two walls, `lives` and `frameskip` in config,
  and an `obs` choice of 84 x 84 grayscale pixels (the default) or a 116-field
  state vector. Golden fixtures, continuation, a test reference in raw integer
  arithmetic and a reference rasterizer, a scripted baseline and step benches.
  In Python it is available through `make` and the Gymnasium adapter.
- **Snake** (`tessera/Snake-v1`, `tessera-rl-classic`): one or two agents
  on a square board of side 6 to 32, simultaneous moves with no slot priority,
  `Discrete(4)` with reversal masked, and an `obs` choice of 84 x 84 grayscale
  pixels (the default) or flat `f32` flags. Fixed state of 4,110 bytes at every
  size. Each collision rule has its own test; golden fixtures, continuation, a
  test reference and rasterizer, a scripted baseline and step benches. In Python
  it is available through `make` and the PettingZoo adapters.
- **Key and door** (`tessera/GridKeyDoor-v1`, `tessera-rl-classic`): a world
  generated at each reset from the instance's stream, seen from each agent's
  position and facing, for 1 to 4 cooperative agents; `Discrete(5)` with masks
  and simultaneous turns resolved in slot order. There is no occlusion, unlike
  MiniGrid-style vision. A solver test plays every sampled world to the end;
  golden fixtures, continuation tests, a test reference, a scripted baseline and
  a step bench. In Python it is available through `make` and the PettingZoo
  adapters.
- **Python search**: `Search.is_privileged`, `False` for every search
  Python can build today.
- **Engine** (`tessera-rl-engine`): `EnvPool::config`, and `EnvPool::contain`,
  which runs author code that reads only the configuration under the
  pool's containment (the canonical floating-point scope, a panic caught
  and reported as `None`).

- **ONNX evaluator** (`tessera-rl-onnx`, ADR-0057; #245, #247, #249, #251, #255, #257): a new crate whose `onnx`
  feature (off by default) loads ONNX Runtime 1.31.1 dynamically from an explicit path, checks its exact version
  through the audited `onnx::sys`, and refuses to load unless `ORT_DISABLE_TELEMETRY=1` is set or telemetry is
  accepted explicitly. `OnnxEvaluator` implements the search `Evaluator` on the CPU provider only, with pinned
  session settings, a model size limit, no external-data files, and provenance (model and library sha256, version,
  threads). In Python, `tessera_rl.search.OnnxEvaluator` and `tessera_rl.OnnxError`, the `[onnx]` extra
  (`onnxruntime==1.31.1`), and inference with the interpreter released. Every release wheel has the feature; the
  library comes from the extra or an explicit `library=` path. Linux and macOS only.
- **Gumbel root search** (`tessera-rl-search`; #226, #230, #234): an exact open-interval Gumbel sampler on the
  search stream, and the Gumbel root (top-k, sequential halving, completed Q, sigma and the improved policy) for
  single searches and search groups, in Rust and in `tessera_rl.search`.
- **RTS** (`tessera-rl-rts`; #141, #143, #144, #146, #155, #156, #161, #175): a new crate with a real-time
  strategy environment: generated maps, a unit table, move resolution over flow fields, production, fog of war with
  explored-cell memory, 11 observation planes, multi-head actions with conditional masks, and scripted bots that
  read only their side's view. In Python, `make_rts` with mask head views and a PettingZoo Parallel wrapper;
  PufferLib and JAX refuse it.
- **Conditional masks** (`tessera-rl-engine`, ADR-0051, #133): entity groups declared at construction, and a
  per-entity illegal-action fallback with the penalty applied once per agent per step.
- **SIMD crate** (`tessera-rl-simd`, #126): a standalone eight-lane CartPole step, scalar, NEON, SSE4.2 and AVX2,
  checked bit for bit against the scalar physics, and frozen lane sine and cosine in `tessera-rl-num`. No pool uses
  it yet, and no speed is claimed.
- **Python actions**: generic `MultiDiscrete` actions (#162), and continuous actions through the binding and the
  Gymnasium `Box` adapter (#123), one f32 head for single-agent environments in Sync, Pipelined and Async.
  PettingZoo, PufferLib and JAX refuse continuous environments.
- **JAX handles** (SPEC 15.2, ADR-0031, #119): reset and step through ordered `io_callback`, valid under `jit` and
  `lax.scan`; vmap and mapped axes are refused. JAX and jaxlib 0.10.2 as an optional extra. CPU only.
- **Linux aarch64 wheel** (#167), built on a GitHub-hosted arm64 runner in the pinned manylinux_2_28 image.

### Performance

- **Async mode** (`tessera-rl-engine`): workers claim and publish runs of ids with one compare-and-swap each
  (ADR-0056 decision 1, #170), and waits are wake-based instead of polling, with timed futex waits on Linux,
  macOS 14.4+ and Windows (ADR-0056 decisions 2 to 4, #187). `AsyncPool::snapshot(id)` needs only that row to be
  idle, not the whole pool (#158).
- **Connect Four `observe`** (#246): writes one fixed-size 84-byte view with a single bounds check, whatever the
  codegen-unit split.
- **Classic environments** (#250): Snake, BrickWall and PaddleDuel no longer clear leaves the engine already
  zeroes (ADR-0058).

### Fixed

- **SPEC 10 authoring example (#64), a 0.1.0 known deviation, resolved:** the SPEC now shows the real API
  (ADR-0053). `type Action` stays an untyped marker (`()`); the example declares `type Header`,
  `observation_space` and `action_space` and reads actions through `Actions<'_, ()>`. It is compiled verbatim
  as `crates/tessera-rl/tests/spec10_example.rs`. Its `register!` line waits for ADR-0054 (#63).
- **Async after a panic or a death** (`tessera-rl-engine`): `snapshot` re-checks
  poison after its quiescence check, so a step that panicked in between cannot expose
  a partly written row; `is_poisoned` is true once a worker died, in Pipelined too, and send, partial writers and `recv` report `WorkerDied`; `wait_ready` no
  longer reports ready for a moment when a receive pops an id before its worker counts it.
- **Async writers right after the last `recv`** (`tessera-rl-engine`): a writer or a
  snapshot could refuse with a retryable `NotQuiescent` although every id had been
  received, because a worker counted an id as stepped after making it visible. It
  now counts first, as ADR-0047 section 3 requires.
- **Async batch cost** (`tessera-rl` Python): gathering a batch searched the
  token per id and per field, quadratic in its size. It now takes one pass, through
  the new `BatchToken::rows`.
- **Fork safety** (`tessera-rl` Python, #93): a forked child using an environment
  created before the fork hung, or crashed on macOS. In the child such an object now
  raises the new `ForkError` from every call except `close`, and neither `close`,
  dropping it nor interpreter exit joins its workers. A new environment made in the
  child works normally.
- **Stale views during a writer** (`tessera-rl` Python, merged in #88):
  borrowed Pipelined arrays reported `is_stale() == False` while `reset` or
  `restore_all` rewrote their rows, because the generations move only after
  the write. Every split-phase array now reads stale from before a writer's
  first write until it ends, and fresh again if it was refused.

## [0.1.0] - 2026-11-04

The first functional release. Six of these crate names were reserved on
crates.io at 0.0.1; `tessera-rl-games` and `tessera-rl-search` are new.

### Added

- **Core** (`tessera-rl-core`): the `Environment` trait and its family
  (`ActionMask`, `Validate`, `Heuristic`, `ImperfectInformation`,
  `Determinize`, `Render`, `EnvAssets`), spaces, `StepOut`, agent slots,
  the canonical encoding and `StateHash`, and the plain-data state
  contract (`Bool8`, `Reserved<N>`, `Pool<T, CAP>`, `StateEnum`).
- **Numerics** (`tessera-rl-num`): the counter-based Philox RNG with
  per-instance identities and branch salts, the `Fx32` and `Fx64` fixed-point
  types, frozen deterministic math, and the floating-point scope guard.
- **Macros** (`tessera-rl-macros`): the `EnvState`, `EnvConfig`,
  `Observation` and `StateEnum` derives, with compile-time rejection of
  padding, missing `#[repr(C)]` and non-plain fields.
- **Engine** (`tessera-rl-engine`): `EnvPool` with the `Inline` and `Sync`
  modes (pinned worker threads, futex waits on Linux), the `NextStep`,
  `SameStep` and `Disabled` auto-reset modes, action masks, `Sequential`
  and `Mixed` turns, poisoning, the validation boundary, and snapshot,
  restore, fork and branched fork of instance records and pool context.
- **Search** (`tessera-rl-search`): the bounded `SearchPool` and a batched
  PUCT MCTS helper with virtual loss and root noise.
- **Environments**: `tessera/CartPole-v1` and `tessera/Tag-v1`
  (`tessera-rl-classic`), and `tessera/Connect4-v1` (`tessera-rl-games`),
  with golden trajectory and continuation tests.
- **Facade** (`tessera-rl`): re-exports of the crates, a `prelude`, and the
  `debug-checks` feature forwarded to the engine
  (`tessera_rl::engine::DEBUG_CHECKS` reports it).
- **Python** (`tessera-rl-py`, built with maturin, abi3, Python 3.11+):
  `tessera_rl.make`, owned and borrowed outputs, a Gymnasium 1.x `Env` and
  `VectorEnv` (`tessera_rl.gym`), and `Search` and `SearchGroup` over a
  Python evaluator.
- **Example**: an AlphaZero-style Connect Four training pipeline
  (`examples/alphazero_connect4`) with a core-hour ledger, a supervised
  phase harness and a remote phase transport.

### Not included in 0.1.0

- **The phase 2 exit gate has not been run.** Whether the AlphaZero-style
  example beats the scripted Connect Four bot in at least 90% of 1,000
  games is not established.
- **The example's training run is incomplete.** It was stopped after 8
  generations (51 of its 200 core-hour budget) by the owner's decision and
  is deferred. The best held-out look so far won 310 of 400 games; the
  looks are a selection aid, not a gate estimate.
- **`SearchGroup` scope.** Its throughput was measured once and it was
  adopted (results in
  `examples/alphazero_connect4/results/group-bench-20261103`), but the
  example's training loop still uses single-tree search. The measurement is
  shared-host throughput on one machine, not a published benchmark.
- **No throughput figure** is published with this release.
- **0.1.0 is unverified on Windows.** CI covers Linux x86-64, Linux
  aarch64 and macOS (aarch64); Windows is not in CI. The last Windows runs
  were on 2026-11-01: a native hand check of the Python Win64
  floating-point boundary at `cc5c97d` (#29), and GitHub's Windows CI of
  the Rust workspace, last green on main at `105be6c`. Both the Rust code
  (search and engine) and the Python native boundary have changed since.
  In particular the Win64 thunk in this release has never run on Windows.
- **Planned, not implemented:** PettingZoo and PufferLib adapters,
  `Async` and `Pipelined` modes, persisted snapshots, Othello, Go 9x9 and
  Leduc hold'em (phase 2); SIMD lanes, JAX and GPU paths, Snake, Pong and
  Breakout (phase 3); the RTS, market simulator, card games, survival world,
  league tooling and viewer (phase 4). Their crates do not exist yet.
- The Python package is not published to PyPI.

### Known deviations from the specification

Code, comments and documents cite a project specification ("SPEC") and
architecture decision records ("ADR-00nn") that are not published in this
repository, so a reader cannot open them. Where the specification and the
code disagree, the specification is the target, and these are open
defects:

- `debug-checks` does not provide the NaN detection and allocation
  tracking the specification requires (#62). This release is knowingly
  non-conformant with that requirement.
- `tessera_rl::register!` is not implemented (#63).
- There is no typed `Discrete<N>` action type, and the specification's
  authoring example does not compile: it also omits `type Header`,
  `observation_space` and `action_space`, which the trait requires (#64).
  The README quick start is a compiled doctest of the real API.

### Known issues

- #43: `alloc_sync` counted 4 allocations in the `Sync` step window once,
  on a GitHub-hosted ubuntu-latest runner with default features. It has not
  been reproduced: 1,200 earlier runs, then 5,000 runs in a local Linux
  container and 10,000 on an x86-64 Linux host (two variants, 4 pinned
  CPUs). The GitHub-hosted environment could not be retested because of
  account billing. The test now records each counted allocation (its size,
  the thread and its name, whether the thread was in a step or reset kernel
  or in neither, and that thread's kernel call count), so a recurrence will
  be diagnosable. The assertion
  stays at zero allocations.

### Notes for dependent crates

- Depend on the facade `tessera-rl` only; the derives resolve their paths
  through it.
- This repository's `clippy.toml` and `rust-toolchain.toml` do not apply
  to a dependent crate. To keep simulation kernels deterministic across
  platforms, configure the same `disallowed_methods` lints (no `mul_add`,
  no `powi`, no `std` transcendental methods; use `tessera_rl::num::math`)
  in your own crate.

[Unreleased]: https://github.com/joaquinbejar/tessera-rl/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/joaquinbejar/tessera-rl/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/joaquinbejar/tessera-rl/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/joaquinbejar/tessera-rl/releases/tag/v0.1.0
