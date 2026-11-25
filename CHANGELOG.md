<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->

# Changelog

All notable changes to this project are documented in this file. The
format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project uses [Semantic Versioning](https://semver.org/). Before 1.0
the API is not stable, and any 0.x release may break it.

## [Unreleased]

### Breaking

- `PoolOptions` gains the public field `schedule: Schedule` (ADR-0076, #357, #439): struct literals without `..` must set it; `Schedule::Static` keeps the previous behaviour.
- `PoolOptions` gains the public field `simd_level: Option<Level>` (ADR-0071, #679): struct literals without `..` must set it; `None` keeps the previous behaviour (the widest level this CPU has, for environments with a lane step).
- `SyncOptions` gains the public field `wait_policy: WaitPolicy` (ADR-0074, #434): struct literals without `..` must set it; `WaitPolicy::Fixed` keeps the previous behaviour.
- `PersistError` gains the variants `Transforms` (ADR-0078, #346, #478) and `AssetMismatch` (ADR-0077, #364, #419): an exhaustive `match` on it needs arms for both.
- `tessera/Grid-v1` is environment version 2 (ADR-0077, #364, #419): a `layout` wall-map asset (`"open"` by default, `"rooms8"` bundled, or bytes). The default reproduces version 1's trajectories (golden step digests unchanged); its config hash and persisted files change.
- `Render` is no longer an empty marker trait; an `impl Render for X {}` must now provide `canvas` and `draw` (ADR-0083, #378, #438).
- Experimental host API (ADR-0062 item 9; ADR-0073, #344, #404): `HostPool::new` takes the configuration by value and keeps it, as `EnvPool::new` does (ADR-0073 needs it after construction). Replace `HostPool::new(&cfg, options)` with `HostPool::new(cfg, options)`. A type-erased pool (`DynHostPool`, `make_host`, `register_host_factory!`, `export_host_env!`) now needs a `Send` configuration. `DecodeLimits` gains the field `max_host_record_bytes`, so struct literals need it (or `..DecodeLimits::default()`).
- Experimental `tessera-rl-pyffi` (ADR-0095, #369, #684): `errors::Kind` gains `Order` (the new `OrderError`), and `Kind` is not `#[non_exhaustive]`, so an exhaustive `match` in a host wheel needs an arm for it.
- `OnnxError::SetObservation` is removed (#505, #523): the ONNX evaluator takes `Set` observations, so the refusal is never returned.
- `SearchError::SetObservation` is removed (#490, #512, ADR-0082 point 7 as amended): `SearchPool` accepts `Set` observations, so the variant is never returned.
- `EnvPool::epoch_barrier` and `PipelinedPool::epoch_barrier` return an `EpochReport` instead of `()` (ADR-0081, #359): episode statistics, and later the curriculum's changes.
- `PoolContext.transforms` moved to `PoolContext.adaptive.normalization` (ADR-0081 item 6, #359, #515): `PoolContext.adaptive` is an `AdaptiveContext`, one entry per adaptive subsystem.
- Replay format version 1 is frozen (#411, #430): the policy map, the curriculum flag, the archive and reset source, and the chunk size are in the header; event tags 7, 10 and 11 are reserved and validated; `ReplaySummary::unclaimed` lists ids an Async file never received.
- `RecordLayout::Adaptive` requires `Send + Sync` (ADR-0081 item 5.3, #359): pools hold adaptive values as per-slot curriculum targets. Generated `<Config>Adaptive` types are plain data and always both; only a hand-written `RecordLayout` impl over a non-`Sync` type stops compiling.
- The `_reserved` padding field of a derived `<Name>Adaptive` struct (`#[derive(EnvConfig)]` with `#[config(adaptive)]` fields) is no longer `pub` (#613, #669): only the generated code builds it. A struct literal that set `_reserved` no longer compiles: build the value with `AdaptiveValues::from_config(&cfg)` (or `StateInit::state_default()`) and set the adaptive fields; patterns use `..`.
- Replay format version 1 reserves the league fields of ADR-0088 item 8 (#384), as it did for ADR-0075, ADR-0079 and ADR-0081: a `has_league:u8` header byte after `has_curriculum`, policy-map group code 4 (`opponent`), and event tag 12 (`League`). Writers write 0 and never code 4 or tag 12; replay refuses a league, code 4 and tag 12 as `Unsupported` until the league lands. `replay::Event` gains a `League { body }` variant and `EventTag` a `League` tag, so exhaustive matches need an arm. No release has shipped replay files.
- `EnvPool::state(id)` returns `Option<State>` by value, not `Option<&State>` (ADR-0090, #345, #543): a lane-wise state is gathered from its columns, and one API serves both stores. `EnvState: Copy` makes it cheap. Callers that took `*pool.state(i).unwrap()` drop the `*`; callers that pass it on take `&`.
- `EnvState` gains `const LANES: bool = false`, `type Fields` and `to_fields` / `from_fields` (ADR-0090, #345, #543). `#[derive(EnvState)]` emits them. A hand-written implementation adds `type Fields = ();`, an empty `to_fields` and a `from_fields` that returns the default state, as `EnvState`'s docs show.
- `EnvState` requires `Send + Sync` (ADR-0075, #363): worker threads read shared archive states. Plain-data state (ADR-0029: no pointers, no interior mutability) is always both, so no environment changes; a hand-written `EnvState` impl on a non-`Sync` type no longer compiles.
- `PoolContext` is `#[non_exhaustive]` and no longer `Copy` (ADR-0075, #363): it gains `reset_source`; build it with `PoolContext::new(branch_counter)`. `restore_pool_context` takes the archive a context names: `restore_pool_context(context, archive)`. ADR-0078 (#346) then adds `transforms: TransformContext`, the transform identity, statistics and freeze flag.

### Added

- Documentation: an API stability page on the site (ADR-0094, #642, #688, #700). It says what before 1.0 means and which APIs are tracked. In Rust, that is the `tessera-rl` facade and what it re-exports, default features plus `onnx`; not hidden items, test-only features, the host-environment API or the tool crates. In Python, it is the names in `tessera_rl.__all__` and the modules `gym`, `pz`, `puffer`, `replay` and `render`, with their stub signatures; not `_` names, the other public modules (`eval`, `jax`, `masks`, `search`, `transforms`, `viewer`) or `HostEnv` and `HostStep`. It explains how the snapshots `api/tessera-rl.api` and `api/tessera-rl-py.api` track them, and how each release finds candidate breaking changes (`git diff <previous tag> -- api/tessera-rl.api api/tessera-rl-py.api`). CONTRIBUTING.md asks for `make facade-api-bless` or `make py-api-bless` in the pull request that changes either API.
- XLA FFI handles for JAX, experimental (SPEC 15.2, ADR-0095, #369, #588, #636, #684): `tessera_rl.jax.ffi_handle(env)` (`FfiHandle`) runs a Sync `Env`'s native step inside XLA, without the interpreter and with one copy of the outputs. Calls are ordered by an explicit token, `token, step = h.step(token, actions)` and `h.reset(token, seed)`, with `h.token()` giving the next value; a stale, replayed or reordered token is refused before anything is stepped, with `tessera_rl.OrderError` in `last_error`. The handle shares the `Env`'s reentrancy guard (`ConcurrencyError`). Refused with `CapabilityError`: architectures other than x86-64, `privileged_state=True`, continuous actions and the split-phase modes; several action heads raise `ArgumentError`. A panic in the native step poisons the handle and the `Env`. `close()` or the context manager ends it.
- Lane steps over Array-of-Structs chunks (ADR-0071, #679, #680). An environment implements `LaneStep` and sets `Environment::LANE_STEP` to `Some(lane_hook::<Self>())` (the default `None` keeps every instance on `Environment::step`). The engine then steps each eligible group of `LANES` (8) consecutive instances of a chunk through `LaneStep::lane_step`, reporting per lane what `step` would (`LaneOut`), and every other instance through `step`; bookkeeping, observe and resets stay per instance. `PoolOptions::simd_level` picks the SIMD level (`Level`, re-exported by `tessera-rl-engine`): `None` takes the widest this CPU has, and a level it lacks is `EngineError::SimdLevel` at construction, never a silent fallback. The level changes speed only, never results. `tessera/CartPole-v1` implements it, with its environment version and goldens unchanged; CI also runs the goldens with the level forced to scalar. Python reads `TESSERA_RL_SIMD` (`scalar`, `sse42`, `avx2`, `neon`) when a pool is made, for tests and A/B runs only.
- ONNX league opponents from Python (#619, ADR-0088 item 2.3 and erratum 2, ADR-0057).
  - `make(..., league=league, onnx_opponents=tessera_rl.search.OnnxOpponents(models=[...]))` plays a league's `("onnx", hash)` policies with the given models, identified by content hash. Every such hash must be served (`ArgumentError` naming it); unused models are allowed. A build without the `onnx` feature refuses `onnx_opponents` (`CapabilityError`).
  - `Env.onnx_provenance` records each model's provenance by content hash, read-only. Results are repeatable only in ADR-0057's class: same machine, same library and model bytes, same batch composition; never across platforms.
- Curricula from Python (ADR-0081 items 3 and 5.5, #613).
  - `make(..., targets=True)` builds a Sync pool with curriculum targets, for an environment with adaptive config fields (Gather's `items` today). Other environments, and Pipelined or Async mode, raise `ArgumentError`.
  - `Env.adaptive_fields` names the fields.
  - `Env.targets()` returns a copy of each instance's targets.
  - `Env.set_targets({instance: {field: value}})` changes some fields of some instances. Every change is checked before any write (`ArgumentError`, or `ConfigError` for a value the config refuses), and each applies at that instance's next reset.
  - `Env.epoch_barrier(curriculum=f)` calls `f(report, targets)` after the barrier, with a copy of the targets, and applies the changes it returns.
  - A recording keeps the changes (tag 7), not the callable, so a Python-curriculum run replays into a `targets=True` pool without it.
- `tessera/Gather-v1` is environment version 2 (ADR-0081, #613, #669): `items` is adaptive (`#[config(adaptive)]`), so a pool with targets places each episode's items from its slot's target, read at reset, and a curriculum can change the difficulty between episodes. At the default, with no targets, version 2 gives version 1's trajectories; only the record and header encoding differ, so its config hash and persisted files change. Gather is new in this release, so no released file is affected.
- Independent multi-head actions in the PufferLib adapter (#617, #657): an environment whose action heads have no conditional masks gets a `MultiDiscrete(nvec)` action space, with flat masks holding each head's columns at offset `sum(nvec[:h])`; hidden heuristic slots and Pipelined halves keep the head axis. Heads with parents (LimitOrderBook-v1) are refused with an `ArgumentError` naming the head. `tessera/Survival-v1` joins `SUPPORTED`.
- Multi-leaf observations in the PufferLib adapter (SPEC 11.1, #423, #551): an agent's row is the flattening of its leaves in layout order (a `Set` leaf's items, then `<leaf>_mask`), with dtype `numpy.result_type` of the leaf dtypes. A layout whose promotion is not lossless (a float whose mantissa cannot hold every value of an integer leaf, or `uint64` with a signed integer) is refused with `ArgumentError`.
- Flagship benchmarks, head 3 (#640): `bench.py --max-load5 B --load-wait S` gates every process run on load5 < B, polling every 30 s within a cumulative budget of S seconds (past it the invocation fails and writes no report), and voids a run whose load5 after it is at least B + k_run (its worker count). The report's `load_rule` records each run's load1 and load5 before and after with both bounds; a void run marks the report `load_void: true`, which the dashboard refuses. `tools/dashboard/envelope.py` wraps a report into a result file whose top-level `loads` carries the start and end load averages and the per-run loads, checked by the dashboard for reports made under the load rule. `compat/machines.toml` gains `ref-x86-alt`, a candidate record for the fallback host. Survival's scaling notes state that Sync at chunk 16 is compared against Inline at chunk 64. Flagship files are named `<UTC date>-<machine>-<short commit>-<flagship>.json`.
- Flagship benchmarks, head 2 (#640): flagship scaling is labelled "efficiency vs k×Inline" (k-worker Sync throughput over k times Inline) in the report notes, the flagship table and the READMEs. Survival's multi-worker configurations use `chunk_size = 16` (16 chunks at N = 256, at least the largest `bench_cpu_set`), which the runner takes as `--chunk-size`; every runner row records the effective `chunk_size`. Each flagship configuration states `illegal_action_noops`, shown as a column of the flagship table. On a full run (no `--only`), a scaling row whose multi-worker result is missing is an error.
- Flagship benchmark configurations (#640, head 1): `tessera/Lanterns-v1` gets a bench entry (`rust/lanterns/inline/n1024`, legal actions under its flat mask), and each phase 4 flagship (RTS, LimitOrderBook, Lanterns, Survival, JobShop, Vrp) has the four SPEC 19.1 metrics at one N: Inline throughput, Sync mode at 4 workers and at the machine's full `bench_cpu_set` (`workers = "full"`) with scaling efficiency against Inline, snapshot costs and memory. Every multi-worker configuration names its `schedule` (Static or Dynamic), which the runner takes as `--schedule` and the report records with the effective worker count. The dashboard builder writes a flagship table (`flagships.html`) of the valid and indicative result files, each row with its machine's `compat/machines.toml` status, commit and `valid_for_claims`. Nothing is measured yet: numbers stay indicative until a reference machine exists (#151).
- `tessera-rl-world` stage 4 (#382): `tessera/Survival-v1` from Python and the facade. `tessera_rl.make` builds it (every config field by name, masks, the `Set` leaf `others` with `others_mask`), `tessera_rl.pz.parallel_env` serves it (Parallel only), and `policy_map` heuristic slots run the scripted baseline from each agent's own view (`InfoHeuristic`). The facade re-exports the crate as `tessera_rl::world`.
- Windows x86-64 wheels (#350, option A): wheels.yml gains a GitHub-hosted `windows` job. It builds `win_amd64` without the `onnx` feature, because ADR-0057 has no Windows loading rows, and runs the golden anchor, the Python suite with FP traps expected, the num tests and the SBOM check, like the other legs. pypi.yml and `scripts/pypi_verify.py` now expect six wheels. The job runs for the first time at the next release.
- Search over accumulated returns (ADR-0096 with erratum 1, #524). `PuctConfig::value` is `ValueMode::Terminal`, the default and ADR-0043 bit for bit, or `ValueMode::Returns { discount }`.
  - Under `Returns`, each edge backs up `r + discount * V`: the mover's step reward plus the discounted value below it, which is the evaluator's at a leaf and 0 at a terminal node. Terminal rewards and evaluator values have no `[-1, 1]` bound.
  - PUCT selection and the Gumbel root use the call's min-max of edge means; a virtual visit counts as a normalized 0.
  - `Returns` needs one mover on every path: a single agent, or an ADR-0093 searcher pool. Anything else is refused as `SearchError::ReturnsNeedOneMover`. A discount that is not finite or not in `[0, 1]` is `SearchError::Options`.
  - A non-finite return, from a reward or from an overflow, is `SearchError::NonFiniteReturn { node, origin }`. It is checked before any edge of the path changes, and the call rolls back.
  - Python: `Search` and `SearchGroup` take `value="terminal" | "returns"` and `discount=` (default 1.0, only with `"returns"`).
  - The order book (`LimitOrderBook-v1`) searcher pool backs up its per-step profit and loss in `Returns` mode.
- `tessera-rl-world` stage 3 (#382): the scripted baseline of `tessera/Survival-v1`, a forager and fighter acting from the agent's own view only (`InfoHeuristic`, and `Heuristic` through it): attack a neighbour while healthy, drink or forage when low, step towards the nearest needed resource in the window, otherwise wander. Every action it plays is legal under the masks. Survival-v1 declares imperfect information; its `InfoState` is the agent's `own` values and window, the observation's bytes. Over 400 steps with 32 agents and no respawn, the baseline keeps 11.2 to 15.9 agents alive on average, against 9.5 to 10.6 for a uniform legal policy.
- `tessera/Survival-v1` version 2, `tessera-rl-world` stage 2 (#382): observations and masks. `window`, a `[4, 2 * view + 1, 2 * view + 1]` `u8` tile window (terrain, ready forest, agent present, its health); `own` as before; `others`, a `Set` of up to 16 visible agents, nearest first (squared distance, then `dx`, then `dy`: no slot order), each `[dx, dy, health]`. Masks: a move into a passable tile with no live agent at the start of the step, forage on a ready forest, drink next to water, attack a live neighbour; stay and none are always legal (the no-op). Version 1 was never released. The goldens hash the observations too.
- `tessera-rl-world` stage 1 (#382): `tessera/Survival-v1`, a massive multi-agent survival world. 16 to 128 agents move, forage, drink and attack at once on a generated tile map (integer value noise with order-statistic thresholds and a connectivity pass). Every phase reads the start-of-phase state, so slot order does not change the outcome: a move into a tile occupied at the start of the step fails, contested targets stay, and mutual attacks both land. Health, food and water are integers; agents respawn after `respawn` steps (on by default) as new agents. The reward is +1 per agent alive; kills are counted, not rewarded. Integer state (20,612 bytes at every config). Observations are a minimal `own` leaf in this stage (the tile window and the visible-agents set come in stage 2). Goldens (16 and 128 agents, every worker count), a map golden, continuation and persisted-bytes tests, slot-permutation tests, a proptest, an allocation test and Criterion benches. Rust only for now.
- Teams and roles (ADR-0079 item 8, #347 stage B4): defaulted `Environment::team(cfg, slot) -> u16` (0) and `Environment::role(cfg, slot) -> &'static str` (`""`), configuration facts constant for the pool's life, read once per slot when a Python pool is built and never on the step path. Python: `layout()["teams"]` (one int per slot) and `layout()["roles"]` (one string per slot); PettingZoo `metadata["teams"]` and `metadata["roles"]`, by agent name, on the Parallel and AEC forms (`infos` are unchanged). Tag: pursuers team 0 with role `"pursuer"`, evaders team 1 with role `"evader"`, unused slots the defaults; PaddleDuel and Snake: team = slot; RTS: team = the slot's side after `seat_swap`. LimitOrderBook keeps the defaults: its config does not distinguish market makers from traders. Host ids get team 0 and role `""` for every slot; host protocol 1 is unchanged. `make(..., policy_map=...)` now runs heuristic slots for PaddleDuel and Snake (their scripted baselines).
- `Env.terminal_observations` (#545, #574): the final observations of the last whole-pool `reset` or `step` in Sync mode, before their same-step reset, as a leaf dict shaped like `Step.observations` (rows of instances that did not end are zero). Borrowed mode gives read-only views with `Step`'s borrow window, owned mode a fresh copy per read; `ArgumentError` without `auto_reset="same_step"`, before the first call and after a Pipelined half or an Async batch. The PufferLib adapter's `terminal_observations` now reads it. `tools/learning_twoplayer.terminal_obs` still reads `Env._last_out`, as that file is inside the calibrated T3b fingerprints; its switch waits for #571.
- `tessera/Vrp-v1` version 2 (#565, #581): a `fleet` observation leaf, `[vehicles, 3]`, holding every vehicle's node, load and distance travelled, the observer first, so a vehicle in a fleet sees the others. `ENV_VERSION` is 2 under the same id; version 1 was never released. The golden and persistence fixtures are re-blessed, and the `vrp` learning calibration is redone on version 2.
- League opponents in the engine (ADR-0088 sections 2, 3, 4.1 and 7 with erratum 1, #384 head B): `PolicyGroup::Opponent` (replay code 4: it round-trips through the replay header's policy map, #347 B2, but recording or replaying a pool with an opponent slot is refused as `Unsupported` until head D); `EnvPool::with_league(slots, view, subsystem)` and `with_info_league`; each instance's opponent policy is drawn at every episode from the `LeagueView` on a new `LEAGUE` RNG stream (a pure function of the record's identity, its episode counter and the view), played by the environment's heuristic or submitted by the caller (`External`); the `assignment()` output; per-instance result rings filled at episode end; the league as the epoch barrier's fourth subsystem (`EpochReport::league`, `freeze_league`); fork keeps the opponent, a plain restore or import redraws it, `snapshot_league` / `restore_with_league` are the exact path; the pool context carries the view and the assignments. `tessera_rl_league::PoolLeague` implements the engine's sealed `LeagueSubsystem`; `tessera-rl-league` now depends on `tessera-rl-engine`. Python, roles and several pools (head E) come later.
- ONNX league opponents (ADR-0088 item 2.3 with erratum 2, #384 head C): `OpponentKind::Onnx { content_hash }` in the `LeagueView`, and `PoolLeague` passes ONNX policies through (`LeagueError::OnnxUnsupported` is removed). `EnvPool::with_onnx_opponents(policies)`, before the first reset and once, takes a sealed `engine::league::OpponentPolicies` object. Each step, after the boundary copy, the pool makes one call per ONNX policy on the calling thread, in policy, instance and slot order, over the observation rows of the previous step. It writes the actions into its own action copy, and the workers validate them as heuristic opponents' (an invalid one, or one that is not an `i32`, poisons with `OpponentAction`; a policy's own failure is `EngineError::OpponentPolicy` and does not poison). A step meeting an ONNX opponent without policies is refused before any change. Pools without ONNX opponents do no extra work. `tessera_rl_onnx::OnnxOpponents` implements it, with one session per model at the evaluator's pinned settings, a model being identified by `content_hash` (the first 16 bytes of the file's sha256). Its inputs are the evaluator's and its output is `actions`, `int64` `[N]` or `[N, 1]`; continuous action heads are refused. Recording or replaying a league pool stays refused until head D.
- League replay (ADR-0088 section 8 with erratum 3, #384 head D): the sync `Recorder` records a league pool. It writes `has_league` 1 and policy-map code 4. The first event is a `League` event (tag 12) holding the league's whole view at recording start (digest, last drawable entry, kinds, thresholds; `replay::MAX_LEAGUE_POLICIES` policies at most). Every keyframe then carries each instance's assignment after the records. `Recorder::epoch_barrier` runs the barrier and writes a `League` event when the view changed; `Recorder::freeze_league` passes through. Replay checks the caller-built pool before any event: a league on one side only (`EngineError::League`), another population (`EngineError::LeagueMismatch`), another table (`EngineError::League`), or a recorded ONNX policy the pool does not serve (`EngineError::OpponentPolicy`). Replay restores keyframes with their assignments, so a recording that starts after barriers keeps the opponents drawn from older views. A later `League` event replaces the view without calling the subsystem, and seeks restore the view and assignments of their keyframe. A `has_league` file whose first event is not `League` is malformed. A keyframe assignment that differs on replay is the new `ReplayError::AssignmentDiverged { event_index, instance, recorded, replayed }`, a keyframe divergence that names the policies (Python reports it as the `"keyframe"` kind). Files without a league are byte-identical to before, pinned by the existing fixtures and a new test. Async and host files with a league stay refused.
- A league shared by several pools (ADR-0088 section 6, #384 head E1): `SharedLeague::new(SharedConfig { roles, .. }, population)` holds one population and one learner per role. A `RoleConfig` sets each role's matchmaking, an optional `SelfPlay { share, tag }` entry and a `SnapshotRule` (`EveryRounds(n)` or `WinRate { threshold, min_games }`, adding `External` entries tagged from `snapshot_tags`, listed in `CommitReport::snapshots`).
  - Pools register with `member(role)`, in a fixed registration order.
  - The barrier is two-phase: a pool's barrier of round `r` hands in its results and gets the view committed at round `r - 1`. The round's last barrier commits, merging in registration order, so the result does not depend on thread timing or barrier order, and every pool installs the new view at its next barrier, where a recorder writes it as a `League` event. A shared league therefore lags one round, one pool included; `PoolLeague` does not.
  - A second barrier before the round commits is refused with `EngineError::LeagueBarrier` naming the pools still to submit. The pool is unchanged and nothing blocks.
  - The self-play entry gets exactly `floor(share x 2^64)` of its role's draws (`Table::with_reserved_share`), and its results count in its pair counts but not in the ratings.
  - `SharedLeague::barrier` runs the members' barriers (`LeagueMember`).
  - A pool frozen with `freeze_league` submits an empty round and keeps its view, so it never stalls the others: the engine's `LeagueSubsystem` gains `frozen_barrier`, called at a frozen pool's barrier, a no-op by default (`PoolLeague` is unchanged).
- League persistence and export (ADR-0088 item 7.5, #384 head E2):
  - `League::save`/`load` and `SharedLeague::save`/`load` write and read a canonical, versioned file (`LEAGUE_MAGIC`, `LEAGUE_FORMAT_VERSION`, an XXH3 checksum checked before anything is parsed). It holds the population, every learner's ratings, the per-pair counts, the snapshot and round state, and the members' roles in registration order.
  - Tables are recomputed at load and must reproduce each role's saved view digest.
  - A shared league saves only between rounds. After a load its pools register again with their saved roles, in the saved order, and barriers are refused until they all have.
  - `check_view(role, view)` checks a restored pool's view against the one its role's pools hold, which is one round behind under the shared league's lag.
  - `export()` gives the JSON (RFC 8259 strings, shortest round-trip floats, non-finite refused, fixed key order).
  - New errors: `LeagueError::Persist`, `LeagueError::ViewMismatch`.
- Leagues from Python (ADR-0088 section 10, #384 head E3):
  - `tessera_rl.League(population, roles, elo_k=, trueskill=, rate_population=)` creates a shared league, with `save()`, `League.load(data)`, `export()`, `round`, `population()` and `ratings(role)`.
  - `make(..., policy_map={slot: "opponent"}, league=league, role=r)` attaches a pool. A league from Python is always shared, so it lags one round, with one pool or many.
  - `Env.assignment` gives each instance's opponent, and PettingZoo's infos carry it as `assignment`. `visible="learner"` hides `"opponent"` slots.
  - A league with an ONNX entry is refused by `make` with `CapabilityError` naming the entry (ONNX opponents from Python come later, #619). So is a league on an environment without in-engine heuristics, and async and pipelined modes refuse it.
  - The Gymnasium adapter refuses a league pool with `CapabilityError`: its opponent rows may be the caller's, so it goes through PettingZoo or `Env`.
  - A pool restored (`restore_pool`, `restore_pool_context`, replay's keyframes) from a record that was never reset, or that waits for its reset, draws its league opponent at its reset exactly as the source pool's instances do.
- `tessera eval` (ADR-0088 section 9 with erratum 4, #384 head E4): `python -m tessera_rl.eval`, with `cargo tessera eval` as a thin wrapper (`$TESSERA_PYTHON` or `python3`).
  - It plays a policy (`--policy MODULE:NAME`, `policy(observations, info) -> actions`, checked) against each opponent of `--population` or `--league-export` (`--ids`). External opponents are played by `--external TAG=MODULE:NAME`; ONNX ones are refused (#619).
  - Each opponent plays exactly `--episodes` episodes from `--seeds`, on `--pools` single-instance pools of a one-entry league, in waves flushed by league rounds. A count other than N, a dropped result, or an episode longer than `--max-steps` (100,000 by default) is a hard error.
  - The JSON gives per opponent the win, draw and loss counts (the engine's outcome rule), Wilson 95% intervals of the win and score rates, mean returns, the setup and the provenance. The same arguments give byte-identical JSON.
- Learning calibrations for `tessera/JobShop-v1` and `tessera/Vrp-v1` (#383 stage 4): seeds 101 to 140 at 500k steps on the nightly host, from a clean tree of main `0cedf91a`. JobShop: threshold -816.4, median -760.9, against -856.4 for a random policy and the MWKR bar -648.15. Vrp: threshold -1237.35, median -1203.7, against -1568.8 for random and the nearest-neighbour bar -792.05. Both learn (`learning_gate.learns`), with no non-learning seed.
- The privileged state buffer (ADR-0079 item 7, #347 stage B3): a `PrivilegedState` capability trait (`state_space`, `write_state`) and `EnvPool::with_privileged_state()`, before the first reset. The engine writes one row per instance, zero while pending, in every call that writes the observations (reset, step, restore, fork, import); the rows are regions of the shared output storage (`OutputField::PrivilegedState`, `Outputs::privileged_state`), with the same borrow windows and write generation. Sync and Inline modes; Pipelined and Async refuse it (`EngineError::PrivilegedState`), as does a search scratch pool. Leduc hold'em (both cards and the public state) and LimitOrderBook-v1 implement it. Python: `make(..., privileged_state=True)`, `Env.state`, `layout()["privileged_state"]`, PettingZoo `state()` and `state_space`, Gymnasium `info["state"]`; other environments and host ids raise `CapabilityError`.
- Search against heuristic opponents (ADR-0093 with erratum 1, #491, #541): a search pool whose scratch pool has ADR-0079 heuristic slots plans for one searcher, `SearchOptions::searcher` (or the determinization agent), against the opponents the engine computes in the scratch step; every ply is the searcher's and backup does not negate. A slot's group is checked when it is alive at the root or acting at an expanded node (`SearcherRequired`, `SearcherMismatch`, `SearcherNotLearner`, `NotSearchable`). A pool without heuristic slots searches exactly as before. LimitOrderBook-v1 is searchable from a determinized root with `with_info_heuristic_slots`; search values stay terminal-only (ADR-0043), so its per-step profit and loss does not enter the tree until #524.
- Replay of a pool with a policy map (ADR-0079 item 6 with erratum B2, #347 stage B2, #554): the sync and async recorders write the pool's map into the replay header (a slot count of `MAX_AGENTS` and one group code per slot; 0 for the default map) instead of refusing it. Replay verifies that the caller-built pool has the recorded map before any event and refuses another one (the default map included) with `ReplayError::Engine(EngineError::PolicyMapMismatch)`, the pool unchanged; keyframe contexts carry the recorded map's digest. `ReplayHeader::policy_map()` (`None` for the default map) and `policy_map_digest()` let callers build the matching pool. Heuristic actions are recomputed from state, so files hold no new event. Python: `replay.run(path, env)` replays a recording made with `make(..., policy_map=...)` into an environment built with the same map, and raises `ConfigError` for another map. Host replay files refuse a non-default map.
- Heuristic slots in Pipelined and Async mode (ADR-0079, #347 stage B1, #548): both split-phase modes build their step context with the pool's heuristic route, so workers compute and validate heuristic actions per chunk as in Sync, and every instance's trajectory equals Sync's for the same seed and map, whatever the worker count or the order Async ids are sent in. The fault cell is one per half (Pipelined) or one per instance (Async), reset at send and read when the unit is returned: an invalid heuristic action poisons with `EngineError::HeuristicAction` naming the pool instance at that `recv`. `PipelinedPool::heuristic_actions` and `AsyncPool::heuristic_actions` (and `heuristic_continuous`) hold each instance's rows from its last returned step, copied on the calling thread at `recv`; `into_pool` hands them back to the pool. `EngineError::HeuristicSlotsUnsupported` is removed. Python: `make(..., mode="pipelined" or "async", policy_map=...)` accepts heuristic groups, and `Env.heuristic_actions` works in both modes.
- The ONNX search evaluator takes `Set` observations (#505, #523): a set `p` is the model inputs `p` (or `p/<key>`), `[N, max, ...]`, and `p_mask`, `bool` `[N, max]`, with absent slots zero; the inputs come from `tessera_rl_search::observation_layout`, now public, so they are exactly the leaves of a search row. `Search(env_id="tessera/Gather-v1")` takes an `OnnxEvaluator`.
- Python `spin` pool option (#152, #332): `make(..., spin=...)` sets the Sync worker spin budget (`SyncOptions::spin`, default 2000) in every mode; Sync mode without workers refuses it with `ArgumentError`. The PufferLib adapter passes it through.
- Rejection notice for host environments (ADR-0069, #285, #391): `HostEnvironment::NOTIFIES_REJECTION` (default `false`, needs `VALIDATES`) and `on_rejected`, called once per rejected call on the reported instance before `EngineError::ActionRejected` is returned; it may write only that instance's state, so snapshots carry the notice.
- An engine registry shared across wheels (ADR-0005, #349, #402): engines from different extension wheels consult one Python-level registry at `sys.modules["tessera_rl._registry"]` (`attach_shared_registry`), so the unpinned-workers fallback sees every active engine.
- Type stubs for the Python package (#348, #407): `.pyi` files in the wheel, checked by `stubtest` and `mypy` (`make py-typecheck`). The `ReplaySummary.unclaimed` stub is included (#411 follow-up, #441).
- `tessera_rl.replay` (ADR-0072, #343, #408): `record(env, path, keyframe_every=0, checkpoint_every=1024)` and `run(path, env)`, returning a `ReplaySummary` and raising `ReplayDivergence`, for first-party environments.
- Async replay (ADR-0072 section 5, #343, #414): Async files log `Recv` events and replay into an `AsyncPool`, receiving one id at a time until every recorded id has come back. A step panic that the replay observes before the file recorded it (worker timing) does not report `UnexpectedPoison`: Replay skips the steps, resets, receives, checkpoints and periodic keyframes up to the recorded `Poisoned`, counts them in the new `ReplaySummary::unchecked_before_poison`, and such a replay is not `verified()`; any other event in between still diverges (#415, ADR-0072 erratum 2).
- Recorder wrappers for `set_lineage` and `import_all` (ADR-0072, #389, #435): `StateWrite` ops 5 and 6.
- Opt-in dynamic chunk claiming (ADR-0076, #357, #439): `Schedule::Dynamic` in the pool options (`schedule="dynamic"` from Python; `static` stays the default).
- Burn integration (#367, #443): `integrations/burn/`, its own workspace and lock (`burn =0.22.1`), with the `tessera-rl-burn` adapter and a Rust PPO example on CartPole (`make burn-check`, `make burn-gate`). It is not in the root workspace.
- Reference training scripts per library (SPEC 14.2, #368, #447; #497, #503, #511, #517): Stable-Baselines3, Tianshou and RLlib (validated against Gymnasium 1.2.2, RLlib's pin) on CartPole, plus an RLlib multi-agent wiring smoke on Tag. Each has its own uv project under `compat/training/`, and a report-only nightly runs them (`reference-libraries.yml`, gating per #501). CleanRL's row is `tools/reference_training.py`.
- `tessera-rl-ops` stage 4 (#383, #555): learning protocols `jobshop` (6x6) and `vrp` (20 customers, one vehicle), tier T3a at 500k steps, with gates whose solved bars are the scripted baselines' mean returns on the evaluation seeds (MWKR -648.15, nearest neighbour -792.05). `tessera_rl.gym` admits `tessera/Vrp-v1` with `vehicles=1` (slot 0, as Snake and PaddleDuel). SPT (the JobShop `Heuristic`, fixed by the golden fixtures) scores below a random policy on makespan, so the JobShop bar is MWKR's. No calibration yet.
- `tessera-rl-ops` stage 3 (#383, #540): `tessera/Vrp-v1` takes 1 to 8 `vehicles` under `Sequential` turns: the vehicle with work left that has travelled the least moves next, ties broken by the lowest slot (one vehicle behaves as before; the fixtures are unchanged), and it declares the PettingZoo AEC adapter. Python: `tessera_rl.make` builds `tessera/JobShop-v1` (also through `tessera_rl.gym.make`) and `tessera/Vrp-v1` (also through `tessera_rl.pz.aec_env`), with their config options and stubs; the facade re-exports the crate as `tessera_rl::ops`.
- `tessera-rl-ops` stage 2 (#383, #522): `tessera/Vrp-v1`, capacitated vehicle routing with one vehicle (`vehicles` 1 for now; the state already holds 8 vehicles under `Sequential` turns). At reset the depot and `customers` customers (2 to 500) get coordinates in `0..grid` and demands in `1..=max_demand`; `capacity` must be at least `max_demand`. `Discrete(customers + 1)` goes to a customer or back to the depot, which refills; the mask allows unserved customers that fit the load and the depot unless the vehicle is there with customers left. Distances are the Euclidean distance rounded to the nearest integer by an exact integer square root on `u64`, computed from the coordinates (no stored matrix; the state is 3,080 bytes); the reward is minus each move's distance. An episode ends when every customer is served and the vehicle is home, within `2 * customers` steps under any legal policy. Nearest feasible neighbour is the `Heuristic`. Golden, continuation, persistence and allocation tests (masks 501 wide) and benches at 20 and 500 customers.
- `tessera-rl-ops`, logistics and scheduling environments (SPEC 16, #383, #513), stage 1: `tessera/JobShop-v1`, job shop scheduling with one dispatching agent. Instances are generated at reset from the instance seed (`jobs` 2 to 64, `machines` 2 to 32, durations `1..=max_duration`); the action dispatches a job's next operation (`Discrete(jobs)`, masked to unfinished jobs); the reward is minus each step's increase of the current makespan (the latest end time scheduled so far), so an episode's return is exactly minus the final makespan; integer state (4,556 bytes) and observations; SPT, MWKR and FIFO dispatching rules (SPT is the `Heuristic`); golden, continuation, persistence and allocation tests and a Criterion bench. Python registration follows with the vehicle routing stages.
- Observation transforms on `Set` leaves (ADR-0092, #413, #534). `Normalize` and `OneHot` apply to set item leaves: absent slots stay zero, and the statistics are pooled over the present slots. A `FrameStack` that names a set stacks all of its leaves. It is exported as a set of `k * MAX` slots, frames oldest first, with a generated `<set>/frame_age` `u8` leaf (0 the newest frame, 0 for absent slots) before `<set>_mask`; a single-leaf set's items move to `<set>/item`. New errors: `TransformError::PartialSetStack` and `TransformError::AfterSetStack`.
- Search over `Set` observations (ADR-0082 point 7 as amended, #490, #512): a set's item leaves and `<name>_mask` leaf are ordinary leaves of the evaluator's rows, absent slots zero; `SearchPool::observation_leaves` and `observation_sets` (`LeafLayout`, `SetLayout`) give each leaf's offset and each set's item and mask leaves; `SearchGroup::bound_bytes` counts set leaves. Python: `Search(env_id="tessera/Gather-v1")` and `Search.sets`. The ONNX evaluator refuses set observations with `OnnxError::SetObservation` until #505. LimitOrderBook-v1 now builds a determinized search pool; its first expansion is refused for simultaneous actors (#491).
- One epoch barrier for pool-level adaptive state (ADR-0081, #359, #515, #521): episode statistics per instance, reduced at the barrier by a pairwise tree over instance order (SPEC 6.3), independent of chunk size, worker count and schedule (`EpisodeStats`: count, return and length sums, return mean and M2 from per-instance Welford partials merged with Chan's formula, host aborts; amends ADR-0081 item 3). `freeze_adaptive` and `freeze_episode_stats`; Pipelined needs both halves joined; `AsyncPool::epoch_barrier` is refused with `EngineError::AdaptiveInAsync` unless every subsystem is frozen; host pools count aborted episodes. Partials restart empty after a context restore.
- Search over `MultiDiscrete` actions (ADR-0086, #454): `SearchPool` takes a `MultiDiscrete` action space as one joint head of `prod(nvec)` actions, mixed radix with the first head most significant, up to `MAX_JOINT_ACTIONS` (4096; above it `SearchError::TooManyJointActions`; `Discrete` stays uncapped). `ActionMask::head_active` (default `true`) declares the conditional heads a parent value does not use; only canonical joints (every inactive head 0) are legal, so each game action has one edge. `tessera_rl_core::space::JointHeads` holds the joint index and the SPEC 11.3 mask offsets; `tessera_rl_search::check_head_activity` tests that an environment's step ignores its inactive heads; `tessera_rl.search.joint_log_prior` builds a joint prior from per-head logits. LimitOrderBook-v1 declares its heads' activity; search on it is still refused, for `Set` observations (#490) and simultaneous actors (#491).
- `tessera-rl-league`, head A of ADR-0088 (#384): the pure league library. It holds a population of opponent policies with stable ids and per-instance result rings, written without allocation, with dropped results counted. It provides Elo (pinned `LN10`) and two-player TrueSkill with a draw margin, plus uniform, PFSP (hard, variance, linear) and Elo matchmaking. Matchmaking builds a cumulative `u64` draw table with a fixed rounding rule. A `League` barrier applies results in instance order, then oldest-first, inside `with_canonical`. The engine wiring follows #347 and #359.
- `tessera_rl_num::math::erf` and `erfc` (ADR-0088 head A, #384, #506): libm 0.2.16's `erf.rs`, vendored as the other frozen functions, with golden vectors on every branch interval. An added function changes no existing result, so the math version stays 1.
- Observation transforms from Python, stage C (ADR-0078, #346): `tessera_rl.make(..., transforms=[...])` with `tessera_rl.transforms.FrameStack`, `OneHot` and `Normalize`; `layout()` and the Gymnasium and PettingZoo spaces describe the transformed observation; `Env.epoch_barrier()` and `Env.freeze_normalization(frozen)` (sync mode; `CapabilityError` otherwise); host environments refuse `transforms` with `CapabilityError`. A refused transform list is a `ConfigError`. `OneHot` now exports a `bool` box (`Space::Box` with `DType::Bool`) instead of `MultiBinary`, the form every adapter exposes. `Normalize` refuses a NaN or infinite `clip` with `ValueError` before the pool is built (#346 review).
- Policy maps and heuristic slots, stage A (ADR-0079, #347, #532; #437 option (a), closes #437): `PolicyGroup` (`Learner`, `External`, `Heuristic`, `PrivilegedHeuristic`) and `InfoHeuristic` (`act_info` on the information state) in the core; `PolicyMap` in the engine, set before the first reset with `EnvPool::with_policy_map` (learner and external groups), `with_heuristic_slots` (`Heuristic` on a perfect-information environment), `with_info_heuristic_slots` (`InfoHeuristic`) or `with_privileged_heuristic_slots` (full-state opponents for testing). The validation boundary skips heuristic rows; workers compute them once per chunk through one routine chosen per pool, on a `StepRng` of the new `HEURISTIC` stream keyed by instance identity, step and slot, and validate them (an invalid one poisons with `EngineError::HeuristicAction`); `EnvPool::heuristic_actions` holds the last step's rows. The map digest joins `PoolContext` (`policy_map`; another digest is `EngineError::PolicyMapMismatch`, nothing written). Tag's baseline now reads only its window (a pursuer chases the nearest evader in view, an evader flees the nearest pursuer in view, the lowest legal action, stay, with none in view) and gains `InfoHeuristic`, as do Leduc (unchanged `act`), Lanterns and the limit order book, so `make("tessera/LimitOrderBook-v1", policy_map=...)` runs its scripted opponents (#470, #539). Python: `make(..., policy_map=[...] or {slot: group})`, `layout()["policy_groups"]`, `Env.policy_groups`, `Env.heuristic_actions`; PettingZoo `infos[agent]["policy_group"]` and `parallel_env(..., agents="learner")`; Gymnasium needs slot 0 a learner; PufferLib hides heuristic slots in Sync and Pipelined mode (#568); host environments refuse `policy_map` (`CapabilityError`).
- `tessera/RTS-v1` heuristic slots (#586, #603): `ImperfectInformation` (an agent's information state is its side's fogged view) and `InfoHeuristic` (the Rush bot on that view, equal to `Heuristic`), so `make("tessera/RTS-v1", policy_map=...)` runs its heuristic slots. The search pool still refuses the environment (no `Determinize`).
- Search on pools with observation transforms, stage B (ADR-0078 item 7, #346): every tree node keeps its frame-stack bytes beside its record (`SearchPool::transform_bytes`), so expansion and root evaluation continue exactly the live pool's observations; `SearchPool::new_with_transforms` and `set_root_with_transforms` take the root's bytes, and a plain root on a stacked scratch pool is refused (`SearchError::StackedRoot`); the scratch pool is frozen and takes the live statistics with `set_transform_context` (another identity is `SearchError::TransformIdentity`); `bytes_per_node_with` and `SearchGroup::bound_bytes_with` count the stacks; `Transforms::exported` gives the exported space and `S` without a pool. Determinized pools refuse frame stacks.
- Observation transforms in the engine, stage A (ADR-0078 and erratum 1, #346): `Transforms` (`FrameStack`, `OneHot`, `Normalize`) set with `EnvPool::with_transforms` before the first reset run on workers after `observe`, on every path that observes, allocation-free. `EnvPool::observation_space` is the exported (transformed) space. Frame stacks live in a per-instance arena beside the records (`transform_state_bytes`, `snapshot_transforms`, `restore_with_transforms` and the `_all` forms are exact continuations; a plain `restore` refills the stacks and logs once; `fork` carries them). Running normalization: per-chunk Welford accumulators merged in chunk order at `epoch_barrier`, `freeze_normalization`, and `TransformContext` in `PoolContext` (another transform identity is refused). Async pools refuse an unfrozen `Normalize`; `encode` (`PersistError::Transforms`) and replay recording refuse pools with transforms; transforms on `Set` leaves are refused (ADR-0082). Search and Python follow in stages B and C.
- `cargo tessera` (#351, SPEC 10.2): a new tool crate, `crates/cargo-tessera` (a workspace member, not a default member; no new dependency). `cargo tessera new <name> [--path] [--id ns/Name-vN] [--agents N] [--turns simultaneous|sequential|mixed] [--snapshots] [--tessera-path]` scaffolds a host environment crate with its own workspace and wheel, pinned to exactly the tool's version of the Tessera crates (or by path); `cargo tessera check [--python] [--bless] [--uv] [--locked] [--offline]` runs the crate's Rust checks and golden hashes, and with `--python` installs the wheel and runs its Gymnasium and PettingZoo adapter tests; `cargo tessera bench [--quick]` runs the criterion bench and prints a summary. The engine's `host_testing` (feature `testing`) gains `check_determinism`, `check_continuation`, `check_snapshot_continuation`, `check_spaces` and `check_invariants`, with `CheckRun`, their reports and `CheckError`. `make cargo-tessera-e2e` runs the tool end to end, offline and locked, in the CI python job.
- `cargo tessera new --kind env` (#628, #643): scaffolds a Rust-only `Environment` crate (`Cargo.toml`, `clippy.toml`, `src/lib.rs`, `tests/check.rs`, `benches/step.rs`, no Python package); `--kind host`, a host environment crate and its wheel, stays the default. `--snapshots` is host only, and `cargo tessera check --python` is refused on an env crate. A failed `cargo tessera check` of an env crate names the comparison kind, the first diverging call (numbered as in `tests/golden.json`, also for the resume and snapshot comparisons) and instance, and a hint at the usual cause (#645, #649, #652).
- The authoring study kit of SPEC 10.3 (#639, #662): `study/` holds the protocol, a facilitator guide, the three timed tasks (T1, a first working environment in one of three variants; T2, a seeded derive error; T3, a seeded determinism divergence) and a grading kit (reference solutions, goldens, `grade.sh` and `acceptance.sh`). The site gains the tutorial "Writing an `Environment`-trait environment", for `cargo tessera new --kind env`. Recruiting participants and publishing results are not part of the kit.
- Live watch (ADR-0083 item 7, #378): `EnvPool::attach_watch`, `PipelinedPool::attach_watch` and `AsyncPool::attach_watch` return a single-consumer `WatchHandle` (`Send`, not `Sync` or `Clone`; `request(instance)` and a non-blocking `take(buf) -> Option<generation>`; `WatchError`), and `detach_watch` drops it. The pool serves a pending request at its next batch boundary where the instance is idle (after `reset`, `step`, `restore` and `fork` in Sync and Inline mode; after `recv`, for the returned half or batch only, in the split-phase modes), on the caller's thread, by encoding one record into a buffer allocated at attach. With no watch the step path pays one pointer test, with an idle one a `Relaxed` load; a poisoned pool serves nothing. `EngineError::WatchAttached`. Python: `Env.watch()` returns `tessera_rl._native.WatchHandle` (`request`, `take() -> (generation, bytes) | None`, `close`), and `tessera_rl.viewer.watch(env)` serves a live Viewer page that polls `/live` every 250 ms and draws each served record through a private one-instance shadow pool; `/live` waits at most 2 s, then answers `{"pending": true}`. The Viewer now logs an unexpected server error's type and message (logger `tessera_rl.viewer`, never the request line or token), and a backward seek finds its keyframe from a map built once.
- Lane-wise states, head 1 of ADR-0090 (#345, #543). `#[state(lanes)]` on a `#[derive(EnvState)]` struct stores it as Struct-of-Arrays, one column per field, each padded to a multiple of 8. Fields may be `f64`, `f32`, `i32`, `u32`, `Fx32`, `Bool8` or `StateEnum` enums, at most 12; arrays, `Pool`, nested states and `Reserved` are refused at compile time. Records stay the only interchange: `snapshot`, `state_hash` and the live watch gather, while `restore`, `import`, `fork` and archive resets scatter. So persisted files, replay files and hashes are byte-identical to an Array-of-Structs store with the same fields. Such a pool needs `chunk_size % 8 == 0` and refuses Async (`LaneEnvUnsupported`). This head steps every instance through a one-instance gather, the scalar step and a scatter; head 2 (#702, below) adds the lane kernels. The column store is the sealed `LaneTuple` in the new audited `core::lane_field`.
- Lane kernels on the column store, head 2 of ADR-0090 (#345, #702). An environment whose state is `#[state(lanes)]` implements `LaneEnvironment` (`step_lanes` and `observe_lanes` over a `LaneGroup` of `LANES` instances' columns) and sets `Environment::LANE_ENV` to `Some(lane_env_hooks::<Self>())`; the default `None` keeps the head 1 path. The engine then steps and observes eligible groups of 8 consecutive instances of a chunk directly on their columns, with no gather or scatter; tails, pending resets and dead agents go through the one-instance path. Each lane must match `step` and `observe` bit for bit (ADR-0090 4.8). The engine uses it only for single-agent, simultaneous, unmasked environments with one discrete head and no observation transforms. `lane_env_hooks` refuses at compile time a state that is not lane-wise and kernels that draw RNG values. A lane-wise state never takes the row lane step of ADR-0071, whose row store it does not have, and one that sets both `LANE_ENV` and `LANE_STEP` is refused at compile time (ADR-0090 erratum, #705). `core::lane_field::LANES` is now the same constant as `lane_step::LANES`.
- The Tessera Viewer (ADR-0085, #378), experimental (#696): the module, its endpoints and the logits side file may change in any release. `tessera_rl.viewer.serve(path, env)` serves a replay file on `127.0.0.1` as a local web page (Canvas 2D scene, timeline, per-agent rewards, flags, masks and observations), and `show(path, env)` returns a Jupyter `IFrame` of it. The backend is Python's standard library, in the wheel, with no new dependency; every request needs the session token in the URL, and the `Host` header must name the loopback port. Scrubbing back seeks to the nearest earlier keyframe and replays forward. `tessera_rl.viewer` is not imported by `import tessera_rl`.
- Policy logits in the Viewer (ADR-0083 item 6, #683, #692). For each agent and action head the Viewer shows the raw logits, the softmax over the legal choices and the legality. `viewer.serve`, `show` and `Viewer` take `logits=` (`"auto"`, the default, reads the side file if present; a side-file path; a callable `fn(call_index, instance)`; or `None`), and `viewer.watch(env, logits=fn)` takes a live callable. `tessera_rl.viewer.logits_log(replay_path, env)` (`LogitsLog.step(logits)`) collects one row per step and writes the side file `<replay>.logits.npz` when its block exits normally (nothing if it raises); open it outside `replay.record`, so the replay is complete when the side file is written. It is bound to its replay by the replay's sha256, never changes the replay file, is read as untrusted input with `allow_pickle=False`, and has format version 1 (any other is refused). Before NumPy reads anything, an entry whose declared uncompressed size is over a cap (1 GiB for `logits.npy`, 1 MiB for any other entry) is refused (#696). A side file that cannot be served is ignored, and `/meta` says why. `ReplaySteps.step_events()` gives the event indices of the file's `step` calls.
- `tessera/LimitOrderBook-v1` from Rust and Python (#381, stage 4, #487).
  - Rust: the facade re-exports `tessera_rl::markets` and registers it.
  - Python: `tessera_rl.make` and `tessera_rl.pz.parallel_env` reach it through `_native.make_lob`, which takes every `LobConfig` field.
  - It declares PettingZoo Parallel only, as the RTS: cycle buffering for a multi-head environment is not designed yet.
  - `pz.aec_env` for an environment that does not declare AEC (RTS, LimitOrderBook) now raises `CapabilityError` instead of `ArgumentError`.
  - Added PettingZoo API tests, a reference PPO protocol (`tools/lob_training.py`, `compat/learning/protocols.toml` entry `lob`: slot 0 against noise traders, no nightly gate), a benchmark entry (`rust/lob/inline/n1024`, 8 agents) and the PGO determinism tool entry.
- Archive resets (ADR-0075, #363): `Archive<E>` (immutable, validated, shared; an XXH3-64 digest), `ResetSource` and `EnvPool::set_reset_source`, `EnvPool::reset_from`, and the reset source in `PoolContext` (`ResetSourceContext`). An archive reset keeps the instance's RNG identity and generations and copies an entry's state and alive slots; entries are drawn by integer weights on the reserved `archive_select` stream. `EngineError::Archive` and `ArchiveMismatch`. In every mode: `PipelinedPool::set_reset_source` and `AsyncPool::set_reset_source` (at quiescence, else `NotQuiescent`, `HalfBorrowed` or `Borrowed`) and `AsyncPool::reset_from` (a partial writer, like `reset`); Pipelined mode has no partial reset and no `reset_from`.
- Replay of archive resets (ADR-0075 item 7, replay format version 1, #363): the header names the reset source at recording start and every keyframe the current one; `Recorder::set_reset_source` and `Recorder::reset_from` log `ResetSource` (tag 10) and `ResetFrom` (tag 11); `Replay::run_with_archive(pool, archive)` replays them, refusing a missing or different archive with `EngineError::ArchiveMismatch`, and a keyframe whose reset source differs is the new divergence `DivergenceKind::ResetSource` (`"reset_source"` in Python). `replay::ResetSourceRecord` is now the engine's `ResetSourceContext`. Async and host replay files do not hold archive sources: the async recorder refuses a pool with one.
- Archive resets from Python (ADR-0075 item 9, as amended by #363 head C): `tessera_rl.Archive(env, snapshots)` builds an archive of `env`'s environment and exact config from `env.snapshot(i)` bytes (`digest`, `len`); `env.set_reset_source(archive, weights=None)` in every mode; `env.reset_from(ids, entries)` in Sync, Inline and Async mode (Pipelined: `CapabilityError`); host environments refuse both. `tessera_rl.replay.run` and `steps` take `archive=`, and `replay.record` logs both calls. The Gymnasium, PettingZoo and PufferLib adapters forward them. An archive of another environment raises `ArgumentError`, of another config `ConfigError`.
- Whole-pool snapshots with their context (ADR-0075 item 7, ADR-0078 item 6, #465): `EnvPool::snapshot_pool` and `restore_pool(state, archive)` (and the Pipelined and Async forms) save and restore every record, every frame stack and the full `PoolContext` (branch counter, reset source, transform identity and statistics), checked before any write. In Python, `env.snapshot_all()` works in every mode and carries the context, and `env.restore_pool(snapshot, archive=None)` restores all of it; `restore_all` still restores records only.
- `tessera/Lanterns-v1` (#380, SPEC 16): a cooperative card game with hidden hands for 2 to 5 players (an original implementation of the public hidden-hand fireworks mechanics, ADR-0034), cross-platform tier. Hints narrow knowledge both ways; `Determinize` deals the observer's hand and the deck exactly uniformly among consistent deals; an information-state scripted baseline. Python: `make_lanterns`, the PettingZoo AEC adapter. From Python, `zero_on_fuse=` takes a bool (#468).
- Derive syntax for a `Set` whose items are a `Dict` (ADR-0082 point 2, #429, #472): `#[space(set(max = .., items = I))]` on an `ObsSetOf<I>` field, where `I` is itself `#[derive(Observation)]` with the same config type, exports one item leaf `p/<field>` (`[MAX, ...]`) per field of `I` in `I`'s declaration order, then `p_mask`, and generates a `SetOf<I>` constant. `ObsWriter::set_of_mut` returns a `SetOfLeaf` (`capacity()`, `set(i, present)`, `item(i, id)`). The derive also generates, for every `ObsLeaf<T>` field of every struct, an `ItemId<Self, T>` constant `<FIELD>_ITEM`; `item` accepts only ids of `I`, so another struct's id does not compile. `Observation::CONTAINS_SET` (default `false`, set by the derive for a struct with a set field; a hand-written impl whose space contains a set must set it) and the new derive-implemented `ObservationLayout` (`LEAVES`, `CONTAINS_SET`, config-free); the derive refuses at compile time, at the field, items that contain a set, and a field named like the set's `p_mask`. The single-leaf `inner(..)` form is unchanged. A derived struct with fields `x` and `x_item` no longer compiles: `X_ITEM` would be defined twice.
- `Set` observation leaves end to end (ADR-0082, #366): a `Set { inner, max }` named `p` exports its item leaves at `p` (single-leaf items) or `p/<inner path>`, shape `[MAX, ...]`, then `p_mask` (`bool`, `[MAX]`); each counts toward the 16-leaf limit, and a set may not contain a set. The engine zeroes, after `observe`, every item row whose mask is false (with `debug-checks`, a write into an absent slot fails a debug assertion). Duplicate exported leaf names are refused with the new `EngineError::DuplicateLeafName`, and by `#[derive(Observation)]` at compile time. The derive gains `#[space(set(max = .., inner(..)))]` on `ObsSet<T>` fields, which generate a `SetId<T>`; `ObsWriter::set_mut` returns a `SetLeaf` (`set(i, present)`, `item(i)`). The Python layout gains `"sets"`, `(path, max, [item leaf indices..., mask index])`; Gymnasium and PettingZoo expose each mask as `MultiBinary(MAX)` (`int8` 0/1) and the items as a `Box`. `SearchPool` refuses set observations with `SearchError::SetObservation`. New environment `tessera/Gather-v1` (`tessera-rl-classic`, `make_gather` in Python). Behaviour change: a host environment that declares a `Set` observation, which loaded and silently lost those leaves, now fails at construction with `EngineError::UnsupportedSpace` (host protocol 1).
- `tessera/LimitOrderBook-v1`: `own_orders` and `trades` are `Set` observation leaves (ADR-0082, #450). Each exports an `i32` item leaf `[MAX, 3]` and a `bool` mask leaf, replacing the `Box` leaves and their `i32` masks; the environment version is unchanged. `sample_with_report` reports the determinization's retries and any use of its last-resort assignment, which is also logged with `tracing`.
- `tessera/LimitOrderBook-v1` (#381, stage 3): `ImperfectInformation` and `Determinize`. The information state holds only the observer's own data, the public fields and the visible levels. A sample constructs the other agents' resting orders from the visible levels, within `MAX_OPEN` and the position limits; their queued items are empty. It also resamples inventories and cash under conservation, and the fundamental within the signal's band. The market maker, noise and momentum baselines (`Heuristic`) read the information state only and are legal under the masks. The privileged state gets its space and writer for ADR-0079. The private signal is drawn from `StepRng` in the step and kept in the state, which also gains a mid history and each slot's last limit-order step. The market closes at settlement. A market order's protection price now reads only the visible levels. Golden values move.
- `tessera/LimitOrderBook-v1` (#381, stage 2): the `Environment` of `tessera-rl-markets`, registered (not yet in the facade or Python). Simultaneous turns, latency through the queue, a seeded arrival permutation per step, `MultiDiscrete([6, 9, 4, 8])` with conditional masks that count open and pending orders against the position and cash limits, market orders with a protection price from the visible book, `i32` observations with a private noisy signal, equity-change rewards, settlement and truncation on the last step, and optional bankruptcy. `Book::ioc` is new.
- `tessera-rl-markets`, stage 1 of `tessera/LimitOrderBook-v1` (#381): `LobConfig` with checked overflow bounds, the 16-byte `Order`, a `Pool`-backed limit order book with price-time priority, immediate-or-cancel market orders, cancels, self-trade prevention, best levels, a market-order cost sweep and per-agent exposure, the latency queue and the fundamental's bounded random walk. No `Environment` yet.
- Video export, stage 4b (ADR-0083 item 5, #378): `tessera_rl.render.export(frames, path, fps=50)` writes `uint8` frames `(n, height, width, 3)` as a looping GIF (encoded natively with the `gif` crate, pinned at 0.14.2; `tessera_rl_render::encode_gif` and native `encode_gif`) or as an MP4 through a user-provided `ffmpeg` started without a shell; no video codec is linked. Video bytes are not golden.
- Release signing and SBOMs (SPEC 20.1, #350). `pypi.yml` gains a `release-assets` job after the PyPI upload. It checks PyPI's PEP 740 attestations for every wheel, extracts each wheel's CycloneDX SBOM, and writes a crates.io SBOM from Cargo.lock (`scripts/crates_sbom.py`). It attests wheels and SBOMs with GitHub's keyless build attestations and uploads the SBOMs and `SHA256SUMS-assets` to the release. `pypi_verify.py` now requires every wheel check, `sbom` included. Signing is keyless; RELEASE-CHECKLIST.md explains verification. Each wheel's SBOM is reproducible (#654, #658): its `serialNumber` is a UUIDv5 of the commit, target, profile and features, and its `metadata.timestamp` is `SOURCE_DATE_EPOCH`, so two builds of one commit give the same SBOM.
- Opt-in adaptive worker wait (ADR-0074, #434): `SyncOptions::wait_policy` with `WaitPolicy::Adaptive { cap }` (cap at most `MAX_ADAPTIVE_CAP`, 1 s). After the fixed spins, a worker keeps spinning while its wait is shorter than `min(cap, 2 x` the EWMA of its past waits`)`, then blocks as before; it keeps caches warm through the caller's gap at the cost of about one core per worker. It applies to Sync, Pipelined and Async workers and to host pools in Rust. Python: `make(..., wait="adaptive", wait_cap_us=10_000)` (cap in microseconds, 1 to 1,000,000), also through `TesseraPufferEnv` and `make_puffer`; host environments refuse it, as they refuse `spin`. The default (`WaitPolicy::Fixed`) is unchanged.
- Rendering in Python, stage 4a-ii (ADR-0083 item 4, #378): `Env.render(ids=None)` returns owned `uint8` frames `(n, height, width, 3)` drawn from snapshots, `Env.scene(index)` the vector scene (primitives in pixels), `Env.canvas` the frame size; sync mode, for environments that implement `Render` (`CapabilityError` otherwise, and for host environments). Gymnasium (`TesseraVectorEnv`, `TesseraEnv`, `gymnasium.make`) and PettingZoo (`parallel_env`, `aec_env`) accept `render_mode="rgb_array"` and list it in `metadata["render_modes"]` only for such environments, with `render_fps` 50; `"human"` is refused. A truncated scene warns with `TesseraRenderWarning` (a `RuntimeWarning`). Native `Env.canvas`, `render` and `scene` (scenes cross as 144-byte records).
- Rendering from snapshots, stage 4a-i (ADR-0083, #378): `Render` now has `canvas(cfg)`, `draw(cfg, &State, &mut SceneWriter)` and `MAX_PRIMITIVES` (default 1024); `tessera_rl_core::render` holds the scene format (`Prim`: rect, circle, line with round caps, polygon of up to 16 points; `Px16` coordinates in 1/16 pixel; `Rgba`; `SceneWriter`, which drops what does not fit and reports it as overflow). The new `tessera-rl-render` crate (facade feature `render`, off by default) has `Renderer` and an integer rasterizer: one sample per pixel centre, no anti-aliasing, integer blending, so frames are bit-exact across platforms; golden frame digests cover CartPole, Grid, GridKeyDoor, Snake, BrickWall, PaddleDuel, Connect4, Othello and Go9, which implement `Render`. Python and the adapters follow in stage 4a-ii.
- Declared adapter capabilities and environment deprecation (ADR-0080, #373): `Adapters` and `Environment::ADAPTERS` (the PettingZoo interfaces an environment declares, empty by default; `tessera_rl.pz` now reads the first-party declarations from the registry, unchanged), `HostEnvironment::ADAPTERS` (derived from `TURNS`), `register!(Env, id = .., deprecated(since = .., note = ..))`, `EnvRegistration::adapters`/`deprecation`, `Deprecation`, `deprecation_of::<E>()`, a `tracing` warning when a pool of a deprecated environment is built, Python `tessera_rl.envs()` (records with `adapters` and `deprecated`; `registered_envs()` unchanged) and `TesseraDeprecationWarning` (a `FutureWarning`) from every entry that builds a pool by a deprecated id. No shipped id is deprecated.
- Conditional heads and entity groups for host environments (#385): `HostEnvironment::head_parents` and `entity_group_heads` (defaulted), so a host pool builds the same mask layout as an `EnvPool` and reports `entity_group_heads` in its layout (host protocol 1, an addition; masks stay outputs, child heads are validated by range). `EntityHeads` and `HeadSpec` in `tessera-rl-core` describe entity-centric heads once (action space, parents, groups, mask rows); the RTS declares its heads with them, with unchanged outputs.
- Replay rebuilds its pool from the file header (ADR-0072, #377): `Replay::build_pool` decodes the canonical config bytes and derives the pool options from the header's identity fields, then hands them to the caller's constructor (which picks masks and workers) and checks the pool; `Replay::run_rebuilt` builds and replays; `Replay::pool_options` returns the options. The canonical config form now decodes: `#[derive(EnvConfig)]` implements `CanonicalDecode`, and `decode_canonical` checks the version, every key, type tag and nested entry count, `validate`, and that the config encodes back to the same bytes (adaptive fields keep their defaults). New `ReplayError::Config`.
- Facade features (SPEC 20.1, #350): `search` and `simd` (default) and `onnx` (opt-in, implies `search`, re-exports `tessera_rl::onnx`). `tessera_rl::simd` re-exports `tessera-rl-simd`. With default features the API is unchanged.
- Per-agent refusal for host environments (ADR-0070, #337): `HostEnvironment::REFUSES_AGENTS` and `refuse(&self, &Actions, &mut Refusals)` in `tessera-rl-core`, with `RefuseCode`, `Refusals` and `Actions::is_refused`/`refuse_code`. `refuse` runs per instance after `validate` accepted the call and before `step`; refused agents are reported in the new `refused` and `refuse_code` outputs (`[N, A]`, refusing host pools only) while every other agent and instance steps. Refusing a slot that is not acting poisons the pool (`EngineError::RefusedNotActing`). Host protocol 2 (protocol 1 plus those outputs and the `refuses_agents` layout key) is spoken by a wheel whose environment refuses agents; `tessera_rl` accepts protocols 1 and 2. `HostStep.refused`/`refuse_code`, Gymnasium `info["refused"]`/`info["refuse_code"]` and PettingZoo `infos[agent]["refused"]`/`["refuse_code"]` report refusals, and both adapters' metadata carry `refuses_agents`. Refusal rows in replay files (ADR-0070 item 9) follow in #339.
- `tessera_rl_engine::physical_cores()` and Python `tessera_rl.physical_cores()`: physical cores this process may use (sysfs topology within the affinity mask on Linux, `hw.physicalcpu` on macOS, processor core records on Windows), capped by `available_parallelism()` so CPU quotas apply. Efficiency cores of hybrid CPUs count as cores (#356, SPEC 6.1).
- `Default` for `SyncOptions` and `PipelinedOptions`: one worker per physical core. A Rust pool still steps inline until it is given workers. `default_workers()` adds the per-chunk cap that Python's `make()` uses.
- Output buffer alignment (SPEC 5.2, #355): the output allocation and every observation and terminal observation leaf start on a 16 KiB boundary (a multiple of the page size: 4 KiB on x86-64, 16 KiB on Apple silicon; `tessera_rl_engine::OUTPUT_ALIGN`); every other output buffer starts on 64 bytes (`OUTPUT_MIN_ALIGN`). Before, buffers were 8-byte aligned. Step times are unchanged within noise on CartPole and Tag at 1024 and 8192 instances.
- Host snapshots (ADR-0073, #344): `HostSnapshot` and `SnapshotReject` in `tessera-rl-core`, and on host pools of such environments `snapshot`, `snapshot_to`, `snapshot_all`, `restore`, `import`, `restore_all`, `fork`, `state_hash`, `encode` and `decode` over `HostRecord`s (identity digest, host header, `SNAPSHOT_VERSION` and the environment's bytes). Every record is checked before any instance changes: identity, version, length, the host header rules of ADR-0068 (r2h) and `check_snapshot`. A poisoned host pool refuses all of them, `restore_all` included; a restored pending instance is not observed; there is no `fork_branched`. Host files use their own magic, `TESHOST\x1a`, under the shared ADR-0011 header. New `EngineError::SnapshotRejected` and `SnapshotTooLong`, `DecodeLimits::max_host_record_bytes`, and `StateHasher::add_bytes`. Through the type-erased pool and Python: `HostPool::with_dyn_snapshots`, `register_host_factory!(Env, snapshots)` and `export_host_env!(Env, module = .., snapshots)` make `DynHostPool`'s snapshot methods (`snapshot_records`, `restore_records`, `restore_all_records`, `fork`, `state_hash`, `encode`, `import_file`; `EngineError::NoSnapshots` otherwise) and the host object's (`snapshot`, `restore`, `restore_all`, `fork`, `state_hash`, `encode`, `import_records`) work, and add the layout key `snapshots` (host protocol 1, an addition). `HostEnv.snapshots`, `snapshot`, `restore`, `restore_all`, `import_record`, `fork`, `state_hash`, `encode` and `load` follow; host modules bind `RecordError` and `PersistError`. The engine feature `testing` adds `host_testing::check_snapshot_contract` for environment crates' tests. The `host_orderbook` example implements `HostSnapshot`, and the fuzz target `host_decode` covers host files and records.

### Changed

- League and opponent-policy errors now raise `ConfigError` and `OnnxError` instead of `TesseraError` (#619). Both subclass `TesseraError`, so `except TesseraError` still catches them. This includes host wheels, which share the error mapping.
- `Env.epoch_barrier()` returns a dict instead of `None` (#384 head E3, ADR-0081 item 1.3): fixed keys `episodes`, `normalization`, `curriculum` and `league`, each `None` for a subsystem the pool does not have (typed `tessera_rl.EpochReport`). Callers that ignored the result are unaffected. The Rust form is the `EpochReport` change under Breaking.
- Behaviour change: exact integer Box bounds in Python (#570, #584): `layout()` leaf bounds are Python ints for integer leaves (exact, so `i64::MAX` is `2**63 - 1`, not `2**63`) and floats for float leaves (bits unchanged). The Gymnasium, PettingZoo and PufferLib adapters build integer observation bounds from those ints without a `float64` round trip, and host leaves span their integer dtype exactly. Shapes, dtypes and space types are unchanged.
- `Schedule::Dynamic` claims chunks owner first (ADR-0087, #436, #578): worker `w` drains the chunks `Static` gives it from its own claim cursor (one per worker, each on its own cache line, allocated once per Sync runtime and per Pipelined half), then steals from the other workers' shares in a fixed rotation. The API, the guarantees and the outputs are unchanged (bit-identical to `Static`); only the claim order changes, to remove contention on the one shared counter and keep `Static`'s instance-to-worker affinity on uniform cost.
- Performance, outputs unchanged:
  - the engine skips the per-agent died and spawned row reads when no agent event fired, and CartPole writes its observation with one leaf copy (#152, #333);
  - the PufferLib adapter counts ended episodes on the `uint32` view of the episode lengths (#294, #341);
  - CartPole lane columns convert by name, so the AVX2 path stays inlined (#335, #353);
  - the state hash is buffered, about 4x faster (#387, #398), and encodes numeric arrays in batches, 1.6x faster again (#387, #492), with identical bytes;
  - Go 9x9 compares superko history entries word by word, without `memcmp` under LTO (#611, #630).
- Behaviour change: the keyword-only native functions (`tessera_rl._native.make_*`, `make_search`, `make_search_group`, `onnx_load` and `Search.attach_onnx`) raise `ArgumentError` for positional arguments; before, they ignored them silently, so `make_cartpole(1)` built a default pool (#376). Every native function and method now carries a text signature (`inspect.signature` works, and stubtest checks the `.pyi` stubs' parameters).
- Content-addressed assets (ADR-0077, #364). Config fields of type `Asset<T>` (`tessera_rl_core::asset`) hold immutable shared data, hashed with XXH3-128 over their canonical bytes. The config's canonical form holds the hash; `#[config(asset = "name")]` names the asset. Environment snapshot files of configs with assets are format version 2, which lists `(name, hash)` entries after `asset_count`; a different asset is `PersistError::AssetMismatch`. Asset-free files stay version 1, byte for byte. Replay files carry the entries in version 1. Host snapshots and host replays refuse configs with assets. `ConfigItem::Bytes` passes an asset's bytes.
- Behaviour change: `tessera_rl.make()` without `workers` now starts one worker per physical core this process may use, at most one per chunk (`tessera_rl_engine::default_workers`, SPEC 6.1), in Sync, Pipelined and Async modes. Before, Sync stepped inline and the split-phase modes used one worker. Pass `workers=0` for the old Sync behaviour. On an x86-64 host with 8 physical cores, the default ran CartPole and BrickWall pool steps in 0.28 to 0.44 of the inline time at 1024 and 8192 instances (#356). The PufferLib backends keep their explicit worker counts.

### Deprecated

- `EnvAssets` (#418): assets are `Asset<T>` config fields with `AssetData` (ADR-0077), and nothing uses the trait. It stays exported, in the prelude too, until its removal in a later minor release.
- `tessera_rl.pz`'s `agents="all"` and `agents="learner"` as the learner filter (#538, #572): use `visible=`, a new keyword of `parallel_env`, `aec_env`, `TesseraParallelEnv` and `TesseraAECEnv` (`"all"` by default, or `"learner"` to hide heuristic slots, ADR-0079 item 5.2). `agents=` is now the environment's own agent count (Snake, GridKeyDoor, PaddleDuel, LimitOrderBook), so both can be set together; the old string spelling keeps working with a `TesseraDeprecationWarning` attributed to the caller, until its removal in 0.5.0. Passing `visible=` and a string `agents=` together is `ArgumentError`. `tools/lob_training.py` now passes `agents=8` and `visible="learner"`.

### Fixed

- Derive macros point their errors at the offending field (#365, #395), and `EnvState` reports a non-plain-data field type at that field only (#388, #461).
- Python: the Gymnasium adapter gives single-head masked environments (Snake with `agents=1`) their mask in `info["action_mask"]` (`[N, n]` `int8`) and `info["action_masks"]`, (#477) as SPEC 11.3 says and multi-head environments already did; before, these keys were missing.

Internal changes with no user-visible effect are not listed: learning protocols, calibrations and nightly gates (#432, #452, #458, #466, #480, #542, #544, #546, #550, #553, #562, #569, #571, #585, #591, #600, #607, #614, #627, #660), the nightly wall-clock gate (#433, #440), PGO evidence and bench harness (#338, #632, #638), test and tool portability and fixtures (#393, #467, #656), lints (#399, #703), the public-API snapshots of the facade and of the Python package, and their checks (ADR-0094 items 2 and 7, #686, #698), scripts (#516, #518, #637), the docs site (#526, #535), lockfile refreshes (#577, #653, #677, #691) and bench-runner entries (#589).

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
