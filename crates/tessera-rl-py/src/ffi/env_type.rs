// SPDX-License-Identifier: MIT OR Apache-2.0

//! The `tessera_rl._native.Env` type: its body, method table and the
//! module-level constructors, as safe impls of the typed layer of
//! `tessera_rl_pyffi` (ADR-0062 item 8).

use tessera_rl_pyffi::ffi::capi::{self, CResult, Gil, Mine, Ref};
use tessera_rl_pyffi::ffi::module_def::{FunctionKw, FunctionRow};
use tessera_rl_pyffi::ffi::object::{
    Body, Method, MethodNoArgs, MethodRow, MethodTable, Release, TypeObject, TypeSpec,
};

use super::inner::Inner;
use crate::api;

/// An `Env` object's body: the pool behind the reentrancy guard.
pub(crate) struct EnvBody(Inner);

impl EnvBody {
    /// The state, refusing an object inherited through a fork before any
    /// lock or flag of it is touched.
    fn guarded(&self, g: Gil<'_>) -> CResult<&Inner> {
        self.0.check_fork(g)?;
        Ok(&self.0)
    }
}

impl Body for EnvBody {
    /// Dropping a pool joins its workers, so it is dropped detached. In a
    /// forked child the copy of the workers does not exist and its locks may
    /// be held forever, so `Drop` could hang or abort: the memory is leaked
    /// by design, for the rest of the child's life.
    fn release(self, r: Release<'_>) {
        if self.0.inherited() {
            core::mem::forget(self);
        } else {
            r.detach(move || drop(self));
        }
    }
}

static TYPE: TypeObject<EnvBody> = TypeObject::new(&SPEC);

static SPEC: TypeSpec<EnvBody> = TypeSpec::new(
    c"tessera_rl._native.Env",
    c"A pool of Tessera environments (use tessera_rl.make).",
    &METHODS,
);

/// Creates the `Env` type and adds it to `module`.
pub(super) fn register(g: Gil<'_>, module: Ref<'_>) -> CResult<()> {
    TYPE.create(g, module, c"Env")
}

/// A method without arguments, calling `$api(g, &Inner)`.
macro_rules! noargs {
    ($name:ident, $api:path) => {
        struct $name;
        impl MethodNoArgs for $name {
            type Body = EnvBody;
            fn call<'py>(g: Gil<'py>, body: &EnvBody) -> CResult<Mine<'py>> {
                $api(g, body.guarded(g)?)
            }
        }
    };
}

/// A method with positional arguments, calling `$api(g, &Inner, args)`.
macro_rules! varargs {
    ($name:ident, $api:path) => {
        struct $name;
        impl Method for $name {
            type Body = EnvBody;
            fn call<'py>(g: Gil<'py>, body: &EnvBody, args: Ref<'py>) -> CResult<Mine<'py>> {
                $api(g, body.guarded(g)?, args)
            }
        }
    };
}

noargs!(LayoutM, api::layout);
noargs!(OutputsM, api::outputs);
noargs!(IsPoisonedM, api::is_poisoned);
varargs!(RecvM, api::recv);
varargs!(RestoreM, crate::async_api::restore);
noargs!(SnapshotAllM, api::snapshot_all);
varargs!(RestoreAllM, api::restore_all);
varargs!(RestorePoolM, api::restore_pool);
varargs!(SendM, api::send);
varargs!(ResetM, api::reset);
varargs!(StepM, api::step);
varargs!(SnapshotM, api::snapshot);
varargs!(IllegalActionsM, api::illegal_actions);
varargs!(StateHashM, api::state_hash);
varargs!(RenderM, api::render);
noargs!(EpochBarrierM, api::epoch_barrier);
varargs!(FreezeNormalizationM, api::freeze_normalization);
varargs!(SceneM, api::scene);
noargs!(CanvasM, api::canvas);
noargs!(HeuristicActionsM, api::heuristic_actions);
varargs!(ImportRecordM, api::import_record);
varargs!(ForkM, api::fork);
varargs!(MakeArchiveM, api::make_archive);
varargs!(SetResetSourceM, api::set_reset_source);
varargs!(ResetFromM, api::reset_from);
varargs!(EncodeM, api::encode);
varargs!(LoadM, api::load);
varargs!(RecordStartM, crate::replay_api::record_start);
noargs!(RecordFinishM, crate::replay_api::record_finish);
noargs!(RecordAbortM, crate::replay_api::record_abort);
varargs!(ReplayM, crate::replay_api::replay);
varargs!(ReplayBeginM, crate::replay_api::replay_begin);
varargs!(ReplayAdvanceM, crate::replay_api::replay_advance);
varargs!(ReplaySeekM, crate::replay_api::replay_seek);
noargs!(ReplayKeyframesM, crate::replay_api::replay_keyframes);
noargs!(ReplayFinishM, crate::replay_api::replay_finish);
noargs!(WatchM, crate::watch_api::watch);

/// `close()`: on an inherited object a no-op, as the deallocation: it
/// neither joins nor refuses, so `with` blocks and cleanup paths in a child
/// still finish.
struct CloseM;

impl MethodNoArgs for CloseM {
    type Body = EnvBody;
    fn call<'py>(g: Gil<'py>, body: &EnvBody) -> CResult<Mine<'py>> {
        if body.0.inherited() {
            return Ok(capi::none(g));
        }
        api::close(g, &body.0)
    }
}

static METHODS: MethodTable<EnvBody> = MethodTable::new(&[
    MethodRow::noargs::<LayoutM>(c"layout", c"layout($self, /)\n--\n\nShapes and dtypes of the outputs."),
    MethodRow::noargs::<OutputsM>(c"outputs", c"outputs($self, /)\n--\n\nCopies of the last call's outputs."),
    MethodRow::noargs::<IsPoisonedM>(c"is_poisoned", c"is_poisoned($self, /)\n--\n\nWhether the pool is poisoned."),
    MethodRow::noargs::<CloseM>(c"close", c"close($self, /)\n--\n\nFrees the pool."),
    MethodRow::varargs::<ResetM>(c"reset", c"reset($self, seed=None, ids=None, delivered=None, /)\n--\n\nreset(seed=None, ids=None)"),
    MethodRow::varargs::<StepM>(c"step", c"step($self, actions, /)\n--\n\nstep(actions: bytes of int32)"),
    MethodRow::varargs::<SnapshotM>(c"snapshot", c"snapshot($self, id, /)\n--\n\nsnapshot(id) -> bytes"),
    MethodRow::varargs::<IllegalActionsM>(
        c"illegal_actions",
        c"illegal_actions($self, id, /)\n--\n\nillegal_actions(id) -> per-agent illegal actions this episode",
    ),
    MethodRow::varargs::<StateHashM>(
        c"state_hash",
        c"state_hash($self, id, /)\n--\n\nstate_hash(id) -> the instance's state hash",
    ),
    MethodRow::noargs::<CanvasM>(
        c"canvas",
        c"canvas($self, /)\n--\n\nThe canvas (width, height) of a pool that renders, or None (ADR-0083).",
    ),
    MethodRow::noargs::<HeuristicActionsM>(
        c"heuristic_actions",
        c"heuristic_actions($self, /)\n--\n\nA copy of the last step's heuristic actions (int32 bytes), or None (ADR-0079).",
    ),
    MethodRow::varargs::<RenderM>(
        c"render",
        c"render($self, ids, /)\n--\n\nrender(ids) -> (RGB8 frames as bytes, per-id overflow flags), drawn from snapshots (ADR-0083).",
    ),
    MethodRow::noargs::<EpochBarrierM>(
        c"epoch_barrier",
        c"epoch_barrier($self, /)\n--\n\nMerges the normalization accumulators into the statistics, in chunk order (ADR-0078); sync mode.",
    ),
    MethodRow::varargs::<FreezeNormalizationM>(
        c"freeze_normalization",
        c"freeze_normalization($self, frozen, /)\n--\n\nStops (True) or restarts (False) normalization accumulation (ADR-0078); sync mode.",
    ),
    MethodRow::varargs::<SceneM>(
        c"scene",
        c"scene($self, id, /)\n--\n\nscene(id) -> (144-byte primitive records, overflowed) (ADR-0083).",
    ),
    MethodRow::varargs::<ImportRecordM>(c"import_record", c"import_record($self, id, record, /)\n--\n\nimport_record(id, record)"),
    MethodRow::varargs::<EncodeM>(c"encode", c"encode($self, ids, /)\n--\n\nencode(ids) -> bytes"),
    MethodRow::varargs::<LoadM>(
        c"load",
        c"load($self, data, ids, max_instances, max_decoded_bytes, /)\n--\n\nload(data, ids, max_instances, max_decoded_bytes)",
    ),
    MethodRow::varargs::<RecordStartM>(
        c"_record_start",
        c"_record_start(keyframe_every, checkpoint_every): see tessera_rl.replay",
    ),
    MethodRow::noargs::<RecordFinishM>(c"_record_finish", c"_record_finish() -> bytes"),
    MethodRow::noargs::<RecordAbortM>(c"_record_abort", c"_record_abort()"),
    MethodRow::varargs::<ReplayM>(c"_replay", c"_replay(data, archive) -> tuple: see tessera_rl.replay"),
    MethodRow::varargs::<ReplayBeginM>(
        c"_replay_begin",
        c"_replay_begin($self, data, archive, /)\n--\n\n_replay_begin(data, archive): starts a stepped replay; see tessera_rl.replay.steps",
    ),
    MethodRow::varargs::<ReplayAdvanceM>(
        c"_replay_advance",
        c"_replay_advance($self, calls, /)\n--\n\n_replay_advance(calls) -> tuple: see tessera_rl.replay.steps",
    ),
    MethodRow::varargs::<ReplaySeekM>(
        c"_replay_seek",
        c"_replay_seek($self, keyframe, /)\n--\n\n_replay_seek(keyframe): imports that keyframe and moves the replay after it",
    ),
    MethodRow::noargs::<ReplayKeyframesM>(
        c"_replay_keyframes",
        c"_replay_keyframes($self, /)\n--\n\n_replay_keyframes() -> tuple[int, ...]: the keyframes _replay_seek accepts",
    ),
    MethodRow::noargs::<ReplayFinishM>(
        c"_replay_finish",
        c"_replay_finish($self, /)\n--\n\n_replay_finish() -> tuple: ends the stepped replay; see tessera_rl.replay.steps",
    ),
    MethodRow::varargs::<SendM>(c"send", c"send($self, half, actions, delivered, /)\n--\n\nsend(half, actions: bytes of int32, delivered)"),
    MethodRow::varargs::<RecvM>(
        c"recv",
        c"recv($self, delivered, /)\n--\n\nrecv(delivered) -> the oldest sent half's outputs",
    ),
    MethodRow::varargs::<RestoreM>(c"restore", c"restore($self, ids, snapshots, delivered, /)\n--\n\nrestore(ids, snapshots, delivered)"),
    MethodRow::noargs::<SnapshotAllM>(
        c"snapshot_all",
        c"snapshot_all($self, /)\n--\n\nsnapshot_all() -> an opaque whole-pool snapshot",
    ),
    MethodRow::varargs::<RestoreAllM>(
        c"restore_all",
        c"restore_all($self, snapshot, /)\n--\n\nrestore_all(snapshot) -> both halves' outputs",
    ),
    MethodRow::varargs::<RestorePoolM>(
        c"restore_pool",
        c"restore_pool($self, snapshot, archive, /)\n--\n\nrestore_pool(snapshot, archive): restores records, frame stacks and the pool context",
    ),
    MethodRow::varargs::<ForkM>(c"fork", c"fork($self, src, dst, /)\n--\n\nfork(src, dst)"),
    MethodRow::noargs::<WatchM>(
        c"watch",
        c"watch($self, /)\n--\n\nwatch() -> WatchHandle: attaches a live watch (ADR-0083 item 7).",
    ),
    MethodRow::varargs::<MakeArchiveM>(c"_make_archive", c"_make_archive($self, snapshots, /)\n--\n\n_make_archive(snapshots)"),
    MethodRow::varargs::<SetResetSourceM>(c"set_reset_source", c"set_reset_source($self, archive, weights, /)\n--\n\nset_reset_source(archive, weights)"),
    MethodRow::varargs::<ResetFromM>(c"reset_from", c"reset_from($self, ids, entries, delivered=None, /)\n--\n\nreset_from(ids, entries, delivered=None)"),
    MethodRow::END,
]);

/// A module function that builds an `Env` from keyword arguments.
macro_rules! maker {
    ($name:ident, $api:path, $label:literal) => {
        pub(super) struct $name;
        impl FunctionKw for $name {
            fn call<'py>(
                g: Gil<'py>,
                args: Ref<'py>,
                kwargs: Option<Ref<'py>>,
            ) -> CResult<Mine<'py>> {
                api::no_positional(g, args, $label)?;
                let inner = $api(g, kwargs)?;
                TYPE.instantiate(g, EnvBody(inner))
            }
        }
    };
}

#[cfg(feature = "test-hooks")]
maker!(
    MakePanicCounter,
    api::make_panic_counter,
    "_make_panic_counter"
);
#[cfg(feature = "test-hooks")]
maker!(
    MakeContinuousProbe,
    api::make_continuous_probe,
    "_make_continuous_probe"
);
#[cfg(feature = "test-hooks")]
maker!(MakeMultiProbe, api::make_multi_probe, "_make_multi_probe");
maker!(MakeCartpole, api::make_cartpole, "make_cartpole");
maker!(MakePendulum, api::make_pendulum, "make_pendulum");
maker!(MakeAcrobot, api::make_acrobot, "make_acrobot");
maker!(MakeMountaincar, api::make_mountaincar, "make_mountaincar");
maker!(MakeTag, api::make_tag, "make_tag");
maker!(MakeRts, api::make_rts, "make_rts");
maker!(MakeLob, api::make_lob, "make_lob");
maker!(MakeSurvival, api::make_survival, "make_survival");
maker!(MakeGrid, api::make_grid, "make_grid");
maker!(MakeKeydoor, api::make_keydoor, "make_keydoor");
maker!(MakeGather, api::make_gather, "make_gather");
maker!(MakeSnake, api::make_snake, "make_snake");
maker!(MakeBrickwall, api::make_brickwall, "make_brickwall");
maker!(MakePaddleduel, api::make_paddleduel, "make_paddleduel");
maker!(MakeConnect4, api::make_connect4, "make_connect4");
maker!(MakeLeduc, api::make_leduc, "make_leduc");
maker!(MakeLanterns, api::make_lanterns, "make_lanterns");
maker!(MakeJobshop, api::make_jobshop, "make_jobshop");
maker!(MakeVrp, api::make_vrp, "make_vrp");
maker!(MakeOthello, api::make_othello, "make_othello");
maker!(MakeGo9, api::make_go9, "make_go9");

/// The module-level constructors, for the module's function table.
pub(super) const MAKERS: [FunctionRow; 21] = [
    FunctionRow::kw::<MakeCartpole>(c"make_cartpole", c"make_cartpole($module, *, mode=None, batch_size=None, gather=None, num_envs=None, seed=None, max_episode_steps=None, auto_reset=None, chunk_size=None, workers=None, spin=None, outputs=None)\n--\n\nmake_cartpole(**pool_options) -> Env"),
    FunctionRow::kw::<MakeAcrobot>(c"make_acrobot", c"make_acrobot($module, *, mode=None, batch_size=None, gather=None, num_envs=None, seed=None, max_episode_steps=None, auto_reset=None, chunk_size=None, workers=None, spin=None, outputs=None)\n--\n\nmake_acrobot(**pool_options) -> Env"),
    FunctionRow::kw::<MakeMountaincar>(
        c"make_mountaincar",
        c"make_mountaincar($module, *, mode=None, batch_size=None, gather=None, num_envs=None, seed=None, max_episode_steps=None, auto_reset=None, chunk_size=None, workers=None, spin=None, outputs=None)\n--\n\nmake_mountaincar(**pool_options) -> Env",
    ),
    FunctionRow::kw::<MakePendulum>(c"make_pendulum", c"make_pendulum($module, *, mode=None, batch_size=None, gather=None, num_envs=None, seed=None, max_episode_steps=None, auto_reset=None, chunk_size=None, workers=None, spin=None, outputs=None)\n--\n\nmake_pendulum(**pool_options) -> Env"),
    FunctionRow::kw::<MakeTag>(
        c"make_tag",
        c"make_tag($module, *, mode=None, batch_size=None, gather=None, num_envs=None, seed=None, max_episode_steps=None, auto_reset=None, chunk_size=None, workers=None, spin=None, outputs=None, width=None, height=None, pursuers=None, evaders=None, slip=None, radius=None)\n--\n\nmake_tag(**pool_options, **tag_config) -> Env",
    ),
    FunctionRow::kw::<MakeRts>(
        c"make_rts",
        c"make_rts($module, *, mode=None, batch_size=None, gather=None, num_envs=None, seed=None, max_episode_steps=None, auto_reset=None, chunk_size=None, workers=None, spin=None, outputs=None, size=None, decision_ticks=None, max_units=None, step_limit=None, seat_swap=None, opponent=None)\n--\n\nmake_rts(**pool_options, **rts_config) -> Env",
    ),
    FunctionRow::kw::<MakeLob>(
        c"make_lob",
        c"make_lob($module, *, mode=None, batch_size=None, gather=None, num_envs=None, seed=None, max_episode_steps=None, auto_reset=None, chunk_size=None, workers=None, spin=None, outputs=None, agents=None, steps=None, depth=None, min_tick=None, max_tick=None, initial_price=None, up_permille=None, down_permille=None, latency=None, size_0=None, size_1=None, size_2=None, size_3=None, max_position=None, initial_cash=None, allow_short_cash=None, bankrupt=None, noise=None, reward_shift=None)\n--\n\nmake_lob(**pool_options, **lob_config) -> Env",
    ),
    FunctionRow::kw::<MakeConnect4>(c"make_connect4", c"make_connect4($module, *, mode=None, batch_size=None, gather=None, num_envs=None, seed=None, max_episode_steps=None, auto_reset=None, chunk_size=None, workers=None, spin=None, outputs=None)\n--\n\nmake_connect4(**pool_options) -> Env"),
    FunctionRow::kw::<MakeLeduc>(c"make_leduc", c"make_leduc($module, *, mode=None, batch_size=None, gather=None, num_envs=None, seed=None, max_episode_steps=None, auto_reset=None, chunk_size=None, workers=None, spin=None, outputs=None)\n--\n\nmake_leduc(**pool_options) -> Env"),
    FunctionRow::kw::<MakeLanterns>(c"make_lanterns", c"make_lanterns($module, *, players=None, zero_on_fuse=None, mode=None, batch_size=None, gather=None, num_envs=None, seed=None, max_episode_steps=None, auto_reset=None, chunk_size=None, workers=None, spin=None, outputs=None)\n--\n\nmake_lanterns(*, players=2, zero_on_fuse=False, **pool_options) -> Env"),
    FunctionRow::kw::<MakeOthello>(c"make_othello", c"make_othello($module, *, mode=None, batch_size=None, gather=None, num_envs=None, seed=None, max_episode_steps=None, auto_reset=None, chunk_size=None, workers=None, spin=None, outputs=None)\n--\n\nmake_othello(**pool_options) -> Env"),
    FunctionRow::kw::<MakeGo9>(
        c"make_go9",
        c"make_go9($module, *, mode=None, batch_size=None, gather=None, num_envs=None, seed=None, max_episode_steps=None, auto_reset=None, chunk_size=None, workers=None, spin=None, outputs=None, komi_half_points=None)\n--\n\nmake_go9(**pool_options, komi_half_points=15) -> Env",
    ),
    FunctionRow::kw::<MakeGrid>(
        c"make_grid",
        c"make_grid($module, *, mode=None, batch_size=None, gather=None, num_envs=None, seed=None, max_episode_steps=None, auto_reset=None, chunk_size=None, workers=None, spin=None, outputs=None, size=None, max_steps=None)\n--\n\nmake_grid(**pool_options, size=16, max_steps=200) -> Env",
    ),
    FunctionRow::kw::<MakeKeydoor>(
        c"make_keydoor",
        c"make_keydoor($module, *, mode=None, batch_size=None, gather=None, num_envs=None, seed=None, max_episode_steps=None, auto_reset=None, chunk_size=None, workers=None, spin=None, outputs=None, size=None, agents=None)\n--\n\nmake_keydoor(**pool_options, size=8, agents=1) -> Env",
    ),
    FunctionRow::kw::<MakeGather>(
        c"make_gather",
        c"make_gather($module, *, mode=None, batch_size=None, gather=None, num_envs=None, seed=None, max_episode_steps=None, auto_reset=None, chunk_size=None, workers=None, spin=None, outputs=None, size=None, items=None, max_steps=None)\n--\n\nmake_gather(**pool_options, size=8, items=8, max_steps=100) -> Env",
    ),
    FunctionRow::kw::<MakeSnake>(
        c"make_snake",
        c"make_snake($module, *, mode=None, batch_size=None, gather=None, num_envs=None, seed=None, max_episode_steps=None, auto_reset=None, chunk_size=None, workers=None, spin=None, outputs=None, size=None, agents=None, obs=None)\n--\n\nmake_snake(**pool_options, size=16, agents=1, obs=\"pixels\") -> Env",
    ),
    FunctionRow::kw::<MakeBrickwall>(
        c"make_brickwall",
        c"make_brickwall($module, *, mode=None, batch_size=None, gather=None, num_envs=None, seed=None, max_episode_steps=None, auto_reset=None, chunk_size=None, workers=None, spin=None, outputs=None, frameskip=None, lives=None, obs=None)\n--\n\nmake_brickwall(**pool_options, frameskip=4, lives=5, obs=\"pixels\") -> Env",
    ),
    FunctionRow::kw::<MakePaddleduel>(
        c"make_paddleduel",
        c"make_paddleduel($module, *, mode=None, batch_size=None, gather=None, num_envs=None, seed=None, max_episode_steps=None, auto_reset=None, chunk_size=None, workers=None, spin=None, outputs=None, agents=None, frameskip=None, max_score=None, bounce=None, obs=None)\n--\n\nmake_paddleduel(**pool_options, agents=1, frameskip=4, max_score=11, bounce=1, obs=\"pixels\") -> Env",
    ),
    FunctionRow::kw::<MakeJobshop>(c"make_jobshop", c"make_jobshop($module, *, jobs=None, machines=None, max_duration=None, mode=None, batch_size=None, gather=None, num_envs=None, seed=None, max_episode_steps=None, auto_reset=None, chunk_size=None, workers=None, spin=None, outputs=None))\n--\n\nmake_jobshop(*, jobs=6, machines=6, max_duration=99, **pool_options) -> Env"),
    FunctionRow::kw::<MakeVrp>(c"make_vrp", c"make_vrp($module, *, customers=None, vehicles=None, grid=None, max_demand=None, capacity=None, mode=None, batch_size=None, gather=None, num_envs=None, seed=None, max_episode_steps=None, auto_reset=None, chunk_size=None, workers=None, spin=None, outputs=None))\n--\n\nmake_vrp(*, customers=20, vehicles=1, grid=100, max_demand=9, capacity=30, **pool_options) -> Env"),
    FunctionRow::kw::<MakeSurvival>(
        c"make_survival",
        c"make_survival($module, *, mode=None, batch_size=None, gather=None, num_envs=None, seed=None, max_episode_steps=None, auto_reset=None, chunk_size=None, workers=None, spin=None, outputs=None, agents=None, size=None, steps=None, view=None, respawn=None, forest_regrow=None, water_permille=None, forest_permille=None, stone_permille=None)\n--\n\nmake_survival(**pool_options, **survival_config) -> Env",
    ),
];

/// The test-hook constructors.
#[cfg(feature = "test-hooks")]
pub(super) const TEST_MAKERS: [FunctionRow; 3] = [
    FunctionRow::kw::<MakePanicCounter>(
        c"_make_panic_counter",
        c"Test hook: a pool whose first step panics.",
    ),
    FunctionRow::kw::<MakeContinuousProbe>(
        c"_make_continuous_probe",
        c"Test hook: a pool with one continuous head.",
    ),
    FunctionRow::kw::<MakeMultiProbe>(
        c"_make_multi_probe",
        c"Test hook: a pool with two discrete heads, MultiDiscrete([3, 5]).",
    ),
];
