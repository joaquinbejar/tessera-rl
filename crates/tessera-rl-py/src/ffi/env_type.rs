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
varargs!(SendM, api::send);
varargs!(ResetM, api::reset);
varargs!(StepM, api::step);
varargs!(SnapshotM, api::snapshot);
varargs!(IllegalActionsM, api::illegal_actions);
varargs!(StateHashM, api::state_hash);
varargs!(ImportRecordM, api::import_record);
varargs!(ForkM, api::fork);
varargs!(EncodeM, api::encode);
varargs!(LoadM, api::load);

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
    MethodRow::noargs::<LayoutM>(c"layout", c"Shapes and dtypes of the outputs."),
    MethodRow::noargs::<OutputsM>(c"outputs", c"Copies of the last call's outputs."),
    MethodRow::noargs::<IsPoisonedM>(c"is_poisoned", c"Whether the pool is poisoned."),
    MethodRow::noargs::<CloseM>(c"close", c"Frees the pool."),
    MethodRow::varargs::<ResetM>(c"reset", c"reset(seed=None, ids=None)"),
    MethodRow::varargs::<StepM>(c"step", c"step(actions: bytes of int32)"),
    MethodRow::varargs::<SnapshotM>(c"snapshot", c"snapshot(id) -> bytes"),
    MethodRow::varargs::<IllegalActionsM>(
        c"illegal_actions",
        c"illegal_actions(id) -> per-agent illegal actions this episode",
    ),
    MethodRow::varargs::<StateHashM>(
        c"state_hash",
        c"state_hash(id) -> the instance's state hash",
    ),
    MethodRow::varargs::<ImportRecordM>(c"import_record", c"import_record(id, record)"),
    MethodRow::varargs::<EncodeM>(c"encode", c"encode(ids) -> bytes"),
    MethodRow::varargs::<LoadM>(
        c"load",
        c"load(data, ids, max_instances, max_decoded_bytes)",
    ),
    MethodRow::varargs::<SendM>(c"send", c"send(half, actions: bytes of int32, delivered)"),
    MethodRow::varargs::<RecvM>(
        c"recv",
        c"recv(delivered) -> the oldest sent half's outputs",
    ),
    MethodRow::varargs::<RestoreM>(c"restore", c"restore(ids, snapshots, delivered)"),
    MethodRow::noargs::<SnapshotAllM>(
        c"snapshot_all",
        c"snapshot_all() -> an opaque whole-pool snapshot",
    ),
    MethodRow::varargs::<RestoreAllM>(
        c"restore_all",
        c"restore_all(snapshot) -> both halves' outputs",
    ),
    MethodRow::varargs::<ForkM>(c"fork", c"fork(src, dst)"),
    MethodRow::END,
]);

/// A module function that builds an `Env` from keyword arguments.
macro_rules! maker {
    ($name:ident, $api:path) => {
        pub(super) struct $name;
        impl FunctionKw for $name {
            fn call<'py>(
                g: Gil<'py>,
                _args: Ref<'py>,
                kwargs: Option<Ref<'py>>,
            ) -> CResult<Mine<'py>> {
                let inner = $api(g, kwargs)?;
                TYPE.instantiate(g, EnvBody(inner))
            }
        }
    };
}

#[cfg(feature = "test-hooks")]
maker!(MakePanicCounter, api::make_panic_counter);
#[cfg(feature = "test-hooks")]
maker!(MakeContinuousProbe, api::make_continuous_probe);
#[cfg(feature = "test-hooks")]
maker!(MakeMultiProbe, api::make_multi_probe);
maker!(MakeCartpole, api::make_cartpole);
maker!(MakePendulum, api::make_pendulum);
maker!(MakeAcrobot, api::make_acrobot);
maker!(MakeMountaincar, api::make_mountaincar);
maker!(MakeTag, api::make_tag);
maker!(MakeRts, api::make_rts);
maker!(MakeGrid, api::make_grid);
maker!(MakeKeydoor, api::make_keydoor);
maker!(MakeSnake, api::make_snake);
maker!(MakeBrickwall, api::make_brickwall);
maker!(MakePaddleduel, api::make_paddleduel);
maker!(MakeConnect4, api::make_connect4);
maker!(MakeLeduc, api::make_leduc);
maker!(MakeOthello, api::make_othello);
maker!(MakeGo9, api::make_go9);

/// The module-level constructors, for the module's function table.
pub(super) const MAKERS: [FunctionRow; 15] = [
    FunctionRow::kw::<MakeCartpole>(c"make_cartpole", c"make_cartpole(**pool_options) -> Env"),
    FunctionRow::kw::<MakeAcrobot>(c"make_acrobot", c"make_acrobot(**pool_options) -> Env"),
    FunctionRow::kw::<MakeMountaincar>(
        c"make_mountaincar",
        c"make_mountaincar(**pool_options) -> Env",
    ),
    FunctionRow::kw::<MakePendulum>(c"make_pendulum", c"make_pendulum(**pool_options) -> Env"),
    FunctionRow::kw::<MakeTag>(
        c"make_tag",
        c"make_tag(**pool_options, **tag_config) -> Env",
    ),
    FunctionRow::kw::<MakeRts>(
        c"make_rts",
        c"make_rts(**pool_options, **rts_config) -> Env",
    ),
    FunctionRow::kw::<MakeConnect4>(c"make_connect4", c"make_connect4(**pool_options) -> Env"),
    FunctionRow::kw::<MakeLeduc>(c"make_leduc", c"make_leduc(**pool_options) -> Env"),
    FunctionRow::kw::<MakeOthello>(c"make_othello", c"make_othello(**pool_options) -> Env"),
    FunctionRow::kw::<MakeGo9>(
        c"make_go9",
        c"make_go9(**pool_options, komi_half_points=15) -> Env",
    ),
    FunctionRow::kw::<MakeGrid>(
        c"make_grid",
        c"make_grid(**pool_options, size=16, max_steps=200) -> Env",
    ),
    FunctionRow::kw::<MakeKeydoor>(
        c"make_keydoor",
        c"make_keydoor(**pool_options, size=8, agents=1) -> Env",
    ),
    FunctionRow::kw::<MakeSnake>(
        c"make_snake",
        c"make_snake(**pool_options, size=16, agents=1, obs=\"pixels\") -> Env",
    ),
    FunctionRow::kw::<MakeBrickwall>(
        c"make_brickwall",
        c"make_brickwall(**pool_options, frameskip=4, lives=5, obs=\"pixels\") -> Env",
    ),
    FunctionRow::kw::<MakePaddleduel>(
        c"make_paddleduel",
        c"make_paddleduel(**pool_options, agents=1, frameskip=4, max_score=11, bounce=1, obs=\"pixels\") -> Env",
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
