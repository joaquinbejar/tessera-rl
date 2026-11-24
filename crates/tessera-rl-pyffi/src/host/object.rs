// SPDX-License-Identifier: MIT OR Apache-2.0

//! The host object (SPEC 14.4.2 draft): a host environment's pool behind
//! the reentrancy guard, with its protocol attributes and methods.

use std::ffi::CString;
use std::sync::{Arc, OnceLock};

use tessera_rl_engine::{ActionInput, DynHostPool, EngineError, OutputStorage, Reseed};
use tessera_rl_num::identity::Seed;

use crate::errors::{self, Kind};
use crate::ffi::boundary::{self, Policy};
use crate::ffi::capi::{self, CResult, Gil, Mine, Ref};
use crate::ffi::inner::Inner;
use crate::ffi::object::{
    Attribute, AttributeRow, AttributeTable, Body, Method, MethodNoArgs, MethodRow, MethodTable,
    Release, TypeObject, TypeSpec,
};

use super::{describe, views};

/// The host protocol version of an object whose environment does not
/// refuse agents (SPEC 14.4.7).
pub const HOST_PROTOCOL: u64 = 1;

/// The host protocol version of an object whose environment refuses agents
/// (`HostEnvironment::REFUSES_AGENTS`): protocol 1 plus the `refused` and
/// `refuse_code` outputs, which a wrapper must read (ADR-0070 item 10).
pub const HOST_PROTOCOL_REFUSING: u64 = 2;

/// The body of a host object: its pool, until `close`, and the output
/// storage its views share, which outlives the pool (ADR-0003 rule 1).
pub struct HostBody {
    pub(super) inner: Inner<dyn DynHostPool>,
    storage: Arc<OutputStorage>,
    /// [`HOST_PROTOCOL`] or [`HOST_PROTOCOL_REFUSING`], fixed by the
    /// environment type.
    protocol: u64,
}

impl HostBody {
    /// A host object's body around `pool`, whose views share `storage`
    /// (the pool's `Outputs::storage`).
    pub(super) fn new(pool: Box<dyn DynHostPool>, storage: Arc<OutputStorage>) -> Self {
        let protocol = if pool.layout().refuses_agents {
            HOST_PROTOCOL_REFUSING
        } else {
            HOST_PROTOCOL
        };
        Self {
            inner: Inner::new(pool, true),
            storage,
            protocol,
        }
    }

    /// The state, refusing an object inherited through a fork before any
    /// lock or flag of it is touched.
    pub(super) fn guarded(&self, g: Gil<'_>) -> CResult<&Inner<dyn DynHostPool>> {
        self.inner.check_fork(g)?;
        Ok(&self.inner)
    }
}

impl Body for HostBody {
    /// Dropping a pool joins its workers, so it is dropped detached. In a
    /// forked child the copy of the workers does not exist and its locks may
    /// be held forever: the memory is leaked by design, as for the
    /// first-party `Env`.
    fn release(self, r: Release<'_>) {
        if self.inner.inherited() {
            core::mem::forget(self);
        } else {
            r.detach(move || drop(self));
        }
    }
}

/// `host_protocol`: 1, or 2 for an environment that refuses agents.
struct HostProtocol;

impl Attribute for HostProtocol {
    type Body = HostBody;
    fn get<'py>(g: Gil<'py>, body: &HostBody) -> CResult<Mine<'py>> {
        capi::int_u64(g, body.protocol)
    }
}

/// `tessera_rl_min`: the version this module was built with.
struct TesseraMin;

impl Attribute for TesseraMin {
    type Body = HostBody;
    fn get<'py>(g: Gil<'py>, _: &HostBody) -> CResult<Mine<'py>> {
        capi::string(g, &describe::version_min())
    }
}

/// `tessera_rl_below`: the next minor version, excluded.
struct TesseraBelow;

impl Attribute for TesseraBelow {
    type Body = HostBody;
    fn get<'py>(g: Gil<'py>, _: &HostBody) -> CResult<Mine<'py>> {
        capi::string(g, &describe::version_below())
    }
}

/// `layout()`: the dict of SPEC 14.4.3 (draft, with ADR-0065's keys).
struct LayoutM;

impl MethodNoArgs for LayoutM {
    type Body = HostBody;
    fn call<'py>(g: Gil<'py>, body: &HostBody) -> CResult<Mine<'py>> {
        body.guarded(g)?.with(g, |pool| describe::layout(g, &*pool))
    }
}

/// `is_poisoned()`: whether a panic poisoned the pool (ADR-0028).
struct IsPoisonedM;

impl MethodNoArgs for IsPoisonedM {
    type Body = HostBody;
    fn call<'py>(g: Gil<'py>, body: &HostBody) -> CResult<Mine<'py>> {
        body.guarded(g)?
            .with(g, |pool| Ok(capi::boolean(g, pool.is_poisoned())))
    }
}

fn engine(g: Gil<'_>, e: &EngineError) -> capi::Raised {
    errors::error(g, errors::engine_kind(e), e.to_string())
}

fn argument(g: Gil<'_>, message: &str) -> capi::Raised {
    errors::error(g, Kind::Argument, message.to_owned())
}

/// A seed: an exact `int` in `[0, 2**64)` (never a `bool`, an `int`
/// subclass such as an `IntEnum`, or a NumPy integer). The error says what
/// the value is: a bool, an int out of range, or not an exact int.
fn seed_of(g: Gil<'_>, v: Ref<'_>, what: &str) -> CResult<Seed> {
    capi::to_u64(g, v, what).map(Seed::new).map_err(|_| {
        capi::clear_pending(g);
        let got = match capi::kind(v) {
            capi::Kind::Bool => "a bool",
            capi::Kind::Int => "an int outside that range",
            _ => "not an exact int",
        };
        argument(
            g,
            &format!("reset: {what} must be an exact int in [0, 2**64); got {got}"),
        )
    })
}

/// The outputs of the call that just ran: views at the storage's current
/// generation. A poisoned pool exports none (its rows are undefined).
fn outputs<'py>(g: Gil<'py>, body: &HostBody, pool: &dyn DynHostPool) -> CResult<Mine<'py>> {
    if pool.is_poisoned() {
        return Err(engine(g, &EngineError::Poisoned));
    }
    views::outputs(g, pool.layout(), &body.storage)
}

/// `reset(seed=None)`: resets every instance and returns the outputs.
/// `seed` is `None` (keep the seeds), an int (instance i gets `seed + i`)
/// or a list of `num_envs` ints (ADR-0007). A full reset clears poison.
struct ResetM;

impl Method for ResetM {
    type Body = HostBody;
    fn call<'py>(g: Gil<'py>, body: &HostBody, args: Ref<'py>) -> CResult<Mine<'py>> {
        let items = capi::tuple_items(g, args, "the arguments")?;
        let seed = match items.as_slice() {
            [] => None,
            [v] if v.is_none() => None,
            [v] => Some(*v),
            _ => return Err(argument(g, "reset takes at most one argument, seed")),
        };
        body.guarded(g)?.with(g, |pool| {
            let n = pool.layout().num_envs;
            let list: Vec<Seed>;
            let reseed = match seed {
                None => Reseed::Keep,
                Some(v) if capi::is_list(v) => {
                    list = capi::list_items(g, v, "seed")?
                        .iter()
                        .map(|s| seed_of(g, s.as_ref(), "each seed"))
                        .collect::<CResult<_>>()?;
                    if list.len() != n {
                        return Err(argument(
                            g,
                            &format!("reset: a seed list needs {n} seeds, got {}", list.len()),
                        ));
                    }
                    Reseed::List(&list)
                }
                Some(v) => Reseed::Base(seed_of(g, v, "seed")?),
            };
            g.detach(|| pool.reset_all(reseed).map(|_| ()))
                .map_err(|e| engine(g, &e))?;
            outputs(g, body, &*pool)
        })
    }
}

/// `step(actions)`: steps every instance with `actions`, exact `bytes` of
/// `num_envs * agents * heads` little-endian `int32` (discrete heads) or
/// `float32` (the one continuous head), and returns the outputs. Invalid
/// actions raise `ActionError` with nothing stepped.
struct StepM;

impl Method for StepM {
    type Body = HostBody;
    fn call<'py>(g: Gil<'py>, body: &HostBody, args: Ref<'py>) -> CResult<Mine<'py>> {
        let [actions] = capi::tuple_items(g, args, "the arguments")?
            .try_into()
            .map_err(|_| argument(g, "step takes exactly one argument, actions"))?;
        let raw = capi::bytes_of(g, actions, "actions").map_err(|_| {
            capi::clear_pending(g);
            argument(g, "step: actions must be bytes")
        })?;
        body.guarded(g)?.with(g, |pool| {
            let continuous = !pool.layout().continuous_heads.is_empty();
            if raw.len() % 4 != 0 {
                let what = if continuous { "float32" } else { "int32" };
                return Err(errors::error(
                    g,
                    Kind::Action,
                    format!("step: actions must be {what} values"),
                ));
            }
            let words = raw.as_chunks::<4>().0.iter();
            let (discrete, floats): (Vec<i32>, Vec<f32>) = if continuous {
                (Vec::new(), words.map(|c| f32::from_le_bytes(*c)).collect())
            } else {
                (words.map(|c| i32::from_le_bytes(*c)).collect(), Vec::new())
            };
            let input = ActionInput {
                discrete: &discrete,
                continuous: &floats,
            };
            g.detach(|| pool.step(input).map(|_| ()))
                .map_err(|e| engine(g, &e))?;
            outputs(g, body, &*pool)
        })
    }
}

/// Instance ids: a list of exact ints in `[0, 2**32)`.
fn ids_of(g: Gil<'_>, v: Ref<'_>, what: &str) -> CResult<Vec<u32>> {
    let items = capi::list_items(g, v, what).map_err(|_| {
        capi::clear_pending(g);
        argument(g, &format!("{what} must be a list of ints"))
    })?;
    items.iter().map(|i| id_of(g, i.as_ref(), what)).collect()
}

/// One instance id: an exact int in `[0, 2**32)`.
fn id_of(g: Gil<'_>, v: Ref<'_>, what: &str) -> CResult<u32> {
    capi::to_u64(g, v, what)
        .ok()
        .and_then(|x| u32::try_from(x).ok())
        .ok_or_else(|| {
            capi::clear_pending(g);
            argument(g, &format!("{what} must be exact ints in [0, 2**32)"))
        })
}

/// The persisted-file error of a snapshot method.
fn persist(g: Gil<'_>, e: &tessera_rl_engine::PersistError) -> capi::Raised {
    errors::error(g, errors::persist_kind(e), e.to_string())
}

/// The arguments of a method, exactly `N` of them.
fn exactly<'a, const N: usize>(g: Gil<'_>, args: Ref<'a>, usage: &str) -> CResult<[Ref<'a>; N]> {
    capi::tuple_items(g, args, "the arguments")?
        .try_into()
        .map_err(|_| argument(g, usage))
}

/// `snapshot(ids)`: one `bytes` record per id (ADR-0073), in order.
struct SnapshotM;

impl Method for SnapshotM {
    type Body = HostBody;
    fn call<'py>(g: Gil<'py>, body: &HostBody, args: Ref<'py>) -> CResult<Mine<'py>> {
        let [ids] = exactly(g, args, "snapshot takes one argument, ids")?;
        let ids = ids_of(g, ids, "snapshot: ids")?;
        body.guarded(g)?.with(g, |pool| {
            let records = g
                .detach(|| pool.snapshot_records(&ids))
                .map_err(|e| engine(g, &e))?;
            let list = capi::list(g)?;
            for r in &records {
                capi::list_append(g, &list, &capi::bytes(g, r)?)?;
            }
            Ok(list)
        })
    }
}

/// The records of a list of `bytes`, borrowed from `items`.
fn records_of<'a>(g: Gil<'_>, items: &'a [Mine<'_>]) -> CResult<Vec<&'a [u8]>> {
    items
        .iter()
        .map(|i| {
            capi::bytes_of(g, i.as_ref(), "each record").map_err(|_| {
                capi::clear_pending(g);
                argument(g, "records must be a list of bytes")
            })
        })
        .collect()
}

/// `restore(ids, records)`: restores the records into `ids`; every record
/// is checked before any instance changes (ADR-0073). Returns `None`.
struct RestoreM;

impl Method for RestoreM {
    type Body = HostBody;
    fn call<'py>(g: Gil<'py>, body: &HostBody, args: Ref<'py>) -> CResult<Mine<'py>> {
        let [ids, records] = exactly(g, args, "restore takes two arguments, ids and records")?;
        let ids = ids_of(g, ids, "restore: ids")?;
        let items = capi::list_items(g, records, "records")?;
        let records = records_of(g, &items)?;
        body.guarded(g)?.with(g, |pool| {
            g.detach(|| pool.restore_records(&ids, &records))
                .map_err(|e| engine(g, &e))?;
            Ok(capi::none(g))
        })
    }
}

/// `restore_all(records)`: one record per instance. Refused on a poisoned
/// pool (ADR-0073: no recovery by restore).
struct RestoreAllM;

impl Method for RestoreAllM {
    type Body = HostBody;
    fn call<'py>(g: Gil<'py>, body: &HostBody, args: Ref<'py>) -> CResult<Mine<'py>> {
        let [records] = exactly(g, args, "restore_all takes one argument, records")?;
        let items = capi::list_items(g, records, "records")?;
        let records = records_of(g, &items)?;
        body.guarded(g)?.with(g, |pool| {
            g.detach(|| pool.restore_all_records(&records))
                .map_err(|e| engine(g, &e))?;
            Ok(capi::none(g))
        })
    }
}

/// `fork(src, dst_ids)`: copies instance `src` into every instance of
/// `dst_ids`. There is no `fork_branched` (ADR-0073).
struct ForkM;

impl Method for ForkM {
    type Body = HostBody;
    fn call<'py>(g: Gil<'py>, body: &HostBody, args: Ref<'py>) -> CResult<Mine<'py>> {
        let [src, dst] = exactly(g, args, "fork takes two arguments, src and dst_ids")?;
        let src = id_of(g, src, "fork: src")?;
        let dst = ids_of(g, dst, "fork: dst_ids")?;
        body.guarded(g)?.with(g, |pool| {
            g.detach(|| pool.fork(src, &dst))
                .map_err(|e| engine(g, &e))?;
            Ok(capi::none(g))
        })
    }
}

/// `state_hash(id)`: the host `StateHash` as 8 little-endian bytes.
struct StateHashM;

impl Method for StateHashM {
    type Body = HostBody;
    fn call<'py>(g: Gil<'py>, body: &HostBody, args: Ref<'py>) -> CResult<Mine<'py>> {
        let [id] = exactly(g, args, "state_hash takes one argument, id")?;
        let id = id_of(g, id, "state_hash: id")?;
        body.guarded(g)?.with(g, |pool| {
            let hash = g
                .detach(|| pool.state_hash(id))
                .map_err(|e| engine(g, &e))?;
            capi::bytes(g, &hash.to_le_bytes())
        })
    }
}

/// `encode(ids)`: a `TESHOST` file holding the records of `ids`.
struct EncodeM;

impl Method for EncodeM {
    type Body = HostBody;
    fn call<'py>(g: Gil<'py>, body: &HostBody, args: Ref<'py>) -> CResult<Mine<'py>> {
        let [ids] = exactly(g, args, "encode takes one argument, ids")?;
        let ids = ids_of(g, ids, "encode: ids")?;
        body.guarded(g)?.with(g, |pool| {
            let file = g.detach(|| pool.encode(&ids)).map_err(|e| persist(g, &e))?;
            capi::bytes(g, &file)
        })
    }
}

/// `import_records(ids, file)`: decodes a `TESHOST` file with the default
/// limits and restores its records into `ids`.
struct ImportRecordsM;

impl Method for ImportRecordsM {
    type Body = HostBody;
    fn call<'py>(g: Gil<'py>, body: &HostBody, args: Ref<'py>) -> CResult<Mine<'py>> {
        let [ids, file] = exactly(g, args, "import_records takes two arguments, ids and file")?;
        let ids = ids_of(g, ids, "import_records: ids")?;
        let file = capi::bytes_of(g, file, "file").map_err(|_| {
            capi::clear_pending(g);
            argument(g, "import_records: file must be bytes")
        })?;
        body.guarded(g)?.with(g, |pool| {
            g.detach(|| pool.import_file(&ids, file, tessera_rl_engine::DecodeLimits::default()))
                .map_err(|e| persist(g, &e))?;
            Ok(capi::none(g))
        })
    }
}

/// `_tessera_current_generation()`: the generation of the current borrow
/// window; it increases on every call that writes outputs.
struct CurrentGenerationM;

impl MethodNoArgs for CurrentGenerationM {
    type Body = HostBody;
    fn call<'py>(g: Gil<'py>, body: &HostBody) -> CResult<Mine<'py>> {
        capi::int_u64(g, body.storage.generation())
    }
}

/// `_tessera_set_fp_policy(policy)`: `"canonicalize"` or `"reject"`
/// (ADR-0041). The policy is this module's, global to every object, view
/// and deleter of it, as `tessera_rl.set_fp_policy` sets it (SPEC 14.4.2).
struct SetFpPolicyM;

impl Method for SetFpPolicyM {
    type Body = HostBody;
    fn call<'py>(g: Gil<'py>, _body: &HostBody, args: Ref<'py>) -> CResult<Mine<'py>> {
        let items = capi::tuple_items(g, args, "the arguments")?;
        let [name] = items.as_slice() else {
            return Err(argument(g, "_tessera_set_fp_policy takes one argument"));
        };
        let name = capi::to_string(g, *name, "the policy").map_err(|_| {
            capi::clear_pending(g);
            argument(g, "the floating-point policy is a str")
        })?;
        let policy = match name.as_str() {
            "canonicalize" => Policy::Canonicalize,
            "reject" => Policy::Reject,
            other => {
                return Err(argument(
                    g,
                    &format!(
                        "the floating-point policy is \"canonicalize\" or \"reject\", not {other:?}"
                    ),
                ));
            }
        };
        boundary::set_policy(policy);
        Ok(capi::none(g))
    }
}

/// `close()`: releases the pool; a second `close` does nothing. Views
/// already exported keep their memory. On an inherited object a no-op.
struct CloseM;

impl MethodNoArgs for CloseM {
    type Body = HostBody;
    fn call<'py>(g: Gil<'py>, body: &HostBody) -> CResult<Mine<'py>> {
        if body.inner.inherited() {
            return Ok(capi::none(g));
        }
        let pool = body.inner.take(g)?;
        g.detach(move || drop(pool));
        Ok(capi::none(g))
    }
}

static METHODS: MethodTable<HostBody> = MethodTable::new(&[
    MethodRow::noargs::<LayoutM>(c"layout", c"The pool's layout (SPEC 14.4.3)."),
    MethodRow::varargs::<ResetM>(
        c"reset",
        c"reset(seed=None): resets every instance and returns the outputs.",
    ),
    MethodRow::varargs::<StepM>(
        c"step",
        c"step(actions: bytes of int32, or float32 for a continuous head): steps every instance and returns the outputs.",
    ),
    MethodRow::noargs::<CurrentGenerationM>(
        c"_tessera_current_generation",
        c"The generation of the current borrow window.",
    ),
    MethodRow::noargs::<IsPoisonedM>(c"is_poisoned", c"Whether a panic poisoned the pool."),
    MethodRow::varargs::<SetFpPolicyM>(
        c"_tessera_set_fp_policy",
        c"_tessera_set_fp_policy(policy): \"canonicalize\" or \"reject\", for this module (ADR-0041).",
    ),
    MethodRow::varargs::<SnapshotM>(
        c"snapshot",
        c"snapshot(ids): one bytes record per id (ADR-0073); CapabilityError without snapshots.",
    ),
    MethodRow::varargs::<RestoreM>(
        c"restore",
        c"restore(ids, records): restores bytes records into ids, all checked first.",
    ),
    MethodRow::varargs::<RestoreAllM>(
        c"restore_all",
        c"restore_all(records): one bytes record per instance.",
    ),
    MethodRow::varargs::<ForkM>(
        c"fork",
        c"fork(src, dst_ids): copies instance src into every instance of dst_ids.",
    ),
    MethodRow::varargs::<StateHashM>(
        c"state_hash",
        c"state_hash(id): the host state hash, 8 little-endian bytes.",
    ),
    MethodRow::varargs::<EncodeM>(
        c"encode",
        c"encode(ids): a TESHOST file holding the records of ids.",
    ),
    MethodRow::varargs::<ImportRecordsM>(
        c"import_records",
        c"import_records(ids, file): restores the records of a TESHOST file into ids.",
    ),
    MethodRow::noargs::<CloseM>(c"close", c"Releases the pool."),
    MethodRow::END,
]);

static ATTRIBUTES: AttributeTable<HostBody> = AttributeTable::new(&[
    AttributeRow::get::<HostProtocol>(
        c"host_protocol",
        c"The host protocol version: 1, or 2 when the environment refuses agents.",
    ),
    AttributeRow::get::<TesseraMin>(
        c"tessera_rl_min",
        c"The lowest tessera_rl version this object works with (MAJOR.MINOR.PATCH).",
    ),
    AttributeRow::get::<TesseraBelow>(
        c"tessera_rl_below",
        c"The first tessera_rl version this object does not work with (MAJOR.MINOR.PATCH).",
    ),
    AttributeRow::END,
]);

static SPEC: TypeSpec<HostBody> = TypeSpec::new(
    c"_tessera_host.HostEnv",
    c"A host environment's pool (host protocol 1, SPEC 14.4).",
    &METHODS,
)
.with_attributes(&ATTRIBUTES);

pub(super) static HOST_TYPE: TypeObject<HostBody> = TypeObject::new(&SPEC);

/// `<module>.<ty>`, the qualified name of a type of the module `module`.
pub(super) fn qualified(module: &core::ffi::CStr, ty: &str) -> Option<CString> {
    CString::new(format!("{}.{ty}", module.to_str().ok()?)).ok()
}

/// The qualified type names, made once per process from the module name
/// `export_host_env!` gives, so a packaged wheel's types are
/// `<package>.<module>.HostEnv` and `.HostView`.
static NAMES: OnceLock<(CString, CString)> = OnceLock::new();

/// Adds the host object and view types to `module`, named after
/// `module_name` (the `module = c"..."` of `export_host_env!`).
pub(super) fn register(
    g: Gil<'_>,
    module: Ref<'_>,
    module_name: &'static core::ffi::CStr,
) -> CResult<()> {
    let (Some(env), Some(view)) = (
        qualified(module_name, "HostEnv"),
        qualified(module_name, "HostView"),
    ) else {
        return Err(capi::raise(g, capi::runtime_error(), "host module name"));
    };
    let names = NAMES.get_or_init(|| (env, view));
    HOST_TYPE.create_as(g, module, c"HostEnv", names.0.as_c_str())?;
    views::register(g, module, names.1.as_c_str())
}

#[cfg(test)]
mod tests {
    use super::qualified;

    #[test]
    fn test_host_type_names_follow_the_module() {
        assert_eq!(
            qualified(c"book_env._tessera_host", "HostEnv")
                .unwrap()
                .as_bytes(),
            b"book_env._tessera_host.HostEnv"
        );
        assert_eq!(
            qualified(c"_tessera_host", "HostView").unwrap().as_bytes(),
            b"_tessera_host.HostView"
        );
    }
}
