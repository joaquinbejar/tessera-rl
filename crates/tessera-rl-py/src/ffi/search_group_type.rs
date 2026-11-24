// SPDX-License-Identifier: MIT OR Apache-2.0

//! The `tessera_rl._native.SearchGroup` type: PUCT searches sharing
//! evaluator batches (ADR-0045), its body, method table and module
//! constructor, as safe impls of the typed layer of `tessera_rl_pyffi`
//! (ADR-0062 item 8).

use tessera_rl_pyffi::ffi::capi::{CResult, Gil, Mine, Ref};
use tessera_rl_pyffi::ffi::module_def::{FunctionKw, FunctionRow};
use tessera_rl_pyffi::ffi::object::{
    Body, Method, MethodNoArgs, MethodRow, MethodTable, Release, TypeObject, TypeSpec,
};

use crate::search_group_api::{self as api, GroupInner};

/// A `SearchGroup` object's body. It holds no Python object: the evaluator
/// is borrowed for one `run` call only.
pub(crate) struct GroupBody(GroupInner);

impl Body for GroupBody {
    /// Dropping the searches frees their pools: detached.
    fn release(self, r: Release<'_>) {
        r.detach(move || drop(self));
    }
}

static TYPE: TypeObject<GroupBody> = TypeObject::new(&SPEC);

static SPEC: TypeSpec<GroupBody> = TypeSpec::new(
    c"tessera_rl._native.SearchGroup",
    c"PUCT searches sharing evaluator batches (use tessera_rl.search.SearchGroup).",
    &METHODS,
);

/// Creates the `SearchGroup` type and adds it to `module`.
pub(super) fn register(g: Gil<'_>, module: Ref<'_>) -> CResult<()> {
    TYPE.create(g, module, c"SearchGroup")
}

/// A method without arguments, calling `$api(g, &GroupInner)`.
macro_rules! noargs {
    ($name:ident, $api:path) => {
        struct $name;
        impl MethodNoArgs for $name {
            type Body = GroupBody;
            fn call<'py>(g: Gil<'py>, body: &GroupBody) -> CResult<Mine<'py>> {
                $api(g, &body.0)
            }
        }
    };
}

/// A method with positional arguments, calling `$api(g, &GroupInner,
/// args)`.
macro_rules! varargs {
    ($name:ident, $api:path) => {
        struct $name;
        impl Method for $name {
            type Body = GroupBody;
            fn call<'py>(g: Gil<'py>, body: &GroupBody, args: Ref<'py>) -> CResult<Mine<'py>> {
                $api(g, &body.0, args)
            }
        }
    };
}

noargs!(LayoutM, api::layout);
noargs!(LastFailureM, api::last_failure);
varargs!(SetRootM, api::set_root);
varargs!(SetActiveM, api::set_active);
varargs!(ActiveM, api::active);
varargs!(RunM, api::run);
varargs!(RootVisitsM, api::root_visits);
varargs!(SelectActionM, api::select_action);
varargs!(ImprovedPolicyM, api::improved_policy);
varargs!(ConsideredM, api::considered);
varargs!(CallsM, api::calls);
varargs!(TreeHashM, api::tree_hash);

static METHODS: MethodTable<GroupBody> = MethodTable::new(&[
    MethodRow::noargs::<LayoutM>(
        c"layout",
        c"layout($self, /)\n--\n\nTrees, actions, observation bytes per leaf and the memory bound.",
    ),
    MethodRow::noargs::<LastFailureM>(
        c"last_failure",
        c"last_failure($self, /)\n--\n\nThe last failed call: None or (slot, message, user_raised).",
    ),
    MethodRow::varargs::<SetRootM>(
        c"set_root",
        c"set_root($self, i, record, /)\n--\n\nset_root(i, record: bytes): replaces slot i's tree by this root.",
    ),
    MethodRow::varargs::<SetActiveM>(
        c"set_active",
        c"set_active($self, i, flag, /)\n--\n\nset_active(i, flag: bool): whether slot i takes part in the next call.",
    ),
    MethodRow::varargs::<ActiveM>(c"active", c"active($self, i, /)\n--\n\nactive(i) -> bool"),
    MethodRow::varargs::<RunM>(c"run", c"run($self, evaluator, /)\n--\n\nrun(evaluator): one group call (all or nothing)."),
    MethodRow::varargs::<RootVisitsM>(
        c"root_visits",
        c"root_visits($self, i, /)\n--\n\nroot_visits(i): slot i's root visit counts.",
    ),
    MethodRow::varargs::<SelectActionM>(
        c"select_action",
        c"select_action($self, i, temperature, /)\n--\n\nselect_action(i, temperature: float) -> int",
    ),
    MethodRow::varargs::<ImprovedPolicyM>(
        c"improved_policy",
        c"improved_policy($self, i, /)\n--\n\nimproved_policy(i): slot i's last Gumbel improved policy, or None.",
    ),
    MethodRow::varargs::<ConsideredM>(
        c"considered",
        c"considered($self, i, /)\n--\n\nconsidered(i): the actions slot i's last Gumbel call considered, or None.",
    ),
    MethodRow::varargs::<CallsM>(c"calls", c"calls($self, i, /)\n--\n\ncalls(i): slot i's completed calls."),
    MethodRow::varargs::<TreeHashM>(c"tree_hash", c"tree_hash($self, i, /)\n--\n\ntree_hash(i): a digest of slot i's tree."),
    MethodRow::END,
]);

/// `make_search_group(**kwargs) -> SearchGroup`.
struct MakeSearchGroup;

impl FunctionKw for MakeSearchGroup {
    fn call<'py>(g: Gil<'py>, args: Ref<'py>, kwargs: Option<Ref<'py>>) -> CResult<Mine<'py>> {
        crate::api::no_positional(g, args, "make_search_group")?;
        let inner = api::make_search_group(g, kwargs)?;
        TYPE.instantiate(g, GroupBody(inner))
    }
}

/// The module-level constructor, for the module's function table.
pub(super) const MAKER: FunctionRow = FunctionRow::kw::<MakeSearchGroup>(
    c"make_search_group",
    c"make_search_group($module, *, trees, slot_options=None, env_id=None, capacity=None, batch=None, eviction=None, c_puct=None, fpu=None, virtual_loss=None, batch_size=None, simulations=None, noise_alpha=None, noise_epsilon=None, seed=None, root=None, gumbel_m=None, c_visit=None, c_scale=None)\n--\n\nmake_search_group(trees, slot_options=None, **search_options) -> SearchGroup",
);
