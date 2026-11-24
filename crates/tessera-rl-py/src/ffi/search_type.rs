// SPDX-License-Identifier: MIT OR Apache-2.0

//! The `tessera_rl._native.Search` type: a PUCT search (ADR-0043), its
//! body, method table and module constructor, as safe impls of the typed
//! layer of `tessera_rl_pyffi` (ADR-0062 item 8).

use tessera_rl_pyffi::ffi::capi::{CResult, Gil, Mine, Ref};
use tessera_rl_pyffi::ffi::module_def::{FunctionKw, FunctionRow};
use tessera_rl_pyffi::ffi::object::{
    Body, Method, MethodKw, MethodNoArgs, MethodRow, MethodTable, Release, TypeObject, TypeSpec,
};

use crate::search_api::{self as api, SearchInner};

/// A `Search` object's body. It holds no Python object: the evaluator is
/// borrowed for one `run` call only.
pub(crate) struct SearchBody(SearchInner);

impl Body for SearchBody {
    /// Dropping the search frees its pools: detached.
    fn release(self, r: Release<'_>) {
        r.detach(move || drop(self));
    }
}

static TYPE: TypeObject<SearchBody> = TypeObject::new(&SPEC);

static SPEC: TypeSpec<SearchBody> = TypeSpec::new(
    c"tessera_rl._native.Search",
    c"A PUCT or Gumbel-root search over an environment (use tessera_rl.search.Search).",
    &METHODS,
);

/// Creates the `Search` type and adds it to `module`.
pub(super) fn register(g: Gil<'_>, module: Ref<'_>) -> CResult<()> {
    TYPE.create(g, module, c"Search")
}

/// A method without arguments, calling `$api(g, &SearchInner)`.
macro_rules! noargs {
    ($name:ident, $api:path) => {
        struct $name;
        impl MethodNoArgs for $name {
            type Body = SearchBody;
            fn call<'py>(g: Gil<'py>, body: &SearchBody) -> CResult<Mine<'py>> {
                $api(g, &body.0)
            }
        }
    };
}

/// A method with positional arguments, calling `$api(g, &SearchInner,
/// args)`.
macro_rules! varargs {
    ($name:ident, $api:path) => {
        struct $name;
        impl Method for $name {
            type Body = SearchBody;
            fn call<'py>(g: Gil<'py>, body: &SearchBody, args: Ref<'py>) -> CResult<Mine<'py>> {
                $api(g, &body.0, args)
            }
        }
    };
}

noargs!(LayoutM, api::layout);
noargs!(RootVisitsM, api::root_visits);
noargs!(ImprovedPolicyM, api::improved_policy);
noargs!(ConsideredM, api::considered);
noargs!(CallsM, api::calls);
noargs!(TreeHashM, api::tree_hash);
noargs!(IsPrivilegedM, api::is_privileged);
noargs!(RunOnnxM, crate::onnx_api::run_onnx);
varargs!(SetRootM, api::set_root);
varargs!(RunM, api::run);
varargs!(SelectActionM, api::select_action);

/// `attach_onnx(**kwargs)`: keyword arguments only.
struct AttachOnnxM;

impl MethodKw for AttachOnnxM {
    type Body = SearchBody;
    fn call<'py>(
        g: Gil<'py>,
        body: &SearchBody,
        args: Ref<'py>,
        kwargs: Option<Ref<'py>>,
    ) -> CResult<Mine<'py>> {
        crate::api::no_positional(g, args, "attach_onnx")?;
        crate::onnx_api::attach_onnx(g, &body.0, kwargs)
    }
}

static METHODS: MethodTable<SearchBody> = MethodTable::new(&[
    MethodRow::noargs::<LayoutM>(c"layout", c"layout($self, /)\n--\n\nActions and observation bytes per leaf."),
    MethodRow::varargs::<SetRootM>(
        c"set_root",
        c"set_root($self, record, /)\n--\n\nset_root(record: bytes): replaces the tree by this root.",
    ),
    MethodRow::varargs::<RunM>(c"run", c"run($self, evaluator, /)\n--\n\nrun(evaluator): one search call (all or nothing)."),
    MethodRow::noargs::<RootVisitsM>(c"root_visits", c"root_visits($self, /)\n--\n\nThe root's visit counts."),
    MethodRow::varargs::<SelectActionM>(
        c"select_action",
        c"select_action($self, temperature, /)\n--\n\nselect_action(temperature: float) -> int",
    ),
    MethodRow::noargs::<ImprovedPolicyM>(
        c"improved_policy",
        c"improved_policy($self, /)\n--\n\nThe last Gumbel call's improved policy, or None.",
    ),
    MethodRow::noargs::<ConsideredM>(
        c"considered",
        c"considered($self, /)\n--\n\nThe actions the last Gumbel call considered, or None.",
    ),
    MethodRow::noargs::<CallsM>(c"calls", c"calls($self, /)\n--\n\nCompleted search calls."),
    MethodRow::noargs::<TreeHashM>(c"tree_hash", c"tree_hash($self, /)\n--\n\nA digest of the whole tree."),
    MethodRow::noargs::<IsPrivilegedM>(
        c"is_privileged",
        c"is_privileged($self, /)\n--\n\nWhether the search reads hidden state (ADR-0046).",
    ),
    MethodRow::kw::<AttachOnnxM>(
        c"attach_onnx",
        c"attach_onnx($self, *, model, library, accept_telemetry=False, max_model_bytes=None)\n--\n\nattach_onnx(model=, library=, accept_telemetry=, max_model_bytes=) -> dict",
    ),
    MethodRow::noargs::<RunOnnxM>(
        c"run_onnx",
        c"run_onnx($self, /)\n--\n\nrun_onnx(): one search call with the attached ONNX evaluator.",
    ),
    MethodRow::END,
]);

/// `make_search(**kwargs) -> Search`.
struct MakeSearch;

impl FunctionKw for MakeSearch {
    fn call<'py>(g: Gil<'py>, args: Ref<'py>, kwargs: Option<Ref<'py>>) -> CResult<Mine<'py>> {
        crate::api::no_positional(g, args, "make_search")?;
        let inner = api::make_search(g, kwargs)?;
        TYPE.instantiate(g, SearchBody(inner))
    }
}

/// `onnx_load(library=, accept_telemetry=) -> dict`: loads ONNX Runtime.
struct OnnxLoad;

impl FunctionKw for OnnxLoad {
    fn call<'py>(g: Gil<'py>, args: Ref<'py>, kwargs: Option<Ref<'py>>) -> CResult<Mine<'py>> {
        crate::api::no_positional(g, args, "onnx_load")?;
        crate::onnx_api::onnx_load(g, kwargs)
    }
}

/// The module-level ONNX Runtime loader (ADR-0057).
pub(super) const ONNX_LOAD: FunctionRow = FunctionRow::kw::<OnnxLoad>(
    c"onnx_load",
    c"onnx_load($module, *, library, accept_telemetry=False)\n--\n\nonnx_load(library=, accept_telemetry=False) -> dict",
);

/// The module-level constructor, for the module's function table.
pub(super) const MAKER: FunctionRow =
    FunctionRow::kw::<MakeSearch>(c"make_search", c"make_search($module, *, env_id=None, capacity=None, batch=None, eviction=None, c_puct=None, fpu=None, virtual_loss=None, batch_size=None, simulations=None, noise_alpha=None, noise_epsilon=None, seed=None, root=None, gumbel_m=None, c_visit=None, c_scale=None)\n--\n\nmake_search(**search_options) -> Search");
