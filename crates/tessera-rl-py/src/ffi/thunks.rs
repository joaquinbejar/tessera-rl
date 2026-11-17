// SPDX-License-Identifier: MIT OR Apache-2.0

//! Every boundary thunk of `_native` (ADR-0041), made with
//! `tessera_rl_pyffi::thunk!` from the shared templates (ADR-0062 item 8).
//! Audited module: each invocation expands to a naked function.

use pyo3::ffi;
use tessera_rl_pyffi::ffi::thunks::{CFunction, CFunctionKw, Destructor, GetBuffer};
use tessera_rl_pyffi::thunk;

use super::{env_type, module, search_group_type, search_type, snapshot_type, view_type};

type Obj = *mut ffi::PyObject;

thunk!(
    @naked
    /// The module initializer CPython looks up by name.
    #[unsafe(no_mangle)]
    PyInit__native => module::init : fn() -> Obj
);

thunk!(VERSION: CFunction = thunk_version => module::version : fn(Obj, Obj) -> Obj);
thunk!(SET_FP_POLICY: CFunction = thunk_set_fp_policy => module::set_fp_policy : fn(Obj, Obj) -> Obj);
thunk!(GET_FP_POLICY: CFunction = thunk_get_fp_policy => module::get_fp_policy : fn(Obj, Obj) -> Obj);
thunk!(FP_NON_CANONICAL_ENTRIES: CFunction = thunk_fp_non_canonical_entries => module::fp_non_canonical_entries : fn(Obj, Obj) -> Obj);
#[cfg(feature = "test-hooks")]
thunk!(FAIL_AFTER: CFunction = thunk_fail_after => module::fail_after : fn(Obj, Obj) -> Obj);
thunk!(FP_BOUNDARY: CFunction = thunk_fp_boundary => module::fp_boundary : fn(Obj, Obj) -> Obj);
thunk!(BUILD_PROFILE: CFunction = thunk_build_profile => module::build_profile : fn(Obj, Obj) -> Obj);
thunk!(REGISTERED_ENVS: CFunction = thunk_registered_envs => module::registered_envs : fn(Obj, Obj) -> Obj);
#[cfg(feature = "test-hooks")]
thunk!(MAKE_PANIC_COUNTER: CFunctionKw = thunk_make_panic_counter => env_type::make_panic_counter : fn(Obj, Obj, Obj) -> Obj);
#[cfg(feature = "test-hooks")]
thunk!(MAKE_CONTINUOUS_PROBE: CFunctionKw = thunk_make_continuous_probe => env_type::make_continuous_probe : fn(Obj, Obj, Obj) -> Obj);
#[cfg(feature = "test-hooks")]
thunk!(MAKE_MULTI_PROBE: CFunctionKw = thunk_make_multi_probe => env_type::make_multi_probe : fn(Obj, Obj, Obj) -> Obj);
thunk!(MAKE_ACROBOT: CFunctionKw = thunk_make_acrobot => env_type::make_acrobot : fn(Obj, Obj, Obj) -> Obj);
thunk!(MAKE_MOUNTAINCAR: CFunctionKw = thunk_make_mountaincar => env_type::make_mountaincar : fn(Obj, Obj, Obj) -> Obj);
#[cfg(feature = "test-hooks")]
thunk!(PENDULUM_REWARDS: CFunction = thunk_pendulum_rewards => module::pendulum_rewards : fn(Obj, Obj) -> Obj);
thunk!(MAKE_PENDULUM: CFunctionKw = thunk_make_pendulum => env_type::make_pendulum : fn(Obj, Obj, Obj) -> Obj);
thunk!(MAKE_CARTPOLE: CFunctionKw = thunk_make_cartpole => env_type::make_cartpole : fn(Obj, Obj, Obj) -> Obj);
thunk!(MAKE_GRID: CFunctionKw = thunk_make_grid => env_type::make_grid : fn(Obj, Obj, Obj) -> Obj);
thunk!(MAKE_KEYDOOR: CFunctionKw = thunk_make_keydoor => env_type::make_keydoor : fn(Obj, Obj, Obj) -> Obj);
thunk!(MAKE_SNAKE: CFunctionKw = thunk_make_snake => env_type::make_snake : fn(Obj, Obj, Obj) -> Obj);
thunk!(MAKE_BRICKWALL: CFunctionKw = thunk_make_brickwall => env_type::make_brickwall : fn(Obj, Obj, Obj) -> Obj);
thunk!(MAKE_PADDLEDUEL: CFunctionKw = thunk_make_paddleduel => env_type::make_paddleduel : fn(Obj, Obj, Obj) -> Obj);
thunk!(MAKE_TAG: CFunctionKw = thunk_make_tag => env_type::make_tag : fn(Obj, Obj, Obj) -> Obj);
thunk!(MAKE_RTS: CFunctionKw = thunk_make_rts => env_type::make_rts : fn(Obj, Obj, Obj) -> Obj);
thunk!(MAKE_CONNECT4: CFunctionKw = thunk_make_connect4 => env_type::make_connect4 : fn(Obj, Obj, Obj) -> Obj);
thunk!(MAKE_LEDUC: CFunctionKw = thunk_make_leduc => env_type::make_leduc : fn(Obj, Obj, Obj) -> Obj);
thunk!(MAKE_OTHELLO: CFunctionKw = thunk_make_othello => env_type::make_othello : fn(Obj, Obj, Obj) -> Obj);
thunk!(MAKE_GO9: CFunctionKw = thunk_make_go9 => env_type::make_go9 : fn(Obj, Obj, Obj) -> Obj);
thunk!(LAYOUT: CFunction = thunk_layout => env_type::layout : fn(Obj, Obj) -> Obj);
thunk!(OUTPUTS: CFunction = thunk_outputs => env_type::outputs : fn(Obj, Obj) -> Obj);
thunk!(IS_POISONED: CFunction = thunk_is_poisoned => env_type::is_poisoned : fn(Obj, Obj) -> Obj);
thunk!(CLOSE: CFunction = thunk_close => env_type::close : fn(Obj, Obj) -> Obj);
thunk!(RESET: CFunction = thunk_reset => env_type::reset : fn(Obj, Obj) -> Obj);
thunk!(STEP: CFunction = thunk_step => env_type::step : fn(Obj, Obj) -> Obj);
thunk!(SNAPSHOT: CFunction = thunk_snapshot => env_type::snapshot : fn(Obj, Obj) -> Obj);
thunk!(STATE_HASH: CFunction = thunk_state_hash => env_type::state_hash : fn(Obj, Obj) -> Obj);
thunk!(ILLEGAL_ACTIONS: CFunction = thunk_illegal_actions => env_type::illegal_actions : fn(Obj, Obj) -> Obj);
thunk!(IMPORT_RECORD: CFunction = thunk_import_record => env_type::import_record : fn(Obj, Obj) -> Obj);
thunk!(FORK: CFunction = thunk_fork => env_type::fork : fn(Obj, Obj) -> Obj);
thunk!(ENCODE: CFunction = thunk_encode => env_type::encode : fn(Obj, Obj) -> Obj);
thunk!(LOAD: CFunction = thunk_load => env_type::load : fn(Obj, Obj) -> Obj);
thunk!(SEND: CFunction = thunk_send => env_type::send : fn(Obj, Obj) -> Obj);
thunk!(RECV: CFunction = thunk_recv => env_type::recv : fn(Obj, Obj) -> Obj);
thunk!(RESTORE: CFunction = thunk_restore => env_type::restore : fn(Obj, Obj) -> Obj);
thunk!(SNAPSHOT_ALL: CFunction = thunk_snapshot_all => env_type::snapshot_all : fn(Obj, Obj) -> Obj);
thunk!(RESTORE_ALL: CFunction = thunk_restore_all => env_type::restore_all : fn(Obj, Obj) -> Obj);
thunk!(SNAPSHOT_DEALLOC: Destructor = thunk_snapshot_dealloc => snapshot_type::dealloc : fn(Obj) -> ());
thunk!(DEALLOC: Destructor = thunk_dealloc => env_type::dealloc : fn(Obj) -> ());
thunk!(MAKE_SEARCH: CFunctionKw = thunk_make_search => search_type::make_search : fn(Obj, Obj, Obj) -> Obj);
thunk!(SEARCH_LAYOUT: CFunction = thunk_search_layout => search_type::layout : fn(Obj, Obj) -> Obj);
thunk!(SEARCH_SET_ROOT: CFunction = thunk_search_set_root => search_type::set_root : fn(Obj, Obj) -> Obj);
thunk!(SEARCH_RUN: CFunction = thunk_search_run => search_type::run : fn(Obj, Obj) -> Obj);
thunk!(SEARCH_ROOT_VISITS: CFunction = thunk_search_root_visits => search_type::root_visits : fn(Obj, Obj) -> Obj);
thunk!(SEARCH_SELECT_ACTION: CFunction = thunk_search_select_action => search_type::select_action : fn(Obj, Obj) -> Obj);
thunk!(SEARCH_IMPROVED_POLICY: CFunction = thunk_search_improved_policy => search_type::improved_policy : fn(Obj, Obj) -> Obj);
thunk!(SEARCH_CONSIDERED: CFunction = thunk_search_considered => search_type::considered : fn(Obj, Obj) -> Obj);
thunk!(SEARCH_CALLS: CFunction = thunk_search_calls => search_type::calls : fn(Obj, Obj) -> Obj);
thunk!(SEARCH_TREE_HASH: CFunction = thunk_search_tree_hash => search_type::tree_hash : fn(Obj, Obj) -> Obj);
thunk!(SEARCH_IS_PRIVILEGED: CFunction = thunk_search_is_privileged => search_type::is_privileged : fn(Obj, Obj) -> Obj);
thunk!(SEARCH_DEALLOC: Destructor = thunk_search_dealloc => search_type::dealloc : fn(Obj) -> ());
thunk!(SEARCH_ATTACH_ONNX: CFunctionKw = thunk_search_attach_onnx => search_type::attach_onnx : fn(Obj, Obj, Obj) -> Obj);
thunk!(SEARCH_RUN_ONNX: CFunction = thunk_search_run_onnx => search_type::run_onnx : fn(Obj, Obj) -> Obj);
thunk!(ONNX_LOAD: CFunctionKw = thunk_onnx_load => search_type::onnx_load : fn(Obj, Obj, Obj) -> Obj);
thunk!(MAKE_SEARCH_GROUP: CFunctionKw = thunk_make_search_group => search_group_type::make_search_group : fn(Obj, Obj, Obj) -> Obj);
thunk!(GROUP_LAYOUT: CFunction = thunk_group_layout => search_group_type::layout : fn(Obj, Obj) -> Obj);
thunk!(GROUP_LAST_FAILURE: CFunction = thunk_group_last_failure => search_group_type::last_failure : fn(Obj, Obj) -> Obj);
thunk!(GROUP_SET_ROOT: CFunction = thunk_group_set_root => search_group_type::set_root : fn(Obj, Obj) -> Obj);
thunk!(GROUP_SET_ACTIVE: CFunction = thunk_group_set_active => search_group_type::set_active : fn(Obj, Obj) -> Obj);
thunk!(GROUP_ACTIVE: CFunction = thunk_group_active => search_group_type::active : fn(Obj, Obj) -> Obj);
thunk!(GROUP_RUN: CFunction = thunk_group_run => search_group_type::run : fn(Obj, Obj) -> Obj);
thunk!(GROUP_ROOT_VISITS: CFunction = thunk_group_root_visits => search_group_type::root_visits : fn(Obj, Obj) -> Obj);
thunk!(GROUP_SELECT_ACTION: CFunction = thunk_group_select_action => search_group_type::select_action : fn(Obj, Obj) -> Obj);
thunk!(GROUP_IMPROVED_POLICY: CFunction = thunk_group_improved_policy => search_group_type::improved_policy : fn(Obj, Obj) -> Obj);
thunk!(GROUP_CONSIDERED: CFunction = thunk_group_considered => search_group_type::considered : fn(Obj, Obj) -> Obj);
thunk!(GROUP_CALLS: CFunction = thunk_group_calls => search_group_type::calls : fn(Obj, Obj) -> Obj);
thunk!(GROUP_TREE_HASH: CFunction = thunk_group_tree_hash => search_group_type::tree_hash : fn(Obj, Obj) -> Obj);
thunk!(GROUP_DEALLOC: Destructor = thunk_group_dealloc => search_group_type::dealloc : fn(Obj) -> ());

// Borrowed output views (ADR-0003). The deleters and the capsule destructor
// are cleanup entries: they never fail or reject.
thunk!(VIEW_DEALLOC: Destructor = thunk_view_dealloc => view_type::dealloc : fn(Obj) -> ());
thunk!(VIEW_GETBUFFER: GetBuffer = thunk_view_getbuffer => view_type::getbuffer : fn(Obj, *mut ffi::Py_buffer, core::ffi::c_int) -> core::ffi::c_int);
thunk!(VIEW_DLPACK: CFunctionKw = thunk_view_dlpack => view_type::dlpack : fn(Obj, Obj, Obj) -> Obj);
thunk!(VIEW_DLPACK_DEVICE: CFunction = thunk_view_dlpack_device => view_type::dlpack_device : fn(Obj, Obj) -> Obj);
thunk!(VIEW_GENERATION: CFunction = thunk_view_generation => view_type::generation : fn(Obj, Obj) -> Obj);
thunk!(VIEW_IS_STALE: CFunction = thunk_view_is_stale => view_type::is_stale : fn(Obj, Obj) -> Obj);
