# tessera-rl-pyffi

The CPython boundary shared by Tessera's Python extensions (ADR-0041,
ADR-0044, ADR-0062 item 8): a safe typed layer (object bodies and method
traits, module definitions, read-only buffer and DLPack exports, C-API
wrappers under a call-scoped interpreter token) over the floating-point
entry thunks, the callouts and the mapping from engine errors to Python
exceptions. No public item is `unsafe`, and none relies on its caller for
memory safety, with two documented exceptions (the crate docs): a read of a
borrowed export racing the next write, or a write through a legacy DLPack
capsule (both left to the caller by SPEC 5.3), and a daemon thread inside
`detach` at interpreter finalization, which CPython 3.11 to 3.13 end with
`pthread_exit`: on glibc its forced unwind stops at pyo3-ffi's
`PyEval_RestoreThread` guard and the thread parks forever, and on macOS
the thread ends without unwinding, so no drop above `detach` runs.

Excluded from the 1.x stability promise (ADR-0094 item 3). Its API serves
Tessera's own extension crates and may change in a minor release. Host
protocols 1 and 2 are stable, and so is `tessera_rl.jax.ffi_handle` as Python
API.
