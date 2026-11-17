# tessera-rl-pyffi

The CPython boundary shared by Tessera's Python extensions (ADR-0041,
ADR-0044, ADR-0062 item 8): a safe typed layer (object bodies and method
traits, module definitions, read-only buffer and DLPack exports, C-API
wrappers under a call-scoped interpreter token) over the floating-point
entry thunks, the callouts and the mapping from engine errors to Python
exceptions. No public item is `unsafe`.

Experimental until 1.0. Its API serves Tessera's own extension crates and
may change in any 0.x release.
