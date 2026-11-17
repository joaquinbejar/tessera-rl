# tessera-rl-pyffi

The CPython boundary shared by Tessera's Python extensions (ADR-0041,
ADR-0044, ADR-0062 item 8): the floating-point entry thunks and their
table, the C-API wrappers and callouts, DLPack and buffer exports, and the
mapping from engine errors to Python exceptions.

Experimental until 1.0. Its API serves Tessera's own extension crates and
may change in any 0.x release.
