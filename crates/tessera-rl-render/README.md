<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->

# tessera-rl-render

Rendering for Tessera (ADR-0017, ADR-0083): an integer, bit-exact rasterizer
of the vector scenes that `Render::draw` writes from a snapshot's state. Off
the simulation path; nothing in `reset`, `step` or `observe` depends on it.
