<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->

# tessera-rl-league

League machinery for Tessera (ADR-0026, ADR-0088): a population of opponent
policies, per-instance match results, Elo and TrueSkill ratings, and
uniform, PFSP and Elo matchmaking tables, all updated only at epoch
barriers under the canonical floating-point scope.

This crate is the pure library (ADR-0088 head A). The engine wiring (the
`Opponent` slot group, assignment at reset, pool context and replays)
lands after the policy map (#347) and the epoch barrier (#359).
