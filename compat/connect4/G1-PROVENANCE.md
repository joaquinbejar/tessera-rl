<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->

# P2-G1 opponent provenance

ADR-0042 pins the P2-G1 opponent by the source hash of the reference bot,
`tools/connect4-ref/src/bot.rs`. P2-M1b ports that bot into the Connect
Four environment crate. The port is not the same file, so its identity
rests on evidence, not on a shared hash: its tests check that it plays
exactly like the reference and reproduce the ADR-0042 calibration.

| Item | Value |
| --- | --- |
| Reference bot (ADR-0042) | `tools/connect4-ref/src/bot.rs`, SHA-256 `b21c14c26fc77054951c7aa44c139237e0299dc6aa376fa417ae1765bc9ef646` |
| Port, as reviewed | `crates/tessera-rl-games/src/connect4/bot.rs`, SHA-256 `ae3fc1bbbde31a6f569440107ee2872247aa37461fd2ef81d85689d2dfe5a1e3`, PR #34 head `86d33f32527d12aad14e435c1cbac9e53c1d3e68` (rebased to `48826935dacc3fc05cf5749c01f622c32730c212`, merged as `fddd7e3fb625c78951e22cd4a4580942a5d90471`) |
| Port, current | SHA-256 `9a4725d962b75def197d078982423cf6a44b6b10409f74099d15a54e42c56559`: the `TIE_BASE` doc comment states the per-opening rule (P2-M1b PR 3); no code change |

## Evidence

`make connect4-calibration` (`tests/bot.rs`, including the ignored solver
games) checks the following.

**Equality with the reference:**
- the same evaluation, root scores and column on the 1,000 calibration
  positions, at depth 4 and depth 1;
- the same on every ply of 300 random games;
- the same on every bot move of the calibration games.

**The ADR-0042 calibration, number for number:**

| Against | Result |
| --- | --- |
| Random play | 1000/0/0 |
| The depth-1 bot | 984/0/16 |
| Agreement with the solver | 844 optimal and 944 in the same outcome class, of 1,000 positions |
| The solver, bot moving first | 82/30/388 |
| The solver, bot moving second | 44/29/427 |

Results are win/draw/loss for the bot.

It passed on these platforms:
- macOS arm64;
- Linux arm64;
- Linux x86-64 (xaskar, in the pinned `rust:1.99.1` image).

Windows is pending, as M1b debt.

A P2-G1 report names this file's current port hash and the commit it ran
on. Any change to the port's hash needs `make connect4-calibration` to
pass, and a new row in the table.
