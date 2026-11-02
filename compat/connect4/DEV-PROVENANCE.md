<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->

# P2-M3 dev openings

`openings-dev.txt` holds 200 Connect Four openings. They are for **P2-M3 pipeline checks only**: the C1-against-C0 head-to-head of the self-play baseline. They are never used for P2-G1 evaluation, held-out checkpoint selection, or tuning against the gate. The set is frozen in the first commit of the P2-M3 self-play PR, before any training. Positions are never selected or edited based on any checkpoint's results.

- **Generator:** `tools/connect4-ref`, `dev-openings` subcommand. It runs the same procedure as ADR-0042 §3: `K = 8` random plies, the `L = 8` forced-win filter, and novelty up to mirror.
- **Seed base:** `0x5032473140000000`, seeds `base + j`. It is 2^30 away from the evaluation base (`0x5032473100000000`) and the held-out base (`0x5032473180000000`).
- **Disjointness:** canonical keys (mirror-normalized) from both existing files are excluded during generation, and the check is printed after it.

```
cargo run --release -p tessera-rl-connect4-ref -- dev-openings \
  compat/connect4/openings-eval.txt compat/connect4/openings-heldout.txt compat/connect4/openings-dev.txt
{"dev": 200, "distinct_up_to_mirror": 200, "shared_with_eval": 0, "shared_with_heldout": 0}
```

The pinned files (ADR-0042 Evidence; unchanged, checked by `tests/python/test_connect4_pins.py`):

| File | SHA-256 |
| --- | --- |
| `compat/connect4/openings-dev.txt` (200, this set) | `6208bf759e5adf712040f37f05da9a33c469e57ce077b1e6eb8a3270faffebd2` |
| `tools/connect4-ref/src/openings.rs` (generator and filter) | `6cdd2ae7dd9d6fbf6b49c54aace88811b214a21e1acf7620fb2deb4d3a38c218` |
| `tools/connect4-ref/src/solver.rs` | `2927b3f6e49c8defcd5b2348989131cfe2733033cefb44fdce121f2de1e9d19c` |
| `tools/connect4-ref/src/board.rs` | `664eb16cbbc6d44ddd58a99cfad0e14af6e7dec6c21976f44a255a28bd7a7040` |
| `tools/connect4-ref/src/rng.rs` | `907f5f89679dab32cd8a6f6e982b59984e4113dae92a7946aaf0287a0c1255b2` |
| `compat/connect4/openings-eval.txt` | `54a1cea7e6fe70d53a039bc27176486953c8bb6a243353175128526702360475` |
| `compat/connect4/openings-heldout.txt` | `3276d2d044c911ed29c36f2d42690697f19a4bac0183e3a77b1af3c5af72cc49` |

The `openings` subcommand still reproduces the two ADR-0042 files byte for byte after the `dev-openings` change, which was checked on macOS arm64.
