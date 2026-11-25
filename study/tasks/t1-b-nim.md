# T1, sheet B: Nim

Write this environment as a new crate on the `Environment` trait, starting
from `cargo tessera new <name> --kind env --agents 2 --turns sequential
--tessera-path <checkout>`. The task ends when `cargo tessera check` passes
and the facilitator's grader passes.

The grader replays your environment from fixed seeds with a fixed action
script and compares every step with a reference, so follow the rules and the
encoding below exactly.

## Rules

- Three heaps, which hold 3, 4 and 5 items at reset. Nim draws no random
  numbers.
- Two players alternate, slot 0 first (sequential turns: one player acts per
  step).
- Actions, `Discrete(9)`: action `a` takes `a % 3 + 1` items from heap
  `a / 3`. It is legal when that heap holds at least that many items. The
  mask has one row per player, and both rows, the waiting player's too,
  mark the actions that are legal on the current heaps. The no-op is the
  lowest-numbered legal action.
- Each step: the player to move takes their items, then the turn passes to
  the other player, and
  - if every heap is empty, the player who just moved wins: reward 1.0 to
    them and -1.0 to the other, and the episode terminates;
  - otherwise both rewards are 0.0, and the episode truncates at step 64,
    counting the steps since reset (configuration field `episode`, default
    64).

## Encoding

- Observation, the same for both players: exactly one leaf,
  `Box(i32, [4])` with bounds 0 to 5: the three heap sizes, then the slot of
  the player to move (after the step).
- The configuration's defaults are the values above.

## Grading interface

End `src/lib.rs` with a type alias naming your environment:

```rust
/// The environment the study's grader runs.
pub type StudyEnv = YourEnvironment;
```

The grader runs `cargo tessera check`, `cargo clippy --all-targets -- -D
warnings`, the conformance replay (observations, rewards, episode flags and
masks) and a lint check that `src/` has no unsafe code and no `unwrap`,
`expect`, `panic` or unchecked indexing.
